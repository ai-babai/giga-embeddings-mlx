from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import mteb
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from giga_embeddings_mlx import load_embedding_model
from giga_embeddings_mlx.prompting import format_query

STS_TASK = "RuSTSBenchmarkSTS"
CLASSIFICATION_TASK = "RuToxicOKMLCUPClassification.v2"
CLASSIFICATION_INSTRUCTION = (
    "Classify whether the Russian text is toxic or non-toxic"
)


def load_tasks():
    sts_task = mteb.get_task(STS_TASK)
    sts_task.load_data()
    classification_task = mteb.get_task(CLASSIFICATION_TASK)
    classification_task.load_data()
    return sts_task, classification_task


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        values = model.encode(texts[start : start + batch_size], max_length=512)
        chunks.append(np.array(values.astype(mx.float32), copy=True))
        del values
        mx.clear_cache()
    return np.concatenate(chunks)


def create_embeddings(
    path: Path, sts, classification, batch_size: int, section_dir: Path
) -> dict:
    model = load_embedding_model(path)
    sts_test = sts.dataset["test"]
    classification_instruction = CLASSIFICATION_INSTRUCTION
    specifications = {
        "sts_sentence1": list(sts_test["sentence1"]),
        "sts_sentence2": list(sts_test["sentence2"]),
        "classification_train": [
            format_query(classification_instruction, text)
            for text in classification.dataset["train"]["text"]
        ],
        "classification_test": [
            format_query(classification_instruction, text)
            for text in classification.dataset["test"]["text"]
        ],
    }
    section_dir.mkdir(parents=True, exist_ok=True)
    values = {}
    for key, texts in specifications.items():
        section_path = section_dir / f"{key}.npy"
        if section_path.exists():
            values[key] = np.load(section_path)
            continue
        print(f"[downstream] encoding {path.name}: {key}", flush=True)
        values[key] = encode(model, texts, batch_size)
        np.save(section_path, values[key])
        print(f"[downstream] cached {section_path}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()
    return values


def load_or_create(cache: Path, path: Path, sts, classification, batch_size: int):
    if cache.exists():
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    values = create_embeddings(
        path, sts, classification, batch_size, cache.with_suffix("")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **values)
    return values


def evaluate(values: dict, sts, classification) -> dict:
    predicted = np.sum(values["sts_sentence1"] * values["sts_sentence2"], axis=1)
    gold = np.asarray(sts.dataset["test"]["score"], dtype=np.float32)
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=826,
        solver="liblinear",
    )
    classifier.fit(
        values["classification_train"], classification.dataset["train"]["label"]
    )
    labels = np.asarray(classification.dataset["test"]["label"])
    predictions = classifier.predict(values["classification_test"])
    return {
        "ru_sts": {
            "spearman": float(spearmanr(predicted, gold).statistic),
            "pearson": float(pearsonr(predicted, gold).statistic),
            "pairs": len(gold),
        },
        "ru_classification": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "train_samples": len(classification.dataset["train"]),
            "test_samples": len(labels),
        },
    }


def with_delta(base: dict, candidate: dict) -> dict:
    result = json.loads(json.dumps(candidate))
    for family in ("ru_sts", "ru_classification"):
        for metric in set(base[family]).intersection(candidate[family]):
            if metric not in {"pairs", "train_samples", "test_samples"}:
                result[family][f"{metric}_delta"] = (
                    candidate[family][metric] - base[family][metric]
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("base", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sts, classification = load_tasks()
    base_values = load_or_create(
        args.cache_dir / f"{args.model}-bf16.npz",
        args.base,
        sts,
        classification,
        args.batch_size,
    )
    base_metrics = evaluate(base_values, sts, classification)
    results = []
    for path in args.candidates:
        values = load_or_create(
            args.cache_dir / f"{path.name}.npz",
            path,
            sts,
            classification,
            args.batch_size,
        )
        results.append(
            {"variant": path.name, **with_delta(base_metrics, evaluate(values, sts, classification))}
        )
    report = {
        "model": args.model,
        "tasks": {
            STS_TASK: {
                "dataset": sts.metadata.dataset,
                "license": str(sts.metadata.license),
                "split": "test",
            },
            CLASSIFICATION_TASK: {
                "dataset": classification.metadata.dataset,
                "license": str(classification.metadata.license),
                "splits": ["train", "test"],
                "instruction": CLASSIFICATION_INSTRUCTION,
            },
        },
        "baseline": base_metrics,
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
