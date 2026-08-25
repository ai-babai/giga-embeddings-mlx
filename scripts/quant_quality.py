from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.stats import spearmanr

from giga_embeddings_mlx import load_embedding_model

TOPICS = [
    ("столица России", "Москва является столицей Российской Федерации."),
    ("фотосинтез растений", "При фотосинтезе растения превращают свет в химическую энергию."),
    ("обучение нейронных сетей", "Нейронную сеть обучают, минимизируя функцию потерь на данных."),
    ("рецепт борща", "Борщ готовят из свёклы, капусты, овощей и бульона."),
    ("квантовая запутанность", "Запутанные квантовые состояния демонстрируют корреляции измерений."),
    ("индексация базы данных", "Индекс базы данных ускоряет поиск строк ценой памяти и записи."),
    ("погода в Арктике", "Арктический климат отличается долгой холодной зимой и коротким летом."),
    ("первая помощь при ожоге", "Небольшой ожог охлаждают прохладной проточной водой."),
    ("история железных дорог", "Железные дороги стали ключевой инфраструктурой промышленной эпохи."),
    ("защита персональных данных", "Персональные данные требуют контроля доступа и минимизации сбора."),
    ("solar panel efficiency", "Photovoltaic efficiency measures how much sunlight becomes electricity."),
    ("database transaction isolation", "Isolation levels control visibility between concurrent transactions."),
    ("how vaccines work", "Vaccines train the immune system to recognize a pathogen."),
    ("Python context manager", "A Python context manager controls setup and cleanup around a block."),
    ("ocean acidification", "Oceans become more acidic as they absorb atmospheric carbon dioxide."),
    ("renewable energy storage", "Batteries and pumped hydro can balance variable renewable generation."),
    ("causes of inflation", "Inflation can reflect demand, supply costs, and monetary conditions."),
    ("binary search algorithm", "Binary search repeatedly halves a sorted search interval."),
    ("Roman architecture", "Roman builders used arches, vaults, concrete, and monumental public spaces."),
    ("protein folding", "A protein's amino-acid sequence shapes its three-dimensional structure."),
]


def build_corpus() -> tuple[list[str], list[str]]:
    queries = [
        f"Instruct: Given a query, retrieve relevant passages\nQuery: {topic}"
        for topic, _ in TOPICS
    ]
    documents = []
    for index, (topic, answer) in enumerate(TOPICS):
        documents.extend(
            [
                answer,
                f"Краткая заметка по теме «{topic}»: {answer}",
                f"Reference entry {index}: this passage discusses {topic} in general terms.",
                f"Unrelated catalog record {index} containing neutral administrative metadata.",
                f"Архивная карточка номер {index}: техническое описание без ответа на запрос.",
            ]
        )
    return queries, documents


def encode(path: Path, texts: list[str], batch_size: int) -> np.ndarray:
    model = load_embedding_model(path)
    chunks = []
    for start in range(0, len(texts), batch_size):
        value = model.encode(texts[start : start + batch_size], max_length=512)
        chunks.append(np.array(value.astype(mx.float32), copy=True))
        del value
        mx.clear_cache()
    result = np.concatenate(chunks)
    del model
    gc.collect()
    mx.clear_cache()
    return result


def upper_triangle(scores: np.ndarray) -> np.ndarray:
    return scores[np.triu_indices(scores.shape[0], k=1)]


def metrics(
    base_all: np.ndarray,
    candidate_all: np.ndarray,
    base_query: np.ndarray,
    candidate_query: np.ndarray,
    base_docs: np.ndarray,
    candidate_docs: np.ndarray,
) -> dict:
    vector_cosine = np.sum(base_all * candidate_all, axis=1)
    base_pairwise = upper_triangle(base_all @ base_all.T)
    candidate_pairwise = upper_triangle(candidate_all @ candidate_all.T)
    base_retrieval = base_query @ base_docs.T
    candidate_retrieval = candidate_query @ candidate_docs.T
    base_top10 = np.argsort(-base_retrieval, axis=1)[:, :10]
    candidate_top10 = np.argsort(-candidate_retrieval, axis=1)[:, :10]
    overlaps = [
        len(set(left).intersection(right)) / 10
        for left, right in zip(base_top10, candidate_top10)
    ]
    return {
        "min_vector_cosine": float(vector_cosine.min()),
        "mean_vector_cosine": float(vector_cosine.mean()),
        "similarity_spearman": float(
            spearmanr(base_pairwise, candidate_pairwise).statistic
        ),
        "similarity_rmse": float(
            np.sqrt(np.mean((base_pairwise - candidate_pairwise) ** 2))
        ),
        "max_abs_similarity_delta": float(
            np.max(np.abs(base_pairwise - candidate_pairwise))
        ),
        "mean_top10_overlap": float(np.mean(overlaps)),
        "min_top10_overlap": float(np.min(overlaps)),
        "top1_changes": int(
            np.sum(np.argmax(base_retrieval, axis=1) != np.argmax(candidate_retrieval, axis=1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("base", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    queries, documents = build_corpus()
    all_texts = queries + documents
    base_all = encode(args.base, all_texts, args.batch_size)
    base_query = base_all[: len(queries)]
    base_docs = base_all[len(queries) :]
    results = []
    for candidate in args.candidates:
        candidate_all = encode(candidate, all_texts, args.batch_size)
        results.append(
            {
                "variant": candidate.name,
                **metrics(
                    base_all,
                    candidate_all,
                    base_query,
                    candidate_all[: len(queries)],
                    base_docs,
                    candidate_all[len(queries) :],
                ),
            }
        )

    report = {
        "model": args.model,
        "queries": len(queries),
        "documents": len(documents),
        "comparison_texts": len(all_texts),
        "baseline": str(args.base),
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
