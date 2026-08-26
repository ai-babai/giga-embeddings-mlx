from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from giga_embeddings_mlx import load_embedding_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def create_trace(
    model_path: Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    output: Path,
) -> None:
    loaded = load_embedding_model(model_path)
    trace_model = loaded.model.model
    if not hasattr(trace_model, "forward_with_router_trace"):
        raise TypeError(f"Model at {model_path} does not expose router traces")

    layers: list[list[np.ndarray]] | None = None
    token_count = 0
    for start in range(0, len(texts), batch_size):
        encoded = loaded.tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        hidden, current = trace_model.forward_with_router_trace(input_ids, attention_mask)
        mx.eval(hidden, *current)
        valid = np.asarray(encoded["attention_mask"], dtype=bool)
        if layers is None:
            layers = [[] for _ in current]
        if len(current) != len(layers):
            raise RuntimeError("Router layer count changed between batches")
        for chunks, indices in zip(layers, current):
            chunks.append(np.asarray(indices)[valid].astype(np.uint8, copy=False))
        token_count += int(valid.sum())
        del hidden, current, input_ids, attention_mask
        mx.clear_cache()

    if layers is None:
        raise ValueError("No calibration texts")
    arrays = {f"layer_{index:02d}": np.concatenate(chunks) for index, chunks in enumerate(layers)}
    arrays["token_count"] = np.asarray(token_count, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    del loaded
    gc.collect()
    mx.clear_cache()


def compare_traces(base_path: Path, candidate_path: Path, experts: int) -> dict:
    with np.load(base_path) as base_file, np.load(candidate_path) as candidate_file:
        layer_keys = sorted(key for key in base_file.files if key.startswith("layer_"))
        if layer_keys != sorted(key for key in candidate_file.files if key.startswith("layer_")):
            raise ValueError("Router layer sets differ")
        per_layer = []
        all_overlap = []
        all_top1_changes = []
        for key in layer_keys:
            base = base_file[key]
            candidate = candidate_file[key]
            if base.shape != candidate.shape:
                raise ValueError(
                    f"Router shape differs at {key}: {base.shape} vs {candidate.shape}"
                )
            matches = (base[:, :, None] == candidate[:, None, :]).any(axis=2)
            overlap = matches.mean(axis=1)
            top1_changes = base[:, 0] != candidate[:, 0]
            base_frequency = np.bincount(base.ravel(), minlength=experts) / base.size
            candidate_frequency = np.bincount(candidate.ravel(), minlength=experts) / candidate.size
            max_frequency_drift_pp = float(
                np.max(np.abs(base_frequency - candidate_frequency)) * 100.0
            )
            per_layer.append(
                {
                    "layer": key,
                    "tokens": int(base.shape[0]),
                    "mean_top4_set_overlap": float(overlap.mean()),
                    "fully_different_top1_fraction": float(top1_changes.mean()),
                    "expert_frequency_max_drift_pp": max_frequency_drift_pp,
                }
            )
            all_overlap.append(overlap)
            all_top1_changes.append(top1_changes)

    return {
        "router_layers": len(per_layer),
        "mean_top4_set_overlap": float(np.concatenate(all_overlap).mean()),
        "fully_different_top1_fraction": float(np.concatenate(all_top1_changes).mean()),
        "expert_frequency_max_drift_pp": max(
            row["expert_frequency_max_drift_pp"] for row in per_layer
        ),
        "per_layer": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    trace = subparsers.add_parser("trace")
    trace.add_argument("model_path", type=Path)
    inputs = trace.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration", type=Path)
    inputs.add_argument("--texts-jsonl", type=Path)
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--batch-size", type=int, default=1)
    trace.add_argument("--max-length", type=int, default=512)
    compare = subparsers.add_parser("compare")
    compare.add_argument("base", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--variant", required=True)
    compare.add_argument("--experts", type=int, default=64)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "trace":
        source = (
            args.calibration / "texts.jsonl" if args.calibration is not None else args.texts_jsonl
        )
        rows = read_jsonl(source)
        create_trace(
            args.model_path,
            [row["text"] for row in rows],
            args.batch_size,
            args.max_length,
            args.output,
        )
        print(args.output)
        return

    report = {
        "variant": args.variant,
        "base_trace": str(args.base),
        "candidate_trace": str(args.candidate),
        **compare_traces(args.base, args.candidate, args.experts),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
