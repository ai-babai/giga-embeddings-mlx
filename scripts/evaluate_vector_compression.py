from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from giga_embeddings_mlx import load_embedding_model
from giga_embeddings_mlx.prompting import format_query

INSTRUCTIONS = {
    "ru": "Given a question, retrieve Wikipedia passages that answer the question",
    "en": "Given a scientific claim, retrieve documents that support or refute the claim",
    "code": "Given a natural language query, retrieve relevant code",
    "multilingual": "Given a question, retrieve Wikipedia passages that answer the question",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def encode_calibration(model_path: Path, rows: list[dict], batch_size: int) -> np.ndarray:
    model = load_embedding_model(model_path)
    texts = [row["text"] for row in rows]
    texts.extend(format_query(INSTRUCTIONS[row["family"]], row["text"]) for row in rows)
    chunks = []
    for start in range(0, len(texts), batch_size):
        values = model.encode(texts[start : start + batch_size], max_length=512)
        chunks.append(np.array(values.astype(mx.float32), copy=True))
        del values
        mx.clear_cache()
    return np.concatenate(chunks)


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def uint8_roundtrip(
    values: np.ndarray, minimum: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, int]:
    raw = np.rint((values - minimum) / scale)
    clipped = int(np.sum((raw < 0) | (raw > 255)))
    quantized = np.clip(raw, 0, 255).astype(np.uint8)
    return normalize(minimum + quantized.astype(np.float32) * scale), clipped


def effectiveness(order: np.ndarray, queries: list[dict], documents: list[dict]):
    document_index = {row["id"]: index for index, row in enumerate(documents)}
    reciprocal_ranks = []
    ndcg10 = []
    for row_index, query in enumerate(queries):
        relevant = {
            document_index[doc_id]
            for doc_id in query["relevant_document_ids"]
            if doc_id in document_index
        }
        ranks = [rank for rank, doc in enumerate(order[row_index], 1) if doc in relevant]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, doc in enumerate(order[row_index, :10], 1)
            if doc in relevant
        )
        ideal = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(len(relevant), 10) + 1)
        )
        ndcg10.append(dcg / ideal if ideal else 0.0)
    return float(np.mean(reciprocal_ranks)), float(np.mean(ndcg10))


def ranking_report(
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    queries: list[dict],
    documents: list[dict],
):
    base_order = np.argsort(-base_scores, axis=1)
    candidate_order = np.argsort(-candidate_scores, axis=1)
    overlap = [
        len(set(left).intersection(right)) / 10
        for left, right in zip(base_order[:, :10], candidate_order[:, :10])
    ]
    base_mrr, base_ndcg = effectiveness(base_order, queries, documents)
    candidate_mrr, candidate_ndcg = effectiveness(candidate_order, queries, documents)
    return {
        "top1_agreement": float(np.mean(base_order[:, 0] == candidate_order[:, 0])),
        "mean_top10_overlap": float(np.mean(overlap)),
        "mrr_delta": candidate_mrr - base_mrr,
        "ndcg_at_10_delta": candidate_ndcg - base_ndcg,
    }


def timed(operation, warmups: int = 2, repetitions: int = 5) -> dict:
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "samples_seconds": samples,
    }


def packed_binary_scores(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    packed_queries = np.packbits(queries >= 0, axis=1)
    packed_documents = np.packbits(documents >= 0, axis=1)
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    scores = np.empty((len(queries), len(documents)), dtype=np.float32)
    for row, query in enumerate(packed_queries):
        distances = popcount[np.bitwise_xor(packed_documents, query)].sum(axis=1)
        scores[row] = 1.0 - 2.0 * distances / queries.shape[1]
    return scores


def binary_rescore_scores(
    queries: np.ndarray,
    documents: np.ndarray,
    base_scores: np.ndarray,
    oversample: int,
) -> np.ndarray:
    binary_scores = packed_binary_scores(queries, documents)
    binary_order = np.argsort(-binary_scores, axis=1)
    rescored = np.full_like(base_scores, -np.inf)
    for row in range(len(queries)):
        candidates = binary_order[row, :oversample]
        rescored[row, candidates] = base_scores[row, candidates]
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-embeddings", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--oversample", type=int, default=100)
    args = parser.parse_args()

    queries = read_jsonl(args.holdout / "queries.jsonl")
    documents = read_jsonl(args.holdout / "documents.jsonl")
    calibration_rows = read_jsonl(args.calibration / "texts.jsonl")
    if args.calibration_cache.exists():
        calibration = np.load(args.calibration_cache)
    else:
        calibration = encode_calibration(
            args.model_path, calibration_rows, args.batch_size
        )
        args.calibration_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.calibration_cache, calibration)

    with np.load(args.holdout_embeddings) as stored:
        original = {key: stored[key] for key in stored.files}

    minimum = calibration.min(axis=0).astype(np.float32)
    maximum = calibration.max(axis=0).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 255.0, 1e-8)
    compressed = {}
    clipped = 0
    total_values = 0
    for key, values in original.items():
        compressed[key], current_clipped = uint8_roundtrip(values, minimum, scale)
        clipped += current_clipped
        total_values += values.size

    aligned_cosine = np.sum(original["aligned"] * compressed["aligned"], axis=1)
    base_scores = original["queries"] @ original["documents"].T
    uint8_scores = compressed["queries"] @ compressed["documents"].T
    uint8_metrics = {
        "min_aligned_cosine": float(aligned_cosine.min()),
        "mean_aligned_cosine": float(aligned_cosine.mean()),
        "clip_fraction": clipped / total_values,
        "storage_reduction_vs_fp32": 0.75,
        **ranking_report(base_scores, uint8_scores, queries, documents),
        "numpy_search_latency": timed(
            lambda: compressed["queries"] @ compressed["documents"].T
        ),
    }

    binary_scores = packed_binary_scores(original["queries"], original["documents"])
    direct_binary = ranking_report(base_scores, binary_scores, queries, documents)
    rescored = binary_rescore_scores(
        original["queries"], original["documents"], base_scores, args.oversample
    )
    rescored_binary = ranking_report(base_scores, rescored, queries, documents)
    binary_metrics = {
        "bits_per_dimension": 1,
        "storage_reduction_vs_fp32": 0.96875,
        "oversample": args.oversample,
        "direct": direct_binary,
        "rescored": rescored_binary,
        "packed_hamming_search_latency": timed(
            lambda: packed_binary_scores(
                original["queries"], original["documents"]
            )
        ),
        "packed_hamming_plus_rescore_latency": timed(
            lambda: binary_rescore_scores(
                original["queries"],
                original["documents"],
                base_scores,
                args.oversample,
            )
        ),
    }

    dimension = original["documents"].shape[1]
    report = {
        "model": args.model,
        "dimension": dimension,
        "calibration_manifest": json.loads(
            (args.calibration / "manifest.json").read_text()
        ),
        "uint8": uint8_metrics,
        "binary": binary_metrics,
        "bytes_per_vector": {
            "fp32": dimension * 4,
            "uint8": dimension,
            "binary": (dimension + 7) // 8,
        },
        "fp32_numpy_search_latency": timed(
            lambda: original["queries"] @ original["documents"].T
        ),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
