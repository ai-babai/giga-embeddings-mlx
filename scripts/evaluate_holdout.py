from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.stats import spearmanr

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


def encode_texts(model, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    values = []
    for start in range(0, len(texts), batch_size):
        batch = model.encode(texts[start : start + batch_size], max_length=max_length)
        values.append(np.array(batch.astype(mx.float32), copy=True))
        del batch
        # Dynamic padding creates many Metal allocation shapes. Release cached
        # intermediates after every evaluation batch while retaining weights.
        mx.clear_cache()
    return np.concatenate(values)


def create_embeddings(
    path: Path,
    aligned: list[dict],
    queries: list[dict],
    documents: list[dict],
    batch_size: int,
    aligned_batch_size: int,
    section_dir: Path,
) -> dict[str, np.ndarray]:
    model = load_embedding_model(path)
    section_dir.mkdir(parents=True, exist_ok=True)
    specifications = {
        "aligned": (
            model,
            [row["text"] for row in aligned],
            aligned_batch_size,
            2048,
        ),
        "queries": (
            model,
            [format_query(INSTRUCTIONS[row["family"]], row["text"]) for row in queries],
            batch_size,
            512,
        ),
        "documents": (
            model,
            [row["text"] for row in documents],
            batch_size,
            512,
        ),
    }
    values = {}
    for key, parameters in specifications.items():
        section_path = section_dir / f"{key}.npy"
        if section_path.exists():
            values[key] = np.load(section_path)
            continue
        print(f"[holdout] encoding {path.name}: {key}", flush=True)
        values[key] = encode_texts(*parameters)
        np.save(section_path, values[key])
        print(f"[holdout] cached {section_path}", flush=True)
    del model
    gc.collect()
    mx.clear_cache()
    return values


def load_or_create(
    cache_path: Path,
    model_path: Path,
    aligned: list[dict],
    queries: list[dict],
    documents: list[dict],
    batch_size: int,
    aligned_batch_size: int,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as stored:
            return {key: stored[key] for key in stored.files}
    values = create_embeddings(
        model_path,
        aligned,
        queries,
        documents,
        batch_size,
        aligned_batch_size,
        cache_path.with_suffix(""),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    return values


def retrieval_effectiveness(
    scores: np.ndarray,
    queries: list[dict],
    documents: list[dict],
) -> dict[str, float]:
    document_index = {row["id"]: index for index, row in enumerate(documents)}
    reciprocal_ranks = []
    ndcg10 = []
    for row_index, query in enumerate(queries):
        relevant = {
            document_index[doc_id]
            for doc_id in query["relevant_document_ids"]
            if doc_id in document_index
        }
        order = np.argsort(-scores[row_index])
        ranks = [rank for rank, doc_index in enumerate(order, start=1) if doc_index in relevant]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, doc_index in enumerate(order[:10], start=1)
            if doc_index in relevant
        )
        ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        ndcg10.append(dcg / ideal if ideal else 0.0)
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcg10)),
    }


def ranking_metrics(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    queries: list[dict],
    documents: list[dict],
) -> dict:
    base_scores = base["queries"] @ base["documents"].T
    candidate_scores = candidate["queries"] @ candidate["documents"].T
    base_top10 = np.argsort(-base_scores, axis=1)[:, :10]
    candidate_top10 = np.argsort(-candidate_scores, axis=1)[:, :10]
    overlap = [
        len(set(left).intersection(right)) / 10 for left, right in zip(base_top10, candidate_top10)
    ]
    base_effectiveness = retrieval_effectiveness(base_scores, queries, documents)
    candidate_effectiveness = retrieval_effectiveness(candidate_scores, queries, documents)
    return {
        "top1_agreement": float(
            np.mean(np.argmax(base_scores, axis=1) == np.argmax(candidate_scores, axis=1))
        ),
        "mean_top10_overlap": float(np.mean(overlap)),
        "min_top10_overlap": float(np.min(overlap)),
        "base_mrr": base_effectiveness["mrr"],
        "candidate_mrr": candidate_effectiveness["mrr"],
        "mrr_delta": candidate_effectiveness["mrr"] - base_effectiveness["mrr"],
        "base_ndcg_at_10": base_effectiveness["ndcg_at_10"],
        "candidate_ndcg_at_10": candidate_effectiveness["ndcg_at_10"],
        "ndcg_at_10_delta": candidate_effectiveness["ndcg_at_10"]
        - base_effectiveness["ndcg_at_10"],
    }


def subset_embeddings(
    values: dict[str, np.ndarray],
    query_indices: list[int],
    document_indices: list[int],
) -> dict[str, np.ndarray]:
    return {
        "queries": values["queries"][query_indices],
        "documents": values["documents"][document_indices],
    }


def ranking_comparison(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    queries: list[dict],
    documents: list[dict],
) -> dict:
    per_family = {}
    for family in INSTRUCTIONS:
        query_indices = [i for i, row in enumerate(queries) if row["family"] == family]
        document_indices = [i for i, row in enumerate(documents) if row["family"] == family]
        family_queries = [queries[i] for i in query_indices]
        family_documents = [documents[i] for i in document_indices]
        per_family[family] = ranking_metrics(
            subset_embeddings(base, query_indices, document_indices),
            subset_embeddings(candidate, query_indices, document_indices),
            family_queries,
            family_documents,
        )
    return {
        "ranking": ranking_metrics(base, candidate, queries, documents),
        "ranking_by_family": per_family,
    }


def compare(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    aligned: list[dict],
    queries: list[dict],
    documents: list[dict],
) -> dict:
    aligned_cosine = np.sum(base["aligned"] * candidate["aligned"], axis=1)
    triangle = np.triu_indices(len(aligned), k=1)
    base_pairwise = (base["aligned"] @ base["aligned"].T)[triangle]
    candidate_pairwise = (candidate["aligned"] @ candidate["aligned"].T)[triangle]
    ranking_results = ranking_comparison(base, candidate, queries, documents)
    return {
        "min_aligned_cosine": float(aligned_cosine.min()),
        "mean_aligned_cosine": float(aligned_cosine.mean()),
        "similarity_spearman": float(spearmanr(base_pairwise, candidate_pairwise).statistic),
        "similarity_rmse": float(np.sqrt(np.mean((base_pairwise - candidate_pairwise) ** 2))),
        "max_abs_similarity_delta": float(np.max(np.abs(base_pairwise - candidate_pairwise))),
        **ranking_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("base", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aligned-batch-size", type=int)
    args = parser.parse_args()

    aligned = read_jsonl(args.holdout / "aligned-texts.jsonl")
    queries = read_jsonl(args.holdout / "queries.jsonl")
    documents = read_jsonl(args.holdout / "documents.jsonl")
    aligned_batch_size = args.aligned_batch_size or args.batch_size
    base = load_or_create(
        args.cache_dir / f"{args.model}-bf16.npz",
        args.base,
        aligned,
        queries,
        documents,
        args.batch_size,
        aligned_batch_size,
    )
    results = []
    for candidate_path in args.candidates:
        candidate = load_or_create(
            args.cache_dir / f"{candidate_path.name}.npz",
            candidate_path,
            aligned,
            queries,
            documents,
            args.batch_size,
            aligned_batch_size,
        )
        results.append(
            {
                "variant": candidate_path.name,
                **compare(base, candidate, aligned, queries, documents),
            }
        )
    report = {
        "model": args.model,
        "holdout_manifest": json.loads((args.holdout / "manifest.json").read_text()),
        "query_instructions": INSTRUCTIONS,
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
