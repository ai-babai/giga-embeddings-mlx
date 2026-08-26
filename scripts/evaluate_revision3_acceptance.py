from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FAMILIES = ("ru", "en", "code", "multilingual")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def metric_check(name: str, value: float, operator: str, threshold: float) -> dict:
    if not math.isfinite(value):
        passed = False
    elif operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def first_relevant_rank(
    order: np.ndarray, relevant_ids: set[str], documents: list[dict]
) -> int | None:
    for rank, document_index in enumerate(order, start=1):
        if documents[int(document_index)]["id"] in relevant_ids:
            return rank
    return None


def top1_audit(report: dict, queries: list[dict], documents: list[dict]) -> list[dict]:
    reference_path = Path(report["reference_cache"])
    candidate_path = Path(report["mlx_cache"])
    if file_sha256(reference_path) != report["reference_cache_sha256"]:
        raise ValueError(f"Reference cache hash mismatch: {reference_path}")
    if file_sha256(candidate_path) != report["mlx_cache_sha256"]:
        raise ValueError(f"Candidate cache hash mismatch: {candidate_path}")

    with np.load(reference_path) as stored:
        reference_queries = stored["queries"]
        reference_documents = stored["documents"]
    with np.load(candidate_path) as stored:
        candidate_queries = stored["queries"]
        candidate_documents = stored["documents"]

    expected_query_shape = (len(queries), reference_queries.shape[1])
    expected_document_shape = (len(documents), reference_documents.shape[1])
    for name, values, expected in (
        ("reference queries", reference_queries, expected_query_shape),
        ("candidate queries", candidate_queries, expected_query_shape),
        ("reference documents", reference_documents, expected_document_shape),
        ("candidate documents", candidate_documents, expected_document_shape),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} != {expected}")

    reference_scores = reference_queries @ reference_documents.T
    candidate_scores = candidate_queries @ candidate_documents.T
    reference_top1 = np.argmax(reference_scores, axis=1)
    candidate_top1 = np.argmax(candidate_scores, axis=1)
    changed = np.flatnonzero(reference_top1 != candidate_top1)
    computed_agreement = 1.0 - len(changed) / len(queries)
    reported_agreement = report["metrics"]["ranking"]["top1_agreement"]
    if not math.isclose(computed_agreement, reported_agreement, abs_tol=1e-12):
        raise ValueError(f"Computed top-1 agreement {computed_agreement} != {reported_agreement}")

    audit = []
    for query_index in changed:
        query = queries[int(query_index)]
        relevant_ids = set(query["relevant_document_ids"])
        base_index = int(reference_top1[query_index])
        candidate_index = int(candidate_top1[query_index])
        base_id = documents[base_index]["id"]
        candidate_id = documents[candidate_index]["id"]
        base_order = np.argsort(-reference_scores[query_index])
        candidate_order = np.argsort(-candidate_scores[query_index])

        audit.append(
            {
                "query_id": query["id"],
                "family": query["family"],
                "reference_top1_document_id": base_id,
                "candidate_top1_document_id": candidate_id,
                "reference_top1_relevant": base_id in relevant_ids,
                "candidate_top1_relevant": candidate_id in relevant_ids,
                "reference_first_relevant_rank": first_relevant_rank(
                    base_order, relevant_ids, documents
                ),
                "candidate_first_relevant_rank": first_relevant_rank(
                    candidate_order, relevant_ids, documents
                ),
                "reference_margin_between_changed_documents": float(
                    reference_scores[query_index, base_index]
                    - reference_scores[query_index, candidate_index]
                ),
                "candidate_margin_between_changed_documents": float(
                    candidate_scores[query_index, candidate_index]
                    - candidate_scores[query_index, base_index]
                ),
            }
        )
    return audit


def evaluate_profile(
    model: str,
    parity_path: Path,
    holdout_path: Path,
    *,
    queries: list[dict],
    documents: list[dict],
    expected_corpus_sha256: str,
    thresholds: dict[str, float],
) -> dict:
    parity = read_json_object(parity_path)
    holdout = read_json_object(holdout_path)
    if parity["model"] != model or holdout["model"] != model:
        raise ValueError(f"Model label mismatch for {model}")
    if parity["source_revision"] != holdout["source_revision"]:
        raise ValueError(f"Source revision mismatch for {model}")
    if parity["corpus_sha256"] != expected_corpus_sha256:
        raise ValueError(f"Parity corpus hash mismatch for {model}")

    padding = parity["padding_invariance"]
    checks = [
        metric_check(
            "minimum pooled-vector cosine",
            parity["min_vector_cosine"],
            ">=",
            thresholds["min_vector_cosine"],
        ),
        metric_check(
            "mean pooled-vector cosine",
            parity["mean_vector_cosine"],
            ">=",
            thresholds["mean_vector_cosine"],
        ),
        metric_check(
            "pairwise similarity Spearman",
            parity["similarity_spearman"],
            ">=",
            thresholds["similarity_spearman"],
        ),
        metric_check(
            "maximum absolute similarity delta",
            parity["max_abs_similarity_delta"],
            "<=",
            thresholds["max_abs_similarity_delta"],
        ),
        metric_check(
            "MLX single/padded cosine",
            padding["mlx_single_vs_padded_cosine"],
            ">=",
            thresholds["mlx_padding_cosine"],
        ),
        metric_check(
            "MLX padding deficit vs PyTorch",
            padding["reference_single_vs_padded_cosine"] - padding["mlx_single_vs_padded_cosine"],
            "<=",
            thresholds["mlx_padding_deficit"],
        ),
    ]

    ranking = holdout["metrics"]["ranking"]
    for metric in ("mrr_delta", "ndcg_at_10_delta"):
        checks.append(
            metric_check(
                f"aggregate {metric}",
                ranking[metric],
                ">=",
                thresholds[metric],
            )
        )

    by_family = holdout["metrics"]["ranking_by_family"]
    if tuple(by_family) != FAMILIES:
        raise ValueError(f"Unexpected family order/set for {model}: {tuple(by_family)}")
    for family in FAMILIES:
        for metric in ("mrr_delta", "ndcg_at_10_delta"):
            checks.append(
                metric_check(
                    f"{family} {metric}",
                    by_family[family][metric],
                    ">=",
                    thresholds[metric],
                )
            )

    hidden_diagnostic = [
        {
            "layer_index": row["layer_index"],
            "global_valid_tokens_flattened_cosine": row["global_valid_tokens_flattened_cosine"],
            "min_per_text_flattened_cosine": row["min_per_text_flattened_cosine"],
            "max_abs_hidden_delta": row["max_abs_hidden_delta"],
        }
        for row in parity["selected_hidden_state_parity"]
    ]
    rank_diagnostic = {
        "top1_agreement": ranking["top1_agreement"],
        "mean_top10_overlap": ranking["mean_top10_overlap"],
        "min_top10_overlap": ranking["min_top10_overlap"],
        "changed_top1": top1_audit(holdout, queries, documents),
    }
    return {
        "model": model,
        "source_revision": parity["source_revision"],
        "parity_report": str(parity_path.resolve()),
        "parity_report_sha256": file_sha256(parity_path),
        "backend_holdout_report": str(holdout_path.resolve()),
        "backend_holdout_report_sha256": file_sha256(holdout_path),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
        "rank_diagnostic": rank_diagnostic,
        "hidden_state_diagnostic": hidden_diagnostic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        nargs=3,
        action="append",
        metavar=("MODEL", "PARITY_REPORT", "HOLDOUT_REPORT"),
        required=True,
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [profile[0] for profile in args.profile]
    if labels != ["480m", "3b", "10b-a1.8b"]:
        raise ValueError(f"Expected profiles in canonical order, got {labels}")

    queries = read_jsonl(args.holdout / "queries.jsonl")
    documents = read_jsonl(args.holdout / "documents.jsonl")
    thresholds = {
        "min_vector_cosine": 0.9985,
        "mean_vector_cosine": 0.9994,
        "similarity_spearman": 0.9994,
        "max_abs_similarity_delta": 0.0055,
        "mlx_padding_cosine": 0.9994,
        "mlx_padding_deficit": 0.0002,
        "mrr_delta": -0.005,
        "ndcg_at_10_delta": -0.005,
    }
    profiles = [
        evaluate_profile(
            model,
            Path(parity_path),
            Path(holdout_path),
            queries=queries,
            documents=documents,
            expected_corpus_sha256=args.expected_corpus_sha256,
            thresholds=thresholds,
        )
        for model, parity_path, holdout_path in args.profile
    ]
    report = {
        "schema_version": 1,
        "criteria_version": "0826-v3",
        "holdout": str(args.holdout.resolve()),
        "expected_corpus_sha256": args.expected_corpus_sha256,
        "thresholds": thresholds,
        "profiles": profiles,
        "overall_pass": all(profile["pass"] for profile in profiles),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
