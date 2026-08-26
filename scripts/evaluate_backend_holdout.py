from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from evaluate_holdout import INSTRUCTIONS, compare, ranking_comparison, read_jsonl
from giga_embeddings_mlx.prompting import format_query


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_reference(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    chunks = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress_interval = max(total_batches // 20, 1)
    started_at = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            chunks.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
            del encoded, hidden, mask, pooled
            completed_batches = start // batch_size + 1
            if (
                completed_batches % progress_interval == 0
                or completed_batches == total_batches
            ):
                elapsed = time.monotonic() - started_at
                rate = completed_batches / elapsed
                remaining = (total_batches - completed_batches) / rate
                print(
                    "[backend-holdout] "
                    f"{completed_batches}/{total_batches} batches "
                    f"({completed_batches / total_batches:.0%}), "
                    f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s",
                    flush=True,
                )
    return np.concatenate(chunks)


def load_or_create_reference(
    cache_path: Path,
    *,
    model_path: Path,
    aligned: list[dict],
    queries: list[dict],
    documents: list[dict],
    device: torch.device,
    batch_size: int,
    aligned_batch_size: int,
    retrieval_only: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as stored:
            return {key: stored[key] for key in stored.files}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    specifications = {
        "queries": (
            [format_query(INSTRUCTIONS[row["family"]], row["text"]) for row in queries],
            batch_size,
            512,
        ),
        "documents": (
            [row["text"] for row in documents],
            batch_size,
            512,
        ),
    }
    if not retrieval_only:
        specifications = {
            "aligned": (
                [row["text"] for row in aligned],
                aligned_batch_size,
                2048,
            ),
            **specifications,
        }
    values = {}
    for section, (texts, section_batch_size, max_length) in specifications.items():
        print(f"[backend-holdout] encoding PyTorch BF16: {section}", flush=True)
        values[section] = encode_reference(
            model,
            tokenizer,
            texts,
            device=device,
            batch_size=section_batch_size,
            max_length=max_length,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    del model, tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--mlx-cache", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aligned-batch-size", type=int, default=1)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    aligned = read_jsonl(args.holdout / "aligned-texts.jsonl")
    queries = read_jsonl(args.holdout / "queries.jsonl")
    documents = read_jsonl(args.holdout / "documents.jsonl")
    reference = load_or_create_reference(
        args.reference_cache,
        model_path=args.model_path,
        aligned=aligned,
        queries=queries,
        documents=documents,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        aligned_batch_size=args.aligned_batch_size,
        retrieval_only=args.retrieval_only,
    )
    with np.load(args.mlx_cache) as stored:
        mlx_values = {key: stored[key] for key in stored.files}

    expected_shapes = {
        "queries": (len(queries), reference["queries"].shape[1]),
        "documents": (len(documents), reference["documents"].shape[1]),
    }
    if not args.retrieval_only:
        expected_shapes = {
            "aligned": (len(aligned), reference["aligned"].shape[1]),
            **expected_shapes,
        }
    for section, expected_shape in expected_shapes.items():
        if reference[section].shape != expected_shape:
            raise ValueError(
                f"Reference {section} shape {reference[section].shape} != {expected_shape}"
            )
        if mlx_values[section].shape != expected_shape:
            raise ValueError(
                f"MLX {section} shape {mlx_values[section].shape} != {expected_shape}"
            )

    report = {
        "model": args.model,
        "source_revision": args.revision,
        "reference_backend": f"pytorch-bfloat16-{args.device}",
        "candidate_backend": "mlx-bfloat16",
        "holdout_manifest": json.loads((args.holdout / "manifest.json").read_text()),
        "query_instructions": INSTRUCTIONS,
        "batch_size": args.batch_size,
        "aligned_batch_size": args.aligned_batch_size,
        "retrieval_only": args.retrieval_only,
        "reference_cache": str(args.reference_cache.resolve()),
        "reference_cache_sha256": file_sha256(args.reference_cache),
        "mlx_cache": str(args.mlx_cache.resolve()),
        "mlx_cache_sha256": file_sha256(args.mlx_cache),
        "metrics": (
            ranking_comparison(reference, mlx_values, queries, documents)
            if args.retrieval_only
            else compare(
                reference,
                mlx_values,
                aligned,
                queries,
                documents,
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
