from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from giga_embeddings_mlx import load_embedding_model
from giga_embeddings_mlx.pooling import pool_and_normalize

TEXTS = [
    "Instruct: Given a query, retrieve relevant passages\nQuery: Где столица России?",
    "Москва — столица Российской Федерации.",
    "Париж — столица Франции.",
    "Instruct: Given a query, retrieve relevant passages\nQuery: How do solar panels work?",
    "Photovoltaic cells convert sunlight directly into electricity.",
    "A short sentence.",
    "Короткое предложение.",
    "Нейронные сети обучаются на данных и минимизируют функцию потерь.",
    "Machine learning systems discover statistical patterns in data.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id;",
    "Чуть более длинный текст нужен для проверки паддинга в батче. " * 16,
]


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def pairwise(values: np.ndarray) -> np.ndarray:
    scores = values @ values.T
    return scores[np.triu_indices(len(values), k=1)]


def ranking(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    left_scores = left @ left.T
    right_scores = right @ right.T
    np.fill_diagonal(left_scores, -np.inf)
    np.fill_diagonal(right_scores, -np.inf)
    width = min(10, len(left) - 1)
    left_order = np.argsort(-left_scores, axis=1)
    right_order = np.argsort(-right_scores, axis=1)
    overlap = [
        len(set(a[:width]).intersection(b[:width])) / width
        for a, b in zip(left_order, right_order)
    ]
    return float(np.mean(overlap)), int(np.sum(left_order[:, 0] != right_order[:, 0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    encoded_batches = [
        tokenizer(
            TEXTS[start : start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        for start in range(0, len(TEXTS), args.batch_size)
    ]
    device_name = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    reference = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch_dtype
    ).to(device).eval()
    reference_chunks = []
    with torch.inference_mode():
        for encoded in encoded_batches:
            encoded_device = {key: value.to(device) for key, value in encoded.items()}
            hidden = reference(**encoded_device).last_hidden_state
            mask = encoded_device["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            reference_chunks.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
            del encoded_device, hidden, pooled
    reference_embeddings = np.concatenate(reference_chunks)

    del reference, reference_chunks
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    mlx_model = load_embedding_model(args.model_path)
    if args.dtype == "float32":
        mlx_model.model.set_dtype(mx.float32)
    mlx_chunks = []
    for encoded in encoded_batches:
        input_ids = mx.array(encoded["input_ids"].numpy())
        attention_mask = mx.array(encoded["attention_mask"].numpy())
        values = pool_and_normalize(
            mlx_model.model(input_ids, attention_mask), attention_mask
        )
        mx.eval(values)
        mlx_chunks.append(np.array(values.astype(mx.float32), copy=True))
        del values, input_ids, attention_mask
        mx.clear_cache()
    mlx_embeddings = np.concatenate(mlx_chunks)
    reference_embeddings = normalize(reference_embeddings)
    mlx_embeddings = normalize(mlx_embeddings)
    row_cosines = np.sum(reference_embeddings * mlx_embeddings, axis=1)
    reference_scores = pairwise(reference_embeddings)
    mlx_scores = pairwise(mlx_embeddings)
    top10_overlap, top1_changes = ranking(reference_embeddings, mlx_embeddings)
    report = {
        "model": args.model,
        "source_revision": args.revision,
        "reference_device": str(device),
        "dtype": args.dtype,
        "texts": len(TEXTS),
        "batch_size": args.batch_size,
        "max_tokens_in_batch": max(
            int(encoded["attention_mask"].sum(axis=1).max())
            for encoded in encoded_batches
        ),
        "min_vector_cosine": float(row_cosines.min()),
        "mean_vector_cosine": float(row_cosines.mean()),
        "max_abs_similarity_delta": float(
            np.max(np.abs(reference_scores - mlx_scores))
        ),
        "similarity_rmse": float(
            np.sqrt(np.mean((reference_scores - mlx_scores) ** 2))
        ),
        "similarity_spearman": float(
            spearmanr(reference_scores, mlx_scores).statistic
        ),
        "mean_top10_overlap": top10_overlap,
        "top1_changes": top1_changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
