from __future__ import annotations

import argparse
import gc
import hashlib
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


def load_corpus(path: Path | None) -> tuple[list[str], list[dict], list[str], str | None]:
    if path is None:
        records = [
            {"id": f"builtin-{index:02d}", "text": text}
            for index, text in enumerate(TEXTS)
        ]
        return TEXTS, records, TEXTS[:3], None

    raw = path.read_bytes()
    payload = json.loads(raw)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 11:
        raise ValueError("Parity corpus must contain at least 11 records for top-10")
    if any(not isinstance(record.get("text"), str) for record in records):
        raise ValueError("Every parity record must contain string field 'text'")
    by_id = {record.get("id"): record for record in records}
    padding_ids = payload.get("padding_control_ids", [])
    if not padding_ids or any(record_id not in by_id for record_id in padding_ids):
        raise ValueError("Parity corpus has invalid padding_control_ids")
    texts = [record["text"] for record in records]
    padding_texts = [by_id[record_id]["text"] for record_id in padding_ids]
    return texts, records, padding_texts, hashlib.sha256(raw).hexdigest()


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


def hidden_metrics(
    reference_chunks: list[np.ndarray],
    candidate_chunks: list[np.ndarray],
    attention_masks: list[np.ndarray],
) -> dict[str, float]:
    per_text_cosines = []
    valid_reference = []
    valid_candidate = []
    for reference, candidate, attention_mask in zip(
        reference_chunks, candidate_chunks, attention_masks, strict=True
    ):
        for row in range(len(reference)):
            valid = attention_mask[row].astype(bool)
            left = reference[row, valid].reshape(-1).astype(np.float64)
            right = candidate[row, valid].reshape(-1).astype(np.float64)
            denominator = max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
            per_text_cosines.append(float(np.dot(left, right) / denominator))
            valid_reference.append(reference[row, valid].reshape(-1))
            valid_candidate.append(candidate[row, valid].reshape(-1))
    left = np.concatenate(valid_reference).astype(np.float64)
    right = np.concatenate(valid_candidate).astype(np.float64)
    denominator = max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
    return {
        "min_per_text_flattened_cosine": float(min(per_text_cosines)),
        "mean_per_text_flattened_cosine": float(np.mean(per_text_cosines)),
        "global_valid_tokens_flattened_cosine": float(
            np.dot(left, right) / denominator
        ),
        "max_abs_hidden_delta": float(
            max(
                np.max(np.abs(left_values - right_values))
                for left_values, right_values in zip(
                    valid_reference, valid_candidate, strict=True
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    texts, records, padding_texts, corpus_sha256 = load_corpus(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    encoded_batches = [
        tokenizer(
            texts[start : start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        for start in range(0, len(texts), args.batch_size)
    ]
    model_token_lengths = [
        int(length)
        for encoded in encoded_batches
        for length in encoded["attention_mask"].sum(axis=1).tolist()
    ]
    content_token_lengths = [
        len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
    ]
    expected_content_lengths = [
        record.get("expected_content_tokens") for record in records
    ]
    expected_model_lengths = [record.get("expected_model_tokens") for record in records]
    if any(expected is not None for expected in expected_content_lengths):
        mismatches = [
            {
                "id": record.get("id"),
                "expected_content_tokens": expected_content,
                "actual_content_tokens": actual_content,
                "expected_model_tokens": expected_model,
                "actual_model_tokens": actual_model,
            }
            for record, expected_content, actual_content, expected_model, actual_model in zip(
                records,
                expected_content_lengths,
                content_token_lengths,
                expected_model_lengths,
                model_token_lengths,
                strict=True,
            )
            if (expected_content is not None and expected_content != actual_content)
            or (expected_model is not None and expected_model != actual_model)
        ]
        if mismatches:
            raise ValueError(f"Parity corpus token-length mismatch: {mismatches}")
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
    layer_count = len(reference.layers)
    selected_indices = sorted({0, layer_count // 2, layer_count - 1})
    reference_hidden_chunks = {index: [] for index in selected_indices}
    handles = []
    for index in selected_indices:
        def capture(_module, _inputs, output, *, layer_index=index):
            reference_hidden_chunks[layer_index].append(
                output.detach().float().cpu().numpy()
            )

        handles.append(reference.layers[index].register_forward_hook(capture))
    reference_chunks = []
    with torch.inference_mode():
        for encoded in encoded_batches:
            encoded_device = {key: value.to(device) for key, value in encoded.items()}
            hidden = reference(**encoded_device).last_hidden_state
            mask = encoded_device["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            reference_chunks.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
            del encoded_device, hidden, pooled
    for handle in handles:
        handle.remove()
    reference_embeddings = np.concatenate(reference_chunks)
    solo_encoded = tokenizer(
        [padding_texts[0]],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        solo_device = {key: value.to(device) for key, value in solo_encoded.items()}
        solo_hidden = reference(**solo_device).last_hidden_state
        solo_mask = solo_device["attention_mask"].unsqueeze(-1).to(solo_hidden.dtype)
        solo_pooled = (solo_hidden * solo_mask).sum(1) / solo_mask.sum(1).clamp(
            min=1e-6
        )
        reference_solo = F.normalize(solo_pooled, dim=-1).float().cpu().numpy()
        padded_encoded = tokenizer(
            padding_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        padded_device = {
            key: value.to(device) for key, value in padded_encoded.items()
        }
        padded_hidden = reference(**padded_device).last_hidden_state
        padded_mask = padded_device["attention_mask"].unsqueeze(-1).to(
            padded_hidden.dtype
        )
        padded_pooled = (padded_hidden * padded_mask).sum(1) / padded_mask.sum(
            1
        ).clamp(min=1e-6)
        reference_padded = F.normalize(padded_pooled, dim=-1).float().cpu().numpy()

    del (
        reference,
        reference_chunks,
        solo_device,
        solo_hidden,
        solo_pooled,
        padded_device,
        padded_hidden,
        padded_pooled,
    )
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    mlx_model = load_embedding_model(args.model_path)
    if args.dtype == "float32":
        mlx_model.model.set_dtype(mx.float32)
    mlx_chunks = []
    mlx_hidden_chunks = {index: [] for index in selected_indices}
    for encoded in encoded_batches:
        input_ids = mx.array(encoded["input_ids"].numpy())
        attention_mask = mx.array(encoded["attention_mask"].numpy())
        # Keep acceptance embeddings independent from diagnostic
        # materialization. Hidden-state capture is a separate pass over the
        # same inputs and cannot substitute its output for the measured path.
        values = pool_and_normalize(
            mlx_model.model(input_ids, attention_mask), attention_mask
        )
        mx.eval(values)
        mlx_chunks.append(np.array(values.astype(mx.float32), copy=True))
        _diagnostic_final, selected_hidden = (
            mlx_model.model.model.forward_with_selected_hidden_states(
                input_ids, attention_mask, set(selected_indices)
            )
        )
        mx.eval(*selected_hidden.values())
        for index in selected_indices:
            mlx_hidden_chunks[index].append(
                np.array(selected_hidden[index].astype(mx.float32), copy=True)
            )
        del values, input_ids, attention_mask, _diagnostic_final, selected_hidden
        mx.clear_cache()
    mlx_embeddings = np.concatenate(mlx_chunks)
    solo_input_ids = mx.array(solo_encoded["input_ids"].numpy())
    solo_attention_mask = mx.array(solo_encoded["attention_mask"].numpy())
    mlx_solo = pool_and_normalize(
        mlx_model.model(solo_input_ids, solo_attention_mask), solo_attention_mask
    )
    mx.eval(mlx_solo)
    mlx_solo = np.array(mlx_solo.astype(mx.float32), copy=True)
    padded_input_ids = mx.array(padded_encoded["input_ids"].numpy())
    padded_attention_mask = mx.array(padded_encoded["attention_mask"].numpy())
    mlx_padded = pool_and_normalize(
        mlx_model.model(padded_input_ids, padded_attention_mask),
        padded_attention_mask,
    )
    mx.eval(mlx_padded)
    mlx_padded = np.array(mlx_padded.astype(mx.float32), copy=True)
    reference_embeddings = normalize(reference_embeddings)
    reference_solo = normalize(reference_solo)
    reference_padded = normalize(reference_padded)
    mlx_embeddings = normalize(mlx_embeddings)
    mlx_solo = normalize(mlx_solo)
    mlx_padded = normalize(mlx_padded)
    row_cosines = np.sum(reference_embeddings * mlx_embeddings, axis=1)
    reference_scores = pairwise(reference_embeddings)
    mlx_scores = pairwise(mlx_embeddings)
    top10_overlap, top1_changes = ranking(reference_embeddings, mlx_embeddings)
    report = {
        "model": args.model,
        "source_revision": args.revision,
        "reference_device": str(device),
        "dtype": args.dtype,
        "texts": len(texts),
        "corpus": str(args.corpus.resolve()) if args.corpus else "builtin",
        "corpus_sha256": corpus_sha256,
        "record_ids": [record.get("id") for record in records],
        "content_token_lengths": content_token_lengths,
        "model_token_lengths": model_token_lengths,
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
        "padding_invariance": {
            "reference_single_vs_padded_cosine": float(
                np.sum(reference_solo[0] * reference_padded[0])
            ),
            "mlx_single_vs_padded_cosine": float(
                np.sum(mlx_solo[0] * mlx_padded[0])
            ),
        },
        "selected_hidden_state_parity": [
            {
                "layer_index": index,
                **hidden_metrics(
                    reference_hidden_chunks[index],
                    mlx_hidden_chunks[index],
                    [encoded["attention_mask"].numpy() for encoded in encoded_batches],
                ),
            }
            for index in selected_indices
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
