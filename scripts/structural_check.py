from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from giga_embeddings_mlx import load_embedding_model

TEXTS = [
    "Короткий текст для проверки паддинга.",
    "A longer English passage used to verify deterministic batching behavior. " * 24,
    "def add(a, b):\n    return a + b",
]


def as_numpy(value) -> np.ndarray:
    return np.asarray(value.astype(mx.float32))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(left * right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--expected-dimension", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    loaded = load_embedding_model(args.model_path)
    if args.dtype == "float32":
        loaded.model.set_dtype(mx.float32)
    single = as_numpy(loaded.encode(TEXTS[0]))[0]
    padded = as_numpy(loaded.encode(TEXTS))[0]
    regular = as_numpy(loaded.encode(TEXTS))
    loaded.tokenizer.padding_side = "left"
    left_padded = as_numpy(loaded.encode(TEXTS))[0]
    loaded.tokenizer.padding_side = "right"
    permuted = as_numpy(loaded.encode([TEXTS[2], TEXTS[0], TEXTS[1]]))[[1, 2, 0]]
    repeated = as_numpy(loaded.encode(TEXTS))
    norms = np.linalg.norm(regular, axis=1)
    report = {
        "model": args.model,
        "model_type": loaded.config["model_type"],
        "dtype": args.dtype,
        "output_dimension": int(regular.shape[1]),
        "expected_dimension": args.expected_dimension,
        "single_vs_padded_cosine": cosine(single, padded),
        "right_vs_left_padded_cosine": cosine(padded, left_padded),
        "min_permuted_batch_cosine": float(np.min(np.sum(regular * permuted, axis=1))),
        "max_deterministic_abs_delta": float(np.max(np.abs(regular - repeated))),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "uses_generation_cache": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
