from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--texts-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    texts = [row["text"] for row in read_jsonl(args.texts_jsonl)]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True, dtype=torch.bfloat16)
        .cpu()
        .eval()
    )
    gate_modules = [layer.mlp.gate for layer in model.layers if hasattr(layer.mlp, "gate")]
    layer_chunks: list[list[np.ndarray]] = [[] for _ in gate_modules]
    current: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(_module, _inputs, output):
        _, weights, indices = output
        order = torch.argsort(weights, dim=-1, descending=True)
        current.append(
            (
                torch.gather(indices, 1, order).detach().cpu(),
                torch.gather(weights, 1, order).detach().cpu(),
            )
        )

    hooks = [module.register_forward_hook(capture) for module in gate_modules]
    token_count = 0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            current.clear()
            model(**encoded)
            if len(current) != len(gate_modules):
                raise RuntimeError(f"Expected {len(gate_modules)} router calls, got {len(current)}")
            valid = encoded["attention_mask"].numpy().astype(bool)
            batch, length = valid.shape
            for chunks, (indices, _weights) in zip(layer_chunks, current):
                shaped = indices.numpy().reshape(batch, length, -1)
                chunks.append(shaped[valid].astype(np.uint8, copy=False))
            token_count += int(valid.sum())

    for hook in hooks:
        hook.remove()
    arrays = {
        f"layer_{index:02d}": np.concatenate(chunks) for index, chunks in enumerate(layer_chunks)
    }
    arrays["token_count"] = np.asarray(token_count, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    del model
    gc.collect()
    print(args.output)


if __name__ == "__main__":
    main()
