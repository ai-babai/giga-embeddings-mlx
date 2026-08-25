from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from giga_embeddings_mlx import load_embedding_model
from giga_embeddings_mlx.pooling import pool_and_normalize

CASES = [
    (1, 128),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (8, 512),
    (8, 1024),
    (16, 512),
    (16, 1024),
    (1, 4096),
]


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def artifact_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=True):
        for name in files:
            total += os.path.getsize(Path(root) / name)
    return total


def make_inputs(tokenizer, batch: int, length: int) -> tuple[mx.array, mx.array]:
    seed = tokenizer.encode(
        "Тест производительности MLX embeddings. Benchmark input sequence.",
        add_special_tokens=True,
    )
    row = np.resize(np.asarray(seed, dtype=np.int32), length)
    ids = mx.array(np.tile(row, (batch, 1)))
    mask = mx.ones((batch, length), dtype=mx.bool_)
    return ids, mask


def synchronize(value: mx.array) -> None:
    mx.eval(value)
    mx.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    environment_before = {
        "swap": command_output(["sysctl", "-n", "vm.swapusage"]),
        "memory_pressure": command_output(["memory_pressure", "-Q"]),
    }
    load_started = time.perf_counter()
    loaded = load_embedding_model(args.model_path)
    cold_load_seconds = time.perf_counter() - load_started

    results = []
    for batch, length in CASES:
        ids, mask = make_inputs(loaded.tokenizer, batch, length)
        try:
            mx.clear_cache()
            mx.reset_peak_memory()
            first_started = time.perf_counter()
            first = pool_and_normalize(loaded.model(ids, mask), mask)
            synchronize(first)
            cold_inference = time.perf_counter() - first_started

            for _ in range(args.warmups):
                synchronize(pool_and_normalize(loaded.model(ids, mask), mask))

            timings = []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                synchronize(pool_and_normalize(loaded.model(ids, mask), mask))
                timings.append(time.perf_counter() - started)

            median = float(np.median(timings))
            results.append(
                {
                    "batch": batch,
                    "sequence_length": length,
                    "status": "ok",
                    "cold_inference_seconds": cold_inference,
                    "median_seconds": median,
                    "p95_seconds": float(np.percentile(timings, 95)),
                    "documents_per_second": batch / median,
                    "tokens_per_second": batch * length / median,
                    "metal_peak_bytes": int(mx.get_peak_memory()),
                    "metal_active_bytes": int(mx.get_active_memory()),
                    "samples_seconds": timings,
                }
            )
        # Preserve every unsupported/OOM profile as a benchmark outcome.
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "batch": batch,
                    "sequence_length": length,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            del ids, mask
            mx.clear_cache()

    report = {
        "variant": args.variant,
        "model_path": str(args.model_path.resolve()),
        "artifact_bytes": artifact_bytes(args.model_path),
        "cold_load_seconds": cold_load_seconds,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment_before": environment_before,
        "environment_after": {
            "swap": command_output(["sysctl", "-n", "vm.swapusage"]),
            "memory_pressure": command_output(["memory_pressure", "-Q"]),
        },
        "cases": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
