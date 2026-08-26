from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from giga_embeddings_mlx import load_embedding_model


VARIANTS = (
    "480m-q8-g64",
    "3b-q8-edges-bf16-g64",
    "10b-a1.8b-q8-g64",
)

SMOKE_TEXTS = [
    "Instruct: Given a question, retrieve passages that answer the question\n"
    "Query: Где находится Москва?",
    "Москва — столица России.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(staging: Path) -> dict[str, Any]:
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)

    if any(value.startswith("/") for value in strings(manifest)):
        raise ValueError(f"Absolute path found in {manifest_path}")
    if list(staging.glob("*.py")):
        raise ValueError(f"Checkpoint Python found in {staging}")
    config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
    if "auto_map" in config:
        raise ValueError(f"auto_map found in {staging / 'config.json'}")

    expected = {item["path"]: item for item in manifest["files"]}
    actual_names = {
        path.name for path in staging.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual_names != set(expected):
        raise ValueError(f"Manifest inventory mismatch in {staging}")
    for name, item in expected.items():
        path = staging / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Manifest hash mismatch for {path}")
    return manifest


def encode(path: Path) -> np.ndarray:
    loaded = load_embedding_model(path, skip_memory_check=True)
    vectors = np.array(loaded.encode(SMOKE_TEXTS).astype(mx.float32))
    del loaded
    mx.clear_cache()
    gc.collect()
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for variant in VARIANTS:
        source = args.source_root / variant
        staging = args.staging_root / variant
        manifest = verify_manifest(staging)
        source_vectors = encode(source)
        staging_vectors = encode(staging)
        cosine = np.sum(source_vectors * staging_vectors, axis=1)
        max_abs_delta = float(np.max(np.abs(source_vectors - staging_vectors)))
        result = {
            "variant": variant,
            "repository": manifest["repository"],
            "shape": list(staging_vectors.shape),
            "finite": bool(np.isfinite(staging_vectors).all()),
            "minimum_same_artifact_cosine": float(cosine.min()),
            "maximum_absolute_delta": max_abs_delta,
            "strict_load": "pass",
            "manifest_sha256": sha256(staging / "manifest.json"),
        }
        if not result["finite"] or max_abs_delta != 0.0:
            raise ValueError(f"Embedding mismatch for {variant}: {result}")
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    report = {
        "schema_version": 1,
        "release_tag": "0826-v0.1.0",
        "checkpoint_python_omitted": True,
        "results": results,
        "overall_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
