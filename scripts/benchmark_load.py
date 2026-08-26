from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path

import mlx.core as mx

from giga_embeddings_mlx import load_embedding_model


def artifact_bytes(path: Path) -> int:
    return sum(
        os.path.getsize(Path(root) / name)
        for root, _, files in os.walk(path, followlinks=True)
        for name in files
    )


def materialized_load(path: Path):
    started = time.perf_counter()
    loaded = load_embedding_model(path)
    mx.eval(loaded.model.parameters())
    mx.synchronize()
    return loaded, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first, first_seconds = materialized_load(args.model_path)
    del first
    gc.collect()
    mx.clear_cache()
    second, warm_seconds = materialized_load(args.model_path)
    del second
    gc.collect()
    mx.clear_cache()

    report = {
        "variant": args.variant,
        "model_path": str(args.model_path.resolve()),
        "artifact_bytes": artifact_bytes(args.model_path),
        "first_process_materialized_load_seconds": first_seconds,
        "warm_reload_seconds": warm_seconds,
        "os_page_cache_flushed": False,
        "interpretation": (
            "The first measurement is process-cold but the OS page cache is not "
            "controlled; warm reload follows a complete materialized load."
        ),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
