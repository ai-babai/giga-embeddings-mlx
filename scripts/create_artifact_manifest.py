from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("variant")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("source_revision")
    parser.add_argument("--role", required=True)
    parser.add_argument("--status", choices=("accepted", "exploratory", "rejected"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.model_path.resolve()
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    report = {
        "schema_version": 1,
        "profile": args.profile,
        "variant": args.variant,
        "source_revision": args.source_revision,
        "role": args.role,
        "status": args.status,
        "artifact_root": str(root),
        "artifact_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "transformers": importlib.metadata.version("transformers"),
            "giga_embeddings_mlx": importlib.metadata.version("giga-embeddings-mlx"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
