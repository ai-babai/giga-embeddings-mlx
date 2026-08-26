from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CONVERTER_COMMIT = "dfbc6a375ccdb637d1932529acbcfbf4db5025b6"
RUNTIME_VERSION = "0.1.0"
RELEASE_TAG = "0826-v0.1.0"

ARTIFACTS = {
    "480m-q8": {
        "source_dir": "480m-q8-g64",
        "manifest": "480m-q8-g64.json",
        "repo_id": "ai-babai/giga-embeddings-0826-480m-mlx-q8-g64",
        "base_model": "ai-sage/Giga-Embeddings-instruct-480M-0826",
        "source_revision": "2d0c1a92716eef0e5b6972df85b5883eb5b4f57a",
        "policy": "uniform affine Q8",
        "group_size": 64,
        "role": "compact",
    },
    "3b-q8": {
        "source_dir": "3b-q8-edges-bf16-g64",
        "manifest": "3b-q8-edges-bf16-g64.json",
        "repo_id": "ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64",
        "base_model": "ai-sage/Giga-Embeddings-instruct-3B-0826",
        "source_revision": "ed7db5c91b900b39381b27b6e9c0a3d31137cd29",
        "policy": "affine Q8 with BF16 embedding/final layers",
        "group_size": 64,
        "role": "balanced-default",
    },
    "10b-a1.8b-q8": {
        "source_dir": "10b-a1.8b-q8-g64",
        "manifest": "10b-a1.8b-q8-g64.json",
        "repo_id": "ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64",
        "base_model": "ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826",
        "source_revision": "1cb3ad3374dbf0eb9130546ca38b262de5f60287",
        "policy": "affine Q8 with BF16 routers/norms",
        "group_size": 64,
        "role": "compact-research",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]


def sanitize_config(source: Path, destination: Path) -> dict[str, str]:
    config = json.loads(source.read_text(encoding="utf-8"))
    removed = {}
    for key in ("auto_map",):
        if key in config:
            removed[key] = json.dumps(config.pop(key), sort_keys=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return removed


def prepare_one(
    alias: str,
    spec: dict[str, Any],
    source_root: Path,
    accepted_manifest_root: Path,
    cards_root: Path,
    staging_root: Path,
    refresh_metadata: bool,
) -> Path:
    source = source_root / spec["source_dir"]
    destination = staging_root / spec["source_dir"]
    if destination.exists() and any(destination.iterdir()) and not refresh_metadata:
        raise FileExistsError(f"Refusing to overwrite non-empty staging directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    accepted = json.loads((accepted_manifest_root / spec["manifest"]).read_text(encoding="utf-8"))
    accepted_files = {item["path"]: item for item in accepted["files"]}

    selected = [
        path
        for path in source.iterdir()
        if path.is_file()
        and (
            path.name in {"tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
            or path.name.endswith(".safetensors")
        )
    ]
    for path in selected:
        expected = accepted_files[path.name]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Accepted manifest mismatch for {path}")
        destination_file = destination / path.name
        if not (
            destination_file.exists()
            and destination_file.stat().st_size == expected["bytes"]
            and sha256(destination_file) == expected["sha256"]
        ):
            shutil.copy2(path, destination_file)

    source_config_sha256 = sha256(source / "config.json")
    removed_config_fields = sanitize_config(source / "config.json", destination / "config.json")
    for name in ("README.md", "README.ru.md"):
        shutil.copy2(cards_root / alias / name, destination / name)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(cards_root.parent / name, destination / name)
    shutil.copy2(cards_root / ".gitattributes", destination / ".gitattributes")

    manifest = {
        "schema_version": 1,
        "release_line": "0826",
        "release_tag": RELEASE_TAG,
        "repository": spec["repo_id"],
        "base_model": spec["base_model"],
        "base_model_relation": "quantized",
        "source_revision": spec["source_revision"],
        "parent_bf16": {
            "repository": spec["base_model"],
            "revision": spec["source_revision"],
        },
        "quantization": {
            "policy": spec["policy"],
            "group_size": spec["group_size"],
            "role": spec["role"],
            "direct_from_bf16": True,
        },
        "converter_commit": CONVERTER_COMMIT,
        "runtime": {
            "package": "giga-embeddings-mlx",
            "version": RUNTIME_VERSION,
            "strict_load": "pass",
            "checkpoint_python_required": False,
        },
        "sanitation": {
            "source_config_sha256": source_config_sha256,
            "removed_config_fields": removed_config_fields,
            "checkpoint_python_omitted": True,
            "tensor_files_changed": False,
        },
        "files": inventory(destination),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accepted-manifest-root", type=Path, required=True)
    parser.add_argument("--cards-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Refresh cards/config/manifest while preserving verified tensor files",
    )
    args = parser.parse_args()

    for alias, spec in ARTIFACTS.items():
        destination = prepare_one(
            alias,
            spec,
            args.source_root,
            args.accepted_manifest_root,
            args.cards_root,
            args.staging_root,
            args.refresh_metadata,
        )
        print(destination)


if __name__ == "__main__":
    main()
