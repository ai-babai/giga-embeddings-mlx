from __future__ import annotations

from pathlib import Path

import pytest

from giga_embeddings_mlx import loading


def test_local_directory_is_resolved_without_hub_access(tmp_path: Path) -> None:
    assert loading.resolve_model_path(tmp_path) == tmp_path.resolve()


def test_default_uses_pinned_release_and_forwards_cache_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    observed: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(loading, "snapshot_download", fake_snapshot_download)

    resolved = loading.resolve_model_path(
        "default", cache_dir=cache_dir, local_files_only=True, token="secret"
    )

    assert resolved == snapshot.resolve()
    assert observed == {
        "repo_id": "ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64",
        "revision": "0826-v0.1.0",
        "cache_dir": str(cache_dir),
        "local_files_only": True,
        "token": "secret",
    }


def test_explicit_hub_repository_accepts_explicit_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    observed: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(loading, "snapshot_download", fake_snapshot_download)

    loading.resolve_model_path("owner/model", revision="immutable-commit")

    assert observed["repo_id"] == "owner/model"
    assert observed["revision"] == "immutable-commit"


def test_unknown_source_has_actionable_error() -> None:
    with pytest.raises(FileNotFoundError, match="neither an existing local path"):
        loading.resolve_model_path("missing")


def test_memory_preflight_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
    monkeypatch.setattr(loading, "_physical_memory_bytes", lambda: 1024)

    with pytest.raises(MemoryError, match="--skip-memory-check"):
        loading._preflight_model_memory(tmp_path)
