from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerBase, Qwen2Tokenizer

from .deepseek_v3_bidirectional import get_model_classes as get_deepseek_model_classes
from .models import MODEL_PROFILES, get_model_profile
from .pooling import pool_and_normalize
from .prompting import format_query
from .qwen3_bidirectional import get_model_classes as get_qwen3_model_classes


@dataclass(slots=True)
class EmbeddingModel:
    model: Any
    tokenizer: PreTrainedTokenizerBase
    config: dict[str, Any]
    path: Path

    def encode(self, texts: str | list[str], *, max_length: int = 8192) -> mx.array:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        hidden = self.model(input_ids, attention_mask)
        embeddings = pool_and_normalize(hidden, attention_mask)
        mx.eval(embeddings)
        return embeddings

    def encode_queries(
        self,
        queries: str | list[str],
        *,
        instruction: str,
        max_length: int = 8192,
    ) -> mx.array:
        if isinstance(queries, str):
            queries = [queries]
        return self.encode(
            [format_query(instruction, query) for query in queries],
            max_length=max_length,
        )

    def encode_documents(self, documents: str | list[str], *, max_length: int = 8192) -> mx.array:
        return self.encode(documents, max_length=max_length)


def _model_classes(config: dict):
    model_type = config.get("model_type")
    if model_type == "qwen3_bidirec":
        return get_qwen3_model_classes(config)
    if model_type == "deepseek_v3_bidirec":
        return get_deepseek_model_classes(config)
    raise ValueError(f"Unsupported embedding model type: {model_type!r}")


def resolve_model_path(
    model_source: str | Path = "default",
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> Path:
    value = str(model_source)
    path = Path(model_source).expanduser()
    if path.exists():
        return path.resolve()

    try:
        profile = get_model_profile(value)
    except ValueError:
        profile = None

    if profile is not None:
        repo_id = profile.repo_id
        resolved_revision = revision or profile.revision
    elif "/" in value and not value.startswith(("./", "../", "/")):
        repo_id = value
        resolved_revision = revision
    else:
        choices = ", ".join(["default", *MODEL_PROFILES])
        raise FileNotFoundError(
            f"Model source {value!r} is neither an existing local path, a Hub "
            f"repository (owner/name), nor a pinned profile. Profiles: {choices}"
        )

    try:
        snapshot = snapshot_download(
            repo_id,
            revision=resolved_revision,
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
            local_files_only=local_files_only,
            token=token,
        )
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"Offline model artifact {repo_id}@{resolved_revision or 'default'} is "
            "not present in the selected Hugging Face cache. Run once without "
            "offline mode, or point cache_dir/--cache-dir at a populated cache."
        ) from exc
    return Path(snapshot).resolve()


def _ensure_supported_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "giga-embeddings-mlx requires Apple Silicon (macOS arm64). "
            "On other platforms, use the upstream ai-sage checkpoints with "
            "their reference PyTorch runtime."
        )


def _physical_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def _preflight_model_memory(path: Path) -> None:
    weight_bytes = sum(file.stat().st_size for file in path.glob("*.safetensors"))
    physical_bytes = _physical_memory_bytes()
    if not weight_bytes or physical_bytes is None:
        return

    # Weight bytes are only part of the unified-memory working set. The 1.5x
    # factor plus 4 GiB OS/runtime reserve is deliberately conservative and is
    # derived from the accepted 0826 Metal peak measurements.
    required_bytes = int(weight_bytes * 1.5) + 4 * 1024**3
    if physical_bytes < required_bytes:
        required_gib = required_bytes / 1024**3
        physical_gib = physical_bytes / 1024**3
        raise MemoryError(
            f"Estimated safe unified-memory requirement is {required_gib:.1f} GiB, "
            f"but this Mac has {physical_gib:.1f} GiB. Choose a smaller profile, "
            "reduce competing memory use, or explicitly set "
            "skip_memory_check=True/--skip-memory-check if you accept swap or OOM risk."
        )


def load_embedding_model(
    model_source: str | Path = "default",
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    lazy: bool = False,
    skip_memory_check: bool = False,
) -> EmbeddingModel:
    _ensure_supported_platform()
    path = resolve_model_path(
        model_source,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    if not skip_memory_check:
        _preflight_model_memory(path)
    model, config = load_model(
        path,
        lazy=lazy,
        strict=True,
        get_model_classes=_model_classes,
    )
    # All three pinned profiles use the Qwen2 tokenizer vocabulary. Loading it
    # directly avoids asking Transformers to resolve the custom model_type and
    # keeps normal inference independent from checkpoint Python code.
    tokenizer = Qwen2Tokenizer.from_pretrained(path)
    return EmbeddingModel(model=model, tokenizer=tokenizer, config=config, path=path)
