from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerBase, Qwen2Tokenizer

from .deepseek_v3_bidirectional import get_model_classes as get_deepseek_model_classes
from .models import MODEL_PROFILES
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

    def encode_documents(
        self, documents: str | list[str], *, max_length: int = 8192
    ) -> mx.array:
        return self.encode(documents, max_length=max_length)


def _model_classes(config: dict):
    model_type = config.get("model_type")
    if model_type == "qwen3_bidirec":
        return get_qwen3_model_classes(config)
    if model_type == "deepseek_v3_bidirec":
        return get_deepseek_model_classes(config)
    raise ValueError(f"Unsupported embedding model type: {model_type!r}")


def resolve_model_path(model_path: str | Path) -> Path:
    value = str(model_path)
    if value in MODEL_PROFILES:
        profile = MODEL_PROFILES[value]
        return Path(
            snapshot_download(profile.repo_id, revision=profile.revision)
        ).resolve()
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {path}. Use a local path or a pinned profile: "
            f"{', '.join(MODEL_PROFILES)}"
        )
    return path


def load_embedding_model(model_path: str | Path, *, lazy: bool = False) -> EmbeddingModel:
    path = resolve_model_path(model_path)
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
