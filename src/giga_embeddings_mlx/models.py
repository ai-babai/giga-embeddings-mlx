from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    repo_id: str
    revision: str
    architecture: str
    model_type: str
    embedding_dimension: int
    source_release: str = "0826"


MODEL_PROFILES: dict[str, ModelProfile] = {
    "480m": ModelProfile(
        name="480m",
        repo_id="ai-sage/Giga-Embeddings-instruct-480M-0826",
        revision="2d0c1a92716eef0e5b6972df85b5883eb5b4f57a",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=1024,
    ),
    "3b": ModelProfile(
        name="3b",
        repo_id="ai-sage/Giga-Embeddings-instruct-3B-0826",
        revision="ed7db5c91b900b39381b27b6e9c0a3d31137cd29",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=2048,
    ),
    "10b-a1.8b": ModelProfile(
        name="10b-a1.8b",
        repo_id="ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826",
        revision="1cb3ad3374dbf0eb9130546ca38b262de5f60287",
        architecture="deepseek_v3_bidirectional",
        model_type="deepseek_v3_bidirec",
        embedding_dimension=1536,
    ),
}


def get_model_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(MODEL_PROFILES)
        raise ValueError(f"Unknown model profile {name!r}; choose one of: {choices}") from exc

