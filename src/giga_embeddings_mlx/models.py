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
    precision: str
    release_role: str
    source_release: str = "0826"


MODEL_PROFILES: dict[str, ModelProfile] = {
    "480m-bf16": ModelProfile(
        name="480m-bf16",
        repo_id="ai-sage/Giga-Embeddings-instruct-480M-0826",
        revision="2d0c1a92716eef0e5b6972df85b5883eb5b4f57a",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=1024,
        precision="BF16",
        release_role="upstream-baseline",
    ),
    "480m-q8": ModelProfile(
        name="480m-q8",
        repo_id="ai-babai/giga-embeddings-0826-480m-mlx-q8-g64",
        revision="cdf24f725e718b909449d3aae3dff61b677b0283",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=1024,
        precision="Q8 / group-size 64",
        release_role="compact",
    ),
    "3b-bf16": ModelProfile(
        name="3b-bf16",
        repo_id="ai-sage/Giga-Embeddings-instruct-3B-0826",
        revision="ed7db5c91b900b39381b27b6e9c0a3d31137cd29",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=2048,
        precision="BF16",
        release_role="upstream-baseline",
    ),
    "3b-q8": ModelProfile(
        name="3b-q8",
        repo_id="ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64",
        revision="829506daf10dff1abe75bd8412e3ad2ab1856123",
        architecture="qwen3_bidirectional",
        model_type="qwen3_bidirec",
        embedding_dimension=2048,
        precision="mixed Q8 / BF16 edges / group-size 64",
        release_role="balanced-default",
    ),
    "10b-a1.8b-bf16": ModelProfile(
        name="10b-a1.8b-bf16",
        repo_id="ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826",
        revision="1cb3ad3374dbf0eb9130546ca38b262de5f60287",
        architecture="deepseek_v3_bidirectional",
        model_type="deepseek_v3_bidirec",
        embedding_dimension=1536,
        precision="BF16",
        release_role="upstream-baseline",
    ),
    "10b-a1.8b-q8": ModelProfile(
        name="10b-a1.8b-q8",
        repo_id="ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64",
        revision="ce54ed6b20eb73d4aeefc5820f4a3463004d45c4",
        architecture="deepseek_v3_bidirectional",
        model_type="deepseek_v3_bidirec",
        embedding_dimension=1536,
        precision="Q8 / group-size 64",
        release_role="compact-research",
    ),
}

DEFAULT_PROFILE = "3b-q8"

PROFILE_ALIASES = {
    "default": DEFAULT_PROFILE,
    "480m": "480m-bf16",
    "3b": "3b-bf16",
    "3b-q8-edges-bf16-g64": "3b-q8",
    "10b-a1.8b": "10b-a1.8b-bf16",
}


def get_model_profile(name: str) -> ModelProfile:
    canonical_name = PROFILE_ALIASES.get(name, name)
    try:
        return MODEL_PROFILES[canonical_name]
    except KeyError as exc:
        choices = ", ".join(["default", *MODEL_PROFILES])
        raise ValueError(f"Unknown model profile {name!r}; choose one of: {choices}") from exc
