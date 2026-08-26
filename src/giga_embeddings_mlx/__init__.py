from ._version import __version__
from .loading import EmbeddingModel, load_embedding_model
from .models import DEFAULT_PROFILE, MODEL_PROFILES, ModelProfile, get_model_profile
from .prompting import format_query

__all__ = [
    "MODEL_PROFILES",
    "DEFAULT_PROFILE",
    "EmbeddingModel",
    "ModelProfile",
    "__version__",
    "format_query",
    "get_model_profile",
    "load_embedding_model",
]
