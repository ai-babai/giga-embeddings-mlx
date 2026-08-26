from __future__ import annotations

import pytest

from giga_embeddings_mlx.models import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
    get_model_profile,
)


def test_release_catalog_contains_six_profiles_and_balanced_default() -> None:
    assert len(MODEL_PROFILES) == 6
    assert DEFAULT_PROFILE == "3b-q8"
    assert get_model_profile("default") is MODEL_PROFILES["3b-q8"]
    assert MODEL_PROFILES["3b-q8"].release_role == "balanced-default"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("480m", "480m-bf16"),
        ("3b", "3b-bf16"),
        ("10b-a1.8b", "10b-a1.8b-bf16"),
        ("3b-q8-edges-bf16-g64", "3b-q8"),
    ],
)
def test_compatibility_aliases(alias: str, canonical: str) -> None:
    assert get_model_profile(alias) is MODEL_PROFILES[canonical]


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unknown model profile"):
        get_model_profile("missing")
