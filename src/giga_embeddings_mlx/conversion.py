from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx_lm.convert import QUANT_RECIPES, mixed_quant_predicate_builder
from mlx_lm.utils import quantize_model, save

from .loading import load_embedding_model


def convert_model(
    source: str | Path,
    destination: str | Path,
    *,
    bits: int | None = None,
    group_size: int = 64,
    recipe: str | None = None,
    policy: str | None = None,
) -> Path:
    """Materialize a native MLX embedding checkpoint, optionally quantized."""

    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    if bits is not None and bits not in {4, 6, 8}:
        raise ValueError("bits must be one of 4, 6, 8")
    if recipe is not None and recipe not in QUANT_RECIPES:
        raise ValueError(f"recipe must be one of: {', '.join(QUANT_RECIPES)}")
    if policy not in {None, "q8-edges-bf16"}:
        raise ValueError("policy must be q8-edges-bf16")
    if sum(value is not None for value in (bits, recipe, policy)) > 1:
        raise ValueError("choose one of uniform bits, a mixed recipe, or a policy")

    loaded = load_embedding_model(source, lazy=True)
    source_path = loaded.path
    model = loaded.model
    config = dict(loaded.config)
    if bits is not None or recipe is not None or policy is not None:
        if recipe is not None:
            predicate = mixed_quant_predicate_builder(recipe, model, group_size)
            quant_bits = 4
        elif policy == "q8-edges-bf16":
            final_layer = len(model.layers) - 1

            def predicate(path, module):
                return not (
                    "embed_tokens" in path
                    or ".layers.0." in f".{path}."
                    or f".layers.{final_layer}." in f".{path}."
                )

            quant_bits = 8
        else:
            predicate = None
            quant_bits = bits
        model, config = quantize_model(
            model,
            config,
            group_size=group_size,
            bits=quant_bits,
            mode="affine",
            quant_predicate=predicate,
        )

    mx.eval(model.parameters())
    save(
        destination_path,
        source_path,
        model,
        loaded.tokenizer,
        config,
        donate_model=True,
    )
    return destination_path
