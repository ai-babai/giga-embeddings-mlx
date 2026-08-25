from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx_lm.models import qwen3


class Qwen3BidirectionalModel(qwen3.Qwen3Model):
    """Qwen3 encoder with full-sequence attention and no KV cache."""

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        # MLX SDPA treats True as an allowed key position. The singleton query
        # axis broadcasts the padding mask across every token, making attention
        # fully bidirectional while excluding padded keys.
        mask = None
        if attention_mask is not None:
            mask = attention_mask.astype(mx.bool_)[:, None, None, :]

        for layer in self.layers:
            hidden = layer(hidden, mask, cache=None)

        return self.norm(hidden)


class Model(nn.Module):
    """Checkpoint-compatible wrapper preserving the upstream ``model.*`` keys."""

    def __init__(self, args: qwen3.ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3BidirectionalModel(args)

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.model(inputs, attention_mask, input_embeddings)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers


ModelArgs = qwen3.ModelArgs


def get_model_classes(config: dict) -> tuple[type[nn.Module], type[qwen3.ModelArgs]]:
    if config.get("model_type") != "qwen3_bidirec":
        raise ValueError(f"Unsupported Qwen3 embedding model type: {config.get('model_type')!r}")
    return Model, ModelArgs
