from __future__ import annotations

import mlx.core as mx


def mean_pool(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape (batch, sequence, hidden), "
            f"got {last_hidden_state.shape}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (batch, sequence), got {attention_mask.shape}"
        )
    if tuple(last_hidden_state.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            "attention_mask does not match hidden-state batch/sequence dimensions: "
            f"{attention_mask.shape} vs {last_hidden_state.shape}"
        )

    # Pool in float32 even when the transformer runs in BF16/quantized mode.
    # This avoids accumulating sequence-length-dependent BF16 rounding error
    # and matches the precision expected by downstream cosine search.
    hidden = last_hidden_state.astype(mx.float32)
    mask = attention_mask.astype(mx.float32)[..., None]
    counts = mx.maximum(mask.sum(axis=1), mx.array(1, dtype=mx.float32))
    return (hidden * mask).sum(axis=1) / counts


def l2_normalize(embeddings: mx.array, eps: float = 1e-12) -> mx.array:
    norms = mx.sqrt(mx.sum(embeddings.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    norms = mx.maximum(norms, mx.array(eps, dtype=mx.float32))
    return embeddings.astype(mx.float32) / norms


def pool_and_normalize(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    return l2_normalize(mean_pool(last_hidden_state, attention_mask))
