from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx_lm.models import deepseek_v3


class Float32MoEGate(deepseek_v3.MoEGate):
    """DeepSeek router matching the upstream FP32 selection contract."""

    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return deepseek_v3.group_expert_select(
            logits,
            self.e_score_correction_bias.astype(mx.float32),
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class DeepseekV3BidirectionalModel(deepseek_v3.DeepseekV3Model):
    """DeepSeek-V3 encoder with full-sequence attention and no KV cache."""

    def __init__(self, args: deepseek_v3.ModelArgs):
        super().__init__(args)
        for layer in self.layers:
            if isinstance(layer.mlp, deepseek_v3.DeepseekV3MoE):
                layer.mlp.gate = Float32MoEGate(args)

    @staticmethod
    def _reference_attention(
        attention: deepseek_v3.DeepseekV3Attention,
        x: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        """Expanded MLA path matching upstream operation ordering.

        MLX-LM's generation-optimized DeepSeek path keeps the KV latent
        compressed through attention. That is algebraically equivalent, but
        it changes BF16 rounding enough to perturb encoder routers. Embedding
        parity uses the upstream expanded key/value ordering instead.
        """

        batch, length, _ = x.shape
        if attention.q_lora_rank is None:
            q = attention.q_proj(x)
        else:
            q = attention.q_b_proj(
                attention.q_a_layernorm(attention.q_a_proj(x))
            )
        q = q.reshape(
            batch, length, attention.num_heads, attention.q_head_dim
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [attention.qk_nope_head_dim], axis=-1)

        compressed_kv = attention.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(
            compressed_kv, [attention.kv_lora_rank], axis=-1
        )
        k_pe = k_pe.reshape(
            batch, length, 1, attention.qk_rope_head_dim
        ).transpose(0, 2, 1, 3)
        kv_latent = attention.kv_a_layernorm(compressed_kv)[:, None, :, :]
        q_pe = attention.rope(q_pe, 0)
        k_pe = attention.rope(k_pe, 0)

        k_nope = attention.embed_q(kv_latent, transpose=False)
        values = attention.unembed_out(kv_latent)
        k_pe = mx.broadcast_to(
            k_pe,
            (
                batch,
                attention.num_heads,
                length,
                attention.qk_rope_head_dim,
            ),
        )
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        output = deepseek_v3.scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=attention.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return attention.o_proj(output)

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        mask = None
        if attention_mask is not None:
            mask = attention_mask.astype(mx.bool_)[:, None, None, :]

        # Embedding inference does not use autoregressive caches or distributed
        # pipeline transport. The stock MLX DeepSeek blocks, MLA and MoE stay
        # unchanged; only the attention mask differs from the causal LM path.
        for layer in self.layers:
            attention_out = self._reference_attention(
                layer.self_attn, layer.input_layernorm(hidden), mask
            )
            residual = hidden + attention_out
            hidden = residual + layer.mlp(
                layer.post_attention_layernorm(residual)
            )

        return self.norm(hidden)

    def forward_with_router_trace(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        """Return hidden states plus routed-expert indices for MoE QA."""

        hidden = self.embed_tokens(inputs)
        mask = (
            attention_mask.astype(mx.bool_)[:, None, None, :]
            if attention_mask is not None
            else None
        )
        router_indices = []
        for layer in self.layers:
            attention_out = self._reference_attention(
                layer.self_attn, layer.input_layernorm(hidden), mask
            )
            residual = hidden + attention_out
            mlp_input = layer.post_attention_layernorm(residual)
            if isinstance(layer.mlp, deepseek_v3.DeepseekV3MoE):
                indices, scores = layer.mlp.gate(mlp_input)
                # Both upstream torch.topk(sorted=False) and MLX argpartition
                # return an unspecified order. Sort only the diagnostic trace
                # by actual routing weight so "top-1" has stable semantics.
                trace_order = mx.argsort(-scores, axis=-1)
                router_indices.append(
                    mx.take_along_axis(indices, trace_order, axis=-1)
                )
                mlp_out = layer.mlp.switch_mlp(mlp_input, indices)
                mlp_out = (mlp_out * scores[..., None]).sum(axis=-2).astype(
                    mlp_out.dtype
                )
                if layer.mlp.config.n_shared_experts is not None:
                    mlp_out = mlp_out + layer.mlp.shared_experts(mlp_input)
            else:
                mlp_out = layer.mlp(mlp_input)
            hidden = residual + mlp_out

        return self.norm(hidden), router_indices

    def forward_with_selected_hidden_states(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None,
        layer_indices: set[int],
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Return selected block outputs without retaining every layer."""

        hidden = self.embed_tokens(inputs)
        mask = (
            attention_mask.astype(mx.bool_)[:, None, None, :]
            if attention_mask is not None
            else None
        )
        selected = {}
        for index, layer in enumerate(self.layers):
            attention_out = self._reference_attention(
                layer.self_attn, layer.input_layernorm(hidden), mask
            )
            residual = hidden + attention_out
            hidden = residual + layer.mlp(
                layer.post_attention_layernorm(residual)
            )
            if index in layer_indices:
                selected[index] = hidden
        return self.norm(hidden), selected


class Model(deepseek_v3.Model):
    """Checkpoint-compatible encoder wrapper reusing MLX-LM sanitization."""

    def __init__(self, args: deepseek_v3.ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV3BidirectionalModel(args)

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.model(inputs, attention_mask, input_embeddings)


ModelArgs = deepseek_v3.ModelArgs


def get_model_classes(
    config: dict,
) -> tuple[type[nn.Module], type[deepseek_v3.ModelArgs]]:
    if config.get("model_type") != "deepseek_v3_bidirec":
        raise ValueError(
            f"Unsupported DeepSeek embedding model type: {config.get('model_type')!r}"
        )
    return Model, ModelArgs
