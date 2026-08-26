---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - moe
  - q8
  - research
inference: false
---

# Giga Embeddings 0826 10B-A1.8B — MLX Q8 g64

[Русская карточка](README.ru.md)

> **Compact/research artifact with a code-retrieval warning.** Aggregate
> retrieval passed, but code-family NDCG@10 changed by `−0.01297` versus native
> MLX BF16. This model is not the default and is not claimed to be near-lossless.

Native-MLX quantization of
[`ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826)
for Apple Silicon. It keeps MoE routers and normalization weights in BF16. This
is an independent `ai-babai` artifact, not an official `ai-sage` release.

## Use

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("10b-a1.8b-q8")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Queries require an explicit instruction; documents have no prefix. Normal
inference uses [`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx)
and does not execute checkpoint Python.

## Artifact and measurements

- Quantization: direct from BF16, affine Q8 eligible weights, BF16 routers and
  norms, group size 64.
- Role: compact / research.
- Embedding dimension: 1536; maximum sequence length: 8192.
- Artifact: 11.144 GB; Metal peak at B16×1024: 14.423 GB.
- M4 Pro 48 GB, B1×512: 0.597 s median / 0.674 s p95.
- M4 Pro 48 GB, B16×1024: 0.76 documents/s and 776 tokens/s.
- Frozen holdout versus native MLX BF16: min/mean cosine
  0.993838/0.999272, top-1 agreement 97.66%, mean top-10 overlap 98.09%,
  aggregate NDCG@10 delta −0.00046.
- Code-family top-1 agreement: 92.19%; code NDCG@10 delta: −0.01297.
- Downstream delta: RuSTS +0.000350; classification accuracy +0.000541 and
  macro-F1 +0.000540. Positive deltas are not an improvement claim.

Full methodology, MoE-router gates, speed samples and source hashes are in the
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 reduced disk and Metal peak but was slower than BF16 at the displayed
workloads.

## Provenance

- Base revision: `1cb3ad3374dbf0eb9130546ca38b262de5f60287`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable file inventory and SHA-256 hashes.
- Model repository Python files and `auto_map` were intentionally omitted.
