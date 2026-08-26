---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-3B-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - q8
inference: false
---

# Giga Embeddings 0826 3B — MLX Q8 + BF16 edges g64

[Русская карточка](README.ru.md)

[Giga Embeddings 0826 MLX Collection](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244)

Balanced native-MLX quantization of
[`ai-sage/Giga-Embeddings-instruct-3B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826)
for Apple Silicon. It keeps embedding and final edge layers in BF16 and uses Q8
for eligible inner weights. This is the disclosed `default` profile in
[`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx), not an
official `ai-sage` release.

## Use

Install the
[`giga-embeddings-mlx` package](https://pypi.org/project/giga-embeddings-mlx/):

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("default")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Queries require an explicit instruction; documents have no prefix. Normal
inference does not execute checkpoint Python.

## Artifact and measurements

- Quantization: direct from BF16, affine Q8 inner weights, BF16 edge layers,
  group size 64.
- Role: balanced default.
- Embedding dimension: 2048; maximum sequence length: 8192.
- Artifact: 3.755 GB; Metal peak at B16×1024: 5.137 GB.
- M4 Pro 48 GB, B1×512: 0.637 s median / 0.659 s p95.
- M4 Pro 48 GB, B16×1024: 0.73 documents/s and 744 tokens/s.
- Frozen holdout versus native MLX BF16: min/mean cosine
  0.992840/0.999575, top-1 agreement 99.61%, mean top-10 overlap 98.52%,
  NDCG@10 delta +0.00181.
- Downstream delta: RuSTS −0.000127; classification accuracy −0.000541 and
  macro-F1 −0.000540.

Full methodology, speed samples, backend gates and evidence hashes are in the
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 reduced disk and Metal peak but was slower than BF16 at the displayed
workloads.

## Provenance

- Base revision: `ed7db5c91b900b39381b27b6e9c0a3d31137cd29`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable file inventory and SHA-256 hashes.
- Model repository Python files and `auto_map` were intentionally omitted.
