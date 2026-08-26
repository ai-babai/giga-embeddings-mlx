---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-480M-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - q8
inference: false
---

# Giga Embeddings 0826 480M — MLX Q8 g64

[Русская карточка](README.ru.md)

[Giga Embeddings 0826 MLX Collection](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244)

Compact native-MLX quantization of
[`ai-sage/Giga-Embeddings-instruct-480M-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826)
for Apple Silicon. This is an independent `ai-babai` artifact, not an official
`ai-sage` release. Normal inference uses
[`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx) and
does not execute checkpoint Python.

## Use

Install the
[`giga-embeddings-mlx` package](https://pypi.org/project/giga-embeddings-mlx/):

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("480m-q8")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Queries require an explicit instruction. Documents have no prefix. Embeddings
use padding-aware FP32 mean pooling and FP32 L2 normalization.

## Artifact and measurements

- Quantization: direct from BF16, affine Q8, group size 64.
- Role: compact; not the default.
- Embedding dimension: 1024; maximum sequence length: 8192.
- Artifact: 0.525 GB; Metal peak at B16×1024: 1.339 GB.
- M4 Pro 48 GB, B1×512: 0.071 s median / 0.072 s p95.
- M4 Pro 48 GB, B16×1024: 6.38 documents/s and 6530 tokens/s.
- Frozen holdout versus native MLX BF16: min/mean vector cosine
  0.992867/0.998627, top-1 agreement 98.44%, mean top-10 overlap 96.72%,
  NDCG@10 delta +0.00289.

Small positive deltas mean no measured regression in this lane; they are not a
claim that quantization improves the model. Full methodology, p95, downstream
results, source hashes and historical revision 1/2 failures are in the
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).

## Provenance

- Base revision: `2d0c1a92716eef0e5b6972df85b5883eb5b4f57a`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable file inventory and SHA-256 hashes.
- Model repository Python files and `auto_map` were intentionally omitted.

Tested on Apple Silicon/macOS. Q8 is primarily a disk/capacity trade-off and
is not guaranteed to be faster than BF16.
