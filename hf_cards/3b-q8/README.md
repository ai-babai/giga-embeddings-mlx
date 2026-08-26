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
  - apple-silicon
  - embeddings
  - text-embeddings
  - semantic-search
  - rag
  - russian
  - local-ai
  - macos
  - quantized
  - q8
  - arxiv:2608.23806
inference: false
---

# Giga Embeddings 0826 3B — recommended MLX Q8 model for Apple Silicon

[Русская карточка](README.ru.md) ·
[All MLX models](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244) ·
[GitHub](https://github.com/ai-babai/giga-embeddings-mlx) ·
[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Original paper](https://arxiv.org/abs/2608.23806)

![Choose a Giga Embeddings 0826 MLX model](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=0.1.1)

The recommended default for local Russian and English semantic search, RAG,
text similarity, clustering, and classification on Apple Silicon. This mixed
Q8/BF16 artifact keeps the embedding and final edge layers in BF16 while
quantizing eligible inner weights to Q8.

## Quality and size at a glance

- **Original model quality:** 74.56 Russian MTEB in the authors'
  [paper](https://arxiv.org/html/2608.23806#S4).
- **Our Q8 preservation check:** aggregate NDCG@10 change `+0.00181` versus the
  native MLX BF16 model—no measured aggregate regression on the frozen set.
- **Download:** 3.755 GB; **peak Metal memory:** 5.137 GB in the 16-text test.
- **Typical time for one 512-token text:** 0.637 s on an M4 Pro.

The official MTEB result belongs to the original BF16 model. We did not rerun
the complete MTEB suite on Q8; our local test separately checks whether the MLX
port and quantization preserve retrieval behavior.

## Choose a released MLX model

| Model | Best for | Download | Original Russian MTEB | Our Q8 retrieval check |
|---|---|---:|---:|---:|
| **[3B Q8 + BF16 edges](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64)** | **recommended default** | **3.755 GB** | **74.56** | **NDCG@10 Δ +0.00181** |
| [480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8-g64) | smallest and fastest | 0.525 GB | 70.98 | NDCG@10 Δ +0.00289 |
| [10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64) | research; code warning | 11.144 GB | 74.98 | aggregate Δ −0.00046 |

## Use

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
scores = queries @ documents.T
```

Queries require an explicit instruction; documents have no prefix. Normal
inference does not execute Python code from this model repository.

## What is inside

- Affine Q8 inner weights, BF16 embedding and final edge layers, group size 64.
- Embedding dimension: 2048; maximum sequence length: 8192.
- Artifact size: 3.755 GB; peak Metal memory: 5.137 GB for 16 long texts.
- Batch speed: 0.73 documents/s for 16 texts of 1024 tokens each.
- Min/mean vector cosine versus native MLX BF16: 0.992840 / 0.999575.
- Top-1 agreement: 99.61%; mean top-10 overlap: 98.52%.
- RuSTS change: −0.000127; classification accuracy change: −0.000541.

Full methodology, speed samples, backend gates, and evidence hashes are in the
[`0826` MLX report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 reduced disk and peak Metal memory, but was slower than BF16 in the displayed
workloads.

## Source and reproducibility

- Original model: [`ai-sage/Giga-Embeddings-instruct-3B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826).
- Original paper: [arXiv:2608.23806](https://arxiv.org/abs/2608.23806).
- Base revision: `ed7db5c91b900b39381b27b6e9c0a3d31137cd29`.
- Weight release: `0826-v0.1.0`; tensor bytes are unchanged in the `0.1.1`
  documentation update.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable inventory and SHA-256 hashes.

Tested on Apple Silicon/macOS. This is an independent `ai-babai` quantization,
not an official `ai-sage` release.
