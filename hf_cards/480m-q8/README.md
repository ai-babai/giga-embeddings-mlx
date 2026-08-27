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

# Giga Embeddings 0826 480M — MLX Q8 for Apple Silicon

[Русская карточка](README.ru.md) ·
[All MLX models](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-for-apple-silicon-mlx-q8-6a8eec40b26f6543f5da3244) ·
[GitHub](https://github.com/ai-babai/giga-embeddings-mlx) ·
[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Original paper](https://arxiv.org/abs/2608.23806)

![Q8 download and peak Metal memory savings for Giga Embeddings 0826](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-q8-savings.png?v=2026-08-27)

The compact choice for local Russian and English semantic search, RAG, text
similarity, clustering, and classification on Apple Silicon. It is the smallest
released Giga Embeddings `0826` MLX model: a 0.525 GB download that used 1.339 GB
of peak Metal memory in our 16-text test.

## Quality and size at a glance

- **Original model quality:** 70.98 Russian MTEB in the authors'
  [paper](https://arxiv.org/html/2608.23806#S4).
- **Our Q8 preservation check:** aggregate NDCG@10 change `+0.00289` versus the
  native MLX BF16 model—no measured aggregate regression on the frozen set.
- **Q8 savings:** download 0.978 → 0.525 GB (**46% smaller**); peak Metal
  memory 1.792 → 1.339 GB (**25% lower**) versus native MLX BF16.
- **Typical time for one 512-token text:** 0.071 s on an M4 Pro.
- **Batch speed:** 6.38 documents/s for 16 texts of 1024 tokens each.

The official MTEB result belongs to the original BF16 model. We did not rerun
the complete MTEB suite on Q8; our local test separately checks whether the MLX
port and quantization preserve retrieval behavior.

## Choose a released MLX model

![Choose a Giga Embeddings 0826 MLX model](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=2026-08-26-2)

| Model | Best for | Q8 download | Original Russian MTEB | Our Q8 retrieval check |
|---|---|---:|---:|---:|
| [3B MLX Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8) | recommended default | 3.755 GB | 74.56 | NDCG@10 Δ +0.00181 |
| **[480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8)** | **smallest and fastest** | **0.525 GB** | **70.98** | **NDCG@10 Δ +0.00289** |
| [10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8) | research; code warning | 11.144 GB | 74.98 | aggregate Δ −0.00046 |

## Use

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
scores = queries @ documents.T
```

Queries require an explicit instruction; documents have no prefix. Normal
inference does not execute Python code from this model repository.

## What is inside

- Direct affine Q8 quantization from BF16, group size 64.
- Embedding dimension: 1024; maximum sequence length: 8192.
- Artifact size: 0.525 GB.
- Peak Metal memory: 1.339 GB for 16 texts of 1024 tokens.
- Min/mean vector cosine versus native MLX BF16: 0.992867 / 0.998627.
- Top-1 agreement: 98.44%; mean top-10 overlap: 96.72%.

Small positive deltas are evidence of no measured regression, not a claim that
quantization improves the original model. Full methodology, downstream results,
hardware details, and evidence hashes are in the
[`0826` MLX report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).

## Source and reproducibility

- Original model: [`ai-sage/Giga-Embeddings-instruct-480M-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826).
- Original paper: [arXiv:2608.23806](https://arxiv.org/abs/2608.23806).
- Base revision: `2d0c1a92716eef0e5b6972df85b5883eb5b4f57a`.
- Weight release: `0826-v0.1.0`; tensor bytes are unchanged in the `0.1.2`
  documentation update.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable inventory and SHA-256 hashes.

Tested on Apple Silicon/macOS. Q8 is primarily a memory and disk optimization,
not a guaranteed speed-up over BF16. This is an independent `ai-babai`
quantization, not an official `ai-sage` release.

## Citation

Please cite both this MLX software release and the original Giga-Embeddings
work:

```bibtex
@software{popkov2026gigaembeddingsmlx,
  author  = {Maksim Popkov},
  title   = {Giga Embeddings MLX},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/ai-babai/giga-embeddings-mlx}
}

@misc{kolodin2026gigaembeddings,
  title         = {Giga-Embeddings: Mixture-of-Experts Encoders for High-Throughput Text Embeddings},
  author        = {Egor Kolodin and Egor Krasnoperov and Evgeniy Kosarev and Fyodor Minkin},
  year          = {2026},
  eprint        = {2608.23806},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.23806}
}
```
