---
license: mit
language:
  - ru
  - en
pipeline_tag: sentence-similarity
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826
base_model_relation: quantized
tags:
  - feature-extraction
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
  - moe
  - q8
  - research
  - arxiv:2608.23806
inference: false
---

# Giga Embeddings 0826 10B-A1.8B — MLX Q8 research model

[Русская карточка](README.ru.md) ·
[All MLX models](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-for-apple-silicon-mlx-q8-6a8eec40b26f6543f5da3244) ·
[GitHub](https://github.com/ai-babai/giga-embeddings-mlx) ·
[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Original paper](https://arxiv.org/abs/2608.23806)

![Q8 download and peak Metal memory savings for Giga Embeddings 0826](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-q8-savings.png?v=2026-08-27)

> **Research model with a code-search warning.** Overall retrieval quality
> passed our acceptance gate, but code-search NDCG@10 changed by `−0.01297`
> versus native MLX BF16. Use the 3B model by default unless this trade-off has
> been evaluated on your own data.

This is the high-capacity sparse Mixture-of-Experts option for local Russian
and English semantic search, RAG, text similarity, clustering, and
classification on Apple Silicon. The Q8 artifact keeps MoE routers and
normalization weights in BF16.

## Quality and size at a glance

- **Original model quality:** 74.98 Russian MTEB—the best result in the Giga
  Embeddings family—in the authors' [paper](https://arxiv.org/html/2608.23806#S4).
- **Our aggregate Q8 check:** NDCG@10 change `−0.00046` versus native MLX BF16.
- **Code-search check:** NDCG@10 change `−0.01297`; top-1 agreement 92.19%.
- **Q8 savings:** download 20.963 → 11.144 GB (**47% smaller**); peak Metal
  memory 23.853 → 14.423 GB (**40% lower**) versus native MLX BF16.

The official MTEB result belongs to the original BF16 model. We did not rerun
the complete MTEB suite on Q8; our local test separately checks whether the MLX
port and quantization preserve retrieval behavior.

## Choose a released MLX model

![Choose a Giga Embeddings 0826 MLX model](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=2026-08-26-2)

| Model | Best for | Q8 download | Original Russian MTEB | Our Q8 retrieval check |
|---|---|---:|---:|---:|
| [3B MLX Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8) | recommended default | 3.755 GB | 74.56 | NDCG@10 Δ +0.00181 |
| [480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8) | smallest and fastest | 0.525 GB | 70.98 | NDCG@10 Δ +0.00289 |
| **[10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8)** | **research; code warning** | **11.144 GB** | **74.98** | **aggregate Δ −0.00046** |

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
scores = queries @ documents.T
```

Queries require an explicit instruction; documents have no prefix. Normal
inference does not execute Python code from this model repository.

## What is inside

- Affine Q8 for eligible weights; BF16 MoE routers and normalization weights;
  group size 64.
- Embedding dimension: 1536; maximum sequence length: 8192.
- Artifact size: 11.144 GB; peak Metal memory: 14.423 GB for 16 long texts.
- Typical time for one 512-token text: 0.597 s on an M4 Pro.
- Batch speed: 0.76 documents/s for 16 texts of 1024 tokens each.
- Min/mean vector cosine versus native MLX BF16: 0.993838 / 0.999272.
- Top-1 agreement: 97.66%; mean top-10 overlap: 98.09%.
- RuSTS change: +0.000350; classification accuracy change: +0.000541. These
  small positive deltas are not improvement claims.

Full methodology, MoE-router checks, speed samples, and evidence hashes are in
the [`0826` MLX report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 reduced disk and peak Metal memory, but was slower than BF16 in the displayed
workloads.

## Source and reproducibility

- Original model: [`ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826).
- Original paper: [arXiv:2608.23806](https://arxiv.org/abs/2608.23806).
- Base revision: `1cb3ad3374dbf0eb9130546ca38b262de5f60287`.
- Weight release: `0826-v0.1.0`; tensor bytes are unchanged in the `0.1.2`
  documentation update.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` records the portable inventory and SHA-256 hashes.

Tested on Apple Silicon/macOS. This is an independent `ai-babai` quantization,
not an official `ai-sage` release.

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
