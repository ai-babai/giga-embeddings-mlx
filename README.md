# Giga Embeddings MLX

[Русская версия](README.ru.md)

Independent native [MLX](https://github.com/ml-explore/mlx) runtime for the
`0826` Giga Embeddings model family on Apple Silicon. This project is maintained
by [ai-babai](https://github.com/ai-babai); it is not an official `ai-sage`
release.

[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Hugging Face Collection](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244) ·
[Benchmarks](docs/benchmarks/0826-results.md) ·
[Changelog](CHANGELOG.md)

The runtime supports the exact upstream
[480M](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826),
[3B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826), and
[10B-A1.8B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826)
revisions, plus three accepted MLX Q8 artifacts. It performs full-sequence
bidirectional attention, padding-aware mean pooling in FP32, and FP32 L2
normalization. Normal inference does not execute Python code from model
repositories.

## Install

Requirements: an Apple Silicon Mac, macOS, and Python 3.12 or 3.13.

```bash
python -m pip install giga-embeddings-mlx
```

The default artifact is `3b-q8`, the balanced 3B Q8 model with BF16 edge
layers. Its download is approximately 3.8 GB. Model files are downloaded from
Hugging Face on first use.

## 60-second quick start

Documents are encoded without a prefix. Queries require an explicit retrieval
instruction; the runtime never invents one.

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("default")

documents = model.encode_documents(["Москва — столица России.", "Париж — столица Франции."])
queries = model.encode_queries(
    "Где находится Москва?",
    instruction="Given a question, retrieve passages that answer the question",
)

scores = queries @ documents.T
print(scores.tolist())
```

The low-level `model.encode(...)` method accepts already prepared text. Prefer
`encode_queries` and `encode_documents` when the retrieval role matters.

CLI:

```bash
giga-embeddings-mlx models
giga-embeddings-mlx encode "Москва — столица России." --document
giga-embeddings-mlx encode "Где находится Москва?" \
  --instruction "Given a question, retrieve passages that answer the question"
```

## Choose a profile

| Alias | Weights | Dimension | Role | Expected download |
|---|---|---:|---|---:|
| `480m-bf16` | upstream BF16 | 1024 | smallest quality baseline | 1.0 GB |
| `480m-q8` | Q8, group 64 | 1024 | compact | 0.5 GB |
| `3b-bf16` | upstream BF16 | 2048 | 3B quality baseline | 6.3 GB |
| `3b-q8` / `default` | Q8 + BF16 edges, group 64 | 2048 | balanced default | 3.8 GB |
| `10b-a1.8b-bf16` | upstream BF16 MoE | 1536 | quality-first, high capacity | 21.0 GB |
| `10b-a1.8b-q8` | Q8, BF16 routers/norms, group 64 | 1536 | compact / research | 11.1 GB |

Quant repositories:
[480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8-g64),
[3B balanced](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64),
and
[10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64).

The BF16 aliases point to immutable upstream commits; weights are not
duplicated under `ai-babai`. Q8 aliases point to the immutable `0826-v0.1.0`
release tag in their dedicated repositories.

The 10B Q8 artifact is not the default and is not described as near-lossless:
its aggregate retrieval gate passed, but the measured code-family NDCG@10
delta was `−0.01297` versus native MLX BF16.

Q4, Q6, uniform 3B Q8, and dominated mixed variants are intentionally not
released.

## Explicit Hub repositories and local artifacts

Aliases, an explicit Hub repository/revision, and a local directory share the
same loader:

```python
from pathlib import Path

from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("480m-q8")

model = load_embedding_model(
    "owner/repository",
    revision="immutable-commit-or-tag",
    cache_dir=Path("models-cache"),
)

model = load_embedding_model(Path("portable-model-directory"))
```

## Cache and offline use

Use a dedicated cache directory when lifecycle control matters:

```python
model = load_embedding_model("3b-q8", cache_dir="models-cache")
model = load_embedding_model(
    "3b-q8",
    cache_dir="models-cache",
    local_files_only=True,
)
```

CLI equivalents are `--cache-dir models-cache` and `--offline`. If the pinned
snapshot is absent, offline mode reports the repository and tells you how to
populate or select a cache.

Only remove a cache directory that you deliberately dedicated to this project,
after confirming that no other Hugging Face application uses it. Do not delete
the global Hugging Face cache merely to remove one model snapshot.

Before loading, the runtime compares artifact bytes with physical unified
memory using a conservative reserve. Choose a smaller profile if the preflight
fails. `skip_memory_check=True` / `--skip-memory-check` is an explicit escape
hatch for users willing to accept swap or out-of-memory risk.

## Local OpenAI-compatible endpoint

Install the optional server dependencies:

```bash
python -m pip install 'giga-embeddings-mlx[server]'
giga-embeddings-mlx serve --model default --served-model-name giga-3b
```

The endpoint is `POST /v1/embeddings`. Input without `instruction` is treated
as a document. A query must provide its instruction explicitly:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "giga-3b",
    "input": ["Где находится Москва?"],
    "instruction": "Given a question, retrieve passages that answer the question"
  }'
```

The server supports `float` and `base64` output. Dimension truncation is not
supported because the upstream `0826` cards do not claim Matryoshka training.
One process serves one model and serializes Metal inference.

## Measured results

Measurements were made on a MacBook Pro with Apple M4 Pro and 48 GB unified
memory, Python 3.12.11, MLX 0.32.2, and MLX-LM 0.31.3. Speed uses 2 warmups and
5 measured repetitions. These numbers do not predict other Macs or compare
directly with upstream H100 results.

Q8 reduced artifact size by about 40–47% for the released choices, but was
12–19% slower than its BF16 counterpart at B16×1024. Treat weight quantization
here primarily as a capacity and disk trade-off.

The full generated report separates artifact size, process RSS, Metal peak,
load time, median/p95 speed, BF16 backend preservation, quantized-vector drift,
ranking, and downstream deltas:

- [human-readable tables](docs/benchmarks/0826-results.md);
- [machine-readable evidence and source hashes](docs/benchmarks/0826-results.json).

The acceptance JSON SHA-256 is
`410b9cf7756e0718816b23a46f0d99e0f3e6574e4eb515cc5a99cff131057316`.
The 512-text / 256-query / 2048-document frozen holdout covers Russian,
English, code, and multilingual families up to 2048 tokens. Small positive
deltas are evidence of no measured regression in this lane, not proof that
quantization improves the model.

## Quality interpretation

The original strict revision 1/2 numerical gates exposed BF16 cross-backend
and dynamic-shape drift and did not pass. Those failures remain documented.
Revision 3 passed an effectiveness-based gate for pooled vectors, padding,
aggregate retrieval and every family with a `−0.005` MRR/NDCG margin, while
retaining rank agreement and hidden-state drift as diagnostics.

This distinction matters: backend-level hidden vectors can drift while the
observable retrieval task remains non-inferior. The public Q8 selection was
therefore based on representation, ranking, downstream quality, disk, memory,
load and speed—not one cosine number.

## Limitations

- Apple Silicon/macOS only; use the upstream reference runtime elsewhere.
- Maximum sequence length is 8192, but memory grows with batch and sequence
  length; the public speed matrix is not a promise for every workload.
- Q8 is not guaranteed to be faster than BF16 on Metal.
- 10B Q8 has the explicit code-retrieval warning above.
- This package does not silently truncate embedding dimensions.
- Output-vector `uint8`/binary compression is a separate index-storage choice
  and is not performed by the weight loader.
- The local benchmark is not an official upstream leaderboard result.

## Development and citation

Conversion and evaluation remain developer-facing and are not exposed by the
end-user CLI. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/EVALUATION.md](docs/EVALUATION.md). Security reports follow
[SECURITY.md](SECURITY.md).

License: MIT for this independent runtime. Upstream model licenses and notices
remain attached to their respective repositories; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is in
[CITATION.cff](CITATION.cff).
