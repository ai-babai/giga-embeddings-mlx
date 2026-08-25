# Giga Embeddings MLX

[Русская версия](README.ru.md)

Local native-MLX runtime under active evaluation for the `0826` source line:

- `ai-sage/Giga-Embeddings-instruct-480M-0826`;
- `ai-sage/Giga-Embeddings-instruct-3B-0826`;
- `ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`.

The current work is governed by `GIGA-EMBEDDINGS-GOAL-001`. It is not a
published release and makes no quality or performance claims yet.

Heavy model weights, datasets, caches, and raw benchmark outputs live outside
this repository under:

```text
<ML_DATA_ROOT>/giga-embeddings/0826/
```

## Development status

The native bidirectional Qwen3 path is implemented for 480M and 3B, and the
DeepSeek-V3 bidirectional MoE path is implemented and measured for 10B.
Quantized artifacts remain experimental because the deliberately strict
revision-1 BF16 cross-backend gate failed despite passing FP32 graph controls.

The runtime uses full-sequence attention, no generation cache, mean pooling in
FP32 over non-padding tokens, and FP32 L2 normalization. It does not execute
checkpoint Python code for normal MLX inference.

## Local setup

```bash
uv sync --frozen
uv run giga-embeddings-mlx models
```

Pinned source profiles can be addressed by `480m`, `3b`, or `10b-a1.8b`.
Weights are resolved at the exact revisions recorded in `models.py` and stored
in the Hugging Face cache rather than in this Git repository.

```bash
uv run giga-embeddings-mlx encode 480m \
  "Москва — столица Российской Федерации." \
  "Париж — столица Франции."
```

For asymmetric retrieval, format only queries as:

```text
Instruct: Given a query, retrieve relevant passages
Query: Где столица России?
```

Documents are encoded without that prefix.

## Local OpenAI-compatible endpoint

Install the optional server dependencies and start one model per process:

```bash
uv sync --frozen --extra server
uv run giga-embeddings-mlx serve 480m --served-model-name giga-embeddings-480m-0826
```

The server binds to `127.0.0.1:8000` by default and implements
`POST /v1/embeddings` with `float` and `base64` encoding formats. Query
instructions are not guessed by the server: clients must pass the exact
`Instruct: ...\nQuery: ...` text for queries, while documents stay unprefixed.
It also exposes `GET /health`. This is a local reference interface, not a
production server or a published compatibility claim.

## Conversion

Uniform affine Q8/Q6/Q4 conversion uses group size 64 by default:

```bash
uv run giga-embeddings-mlx convert 3b /path/to/3b-q8-g64 --bits 8
```

MLX-LM mixed recipes can be screened explicitly:

```bash
uv run giga-embeddings-mlx convert 3b /path/to/3b-mixed4-6-g64 \
  --recipe mixed_4_6
```

For a quality-oriented Q8 candidate, keep token embeddings and the first/last
transformer blocks in BF16:

```bash
uv run giga-embeddings-mlx convert 3b /path/to/3b-q8-edges-bf16 \
  --policy q8-edges-bf16
```

An output directory must not already exist. Conversion never uploads or
publishes artifacts.

## Evaluation status

The frozen evaluation compares every quant directly with its MLX BF16 source
and keeps weight quantization separate from output-vector compression. Reports
cover aligned-vector drift, retrieval ranking, Russian semantic similarity and
classification, artifact size, load time, warm median/p95 latency, tokens/s,
documents/s, process RSS, peak Metal allocation, memory pressure, and swap.

The measured BF16 backend drift does not satisfy the deliberately strict
pre-registered cross-backend gate even though FP32 graph controls match. Full
holdout, downstream, MoE router, speed/load, and output-vector compression
lanes are complete. No artifact should be described as accepted or released
until a new criteria revision is decided explicitly.

Reproducible commands and the fixed speed matrix are documented in
[`docs/EVALUATION.md`](docs/EVALUATION.md).
