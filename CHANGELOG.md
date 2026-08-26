# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-26

### Changed

- Shortened the public Hugging Face repository names while retaining the
  weight-version, model-size, MLX, and Q8 identifiers.
- Updated runtime aliases, documentation, model cards, and release tooling to
  use the shorter canonical repository names. The previous Hugging Face URLs
  remain redirects.
- Simplified the public 3B label; its mixed Q8/BF16 policy remains documented
  in the model card and manifest instead of the repository name.

### Unchanged

- Runtime behavior, quantization policies, tensor bytes, and immutable
  model-weight revisions are unchanged from `0.1.0` and `0.1.1`.

## [0.1.1] - 2026-08-26

### Changed

- Reworked the GitHub and PyPI landing page around local semantic search, RAG,
  direct links to the released MLX models, and plain-language model selection.
- Added the original Giga-Embeddings paper and official MTEB quality results,
  clearly separated from the local MLX backend and Q8 preservation checks.
- Added a deterministic, accessible model-choice graphic and human-readable
  speed and memory labels.
- Updated all English and Russian Hugging Face model cards and narrowed the MLX
  Collection to the three released `ai-babai` artifacts.

### Unchanged

- Runtime behavior, model registry, quantization policies, tensor bytes, and
  immutable model-weight revisions are unchanged from `0.1.0`.

## [0.1.0] - 2026-08-26

### Added

- Native MLX inference for the exact Giga Embeddings `0826` 480M, 3B, and
  10B-A1.8B revisions.
- Full-sequence bidirectional Qwen3 and DeepSeek-V3 MoE execution without
  running checkpoint Python during normal inference.
- Three immutable upstream BF16 profiles and three accepted Q8 profiles, with
  `3b-q8` as the disclosed balanced default; every profile resolves by an
  exact verified Hugging Face commit SHA.
- Explicit query/document APIs, CLI encoding, local OpenAI-compatible
  `/v1/embeddings`, offline/cache support, and conservative memory preflight.
- Reproducible public benchmark JSON and tables for quality, artifact size,
  process RSS, Metal peak, load time, median/p95 latency, documents/s, and
  tokens/s.
- Python 3.12/3.13 macOS CI, package smoke checks, and PyPI Trusted Publishing
  workflow.

### Not released

- Q4, Q6, uniform 3B Q8, and dominated experimental policies.
- Automatic embedding-dimension truncation or output-vector compression.

[0.1.2]: https://github.com/ai-babai/giga-embeddings-mlx/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ai-babai/giga-embeddings-mlx/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ai-babai/giga-embeddings-mlx/releases/tag/v0.1.0
