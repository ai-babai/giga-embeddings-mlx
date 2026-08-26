# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-26

### Added

- Native MLX inference for the exact Giga Embeddings `0826` 480M, 3B, and
  10B-A1.8B revisions.
- Full-sequence bidirectional Qwen3 and DeepSeek-V3 MoE execution without
  running checkpoint Python during normal inference.
- Three immutable upstream BF16 profiles and three accepted Q8 profiles, with
  `3b-q8` as the disclosed balanced default.
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

[0.1.0]: https://github.com/ai-babai/giga-embeddings-mlx/releases/tag/v0.1.0
