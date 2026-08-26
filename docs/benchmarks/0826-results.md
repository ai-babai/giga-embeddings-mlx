# Giga Embeddings 0826 MLX — measured results

This file is generated from the Goal 001 JSON evidence by `tools/build_release_benchmarks.py`. Do not edit measured values by hand.

## Measurement context

- Machine: MacBook Pro, Apple M4 Pro, 48 GB unified memory.
- Runtime: Python 3.12.11, MLX 0.32.2, MLX-LM 0.31.3.
- Speed: 2 warmups, 5 measured repetitions; median and p95 are reported.
- Load: first process-materialized and warm reload; OS page cache was not flushed.
- Lower is better for seconds/bytes; higher is better for docs/s, tok/s and quality.

## Disk, memory, load and speed

| Alias | Role | Artifact (GB) ↓ | Process max RSS (GB) ↓ | Metal peak B16×1024 (GB) ↓ | Load first / warm (s) ↓ | B1×512 median / p95 (s) ↓ | B16×1024 docs/s ↑ | B16×1024 tok/s ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `480m-bf16` | upstream baseline | 0.978 | 1.647 | 1.792 | 0.327 / 0.322 | 0.069 / 0.070 | 7.46 | 7641 |
| `480m-q8` | compact | 0.525 | 1.206 | 1.339 | 0.378 / 0.312 | 0.071 / 0.072 | 6.38 | 6530 |
| `3b-bf16` | upstream baseline | 6.312 | 6.987 | 7.694 | 0.492 / 0.485 | 0.464 / 0.465 | 0.89 | 915 |
| `3b-q8` | balanced default | 3.755 | 4.426 | 5.137 | 0.875 / 0.438 | 0.637 / 0.659 | 0.73 | 744 |
| `10b-a1.8b-bf16` | upstream baseline | 20.963 | 3.654 | 23.853 | 2.697 / 2.741 | 0.356 / 0.379 | 0.86 | 886 |
| `10b-a1.8b-q8` | compact / research | 11.144 | 11.815 | 14.423 | 1.998 / 0.626 | 0.597 / 0.674 | 0.76 | 776 |

Artifact size, process RSS and Metal peak are distinct measurements and must not be added.
On this M4 Pro, Q8 is a capacity/disk optimization, not a guaranteed speed-up.

## BF16 backend preservation

PyTorch BF16 reference versus native MLX BF16. Deltas are MLX minus reference; the acceptance margin for aggregate and each family was −0.005.

| Profile | Min pooled cosine ↑ | MLX single/padded cosine ↑ | MRR Δ ↑ | NDCG@10 Δ ↑ | Worst family MRR Δ ↑ | Worst family NDCG@10 Δ ↑ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `480m` | 0.999459 | 0.999880 | +0.007427 | +0.006695 | +0.000000 | +0.000000 | pass |
| `3b` | 0.999740 | 0.999909 | -0.003587 | -0.002735 | -0.000024 | +0.000000 | pass |
| `10b-a1.8b` | 0.998783 | 0.999936 | -0.002442 | -0.001905 | -0.001302 | -0.001083 | pass |

Revision 1/2 stricter numerical diagnostics failed and remain part of the historical record. Revision 3 accepts observable retrieval effectiveness while retaining rank and hidden-state drift as diagnostics.

## Quantized weight preservation

All deltas are against the corresponding native MLX BF16 artifact.

| Alias | Min / mean cosine ↑ | Spearman ↑ | Top-1 / top-10 agreement ↑ | NDCG@10 Δ ↑ | Worst family NDCG@10 Δ ↑ | RuSTS Δ ↑ | Classification acc. / macro-F1 Δ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `480m-q8` | 0.992867 / 0.998627 | 0.999087 | 98.44% / 96.72% | +0.00289 | +0.00000 | +0.000662 | +0.001081 / +0.001082 |
| `3b-q8` | 0.992840 / 0.999575 | 0.999777 | 99.61% / 98.52% | +0.00181 | -0.00048 | -0.000127 | -0.000541 / -0.000540 |
| `10b-a1.8b-q8` | 0.993838 / 0.999272 | 0.999599 | 97.66% / 98.09% | -0.00046 | -0.01297 | +0.000350 | +0.000541 / +0.000540 |

The 10B Q8 code-family NDCG@10 delta is −0.01297. It is therefore a compact/research artifact, not the default and not a near-lossless claim.

## Reproducibility

Acceptance JSON SHA-256: `410b9cf7756e0718816b23a46f0d99e0f3e6574e4eb515cc5a99cff131057316`.
The companion `0826-results.json` includes SHA-256 for every source evidence file used here.
Raw datasets, model weights and benchmark outputs are intentionally not stored in this Git repository.
