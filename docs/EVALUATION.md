# Evaluation and benchmarking

The scripts in `scripts/` are the reproducible part of the `0826-v1` local
evaluation. Model weights, datasets, embedding caches, and JSON outputs must
live outside this Git repository.

Install all optional dependencies used by the evaluation lane:

```bash
uv sync --frozen --extra reference --extra evaluation --extra server
```

The pre-registered run compares each quantized checkpoint directly with its
MLX BF16 source. It does not cascade lossy conversions. Query prompts have the
exact form `Instruct: ...\nQuery: ...`; documents have no prefix. Mean pooling
excludes padding and normalization is performed in FP32.

## Structural and cross-backend parity

```bash
uv run python scripts/structural_check.py 480m "$SOURCE_480M" \
  --expected-dimension 1024 --output "$RUN/structural-480m.json"

uv run python scripts/evaluate_parity.py 480m "$SOURCE_480M" \
  2d0c1a92716eef0e5b6972df85b5883eb5b4f57a \
  --dtype bfloat16 --output "$RUN/parity-480m-bfloat16.json"
```

`float32` is a diagnostic graph-control lane. It must never be substituted for
the pre-registered BF16 acceptance result.

## Quant screening and frozen holdout

```bash
uv run python scripts/quant_quality.py 3b "$SOURCE_3B" \
  "$MLX/3b-q8-g64" "$MLX/3b-q6-g64" "$MLX/3b-q4-g64" \
  --batch-size 8 --output "$RUN/quant-quality-3b.json"

uv run python scripts/evaluate_holdout.py 3b "$SOURCE_3B" \
  "$MLX/3b-q8-g64" "$MLX/3b-q6-g64" \
  --holdout "$HOLDOUT" --cache-dir "$RUN/embeddings/3b" \
  --batch-size 8 --aligned-batch-size 1 \
  --output "$RUN/holdout-3b.json"
```

The full holdout has 512 aligned texts, 256 queries, and 2048 documents, with
RU, EN, code, and multilingual families and four length buckets. The
incremental embedding cache allows a safety-stopped run to resume by section.

## Speed and memory

```bash
uv run python scripts/benchmark_speed.py 3b-bf16 "$SOURCE_3B" \
  --warmups 2 --repetitions 5 --output "$RUN/speed-3b-bf16.json"

uv run python scripts/benchmark_load.py 3b-bf16 "$SOURCE_3B" \
  --output "$RUN/load-3b-bf16.json"
```

The fixed matrix includes batch 1 at lengths 128/512/1024/2048, batch 8 and 16
at 512/1024, and a separately marked batch-1 4096-token probe. The report
records cold load/inference, warm median and p95 latency, documents/s,
tokens/s, peak Metal allocation, process RSS, memory pressure, and swap state.
Runs are sequential to avoid mixing models in unified memory.
The load probe materializes all parameters twice and explicitly records that
the macOS page cache was not flushed, so its first load is process-cold rather
than a synthetic machine-cold claim.

## MoE routers and output-vector compression

For the 10B profile, capture the BF16 and candidate router traces in separate
processes before comparing them:

```bash
uv run python scripts/router_trace.py trace "$SOURCE_10B" \
  --calibration "$CALIBRATION" --batch-size 1 \
  --output "$RUN/router-10b-bf16.npz"

uv run python scripts/router_trace.py compare \
  "$RUN/router-10b-bf16.npz" "$RUN/router-10b-q8.npz" \
  --variant 10b-q8-g64 --output "$RUN/router-10b-q8.json"
```

`evaluate_vector_compression.py` uses a disjoint calibration corpus for uint8
ranges and reports binary search both directly and with oversampling plus FP32
rescoring. Those results describe index storage/search, not weight
quantization.

## Interpretation

An artifact is recommended only when it passes the representation, ranking,
downstream, and resource gates registered for its role. Safety stops, failed
thresholds, and slower quantized kernels remain visible results. Upstream H100
numbers are not mixed with local Apple M4 Pro measurements.
