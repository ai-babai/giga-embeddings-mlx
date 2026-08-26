# Contributing

Thank you for helping improve this independent Giga Embeddings MLX port.

## Development setup

Use Python 3.12 or 3.13 on an Apple Silicon Mac. Keep model weights, datasets,
virtual environments, Hugging Face caches, and raw benchmark outputs outside
the Git repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'
pytest
ruff format --check .
ruff check .
```

The production `giga-embeddings-mlx` CLI contains `models`, `encode`, and
`serve` only. Conversion, parity, quality evaluation, and benchmarks are
developer-facing. Checkpoint conversion is available through:

```bash
python -m giga_embeddings_mlx.dev_cli --help
```

Reference-backend or MTEB work uses the `[reference]` and `[evaluation]`
extras as needed.

## Pull requests

- Describe observable behavior and user impact.
- Add or update behavior tests for changed public behavior.
- Do not commit weights, datasets, caches, raw outputs, secrets, local paths,
  or private/user text.
- Preserve upstream attribution and immutable revisions.
- Keep query instructions explicit and documents unprefixed.
- If tensor bytes or execution semantics change, rerun strict load, backend
  parity, frozen holdout, downstream, resource, and speed gates.
- Do not broaden the public release matrix without new evidence and an explicit
  acceptance decision.

Security vulnerabilities should follow [`SECURITY.md`](SECURITY.md), not a
public issue.
