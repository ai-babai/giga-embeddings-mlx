# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| older or unreleased snapshots | no |

## Reporting a vulnerability

Use GitHub private vulnerability reporting for
`ai-babai/giga-embeddings-mlx`. Do not open a public issue containing an
unpatched vulnerability, credentials, private text, model inputs, or local
filesystem data.

Include the affected version, Apple Silicon/macOS version, reproduction steps,
and impact. Reports are acknowledged as soon as practical; a disclosure
timeline is agreed after validation.

The project never requests Hugging Face or package-registry tokens in an issue.
Release workflows use GitHub OIDC for PyPI and narrowly scoped credentials for
Hugging Face.

Normal model loading uses the native runtime and does not execute Python code
from model repositories. Reports involving altered model files should include
the repository, immutable revision, and manifest hash without attaching
private weights.
