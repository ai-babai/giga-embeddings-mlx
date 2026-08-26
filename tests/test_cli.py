from __future__ import annotations

import pytest

from giga_embeddings_mlx.cli import build_parser


def test_encode_uses_balanced_default() -> None:
    args = build_parser().parse_args(["encode", "hello", "--document"])

    assert args.model == "default"
    assert args.texts == ["hello"]
    assert args.offline is False


def test_query_requires_an_explicit_instruction() -> None:
    args = build_parser().parse_args(["encode", "hello", "--instruction", "Find relevant passages"])

    assert args.instruction == "Find relevant passages"
    assert args.document is False


def test_public_cli_does_not_expose_checkpoint_conversion() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["convert"])
