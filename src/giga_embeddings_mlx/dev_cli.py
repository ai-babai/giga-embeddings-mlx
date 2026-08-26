from __future__ import annotations

import argparse

from .conversion import convert_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m giga_embeddings_mlx.dev_cli",
        description="Developer-only checkpoint conversion utilities.",
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--bits", type=int, choices=(4, 6, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--recipe",
        choices=("mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"),
    )
    parser.add_argument("--policy", choices=("q8-edges-bf16",))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = convert_model(
        args.source,
        args.destination,
        bits=args.bits,
        group_size=args.group_size,
        recipe=args.recipe,
        policy=args.policy,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
