from __future__ import annotations

import argparse
import json

import mlx.core as mx

from .conversion import convert_model
from .loading import load_embedding_model
from .models import MODEL_PROFILES


def _models_command() -> int:
    rows = [
        {
            "profile": profile.name,
            "repo_id": profile.repo_id,
            "revision": profile.revision,
            "architecture": profile.architecture,
            "embedding_dimension": profile.embedding_dimension,
            "source_release": profile.source_release,
        }
        for profile in MODEL_PROFILES.values()
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="giga-embeddings-mlx")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="List pinned model profiles")
    encode = subparsers.add_parser("encode", help="Encode one or more texts")
    encode.add_argument("model")
    encode.add_argument("texts", nargs="+")
    encode.add_argument("--max-length", type=int, default=8192)
    convert = subparsers.add_parser("convert", help="Create a native MLX checkpoint")
    convert.add_argument("source")
    convert.add_argument("destination")
    convert.add_argument("--bits", type=int, choices=(4, 6, 8))
    convert.add_argument("--group-size", type=int, default=64)
    convert.add_argument(
        "--recipe",
        choices=("mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"),
    )
    convert.add_argument("--policy", choices=("q8-edges-bf16",))
    serve = subparsers.add_parser(
        "serve", help="Serve a local OpenAI-compatible embeddings endpoint"
    )
    serve.add_argument("model")
    serve.add_argument("--served-model-name")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "models":
        return _models_command()
    if args.command == "encode":
        model = load_embedding_model(args.model)
        embeddings = model.encode(args.texts, max_length=args.max_length)
        print(json.dumps(embeddings.astype(mx.float32).tolist(), ensure_ascii=False))
        return 0
    if args.command == "convert":
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
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                'Server dependencies are missing; install with `uv sync --extra server`.'
            ) from exc
        from .server import create_app

        uvicorn.run(
            create_app(args.model, served_model_name=args.served_model_name),
            host=args.host,
            port=args.port,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
