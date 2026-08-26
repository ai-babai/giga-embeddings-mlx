from __future__ import annotations

import argparse
import json

import mlx.core as mx

from ._version import __version__
from .loading import load_embedding_model
from .models import DEFAULT_PROFILE, MODEL_PROFILES


def _models_command() -> int:
    rows = [
        {
            "profile": profile.name,
            "repo_id": profile.repo_id,
            "revision": profile.revision,
            "architecture": profile.architecture,
            "embedding_dimension": profile.embedding_dimension,
            "precision": profile.precision,
            "release_role": profile.release_role,
            "default": profile.name == DEFAULT_PROFILE,
            "source_release": profile.source_release,
        }
        for profile in MODEL_PROFILES.values()
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="giga-embeddings-mlx")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="List pinned model profiles")
    encode = subparsers.add_parser("encode", help="Encode one or more texts")
    encode.add_argument("texts", nargs="+")
    _add_model_source_arguments(encode)
    role = encode.add_mutually_exclusive_group(required=True)
    role.add_argument(
        "--document",
        action="store_true",
        help="Encode input as documents without an instruction prefix",
    )
    role.add_argument(
        "--instruction",
        help="Encode input as queries using this explicit retrieval instruction",
    )
    encode.add_argument("--max-length", type=int, default=8192)
    serve = subparsers.add_parser(
        "serve", help="Serve a local OpenAI-compatible embeddings endpoint"
    )
    _add_model_source_arguments(serve)
    serve.add_argument("--served-model-name")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _add_model_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="default")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only artifacts already present in the Hugging Face cache",
    )
    parser.add_argument(
        "--skip-memory-check",
        action="store_true",
        help="Bypass the conservative unified-memory safety check",
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "models":
        return _models_command()
    if args.command == "encode":
        model = load_embedding_model(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
            skip_memory_check=args.skip_memory_check,
        )
        if args.instruction:
            embeddings = model.encode_queries(
                args.texts,
                instruction=args.instruction,
                max_length=args.max_length,
            )
        else:
            embeddings = model.encode_documents(args.texts, max_length=args.max_length)
        print(json.dumps(embeddings.astype(mx.float32).tolist(), ensure_ascii=False))
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "Server dependencies are missing; install with "
                "`pip install 'giga-embeddings-mlx[server]'`."
            ) from exc
        from .server import create_app

        uvicorn.run(
            create_app(
                args.model,
                served_model_name=args.served_model_name,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_files_only=args.offline,
                skip_memory_check=args.skip_memory_check,
            ),
            host=args.host,
            port=args.port,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
