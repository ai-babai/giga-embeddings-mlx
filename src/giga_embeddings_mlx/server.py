import base64
import struct
import threading
import time
from typing import Literal

import mlx.core as mx

from .loading import EmbeddingModel, load_embedding_model


def _float32_base64(values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}f", *values)
    return base64.b64encode(raw).decode("ascii")


def create_app(model_path: str, *, served_model_name: str | None = None):
    """Create a local OpenAI-compatible embeddings application.

    FastAPI is imported lazily so the core runtime does not require server
    dependencies. The model is loaded once when the application is created.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, ConfigDict
    except ImportError as exc:
        raise RuntimeError(
            'Server dependencies are missing; install with `uv sync --extra server`.'
        ) from exc

    loaded: EmbeddingModel = load_embedding_model(model_path)
    public_name = served_model_name or model_path
    inference_lock = threading.Lock()

    class EmbeddingsRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        input: str | list[str]
        model: str
        encoding_format: Literal["float", "base64"] = "float"
        dimensions: int | None = None
        user: str | None = None

    app = FastAPI(title="Giga Embeddings MLX", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": public_name}

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingsRequest) -> dict:
        if request.dimensions is not None:
            raise HTTPException(
                status_code=400,
                detail="Dimension truncation is not supported by this model family.",
            )
        if request.model != public_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model!r} is not served by this process.",
            )

        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")

        # One model owns one Metal execution path. Serializing requests avoids
        # interleaved allocator state and gives predictable local behavior.
        with inference_lock:
            tokenized = loaded.tokenizer(texts, add_special_tokens=True)
            prompt_tokens = sum(len(ids) for ids in tokenized["input_ids"])
            vectors = loaded.encode(texts).astype(mx.float32).tolist()
        if request.encoding_format == "base64":
            payload = [_float32_base64(vector) for vector in vectors]
        else:
            payload = vectors

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vector, "index": index}
                for index, vector in enumerate(payload)
            ],
            "model": public_name,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            "created": int(time.time()),
        }

    return app
