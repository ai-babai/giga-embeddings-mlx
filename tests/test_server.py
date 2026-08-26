from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from giga_embeddings_mlx import server


@dataclass
class FakeEmbeddingModel:
    encoded_batches: list[list[str]] = field(default_factory=list)

    def tokenizer(self, texts: list[str], *, add_special_tokens: bool) -> dict:
        assert add_special_tokens is True
        return {"input_ids": [[1, 2] for _ in texts]}

    def encode(self, texts: list[str]) -> mx.array:
        self.encoded_batches.append(texts)
        return mx.array([[1.0, 0.0] for _ in texts])


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeEmbeddingModel]:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(server, "load_embedding_model", lambda *args, **kwargs: model)
    app = server.create_app("default", served_model_name="giga-3b")
    return TestClient(app), model


def test_embedding_endpoint_treats_unadorned_input_as_documents(
    app_client: tuple[TestClient, FakeEmbeddingModel],
) -> None:
    client, model = app_client

    response = client.post(
        "/v1/embeddings",
        json={"model": "giga-3b", "input": ["document"]},
    )

    assert response.status_code == 200
    assert model.encoded_batches == [["document"]]
    assert response.json()["usage"] == {"prompt_tokens": 2, "total_tokens": 2}


def test_embedding_endpoint_applies_explicit_query_instruction(
    app_client: tuple[TestClient, FakeEmbeddingModel],
) -> None:
    client, model = app_client

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "giga-3b",
            "input": "query",
            "instruction": "Find relevant passages",
        },
    )

    assert response.status_code == 200
    assert model.encoded_batches == [["Instruct: Find relevant passages\nQuery: query"]]


def test_embedding_endpoint_rejects_dimension_truncation(
    app_client: tuple[TestClient, FakeEmbeddingModel],
) -> None:
    client, _ = app_client

    response = client.post(
        "/v1/embeddings",
        json={"model": "giga-3b", "input": "document", "dimensions": 512},
    )

    assert response.status_code == 400
