---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-3B-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - q8
inference: false
---

# Giga Embeddings 0826 3B — MLX Q8 + BF16 edges g64

[English card](README.md)

[Коллекция Giga Embeddings 0826 MLX](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244)

Сбалансированная native-MLX квантизация
[`ai-sage/Giga-Embeddings-instruct-3B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826)
для Apple Silicon. Embedding и финальные edge layers сохранены в BF16, eligible
inner weights используют Q8. Это раскрытый профиль `default` в
[`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx), а не
официальный release `ai-sage`.

## Использование

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("default")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Для запросов нужна явная инструкция; документы не имеют префикса. Обычный
inference не исполняет Python-код checkpoint.

## Артефакт и измерения

- Квантизация: напрямую из BF16, affine Q8 для inner weights, BF16 edge layers,
  group size 64.
- Роль: balanced default.
- Размерность: 2048; максимальная длина: 8192.
- Артефакт: 3,755 GB; Metal peak на B16×1024: 5,137 GB.
- M4 Pro 48 GB, B1×512: 0,637 s median / 0,659 s p95.
- M4 Pro 48 GB, B16×1024: 0,73 documents/s и 744 tokens/s.
- Frozen holdout против native MLX BF16: min/mean cosine
  0,992840/0,999575, top-1 agreement 99,61%, mean top-10 overlap 98,52%,
  NDCG@10 delta +0,00181.
- Downstream delta: RuSTS −0,000127; classification accuracy −0,000541 и
  macro-F1 −0,000540.

Методика, speed samples, backend gates и evidence hashes приведены в
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 уменьшил диск и Metal peak, но был медленнее BF16 на показанных workload.

## Provenance

- Base revision: `ed7db5c91b900b39381b27b6e9c0a3d31137cd29`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` содержит portable inventory и SHA-256.
- Python-файлы model repository и `auto_map` намеренно исключены.
