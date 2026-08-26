---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - moe
  - q8
  - research
inference: false
---

# Giga Embeddings 0826 10B-A1.8B — MLX Q8 g64

[English card](README.md)

[Коллекция Giga Embeddings 0826 MLX](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244)

> **Compact/research артефакт с предупреждением по code retrieval.** Aggregate
> retrieval прошёл, но code-family NDCG@10 изменился на `−0,01297` относительно
> native MLX BF16. Это не default и не near-lossless claim.

Native-MLX квантизация
[`ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826)
для Apple Silicon. MoE routers и normalization weights сохранены в BF16. Это
независимый артефакт `ai-babai`, а не официальный release `ai-sage`.

## Использование

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("10b-a1.8b-q8")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Для запросов нужна явная инструкция; документы не имеют префикса. Обычный
inference использует
[`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx) и не
исполняет Python-код checkpoint.

## Артефакт и измерения

- Квантизация: напрямую из BF16, affine Q8 для eligible weights, BF16 routers и
  norms, group size 64.
- Роль: compact / research.
- Размерность: 1536; максимальная длина: 8192.
- Артефакт: 11,144 GB; Metal peak на B16×1024: 14,423 GB.
- M4 Pro 48 GB, B1×512: 0,597 s median / 0,674 s p95.
- M4 Pro 48 GB, B16×1024: 0,76 documents/s и 776 tokens/s.
- Frozen holdout против native MLX BF16: min/mean cosine
  0,993838/0,999272, top-1 agreement 97,66%, mean top-10 overlap 98,09%,
  aggregate NDCG@10 delta −0,00046.
- Code-family top-1 agreement: 92,19%; code NDCG@10 delta: −0,01297.
- Downstream delta: RuSTS +0,000350; classification accuracy +0,000541 и
  macro-F1 +0,000540. Положительные delta не считаются улучшением.

Методика, MoE-router gates, speed samples и source hashes приведены в
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 уменьшил диск и Metal peak, но был медленнее BF16 на показанных workload.

## Provenance

- Base revision: `1cb3ad3374dbf0eb9130546ca38b262de5f60287`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` содержит portable inventory и SHA-256.
- Python-файлы model repository и `auto_map` намеренно исключены.
