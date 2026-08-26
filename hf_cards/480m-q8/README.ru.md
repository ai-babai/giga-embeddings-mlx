---
license: mit
language:
  - ru
  - en
pipeline_tag: feature-extraction
library_name: giga-embeddings-mlx
base_model: ai-sage/Giga-Embeddings-instruct-480M-0826
base_model_relation: quantized
tags:
  - mlx
  - embeddings
  - apple-silicon
  - q8
inference: false
---

# Giga Embeddings 0826 480M — MLX Q8 g64

[English card](README.md)

[Коллекция Giga Embeddings 0826 MLX](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244)

Компактная native-MLX квантизация
[`ai-sage/Giga-Embeddings-instruct-480M-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826)
для Apple Silicon. Это независимый артефакт `ai-babai`, а не официальный
release `ai-sage`. Обычный inference использует
[`giga-embeddings-mlx`](https://github.com/ai-babai/giga-embeddings-mlx) и не
исполняет Python-код checkpoint.

## Использование

```bash
python -m pip install giga-embeddings-mlx
```

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("480m-q8")
documents = model.encode_documents(["Москва — столица России."])
queries = model.encode_queries(
    ["Где находится Москва?"],
    instruction="Given a question, retrieve passages that answer the question",
)
```

Для запросов нужна явная инструкция. Документы не имеют префикса. Используются
padding-aware FP32 mean pooling и L2-нормализация в FP32.

## Артефакт и измерения

- Квантизация: напрямую из BF16, affine Q8, group size 64.
- Роль: compact; не default.
- Размерность: 1024; максимальная длина: 8192.
- Артефакт: 0,525 GB; Metal peak на B16×1024: 1,339 GB.
- M4 Pro 48 GB, B1×512: 0,071 s median / 0,072 s p95.
- M4 Pro 48 GB, B16×1024: 6,38 documents/s и 6530 tokens/s.
- Frozen holdout против native MLX BF16: min/mean cosine
  0,992867/0,998627, top-1 agreement 98,44%, mean top-10 overlap 96,72%,
  NDCG@10 delta +0,00289.

Малые положительные delta означают отсутствие измеренной деградации в этом
lane, а не улучшение модели от квантизации. Методика, p95, downstream,
хэши источников и исторические failures revision 1/2 приведены в
[`0826` report](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).

## Provenance

- Base revision: `2d0c1a92716eef0e5b6972df85b5883eb5b4f57a`.
- Release tag: `0826-v0.1.0`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` содержит portable inventory и SHA-256.
- Python-файлы model repository и `auto_map` намеренно исключены.

Проверено на Apple Silicon/macOS. Q8 — прежде всего компромисс по диску и
ёмкости, а не гарантированное ускорение относительно BF16.
