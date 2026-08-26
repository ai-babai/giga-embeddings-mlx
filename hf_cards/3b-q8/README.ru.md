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
  - apple-silicon
  - embeddings
  - text-embeddings
  - semantic-search
  - rag
  - russian
  - local-ai
  - macos
  - quantized
  - q8
  - arxiv:2608.23806
inference: false
---

# Giga Embeddings 0826 3B — рекомендуемая MLX Q8 для Apple Silicon

[English card](README.md) ·
[Все MLX-модели](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-for-apple-silicon-mlx-q8-6a8eec40b26f6543f5da3244) ·
[GitHub](https://github.com/ai-babai/giga-embeddings-mlx) ·
[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Оригинальная статья](https://arxiv.org/abs/2608.23806)

![Какую модель Giga Embeddings 0826 MLX выбрать](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=0.1.2)

Рекомендуемая модель по умолчанию для локального семантического поиска, RAG,
сравнения текстов, кластеризации и классификации на русском и английском языках.
В этом смешанном Q8/BF16-артефакте embedding и финальные edge layers сохранены
в BF16, а подходящие внутренние веса квантизованы до Q8.

## Качество и размер

- **Качество исходной модели:** 74,56 на русском MTEB в
  [статье авторов](https://arxiv.org/html/2608.23806#S4).
- **Наша проверка Q8:** совокупная NDCG@10 изменилась на `+0,00181` относительно
  нативной MLX BF16 — ухудшение на замороженном наборе не обнаружено.
- **Размер загрузки:** 3,755 GB; **пиковая Metal-память:** 5,137 GB в тесте на
  16 текстах.
- **Типичное время одного текста на 512 токенов:** 0,637 с на M4 Pro.

Официальный MTEB относится к исходной BF16-модели. Мы не перезапускали полный
MTEB на Q8: наша отдельная проверка измеряет сохранение поискового поведения при
переносе на MLX и квантизации.

## Какую выпущенную MLX-модель выбрать

| Модель | Для чего подходит | Размер | Русский MTEB исходной модели | Наша проверка Q8 |
|---|---|---:|---:|---:|
| **[3B MLX Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8)** | **рекомендуемая по умолчанию** | **3,755 GB** | **74,56** | **NDCG@10 Δ +0,00181** |
| [480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8) | самая компактная и быстрая | 0,525 GB | 70,98 | NDCG@10 Δ +0,00289 |
| [10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8) | research; предупреждение по коду | 11,144 GB | 74,98 | совокупная Δ −0,00046 |

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
scores = queries @ documents.T
```

Для запросов нужна явная инструкция; документы кодируются без префикса. При
обычном инференсе Python-код из этого model repository не исполняется.

## Что находится внутри

- Affine Q8 для внутренних весов, BF16 embedding и финальные edge layers,
  group size 64.
- Размерность: 2048; максимальная длина: 8192.
- Размер артефакта: 3,755 GB; пиковая Metal-память: 5,137 GB для 16 длинных
  текстов.
- Скорость пакета: 0,73 документа/с для 16 текстов по 1024 токена.
- Min/mean cosine относительно нативного MLX BF16: 0,992840 / 0,999575.
- Top-1 agreement: 99,61%; среднее top-10 overlap: 98,52%.
- Изменение RuSTS: −0,000127; изменение classification accuracy: −0,000541.

Методика, speed samples, backend gates и evidence hashes приведены в
[`0826` MLX-отчёте](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 уменьшила место на диске и пиковую Metal-память, но была медленнее BF16 в
показанных нагрузках.

## Источник и воспроизводимость

- Исходная модель: [`ai-sage/Giga-Embeddings-instruct-3B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826).
- Оригинальная статья: [arXiv:2608.23806](https://arxiv.org/abs/2608.23806).
- Base revision: `ed7db5c91b900b39381b27b6e9c0a3d31137cd29`.
- Релиз весов: `0826-v0.1.0`; tensor bytes не менялись в документационном
  обновлении `0.1.2`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` содержит portable inventory и SHA-256.

Проверено на Apple Silicon/macOS. Это независимая квантизация `ai-babai`, а не
официальный релиз `ai-sage`.
