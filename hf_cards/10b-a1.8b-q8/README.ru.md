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
  - apple-silicon
  - embeddings
  - text-embeddings
  - semantic-search
  - rag
  - russian
  - local-ai
  - macos
  - quantized
  - moe
  - q8
  - research
  - arxiv:2608.23806
inference: false
---

# Giga Embeddings 0826 10B-A1.8B — исследовательская MLX Q8

[English card](README.md) ·
[Все MLX-модели](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244) ·
[GitHub](https://github.com/ai-babai/giga-embeddings-mlx) ·
[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Оригинальная статья](https://arxiv.org/abs/2608.23806)

![Какую модель Giga Embeddings 0826 MLX выбрать](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=0.1.1)

> **Исследовательская модель с предупреждением по поиску в коде.** Совокупное
> качество поиска прошло порог приёмки, но NDCG@10 для кода изменился на
> `−0,01297` относительно нативной MLX BF16. По умолчанию используйте 3B, пока
> не проверите этот компромисс на собственных данных.

Высокоёмкий sparse Mixture-of-Experts вариант для локального семантического
поиска, RAG, сравнения текстов, кластеризации и классификации на русском и
английском языках. В Q8-артефакте MoE routers и normalization weights сохранены
в BF16.

## Качество и размер

- **Качество исходной модели:** 74,98 на русском MTEB — лучший результат
  семейства Giga Embeddings в [статье авторов](https://arxiv.org/html/2608.23806#S4).
- **Наша совокупная проверка Q8:** NDCG@10 изменилась на `−0,00046` относительно
  нативной MLX BF16.
- **Поиск в коде:** NDCG@10 изменилась на `−0,01297`; top-1 agreement 92,19%.
- **Размер загрузки:** 11,144 GB; **пиковая Metal-память:** 14,423 GB в тесте на
  16 текстах.

Официальный MTEB относится к исходной BF16-модели. Мы не перезапускали полный
MTEB на Q8: наша отдельная проверка измеряет сохранение поискового поведения при
переносе на MLX и квантизации.

## Какую выпущенную MLX-модель выбрать

| Модель | Для чего подходит | Размер | Русский MTEB исходной модели | Наша проверка Q8 |
|---|---|---:|---:|---:|
| [3B Q8 + BF16 edges](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64) | рекомендуемая по умолчанию | 3,755 GB | 74,56 | NDCG@10 Δ +0,00181 |
| [480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8-g64) | самая компактная и быстрая | 0,525 GB | 70,98 | NDCG@10 Δ +0,00289 |
| **[10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64)** | **research; предупреждение по коду** | **11,144 GB** | **74,98** | **совокупная Δ −0,00046** |

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
scores = queries @ documents.T
```

Для запросов нужна явная инструкция; документы кодируются без префикса. При
обычном инференсе Python-код из этого model repository не исполняется.

## Что находится внутри

- Affine Q8 для подходящих весов, BF16 MoE routers и normalization weights,
  group size 64.
- Размерность: 1536; максимальная длина: 8192.
- Размер артефакта: 11,144 GB; пиковая Metal-память: 14,423 GB для 16 длинных
  текстов.
- Типичное время одного текста на 512 токенов: 0,597 с на M4 Pro.
- Скорость пакета: 0,76 документа/с для 16 текстов по 1024 токена.
- Min/mean cosine относительно нативного MLX BF16: 0,993838 / 0,999272.
- Top-1 agreement: 97,66%; среднее top-10 overlap: 98,09%.
- Изменение RuSTS: +0,000350; classification accuracy: +0,000541. Малые
  положительные значения не считаются улучшением.

Методика, проверки MoE-router, speed samples и evidence hashes приведены в
[`0826` MLX-отчёте](https://github.com/ai-babai/giga-embeddings-mlx/blob/main/docs/benchmarks/0826-results.md).
Q8 уменьшила место на диске и пиковую Metal-память, но была медленнее BF16 в
показанных нагрузках.

## Источник и воспроизводимость

- Исходная модель: [`ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826).
- Оригинальная статья: [arXiv:2608.23806](https://arxiv.org/abs/2608.23806).
- Base revision: `1cb3ad3374dbf0eb9130546ca38b262de5f60287`.
- Релиз весов: `0826-v0.1.0`; tensor bytes не менялись в документационном
  обновлении `0.1.1`.
- Converter commit: `dfbc6a375ccdb637d1932529acbcfbf4db5025b6`.
- `manifest.json` содержит portable inventory и SHA-256.

Проверено на Apple Silicon/macOS. Это независимая квантизация `ai-babai`, а не
официальный релиз `ai-sage`.
