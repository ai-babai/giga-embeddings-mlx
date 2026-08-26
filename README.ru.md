# Giga Embeddings MLX

[English version](README.md)

Локальный нативный MLX runtime для трех моделей линии `0826`:

- `ai-sage/Giga-Embeddings-instruct-480M-0826`;
- `ai-sage/Giga-Embeddings-instruct-3B-0826`;
- `ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`.

Это рабочий результат цели `GIGA-EMBEDDINGS-GOAL-001`, а не опубликованный
релиз. Тяжелые веса, датасеты, кэши и сырые результаты не хранятся в Git.

## Состояние

Для 480M и 3B реализован общий bidirectional Qwen3-путь. Для 10B реализован и
полностью измерен bidirectional DeepSeek-V3 MoE-путь с отдельной структурной,
численной и router-проверкой. Runtime использует полнонаправленное внимание без
generation cache, усреднение в FP32 только по непустым токенам и L2-нормализацию
в FP32. Python-код из checkpoint не исполняется при обычном MLX-инференсе.

## Установка и инференс

```bash
uv sync --frozen
uv run giga-embeddings-mlx models
uv run giga-embeddings-mlx encode 480m \
  "Москва — столица Российской Федерации." \
  "Париж — столица Франции."
```

Доступны закрепленные профили `480m`, `3b` и `10b-a1.8b`. Для поиска только
запрос получает точный префикс:

```text
Instruct: Given a query, retrieve relevant passages
Query: Где столица России?
```

Документы кодируются без этого префикса.

## Локальный OpenAI-compatible endpoint

```bash
uv sync --frozen --extra server
uv run giga-embeddings-mlx serve 480m \
  --served-model-name giga-embeddings-480m-0826
```

По умолчанию сервер слушает только `127.0.0.1:8000`. Реализованы
`POST /v1/embeddings` с форматами `float` и `base64`, а также `GET /health`.
Сервер не угадывает тип входа: клиент сам добавляет instruction к запросам и
не добавляет его к документам. Это локальный reference interface, не
production deployment.

## Конвертация

Каждый lossy-вариант создается прямо из BF16 source, а не из другого кванта:

```bash
uv run giga-embeddings-mlx convert 3b /path/to/3b-q8-g64 --bits 8
uv run giga-embeddings-mlx convert 3b /path/to/3b-q6-g64 --bits 6
uv run giga-embeddings-mlx convert 3b /path/to/3b-q4-g64 --bits 4
```

Поддержаны и явные experimental mixed/policy варианты, но их наличие не
означает, что они рекомендованы. Converter не перезаписывает существующий
каталог и ничего не загружает во внешние сервисы.

## Как принимается решение

Каждый квант сравнивается со своим MLX BF16 baseline на frozen holdout. В
отчет входят дрейф векторов, retrieval ranking, русские STS/classification,
размер, время загрузки, warm median/p95 latency, documents/s, tokens/s, RSS,
peak Metal allocation, pressure и swap. Квантование весов и сжатие выходных
векторов оцениваются раздельно.

Строгие диагностические gates revision 1/2 не пройдены из-за воспроизводимого
BF16 rounding drift, хотя FP32 graph controls совпадают. Согласованная
effectiveness-based revision 3 прошла для всех трех профилей: pooled-vector и
padding checks, а также aggregate и каждая RU/EN/code/multilingual family по
MRR/NDCG@10 находятся в заданных пределах. Локально приняты BF16 baselines,
`480m-q8-g64`, `3b-q8-edges-bf16-g64` и `10b-a1.8b-q8-g64`. Эти артефакты еще
не опубликованы и не должны называться released.

Воспроизводимые команды и фиксированная speed-матрица описаны в
[`docs/EVALUATION.md`](docs/EVALUATION.md).
