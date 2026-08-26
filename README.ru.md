# Giga Embeddings 0826 для Apple Silicon — MLX

[English version](README.md)

Локальные приватные текстовые эмбеддинги для семантического поиска, RAG,
кластеризации, классификации и сравнения текстов на русском и английском языках.
Проект включает нативный runtime на [MLX](https://github.com/ml-explore/mlx) и
три проверенные Q8-версии семейства Giga Embeddings `0826`. Для обычного
инференса не нужны PyTorch, облачный API и Python-код из репозиториев моделей.

[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[MLX-модели](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-for-apple-silicon-mlx-q8-6a8eec40b26f6543f5da3244) ·
[Оригинальная статья](https://arxiv.org/abs/2608.23806) ·
[Полный MLX-бенчмарк](docs/benchmarks/0826-results.md) ·
[Последний релиз](https://github.com/ai-babai/giga-embeddings-mlx/releases/latest)

![Какую модель Giga Embeddings 0826 MLX выбрать для Apple Silicon](https://raw.githubusercontent.com/ai-babai/giga-embeddings-mlx/main/docs/giga-embeddings-0826-mlx-choice.png?v=2026-08-26-2)

## Зачем нужен этот MLX-порт?

- Исходные модели набирают от **70,98 до 74,98 на русском MTEB** в оценке
  авторов.
- Наши отдельные тесты проверяют перенос исходных BF16-весов на MLX и измеряют,
  что меняется после Q8-квантизации. У выпущенных 480M и 3B не обнаружено
  снижения совокупного качества поиска на замороженном тестовом наборе.
- Рекомендуемая 3B-модель скачивается в размере 3,76 GB и использовала до
  5,14 GB Metal-памяти в нашем тесте. Для BF16 это 6,31 GB и 7,69 GB.
- В комплекте есть Python API, командная строка, offline-кеш и локальный
  OpenAI-совместимый endpoint `/v1/embeddings`.

Это независимый порт [ai-babai](https://github.com/ai-babai), а не официальный
релиз `ai-sage`.

## Установка и первый запуск

Нужны Mac с Apple Silicon, macOS и Python 3.12 или 3.13.

```bash
python -m pip install giga-embeddings-mlx
```

Документы кодируются без префикса. Для поискового запроса нужна явная инструкция:

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("default")

documents = model.encode_documents(
    [
        "Москва — столица России.",
        "Париж — столица Франции.",
    ]
)
queries = model.encode_queries(
    "Где находится Москва?",
    instruction="Given a question, retrieve passages that answer the question",
)

scores = queries @ documents.T
print(scores.tolist())
```

`default` — рекомендуемая 3B Q8-модель. При первом запуске она скачивается с
Hugging Face, затем используется локальный кеш.

## Какую MLX-модель выбрать

| MLX-модель | Для чего подходит | Русский MTEB исходной модели¹ | Наша проверка Q8² | Размер загрузки | Пиковая память³ |
|---|---|---:|---:|---:|---:|
| **[3B MLX Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8)** | **рекомендуемая по умолчанию** | **74,56** | **NDCG@10 Δ +0,00181** | **3,755 GB** | **5,137 GB** |
| [480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8) | самая компактная и быстрая | 70,98 | NDCG@10 Δ +0,00289 | 0,525 GB | 1,339 GB |
| [10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8) | исследовательские задачи с большим запасом качества | 74,98 | совокупная Δ −0,00046 | 11,144 GB | 14,423 GB |

1. Task-macro MTEB из
   [статьи Giga-Embeddings](https://arxiv.org/html/2608.23806#S4). Это результат
   исходной BF16-модели; полный MTEB на нашем Q8-артефакте не перезапускался.
2. Разница между Q8 и нативным MLX BF16 на отдельном замороженном наборе для
   поиска. Малое положительное значение означает отсутствие измеренного
   ухудшения, а не улучшение модели от квантизации.
3. Максимальное выделение Metal-памяти при обработке 16 текстов длиной по 1024
   токена на M4 Pro с 48 GB объединённой памяти.

10B Q8 прошла совокупный порог, но на поиске по коду NDCG@10 изменился на
`−0,01297`. Поэтому эта модель помечена как исследовательская и не является
вариантом по умолчанию.

Выбрать модель можно без запоминания имени репозитория:

```bash
giga-embeddings-mlx models
giga-embeddings-mlx encode "Москва — столица России." --document
giga-embeddings-mlx encode "Где находится Москва?" \
  --instruction "Given a question, retrieve passages that answer the question"
```

## Насколько сохраняется исходное качество

В оригинальной статье приведён task-macro MTEB по 41 английской, 23 русским,
131 мультиязычной и 12 задачам по коду:

| Исходная BF16-модель | Английский | Русский | Мультиязычный | Код |
|---|---:|---:|---:|---:|
| 480M | 69,52 | 70,98 | 56,97 | 72,87 |
| 3B | 71,93 | 74,56 | 63,89 | 76,93 |
| 10B-A1.8B | **72,23** | **74,98** | **65,64** | **78,41** |

Источник: [Giga-Embeddings: Mixture-of-Experts Encoders for High-Throughput
Text Embeddings](https://arxiv.org/html/2608.23806#S4). Авторы предупреждают,
что каждая модель оценивалась один раз, поэтому различия меньше одного пункта
следует трактовать осторожно.

Мы не выдаём Q8-артефакты за новые официальные результаты MTEB. Вместо этого
проверены два преобразования, от которых зависит порт:

| Семейство | Исходный BF16 → MLX BF16 | MLX BF16 → выпущенный Q8 | Худшая измеренная группа Q8 |
|---|---|---:|---:|
| 480M | порог качества поиска пройден | NDCG@10 Δ +0,00289 | +0,00000 |
| 3B | порог качества поиска пройден | NDCG@10 Δ +0,00181 | −0,00048 |
| 10B-A1.8B | порог качества поиска пройден | совокупная NDCG@10 Δ −0,00046 | код: −0,01297 |

Эта цепочка подтверждает сохранение качества на нашем зафиксированном наборе
русских, английских, мультиязычных задач и поиска по коду. Она не превращает
локальную проверку в официальный MTEB. В
[полном отчёте](docs/benchmarks/0826-results.md) приведены сравнения векторов,
рангов, отдельных групп и downstream-задач.

## Скорость и память понятными словами

Измерения выполнены на MacBook Pro с Apple M4 Pro и 48 GB объединённой памяти.
Для каждого замера скорости сделано 2 прогрева и 5 измеряемых повторов.

| MLX-модель | Типичное время для одного текста на 512 токенов | Скорость пакета из 16 длинных текстов | Пиковая память для такого пакета |
|---|---:|---:|---:|
| 480M Q8 | 0,071 с | 6,38 документа/с | 1,339 GB |
| 3B Q8 | 0,637 с | 0,73 документа/с | 5,137 GB |
| 10B-A1.8B Q8 | 0,597 с | 0,76 документа/с | 14,423 GB |

Под «длинным текстом» здесь понимается 1024 токена. Токены — части текста,
которыми оперирует модель; они не всегда совпадают с целыми словами. Q8
уменьшила место на диске и пиковую Metal-память примерно на 25–40%, но в тесте
на 16 текстах оказалась на 12–19% медленнее BF16. Q8 стоит выбирать прежде всего
для экономии памяти и места, а не ради гарантированного ускорения.

[Полный отчёт](docs/benchmarks/0826-results.md) содержит median и p95, время
загрузки, документов и токенов в секунду, память процесса, Metal-память,
изменения качества, команды и хеши доказательств.

## Исходные BF16-модели через MLX

Тот же runtime умеет загружать точные исходные BF16-веса без их повторной
публикации под `ai-babai`:

| Алиас runtime | Исходная модель | Размер загрузки |
|---|---|---:|
| `480m-bf16` | [ai-sage/Giga-Embeddings-instruct-480M-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826) | 1,0 GB |
| `3b-bf16` | [ai-sage/Giga-Embeddings-instruct-3B-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826) | 6,3 GB |
| `10b-a1.8b-bf16` | [ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826) | 21,0 GB |

В MLX Collection остаются только три готовых Q8-артефакта. Исходные веса
связаны с ними здесь и в model cards как источник и эталон качества.

## Явные репозитории и локальные артефакты

Алиас, явный Hugging Face repository/revision и локальная директория используют
один загрузчик:

```python
from pathlib import Path

from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("480m-q8")
model = load_embedding_model(
    "owner/repository",
    revision="immutable-commit-or-tag",
    cache_dir=Path("models-cache"),
)
model = load_embedding_model(Path("portable-model-directory"))
```

Встроенные алиасы разрешаются в точные проверенные коммиты. Читаемые теги
обозначают релизы, а неизменяемые коммиты обеспечивают воспроизводимость.

## Кеш и работа без сети

```python
model = load_embedding_model("3b-q8", cache_dir="models-cache")
model = load_embedding_model(
    "3b-q8",
    cache_dir="models-cache",
    local_files_only=True,
)
```

Эквиваленты в CLI: `--cache-dir models-cache` и `--offline`. Удаляйте только
кеш, который вы намеренно выделили этому проекту. Не удаляйте глобальный кеш
Hugging Face ради одного snapshot.

Перед загрузкой runtime консервативно сравнивает размер весов с физической
объединённой памятью. Если проверка не проходит, выберите модель меньше.
`skip_memory_check=True` / `--skip-memory-check` — явный обход для пользователей,
готовых принять риск swap или нехватки памяти.

## Локальный OpenAI-совместимый endpoint

```bash
python -m pip install 'giga-embeddings-mlx[server]'
giga-embeddings-mlx serve --model default --served-model-name giga-3b
```

Endpoint: `POST /v1/embeddings`. Ввод без `instruction` считается документом.
Для запроса инструкцию нужно передать явно:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "giga-3b",
    "input": ["Где находится Москва?"],
    "instruction": "Given a question, retrieve passages that answer the question"
  }'
```

Сервер поддерживает выдачу `float` и `base64`. Усечение размерности не
поддерживается: модели `0826` не заявляют Matryoshka-обучение. Один процесс
обслуживает одну модель и последовательно выполняет Metal-инференс.

## Совместимость и ограничения

- Только Apple Silicon/macOS; на других платформах используйте исходный runtime.
- Максимальная длина — 8192 токена, но расход памяти растёт с размером пакета и
  длиной текста.
- Q8 не гарантирует ускорение относительно BF16 на Metal.
- Предупреждение по поиску в коде для 10B существенно; для общих задач лучше 3B.
- Пакет не усекает размерность эмбеддинга молча.
- `uint8` или бинарное сжатие готовых векторов — отдельный выбор для хранения
  индекса и не выполняется загрузчиком весов.
- Локальная проверка качества не является официальным leaderboard result.

## Разработка, лицензия и цитирование

Конвертация и evaluation остаются инструментами разработчика и не входят в
пользовательский CLI. См. [CONTRIBUTING.md](CONTRIBUTING.md),
[docs/EVALUATION.md](docs/EVALUATION.md) и [SECURITY.md](SECURITY.md).

Независимый MLX-runtime распространяется по MIT. Лицензии и уведомления
исходных моделей остаются в соответствующих репозиториях; см.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). При использовании укажите
[оригинальную статью Giga-Embeddings](https://arxiv.org/abs/2608.23806) и этот
проект через [CITATION.cff](CITATION.cff).
