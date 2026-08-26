# Giga Embeddings MLX

[English version](README.md)

Независимый нативный runtime на
[MLX](https://github.com/ml-explore/mlx) для семейства Giga Embeddings `0826`
на Apple Silicon. Проект поддерживает
[ai-babai](https://github.com/ai-babai); это не официальный выпуск `ai-sage`.

[PyPI](https://pypi.org/project/giga-embeddings-mlx/) ·
[Коллекция Hugging Face](https://huggingface.co/collections/ai-babai/giga-embeddings-0826-mlx-6a8eec40b26f6543f5da3244) ·
[Бенчмарки](docs/benchmarks/0826-results.md) ·
[История изменений](CHANGELOG.md)

Runtime поддерживает точные upstream-ревизии
[480M](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826),
[3B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826) и
[10B-A1.8B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826),
а также три принятых MLX Q8-артефакта. Используются двунаправленный attention
по всей последовательности, padding-aware mean pooling в FP32 и L2-нормализация
в FP32. При обычном inference Python-код из model repositories не исполняется.

## Установка

Требования: Mac на Apple Silicon, macOS и Python 3.12 или 3.13.

```bash
python -m pip install giga-embeddings-mlx
```

По умолчанию используется `3b-q8` — сбалансированная 3B Q8-модель с крайними
слоями в BF16. Размер загрузки — примерно 3,8 GB. При первом запуске файлы
модели скачиваются с Hugging Face.

## Быстрый старт за минуту

Документы кодируются без префикса. Для запроса требуется явная retrieval-
инструкция; runtime не придумывает её самостоятельно.

```python
from giga_embeddings_mlx import load_embedding_model

model = load_embedding_model("default")

documents = model.encode_documents(["Москва — столица России.", "Париж — столица Франции."])
queries = model.encode_queries(
    "Где находится Москва?",
    instruction="Given a question, retrieve passages that answer the question",
)

scores = queries @ documents.T
print(scores.tolist())
```

Низкоуровневый метод `model.encode(...)` принимает уже подготовленный текст.
Если важна retrieval-роль, используйте `encode_queries` и `encode_documents`.

CLI:

```bash
giga-embeddings-mlx models
giga-embeddings-mlx encode "Москва — столица России." --document
giga-embeddings-mlx encode "Где находится Москва?" \
  --instruction "Given a question, retrieve passages that answer the question"
```

## Выбор профиля

| Алиас | Веса | Размерность | Роль | Ожидаемая загрузка |
|---|---|---:|---|---:|
| `480m-bf16` | upstream BF16 | 1024 | самый малый quality baseline | 1,0 GB |
| `480m-q8` | Q8, group 64 | 1024 | compact | 0,5 GB |
| `3b-bf16` | upstream BF16 | 2048 | quality baseline для 3B | 6,3 GB |
| `3b-q8` / `default` | Q8 + крайние слои BF16, group 64 | 2048 | balanced default | 3,8 GB |
| `10b-a1.8b-bf16` | upstream BF16 MoE | 1536 | quality-first при большом запасе памяти | 21,0 GB |
| `10b-a1.8b-q8` | Q8, routers/norms BF16, group 64 | 1536 | compact / research | 11,1 GB |

Репозитории квантованных моделей:
[480M Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-480m-mlx-q8-g64),
[3B balanced](https://huggingface.co/ai-babai/giga-embeddings-0826-3b-mlx-q8-edges-bf16-g64)
и
[10B-A1.8B Q8](https://huggingface.co/ai-babai/giga-embeddings-0826-10b-a1.8b-mlx-q8-g64).

BF16-алиасы указывают на неизменяемые upstream-коммиты; веса не дублируются в
`ai-babai`. Q8-алиасы также указывают прямо на точные проверенные Hugging Face
коммиты этого выпуска. Человекочитаемые теги `0826-v0.1.0` обозначают те же
артефакты, но не требуются runtime для разрешения модели.

10B Q8 не является default-вариантом и не позиционируется как near-lossless:
aggregate retrieval gate пройден, но измеренная разница NDCG@10 для code-family
составила `−0,01297` относительно native MLX BF16.

Q4, Q6, uniform 3B Q8 и dominated mixed-варианты намеренно не публикуются.

## Явный Hub repository и локальные артефакты

Алиасы, явный Hub repository/revision и локальная директория используют один
загрузчик:

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

## Кэш и offline-режим

Если жизненным циклом файлов нужно управлять отдельно, выделите кэш проекта:

```python
model = load_embedding_model("3b-q8", cache_dir="models-cache")
model = load_embedding_model(
    "3b-q8",
    cache_dir="models-cache",
    local_files_only=True,
)
```

Эквиваленты в CLI: `--cache-dir models-cache` и `--offline`. Если закреплённого
snapshot в кэше нет, ошибка offline-режима показывает repository и объясняет,
как заполнить или выбрать кэш.

Удаляйте только директорию, которую вы намеренно выделили этому проекту, и
сначала убедитесь, что её не используют другие приложения Hugging Face. Не
удаляйте глобальный Hugging Face cache ради одного snapshot.

До загрузки runtime консервативно сопоставляет размер весов с физической unified
memory. Если preflight не проходит, выберите меньший профиль.
`skip_memory_check=True` / `--skip-memory-check` — явный обход для пользователей,
готовых принять риск swap или out-of-memory.

## Локальный OpenAI-compatible endpoint

Установите опциональные серверные зависимости:

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
поддерживается: upstream-карточки `0826` не заявляют Matryoshka-обучение. Один
процесс обслуживает одну модель и сериализует Metal inference.

## Результаты измерений

Измерения выполнены на MacBook Pro с Apple M4 Pro и 48 GB unified memory,
Python 3.12.11, MLX 0.32.2 и MLX-LM 0.31.3. Для скорости использованы 2 warmup
и 5 измеряемых повторов. Эти числа не предсказывают результат на других Mac и
не сравниваются напрямую с upstream-результатами H100.

Для выпускаемых вариантов Q8 уменьшил размер артефакта примерно на 40–47%, но
на B16×1024 оказался на 12–19% медленнее соответствующего BF16. Рассматривайте
weight quantization здесь прежде всего как компромисс по ёмкости и диску.

Полный генерируемый отчёт отдельно показывает artifact size, process RSS,
Metal peak, load time, median/p95 скорости, сохранение качества BF16 между
backend, drift квантованных vectors, ranking и downstream delta:

- [читаемые таблицы](docs/benchmarks/0826-results.md);
- [machine-readable evidence и хэши источников](docs/benchmarks/0826-results.json).

SHA-256 итогового acceptance JSON:
`410b9cf7756e0718816b23a46f0d99e0f3e6574e4eb515cc5a99cff131057316`.
Замороженный holdout на 512 текстах / 256 запросах / 2048 документах покрывает
русский, английский, code и multilingual до 2048 токенов. Малые положительные
delta означают отсутствие измеренной деградации в этом lane, а не улучшение
модели от квантизации.

## Как трактовать качество

Исходные строгие numerical gates revision 1/2 выявили BF16 cross-backend и
dynamic-shape drift и не прошли. Эти failures сохранены в истории. Revision 3
прошла effectiveness-based gate для pooled vectors, padding, aggregate
retrieval и каждой family с допуском `−0,005` по MRR/NDCG; rank agreement и
hidden-state drift остались обязательной диагностикой.

Это различие существенно: внутренние vectors разных backend могут расходиться,
пока наблюдаемая retrieval-задача остаётся non-inferior. Поэтому публичный выбор
Q8 учитывал representation, ranking, downstream quality, диск, память, загрузку
и скорость, а не одно значение cosine.

## Ограничения

- Только Apple Silicon/macOS; на других платформах используйте upstream runtime.
- Максимальная длина — 8192, но расход памяти растёт с batch и длиной; публичная
  матрица скорости не является обещанием для любого workload.
- Q8 не гарантирует ускорение относительно BF16 на Metal.
- Для 10B Q8 действует явное предупреждение по code retrieval выше.
- Пакет не усекает размерность embedding молча.
- `uint8`/binary-компрессия выходных vectors — отдельный выбор хранения индекса,
  который не выполняется загрузчиком весов.
- Локальный benchmark не является официальным upstream leaderboard result.

## Разработка и цитирование

Conversion и evaluation остаются developer-facing и не входят в
пользовательский CLI. См. [CONTRIBUTING.md](CONTRIBUTING.md) и
[docs/EVALUATION.md](docs/EVALUATION.md). Порядок отправки security reports — в
[SECURITY.md](SECURITY.md).

Этот независимый runtime распространяется по MIT. Лицензии upstream-моделей и
уведомления сохраняются в соответствующих repositories; см.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Метаданные для цитирования — в
[CITATION.cff](CITATION.cff).
