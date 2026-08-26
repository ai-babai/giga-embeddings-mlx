from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import Qwen2Tokenizer

LENGTH_SPECS = [
    ("length-en-0001", "en", 1, "Hello"),
    (
        "length-ru-0008",
        "ru",
        8,
        "Москва — столица России. Санкт-Петербург — культурная столица.",
    ),
    (
        "length-code-0032",
        "code",
        32,
        "def merge(left, right):\n    return sorted(left + right)\n",
    ),
    (
        "length-multilingual-0128",
        "multilingual",
        128,
        "Русский текст, English text, código Python и данные multilingual retrieval. ",
    ),
    (
        "length-ru-0512",
        "ru",
        512,
        "Нейронные сети преобразуют входные данные в векторные представления для поиска. ",
    ),
    (
        "length-en-1024",
        "en",
        1024,
        "Dense retrieval systems encode queries and documents into normalized vector representations. ",
    ),
    (
        "length-code-2048",
        "code",
        2048,
        (
            "def process_record(record):\n"
            "    tokens = tokenize(record)\n"
            "    return normalize(encode(tokens))\n"
        ),
    ),
]

BASELINE_RECORDS = [
    (
        "query-ru-capital",
        "ru",
        "Instruct: Given a query, retrieve relevant passages\nQuery: Где столица России?",
    ),
    ("document-ru-moscow", "ru", "Москва — столица Российской Федерации."),
    ("document-fr-paris", "multilingual", "Париж — столица Франции."),
    (
        "query-en-solar",
        "en",
        "Instruct: Given a query, retrieve relevant passages\nQuery: How do solar panels work?",
    ),
    (
        "document-en-solar",
        "en",
        "Photovoltaic cells convert sunlight directly into electricity.",
    ),
    (
        "code-fibonacci",
        "code",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    ),
    (
        "code-sql-aggregation",
        "code",
        "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id;",
    ),
    (
        "multilingual-short",
        "multilingual",
        "Поиск embeddings across languages должен сохранять semantic geometry.",
    ),
]


def exact_length_text(tokenizer, seed: str, target: int) -> str:
    repeated = (seed + "\n") * (target // 2 + 4)
    token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:target]
    if len(token_ids) != target:
        raise ValueError(f"Seed produced only {len(token_ids)} tokens for target {target}")
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = tokenizer.encode(text, add_special_tokens=False)
    if roundtrip != token_ids:
        raise ValueError(f"Tokenizer roundtrip changed target-length record {target}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_path)
    special_tokens_per_record = len(tokenizer.encode("", add_special_tokens=True)) - len(
        tokenizer.encode("", add_special_tokens=False)
    )
    records = []
    for record_id, family, target, seed in LENGTH_SPECS:
        text = exact_length_text(tokenizer, seed, target)
        records.append(
            {
                "id": record_id,
                "family": family,
                "expected_content_tokens": target,
                "expected_model_tokens": target + special_tokens_per_record,
                "length_control": True,
                "text": text,
            }
        )
    for record_id, family, text in BASELINE_RECORDS:
        records.append(
            {
                "id": record_id,
                "family": family,
                "expected_content_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "expected_model_tokens": len(tokenizer.encode(text, add_special_tokens=True)),
                "length_control": False,
                "text": text,
            }
        )

    payload = {
        "schema_version": 1,
        "corpus_id": "giga-embeddings-0826-parity-v2",
        "source_tokenizer_revision": args.source_revision,
        "required_content_token_lengths": [spec[2] for spec in LENGTH_SPECS],
        "special_tokens_per_record": special_tokens_per_record,
        "padding_control_ids": [spec[0] for spec in LENGTH_SPECS[:3]],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()

    family_counts = {}
    for record in records:
        family = record["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    manifest = {
        "schema_version": 1,
        "corpus_id": payload["corpus_id"],
        "created_at": "2026-08-26",
        "source_tokenizer_revision": args.source_revision,
        "sha256": digest,
        "records": len(records),
        "required_content_token_lengths": payload["required_content_token_lengths"],
        "special_tokens_per_record": special_tokens_per_record,
        "family_counts": family_counts,
        "padding_control_ids": payload["padding_control_ids"],
        "license": "synthetic-evaluation-text",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
