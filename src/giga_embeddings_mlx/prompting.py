from __future__ import annotations


def format_query(instruction: str, query: str) -> str:
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    return f"Instruct: {instruction.strip()}\nQuery: {query}"

