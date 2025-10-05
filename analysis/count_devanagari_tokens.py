"""Count tokens containing Devanagari characters in OpenAI tokenizers.

This script scans the published merge tables exposed via `tiktoken`
for `cl100k_base` and `o200k_base`, counting tokens whose UTF-8
renderings include code points in the Devanagari block (\u0900-\u097F).
Run with Python 3.10+.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken


def has_devanagari(token_bytes: bytes) -> bool:
    """Return True if *token_bytes* decodes to any Devanagari code point."""

    if not token_bytes:
        return False
    try:
        text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def count_tokens(encoding_name: str) -> int:
    """Count Devanagari-bearing tokens for the named encoding."""

    encoding = tiktoken.get_encoding(encoding_name)
    core = encoding._core_bpe  # type: ignore[attr-defined]

    total = 0
    for token_bytes in core.token_byte_values():
        if has_devanagari(token_bytes):
            total += 1

    # Include special tokens if their string form contains Devanagari.
    for special in encoding._special_tokens:  # type: ignore[attr-defined]
        if any("\u0900" <= ch <= "\u097F" for ch in special):
            total += 1

    return total


def main() -> None:
    counts = {
        name: count_tokens(name) for name in ("cl100k_base", "o200k_base")
    }
    counts["delta"] = counts["o200k_base"] - counts["cl100k_base"]
    print(json.dumps(counts, indent=2))

    out_path = Path("analysis") / "devanagari_token_counts.json"
    out_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"Saved counts to {out_path}")


if __name__ == "__main__":
    main()
