"""Ad-hoc merge pattern sampling for Hindi baseline rows.

This script draws a reproducible sample of token IDs for
`cl100k_base` and `o200k_base` from the pre-exported
`hindi-fast.csv`, decodes each token via tiktoken, and summarises
high-level categories that explain why `o200k_base` is more compact.

The resulting counts and exemplar tokens are meant to support the
"Merge Pattern Analysis" subsection in the paper.
"""
from __future__ import annotations

import csv
import json
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import tiktoken

CSV_PATH = Path("tokenizer-output/hindi-fast.csv")
BASELINE_SWEEP_VALUE = "baseline"
CL_TOKENIZER_ID = "openai/tiktoken/cl100k_base"
O2_TOKENIZER_ID = "openai/tiktoken/o200k_base"
SAMPLE_SIZE = 50
RNG_SEED = 20241004


def load_token_stream(tokenizer_id: str) -> List[int]:
    """Flatten all token IDs from baseline rows for a tokenizer."""

    tokens: list[int] = []
    with CSV_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sweep_axis"] != BASELINE_SWEEP_VALUE:
                continue
            if row["tokenizer_id"] != tokenizer_id:
                continue
            payload_raw = row.get("debug_token_ids_json")
            if not payload_raw:
                continue
            token_payload = json.loads(payload_raw)
            if isinstance(token_payload, dict):
                token_ids = token_payload.get("values", [])
            else:
                token_ids = token_payload
            tokens.extend(int(tok) for tok in token_ids)
    if not tokens:
        raise RuntimeError(f"No tokens collected for {tokenizer_id}")
    return tokens


def load_token_strings(tokenizer_id: str) -> list[str]:
    strings: list[str] = []
    with CSV_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sweep_axis"] != BASELINE_SWEEP_VALUE:
                continue
            if row["tokenizer_id"] != tokenizer_id:
                continue
            payload_raw = row.get("debug_token_strings_json")
            if not payload_raw:
                continue
            token_payload = json.loads(payload_raw)
            if isinstance(token_payload, dict):
                token_values = token_payload.get("values", [])
            else:
                token_values = token_payload
            strings.extend(str(tok) for tok in token_values)
    if not strings:
        raise RuntimeError(f"No token strings collected for {tokenizer_id}")
    return strings


def get_encoding(tokenizer_id: str) -> tiktoken.Encoding:
    if tokenizer_id.endswith("cl100k_base"):
        return tiktoken.get_encoding("cl100k_base")
    if tokenizer_id.endswith("o200k_base"):
        return tiktoken.get_encoding("o200k_base")
    raise ValueError(f"Unsupported tokenizer id: {tokenizer_id}")


@dataclass
class TokenObservation:
    token_id: int
    text: str
    token_bytes: bytes
    byte_length: int
    categories: tuple[str, ...]


DEVANAGARI_BLOCK = range(0x0900, 0x0980)
HALANT = "\u094d"
SPACE = " "


def categorise_token(token_bytes: bytes, rendered: str) -> tuple[str, ...]:
    categories: list[str] = []

    if len(token_bytes) == 1 and token_bytes[0] >= 0x80:
        categories.append("byte_fallback")

    codepoints = [ord(ch) for ch in rendered]

    if len(rendered) == 1 and codepoints and codepoints[0] in DEVANAGARI_BLOCK:
        cat = unicodedata.category(rendered)
        if cat.startswith("L"):
            categories.append("single_devanagari_letter")

    if any(unicodedata.category(ch).startswith("M") for ch in rendered):
        base_chars = [ch for ch in rendered if unicodedata.category(ch).startswith("L")]
        if base_chars:
            categories.append("base_plus_matra")

    if HALANT in rendered:
        categories.append("conjunct_or_halant")

    if SPACE in rendered:
        categories.append("contains_space")

    return tuple(categories)


def sample_observations(token_ids: Iterable[int], encoding: tiktoken.Encoding, *, sample_size: int, rng: random.Random) -> list[TokenObservation]:
    token_list = list(token_ids)
    if len(token_list) < sample_size:
        raise ValueError("Not enough tokens available to satisfy the sample size")
    indices = rng.sample(range(len(token_list)), sample_size)
    observations: list[TokenObservation] = []
    for idx in indices:
        token_id = token_list[idx]
        token_bytes = encoding.decode_single_token_bytes(token_id)
        rendered = token_bytes.decode("utf-8", errors="replace")
        categories = categorise_token(token_bytes, rendered)
        observations.append(TokenObservation(token_id, rendered, token_bytes, len(token_bytes), categories))
    return observations


def sanitise(text: str) -> str:
    return text.strip()


def escape(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def summarise(observations: Iterable[TokenObservation]):
    cat_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for obs in observations:
        if obs.categories:
            for cat in obs.categories:
                cat_counter[cat] += 1
                if len(examples[cat]) < 10:
                    snippet = sanitise(obs.text)
                    if cat == "byte_fallback" and obs.token_bytes:
                        hex_repr = " ".join(f"0x{b:02x}" for b in obs.token_bytes)
                        snippet = f"{snippet or '<byte>'} [{hex_repr}]"
                    examples[cat].append(snippet)
        token_counter[obs.text] += 1
    return cat_counter, token_counter, examples


def main() -> None:
    rng = random.Random(RNG_SEED)

    cl_tokens = load_token_stream(CL_TOKENIZER_ID)
    o2_tokens = load_token_stream(O2_TOKENIZER_ID)

    cl_enc = get_encoding(CL_TOKENIZER_ID)
    o2_enc = get_encoding(O2_TOKENIZER_ID)

    cl_obs = sample_observations(cl_tokens, cl_enc, sample_size=SAMPLE_SIZE, rng=rng)
    o2_obs = sample_observations(o2_tokens, o2_enc, sample_size=SAMPLE_SIZE, rng=rng)

    cl_cats, cl_top, cl_examples = summarise(cl_obs)
    o2_cats, o2_top, o2_examples = summarise(o2_obs)

    print("cl100k_base category counts:")
    for cat, count in cl_cats.most_common():
        print(f"  {cat:24s}: {count}")
        if cat in cl_examples:
            joined = ", ".join(escape(example) or "<space>" for example in cl_examples[cat])
            print(f"    e.g., {joined}")
    print("o200k_base category counts:")
    for cat, count in o2_cats.most_common():
        print(f"  {cat:24s}: {count}")
        if cat in o2_examples:
            joined = ", ".join(escape(example) or "<space>" for example in o2_examples[cat])
            print(f"    e.g., {joined}")

    print("\ncl100k_base sample tokens:")
    for text, freq in cl_top.most_common(15):
        safe_text = escape(text)
        print(f"  {safe_text}: {freq}")

    print("\no200k_base sample tokens:")
    for text, freq in o2_top.most_common(15):
        safe_text = escape(text)
        print(f"  {safe_text}: {freq}")

    cl_strings = load_token_strings(CL_TOKENIZER_ID)
    o2_strings = load_token_strings(O2_TOKENIZER_ID)
    cl_string_counts = Counter(cl_strings)
    o2_string_counts = Counter(o2_strings)

    phrase = " के लिए"
    print("\nPhrase coverage (leading space included):")
    print(
        f"  {escape(phrase)}: cl100k_base={cl_string_counts[phrase]} tokens, "
        f"o200k_base={o2_string_counts[phrase]} tokens"
    )

    conjuncts = [" क्ष", " ज्ञ"]
    for token in conjuncts:
        print(
            f"  {escape(token)}: cl100k_base={cl_string_counts[token]} tokens, "
            f"o200k_base={o2_string_counts[token]} tokens"
        )

    matras = ["\u093f", "\u0940", "\u0947"]
    print("\nStandalone matra tokens:")
    for matra in matras:
        print(
            f"  {escape(matra)}: cl100k_base={cl_string_counts[matra]}, "
            f"o200k_base={o2_string_counts[matra]}"
        )

    print("\nDirect encoding comparison (prepended with a space to match BPE conventions):")
    for snippet in [" \u0915\u094d\u0937", " \u091c\u094d\u091e", " \u0915\u0940", " \u0924\u0947"]:
        cl_len = len(cl_enc.encode(snippet))
        o2_len = len(o2_enc.encode(snippet))
        print(
            f"  {escape(snippet)}: cl100k_base={cl_len} tokens, "
            f"o200k_base={o2_len} tokens"
        )

if __name__ == "__main__":
    main()
