from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tiktoken
from transformers import AutoTokenizer

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x0980
LEADING_MARKERS = (" ", "\u2581", "\u0120", "\u00A0")  # space, ▁, Ġ, nbsp
RNG = random.Random(20241005)

LANG_SPECS = {
    "Hindi": "tokenizer-output/hindi-fast.csv",
    "Hinglish": "tokenizer-output/hinglish-fast.csv",
}

EFFICIENCY_FILES = {
    "English": "tokenizer-output/english-fast.csv",
    "Hindi": "tokenizer-output/hindi-fast.csv",
    "Kannada": "tokenizer-output/kannada-fast.csv",
    "Tamil": "tokenizer-output/tamil-fast.csv",
    "Hinglish": "tokenizer-output/hinglish-fast.csv",
}

TOKENIZER_IDS = [
    "openai/tiktoken/o200k_base",
    "openai/tiktoken/cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only",
]

OPTIONAL_TOKENIZERS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


@dataclass
class TokenData:
    tokens: list[str]
    elapsed_ms: float


class TokenizerAdapter:
    def __init__(self, tokenizer_id: str, load_optional: bool = False) -> None:
        self.tokenizer_id = tokenizer_id
        self.kind = "hf"
        self.encoder = None
        self.tokenizer = None

        if tokenizer_id.endswith("o200k_base"):
            self.kind = "tiktoken"
            self.encoder = tiktoken.get_encoding("o200k_base")
        elif tokenizer_id.endswith("cl100k_base"):
            self.kind = "tiktoken"
            self.encoder = tiktoken.get_encoding("cl100k_base")
        elif tokenizer_id in OPTIONAL_TOKENIZERS and not load_optional:
            raise RuntimeError(
                f"Tokenizer {tokenizer_id} requested but optional adapters disabled"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                use_fast=True,
                trust_remote_code=True,
            )

    def tokenize(self, text: str) -> list[str]:
        if self.kind == "tiktoken":
            assert self.encoder is not None
            token_ids = self.encoder.encode(
                text,
                allowed_special=set(),
                disallowed_special=()
            )
            tokens: list[str] = []
            for tok in token_ids:
                token_bytes = self.encoder.decode_single_token_bytes(tok)
                decoded = token_bytes.decode("utf-8", errors="replace")
                tokens.append(decoded)
            return tokens
        assert self.tokenizer is not None
        pieces = self.tokenizer.tokenize(text, add_special_tokens=False)
        normalized: list[str] = []
        for piece in pieces:
            if piece.startswith("##"):
                normalized.append(piece[2:])
            else:
                normalized.append(piece)
        return normalized


def is_devanagari(ch: str) -> bool:
    code = ord(ch)
    return DEVANAGARI_START <= code < DEVANAGARI_END


def is_latin(ch: str) -> bool:
    return "a" <= ch.lower() <= "z"


def classify_script(ch: str) -> str:
    if is_devanagari(ch):
        return "deva"
    if is_latin(ch):
        return "latin"
    return "other"


def strip_leading_markers(token: str) -> tuple[str, bool]:
    stripped = token
    marker_found = False
    while stripped and stripped[0] in LEADING_MARKERS:
        stripped = stripped[1:]
        marker_found = True
    return stripped, marker_found


def boundary_is_covered(tokens: Sequence[str], first_char: str) -> bool:
    for token in tokens:
        core, marker = strip_leading_markers(token)
        if not marker:
            continue
        if core and core[0] == first_char:
            return True
    return False


def compute_row_metrics(text: str, tokens: Sequence[str]) -> dict[str, float]:
    lsar_num = 0
    lsar_den = 0
    csbc_num = 0
    csbc_den = 0
    deva_codepoints = 0
    deva_token_count = 0

    for token in tokens:
        chars = [ch for ch in token if ch and not ch.isspace()]
        has_deva = any(is_devanagari(ch) for ch in chars)
        if has_deva:
            deva_token_count += 1
            deva_codepoints += sum(1 for ch in chars if is_devanagari(ch))

    words = list(re.finditer(r"\S+", text))
    prev_script: str | None = None

    for match in words:
        token_text = match.group(0)
        first_char = next((ch for ch in token_text if classify_script(ch) != "other"), None)
        if not first_char:
            continue
        script = classify_script(first_char)
        if script == "deva" and match.start() > 0:
            lsar_den += 1
            if boundary_is_covered(tokens, first_char):
                lsar_num += 1

        if prev_script and prev_script != script and prev_script != "other" and script != "other" and match.start() > 0:
            csbc_den += 1
            if boundary_is_covered(tokens, first_char):
                csbc_num += 1

        prev_script = script

    spr = (deva_codepoints / deva_token_count) if deva_token_count else 0.0
    lsar = (lsar_num / lsar_den) if lsar_den else 0.0
    csbc = (csbc_num / csbc_den) if csbc_den else 0.0

    return {
        "lsar_num": lsar_num,
        "lsar_den": lsar_den,
        "csbc_num": csbc_num,
        "csbc_den": csbc_den,
        "spr_sum": deva_codepoints,
        "spr_count": deva_token_count,
        "lsar": lsar,
        "csbc": csbc,
        "spr": spr,
    }


def load_baseline_texts(csv_path: Path) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sweep_axis"] != "baseline":
                continue
            template_id = row["template_id"]
            if template_id in seen:
                continue
            seen[template_id] = {
                "template_id": template_id,
                "text": row["text"],
                "ascii_ratio": float(row.get("ascii_ratio_bytes", 0.0) or 0.0),
            }
    return list(seen.values())


def measure_throughput(adapter: TokenizerAdapter, texts: Sequence[str]) -> float:
    start = time.perf_counter()
    total_tokens = 0
    for text in texts:
        tokens = adapter.tokenize(text)
        total_tokens += len(tokens)
    elapsed = time.perf_counter() - start
    if elapsed == 0:
        return float("inf")
    return total_tokens / elapsed


def cliffs_delta(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    greater = 0
    lesser = 0
    for a in values_a:
        for b in values_b:
            if a > b:
                greater += 1
            elif a < b:
                lesser += 1
    total = len(values_a) * len(values_b)
    if total == 0:
        return 0.0
    return (greater - lesser) / total


def bootstrap_ci(values: Sequence[float], *, iterations: int = 5000, seed: int = 20241005) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    medians = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    lower = medians[int(0.025 * len(medians))]
    upper = medians[int(0.975 * len(medians))]
    return lower, upper


def compute_convergence_curve(deltas: Sequence[float]) -> list[dict[str, float]]:
    results = []
    if not deltas:
        return results
    n = len(deltas)
    for k in range(10, n + 1, 10):
        medians = []
        for _ in range(500):
            sample = RNG.sample(deltas, k)
            medians.append(statistics.median(sample))
        medians.sort()
        results.append(
            {
                "k": k,
                "median": float(np.mean(medians)),
                "ci_low": float(medians[int(0.025 * len(medians))]),
                "ci_high": float(medians[int(0.975 * len(medians))]),
            }
        )
    return results


def segmented_regression(ascii_values: Sequence[float], deltas: Sequence[float]) -> dict[str, float]:
    if len(ascii_values) != len(deltas) or len(ascii_values) < 4:
        return {"breakpoint": math.nan, "slope_left": math.nan, "slope_right": math.nan}
    xs = np.array(ascii_values)
    ys = np.array(deltas)
    unique_breaks = np.unique(xs)
    best_sse = math.inf
    best_break = None
    best_slopes = (math.nan, math.nan)

    for bp in unique_breaks[1:-1]:
        left_mask = xs <= bp
        right_mask = xs >= bp
        if left_mask.sum() < 2 or right_mask.sum() < 2:
            continue
        X_left = np.vstack([xs[left_mask], np.ones(left_mask.sum())]).T
        X_right = np.vstack([xs[right_mask], np.ones(right_mask.sum())]).T
        beta_left, _, _, _ = np.linalg.lstsq(X_left, ys[left_mask], rcond=None)
        beta_right, _, _, _ = np.linalg.lstsq(X_right, ys[right_mask], rcond=None)
        pred_left = X_left @ beta_left
        pred_right = X_right @ beta_right
        sse = float(np.sum((ys[left_mask] - pred_left) ** 2) + np.sum((ys[right_mask] - pred_right) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_break = float(bp)
            best_slopes = (float(beta_left[0]), float(beta_right[0]))

    return {
        "breakpoint": best_break if best_break is not None else math.nan,
        "slope_left": best_slopes[0],
        "slope_right": best_slopes[1],
    }


def load_template_deltas(csv_path: Path) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    ascii_values: list[float] = []
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sweep_axis"] != "baseline":
                continue
            template = row["template_id"]
            tokenizer = row["tokenizer_id"]
            grouped[template][tokenizer] = row

    for template, by_tok in grouped.items():
        cl = by_tok.get("openai/tiktoken/cl100k_base")
        o2 = by_tok.get("openai/tiktoken/o200k_base")
        if not cl or not o2:
            continue
        delta = float(cl["tokens_per_100_chars"]) - float(o2["tokens_per_100_chars"])
        deltas.append(delta)
        ascii_raw = cl.get("ascii_ratio_bytes") or o2.get("ascii_ratio_bytes") or "0.0"
        ascii_values.append(float(ascii_raw))

    return deltas, ascii_values


def wilcoxon_paired(deltas: Sequence[float]) -> float:
    filtered: list[tuple[float, int]] = []
    for val in deltas:
        if val > 0:
            filtered.append((abs(val), 1))
        elif val < 0:
            filtered.append((abs(val), -1))
    n = len(filtered)
    if n == 0:
        return math.nan
    # Rank absolute values with average ranks for ties (1-indexed)
    sorted_items = sorted(enumerate(filtered), key=lambda item: item[1][0])
    ranks = [0.0] * n
    tie_counts: list[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and math.isclose(sorted_items[j][1][0], sorted_items[i][1][0]):
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[sorted_items[k][0]] = avg_rank
        tie_counts.append(j - i)
        i = j
    W_plus = sum(rank for rank, (_, sign) in zip(ranks, filtered) if sign > 0)
    mu = n * (n + 1) / 4
    tie_adjust = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts if t > 1)
    sigma_sq = (n * (n + 1) * (2 * n + 1) - tie_adjust) / 24
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    z = (W_plus - mu - 0.5) / sigma
    p = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return max(min(p, 1.0), 0.0)


def analyse_language_deltas(lang_paths: dict[str, Path]) -> list[dict[str, float]]:
    summary = []
    for lang, path in lang_paths.items():
        deltas, ascii_vals = load_template_deltas(path)
        if not deltas:
            continue
        lower, upper = bootstrap_ci(deltas)
        p_value = wilcoxon_paired(deltas)
        summary.append(
            {
                "language": lang,
                "median_delta": float(statistics.median(deltas)),
                "ci_low": float(lower),
                "ci_high": float(upper),
                "cliffs_delta": float(cliffs_delta(deltas, [0.0] * len(deltas))),
                "wilcoxon_p": p_value,
                "convergence": compute_convergence_curve(deltas),
                "ascii": segmented_regression(ascii_vals, deltas),
            }
        )
    return summary


def main(load_optional: bool = False) -> None:
    out_dir = Path("analysis/out")
    out_dir.mkdir(exist_ok=True, parents=True)

    adapters = {
        tok_id: TokenizerAdapter(tok_id, load_optional=load_optional)
        for tok_id in TOKENIZER_IDS + ([] if not load_optional else OPTIONAL_TOKENIZERS)
    }

    row_records = []
    aggregate_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    token_counters: dict[str, Counter[str]] = defaultdict(Counter)
    throughput = {}

    for lang, rel_path in LANG_SPECS.items():
        csv_path = Path(rel_path)
        baselines = load_baseline_texts(csv_path)
        texts = [row["text"] for row in baselines]

        for tok_id, adapter in adapters.items():
            if tok_id not in TOKENIZER_IDS and not load_optional:
                continue

            lsar_total_num = 0
            lsar_total_den = 0
            csbc_total_num = 0
            csbc_total_den = 0
            spr_sum = 0
            spr_count = 0

            leading_counter = token_counters[tok_id]

            for base in baselines:
                tokens = adapter.tokenize(base["text"])
                metrics = compute_row_metrics(base["text"], tokens)

                lsar_total_num += metrics["lsar_num"]
                lsar_total_den += metrics["lsar_den"]
                csbc_total_num += metrics["csbc_num"]
                csbc_total_den += metrics["csbc_den"]
                spr_sum += metrics["spr_sum"]
                spr_count += metrics["spr_count"]

                if metrics["lsar_num"]:
                    for token in tokens:
                        core, marker = strip_leading_markers(token)
                        if marker and any(is_devanagari(ch) for ch in core):
                            leading_counter[token] += 1

                row_records.append(
                    {
                        "language": lang,
                        "tokenizer_id": tok_id,
                        "template_id": base["template_id"],
                        "ascii_ratio": base["ascii_ratio"],
                        "lsar": metrics["lsar"],
                        "csbc": metrics["csbc"],
                        "spr": metrics["spr"],
                    }
                )

            throughput[(lang, tok_id)] = measure_throughput(adapter, texts)

            aggregate = aggregate_stats[tok_id]
            aggregate[f"{lang}_lsar_num"] += lsar_total_num
            aggregate[f"{lang}_lsar_den"] += lsar_total_den
            aggregate[f"{lang}_csbc_num"] += csbc_total_num
            aggregate[f"{lang}_csbc_den"] += csbc_total_den
            aggregate[f"{lang}_spr_sum"] += spr_sum
            aggregate[f"{lang}_spr_count"] += spr_count

    summary_rows = []
    for tok_id, stats_map in aggregate_stats.items():
        row = {"tokenizer_id": tok_id}
        for lang in LANG_SPECS:
            lsar_num = stats_map.get(f"{lang}_lsar_num", 0.0)
            lsar_den = stats_map.get(f"{lang}_lsar_den", 0.0)
            csbc_num = stats_map.get(f"{lang}_csbc_num", 0.0)
            csbc_den = stats_map.get(f"{lang}_csbc_den", 0.0)
            spr_sum = stats_map.get(f"{lang}_spr_sum", 0.0)
            spr_count = stats_map.get(f"{lang}_spr_count", 0.0)
            row[f"{lang}_lsar"] = (lsar_num / lsar_den) if lsar_den else 0.0
            row[f"{lang}_csbc"] = (csbc_num / csbc_den) if csbc_den else 0.0
            row[f"{lang}_spr"] = (spr_sum / spr_count) if spr_count else 0.0
        summary_rows.append(row)

    (Path(out_dir) / "abugida_metrics_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    row_lines = [
        f"{r['language']},{r['tokenizer_id']},{r['template_id']},{r['ascii_ratio']:.6f},{r['lsar']:.6f},{r['csbc']:.6f},{r['spr']:.6f}"
        for r in row_records
    ]
    (Path(out_dir) / "abugida_row_metrics.csv").write_text(
        "language,tokenizer_id,template_id,ascii_ratio,lsar,csbc,spr\n" + "\n".join(row_lines),
        encoding="utf-8",
    )

    examples_payload = {
        tok_id: counter.most_common(12)
        for tok_id, counter in token_counters.items()
    }
    (Path(out_dir) / "leading_space_examples.json").write_text(
        json.dumps(examples_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    throughput_rows = [
        {
            "language": lang,
            "tokenizer_id": tok_id,
            "tokens_per_sec": throughput[(lang, tok_id)],
        }
        for (lang, tok_id) in throughput
    ]
    (Path(out_dir) / "tokenizer_throughput.json").write_text(
        json.dumps(throughput_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    efficiency_paths = {lang: Path(path) for lang, path in EFFICIENCY_FILES.items()}
    delta_summary = analyse_language_deltas(efficiency_paths)
    (Path(out_dir) / "efficiency_deltas.json").write_text(
        json.dumps(delta_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-optional",
        action="store_true",
        help="Include optional tokenizers like Meta-Llama",
    )
    args = parser.parse_args()
    main(load_optional=args.load_optional)
