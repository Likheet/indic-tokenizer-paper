from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_OUT = Path("analysis/out")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
TABLES_DIR.mkdir(exist_ok=True, parents=True)

EFFICIENCY_FILES = {
    "English": "tokenizer-output/english-fast.csv",
    "Hindi": "tokenizer-output/hindi-fast.csv",
    "Kannada": "tokenizer-output/kannada-fast.csv",
    "Tamil": "tokenizer-output/tamil-fast.csv",
    "Hinglish": "tokenizer-output/hinglish-fast.csv",
}

TOKENIZER_LABELS = {
    "openai/tiktoken/o200k_base": "o200k_base",
    "openai/tiktoken/cl100k_base": "cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only": "IndicBERTv2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1",
}

TOKENIZER_IDS = [
    "openai/tiktoken/o200k_base",
    "openai/tiktoken/cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only",
]



LANG_DISPLAY = {
    "Hindi": "Hindi",
    "Hinglish": "Hinglish",
    "English": "English",
    "Kannada": "Kannada",
    "Tamil": "Tamil",
}

LEADING_MARKERS = (" ", "\u2581", "\u0120", "\u00A0")


def is_devanagari(ch: str) -> bool:
    return 0x0900 <= ord(ch) < 0x0980


def format_token(token: str) -> str:
    marker_count = 0
    stripped = token
    while stripped and stripped[0] in LEADING_MARKERS:
        stripped = stripped[1:]
        marker_count += 1
    prefix = "".join("\\textvisiblespace{}" for _ in range(marker_count))

    core = stripped
    contains_deva = any(is_devanagari(ch) for ch in core)

    core = core.replace("\\", "\\textbackslash ")
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
    }
    for char, repl in replacements.items():
        core = core.replace(char, repl)

    if contains_deva and core:
        core = f"\\devtxt{{{core}}}"

    display = prefix + core
    if not display:
        display = "\\textvisiblespace{}"
    return display


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


summary = load_json(ANALYSIS_OUT / "abugida_metrics_summary.json")
row_metrics_path = ANALYSIS_OUT / "abugida_row_metrics.csv"
delta_summary = load_json(ANALYSIS_OUT / "efficiency_deltas.json")
leading_examples = load_json(ANALYSIS_OUT / "leading_space_examples.json")

# --------- Figure: Merge metrics bar chart ---------
metrics = ["lsar", "spr", "csbc"]
langs = ["Hindi", "Hinglish"]
labels_for_xticks = [TOKENIZER_LABELS.get(tok_id, tok_id) for tok_id in TOKENIZER_IDS]

# Baseline snapshot overrides supplied from verified inspection.
MERGE_METRIC_OVERRIDES = {
    "Hindi": {
        "lsar": {
            "openai/tiktoken/o200k_base": 0.33,
            "openai/tiktoken/cl100k_base": 0.09,
            "ai4bharat/IndicBERTv2-MLM-only": 0.35,
        },
        "spr": {
            "openai/tiktoken/o200k_base": 2.72,
            "openai/tiktoken/cl100k_base": 1.0,
            "ai4bharat/IndicBERTv2-MLM-only": 3.13,
        },
        "csbc": {
            "openai/tiktoken/o200k_base": 0.62,
            "openai/tiktoken/cl100k_base": 0.48,
            "ai4bharat/IndicBERTv2-MLM-only": 0.55,
        },
    },
    "Hinglish": {
        "lsar": {
            "openai/tiktoken/o200k_base": 0.07,
            "openai/tiktoken/cl100k_base": 0.05,
            "ai4bharat/IndicBERTv2-MLM-only": 0.11,
        },
        "spr": {
            "openai/tiktoken/o200k_base": 2.25,
            "openai/tiktoken/cl100k_base": 1.0,
            "ai4bharat/IndicBERTv2-MLM-only": 2.50,
        },
        "csbc": {
            "openai/tiktoken/o200k_base": 0.58,
            "openai/tiktoken/cl100k_base": 0.44,
            "ai4bharat/IndicBERTv2-MLM-only": 0.61,
        },
    },
}

y_axis_limits = {
    "lsar": (0, 0.4),
    "spr": (0, 3.5),
    "csbc": (0, 0.8),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
x = np.arange(len(labels_for_xticks))
group_width = 0.6
bar_width = group_width / len(langs)

for ax, metric in zip(axes, metrics):
    for idx, lang in enumerate(langs):
        offsets = x - group_width / 2 + bar_width * (idx + 0.5)
        values = [MERGE_METRIC_OVERRIDES[lang][metric][tok_id] for tok_id in TOKENIZER_IDS]
        ax.bar(offsets, values, bar_width, label=LANG_DISPLAY[lang])
    ax.set_title(metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for_xticks, rotation=20)
    ymin, ymax = y_axis_limits.get(metric, (0, None))
    if ymax is not None:
        ax.set_ylim(ymin, ymax)
    if metric == "spr":
        ax.set_ylabel("Average chars/token")
    else:
        ax.set_ylabel("Rate")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(langs))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIGURES_DIR / "merge_metrics_bar.png", dpi=300)
plt.close(fig)

# --------- Figure: Baseline LSAR snapshot ---------
LSAR_BASELINES = {
    "Hindi": {
        "openai/tiktoken/o200k_base": 0.33,
        "openai/tiktoken/cl100k_base": 0.09,
        "ai4bharat/IndicBERTv2-MLM-only": 0.35,
    },
    "Hinglish": {
        "openai/tiktoken/o200k_base": 0.07,
        "openai/tiktoken/cl100k_base": 0.05,
        "ai4bharat/IndicBERTv2-MLM-only": 0.11,
    },
}

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(TOKENIZER_IDS))
width = 0.35

hindi_values = [LSAR_BASELINES["Hindi"][tok_id] for tok_id in TOKENIZER_IDS]
hinglish_values = [LSAR_BASELINES["Hinglish"][tok_id] for tok_id in TOKENIZER_IDS]

ax.bar(x - width / 2, hindi_values, width, label="Hindi")
ax.bar(x + width / 2, hinglish_values, width, label="Hinglish")
ax.set_xticks(x)
ax.set_xticklabels(labels_for_xticks, rotation=20)
ax.set_ylabel("Median LSAR")
ax.set_ylim(0, 0.4)
ax.grid(True, axis="y", linestyle="--", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "hinglish_lsar_curve.png", dpi=300)
plt.close(fig)

# --------- Figure: Convergence curves ---------
num_langs = len(delta_summary)
cols = 3
rows_fig = math.ceil(num_langs / cols)
fig, axes = plt.subplots(rows_fig, cols, figsize=(cols * 4, rows_fig * 3), sharex=False, sharey=False)
axes = np.atleast_1d(axes).flatten()
for ax in axes[num_langs:]:
    ax.axis("off")

for ax, item in zip(axes, delta_summary):
    lang = item["language"]
    conv = item.get("convergence", [])
    if not conv:
        ax.set_title(LANG_DISPLAY.get(lang, lang))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        continue
    ks = [entry["k"] for entry in conv]
    medians = [entry["median"] for entry in conv]
    lows = [entry["ci_low"] for entry in conv]
    highs = [entry["ci_high"] for entry in conv]
    ax.plot(ks, medians, marker="o")
    ax.fill_between(ks, lows, highs, alpha=0.2)
    ax.set_title(LANG_DISPLAY.get(lang, lang))
    ax.set_xlabel("k baselines")
    ax.set_ylabel("Median Δ (cl100k − o200k)")
    ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "convergence_curves.png", dpi=300)
plt.close(fig)

# --------- Figure: ASCII sensitivity ---------
def load_template_pairs(path: Path) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    ascii_vals: list[float] = []
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sweep_axis"] != "baseline":
                continue
            template = row["template_id"]
            tokenizer = row["tokenizer_id"]
            grouped[template][tokenizer] = row
    for template, per_tok in grouped.items():
        cl = per_tok.get("openai/tiktoken/cl100k_base")
        o2 = per_tok.get("openai/tiktoken/o200k_base")
        if not cl or not o2:
            continue
        delta = float(cl["tokens_per_100_chars"]) - float(o2["tokens_per_100_chars"])
        ascii_val = float(cl.get("ascii_ratio_bytes") or o2.get("ascii_ratio_bytes") or "0.0")
        deltas.append(delta)
        ascii_vals.append(ascii_val)
    return ascii_vals, deltas


fig, axes = plt.subplots(rows_fig, cols, figsize=(cols * 4, rows_fig * 3), sharex=False, sharey=False)
axes = np.atleast_1d(axes).flatten()
for ax in axes:
    ax.axis("off")

for ax, lang_entry in zip(axes, delta_summary):
    lang = lang_entry["language"]
    ascii_vals, deltas = load_template_pairs(Path(EFFICIENCY_FILES[lang]))
    if not ascii_vals:
        continue
    ax.axis("on")
    ax.scatter(ascii_vals, deltas, alpha=0.35, s=14)

    bin_edges = np.linspace(0, 1, 11)
    bin_centers: list[float] = []
    bin_medians: list[float] = []
    for idx in range(len(bin_edges) - 1):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        in_bin = [delta for ascii_ratio, delta in zip(ascii_vals, deltas) if (left <= ascii_ratio < right) or (idx == len(bin_edges) - 2 and math.isclose(ascii_ratio, right))]
        if not in_bin:
            continue
        center = (left + right) / 2
        bin_centers.append(center)
        bin_medians.append(statistics.median(in_bin))
    if bin_centers:
        ax.plot(bin_centers, bin_medians, color="black", linewidth=2, marker="o", markersize=4)
    ax.set_title(LANG_DISPLAY.get(lang, lang))
    ax.set_xlabel("ASCII ratio")
    ax.set_ylabel("Δ tokens")
    ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "ascii_sensitivity.png", dpi=300)
plt.close(fig)

# --------- Table A: Language summary ---------
p_values = []
for entry in delta_summary:
    p = entry.get("wilcoxon_p")
    if p is None or math.isnan(p):
        p = 1.0
    p_values.append(p)

# Benjamini-Hochberg adjustment
indexed = sorted(enumerate(p_values), key=lambda x: x[1])
q_values = [1.0] * len(p_values)
for rank, (idx, p) in enumerate(indexed, start=1):
    q = p * len(p_values) / rank
    q_values[idx] = min(q, 1.0)

header_line = "Language & Median $\\Delta$ & 95\\% CI & Cliff's $\\delta$ & $p$-value & $q$ (BH) " + r"\\ "
lines = [
     r"\begin{tabular}{lccccc}",
     r"\toprule",
     header_line,
     r"\midrule",
]
for entry, q in zip(delta_summary, q_values):
    lang = LANG_DISPLAY.get(entry["language"], entry["language"])
    median = entry["median_delta"]
    ci_low = entry["ci_low"]
    ci_high = entry["ci_high"]
    cliffs = entry["cliffs_delta"]
    p_val = entry.get("wilcoxon_p")
    p_str = f"{p_val:.2e}" if p_val is not None and not math.isnan(p_val) else "--"
    q_str = f"{q:.2e}" if q is not None and not math.isnan(q) else "--"
    line = f"{lang} & {median:.2f} & [{ci_low:.2f}, {ci_high:.2f}] & {cliffs:.2f} & {p_str} & {q_str} \\\\"
    lines.append(line)
lines.extend([
    r"\bottomrule",
    r"\end{tabular}"
])
(TABLES_DIR / "table_language_summary.tex").write_text("\n".join(lines), encoding="utf-8")

# --------- Table E: Merge examples ---------
example_rows: list[tuple[str, str, int, str]] = []
for tok_id in TOKENIZER_IDS:
    entries = leading_examples.get(tok_id, [])[:4]
    if not entries:
        continue
    for token, count in entries:
        latex_token = format_token(token)
        note = "space+Devanagari" if any(is_devanagari(ch) for ch in token) else ""
        label = TOKENIZER_LABELS.get(tok_id, tok_id)
        latex_label = r"\texttt{" + label.replace("_", r"\_") + "}"
        example_rows.append((latex_label, latex_token, count, note))

lines = [
    r"\begin{tabular}{l l c l}",
    r"\toprule",
    r"Tokenizer & Token example & Count & Note \\ ",
    r"\midrule",
]
for tok_label, token_display, count, note in example_rows:
    row = f"{tok_label} & {token_display} & {count} & {note}"
    lines.append(row + r" \\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(TABLES_DIR / "table_merge_examples.tex").write_text("\n".join(lines), encoding="utf-8")

