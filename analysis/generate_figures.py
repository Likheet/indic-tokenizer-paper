"""Generate publication figures for baseline medians and ASCII-ratio sweeps."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Colour palette shared across plots
COLOURS: Dict[str, str] = {
    "openai/tiktoken/o200k_base": "#1b9e77",  # green
    "openai/tiktoken/cl100k_base": "#d95f02",  # orange
    "ai4bharat/IndicBERTv2-MLM-only": "#7570b3",  # purple
}
LABELS: Dict[str, str] = {
    "openai/tiktoken/o200k_base": "o200k_base",
    "openai/tiktoken/cl100k_base": "cl100k_base",
    "ai4bharat/IndicBERTv2-MLM-only": "IndicBERTv2",
}
LANGUAGE_ORDER: List[str] = ["English", "Hindi", "Kannada", "Tamil", "Script-Mixed Hindi-English"]


def load_baseline_summary() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "all_languages_baseline_summary.csv")
    wanted = df["tokenizer_id"].isin(COLOURS.keys())
    subset = df.loc[wanted].copy()
    if subset.empty:
        raise RuntimeError("Selected tokenizers not found in baseline summary")
    subset["tokenizer_label"] = subset["tokenizer_id"].map(LABELS)
    subset.sort_values(["language", "tokenizer_label"], inplace=True)
    subset["language"].replace({"Hinglish": "Script-Mixed Hindi-English"}, inplace=True)
    return subset


def plot_baseline_bars(df: pd.DataFrame) -> Path:
    pivot = (
        df.pivot(index="language", columns="tokenizer_label", values="tokens_per_100_chars_median")
        .reindex(LANGUAGE_ORDER)
    )
    # Ensure all requested tokenizers exist for each language
    missing = [tok for tok in LABELS.values() if tok not in pivot.columns]
    if missing:
        raise RuntimeError(f"Missing baseline medians for: {', '.join(missing)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    x = range(len(pivot.index))
    total = len(LABELS)
    bar_width = 0.2
    offsets = {
        label: (i - (total - 1) / 2) * bar_width for i, label in enumerate(LABELS.values())
    }

    for label, offset in offsets.items():
        ax.bar(
            [i + offset for i in x],
            pivot[label],
            width=bar_width,
            label=label,
            color=COLOURS[next(k for k, v in LABELS.items() if v == label)],
        )

    ax.set_ylabel("Median tokens per 100 chars")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, math.ceil(float(pivot.values.max()) / 20) * 20)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Baseline token cost by language")
    ax.set_xlabel("Language slice")

    fig.tight_layout()
    output_path = FIGURES_DIR / "baseline_tokens_bar.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def load_ascii_sweep(language: str = "Script-Mixed Hindi-English") -> pd.DataFrame:
    slug_map = {"Script-Mixed Hindi-English": "hinglish"}
    slug = slug_map.get(language, language.lower().replace(" ", "_"))
    csv_path = ROOT / "tokenizer-output" / f"{slug}-fast.csv"
    usecols = ["sweep_axis", "tokenizer_id", "x_value", "tokens_per_100_chars"]
    df = pd.read_csv(csv_path, usecols=usecols)
    ascii_df = df[(df["sweep_axis"] == "ascii_ratio") & (df["tokenizer_id"].isin(COLOURS.keys()))].copy()
    if ascii_df.empty:
        raise RuntimeError(f"ASCII-ratio sweep rows missing for {language}")
    ascii_df["ascii_ratio"] = ascii_df["x_value"].astype(float)
    ascii_df["tokens_per_100_chars"] = ascii_df["tokens_per_100_chars"].astype(float)
    return ascii_df


def plot_ascii_panel(df: pd.DataFrame) -> Path:
    grouped = df.groupby(["tokenizer_id", "ascii_ratio"])["tokens_per_100_chars"].median().reset_index()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for tokenizer_id, group in grouped.groupby("tokenizer_id"):
        label = LABELS[tokenizer_id]
        group_sorted = group.sort_values("ascii_ratio")
        ax.plot(
            group_sorted["ascii_ratio"],
            group_sorted["tokens_per_100_chars"],
            marker="o",
            linewidth=2.0,
            label=label,
            color=COLOURS[tokenizer_id],
        )

    ax.set_xlabel("ASCII ratio of input line")
    ax.set_ylabel("Median tokens per 100 chars")
    ax.set_title("ASCII mix vs token cost (Script-Mixed Hindi-English sweep)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = FIGURES_DIR / "ascii_ratio_panel.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    baseline = load_baseline_summary()
    baseline_path = plot_baseline_bars(baseline)

    ascii_df = load_ascii_sweep(language="Script-Mixed Hindi-English")
    ascii_path = plot_ascii_panel(ascii_df)

    print("Generated figures:")
    print(f" - {baseline_path.relative_to(ROOT)}")
    print(f" - {ascii_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
