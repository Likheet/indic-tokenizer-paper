import csv
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tokenizer-output"
TARGETS = (
    "openai/tiktoken/cl100k_base",
    "openai/tiktoken/o200k_base",
)
LANGUAGES = {
    "English": "english-fast.csv",
    "Script-Mixed Hindi-English": "hinglish-fast.csv",
    "Hindi": "hindi-fast.csv",
    "Kannada": "kannada-fast.csv",
    "Tamil": "tamil-fast.csv",
}


def load_language(filename: str):
    per_tokenizer: dict[str, list[float]] = {}
    per_template: dict[str, dict[str, float]] = {}

    csv_path = ROOT / filename
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["sweep_axis"] != "baseline":
                continue
            tokenizer = row["tokenizer_id"]
            value = float(row["tokens_per_100_chars"])
            per_tokenizer.setdefault(tokenizer, []).append(value)
            if tokenizer in TARGETS:
                per_template.setdefault(row["template_id"], {})[tokenizer] = value

    return per_tokenizer, per_template


def safe_median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty sequence")
    return statistics.median(values)


def compute_summary():
    rows = []
    for language, filename in LANGUAGES.items():
        per_tokenizer, per_template = load_language(filename)
        cl_values = per_tokenizer.get(TARGETS[0], [])
        o_values = per_tokenizer.get(TARGETS[1], [])
        if not cl_values or not o_values:
            raise RuntimeError(f"Missing baseline rows for {language}")

        cl_median = safe_median(cl_values)
        o_median = safe_median(o_values)
        delta = cl_median - o_median
        pct_savings = (delta / cl_median * 100) if cl_median else 0.0
        cost_cl = cl_median * 0.1
        cost_o = o_median * 0.1

        dominant = 0
        ties = 0
        total = 0
        for template_id, pair in per_template.items():
            if TARGETS[0] in pair and TARGETS[1] in pair:
                total += 1
                if pair[TARGETS[0]] > pair[TARGETS[1]]:
                    dominant += 1
                elif pair[TARGETS[0]] == pair[TARGETS[1]]:
                    ties += 1
        rows.append(
            {
                "language": language,
                "cl_median": cl_median,
                "o_median": o_median,
                "delta": delta,
                "%_savings": pct_savings,
                "cost_cl": cost_cl,
                "cost_o": cost_o,
                "dominant": dominant,
                "ties": ties,
                "total": total,
                "notes": "—" if language == "English" or total == 0 else f"{dominant} / {total}",
            }
        )
    return rows


def round_decimal(value: float, places: int) -> Decimal:
    quant = Decimal("1").scaleb(-places)
    return Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)


def format_float(value: float, places: int = 2) -> str:
    return f"{round_decimal(value, places):.{places}f}"


def main():
    rows = compute_summary()
    header = [
        "Language",
        "cl100k tokens/100",
        "o200k tokens/100",
        "Δ tokens/100",
        "% savings",
        "Cost cl ($/1M chars)",
        "Cost o ($/1M chars)",
        "Dominant pairs",
        "Ties",
        "Total pairs",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    row["language"],
                    format_float(row["cl_median"]),
                    format_float(row["o_median"]),
                    format_float(row["delta"]),
                    format_float(row["%_savings"]),
                    format_float(row["cost_cl"]),
                    format_float(row["cost_o"]),
                    row["notes"] if row["notes"] != "—" else row["notes"],
                    str(row["ties"]),
                    str(row["total"]),
                ]
            )
        )

    print("\nMarkdown table:\n")
    print("|Language|cl100k tokens/100 (median)|o200k tokens/100 (median)|Δ tokens/100|% savings vs cl100k|Cost per 1M chars (cl)$|Cost per 1M chars (o)$|Dominant pairs|")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        percent = format_float(row["%_savings"], 1)
        markdown_row = "|{language}|{cl}|{o}|{delta}|{pct}%|{cost_cl}|{cost_o}|{pairs}|".format(
            language=row["language"],
            cl=format_float(row["cl_median"]),
            o=format_float(row["o_median"]),
            delta=format_float(row["delta"]),
            pct=percent,
            cost_cl=format_float(row["cost_cl"]),
            cost_o=format_float(row["cost_o"]),
            pairs=row["notes"] if row["notes"] != "—" else "—",
        )
        print(markdown_row)


if __name__ == "__main__":
    main()
