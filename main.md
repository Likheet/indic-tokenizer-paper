---
title: "Tokenizer Benchmarking Across Indic and Code-Mixed Scripts"
author: "Likheet Shetty"
bibliography: refs.bib
csl: ieee.csl
---

# Tokenizer Benchmarking Across Indic and Code-Mixed Scripts

---
title: "Tokenizer Benchmarking Across Indic and Code-Mixed Scripts"
author: "Likheet Shetty"
bibliography: refs.bib
csl: ieee.csl
---

# Tokenizer Benchmarking Across Indic and Code-Mixed Scripts

**Author:** Likheet Shetty, Upcoming MS in IT (AI), UNSW

## Abstract

**Tokenization** determines cost, context budgets, and prompt robustness in LLM deployments, yet production tokenizers remain uncompared across **Indic scripts** and **code-mixed** text. We benchmarked 11 widely-used tokenizers (OpenAI, Meta, Mistral, and Indic-trained models) on English, Hindi, Kannada, Tamil, and script-mixed Hinglish using a browser-based measurement harness.

**Key findings.** OpenAI's `o200k_base` reduces token counts by roughly 55-70% compared to `cl100k_base` on Indic text (HI 54.7%, KN 69.7%, TA 60.6%), ~29.3% on Hinglish, and ~4.1% on English (paired Wilcoxon tests, BH-corrected). Byte-fallback tokenizers produced zero `[UNK]` on the baselines and showed stable behavior under noisy inputs (URLs, emoji, normalization variants). Indic-trained WordPiece models are most compact on pure Indic text but fragment under ASCII mixing. We provide routing logic and monitoring counters for production deployment focused on within-line script mixing (Devanagari + Latin).

All results derive from released CSV artifacts; the browser-based harness enables replication without GPUs or servers.

## Introduction

Tokenization is the first irreversible decision in an LLM pipeline. Once text is segmented, downstream steps, such as attention budget allocation, truncation risk, and cost, are determined. Inefficient tokenization inflates costs and reduces usable context. Brittle tokenization (high `[UNK]` or erratic fragmentation) harms robustness for real-world inputs containing romanized words, code-mixing, emoji, hashtags, URLs, or mixed Unicode normalization. We evaluate common production tokenizers on Indic and code-mixed inputs to provide model-agnostic operational guidance.

### Research questions
- RQ1: How do byte-fallback and WordPiece allocations trade off efficiency and robustness on Indic and code-mixed baselines?
- RQ2: How does ASCII mixing shift token budgets across evaluated tokenizers?
- RQ3: Which merge patterns (e.g., leading-space + akshara tokens) correlate with reduced fragmentation?

### Hypotheses
- H1: Byte-fallback tokenizers guarantee coverage and stabilize token budgets under noisy or mixed-script inputs.
- H2: Indic WordPiece vocabularies achieve optimal compactness on pure Indic text but degrade as ASCII ratio increases.
- H3: Higher leading-space akshara and cross-script bigram coverage correlate with lower code-mixed fragmentation.

## Related work

Summarized background on BPE, WordPiece, SentencePiece, and multilingual tokenizer studies. See `refs.bib` for citations.

## Methods

### Measurement pipeline
We follow a CSV-first approach: results are computed from exported CSVs produced by a browser-only harness that loads tokenizers via Transformers.js or `tiktoken` (WASM). Per-row metrics include: tokens per 100 characters, bytes per token, `[UNK]` incidence, fragmentation entropy, and timing metadata. Each export is versioned and checksummed; every row records tokenizer id, artifact hash, library versions, commit SHA, OS, and timestamp for offline analysis.

### Baseline corpus and sampling
Input pools live under `tokenizer-input/<language>.txt` and consist of PII-scrubbed snippets (support text, short news, product help). We normalize to NFC, preserve ZWJ, deduplicate deterministically, and sample 100 unique baselines per language (fixed seed). Slices and filters:

- EN / HI / KN / TA: target script covers ≥ 90% of Unicode code points.
- Hinglish: each line contains ≥ 3 Devanagari and ≥ 3 Latin code points.
- All slices: trim whitespace, enforce NFC, forbid ZWJ stripping.

Baseline length (median; 10th-90th): EN 45.0 (37.0/53.0), HI 47.5 (37.0/60.1), Hinglish 34.0 (25.9/43.2), KN 46.5 (36.9/59.0), TA 49.0 (38.0/67.0). Appendix A lists per-template metadata and ASCII ratios.

**Definitions.** ASCII ratio (bytes) = ASCII byte count / total bytes. HLΔ is the Hodges-Lehmann median paired difference for per-template token-count differences between `cl100k_base` and `o200k_base`.

### Tokenizer inventory
We evaluate 11 production tokenizers across BPE-family and Indic-focused WordPiece/SentencePiece models. Key candidates include:

- BPE-family: `openai/tiktoken/o200k_base`, `openai/tiktoken/cl100k_base`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Xenova/distilgpt2`.
- Indic-focused: `ai4bharat/IndicBERTv2-MLM-only`, `InvincibleSloth/muril-tokenizer`, `Xenova/xlm-roberta-base`, `Xenova/bert-base-multilingual-uncased`, `Xenova/bert-base-uncased`, `Xenova/t5-small`.

Tokenizers must load deterministically in-browser, be operationally relevant, and have compatible licensing for redistribution or runtime fetching. Hosted or frontier-only tokenizers (GPT-4o, Claude, Gemini) and models that need native extensions are excluded.

### Preset configuration
All exports use the **Full** preset: `sampleLines=100`, `repeats=5` (collapsed by median), baseline condition (NFC, ZWJ stressor off, emoji/URL off, `add_special_tokens=false`), and sweeps for ASCII ratio (11 bins), emoji counts, URL toggles, normalization flips, ZWJ toggles, and light perturbations. Each tokenizer–slice yields 2,230 rows (24,530 per slice with the sweep). Baseline rows are `sweep_axis == "baseline"` and stressors are analyzed as deltas relative to baseline medians.

## Results

Figure: Baseline median tokens/100 for `o200k_base`, `cl100k_base`, and `IndicBERTv2` across EN/HI/KN/TA/Hinglish.

![Baseline medians](figures/baseline_tokens_bar.png)

### Baseline medians
`o200k_base` shows superior compactness relative to `cl100k_base` on Hindi, Kannada, Tamil, and Hinglish, while remaining near parity on English. For EN, 74/100 pairs tie and 26/100 favor `o200k_base`; HLΔ equals 0.0 tokens/100 with p ≈ 4.135×10⁻⁶ (paired Wilcoxon, BH-corrected). Indic-aware tokenizers (IndicBERTv2, MuRIL) are most compact on pure Indic text. See Appendix A for full tables (A1-A5).

#### Cost example (USD per 1M characters at $0.01 per 1k tokens)

| Tokenizer | EN | HI | KN | TA | Hinglish |
|---|---:|---:|---:|---:|---:|
| `cl100k_base` | 3.51 | 12.43 | 19.53 | 16.67 | 6.00 |
| `o200k_base` | 3.36 | 5.63 | 5.93 | 6.57 | 4.24 |

### Effect sizes
Median paired differences (cl100k − o200k), bootstrap intervals, Cliff's δ, and Wilcoxon p-values are reported in Appendix A. Reductions exceed ~66 tokens/100 on Hindi and ~133 tokens/100 on Kannada.

## Analysis

### Vocabulary allocation metrics
`o200k_base` emits frequent space-prefixed Devanagari merges (e.g., leading-space + क्ष), bundling whitespace with aksharas. `cl100k_base` tends to produce single-codepoint or byte splits. IndicBERTv2 emits fewer space-prefixed merges.

![Merge metrics snapshot](figures/merge_metrics_bar.png)

### ASCII sensitivity and routing
Tokens/100 differences (cl100k − o200k) decrease as ASCII ratio increases but remain positive across bins. Treat ASCII ratio as a monotonic cost signal rather than a hard routing breakpoint.

![ASCII sensitivity](figures/ascii_sensitivity.png)

### Convergence
Medians stabilize well before the n=100 template budget (see `figures/convergence_curves.png`).

## Discussion

Two factors explain `o200k_base`'s advantage:

1. **Merge coverage.** The larger merge vocabulary (200k) captures multi-byte Indic clusters, emoji, and URL substrings that would otherwise be byte-split.
2. **Byte fallback.** Guaranteed coverage yields zero `[UNK]` even for noisy or mixed inputs.

Operational recommendations:

- **Default:** use `o200k_base` universally.
- **Optional router:** route single-script Indic lines with ASCII ratio < 0.25 to `ai4bharat/IndicBERTv2-MLM-only`; auto-fallback to `o200k_base` on `[UNK]`.
- **Unicode hygiene:** normalize to NFC, preserve ZWJ, leave emoji/URLs untouched.
- **Monitoring:** tokens/100 median, `[UNK]` incidence, fragmentation dispersion per slice.

## Limitations

- **Scope:** intrinsic metrics only (tokens/100, bytes/token, `[UNK]`, fragmentation); no throughput or downstream accuracy reported.
- **Coverage:** five slices (EN, HI, KN, TA, Hinglish). Other scripts and Romanized Hindi require new data.
- **Input length:** baselines are short-to-medium lines; longer documents may differ.
- **Library/version dependence:** results are tied to specific library versions (e.g., `transformersjs=2.17.2`, `tiktoken=1.0.22`).

## Conclusion

TokenizerLab's CSV-first benchmark shows `o200k_base` is the best single-tokenizer choice for Indic and code-mixed deployments: it is `[UNK]`-free, reduces costs vs `cl100k_base` on Indic scripts, and preserves parity on English. A lightweight router can optionally direct pure Indic text to IndicBERTv2 for further cost gains. Follow NFC normalization and preserve ZWJ; track tokens/100, `[UNK]`, and dispersion to detect drift.

## Appendix A — Per-language efficiency summary

Median Δ (tokens/100 chars), 95% CI, Cliff's δ, and p/q-values (BH) for `o200k_base` vs `cl100k_base`:

| Language | Median Δ | 95% CI | Cliff's δ | p-value | q (BH) |
|---|---:|---|---:|---:|---:|
| English | 0.00 | [0.00, 0.00] | 0.26 | 4.26e-06 | 4.26e-06 |
| Hindi | 66.15 | [61.38, 69.70] | 1.00 | 0.00e+00 | 0.00e+00 |
| Kannada | 133.81 | [126.67, 145.30] | 1.00 | 0.00e+00 | 0.00e+00 |
| Tamil | 101.43 | [94.52, 106.06] | 1.00 | 0.00e+00 | 0.00e+00 |
| Hinglish | 16.67 | [15.38, 18.18] | 1.00 | 0.00e+00 | 0.00e+00 |

## Appendix B — Merge exemplars

Frequent leading-space merges observed across Hindi/Hinglish baselines (token counts from debug sampling):

| Tokenizer | Token example | Count | Note |
|---|---|---:|---|
| `o200k_base` | `<space>क्ष` | 200 | space + Devanagari |
| `o200k_base` | `<space>है` | 42 | space + Devanagari |
| `o200k_base` | `<space>करो` | 40 | space + Devanagari |
| `o200k_base` | `<space>पर` | 34 | space + Devanagari |
| `cl100k_base` | `<space>क` | 252 | space + Devanagari |
| `cl100k_base` | `<space>ह` | 70 | space + Devanagari |
| `cl100k_base` | `<space>म` | 55 | space + Devanagari |
| `cl100k_base` | `<space>प` | 34 | space + Devanagari |

Bibliography is available in `refs.bib`.

## Reproducibility & artifacts

Input CSVs used: `tokenizer-output/english-fast.csv`, `hindi-fast.csv`, `hinglish-fast.csv`, `kannada-fast.csv`, `tamil-fast.csv`.

For Pandoc/Quarto renders include `refs.bib` when compiling to enable citations.

| `cl100k_base` | `<space>म` | 55 | space + Devanagari |
| `cl100k_base` | `<space>प` | 34 | space + Devanagari |
% NeurIPS 2025-compliant main.tex (MD-aligned content)
% This version aligns wording/emphasis with the provided main.md while
% keeping NeurIPS formatting and robust compilation under XeLaTeX/LuaLaTeX.

\documentclass{article}
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage[preprint]{neurips_2025}

% Encoding, fonts, links
\usepackage{iftex}

\ifPDFTeX%
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \PackageError{main}{This document must be compiled with XeLaTeX or LuaLaTeX to render Indic characters}{Use xelatex main.tex}
\else
    \usepackage{fontspec}
    \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}
    \setmainfont{Times New Roman}
    \setsansfont{Arial}
    \setmonofont{Consolas}
    \newfontfamily\devanagarifont[Script=Devanagari,ItalicFont=Nirmala UI]{Nirmala UI}
    \newcommand{\devtxt}[1]{{\devanagarifont#1}}
    \newcommand{\devcode}[1]{\texttt{{\devanagarifont#1}}}
\fi

\usepackage[hidelinks]{hyperref}
\usepackage{url}
\urlstyle{same}

% Math, tables, graphics, spacing
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage{array}
\usepackage{textcomp}
\renewcommand{\textvisiblespace}{\texttt{\textless{}space\textgreater{}}}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
% Prefer ragged-right captions to avoid tight full-justification that often
% produces underfull/overfull warnings in narrow floats. Keep single-line
% captions left-aligned as well.
\captionsetup{justification=raggedright,singlelinecheck=false}
\usepackage{titlesec}

\makeatletter
\renewcommand{\@noticestring}{}
\makeatother

% Improve line-breaking tolerance to reduce underfull/overfull boxes in tight layouts
% This is a coarse but safe global setting; can be removed if you prefer stricter layout.
\emergencystretch=2em

\captionsetup[table]{skip=6pt}
\captionsetup[figure]{skip=6pt}
 % Removed custom title formats to rely on default section styling


\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
% Slightly reduce inter-column padding to give tables more horizontal room
% and reduce the chance of overfull hboxes in tight tables.
\setlength{\tabcolsep}{3pt}
\setlength{\bibsep}{0pt plus 0.3pt}
\setlength{\bibhang}{1.0em}

% Unicode guards for a few common glyphs (pdfLaTeX only)
\ifPDFTeX%
    \DeclareUnicodeCharacter{0394}{\ensuremath{\Delta}} % Δ
    \DeclareUnicodeCharacter{2248}{\ensuremath{\approx}} % ≈
    \DeclareUnicodeCharacter{2011}{-} % non-breaking hyphen
    \DeclareUnicodeCharacter{2212}{-} % minus sign
\fi

% Figures live in ./figures
\graphicspath{{figures/}}

% Macros
\newcommand{\unk}{\texttt{[UNK]}}
---
title: "Tokenizer Benchmarking Across Indic and Code-Mixed Scripts"
author: "Likheet Shetty — Upcoming MS in IT (AI), UNSW — likheet.s@gmail.com"
date: ""
---

# Tokenizer Benchmarking Across Indic and Code-Mixed Scripts

**Authors:** Likheet Shetty, Upcoming MS in IT (AI), UNSW

## Abstract

**Tokenization** critically determines cost, context budgets, and prompt robustness in LLM deployments, yet production tokenizers remain uncompared across **Indic scripts** and **code-mixed** text. We benchmark 11 widely-used tokenizers (OpenAI, Meta, Mistral, and Indic-trained models) on English, Hindi, Kannada, Tamil, and script-mixed Hinglish using a browser-based measurement harness.

**Key findings.** OpenAI's `o200k_base` requires roughly 55–70% fewer tokens than `cl100k_base` on Indic text (HI 54.7%, KN 69.7%, TA 60.6%), ~29.3% fewer on Hinglish, and ~4.1% fewer on English (paired Wilcoxon tests, BH-corrected). Byte-fallback tokenizers have zero `[UNK]` on the baselines and show stable behavior under noisy inputs (URLs, emoji, normalization variants). Indic-trained WordPiece models are most compact on pure Indic text but fragment under ASCII mixing. We provide routing logic and monitoring counters for production deployment focused on within-line script mixing (Devanagari + Latin).

All results derive from released CSV artifacts; the browser-based harness enables replication without GPUs or servers.

## Introduction

Tokenization is the first irreversible decision in an LLM pipeline. Once text is segmented, downstream steps, such as attention budget allocation, truncation risk, and cost, are determined. Inefficient tokenization inflates token counts, increasing cost and reducing usable context. Brittle tokenization (high `[UNK]` or erratic fragmentation) harms robustness for real-world inputs containing romanized words, code-mixing, emoji, hashtags, URLs, or mixed Unicode normalization. This paper evaluates common production tokenizers on Indic and code-mixed inputs to provide model-agnostic operational guidance.

### Research questions
- RQ1: How do byte-fallback and WordPiece allocations trade off efficiency and robustness on Indic and code-mixed baselines?
- RQ2: How does ASCII mixing shift token budgets across evaluated tokenizers?
- RQ3: Which merge patterns (e.g., leading-space + akshara tokens) correlate with reduced fragmentation?

### Hypotheses
- H1: Byte-fallback tokenizers guarantee coverage and stabilize token budgets under noisy or mixed-script inputs.
- H2: Indic WordPiece vocabularies achieve optimal compactness on pure Indic text but degrade as ASCII ratio increases.
- H3: Higher leading-space akshara and cross-script bigram coverage correlate with lower code-mixed fragmentation.

## Related work

### Tokenization algorithms
Byte Pair Encoding (BPE) fits a fixed merge table from a corpus. Byte-level BPE (GPT-2 style) operates on UTF-8 bytes to guarantee coverage. WordPiece trains merges optimized for LM likelihood and can favor granularity for morphologically rich scripts. SentencePiece trains using a unigram language model and optionally supports byte-fallback and normalization.

### Multilingual and Indic tokenization
Work such as IndicBERT, MuRIL, and XLM‑R report downstream gains across Indic languages but typically omit intrinsic tokenization diagnostics or code-mixed fragmentation analysis. This study fills that gap by focusing on segmentation metrics (tokens per 100 chars, bytes/token, `[UNK]`, fragmentation) across five slices.

### Production tokenization notes
Open-source tokenizer release notes (e.g., `tiktoken`) describe practical trade-offs; recent releases introduced `o200k_base` for long-context deployments. Our benchmark isolates segmentation behavior on mixed-script inputs and proposes simple operational policies.

## Methods

### Measurement pipeline
We follow a CSV-first approach: results are computed from exported CSVs produced by a browser-only harness that loads tokenizers via Transformers.js or `tiktoken` (WASM). Per-row metrics include: tokens per 100 characters, bytes per token, `[UNK]` incidence, fragmentation entropy, and timing metadata. Each export is versioned and checksummed; every row records tokenizer id, artifact hash, library versions, commit SHA, OS, and timestamp for offline analysis.

### Baseline corpus and sampling
Input pools live under `tokenizer-input/<language>.txt` and consist of PII-scrubbed snippets (support text, short news, product help). We normalize to NFC, preserve ZWJ, deduplicate deterministically, and sample 100 unique baselines per language (fixed seed). Slices and filters:

- EN / HI / KN / TA: target script covers ≥ 90% of Unicode code points.
- Hinglish: each line contains ≥ 3 Devanagari and ≥ 3 Latin code points.
- All slices: trim whitespace, enforce NFC, forbid ZWJ stripping.

Baseline length (median; 10th-90th): EN 45.0 (37.0/53.0), HI 47.5 (37.0/60.1), Hinglish 34.0 (25.9/43.2), KN 46.5 (36.9/59.0), TA 49.0 (38.0/67.0). Appendix A lists per-template metadata and ASCII ratios.

**Definitions.** ASCII ratio (bytes) = ASCII byte count / total bytes. HL$\Delta$ is the Hodges-Lehmann median paired difference for per-template token-count differences between `cl100k_base` and `o200k_base`.

### Tokenizer inventory
We evaluate 11 production tokenizers across BPE-family and Indic-focused WordPiece/SentencePiece models. Key candidates include:

- BPE-family: `openai/tiktoken/o200k_base`, `openai/tiktoken/cl100k_base`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Xenova/distilgpt2`.
- Indic-focused: `ai4bharat/IndicBERTv2-MLM-only`, `InvincibleSloth/muril-tokenizer`, `Xenova/xlm-roberta-base`, `Xenova/bert-base-multilingual-uncased`, `Xenova/bert-base-uncased`, `Xenova/t5-small`.

Tokenizers must load deterministically in-browser, be operationally relevant, and have compatible licensing for redistribution or runtime fetching. Hosted or frontier-only tokenizers (GPT-4o, Claude, Gemini) and models that need native extensions are excluded.

### Preset configuration
All exports use the **Full** preset: `sampleLines=100`, `repeats=5` (collapsed by median), baseline condition (NFC, ZWJ stressor off, emoji/URL off, `add_special_tokens=false`), and sweeps for ASCII ratio (11 bins), emoji counts, URL toggles, normalization flips, ZWJ toggles, and light perturbations. Each tokenizer–slice yields 2,230 rows (24,530 per slice with the sweep). Baseline rows are `sweep_axis == "baseline"` and stressors are analyzed as deltas relative to baseline medians.

## Results

Figure: Baseline median tokens/100 for `o200k_base`, `cl100k_base`, and `IndicBERTv2` across EN/HI/KN/TA/Hinglish.

![Baseline medians](figures/baseline_tokens_bar.png)

### Baseline medians
`o200k_base` shows superior compactness relative to `cl100k_base` on Hindi, Kannada, Tamil, and Hinglish, while remaining near parity on English. For EN, 74/100 pairs tie and 26/100 favor `o200k_base`; HL$\Delta$ equals 0.0 tokens/100 with $p\approx 4.135\times10^{-6}$ (paired Wilcoxon, BH-corrected). Indic-aware tokenizers (IndicBERTv2, MuRIL) are most compact on pure Indic text. See Appendix A for full tables (A1-A5).

#### Cost example (USD per 1M characters at $0.01 per 1k tokens)

| Tokenizer | EN | HI | KN | TA | Hinglish |
|---|---:|---:|---:|---:|---:|
| `cl100k_base` | 3.51 | 12.43 | 19.53 | 16.67 | 6.00 |
| `o200k_base` | 3.36 | 5.63 | 5.93 | 6.57 | 4.24 |

### Effect sizes
Median paired differences (cl100k − o200k), bootstrap intervals, Cliff's δ, and Wilcoxon p-values are reported in Appendix A. Reductions exceed ~66 tokens/100 on Hindi and ~133 tokens/100 on Kannada.

## Analysis

### Vocabulary allocation metrics
`o200k_base` emits frequent space-prefixed Devanagari merges (e.g., leading-space + क्ष), bundling whitespace with aksharas. `cl100k_base` tends to produce single-codepoint or byte splits. IndicBERTv2 emits fewer space-prefixed merges.

![Merge metrics snapshot](figures/merge_metrics_bar.png)

### ASCII sensitivity and routing
Tokens/100 differences (cl100k − o200k) decrease as ASCII ratio increases but remain positive across bins. Treat ASCII ratio as a monotonic cost signal rather than a hard routing breakpoint.

![ASCII sensitivity](figures/ascii_sensitivity.png)

### Convergence
Medians stabilize well before the n=100 template budget (see `figures/convergence_curves.png`).

## Discussion

Two factors explain `o200k_base`'s advantage:

1. **Merge coverage.** The larger merge vocabulary (200k) captures multi-byte Indic clusters, emoji, and URL substrings that would otherwise be byte-split.
2. **Byte fallback.** Guaranteed coverage yields zero `[UNK]` even for noisy or mixed inputs.

Operationally:

- Default: use `o200k_base` universally.
- Optional router: route single-script Indic lines with ASCII ratio < 0.25 to `ai4bharat/IndicBERTv2-MLM-only`; auto-fallback to `o200k_base` on `[UNK]`.
- Unicode hygiene: normalize to NFC, preserve ZWJ, leave emoji/URLs untouched.
- Monitoring: tokens/100 median, `[UNK]` incidence, fragmentation dispersion per slice.

## Limitations

- Scope: intrinsic metrics only (tokens/100, bytes/token, `[UNK]`, fragmentation); no throughput or downstream accuracy reported.
- Coverage: five slices (EN, HI, KN, TA, Hinglish). Other scripts and Romanized Hindi require new data.
- Input length: baselines are short-to-medium lines; longer documents may differ.
- Library/version dependence: results are tied to specific library versions (e.g., `transformersjs=2.17.2`, `tiktoken=1.0.22`).

## Conclusion

TokenizerLab's CSV-first benchmark shows `o200k_base` is the best single-tokenizer choice for Indic and code-mixed deployments: it is `[UNK]`-free, reduces costs vs `cl100k_base` on Indic scripts, and preserves parity on English. A lightweight router can optionally direct pure Indic text to IndicBERTv2 for further cost gains. Follow NFC normalization and preserve ZWJ; track tokens/100, `[UNK]`, and dispersion to detect drift.

## Appendices

- Appendix A: Baseline medians, IQRs, cost tables, and `[UNK]` incidence (CSV-derived tables A1-A5).
- Appendix B: Validity, ethics, risk assessment, Unicode hygiene guidelines.
- Appendix C: Implementation guide (policy, monitoring, deployment sequence, incident response).
- Appendix D: Reproducibility protocol and instructions to regenerate tables/figures from the CSVs in `tokenizer-output/`.

## Reproducibility & artifacts

Input CSVs used: `tokenizer-output/english-fast.csv`, `hindi-fast.csv`, `hinglish-fast.csv`, `kannada-fast.csv`, `tamil-fast.csv`.

## References

Bibliography is available in `refs.bib`. For LaTeX/Pandoc workflows you can include the bibliography file when rendering.

Tokenizers must (i) load deterministically in the browser, (ii) have operational relevance, and (iii) expose licensing permitting redistribution or runtime fetching. Frontier-only or hosted tokenizers (GPT-4o, Claude, Gemini) and models requiring native extensions are excluded; Appendix~B summarizes the practical implications of these exclusions.

\subsection{Preset configuration}
All exports use the \textbf{Full} preset: \texttt{sampleLines=100}, \texttt{repeats=5} (each template tokenized five times to probe non-determinism; all repeats were identical in this release), baseline condition (NFC, ZWJ stressor off—no injection or removal—, no emoji/URL, \texttt{add\_special\_tokens=false}), plus sweeps for ASCII ratio (11 bins), emoji \textbf{(0/1/2/3/4/5)}, URL toggles, normalization flips (NFC/NFD), zero-width joiner toggles, and light perturbations. Each tokenizer\textendash{}slice pair yields \textbf{2,230} rows (\textbf{24,530} per slice). Baseline rows are filtered by \texttt{sweep\_axis == ``baseline''}; stressor rows are analyzed via deltas relative to the baseline median.

For all analyses, five repeats per \texttt{template\_id} were collapsed by the median, yielding 100 independent baselines per slice.


% ===================== RESULTS =====================

\section{Results}
Figure~\ref{fig:baseline-tokens} summarizes baseline medians for three deployment candidates (\texttt{o200k\_base}, \texttt{cl100k\_base}, and \texttt{IndicBERTv2}). Detailed per-tokenizer tables, cost micro-tables, and stressor snapshots are presented in Appendix~A.\@

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/baseline_tokens_bar.png}
\caption{Baseline median tokens/100 for \texttt{o200k\_base}, \texttt{cl100k\_base}, and \texttt{IndicBERTv2} across EN/HI/KN/TA/Hinglish. \texttt{o200k\_base} achieves \textasciitilde55\textendash{}70\% reductions on Indic scripts while maintaining near parity on English. Error bars represent the IQR.\@ Medians are computed on baseline rows only.}\label{fig:baseline-tokens}
\end{figure}

\subsection{Baseline medians}
	exttt{o200k\_base} demonstrates superior compactness compared to \texttt{cl100k\_base} on Hindi, Kannada, Tamil, and Hinglish (script-mixed Hindi\textendash{}English), while maintaining near parity on English. For English, 74/100 paired baselines show exact ties and 26/100 favor \texttt{o200k\_base}; the Hodges\textendash{}Lehmann estimator (HL$\Delta$) equals 0.0 tokens/100 with $p = 4.135\times10^{-6}$ (Wilcoxon, one-sided, paired; BH correction yields $q = 4.135\times10^{-6}$ for EN and all $q < 1\times10^{-8}$ for HI/KN/TA/Hinglish). Although Indic-aware tokenizers (IndicBERTv2, MuRIL) produce the fewest tokens on pure Indic baselines, median tokens/100 and bytes/token for all 11 tokenizers per slice are tabulated in Appendix~A (Tables A1\textendash{}A5), alongside cost per 1M characters computed as $0.1 \times$ (median tokens per 100 chars) at $\$0.01 / 1\text{k}$ tokens.

\begin{table}[t]
\centering
\caption{Cost per 1M characters (USD) at $\$0.01$ per 1k tokens.}
\begin{tabular}{lccccc}
\toprule
Tokenizer & EN & HI & KN & TA & Hinglish \\
\midrule
\texttt{cl100k\_base} & 3.51 & 12.43 & 19.53 & 16.67 & 6.00 \\
\texttt{o200k\_base} & 3.36 & 5.63 & 5.93 & 6.57 & 4.24 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Effect sizes across languages}
Table~\ref{tab:language-summary} presents median paired differences (\texttt{cl100k} $-$ \texttt{o200k}) with 95\% bootstrap intervals, Cliff's $\delta$, and Wilcoxon p-values (BH-corrected across slices). Reductions exceed 66 tokens/100 on Hindi and 133 tokens/100 on Kannada, with Hinglish showing a +16.67 tokens/100 shift, consistent with the \textasciitilde29\% reduction reported in the abstract.
\begin{table}[t]
    \centering
    \caption{Per-language efficiency summary for \texttt{o200k\_base} relative to \texttt{cl100k\_base}. Median $\Delta$ is reported in tokens/100 characters; Cliff's $\delta$ measures paired effect size.}\label{tab:language-summary}
    \input{tables/table_language_summary.tex}
\end{table}

% ===================== ANALYSIS =====================
\section{Analysis}

\subsection{Vocabulary allocation metrics}
Figure~\ref{fig:merge-metrics} compares qualitative leading-space behavior across Hindi and Hinglish samples. Qualitatively, \texttt{o200k\_base} emits frequent space-prefixed Devanagari merges (e.g., {\textvisiblespace}\devtxt{क्ष}, {\textvisiblespace}\devtxt{कर}), bundling whitespace with aksharas; \texttt{cl100k\_base} predominantly yields single-codepoint or byte-split pieces; \texttt{IndicBERTv2} does not emit space-prefixed merges. We omit numeric LSAR/SPR/CSBC values because our debug sampling under-reports absolute coverage.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/merge_metrics_bar.png}
    \caption{Vocabulary allocation snapshots across tokenizers. Each bar reflects qualitative counts drawn from debug sampling; we highlight frequent space-prefixed merges but omit numeric LSAR/SPR/CSBC rates due to sampling under-coverage.}\label{fig:merge-metrics}
\end{figure}

\subsection{ASCII sensitivity and routing threshold}
\begin{figure}[t]
    \centering
    \includegraphics[width=0.75\linewidth]{figures/hinglish_lsar_curve.png}
    \caption{Baseline LSAR medians for Hindi and Hinglish across the three deployment tokenizers. Byte-fallback encoders retain some leading-space merges, while IndicBERTv2 maintains the highest coverage on script-mixed text.}\label{fig:hinglish-lsar}
\end{figure}
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/ascii_sensitivity.png}
    \caption{Median tokens/100 differences (\texttt{cl100k} minus \texttt{o200k}) versus ASCII ratio. Deltas decrease as ASCII increases and remain $>0$ across all bins for all slices; no single break-even point is observed.}\label{fig:ascii-sensitivity}
\end{figure}
The trends in Figure~\ref{fig:ascii-sensitivity} show a monotonic decrease in paired deltas while remaining strictly positive; we therefore treat ASCII ratio as a monotonic cost signal rather than a hard routing breakpoint.

\subsection{Sample convergence}
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/convergence_curves.png}
    \caption{Convergence of median (\texttt{cl100k} $-$ \texttt{o200k}) as the number of paired baselines $k$ increases. Medians stabilize as $k$ grows, supporting the $n=100$ design.}\label{fig:convergence}
\end{figure}

Median deltas stabilize well before the 100-template budget.

\subsection{Merge exemplars}
Table~\ref{tab:merge-examples} lists frequent leading-space merges observed across Hindi and Hinglish baselines. \texttt{o200k\_base} emits high-frequency tokens such as {\textvisiblespace}\devtxt{क्ष} and {\textvisiblespace}\devtxt{करो}, which collapse whitespace and aksharas; \texttt{cl100k\_base} primarily yields single letters with prefixed spaces.
\begin{table}[t]
    \centering
    \caption{Frequent space-prefixed merges observed across Hindi/Hinglish baselines. {\textvisiblespace} denotes a leading space.}\label{tab:merge-examples}
    \input{tables/table_merge_examples.tex}
\end{table}

% ===================== DISCUSSION =====================

\section{Discussion}

\subsection{Why \texttt{o200k\_base} dominates}
Two design choices explain its consistent advantage:
\begin{enumerate}
    \item \textbf{Merge coverage.} \texttt{o200k\_base} doubles the merge vocabulary (200k) relative to \texttt{cl100k\_base}, capturing multi-byte Indic clusters, emoji, and URL substrings that otherwise fall back to byte-level splits.
    \item \textbf{Byte fallback.} Guaranteed coverage yields zero \texttt{[UNK]} even when inputs are noisy or code-mixed. While Indic-trained WordPiece models achieve optimal compactness on pure Indic text and remain competitive under moderate code-mixing, byte-fallback tokenizers guarantee coverage and show low variance under noisy inputs (emoji/URLs/normalization).
\end{enumerate}

\subsection{Operational takeaways}
A simple policy captures the observed gains:
\begin{itemize}
    \item \textbf{Default:} use \texttt{o200k\_base} universally.
    \item \textbf{Optional router:} route single-script Indic lines with ASCII ratio $<0.25$ to \texttt{ai4bharat/IndicBERTv2-MLM-only}; auto-fallback to \texttt{o200k\_base} if \texttt{[UNK]} appears.
    \item \textbf{Unicode hygiene:} normalize to NFC, preserve ZWJ, keep emoji/URLs untouched.
    \item \textbf{Monitoring:} track median tokens/100, \texttt{[UNK]} incidence, and fragmentation dispersion per slice to detect drift.
\end{itemize}

\subsection{Future considerations}
Results are conditional on the five slices studied. Additional Indic scripts (Bengali, Telugu, Malayalam, etc.), Romanized Hindi, or long-document workloads may shift magnitude but are unlikely to overturn the relative strengths of byte-fallback versus script-aware tokenizers. Library updates or merge-table revisions warrant re-exporting CSVs with the pinned pipeline.

% ===================== LIMITATIONS =====================

\section{Limitations}

Our study has several limitations (detailed in Appendix~B):

\begin{itemize}
    \item \textbf{Scope:} We focus primarily on intrinsic metrics (tokens/100, bytes/token, \texttt{[UNK]}, fragmentation). We do not report absolute throughput; any such measurements would be CPU-only proxies rather than comprehensive performance benchmarks, and we do not evaluate downstream perplexity or task accuracy.
    
    \item \textbf{Coverage:} Our evaluation encompasses five language slices (EN, HI, KN, TA, Hinglish). Analysis of additional Indic scripts and Romanized Hindi would require new data collection.
    
    \item \textbf{Input length:} Baselines consist of short-to-medium length text lines; tokenization behavior may differ for longer documents.
    
    \item \textbf{Library/version dependence:} Results are contingent on specific library versions (\texttt{transformersjs=2.17.2}, \texttt{tiktoken=1.0.22}, app 0.1.0, commit \texttt{b625b2ea}, OS \texttt{Win32}).
    
    \item \textbf{Tokenizer equivalence:} The observed equivalence between \texttt{cl100k\_base} and \texttt{Meta{-}Llama-3.1{-}8B{-}Instruct} holds for all English, Kannada, and Tamil baselines (100/100), though this equivalence breaks under several stressor conditions.
\end{itemize}


% ===================== CONCLUSION =====================

\section{Conclusion}
TokenizerLab's CSV-first benchmark demonstrates that \texttt{o200k\_base} is the optimal single-tokenizer choice for Indic and code-mixed deployments: it remains \texttt{[UNK]}-free, reduces costs by approximately half compared to \texttt{cl100k\_base} on Indic scripts, and maintains performance parity with English. For further cost optimization, a lightweight router can optionally direct pure Indic lines to IndicBERTv2. By adopting NFC normalization, ZWJ preservation, and monitoring three key counters (tokens/100, \texttt{[UNK]} incidence, and dispersion), teams can deploy robust tokenization pipelines without re-running the benchmark. Appendices A\textendash{}D provide the complete reproducibility protocol, deployment guide, and extended risk register, while operational checklists are available in the supplementary README.\@

% ===================== APPENDICES =====================

\appendix

\section{Supplemental Materials (Appendix A)}
\begin{itemize}
    \item Tables A1\textendash{}A5: Baseline medians, IQRs, cost summaries, and \texttt{[UNK]} incidence for all tokenizers per language slice.
    \item Tables A6\textendash{}A8: ASCII, emoji, URL, normalization, and perturbation deltas for \{\texttt{o200k\_base}, \texttt{cl100k\_base}, \texttt{IndicBERTv2}\}.
    \item Figures A1\textendash{}A3: Fragmentation entropy distributions and dispersion visualizations.
    \item Figure A4: Stressor deltas for Hinglish (script-mixed Hindi\textendash{}English).
\end{itemize}

\section{Validity, Ethics, and Risk Assessment (Appendix B)}
\begin{itemize}
    \item Comprehensive analysis of internal, external, construct, and conclusion validity.
    \item Guidelines for Unicode hygiene, normalization, and ZWJ handling.
    \item Data curation methodology, fairness considerations, environmental impact assessment, and privacy protocols.
    \item Negative findings and potential falsifiers that could challenge current conclusions.
\end{itemize}

\section{Implementation Guide (Appendix C)}
Practical deployment guidance consolidated from earlier sections:
\begin{enumerate}
    \item \textbf{Policy framework:} Default configurations, routing thresholds, Unicode hygiene protocols, and \texttt{[UNK]} failsafe mechanisms.
    \item \textbf{Monitoring protocol:} Metrics for tokens/100, \texttt{[UNK]}\%, and dispersion; recommended alert thresholds.
    \item \textbf{Deployment sequence:} shadow $\to$ canary $\to$ ramp $\to$ baseline freeze.
    \item \textbf{Incident response:} Normalization audit procedures, version verification, golden-set rerun protocols, and fallback rules.
    \item \textbf{Privacy safeguards:} Metric-only logging approach, salted hash implementation for audit requirements.
\end{enumerate}

\section{Reproducibility Protocol (Appendix D)}
All tables, figures, and statistical analyses are regenerable from the released CSVs through the following procedure:
\begin{enumerate}
    \item \textbf{Input datasets:} \texttt{english-fast.csv}, \texttt{hindi-fast.csv}, \texttt{hinglish-fast.csv}, \texttt{kannada-fast.csv}, \texttt{tamil-fast.csv}.
    \item \textbf{Baseline filtering:} \texttt{sweep\_axis == ``baseline''}, NFC normalization, ZWJ stressor disabled, \texttt{emoji\_count=0}, \texttt{url\_applied=0}, \texttt{add\_special\_tokens=false}; use the published \texttt{baselines/<lang>\_baseline\_template\_ids.txt} manifests (with a fixed sampling seed for reproducibility).
    \item \textbf{Tokenizer-specific summaries:} Compute medians and IQRs for \texttt{tokens\_per\_100\_chars}, \texttt{bytes\_per\_token}, and \texttt{[UNK]\%}; calculate ``Rows with UNK \%'' as the incidence of \texttt{unk\_count > 0}.
    \item \textbf{Cost analysis:} Transform medians using $0.1 \times$ tokens/100 (at $\$0.01/1k$ tokens); optionally bootstrap the median (e.g., $B=5{,}000$) for 95\% confidence intervals.
    \item \textbf{Paired statistical analysis:} Join baseline rows for \texttt{cl100k\_base} and \texttt{o200k\_base} by \texttt{template\_id}; compute differences; apply a one-sided Wilcoxon test ($H_1$: \texttt{cl100k > o200k}); report $W$, $n$, $z$, $p$, effect size $r$, and Hodges\textendash{}Lehmann $\Delta$. Apply Benjamini\textendash{}Hochberg across HI/KN/TA/Hinglish. In our data: for EN, $p \approx 4.40\times10^{-6}$; for HI/KN/TA/Hinglish, $p < 2.2\times10^{-16}$ (machine-precision floor under the normal approximation). Corresponding $q$-values remain significant (EN $q \approx 4.40\times10^{-6}$; others $< 1\times10^{-8}$).
    \item \textbf{Stressor analysis:} Summarize deltas relative to baseline for ASCII, emoji, URL, normalization, ZWJ, and perturbations as required.
\end{enumerate}

Additional reference materials:
\begin{itemize}
    \item Precise formulas for all metrics, cost transformations, Wilcoxon tests, and BH correction.
    \item CSV schema documentation, baseline filtering criteria, and stressor definitions.
    \item Baseline template ID manifests in \texttt{baselines/<lang>\_baseline\_template\_ids.txt} for exact replication of sampled sets.
    \item Alias screening methodology ($\geq 99\%$ identical baseline rows) with perturbation stress tests.
    \item Abugida metric and visualization scripts defining LSAR, SPR, CSBC, ASCII sensitivity summaries, and convergence curves.
    \item Replication modes: CSV-only (expedited) and full browser rerun (comprehensive).
    \item Analysis scripts and diagnostics, including Devanagari merge counts and the reproducibility notebook.
    \item Data licensing statement (expanded from main text), archive structure, and citation template.
    \item Reviewer/practitioner FAQ (condensed) and administrative documentation.
\end{itemize}

TokenizerLab (browser application), the five language slice CSVs, schema documentation, and checksums are released under permissive licenses; see this appendix for file structure, licensing information, and citation guidelines. All assets are fixed at archive tag (v1.0.0).

All supplementary materials referenced in the main text are hosted alongside the artifact tag.

\bibliographystyle{plainnat}
\bibliography{refs}


\end{document}
