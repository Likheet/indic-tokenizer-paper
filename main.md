# Abstract

We benchmark 11 production tokenizers (OpenAI, Meta, Mistral, Indic-focused WordPiece/SentencePiece models) on English, Hindi, Kannada, Tamil, and script-mixed Hinglish using a browser-only measurement harness. Byte-fallback `o200k_base` halves Indic token counts and stays `[UNK]`-free, while `cl100k_base` and Indic WordPiece trade efficiency for robustness. New diagnostics quantify vocabulary allocation (leading-space akshara rate, syllable preservation, cross-script coverage), ASCII sensitivity, and convergence of the paired baselines. All statistics, figures, and tables are regenerated from released CSVs and Python scripts.

## 1 Introduction

Tokenization is the first irreversible decision in an LLM pipeline: once text is segmented, attention budgets, truncation risk, and cost follow. Devanagari+Latin code-mixing is common in South Asian deployments, yet most tokenizers are tuned for monolingual English. We therefore analyse how production tokenizers behave on Indic and mixed-script workloads using the same sampled baselines.

### Research Questions

- **RQ1** How do byte-fallback versus WordPiece vocabularies trade off efficiency and robustness on Indic and code-mixed inputs 
- **RQ2** How does ASCII mixing shift token budgets per tokenizer 
- **RQ3** Which merge patterns (e.g., space+akshara) correlate with lower fragmentation 

### Hypotheses

- **H1** Byte-fallback tokenizers guarantee coverage and stabilise budgets under noisy inputs.
- **H2** Indic WordPiece vocabularies are most compact on pure Indic text but degrade as ASCII ratio increases.
- **H3** Higher leading-space akshara and cross-script coverage correlate with better code-mixed efficiency.

## 2 Related Work

- **Tokenization algorithms.** BPE [Sennrich et al. 2016], byte-level BPE [Radford et al. 2019], WordPiece [Schuster & Nakajima 2012], SentencePiece [Kudo & Richardson 2018].
- **Multilingual/Indic benchmarks.** IndicBERT [Kakwani et al. 2020], MuRIL [Khanuja et al. 2021], XLM-R [Conneau et al. 2020] emphasise downstream tasks rather than intrinsic segmentation.
- **Production reports.** `tiktoken`, LLaMA, and Claude notes discuss merge tables, byte fallback, and routing trade-offs.

## 3 Methods

- **Measurement pipeline.** Browser-only harness (Transformers.js + `tiktoken` WASM); five repeats per template collapsed by the median; outputs logged to CSV with provenance metadata.
- **Baseline sampling.** 100 NFC-normalised templates per slice (seed 20241004); Hinglish enforces ≥3 Devanagari and ≥3 Latin code points per line.
- **Tokenizers.** Byte-fallback BPE (`o200k_base`, `cl100k_base`, LLaMA, Mistral) and Indic WordPiece/SentencePiece (IndicBERTv2, MuRIL, mBERT, etc.); hosted tokenizers omitted (Appendix B).
- **Definitions.** ASCII ratio = ASCII bytes / total bytes; HLΔ = Hodges–Lehmann median paired difference; LSAR/SPR/CSBC computed via `analysis/compute_abugida_metrics.py`.

## 4 Results

- **Baseline medians.** `o200k_base` matches English and reduces Indic token counts by 55–70% (HLΔ = 66 tokens/100 on Hindi, 134 on Kannada, 101 on Tamil, 16.7 on Hinglish). Figure 1 (`figures/baseline_tokens_bar.png`) and Table 1 summarise costs per 1M characters.
- **Effect sizes.** Table 2 (`tables/table_language_summary.tex`) reports median deltas, 95% bootstrap intervals, Cliff's δ, Wilcoxon p-values, and ASCII breakpoints (~0.10 for Hindi, ~0.16 for Kannada/Tamil, ~0.54 for Hinglish). Deltas stabilise by k=40 baselines (Figure `figures/convergence_curves.png`).

## 5 Analysis

- **Vocabulary allocation metrics.** Figure `figures/merge_metrics_bar.png` shows LSAR/SPR/CSBC: `o200k_base` reaches LSAR=1.0 (SPR 2.44 Hindi / 2.05 Hinglish), `cl100k_base` only 0.52 LSAR on Hindi (SPR 1.02), IndicBERTv2 LSAR/CSBC=0 despite SPR≈3.3.
- **ASCII sensitivity.** Figure `figures/hinglish_lsar_curve.png` (LSAR vs ASCII) and `figures/ascii_sensitivity.png` (segmented regression) underpin the 25% routing threshold.
- **Merge exemplars.** Table 3 (`tables/table_merge_examples.tex`) lists frequent space+akshara merges (e.g., spaceक्ष, spaceकर) produced by `o200k_base`; `cl100k_base` yields mostly single Devanagari letters with prefixed spaces.

## 6 Discussion

- **Why `o200k_base` wins.** Larger merge vocabularies eliminate byte fallbacks, preserving aksharas; byte fallback guarantees zero `[UNK]` under mixed inputs.
- **Operational guidance.** Default to `o200k_base`, optionally route ASCII<0.25 Indic lines to IndicBERTv2, monitor tokens/100 + `[UNK]` + dispersion, maintain NFC/ZWJ hygiene. CPU microbenchmarks: 4.1×10⁵ tokens/s (`o200k_base`) vs 7.4×10⁵ (`cl100k_base`) vs 1.5×10⁵ (IndicBERTv2) on Hindi; Hinglish exhibits the same ordering.
- **Future work.** Extend to additional Indic scripts, Romanized Hindi, long documents, GPU latency, and downstream accuracy.

## 7 Limitations

Intrinsic metrics dominate; throughput numbers are CPU proxies, and no perplexity or task evaluations are reported. Coverage is limited to five slices; expanding to other Indic scripts or genre distributions may change magnitudes but not qualitative patterns.

## 8 Conclusion

`o200k_base` is the safest deployment default for Indic/Hinglish workloads: it halves cost relative to `cl100k_base`, preserves `[UNK]`-free behaviour, and remains throughput-competitive. A lightweight router can hand pure Indic lines to IndicBERTv2 to squeeze extra savings. Appendices and scripts reproduce every table, figure, and statistic.

### Appendix Snapshot

- **Appendix A**: Baseline tables, stressor deltas, supplementary figures.
- **Appendix B**: Extended limitations, ethics, and risk mitigations.
- **Appendix C**: Deployment guide (routing, monitoring, incident response).
- **Appendix D**: Statistical methods, reproducibility checklist (inputs, filters, paired stats, stressors), CSV schema, abugida metrics scripts (`analysis/compute_abugida_metrics.py`, `analysis/plot_benchmark_figures.py`), plus licensing and data-release notes.








