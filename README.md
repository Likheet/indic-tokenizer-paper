# TokenizerLab — Tokenizer Benchmarking Across Indic and Code‑Mixed Scripts

This repository contains the paper, analysis code, figures, and dataset manifests for "Tokenizer Benchmarking Across Indic and Code‑Mixed Scripts" (Likheet Shetty, 2025). It provides CSV artifacts, plotting scripts, tables used in the paper, and the reproducibility pipeline.

Short description
- Title: Tokenizer Benchmarking Across Indic and Code‑Mixed Scripts
- Focus: intrinsic tokenizer metrics (tokens/100, bytes/token, [UNK], fragmentation) on EN/HI/KN/TA/Hinglish

Contents
- `main.pdf`, `main.tex`, `refs.bib` — paper source and compiled PDF
- `analysis/` — analysis and plotting scripts used to generate figures/tables
- `figures/`, `tables/` — generated visualizations and table LaTeX fragments
- `tokenizer-input/` — input template pools (may be large)
- `tokenizer-output/` — CSV exports (large; see `DATA_README.md`)
- `baselines/` — lists of template IDs used for reproducibility
- `checksums/sha256sum.csv` — checksums for large artifacts

Cite
If you use this artifact, please cite the Zenodo release (placeholder until DOI minted):

Likheet Shetty (2025). Tokenizer Benchmarking Across Indic and Code‑Mixed Scripts. Zenodo. DOI: 10.5281/zenodo.YOURID

License
- Paper/text: CC‑BY 4.0 (recommended)
- Code: MIT (recommended)
- Data: see `LICENSE-data.txt` in the repository for the dataset license

Getting the large files
- This repository currently contains dataset CSVs in `tokenizer-output/`. These files are large (~60 MB each). If you prefer not to fetch them with Git, we provide checksums in `checksums/sha256sum.csv` and a release archive available via the GitHub Releases page or Zenodo. See `DATA_README.md` for verification commands.

Reproducibility
- See `Reproducibility Protocol (Appendix D)` in the paper and the `analysis/` scripts. To regenerate figures, run the plotting scripts (Python 3.9+; see `analysis/` for dependencies). Example:

```powershell
# from repo root
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r analysis/requirements.txt
python analysis/generate_figures.py
```

Release and DOI
- Create a GitHub release (tag `v1.0.0`) including `main.pdf` and a source zip. Link GitHub releases to Zenodo to mint a DOI (see `.zenodo.json`).

Contact
- Likheet Shetty — likheet.s@gmail.com

Tokenizer app
- The browser-based TokenizerLab application that produced the measured tokenizers is available at: https://github.com/Likheet/tokenizer-lab
