# Build Instructions

This project uses LaTeX (specifically `xelatex`) to compile the research paper. `xelatex` is required to correctly render Indic scripts (Hindi, Kannada, Tamil).

## Prerequisites

- A TeX distribution (e.g., TeX Live, MiKTeX)
- Python 3.9+ (for analysis scripts and release prep)

## Compilation

To compile the `main.tex` file into a PDF:

1.  **Run XeLaTeX** (first pass):
    ```bash
    xelatex main.tex
    ```

2.  **Run BibTeX** (to process references):
    ```bash
    bibtex main
    ```

3.  **Run XeLaTeX** (second pass, to link references):
    ```bash
    xelatex main.tex
    ```

4.  **Run XeLaTeX** (third pass, to resolve cross-references):
    ```bash
    xelatex main.tex
    ```

You should now have a `main.pdf` file in the directory.

## Cleaning Up

To remove temporary build files (`.aux`, `.log`, etc.) and prepare the repository for a clean commit, run the provided Python script:

```bash
python prepare_release.py
```

This script will:
1.  Delete standard LaTeX build artifacts.
2.  Warn you if any file is larger than 50MB (to avoid GitHub reject issues).
3.  Verify that `main.pdf` exists.
