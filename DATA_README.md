# Data files and verification

This project includes large CSV exports under `tokenizer-output/`. If you cloned the repository with these files, verify integrity using the provided checksums in `checksums/sha256sum.csv`.

Verification (PowerShell):

```powershell
# from repo root
Get-FileHash -Algorithm SHA256 tokenizer-output\english-fast.csv | Format-List
# Compare the hash with the entry in checksums/sha256sum.csv
```

If you prefer not to store these files in Git, download the release archive from the GitHub Releases page or from Zenodo once the DOI/release is published. The repository includes `checksums/sha256sum.csv` for verification.

If the files are large and you encounter Git cloning slowness, consider using Git LFS or DVC for dataset hosting (see README and project notes).
