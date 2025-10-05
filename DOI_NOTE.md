## DOI guidance

Zenodo provides two DOI types:

- Concept DOI: a stable DOI that represents the entire project across all versions. Use this when you want a citation that always resolves to the latest version. (Example: `10.5281/zenodo.17273988`)
- Version DOI: a DOI minted for a specific release (example: `10.5281/zenodo.17273989` for v1.0.0 and `10.5281/zenodo.17274005` for v1.0.1).

Recommendation
- For academic citations that should always point to the most recent version, cite the concept DOI: `10.5281/zenodo.17273988`.
- If you need to reference a specific released snapshot (for reproducibility), cite the version DOI for that release.
## DOI and release note

During the initial release we created two GitHub/Zendodo deposits in quick succession. Zenodo minted two DOIs:

- 10.5281/zenodo.17273989 (v1.0.0)
- 10.5281/zenodo.17274005 (v1.0.1)

We recommend using the later DOI (10.5281/zenodo.17274005, v1.0.1) as the canonical citation for this repository and paper. The repository `CITATION.cff` and `README.md` reference the canonical DOI.

If you prefer, you can consolidate releases on Zenodo by editing the older record to point to the newer version or by contacting Zenodo support; however, keeping both records is fine — Zenodo will maintain a concept DOI grouping these releases automatically.
