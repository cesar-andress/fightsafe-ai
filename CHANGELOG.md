# Changelog

## [1.0.0] — 2026-07-30

### Canonical EAAI reproducibility artefact

- Repository root is the official v1.0.0 artefact (no nested Zenodo staging package).
- Frozen checkpoint `run_20260730_005150` under `canonical_results/`.
- EAAI manuscript sources and PDF under `paper/`.
- Tier A regeneration via `scripts/generate_eaai_assets.py` and `scripts/validate_tier_a.py`.
- BoxingVI punch-interval proxy annotations under `annotations/boxingvi/`.
- Repository hygiene: removed obsolete ESWA/JSS doc stubs, exploration notebook, duplicate `annotations/boxingvi_root` shim, and unused `data/paper2_human_study.csv`; tightened `.gitignore`.
- Official Zenodo version DOI for v1.0.0: https://doi.org/10.5281/zenodo.21698326 (concept DOI `10.5281/zenodo.20622868`; earlier deposit `10.5281/zenodo.20622869` is not the v1.0.0 artefact).

## Prior tags

Historical tags `v0.1.x` / `v0.2.0` remain in git history for provenance and must not be treated as the current EAAI artefact.
