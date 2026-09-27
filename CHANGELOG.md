# Changelog

## [2.0.0] — 2026-09-28

### Public software identity

- Journal-neutral project presentation: FightSafe AI as an independent research-software artefact for an availability-aware interpretable temporal event pipeline.
- Public metadata (`README.md`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml`) updated to version **2.0.0**.
- Official Zenodo **version** DOI for v2.0.0: https://doi.org/10.5281/zenodo.23003746 (concept / all-versions DOI `10.5281/zenodo.20622868`).

### Pipeline and reproducibility (unchanged core artefact)

- Frozen canonical run `run_20260730_005150` retained under `canonical_results/`.
- Explicit channel-availability handling, interpretable aggregation, interaction-rule firings and temporal consolidation remain the Tier A evaluation surface.
- Historical script names (e.g. `generate_eaai_assets.py`, `run_eaai_checkpoint.py`) are kept for reproducibility path stability.

## [1.0.0] — 2026-07-30

### Canonical reproducibility artefact

- Repository root is the official v1.0.0 artefact (no nested Zenodo staging package).
- Frozen checkpoint `run_20260730_005150` under `canonical_results/`.
- Companion manuscript sources lived in sibling monorepo `paper1/` (not shipped inside this GitHub software repository).
- Tier A regeneration via `scripts/generate_eaai_assets.py` (writes into `../paper1/`) and `scripts/validate_tier_a.py`.
- BoxingVI punch-interval proxy annotations under `annotations/boxingvi/`.
- Official Zenodo **version** DOI for v1.0.0: https://doi.org/10.5281/zenodo.21698326 (concept DOI `10.5281/zenodo.20622868`).

## Prior tags

Historical tags `v0.1.x` / `v0.2.0` remain in git history for provenance and must not be treated as the current public artefact.
