# paper1 — EAAI manuscript workspace

## Purpose

LaTeX workspace for the manuscript targeting *Engineering Applications of Artificial Intelligence*:

**Engineering an Interpretable Temporal Event Pipeline with Explicit Channel Availability: A Combat-Sports Case Study**

## Numerical source of truth

Canonical checkpoint identifier: `run_20260730_005150`  
Frozen CSVs in this repository: `../release/eaai_zenodo_staging/results/run_20260730_005150/`

## Build

```bash
python3.12 scripts/generate_assets.py
make
```

Appendix tables are printed after the bibliography (`\appendix` follows `\bibliography{refs}`).

## Public reproducibility package

See [`../release/eaai_zenodo_staging/`](../release/eaai_zenodo_staging/) (Tier A).  
Tier B feature caches are not redistributed with the public artefact.
