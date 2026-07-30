# FightSafe AI — EAAI reproducibility package

## 1. Purpose
Tier A artefact to **inspect, verify and regenerate** the tables and figures of the EAAI paper from
frozen canonical results. This is a controlled engineering evaluation package, **not** a
safety-certified system and **not** a state-of-the-art detector claim.

## 2. Paper title
Engineering an Interpretable Temporal Event Pipeline with Explicit Channel Availability: A Combat-Sports Case Study

## 3. Canonical run identifier
`run_20260730_005150`

## 4. Tier A vs Tier B
- **Tier A (this package, mandatory):** regenerate/verify figures and tables from frozen CSVs;
  inspect configs, aggregation code, environment metadata and checksums.
- **Tier B (optional):** re-run the experimental layer from `features_cache` + strike baselines.
  Feature pickles are **not included** (pending institutional approval). See `optional_tier_b/`.

## 5. Directory structure
```
README.md, LICENSE, CITATION.cff, zenodo.json, NOTICE_DATA.md, pyproject.toml
environment/          # freeze + Tier A requirements
src/                  # FightSafe AI library (includes aggregation_schemes.py)
configs/              # risk_fusion.yaml, risk_rules.yaml
scripts/              # generate_assets.py, verify_checksums.py, run_eaai_checkpoint.py
results/run_20260730_005150/     # canonical CSVs, PROTOCOL, checksums, parquet matrices
analysis/             # descriptive CSVs used by figure/table generation
figures/, tables/, supplementary/
manuscript/           # main.tex, refs.bib, Makefile
annotations/boxingvi/ # punch-interval labels (with NOTICE_DATA.md)
checksums/, manifests/
optional_tier_b/      # placeholders + strike baselines
validation/           # filled by validate_tier_a.py
```

## 6. Quick start (Tier A)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r environment/requirements-tierA.txt
python scripts/verify_checksums.py
python scripts/generate_assets.py
PYTHONPATH=src python -m pytest tests/unit/test_aggregation_schemes.py -q
```

## 7. Figure regeneration
```bash
python scripts/generate_assets.py
# writes figures/fig_*.pdf|png
```

## 8. Table regeneration
```bash
python scripts/generate_assets.py
# writes tables/tab_*.tex and supplementary/tab_si_*.tex
```

## 9. Manuscript build
```bash
cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex
```

## 10. Checksum verification
```bash
python scripts/verify_checksums.py
```

## 11. Environment
Python **3.12**. Prefer `environment/requirements-tierA.txt`.
The full lab `environment/pip_freeze.txt` is archival and includes non-public editable installs.

## 12. Data availability
See `NOTICE_DATA.md`. Skeleton/video/`features_cache` are not redistributed here.

## 13. Licence
MIT for software. BoxingVI labels: cite upstream; see notice.

## 14. What is not included
Legacy archives, aborted runs, ISWA/Information Fusion manuscripts, raw video, skeletons,
unapproved feature caches, dashboards, Git history, LaTeX auxiliaries.

## 15. Known limitations
- Natural α≡1 on held-fixed matrices; missingness is synthetic.
- Combined timeline uses a fixed strike component.
- Exact recreation of the full lab pip freeze is not claimed.
- Tier B re-execution is not validated in this deposit.

## 16. Citation
See `CITATION.cff`. Zenodo DOI placeholder: `10.5281/zenodo.PENDING`.
