# FightSafe AI

**v1.0.0 — canonical EAAI reproducibility artefact**

Research software for interpretable multi-source temporal event pipelines, with an accompanying manuscript for *Engineering Applications of Artificial Intelligence* (EAAI):

*Engineering an Interpretable Temporal Event Pipeline with Explicit Channel Availability: A Combat-Sports Case Study*

This GitHub repository **is** the official v1.0.0 artefact. There is no nested staging or release subdirectory.

| Resource | Location |
|----------|----------|
| Source code | [https://github.com/cesar-andress/fightsafe-ai](https://github.com/cesar-andress/fightsafe-ai) |
| Zenodo DOI | [https://doi.org/10.5281/zenodo.21698326](https://doi.org/10.5281/zenodo.21698326) |
| Tag | `v1.0.0` |

**Not** a medical device, clinical diagnostic tool, autonomous officiating system, or deployment-ready safety product.

---

## Engineering problem

Perception modules emit soft channel confidences at uneven reliability; channels may be unavailable; and review workflows need intervals that remain inspectable. FightSafe AI separates held-fixed perception features from interpretable aggregation, optional interaction rules, score banding and temporal consolidation, with an explicit availability mask \(\alpha\) distinct from zero-valued evidence.

The EAAI case study freezes perception, varies aggregation / interaction / synthetic availability encoding, and reports protocol-limited findings on BoxingVI punch/impact *proxy* labels (canonical checkpoint `run_20260730_005150`).

---

## Repository layout

```
README.md, CITATION.cff, LICENSE, CHANGELOG.md, NOTICE_DATA.md
pyproject.toml, .zenodo.json, environment.yml, requirements.txt

src/fightsafe_ai/          # Python package
tests/                     # unit / integration / e2e tests
configs/                   # YAML rules and weights (incl. risk_fusion.yaml)
scripts/                   # EAAI asset regeneration and Tier A validation
annotations/               # BoxingVI punch-interval proxies + case-study labels
canonical_results/         # frozen CSVs/matrices for run_20260730_005150
checksums/                 # SHA-256 manifests for the public tree
optional_tier_b/           # strike baselines; features_cache NOT redistributed
docs/                      # architecture and reproducibility notes
environment/               # Tier A freeze notes
```

The EAAI LaTeX manuscript lives in the sibling monorepo directory `../paper1/` (not in this GitHub repository).

---

## Installation

**Requirements:** Python 3.12, FFmpeg on `PATH`, Git.

```bash
git clone https://github.com/cesar-andress/fightsafe-ai.git
cd fightsafe-ai
git checkout v1.0.0
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Verify:

```bash
fightsafe --help
make test-unit
```

---

## Tier A reproducibility (EAAI)

Tier A regenerates and verifies manuscript tables/figures from **frozen** canonical CSVs. It does **not** re-run perception or require skeleton keypoints / raw video / `features_cache`.

```bash
# Regenerate manuscript figures/tables into sibling ../paper1/
python3.12 scripts/generate_eaai_assets.py

# Verify package checksums
python3.12 scripts/verify_checksums.py

# End-to-end Tier A checks (imports, numbers, regeneration, tests; builds paper1 if present)
python3.12 scripts/validate_tier_a.py

# Aggregation unit tests
PYTHONPATH=src python3.12 -m pytest tests/unit/test_aggregation_schemes.py -q
```

Canonical path: `canonical_results/run_20260730_005150/`.  
Frozen numbers mirror: `canonical_results/analysis/numbers.json`.

Compile the manuscript (sibling workspace):

```bash
cd ../paper1
latexmk -pdf -interaction=nonstopmode main.tex
# bibliography: bibtex main   (not bibtex main.aux)
```

Override manuscript location with `FIGHTSAFE_PAPER1_DIR` if needed.

---

## Restricted data (not redistributed)

See `NOTICE_DATA.md`. Excluded from this repository:

- raw BoxingVI video / frames;
- BoxingVI skeleton keypoints;
- derived `features_cache/*.pkl` (Tier B).

Strike baselines under `optional_tier_b/inputs/strike_baselines/` are included for combined-timeline interpretation.

---

## Testing

```bash
make test-unit
# or
pytest tests/unit -q
```

---

## Limitations (summary)

- \(n{=}10\) BoxingVI stems; limited statistical power;
- pooled metrics dominated by stem V6;
- combined timeline includes a fixed strike component;
- natural availability \(\alpha{\equiv}1\); missingness results are synthetic;
- proxy punch/impact labels, not clinical ground truth;
- not a deployment or operator-outcome study.

Full limits are stated in the manuscript.

---

## Citation

```bibtex
@misc{fightsafe_ai_2026,
  author       = {Andr\'{e}s, C\'{e}sar and Martin Moncunill, David},
  title        = {{FightSafe AI} --- {EAAI} reproducibility artefact (availability-aware temporal event pipeline)},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21698326},
  url          = {https://doi.org/10.5281/zenodo.21698326},
  note         = {GitHub: https://github.com/cesar-andress/fightsafe-ai (tag v1.0.0)}
}
```

Also see `CITATION.cff`.

---

## Licence

MIT License — see `LICENSE`.

Copyright (c) 2026 David Martin Moncunill, César Andrés, Camilo José Cela University (UCJC), Spain.

César Andrés — cesar.andress@ucjc.edu ([ORCID 0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404))  
David Martin Moncunill — david.martinm@ucjc.edu ([ORCID 0000-0003-2422-9005](https://orcid.org/0000-0003-2422-9005))
