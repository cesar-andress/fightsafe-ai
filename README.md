# FightSafe AI

**v2.0.0 — availability-aware interpretable temporal event pipeline**

Research software for interpretable multi-channel temporal event processing from video-derived evidence.
Upstream perception features can be held fixed while downstream aggregation, optional interaction rules, score banding, temporal consolidation and availability encoding are varied under an explicit availability mask \(\alpha\) distinct from observed zero-valued evidence.

| Resource | Location |
|----------|----------|
| Source code | [https://github.com/cesar-andress/fightsafe-ai](https://github.com/cesar-andress/fightsafe-ai) |
| Zenodo (all versions) | [https://doi.org/10.5281/zenodo.20622868](https://doi.org/10.5281/zenodo.20622868) |
| Tag | `v2.0.0` |

**Not** a medical device, clinical diagnostic tool, autonomous officiating system, or deployment-ready safety product.

---

## What the software implements

Perception modules emit soft channel confidences at uneven reliability; channels may be unavailable; and review workflows need intervals that remain inspectable. FightSafe AI provides:

- named evidence channels with soft confidence \(c\) and binary availability \(\alpha\);
- availability-aware equal, weighted and max aggregation;
- optional configured interaction-rule boosts with exportable firings;
- HIGH/CRITICAL banding and temporal consolidation into risk intervals;
- a separate strike-interval path that can be held fixed while the risk path is re-aggregated;
- combined-timeline matching against proxy temporal labels for controlled evaluation.

A frozen BoxingVI case-study run (`canonical_results/run_20260730_005150`) supports Tier A regeneration of tables/figures from canonical CSVs under synthetic channel dropout. Findings are protocol-limited (\(n{=}10\) videos; proxy punch/impact labels).

---

## Repository layout

```
README.md, CITATION.cff, LICENSE, CHANGELOG.md, NOTICE_DATA.md
pyproject.toml, .zenodo.json, environment.yml, requirements.txt

src/fightsafe_ai/          # Python package
tests/                     # unit / integration / e2e tests
configs/                   # YAML rules and weights (incl. risk_fusion.yaml)
scripts/                   # asset regeneration and Tier A validation
annotations/               # BoxingVI punch-interval proxies + case-study labels
canonical_results/         # frozen CSVs/matrices for run_20260730_005150
checksums/                 # SHA-256 manifests for the public tree
optional_tier_b/           # strike baselines; features_cache NOT redistributed
docs/                      # architecture and reproducibility notes
environment/               # Tier A freeze notes
```

Companion LaTeX manuscripts, when used, live outside this GitHub software repository (sibling workspaces in a local monorepo layout).

---

## Installation

**Requirements:** Python 3.12, FFmpeg on `PATH`, Git.

```bash
git clone https://github.com/cesar-andress/fightsafe-ai.git
cd fightsafe-ai
git checkout v2.0.0
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Verify:

```bash
fightsafe --help
fightsafe --version
make test-unit
```

---

## Tier A reproducibility

Tier A regenerates and verifies reported tables/figures from **frozen** canonical CSVs. It does **not** re-run perception or require skeleton keypoints / raw video / `features_cache`.

```bash
# Regenerate manuscript figures/tables into sibling ../paper1/ (if present)
python3.12 scripts/generate_eaai_assets.py

# Verify package checksums
python3.12 scripts/verify_checksums.py

# End-to-end Tier A checks (imports, numbers, regeneration, tests)
python3.12 scripts/validate_tier_a.py

# Aggregation unit tests
PYTHONPATH=src python3.12 -m pytest tests/unit/test_aggregation_schemes.py -q
```

Canonical path: `canonical_results/run_20260730_005150/`.  
Frozen numbers mirror: `canonical_results/analysis/numbers.json`.

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
- proxy punch/impact labels, not validated safety ground truth;
- not a deployment or operator-outcome study.

---

## Citation

Cite the software version you used. The Zenodo concept DOI resolves to the latest archived version:

```bibtex
@software{fightsafe_ai_2026,
  author       = {Andr\'{e}s, C\'{e}sar and Martin Moncunill, David},
  title        = {{FightSafe AI}: Availability-Aware Interpretable Temporal Event Pipeline},
  year         = {2026},
  version      = {2.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20622868},
  url          = {https://doi.org/10.5281/zenodo.20622868},
  note         = {GitHub: https://github.com/cesar-andress/fightsafe-ai (tag v2.0.0); concept/all-versions DOI}
}
```

Also see `CITATION.cff`. After Zenodo archives this tag, prefer the version-specific DOI shown on the Zenodo record for that release.

---

## Licence

MIT License — see `LICENSE`.

Copyright (c) 2026 David Martin Moncunill, César Andrés, Camilo José Cela University (UCJC), Spain.

César Andrés — cesar.andress@ucjc.edu ([ORCID 0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404))  
David Martin Moncunill — david.martinm@ucjc.edu ([ORCID 0000-0003-2422-9005](https://orcid.org/0000-0003-2422-9005))
