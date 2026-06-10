# Reproducibility guide

This document maps each companion manuscript to the software commands, input data, and expected outputs in this repository.

**Layout assumption (monorepo):**

```
papers/fightsafe-ai/          ← this repository (software)
papers/fusion2026/            ← Information Fusion manuscript
papers/eswa2026/            ← TapKO / HITL manuscript
papers/sports/                ← FightSafe-Bench manuscript
```

Override paths with environment variables (see [Environment variables](#environment-variables)).

---

## Quick start

```bash
cd fightsafe-ai
pip install -e ".[dev]"

# All three papers (reference mode for eswa when video is absent)
make reproduce-all

# Or individually
make reproduce-fusion
make reproduce-eswa
make reproduce-sports

# Verify bundled reference snapshots
make verify-repro
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_DIR` | `../fusion2026` | Information Fusion LaTeX root |
| `ESWA_DIR` | `../eswa2026` | TapKO / HITL LaTeX root |
| `SPORTS_DIR` | `../sports` | FightSafe-Bench LaTeX root |
| `REPO_ROOT` | auto | Software repository root |
| `REPRO_OUTPUT_ROOT` | `outputs/repro` | Generated reproduction artefacts |
| `TAPKO_VIDEO` | `data/tapko/videos/jedi_submissions.mp4` | TapKO pilot video |
| `REPRO_USE_REFERENCE` | `0` (auto `1` in `reproduce_all` if video missing) | Use bundled TapKO predictions |
| `BENCHMARK_POSE_CSV` | `outputs/tapko/.../pose_keypoints.csv` | Pose CSV for benchmark features |

---

## fusion2026 — Information Fusion

### Scientific artefacts

| Artefact | Manuscript path | Reproduced by | Expected output |
|----------|-----------------|---------------|-----------------|
| Ablation table A–F | `tables/ablation_selected_results.tex` | `scripts/export_fusion2026_assets.py` | `outputs/repro/fusion2026/tables/ablation_selected_results.tex` |
| Quantitative summary | `tables/quantitative_observations.tex` | same | `outputs/repro/fusion2026/tables/quantitative_observations.tex` |
| Ablation figures | `figures/ablation_*.pdf` | `../fusion2026/scripts/regenerate_figures.py` | `outputs/repro/fusion2026/figures/` + `fusion2026/figures/` |
| Pipeline figures | `figures/fig01_architecture.pdf`, etc. | `fusion2026/figures/build_pipeline_figures.sh` | `fusion2026/figures/` |
| BoxingVI baselines | `tables/baseline_comparison.tex` | `make fusion-all` (needs skeleton) | `outputs/evaluation/boxingvi_batch/baseline_comparison.tex` |
| BoxingVI pooled | `tables/boxingvi_pooled.tex` | editorial snapshot / batch exports | reference CSV in `data/repro/fusion2026/reference/boxingvi/` |
| Sweep summary | `tables/sweep_summary.tex` | `python -m fightsafe_ai.evaluation.summarize_sweeps` | reference CSV in `data/repro/.../sweeps/` |
| PDF | `main.pdf` | `make fusion-pdf` | `outputs/repro/fusion2026/main.pdf` |

### Bundled inputs (in Git)

| Path | Description |
|------|-------------|
| `runs/case_studies/ablation_summary/ablation_all_runs.csv` | Aggregate ablation metrics (cases A–F) |
| `runs/case_studies/ablation_summary/case_*/risk_series_*.csv` | Per-case risk traces |
| `data/boxingvi/annotations/V*.xlsx` | BoxingVI punch-interval labels |
| `annotations/case_*.json` | Manual interval labels for case studies |
| `data/repro/fusion2026/reference/boxingvi/*.csv` | Reference BoxingVI aggregate metrics |

### External inputs (not in Git)

| Path | Size | How to obtain |
|------|------|---------------|
| `data/boxingvi/skeleton/V*.npy` | ~50 MB | BoxingVI dataset authors — see `data/README.md` |
| Case-study videos | varies | YouTube URLs in `configs/case_studies.yaml` + `fightsafe run-case-studies` |

### Commands

```bash
# Ablation tables + figures (no BoxingVI skeleton required)
bash scripts/reproduce_fusion2026.sh

# Full BoxingVI batch (when skeleton is available)
make fusion-all FUSION_DIR=../fusion2026

# Regenerate ablation only
python scripts/export_fusion2026_assets.py --fusion-dir ../fusion2026
```

### Notes on manuscript numbers

- **Ablation tables** can be regenerated from `ablation_all_runs.csv`. The committed `fusion2026/tables/ablation_selected_results.tex` uses an editorial layout; regenerated tables land in `outputs/repro/fusion2026/tables/` unless you pass `--install-tables`.
- **BoxingVI pooled/baseline prose** in `sections/08_results.tex` includes editorial aggregates. Reference CSV snapshots document the software export used during development; full re-execution requires BoxingVI skeleton keypoints.

---

## eswa2026 — TapKO / HITL workflow demonstration

### Scientific artefacts

| Artefact | Manuscript path | Reproduced by | Expected output |
|----------|-----------------|---------------|-----------------|
| Pilot results table | `tables/tapko_pilot_results.tex` | `scripts/export_eswa2026_tables.py --install` | `outputs/repro/eswa2026/tables/tapko_pilot_results.tex` |
| Per-class table | `tables/tapko_pilot_per_class.tex` | same | `outputs/repro/eswa2026/tables/tapko_pilot_per_class.tex` |
| Evaluator CSV | (supporting) | `fightsafe tapko-evaluate` | `outputs/tapko/jedi_submissions_eval/tapko_results.csv` |
| Error analysis | cited in text | evaluator | `outputs/tapko/jedi_submissions_eval/tapko_error_analysis.md` |
| Architecture figures | `figures/fig*_tikz.tex` | LaTeX/TikZ (no code run) | compiled in PDF |
| PDF | `main.pdf` | `reproduce_eswa2026.sh` | `outputs/repro/eswa2026/main.pdf` |

### Bundled inputs

| Path | Description |
|------|-------------|
| `data/tapko/annotations/jedi_submissions.json` | Draft reference windows (10 intervals) |
| `data/repro/eswa2026/reference/tapko_predictions.json` | Reference detector export (verification mode) |
| `data/repro/eswa2026/reference/tapko_results.csv` | Reference evaluator metrics |

### External inputs

| Path | Description |
|------|-------------|
| `data/tapko/videos/jedi_submissions.mp4` | ~200 MB instructional clip — `yt-dlp` per `data/README.md` |

### Commands

```bash
# Full pipeline (requires video)
bash scripts/reproduce_eswa2026.sh

# Without video — reference predictions + table sync
REPRO_USE_REFERENCE=1 bash scripts/reproduce_eswa2026.sh

# Export tables only
python scripts/export_eswa2026_tables.py --install --eswa-dir ../eswa2026
```

### Expected metrics (micro row, draft references)

| Metric | Value |
|--------|-------|
| TP | 1 |
| FP | 336 |
| FN | 9 |
| Precision | 0.0030 |
| Recall | 0.1000 |
| F1 | 0.0058 |
| FP/min | 30.1135 |

---

## sports — FightSafe-Bench

### Scientific artefacts

| Artefact | Manuscript path | Status |
|----------|-----------------|--------|
| Dataset statistics | `sections/03_dataset_construction.tex` | Pilot via `build_fightsafe_bench.py` |
| Result tables (mAP, agreement) | `sections/09_experimental_results.tex` | **TBD** — skeleton manuscript |
| Feature export | (supporting) | `scripts/extract_benchmark_features.py` |

### Bundled inputs

| Path | Description |
|------|-------------|
| `data/FightSafeBench/annotations/jedi_submissions_alumno001.xlsx` | Sample annotation spreadsheet |
| `data/FightSafeBench/events.csv` | Merged pilot events |

### Commands

```bash
bash scripts/reproduce_sports.sh
```

Produces `data/FightSafeBench/events.csv`, `dataset_statistics.json`, `../sports/dataset_summary.md`, and optionally `outputs/repro/sports/benchmark_features.csv` when pose CSV exists.

---

## Verification

```bash
python scripts/verify_paper_outputs.py --paper all
```

Checks:

- fusion: ablation CSV completeness
- eswa: reproduced metrics vs `data/repro/eswa2026/reference/tapko_results.csv`
- sports: `dataset_statistics.json` present

---

## Output directory layout

```
outputs/repro/
├── fusion2026/
│   ├── tables/           # regenerated ablation TeX
│   ├── figures/          # copies of ablation PDF/PNG
│   └── main.pdf          # optional PDF copy
├── eswa2026/
│   ├── tables/           # tapko_pilot_*.tex
│   ├── tapko_results.csv
│   └── main.pdf
└── sports/
    ├── events.csv
    ├── dataset_statistics.json
    └── benchmark_features.csv
```

---

## See also

- `data/README.md` — large media download policy
- `scripts/README.md` — script index
- `README.md` — installation and citation
- [`ESWA2026_REPRODUCIBILITY.md`](ESWA2026_REPRODUCIBILITY.md) — eswa2026 paper-specific reproduction
- [`ESWA2026_TRACEABILITY_MATRIX.md`](ESWA2026_TRACEABILITY_MATRIX.md) — manuscript-to-artifact map for eswa2026
