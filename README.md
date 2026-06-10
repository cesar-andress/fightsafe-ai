# FightSafe AI

A research software artifact implementing machine-generated alert export, traceability, auditability, protocol-defined bookkeeping, and governance constraints for safety-alert review workflows.

Open-source software for **combat-sports safety monitoring**: monocular video → pose → interpretable features → temporal fusion → schema-typed alert-candidate exports for **specified human-oversight** review paths.

**Not** a medical device, clinical diagnostic tool, or autonomous officiating system.

| Resource | URL |
|----------|-----|
| Source code | [https://github.com/cesar-andress/fightsafe-ai](https://github.com/cesar-andress/fightsafe-ai) |
| Zenodo archive | [https://doi.org/10.5281/zenodo.20622869](https://doi.org/10.5281/zenodo.20622869) |
| Companion manuscripts | See [Research outputs](#research-outputs) below |

---

## Scope of the Artifact

This repository ships **research software** for export generation, traceability bookkeeping, and governance-oriented metadata—not a validated operator-facing product.

- **Reviewer decisions** (`H_{t,k}`) were **not** logged in the reported machine-side stress test.
- **Gate outcomes** (`O_{t,k}`) were **not** logged in the reported run.
- The published stress test (**v0.1.3**, `jedi_submissions` clip) is **machine-side only**: detector exports, evaluator CSV, protocol error tags, and manuscript tables under frozen matching defaults.
- The artifact **supports future audit-schema exercises** (append-only candidate, state, decision, and gate record types) but does not claim confirmation-gate execution, operator benefit, or deployment readiness without separate logged studies.

Script and directory names such as `reproduce_eswa2026.sh` and `data/repro/eswa2026/` reflect the **project-internal paper folder name at artifact creation**; the active traceability manuscript targets the *Journal of Systems and Software* (`../jss2026/`).

---

## Scope and disclaimer

FightSafe AI is a **reproducible research prototype**. It produces interpretable, auditable **candidate** exports for specified review workflows. Qualified humans retain authority over stoppages, medical response, and competitive outcomes. The software does **not** claim improved human decisions, reduced operator workload, or validated human–AI collaboration unless backed by a separate logged study.

---

## Research outputs

Three companion manuscripts share terminology but target different scientific objects. LaTeX sources live in sibling directories when using the monorepo layout (`../fusion2026`, `../jss2026`, `../sports`).

| Manuscript | Directory | Scientific focus | Software entry points |
|------------|-----------|------------------|------------------------|
| **Information Fusion** | `../fusion2026/` | Multi-source temporal fusion, mask ablations, BoxingVI interval evaluation | `make reproduce-fusion`, `fightsafe risk-ablation-all` |
| **JSS / traceability architecture** | `../jss2026/` | Formal specification, traceability architecture, audit schemas, machine-side stress test | `make reproduce-eswa`, `fightsafe tapko-detect` |
| **FightSafe-Bench** | `../sports/` | Benchmark dataset design, annotation protocol, baseline tasks | `make reproduce-sports`, `scripts/build_fightsafe_bench.py` |

---

## Installation

**Requirements:** Python 3.12, FFmpeg on `PATH`, Git.

```bash
git clone https://github.com/cesar-andress/fightsafe-ai.git
cd fightsafe-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Alternative environment file:

```bash
conda env create -f environment.yml
conda activate fightsafe-ai
pip install -e ".[dev]"
```

Verify:

```bash
fightsafe --help
make test-unit
```

---

## Quick start

```bash
# Offline pipeline on a local clip
fightsafe run-pipeline --video path/to/clip.mp4 --output runs/demo/

# TapKO candidate detection (offline)
fightsafe tapko-detect --source path/to/clip.mp4 --output-dir outputs/tapko/run/ --fps 30

# TapKO evaluation against annotations
fightsafe tapko-evaluate \
  --annotations data/tapko/annotations/jedi_submissions.json \
  --predictions outputs/tapko/run/tapko_predictions.json \
  --output-dir outputs/tapko/run_eval/
```

See [`docs/architecture.md`](docs/architecture.md), [`docs/evaluation.md`](docs/evaluation.md), [`docs/troubleshooting.md`](docs/troubleshooting.md), and the full [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) guide.

---

## Repository structure

```
fightsafe-ai/
├── src/fightsafe_ai/     # Python package (CLI, fusion, TapKO, evaluation)
├── configs/              # YAML fusion rules, case-study definitions
├── data/                 # Small curated samples; large media downloaded separately
├── docs/                 # Architecture, evaluation, release checklist, dataset policy
├── scripts/              # Reproduction helpers and dataset builders
├── tools/                # LaTeX table/figure generators for fusion manuscript
├── tests/                # Unit, integration, and e2e tests
├── examples/             # Minimal JSON examples
├── annotations/          # Case-study interval labels (clips A–F)
├── Makefile              # install, test, ci, reproduce-* targets
├── pyproject.toml        # Package metadata and dependencies
├── requirements.txt      # Minimal runtime pins for CI/containers
├── environment.yml       # Conda environment specification
├── CITATION.cff          # Software citation metadata (Zenodo-ready)
└── LICENSE               # MIT License
```

Generated at runtime (ignored by Git): `outputs/`, `runs/`, `.venv/`, caches.

---

## Reproducing experiments

All reproduction targets are available via `make`:

```bash
make reproduce-fusion    # fusion2026 manuscript assets + PDF (when data present)
make reproduce-eswa      # JSS traceability stress test (TapKO pilot pipeline)
make reproduce-sports    # sports / FightSafe-Bench dataset exports
make reproduce-all       # run all three (best-effort; skips missing data)
```

Or run the shell scripts directly under `scripts/`.

### 1. fusion2026 (Information Fusion)

**Goal:** Regenerate ablation tables, BoxingVI evaluation tables, figures, and compile `../fusion2026/main.pdf`.

**Prerequisites:**

1. BoxingVI annotations (included): `data/boxingvi/annotations/`
2. BoxingVI skeleton keypoints (~50 MB, not in Git): place under `data/boxingvi/skeleton/` per [`data/README.md`](data/README.md)
3. Case-study ablation exports (optional for full table regeneration): `runs/case_studies/ablation_summary/ablation_all_runs.csv` — generate with case-study clips and `fightsafe risk-ablation-all`
4. TeX Live (`pdflatex`, `bibtex`, `elsarticle` class)

```bash
# Assets only (tables + figures into ../fusion2026)
make fusion-assets

# Full pipeline: tests + BoxingVI batch eval + assets + PDF
make reproduce-fusion

# Force recomputation of BoxingVI predictions
make fusion-all-force
```

Key CLI commands used internally:

```bash
fightsafe risk-ablation-all --base-dir runs/case_studies --summary-dir runs/case_studies/ablation_summary
python scripts/generate_paper_assets.py --paper-dir ../fusion2026
python ../fusion2026/scripts/regenerate_figures.py
```

Precomputed LaTeX fragments and figures are already committed under `../fusion2026/tables/` and `../fusion2026/figures/` for PDF-only builds.

### 2. jss2026 (machine-side traceability stress test)

**Manuscript:** *A Formal Traceability Architecture and Auditable Human-Oversight Specification for Machine-Generated Safety Alerts* (`../jss2026/`)

**Software archive:** Zenodo [10.5281/zenodo.20622869](https://doi.org/10.5281/zenodo.20622869), release **v0.1.3**.

**Goal:** Reproduce the `jedi_submissions` machine-side stress test (export traceability and protocol-defined bookkeeping, not operator-outcome evaluation).

Paper-specific guide: [`docs/JSS2026_REPRODUCIBILITY.md`](docs/JSS2026_REPRODUCIBILITY.md) · traceability matrix: [`docs/JSS2026_TRACEABILITY_MATRIX.md`](docs/JSS2026_TRACEABILITY_MATRIX.md)

**Prerequisites:**

1. Draft annotations (included): `data/tapko/annotations/jedi_submissions.json`
2. Source video (not in Git, ~200 MB): download to `data/tapko/videos/jedi_submissions.mp4` — see [`data/README.md`](data/README.md). The instructional clip `jedi_submissions` is not redistributed because of rights restrictions.
3. Reference predictions (verification without video): `data/repro/eswa2026/reference/tapko_predictions.json` and `data/repro/eswa2026/reference/tapko_results.csv`

**Reproduction commands** (from repository root):

```bash
# Full pipeline (requires local video)
bash scripts/reproduce_eswa2026.sh
# equivalent: make reproduce-eswa

# Reference mode — bundled predictions when the video is unavailable
REPRO_USE_REFERENCE=1 bash scripts/reproduce_eswa2026.sh

# Point table export at the JSS manuscript directory (script default may be ../eswa2026)
ESWA_DIR=../jss2026 bash scripts/reproduce_eswa2026.sh
```

Reference mode copies bundled detector/evaluator exports from `data/repro/eswa2026/reference/`, regenerates manuscript Tables I–II, and verifies metrics against the reference CSV. It does **not** re-run pose inference on withheld video.

**Expected outputs:**

| Path | Description |
|------|-------------|
| `outputs/tapko/jedi_submissions/tapko_predictions.json` | Detector export (337 candidate intervals) |
| `outputs/tapko/jedi_submissions_eval/tapko_results.csv` | Evaluator metrics (micro / per-class) |
| `outputs/tapko/jedi_submissions_eval/tapko_error_analysis.md` | Error taxonomy digest |
| `../jss2026/tables/tapko_pilot_results.tex` | Table I (`tab:tapko_pilot_results`) |
| `../jss2026/tables/tapko_pilot_per_class.tex` | Table II (`tab:tapko_pilot_per_class`) |
| `outputs/repro/eswa2026/` | Repro bundle (tables, CSV copy, optional PDF copy) |

Verification: `python scripts/verify_paper_outputs.py --paper eswa` (micro row: TP=1, FP=336, FN=9).

Manual steps:

```bash
fightsafe tapko-validate-annotations --annotations data/tapko/annotations/jedi_submissions.json
fightsafe tapko-detect \
  --source data/tapko/videos/jedi_submissions.mp4 \
  --output-dir outputs/tapko/jedi_submissions --fps 30 --pose-backend mediapipe
fightsafe tapko-evaluate \
  --annotations data/tapko/annotations/jedi_submissions.json \
  --predictions outputs/tapko/jedi_submissions/tapko_predictions.json \
  --output-dir outputs/tapko/jedi_submissions_eval \
  --tolerance-seconds 0.5 --match-mode family
```

Compile the manuscript:

```bash
cd ../jss2026 && bash build.sh
```

### 3. sports (FightSafe-Bench)

**Goal:** Rebuild benchmark CSV/JSON exports from annotation spreadsheets and extract per-frame features.

**Prerequisites:** Annotation spreadsheets in `data/FightSafeBench/annotations/` (sample included).

```bash
make reproduce-sports
```

Manual steps:

```bash
python scripts/build_fightsafe_bench.py \
  --input-dir data/FightSafeBench/annotations \
  --output-dir data/FightSafeBench \
  --summary-path ../sports/dataset_summary.md
python scripts/extract_benchmark_features.py
```

The sports manuscript (`../sports/main.tex`) is a **skeleton** with placeholder (TBD) result tables; dataset statistics are real but the full benchmark corpus is not yet locked.

---

## Data policy

Large videos, skeleton exports, and experiment runs are **excluded from Git** to keep the repository Zenodo-friendly. See [`data/README.md`](data/README.md) for download instructions, checksums, and redistribution notes.

---

## Development

```bash
make install      # editable install with dev extras
make ci           # ruff + mypy + pytest (74% coverage floor)
make format       # apply ruff formatter
make lint         # ruff check
```

Pre-commit hooks: `pre-commit install` then `make pre-commit`.

---

## Citation

If you use this software, please cite the Zenodo archive and the GitHub repository. Companion manuscripts use the same BibTeX key `fightsafe_ai_2026` in the shared bibliography at `../../bibliography.bib` (monorepo layout).

### Software (this repository)

```bibtex
@software{fightsafe_ai_2026,
  author       = {Martin Moncunill, David and Andr{\'e}s, C{\'e}sar},
  title        = {FightSafe AI: Traceability and Auditability Software for Safety-Alert Review Workflows},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {0.1.3},
  doi          = {10.5281/zenodo.20622869},
  url          = {https://doi.org/10.5281/zenodo.20622869}
}
```

- GitHub reads [`CITATION.cff`](CITATION.cff) for the **Cite this repository** button.
- Release workflow documentation: [`docs/release_checklist.md`](docs/release_checklist.md).

### Companion manuscripts

Cite the relevant manuscript when using methods or results from that line of work:

- **Fusion:** *Explainable Multi-Source Temporal Information Fusion for Combat-Sports Safety Intelligence* (`../fusion2026/`)
- **Traceability / JSS:** *A Formal Traceability Architecture and Auditable Human-Oversight Specification for Machine-Generated Safety Alerts* (`../jss2026/`)
- **Benchmark:** *FightSafe-Bench: A Benchmark for Temporal Safety Event Detection under Partial Observability* (`../sports/`)

---

## License

MIT License — see [`LICENSE`](LICENSE).

Copyright (c) 2026 David Martin Moncunill, César Andrés, Camilo José Cela University (UCJC), Spain.

---

## Authors

David Martin Moncunill — david.martinm@ucjc.edu (corresponding)  
César Andrés — cesar.andress@ucjc.edu ([ORCID 0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404))  

CRIA-BDHS Research Group, Camilo José Cela University, Madrid, Spain.
