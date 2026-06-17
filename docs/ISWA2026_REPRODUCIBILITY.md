# iswa2026 reproducibility guide

Documentation for the ISWA 2026 traceability manuscript (*Information Systems and e-Business Management*). Documentation-only; release **v0.1.4** ships the aligned reproducibility bundle.

Paper-specific instructions for reproducing the **machine-side traceability protocol demonstration** reported in:

**A Formal Traceability Architecture and Auditable Human-Oversight Specification for Machine-Generated Safety Alerts**

LaTeX sources: `../iswa2026/` (monorepo layout: `papers/fightsafe-ai/iswa2026/`).

General installation, environment variables, and cross-manuscript notes: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Software archive

| Field | Value |
|-------|-------|
| Artifact title | **FightSafe AI: Traceability and Auditability Software for Safety-Alert Review Workflows** |
| Repository release | **v0.1.4** (`pyproject.toml`, `CITATION.cff`, `src/fightsafe_ai/__version__.py`) |
| Zenodo DOI | [10.5281/zenodo.20622869](https://doi.org/10.5281/zenodo.20622869) |
| GitHub tag | `v0.1.4` (see [`docs/release_checklist.md`](release_checklist.md)) |

Reproducibility entry points: `reproduce_iswa2026.sh`, bundled reference exports under `data/repro/iswa2026/`, and manuscript tables installed to `../iswa2026/tables/`.

---

## Scope

Reproduce the **`jedi_submissions` machine-side protocol demonstration**: detector exports, protocol-defined evaluator bookkeeping, and manuscript Tables I–II (`tapko_pilot_results`, `tapko_pilot_per_class`).

This is **export traceability and queue-load bookkeeping**, not fusion-accuracy validation, operator-outcome evaluation, or deployment readiness. Draft reference intervals are research bookkeeping only (see [`data/README.md`](../data/README.md)).

**Not logged in the reported run:** reviewer decisions (`H_{t,k}`), gate outcomes (`O_{t,k}`), confirmation-gate execution on footage.

**Evaluator matching defaults** (fixed in the reported run; also used by `scripts/reproduce_iswa2026.sh`):

| Parameter | Value |
|-----------|-------|
| Temporal IoU threshold | `0.3` (evaluator default) |
| Symmetric tolerance | `0.5` s (`--tolerance-seconds 0.5`) |
| Match mode | `family` (`--match-mode family`) |
| Pose backend (full pipeline) | `mediapipe` |
| Detection FPS | `30` |

---

## Prerequisites

From the repository root (`fightsafe-ai/`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Optional PDF rebuild: TeX Live with `pdflatex` and `bibtex`.

---

## Required inputs

### Bundled in Git

| Path | Description |
|------|-------------|
| `data/tapko/annotations/jedi_submissions.json` | Draft TapKO reference windows (10 intervals) |
| `data/repro/iswa2026/reference/tapko_predictions.json` | Reference detector export (`jedi_submissions`) |
| `data/repro/iswa2026/reference/tapko_results.csv` | Reference evaluator metrics |
| `data/repro/iswa2026/reference/tapko_error_analysis.md` | Reference error category counts |
| `data/repro/iswa2026/reference/tapko_manifest.json` | Optional manifest (copied in reference mode when present) |

### External (full pipeline only)

| Path | Description |
|------|-------------|
| `data/tapko/videos/jedi_submissions.mp4` | ~200 MB instructional clip — **not redistributed** in this repository because of rights restrictions; obtain a licensed local copy per [`data/README.md`](../data/README.md) |

Override paths when needed:

| Variable | Default | Notes |
|----------|---------|-------|
| `ISWA_DIR` | `../iswa2026` | Traceability manuscript LaTeX root |
| `TAPKO_VIDEO` | `data/tapko/videos/jedi_submissions.mp4` | |
| `TAPKO_ANNOTATIONS` | `data/tapko/annotations/jedi_submissions.json` | |
| `REPRO_USE_REFERENCE` | `0` | |
| `ISWA_DETECT_DIR` | `outputs/tapko/jedi_submissions` | |
| `ISWA_EVAL_DIR` | `outputs/tapko/jedi_submissions_eval` | |
| `ISWA_REPRO_DIR` | `outputs/repro/iswa2026` | |

---

## Reproduction commands

Run from the **repository root** (`fightsafe-ai/`).

### Full pipeline (requires local video)

```bash
ISWA_DIR=../iswa2026 make reproduce-iswa
```

Equivalent:

```bash
ISWA_DIR=../iswa2026 bash scripts/reproduce_iswa2026.sh
```

This runs, in order:

1. `fightsafe tapko-validate-annotations --annotations data/tapko/annotations/jedi_submissions.json`
2. `fightsafe tapko-detect --source data/tapko/videos/jedi_submissions.mp4 --output-dir outputs/tapko/jedi_submissions --fps 30 --pose-backend mediapipe`
3. `fightsafe tapko-evaluate --annotations data/tapko/annotations/jedi_submissions.json --predictions outputs/tapko/jedi_submissions/tapko_predictions.json --output-dir outputs/tapko/jedi_submissions_eval --tolerance-seconds 0.5 --match-mode family`
4. `python scripts/export_iswa2026_tables.py --install --iswa-dir ../iswa2026`
5. Optional: compile `../iswa2026/main.pdf` when `pdflatex` is available
6. `python scripts/verify_paper_outputs.py --paper iswa`

### Reference mode (no video)

```bash
REPRO_USE_REFERENCE=1 ISWA_DIR=../iswa2026 bash scripts/reproduce_iswa2026.sh
```

Reference mode copies bundled detector/evaluator exports, regenerates tables, and verifies metrics without re-running pose inference on withheld video.

### Verify metrics against bundled reference

```bash
python scripts/verify_paper_outputs.py --paper iswa
```

### Compile manuscript PDF (optional)

```bash
cd ../iswa2026 && bash build.sh
```

---

## Expected metrics (micro row, draft references)

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

## Limitations

- **Video not redistributed:** full end-to-end detection requires a licensed local copy ([`data/README.md`](../data/README.md)).
- **Reference mode:** verifies table regeneration and metric consistency using bundled exports; does not re-run pose inference on withheld video.
- **Machine-side only:** reviewer and gate audit records were not logged; the artifact supports future audit-schema exercises but does not demonstrate operator benefit.
- **Single-clip protocol demonstration:** documents export bookkeeping under partial observability, not operational deployment readiness.

---

## See also

- [`README.md`](../README.md) — installation and iswa2026 quick reference
- [`ISWA2026_TRACEABILITY_MATRIX.md`](ISWA2026_TRACEABILITY_MATRIX.md) — manuscript-to-artifact map
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — all companion manuscripts
