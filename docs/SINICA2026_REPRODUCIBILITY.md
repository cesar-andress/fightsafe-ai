# sinica2026 reproducibility guide

Post-submission documentation for the JAS / TapKO HITL manuscript. Documentation-only; no software or metric changes beyond what **v0.1.3** already ships.

Paper-specific instructions for reproducing the TapKO / HITL workflow demonstration reported in:

**Human Supervisory Control for Explainable Escalation of AI-Generated Safety Alerts Under Partial Observability**

LaTeX sources: `../sinica2026/` (monorepo layout: `papers/sinica2026/`).

General installation, environment variables, and cross-manuscript notes: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Software archive

| Field | Value |
|-------|-------|
| Repository release | **v0.1.3** (`pyproject.toml`, `CITATION.cff`, `src/fightsafe_ai/__version__.py`) |
| Zenodo DOI | [10.5281/zenodo.20622869](https://doi.org/10.5281/zenodo.20622869) |
| GitHub tag | `v0.1.3` (see [`docs/release_checklist.md`](release_checklist.md)) |

---

## Scope

Reproduce the **`jedi_submissions` diagnostic pilot**: pipeline traceability, evaluator exports, and manuscript Tables I–II (`tapko_pilot_results`, `tapko_pilot_per_class`). This is **not** headline accuracy validation; draft reference intervals are research bookkeeping only (see [`data/README.md`](../data/README.md)).

**Evaluator matching defaults** (fixed in the reported pilot; also used by `scripts/reproduce_sinica2026.sh`):

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
| `data/repro/sinica2026/reference/tapko_predictions.json` | Reference detector export (`jedi_submissions`) |
| `data/repro/sinica2026/reference/tapko_results.csv` | Reference evaluator metrics |
| `data/repro/sinica2026/reference/tapko_error_analysis.md` | Reference error category counts |
| `data/repro/sinica2026/reference/tapko_manifest.json` | Optional manifest (copied in reference mode when present) |

### External (full pipeline only)

| Path | Description |
|------|-------------|
| `data/tapko/videos/jedi_submissions.mp4` | ~200 MB instructional clip — **not redistributed** in this repository because of rights restrictions; obtain a licensed local copy per [`data/README.md`](../data/README.md) (`video_url` field in `jedi_submissions.json`, `yt-dlp` example provided there) |

Override paths when needed:

| Variable | Default |
|----------|---------|
| `SINICA_DIR` | `../sinica2026` |
| `TAPKO_VIDEO` | `data/tapko/videos/jedi_submissions.mp4` |
| `TAPKO_ANNOTATIONS` | `data/tapko/annotations/jedi_submissions.json` |
| `REPRO_USE_REFERENCE` | `0` |
| `SINICA_DETECT_DIR` | `outputs/tapko/jedi_submissions` |
| `SINICA_EVAL_DIR` | `outputs/tapko/jedi_submissions_eval` |
| `SINICA_REPRO_DIR` | `outputs/repro/sinica2026` |

---

## Reproduction commands

Run from the **repository root** (`fightsafe-ai/`).

### Full pipeline (requires local video)

```bash
make reproduce-sinica
```

Equivalent:

```bash
bash scripts/reproduce_sinica2026.sh
```

This runs, in order:

1. `fightsafe tapko-validate-annotations --annotations data/tapko/annotations/jedi_submissions.json`
2. `fightsafe tapko-detect --source data/tapko/videos/jedi_submissions.mp4 --output-dir outputs/tapko/jedi_submissions --fps 30 --pose-backend mediapipe`
3. `fightsafe tapko-evaluate --annotations data/tapko/annotations/jedi_submissions.json --predictions outputs/tapko/jedi_submissions/tapko_predictions.json --output-dir outputs/tapko/jedi_submissions_eval --tolerance-seconds 0.5 --match-mode family`
4. `python scripts/export_sinica2026_tables.py --install --sinica-dir ../sinica2026` (paths resolved by the script from evaluator outputs)
5. Optional: compile `../sinica2026/main.pdf` when `pdflatex` is available
6. `python scripts/verify_paper_outputs.py --paper sinica`

### Reference mode (no video)

When the source video is unavailable, bundled reference predictions regenerate the manuscript tables and metrics:

```bash
REPRO_USE_REFERENCE=1 bash scripts/reproduce_sinica2026.sh
```

Reference mode:

1. Copies `data/repro/sinica2026/reference/tapko_predictions.json` (and `tapko_manifest.json` when present) into `outputs/tapko/jedi_submissions/`
2. Copies `data/repro/sinica2026/reference/tapko_results.csv` into `outputs/tapko/jedi_submissions_eval/` when present
3. Re-runs `fightsafe tapko-evaluate` if no evaluator CSV is available
4. Exports and installs LaTeX tables, then runs verification

`make reproduce-all` invokes sinica reproduction in reference mode automatically when the video file is missing (`scripts/reproduce_all.sh`).

### Export tables only

```bash
python scripts/export_sinica2026_tables.py --install --sinica-dir ../sinica2026
```

Requires existing evaluator outputs at `outputs/tapko/jedi_submissions_eval/tapko_results.csv` and `outputs/tapko/jedi_submissions/tapko_predictions.json`.

### Verify metrics against bundled reference

```bash
make verify-repro
# or:
python scripts/verify_paper_outputs.py --paper sinica
```

### Compile manuscript PDF (optional)

```bash
cd ../sinica2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Expected outputs

### Intermediate (TapKO run)

| Path | Produced by |
|------|-------------|
| `outputs/tapko/jedi_submissions/tapko_predictions.json` | `fightsafe tapko-detect` or reference copy |
| `outputs/tapko/jedi_submissions_eval/tapko_results.csv` | `fightsafe tapko-evaluate` |
| `outputs/tapko/jedi_submissions_eval/tapko_error_analysis.md` | `fightsafe tapko-evaluate` |

### Reproduction bundle

| Path | Description |
|------|-------------|
| `outputs/repro/sinica2026/tables/tapko_pilot_results.tex` | Regenerated pilot results table |
| `outputs/repro/sinica2026/tables/tapko_pilot_per_class.tex` | Regenerated per-class table |
| `outputs/repro/sinica2026/tapko_results.csv` | Copy of evaluator CSV |
| `outputs/repro/sinica2026/tapko_error_analysis.md` | Copy of error analysis (when produced) |
| `outputs/repro/sinica2026/main.pdf` | Copy of compiled manuscript (when `pdflatex` succeeds) |

Architecture figures (`figures/fig*_tikz.tex`) are LaTeX/TikZ sources compiled inside the PDF; no software run is required to regenerate them.

---

## Generated manuscript tables

With `--install`, `scripts/export_sinica2026_tables.py` copies regenerated fragments into the manuscript tree:

| Manuscript path | Label |
|-----------------|-------|
| `../sinica2026/tables/tapko_pilot_results.tex` | `tab:tapko_pilot_results` |
| `../sinica2026/tables/tapko_pilot_per_class.tex` | `tab:tapko_pilot_per_class` |

---

## Expected metrics (micro row, draft references)

Verification compares the reproduced `micro` row in `tapko_results.csv` against `data/repro/sinica2026/reference/tapko_results.csv`:

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

- **Video not redistributed:** the instructional clip `jedi_submissions` is excluded from Git and Zenodo because of rights restrictions. Full end-to-end detection requires a licensed local copy at `data/tapko/videos/jedi_submissions.mp4` ([`data/README.md`](../data/README.md)).
- **Reference mode:** `REPRO_USE_REFERENCE=1` verifies table regeneration and metric consistency using bundled detector/evaluator exports; it does not re-run pose inference on withheld video.
- **Draft references:** `jedi_submissions.json` intervals are not visually confirmed ground truth ([`data/README.md`](../data/README.md)).
- **Single-clip pilot:** reproduction covers one diagnostic demonstration (`jedi_submissions`); it documents workflow traceability under partial observability, not operational deployment readiness.

---

## See also

- [`README.md`](../README.md) — installation and sinica2026 quick reference
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — all companion manuscripts
- [`data/repro/README.md`](../data/repro/README.md) — reference snapshot policy
- [`scripts/README.md`](../scripts/README.md) — script index
