# eswa2026 traceability matrix

Post-submission documentation for the JAS / TapKO HITL manuscript. Maps manuscript claims to repository artefacts; no software or metric changes beyond **v0.1.3**.

Manuscript-to-artifact map for:

**A Traceable Machine-Side Alert Ranking and Audit Specification for Human Oversight**

LaTeX root: `eswa2026/` (sibling of `fightsafe-ai/` in the monorepo layout `papers/fightsafe-ai/`).

**Purpose:** Every empirical table, figure, metric, and experiment reported in the manuscript should link to a public repository artefact or an explicit authorial/specification source. Reproduction entry point: `bash scripts/reproduce_eswa2026.sh` (see [`ESWA2026_REPRODUCIBILITY.md`](ESWA2026_REPRODUCIBILITY.md)).

**Legend**

| Traceability class | Meaning |
|--------------------|---------|
| **Computed** | Value produced by software from bundled or locally obtained inputs |
| **Authorial** | Specification or roadmap written in LaTeX; no software run required |
| **Reference snapshot** | Bundled export used for verification when video is withheld |
| **Not logged** | Claim stated in text; no artefact was recorded in the pilot run |

---

## Summary

| Category | Count | Public artefact coverage |
|----------|-------|--------------------------|
| Tables (all) | 6 | 2 computed + 4 authorial |
| Figures | 3 | 3 authorial TikZ (compiled in PDF) |
| Reported pilot metrics | 18+ scalars | All trace to `tapko_results.csv` / `tapko_predictions.json` |
| Experiments executed | 1 | `jedi_submissions` workflow demonstration |
| Experiments planned only | E1–E5 | Roadmap table only (`tab:eval-programme`) |

---

## Master matrix

Paths below use repository roots **`fightsafe-ai/`** (software) and **`eswa2026/`** (manuscript) unless noted.

### Tables

| Label | Manuscript claim | Source file (LaTeX) | Script / generator | Input data | Output artefact | Repository location |
|-------|------------------|---------------------|--------------------|------------|-----------------|---------------------|
| `tab:tapko_pilot_results` | Pilot interval bookkeeping (duration, counts, TP/FP/FN, P/R/F1/F2, latency, FP/min) | `eswa2026/tables/tapko_pilot_results.tex` ← `\input` in `sections/09_preliminary_tapko_experiments.tex` | `fightsafe-ai/scripts/export_eswa2026_tables.py` (`write_pilot_results_tex`) | `fightsafe-ai/data/tapko/annotations/jedi_submissions.json`; `outputs/tapko/jedi_submissions/tapko_predictions.json` (or reference copy); `outputs/tapko/jedi_submissions_eval/tapko_results.csv` from `fightsafe tapko-evaluate` | `eswa2026/tables/tapko_pilot_results.tex`; copy under `fightsafe-ai/outputs/repro/eswa2026/tables/` | **Computed** — evaluator: `fightsafe-ai/src/fightsafe_ai/evaluation/tapko_evaluator.py`; matching: `evaluation/event_matching.py`, `evaluation/event_metrics.py` |
| `tab:tapko_pilot_per_class` | Per-channel TP/FP/FN/P/R/F1 (NID, FT, HT) | `eswa2026/tables/tapko_pilot_per_class.tex` ← `\input` in `sections/09_preliminary_tapko_experiments.tex` | `fightsafe-ai/scripts/export_eswa2026_tables.py` (`write_per_class_tex`) | Same CSV + predictions as above | `eswa2026/tables/tapko_pilot_per_class.tex`; copy under `fightsafe-ai/outputs/repro/eswa2026/tables/` | **Computed** — per-class rows in `tapko_results.csv` (`scope=per_class`) |
| `tab:formal-symbols` | Formal notation ($V_t$, $A_t$, $H_t$, …) | Inline in `eswa2026/sections/method.tex` | — | — | Compiled PDF only | **Authorial** |
| `tab:design-decisions` | D-TK000–D-TK010 supervisory commitments (incl. D-TK009 matching parameters) | Inline in `eswa2026/sections/11_design_decisions.tex` | — | — | Compiled PDF only | **Authorial** — D-TK009 parameters enforced by `fightsafe tapko-evaluate --tolerance-seconds 0.5 --match-mode family` (IoU default `0.3`) |
| `tab:reg-mapping` | Validation-domain rule → proxy → channel → HITL mapping | Inline in `eswa2026/sections/regulatory_background.tex` | — | External rule citations (ABC, UWW, IBJJF, ADCC) | Compiled PDF only | **Authorial** — event namespaces align with `fightsafe-ai/src/fightsafe_ai/annotation/tapko_schema.py` |
| `tab:eval-programme` | Future stages E1–E5 (not executed) | Inline in `eswa2026/sections/10_future_work.tex` | — | — | Compiled PDF only | **Authorial** — no experiment run |

### Figures

| Label | Manuscript claim | Source file (LaTeX) | Script / generator | Input data | Output artefact | Repository location |
|-------|------------------|---------------------|--------------------|------------|-----------------|---------------------|
| `fig:hitl-workflow` | Supervisory path $A_t$ → queue → $H_t$ → $O_t$ | `eswa2026/figures/fig01_hitl_review_workflow_tikz.tex` ← `\input` in `sections/01_introduction.tex` | LaTeX/TikZ (`pdflatex`) | — | Compiled in `eswa2026/main.pdf` | **Authorial** — `eswa2026/figures/fig01_hitl_review_workflow_tikz.tex` |
| `fig:tapko-architecture` | Companion $A_t$ formation vs supervisory $A_t \rightarrow H_t \rightarrow O_t$ | `eswa2026/figures/fig02_tapko_architecture_tikz.tex` ← `\input` in `sections/method.tex` | LaTeX/TikZ | — | Compiled in `eswa2026/main.pdf` | **Authorial** — `eswa2026/figures/fig02_tapko_architecture_tikz.tex` |
| `fig:tapko-taxonomy` | Dual-channel taxonomy + confirmation gate | `eswa2026/figures/fig03_tapko_taxonomy_tikz.tex` ← `\input` in `sections/04_problem_definition.tex` | LaTeX/TikZ | — | Compiled in `eswa2026/main.pdf` | **Authorial** — `eswa2026/figures/fig03_tapko_taxonomy_tikz.tex`; optional SVG drafts in `eswa2026/figures/*.svg` (not `\includegraphics`'d in build) |

### Metrics (pilot demonstration, §`sec:tapko-pilot`)

All values below appear in `sections/09_preliminary_tapko_experiments.tex` and/or `tab:tapko_pilot_results` / `tab:tapko_pilot_per_class`.

| Metric / claim | Reported value | Primary artefact | Script / command | Input data | Repository location |
|----------------|----------------|------------------|------------------|------------|---------------------|
| Video ID | `jedi_submissions` | `jedi_submissions.json` | — | Bundled annotation | `fightsafe-ai/data/tapko/annotations/jedi_submissions.json` |
| Source | YouTube instructional | `source_uri` field | — | Metadata in annotation | Same file (`https://www.youtube.com/watch?v=ALEeReC3u5Y`) |
| Duration (min) | $11.1578$ | `tapko_results.csv` (`scope=micro`, `total_video_duration_min`) | `fightsafe tapko-evaluate` | Predictions + annotations | `fightsafe-ai/data/repro/eswa2026/reference/tapko_results.csv` (reference); regenerated at `outputs/tapko/jedi_submissions_eval/tapko_results.csv` |
| Reference intervals | $10$ | `jedi_submissions.json` (`events` array, length 10) | `fightsafe tapko-validate-annotations` | Bundled annotation | `fightsafe-ai/data/tapko/annotations/jedi_submissions.json` |
| Predicted candidates | $337$ | `tapko_predictions.json` (list length) | `fightsafe tapko-detect` **or** reference copy | Video **or** reference JSON | `fightsafe-ai/data/repro/eswa2026/reference/tapko_predictions.json`; `export_eswa2026_tables.py` counts rows |
| TP | $1$ | `tapko_results.csv` micro row | `fightsafe tapko-evaluate` | Annotations + predictions | Reference: `data/repro/eswa2026/reference/tapko_results.csv` |
| FP | $336$ | same | same | same | same |
| FN | $9$ | same | same | same | same |
| Precision | $0.0030$ | same | same | same | same |
| Recall | $0.1000$ | same | same | same | same |
| F1 | $0.0058$ | same | same | same | same |
| F2 | $0.0133$ | Table I only | same | same | same |
| Mean onset latency (s) | $2.5000$ | Table I / CSV | same | same | same |
| False positives / min | $30.1135$ | same | same | same | same |
| NID alerts (FP, zero overlap) | $198$ | `tapko_results.csv` per-class + Table II | same | same | `scope=per_class`, `label=extreme_vulnerability.no_intelligent_defense` |
| FT alerts (FP) | $106$ | same | same | same | `label=submission_signal.foot_tap` |
| HT TP/FP/FN | $1$ / $32$ / $9$ | Table II | same | same | `label=submission_signal.hand_tap` |
| Error tag: false\_positive | $335$ | `tapko_error_analysis.md` | `fightsafe tapko-evaluate` | same | `fightsafe-ai/data/repro/eswa2026/reference/tapko_error_analysis.md` |
| Error tag: missed\_event | $8$ | same | same | same | same |
| Error tag: wrong\_subtype | $2$ | same | same | same | same |
| Error tag: late\_detection | $1$ | same | same | same | same |

**Verification:** `python fightsafe-ai/scripts/verify_paper_outputs.py --paper eswa` compares reproduced `micro` row to `data/repro/eswa2026/reference/tapko_results.csv`.

### Experiments

| Experiment ID | Manuscript section | Description | Orchestration script | Pipeline steps | Inputs | Outputs | Repository location |
|---------------|-------------------|-------------|----------------------|----------------|--------|---------|---------------------|
| **EXP-1** `jedi_submissions` pilot | `sec:tapko-pilot`, `sec:eval` | Single-clip workflow traceability demonstration (detect → evaluate → table export); not human-subject study | `fightsafe-ai/scripts/reproduce_eswa2026.sh` | See below | See below | See below | Full chain under `fightsafe-ai/` |

#### EXP-1 pipeline steps

| Step | Command / module | Input | Output artefact | Location |
|------|------------------|-------|-----------------|----------|
| 0 (optional) | `fightsafe tapko-validate-annotations` → `annotation/tapko_schema.py` | `data/tapko/annotations/jedi_submissions.json` | stdout validation | `fightsafe-ai/data/tapko/annotations/jedi_submissions.json` |
| 1a (full) | `fightsafe tapko-detect` → `tapko/detect_run.py`, `events/tap_detector.py`, `events/vulnerability_detector.py` | `data/tapko/videos/jedi_submissions.mp4` (not in Git); `--fps 30 --pose-backend mediapipe` | `tapko_predictions.json`, `tapko_predictions.csv`, `tapko_report.md`, `tapko_manifest.json`, `pose_keypoints.csv` | `fightsafe-ai/outputs/tapko/jedi_submissions/` |
| 1b (reference) | Copy bundled reference | `data/repro/eswa2026/reference/tapko_predictions.json` | Same filenames under detect dir | `fightsafe-ai/data/repro/eswa2026/reference/` → `outputs/tapko/jedi_submissions/` |
| 2 | `fightsafe tapko-evaluate` → `evaluation/tapko_evaluator.py` | Annotations + predictions; `--tolerance-seconds 0.5 --match-mode family` (IoU threshold default `0.3`) | `tapko_results.csv`, `tapko_results.tex`, `tapko_error_analysis.md` | `fightsafe-ai/outputs/tapko/jedi_submissions_eval/`; reference: `data/repro/eswa2026/reference/` |
| 3 | `scripts/export_eswa2026_tables.py --install` | Evaluator CSV + predictions JSON | `tapko_pilot_results.tex`, `tapko_pilot_per_class.tex` | `eswa2026/tables/`; repro copy: `outputs/repro/eswa2026/tables/` |
| 4 | `pdflatex` + `bibtex` (optional, in repro script) | `eswa2026/main.tex`, `sections/*.tex`, tables, figures | `eswa2026/main.pdf` | `eswa2026/main.pdf`; copy: `outputs/repro/eswa2026/main.pdf` |
| 5 | `scripts/verify_paper_outputs.py --paper eswa` | Reproduced vs reference CSV | stdout pass/fail | `fightsafe-ai/scripts/verify_paper_outputs.py` |

**Matching configuration** (fixed for reported results): documented in `eswa2026/sections/07_evaluation_protocol.tex` and `sections/11_design_decisions.tex` (D-TK009); implemented via CLI flags above and `TapkoEvalConfig` in `tapko_evaluator.py`.

---

## Supporting artefacts (not standalone tables/figures)

| Artefact | Role in manuscript | Generator | Repository location |
|----------|-------------------|-----------|---------------------|
| `tapko_predictions.json` | Machine-side $A_t$ export; 337 candidate intervals | `fightsafe tapko-detect` | `fightsafe-ai/data/repro/eswa2026/reference/tapko_predictions.json` |
| `tapko_manifest.json` | Run metadata (fps, pose source, video path) | `tapko/detect_run.py` | `fightsafe-ai/data/repro/eswa2026/reference/tapko_manifest.json` |
| `tapko_results.csv` | Evaluator metrics (micro/macro/per_class) | `fightsafe tapko-evaluate` | `fightsafe-ai/data/repro/eswa2026/reference/tapko_results.csv` |
| `tapko_error_analysis.md` | Error taxonomy counts + examples (§Explainable error analysis) | `fightsafe tapko-evaluate` | `fightsafe-ai/data/repro/eswa2026/reference/tapko_error_analysis.md` |
| `jedi_submissions.json` | Draft reference windows (10 intervals) | Manual / transcript-derived draft | `fightsafe-ai/data/tapko/annotations/jedi_submissions.json` |
| TapKO schema docs | Annotation protocol semantics | — | `fightsafe-ai/docs/tapko_annotation.md` |
| Software release | Data & Code Availability cite | Zenodo/GitHub tag `v0.1.3` | `fightsafe-ai/CITATION.cff`, DOI `10.5281/zenodo.20622869` |

---

## Claims without logged public artefacts

These statements appear in the manuscript but were **not** instrumented in EXP-1 (explicitly acknowledged in text):

| Claim domain | Manuscript location | Traceability status |
|--------------|--------------------|--------------------|
| Human decisions $H_t$ | `sec:tapko-pilot`, abstract, `sec:limitations` | **Not logged** — confirmation workflow specified, not exercised on footage |
| Gated outcomes $O_t$ | same | **Not logged** |
| Operator performance / trust | `sec:discussion`, `sec:future` | **Not logged** — no human-subject study |
| Multi-rater $\kappa$ / $\alpha$ | `sec:annotation`, E1 | **Not executed** — protocol only |
| E1–E5 programme results | `tab:eval-programme`, `sec:future` | **Not executed** — roadmap only |
| Full pipeline re-run without video | Data Availability | Use **reference mode**: `REPRO_USE_REFERENCE=1 bash scripts/reproduce_eswa2026.sh` |

---

## Reproduction commands (quick reference)

```bash
cd fightsafe-ai

# Full pipeline (requires local video at data/tapko/videos/jedi_submissions.mp4)
bash scripts/reproduce_eswa2026.sh

# Reference mode (bundled predictions; regenerates Tables I–II and verifies metrics)
REPRO_USE_REFERENCE=1 bash scripts/reproduce_eswa2026.sh

# Manuscript PDF only
cd ../eswa2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Coverage assessment

| Requirement | Status |
|-------------|--------|
| Every **empirical table** traces to a CSV/JSON + export script | **Yes** (`tab:tapko_pilot_*`) |
| Every **figure** traces to a versioned source file | **Yes** (TikZ under `eswa2026/figures/`) |
| Every **reported pilot metric** traces to evaluator output | **Yes** (`tapko_results.csv`, `tapko_error_analysis.md`, `tapko_predictions.json`) |
| The **single executed experiment** traces to reproducible script chain | **Yes** (`reproduce_eswa2026.sh`) |
| Specification / roadmap tables trace to authorial LaTeX | **Yes** (no software artefact expected) |
| Withheld video documented | **Yes** — `data/README.md`; reference snapshots ship without video |

**Conclusion:** All quantitative claims in the eswa2026 pilot demonstration map to public artefacts in `fightsafe-ai/data/` and `fightsafe-ai/data/repro/eswa2026/reference/`, regenerated via `scripts/reproduce_eswa2026.sh`. Non-quantitative architectural claims map to TikZ figures and inline specification tables in `eswa2026/sections/`.

---

## See also

- [`ESWA2026_REPRODUCIBILITY.md`](ESWA2026_REPRODUCIBILITY.md) — step-by-step reproduction
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — cross-manuscript overview
- [`../data/repro/README.md`](../data/repro/README.md) — reference snapshot policy
