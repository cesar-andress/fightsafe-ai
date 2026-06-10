# jss2026 traceability matrix

Manuscript-to-artifact map for the *Journal of Systems and Software* submission:

**A Formal Traceability Architecture and Auditable Human-Oversight Specification for Machine-Generated Safety Alerts**

LaTeX root: `jss2026/` (sibling of `fightsafe-ai/` in the monorepo layout `papers/fightsafe-ai/`).

**Purpose:** Every empirical table, figure, metric, and experiment reported in the manuscript should link to a public repository artefact or an explicit specification source. Reproduction entry point: `ESWA_DIR=../jss2026 bash scripts/reproduce_eswa2026.sh` (see [`JSS2026_REPRODUCIBILITY.md`](JSS2026_REPRODUCIBILITY.md)).

**Legend**

| Traceability class | Meaning |
|--------------------|---------|
| **Computed** | Value produced by software from bundled or locally obtained inputs |
| **Authorial** | Specification written in LaTeX; no software run required |
| **Reference snapshot** | Bundled export used for verification when video is withheld |
| **Not logged** | Claim stated in specification text; no artefact recorded in the stress test |

---

## Summary

| Category | Count | Public artefact coverage |
|----------|-------|--------------------------|
| Tables (all) | 6+ | 2 computed + remainder authorial |
| Figures | 5+ | TikZ / compiled in PDF |
| Reported stress-test metrics | 18+ scalars | All trace to `tapko_results.csv` / `tapko_predictions.json` |
| Experiments executed | 1 | `jedi_submissions` machine-side stress test |
| Review-interface / gate logs | 0 | **Not logged** |

---

## Key tables

| Label | Claim | Manuscript path | Generator | Artefact |
|-------|-------|-----------------|-----------|----------|
| `tab:tapko_pilot_results` | Interval bookkeeping, FP/min | `jss2026/tables/tapko_pilot_results.tex` | `export_eswa2026_tables.py` | **Computed** |
| `tab:tapko_pilot_per_class` | Per-channel TP/FP/FN | `jss2026/tables/tapko_pilot_per_class.tex` | same | **Computed** |
| `tab:formal-symbols` | Formal notation | `jss2026/sections/method.tex` | — | **Authorial** |
| `tab:design-decisions` | D-TK000–D-TK010 governance commitments | `jss2026/sections/11_design_decisions.tex` | — | **Authorial** |

Bundled reference snapshots: `fightsafe-ai/data/repro/eswa2026/reference/` (internal folder name at artifact creation).

---

## Claims without logged public artefacts

| Claim domain | Status |
|--------------|--------|
| Reviewer decisions `H_{t,k}` | **Not logged** |
| Gate outcomes `O_{t,k}` | **Not logged** |
| Operator performance / trust / workload | **Not evaluated** — no human-subject study |
| Deployment readiness | **Not claimed** |
| E1–E5 programme results | **Not executed** — roadmap only |

---

## Reproduction (quick reference)

```bash
cd fightsafe-ai
REPRO_USE_REFERENCE=1 ESWA_DIR=../jss2026 bash scripts/reproduce_eswa2026.sh
python scripts/verify_paper_outputs.py --paper eswa
cd ../jss2026 && bash build.sh
```

---

## See also

- [`JSS2026_REPRODUCIBILITY.md`](JSS2026_REPRODUCIBILITY.md)
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- [`../data/repro/README.md`](../data/repro/README.md)
