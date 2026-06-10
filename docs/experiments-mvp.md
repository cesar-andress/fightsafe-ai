# Experiments (FightSafe AI — MVP Protocol and Reporting Template)

**Authors (context).** David Martin Moncunill, César Andrés Sánchez; Camilo José Cela University (UCJC), Madrid, Spain.

**Important.** The FightSafe AI repository **does not ship evaluation videos or labels** (see `docs/dataset.md`). This document is a **template** for a realistic experiment section: **procedures, metrics, and ablation definitions** are specified; **numerical results and clip counts are not invented**—use **TBD** until you have real measurements, or omit quantitative subsections.

**Compliance.** Outputs are **decision-support** only—not medical or governing-body ground truth.

---

## 1. Experimental setup

### 1.1 Dataset description

- **Provenance.** Curate combat-sports clips from sources you are licensed to use (e.g. public material with platform terms respected, or controlled capture with consent). Store raw video outside public version control; keep a metadata sheet with source, date, and anonymization policy (`docs/dataset.md`).
- **Inclusion (example).** Clips (e.g. 10–60 s) that contain striking or grappling phases where balance and posture vary; prefer multiple camera angles when feasible.
- **Exclusion (example).** Clips failing quality control (frame–keypoint misalignment, unreadable sync) or, if the study defines it, heavily unusable occlusion—document **drops** and **reasons** (counts: TBD per study).

**Table: study scale (placeholders only).**

| Item | Value |
|------|--------|
| Number of clips in the evaluation set | **TBD** |
| Train / validation / test split (if any) | **TBD** |
| Frame extraction FPS, resolution, codec | **TBD** (fixed per run) |

### 1.2 Types of events (system + reference)

**System-side (heuristic indicators).** The rule engine uses named indicators configured in `configs/risk_rules.yaml`, for example: `fast_downward_motion`, `large_torso_angle`, `prolonged_low_posture`, `high_instability`, `post_fall_low_movement`. These are **engineering labels** for traceability, not clinical diagnoses.

**Reference (annotation) categories** for evaluation should be defined in a **codebook** agreed before labeling, e.g.:

| Code | Description (example) |
|------|------------------------|
| E1 | Sustained bent / low torso posture (visible sustained stoop or crouch) |
| E2 | Rapid downward CoM / hip dynamics (e.g. sudden level change) |
| E3 | Erratic or unstable base within the clip |
| E4 | Prolonged low posture or very low kinematic activity (aligned with heuristic “near ground + low speed” sense) |
| (opt.) | Absence of event of interest (for negative spans, if explicitly annotated) |

**Rater protocol.** Number of annotators, independence vs. consensus, blinding to run IDs, and disagreement resolution (e.g. κ or adjudication): **TBD** and **preregistered** where possible. Do not report inter-rater numbers until measured.

---

## 2. Evaluation metrics

Metrics are defined **with respect to** your annotation codebook and time alignment—not as absolute “accuracy of safety.”

### 2.1 Event detection: precision, recall, F1

- **Preferred: event-level matching.** Convert system output to time intervals (merged events as produced by the pipeline) and compare to expert intervals. A **match** if temporal **IoU** ≥ *t* (e.g. 0.3 or 0.5; preregister *t*), optionally with onset/offset tolerance ±δ ms. Unmatched predictions → FP; unmatched references → FN.
- **Auxiliary: frame-level metrics** (optional). Compare per-frame risk or flag to a frame-wise mask derived from expert spans—use cautiously; combat video is highly correlated in time.

**Definitions.** Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = harmonic mean. Report macro- or micro-averaging as appropriate for imbalance. All **numeric** results: **TBD** until computed on your labeled set.

### 2.2 Qualitative validation (essential for MVP)

- **Overlay review:** random subset of *K* clips (TBD) with a fixed checklist (e.g. “Is the triggered rule visually plausible for this segment?” yes/no).
- **Error taxonomy (no counts without a study):** occlusion, motion blur, tight framing, 2D pose failure, mismatch between heuristic “risk” and referee concepts of foul or unsafety.
- **Optional LLM text:** assess faithfulness to tabulated fields separately from detector F1 (human review).

---

## 3. Results (template — no fabricated numbers)

Use **consistent figure numbering** with `docs/methodology-imrad.md`: **Figure 1** (architecture), **Figure 2** (pipeline flow), **Figure 3** (risk timeline), **Figure 4** (event detection), **Figure 5** (pose features), **Figure 6** (risk levels / example signals). Rebuild schematics with `python docs/figures/generate_paper_figures.py`, or add run-specific figures from real evaluations (with consent and redaction as required).

### 3.1 Quantitative (fill after labeling)

- Number of **predicted** events (total and by indicator class): **TBD**
- Number of **reference** events: **TBD**
- Precision, recall, F1 (and optional stratification by event type or occlusion): **TBD**

Complementary **plots** (not a substitute for tables) may show risk vs. time in the same **layout** as **Figure 3**; **caption** clearly whether each plot is a **schematic** (synthetic) or a **real** run (e.g. *“Representative output from the evaluation set, not the synthetic template in Figure 3”*).

### 3.2 Example cases and failure cases (qualitative)

- **Illustrative positives:** 2–3 clips (with permissions / redaction) describing overlay and risk timeline—**only** with real study examples. The narrative can parallel **Figure 3** (risk over time) and **Figure 4** (merging into event intervals, qualitatively).
- **Failure modes (described, not counted without data):**
  - **FP:** aggressive tuning may flag fast lateral movement or feints as instability.
  - **FN:** heavy occlusion of legs/hips degrades landmarks; rules may miss a concerning phase.
  - **Concept gap:** heuristic “risk” ≠ official foul or medical “unsafety”—expected under MVP.

**Figures:** `docs/figures/` **Figures 1–6** are publication-ready schematics. For a **Results** section backed by data, add run-specific time series (Figure 3–style), event-alignment or IoU schematics (Figure 4–style), and keypoint or overlay frames (Figure 5–style or actual frame grabs).

---

## 4. Ablation (simple)

Ablations toggles **one** group at a time on the **same** clips, with the same postprocessing. Operational mapping references `configs/risk_rules.yaml` and the feature columns consumed by `fightsafe_ai.risk` / `fightsafe_ai.features` (see code and config comments).

| ID | What is removed or reduced | Expected qualitative effect (not a number without an experiment) |
|----|------------------------------|-------------------------------------------------------------------|
| **A1** | Temporal context: e.g. reduce rolling windows or streak-based logic to minimum | Less persistence; more one-frame noise; worse capture of **sustained** states. |
| **A2** | **Instability:** zero weight or disable `high_instability` in aggregation | Fewer alerts on erratic motion; may miss some unstable patterns. |
| **A3** | **Posture / duration:** disable or tighten `prolonged_low_posture` and related ground-streak heuristics | Fewer long-bent-posture flags; more risk of missing sustained low postures. |

**Reporting:** P/R/F1 or agreement per ablation on the test split, or state “not yet evaluated.”

*Note:* Feature groups are **correlated**; ablations are **not** independent causal factors.

---

## 5. Discussion (strengths and weaknesses)

### 5.1 Strengths (MVP-realistic)

- **Interpretability and audit:** YAML rules and triggered indicators support post-hoc inspection.
- **Reproducibility:** versioned config and tabular exports (CSV/JSON).
- **Human-in-the-loop:** designed for review workflows, not autonomous sanctions.

### 5.2 Weaknesses (intrinsic to MVP + monocular heuristics)

- **No in-repo benchmark**—quantitative claims require a completed annotation effort.
- **2D pose** and **heuristic rules** are limited by **occlusion, viewpoint,** and **domain shift.**
- **Ablations** are correlated; interpret marginal differences cautiously.
- **Optional LLM** explanations must be evaluated for **factual grounding** separately from event metrics.

---

## Using this in a manuscript

1. **No labels yet:** Include Sections 1, 2, 4, and 5; in Section 3, one sentence: *“Quantitative results are pending a curated, preregistered annotation study.”*
2. **Pilot complete:** Replace every **TBD** with measured values, add CIs if appropriate, and a supplementary metadata table (no re-identifying links).
3. **Artifacts:** provide evaluation scripts and config hashes; raw video only per ethics and license.

---

*This document does not replace institutional ethics review, informed consent, or journal-specific reporting standards.*
