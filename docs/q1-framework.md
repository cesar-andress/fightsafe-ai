# FightSafe AI as a Q1-Level Research Framework

**Authors (software):** David Martín Moncunill, César Andrés Sánchez

**Affiliation:** Camilo José Cela University (UCJC), Madrid, Spain

This note positions **FightSafe AI** as a **Q1-level research framework**: an early-stage, modular system intended to **structure hypotheses**, **expose extension points**, and **support human-in-the-loop** analysis in combat-sports video—not as a validated product for real-time officiating or clinical use. Scope and claims are deliberately conservative.

---

## 1. Motivation

Combat and striking sports place heavy cognitive load on referees and safety personnel. **Computer vision** can, in principle, offer **repetitive, time-aligned cues** (pose, motion, proximity to the canvas) that complement human observation. However, full automation of stoppages or medical triage is **ethically and technically** inappropriate in current open research. FightSafe is motivated by a narrower goal: to **organize** perception pipelines into testable components that emit **interpretable, auditable signals** for **off-line or lab-side** review, training, and method comparison. The framework does not assert regulatory readiness or injury prediction.

---

## 2. Literature-informed design

The architecture is **informed by** ongoing trends in **sports and activity analysis**, without implying that the implementation replicates the state of the art in every layer:

- **Sports human pose estimation** and **combat-sports pose** work suggest that 2D/3D landmarks are a practical common currency for video analytics, subject to domain shift and occlusion.
- **Skeleton-based action recognition** motivates treating temporal sequences of joints as features for heuristics or classifiers, while acknowledging that small datasets and class imbalance remain open problems.
- **Sports multi-object tracking (MOT)** underpins the need to associate people with **consistent identities** across frames when multiple fighters (and entourage) are visible.
- **Fall detection** and related **inactivity** or **collapse** indicators from video or pose motivate **anomaly-style** modules as *signals*, not ground truth.
- **Video anomaly detection** encourages framing rare or abrupt motion as **candidates** for review rather than as definitive events.
- **Explainable decision-support** literature supports **post-hoc** narratives and **constraint-aware** text generation so that model outputs remain **legible to humans** without overconfident language.

The codebase reflects these influences as **pluggable modules** and **documented heuristics**, not as a closed theory.

---

## 3. Framework architecture

FightSafe AI is organized as a **directed processing pipeline** with **stable join keys** (frame index, derived time) between stages: **ingestion and clipping** → **frame extraction** → **pose estimation** → **feature engineering** → **multi-signal processing** (tracking, action, safety anomalies) → **risk fusion** → **referee-oriented HCI** → **optional** natural-language explanation. Configuration is **declarative** (YAML) where possible, so that experiments can vary components without rewrites of unrelated layers. A detailed stage-wise description of the **MVP** path appears in `docs/architecture.md` and a compact module map in `docs/framework.md`.

---

## 4. Component taxonomy

| Concern | Package (illustrative) | Function |
|--------|-------------------------|----------|
| Video | `fightsafe_ai.video` | Ingest, clip, extract frames |
| Keypoints I/O | `fightsafe_ai.keypoints` | Load and normalize pose tables |
| Pose | `fightsafe_ai.pose` | `BasePoseEstimator` and backends (e.g., MediaPipe) |
| Features | `fightsafe_ai.features` | Biomechanical proxies from landmarks |
| Tracking | `fightsafe_ai.tracking` | `BaseTracker`, `SportsTracker`, identity helpers |
| Action | `fightsafe_ai.action` | Heuristic strike/guard signals, temporal MVP helpers |
| Anomaly / safety | `fightsafe_ai.anomaly` | Fall, inactivity, limb, surrender *heuristics* |
| Risk | `fightsafe_ai.risk` | Rules, levels, `fusion` for multi-signal experiments |
| HCI | `fightsafe_ai.hci` | `RefereeAlert`, messaging aligned to HITL |
| LLM | `fightsafe_ai.llm` | Optional Ollama client and explainers (post-hoc) |
| Evaluation | `fightsafe_ai.evaluation` | Metrics and comparison helpers |
| Datasets / QA / visualization | `fightsafe_ai.datasets`, `fightsafe_ai.qa`, `fightsafe_ai.visualization` | Metadata, quality checks, overlays |

This taxonomy is **evolutionary**: new backends can sit behind the same abstractions when schemas remain compatible.

---

## 5. Pose estimation layer

The **pose layer** recovers a **skeleton representation** (typically 2D landmarks with visibility) per video frame. The default path uses **MediaPipe**-style conventions, normalized coordinates, and **CSV** consolidation for traceability. Alternative estimators are expected to subclass a small **estimator base** and emit data consumable by shared loaders. The layer does **not** by itself perform semantic “understanding” of a bout; it merely supplies geometry for downstream features. Limitations (motion blur, self-occlusion, unusual camera angles) are inherited from monocular sports footage and should be treated as **error sources** in any evaluation.

---

## 6. Fighter tracking layer

When **multiple** people appear in the scene, **per-frame** pose or boxes must be **linked over time** for fighter-specific reasoning. The tracking package provides **abstractions** (e.g., `BaseTracker`, `SportsTracker`) and utilities such as **greedy identity assignment** via **IoU**, reflecting common **sports MOT** practice at a **research MVP** level. The current implementation is **not** presented as competition-grade tracking; it exists to keep **identity** explicit in the stack and to allow **replacement** with stronger detectors or association logic in future work.

---

## 7. Action recognition layer

The **action** layer groups **skeleton- and heuristics-based** components that emit **soft signals** (e.g., guard posture, turned back, punch/kick activity proxies) over time. Interfaces such as `BaseActionRecognizer` separate **algorithms** from feature plumbing. The emphasis is on **transparency and extensibility** rather than on a single pre-trained model for all combat rulesets. This aligns with **skeleton-based action recognition** in spirit, while the shipped heuristics should be read as **baselines** subject to **dataset-specific** calibration.

---

## 8. Safety anomaly detection layer

The **anomaly** module aggregates **research-grade** indicators—**fall** likelihood, **inactivity**, **limb** non-normative motion, and **surrender**-like gestures—into structured outputs. **Naming is functional**: these are **software signals** derived from pose and motion, **not** medical diagnoses, regulatory categories, or verified fight-ending events. The design draws on ideas from **fall detection** and **video anomaly** literature (rare, abrupt, or prolonged deviations) but does not claim **clinical** validity.

---

## 9. Risk fusion layer

**Risk fusion** combines heterogeneous cues into a **bounded** summary (e.g., a scalar in \([0,1]\) and a discrete **risk band** with **triggered** rule labels). The intent is **interpretable composition**: thresholds and weights are **inspected** in configuration, supporting **ablation** and error analysis. Fusion is a **laboratory construct** to compare pipeline variants; it is **not** a claim of optimal or universal risk. Multi-signal extensions are documented in code (e.g., `fightsafe_ai.risk.fusion`) and should be **reported with** the **exact** rule set used in each study.

---

## 10. Human-in-the-loop referee alerts

The **HCI** layer maps fused outputs to **referee-facing** messages that emphasize **recommendation to review** rather than **mandatory** outcomes. **Alert levels** and copy are **decision-support** oriented; they do **not** command stoppage or assert injury. This mirrors the ethical stance of **human-in-the-loop** systems: automation suggests **where** a human should look, not **what** the result of a contest must be.

---

## 11. Optional LLM explainability

An **optional** local **LLM** path (e.g., via Ollama) can generate **post-hoc** text summaries from **structured** context (risk, alert, signals, confidences, time span). By design, the LLM **does not** drive core detection, **does not** issue referee commands, and **must not** be read as a source of **medical** or **factual** certainty. When the model is **disabled** or **fails**, **deterministic** text preserves **auditability**—a pattern consistent with **constrained, explainable** decision-support rather than end-to-end opaque narratives.

---

## 12. Evaluation strategy

`fightsafe_ai.evaluation` is intended to support **quantitative** comparison: for example, **per-frame** or **per-interval** agreement with **sparse** or **imperfect** human labels, **temporal** overlap (IoU-style) for events, and **sensitivity** to ablations (pose backend, rules, missing tracks). A serious study should pre-register (even informally) **splits**, **inclusion** of difficult clips, and **failure** analysis (false positives in crowded scenes, **cross-domain** video). The framework **facilitates** these analyses; it does **not** by itself provide benchmark leadership.

---

## 13. Limitations

- **Data and domain shift:** Training-free or weakly specified domains (amateur video, new camera placements) can degrade all layers.
- **Heuristic risk:** Default rules and fusion weights are **tunable** and may **over-** or **under-react** to specific rule sets and promotions.
- **No clinical or regulatory use:** Outputs are **not** validated for medical, insurance, or sanctioning decisions.
- **Tracker and action MVP scope:** Current tracking and action components are **research scaffolds**, not claims of detection parity with large supervised baselines.
- **LLM risk:** Explanations can be **unstable** or **verbose**; deterministic fallbacks are preferred when reproducibility is paramount.

---

## 14. Future work

Plausible directions, consistent with a **Q1** trajectory, include: **stronger** tracking and **learned** action models under the same I/O contracts; **curated** event-level annotations in partner datasets; **calibration** and **uncertainty** reporting for fused scores; **adversarial** and **robustness** analysis on pose estimators; **user studies** with domain experts to validate **alert** wording; and **open** reporting of **failures** as **first-class** research outcomes.

---

## Document scope

This document is **descriptive** and **non-contractual**. It is meant for **academic and technical** audiences evaluating whether FightSafe AI is a **suitable** experimental substrate. For implementation details, see `docs/architecture.md`, `docs/framework.md`, and the package source under `src/fightsafe_ai/`.
