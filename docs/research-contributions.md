# Scientific Contribution (FightSafe AI)

Draft for Q1-style manuscripts. Phrasing is intended to be **defensible** and **bounded**—appropriate for venues in computer vision, sports informatics, or human–machine systems.

**Authors (context).** David Martin Moncunill, César Andrés Sánchez; Camilo José Cela University (UCJC), Madrid, Spain.

**Scope.** FightSafe AI is a **decision-support** and **research** toolkit, not a medical device. Contributions concern methodology, system design, and review workflows—not clinical diagnosis, injury prediction, or autonomous officiation.

---

## Contributions

We state **six** principal contributions, phrased to be **specific** and **technically grounded** while avoiding over-claim on accuracy, clinical validity, or match authority. We distinguish **problem framing**, **open implementation**, and **governable heuristics** as distinct from a single-benchmark *state-of-the-art* claim in detection or injury prediction.

1. **Safety-first problem framing for combat-sports review.** We position monocular video analysis toward **screening** and **time-localised** follow-up, rather than optimising only for scoring, highlights, or foul labels. **Multi-level** risk (ordered bands on a continuous score) refines a purely binary *alert / no-alert* view: it is intended for **triage and audit**, not to encode a league’s stop-fight policy. The novelty is **framing- and design-level** and should be supported by a related-work discussion that separates **assistance to officials** from **autonomous officiation**.

2. **End-to-end open-source pipeline and event-scale artifacts.** We provide a coherent stack from (optional) clipping through 2D pose, tabular **features**, **per-frame** risk, **temporal** **event** aggregation, overlays, and **reports**, with versioned **YAML** and **stable** column **schemas**. The contribution is **reproducibility** and **ablation** convenience for sports-vision research, **not** a top ranking on a single public benchmark as the paper’s main claim.

3. **A multi-level, interpretable risk framework with explicit pre-critical cues.** We connect pose-derived signals to a **rule-based** fusion that outputs a **bounded** score, **categorical** bands (e.g. **LOW**–**CRITICAL** in software), and **per-rule** **components** where enabled. The implementation foregrounds **interpretability**: named cues include, among others, **low-guard** (hands relative to the head) and **rolling** **instability** proxies—**pre-critical** in the sense of *raising attention before* a severe composite fires, not in the sense of a validated precursor of injury. **We do not** claim that these heuristics are **universal** or **sufficient** for all venues.

4. **Specialised, prototype heuristics beyond the core rule set.** (i) *Surrender* (tap-out) **gesture** detection is implemented as a **wrist-trajectory** heuristic over a short window to *suggest* a tap-out–style action for human confirmation; it is a **prototype**, not a certifying body’s official classifier, and is prone to false positives. (ii) A separate **heuristic** **anomaly** block (e.g. bilateral asymmetry in 2D knee flexion, frame-to-frame change, rapid ankle drop in image coordinates) can nudge the interpretable risk score; it is **not** medically validated and must **not** be read as injury diagnosis—only as a transparent engineered signal for research and caution-first review narration.

5. **Human-in-the-loop alert design for referees (recommendation-only).** A dedicated **HCI** layer maps internal risk bands to operator-facing alert gradients (e.g. *INFO* / *WATCH* / *WARNING* / *STOP* in the codebase) with traceable causes tied to rule keys and optional paraphrases. The system *recommends* attention and preparation to intervene; it does not issue stoppages, match scores, or medical verdicts. HITL as a principle is not novel; the contribution is a concrete separation of (a) deterministic analytics from (b) referee-oriented messaging in a safety-sensitive domain and toolkit.

6. **Interpretable rules with an optional, score-preserving LLM path.** A local (e.g. Ollama) LLM may narrate or template explanations from the *same* structured fields (features, triggered rules, event windows) with template fallback when the LLM is unavailable; numeric risk, level bands, and merging rules are not altered by the LLM. The integration contribution is (i) governability of the deterministic core and (ii) downstream narration that reuses exposed semantics rather than substituting opaque prose—without asserting that LLM text is exhaustive or factual beyond the supplied tables.

**For reviewers:** Defend novelty as the *conjunction* of: a multilevel governable risk framework; named pre-critical and safety-oriented cues; strictly bounded special heuristics (surrender, limb anomaly); schema-typed alert exports with confirmation-gate metadata; and optional local LLM narration grounded in the same tables—within one reproducible codebase and with explicit non-claims on medicine, match authority, operator benefit, and deployment readiness.

---

## Research Questions

The following RQs ground **empirical** follow-on studies; a methods-only or systems paper may **state** them without **fully** answering them.

- **RQ1 (validity of kinematic proxies).** To what extent do 2D pose-derived biomechanical features and temporal smoothing, as implemented in this stack, agree with expert-identified episodes of reduced stability or impaired control in combat-sports video, under a fixed annotation rubric and inter-coder or consensus analysis?

- **RQ2 (heuristic event alignment).** Under operating points tuned on held-out clips, can rule-based per-frame risk and event merging align with reference safety-relevant event labels (e.g. per-rater or committee-defined) at a preregistered F1 or temporal IoU level—and where do systematic error modes (e.g. occlusion-driven false positives) concentrate?

- **RQ3 (explainability and human outcomes).** Under comparable exposure time, does exposure to short structured explanations (rule-linked or LLM-narrated from the same feature and time structure) change reviewer outcomes (e.g. time-to-decision, agreement with consensus, usability) relative to overlays or tabular scores alone, in a preregistered user-study or expert-review protocol?

Ethics review and preregistration apply where human subjects are involved. Metrics and baselines should be defined per venue.

---

## Hypotheses

These are candidate **falsifiable** claims for future work, not results of a software description alone.

- **H1 (non-trivial alignment).** For in-domain tuning or calibration splits, with preregistered ablations (e.g. by occlusion or landmark visibility), per-frame risk and derived event intervals will exceed a preregistered chance or simple baseline (e.g. majority class or a trivial heuristic) in agreement with expert or consensus safety-relevant labels, **without** implying clinical or regulatory validity.

- **H2 (explanations without loss of decision calibration).** Given the same underlying tabulated and timeline information, adding constrained narration that reuses or maps to exposed fields will increase at least one preregistered measure of explanation sufficiency or self-reported understanding, while not worsening a preregistered binary or ordinal calibration metric on a concrete review task, under a defined *n* and design.

---

## Future work

- **Learned risk or event models** (supervised or semi-supervised) on the same engineered features or on pose embeddings with a path to introspection, with ablation vs. heuristics, cross-context fairness analysis, and deployment limited to **research** contexts until governance criteria are met.

- **Larger, diverse, consented datasets** across weight classes, venues, and camera placements, with provenance and licenses suitable for peer review and community release.

- **Real-time or near real-time** inference (edge/venue hardware), with documented throughput, latency, and failure under occlusion and compression; and explicit policy that live outputs are **non-definitive** review aids.

- **Validation with referees, safety officers, or expert coaches** under approved protocols (agreement, time-to-review, usability), without conflating these measures with medical or regulatory validation of the system.

- **Multiview video** or **complementary sensors** (e.g. IMU, time-synchronised side cameras) to reduce monocular depth and ambiguity, where practical and ethical.

---

## Suggested one-paragraph contribution block (for abstract or introduction)

FightSafe AI contributes: (1) a safety- and review-oriented problem framing, including multilevel risk for *assisting* (not *replacing*) officials; (2) a reproducible end-to-end pipeline to interpretable per-frame risk, event artifacts, and overlays; (3) named pre-critical cues (e.g. low-guard and instability proxies) within a governable rule system; (4) prototype surrender detection and heuristic limb-level anomaly signals with explicit non-clinical scope; (5) a recommendation-only referee alert layer; and (6) integration of interpretable rules with an optional local LLM path that does not modify numeric risk—together with a preregisterable research agenda, without conflating the system with autonomous officiation or medical use.

---

*This document is a drafting aid. The final submission must support comparative claims with peer-reviewed citations; adjust tone and length to the target Q1 journal (e.g. sports informatics, multimedia, or HCI with applications).*
