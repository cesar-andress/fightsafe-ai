# FightSafe AI: Methodology (IMRaD Draft)

**Authors:** David Martin Moncunill, César Andrés Sánchez
**Affiliation:** Camilo José Cela University (UCJC), Madrid, Spain

*This draft follows the Introduction–Methods–Results–Discussion (IMRaD) structure. FightSafe AI is described as a **decision-support** and **computer-vision** toolkit for **highlighting candidate safety-relevant intervals** in combat-sports video. It is **not** a medical device, does **not** diagnose injury or concussion, and is **not** a substitute for qualified medical staff, officials, or governing-body procedures.*

---

## 1. Introduction

### 1.1 Motivation: safety in combat sports

Combat sports pair high movement speeds with impact and fatigue, creating environments in which adverse outcomes—falls, head impacts, loss of balance, and other moments of concern—can occur in seconds and may be easy to miss in real time. Stakeholders including coaches, event staff, and researchers need **repeatable, reviewable** ways to **screen** long recordings and **direct attention** to segments that may warrant human follow-up, without replacing professional judgment.

### 1.2 Limitations of current refereeing and broadcast-centric workflows

Human officials must prioritize **rule compliance and match flow**; they cannot attend equally to all biomechanical cues on every frame. Commercial scoring or highlight systems are often **optimized for entertainment or performance metrics**, not for **safety-oriented review**. Recordings from a **single camera** are further limited by **occlusion**, **motion blur**, and **viewpoint bias**, so any automated signal should be treated as a **noisy, approximate cue** subject to expert interpretation.

### 1.3 Gap: AI for safety versus AI for scoring

Much of the sports–AI literature emphasizes **scoring, tactics, or highlight generation**. Relatively less attention is given to **transparent, auditable** pipelines that output **interpretable** frame-level and event-level **risk proxies** together with **artifacts** (tables, plots, overlay video) that support **traceable** review. There is a need for systems that **separate** (i) **core, offline-capable** processing (pose, features, rules) from (ii) **optional** language-based summaries, so that the **analytic core** remains verifiable and does not depend on generative models.

### 1.4 Objective of the paper

The objective of this work is to present **FightSafe AI**, a modular **decision-support** system that:

1. Ingests **combat-sports video** and estimates **2D body pose** over time.
2. Derives **biomechanically motivated features** and **rule-based, interpretable** per-frame **risk** scores.
3. **Aggregates** frame-level evidence into **temporal events** for review and reporting.
4. Optionally enriches outputs with **local** large-language-model (**LLM**) **explanations** (e.g., via Ollama) while **preserving** a non-LLM path suitable for **reproducible** analysis.

We report the **methodology** and **qualitative** output types; we do **not** claim clinical efficacy or replace institutional safety protocols.

---

## 2. Methods

*Figure references below point to static illustrations under `docs/figures/` (PNG/SVG; schematic or synthetic, not a substitute for data from a specific run).*

### 2.1 System overview

**Figure 1** shows the FightSafe AI system architecture: the **deterministic** processing chain (video through pose, features, temporal analysis, risk scoring, event detection, and visualization), an **optional** large-language-model (**LLM**) **explanation** path (e.g. local Ollama), and the **human-in-the-loop** decision point. **Figure 2** summarizes the **end-to-end logical flow** of artifacts—raw video, frames, keypoints, features, per-frame risk, event aggregation, and report/overlay outputs.

FightSafe AI implements an end-to-end **processing chain** with stable **frame indexing** and time alignment when a single sampling rate is used throughout:

**Raw (or clipped) video → frame extraction → pose keypoints → tabular features → per-frame risk score and multi-level banding (optionally: a binary “review” cut) → event intervals → reports, operator-facing HCI alerts (recommendation-only), and visualization.**

The **core path** (through risk and events) is **deterministic** given fixed inputs and configuration; an **optional LLM** stage only **narrates** or **paraphrases** existing structured outputs and **must not** change numeric risk. A **human-in-the-loop** step is assumed for any operational use: reviewers interpret overlays, event lists, and text in light of **context** and **organizational** rules.

### 2.2 Data acquisition

**Sources.** In our research setting, material may include **publicly available** recordings (e.g., **YouTube**), subject to platform terms and institutional policies. Download and trimming are **auxiliary** steps; the **methodological** focus is on **repeatable** clip definitions once a source file exists.

**Manual clip selection.** Long broadcasts are **reduced** to **short clips** that contain exchanges or balance-critical moments, to bound compute cost and align analysis with **analyst interest**. Clips are chosen **manually** or with simple heuristics; this introduces **selection bias** (discussed below).

**Local dataset creation.** Processed data reside in a **project-structured** layout: extracted frames, consolidated or per-frame keypoint tables, feature and risk **CSVs**, **JSON** event lists, and optional **Markdown/HTML** reports. The design favors **reproducibility** (versioned **YAML** configuration, stable column names) over ad hoc scripts.

### 2.3 Pose estimation

**Backend.** We use **MediaPipe**-style 2D pose estimation (e.g., BlazePose conventions) to produce **normalized** image-plane landmarks with **per-keypoint** visibility (or confidence) when available. Landmarks are exported to **tabular** form keyed by **frame order**, consistent with natural sorting of frame filenames.

**Keypoint extraction.** For each frame, a vector of **named joints** (e.g., shoulders, hips, knees) is stored. The pipeline enforces **alignment** between the video frame index and the keypoint **row index**; dropped or unreadable frames are handled explicitly in downstream code to avoid silent **mis-joins** with risk tables.

**Limitation.** Monocular 2D pose is an **approximation** of 3D kinematics; we do not infer medical diagnoses from landmarks.

### 2.4 Feature engineering

From sequences of 2D landmarks, we compute **per-frame** biomechanical **proxies**, including (non-exclusively):

- **Torso angle (or tilt).** Deviation of the upper-body segment relative to a **vertical** reference, as a simple indicator of **postural configuration** and loading patterns.
- **Hip velocity (or center-of-mass proxy).** Finite differences of hip-related coordinates over time, scaled by the **sampling period**, to capture **translational** dynamics.
- **Instability (erratic motion).** Rolling **variance**-style statistics over selected joints or combined signals to capture **irregular** movement **relative** to a local baseline.
- **Posture or state duration.** **Sustained** intervals in which the athlete remains in configurations associated with **elevated** concern (e.g., **prolonged** low posture or **near-ground** proximity according to **rule-defined** height proxies).

Features are **documented in configuration**; missing detections propagate as **missing** values, and the risk module defines how gaps are **treated** (e.g., forward-fill, omission, or flagging). **Figure 5** illustrates, in schematic form, a 2D skeleton with example geometry-based cues (e.g. torso orientation relative to vertical, hip center, head position, and instantaneous displacement)—**illustrative** only; exact feature names and columns are defined in configuration and code.

### 2.5 Risk scoring

**Paradigm.** Risk is implemented as a **rule-based** fusion of feature columns: **thresholds**, **rolling windows**, and **weights** declared in **versioned** configuration (e.g., separate YAML for risk). The system outputs, per frame, a **continuous score** in a bounded range (e.g., **\[0, 1\]**) and a **multi-level** categorical **band** derived from that score (see Section 2.6). A **binary** or single-threshold “flag” view remains available for **compatibility** with early prototypes and for **simplified** event logic, but **graded** reporting is treated as the **primary** design for **safety-oriented** triage: it preserves **gradual** escalation before any binary cut.

**Interpretability.** Each component maps to **inspectable** rules: analysts can **trace** why a frame entered a given band by reading **feature** columns, **per-rule** **component** columns (where exposed), and **rule** **parameters** in configuration. This supports **governance** and **calibration** on new footage **without** requiring a retrained black-box model for the baseline.

**Non-claims.** The score is a **heuristic risk proxy** for **workflow prioritization**, not a probability of injury, illness, or foul. For **visual** communication in papers and reports, a **per-frame** risk trace can be plotted over time with **banded** regions; **Figure 3** shows a **synthetic** example of such a timeline with vertical markers for **detected** event times—**not** a claim about any specific contest or athlete.

### 2.6 Multi-level risk modeling

**Design rationale (from binary to graded outputs).** Early design sketches used a high–low or “flagged / not flagged” summary to ease integration with event lists and on-screen encodings. For safety review, a binarized view erodes distinctions that matter operationally: routine monitoring, emerging concern, acute hazard, and *urgent human review* as separate stages. The implemented system therefore maps a continuous per-frame score on **[0, 1]** to ordered **risk levels** (typically **LOW**, **MEDIUM**, **HIGH**, **CRITICAL**) via configurable cutoffs, augmented by *composite* rules (multi-cue fusions) that can raise severity when conjunctions of signals align. Weights and thresholds are versioned (e.g., in YAML) so that level transitions are auditable and can be critiqued in post hoc analysis.

**Aggregation behavior.** The global score is a *weighted* blend of *active* rule components; if a feature column is absent, that rule is *deactivated* and its weight is renormalized over remaining terms rather than imputing a high value—an explicit **safety-first** device against spurious inflation under partial sensing. Tuning the boundaries between bands remains an empirical and context-dependent exercise; repository defaults are for software validation, not universal sanctioning policy.

**Link to later subsections.** The graded banding supports human-readable **alerts** (Section 2.7) and is compatible with **optional** generative **narration** (Section 2.10) that must echo the same structured evidence, not override it.

### 2.7 Human-in-the-loop interaction

**Support, not replacement.** The pipeline is a **decision-support** artifact: it is intended to sharpen attention during review, not to replace a referee, ringside physician, or governing body. A dedicated **human–computer interaction (HCI) layer** therefore *re-frames* per-frame risk for operators by issuing **recommendation-style alerts** (e.g., mapped vocabulary such as *INFO* / *WATCH* / *WARNING* / *STOP* in the codebase) with explicit *reasons* (triggered rules and human-paraphrased labels). The software **does not** command match stoppage, medical action, or scoring; enforcement authority remains with humans under applicable rules.

**Alert levels and careful wording.** *INFO* (normal tracking) aligns with the lowest risk band; *WATCH* signals that sustained attention is advisable; *WARNING* that intervention should be *prepared*; *STOP* denotes maximum-priority *review* suggestion—not an automated stoppage. Wording is chosen to privilege clarity and a **safety-first** stance over confidence in a single model channel.

**Interaction design and interpretability.** Alerts bundle **(i)** a level, **(ii)** a timestamp, **(iii)** a short message, and **(iv)** **traceable** causes tied to rule keys (optionally with plain-language paraphrases). This keeps **interface copy** *grounded* in inspectable rules (Sections 2.4–2.5) and reduces “uncanny certainty.” Operators should be able to dismiss, annotate, or override signals *without* the system treating such actions as *training feedback* in the default offline pipeline. Optional LLM text (Section 2.10) is strictly *downstream* of the same **structured** trail.

### 2.8 Special event detection

**Scope.** The generic temporal merger in Section 2.9 groups intervals from frame-wise risk. *Additional* passes address behaviorally salient, **interpretable** modes that are useful for **safety** narratives and for **reproducible** ablation in research, without replacing the main rule stack.

**Surrender (tap-out) detection.** A hand–trajectory **heuristic** (short-window wrist motion) can, when a stereotyped tap-out gesture is suspected, elevate the displayed severity on affected frames subject to confidence gating and override logic in the scorer. The module is a **culturally common** gesture **proxy** for *human* confirmation; it is *not* a certifying body’s official classifier and may false-positive on confusable actions (e.g., glove adjustment). The methodology therefore stresses **disclosure** in interface text and in methods, to **limit** over-reliance on a single channel.

**Anomaly detection (non-clinical limb proxies).** A parallel **engineered** feature block scores **bilateral asymmetry** in 2D knee flexion, abrupt frame-to-frame joint change, and ankle *drop* in image coordinates, yielding a bounded *anomaly* score and a discrete *type* label (e.g., collapse- or asymmetry-dominant heuristics in software). The output may feed the interpretable risk *weights* and, where configured, a *post hoc* tier nudge toward **HIGH** or **CRITICAL** to align operator attention with unexpected 2D kinematic patterns—still a **proxy**, not an injury *diagnosis*. The absence of **clinical** validation is explicit in the implementation: the block supports *safety-first* triage narration and transparent on/off ablations, not *medical* use.

**Cross-cutting note.** All such passes are documented in code and configuration, avoid end-to-end opacity, and *complement* the core rule semantics of Sections 2.5 and 2.6 rather than substituting for them.

### 2.9 Event detection

**Goal.** **Frame-level** flags are too granular for long matches; we **group** time into **events**—**intervals** on the time axis that summarize sustained or salient **HIGH**- or **CRITICAL**-class behavior under project settings.

**Temporal grouping.** Consecutive or nearly consecutive **flagged** frames are **merged** into segments; **gaps** shorter than a **minimum** merge or **hysteresis** window may be **bridged** to avoid **fragmentation**, depending on configuration (e.g., `event` aggregation parameters).

**Outputs.** Event lists include **start/end** times (or frame indices) and **labels** suitable for **bookmarks**, **explanations**, and **qualitative** comparison with **human** annotations when available. Events are **candidates for review**, not **verified** incidents. **Figure 4** demonstrates, at a high level, how **per-frame** risk can be **grouped** into **event** **intervals** (sampled frames, a risk trace, and merged segments—**illustrative** **synthetic** layout).

### 2.10 Optional LLM layer (Ollama) and human-in-the-loop

**Ollama.** A **local** server (Ollama) can generate **short natural-language** text conditioned on **structured** event and risk **summaries** already present in the pipeline. **No** online API is **required** for the LLM if models are served locally; the **analytic** core can still run **without** Ollama.

**Explanation generation.** The LLM is used for **narration** and **templating**-style help (e.g., bullet lists of “why this interval was flagged” from **known** feature **names** and **thresholds**). If the LLM is **unavailable** or **fails**, the system can fall back to **template**-based or **rule-derived** text so that **reports** remain **complete**. Generated prose must not alter **numeric** risk: it rephrases the same **evidence** trail already exposed to operators through rules and, where present, the HCI *alert* layer (Section 2.7).

**Positioning with respect to Sections 2.1 and 2.7.** The intended end-to-end workflow is: **(1)** **deterministic** signals and **artifacts** → **(2)** *optional* **natural-language** summary via Ollama → **(3)** *human* decision (e.g., ignore, **escalate** for follow-up, rewatch, or use only in **research**), consistent with the *referee-first* and *safety-first* framing in Section 2.7. The system **does not** close the loop to **autonomous** sanctions, **medical** triage, or **match** stoppage.

---

## 3. Results (outputs and qualitative examples)

This section describes **expected** system outputs; quantitative benchmarks on a public benchmark are **out of scope** for this methodology paper. The **figures** referenced here are the same **schematic** assets as in Section 2; empirical studies should add **run-specific** plots and frames from **authorized** **clips**.

### 3.1 Risk timeline

For a processed clip, analysts obtain a **time series** of **per-frame** risk **scores** and optional **multi-level** categorical bands (e.g., **LOW** / **MEDIUM** / **HIGH** / **CRITICAL**), typically plotted as a **line chart** with **shaded** regions. This visualization supports **spot-checks** of **stability** (e.g., spurious spikes) and **alignment** with **video**. **As illustrated in Figure 3**, risk scores can **increase sharply** in short time windows, with **horizontal** bands (including **MEDIUM-** and **high-** tier regions as configured) and **vertical** markers at **putative** event times; the illustration is **synthetic** and is **not** a measured trace from a specific competition or subject.

### 3.2 Detected events

**JSON** (or equivalent) event files list **temporal** intervals and metadata for **downstream** reporting. Events **compress** the timeline into **actionable** segments for **coaches** and **analysts**, and can be compared—qualitatively—to **manual** labels where available. **Figure 4** demonstrates event grouping from frame-level inputs to interval-level outputs; formal precision and recall require a labeled corpus (see `docs/experiments-mvp.md`).

### 3.3 Overlay video

**Rendered** **MP4** (or similar) with **skeleton** overlay and **on-screen** risk or flag signals allows **in-context** inspection. Skeleton style and joint layout are conceptually aligned with the schematic in **Figure 5**; on-screen encoding of risk (colors, borders) follows project visualization configuration. Overlays are **illustrative**; compression artifacts and 2D projection limits remain visible in **side-by-side** review with raw footage.

### 3.4 Qualitative examples

Representative **clips** (not reported here as a quantitative dataset) show **(i)** stable baselines in upright stance, **(ii)** **brief** high-score spikes at fast motion, and **(iii)** **sustained** event intervals where **rule** **fusion** and **merging** produce a **coherent** segment. These examples **illustrate** **behavior** under default or study-specific **YAML**; they **do not** establish **ground-truth** safety. Qualitative discussion can cross-reference **Figure 3** (timeline shape), **Figure 4** (interval semantics), and **Figure 1** (where reviewers sit in the loop).

---

## 4. Discussion

### 4.1 Strengths

- **Graded risk and HCI separation.** Multi-level banding preserves *nuance* for triage, while the **HCI** vocabulary can be mapped *independently* from internal risk labels so that operator copy remains *cautious* and *non-prescriptive* (see Section 2.7).
- **Interpretability.** Rule-based **risk** and **tabular** features make **disagreements** between the system and a human **debuggable** by **inspection**.
- **Modularity.** **Pose**, **features**, and **rules** are **separable**; future **learned** components can replace **aggregation** while retaining **I/O** contracts.
- **Real-time potential.** The design **targets** efficient frame-wise processing; **end-to-end** **latency** depends on hardware, resolution, and FPS—**prospective** engineering for **live** use is possible but **not** evaluated here.
- **Optional LLM.** Explanations can be **switched off**, supporting **reproducible** studies where **only** **deterministic** code paths matter.

### 4.2 Limitations

- **Small or opportunistic datasets.** Clips from **YouTube** and ad hoc selection introduce **bias** and limit **generality** across **venues**, **angles**, and **athlete** populations.
- **Rule-based** **risk** may **underfit** **rich** spatio-temporal patterns that **supervised** models could capture, at the cost of **transparency** and **data** **requirements**.
- **No clinical or regulatory validation** has been performed; **outputs** must not be used for **diagnosis** or **medical** **decisions**.
- **Monocular** 2D pose and **heuristic** features **do not** capture **true** 3D forces or **contact** **physics**; **occlusion** can **degrade** landmarks.
- **“Safety”** is **operationalized** by **configuration**; **leagues** and **cultures** differ, so **default** **thresholds** are **illustrative**, not **universal**.

---

## 5. Conclusion

We presented a **modular, interpretable** **decision-support** **pipeline**—FightSafe AI—for **combat-sports** **video**: **2D** **pose**, **biomechanical** **features**, **rule-based** **per-frame** **risk** with **multi-level** banding, **special** **heuristics** for **surrender**-like gestures and **non**-**clinical** **limb** **anomalies**, an **operator**-facing **HCI** **layer** that **recommends** **attention** **without** **replacing** **referees**, **temporal** **event** **grouping**, **optional** **local** **LLM** **narration**, and **human** **review**. The approach is **feasible** for **research** and **analyst** **workflows** that prioritize **auditability** and **caution** over **fully** **autonomous** **adjudication**.

**Potential** **impact** lies in **faster** **browsing** of long recordings, **common** **language** for **coaches** and **engineers** around **“risk”** **proxies**, and a **path** to **pair** **transparent** heuristics with **future** **data-driven** **models** under **governance**.

**Future work** includes: **(1)** **curated**, **diverse** **label** sets and **agreement** studies with **experts**; **(2)** **calibrated** **learned** **risk** heads that retain **explanations**; **(3)** **multiview** or **IMU** **fusion** where **feasible**; **(4)** **prospective** **evaluation** of **workflow** **utility** (e.g., **time-to-review**) **without** conflating **usability** with **medical** **outcomes**; and **(5)** **ethics** and **fairness** **review** across **demographics** and **competition** **settings**.

---

## List of figures (repository assets)

| Figure | File(s) in `docs/figures/` | Content |
|--------|----------------------------|---------|
| **Figure 1** | `architecture.png` / `.svg` | System architecture: core pipeline, optional LLM layer, human-in-the-loop. |
| **Figure 2** | `pipeline_flow.png` / `.svg` | End-to-end data flow: raw video → frames → keypoints → features → risk → events → report. |
| **Figure 3** | `risk_timeline.png` / `.svg` | Synthetic risk score over time with multi-level risk bands and event markers. |
| **Figure 4** | `event_detection.png` / `.svg` | Schematic: sampled frames, per-frame risk, merged event intervals. |
| **Figure 5** | `pose_features.png` / `.svg` | Schematic skeleton with example geometry / kinematic cues. |
| **Figure 6** | `risk_levels.png` / `.svg` | Ordered risk bands (LOW–CRITICAL) and example interpretable signals (schematic). |

*Regenerate or replace with run-specific figures as required; `python docs/figures/generate_paper_figures.py` rebuilds the current schematics.*

---

## References (placeholder)

*To be completed: citations for MediaPipe, Ollama, and relevant sports-safety and interpretable-ML literature.*

---

## Suggested running title

*FightSafe AI: An Interpretable, Human-in-the-Loop Pipeline for Safety-Oriented Risk Cues in Combat-Sports Video*

---

## Compliance note

> FightSafe AI is a **decision-support** and **research** **toolkit**. It does **not** provide **medical** **advice**, is **not** a **medical** **device**, and does **not** **replace** **qualified** **officials**, **medical** **personnel**, or **organizational** **safety** **procedures**.
