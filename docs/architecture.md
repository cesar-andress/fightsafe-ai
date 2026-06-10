# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Architecture

**Documentation note:** FightSafe AI is a **research software artifact** for traceability and auditability, not a certified medical or officiating system. It targets **combat sports** video with **event-level safety alert exports** and **protocol-defined evaluation**; it does **not** certify medical safety, replace referees, or claim validated human–AI collaboration without an explicit logged study.

FightSafe AI is structured as a **modular research toolkit**: each stage has a narrow responsibility, explicit inputs/outputs, and stable join keys (typically **frame index** and **time** derived from video frame rate). Configuration is declared in YAML under `configs/`; researchers can run batch workflows through the `fightsafe` CLI or compose stages programmatically via `fightsafe_ai`.

This document explains the **logical architecture** (including stages that may be implemented outside the core package today, e.g., clip cutting via external tools) and the **design rationale** for **human-in-the-loop** operation rather than autonomous officiating.

**Terminology (consistent with the README and paper):** **event-level safety alerts** (candidate intervals merged on the `EventBus`), **export traceability** (schema-typed tuples and manifests), **confirmation-gate semantics** (metadata flags and specification-level routing), and **protocol-defined bookkeeping** (evaluator CSV under frozen matching defaults—not operator-outcome evaluation).

---

## Real-time monitoring path

When operators run **`python -m fightsafe_ai.live.live_runner`** (OpenCV **live mode**), the stack follows the same **semantic pipeline** as **batch mode**—**pose → biomechanical / temporal features → real-time risk estimation → anomaly soft signals → event-level safety alerts**—but executes **inside `LivePipeline`** on each decoded frame instead of writing intermediate CSVs for every stage. The optional **web dashboard mode** (`fightsafe_ai.api.app`) exposes the same fused semantics with browser controls, **event export**, **feedback** logging, and optional **GPU monitoring**.

- **`VideoSource` / `open_video_source`** wrap OpenCV capture from a **file path** (pseudo-stream) or **camera index**. Optional **`--realtime`** paces playback to wall-clock time.
- **`LivePipeline`** calls `create_runtime_pose_estimator`, maintains short rolling buffers for smoothing and biomechanics, and feeds the same combat MVP risk helpers used offline.
- **`EventBus`** receives **`SafetyEvent`** instances: it applies **cooldown**, **merge gap**, and optional **visual expiry** so the UI and exports do not flicker on single-frame spikes.
- **Latency:** decoding and pose inference share the critical path; the reference implementation may use a **worker queue** so OpenCV preview stays responsive. End-to-end latency depends on hardware, backend (CPU vs GPU), frame size, and **`max_infer_hz`** throttling—treat numbers as **engineering measurements**, not latency guarantees for officiating.

Artifacts default to **`outputs/live/events.json`** and **`outputs/live/events.csv`** for streamed runs; the batch pipeline still writes **`events.json`** and **`risk_scores.csv`** under the chosen run directory.

---

## Modular components

The system is organized into the following modules. Package paths refer to the current Python layout under `src/fightsafe_ai/` unless noted.

### 1. Video ingestion

**Role.** Acquire raw video from files or supported URLs and normalize access for downstream processing.

**Implementation notes.** The core package provides helpers such as download via the external **`yt-dlp`** CLI (`fightsafe_ai.video.downloader`) and reading streams with **OpenCV** (`cv2.VideoCapture`). Ingestion is deliberately thin: it avoids embedding site-specific policies beyond what operators configure for their environment.

**Outputs.** Local media files on disk or in-memory frames when reading directly from file paths.

---

### 2. Clip cutting

**Role.** Reduce **long continuous recordings** to **analysis windows** (temporal clips) that bound computational cost and focus evaluation on relevant segments (e.g., rounds, exchanges, or manually marked intervals).

**Implementation notes.** The package includes **`fightsafe_ai.video.cutter`** (`cut_clip`), built on **FFmpeg** via **`ffmpeg-python`**; it requires the **`ffmpeg`** binary on `PATH`. Teams may still use external editors or scripts when preferred. Architecturally, this stage defines what “raw video” means for the pipeline—often a shorter file that downstream stages treat as the canonical source.

**Outputs.** One or more clip files (or in-memory frame ranges) passed to frame extraction.

---

### 3. Frame extraction

**Role.** Sample the clip onto a **discrete time lattice** at a chosen frame rate (or stride), producing still images suitable for pose estimation.

**Implementation notes.** Implemented in `fightsafe_ai.video.frame_extractor` using OpenCV; frames are named ``frame_000001.jpg`` onward in chronological order. Parameters such as **target FPS** must stay **consistent** across later stages when converting frame indices to seconds for biomechanics and reporting.

**Outputs.** A time-ordered sequence of raster images (e.g., `frame_0001.jpg`, …) with stable **natural sorting** by filename.

---

### 4. Pose estimation

**Role.** Estimate **2D body landmarks** per frame from RGB appearance. The default backend follows **MediaPipe BlazePose** conventions: normalized coordinates in \([0,1]\) with **visibility** scores.

**Implementation notes.** `fightsafe_ai.pose.backends.mediapipe_backend.MediaPipePoseEstimator` processes image folders and writes one **consolidated CSV** (`frame_id`, `keypoint_name`, `x`, `y`, `z`, `visibility`). Alternative backends subclass `fightsafe_ai.pose.base.BasePoseEstimator` and emit compatible tables for the `fightsafe_ai.keypoints.io` loaders.

**Outputs.** Per-frame keypoint tables (`keypoints/` CSVs). Shared serialization helpers live under `fightsafe_ai.keypoints`.

---

### 5. Feature engineering

**Role.** Transform landmark sequences into **biomechanical proxies** and stability cues aligned **one row per frame**: e.g., torso orientation relative to vertical, hip vertical velocity (from differentiated normalized coordinates scaled by time), rolling variance of selected joints (stability proxy), and ground-proximity indicators derived from foot/ankle geometry.

**Implementation notes.** `fightsafe_ai.features.biomechanics.compute_pose_features` consumes the keypoint directory and emits a **pandas** table (or CSV on disk). Missing detections propagate as missing values; downstream stages must handle sparse sequences.

**Outputs.** A **feature matrix** indexed by frame order, suitable for tabular analysis and risk scoring.

---

### 6. Risk scoring

**Role.** Map frame-aligned features to a bounded **risk signal** using **transparent rules** (thresholds, weights, rolling statistics) loaded from `configs/risk_rules.yaml`.

**Implementation notes.** `fightsafe_ai.risk.engine.detect_risk_events` and `fightsafe_ai.risk.models.RiskRuleParams` implement a **heuristic fusion** (e.g., weighted combination of tilt/velocity cues, sustained near-ground duration, erratic motion proxies). The result includes **`risk_score ∈ [0,1]`** and a boolean **`risk_flag`** per frame under an operating point.

**Outputs.** Augmented table with risk columns; intended for review and calibration—not standalone adjudication.

---

### 7. Event detection

**Role.** Elevate **frame-level signals** to **temporal events** analysts care about: contiguous intervals above a threshold, peaks, or minimum-duration segments suitable for logging, clip bookmarks, or evaluation against labels.

**Implementation notes.** In the MVP, “events” can be derived **directly** from **`risk_flag`** sequences (run-length encoding, merging neighboring positives) or from **`risk_score`** peaks with hysteresis—implemented in notebooks or small utilities rather than a mandatory core service. The architectural boundary remains clear: **frames → scores → candidate events** for human confirmation.

**Outputs.** Event lists (intervals, peak times, or frame spans) for downstream reporting—not automatic penalties.

---

### TapKO candidate detectors (parallel track)

**Role.** Emit **TapKO-aligned** interval candidates—**not** commission rulings or clinical labels—for two **event families** used in the TapKO research track:

| Family | Meaning (developer shorthand) |
|--------|--------------------------------|
| `submission_signal` | Tap-adjacent pose proxies (e.g. hand/foot tap candidates from motion rhythm; verbal/technical types reserved for schema and future multimodal cues). |
| `extreme_vulnerability` | Fight-ending–style **candidates** (e.g. collapse, guard-loss, post-impact inactivity, choke-context proxy)—always **requires human confirmation** in product language. |

**Implementation notes.**

- **Detectors:** `fightsafe_ai.events.tap_detector` and `fightsafe_ai.events.vulnerability_detector` consume a **single-person COCO-17** stack `(T, 17, 2)` in normalized image coordinates (same pose contract as the MVP CSV export).
- **Live dashboard:** `fightsafe_ai.live.tapko_live_events` maps detector outputs to `SafetyEvent` rows (`event_type` like `submission_signal.hand_tap`, `extreme_vulnerability.ko_collapse`) with `metadata.tapko_family` / `tapko_subtype` for the UI.
- **Offline CLI (no database):** `fightsafe tapko-detect` runs pose (or accepts `--pose-csv`) and writes `tapko_predictions.json` / `.csv` / `tapko_report.md` under a run directory. See [`evaluation.md`](evaluation.md) for matching predictions to annotations.

**Schema and evaluation docs:** [`tapko_annotation.md`](tapko_annotation.md) (annotation JSON), [`evaluation.md`](evaluation.md) (TapKO evaluator), [`datasets.md`](datasets.md) (collection policy). **Limitations** (monocular pose, false positives, audio not used unless explicitly modeled) are summarized in [`evaluation.md`](evaluation.md) and the README—not duplicated here.

---

### 8. Visualization

**Role.** Communicate model behavior to humans: **skeleton overlay**, on-screen **risk score**, and **visual emphasis** when flags activate (e.g., border/tint), optionally driven by `configs/default.yaml` styling.

**Implementation notes.** `fightsafe_ai.visualization.overlay.render_risk_overlay_video` aligns **video frame index** with sorted keypoint CSVs and risk rows. Misalignment between clip length and exported keypoints should surface as warnings in logs.

**Outputs.** Annotated video files for qualitative audit and presentations.

---

### 9. Future ML models

**Role.** Replace or augment heuristic fusion with **learned estimators** trained on curated labels while preserving the same **tabular interfaces** (features → score/event).

**Implementation notes.** Planned extension points include:

- **Supervised models** consuming fixed-length windows of engineered features or learned pose embeddings.
- **Calibration layers** mapping raw logits to probability estimates with explicit evaluation protocols.
- **Fairness and robustness** analysis across camera angles, weight classes, and environments—reported alongside accuracy.

The intent is to swap **only** the risk aggregation module (or add parallel heads) without rewriting ingestion or pose I/O, subject to schema compatibility.

---

## Data flow (end-to-end)

The canonical processing chain is:

```text
Raw video → clips → frames → pose keypoints → biomechanical features → risk score → visual output
```

More explicitly:

| Stage | Artifact |
|--------|-----------|
| Raw video | Source file or stream |
| Clips | Trimmed temporal segments (optional but recommended at scale) |
| Frames | Raster images sampled at target FPS |
| Pose keypoints | Consolidated or per-frame landmark CSV |
| Biomechanical features | Frame-indexed feature table |
| Risk score | Per-frame `risk_score` / `risk_flag` |
| Visual output | Overlay video (and optional event logs) |

**Join discipline.** Frame \(i\) in the video reader corresponds to the \(i\)-th naturally sorted keypoint CSV and the \(i\)-th row of the feature/risk tables **when pipelines share the same extraction FPS and ordering**. Any resampling or dropped frames must be logged to avoid silent misalignment.

---

## Human-in-the-loop by design

FightSafe AI is **not** architected as an **autonomous referee** or medical device. Reasons include:

1. **Epistemic limits of monocular video.** 2D lifting, occlusion, motion blur, and viewpoint bias make physically interpretable quantities **approximate**. Automated scores are **cues**, not ground truth for causality of injury or foul play.

2. **Normative ambiguity.** “Safety” and “risk” are **operationalized** differently across leagues, weight classes, rule sets, and coaching philosophies. A deployable system requires **domain governance** and **threshold calibration** on representative data—typically done by experts, not fixed defaults in code.

3. **Accountability.** Competitive sports and athlete welfare decisions demand **traceable** rationale. Heuristic YAML rules and per-frame tables support **audit**; black-box automation without review increases liability and erodes trust.

4. **Dataset bias.** Models—even heuristic ones—inherit biases from cameras, demographics, and labeling practices. Human reviewers mitigate **false positives** that could unfairly flag athletes or teams.

Accordingly, visualization and tabular outputs are optimized for **review workflows**: highlight **candidate** intervals, display scores, and leave **final judgment** to qualified humans using institutional procedures.

---

## Configuration and extension

- **`configs/default.yaml`**: sampling, pose parameters, feature lists, visualization styling.
- **`configs/risk_rules.yaml`**: heuristic thresholds and aggregation weights.

Extension hooks are described in `README.md` (protocols, schema stability). Changes that alter CSV schemas should be versioned and documented for reproducibility.
