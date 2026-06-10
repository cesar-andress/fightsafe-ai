# Troubleshooting

## MediaPipe / TensorFlow Lite console output (CPU)

While running the default **MediaPipe** pose path, you may see messages from
**TensorFlow Lite** and **MediaPipe** on the terminal. On a typical **CPU** run
these are **often normal** and **do not** mean the FightSafe pipeline failed.

- **XNNPACK delegate** — TFLite may report that inference uses the
  [XNNPACK](https://github.com/google/XNNPACK) path. This is **informational**
  in most local setups, not a FightSafe error.
- **Feedback manager** (or similar MediaPipe runtime line) — Usually
  **harmless**; it does not by itself invalidate `pose_keypoints.csv` or other
  outputs.
- **NORM_RECT** — May appear in relation to **ROI** / input rectangle handling
  inside MediaPipe’s graph. It does **not** by itself mean landmarks are
  unusable; assess outputs (CSV, overlays, `qa_report.json`) instead of relying
  on that line alone.

## How to know if a run really succeeded

**Do not** treat the messages above as pass/fail signals. Pipeline success
should be judged from:

1. **Generated artefacts** at the run root: e.g. `pose_keypoints.csv`,
   `features.csv`, `risk_scores.csv`, `events.json`, `qa_report.json`, overlay
   `output_overlay.mp4` when that stage is enabled, etc.
2. **QA report** — `fightsafe qa` (or the `qa_report.json` written in the
   pipeline) and its `passed` / `failed_checks` / `warnings` tell you whether
   the automated checks for that run are acceptable, independent of
   third-party **stderr** noise from native libraries.

If those outputs are **missing** or `qa_report.json` reports a **failed** state
the operator cares about, **then** there is a problem worth investigating—not
merely because XNNPACK or MediaPipe printed a line to the console.

## Ollama / optional LLM failures (HTTP 500, model load, memory)

When **per-event explanations** use a local **Ollama** model (`--explain-events` with
Ollama enabled in `configs/llm.yaml`), the server may return **HTTP 500** or similar
errors while **loading a large model** or when **VRAM/RAM** is exhausted.

**FightSafe behavior:**

- The pipeline **does not fail** because of LLM errors. **Deterministic risk** scores,
  `risk_scores.csv`, `events.json`, and overlays are produced **without** the LLM.
- On the **first** failure that looks like a **model load or resource** problem, the
  run **disables further Ollama calls** for that run, emits **one** concise warning,
  and writes **template / rule-based** explanations for **all remaining** events (same
  fallback as when Ollama is off).
- The run root may include `llm_explanation_state.json`. **`summary.json`** and
  **`qa_report.json` metrics** can include:
  `llm_requested`, `llm_available`, `llm_fallback`, `llm_error` (e.g. `"model failed to load"`).

LLM or Ollama outages **do not change** numeric risk fusion or event extraction; they
only affect optional narrative text for human review.

---

## TapKO CLI and evaluation

**Empty or sparse `tapko_predictions.json` after `fightsafe tapko-detect`**

- Confirm the clip produced **enough frames** (detectors need a minimum pose history; very short clips may yield zero candidates).
- Check **pose coverage** (`pose_keypoints.csv` in the run dir when not using `--pose-csv`): widespread dropout or wrong FPS makes motion gates unreliable.
- **FPS** passed to `tapko-detect` must match how you **extracted** frames if you reuse a prior `pose_keypoints.csv` from another run.

**`tapko-evaluate` yields zero matches or odd FP/FN**

- **`video_id`** in the annotation file must **match** the string embedded in predictions (default: stem of the video file for `tapko-detect`; override with `--video-id`).
- Reported metrics depend on **`--iou-threshold`**, **`--tolerance-seconds`**, and **`--match-mode`** (`exact` vs `family`)—tune and document them together (see [`evaluation.md`](evaluation.md)).

**Dashboard TapKO cards**

- Events with types under `submission_signal.*` / `extreme_vulnerability.*` are **candidates** only; if the UI shows “Human confirmation required”, that reflects design—not a certified alert.

**PostgreSQL**

- TapKO **CLI and evaluator** paths do **not** require a database; optional dashboard DB hooks are separate.
