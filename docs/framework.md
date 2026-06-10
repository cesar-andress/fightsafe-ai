# FightSafe AI — research framework

**Documentation note:** FightSafe AI is a **research prototype for decision support**, not a certified medical or officiating system.

FightSafe is a **modular decision support** stack for **combat sports** safety intelligence. It emphasizes **real-time risk estimation**, **event-level safety alerts**, and **human-in-the-loop feedback**. It does **not** replace referees or medical staff, certify clinical safety, or provide validated knockout detection. Outputs are signals for **human-in-the-loop** review, research, and training workflows.

## Design principles

1. **MVP compatibility** — The default path (frames → MediaPipe pose → biomechanics →
   interpretable risk → overlay → report) remains the supported baseline.
2. **Pluggable components** — Pose, tracking, action, anomaly, and risk *fusion* layers expose
   small abstract interfaces so labs can swap implementations without forking the pipeline.
3. **Testability** — `mock` pose backends, pure NumPy heuristics, and offline dataset helpers keep
   CI free of real video, network, or GPU requirements.
4. **Explainability** — Optional Ollama-based text (see `fightsafe_ai.llm`) runs **after** risk
   scoring and never controls detection.

## High-level map

| Area | Package | Role |
|------|---------|------|
| Pose | `fightsafe_ai.pose` | `BasePoseEstimator` implementations: **MediaPipe** (default), **YOLO** (optional ultralytics), **Mock** (tests) |
| Tracking | `fightsafe_ai.tracking` | `BaseTracker` + `SportsTracker` placeholder for multi-fighter boxes / IDs |
| Action | `fightsafe_ai.action` | Heuristics and `BaseActionRecognizer` for strikes / guard (extensible) |
| Anomaly | `fightsafe_ai.anomaly` | Fall / inactivity / limb / surrender *signals* (no clinical claim) |
| Risk | `fightsafe_ai.risk` | Interpretable rules, `RiskLevelName`, `fusion` helpers for experiments |
| HCI | `fightsafe_ai.hci` | `Alert`, `RefereeAlertManager`, English `referee_messages` copy |
| Datasets | `fightsafe_ai.datasets` | `DatasetMetadata` registry, COCO / YOLO *file* helpers (no download); see `docs/datasets.md` |
| Evaluation | `fightsafe_ai.evaluation` | Frame / event metrics, ablation presets; **TapKO** interval metrics in `tapko_evaluator`; see `docs/evaluation.md` |
| TapKO | `fightsafe_ai.events`, `fightsafe_ai.tapko`, `fightsafe_ai.evaluation.tapko_evaluator` | **`submission_signal.*`** / **`extreme_vulnerability.*`** pose candidates; offline **`fightsafe tapko-detect`** → predictions JSON; **`fightsafe tapko-evaluate`** vs TapKO annotations. Schema: `annotation.tapko_schema`; docs: [`tapko_annotation.md`](tapko_annotation.md), [`evaluation.md`](evaluation.md) |
| LLM | `fightsafe_ai.llm` | `OllamaClient`, prompts, `explainer` (post-hoc text) |

## Configuration

- `configs/framework.yaml` — optional keys such as `pose.backend` (default `mediapipe`). Loaded
  by `fightsafe_ai.config.framework.load_framework_config` (deep-merge with defaults).
- `configs/risk_rules.yaml` — tunable **interpretable** rule weights (unchanged).
- `configs/llm.yaml` — Ollama toggles and models for explanations / narratives.

CLI `run-pipeline` accepts `--pose-backend` to override the default for a run without editing YAML.

## Safety and ethics

- Outputs are **advisory**; sanctioning, medical diagnosis, and fight results remain with
  qualified humans and governing bodies.
- “Surrender”, “fall”, and “injury anomaly” language refers to **heuristic, research-grade**
  software signals, not medical fact.
- **TapKO** outputs (tap-out–like or vulnerability **candidates**) must not be presented as
  official outcomes; see README disclaimer and [`evaluation.md`](evaluation.md) (TapKO limitations).

## Citation and extension

When publishing, cite the software version (`fightsafe_ai.__version__`) and document which
backend, tracker, and action modules you enabled. Prefer registering custom datasets in
`fightsafe_ai.datasets.registry.BUILTIN_REGISTRY` (see `docs/datasets.md`) for reproducibility in your environment.
