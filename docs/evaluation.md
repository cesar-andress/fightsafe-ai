# Evaluation metrics (FightSafe AI)

**Documentation note:** FightSafe AI is a **research software artifact** for traceability and auditability; high metric scores on a private split do **not** imply certified medical safety, replacement of referees, operator benefit, or deployment readiness without an explicit protocol and independent review.

This document explains how **offline** evaluation helpers in `fightsafe_ai.evaluation` should be used in research. The code does **not** perform automated benchmark submission; it provides **definitions** and **pure functions** so that teams can report metrics consistently for **real-time risk estimation** and **event-level safety alerts** against curated labels.

## Scope and limitations

- Metrics are **agnostic** to video codecs, frame rates, and rule sets unless you **align** labels and predictions to a **common time base** (e.g. seconds or frame indices).
- **Ground truth** for rare safety events is often **sparse** and **subjective**; reported numbers are only as sound as the annotation protocol. For a **file-based** manual label format (JSON) and the `fightsafe annotate-template` / `validate-annotations` commands, see [`docs/annotation.md`](annotation.md).
- **LLM explanations** do not change detection scores; ablations that add the LLM row measure **narrative** cost and (optionally) human preference, not detection F1.

## Frame-level metrics (`metrics.py`)

Use when you have **per-frame** binary (or positive-class) labels after thresholding a risk flag, alert, or segment mask.

| Function | Meaning |
|----------|---------|
| `frame_level_precision` | Among frames predicted positive, fraction that are true positives. |
| `frame_level_recall` | Among ground-truth positive frames, fraction predicted positive. |
| `frame_level_f1` | Harmonic mean of precision and recall. |
| `precision_recall_f1` | Same, for parallel `list[int]` with an integer `positive_label`. |
| `false_alarm_rate_frames` | FP / count(ground-truth **negative** frames). If there are no GT negatives, the function returns `0.0` (convention for undefined). |
| `pose_coverage` | Fraction of frames marked as having **valid** pose (e.g. detection success flags). |

**Usage:** Build `y_true` and `y_pred` with the **same length** and **aligned indices** (same FPS and crop). For class imbalance, pair frame metrics with **event-level** metrics and **error analysis** (false alarms on crowd shots, etc.).

## Event-level metrics (`event_metrics.py`)

Use when labels and outputs are **intervals** `[start, end]` in seconds or frames (`EventWindow`).

| Function | Meaning |
|----------|---------|
| `temporal_iou` | 1D temporal intersection-over-union of two intervals. |
| `match_events_greedy_iou` | Greedy one-to-one matching with a minimum IoU; **not** a global optimum. |
| `event_level_metrics` | Precision / recall / F1 from matched pairs (TP = number of matches; FP = unmatched predictions; FN = unmatched references). |
| `event_level_precision` / `event_level_recall` / `event_level_f1` | Convenience accessors. |
| `mean_absolute_onset_error` | Mean abs (`pred.start - ref.start`) over matched pairs. |
| `false_alarm_rate_events` | Unmatched predictions / total predicted events (spurious event rate among **predicted** segments). |
| `mean_time_to_alert_seconds` | For each reference interval, delay from its onset to the first alert time ≥ onset; **mean** over refs that receive an alert. |
| `alert_consistency` | Exact **string** match rate between two **aligned** per-frame alert-level sequences (e.g. two runs or two thresholds). |
| `best_match` | Legacy greedy match (pred-centric); prefer `match_events_greedy_iou` for symmetric TP/FP/FN. |

**IoU threshold** is a **hyperparameter**; report the value you use. Higher thresholds demand tighter localization.

### `event_matching.py` and `fightsafe evaluate` (ground truth vs `events.json`)

Use this when you have a **manually** labelled JSON (see [annotation](annotation.md)) and a run’s **`events.json`** to compare at **segment** level with temporal overlap (not per-frame risk).

| Item | Meaning |
|------|---------|
| `match_events` | Greedily pairs predicted :class:`EventWindow` rows with reference rows by IoU, optional **label agreement**, and optional **tolerance** (dilate intervals by `tolerance_s/2` on each end before testing overlap). |
| `evaluate_event_prediction` | Produces :class:`EventEvaluationResult` with **precision, recall, F1**, false positive / false negative **counts**, and per-match **onset/offset delay** (pred start − ref start) for detection lag. |
| `events_json_to_windows` / `annotation_file_to_ground_truth_windows` | Load a pipeline `events.json` and an annotation file into a common :class:`EventWindow` list. |
| `event_evaluation_to_json_dict` | Serialize metrics for a JSON report. |

**CLI (writes `metrics.json` by default, prints a summary on stdout):**

```bash
fightsafe evaluate --predicted runs/demo/events.json --ground-truth annotations/demo_annotations.json
fightsafe evaluate -p runs/demo/events.json -g annotations/demo_annotations.json -o metrics.json \
  --iou-threshold 0.15 --tolerance-seconds 0.2
```

- **`--tolerance-seconds`**: use when small clock/annotation drift should still allow a **match**; overlap is checked on *dilated* segments (symmetric).
- **`--require-same-label`**: if set, a prediction and reference are candidates only if their :class:`EventWindow.label` strings are equal (case-insensitive). Default is **off**, because the pipeline’s `event_level` (e.g. HIGH) often **does not** line up with hand `event_type` (e.g. KO) until you add a protocol mapping.

**Example (library):**

```python
from fightsafe_ai.evaluation import (
    evaluate_event_prediction,
    event_evaluation_to_json_dict,
    events_json_to_windows,
    annotation_file_to_ground_truth_windows,
)
pred = events_json_to_windows("runs/demo/events.json")
ref = annotation_file_to_ground_truth_windows("annotations/demo_annotations.json")
r = evaluate_event_prediction(
    pred, ref, iou_threshold=0.1, tolerance_seconds=0.0, require_same_label=False
)
print(r.precision, r.recall, r.f1, r.mean_onset_delay_seconds)
print(event_evaluation_to_json_dict(r))
```

## Ablation presets (`ablation.py`)

`AblationScenario` names **rows** in a study matrix; `ablation_param_template` supplies suggested booleans for logging. Your experiment runner must map these to real **config** (framework YAML, risk rules, LLM on/off).

| Scenario | Intent |
|----------|--------|
| `baseline_biomechanics_only` | Biomechanical features only; no action/anomaly layers. |
| `biomechanics_action` | Add action-signal path and fusion that ingests it. |
| `biomechanics_anomaly` | Add anomaly path and fusion that ingests it. |
| `full_fusion` | Action + anomaly + multi-signal fusion. |
| `full_fusion_llm` | Full fusion **and** post-hoc LLM explanation (scores unchanged). |

Use `make_ablation_row` to create an `AblationRow` and fill `metrics` from your eval script; `sort_rows_by_metric` orders rows for tables.

## Recommended reporting

1. **Time alignment** — document FPS, clip boundaries, and any subsampling.
2. **Thresholds** — event IoU, frame risk threshold, and alert level mapping.
3. **Counts** — number of clips, positive prevalence, and excluded failures (e.g. empty pose).
4. **Ablations** — use the same test split for every `AblationScenario` row.
5. **LLM** — if reporting “+ LLM”, clarify that metrics on **detection** are identical unless you measure **separate** human factors (readability, time to decision).

## Publication figures

Vector and raster assets for articles and slides live under `docs/figures/`. Key extended-framework diagrams (PNG + SVG at 300 dpi for raster):

- `framework_architecture` — pose, tracking, action, anomaly, fusion, HCI, optional LLM.
- `risk_fusion_model` — signals → aggregation → LOW / MEDIUM / HIGH / CRITICAL.
- `human_in_the_loop_alerts` — advisory alerts vs. human authority (no autonomous outcome).
- `combat_safety_signal_taxonomy` — pre-critical, critical, contextual groupings.
- `evaluation_protocol` — dataset → annotations → metrics → ablation.

Regenerate all figures from the repository root:

`python docs/figures/generate_paper_figures.py`

## TapKO evaluation (submission_signal / extreme_vulnerability)

**Scope.** The TapKO **evaluator** compares **time-bounded** predicted intervals to **manual TapKO annotations** (same clip `video_id`, times in **seconds**). It is **offline** library code plus CLI wrappers—**no PostgreSQL** required.

| Piece | Location / command |
|-------|---------------------|
| **Matcher + metrics** | `fightsafe_ai.evaluation.tapko_evaluator` (`evaluate_tapko`, `run_tapko_evaluation_and_write`) |
| **Prediction file** | JSON **array** of objects with at least `video_id`, `start_time`, `end_time`, `event_type` (as produced by `fightsafe tapko-detect` → `tapko_predictions.json`) |
| **Ground-truth file** | TapKO annotation document (JSON) validated by `fightsafe_ai.annotation.tapko_schema` — see [`tapko_annotation.md`](tapko_annotation.md) |
| **CLI** | `fightsafe tapko-evaluate --annotations GT.json --predictions PREDS.json --output-dir OUT` writes `tapko_results.csv`, `tapko_results.tex`, `tapko_error_analysis.md` |
| **Validate labels** | `fightsafe tapko-validate-annotations --annotations FILE` |
| **Example JSON** | `fightsafe tapko-export-examples --output-dir OUT` |
| **End-to-end detect** | `fightsafe tapko-detect --source VIDEO --output-dir RUN --fps FPS` (optional `--pose-csv` to skip re-estimating pose) |

**Families and labels.** Predictions and references use **dot-separated** types (e.g. `submission_signal.hand_tap`, `extreme_vulnerability.ko_collapse`). The evaluator supports **`exact`** label match or **`family`** match (first segment of the type string: `submission_signal` vs `extreme_vulnerability`). Hard-negative intervals in annotations (`negative.*`) are **not** counted as positives but can inform FP analysis—see module docstrings.

**TapKO limitations (evaluation-facing).**

- **Pose-only path** does not hear **verbal** taps; schema allows `submission_signal.verbal_tap` for labels and future audio-aligned runs.
- **Temporal IoU** and tolerance are **hyperparameters**—report `iou_threshold`, `tolerance_seconds`, and `match_mode` with every table.
- Metrics are **not** regulatory or medical validation; they measure agreement under a **protocol**, not real-world officiating correctness.

For generic **MVP** `events.json` vs hand annotations (non-TapKO schema), use `fightsafe evaluate` and [`annotation.md`](annotation.md) as elsewhere in this document.

## See also

- `docs/architecture.md` — pipeline stages, TapKO detector placement, data flow.
- `docs/tapko_annotation.md` — TapKO JSON schema and labeling guidance.
- `docs/datasets.md` — dataset registry and TapKO collection policy.
- `docs/q1-framework.md` — research framing and limitations.
