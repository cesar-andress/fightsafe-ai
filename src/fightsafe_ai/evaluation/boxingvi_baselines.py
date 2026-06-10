"""
BoxingVI offline baselines: same JSON schema as :mod:`fightsafe_ai.evaluation.boxingvi_skeleton_runner`,
evaluated with :func:`fightsafe_ai.evaluation.boxingvi_evaluator.evaluate_boxingvi_video`.

Baselines (decision-support only; no ground truth inside detectors):

1. **velocity_threshold** — wrist-speed magnitude peaks only (no smoothing, no acceleration term,
   no risk/anomaly/strike fusion).
2. **anomaly_only** — sequential anomaly detectors only; **HIGH** / **CRITICAL** episodes mapped to
   impact-like candidates (no strike detector, no fuse rules).
3. **strike_detector** — :func:`fightsafe_ai.evaluation.boxingvi_strike_detector.detect_strike_events`.
4. **full_fusion** — skeleton runner (strike + risk + anomaly), then **timeline merge**: impact-like
   intervals from ``events`` and ``anomaly_events`` are merged when the gap (seconds) is
   ≤ ``strike_merge_frames / fps`` (same gap semantics as strike segment merge).

CLI::

    python -m fightsafe_ai.evaluation.boxingvi_baselines \\
      --dataset-root data/boxingvi \\
      --video-ids V1 V2 ... \\
      --fps 30 \\
      --output-dir outputs/evaluation/baselines \\
      --force
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from fightsafe_ai.anomaly.fall_detector import FallDetector
from fightsafe_ai.anomaly.inactivity_detector import InactivityDetector
from fightsafe_ai.anomaly.limb_anomaly import LimbAnomalyDetector
from fightsafe_ai.datasets.boxingvi import BoxingVIDataset
from fightsafe_ai.evaluation.boxingvi_batch_eval import (
    _compute_micro_macro,
    _write_baseline_comparison_csv,
    _write_baseline_comparison_tex,
)
from fightsafe_ai.evaluation.boxingvi_evaluator import (
    BoxingVIEvalResult,
    _qualifies_impact_like_prediction,
    evaluate_boxingvi_video,
    write_boxingvi_results_csv,
)
from fightsafe_ai.evaluation.boxingvi_skeleton_runner import (
    _anomaly_signals_to_events,
    _safety_event_to_json,
    build_landmark_sequence_from_skeleton,
    run_boxingvi_skeleton_evaluation,
)
from fightsafe_ai.evaluation.boxingvi_strike_detector import (
    _EPS,
    _count_valid_focus,
    _ensure_thpj2,
    _merge_segments,
    _person_strike_score,
    _segments_above_threshold,
    detect_strike_events,
)
from fightsafe_ai.live.event_bus import SafetyEvent


logger = logging.getLogger(__name__)

BaselineName = Literal[
    "velocity_threshold",
    "anomaly_only",
    "strike_detector",
    "full_fusion",
]

BASELINE_ORDER: Final[tuple[str, ...]] = (
    "velocity_threshold",
    "anomaly_only",
    "strike_detector",
    "full_fusion",
)


def detect_velocity_threshold_events(
    skeleton: np.ndarray,
    *,
    fps: float,
    percentile: float,
    merge_frames: int,
    min_valid_keypoints: int,
) -> list[dict[str, Any]]:
    """
    Wrist **velocity** (per-frame max L/R speed magnitude), threshold at percentile on non-zero scores.

    No smoothing window, no acceleration features, no fusion.
    """
    arr = _ensure_thpj2(skeleton)
    t_max, p_max, _, _ = arr.shape
    if t_max < 2:
        return []

    fd = float(fps)
    if fd <= 0:
        raise ValueError("fps must be positive.")
    pct = float(percentile)
    if not 0.0 < pct < 100.0:
        raise ValueError("percentile must lie strictly between 0 and 100.")

    raw_score = np.zeros(t_max, dtype=np.float64)
    for ti in range(1, t_max):
        prev_f = arr[ti - 1]
        curr_f = arr[ti]
        ranked_pids: list[tuple[int, int]] = []
        for pid in range(p_max):
            nvalid = _count_valid_focus(curr_f[pid])
            if nvalid >= min_valid_keypoints:
                ranked_pids.append((nvalid, pid))
        ranked_pids.sort(key=lambda x: (-x[0], x[1]))
        best_scores: list[float] = []
        for _nv, pid in ranked_pids[:2]:
            sc = _person_strike_score(prev_f[pid], curr_f[pid], fd, min_valid_keypoints)
            if sc is not None:
                best_scores.append(float(sc))
        if best_scores:
            raw_score[ti] = max(best_scores)

    positive = raw_score[raw_score > _EPS]
    if positive.size == 0:
        return []

    thr = float(np.percentile(positive, pct))
    mask = raw_score > max(thr, _EPS)
    segs = _segments_above_threshold(mask)
    segs = _merge_segments(segs, merge_frames)
    if not segs:
        return []

    pad_w = max(4, len(str(t_max - 1)))
    out: list[dict[str, Any]] = []
    for k, (sf, ef) in enumerate(segs):
        seg_scores = raw_score[sf : ef + 1]
        peak = float(np.max(seg_scores)) if seg_scores.size else 0.0
        t0 = float(sf) / fd
        t1 = float(ef + 1) / fd
        eid = f"velocity_peak_{k + 1:05d}"
        sf_s = str(sf).zfill(pad_w)
        ef_s = str(ef).zfill(pad_w)
        desc = f"Wrist velocity threshold only (no smoothing; percentile={pct:.1f}; merge_gap≤{merge_frames})."
        out.append(
            {
                "event_id": eid,
                "start_frame": sf_s,
                "end_frame": ef_s,
                "start_time": t0,
                "end_time": t1,
                "max_risk_score": peak,
                "score": peak,
                "event_level": "HIGH",
                "level": "HIGH",
                "category": "impact",
                "event_type": "boxingvi.velocity_threshold",
                "title": "Velocity threshold candidate",
                "description": desc,
            }
        )
    return out


def _collect_sequential_anomaly_events(
    per_frame_lm: list[Any],
    fps: float,
) -> list[SafetyEvent]:
    """Same sequential causal window as :func:`run_boxingvi_skeleton_evaluation` (no risk rows)."""
    times = [i / float(fps) for i in range(len(per_frame_lm))]
    frame_dicts: list[dict[str, tuple[float, float]]] = [
        dict(lm) if lm is not None else {} for lm in per_frame_lm
    ]
    anomaly_events: list[SafetyEvent] = []
    fall_det = FallDetector()
    limb_det = LimbAnomalyDetector()
    inact_det = InactivityDetector()
    for t in range(len(frame_dicts)):
        prefix_t = t + 1
        sub_times = times[:prefix_t]
        sub_frames = frame_dicts[:prefix_t]
        for det in (fall_det, limb_det, inact_det):
            for sig in det.analyze(sub_times, sub_frames, "fighter_0"):
                anomaly_events.extend(_anomaly_signals_to_events([sig]))
    return anomaly_events


def _impact_candidates_from_anomalies(events: list[SafetyEvent]) -> list[dict[str, Any]]:
    """Keep HIGH/CRITICAL only; map to impact-like dicts for :func:`load_prediction_impact_windows`."""
    out: list[dict[str, Any]] = []
    for ev in events:
        lvl = str(ev.level).strip().upper()
        if lvl not in {"HIGH", "CRITICAL"}:
            continue
        d = _safety_event_to_json(ev)
        d["event_level"] = lvl
        d["category"] = "impact"
        d["event_type"] = f"{ev.event_type}.impact_candidate"
        d["title"] = str(ev.title)[:120]
        out.append(d)
    return out


def write_predictions_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _base_summary(
    *,
    video_id: str,
    fps: float,
    skeleton_path: Path,
    arr: np.ndarray,
    baseline: str,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "fps": float(fps),
        "baseline": baseline,
        "skeleton_path": str(skeleton_path),
        "skeleton_shape": list(arr.shape),
    }


def build_payload_velocity_threshold(
    dataset_root: Path,
    video_id: str,
    arr: np.ndarray,
    *,
    fps: float,
    strike_percentile: float,
    strike_merge_frames: int,
    min_valid_keypoints: int,
) -> dict[str, Any]:
    evs = detect_velocity_threshold_events(
        arr,
        fps=float(fps),
        percentile=float(strike_percentile),
        merge_frames=int(strike_merge_frames),
        min_valid_keypoints=int(min_valid_keypoints),
    )
    sk_path = Path(dataset_root) / "skeleton" / f"{video_id}.npy"
    summary = _base_summary(
        video_id=video_id,
        fps=fps,
        skeleton_path=sk_path,
        arr=arr,
        baseline="velocity_threshold",
    )
    return {
        **summary,
        "risk_rules_yaml": None,
        "events": evs,
        "events_risk_only": [],
        "anomaly_events": [],
        "strike_events": [],
        "n_frames_risk": 0,
        "n_anomaly_signals": 0,
        "n_strike_events": 0,
    }


def build_payload_anomaly_only(
    dataset_root: Path,
    video_id: str,
    arr: np.ndarray,
    *,
    fps: float,
    min_valid_keypoints: int,
) -> dict[str, Any]:
    _, per_frame_lm = build_landmark_sequence_from_skeleton(
        arr, min_valid_keypoints=min_valid_keypoints
    )
    raw_anom = _collect_sequential_anomaly_events(per_frame_lm, float(fps))
    impact_like = _impact_candidates_from_anomalies(raw_anom)
    sk_path = Path(dataset_root) / "skeleton" / f"{video_id}.npy"
    summary = _base_summary(
        video_id=video_id,
        fps=fps,
        skeleton_path=sk_path,
        arr=arr,
        baseline="anomaly_only",
    )
    return {
        **summary,
        "risk_rules_yaml": None,
        "events": impact_like,
        "events_risk_only": [],
        "anomaly_events": [],
        "strike_events": [],
        "n_frames_risk": 0,
        "n_anomaly_signals": len(raw_anom),
        "n_strike_events": 0,
    }


def build_payload_strike_detector(
    dataset_root: Path,
    video_id: str,
    arr: np.ndarray,
    *,
    fps: float,
    strike_percentile: float,
    strike_merge_frames: int,
    min_valid_keypoints: int,
) -> dict[str, Any]:
    strikes = detect_strike_events(
        arr,
        fps=float(fps),
        percentile=float(strike_percentile),
        merge_frames=int(strike_merge_frames),
        min_valid_keypoints=int(min_valid_keypoints),
    )
    sk_path = Path(dataset_root) / "skeleton" / f"{video_id}.npy"
    summary = _base_summary(
        video_id=video_id,
        fps=fps,
        skeleton_path=sk_path,
        arr=arr,
        baseline="strike_detector",
    )
    return {
        **summary,
        "risk_rules_yaml": None,
        "events": [],
        "events_risk_only": [],
        "anomaly_events": [],
        "strike_events": strikes,
        "n_frames_risk": 0,
        "n_anomaly_signals": 0,
        "n_strike_events": len(strikes),
    }


def run_full_fusion_runner(
    *,
    dataset_root: Path,
    video_id: str,
    output_dir: Path,
    fps: float,
    rolling_window: int,
    min_valid_keypoints: int,
    strike_percentile: float,
    strike_merge_frames: int,
    rules_yaml: Path | None,
) -> dict[str, Any]:
    """Delegates to :func:`run_boxingvi_skeleton_evaluation` (writes JSON + frame CSV under ``output_dir``)."""
    return run_boxingvi_skeleton_evaluation(
        dataset_root=dataset_root,
        video_id=video_id,
        fps=float(fps),
        rolling_window=int(rolling_window),
        min_valid_keypoints=int(min_valid_keypoints),
        output_dir=output_dir,
        rules_yaml=rules_yaml,
        enable_strike_detector=True,
        strike_percentile=float(strike_percentile),
        strike_merge_frames=int(strike_merge_frames),
    )


def apply_full_fusion_timeline_merge(
    pred_json: Path,
    *,
    fps: float,
    merge_frames: int,
) -> None:
    """
    After the skeleton runner, **full_fusion** should present one deduplicated timeline to the
    evaluator: union impact-like rows from ``events`` and ``anomaly_events``, then merge
    adjacent/overlapping windows when the start of the next event is within ``merge_frames`` worth
    of time after the current end (gap ≤ ``merge_frames / fps`` seconds).
    Merged result is written to ``events``; ``anomaly_events`` is cleared to avoid double counting
    (same rule as :func:`load_prediction_impact_windows` for ``full_fusion``).
    """
    path = Path(pred_json).expanduser().resolve()
    if not path.is_file():
        return
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    fd = float(fps)
    if fd <= 0:
        return
    gap_sec = max(0.0, float(merge_frames) / fd)

    candidates: list[dict[str, Any]] = []
    for key in ("events", "anomaly_events"):
        block = raw.get(key)
        if not isinstance(block, list):
            continue
        for ev in block:
            if not isinstance(ev, dict):
                continue
            if not _qualifies_impact_like_prediction(ev):
                continue
            t0, t1 = ev.get("start_time"), ev.get("end_time")
            if t0 is None or t1 is None:
                continue
            candidates.append(dict(ev))

    if not candidates:
        raw["anomaly_events"] = []
        raw["full_fusion_timeline_merge_frames"] = int(merge_frames)
        path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        return

    candidates.sort(key=lambda d: (float(d["start_time"]), float(d["end_time"])))
    merged: list[dict[str, Any]] = []
    cur: dict[str, Any] = dict(candidates[0])
    cur_e = float(cur["end_time"])
    for nxt in candidates[1:]:
        s = float(nxt["start_time"])
        e = float(nxt["end_time"])
        if s <= cur_e + gap_sec:
            cur_e = max(cur_e, e)
            cur["end_time"] = cur_e
            for sk in ("score", "max_risk_score"):
                try:
                    cur[sk] = max(
                        float(cur.get(sk) or 0.0),
                        float(nxt.get(sk) or 0.0),
                    )
                except (TypeError, ValueError):
                    pass
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur_e = float(cur["end_time"])
    merged.append(cur)

    raw["events"] = merged
    raw["anomaly_events"] = []
    raw["full_fusion_timeline_merge_frames"] = int(merge_frames)
    path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")


def _prediction_subset_for_baseline(name: str) -> str:
    if name == "strike_detector":
        return "strike_only"
    return "full_fusion"


def run_baselines_for_video(
    *,
    dataset_root: Path,
    video_id: str,
    output_root: Path,
    fps: float,
    tolerance_seconds: float,
    iou_threshold: float,
    rolling_window: int,
    strike_percentile: float,
    strike_merge_frames: int,
    min_valid_keypoints: int,
    rules_yaml: Path | None,
    force: bool,
) -> dict[str, BoxingVIEvalResult | None]:
    """
    For each baseline, write predictions JSON and results CSV under ``output_root/<baseline>/``.

    Returns mapping baseline_name -> result or None if evaluation failed / skipped.
    """
    ds = BoxingVIDataset(dataset_root)
    stem = str(video_id).strip()
    results: dict[str, BoxingVIEvalResult | None] = {}

    try:
        arr = ds.load_skeleton(stem)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Skip %s: cannot load skeleton (%s)", stem, exc)
        for b in BASELINE_ORDER:
            results[b] = None
        return results

    for baseline in BASELINE_ORDER:
        out_dir = output_root / baseline
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"boxingvi_predictions_{stem}.json"
        res_path = out_dir / f"boxingvi_results_{stem}.csv"

        need_write = force or not pred_path.is_file()
        try:
            if need_write:
                if baseline == "velocity_threshold":
                    payload = build_payload_velocity_threshold(
                        dataset_root,
                        stem,
                        arr,
                        fps=fps,
                        strike_percentile=strike_percentile,
                        strike_merge_frames=strike_merge_frames,
                        min_valid_keypoints=min_valid_keypoints,
                    )
                    write_predictions_json(pred_path, payload)
                elif baseline == "anomaly_only":
                    payload = build_payload_anomaly_only(
                        dataset_root,
                        stem,
                        arr,
                        fps=fps,
                        min_valid_keypoints=min_valid_keypoints,
                    )
                    write_predictions_json(pred_path, payload)
                elif baseline == "strike_detector":
                    payload = build_payload_strike_detector(
                        dataset_root,
                        stem,
                        arr,
                        fps=fps,
                        strike_percentile=strike_percentile,
                        strike_merge_frames=strike_merge_frames,
                        min_valid_keypoints=min_valid_keypoints,
                    )
                    write_predictions_json(pred_path, payload)
                elif baseline == "full_fusion":
                    run_full_fusion_runner(
                        dataset_root=dataset_root,
                        video_id=stem,
                        output_dir=out_dir,
                        fps=fps,
                        rolling_window=rolling_window,
                        min_valid_keypoints=min_valid_keypoints,
                        strike_percentile=strike_percentile,
                        strike_merge_frames=strike_merge_frames,
                        rules_yaml=rules_yaml,
                    )
                else:
                    raise ValueError(f"Unknown baseline {baseline!r}")
            else:
                logger.info("Reuse predictions %s (pass --force to regenerate)", pred_path)
        except Exception as exc:
            logger.exception("Baseline %s failed for %s: %s", baseline, stem, exc)
            results[baseline] = None
            continue

        if not pred_path.is_file():
            logger.warning("Missing predictions JSON %s", pred_path)
            results[baseline] = None
            continue

        if baseline == "full_fusion":
            try:
                apply_full_fusion_timeline_merge(
                    pred_path, fps=float(fps), merge_frames=int(strike_merge_frames)
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("full_fusion timeline merge failed for %s: %s", pred_path, exc)

        try:
            er = evaluate_boxingvi_video(
                dataset_root=dataset_root,
                video_id=stem,
                predictions_path=pred_path,
                fps=float(fps),
                tolerance_seconds=float(tolerance_seconds),
                iou_threshold=float(iou_threshold),
                prediction_subset=_prediction_subset_for_baseline(baseline),
            )
            write_boxingvi_results_csv(res_path, er, append=False)
            results[baseline] = er
        except Exception as exc:
            logger.warning("Evaluate failed %s / %s: %s", baseline, stem, exc)
            results[baseline] = None

    return results


def run_all_videos_and_aggregate(
    *,
    dataset_root: Path,
    video_ids: list[str],
    output_root: Path,
    fps: float,
    tolerance_seconds: float,
    iou_threshold: float,
    rolling_window: int,
    strike_percentile: float,
    strike_merge_frames: int,
    min_valid_keypoints: int,
    rules_yaml: Path | None,
    force: bool,
) -> list[dict[str, Any]]:
    """
    Run every baseline for every video; write ``baseline_comparison.{csv,tex}`` under ``output_root``.
    """
    per_baseline: dict[str, list[BoxingVIEvalResult]] = {b: [] for b in BASELINE_ORDER}

    for vid in video_ids:
        row = run_baselines_for_video(
            dataset_root=dataset_root,
            video_id=str(vid).strip(),
            output_root=output_root,
            fps=fps,
            tolerance_seconds=tolerance_seconds,
            iou_threshold=iou_threshold,
            rolling_window=rolling_window,
            strike_percentile=strike_percentile,
            strike_merge_frames=strike_merge_frames,
            min_valid_keypoints=min_valid_keypoints,
            rules_yaml=rules_yaml,
            force=force,
        )
        for b, res in row.items():
            if res is not None:
                per_baseline[b].append(res)

    comparison_rows: list[dict[str, Any]] = []
    for b in BASELINE_ORDER:
        ok = per_baseline[b]
        if not ok:
            comparison_rows.append(
                {
                    "baseline": b,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "micro_precision": 0.0,
                    "micro_recall": 0.0,
                    "micro_f1": 0.0,
                    "macro_precision": 0.0,
                    "macro_recall": 0.0,
                    "macro_f1": 0.0,
                    "mean_latency": 0.0,
                }
            )
            continue
        summary = _compute_micro_macro(ok)
        comparison_rows.append(
            {
                "baseline": b,
                "TP": summary["micro_tp"],
                "FP": summary["micro_fp"],
                "FN": summary["micro_fn"],
                "micro_precision": summary["micro_precision"],
                "micro_recall": summary["micro_recall"],
                "micro_f1": summary["micro_f1"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "mean_latency": summary["mean_latency_mean"],
            }
        )

    csv_path = output_root / "baseline_comparison.csv"
    tex_path = output_root / "baseline_comparison.tex"
    _write_baseline_comparison_csv(csv_path, comparison_rows)
    _write_baseline_comparison_tex(tex_path, comparison_rows)
    logger.info("Wrote %s and %s", csv_path, tex_path)
    return comparison_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--video-ids", nargs="+", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tolerance-seconds", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.01)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rolling-window", type=int, default=5)
    p.add_argument("--strike-percentile", type=float, default=85.0)
    p.add_argument("--strike-merge-frames", type=int, default=8)
    p.add_argument("--min-valid-keypoints", type=int, default=5)
    p.add_argument(
        "--rules-yaml",
        type=Path,
        default=None,
        help="Risk rules YAML for full_fusion only (default: repo configs/risk_rules.yaml)",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    rules = args.rules_yaml.expanduser().resolve() if args.rules_yaml else None

    run_all_videos_and_aggregate(
        dataset_root=dataset_root,
        video_ids=list(args.video_ids),
        output_root=output_root,
        fps=float(args.fps),
        tolerance_seconds=float(args.tolerance_seconds),
        iou_threshold=float(args.iou_threshold),
        rolling_window=int(args.rolling_window),
        strike_percentile=float(args.strike_percentile),
        strike_merge_frames=int(args.strike_merge_frames),
        min_valid_keypoints=int(args.min_valid_keypoints),
        rules_yaml=rules,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "BASELINE_ORDER",
    "BaselineName",
    "apply_full_fusion_timeline_merge",
    "build_payload_anomaly_only",
    "build_payload_strike_detector",
    "build_payload_velocity_threshold",
    "detect_velocity_threshold_events",
    "main",
    "run_all_videos_and_aggregate",
    "run_baselines_for_video",
    "run_full_fusion_runner",
    "write_predictions_json",
]
