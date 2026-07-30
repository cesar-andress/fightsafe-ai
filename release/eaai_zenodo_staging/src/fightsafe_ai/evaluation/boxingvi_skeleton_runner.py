"""
Offline **skeleton-only** pipeline for BoxingVI: COCO-17 ``.npy`` → features → risk → events.

No RGB or pose models are required. Input is expected under ``<dataset_root>/skeleton/<video_id>.npy``.

Typical array layouts:

- ``(T, 17, 2)`` — one person, x/y per joint
- ``(T, 17, 3)`` — x, y, confidence
- ``(T, P, 17, 2|3)`` — multiple people; the runner picks the best-scoring person per frame
- ``(T, 51)`` — flattened ``17 * 3`` (x, y, conf) per row
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from fightsafe_ai.anomaly.fall_detector import FallDetector
from fightsafe_ai.anomaly.inactivity_detector import InactivityDetector
from fightsafe_ai.anomaly.limb_anomaly import LimbAnomalyDetector
from fightsafe_ai.datasets.boxingvi import BoxingVIDataset
from fightsafe_ai.evaluation.boxingvi_strike_detector import detect_strike_events
from fightsafe_ai.features.anomaly import add_limb_anomaly_columns
from fightsafe_ai.features.biomechanics import (
    LandmarkMap,
    build_biomechanical_mvp_dataframe_from_landmark_sequence,
)
from fightsafe_ai.features.temporal import compute_temporal_features
from fightsafe_ai.live.event_bus import EventCategory, SafetyEvent, SafetyLevel
from fightsafe_ai.risk.events import (
    COL_FRAME_ID,
    RiskEventExtractionConfig,
    frame_risk_to_events_list,
)
from fightsafe_ai.risk.scorer import (
    COL_RISK_LEVEL,
    COL_RISK_SCORE,
    COL_TRIGGERED,
    build_combat_mvp_frame_risk,
)


logger = logging.getLogger(__name__)

# COCO-17 person keypoints (same order as :data:`fightsafe_ai.pose.backends.yolo_pose_backend._COCO17`).
_COCO17_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_DEFAULT_MIN_VALID_KP: Final[int] = 4
_CONF_THRESHOLD: Final[float] = 0.05


def _repo_root_configs_risk_rules() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *list(here.parents)]:
        p = parent / "configs" / "risk_rules.yaml"
        if p.is_file():
            return p
    return None


def _reshape_skeleton_array(arr: np.ndarray) -> np.ndarray:
    """Return array with shape starting with ``(T, ...)`` and trailing ``17`` joints."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 51:
        return a.reshape(a.shape[0], 17, 3)
    if a.ndim == 2 and a.shape[1] == 34:
        return a.reshape(a.shape[0], 17, 2)
    return a


def _person_quality(xy: np.ndarray, conf: np.ndarray | None) -> float:
    """Higher is better: valid keypoint count + spatial spread proxy."""
    if xy.size == 0:
        return -1.0
    if xy.ndim != 2 or xy.shape[0] != 17:
        return -1.0
    if conf is None:
        conf = np.ones(17, dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    nonzero = np.any(np.abs(xy) > 1e-8, axis=1)
    cmask = conf > _CONF_THRESHOLD
    valid = finite & nonzero & cmask
    n = int(np.sum(valid))
    if n < 2:
        return float(n) - 1.0
    pts = xy[valid]
    spread = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]))
    return float(n) * 1_000_000.0 + spread


def _select_person_slice(frame_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    From one frame's raw slice, return ``(xy (17,2), conf (17,) | None)``.
    Handles ``(17,2|3)``, ``(P,17,2|3)``, or malformed input (returns zeros).
    """
    ft = np.asarray(frame_tensor)
    if ft.ndim == 2 and ft.shape[0] == 17:
        xy = ft[:, :2].astype(np.float64, copy=False)
        conf = ft[:, 2].astype(np.float64, copy=False) if ft.shape[1] >= 3 else None
        return xy, conf
    if ft.ndim == 3:
        # (P, 17, D)
        best_sl: np.ndarray | None = None
        best_q = -1.0
        for p in range(ft.shape[0]):
            sl = ft[p]
            xy_p = sl[:, :2].astype(np.float64, copy=False)
            conf_p = sl[:, 2].astype(np.float64, copy=False) if sl.shape[1] >= 3 else None
            q = _person_quality(xy_p, conf_p)
            if q > best_q:
                best_q = q
                best_sl = sl
        if best_sl is None:
            return np.full((17, 2), np.nan), None
        xy = best_sl[:, :2].astype(np.float64, copy=False)
        conf = best_sl[:, 2].astype(np.float64, copy=False) if best_sl.shape[1] >= 3 else None
        return xy, conf
    return np.full((17, 2), np.nan), None


def _normalize_xy_unit_square(xy: np.ndarray, conf: np.ndarray | None) -> np.ndarray:
    """
    Map valid keypoints into approximately ``[0, 1]``.

    If coordinates already look normalized (max extent ``<= 1.05``), clip only.
    Otherwise apply per-frame axis-aligned bounding box normalization.
    """
    finite = np.isfinite(xy).all(axis=1)
    if conf is not None:
        finite = finite & (conf > _CONF_THRESHOLD)
    nonzero = np.any(np.abs(xy) > 1e-8, axis=1)
    valid = finite & nonzero
    if not np.any(valid):
        return xy
    sub = xy[valid]
    xmin, ymin = float(np.min(sub[:, 0])), float(np.min(sub[:, 1]))
    xmax, ymax = float(np.max(sub[:, 0])), float(np.max(sub[:, 1]))
    span_x = xmax - xmin
    span_y = ymax - ymin
    ext = max(span_x, span_y, 1e-6)
    if xmax <= 1.05 and ymax <= 1.05 and xmin >= -0.05 and ymin >= -0.05 and ext <= 1.05:
        out = np.clip(xy, 0.0, 1.0)
        return cast("np.ndarray", out)
    out = xy.copy()
    out[:, 0] = (xy[:, 0] - xmin) / ext
    out[:, 1] = (xy[:, 1] - ymin) / ext
    return cast("np.ndarray", np.clip(out, 0.0, 1.0))


def _coco_frame_to_landmark_map(
    xy: np.ndarray,
    conf: np.ndarray | None,
    *,
    min_valid: int,
) -> LandmarkMap | None:
    """Build :class:`LandmarkMap` from COCO-17 arrays; drop invalid joints."""
    out: LandmarkMap = {}
    for i, name in enumerate(_COCO17_NAMES):
        x_i, y_i = float(xy[i, 0]), float(xy[i, 1])
        if conf is not None and float(conf[i]) <= _CONF_THRESHOLD:
            continue
        if not (np.isfinite(x_i) and np.isfinite(y_i)):
            continue
        if abs(x_i) < 1e-8 and abs(y_i) < 1e-8:
            continue
        out[name] = (x_i, y_i)
    if len(out) < min_valid:
        return None
    return out


def _parse_skeleton_sequence(arr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """One entry per time step: raw ``(17,2|3)`` before global normalization."""
    a = _reshape_skeleton_array(arr)
    if a.ndim < 2:
        return []
    t = a.shape[0]
    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    for ti in range(t):
        xy, conf = _select_person_slice(a[ti])
        out.append((xy, conf))
    return out


def build_landmark_sequence_from_skeleton(
    arr: np.ndarray,
    *,
    min_valid_keypoints: int = _DEFAULT_MIN_VALID_KP,
) -> tuple[list[tuple[str, LandmarkMap | None]], list[LandmarkMap | None]]:
    """
    Convert a BoxingVI numpy skeleton tensor into FightSafe landmark frames.

    Returns
    -------
    frames_for_bio
        ``(frame_id, landmarks)`` pairs with **zero-padded** numeric ``frame_id`` for stable sorting.
    per_frame_maps
        Length ``T`` list (``None`` or map) for anomaly detectors aligned with wall-clock indices.
    """
    raw = _parse_skeleton_sequence(arr)
    t_len = len(raw)
    per_frame: list[LandmarkMap | None] = [None] * t_len
    for i, (xy, conf) in enumerate(raw):
        xyn = _normalize_xy_unit_square(xy, conf)
        lm = _coco_frame_to_landmark_map(xyn, conf, min_valid=min_valid_keypoints)
        per_frame[i] = lm

    frames_for_bio: list[tuple[str, LandmarkMap | None]] = []
    for i, lm in enumerate(per_frame):
        fid = f"{i:08d}"
        frames_for_bio.append((fid, lm))
    return frames_for_bio, per_frame


def _timestamps_from_frame_ids(frame_ids: Sequence[Any], fps: float) -> np.ndarray:
    def one(fid: Any) -> float:
        s = str(fid).strip()
        if s.isdigit():
            return int(s) / float(fps)
        if s.startswith("f") and len(s) > 1 and s[1:].isdigit():
            return int(s[1:]) / float(fps)
        return float("nan")

    return np.array([one(x) for x in frame_ids], dtype=np.float64)


def _safety_event_to_json(ev: SafetyEvent) -> dict[str, Any]:
    d = asdict(ev)
    d["category"] = str(ev.category)
    d["level"] = str(ev.level)
    return d


def _category_from_anomaly(name: str) -> EventCategory:
    u = name.upper()
    if "FALL" in u:
        return EventCategory.FALL
    if "INACTIVITY" in u:
        return EventCategory.INACTIVITY
    if "LIMB" in u or "ASYMM" in u:
        return EventCategory.IMBALANCE
    return EventCategory.UNKNOWN


def _anomaly_signals_to_events(signals: list[Any]) -> list[SafetyEvent]:
    from fightsafe_ai.anomaly.base import AnomalySignal

    out: list[SafetyEvent] = []
    for sig in signals:
        if not isinstance(sig, AnomalySignal):
            continue
        ts = float(sig.timestamp)
        cat = _category_from_anomaly(sig.anomaly_type.value)
        out.append(
            SafetyEvent(
                event_type=f"anomaly.{sig.anomaly_type.value.lower()}",
                category=cat,
                start_time=ts,
                end_time=ts,
                level=SafetyLevel.WARNING if float(sig.confidence) < 0.55 else SafetyLevel.HIGH,
                score=float(sig.confidence),
                title=str(sig.anomaly_type.value)[:120],
                description=",".join(f"{k}={v}" for k, v in list(sig.evidence.items())[:4])[:500],
                explanation=f"Heuristic {sig.anomaly_type.value.replace('_', ' ').lower()} (confidence {sig.confidence:.2f})."[
                    :500
                ],
                source=f"anomaly.{sig.anomaly_type.value.split('_')[0].lower()}",
            )
        )
    return out


def run_boxingvi_skeleton_evaluation(
    *,
    dataset_root: Path,
    video_id: str,
    fps: float = 30.0,
    rolling_window: int = 5,
    min_valid_keypoints: int = _DEFAULT_MIN_VALID_KP,
    output_dir: Path | None = None,
    rules_yaml: Path | None = None,
    enable_strike_detector: bool = False,
    strike_percentile: float = 85.0,
    strike_merge_frames: int = 8,
) -> dict[str, Any]:
    """
    Load skeleton ``V*.npy``, run feature + risk (+ heuristic anomaly) scoring, write JSON/CSV.

    Returns a summary dict including paths written and row/event counts.
    """
    ds = BoxingVIDataset(dataset_root)
    sk_path = ds.skeleton_dir / f"{video_id}.npy"
    arr = ds.load_skeleton(video_id)

    strike_events: list[dict[str, Any]] = []
    if enable_strike_detector and arr.shape[0] >= 2:
        strike_events = detect_strike_events(
            arr,
            fps=float(fps),
            percentile=float(strike_percentile),
            merge_frames=int(strike_merge_frames),
            min_valid_keypoints=int(min_valid_keypoints),
        )

    out_root = Path(output_dir or Path("outputs/evaluation")).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"boxingvi_predictions_{video_id}.json"
    csv_path = out_root / f"boxingvi_predictions_{video_id}.csv"

    rules = rules_yaml or _repo_root_configs_risk_rules()

    frames_bio, per_frame_lm = build_landmark_sequence_from_skeleton(
        arr,
        min_valid_keypoints=min_valid_keypoints,
    )

    mvp = build_biomechanical_mvp_dataframe_from_landmark_sequence(
        frames_bio,
        fps=float(fps),
        rolling_window=int(rolling_window),
    )

    summary: dict[str, Any] = {
        "video_id": video_id,
        "fps": float(fps),
        "skeleton_path": str(sk_path),
        "skeleton_shape": list(arr.shape),
        "rolling_window": int(rolling_window),
        "min_valid_keypoints": int(min_valid_keypoints),
        "frames_total": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        "frames_with_landmarks": sum(1 for lm in per_frame_lm if lm),
    }

    if mvp.empty or len(mvp) == 0:
        logger.warning(
            "No biomechanical rows produced (empty or invalid skeleton). Writing empty outputs."
        )
        empty_risk = pd.DataFrame(
            columns=[
                COL_FRAME_ID,
                "timestamp",
                COL_RISK_SCORE,
                COL_RISK_LEVEL,
                COL_TRIGGERED,
            ]
        )
        empty_risk.to_csv(csv_path, index=False)
        merged_events = list(strike_events)
        payload = {
            **summary,
            "risk_rules_yaml": str(rules) if rules else None,
            "events": merged_events,
            "events_risk_only": [],
            "anomaly_events": [],
            "strike_events": strike_events,
            "per_frame_rows": 0,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
        summary["n_events"] = len(merged_events)
        summary["n_frames_risk"] = 0
        summary["n_strike_events"] = len(strike_events)
        return summary

    fps_i = int(max(1, round(float(fps))))
    temp = compute_temporal_features(mvp, fps_i, rolling_window_frames=int(rolling_window))
    with_limb = add_limb_anomaly_columns(temp, float(fps))
    risk_df = build_combat_mvp_frame_risk(
        with_limb,
        float(fps),
        rules_yaml=rules,
        pose_per_frame=None,
    )

    if COL_FRAME_ID in risk_df.columns:
        risk_df = risk_df.copy()
        risk_df["timestamp"] = _timestamps_from_frame_ids(
            risk_df[COL_FRAME_ID].tolist(), float(fps)
        )

    # --- Optional anomaly pass (same detectors as live; prefix windows per frame) ---
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

    ev_cfg = RiskEventExtractionConfig(fps=float(fps))
    events_risk = frame_risk_to_events_list(risk_df, config=ev_cfg)

    # CSV: per-frame risk (primary table for evaluation scripts)
    export_risk = risk_df.copy()
    if COL_TRIGGERED in export_risk.columns:
        export_risk["triggered_rules"] = export_risk[COL_TRIGGERED].apply(
            lambda x: ";".join(str(i) for i in x) if isinstance(x, list) else str(x)
        )
        export_risk = export_risk.drop(columns=[COL_TRIGGERED], errors="ignore")
    export_risk.to_csv(csv_path, index=False)

    merged_events = list(events_risk) + list(strike_events)
    payload = {
        **summary,
        "risk_rules_yaml": str(rules) if rules else None,
        "events": merged_events,
        "events_risk_only": events_risk,
        "anomaly_events": [_safety_event_to_json(e) for e in anomaly_events],
        "strike_events": strike_events,
        "n_frames_risk": len(risk_df),
    }

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    summary["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
    summary["n_events"] = len(merged_events)
    summary["n_frames_risk"] = len(risk_df)
    summary["n_anomaly_signals"] = len(anomaly_events)
    summary["n_strike_events"] = len(strike_events)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BoxingVI skeleton-only FightSafe evaluation runner."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="e.g. data/boxingvi")
    parser.add_argument(
        "--video-id", type=str, required=True, help="Stem of skeleton/V*.npy (e.g. V1)"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Nominal frame rate (default: 30)")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window for biomechanical/temporal features (default: 5, matches live MVP)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for boxingvi_predictions_*.{json,csv} (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--rules-yaml", type=Path, default=None, help="Override path to risk_rules.yaml"
    )
    parser.add_argument(
        "--min-valid-keypoints",
        type=int,
        default=_DEFAULT_MIN_VALID_KP,
        help=f"Minimum valid joints per frame to keep the frame (default: {_DEFAULT_MIN_VALID_KP})",
    )
    parser.add_argument(
        "--enable-strike-detector",
        action="store_true",
        help="Append heuristic wrist-speed strike candidates (no annotations).",
    )
    parser.add_argument(
        "--strike-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold on non-zero smoothed wrist-speed scores (default: 85)",
    )
    parser.add_argument(
        "--strike-merge-frames",
        type=int,
        default=8,
        help="Merge strike segments when gap (frames) is at most this value (default: 8)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_boxingvi_skeleton_evaluation(
        dataset_root=args.dataset_root.expanduser().resolve(),
        video_id=str(args.video_id).strip(),
        fps=float(args.fps),
        rolling_window=int(args.rolling_window),
        min_valid_keypoints=int(args.min_valid_keypoints),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        rules_yaml=args.rules_yaml.expanduser().resolve() if args.rules_yaml else None,
        enable_strike_detector=bool(args.enable_strike_detector),
        strike_percentile=float(args.strike_percentile),
        strike_merge_frames=int(args.strike_merge_frames),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
