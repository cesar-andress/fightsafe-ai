"""
Surrender / **tap-out**-like motion patterns (vision-only, not a rules engine).

**Limitations (read before use):** see :mod:`fightsafe_ai.risk.surrender` for the core heuristics.
This module adds a thin **AnomalySignal** layer and sub-signal tags; it does not hear the referee
or know local rules. False positives: clinch, parry, corner work. False negatives: off-camera taps.

**Not** a medical or consciousness assessment.
"""

from __future__ import annotations

import numpy as np

from fightsafe_ai.anomaly.base import (
    AnomalySignal,
    AnomalyType,
    BaseTimeSeriesAnomalyDetector,
    pose_sequence_from_landmark_dicts,
)
from fightsafe_ai.risk.surrender import (
    COL_SURRENDER_CONFIDENCE,
    SurrenderDetectionResult,
    SurrenderHeuristicConfig,
    apply_surrender_overrides_to_risk_dataframe,
    detect_surrender,
)


__all__ = [
    "COL_SURRENDER_CONFIDENCE",
    "SurrenderAnomalyDetector",
    "SurrenderDetectionResult",
    "SurrenderHeuristicConfig",
    "apply_surrender_overrides_to_risk_dataframe",
    "detect_surrender",
]


def _wrist_mids(
    frames: list[dict[str, tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    wx = np.full(n, np.nan, dtype=np.float64)
    wy = np.full(n, np.nan, dtype=np.float64)
    for i, f in enumerate(frames):
        got: list[tuple[float, float]] = []
        for k in ("left_wrist", "right_wrist"):
            p = f.get(k)
            if p is not None:
                got.append((p[0], p[1]))
        if not got:
            continue
        wx[i] = float(np.mean([a[0] for a in got]))
        wy[i] = float(np.mean([a[1] for a in got]))
    for arr in (wx, wy):
        for i2 in range(1, n):
            if not np.isfinite(arr[i2]):
                arr[i2] = arr[i2 - 1]
        for i2 in range(n - 2, -1, -1):
            if not np.isfinite(arr[i2]):
                arr[i2] = arr[i2 + 1] if i2 + 1 < n else arr[i2]
    return wx, wy


def _reversal_count_1d(y: np.ndarray) -> int:
    v = np.diff(y, prepend=y[0])
    c = 0
    for i in range(1, v.size):
        if v[i - 1] == 0.0 or v[i] == 0.0:
            continue
        if v[i] * v[i - 1] < 0.0:
            c += 1
    return c


def _rhythmicity_score(wy: np.ndarray) -> float:
    """MVP proxy: reversals + local variation relative to span (not a beat tracker)."""
    if wy.size < 4:
        return 0.0
    s = float(np.std(wy))
    span = float(np.ptp(wy)) + 1e-6
    rev = _reversal_count_1d(wy)
    a = min(1.0, s / 0.06) * min(1.0, float(rev) / 4.0)
    return float(np.clip(a * (1.0 - 0.15 * abs(wy[-1] - wy[0]) / span), 0.0, 1.0))


def _oscillation_strength(wy: np.ndarray, wx: np.ndarray) -> float:
    if wy.size == 0 or wx.size == 0:
        return 0.0
    return float(min(1.0, (float(np.std(wx) + np.std(wy)) / 0.12)))


class SurrenderAnomalyDetector(BaseTimeSeriesAnomalyDetector):
    """
    Emits oscillation / tap-like / rhythmic sub-signals and can mirror aggregate ``detect_surrender``.

    **Opponent proximity** is not observed; *near floor* is taken as high ``y`` in image space.
    """

    def __init__(self, config: SurrenderHeuristicConfig | None = None) -> None:
        self.config = config or SurrenderHeuristicConfig()

    def analyze(
        self,
        times: list[float],
        frames: list[dict[str, tuple[float, float]]],
        fighter_id: str,
    ) -> list[AnomalySignal]:
        if len(frames) < 2 or len(frames) != len(times):
            return []
        cfg = self.config
        t_end = float(times[-1])
        poses = pose_sequence_from_landmark_dicts(frames, id_prefix="a")
        det = detect_surrender(poses, config=cfg)
        wx, wy = _wrist_mids(frames)
        out: list[AnomalySignal] = []
        c_osc = _oscillation_strength(wy, wx)
        c_rhy = _rhythmicity_score(wy) if np.isfinite(wy).all() and wy.size >= 4 else 0.0
        yd = 0.0
        if wy.size >= 2 and np.isfinite(wy[0]) and np.isfinite(wy[-1]):
            yd = max(0.0, float(wy[-1] - wy[0]))
        med_y = float(np.nanmedian(wy)) if np.isfinite(wy).any() else 0.0
        near_floor = 1.0 if med_y > 0.58 else 0.4
        c_tap = float(
            min(
                1.0,
                0.55 * min(1.0, yd / 0.08) + 0.45 * c_osc * near_floor,
            )
        )
        c_floor_osc = float(min(1.0, c_osc * (0.5 + 0.5 * near_floor)))
        th = 0.28
        boost = 0.4 + 0.6 * float(det.confidence)

        if c_osc >= th:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.SURRENDER_OSCILLATION,
                    min(1.0, c_osc * boost),
                    {
                        "oscillation_var_proxy": c_osc,
                        "floor_oscillation": c_floor_osc,
                        "detect_surrender_conf": det.confidence,
                    },
                )
            )
        if c_tap >= th:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.SURRENDER_TAP_LIKE,
                    c_tap,
                    {
                        "wrist_median_y": med_y,
                        "wrist_y_span": yd,
                    },
                )
            )
        if c_rhy >= th and len(frames) >= max(4, cfg.min_frames - 2):
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.SURRENDER_RHYTHMIC_HANDS,
                    c_rhy,
                    {
                        "reversal_count_proxy": float(_reversal_count_1d(wy)) if wy.size else 0.0,
                    },
                )
            )
        if det.surrender_detected and not out:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.SURRENDER_OSCILLATION,
                    min(1.0, det.confidence),
                    {
                        "source": "detect_surrender_aggregate",
                        "conf": det.confidence,
                    },
                )
            )
        return out
