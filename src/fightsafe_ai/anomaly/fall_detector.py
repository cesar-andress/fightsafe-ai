"""
Heuristic **fall / near-ground** cues from 2D pose time series (decision-support only).

**Does not** declare a knockdown, injury, or medical event. Camera tilt, zoom, and cropped frames
can mimic vertical motion; referee confirmation is always required.

This module keeps :func:`fall_likelihood_from_y_coords` for scalar tabular pipelines and adds
:class:`FallDetector` for structured :class:`~fightsafe_ai.anomaly.base.AnomalySignal` outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fightsafe_ai.anomaly.base import AnomalySignal, AnomalyType, BaseTimeSeriesAnomalyDetector
from fightsafe_ai.features.biomechanics import compute_body_centers, compute_torso_angle


def fall_likelihood_from_y_coords(
    head_y: float | None,
    hip_y: float | None,
    ground_y: float = 0.82,
) -> float:
    """
    Return a [0,1] value where high means head/hip is *low* in the frame (near the mat line).

    Uses normalized **y** (increasing downward). If both inputs are missing, returns ``0.0``.

    **Decision-support only** — not a licensed knockdown detector.
    """
    vals: list[float] = []
    for y in (head_y, hip_y):
        if y is None or not math.isfinite(y):
            continue
        vals.append(float(y))
    if not vals:
        return 0.0
    m = max(vals)
    g = float(ground_y)
    if m < g:
        return 0.0
    span = max(1e-6, 1.0 - g)
    return min(1.0, (m - g) / span)


def _mid_hip_y(f: dict[str, tuple[float, float]]) -> float | None:
    lh, rh = f.get("left_hip"), f.get("right_hip")
    if lh is None and rh is None:
        return None
    if lh is None:
        assert rh is not None
        return float(rh[1])
    if rh is None:
        return float(lh[1])
    return float((lh[1] + rh[1]) * 0.5)


def _nose_y(f: dict[str, tuple[float, float]]) -> float | None:
    n = f.get("nose")
    if n is None:
        return None
    return float(n[1])


@dataclass
class FallDetectorConfig:
    """Tuning only; not validated on broadcast or multi-sport data."""

    # downward speed in normalized y / second
    min_descent_vy: float = 0.9
    min_descent_confidence: float = 0.32
    # |delta torso angle| (deg) over one step
    torso_collapse_delta_deg: float = 12.0
    min_torso_collapse_confidence: float = 0.35
    # prolonged: fall_likelihood on mid-hip / head
    ground_y: float = 0.78
    prolonged_likelihood_threshold: float = 0.55
    min_prolonged_frames: int = 4
    min_prolonged_confidence: float = 0.36


class FallDetector(BaseTimeSeriesAnomalyDetector):
    """
    Three independent MVP flags (any may fire from the same short window).

    **Limitations:** 2D only; does not know if the athlete is in a *legal* shoot or slip;
    high ``y`` can be camera panning, not a fall.
    """

    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self.config = config or FallDetectorConfig()

    def analyze(
        self,
        times: list[float],
        frames: list[dict[str, tuple[float, float]]],
        fighter_id: str,
    ) -> list[AnomalySignal]:
        cfg = self.config
        out: list[AnomalySignal] = []
        n = len(frames)
        if n < 1 or n != len(times):
            return out
        t_end = float(times[-1])
        if n < 2:
            return out

        # --- 1) Fast downward head/hip in last step ---
        f0, f1 = frames[-2], frames[-1]
        t0, t1 = float(times[-2]), float(times[-1])
        dt = max(1e-4, t1 - t0)
        hy0, hy1 = _mid_hip_y(f0), _mid_hip_y(f1)
        ny0, ny1 = _nose_y(f0), _nose_y(f1)
        v_hip = 0.0
        v_head = 0.0
        if hy0 is not None and hy1 is not None:
            v_hip = (hy1 - hy0) / dt
        if ny0 is not None and ny1 is not None:
            v_head = (ny1 - ny0) / dt
        v_down = max(v_hip, v_head)  # y increasing downward
        c_desc = 0.0
        if v_down > 0.0:
            c_desc = float(
                np.clip(
                    (v_down - 0.25 * cfg.min_descent_vy) / (cfg.min_descent_vy + 1e-6),
                    0.0,
                    1.0,
                )
            )
        if c_desc >= cfg.min_descent_confidence:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.FALL_RAPID_DOWNWARD,
                    c_desc,
                    {
                        "hip_vy": float(v_hip),
                        "head_vy": float(v_head),
                        "dt": float(dt),
                    },
                )
            )

        # --- 2) Torso angle collapse in last step ---
        def torso_deg(f: dict[str, tuple[float, float]]) -> float:
            c = compute_body_centers(f)
            sxy = (c["shoulder_center_x"], c["shoulder_center_y"])
            hxy = (c["hip_center_x"], c["hip_center_y"])
            if not all(math.isfinite(x) for x in (sxy[0], sxy[1], hxy[0], hxy[1])):
                return float("nan")
            return float(compute_torso_angle(sxy, hxy))

        a0, a1 = torso_deg(f0), torso_deg(f1)
        c_torso = 0.0
        if math.isfinite(a0) and math.isfinite(a1):
            da = abs(a1 - a0)
            if da >= cfg.torso_collapse_delta_deg * 0.4:
                c_torso = float(
                    np.clip(
                        (da - 0.5 * cfg.torso_collapse_delta_deg)
                        / (1.2 * cfg.torso_collapse_delta_deg + 1e-6),
                        0.0,
                        1.0,
                    )
                )
        if c_torso >= cfg.min_torso_collapse_confidence:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.FALL_TORSO_ANGLE_COLLAPSE,
                    c_torso,
                    {
                        "angle_prev_deg": float(a0) if math.isfinite(a0) else -1.0,
                        "angle_cur_deg": float(a1) if math.isfinite(a1) else -1.0,
                    },
                )
            )

        # --- 3) Prolonged low (tail of window) ---
        lks: list[float] = []
        tail = min(n, 12)
        for f in frames[-tail:]:
            h = _nose_y(f)
            hip = _mid_hip_y(f)
            lk = fall_likelihood_from_y_coords(h, hip, ground_y=cfg.ground_y)
            lks.append(lk)
        if len(lks) >= cfg.min_prolonged_frames:
            mean_lk = float(np.mean(lks))
            c_low = 0.0
            if mean_lk >= cfg.prolonged_likelihood_threshold * 0.85:
                c_low = float(
                    np.clip(
                        (mean_lk - 0.35) / 0.55,
                        0.0,
                        1.0,
                    )
                )
            if (
                c_low >= cfg.min_prolonged_confidence
                and int(np.sum(np.array(lks) > 0.2)) >= cfg.min_prolonged_frames
            ):
                out.append(
                    AnomalySignal(
                        t_end,
                        fighter_id,
                        AnomalyType.FALL_PROLONGED_LOW_POSTURE,
                        c_low,
                        {"window_frames": float(len(lks)), "mean_fall_likelihood": float(mean_lk)},
                    )
                )
        return out
