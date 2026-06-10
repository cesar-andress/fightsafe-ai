"""
Inactivity / low-motion heuristics from pose keypoint trajectories (not consciousness or health).

**Limitations:** stillness from clinching, feints, or camera freeze can look like inactivity. The
MVP does **not** use heart rate, eyes, or audio. "After fall" is a *description* of intended use;
this detector does not require a prior :class:`FallDetector` result — it only measures low movement
in the current window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fightsafe_ai.anomaly.base import AnomalySignal, AnomalyType, BaseTimeSeriesAnomalyDetector


def inactivity_score(motion_energies: list[float] | np.ndarray, threshold: float = 0.02) -> float:
    """
    Return a [0,1] inactivity score (higher = less motion) from per-frame non-negative values.

    Uses a simple **mean** compared to ``threshold`` (relative scale, dataset-dependent).
    """
    arr = np.asarray(motion_energies, dtype=np.float64).ravel()
    if arr.size == 0:
        return 1.0
    m = float(np.nanmean(np.clip(arr, 0.0, None)))
    if m >= threshold:
        return 0.0
    return float(np.clip(1.0 - m / threshold, 0.0, 1.0))


@dataclass
class InactivityDetectorConfig:
    """Scale depends on normalized image coordinates; retune for your capture setup."""

    min_keypoint_displacement: float = 0.002
    min_com_displacement: float = 0.0015
    low_motion_inactivity_min: float = 0.45
    min_confidence: float = 0.32
    min_window_frames: int = 3
    #: Require at least this many seconds of history (clip time) before flagging inactivity.
    min_duration_seconds: float = 2.0
    max_window_frames: int = 120


def _com_xy(f: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    ls, rs, lh, rh = (
        f.get("left_shoulder"),
        f.get("right_shoulder"),
        f.get("left_hip"),
        f.get("right_hip"),
    )
    if ls is None or rs is None or lh is None or rh is None:
        return None
    x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0
    y = (ls[1] + rs[1] + lh[1] + rh[1]) / 4.0
    return (float(x), float(y))


def _mean_keypoint_step(
    a: dict[str, tuple[float, float]],
    b: dict[str, tuple[float, float]],
) -> float:
    common = [k for k in a if k in b]
    if not common:
        return 0.0
    d = 0.0
    for k in common:
        p, q = a[k], b[k]
        d += float(np.hypot(q[0] - p[0], q[1] - p[1]))
    return d / float(len(common))


def _com_path_length(frames: list[dict[str, tuple[float, float]]]) -> float:
    if len(frames) < 2:
        return 0.0
    c0 = _com_xy(frames[0])
    s = 0.0
    for i in range(1, len(frames)):
        c1 = _com_xy(frames[i])
        if c0 is None or c1 is None:
            continue
        s += float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
        c0 = c1
    return s


class InactivityDetector(BaseTimeSeriesAnomalyDetector):
    """
    Flags very low per-frame keypoint travel and low COM path length in the *last* window.

    **Not** a "knocked out" or medical assessment.
    """

    def __init__(self, config: InactivityDetectorConfig | None = None) -> None:
        self.config = config or InactivityDetectorConfig()

    def analyze(
        self,
        times: list[float],
        frames: list[dict[str, tuple[float, float]]],
        fighter_id: str,
    ) -> list[AnomalySignal]:
        cfg = self.config
        if len(frames) < cfg.min_window_frames or len(times) != len(frames):
            return []
        t_end = float(times[-1])
        n = len(times)
        start_idx = 0
        for j in range(n - 1, -1, -1):
            span = t_end - float(times[j])
            if span >= float(cfg.min_duration_seconds):
                start_idx = j
                break
            if (n - 1 - j) >= int(cfg.max_window_frames):
                start_idx = j
                break
        w = frames[start_idx:]
        tw = times[start_idx:]
        if len(w) < cfg.min_window_frames:
            return []
        win_dur_pre = max(0.0, float(tw[-1] - tw[0])) if len(tw) > 1 else 0.0
        if win_dur_pre + 1e-9 < float(cfg.min_duration_seconds):
            return []
        steps: list[float] = []
        for i in range(1, len(w)):
            steps.append(_mean_keypoint_step(w[i - 1], w[i]))
        mean_step = float(np.mean(steps)) if steps else 0.0
        path_com = _com_path_length(w)
        win_dur = max(1e-4, float(tw[-1] - tw[0])) if len(tw) > 1 else 1.0
        inact_kp = inactivity_score(steps, threshold=cfg.min_keypoint_displacement)
        c_kp = 0.0
        if (
            mean_step < 1.2 * cfg.min_keypoint_displacement
            and inact_kp >= cfg.low_motion_inactivity_min
        ):
            c_kp = float(
                np.clip(1.0 - mean_step / (3.0 * cfg.min_keypoint_displacement + 1e-8), 0.0, 1.0)
            )
        c_com = 0.0
        com_speed = path_com / win_dur
        if com_speed < 2.5 * cfg.min_com_displacement and path_com < 0.04:
            c_com = float(
                np.clip(1.0 - com_speed / (1.0 * cfg.min_com_displacement + 1e-8), 0.0, 1.0)
            )
        out: list[AnomalySignal] = []
        if c_kp >= cfg.min_confidence:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.INACTIVITY_LOW_KEYPOINT_MOTION,
                    c_kp,
                    {
                        "mean_keypoint_step": float(mean_step),
                        "inactivity_index": float(inact_kp),
                    },
                )
            )
        if c_com >= cfg.min_confidence:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.INACTIVITY_LOW_COM_MOTION,
                    c_com,
                    {
                        "com_path_length": float(path_com),
                        "com_speed_proxy": float(com_speed),
                    },
                )
            )
        return out
