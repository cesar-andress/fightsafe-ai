"""
Prototype **tap-out / surrender gesture** detection (decision-support, not automated officiating).

**Not trained on labeled tap-out data; not validated for broadcast angles or all rule sets.**
This MVP uses only **heuristic, interpretable** signals on 2D wrist (and optional index) motion:

- Fast **oscillation** (repeated hand movement) — high variance in wrist position in a short window
- **Direction reversals** in vertical motion (typical of repeated mat taps or frantic waves)
- Net motion **toward the ground** in image space (y-down), common when tapping
- **Impulse-like** transients in acceleration (simplified "contact" proxy — very noisy in monocular video)

**Limitations (read before trusting outputs):**
- **False positives** — clinching, parries, and corner-work can look like rapid hand motion
- **False negatives** — subtle taps, camera occlusion, and single-frame drops break the sequence
- **2D only** — cannot distinguish a real tap on canvas from a gesture in air at similar image speed
- **No audio / referee** — this is a vision-only prototype; the official stop is always human
- **Rule-set variance** — some orgs do not use a mat tap; the word "surrender" is used generically

Downstream, :func:`apply_surrender_overrides_to_risk_dataframe` can force **CRITICAL** for human review;
it does not replace a referee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fightsafe_ai.pose.keypoints import PoseResult


# Human-readable and stable key for ``triggered_rules`` (not part of ALL_RULE_NAMES weights).
SURRENDER_RULE_KEY: str = "surrender_gesture"

_COL_WRISTS = ("left_wrist", "right_wrist")


@dataclass(frozen=True)
class SurrenderHeuristicConfig:
    """Tunable bounds for the MVP (no learning — edit here or wrap from YAML later)."""

    min_frames: int = 8
    """Sequence shorter than this → no positive detection (insufficient signal)."""
    detect_confidence_threshold: float = 0.52
    weight_oscillation: float = 0.32
    weight_reversals: float = 0.30
    weight_ground_trend: float = 0.20
    weight_jerk_pulses: float = 0.18
    oscillation_var_scale: float = 0.10
    """``std(x)+std(y)`` above this (normalized) contributes fully to the oscillation sub-score."""
    min_velocity_reversals: int = 2
    """At least this many sign changes in vertical velocity → contribute to ``reversals`` term."""
    jerk_pulse_threshold: float = 0.04
    """Magnitude of vertical acceleration (per frame) treated as a sharp impulse."""


@dataclass(frozen=True)
class SurrenderDetectionResult:
    """``detect_surrender`` output; the boolean is ``surrender_detected`` (thresholded confidence)."""

    surrender_detected: bool
    confidence: float


def _wrist_mid_xy(poses: list[PoseResult]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    One (wx, wy) per frame, normalized-image coordinates. Missing frames forward-fill; all-nan → bad.
    """
    n = len(poses)
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0,
        )
    wx = np.full(n, np.nan, dtype=np.float64)
    wy = np.full(n, np.nan, dtype=np.float64)
    raw_hits = 0
    for i, pr in enumerate(poses):
        d: dict[str, tuple[float, float]] = {k.name: (k.x, k.y) for k in pr.keypoints}
        got: list[tuple[float, float]] = []
        for kname in _COL_WRISTS:
            t = d.get(kname)
            if t is not None:
                got.append(t)
        if not got:
            continue
        raw_hits += 1
        xs = [a[0] for a in got]
        ys = [a[1] for a in got]
        wx[i] = float(np.mean(xs))
        wy[i] = float(np.mean(ys))
    for arr in (wx, wy):
        for i2 in range(1, n):
            if not np.isfinite(arr[i2]):
                arr[i2] = arr[i2 - 1]
        for i2 in range(n - 2, -1, -1):
            if not np.isfinite(arr[i2]):
                arr[i2] = arr[i2 + 1] if i2 + 1 < n else arr[i2]
    ok = raw_hits
    return wx, wy, ok


def _count_velocity_zero_crossings(wy: np.ndarray) -> int:
    """Count sign changes in vertical *velocity* (1st diff of y) — many suggests oscillation."""
    v = np.diff(wy, prepend=wy[0])
    m = v.size
    if m < 2:
        return 0
    c = 0
    for i in range(1, m):
        if v[i - 1] == 0.0 or v[i] == 0.0:
            continue
        if v[i] * v[i - 1] < 0.0:
            c += 1
    return c


def detect_surrender(
    pose_sequence: list[PoseResult],
    *,
    config: SurrenderHeuristicConfig | None = None,
) -> SurrenderDetectionResult:
    """
    Run temporal heuristics on a *window* of :class:`PoseResult` (same athlete, time order).

    This is a **Boolean + confidence** wrapper; the primary signal is ``SurrenderDetectionResult``.

    For full-match lists, use rolling windows in :func:`apply_surrender_overrides_to_risk_dataframe`
    rather than a single call on the entire list.

    **Does not** use learning; thresholds are hand-set for development.
    """
    cfg = config or SurrenderHeuristicConfig()
    n = len(pose_sequence)
    if n < cfg.min_frames:
        return SurrenderDetectionResult(surrender_detected=False, confidence=0.0)

    wx, wy, n_ok = _wrist_mid_xy(pose_sequence)
    if n_ok < cfg.min_frames or not (np.isfinite(wx).all() and np.isfinite(wy).all()):
        return SurrenderDetectionResult(surrender_detected=False, confidence=0.0)
    if n_ok < n * 0.4:
        # Too many frames without any wrist keypoint — fill is untrustworthy.
        return SurrenderDetectionResult(surrender_detected=False, confidence=0.0)

    osc = float(np.std(wx) + np.std(wy))
    c_osc = min(1.0, osc / max(cfg.oscillation_var_scale, 1e-9))
    revs = _count_velocity_zero_crossings(wy)
    c_rev = min(1.0, float(revs) / 8.0) if revs >= cfg.min_velocity_reversals else 0.0
    c_down = 0.0
    y_span = float(wy[-1] - wy[0])
    if y_span > 0.02:
        c_down = min(1.0, y_span / 0.12)
    d1 = np.diff(wy, prepend=wy[0])
    d2 = np.diff(d1)  # length n-1; 2nd difference magnitude ~ jerk on vertical motion
    pulses = int(np.sum(np.abs(d2) > float(cfg.jerk_pulse_threshold)))
    c_imp = min(1.0, pulses / 4.0)

    conf = float(
        np.clip(
            cfg.weight_oscillation * c_osc
            + cfg.weight_reversals * c_rev
            + cfg.weight_ground_trend * c_down
            + cfg.weight_jerk_pulses * c_imp,
            0.0,
            1.0,
        )
    )
    det = bool(conf >= float(cfg.detect_confidence_threshold))
    return SurrenderDetectionResult(surrender_detected=det, confidence=conf)


# Column added by :func:`apply_surrender_overrides_to_risk_dataframe`
COL_SURRENDER_CONFIDENCE: str = "surrender_confidence"


def apply_surrender_overrides_to_risk_dataframe(
    out: pd.DataFrame,
    pose_per_frame: list[PoseResult] | None,
    *,
    window_frames: int = 22,
    config: SurrenderHeuristicConfig | None = None,
) -> pd.DataFrame:
    """
    For each row *i*, run :func:`detect_surrender` on ``pose_per_frame[i-window+1 : i+1]``.

    If the window yields ``surrender_detected``:

    - **risk_level** = ``"CRITICAL"`` (overrides the interpretable level)
    - **risk_score** = at least ``1.0`` (after clip)
    - **triggered_rules** = existing list plus :data:`SURRENDER_RULE_KEY`

    **Limitations:** requires ``len(pose_per_frame) == len(out)``; if lengths differ or
    ``pose_per_frame`` is None, the row-level risk is **unchanged** and
    ``surrender_confidence`` is 0.0.
    """
    o = out.copy()
    n = len(o)
    o[COL_SURRENDER_CONFIDENCE] = 0.0
    if pose_per_frame is None or len(pose_per_frame) != n or n == 0:
        return o

    cfg = config or SurrenderHeuristicConfig()
    w = max(1, int(window_frames))
    if "risk_level" not in o.columns:
        o["risk_level"] = np.full(n, "LOW", dtype=object)
    levels: list[object] = o["risk_level"].astype(object).tolist()
    score_arr = (
        o["risk_score"].to_numpy(dtype=float, copy=True) if "risk_score" in o.columns else None
    )
    if score_arr is None or len(score_arr) != n:
        score_arr = np.zeros(n, dtype=float)
    if "triggered_rules" in o.columns:
        trig = [
            list(t) if isinstance(t, (list, tuple)) else [] for t in o["triggered_rules"].tolist()
        ]
    else:
        trig = [[] for _ in range(n)]
    if len(trig) != n:
        trig = [[] for _ in range(n)]

    confs = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - w + 1)
        sub: list[PoseResult] = [pose_per_frame[j] for j in range(lo, i + 1)]
        r = detect_surrender(sub, config=cfg)
        confs[i] = r.confidence
        if r.surrender_detected:
            levels[i] = "CRITICAL"
            score_arr[i] = min(1.0, max(float(score_arr[i]), 1.0))
            row_rules = list(trig[i]) if i < len(trig) else []
            if SURRENDER_RULE_KEY not in row_rules:
                row_rules.append(SURRENDER_RULE_KEY)
            trig[i] = row_rules

    o[COL_SURRENDER_CONFIDENCE] = confs
    o["risk_level"] = pd.Series(levels, dtype=object, index=o.index)
    o["risk_score"] = np.clip(score_arr, 0.0, 1.0)
    o["triggered_rules"] = trig
    return o
