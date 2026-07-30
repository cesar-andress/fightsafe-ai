"""
**Heuristic limb / joint anomaly flags for decision-support (MVP).**

This module does **not** perform clinical assessment, is **not** medically validated, and
must not be used to **diagnose injury** or to replace professional judgment. Thresholds
are ad hoc engineering scalings on 2D pose, not human biomechanics or radiology.

Outputs ``anomaly_score`` in ``[0, 1]`` and ``anomaly_type`` to explain the dominant
**non-clinical** signal for debugging and tuning only. Tier bumps (HIGH/CRITICAL) on high
``anomaly_score`` are in :mod:`fightsafe_ai.risk.limb_tier` after weighted rule scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public column names (per-frame, aligned with ``features_df`` rows)
# ---------------------------------------------------------------------------
COL_ANOMALY_SCORE: str = "anomaly_score"
# ``str`` per row: ``"none"`` or a short token describing the *dominant* heuristic.
COL_ANOMALY_TYPE: str = "anomaly_type"


def _knee_bend_stress_01(deg: np.ndarray) -> np.ndarray:
    """
    Map knee flexion (deg) to [0,1] stress. **Tuned 2D heuristic**, not normative data.

    Small flexion near 0 + very deep flexion both increase the score; missing → 0.
    """
    d = deg.astype(float)
    m = np.isfinite(d)
    out = np.zeros_like(d, dtype=float)
    # "Hyperextension" / over-straight: flexion in [0, 5] (model noise sensitive).
    over_straight = np.clip((5.0 - d) / 5.0, 0.0, 1.0)
    # "Very deep" bend in sagittal proxy (crouch/impact-like): flexion above ~100°.
    deep = np.clip((d - 100.0) / 45.0, 0.0, 1.0)
    tmp = np.maximum(over_straight, deep)
    out = np.where(m, tmp, 0.0)
    return out


def _bilateral_knee_asym_01(deg_l: np.ndarray, deg_r: np.ndarray) -> np.ndarray:
    """
    |L−R| knee flexion asymmetry, scaled. **Not** a clinical varus/valgus test.
    """
    fl = deg_l.astype(float)
    fr = deg_r.astype(float)
    m = np.isfinite(fl) & np.isfinite(fr)
    asy = np.abs(fl - fr)
    scored = np.clip(asy / 40.0, 0.0, 1.0)
    return np.where(m, scored, 0.0)


def _leg_collapse_01(
    deg_l: np.ndarray,
    deg_r: np.ndarray,
    ankle_y: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Sudden one-frame changes in flexion and downward ankle shift (y increases down).

    **MVP only:** a sharp proxy for "one leg buckles" in 2D, not a laboratory drop-jump.
    """
    n = len(deg_l)
    if n < 1:
        return np.zeros(0, dtype=float)
    fl = deg_l.astype(float)
    fr = deg_r.astype(float)
    a = ankle_y.astype(float)
    d_l = np.zeros(n, dtype=float)
    d_r = np.zeros(n, dtype=float)
    d_a = np.zeros(n, dtype=float)
    for k in range(1, n):
        if np.isfinite(fl[k - 1]) and np.isfinite(fl[k]):
            d_l[k] = float(abs(fl[k] - fl[k - 1]))
        if np.isfinite(fr[k - 1]) and np.isfinite(fr[k]):
            d_r[k] = float(abs(fr[k] - fr[k - 1]))
        if np.isfinite(a[k - 1]) and np.isfinite(a[k]):
            d_a[k] = float(abs(a[k] - a[k - 1]))
    jchange = np.maximum(d_l, d_r)
    j_scale = 25.0
    # y increases downward: positive ``d_a * fps`` = downward move per second (normalized y).
    drop = np.zeros(n, dtype=float)
    fp = max(fps, 1e-6)
    t_drop = 0.5  # y-units / sec that maps near 1.0 in ``drop`` (tunable MVP constant).
    for k in range(1, n):
        if np.isfinite(a[k - 1]) and np.isfinite(a[k]) and a[k] >= a[k - 1]:
            drop[k] = float(np.clip((d_a[k] * fp) / t_drop, 0.0, 1.0))
    c = 0.55 * np.clip(jchange / j_scale, 0.0, 1.0) + 0.45 * drop
    return np.clip(c, 0.0, 1.0)


def _anomaly_type_row(
    collapse: float,
    asym: float,
    knee_l: float,
    knee_r: float,
    combined: float,
) -> str:
    if collapse > 0.5:
        return "sudden_leg_collapse"
    if asym > 0.5:
        return "bilateral_leg_asymmetry"
    if max(knee_l, knee_r) > 0.45:
        return "extreme_knee_configuration"
    if combined > 0.2:
        return "combined_limb_stress"
    return "none"


def add_limb_anomaly_columns(
    df: pd.DataFrame,
    fps: float,
    *,
    col_knee_l: str = "knee_flexion_left_deg",
    col_knee_r: str = "knee_flexion_right_deg",
    col_ankle: str = "ankle_y_min",
) -> pd.DataFrame:
    """
    Add ``anomaly_score`` in ``[0,1]`` and ``anomaly_type`` to a feature frame.

    If knee/ankle columns are missing, all scores are 0 and types ``"none"`` (no-op safe path).
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        out[COL_ANOMALY_SCORE] = pd.Series(dtype=float)
        out[COL_ANOMALY_TYPE] = pd.Series(dtype=object)
        return out
    for c in (col_knee_l, col_knee_r, col_ankle):
        if c not in out.columns:
            out[COL_ANOMALY_SCORE] = 0.0
            out[COL_ANOMALY_TYPE] = "none"
            return out

    kl = out[col_knee_l].to_numpy(dtype=float, copy=False)
    kr = out[col_knee_r].to_numpy(dtype=float, copy=False)
    ay = out[col_ankle].to_numpy(dtype=float, copy=False)
    s_kl = _knee_bend_stress_01(kl)
    s_kr = _knee_bend_stress_01(kr)
    s_asy = _bilateral_knee_asym_01(kl, kr)
    s_col = _leg_collapse_01(kl, kr, ay, float(fps))
    w_k = 0.3
    w_a = 0.3
    w_c = 0.4
    combined = np.clip(w_k * np.maximum(s_kl, s_kr) + w_a * s_asy + w_c * s_col, 0.0, 1.0)
    out[COL_ANOMALY_SCORE] = combined
    types: list[str] = []
    for i in range(n):
        types.append(
            _anomaly_type_row(
                float(s_col[i]),
                float(s_asy[i]),
                float(s_kl[i]),
                float(s_kr[i]),
                float(combined[i]),
            )
        )
    out[COL_ANOMALY_TYPE] = types
    return out
