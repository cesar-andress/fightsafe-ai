"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Rule-based risk aggregation over biomechanical feature tables.

Outputs ``risk_score ∈ [0, 1]`` and boolean ``risk_flag`` per frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fightsafe_ai.risk.models import RiskRuleParams


def _consecutive_true_streak(mask: pd.Series) -> pd.Series:
    streak = 0
    out: list[float] = []
    for v in mask.fillna(False).astype(bool):
        if v:
            streak += 1
        else:
            streak = 0
        out.append(float(streak))
    return pd.Series(out, index=mask.index, dtype=float)


def detect_risk_events(
    features: pd.DataFrame,
    params: RiskRuleParams | None = None,
) -> pd.DataFrame:
    """
    Add risk columns to a copy of ``features``.

    Required columns: ``torso_angle_deg``, ``hip_vertical_velocity``,
    ``keypoint_position_variance``, ``near_ground``.
    """
    p = params or RiskRuleParams()
    required = [
        "torso_angle_deg",
        "hip_vertical_velocity",
        "keypoint_position_variance",
        "near_ground",
    ]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = features.copy()
    if len(df) == 0:
        df["risk_from_tilt_velocity"] = pd.Series(dtype=float)
        df["risk_from_ground"] = pd.Series(dtype=float)
        df["risk_from_erratic"] = pd.Series(dtype=float)
        df["risk_score"] = pd.Series(dtype=float)
        df["risk_flag"] = pd.Series(dtype=bool)
        return df

    angle = df["torso_angle_deg"].astype(float).abs()
    vel = df["hip_vertical_velocity"].astype(float).abs()

    both = (angle > p.torso_angle_threshold_deg) & (vel > p.hip_velocity_threshold)
    excess_a = np.maximum(0.0, angle - p.torso_angle_threshold_deg) / max(
        p.tilt_velocity_angle_scale, 1e-6
    )
    excess_v = np.maximum(0.0, vel - p.hip_velocity_threshold) / max(
        p.tilt_velocity_speed_scale, 1e-6
    )
    intensity = np.clip(np.nan_to_num(excess_a * excess_v, nan=0.0, posinf=1.0), 0.0, 1.0)
    r_tilt = np.where(both, np.maximum(intensity, 0.35), 0.0).astype(float)

    near = df["near_ground"].fillna(False).astype(bool)
    streak = _consecutive_true_streak(near)
    progress = np.clip(streak / max(float(p.near_ground_min_frames), 1.0), 0.0, 1.0)
    r_ground = np.where(
        ~near,
        0.0,
        np.where(
            streak > p.near_ground_min_frames,
            1.0,
            (progress**2) * 0.45,
        ),
    )

    var = df["keypoint_position_variance"].astype(float)
    baseline = var.rolling(p.erratic_variance_window, min_periods=2).median()
    ratio = var / (baseline + 1e-9)
    r_var = np.clip(
        (ratio - 1.0) / max(p.erratic_variance_factor - 1.0, 1e-6),
        0.0,
        1.0,
    )
    r_var = np.where(var.notna() & baseline.notna(), r_var, 0.0)

    jerk = vel.diff().abs()
    r_jerk = np.clip(np.nan_to_num(jerk, nan=0.0) / max(p.jerk_threshold, 1e-6), 0.0, 1.0)
    r_err = np.maximum(r_var, r_jerk)

    w_sum = p.weight_tilt_velocity + p.weight_ground + p.weight_erratic
    if w_sum <= 0:
        w_sum = 1.0
    wt = p.weight_tilt_velocity / w_sum
    wg = p.weight_ground / w_sum
    we = p.weight_erratic / w_sum

    risk_score = wt * r_tilt + wg * r_ground + we * r_err
    risk_score = np.clip(np.nan_to_num(risk_score, nan=0.0), 0.0, 1.0)

    df["risk_from_tilt_velocity"] = r_tilt
    df["risk_from_ground"] = r_ground
    df["risk_from_erratic"] = r_err
    df["risk_score"] = risk_score
    df["risk_flag"] = df["risk_score"] >= p.risk_flag_threshold
    return df


class RiskEngine:
    """Thin wrapper to keep a :class:`RiskRuleParams` instance for batch jobs."""

    def __init__(self, params: RiskRuleParams | None = None) -> None:
        self.params = params or RiskRuleParams()

    def run(self, features: pd.DataFrame) -> pd.DataFrame:
        """Delegate to :func:`detect_risk_events`."""
        return detect_risk_events(features, self.params)
