"""
Optional interpretable-rule tweaks for **live / dashboard** sessions only.

``medium`` leaves YAML-loaded parameters unchanged. ``high`` / ``low`` apply bounded
multipliers for UI validation. Batch pipelines and default CLI runs use scientific
defaults from ``configs/risk_rules.yaml`` unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from fightsafe_ai.risk.rules import InterpretableRiskConfig


SensitivityLevel = Literal["low", "medium", "high"]


def apply_interpretable_sensitivity(
    cfg: InterpretableRiskConfig,
    sensitivity: SensitivityLevel | str,
) -> InterpretableRiskConfig:
    """
    Return a shallow-adjusted copy of ``cfg`` for live preview only.

    Parameters
    ----------
    sensitivity
        ``medium`` → identical ``cfg``. ``high`` → easier triggers (lower motion /
        angle thresholds, softer band cutoffs). ``low`` → harder triggers (inverse).
    """
    s = str(sensitivity).strip().lower()
    if s == "medium":
        return cfg

    if s == "high":
        agg = replace(
            cfg.aggregation,
            trigger_epsilon=max(0.02, float(cfg.aggregation.trigger_epsilon) * 0.82),
            level_medium_min=max(0.05, float(cfg.aggregation.level_medium_min) * 0.88),
            level_high_min=max(0.12, float(cfg.aggregation.level_high_min) * 0.88),
            level_critical_min=max(0.28, float(cfg.aggregation.level_critical_min) * 0.92),
        )
        fd = replace(
            cfg.fast_downward,
            velocity_threshold=float(cfg.fast_downward.velocity_threshold) * 0.86,
        )
        lt = replace(
            cfg.large_torso,
            abs_angle_threshold_deg=float(cfg.large_torso.abs_angle_threshold_deg) * 0.90,
        )
        hi = replace(
            cfg.high_instability,
            instability_threshold=float(cfg.high_instability.instability_threshold) * 0.85,
        )
        lp = replace(
            cfg.prolonged_low_posture,
            min_frames_to_score=max(2, int(cfg.prolonged_low_posture.min_frames_to_score) - 1),
        )
        lg = replace(cfg.low_guard, level_threshold=float(cfg.low_guard.level_threshold) * 0.88)
        fa = replace(cfg.facing_away, level_threshold=float(cfg.facing_away.level_threshold) * 0.88)
        istrike = replace(
            cfg.incoming_strike,
            level_threshold=float(cfg.incoming_strike.level_threshold) * 0.88,
        )
        return replace(
            cfg,
            aggregation=agg,
            fast_downward=fd,
            large_torso=lt,
            high_instability=hi,
            prolonged_low_posture=lp,
            low_guard=lg,
            facing_away=fa,
            incoming_strike=istrike,
        )

    if s == "low":
        agg = replace(
            cfg.aggregation,
            trigger_epsilon=min(0.25, float(cfg.aggregation.trigger_epsilon) * 1.18),
            level_medium_min=min(0.45, float(cfg.aggregation.level_medium_min) * 1.12),
            level_high_min=min(0.65, float(cfg.aggregation.level_high_min) * 1.10),
            level_critical_min=min(0.92, float(cfg.aggregation.level_critical_min) * 1.06),
        )
        fd = replace(
            cfg.fast_downward,
            velocity_threshold=float(cfg.fast_downward.velocity_threshold) * 1.18,
        )
        lt = replace(
            cfg.large_torso,
            abs_angle_threshold_deg=float(cfg.large_torso.abs_angle_threshold_deg) * 1.12,
        )
        hi = replace(
            cfg.high_instability,
            instability_threshold=float(cfg.high_instability.instability_threshold) * 1.22,
        )
        lp = replace(
            cfg.prolonged_low_posture,
            min_frames_to_score=int(cfg.prolonged_low_posture.min_frames_to_score) + 2,
        )
        lg = replace(
            cfg.low_guard,
            level_threshold=min(0.55, float(cfg.low_guard.level_threshold) * 1.15),
        )
        fa = replace(
            cfg.facing_away,
            level_threshold=min(0.55, float(cfg.facing_away.level_threshold) * 1.15),
        )
        istrike = replace(
            cfg.incoming_strike,
            level_threshold=min(0.55, float(cfg.incoming_strike.level_threshold) * 1.15),
        )
        return replace(
            cfg,
            aggregation=agg,
            fast_downward=fd,
            large_torso=lt,
            high_instability=hi,
            prolonged_low_posture=lp,
            low_guard=lg,
            facing_away=fa,
            incoming_strike=istrike,
        )

    raise ValueError(f"Unknown sensitivity level: {sensitivity!r}")


__all__ = ["SensitivityLevel", "apply_interpretable_sensitivity"]
