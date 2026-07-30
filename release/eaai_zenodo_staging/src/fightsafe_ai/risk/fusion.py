"""
Fusion of biomechanical, action, anomaly, surrender, and inactivity cues into a single
:class:`RiskDecision` (interpretable, not a clinical assessment).

**Does not** replace the weighted interpretable :mod:`fightsafe_ai.risk.rules` pipeline; this
module is a **composable** rule-governed max-level fusion for experiments, HCI, and downstream
orchestrators. Keep :func:`fuse_weighted_mean` for ablations and scalar blend studies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fightsafe_ai.action.base import ActionType
from fightsafe_ai.anomaly.base import AnomalyType
from fightsafe_ai.risk.levels import (
    RISK_LEVEL_ORDER,
    RiskLevelName,
    max_risk_level,
    risk_level_rank,
)


# ---------------------------------------------------------------------------
# Weighted mean (legacy / ablations)
# ---------------------------------------------------------------------------


def fuse_weighted_mean(
    components: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Weighted mean in [0,1] if all components and weights are non-negative.

    Keys must align; missing keys in ``weights`` are treated as 0. If the total weight is 0,
    returns 0.0.
    """
    wsum = 0.0
    acc = 0.0
    for k, v in components.items():
        w = float(weights.get(k, 0.0))
        if w <= 0.0:
            continue
        x = max(0.0, min(1.0, float(v)))
        acc += x * w
        wsum += w
    if wsum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, acc / wsum))


# ---------------------------------------------------------------------------
# Input bundle (per fighter, per instant)
# ---------------------------------------------------------------------------

_FALL_ANY: frozenset[AnomalyType] = frozenset(
    {
        AnomalyType.FALL_RAPID_DOWNWARD,
        AnomalyType.FALL_TORSO_ANGLE_COLLAPSE,
        AnomalyType.FALL_PROLONGED_LOW_POSTURE,
    }
)
_INACTIVITY_TYPES: frozenset[AnomalyType] = frozenset(
    {
        AnomalyType.INACTIVITY_LOW_KEYPOINT_MOTION,
        AnomalyType.INACTIVITY_LOW_COM_MOTION,
    }
)
_LIMB_TYPES: frozenset[AnomalyType] = frozenset(
    {
        AnomalyType.LIMB_JOINT_ANGULAR_ANOMALY,
        AnomalyType.LIMB_BILATERAL_ASYMMETRY,
        AnomalyType.LIMB_SUDDEN_SUPPORT_LOSS,
    }
)
_SURRENDER_LIKE: frozenset[AnomalyType] = frozenset(
    {
        AnomalyType.SURRENDER_OSCILLATION,
        AnomalyType.SURRENDER_TAP_LIKE,
        AnomalyType.SURRENDER_RHYTHMIC_HANDS,
    }
)


@dataclass(frozen=True, slots=True)
class RiskFusionInput:
    """
    Aggregated *maximum* (or last-in-window) confidences. Callers are responsible for
    time alignment and fighter identity; this struct is pure for testing.
    """

    timestamp: float
    fighter_id: str
    # max confidence in [0,1] per :class:`ActionType` in the consider window
    action_conf: dict[ActionType, float] = field(default_factory=dict)
    # max confidence per :class:`AnomalyType` in the window
    anomaly_conf: dict[AnomalyType, float] = field(default_factory=dict)
    surrender_detected: bool = False
    surrender_confidence: float = 0.0
    # biomechanical: unstable stance / rebalancing proxy [0,1] (e.g. from feature pipeline)
    instability_score: float = 0.0
    # optional: hip vertical speed (y-down coords; **positive** = moving down) for fall rule
    hip_vertical_velocity: float | None = None
    # aggregate inactivity in [0,1] (from ``inactivity_score`` helper or hand-built)
    inactivity_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Fused, human-readable result for a single (fighter, time) key."""

    timestamp: float
    fighter_id: str
    risk_score: float
    risk_level: RiskLevelName
    triggered_signals: tuple[str, ...] = ()
    explanation_facts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0,1]")


# ---------------------------------------------------------------------------
# Thresholds (documented; tunable)
# ---------------------------------------------------------------------------

_TH_LOW_GUARD: float = 0.3
_TH_TURNED_BACK: float = 0.3
_TH_INSTAB_MED: float = 0.4
_TH_INSTAB_HIGH: float = 0.66
_TH_FALL_INACT_AN: float = 0.3
_TH_INACT_AGG: float = 0.32
_TH_SURR: float = 0.48
_TH_SURRENDER_ANOM: float = 0.46
_TH_FALL_RAPID_V: float = 0.38
_TH_HIP_V: float = 0.48
_TH_LIMB_HIGH: float = 0.52
_TH_LIMB_CRIT: float = 0.78
_TH_FALL_RAPID_MED: float = 0.36
_FANY: float = 0.16

_CONF_EPS: float = 1e-4


def _fmax(m: dict[Any, float], keys: frozenset[Any]) -> float:
    if not m:
        return 0.0
    return max(0.0, max((float(m.get(k, 0.0) or 0.0) for k in keys), default=0.0))


def _anomaly_max_fall(fi: RiskFusionInput) -> float:
    return _fmax(fi.anomaly_conf, _FALL_ANY)


def _any_fall(fi: RiskFusionInput) -> bool:
    return any(float(fi.anomaly_conf.get(t, 0.0) or 0.0) > _FANY for t in _FALL_ANY)


def _inactivity_strength(fi: RiskFusionInput) -> float:
    am = 0.0
    for t in _INACTIVITY_TYPES:
        am = max(am, float(fi.anomaly_conf.get(t, 0.0) or 0.0))
    ia = max(0.0, min(1.0, float(fi.inactivity_score)))
    return max(am, ia)


def _limb_max(fi: RiskFusionInput) -> float:
    return _fmax(fi.anomaly_conf, _LIMB_TYPES)


def _surrender_like_max(fi: RiskFusionInput) -> float:
    return _fmax(fi.anomaly_conf, _SURRENDER_LIKE)


def _score_from_level(level: RiskLevelName, strength: float) -> float:
    s = max(0.0, min(1.0, float(strength)))
    base: dict[RiskLevelName, float] = {
        RiskLevelName.LOW: 0.1,
        RiskLevelName.MEDIUM: 0.4,
        RiskLevelName.HIGH: 0.68,
        RiskLevelName.CRITICAL: 0.86,
    }
    j = 0.08 * (1.0 + s) * (1 + risk_level_rank(level))
    return min(1.0, max(0.0, base.get(level, 0.0) + j))


def compute_risk_decision(fi: RiskFusionInput) -> RiskDecision:
    """
    Apply max-severity rule fusion. Order is chosen so the **documented** priorities hold;
    the returned level is the max across **all** applicable rule bands.

    See module docstring for human-readable rule names (also in ``triggered_signals``).
    """
    level = RiskLevelName.LOW
    triggers: list[str] = []
    facts: list[str] = []
    s_for_score = 0.0

    surr = bool(fi.surrender_detected) and float(fi.surrender_confidence) + _CONF_EPS >= _TH_SURR
    surr_an = _surrender_like_max(fi) + _CONF_EPS >= _TH_SURRENDER_ANOM
    if surr or surr_an:
        level = RiskLevelName.CRITICAL
        s_for_score = max(s_for_score, float(fi.surrender_confidence), _surrender_like_max(fi))
        if surr:
            triggers.append("surrender_detected_critical")
            facts.append("Surrender or tap-out flag/gesture (vision-only; confirm with referee).")
        elif surr_an:
            triggers.append("surrender_like_vision_critical")
            facts.append("Surrender-like hand motion in vision; not audio- or rules-complete.")

    fall = _any_fall(fi)
    inact_s = _inactivity_strength(fi)
    inact_hit = inact_s + _CONF_EPS >= _TH_INACT_AGG or any(
        float(fi.anomaly_conf.get(t, 0.0) or 0.0) + _CONF_EPS >= _TH_FALL_INACT_AN
        for t in _INACTIVITY_TYPES
    )
    if fall and inact_hit:
        level = max_risk_level(level, RiskLevelName.CRITICAL)
        s_for_score = max(s_for_score, min(1.0, 0.5 * inact_s + 0.5 * _anomaly_max_fall(fi)))
        triggers.append("fall_and_inactivity_critical")
        facts.append(
            "Fall/near-ground cue with very low movement proxy -> CRITICAL review (not a KO claim)."
        )

    limb = _limb_max(fi)
    if limb + _CONF_EPS >= _TH_LIMB_CRIT:
        level = max_risk_level(level, RiskLevelName.CRITICAL)
        s_for_score = max(s_for_score, limb)
        triggers.append("limb_anomaly_critical")
        facts.append(
            "Limb/joint configuration anomaly (2D) at CRITICAL band; not clinical goniometry."
        )

    fr = float(fi.anomaly_conf.get(AnomalyType.FALL_RAPID_DOWNWARD, 0.0) or 0.0)
    hv = 0.0
    if fi.hip_vertical_velocity is not None:
        hv = max(0.0, float(fi.hip_vertical_velocity))
    high_down = (
        (fr + _CONF_EPS >= _TH_FALL_RAPID_MED and hv + _CONF_EPS >= _TH_HIP_V)
        or (fr + _CONF_EPS >= _TH_FALL_RAPID_V)
        or (hv + _CONF_EPS >= _TH_HIP_V and fall)
    )
    if fall and high_down and level is not RiskLevelName.CRITICAL:
        level = max_risk_level(level, RiskLevelName.HIGH)
        s_for_score = max(s_for_score, fr, min(1.0, 0.35 * hv + 0.3 * fr))
        triggers.append("fall_high_downward_velocity_high")
        facts.append(
            "Rapid fall / downward head-hip motion (image space) -> HIGH (camera-dependent)."
        )

    if _TH_LIMB_HIGH - _CONF_EPS <= limb < _TH_LIMB_CRIT - _CONF_EPS and risk_level_rank(
        level
    ) < risk_level_rank(RiskLevelName.CRITICAL):
        level = max_risk_level(level, RiskLevelName.HIGH)
        s_for_score = max(s_for_score, limb)
        if "limb_anomaly_elevated_high" not in triggers:
            triggers.append("limb_anomaly_elevated_high")
        facts.append("Limb/joint proxy in HIGH band (asymmetry or support-loss possible).")

    inst = max(0.0, min(1.0, float(fi.instability_score)))
    tb = float(fi.action_conf.get(ActionType.TURNED_BACK, 0.0) or 0.0)
    if tb + _CONF_EPS >= _TH_TURNED_BACK and inst + _CONF_EPS >= _TH_INSTAB_MED:
        r_tb = RiskLevelName.HIGH if inst + _CONF_EPS >= _TH_INSTAB_HIGH else RiskLevelName.MEDIUM
        level = max_risk_level(level, r_tb)
        s_for_score = max(s_for_score, 0.5 * (tb + inst))
        triggers.append("turned_back_instability_medium_or_high")
        facts.append("Turning away proxy plus instability score (not balance lab data).")

    if (
        float(fi.action_conf.get(ActionType.LOW_GUARD, 0.0) or 0.0) + _CONF_EPS >= _TH_LOW_GUARD
        and level is RiskLevelName.LOW
    ):
        level = max_risk_level(level, RiskLevelName.MEDIUM)
        s_for_score = max(
            s_for_score,
            float(fi.action_conf.get(ActionType.LOW_GUARD, 0.0) or 0.0) * 0.7,
        )
        triggers.append("low_guard_elevated_medium")
        facts.append("Hands low vs head in 2D; raise floor to MEDIUM (rule-only).")

    if s_for_score <= 0.0:
        s_for_score = max(
            0.12,
            float(fi.action_conf.get(ActionType.LOW_GUARD, 0.0) or 0.0) * 0.4,
            limb * 0.2,
            _anomaly_max_fall(fi) * 0.3,
        )
    score = _score_from_level(level, s_for_score)
    ex = (
        facts
        if facts
        else (
            f"Tier {level.value} on the {len(RISK_LEVEL_ORDER)}-band scale (no specific fusion facts).",
        )
    )
    return RiskDecision(
        float(fi.timestamp),
        str(fi.fighter_id),
        min(1.0, max(0.0, score)),
        level,
        tuple(triggers),
        tuple(ex),
    )
