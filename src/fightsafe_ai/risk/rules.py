"""
Transparent, rule-based heuristics for a **non-clinical** combat-sports **decision-support** score.

**Not a medical or diagnostic product:** hand-tuned thresholds and simple signal shapes.
It does **not** detect injury, concussion, or any medical condition. Outputs support
human review (officials, researchers, coaches) and engineering visualization only;
they do not replace professional judgment or medical protocols.

For tuning, edit ``configs/risk_rules.yaml`` and reload; avoid embedding thresholds in code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import yaml

from fightsafe_ai.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Public rule keys (used in ``triggered_rules`` and in logs)
# ---------------------------------------------------------------------------

RULE_FAST_DOWNWARD: Final[str] = "fast_downward_motion"
RULE_LARGE_TORSO: Final[str] = "large_torso_angle"
RULE_LOW_POSTURE: Final[str] = "prolonged_low_posture"
RULE_INSTABILITY: Final[str] = "high_instability"
RULE_POST_FALL: Final[str] = "post_fall_low_movement"
# Atomic signals (features: guard_level, facing_away_score, reaction_delay_score)
RULE_LOW_GUARD: Final[str] = "low_guard"
RULE_FACING_AWAY: Final[str] = "facing_away"
RULE_REACTION_DELAY: Final[str] = "reaction_delay_after_impact"
# Composites (AND-style; explain multi-level alert semantics in YAML)
RULE_LOSS_OF_CONTROL: Final[str] = "loss_of_control"
RULE_CLEAR_DANGER_FALL: Final[str] = "clear_danger_fall"
RULE_INTERVENTION_URGENT: Final[str] = "intervention_urgent"
# Composite: low_guard rule component * mapped strike_incoming_proxy (guard low + fast limb inbound).
RULE_HIGH_RISK_GUARD_STRIKE: Final[str] = "high_risk_guard_strike"
# Heuristic 2D limb/joint configuration anomaly (``anomaly_score``; not clinical).
RULE_LIMB_ANOMALY: Final[str] = "limb_anomaly"

ALL_RULE_NAMES: tuple[str, ...] = (
    RULE_FAST_DOWNWARD,
    RULE_LARGE_TORSO,
    RULE_LOW_POSTURE,
    RULE_INSTABILITY,
    RULE_POST_FALL,
    RULE_LOW_GUARD,
    RULE_FACING_AWAY,
    RULE_REACTION_DELAY,
    RULE_LOSS_OF_CONTROL,
    RULE_CLEAR_DANGER_FALL,
    RULE_INTERVENTION_URGENT,
    RULE_HIGH_RISK_GUARD_STRIKE,
    RULE_LIMB_ANOMALY,
)

#: Maps YAML / ``triggered_rules`` keys to human-facing indicator names (decision-support only).
COMBAT_MVP_INDICATOR_LABELS: dict[str, str] = {
    RULE_FAST_DOWNWARD: "Fast downward hip movement (fall / strong imbalance proxy)",
    RULE_LARGE_TORSO: "Large torso angle (vs. vertical)",
    RULE_LOW_POSTURE: "Prolonged low posture",
    RULE_POST_FALL: "Low movement after sustained near-ground contact",
    RULE_INSTABILITY: "High joint/angle instability (rolling window)",
    RULE_LOW_GUARD: "Low guard (hands low relative to head)",
    RULE_FACING_AWAY: "Body turned away from opponent (lateral head–shoulder offset)",
    RULE_REACTION_DELAY: "Stillness after sudden deceleration (reaction delay proxy)",
    RULE_LOSS_OF_CONTROL: "Loss of control: (low guard or facing away) with instability (MEDIUM-tier)",
    RULE_CLEAR_DANGER_FALL: "Fall-like cue: large torso angle AND fast downward motion (HIGH-tier proxy)",
    RULE_INTERVENTION_URGENT: "Urgent: near-ground stillness with prolonged low posture (CRITICAL-tier)",
    RULE_HIGH_RISK_GUARD_STRIKE: "High risk: low guard with incoming strike proxy (fast inbound limbs)",
    RULE_LIMB_ANOMALY: "Limb / knee proxy anomaly (heuristic 0–1; not a clinical exam)",
}

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class FastDownwardConfig:
    """Hip moving fast downward (positive ``hip_vertical_velocity`` in image coords)."""

    velocity_threshold: float = 0.45
    velocity_scale: float = 0.9


@dataclass(frozen=True)
class LargeTorsoConfig:
    """Large absolute torso angle vs. vertical (degrees)."""

    abs_angle_threshold_deg: float = 28.0
    excess_scale_deg: float = 40.0


@dataclass(frozen=True)
class ProlongedLowPostureConfig:
    """Use rolling low-posture frame count (e.g. from temporal features)."""

    min_frames_to_score: int = 4
    full_risk_at_frames: int = 30


@dataclass(frozen=True)
class HighInstabilityConfig:
    """Rolling instability metric (e.g. ``instability_score`` from temporal features)."""

    instability_threshold: float = 0.04
    excess_scale: float = 0.10


@dataclass(frozen=True)
class PostFallLowMovementConfig:
    """
    After sustained near-ground contact, very low combined speed increases concern.

    This is a **rough** proxy for “on the ground and barely moving” — not a diagnosis
    of unconsciousness or injury.
    """

    min_near_ground_streak_frames: int = 8
    max_combined_speed: float = 0.15
    speed_buffer: float = 0.05


@dataclass(frozen=True)
class Scaled01SignalConfig:
    """Map a 0–1 feature column to a 0–1 rule component (threshold + scale above it)."""

    level_threshold: float = 0.2
    excess_scale: float = 0.35


@dataclass(frozen=True)
class InterpretableAggregationConfig:
    """Weights and level cutoffs for the MVP score (multi-level alert aggregation)."""

    # 12 non-composite/atomic + 3 composite weights originally summed to 1.0; ``limb_anomaly`` added
    # at 0.08 and the rest scaled by 0.92 (see project ``configs/risk_rules.yaml``).
    weight_fast_downward: float = 0.0828
    weight_large_torso: float = 0.0828
    weight_low_posture: float = 0.0828
    weight_instability: float = 0.0828
    weight_post_fall: float = 0.0828
    weight_low_guard: float = 0.0736
    weight_facing_away: float = 0.0736
    weight_reaction_delay: float = 0.0736
    weight_loss_of_control: float = 0.092
    weight_clear_danger_fall: float = 0.092
    weight_intervention_urgent: float = 0.1012
    # Default 0; set in ``configs/risk_rules.yaml`` (rule inactive until nonzero — weights renormalize).
    weight_high_risk_guard_strike: float = 0.0
    weight_limb_anomaly: float = 0.08
    trigger_epsilon: float = 0.08
    level_medium_min: float = 0.25
    level_high_min: float = 0.5
    level_critical_min: float = 0.75


@dataclass(frozen=True)
class InterpretableRiskConfig:
    """Full interpretable ruleset (subset of ``configs/risk_rules.yaml``)."""

    fast_downward: FastDownwardConfig
    large_torso: LargeTorsoConfig
    prolonged_low_posture: ProlongedLowPostureConfig
    high_instability: HighInstabilityConfig
    post_fall: PostFallLowMovementConfig
    low_guard: Scaled01SignalConfig
    facing_away: Scaled01SignalConfig
    reaction_delay_signal: Scaled01SignalConfig
    limb_anomaly: Scaled01SignalConfig
    incoming_strike: Scaled01SignalConfig
    aggregation: InterpretableAggregationConfig

    @staticmethod
    def default() -> InterpretableRiskConfig:
        return load_interpretable_risk_config()


def _as_float(m: Mapping[str, Any], key: str, default: float) -> float:
    v = m.get(key, default)
    if v is None:
        return float(default)
    return float(v)


def _as_int(m: Mapping[str, Any], key: str, default: int) -> int:
    v = m.get(key, default)
    if v is None:
        return int(default)
    return int(v)


def load_interpretable_risk_config(path: Path | None = None) -> InterpretableRiskConfig:
    """
    Load interpretable MVP parameters from ``configs/risk_rules.yaml`` (or an override path).

    Unknown keys are ignored. Missing files raise :class:`ConfigurationError` when
    ``path`` is given; if ``path`` is ``None``, the default factory values are used
    (callers may pass a project path explicitly).
    """
    if path is None:
        return InterpretableRiskConfig(
            fast_downward=FastDownwardConfig(),
            large_torso=LargeTorsoConfig(),
            prolonged_low_posture=ProlongedLowPostureConfig(),
            high_instability=HighInstabilityConfig(),
            post_fall=PostFallLowMovementConfig(),
            low_guard=Scaled01SignalConfig(),
            facing_away=Scaled01SignalConfig(),
            reaction_delay_signal=Scaled01SignalConfig(),
            limb_anomaly=Scaled01SignalConfig(),
            incoming_strike=Scaled01SignalConfig(level_threshold=0.15, excess_scale=0.45),
            aggregation=InterpretableAggregationConfig(),
        )
    if not path.is_file():
        raise ConfigurationError(f"Risk rules file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None or not isinstance(raw, Mapping):
        return load_interpretable_risk_config(None)

    fd = raw.get("fast_downward_motion", {}) or {}
    lt = raw.get("large_torso_angle", {}) or {}
    lp = raw.get("prolonged_low_posture", {}) or {}
    hi = raw.get("high_instability", {}) or {}
    pf = raw.get("post_fall_low_movement", {}) or {}
    lg = raw.get("low_guard", {}) or {}
    fa = raw.get("facing_away", {}) or {}
    rd = raw.get("reaction_delay_after_impact", {}) or {}
    la = raw.get("limb_anomaly", {}) or {}
    inc_strike = raw.get("incoming_strike", {}) or {}
    agg = raw.get("interpretable_aggregation", {}) or {}

    return InterpretableRiskConfig(
        fast_downward=FastDownwardConfig(
            velocity_threshold=_as_float(fd, "velocity_threshold", 0.45),
            velocity_scale=_as_float(fd, "velocity_scale", 0.9),
        ),
        large_torso=LargeTorsoConfig(
            abs_angle_threshold_deg=_as_float(lt, "abs_angle_threshold_deg", 28.0),
            excess_scale_deg=_as_float(lt, "excess_scale_deg", 40.0),
        ),
        prolonged_low_posture=ProlongedLowPostureConfig(
            min_frames_to_score=_as_int(lp, "min_frames_to_score", 4),
            full_risk_at_frames=_as_int(lp, "full_risk_at_frames", 30),
        ),
        high_instability=HighInstabilityConfig(
            instability_threshold=_as_float(hi, "instability_threshold", 0.04),
            excess_scale=_as_float(hi, "excess_scale", 0.10),
        ),
        post_fall=PostFallLowMovementConfig(
            min_near_ground_streak_frames=_as_int(pf, "min_near_ground_streak_frames", 8),
            max_combined_speed=_as_float(pf, "max_combined_speed", 0.15),
            speed_buffer=_as_float(pf, "speed_buffer", 0.05),
        ),
        low_guard=Scaled01SignalConfig(
            level_threshold=_as_float(lg, "level_threshold", 0.2),
            excess_scale=_as_float(lg, "excess_scale", 0.35),
        ),
        facing_away=Scaled01SignalConfig(
            level_threshold=_as_float(fa, "level_threshold", 0.2),
            excess_scale=_as_float(fa, "excess_scale", 0.35),
        ),
        reaction_delay_signal=Scaled01SignalConfig(
            level_threshold=_as_float(rd, "level_threshold", 0.2),
            excess_scale=_as_float(rd, "excess_scale", 0.35),
        ),
        limb_anomaly=Scaled01SignalConfig(
            level_threshold=_as_float(la, "level_threshold", 0.2),
            excess_scale=_as_float(la, "excess_scale", 0.35),
        ),
        incoming_strike=Scaled01SignalConfig(
            level_threshold=_as_float(inc_strike, "level_threshold", 0.15),
            excess_scale=_as_float(inc_strike, "excess_scale", 0.45),
        ),
        aggregation=InterpretableAggregationConfig(
            weight_fast_downward=_as_float(agg, "weight_fast_downward", 0.0828),
            weight_large_torso=_as_float(agg, "weight_large_torso", 0.0828),
            weight_low_posture=_as_float(agg, "weight_low_posture", 0.0828),
            weight_instability=_as_float(agg, "weight_instability", 0.0828),
            weight_post_fall=_as_float(agg, "weight_post_fall", 0.0828),
            weight_low_guard=_as_float(agg, "weight_low_guard", 0.0736),
            weight_facing_away=_as_float(agg, "weight_facing_away", 0.0736),
            weight_reaction_delay=_as_float(agg, "weight_reaction_delay", 0.0736),
            weight_loss_of_control=_as_float(agg, "weight_loss_of_control", 0.092),
            weight_clear_danger_fall=_as_float(agg, "weight_clear_danger_fall", 0.092),
            weight_intervention_urgent=_as_float(agg, "weight_intervention_urgent", 0.1012),
            weight_high_risk_guard_strike=_as_float(agg, "weight_high_risk_guard_strike", 0.0),
            weight_limb_anomaly=_as_float(agg, "weight_limb_anomaly", 0.08),
            trigger_epsilon=_as_float(agg, "trigger_epsilon", 0.08),
            level_medium_min=_as_float(agg, "level_medium_min", 0.25),
            level_high_min=_as_float(agg, "level_high_min", 0.5),
            level_critical_min=_as_float(agg, "level_critical_min", 0.75),
        ),
    )


def _consecutive_true_streak(mask: pd.Series) -> np.ndarray:
    streak = 0
    out: list[float] = []
    for v in mask.fillna(False).astype(bool):
        if v:
            streak += 1
        else:
            streak = 0
        out.append(float(streak))
    return np.asarray(out, dtype=float)


def _clip01(x: np.ndarray) -> np.ndarray:
    clipped: np.ndarray = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return clipped


def component_fast_downward(
    hip_vertical_velocity: np.ndarray,
    cfg: FastDownwardConfig,
) -> np.ndarray:
    # Downward in image = positive hip_vertical_velocity (see biomechanics / temporal).
    v = np.maximum(0.0, hip_vertical_velocity)
    scale = max(cfg.velocity_scale, 1e-9)
    thr = max(cfg.velocity_threshold, 0.0)
    excess = np.maximum(0.0, v - thr) / scale
    return _clip01(excess)


def component_large_torso(
    torso_angle_deg: np.ndarray,
    cfg: LargeTorsoConfig,
) -> np.ndarray:
    a = np.abs(torso_angle_deg)
    scale = max(cfg.excess_scale_deg, 1e-9)
    thr = max(0.0, cfg.abs_angle_threshold_deg)
    excess = np.maximum(0.0, a - thr) / scale
    return _clip01(excess)


def component_prolonged_low_posture(
    low_posture_duration_frames: np.ndarray,
    cfg: ProlongedLowPostureConfig,
) -> np.ndarray:
    d = low_posture_duration_frames.astype(float)
    d0 = max(float(cfg.min_frames_to_score), 0.0)
    d1 = max(float(cfg.full_risk_at_frames), d0 + 1e-6)
    span = d1 - d0
    scored = np.maximum(0.0, d - d0) / span
    m = d >= d0
    return np.where(m, _clip01(scored), 0.0)


def component_high_instability(
    instability_score: np.ndarray,
    cfg: HighInstabilityConfig,
) -> np.ndarray:
    s = instability_score.astype(float)
    scale = max(cfg.excess_scale, 1e-9)
    thr = max(0.0, cfg.instability_threshold)
    excess = np.maximum(0.0, s - thr) / scale
    return _clip01(excess)


def component_post_fall_low_movement(
    near_ground: np.ndarray,
    hip_vel: np.ndarray,
    head_vel: np.ndarray,
    cfg: PostFallLowMovementConfig,
) -> np.ndarray:
    """
    When ``near_ground`` has been true for ``min_near_ground_streak_frames``,
    map low combined speed to high component (0–1).
    """
    mask = near_ground.astype(bool)
    streak = _consecutive_true_streak(pd.Series(mask))
    on_long_enough = streak >= float(cfg.min_near_ground_streak_frames)
    speed = np.abs(hip_vel) + np.abs(head_vel)
    # Band: full concern at speed 0, zero when speed >= max_combined + buffer
    smax = max(float(cfg.max_combined_speed), 1e-9)
    buf = max(float(cfg.speed_buffer), 0.0)
    upper = smax + buf
    # 1 at 0, 0 at >= upper, linear in between
    t = 1.0 - np.clip(speed / upper, 0.0, 1.0)
    return np.where(on_long_enough & mask, _clip01(t), 0.0)


def component_scaled_01_signal(
    x: np.ndarray,
    cfg: Scaled01SignalConfig,
) -> np.ndarray:
    """
    Map a 0–1 *feature* (guard / facing / reaction) to a 0–1 rule response above a neutral band.
    """
    t = max(0.0, float(cfg.level_threshold))
    s = max(float(cfg.excess_scale), 1e-9)
    v = x.astype(float)
    excess = np.maximum(0.0, v - t) / s
    return _clip01(excess)


def component_loss_of_control(
    c_guard: np.ndarray,
    c_facing: np.ndarray,
    c_inst: np.ndarray,
) -> np.ndarray:
    """
    MEDIUM-oriented composite: (low **or** turned away) **and** instability, using active components.
    ``sqrt`` softens the AND so both need to be meaningfully on.
    """
    mx = np.maximum(c_guard, c_facing)
    return _clip01(np.sqrt(np.clip(mx * c_inst, 0.0, 1.0)))


def component_clear_danger_fall(c_torso: np.ndarray, c_fast: np.ndarray) -> np.ndarray:
    """HIGH-tier fall proxy: large torso angle **and** fast downward motion (policy AND)."""
    return _clip01(c_torso * c_fast)


def component_high_risk_guard_strike(c_low_guard: np.ndarray, c_strike: np.ndarray) -> np.ndarray:
    """HIGH-oriented: exposed guard (mapped) **and** inbound strike proxy."""
    return _clip01(c_low_guard * c_strike)


def component_intervention_urgent(c_post: np.ndarray, c_lowp: np.ndarray) -> np.ndarray:
    """
    CRITICAL-oriented composite: post-fall stillness **and** sustained low posture (referee
    “stop the fight / check fighter” **proxy** only).
    """
    return _clip01(c_post * c_lowp)


def _normalize_level_cutoffs(agg: InterpretableAggregationConfig) -> tuple[float, float, float]:
    """Ensure LOW < MEDIUM < HIGH < CRITICAL banding."""
    m = float(agg.level_medium_min)
    h = float(agg.level_high_min)
    c = float(agg.level_critical_min)
    if not (m < h < c):
        raise ValueError(
            f"Need level_medium_min < level_high_min < level_critical_min, got {m}, {h}, {c}"
        )
    return m, h, c


def map_score_to_risk_level(
    risk_score: np.ndarray,
    agg: InterpretableAggregationConfig,
) -> list[RiskLevel]:
    m, h, cr = _normalize_level_cutoffs(agg)
    out: list[RiskLevel] = []
    for s in risk_score:
        if not np.isfinite(s) or s < m:
            out.append("LOW")
        elif s < h:
            out.append("MEDIUM")
        elif s < cr:
            out.append("HIGH")
        else:
            out.append("CRITICAL")
    return out


def _renormalize_weights(w: dict[str, float], active: dict[str, bool]) -> dict[str, float]:
    w_adj = {k: (w[k] if active.get(k, False) else 0.0) for k in w}
    s = float(sum(w_adj.values()))
    if s <= 0.0:
        return {k: 1.0 / max(len(w), 1) for k in w}
    return {k: w_adj[k] / s for k in w}


def build_rule_components(
    df: pd.DataFrame,
    cfg: InterpretableRiskConfig,
    *,
    column_hip_velocity: str = "hip_vertical_velocity",
    column_head_velocity: str = "head_vertical_velocity",
    column_torso: str | None = None,
    column_low_posture_count: str = "low_posture_duration_frames",
    column_instability: str = "instability_score",
    column_near_ground: str = "near_ground",
    column_guard: str = "guard_level",
    column_facing: str = "facing_away_score",
    column_reaction_delay: str = "reaction_delay_score",
    column_anomaly_score: str = "anomaly_score",
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """
    Compute each rule’s 0–1 component on ``df`` and return availability flags.

    If a required column is missing, that rule is skipped and its weight should be
    renormalized in the scorer.
    """
    n = len(df)
    zero = np.zeros(n, dtype=float)
    active: dict[str, bool] = {}
    comp: dict[str, np.ndarray] = {}

    if column_hip_velocity in df.columns:
        v = df[column_hip_velocity].to_numpy(dtype=float, copy=False)
        comp[RULE_FAST_DOWNWARD] = component_fast_downward(v, cfg.fast_downward)
        active[RULE_FAST_DOWNWARD] = True
    else:
        comp[RULE_FAST_DOWNWARD] = zero
        active[RULE_FAST_DOWNWARD] = False

    torso_col = column_torso
    if torso_col is None:
        if "torso_angle_deg" in df.columns:
            torso_col = "torso_angle_deg"
        elif "torso_angle_degrees" in df.columns:
            torso_col = "torso_angle_degrees"
    if torso_col and torso_col in df.columns:
        t = df[torso_col].to_numpy(dtype=float, copy=False)
        comp[RULE_LARGE_TORSO] = component_large_torso(t, cfg.large_torso)
        active[RULE_LARGE_TORSO] = True
    else:
        comp[RULE_LARGE_TORSO] = zero
        active[RULE_LARGE_TORSO] = False

    if column_low_posture_count in df.columns:
        d = df[column_low_posture_count].to_numpy(dtype=float, copy=False)
        comp[RULE_LOW_POSTURE] = component_prolonged_low_posture(d, cfg.prolonged_low_posture)
        active[RULE_LOW_POSTURE] = True
    else:
        comp[RULE_LOW_POSTURE] = zero
        active[RULE_LOW_POSTURE] = False

    if column_instability in df.columns:
        inst = df[column_instability].to_numpy(dtype=float, copy=False)
        comp[RULE_INSTABILITY] = component_high_instability(inst, cfg.high_instability)
        active[RULE_INSTABILITY] = True
    else:
        comp[RULE_INSTABILITY] = zero
        active[RULE_INSTABILITY] = False

    if column_near_ground in df.columns and column_hip_velocity in df.columns:
        ng = df[column_near_ground].to_numpy()
        hv = df[column_hip_velocity].to_numpy(dtype=float, copy=False)
        if column_head_velocity in df.columns:
            hdv = df[column_head_velocity].to_numpy(dtype=float, copy=False)
        else:
            # Without head velocity, use hip-only combined speed (documented limitation).
            hdv = np.zeros(n, dtype=float)
        comp[RULE_POST_FALL] = component_post_fall_low_movement(ng, hv, hdv, cfg.post_fall)
        active[RULE_POST_FALL] = True
    else:
        comp[RULE_POST_FALL] = zero
        active[RULE_POST_FALL] = False

    if column_guard in df.columns:
        g = df[column_guard].to_numpy(dtype=float, copy=False)
        comp[RULE_LOW_GUARD] = component_scaled_01_signal(g, cfg.low_guard)
        active[RULE_LOW_GUARD] = True
    else:
        comp[RULE_LOW_GUARD] = zero
        active[RULE_LOW_GUARD] = False

    if column_facing in df.columns:
        fa = df[column_facing].to_numpy(dtype=float, copy=False)
        comp[RULE_FACING_AWAY] = component_scaled_01_signal(fa, cfg.facing_away)
        active[RULE_FACING_AWAY] = True
    else:
        comp[RULE_FACING_AWAY] = zero
        active[RULE_FACING_AWAY] = False

    if column_reaction_delay in df.columns:
        r = df[column_reaction_delay].to_numpy(dtype=float, copy=False)
        comp[RULE_REACTION_DELAY] = component_scaled_01_signal(r, cfg.reaction_delay_signal)
        active[RULE_REACTION_DELAY] = True
    else:
        comp[RULE_REACTION_DELAY] = zero
        active[RULE_REACTION_DELAY] = False

    comp[RULE_LOSS_OF_CONTROL] = component_loss_of_control(
        comp[RULE_LOW_GUARD],
        comp[RULE_FACING_AWAY],
        comp[RULE_INSTABILITY],
    )
    active[RULE_LOSS_OF_CONTROL] = bool(
        active[RULE_INSTABILITY] and (active[RULE_LOW_GUARD] or active[RULE_FACING_AWAY])
    )

    comp[RULE_CLEAR_DANGER_FALL] = component_clear_danger_fall(
        comp[RULE_LARGE_TORSO],
        comp[RULE_FAST_DOWNWARD],
    )
    active[RULE_CLEAR_DANGER_FALL] = bool(active[RULE_LARGE_TORSO] and active[RULE_FAST_DOWNWARD])

    strike_col = "strike_incoming_proxy"
    if strike_col in df.columns:
        raw_strike = df[strike_col].to_numpy(dtype=float, copy=False)
        c_strike = component_scaled_01_signal(raw_strike, cfg.incoming_strike)
        comp[RULE_HIGH_RISK_GUARD_STRIKE] = component_high_risk_guard_strike(
            comp[RULE_LOW_GUARD],
            c_strike,
        )
        active[RULE_HIGH_RISK_GUARD_STRIKE] = bool(active[RULE_LOW_GUARD])
    else:
        comp[RULE_HIGH_RISK_GUARD_STRIKE] = zero
        active[RULE_HIGH_RISK_GUARD_STRIKE] = False

    comp[RULE_INTERVENTION_URGENT] = component_intervention_urgent(
        comp[RULE_POST_FALL],
        comp[RULE_LOW_POSTURE],
    )
    active[RULE_INTERVENTION_URGENT] = bool(active[RULE_POST_FALL] and active[RULE_LOW_POSTURE])

    if column_anomaly_score in df.columns:
        ax = df[column_anomaly_score].to_numpy(dtype=float, copy=False)
        comp[RULE_LIMB_ANOMALY] = component_scaled_01_signal(ax, cfg.limb_anomaly)
        active[RULE_LIMB_ANOMALY] = True
    else:
        comp[RULE_LIMB_ANOMALY] = zero
        active[RULE_LIMB_ANOMALY] = False

    return comp, active
