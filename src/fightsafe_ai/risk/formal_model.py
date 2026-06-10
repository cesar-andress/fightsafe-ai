"""
Formal **interpretable multi-signal risk fusion** (decision-support only).

This module is a **documented, testable** alternative view of the same heuristics the MVP scorer
uses: it exposes weighted signals, explicit interaction boosts, level thresholds, and an audit
trail. It does **not** diagnose injury or replace referees.

See ``configs/risk_fusion.yaml`` for defaults; ``configs/risk_rules.yaml`` remains the MVP rule source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fightsafe_ai.exceptions import ConfigurationError


SignalGroup = Literal[
    "biomechanics", "posture", "action", "anomaly", "inactivity", "surrender", "vlm"
]
Polarity = Literal["risk_increasing", "risk_decreasing"]

__all__ = [
    "EPS_ACTIVE",
    "InteractionRule",
    "Polarity",
    "RiskFusionConfig",
    "RiskFusionResult",
    "RiskSignal",
    "SignalGroup",
    "apply_interaction_rules",
    "build_audit_trail",
    "compute_base_weighted_score",
    "compute_fused_risk_score",
    "compute_pre_interaction_score",
    "default_risk_fusion_config",
    "fusion_full_result",
    "load_risk_fusion_config",
    "map_score_to_levels",
]

EPS_ACTIVE = 1e-9


@dataclass(frozen=True, slots=True)
class RiskSignal:
    """One interpretable input channel at an instant."""

    name: str
    group: SignalGroup
    confidence: float
    weight: float
    polarity: Polarity
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InteractionRule:
    """Explicit boost when **all** named signals are simultaneously active."""

    name: str
    required_signals: tuple[str, ...]
    boost: float
    rationale: str
    signal_threshold: float = 0.08


@dataclass(frozen=True, slots=True)
class RiskFusionConfig:
    """Loaded fusion parameters (thresholds, weights, interactions, smoothing metadata)."""

    signal_weights: dict[str, float]
    level_thresholds: dict[str, float]
    interaction_rules: tuple[InteractionRule, ...]
    minimum_event_duration_seconds: float
    smoothing_window_frames: int
    interaction_signal_threshold: float
    vlm_can_boost_deterministic_risk: bool
    vlm_max_boost: float
    source_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RiskFusionResult:
    """Single-frame fusion output suitable for logging and audit."""

    timestamp: float
    fighter_id: str | None
    risk_score: float
    risk_level: str
    active_signals: tuple[str, ...]
    triggered_signals: tuple[str, ...]
    explanation_facts: tuple[str, ...]
    audit_trail: dict[str, Any]


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def compute_pre_interaction_score(signals: list[RiskSignal], config: RiskFusionConfig) -> float:
    """
    Weighted mean over **risk_increasing** signals (normalized weights), then a bounded dampening
    term from **risk_decreasing** signals (does not model a calibrated ``NOT`` operator).
    """
    if not signals:
        return 0.0

    increasing = [
        s
        for s in signals
        if s.polarity == "risk_increasing"
        and s.group != "vlm"
        and float(config.signal_weights.get(s.name, s.weight)) > 0.0
    ]
    decreasing = [
        s
        for s in signals
        if s.polarity == "risk_decreasing"
        and s.group != "vlm"
        and float(config.signal_weights.get(s.name, s.weight)) > 0.0
    ]

    w_sum = sum(float(config.signal_weights.get(s.name, s.weight)) for s in increasing)
    base = 0.0
    if w_sum > EPS_ACTIVE:
        for s in increasing:
            wn = float(config.signal_weights.get(s.name, s.weight)) / w_sum
            base += wn * _clamp01(s.confidence)
    base = _clamp01(base)

    if decreasing:
        dsum = sum(float(config.signal_weights.get(s.name, s.weight)) for s in decreasing)
        if dsum > EPS_ACTIVE:
            dec_mean = 0.0
            for s in decreasing:
                wn = float(config.signal_weights.get(s.name, s.weight)) / dsum
                dec_mean += wn * _clamp01(s.confidence)
            base = _clamp01(base * (1.0 - 0.35 * min(1.0, dec_mean)))
    return base


def compute_fused_risk_score(signals: list[RiskSignal], config: RiskFusionConfig) -> float:
    """
    Pure weighted fusion with explicit interaction boosts.

    - ``compute_pre_interaction_score`` for increasing/decreasing channels (excluding VLM).
    - Adds interaction boosts only when every required signal is above threshold.
    - Optional VLM boost only if ``config.vlm_can_boost_deterministic_risk`` is True.
    - Returns value in ``[0, 1]``.
    """
    if not signals:
        return 0.0

    pre = compute_pre_interaction_score(signals, config)
    boost_total, _ = apply_interaction_rules(signals, config)

    vlm_signals = [s for s in signals if s.group == "vlm"]
    vlm_part = 0.0
    if (
        config.vlm_can_boost_deterministic_risk
        and vlm_signals
        and float(config.vlm_max_boost) > 0.0
    ):
        vmax = max((_clamp01(s.confidence) for s in vlm_signals), default=0.0)
        vlm_part = min(float(config.vlm_max_boost), float(config.vlm_max_boost) * vmax)

    return _clamp01(pre + boost_total + vlm_part)


def map_score_to_levels(score: float, config: RiskFusionConfig) -> str:
    """Map ``[0,1]`` score to ``LOW`` / ``MEDIUM`` / ``HIGH`` / ``CRITICAL`` using YAML thresholds."""
    lt = config.level_thresholds
    m = float(lt.get("medium_min", lt.get("level_medium_min", 0.25)))
    h = float(lt.get("high_min", lt.get("level_high_min", 0.5)))
    c = float(lt.get("critical_min", lt.get("level_critical_min", 0.75)))
    if not (m < h < c):
        raise ValueError(f"Invalid thresholds: medium={m}, high={h}, critical={c}")
    s = _clamp01(score)
    if s < m:
        return "LOW"
    if s < h:
        return "MEDIUM"
    if s < c:
        return "HIGH"
    return "CRITICAL"


def build_audit_trail(
    signals: list[RiskSignal],
    config: RiskFusionConfig,
    base_score: float,
    fused_score: float,
    applied_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "base_score_before_interactions_and_vlm": base_score,
        "fused_risk_score": fused_score,
        "signal_weights_used": {k: float(v) for k, v in config.signal_weights.items()},
        "level_thresholds": dict(config.level_thresholds),
        "signals": [
            {
                "name": s.name,
                "group": s.group,
                "confidence": s.confidence,
                "weight_config": float(config.signal_weights.get(s.name, s.weight)),
                "polarity": s.polarity,
            }
            for s in signals
        ],
        "interaction_rules_applied": applied_rules,
        "vlm_can_boost_deterministic_risk": config.vlm_can_boost_deterministic_risk,
    }


def apply_interaction_rules(
    signals: list[RiskSignal],
    config: RiskFusionConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Return sum of boosts and a list of audit dicts for rules that fired."""
    by_name: dict[str, float] = {s.name: _clamp01(s.confidence) for s in signals}
    thr_default = float(config.interaction_signal_threshold)
    total = 0.0
    applied: list[dict[str, Any]] = []
    for rule in config.interaction_rules:
        th = float(rule.signal_threshold or thr_default)
        if all(by_name.get(n, 0.0) > th for n in rule.required_signals):
            b = float(rule.boost)
            total += b
            applied.append(
                {
                    "rule_name": rule.name,
                    "boost": b,
                    "required_signals": list(rule.required_signals),
                    "rationale": rule.rationale,
                    "threshold_used": th,
                }
            )
    return total, applied


def compute_base_weighted_score(signals: list[RiskSignal], config: RiskFusionConfig) -> float:
    """Alias for pre-interaction fused score (increasing + decreasing dampening, no rule boosts)."""
    return compute_pre_interaction_score(signals, config)


def fusion_full_result(
    *,
    timestamp: float,
    fighter_id: str | None,
    signals: list[RiskSignal],
    config: RiskFusionConfig,
) -> RiskFusionResult:
    """
    Full fusion: base score, interaction boosts, optional VLM boost, level mapping, explanations.

    ``explanation_facts`` include human-readable strings; ``triggered_signals`` lists signal names
    with confidence above ``interaction_signal_threshold``.
    """
    base_pre = compute_pre_interaction_score(signals, config)
    fused = compute_fused_risk_score(signals, config)
    thr = float(config.interaction_signal_threshold)
    boost_sum, applied = apply_interaction_rules(signals, config)

    active = tuple(sorted({s.name for s in signals if _clamp01(s.confidence) > thr}))
    triggered = tuple(sorted({s.name for s in signals if _clamp01(s.confidence) > thr}))

    facts: list[str] = []
    for s in signals:
        if s.group == "vlm":
            facts.append(
                f"VLM note ({s.name}): non-authoritative review hint (confidence {_clamp01(s.confidence):.2f})."
            )
    for a in applied:
        facts.append(f"Interaction '{a['rule_name']}': +{a['boost']:.3f} — {a['rationale']}")

    level = map_score_to_levels(fused, config)
    audit = build_audit_trail(signals, config, base_pre, fused, applied)
    audit["interaction_boost_sum"] = boost_sum
    audit["base_weighted_mean"] = base_pre

    return RiskFusionResult(
        timestamp=float(timestamp),
        fighter_id=fighter_id,
        risk_score=fused,
        risk_level=level,
        active_signals=active,
        triggered_signals=triggered,
        explanation_facts=tuple(facts),
        audit_trail=audit,
    )


def _parse_interaction_rules(raw: list[dict[str, Any]] | None) -> tuple[InteractionRule, ...]:
    if not raw:
        return ()
    out: list[InteractionRule] = []
    for item in raw:
        req = item.get("required_signals") or item.get("requires") or []
        if isinstance(req, str):
            req = [req]
        out.append(
            InteractionRule(
                name=str(item["name"]),
                required_signals=tuple(str(x) for x in req),
                boost=float(item.get("boost", 0.0)),
                rationale=str(item.get("rationale", "")),
                signal_threshold=float(item.get("signal_threshold", item.get("threshold", 0.08))),
            )
        )
    return tuple(out)


def load_risk_fusion_config(path: Path | None) -> RiskFusionConfig:
    """Load ``RiskFusionConfig`` from YAML; fall back to packaged numeric defaults if missing."""
    default_root = Path(__file__).resolve().parents[3] / "configs" / "risk_fusion.yaml"
    p = path if path is not None else default_root
    if not p.is_file():
        raise ConfigurationError(f"risk fusion config not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sw = data.get("signal_weights") or {}
    if not isinstance(sw, dict):
        raise ConfigurationError("signal_weights must be a mapping")

    lt = data.get("level_thresholds") or {}
    if not lt:
        lt = {
            "medium_min": float(data.get("level_medium_min", 0.25)),
            "high_min": float(data.get("level_high_min", 0.5)),
            "critical_min": float(data.get("level_critical_min", 0.75)),
        }

    rules_raw = data.get("interaction_rules") or []
    rules = _parse_interaction_rules(rules_raw if isinstance(rules_raw, list) else [])

    smooth = data.get("smoothing") or {}
    vlm = data.get("vlm") or {}

    return RiskFusionConfig(
        signal_weights={str(k): float(v) for k, v in sw.items()},
        level_thresholds={str(k): float(v) for k, v in lt.items()},
        interaction_rules=rules,
        minimum_event_duration_seconds=float(data.get("minimum_event_duration_seconds", 0.5)),
        smoothing_window_frames=int(
            data.get("smoothing_window_frames", smooth.get("window_frames", 3))
        ),
        interaction_signal_threshold=float(data.get("interaction_signal_threshold", 0.08)),
        vlm_can_boost_deterministic_risk=bool(vlm.get("can_boost_deterministic_risk", False)),
        vlm_max_boost=float(vlm.get("max_boost", 0.0)),
        source_path=p.resolve(),
    )


def default_risk_fusion_config() -> RiskFusionConfig:
    """``configs/risk_fusion.yaml`` next to project root when present."""
    root = Path(__file__).resolve().parents[3] / "configs" / "risk_fusion.yaml"
    return load_risk_fusion_config(root)
