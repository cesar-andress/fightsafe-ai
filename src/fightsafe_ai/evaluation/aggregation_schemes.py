"""
Alternative frame-wise aggregation schemes over formal :class:`RiskSignal` channels.

Used for robustness comparisons only; production fusion remains ``compute_pre_interaction_score``.
"""

from __future__ import annotations

from typing import Literal

from fightsafe_ai.risk.formal_model import (
    EPS_ACTIVE,
    RiskFusionConfig,
    RiskSignal,
    _clamp01,
    apply_interaction_rules,
    compute_pre_interaction_score,
)


AggregationScheme = Literal["weighted", "equal", "reliability_discounted", "max"]

SCHEME_ORDER: tuple[AggregationScheme, ...] = (
    "weighted",
    "equal",
    "reliability_discounted",
    "max",
)

SCHEME_LABELS: dict[AggregationScheme, str] = {
    "weighted": "Weighted average (current)",
    "equal": "Equal-weight average",
    "reliability_discounted": "Reliability-discounted average",
    "max": "Max-score aggregation",
}

DECREASING_ATTENUATION = 0.35


def _increasing_signals(signals: list[RiskSignal], config: RiskFusionConfig) -> list[RiskSignal]:
    return [
        s
        for s in signals
        if s.polarity == "risk_increasing"
        and s.group != "vlm"
        and float(config.signal_weights.get(s.name, s.weight)) > 0.0
    ]


def _decreasing_signals(signals: list[RiskSignal], config: RiskFusionConfig) -> list[RiskSignal]:
    return [
        s
        for s in signals
        if s.polarity == "risk_decreasing"
        and s.group != "vlm"
        and float(config.signal_weights.get(s.name, s.weight)) > 0.0
    ]


def _apply_decreasing_attenuation(
    base: float,
    signals: list[RiskSignal],
    config: RiskFusionConfig,
    *,
    scheme: AggregationScheme,
    active: dict[str, bool] | None,
) -> float:
    decreasing = _decreasing_signals(signals, config)
    if not decreasing:
        return _clamp01(base)
    if scheme == "equal":
        dvals = [_clamp01(s.confidence) for s in decreasing]
        dec_mean = sum(dvals) / len(dvals) if dvals else 0.0
    elif scheme == "max":
        dec_mean = max((_clamp01(s.confidence) for s in decreasing), default=0.0)
    elif scheme == "reliability_discounted":
        dsum = 0.0
        dec_mean = 0.0
        for s in decreasing:
            rel = 1.0 if active is None or active.get(s.name, True) else 0.0
            w = float(config.signal_weights.get(s.name, s.weight)) * rel
            if w <= 0.0:
                continue
            dsum += w
            dec_mean += w * _clamp01(s.confidence)
        dec_mean = dec_mean / dsum if dsum > EPS_ACTIVE else 0.0
    else:
        dsum = sum(float(config.signal_weights.get(s.name, s.weight)) for s in decreasing)
        dec_mean = 0.0
        if dsum > EPS_ACTIVE:
            for s in decreasing:
                wn = float(config.signal_weights.get(s.name, s.weight)) / dsum
                dec_mean += wn * _clamp01(s.confidence)
    return _clamp01(base * (1.0 - DECREASING_ATTENUATION * min(1.0, dec_mean)))


def compute_pre_interaction_with_scheme(
    signals: list[RiskSignal],
    config: RiskFusionConfig,
    scheme: AggregationScheme,
    *,
    active: dict[str, bool] | None = None,
) -> float:
    """Alternative pre-interaction aggregators over the same evidence channels."""
    if not signals:
        return 0.0
    if scheme == "weighted":
        return compute_pre_interaction_score(signals, config)

    increasing = _increasing_signals(signals, config)
    if active is not None:
        increasing = [s for s in increasing if active.get(s.name, True)]

    base = 0.0
    if increasing:
        if scheme == "equal":
            base = sum(_clamp01(s.confidence) for s in increasing) / float(len(increasing))
        elif scheme == "max":
            base = max(_clamp01(s.confidence) for s in increasing)
        elif scheme == "reliability_discounted":
            w_sum = 0.0
            for s in increasing:
                rel = 1.0 if active is None or active.get(s.name, True) else 0.0
                w = float(config.signal_weights.get(s.name, s.weight)) * rel
                if w <= 0.0:
                    continue
                w_sum += w
                base += w * _clamp01(s.confidence)
            base = base / w_sum if w_sum > EPS_ACTIVE else 0.0
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
    base = _clamp01(base)
    return _apply_decreasing_attenuation(base, signals, config, scheme=scheme, active=active)


def compute_fused_risk_with_scheme(
    signals: list[RiskSignal],
    config: RiskFusionConfig,
    scheme: AggregationScheme,
    *,
    active: dict[str, bool] | None = None,
) -> float:
    """Same interaction boosts and VLM handling as production; only pre-score differs."""
    if not signals:
        return 0.0
    pre = compute_pre_interaction_with_scheme(signals, config, scheme, active=active)
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


__all__ = [
    "DECREASING_ATTENUATION",
    "SCHEME_LABELS",
    "SCHEME_ORDER",
    "AggregationScheme",
    "compute_fused_risk_with_scheme",
    "compute_pre_interaction_with_scheme",
]
