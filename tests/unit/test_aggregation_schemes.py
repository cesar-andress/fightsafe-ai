"""Minimal tests for equal / weighted / max aggregation and availability masking."""

from __future__ import annotations

from fightsafe_ai.evaluation.aggregation_schemes import (
    SCHEME_ORDER,
    compute_fused_risk_with_scheme,
    compute_pre_interaction_with_scheme,
)
from fightsafe_ai.risk.formal_model import RiskFusionConfig, RiskSignal


def _sig(name: str, confidence: float, weight: float = 1.0) -> RiskSignal:
    return RiskSignal(
        name=name,
        confidence=confidence,
        weight=weight,
        polarity="risk_increasing",
        group="biomechanics",
    )


def _cfg(weights: dict[str, float]) -> RiskFusionConfig:
    return RiskFusionConfig(
        signal_weights=weights,
        level_thresholds={"medium_min": 0.25, "high_min": 0.5, "critical_min": 0.75},
        interaction_rules=(),
        minimum_event_duration_seconds=0.0,
        smoothing_window_frames=1,
        interaction_signal_threshold=0.08,
        vlm_can_boost_deterministic_risk=False,
        vlm_max_boost=0.0,
    )


def test_scheme_order_contains_paper_schemes() -> None:
    assert "equal" in SCHEME_ORDER
    assert "weighted" in SCHEME_ORDER
    assert "max" in SCHEME_ORDER


def test_equal_weighted_max_differ_with_unequal_confidences() -> None:
    signals = [_sig("a", 1.0, 2.0), _sig("b", 0.0, 1.0)]
    cfg = _cfg({"a": 2.0, "b": 1.0})
    equal = compute_pre_interaction_with_scheme(signals, cfg, "equal")
    weighted = compute_pre_interaction_with_scheme(signals, cfg, "weighted")
    max_s = compute_pre_interaction_with_scheme(signals, cfg, "max")
    assert abs(equal - 0.5) < 1e-9
    assert abs(max_s - 1.0) < 1e-9
    assert weighted > equal  # weight-biased toward a=1.0


def test_unavailable_channel_excluded_when_active_false() -> None:
    signals = [_sig("a", 1.0), _sig("b", 0.0)]
    cfg = _cfg({"a": 1.0, "b": 1.0})
    active = {"a": False, "b": True}
    # Only b available → equal mean of {0.0}
    equal = compute_pre_interaction_with_scheme(signals, cfg, "equal", active=active)
    assert abs(equal - 0.0) < 1e-9
    # Max over available only
    max_s = compute_pre_interaction_with_scheme(signals, cfg, "max", active=active)
    assert abs(max_s - 0.0) < 1e-9


def test_zero_confidence_with_available_is_not_masked_out() -> None:
    """α=1, c=0 is zero-valued evidence, not unavailable."""
    signals = [_sig("a", 0.0), _sig("b", 1.0)]
    cfg = _cfg({"a": 1.0, "b": 1.0})
    active = {"a": True, "b": True}
    equal = compute_pre_interaction_with_scheme(signals, cfg, "equal", active=active)
    assert abs(equal - 0.5) < 1e-9


def test_fused_matches_pre_without_interactions() -> None:
    signals = [_sig("a", 0.4), _sig("b", 0.8)]
    cfg = _cfg({"a": 1.0, "b": 1.0})
    for scheme in ("equal", "weighted", "max"):
        pre = compute_pre_interaction_with_scheme(signals, cfg, scheme)
        fused = compute_fused_risk_with_scheme(signals, cfg, scheme)
        assert abs(pre - fused) < 1e-9
