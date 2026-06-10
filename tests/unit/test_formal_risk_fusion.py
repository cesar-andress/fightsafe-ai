"""Formal interpretable risk fusion (pure math, YAML, ablation) — no network or video."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.evaluation.risk_ablation import (
    ABLATION_MODES,
    config_for_ablation_mode,
    filter_signals_for_ablation,
    formal_risk_timeseries,
    run_risk_ablation,
)
from fightsafe_ai.risk.adapters import signals_from_feature_row
from fightsafe_ai.risk.formal_model import (
    InteractionRule,
    RiskFusionConfig,
    RiskSignal,
    apply_interaction_rules,
    compute_fused_risk_score,
    compute_pre_interaction_score,
    default_risk_fusion_config,
    load_risk_fusion_config,
    map_score_to_levels,
)
from fightsafe_ai.risk.rules import InterpretableRiskConfig, load_interpretable_risk_config
from fightsafe_ai.risk.surrender import SURRENDER_RULE_KEY


pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fusion_cfg() -> RiskFusionConfig:
    return load_risk_fusion_config(_REPO_ROOT / "configs" / "risk_fusion.yaml")


@pytest.fixture
def interpret_cfg() -> InterpretableRiskConfig:
    return load_interpretable_risk_config(_REPO_ROOT / "configs" / "risk_rules.yaml")


def _signal(
    name: str,
    group: str,
    conf: float,
    *,
    pol: str = "risk_increasing",
    w: float = 1.0,
) -> RiskSignal:
    return RiskSignal(
        name=name,
        group=group,  # type: ignore[arg-type]
        confidence=conf,
        weight=w,
        polarity=pol,  # type: ignore[arg-type]
        evidence={},
    )


def test_fused_score_in_01(fusion_cfg: RiskFusionConfig) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(0, 12))
        sigs = [_signal(f"s{i}", "biomechanics", float(rng.random())) for i in range(n)]
        s = compute_fused_risk_score(sigs, fusion_cfg)
        assert 0.0 <= s <= 1.0


def test_empty_signals_zero(fusion_cfg: RiskFusionConfig) -> None:
    assert compute_fused_risk_score([], fusion_cfg) == 0.0


def test_weights_normalize_single_active_rule(fusion_cfg: RiskFusionConfig) -> None:
    cfg = replace(
        fusion_cfg,
        signal_weights={"fast_downward_motion": 2.0},
        interaction_rules=(),
    )
    only = [_signal("fast_downward_motion", "biomechanics", 1.0)]
    pre = compute_pre_interaction_score(only, cfg)
    assert abs(pre - 1.0) < 1e-6


def test_missing_signals_do_not_break(fusion_cfg: RiskFusionConfig) -> None:
    cfg = replace(fusion_cfg, interaction_rules=())
    s = compute_fused_risk_score(
        [_signal("fast_downward_motion", "biomechanics", 0.3)],
        cfg,
    )
    assert 0.0 <= s <= 1.0


def test_interaction_only_when_all_required(fusion_cfg: RiskFusionConfig) -> None:
    rule = InteractionRule(
        name="t",
        required_signals=("low_guard", "high_instability"),
        boost=0.1,
        rationale="test",
        signal_threshold=0.08,
    )
    cfg = replace(fusion_cfg, interaction_rules=(rule,), vlm_max_boost=0.0)
    a = [
        _signal("low_guard", "posture", 0.9),
        _signal("high_instability", "biomechanics", 0.9),
    ]
    b0, ar0 = apply_interaction_rules(a, cfg)
    assert b0 > 0 and len(ar0) == 1
    b1, ar1 = apply_interaction_rules(
        [_signal("low_guard", "posture", 0.9)],
        cfg,
    )
    assert b1 == 0.0 and ar1 == []


def test_vlm_does_not_boost_by_default(fusion_cfg: RiskFusionConfig) -> None:
    assert fusion_cfg.vlm_can_boost_deterministic_risk is False
    base = [
        _signal("fast_downward_motion", "biomechanics", 0.2),
    ]
    s0 = compute_fused_risk_score(base, fusion_cfg)
    with_vlm = [
        *base,
        RiskSignal(
            name="vlm_review_hint",
            group="vlm",
            confidence=1.0,
            weight=0.0,
            polarity="risk_increasing",
            evidence={},
        ),
    ]
    s1 = compute_fused_risk_score(with_vlm, fusion_cfg)
    assert abs(s0 - s1) < 1e-9


def test_map_levels_use_yaml_defaults(fusion_cfg: RiskFusionConfig) -> None:
    assert map_score_to_levels(0.0, fusion_cfg) == "LOW"
    assert map_score_to_levels(0.24, fusion_cfg) == "LOW"
    assert map_score_to_levels(0.25, fusion_cfg) == "MEDIUM"
    assert map_score_to_levels(0.75, fusion_cfg) == "CRITICAL"


def test_ablation_modes_change_scores(
    fusion_cfg: RiskFusionConfig, interpret_cfg: InterpretableRiskConfig
) -> None:
    row = {
        "instability_score": 0.95,
        "guard_level": 0.9,
        "hip_vertical_velocity": 0.0,
        "torso_angle_deg": 30.0,
        "low_posture_duration_frames": 0.0,
        "near_ground": 0.0,
        "facing_away_score": 0.0,
        "reaction_delay_score": 0.0,
        "anomaly_score": 0.0,
    }
    sig_full = signals_from_feature_row(
        row, interpretable_config=interpret_cfg, fusion_config=fusion_cfg
    )
    sig_bio = filter_signals_for_ablation(sig_full, "biomechanics_only")
    assert compute_fused_risk_score(sig_full, fusion_cfg) != compute_fused_risk_score(
        sig_bio, fusion_cfg
    )


def test_surrender_ablation_never_includes_surrender_signal(
    fusion_cfg: RiskFusionConfig, interpret_cfg: InterpretableRiskConfig
) -> None:
    row = {"surrender_confidence": 0.9}
    sig = signals_from_feature_row(
        row, interpretable_config=interpret_cfg, fusion_config=fusion_cfg, surrender_confidence=0.9
    )
    sur = [s for s in sig if s.name == SURRENDER_RULE_KEY]
    assert sur
    filt = filter_signals_for_ablation(sig, "full_fusion_with_surrender_disabled")
    assert not any(s.name == SURRENDER_RULE_KEY for s in filt)


def test_config_without_interactions(fusion_cfg: RiskFusionConfig) -> None:
    assert (
        config_for_ablation_mode(fusion_cfg, "full_fusion_without_interactions").interaction_rules
        == ()
    )


def test_formal_timeseries_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "features.csv"
    df = pd.DataFrame(
        {
            "instability_score": [0.0, 0.5, 0.0],
            "guard_level": [0.0, 0.0, 0.0],
        }
    )
    df.to_csv(p, index=False)
    a = formal_risk_timeseries(
        p,
        fusion_yaml=_REPO_ROOT / "configs" / "risk_fusion.yaml",
        rules_yaml=None,
        mode="full_fusion",
        fps=10.0,
    )
    b = formal_risk_timeseries(
        p,
        fusion_yaml=_REPO_ROOT / "configs" / "risk_fusion.yaml",
        rules_yaml=None,
        mode="full_fusion",
        fps=10.0,
    )
    pd.testing.assert_frame_equal(a, b)


def test_run_risk_ablation_all_modes_produce_files(tmp_path: Path) -> None:
    p = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "instability_score": [0.1, 0.2],
            "guard_level": [0.0, 0.1],
            "hip_vertical_velocity": [0.0, 0.0],
        }
    ).to_csv(p, index=False)
    out = run_risk_ablation(
        p,
        tmp_path / "out",
        fusion_yaml=_REPO_ROOT / "configs" / "risk_fusion.yaml",
    )
    assert (out / "ablation_results.csv").is_file()
    assert (out / "ablation_results.tex").is_file()
    for m in ABLATION_MODES:
        assert (out / f"risk_series_{m}.csv").is_file()
    h1 = hashlib.sha256((out / "ablation_results.csv").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out / "ablation_results.csv").read_bytes()).hexdigest()
    assert h1 == h2


def test_default_config_loads() -> None:
    c = default_risk_fusion_config()
    assert c.source_path is not None
    assert c.level_thresholds
