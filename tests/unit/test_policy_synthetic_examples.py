"""
Policy-driven **examples** for **docs/testing-policy.md**.

Each test is offline, uses :mod:`tests.fixtures.synthetic`, and loads code via
:mod:`tests.support.isolated` to avoid optional heavy imports. For broader coverage
of the same domains, see ``tests/unit/`` (e.g. ``test_biomechanics_synthetic``,
``test_temporal_synthetic``, ``test_risk_rules_synthetic``, ``test_risk_events``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.synthetic import (
    make_frame_risk_for_event_merge,
    make_interpretable_risk_feature_frame,
    make_keypoint_long_format_tiny,
    make_temporal_feature_input_small,
)
from tests.support.isolated import (
    load_biomechanics,
    load_risk_events,
    load_risk_rules,
    load_risk_scorer,
    load_temporal,
)


def test_example_biomechanical_features_pure_table() -> None:
    """Biomechanical features from a tiny long-format keypoint table (unit-style)."""
    bio = load_biomechanics()
    a = make_keypoint_long_format_tiny(hip_y=0.5, frame_id="0")
    b = make_keypoint_long_format_tiny(hip_y=0.55, frame_id="1")
    out = bio.compute_biomechanical_features(pd.concat([a, b], ignore_index=True))
    assert len(out) == 2
    assert float(out.iloc[0]["hip_vertical_position"]) < float(out.iloc[1]["hip_vertical_position"])


def test_example_temporal_features_deterministic() -> None:
    """Temporal features on a fixed synthetic table (no video, no network)."""
    t = load_temporal()
    df = make_temporal_feature_input_small(n=10)
    out = t.compute_temporal_features(df, fps=30, rolling_window_frames=3, min_periods=1)
    assert len(out) == 10
    v = out["hip_vertical_velocity"].to_numpy()
    assert np.isfinite(v).all()


def test_example_risk_scoring_interpretable() -> None:
    """Frame-wise interpretable risk score (0–1) and levels from feature columns."""
    rules = load_risk_rules()
    scorer = load_risk_scorer()
    cfg = rules.InterpretableRiskConfig.default()
    df = make_interpretable_risk_feature_frame()
    out = scorer.compute_interpretable_risk(df, config=cfg)
    assert len(out) == 3
    assert (out["risk_score"] >= 0).all() and (out["risk_score"] <= 1.0 + 1e-9).all()
    assert out["risk_level"].iloc[1] in ("HIGH", "CRITICAL", "MEDIUM")


def test_example_event_merging_merges_runs() -> None:
    """Frame-level risk → merged events; gap threshold controls merge behavior."""
    ev = load_risk_events()
    df = make_frame_risk_for_event_merge()
    cfg = ev.RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0)
    merged = ev.frame_risk_to_events(df, cfg)
    assert len(merged) == 1
    assert float(merged.iloc[0]["duration_seconds"]) == pytest.approx(0.4, rel=1e-6)
    split = ev.frame_risk_to_events(
        df, ev.RiskEventExtractionConfig(merge_gap_frames=1, min_duration_seconds=0.0)
    )
    assert len(split) == 2
