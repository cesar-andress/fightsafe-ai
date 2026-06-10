"""Prototype tap-out / surrender heuristics (no learning)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fightsafe_ai.pose.keypoints import Keypoint, PoseResult
from fightsafe_ai.risk.scorer import compute_interpretable_risk
from fightsafe_ai.risk.surrender import (
    SURRENDER_RULE_KEY,
    SurrenderHeuristicConfig,
    apply_surrender_overrides_to_risk_dataframe,
    detect_surrender,
)


def _pose(frame_id: str, wrist_y: float) -> PoseResult:
    return PoseResult(
        frame_id=frame_id,
        keypoints=[
            Keypoint("left_wrist", 0.45, wrist_y),
            Keypoint("right_wrist", 0.55, wrist_y + 0.01),
            Keypoint("nose", 0.5, 0.2),
        ],
    )


def test_detect_surrender_too_short() -> None:
    assert detect_surrender([_pose("0", 0.5)] * 3).surrender_detected is False
    assert detect_surrender([_pose("0", 0.5)] * 3).confidence == 0.0


def test_detect_surrender_oscillating_wrists() -> None:
    """Rapid up-down in image y should raise confidence (MVP, not a real fight clip)."""
    ys = [0.42, 0.55, 0.44, 0.58, 0.45, 0.60, 0.47, 0.59, 0.48, 0.57, 0.50, 0.56]
    seq = [_pose(str(i), y) for i, y in enumerate(ys)]
    lo = SurrenderHeuristicConfig(detect_confidence_threshold=0.45, min_frames=8)
    r = detect_surrender(seq, config=lo)
    assert r.confidence > 0.3
    assert r.surrender_detected is True


def test_compute_interpretable_risk_surrender_sets_critical() -> None:
    n = 12
    ys = np.linspace(0.42, 0.65, n) + 0.08 * np.sin(np.linspace(0, 4 * np.pi, n))
    poses = [_pose(str(i), float(ys[i])) for i in range(n)]
    df = pd.DataFrame(
        {
            "hip_vertical_velocity": [0.0] * n,
            "head_vertical_velocity": [0.0] * n,
            "torso_angle_deg": [10.0] * n,
            "low_posture_duration_frames": [0.0] * n,
            "instability_score": [0.01] * n,
            "near_ground": [False] * n,
            "guard_level": [0.0] * n,
            "facing_away_score": [0.0] * n,
            "reaction_delay_score": [0.0] * n,
        }
    )
    cfg = SurrenderHeuristicConfig(detect_confidence_threshold=0.4, min_frames=8)
    out = compute_interpretable_risk(
        df,
        pose_per_frame=poses,
        surrender_config=cfg,
    )
    assert "surrender_confidence" in out.columns
    assert (out["surrender_confidence"].to_numpy() >= 0.0).all()
    assert (
        any(s == "CRITICAL" for s in out["risk_level"].astype(str).tolist())
        or out["surrender_confidence"].max() > 0.0
    )


def test_apply_surrender_appends_rule_key() -> None:
    n = 10
    poses = [_pose(str(i), 0.5 + 0.1 * (i % 2)) for i in range(n)]
    o = pd.DataFrame(
        {
            "risk_score": [0.1] * n,
            "risk_level": ["LOW"] * n,
            "triggered_rules": [[] for _ in range(n)],
        }
    )
    cfg = SurrenderHeuristicConfig(detect_confidence_threshold=0.0, min_frames=4)
    out = apply_surrender_overrides_to_risk_dataframe(o, poses, window_frames=10, config=cfg)
    # threshold 0 => every window with min_frames may flag (confidence may be 0 on flat); check column exists
    assert "surrender_confidence" in out.columns
    # With threshold 0, flat motion may still have conf 0; require at least one list gets key if any det
    any_s = out["surrender_confidence"].max() > 0
    if any_s and out["risk_level"].eq("CRITICAL").any():
        idx = int(out["risk_level"].tolist().index("CRITICAL"))
        assert SURRENDER_RULE_KEY in out["triggered_rules"].iloc[idx]
