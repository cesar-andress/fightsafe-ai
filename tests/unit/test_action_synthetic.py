"""Synthetic BlazePose-style landmark sequences for action heuristics (MVP, no I/O)."""

from __future__ import annotations

import pytest

from fightsafe_ai.action.base import (
    ActionSignal,
    ActionType,
    landmark_map_xy,
)
from fightsafe_ai.action.defense import (
    defensive_incapacity_confidence,
    low_guard_confidence,
    turned_back_confidence,
)
from fightsafe_ai.action.punch_kick import (
    body_scale,
    limb_motion_features,
    punch_activity_confidence,
)
from fightsafe_ai.action.temporal_classifier import (
    HeuristicMVPActionDetector,
    HeuristicMVPConfig,
    run_sequence_mvp,
)
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult


pytestmark = pytest.mark.unit

# Y-down normalized coords; same naming as `fightsafe_ai.pose.blazepose.BLAZEPOSE_33` subset


def _person_front(
    *,
    hand_y: float = 0.32,
    shoulder_scale: float = 0.2,
    hip_x_span: float = 0.16,
) -> dict[str, tuple[float, float]]:
    y_s = 0.25
    y_h = 0.5
    cx = 0.5
    hsw = shoulder_scale * 0.5
    hhx = hip_x_span * 0.5
    return {
        "nose": (0.5, 0.15),
        "left_shoulder": (cx - hsw, y_s),
        "right_shoulder": (cx + hsw, y_s),
        "left_hip": (cx - hhx, y_h),
        "right_hip": (cx + hhx, y_h),
        "left_wrist": (0.4, hand_y),
        "right_wrist": (0.6, hand_y),
        "left_ankle": (0.43, 0.9),
        "right_ankle": (0.57, 0.9),
    }


def _person_profile_turned(shoulder_w: float) -> dict[str, tuple[float, float]]:
    """Hips wide, shoulders very narrow in x → high turned-back score."""
    return {
        "nose": (0.5, 0.15),
        "left_shoulder": (0.5 - shoulder_w, 0.25),
        "right_shoulder": (0.5 + shoulder_w, 0.25),
        "left_hip": (0.4, 0.5),
        "right_hip": (0.6, 0.5),
        "left_wrist": (0.4, 0.35),
        "right_wrist": (0.6, 0.35),
        "left_ankle": (0.42, 0.9),
        "right_ankle": (0.58, 0.9),
    }


def test_landmark_map_from_pose() -> None:
    pr = PoseResult(
        "f0",
        [
            Keypoint("nose", 0.0, 0.0, None, 1.0),
            Keypoint("left_wrist", 0.1, 0.2, None, 1.0),
        ],
    )
    m = landmark_map_xy(pr.keypoints)
    assert m["nose"] == (0.0, 0.0)
    assert m["left_wrist"] == (0.1, 0.2)


def test_action_signal_confidence_in_range() -> None:
    a = ActionSignal(0.0, "0", ActionType.LOW_GUARD, 0.5, {})
    assert a.confidence == 0.5
    with pytest.raises(ValueError, match="confidence"):
        ActionSignal(0.0, "0", ActionType.LOW_GUARD, 1.5, {})


def test_limb_features_zero_without_previous() -> None:
    a = _person_front()
    fe = limb_motion_features(None, a, 0.03)
    assert fe.max_wrist_speed == 0.0
    sc = body_scale(a)
    assert 0.05 < sc < 0.3


def test_punch_synthetic() -> None:
    t0 = _person_front()
    t1 = {**_person_front(), "left_wrist": (0.4, 0.12)}  # fast vertical move vs t0
    fe = limb_motion_features(t0, t1, dt=0.05)
    p = punch_activity_confidence(fe, body_scale(t1), vel_over_scale_threshold=2.5)
    assert p > 0.3


def test_low_guard() -> None:
    low = _person_front(hand_y=0.6)
    assert low_guard_confidence(low) > 0.4


def test_turned_back() -> None:
    assert turned_back_confidence(_person_profile_turned(0.01)) > 0.3
    assert turned_back_confidence(_person_front()) < 0.1


def test_defensive_incapacity() -> None:
    low = low_guard_confidence(_person_front(hand_y=0.62))
    m = limb_motion_features(_person_front(hand_y=0.62), _person_front(hand_y=0.62), 0.05)
    d = defensive_incapacity_confidence(low, m, static_speed_threshold=0.45)
    assert d > 0.2


def test_mvp_emits_punch() -> None:
    t0 = _person_front()
    t1 = {**_person_front(), "left_wrist": (0.4, 0.12)}
    det = HeuristicMVPActionDetector(HeuristicMVPConfig(min_confidence=0.28))
    sigs = det.process_frame(0.1, "x", t1, t0, 0.05)
    assert any(s.action_type == ActionType.PUNCH_ACTIVITY for s in sigs)
    pconfs = [s for s in sigs if s.action_type == ActionType.PUNCH_ACTIVITY]
    assert pconfs[0].evidence.get("max_wrist_speed", 0) is not None


def test_mvp_kick() -> None:
    t0 = _person_front()
    t1 = {**_person_front(), "left_ankle": (0.2, 0.18), "right_ankle": (0.59, 0.89)}
    det = HeuristicMVPActionDetector(HeuristicMVPConfig(min_confidence=0.3))
    sigs = det.process_frame(0.0, "1", t1, t0, 0.04)
    assert any(s.action_type == ActionType.KICK_ACTIVITY for s in sigs)


def test_run_sequence_mvp() -> None:
    seq = [
        _person_front(),
        {**_person_front(), "left_wrist": (0.4, 0.1)},
    ]
    times = [0.0, 0.05]
    d = HeuristicMVPActionDetector(HeuristicMVPConfig(min_confidence=0.3))
    out = run_sequence_mvp(times, seq, fighter_id="0", detector=d)
    have_punch = any(s.action_type == ActionType.PUNCH_ACTIVITY for s in out)
    assert have_punch


def test_types_literal_values() -> None:
    for name in (
        "PUNCH_ACTIVITY",
        "KICK_ACTIVITY",
        "LOW_GUARD",
        "TURNED_BACK",
        "DEFENSIVE_INCAPACITY",
    ):
        assert name in {x.value for x in ActionType}
