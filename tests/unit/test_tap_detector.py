"""
Synthetic skeleton tests for interpretable tap-out candidate detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.events.tap_detector import (
    EVENT_FOOT_TAP,
    EVENT_HAND_TAP,
    TapCandidateEvent,
    TapDetectorConfig,
    detect_tap_candidates,
)


def _assert_tap_candidate_contract(ev: TapCandidateEvent) -> None:
    """Every emitted candidate must expose TapKO audit fields and negative-aware evidence."""
    d = ev.to_dict()
    assert d["requires_human_confirmation"] is True
    for key in (
        "event_type",
        "start_time",
        "end_time",
        "score",
        "repetition_count",
        "evidence",
        "explanation",
    ):
        assert key in d
    evd = d["evidence"]
    for key in (
        "hand_posting_risk",
        "scramble_risk",
        "single_contact_rejected",
        "contact_band_passed",
    ):
        assert key in evd


# COCO-17 indices (must match tap_detector).
_L_SH, _R_SH = 5, 6
_L_EL, _R_EL = 7, 8
_L_WR, _R_WR = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KN, _R_KN = 13, 14
_L_ANK, _R_ANK = 15, 16


def _base_pose_row(
    *,
    wrist_y: float,
    shoulder_y: float = 0.34,
    hip_y: float = 0.54,
    foot_y: float = 0.91,
    x_center: float = 0.52,
) -> np.ndarray:
    xy = np.zeros((17, 2), dtype=np.float64)
    for ix in range(17):
        xy[ix, 0] = x_center + (ix - 8) * 0.002
    xy[_L_SH, 1] = shoulder_y
    xy[_R_SH, 1] = shoulder_y
    xy[_L_EL, 1] = (shoulder_y + wrist_y) * 0.5
    xy[_R_EL, 1] = (shoulder_y + wrist_y) * 0.5
    xy[_L_WR, 1] = wrist_y
    xy[_R_WR, 1] = wrist_y
    xy[_L_HIP, 1] = hip_y
    xy[_R_HIP, 1] = hip_y
    xy[_L_KN, 1] = (hip_y + foot_y) * 0.5
    xy[_R_KN, 1] = (hip_y + foot_y) * 0.5
    xy[_L_ANK, 1] = foot_y
    xy[_R_ANK, 1] = foot_y
    return xy


def _inject_hand_spike(kp: np.ndarray, t0: int, t1: int, spike_y: float) -> None:
    """Sharp downward wrist motion (image y increases) then recovery."""
    for t in range(t0, min(t1 + 1, kp.shape[0])):
        alpha = (t - t0) / max(1, (t1 - t0))
        y = kp[t, _L_WR, 1] * (1 - alpha) + spike_y * alpha
        kp[t, _L_WR, 1] = y
        kp[t, _R_WR, 1] = y


def test_hand_tap_two_impulses() -> None:
    """Two repeated downward wrist bursts while hands remain in mat-contact band."""
    fps = 30.0
    t_n = 120
    kp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(t_n)])
    _inject_hand_spike(kp, 22, 24, 0.93)
    _inject_hand_spike(kp, 52, 54, 0.92)

    cfg = TapDetectorConfig(
        min_score_emit=0.06,
        repetition_window_sec=2.0,
        min_hand_mat_band_ratio=0.48,
        peak_percentile_hand=82.0,
    )
    ev = detect_tap_candidates(kp, fps, config=cfg)
    hand = [e for e in ev if e.event_type == EVENT_HAND_TAP]
    assert hand, "expected hand tap candidate"
    assert hand[0].repetition_count >= 2
    assert hand[0].requires_human_confirmation is True
    assert "not an official" in hand[0].explanation.lower()
    assert hand[0].score > 0.0
    _assert_tap_candidate_contract(hand[0])
    assert hand[0].evidence["single_contact_rejected"] is False


def test_foot_tap_two_impulses_arms_trapped_context() -> None:
    """Ankle oscillation bursts while wrists are pinned near shoulders (foot tap proxy)."""
    fps = 30.0
    t_n = 140
    kp = np.zeros((t_n, 17, 2), dtype=np.float64)
    for t in range(t_n):
        row = _base_pose_row(wrist_y=0.37, shoulder_y=0.34, hip_y=0.52, foot_y=0.90, x_center=0.5)
        row[_L_EL, 1] = 0.39
        row[_R_EL, 1] = 0.39
        kp[t] = row
    # Two ankle ``stomp'' oscillations
    for t in range(38, 42):
        kp[t, _L_ANK, 1] = 0.91 + 0.04 * np.sin((t - 38) * 1.2)
        kp[t, _R_ANK, 1] = 0.91 + 0.04 * np.sin((t - 38) * 1.2)
    for t in range(72, 76):
        kp[t, _L_ANK, 1] = 0.91 + 0.045 * np.sin((t - 72) * 1.3)
        kp[t, _R_ANK, 1] = 0.91 + 0.045 * np.sin((t - 72) * 1.3)

    cfg = TapDetectorConfig(
        min_score_emit=0.05,
        repetition_window_sec=2.5,
        peak_percentile_foot=78.0,
        velocity_percentile_foot=65.0,
    )
    ev = detect_tap_candidates(kp, fps, config=cfg)
    foot = [e for e in ev if e.event_type == EVENT_FOOT_TAP]
    assert foot, "expected foot tap candidate"
    assert foot[0].repetition_count >= 2
    _assert_tap_candidate_contract(foot[0])
    assert foot[0].evidence["single_contact_rejected"] is False


def test_single_slap_rejected_no_hand_tap_event() -> None:
    """A lone impulse yields fewer clustered peaks than ``min_repetitions`` (single slap path)."""
    fps = 30.0
    t_n = 100
    kp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(t_n)])
    kp[45, _L_WR, 1] = 0.93
    kp[45, _R_WR, 1] = 0.93

    cfg = TapDetectorConfig(
        min_score_emit=0.06,
        repetition_window_sec=2.0,
        min_hand_mat_band_ratio=0.48,
        peak_percentile_hand=82.0,
        smooth_window=1,
        min_repetitions=3,
    )
    ev = detect_tap_candidates(kp, fps, config=cfg)
    assert not [e for e in ev if e.event_type == EVENT_HAND_TAP]


def test_foot_movement_escape_rejected_no_foot_tap() -> None:
    """Busy ankle jitter with arms extended (not trapped) — no rhythmic foot-tap cluster."""
    fps = 30.0
    t_n = 120
    kp = np.zeros((t_n, 17, 2), dtype=np.float64)
    rng = np.random.default_rng(2)
    for t in range(t_n):
        row = _base_pose_row(wrist_y=0.50, shoulder_y=0.34, hip_y=0.52, foot_y=0.90, x_center=0.5)
        row[_L_WR, 0] = 0.28
        row[_R_WR, 0] = 0.72
        row[_L_WR, 1] = 0.56
        row[_R_WR, 1] = 0.56
        kp[t] = row
        kp[t, _L_ANK, 1] = 0.90 + rng.normal(0, 0.015)
        kp[t, _R_ANK, 1] = 0.90 + rng.normal(0, 0.015)

    cfg = TapDetectorConfig(
        min_score_emit=0.40,
        peak_percentile_foot=99.0,
        velocity_percentile_foot=88.0,
        peak_height_fraction_of_gate=0.55,
    )
    ev = detect_tap_candidates(kp, fps, config=cfg)
    assert not [e for e in ev if e.event_type == EVENT_FOOT_TAP]


def test_hand_posting_rejected_no_hand_tap_event() -> None:
    """Slow posting motion: no repeated impulses above threshold."""
    fps = 30.0
    t_n = 100
    kp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(t_n)])
    for t in range(t_n):
        kp[t, _L_WR, 1] = 0.62 + 0.002 * t  # gentle drift
        kp[t, _R_WR, 1] = 0.62 + 0.002 * t

    cfg = TapDetectorConfig(min_score_emit=0.15, peak_percentile_hand=95.0)
    ev = detect_tap_candidates(kp, fps, config=cfg)
    assert not [e for e in ev if e.event_type == EVENT_HAND_TAP]


def test_normal_scramble_rejected_no_hand_tap_event() -> None:
    """High wrist speed but wrists stay mid-torso — mat gate suppresses hand tap channel."""
    fps = 30.0
    t_n = 90
    kp = np.stack(
        [_base_pose_row(wrist_y=0.48, shoulder_y=0.34, hip_y=0.54, foot_y=0.91) for _ in range(t_n)]
    )
    rng = np.random.default_rng(0)
    for t in range(5, t_n - 5):
        jitter = rng.normal(0, 0.06, size=(2,))
        kp[t, _L_WR, :2] += jitter
        kp[t, _R_WR, :2] += jitter

    cfg = TapDetectorConfig(min_score_emit=0.05)
    ev = detect_tap_candidates(kp, fps, config=cfg)
    assert not [e for e in ev if e.event_type == EVENT_HAND_TAP]


def test_opponent_keypoints_shape_mismatch_raises() -> None:
    kp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(40)])
    bad_opp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(10)])
    with pytest.raises(ValueError, match="same T"):
        detect_tap_candidates(kp, 30.0, opponent_keypoints=bad_opp)
