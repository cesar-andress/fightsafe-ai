"""Synthetic pose sequences for anomaly MVP detectors (no video, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.anomaly.base import AnomalyType, pose_sequence_from_landmark_dicts
from fightsafe_ai.anomaly.fall_detector import (
    FallDetector,
    FallDetectorConfig,
    fall_likelihood_from_y_coords,
)
from fightsafe_ai.anomaly.inactivity_detector import (
    InactivityDetector,
    InactivityDetectorConfig,
    inactivity_score,
)
from fightsafe_ai.anomaly.limb_anomaly import LimbAnomalyDetector, LimbAnomalyDetectorConfig
from fightsafe_ai.anomaly.surrender_detector import SurrenderAnomalyDetector
from fightsafe_ai.risk.surrender import SurrenderHeuristicConfig


pytestmark = pytest.mark.unit


def _body(
    *,
    hip_y: float = 0.5,
    shoulder_y: float = 0.25,
    wrist_y: float = 0.4,
) -> dict[str, tuple[float, float]]:
    return {
        "nose": (0.5, 0.12),
        "left_shoulder": (0.4, shoulder_y),
        "right_shoulder": (0.6, shoulder_y),
        "left_elbow": (0.38, 0.35),
        "right_elbow": (0.62, 0.35),
        "left_wrist": (0.36, wrist_y),
        "right_wrist": (0.64, wrist_y),
        "left_hip": (0.42, hip_y),
        "right_hip": (0.58, hip_y),
        "left_knee": (0.43, 0.68),
        "right_knee": (0.57, 0.68),
        "left_ankle": (0.44, 0.9),
        "right_ankle": (0.56, 0.9),
    }


def test_fall_likelihood_unchanged() -> None:
    assert fall_likelihood_from_y_coords(0.5, 0.5, ground_y=0.82) == 0.0
    assert fall_likelihood_from_y_coords(0.9, None, ground_y=0.82) > 0.0


def test_inactivity_score_unchanged() -> None:
    assert inactivity_score([0.0, 0.0, 0.0], threshold=0.02) == 1.0
    assert inactivity_score([1.0, 1.0], threshold=0.02) == 0.0


def test_pose_sequence_from_dicts() -> None:
    seq = pose_sequence_from_landmark_dicts([_body(), _body(hip_y=0.55)])
    assert len(seq) == 2
    assert any(k.name == "nose" for k in seq[0].keypoints)


def test_fall_rapid_and_prolonged() -> None:
    t0, t1 = 0.0, 0.1
    f0 = _body(hip_y=0.4)
    f1 = {
        **_body(hip_y=0.4),
        "left_hip": (0.42, 0.9),
        "right_hip": (0.58, 0.9),
        "nose": (0.5, 0.85),
    }
    d = FallDetector(
        FallDetectorConfig(
            min_descent_vy=0.5,
            min_descent_confidence=0.2,
            min_prolonged_confidence=0.25,
            min_prolonged_frames=2,
        )
    )
    sigs = d.analyze([t0, t1], [f0, f1], "0")
    assert any(s.anomaly_type == AnomalyType.FALL_RAPID_DOWNWARD for s in sigs)

    # prolonged low: several very low (high y) postures
    long_frames = [
        {**_body(hip_y=0.9), "nose": (0.5, 0.9)},
    ] * 5
    times = [i * 0.04 for i in range(5)]
    s2 = d.analyze(times, long_frames, "1")
    assert any(s.anomaly_type == AnomalyType.FALL_PROLONGED_LOW_POSTURE for s in s2) or any(
        s.anomaly_type == AnomalyType.FALL_RAPID_DOWNWARD for s in s2
    )


def test_fall_torso_collapse() -> None:
    d = FallDetector(
        FallDetectorConfig(min_torso_collapse_confidence=0.12, torso_collapse_delta_deg=4.0)
    )
    # Asymmetric shoulders shift shoulder_center.x vs fixed hips -> torso angle per frame changes.
    f0 = _body(hip_y=0.5)
    f1 = {**f0, "left_shoulder": (0.2, 0.22), "right_shoulder": (0.35, 0.35)}
    sigs = d.analyze([0.0, 0.05], [f0, f1], "x")
    assert any(s.anomaly_type == AnomalyType.FALL_TORSO_ANGLE_COLLAPSE for s in sigs)


def test_inactivity() -> None:
    # Detector requires >= min_duration_seconds (default 2s) of clip history before flagging.
    n = 25
    stat = [_body(hip_y=0.5 + i * 0.0001) for i in range(n)]
    t = [i * 0.1 for i in range(n)]
    det = InactivityDetector(
        InactivityDetectorConfig(min_keypoint_displacement=0.08, min_confidence=0.1)
    )
    sigs = det.analyze(t, stat, "0")
    assert any(s.anomaly_type == AnomalyType.INACTIVITY_LOW_KEYPOINT_MOTION for s in sigs) or any(
        s.anomaly_type == AnomalyType.INACTIVITY_LOW_COM_MOTION for s in sigs
    )


def test_limb_asym_and_support() -> None:
    cfg = LimbAnomalyDetectorConfig(
        knee_stress_threshold=0.99,
        asym_threshold=0.2,
        collapse_threshold=0.1,
        min_confidence=0.15,
    )
    det = LimbAnomalyDetector(cfg)
    a = _body(hip_y=0.5)
    b = {
        **a,
        "left_knee": (0.43, 0.5),
        "right_knee": (0.57, 0.5),
    }
    # one frame jump in knee config + sudden ankle
    a2 = b
    b2 = {
        **b,
        "left_ankle": (0.44, 0.99),
        "right_ankle": (0.56, 0.99),
    }
    s = det.analyze([0.0, 0.05], [a2, b2], "0")
    assert s and any(
        t
        in (
            AnomalyType.LIMB_SUDDEN_SUPPORT_LOSS,
            AnomalyType.LIMB_BILATERAL_ASYMMETRY,
            AnomalyType.LIMB_JOINT_ANGULAR_ANOMALY,
        )
        for t in (x.anomaly_type for x in s)
    )

    a3 = {**_body(hip_y=0.5), "left_knee": (0.4, 0.55), "right_knee": (0.6, 0.5)}
    b3 = {
        **a3,
        "left_knee": (0.4, 0.3),
        "right_knee": (0.6, 0.5),
    }
    s2 = det.analyze([0.0, 0.05], [a3, b3], "0")
    assert s2  # at least one limb signal
    assert any(
        t in (x.anomaly_type for x in s2)
        for t in (
            AnomalyType.LIMB_BILATERAL_ASYMMETRY,
            AnomalyType.LIMB_SUDDEN_SUPPORT_LOSS,
            AnomalyType.LIMB_JOINT_ANGULAR_ANOMALY,
        )
    )


def test_surrender_oscillation() -> None:
    n = 12
    t = [i * 0.04 for i in range(n)]
    fr: list[dict[str, tuple[float, float]]] = []
    for i in range(n):
        wy = 0.55 + 0.12 * float(np.sin(i * 1.2)) + 0.02 * i
        f = {
            **_body(hip_y=0.5, wrist_y=wy - 0.1),
            "left_wrist": (0.4, wy),
            "right_wrist": (0.6, wy),
        }
        fr.append(f)
    d = SurrenderAnomalyDetector(
        SurrenderHeuristicConfig(min_frames=4, detect_confidence_threshold=0.45)
    )
    sigs = d.analyze(t, fr, "0")
    assert any(
        s.anomaly_type
        in (
            AnomalyType.SURRENDER_OSCILLATION,
            AnomalyType.SURRENDER_TAP_LIKE,
            AnomalyType.SURRENDER_RHYTHMIC_HANDS,
        )
        for s in sigs
    )
