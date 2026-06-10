"""
Unit tests: torso-shoulder angle and hip center geometry (synthetic, deterministic data).
"""

from __future__ import annotations

import math

import pytest

from fightsafe_ai.features.biomechanics import compute_body_centers, compute_torso_angle


def test_torso_angle_matches_arctan2_geometry() -> None:
    """Torso angle (hip to shoulder) vs. vertical should match atan2(dx, -dy) in degrees."""
    # Upright: shoulder above hip in image coordinates (y down)
    sm = (0.5, 0.2)
    hm = (0.5, 0.7)
    deg = compute_torso_angle(sm, hm)
    assert deg == pytest.approx(0.0, abs=1e-6)

    # 45°: hip to shoulder (dx=1, dy=-1) → atan2(1,1) = 45°
    hip = (0.0, 0.0)
    shoulder = (1.0, -1.0)
    assert compute_torso_angle(shoulder, hip) == pytest.approx(45.0, abs=1e-5)

    # 90°: pure horizontal (sign depends on left/right)
    assert abs(compute_torso_angle((0.0, 0.5), (0.5, 0.5))) == pytest.approx(90.0, abs=1e-5)

    # Missing point: NaN
    assert math.isnan(compute_torso_angle(None, (0.0, 0.0)))


def test_hip_center_is_midpoint_of_bilateral_hips() -> None:
    """hip_center_x / hip_center_y are the mean of left_hip and right_hip (pair required)."""
    c = compute_body_centers(
        {
            "left_hip": (0.0, 0.40),
            "right_hip": (0.60, 0.40),
            "left_shoulder": (0.1, 0.1),
            "right_shoulder": (0.9, 0.1),
            "nose": (0.5, 0.05),
        }
    )
    assert c["hip_center_x"] == pytest.approx(0.3)
    assert c["hip_center_y"] == pytest.approx(0.40)

    # Without a hip pair: no hip center (NaN)
    partial = compute_body_centers(
        {
            "left_hip": (0.0, 0.5),
            "nose": (0.5, 0.0),
        }
    )
    assert math.isnan(partial["hip_center_x"])
    assert math.isnan(partial["hip_center_y"])


def test_torso_angle_uses_shoulder_hip_from_centers_pipeline() -> None:
    """Flow: landmarks → body centers → angle matches synthetic layout."""
    lm = {
        "left_shoulder": (0.4, 0.2),
        "right_shoulder": (0.6, 0.2),
        "left_hip": (0.45, 0.7),
        "right_hip": (0.55, 0.7),
        "nose": (0.5, 0.1),
    }
    centers = compute_body_centers(lm)
    sm = (centers["shoulder_center_x"], centers["shoulder_center_y"])
    hm = (centers["hip_center_x"], centers["hip_center_y"])
    ang = compute_torso_angle(
        (float(sm[0]), float(sm[1])),
        (float(hm[0]), float(hm[1])),
    )
    # Centered shoulders (0.5,0.2) over hips (0.5,0.7) → near vertical, ~0°
    assert ang == pytest.approx(0.0, abs=1e-5)
    # Nudge left shoulder on X: shoulder center shifts; angle is non-zero
    lm2 = {**lm, "left_shoulder": (0.0, 0.2)}
    c2 = compute_body_centers(lm2)
    a2 = compute_torso_angle(
        (c2["shoulder_center_x"], c2["shoulder_center_y"]),
        (c2["hip_center_x"], c2["hip_center_y"]),
    )
    assert not math.isnan(a2) and abs(a2) > 0.1
