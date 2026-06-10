"""Unit tests for pure biomechanics helpers (isolated import, no MediaPipe on import)."""

import math

import pandas as pd
import pytest
from tests.support.isolated import load_biomechanics


_m = load_biomechanics()
compute_body_centers = _m.compute_body_centers
compute_body_height_proxy = _m.compute_body_height_proxy
compute_is_low_posture = _m.compute_is_low_posture
compute_torso_angle = _m.compute_torso_angle
compute_biomechanical_features = _m.compute_biomechanical_features
compute_guard_and_facing_scores = _m.compute_guard_and_facing_scores


def test_compute_guard_and_facing_scores() -> None:
    lm = {
        "nose": (0.5, 0.2),
        "left_shoulder": (0.4, 0.3),
        "right_shoulder": (0.6, 0.3),
        "left_wrist": (0.45, 0.45),
        "right_wrist": (0.55, 0.45),
    }
    g, f = compute_guard_and_facing_scores(lm, 0.5)
    assert 0.0 <= g <= 1.0 and 0.0 <= f <= 1.0
    assert g > 0.05
    assert compute_guard_and_facing_scores(None, 0.5) == (0.0, 0.0)


def test_compute_body_centers_full() -> None:
    lm = {
        "left_shoulder": (0.4, 0.3),
        "right_shoulder": (0.6, 0.3),
        "left_hip": (0.42, 0.55),
        "right_hip": (0.58, 0.55),
        "nose": (0.5, 0.25),
    }
    c = compute_body_centers(lm)
    assert c["shoulder_center_x"] == pytest.approx(0.5)
    assert c["shoulder_center_y"] == pytest.approx(0.3)
    assert c["hip_center_x"] == pytest.approx(0.5)
    assert c["hip_center_y"] == pytest.approx(0.55)
    assert c["head_x"] == pytest.approx(0.5)
    assert c["head_y"] == pytest.approx(0.25)


def test_compute_body_centers_missing_pair() -> None:
    c = compute_body_centers({"left_shoulder": (0.4, 0.3), "nose": (0.5, 0.25)})
    assert math.isnan(c["shoulder_center_x"])
    assert math.isnan(c["hip_center_y"])


def test_compute_torso_angle_upright() -> None:
    sm = (0.5, 0.25)
    hm = (0.5, 0.55)
    deg = compute_torso_angle(sm, hm)
    assert deg == pytest.approx(0.0, abs=1e-6)


def test_compute_torso_angle_missing() -> None:
    assert math.isnan(compute_torso_angle(None, (0.5, 0.5)))
    assert math.isnan(compute_torso_angle((0.5, 0.5), None))


def test_compute_body_height_proxy() -> None:
    lm = {
        "nose": (0.5, 0.2),
        "left_shoulder": (0.4, 0.3),
        "right_shoulder": (0.6, 0.3),
        "left_hip": (0.45, 0.6),
        "right_hip": (0.55, 0.6),
        "left_ankle": (0.46, 0.95),
        "right_ankle": (0.54, 0.95),
    }
    h = compute_body_height_proxy(lm)
    assert h == pytest.approx(0.95 - 0.2)


def test_compute_is_low_posture() -> None:
    assert compute_is_low_posture(0.7, threshold=0.58) is True
    assert compute_is_low_posture(0.4, threshold=0.58) is False
    assert compute_is_low_posture(float("nan"), threshold=0.58) is False


def test_compute_biomechanical_features_long_df() -> None:
    df = pd.DataFrame(
        [
            {"frame_id": "a", "keypoint_name": "nose", "x": 0.5, "y": 0.2},
            {"frame_id": "a", "keypoint_name": "left_shoulder", "x": 0.4, "y": 0.3},
            {"frame_id": "a", "keypoint_name": "right_shoulder", "x": 0.6, "y": 0.3},
            {"frame_id": "a", "keypoint_name": "left_hip", "x": 0.42, "y": 0.55},
            {"frame_id": "a", "keypoint_name": "right_hip", "x": 0.58, "y": 0.55},
            {"frame_id": "a", "keypoint_name": "left_ankle", "x": 0.44, "y": 0.9},
            {"frame_id": "a", "keypoint_name": "right_ankle", "x": 0.56, "y": 0.9},
        ]
    )
    out = compute_biomechanical_features(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["frame_id"] == "a"
    assert not math.isnan(row["torso_angle_degrees"])
    assert row["hip_vertical_position"] == pytest.approx(0.55)
    assert row["head_vertical_position"] == pytest.approx(0.2)


def test_compute_biomechanical_features_requires_columns() -> None:
    with pytest.raises(ValueError, match="must contain"):
        compute_biomechanical_features(pd.DataFrame({"frame_id": [1]}))
