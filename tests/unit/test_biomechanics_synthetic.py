"""
Biomechanical pipeline on synthetic long-format keypoint tables (isolated import).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tests.support.isolated import load_biomechanics


_bio: Any = None


def _bio_mod() -> Any:
    global _bio
    if _bio is None:
        _bio = load_biomechanics()
    return _bio


def _rows_for_frame(
    fid: str,
    hip_y: float,
    *,
    include_ankles: bool = True,
) -> list[dict[str, Any]]:
    base = [
        {"frame_id": fid, "keypoint_name": "nose", "x": 0.5, "y": 0.2},
        {"frame_id": fid, "keypoint_name": "left_shoulder", "x": 0.4, "y": 0.3},
        {"frame_id": fid, "keypoint_name": "right_shoulder", "x": 0.6, "y": 0.3},
        {"frame_id": fid, "keypoint_name": "left_hip", "x": 0.42, "y": hip_y},
        {"frame_id": fid, "keypoint_name": "right_hip", "x": 0.58, "y": hip_y},
    ]
    if include_ankles:
        base += [
            {"frame_id": fid, "keypoint_name": "left_ankle", "x": 0.44, "y": 0.92},
            {"frame_id": fid, "keypoint_name": "right_ankle", "x": 0.56, "y": 0.92},
        ]
    return base


def test_natural_sort_frame_ids() -> None:
    b = _bio_mod()
    rows: list[dict[str, str | float]] = []
    for fid in ("2", "10", "1"):
        rows.extend(_rows_for_frame(fid, 0.55))
    out = b.compute_biomechanical_features(pd.DataFrame(rows))
    order = out["frame_id"].astype(str).tolist()
    assert order == ["1", "2", "10"]


def test_multi_frame_consistent_geometry() -> None:
    b = _bio_mod()
    rows = _rows_for_frame("f0", 0.5) + _rows_for_frame("f1", 0.52)
    out = b.compute_biomechanical_features(pd.DataFrame(rows))
    assert len(out) == 2
    assert out.iloc[0]["hip_vertical_position"] < out.iloc[1]["hip_vertical_position"]


def test_low_posture_threshold_respected() -> None:
    b = _bio_mod()
    low = _rows_for_frame("a", 0.65)
    high = _rows_for_frame("b", 0.4)
    out = b.compute_biomechanical_features(pd.DataFrame(low + high), low_posture_hip_threshold=0.58)
    m = {str(r["frame_id"]): r["is_low_posture"] for _, r in out.iterrows()}
    assert m["a"] is True and m["b"] is False


def test_empty_long_format() -> None:
    b = _bio_mod()
    out = b.compute_biomechanical_features(pd.DataFrame())
    assert len(out) == 0
    assert "torso_angle_degrees" in out.columns
