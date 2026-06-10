"""Indexed landmark loading (MediaPipe names) for :mod:`fightsafe_ai.keypoints.io`."""

from __future__ import annotations

from pathlib import Path

import pytest

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.keypoints.io import (
    load_indexed_sequence,
    load_keypoint_csv,
    load_keypoint_csv_indexed,
    load_landmark_maps_ordered,
)


pytestmark = pytest.mark.unit


def _require_mediapipe_pose() -> None:
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker

        _ = PoseLandmarker
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"mediapipe PoseLandmarker (Tasks) not available: {exc}")


def test_load_keypoint_csv_indexed_maps_mediapipe_names(tmp_path: Path) -> None:
    _require_mediapipe_pose()
    p = tmp_path / "f1.csv"
    p.write_text(
        "landmark,x,y,visibility\nnose,0.5,0.1,0.99\nleft_hip,0.4,0.6,0.9\nright_hip,0.6,0.6,0.9\n",
        encoding="utf-8",
    )
    m = load_keypoint_csv_indexed(p)
    assert len(m) >= 3
    assert all(isinstance(v, tuple) and len(v) == 3 for v in m.values())


def test_load_keypoint_csv_keypoint_name_column(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_text("keypoint_name,x,y\nnose,0.2,0.3\n", encoding="utf-8")
    m = load_keypoint_csv(p)
    assert m is not None and m.get("nose") == (0.2, 0.3)


def test_load_indexed_sequence_per_frame_dir(tmp_path: Path) -> None:
    _require_mediapipe_pose()
    d = tmp_path / "seq"
    d.mkdir()
    (d / "frame_2.csv").write_text("landmark,x,y\nnose,0.1,0.1\n", encoding="utf-8")
    (d / "frame_10.csv").write_text("landmark,x,y\nnose,0.2,0.2\n", encoding="utf-8")
    seq = load_indexed_sequence(d, glob_pattern="*.csv")
    assert len(seq) == 2
    assert len(seq[0]) >= 1 and len(seq[1]) >= 1


def test_load_indexed_sequence_consolidated_csv(tmp_path: Path) -> None:
    _require_mediapipe_pose()
    c = tmp_path / "pose.csv"
    c.write_text(
        "frame_id,keypoint_name,x,y\n"
        "0,nose,0.5,0.1\n"
        "0,left_hip,0.4,0.5\n"
        "1,nose,0.51,0.11\n"
        "1,left_hip,0.41,0.51\n",
        encoding="utf-8",
    )
    seq = load_indexed_sequence(c)
    assert len(seq) == 2
    assert len(seq[0]) >= 1


def test_load_landmark_maps_rejects_non_csv_path(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(VideoIOError, match=r"\.csv"):
        load_landmark_maps_ordered(p)
