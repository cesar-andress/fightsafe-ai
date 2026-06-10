"""Tests for :mod:`fightsafe_ai.keypoints.io` paths that avoid MediaPipe indexed loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.keypoints.io import load_keypoint_csv, load_landmark_maps_ordered


def test_load_keypoint_csv_reads_legacy_columns(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_text("landmark,x,y\nnose,0.5,0.4\n", encoding="utf-8")
    m = load_keypoint_csv(p)
    assert m is not None
    assert m["nose"] == (0.5, 0.4)


def test_load_keypoint_csv_accepts_keypoint_name_column(tmp_path: Path) -> None:
    p = tmp_path / "g.csv"
    p.write_text("keypoint_name,x,y\nnose,0.2,0.3\n", encoding="utf-8")
    m = load_keypoint_csv(p)
    assert m is not None
    assert m["nose"] == (0.2, 0.3)


def test_load_keypoint_csv_skips_unparseable_xy(tmp_path: Path) -> None:
    p = tmp_path / "badxy.csv"
    p.write_text("landmark,x,y\nnose,notafloat,0.1", encoding="utf-8")
    assert load_keypoint_csv(p) is None


def test_load_keypoint_csv_skips_empty_name_row(tmp_path: Path) -> None:
    p = tmp_path / "emptynam.csv"
    p.write_text("landmark,x,y\n,0.1,0.2\nnose,0.3,0.4\n", encoding="utf-8")
    m = load_keypoint_csv(p)
    assert m is not None
    assert list(m.keys()) == ["nose"] and m["nose"] == (0.3, 0.4)


def test_load_keypoint_csv_empty_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "e.csv"
    p.write_text("landmark,x,y\n", encoding="utf-8")
    assert load_keypoint_csv(p) is None


def test_load_keypoint_csv_missing_path_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "this_file_does_not_exist.csv"
    assert load_keypoint_csv(p) is None


def test_load_landmark_maps_keypoints_path_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(VideoIOError, match=r"directory or a \.csv file"):
        load_landmark_maps_ordered(missing)


def test_load_landmark_maps_rejects_consolidated_missing_columns(tmp_path: Path) -> None:
    c = tmp_path / "bad.csv"
    c.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(VideoIOError, match="Consolidated CSV missing columns"):
        load_landmark_maps_ordered(c)


def test_load_landmark_maps_ordered_directory(tmp_path: Path) -> None:
    f1 = tmp_path / "frame_001.csv"
    f2 = tmp_path / "frame_002.csv"
    f1.write_text("landmark,x,y\nnose,0.1,0.1\n", encoding="utf-8")
    f2.write_text("landmark,x,y\nnose,0.2,0.2\n", encoding="utf-8")
    out = load_landmark_maps_ordered(tmp_path, glob_pattern="*.csv")
    assert len(out) == 2
    assert out[0][0] == "frame_001.csv"


def test_load_landmark_maps_ordered_consolidated(tmp_path: Path) -> None:
    c = tmp_path / "all.csv"
    c.write_text(
        "frame_id,keypoint_name,x,y\na,nose,0.1,0.2\na,left_shoulder,0.2,0.3\n",
        encoding="utf-8",
    )
    out = load_landmark_maps_ordered(c)
    assert len(out) == 1
    assert out[0][0] == "a"
    assert out[0][1] is not None
    assert "nose" in out[0][1]


def test_load_landmark_maps_ordered_rejects_non_csv_file(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(VideoIOError, match=r"must be a \.csv file"):
        load_landmark_maps_ordered(p)
