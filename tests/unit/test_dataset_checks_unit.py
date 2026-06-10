""":mod:`fightsafe_ai.qa.dataset_checks` (small on-disk fixtures)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fightsafe_ai.qa.dataset_checks import (
    count_frame_images,
    coverage_to_results,
    pose_coverage_metrics,
)


pytestmark = pytest.mark.unit


def test_count_frame_images_not_a_dir() -> None:
    assert count_frame_images(Path("/nonexistent/absolute/path/frames")) == 0


def test_count_frame_images_counts_extensions(tmp_path: Path) -> None:
    d = tmp_path / "f"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"1")
    (d / "b.png").write_bytes(b"1")
    (d / "c.txt").write_text("x", encoding="utf-8")
    assert count_frame_images(d) == 2


def test_pose_coverage_no_images(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.mkdir()
    pc, m = pose_coverage_metrics(f, tmp_path / "pose.csv")
    assert pc is None and m.get("reason") == "no_jpeg_in_frames"


def test_pose_coverage_missing_csv(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.mkdir()
    (f / "a.jpg").write_bytes(b"0")
    pc, m = pose_coverage_metrics(f, tmp_path / "nope.csv")
    assert pc is None and m.get("reason") == "pose_csv_missing"


def test_pose_coverage_no_frame_id_column(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.mkdir()
    (f / "a.jpg").write_bytes(b"0")
    other = tmp_path / "o.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    pc, m = pose_coverage_metrics(f, other)
    assert pc == 0.0 and m.get("reason") == "no_frame_id_or_empty_pose"


@patch("fightsafe_ai.qa.dataset_checks.pd.read_csv", side_effect=ValueError("table"))
def test_pose_coverage_read_exception(mock_read_csv: object, tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.mkdir()
    (f / "a.jpg").write_bytes(b"0")
    c = tmp_path / "p.csv"
    c.write_text("ok\n", encoding="utf-8")
    pc, m = pose_coverage_metrics(f, c)
    assert pc is None and "ValueError" in m.get("error", "")


def test_coverage_to_results_branches() -> None:
    r0 = coverage_to_results(None, {"n_frame_image_files": 0, "reason": "no_jpeg_in_frames"})
    assert r0[0].status == "fail"
    r1 = coverage_to_results(40.0, {"n_frame_image_files": 3})
    assert r1[0].status == "warn" and "50" in r1[0].message
    r2 = coverage_to_results(90.0, {"n_frame_image_files": 2})
    assert "not all" in r2[0].message
    r3 = coverage_to_results(100.0, {"n_frame_image_files": 1})
    assert r3[0].status == "pass"


def test_pose_coverage_unique_exceeds_images(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.mkdir()
    (f / "a.jpg").write_bytes(b"0")
    (f / "b.jpg").write_bytes(b"0")
    c = tmp_path / "p.csv"
    c.write_text("frame_id\nf0\nf1\nf2\n", encoding="utf-8")
    pc, m = pose_coverage_metrics(f, c)
    assert pc == 100.0
    assert "n_unique" in m.get("note", "").lower() or m.get("n_unique_pose_frame_ids", 0) == 3
