""":func:`fightsafe_ai.video.writer.stitch_jpeg_folder_to_mp4` (OpenCV, small JPEGs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fightsafe_ai.video.writer import stitch_jpeg_folder_to_mp4


pytestmark = pytest.mark.unit


def test_stitch_jpeg_folder_creates_mp4(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    d = tmp_path / "jpegs"
    d.mkdir()
    for i, name in enumerate(["frame_01.jpg", "frame_02.jpg", "frame_10.jpg"]):
        img = np.full((32, 48, 3), (i * 40) % 255, dtype=np.uint8)
        assert cv2.imwrite(str(d / name), img)
    out = tmp_path / "out.mp4"
    p = stitch_jpeg_folder_to_mp4(d, out, fps=5.0)
    assert p.is_file() and p.stat().st_size > 0


def test_stitch_raises_without_jpegs(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(Exception) as exc:
        stitch_jpeg_folder_to_mp4(d, tmp_path / "x.mp4", fps=10.0)
    assert type(exc.value).__name__ == "VideoIOError"
    assert "JPEG" in str(exc.value) and "preview" in str(exc.value).lower()


def test_stitch_rejects_non_positive_fps(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="fps must be positive"):
        stitch_jpeg_folder_to_mp4(d, tmp_path / "o.mp4", fps=0.0)
