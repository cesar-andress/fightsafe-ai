"""
Extract frames from a tiny on-disk MP4 (OpenCV only; no network).

Covers :func:`fightsafe_ai.video.frame_extractor.extract_frames` in a real I/O path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.synthetic_video import write_tiny_mp4

from fightsafe_ai.exceptions import VideoIOError


pytestmark = [pytest.mark.integration]


def test_extract_frames_from_tiny_synthetic_video(tmp_path: Path) -> None:
    from fightsafe_ai.video.frame_extractor import extract_frames

    try:
        vid = write_tiny_mp4(tmp_path / "clip.mp4", n_frames=8, fps=20.0)
    except OSError as exc:
        pytest.skip(f"OpenCV VideoWriter unavailable: {exc}")

    out = tmp_path / "frames"
    out.mkdir()
    try:
        paths = extract_frames(vid, out, fps=2)
    except VideoIOError as e:
        pytest.skip(f"OpenCV cannot read synthetic mp4: {e}")
    assert len(paths) >= 1
    assert all(p.suffix.lower() == ".jpg" for p in paths)
    first = out / "frame_000001.jpg"
    assert first.is_file() and first.stat().st_size > 0
