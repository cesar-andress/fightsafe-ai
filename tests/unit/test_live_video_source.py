"""Unit tests for FileVideoSource."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fightsafe_ai.live.video_source import (
    FileVideoSource,
    VideoFrameMeta,
    WebcamSource,
    open_video_source,
)


pytestmark = pytest.mark.unit


def _write_minimal_avi(path: Path, *, n_frames: int = 5, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    h, w = 48, 64
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert out.isOpened(), "VideoWriter failed (codec/environment)"
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_file_video_source_reads_frames_and_meta(tmp_path: Path) -> None:
    avi = tmp_path / "tiny.avi"
    _write_minimal_avi(avi, n_frames=5, fps=10.0)

    src = FileVideoSource(avi, realtime=False, fps_fallback=30.0)
    try:
        assert src.width == 64
        assert src.height == 48
        assert abs(src.fps - 10.0) < 0.05
        assert src.frame_index == -1
        if src.total_frames is not None:
            assert src.total_frames == 5
        if src.duration_seconds is not None:
            assert src.duration_seconds == pytest.approx(0.5, abs=0.05)

        frames_read = 0
        while True:
            frame, meta = src.read_frame()
            if frame is None:
                break
            assert isinstance(meta, VideoFrameMeta)
            assert meta.frame_index == frames_read
            assert meta.timestamp_seconds == pytest.approx(frames_read / src.fps)
            assert frame.shape == (48, 64, 3)
            frames_read += 1
        assert frames_read == 5
        assert src.frame_index == 4
        assert src.timestamp_seconds == pytest.approx(4 / src.fps)
    finally:
        src.close()


def test_read_alias_matches_read_frame(tmp_path: Path) -> None:
    avi = tmp_path / "readalias.avi"
    _write_minimal_avi(avi, n_frames=1)
    src_a = FileVideoSource(avi, realtime=False)
    src_b = FileVideoSource(avi, realtime=False)
    try:
        a, ma = src_a.read()
        b, mb = src_b.read_frame()
        assert a is not None and b is not None
        assert np.array_equal(a, b)
        assert ma == mb
    finally:
        src_a.close()
        src_b.close()


def test_file_video_source_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        FileVideoSource("/nonexistent/path/video_xyz.mp4")


def test_realtime_invokes_sleep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    avi = tmp_path / "t2.avi"
    _write_minimal_avi(avi, n_frames=3, fps=100.0)

    sleeps: list[float] = []
    monkeypatch.setattr("fightsafe_ai.live.video_source.time.sleep", sleeps.append)

    src = FileVideoSource(avi, realtime=True, fps_fallback=30.0)
    try:
        src.read_frame()
        src.read_frame()
        assert len(sleeps) >= 1
    finally:
        src.close()


def test_context_manager_closes(tmp_path: Path) -> None:
    avi = tmp_path / "t3.avi"
    _write_minimal_avi(avi, n_frames=1)
    with FileVideoSource(avi) as src:
        frame, meta = src.read_frame()
        assert frame is not None and meta is not None
    with FileVideoSource(avi) as src2:
        assert src2.read_frame()[0] is not None


def test_open_video_source_digit_selects_webcam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numeric-only token opens WebcamSource (cv2.VideoCapture(index))."""

    class FakeCapture:
        def __init__(self, index: int) -> None:
            self.index = index

        def isOpened(self) -> bool:
            return True

        def get(self, prop_id: int) -> float:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 320.0
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 240.0
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((240, 320, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("fightsafe_ai.live.video_source.cv2.VideoCapture", FakeCapture)

    src = open_video_source("0", realtime=False)
    assert isinstance(src, WebcamSource)
    assert src.device_index == 0
    frame, meta = src.read_frame()
    assert frame is not None and meta is not None
    assert meta.frame_index == 0
    src.close()


def test_open_video_source_path_selects_file(tmp_path: Path) -> None:
    avi = tmp_path / "openme.avi"
    _write_minimal_avi(avi, n_frames=2)
    src = open_video_source(str(avi))
    assert isinstance(src, FileVideoSource)
    src.close()
