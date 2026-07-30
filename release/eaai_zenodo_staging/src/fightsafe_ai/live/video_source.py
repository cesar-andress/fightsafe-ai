"""
Abstract video ingestion for live runners (disk files or webcam).

Uses OpenCV :class:`cv2.VideoCapture` for files (FFmpeg) and camera indices.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoFrameMeta:
    """Metadata for one decoded frame."""

    frame_index: int
    timestamp_seconds: float
    fps: float


class VideoSource(ABC):
    """Abstract frame source for live runners."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second (may be inferred from the container)."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""

    @property
    @abstractmethod
    def frame_index(self) -> int:
        """Zero-based index of the **last** successfully read frame, or ``-1`` before first read."""

    @property
    @abstractmethod
    def timestamp_seconds(self) -> float:
        """Timestamp of the **last** frame on the clip timeline (seconds)."""

    @abstractmethod
    def read_frame(self) -> tuple[np.ndarray | None, VideoFrameMeta | None]:
        """
        Read the next frame.

        Returns ``(None, None)`` at end-of-stream.
        """

    def read(self) -> tuple[np.ndarray | None, VideoFrameMeta | None]:
        """Same contract as :meth:`read_frame` (alias for OpenCV-style naming)."""
        return self.read_frame()

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""


class FileVideoSource(VideoSource):
    """
    Read a local video file sequentially with OpenCV.

    Decodes **one frame at a time** via :meth:`read_frame`; the file is **not** loaded
    fully into memory (only the current frame buffer and codec state).

    Parameters
    ----------
    path
        Path to a video file (e.g. MP4, AVI).
    realtime
        If True, sleep between frames so playback follows nominal FPS (wall-clock).
    fps_fallback
        Used when the container reports FPS as 0.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        realtime: bool = False,
        fps_fallback: float = 30.0,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._realtime = realtime
        self._fps_fallback = float(fps_fallback)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self._path}")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps_raw if fps_raw > 1e-6 else self._fps_fallback
        self._width = max(w, 1)
        self._height = max(h, 1)

        self._frame_index = -1
        self._timestamp_seconds = 0.0
        self._last_tick: float | None = None

        fc_raw = float(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fc_raw > 0.5:
            self._total_frames: int | None = round(fc_raw)
        else:
            self._total_frames = None

        self._duration_seconds: float | None = None
        if self._total_frames is not None and self._fps > 1e-6:
            self._duration_seconds = float(self._total_frames) / self._fps
        else:
            dur_prop = getattr(cv2, "CAP_PROP_DURATION", None)
            if dur_prop is not None:
                dur_ms = self._cap.get(dur_prop)
                if dur_ms is not None:
                    d = float(dur_ms)
                    if d > 0:
                        # Many backends report milliseconds; very small values may be seconds.
                        self._duration_seconds = d / 1000.0 if d > 1.0 else d

    @property
    def path(self) -> Path:
        return self._path

    @property
    def total_frames(self) -> int | None:
        """Reported frame count from the container, or ``None`` if unknown."""

        return self._total_frames

    @property
    def duration_seconds(self) -> float | None:
        """Clip duration from ``frame_count / fps`` or container hint; ``None`` if unknown."""

        return self._duration_seconds

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp_seconds(self) -> float:
        return self._timestamp_seconds

    def read_frame(self) -> tuple[np.ndarray | None, VideoFrameMeta | None]:
        if self._realtime and self._frame_index >= 0 and self._last_tick is not None:
            dt_wall = time.perf_counter() - self._last_tick
            slot = 1.0 / self._fps
            sleep_for = slot - dt_wall
            if sleep_for > 0:
                time.sleep(sleep_for)

        ok, frame = self._cap.read()
        self._last_tick = time.perf_counter()

        if not ok or frame is None:
            return None, None

        self._frame_index += 1
        self._timestamp_seconds = self._frame_index / self._fps
        meta = VideoFrameMeta(
            frame_index=self._frame_index,
            timestamp_seconds=self._timestamp_seconds,
            fps=self._fps,
        )
        return frame, meta

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> FileVideoSource:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class WebcamSource(VideoSource):
    """
    Live capture from a webcam using ``cv2.VideoCapture(index)`` (e.g. ``0`` for default camera).

    Behaviour matches :class:`FileVideoSource` for :meth:`read_frame` metadata semantics.
    Stream does not end until capture fails (returns ``(None, None)``).
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        realtime: bool = False,
        fps_fallback: float = 30.0,
    ) -> None:
        self._device_index = int(device_index)
        self._realtime = realtime
        self._fps_fallback = float(fps_fallback)
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open webcam device index {self._device_index}")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps_raw if fps_raw > 1e-6 else self._fps_fallback
        self._width = max(w, 1)
        self._height = max(h, 1)

        self._frame_index = -1
        self._timestamp_seconds = 0.0
        self._last_tick: float | None = None

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp_seconds(self) -> float:
        return self._timestamp_seconds

    def read_frame(self) -> tuple[np.ndarray | None, VideoFrameMeta | None]:
        if self._realtime and self._frame_index >= 0 and self._last_tick is not None:
            dt_wall = time.perf_counter() - self._last_tick
            slot = 1.0 / self._fps
            sleep_for = slot - dt_wall
            if sleep_for > 0:
                time.sleep(sleep_for)

        ok, frame = self._cap.read()
        self._last_tick = time.perf_counter()

        if not ok or frame is None:
            return None, None

        self._frame_index += 1
        self._timestamp_seconds = self._frame_index / self._fps
        meta = VideoFrameMeta(
            frame_index=self._frame_index,
            timestamp_seconds=self._timestamp_seconds,
            fps=self._fps,
        )
        return frame, meta

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> WebcamSource:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def open_video_source(
    source: str | Path,
    *,
    realtime: bool = False,
    fps_fallback: float = 30.0,
) -> VideoSource:
    """
    Build a :class:`VideoSource` from a path or a numeric webcam index string.

    If ``str(source).strip()`` is all digits, opens :class:`WebcamSource` with that index;
    otherwise opens :class:`FileVideoSource` for the given path.
    """
    token = str(source).strip()
    if token.isdigit():
        return WebcamSource(int(token), realtime=realtime, fps_fallback=fps_fallback)
    return FileVideoSource(Path(token), realtime=realtime, fps_fallback=fps_fallback)


__all__ = ["FileVideoSource", "VideoFrameMeta", "VideoSource", "WebcamSource", "open_video_source"]
