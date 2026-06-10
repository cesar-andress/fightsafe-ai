"""
Tiny video files built with OpenCV only (no network, no large assets).

Used by integration tests for :mod:`fightsafe_ai.video.frame_extractor`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_tiny_mp4(path: Path, *, n_frames: int = 6, fps: float = 10.0, size: int = 48) -> Path:
    """
    Write a short solid-color MP4 using ``mp4v`` (OpenCV).

    Returns the resolved path. Skips or fails at runtime if the writer cannot be opened
    (codec issues on some CI images are possible; tests may catch and skip).
    """
    import cv2

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size, size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise OSError(f"VideoWriter could not open: {path}")
    try:
        for i in range(n_frames):
            color = (i * 40 % 255, 64, 128)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = color
            writer.write(frame)
    finally:
        writer.release()
    return path.resolve()
