"""
Write or assemble video files from image sequences (OpenCV).

Library code uses logging. These helpers support preview generation and
decision-support overlays for human review (not medical diagnosis).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.utils.sorting import natural_sort_paths


logger = logging.getLogger(__name__)


def stitch_jpeg_folder_to_mp4(frames_dir: Path, output_mp4: Path, *, fps: float) -> Path:
    """
    Assemble a video from all ``*.jpg`` and ``*.jpeg`` files under ``frames_dir``.

    Frames are ordered by natural filename sort. Output dimensions match the first
    readable image; later frames are resized to that size if needed.

    Parameters
    ----------
    frames_dir:
        Directory containing JPEG frame files.
    output_mp4:
        Output path (``.mp4``). Parent directories are created if missing.
    fps:
        Frame rate of the output video. Must be positive.

    Returns
    -------
    Path
        Resolved path to the written ``.mp4`` file.

    Raises
    ------
    ValueError
        If ``fps`` is not positive.
    VideoIOError
        If no images are found, the first image cannot be read, or the writer fails.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")

    frames_dir = frames_dir.expanduser().resolve()
    output_mp4 = output_mp4.expanduser().resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    images = natural_sort_paths(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
    )
    if not images:
        msg = f"No JPEG frames under {frames_dir} — cannot build preview video."
        logger.error(msg)
        raise VideoIOError(msg)

    first = cv2.imread(str(images[0]))
    if first is None:
        raise VideoIOError(f"Could not read first frame: {images[0]}")
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_mp4), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise VideoIOError(f"Cannot create video writer: {output_mp4}")

    try:
        for p in images:
            im = cv2.imread(str(p))
            if im is None:
                logger.warning("Skip unreadable image: %s", p)
                continue
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(im)
    finally:
        writer.release()

    return output_mp4.resolve()


__all__ = ["stitch_jpeg_folder_to_mp4"]
