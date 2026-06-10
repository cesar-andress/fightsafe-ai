"""
Extract frames from video files at a target sampling rate using OpenCV.

To assemble a preview ``.mp4`` from extracted JPEGs, use
:func:`fightsafe_ai.video.writer.stitch_jpeg_folder_to_mp4`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from fightsafe_ai.exceptions import VideoIOError


logger = logging.getLogger(__name__)

FRAME_FILENAME_TEMPLATE = "frame_{index:06d}.jpg"

# Shown when OpenCV reads zero frames (often AV1/VP9/HEVC from web downloads).
NO_FRAMES_USER_HINT = (
    "Internet downloads (e.g. YouTube) often use AV1/VP9/HEVC; OpenCV may decode zero frames. "
    "Re-encode to H.264: "
    "ffmpeg -y -i IN.mp4 -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac OUT.mp4  "
    "See docs/internet-video-codecs.md in the repository."
)


def extract_frames(video_path: Path, output_dir: Path, fps: int = 10) -> list[Path]:
    """
    Sample frames from ``video_path`` at approximately ``fps`` frames per second of source time.

    Frames are written as ``frame_000001.jpg``, ``frame_000002.jpg``, … in **chronological**
    order (reading order of the source video). Returned paths are resolved absolute paths in
    that same order.

    Parameters
    ----------
    video_path:
        Path to a readable video file.
    output_dir:
        Directory to create (including parents) and write JPEG frames into.
    fps:
        Target sampling rate relative to wall-clock time in the video (must be positive).

    Returns
    -------
    list[Path]
        Paths to each saved frame, ordered from earliest to latest.

    Raises
    ------
    VideoIOError
        If the file is missing, OpenCV cannot open it, or a frame cannot be written.
    ValueError
        If ``fps`` is not positive.

    Notes
    -----
    Uses the container's reported FPS; if missing or invalid, assumes **30** FPS.
    Requires OpenCV built with appropriate codecs for the input format.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        msg = f"Video file not found or not a file: {video_path}"
        logger.error(msg)
        raise VideoIOError(msg)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = (
            f"Cannot open video (missing codec, corrupt file, or unsupported format): {video_path}"
        )
        logger.error(msg)
        raise VideoIOError(msg)

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        logger.warning(
            "CAP_PROP_FPS missing or invalid for %s; assuming 30.0 FPS.",
            video_path,
        )
        src_fps = 30.0

    interval = src_fps / float(fps)
    next_frame_at = 0.0
    frame_index = 0
    saved = 0
    paths: list[Path] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index + 1e-6 >= next_frame_at:
                saved += 1
                name = FRAME_FILENAME_TEMPLATE.format(index=saved)
                out_path = output_dir / name
                if not cv2.imwrite(str(out_path), frame):
                    raise VideoIOError(f"Failed to write frame image: {out_path}")
                paths.append(out_path.resolve())
                next_frame_at += interval
            frame_index += 1
    finally:
        cap.release()

    if not paths:
        logger.warning("No frames extracted from %s (empty or unreadable stream).", video_path)
        logger.warning("%s", NO_FRAMES_USER_HINT)

    return paths
