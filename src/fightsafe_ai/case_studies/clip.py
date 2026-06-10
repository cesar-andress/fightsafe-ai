"""Prepare ``input_clip.mp4`` for a case-study run (optionally cut, or full clip)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from fightsafe_ai.exceptions import VideoCutError
from fightsafe_ai.video.cutter import cut_clip


logger = logging.getLogger(__name__)


def ffprobe_duration_seconds(video: Path) -> float:
    """Return container duration in seconds; requires ``ffprobe`` on ``PATH``."""
    p = video.expanduser().resolve()
    if not p.is_file():
        raise VideoCutError(f"ffprobe: file not found: {p}")
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(p),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise VideoCutError(f"ffprobe failed: {r.stderr or r.stdout}")
    try:
        d = json.loads(r.stdout)
        return float((d.get("format") or {}).get("duration", 0.0))
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as e:
        raise VideoCutError(f"ffprobe: could not read duration: {e}") from e


def _remux_or_copy_to_mp4(src: Path, dst: Path) -> None:
    dst = dst.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = src.expanduser().resolve()
    if s.suffix.lower() == ".mp4":
        shutil.copy2(s, dst)
        if dst.is_file() and dst.stat().st_size > 0:
            return
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(s),
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not dst.is_file():
        msg = f"ffmpeg remux failed (exit {r.returncode}): {r.stderr or r.stdout}"
        logger.error(msg)
        raise VideoCutError(msg)


def prepare_input_clip_full(
    full_download: Path,
    run_root: Path,
    *,
    clip_basename: str = "input_clip.mp4",
) -> Path:
    """
    For a **full** download: build ``<run_root>/input_clip.mp4`` — copy if already {MP4},
    else re-encode to {H.264} for OpenCV / pipeline compatibility.
    """
    s = full_download.expanduser().resolve()
    if not s.is_file():
        raise VideoCutError(f"Missing download: {s}")
    d = s.suffix.lower()
    out = run_root.expanduser().resolve() / clip_basename
    if d == ".mp4":
        try:
            shutil.copy2(s, out)
            if out.is_file() and out.stat().st_size > 0:
                return out
        except OSError as e:
            logger.warning("MP4 copy failed, trying re-encode: %s", e)
    _remux_or_copy_to_mp4(s, out)
    if not out.is_file():
        raise VideoCutError(f"Could not build input clip: {out}")
    return out


def prepare_input_clip(
    full_download: Path,
    run_root: Path,
    start_time: str | None,
    end_time: str | None,
    *,
    clip_basename: str = "input_clip.mp4",
) -> Path:
    """
    If both ``start_time`` and ``end_time`` are non-empty: cut with FFmpeg.
    If both are null/empty: full clip via :func:`prepare_input_clip_full`.
    """
    run_root = run_root.expanduser().resolve()
    out = run_root / clip_basename
    s0 = (start_time or "").strip()
    s1 = (end_time or "").strip()
    if s0 and s1:
        return cut_clip(full_download, s0, s1, out)
    if (s0 and not s1) or (s1 and not s0):
        raise ValueError("Provide both start_time and end_time, or set both to null for full clip.")
    return prepare_input_clip_full(full_download, run_root, clip_basename=clip_basename)


__all__ = [
    "ffprobe_duration_seconds",
    "prepare_input_clip",
    "prepare_input_clip_full",
]
