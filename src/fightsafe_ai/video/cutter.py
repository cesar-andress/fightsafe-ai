"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Cut a segment from a video file using FFmpeg via ``ffmpeg-python``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import ffmpeg

from fightsafe_ai.exceptions import VideoCutError


logger = logging.getLogger(__name__)

_TIME_SPLIT = re.compile(r":")


def parse_timecode(value: str) -> float:
    """
    Parse a timestamp into seconds.

    Accepts:

    * Seconds as a decimal string, e.g. ``\"83.5\"``, ``\"120\"``.
    * ``MM:SS`` or ``HH:MM:SS`` with optional fractional seconds on the last part,
      e.g. ``\"01:23\"``, ``\"00:01:23\"``, ``\"00:01:23.500\"``.
    """
    value = value.strip()
    if not value:
        raise ValueError("timestamp must be non-empty")

    if ":" not in value:
        return float(value)

    parts = _TIME_SPLIT.split(value)
    if len(parts) > 3:
        raise ValueError(f"too many ':' segments in timecode: {value!r}")

    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"invalid numeric segment in timecode: {value!r}") from exc

    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    raise ValueError(f"unsupported timecode format: {value!r}")


def cut_clip(
    input_video: Path,
    start_time: str,
    end_time: str,
    output_path: Path,
) -> Path:
    """
    Extract ``[start_time, end_time)`` from ``input_video`` and write ``output_path``.

    Requires the ``ffmpeg`` executable on ``PATH`` (installed separately from this package).
    Uses stream copy (``-c:v copy -c:a copy``) when possible for speed and lossless muxing;
    both streams must exist and be compatible—otherwise FFmpeg may fail (see stderr).

    Parameters
    ----------
    input_video:
        Existing media file readable by FFmpeg.
    start_time, end_time:
        Bounds parsed by :func:`parse_timecode`.
    output_path:
        Destination file; parent directories are created if missing.

    Returns
    -------
    Path
        Resolved ``output_path`` if the file exists after a successful run.

    Raises
    ------
    VideoCutError
        On missing input, invalid times, FFmpeg failures, or missing output file.
    """
    input_video = input_video.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_video.is_file():
        msg = f"Input video not found: {input_video}"
        logger.error(msg)
        raise VideoCutError(msg)

    try:
        start_sec = parse_timecode(start_time)
        end_sec = parse_timecode(end_time)
    except ValueError as exc:
        raise VideoCutError(f"Invalid timecode: {exc}") from exc

    if start_sec < 0 or end_sec < 0:
        raise VideoCutError("start_time and end_time must be non-negative.")

    if end_sec <= start_sec:
        raise VideoCutError(
            f"end_time must be greater than start_time (got start={start_sec}, end={end_sec})."
        )

    duration = end_sec - start_sec
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Cutting clip: %s [%.3fs .. %.3fs] -> %s",
        input_video,
        start_sec,
        end_sec,
        output_path,
    )

    try:
        inp = ffmpeg.input(str(input_video), ss=start_sec)
        out = ffmpeg.output(
            inp,
            str(output_path),
            t=duration,
            vcodec="copy",
            acodec="copy",
        )
        ffmpeg.run(
            out,
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True,
        )
    except ffmpeg.Error as exc:
        stderr = getattr(exc, "stderr", None)
        if stderr:
            text = stderr.decode(errors="replace")[-4000:]
        else:
            text = str(exc)
        logger.error("ffmpeg failed: %s", text)
        raise VideoCutError(f"ffmpeg failed while cutting clip: {text[:1500]}") from exc
    except OSError as exc:
        logger.exception("Could not run ffmpeg")
        raise VideoCutError(f"Could not execute ffmpeg: {exc}") from exc

    if not output_path.is_file():
        raise VideoCutError(f"Expected output file was not created: {output_path}")

    logger.info("Wrote clip: %s", output_path)
    return output_path
