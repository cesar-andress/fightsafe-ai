"""Download video from URLs using the ``yt-dlp`` command-line tool."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from fightsafe_ai.exceptions import VideoDownloadError


logger = logging.getLogger(__name__)

_YTDLP_FORMAT = "bestvideo+bestaudio/best"
_MERGE_FORMAT = "mp4"


def download_video(
    url: str,
    output_dir: Path,
    filename: str | None = None,
) -> Path:
    """
    Download media from ``url`` with ``yt-dlp``, preferring a merged MP4 container.

    Creates ``output_dir`` if it does not exist. Uses ``ffmpeg`` (when available
    to ``yt-dlp``) to mux streams into MP4 where the source permits.

    Parameters
    ----------
    url:
        Page or stream URL understood by ``yt-dlp``.
    output_dir:
        Directory where the final file is written.
    filename:
        Optional output **file name** (not a full path). If omitted, a template
        ``title`` and ``id`` from metadata is used. If the name has no extension,
        ``.mp4`` is appended so merging targets MP4 consistently.

    Returns
    -------
    Path
        Absolute path to the downloaded media file.

    Raises
    ------
    VideoDownloadError
        If ``yt-dlp`` is missing, the process exits non-zero, or the output file
        cannot be located after a reported success.

    Notes
    -----
    Requires the ``yt-dlp`` executable on ``PATH`` (e.g. ``pip install yt-dlp``).
    This module invokes the CLI via subprocess; it does not import ``yt_dlp`` Python API.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("yt-dlp")
    if not executable:
        msg = (
            "Executable 'yt-dlp' not found on PATH. Install with: pip install yt-dlp "
            "(and ensure the Scripts/bin directory is on PATH)."
        )
        logger.error(msg)
        raise VideoDownloadError(msg)

    out_template: str
    expected_path: Path | None = None

    if filename:
        base = filename.strip()
        if not base:
            raise VideoDownloadError("filename must be non-empty when provided.")
        if Path(base).name != base:
            raise VideoDownloadError(
                "filename must be a single file name, not a path or contain separators."
            )
        path = output_dir / base
        if path.suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".m4a"}:
            path = path.with_suffix(".mp4")
            logger.debug("Normalized filename to target mp4: %s", path.name)
        out_template = str(path)
        expected_path = path
    else:
        out_template = str(output_dir / "%(title)s [%(id)s].%(ext)s")

    before_videos: set[Path] = _video_files_in_dir(output_dir)

    cmd: list[str] = [
        executable,
        "-f",
        _YTDLP_FORMAT,
        "--merge-output-format",
        _MERGE_FORMAT,
        "--no-playlist",
        "-o",
        out_template,
        # Default --print stage is "video", where %(filepath)s is often undefined (prints "NA").
        # Use after_move so the path reflects the final merged file on disk.
        "--print",
        "after_move:%(filepath)s",
        url,
    ]

    logger.info("Starting yt-dlp download into %s", output_dir)
    logger.debug("yt-dlp command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=None,
        )
    except OSError as exc:
        logger.exception("Failed to spawn yt-dlp")
        raise VideoDownloadError(f"Could not run yt-dlp: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        logger.error("yt-dlp exited with code %s: %s", proc.returncode, err[:2000])
        raise VideoDownloadError(
            f"yt-dlp failed with exit code {proc.returncode}. stderr: {err[:1500]}"
        )

    printed = _parse_print_filepath(proc.stdout)
    if printed:
        resolved = Path(printed.strip()).expanduser().resolve()
        if resolved.is_file():
            logger.info("Download finished: %s", resolved)
            return resolved
        logger.warning("yt-dlp printed filepath not found on disk: %s", resolved)

    if expected_path is not None and expected_path.is_file():
        logger.info("Download finished: %s", expected_path)
        return expected_path.resolve()

    discovered = _discover_new_video(output_dir, before_videos)
    if discovered is not None:
        logger.info("Resolved download path via directory scan: %s", discovered)
        return discovered.resolve()

    raise VideoDownloadError(
        "Download appeared to succeed but output file could not be resolved. "
        "Check yt-dlp stdout/stderr and output directory contents."
    )


def _parse_print_filepath(stdout: str) -> str | None:
    """Return last non-empty line from ``--print`` output, if any."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    # yt-dlp may print multiple lines; last filepath is typically the final merged file
    last = lines[-1]
    # Pre-merge / wrong-stage prints sometimes emit literal "NA" for missing template fields
    if last.upper() in {"NA", "N/A", "NONE"}:
        return None
    return last


def _video_files_in_dir(output_dir: Path) -> set[Path]:
    """Collect likely video artifacts in ``output_dir`` (non-recursive)."""
    out: set[Path] = set()
    for pat in ("*.mp4", "*.webm", "*.mkv", "*.mov"):
        out.update(output_dir.glob(pat))
    return out


def _discover_new_video(output_dir: Path, before: set[Path]) -> Path | None:
    """Pick a new video file in ``output_dir`` after download (prefer ``.mp4``)."""
    after = _video_files_in_dir(output_dir)
    new_files = after - before
    mp4s = {p for p in new_files if p.suffix.lower() == ".mp4"}
    if len(mp4s) == 1:
        return next(iter(mp4s))
    if len(new_files) == 1:
        return next(iter(new_files))
    if len(new_files) > 1:
        prefer = [p for p in new_files if p.suffix.lower() == ".mp4"]
        pool = prefer if prefer else list(new_files)
        return max(pool, key=lambda p: p.stat().st_mtime)
    if len(after) == 1 and len(before) == 0:
        return next(iter(after))
    return None
