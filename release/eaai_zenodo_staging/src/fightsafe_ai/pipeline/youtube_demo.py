"""
YouTube (or any yt-dlp URL) demo: download into ``<run>/source/``, cut to ``input_clip.mp4``,
then :func:`fightsafe_ai.pipeline.demo.run_e2e_demo` on the clip.

All video artifacts stay under the run root; no network in unit tests (mock this module's dependencies).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fightsafe_ai.exceptions import VideoCutError, VideoDownloadError
from fightsafe_ai.pipeline.demo import run_e2e_demo
from fightsafe_ai.pipeline.output_paths import MVPOutputPaths
from fightsafe_ai.qa.quality_report import QualityReport
from fightsafe_ai.video.cutter import cut_clip
from fightsafe_ai.video.downloader import download_video


logger = logging.getLogger(__name__)

# On-disk layout under the run root (do not place media outside it).
DEMO_YOUTUBE_SOURCE_DIRNAME = "source"
DEMO_YOUTUBE_INPUT_CLIP = "input_clip.mp4"

_id_re = re.compile(r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{6,})")
_standalone_id = re.compile(r"^([a-zA-Z0-9_-]{11})$")


def video_id_hint_for_url(url: str) -> str:
    """
    Best-effort YouTube-style video id for default download filenames; ``\"video\"`` if unknown.
    """
    m = _id_re.search(url)
    if m:
        return m.group(1)
    m2 = _standalone_id.match((url or "").strip())
    if m2:
        return m2.group(1)
    return "video"


def _resolve_download_basename(url: str, download_filename: str | None) -> str:
    """Return a single file name for yt-dlp (``--download-name``)."""
    if download_filename is not None and download_filename.strip():
        base = download_filename.strip()
        if Path(base).name != base:
            raise ValueError("download_filename must be a file name, not a path")
        if Path(base).suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".m4a"}:
            return f"{base}.mp4"
        return base
    return f"{video_id_hint_for_url(url)}.mp4"


def _assert_path_under_run(path: Path, run_root: Path) -> None:
    """Ensure a resolved file path is inside the run directory (no escape via symlinks)."""
    p = path.resolve()
    r = run_root.resolve()
    if not p.is_relative_to(r):
        raise VideoDownloadError(
            f"Download path is outside the run directory: {p} (expected under {r})"
        )


def run_demo_youtube(
    url: str,
    start: str,
    end: str,
    output_root: Path,
    *,
    download_filename: str | None = None,
    rules_yaml: Path | None = None,
    fps: int = 10,
    rolling_window: int = 5,
    ground_y: float = 0.82,
    model_complexity: int = 1,
    min_detection: float = 0.5,
    use_ollama: bool = False,
    ollama_explain_model: str | None = None,
    ollama_explain_temperature: float | None = None,
    ollama_force_enabled: bool = False,
    llm_config: Path | None = None,
) -> tuple[MVPOutputPaths, bool, QualityReport]:
    """
    1) Download with yt-dlp into ``output_root/source/``.
    2) Cut ``[start, end)`` to ``output_root/input_clip.mp4``.
    3) Run the full :func:`~fightsafe_ai.pipeline.demo.run_e2e_demo` on that clip.

    Raises
    ------
    VideoDownloadError
        If yt-dlp fails (caller should exit non-zero with a user-facing message).
    VideoCutError
        If FFmpeg cut fails.
    """
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_dir = output_root / DEMO_YOUTUBE_SOURCE_DIRNAME
    source_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_root / DEMO_YOUTUBE_INPUT_CLIP

    s0, t0 = (start.strip(), end.strip())
    if not s0 or not t0:
        raise ValueError("start and end must be non-empty")

    try:
        dl_basename = _resolve_download_basename(url, download_filename)
    except ValueError as exc:
        raise ValueError(f"Invalid download filename: {exc}") from exc

    # Step 1: download (all files under output_root/source/)
    logger.info(
        "[demo-youtube] Step 1/3: downloading URL into %s (yt-dlp)",
        source_dir,
    )
    try:
        full_path = download_video(url, source_dir, filename=dl_basename)
    except VideoDownloadError:
        logger.exception("[demo-youtube] Step 1/3: download failed (see error above)")
        raise
    _assert_path_under_run(full_path, output_root)
    logger.info("[demo-youtube] Step 1/3: finished - %s", full_path)

    # Step 2: cut clip next to run artifacts
    logger.info(
        "[demo-youtube] Step 2/3: cutting segment [%s, %s) -> %s (ffmpeg)",
        s0,
        t0,
        clip_path,
    )
    try:
        cut_clip(full_path, s0, t0, clip_path)
    except VideoCutError:
        logger.exception("[demo-youtube] Step 2/3: cut failed (see error above)")
        raise
    if not clip_path.is_file():
        raise VideoCutError(f"Cut did not produce output file: {clip_path}")
    _assert_path_under_run(clip_path, output_root)
    logger.info("[demo-youtube] Step 2/3: finished - %s", clip_path.resolve())

    # Step 3: full pipeline (works offline: local clip only; optional Ollama is best-effort)
    logger.info(
        "[demo-youtube] Step 3/3: running full pipeline (frames, pose, risk, QA, reports) in %s",
        output_root,
    )
    paths, qa_ok, qreport = run_e2e_demo(
        clip_path,
        output_root,
        rules_yaml=rules_yaml,
        fps=fps,
        rolling_window=rolling_window,
        ground_y=ground_y,
        model_complexity=model_complexity,
        min_detection=min_detection,
        use_ollama=use_ollama,
        ollama_explain_model=ollama_explain_model,
        ollama_explain_temperature=ollama_explain_temperature,
        ollama_force_enabled=ollama_force_enabled,
        llm_config=llm_config,
    )
    logger.info("[demo-youtube] Step 3/3: pipeline completed (qa_ok=%s)", qa_ok)
    return paths, qa_ok, qreport


__all__ = [
    "DEMO_YOUTUBE_INPUT_CLIP",
    "DEMO_YOUTUBE_SOURCE_DIRNAME",
    "run_demo_youtube",
    "video_id_hint_for_url",
]
