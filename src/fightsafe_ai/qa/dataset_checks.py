"""
Dataset-style metrics (e.g. pose coverage vs. extracted frame images) for QA reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

from fightsafe_ai.qa.validators import QualityCheckResult, Status


logger = logging.getLogger(__name__)

_IMAGE_EXTS: Final[set[str]] = {".jpg", ".jpeg", ".png"}


def count_frame_images(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    n = 0
    for p in frames_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            n += 1
    return n


def pose_coverage_metrics(
    frames_dir: Path,
    pose_path: Path,
) -> tuple[float | None, dict[str, Any]]:
    """
    Compare unique ``frame_id`` in pose export to number of JPEG/PNG under ``frames_dir``.

    Returns
    -------
    (coverage_percent, meta)
        ``None`` for coverage if the metric is undefined (e.g. no images).
    """
    meta: dict[str, Any] = {}
    frames_dir = frames_dir.resolve()
    pose_path = pose_path.resolve()

    n_img = count_frame_images(frames_dir)
    meta["n_frame_image_files"] = n_img

    if n_img == 0:
        meta["reason"] = "no_jpeg_in_frames"
        return None, meta
    if not pose_path.is_file():
        meta["reason"] = "pose_csv_missing"
        return None, meta

    try:
        head = pd.read_csv(pose_path, nrows=0)
        if "frame_id" in head.columns:
            df = pd.read_csv(pose_path, usecols=["frame_id"])
        else:
            df = pd.read_csv(pose_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("pose_coverage: could not read %s: %s", pose_path, exc)
        meta["error"] = repr(exc)
        return None, meta

    if "frame_id" not in df.columns or len(df) == 0:
        meta["reason"] = "no_frame_id_or_empty_pose"
        return 0.0, meta

    n_unique = int(df["frame_id"].astype(str).nunique())
    meta["n_unique_pose_frame_ids"] = n_unique
    # Coverage: min unique pose frames vs. number of image files in frames/
    pct = 100.0 * min(n_unique, n_img) / n_img
    if n_unique > n_img:
        meta["note"] = "n_unique_frame_id exceeds n_image_files; min() caps at 100%"
    return float(pct), meta


def coverage_to_results(
    percent: float | None,
    meta: dict[str, Any],
) -> list[QualityCheckResult]:
    """Map coverage to :class:`QualityCheckResult` list (typically a single item)."""
    n_img = int(meta.get("n_frame_image_files", 0) or 0)
    if percent is None:
        reason = str(meta.get("reason", "unknown"))
        st: Status = "warn"
        if reason in ("no_jpeg_in_frames",) and n_img == 0:
            st = "fail"
        return [
            QualityCheckResult(
                "pose_coverage",
                st,
                f"Pose coverage could not be computed ({reason})",
                details=str(meta)[:2000],
            )
        ]
    if n_img > 0 and percent < 50.0:
        return [
            QualityCheckResult(
                "pose_coverage",
                "warn",
                f"Low pose coverage (<50%): {percent:.1f}% of extracted images have a pose row.",
                details=str(meta)[:2000],
            )
        ]
    if n_img > 0 and percent < 100.0:
        return [
            QualityCheckResult(
                "pose_coverage",
                "warn",
                f"Unique pose frames / extracted images = {percent:.1f}% — not all image slots have pose",
                details=str(meta)[:2000],
            )
        ]
    return [
        QualityCheckResult(
            "pose_coverage",
            "pass",
            f"Unique pose frame coverage vs. extracted images: {percent:.1f}%",
        )
    ]


__all__ = [
    "count_frame_images",
    "coverage_to_results",
    "pose_coverage_metrics",
]
