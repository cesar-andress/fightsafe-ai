"""
Discrete pipeline stages for the end-to-end combat-safety run.

Each public ``step*`` function implements one numbered stage in
:func:`fightsafe_ai.pipeline.runner.run_pipeline`.
All paths follow the **standard run layout** (see :class:`MVPOutputPaths`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fightsafe_ai.config.framework import load_framework_config, pose_init_kwargs_for_backend
from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.features.anomaly import add_limb_anomaly_columns
from fightsafe_ai.features.biomechanics import (
    build_biomechanical_mvp_dataframe,
    compute_pose_features,
)
from fightsafe_ai.features.temporal import compute_temporal_features
from fightsafe_ai.pipeline.artifact_io import (
    COL_RISK_LEVEL,
    COL_RISK_SCORE,
    risk_scores_dataframe_for_csv,
    sanitize_for_json,
)
from fightsafe_ai.pipeline.output_paths import MVPOutputPaths
from fightsafe_ai.pose.factory import create_pose_estimator
from fightsafe_ai.qa.quality_report import (
    QualityReport,
    run_quality_checks,
    write_qa_report_json,
)
from fightsafe_ai.reports import write_all_default_reports
from fightsafe_ai.risk.events import RiskEventExtractionConfig, frame_risk_to_events_list
from fightsafe_ai.risk.scorer import build_combat_mvp_frame_risk
from fightsafe_ai.video.frame_extractor import NO_FRAMES_USER_HINT, extract_frames
from fightsafe_ai.video.writer import stitch_jpeg_folder_to_mp4
from fightsafe_ai.visualization.overlay import render_risk_overlay
from fightsafe_ai.visualization.plots import plot_events_timeline, plot_risk_timeline


logger = logging.getLogger(__name__)

__all__ = [
    "build_full_feature_dataframe",
    "step01_extract_frames",
    "step02_estimate_pose",
    "step03_build_biomechanical_dataframe",
    "step04_add_temporal_features",
    "step05_write_features_csv",
    "step06_compute_risk_write_scores_and_events",
    "step07_generate_overlay_video",
    "step08_run_qa",
    "step09_generate_plots",
    "step10_generate_report_bundle",
]


def step01_extract_frames(video: Path, paths: MVPOutputPaths, *, fps: int) -> list[Path]:
    """1. Sample frames to ``frames/`` (JPEG)."""
    frame_paths = extract_frames(video, paths.frames_dir, fps=fps)
    if not frame_paths:
        raise VideoIOError(
            f"No frames extracted from {video} (empty or unreadable for OpenCV).\n{NO_FRAMES_USER_HINT}"
        )
    return frame_paths


def step02_estimate_pose(
    paths: MVPOutputPaths,
    *,
    model_complexity: int = 1,
    min_detection: float = 0.5,
    pose_backend: str = "mediapipe",
    framework_config: Path | None = None,
) -> None:
    """
    2. Pose model → ``pose_keypoints.csv`` (long / indexed format). Default: MediaPipe.

    With the default **MediaPipe** backend, **TensorFlow Lite** and **MediaPipe**
    may print **XNNPACK**, **Feedback Manager**, or **NORM_RECT**-related lines
    to the console in **CPU mode**; that third-party log noise is **expected** and
    is **not** a pipeline failure. Success of this step is whether
    ``pose_keypoints.csv`` (and later artifacts) are produced, combined with
    :func:`fightsafe_ai.qa.validators.run_quality_checks` on the run root—not
    stderr from native libraries. See ``docs/troubleshooting.md``.
    """
    fw = load_framework_config(framework_config)
    extra = pose_init_kwargs_for_backend(fw, pose_backend)
    create_pose_estimator(
        pose_backend,
        model_complexity=int(model_complexity),
        min_detection_confidence=float(min_detection),
        **extra,
    ).estimate_folder(paths.frames_dir, paths.pose_keypoints_csv)


def step03_build_biomechanical_dataframe(
    paths: MVPOutputPaths,
    *,
    fps: float,
    rolling_window: int = 5,
    ground_y: float = 0.82,
) -> pd.DataFrame:
    """3. Biomechanics and per-frame derived columns (before rolling temporal block)."""
    return build_biomechanical_mvp_dataframe(
        paths.pose_keypoints_csv,
        fps=float(fps),
        rolling_window=rolling_window,
        ground_y_threshold=ground_y,
    )


def step04_add_temporal_features(
    feat_df: pd.DataFrame,
    *,
    fps: int,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """4. Temporal signals + limb anomaly columns (matches :func:`compute_pose_features`)."""
    if feat_df.empty:
        return feat_df
    temp = compute_temporal_features(
        feat_df,
        int(max(1, round(float(fps)))),
        rolling_window_frames=rolling_window,
    )
    return add_limb_anomaly_columns(temp, float(fps))


def step05_write_features_csv(feat_df: pd.DataFrame, paths: MVPOutputPaths) -> None:
    """5. Persist ``features.csv``."""
    feat_df.to_csv(paths.features_csv, index=False)


def step06_compute_risk_write_scores_and_events(
    feat_df: pd.DataFrame,
    paths: MVPOutputPaths,
    *,
    fps: float,
    rules_yaml: Path | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    6. Per-frame risk scores; write ``risk_scores.csv`` and ``events.json`` (detected events list).
    """
    work_ts = build_combat_mvp_frame_risk(feat_df, float(fps), rules_yaml=rules_yaml)
    risk_csv_ready = risk_scores_dataframe_for_csv(work_ts)
    risk_csv_ready.to_csv(paths.risk_scores_csv, index=False)
    events_list: list[dict[str, Any]] = []
    if (
        len(work_ts) > 0
        and {"frame_id", "timestamp"}.issubset(work_ts.columns)
        and COL_RISK_SCORE in work_ts.columns
        and COL_RISK_LEVEL in work_ts.columns
    ):
        events_list = frame_risk_to_events_list(
            work_ts,
            config=RiskEventExtractionConfig(fps=float(fps)),
        )
    events_payload = [sanitize_for_json(e) for e in events_list]
    paths.events_json.write_text(
        json.dumps(events_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return work_ts, events_list


def step07_generate_overlay_video(paths: MVPOutputPaths, *, fps: float) -> None:
    """7. Stitched sequence + risk overlay → ``output_overlay.mp4`` (stitched preview file removed after)."""
    stitch_jpeg_folder_to_mp4(
        paths.frames_dir,
        paths.stitched_preview_mp4,
        fps=float(fps),
    )
    try:
        render_risk_overlay(
            paths.stitched_preview_mp4,
            paths.pose_keypoints_csv,
            paths.risk_scores_csv,
            paths.output_overlay_mp4,
        )
    finally:
        try:
            paths.stitched_preview_mp4.unlink(missing_ok=True)
        except OSError:
            pass


def step08_run_qa(paths: MVPOutputPaths) -> tuple[QualityReport, bool]:
    """8. ``qa_report.json`` (structured checks; CLI may use pass/fail)."""
    report = run_quality_checks(paths.root, require_frames=True)
    write_qa_report_json(paths.root / "qa_report.json", report)
    return report, bool(report.passed)


def step09_generate_plots(paths: MVPOutputPaths) -> None:
    """9. Matplotlib timeline PNGs under the run root."""
    plot_risk_timeline(paths.root, paths.root / "risk_timeline.png")
    plot_events_timeline(paths.root, paths.root / "events_timeline.png")


def step10_generate_report_bundle(paths: MVPOutputPaths) -> None:
    """10. Regenerate ``report.md``, ``report.html``, ``summary.json`` (polished bundle)."""
    write_all_default_reports(paths.root)


def build_full_feature_dataframe(
    paths: MVPOutputPaths,
    *,
    fps: float,
    rolling_window: int = 5,
    ground_y: float = 0.82,
) -> pd.DataFrame:
    """
    Convenience: one-shot biomechanics + temporal, equivalent to
    :func:`fightsafe_ai.features.biomechanics.compute_pose_features` on ``pose_keypoints.csv``.
    """
    return compute_pose_features(
        paths.pose_keypoints_csv,
        fps=float(fps),
        rolling_window=rolling_window,
        ground_y_threshold=ground_y,
    )
