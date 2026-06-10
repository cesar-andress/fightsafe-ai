"""
End-to-end orchestration: a single :func:`run_pipeline` for all run artifacts.

The numbered stages match :mod:`fightsafe_ai.pipeline.steps` and always use
:class:`~fightsafe_ai.pipeline.output_paths.MVPOutputPaths` / :func:`paths_for_run_root`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.llm.reporting import generate_pipeline_event_explanations
from fightsafe_ai.pipeline import steps
from fightsafe_ai.pipeline.mvp_report import MVPReportConfig, write_mvp_report
from fightsafe_ai.pipeline.optional_ollama import try_optional_ollama_narrative
from fightsafe_ai.pipeline.output_paths import MVPOutputPaths, paths_for_run_root
from fightsafe_ai.qa.quality_report import QualityReport


logger = logging.getLogger(__name__)


@dataclass
class RunPipelineConfig:
    """Optional knobs for a full run. Defaults run all post-processing (QA, plots, report bundle)."""

    rules_yaml: Path | None = None
    pose_backend: str = "mediapipe"
    fps: int = 10
    rolling_window: int = 5
    ground_y: float = 0.82
    model_complexity: int = 1
    min_detection: float = 0.5
    explain_events: bool = False
    explanations_use_ollama: bool = True
    ollama_explain_model: str | None = None
    ollama_explain_temperature: float | None = None
    ollama_force_enabled: bool = False
    llm_config: Path | None = None
    report_ollama: bool = False
    include_mvp_report: bool = True
    include_qa: bool = True
    include_plots: bool = True
    include_report_bundle: bool = True


@dataclass
class RunPipelineResult:
    """Outputs of :func:`run_pipeline`."""

    paths: MVPOutputPaths
    frame_paths: list[Path]
    feat_df: pd.DataFrame
    n_events: int
    quality_report: QualityReport | None = None


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    config: RunPipelineConfig | None = None,
) -> RunPipelineResult:
    """
    Run the canonical pipeline: frames → pose → features → risk → events → overlay →
    optional MVP report → optional QA, plots, and static report bundle.

    Parameters
    ----------
    video_path
        Input video file.
    output_dir
        Run root directory (created). Layout is :func:`paths_for_run_root` / :class:`MVPOutputPaths`.
    config
        Tuning and feature flags; if omitted, defaults run steps 1–7 plus QA, plots, and report bundle
        (full ten-step end-to-end).
    """
    cfg = config or RunPipelineConfig()
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for_run_root(output_dir)

    total = 7 + (1 if cfg.explain_events else 0) + (1 if cfg.include_mvp_report else 0)
    total += sum(1 for x in (cfg.include_qa, cfg.include_plots, cfg.include_report_bundle) if x)
    step_i = 0

    def _log(msg: str) -> None:
        nonlocal step_i
        step_i += 1
        logger.info("[%s/%s] %s", step_i, total, msg)

    # 1-2: frames, pose
    _log(f"extract-frames -> {paths.frames_dir} @ {cfg.fps} FPS")
    try:
        frame_paths = steps.step01_extract_frames(video_path, paths, fps=cfg.fps)
    except VideoIOError:
        raise

    _log(f"pose ({cfg.pose_backend}) -> {paths.pose_keypoints_csv}")
    steps.step02_estimate_pose(
        paths,
        model_complexity=cfg.model_complexity,
        min_detection=cfg.min_detection,
        pose_backend=cfg.pose_backend,
    )

    # 3-4: features (biomech + temporal) + 5: write CSV
    _log(f"compute-features (biomechanics + temporal) -> {paths.features_csv}")
    feat_df = steps.step03_build_biomechanical_dataframe(
        paths,
        fps=float(cfg.fps),
        rolling_window=cfg.rolling_window,
        ground_y=cfg.ground_y,
    )
    feat_df = steps.step04_add_temporal_features(
        feat_df,
        fps=cfg.fps,
        rolling_window=cfg.rolling_window,
    )
    steps.step05_write_features_csv(feat_df, paths)

    # 6: risk + events JSON
    _log(
        f"combat MVP risk (timestamp + score + levels) -> {paths.risk_scores_csv} "
        f"and {paths.events_json}"
    )
    _work_ts, events_list = steps.step06_compute_risk_write_scores_and_events(
        feat_df,
        paths,
        fps=float(cfg.fps),
        rules_yaml=cfg.rules_yaml,
    )

    explanations_dir = output_dir / "explanations"
    if cfg.explain_events:
        _log(f"per-event explanations -> {explanations_dir} …")
        generate_pipeline_event_explanations(
            feat_df,
            float(cfg.fps),
            cfg.rules_yaml,
            explanations_dir,
            use_ollama=cfg.explanations_use_ollama,
            ollama_model=cfg.ollama_explain_model,
            ollama_temperature=cfg.ollama_explain_temperature,
            ollama_force_enabled=cfg.ollama_force_enabled,
            llm_config=cfg.llm_config,
        )

    # 7: overlay
    _log(f"stitch + overlay -> {paths.output_overlay_mp4}")
    steps.step07_generate_overlay_video(paths, fps=float(cfg.fps))

    n_events = len(events_list)

    if cfg.include_mvp_report:
        _log(f"report (Markdown) -> {paths.report_md}")
        narrative = (
            try_optional_ollama_narrative(
                n_events,
                cfg.llm_config,
                model=cfg.ollama_explain_model,
                temperature=cfg.ollama_explain_temperature,
                force_enabled=cfg.ollama_force_enabled,
            )
            if cfg.report_ollama
            else ""
        )
        write_mvp_report(
            paths.report_md,
            MVPReportConfig(
                video_path=video_path,
                output_root=output_dir,
                events_path=paths.events_json,
                risk_scores_path=paths.risk_scores_csv,
                rules_config_path=cfg.rules_yaml
                if cfg.rules_yaml and cfg.rules_yaml.is_file()
                else None,
                sampling_fps=float(cfg.fps),
                n_sampled_frames=len(frame_paths),
                optional_ollama_narrative=narrative,
                explanations_dir=explanations_dir if cfg.explain_events else None,
            ),
        )

    qreport: QualityReport | None = None
    if cfg.include_qa:
        _log("QA (quality checks) -> qa_report.json")
        qreport, _ = steps.step08_run_qa(paths)
    if cfg.include_plots:
        _log("plots (matplotlib) -> risk_timeline.png, events_timeline.png")
        steps.step09_generate_plots(paths)
    if cfg.include_report_bundle:
        _log("reports (Markdown / HTML / summary) -> report bundle")
        steps.step10_generate_report_bundle(paths)

    return RunPipelineResult(
        paths=paths,
        frame_paths=frame_paths,
        feat_df=feat_df,
        n_events=n_events,
        quality_report=qreport,
    )


__all__ = [
    "RunPipelineConfig",
    "RunPipelineResult",
    "run_pipeline",
]
