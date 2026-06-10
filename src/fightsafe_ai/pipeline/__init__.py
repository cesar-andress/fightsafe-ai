"""
MVP end-to-end pipeline composition (frames → pose → features → risk → events → overlay → report).
"""

from fightsafe_ai.pipeline.artifact_io import risk_scores_dataframe_for_csv, sanitize_for_json
from fightsafe_ai.pipeline.demo import run_e2e_demo
from fightsafe_ai.pipeline.mvp import MVPOutputPaths, run_mvp_pipeline
from fightsafe_ai.pipeline.mvp_report import (
    MVPReportConfig,
    generate_mvp_report_markdown,
    write_mvp_report,
)
from fightsafe_ai.pipeline.output_paths import paths_for_run_root
from fightsafe_ai.pipeline.runner import RunPipelineConfig, RunPipelineResult, run_pipeline
from fightsafe_ai.pipeline.youtube_demo import (
    DEMO_YOUTUBE_INPUT_CLIP,
    DEMO_YOUTUBE_SOURCE_DIRNAME,
    run_demo_youtube,
    video_id_hint_for_url,
)


__all__ = [
    "DEMO_YOUTUBE_INPUT_CLIP",
    "DEMO_YOUTUBE_SOURCE_DIRNAME",
    "MVPOutputPaths",
    "MVPReportConfig",
    "RunPipelineConfig",
    "RunPipelineResult",
    "generate_mvp_report_markdown",
    "paths_for_run_root",
    "risk_scores_dataframe_for_csv",
    "run_demo_youtube",
    "run_e2e_demo",
    "run_mvp_pipeline",
    "run_pipeline",
    "sanitize_for_json",
    "video_id_hint_for_url",
    "write_mvp_report",
]
