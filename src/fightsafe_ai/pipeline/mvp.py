"""
MVP entry point: :func:`run_mvp_pipeline` is a thin wrapper over :func:`run_pipeline`
with post-MVP steps (QA, plots, report bundle) disabled.

Re-exports :class:`MVPOutputPaths` and JSON/CSV helpers for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.pipeline.artifact_io import (
    risk_scores_dataframe_for_csv,
    sanitize_for_json,
)
from fightsafe_ai.pipeline.output_paths import MVPOutputPaths
from fightsafe_ai.pipeline.runner import RunPipelineConfig, run_pipeline


__all__ = [
    "MVPOutputPaths",
    "risk_scores_dataframe_for_csv",
    "run_mvp_pipeline",
    "sanitize_for_json",
]


def run_mvp_pipeline(
    video: Path,
    output_root: Path,
    *,
    rules_yaml: Path | None = None,
    fps: int = 10,
    rolling_window: int = 5,
    ground_y: float = 0.82,
    model_complexity: int = 1,
    min_detection: float = 0.5,
    explain_events: bool = False,
    explanations_use_ollama: bool = True,
    ollama_explain_model: str | None = None,
    ollama_explain_temperature: float | None = None,
    ollama_force_enabled: bool = False,
    llm_config: Path | None = None,
    report_ollama: bool = False,
) -> MVPOutputPaths:
    """
    Run the core MVP: frames through overlay and ``report.md`` (no QA, plots, or static bundle).
    """
    r = run_pipeline(
        video,
        output_root,
        RunPipelineConfig(
            rules_yaml=rules_yaml,
            fps=fps,
            rolling_window=rolling_window,
            ground_y=ground_y,
            model_complexity=model_complexity,
            min_detection=min_detection,
            explain_events=explain_events,
            explanations_use_ollama=explanations_use_ollama,
            ollama_explain_model=ollama_explain_model,
            ollama_explain_temperature=ollama_explain_temperature,
            ollama_force_enabled=ollama_force_enabled,
            llm_config=llm_config,
            report_ollama=report_ollama,
            include_mvp_report=True,
            include_qa=False,
            include_plots=False,
            include_report_bundle=False,
        ),
    )
    return r.paths


# Back-compat for code doing `from fightsafe_ai.pipeline.mvp import COL_*` (rare)
COL_RISK_SCORE = "risk_score"
COL_RISK_LEVEL = "risk_level"
COL_TRIGGERED = "triggered_rules"
