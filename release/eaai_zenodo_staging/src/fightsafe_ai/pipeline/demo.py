"""
End-to-end demo orchestration: one :func:`~fightsafe_ai.pipeline.runner.run_pipeline` call
with QA, plots, and the static report bundle enabled.
"""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.pipeline.output_paths import MVPOutputPaths
from fightsafe_ai.pipeline.runner import RunPipelineConfig, run_pipeline
from fightsafe_ai.qa.quality_report import QualityReport


def run_e2e_demo(
    video: Path,
    output_root: Path,
    *,
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
    Run the full demo sequence via the unified pipeline:

    1. Same processing as :func:`~fightsafe_ai.pipeline.mvp.run_mvp_pipeline` (with per-event
       explanations; optional Ollama when ``use_ollama`` is True and the server is available)
    2. QA, Matplotlib PNG timelines, and :func:`~fightsafe_ai.reports.write_all_default_reports`.
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
            explain_events=True,
            explanations_use_ollama=use_ollama,
            ollama_explain_model=ollama_explain_model,
            ollama_explain_temperature=ollama_explain_temperature,
            ollama_force_enabled=ollama_force_enabled,
            llm_config=llm_config,
            report_ollama=use_ollama,
            include_mvp_report=True,
            include_qa=True,
            include_plots=True,
            include_report_bundle=True,
        ),
    )
    if r.quality_report is None:
        raise RuntimeError("run_pipeline: expected quality report when include_qa is True")
    qa_ok = bool(r.quality_report.passed)
    return r.paths, qa_ok, r.quality_report


__all__ = [
    "run_e2e_demo",
]
