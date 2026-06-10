"""Unit tests for paper LaTeX helpers and run metrics (no real runs in the repo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.reports.paper_assets import (
    compute_paper_run_metrics,
    latex_escape,
    write_artifacts_tex,
    write_events_tex,
    write_summary_tex,
)


def test_latex_escape_basic() -> None:
    assert latex_escape("a & b%") == r"a \& b\%"
    assert r"\_" in latex_escape("a_b")


def test_metrics_and_tex_snippets(tmp_path: Path) -> None:
    run = tmp_path / "mini"
    run.mkdir()
    (run / "risk_scores.csv").write_text(
        "frame_id,timestamp,risk_score,risk_level\n"
        "f0,0.0,0.1,LOW\n"
        "f1,0.1,0.2,LOW\n"
        "f2,0.2,0.4,LOW\n",
        encoding="utf-8",
    )
    (run / "pose_keypoints.csv").write_text(
        "frame_id,keypoint_name,x,y,z,visibility\nf0,nose,0,0,0,1\nf1,nose,0,0,0,1\n",
        encoding="utf-8",
    )
    (run / "events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": 1,
                    "event_level": "HIGH",
                    "event_type": "test",
                    "start_time": 0.0,
                    "end_time": 0.2,
                    "max_risk_score": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    m = compute_paper_run_metrics(run, run_path_display="runs/mini")
    assert m["n_risk_rows"] == 3
    assert m["n_frames_with_pose"] == 2
    assert m["pose_coverage_denom"] == 3
    assert m["n_events"] == 1

    outd = tmp_path / "t"
    outd.mkdir()
    write_summary_tex(outd / "mini_summary.tex", tag="mini", m=m)
    write_artifacts_tex(outd / "mini_artifacts.tex", tag="mini", m=m)
    write_events_tex(outd / "mini_events.tex", tag="mini", run_dir=run)
    t = (outd / "mini_summary.tex").read_text(encoding="utf-8")
    assert r"\begin{table}" in t
    assert "tab:paper-mini-summary" in t
    assert (outd / "mini_events.tex").read_text(encoding="utf-8").count("HIGH") >= 1


@pytest.mark.slow
def test_plots_on_mini_run(tmp_path: Path) -> None:
    from fightsafe_ai.visualization.plots import plot_pose_coverage, plot_risk_timeline

    run = tmp_path / "r"
    run.mkdir()
    (run / "risk_scores.csv").write_text(
        "frame_id,timestamp,risk_score,risk_level\n0,0.0,0.0,LOW\n1,0.1,0.0,LOW\n",
        encoding="utf-8",
    )
    (run / "pose_keypoints.csv").write_text(
        "frame_id,keypoint_name,x,y,z,visibility\n0,nose,0,0,0,1\n",
        encoding="utf-8",
    )
    (run / "events.json").write_text("[]", encoding="utf-8")
    out1 = tmp_path / "a.png"
    out2 = tmp_path / "b.png"
    plot_risk_timeline(run, output_path=out1)
    plot_pose_coverage(run, output_path=out2)
    assert out1.is_file() and out1.stat().st_size > 100
    assert out2.is_file() and out2.stat().st_size > 100
