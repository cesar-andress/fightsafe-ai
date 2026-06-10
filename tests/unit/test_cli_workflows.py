"""
CLI workflow tests (Typer :class:`typer.testing.CliRunner`): no network, no Ollama.

Exercises ``fightsafe --help``, ``qa``, and ``generate-report`` on a synthetic run tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.mvp_runs import write_minimal_pipeline_run, write_mvp_qa_passing_run
from typer.testing import CliRunner

from fightsafe_ai.cli import app


pytestmark = pytest.mark.unit


def test_fightsafe_root_help_lists_primary_commands() -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    out = (r.stdout or "").lower()
    assert "download" in out and "demo" in out and "generate-report" in out
    assert "report" in out


def test_fightsafe_qa_passes_on_synthetic_mvp_run(tmp_path: Path) -> None:
    run = write_mvp_qa_passing_run(tmp_path / "run_a")
    runner = CliRunner()
    r = runner.invoke(app, ["qa", "--run", str(run)])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    out = (r.stdout or "") + (r.stderr or "")
    assert "PASS" in out or "status:" in out.lower()


def test_fightsafe_qa_fails_cleanly_on_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    runner = CliRunner()
    r = runner.invoke(app, ["qa", "--run", str(empty)])
    assert r.exit_code == 1


def test_fightsafe_generate_report_writes_artifacts(tmp_path: Path) -> None:
    run = write_minimal_pipeline_run(tmp_path / "r1", include_mvp_artifacts=False)
    runner = CliRunner()
    r = runner.invoke(app, ["generate-report", "--run", str(run)])
    assert r.exit_code == 0
    assert (run / "report.md").is_file() or "Wrote" in (r.stdout or "")


def test_fightsafe_generate_report_only_html(tmp_path: Path) -> None:
    run = write_minimal_pipeline_run(tmp_path / "r2", include_mvp_artifacts=False)
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["generate-report", "--run", str(run), "--only", "html"],
    )
    assert r.exit_code == 0
    assert (run / "report.html").is_file()


def test_fightsafe_build_events_writes_json(tmp_path: Path) -> None:
    """Frame risk CSV is merged into a JSON event list via the Typer CLI."""
    risk = tmp_path / "risk_scores.csv"
    risk.write_text(
        "frame_id,timestamp,risk_score,risk_level\n0,0.0,0.1,LOW\n1,0.1,0.9,HIGH\n2,0.2,0.2,LOW\n",
        encoding="utf-8",
    )
    out = tmp_path / "events.json"
    runner = CliRunner()
    r = runner.invoke(app, ["build-events", str(risk), "-o", str(out), "--min-duration", "0"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "event_id" in text or "start_time" in text


def test_fightsafe_report_html_writes_artifact(tmp_path: Path) -> None:
    """`fightsafe report html` matches generate-report --only html."""
    run = write_minimal_pipeline_run(tmp_path / "r3", include_mvp_artifacts=False)
    runner = CliRunner()
    r = runner.invoke(app, ["report", "html", "--run", str(run)])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    assert (run / "report.html").is_file()
