"""Offline smoke tests for the Typer CLI (``--help`` only; no videos, no network)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fightsafe_ai.cli import app


pytestmark = pytest.mark.unit

_SUBCOMMANDS = (
    ["--help"],
    ["download", "--help"],
    ["cut", "--help"],
    ["extract-frames", "--help"],
    ["estimate-pose", "--help"],
    ["bench-pose-backends", "--help"],
    ["export-rtmpose-onnx", "--help"],
    ["compute-features", "--help"],
    ["detect-risk", "--help"],
    ["build-events", "--help"],
    ["render-overlay", "--help"],
    ["explain-event", "--help"],
    ["run-pipeline", "--help"],
    ["demo", "--help"],
    ["demo-youtube", "--help"],
    ["qa", "--help"],
    ["generate-report", "--help"],
    ["report", "--help"],
    ["report", "html", "--help"],
    ["plot-risk", "--help"],
    ["risk-ablation", "--help"],
    ["risk-ablation-all", "--help"],
    ["suggest-annotations", "--help"],
    ["run-case-studies", "--help"],
    ["evaluate", "--help"],
    ["evaluate-case-studies", "--help"],
)


@pytest.mark.parametrize("args", _SUBCOMMANDS)
def test_cli_help_exits_zero(args: list[str]) -> None:
    runner = CliRunner()
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output + ((r.exception and str(r.exception)) or "")
    assert "FightSafe" in (r.stdout or "") or "fightsafe" in (r.stdout or "").lower()


def test_generate_report_exposes_sub_help_flags() -> None:
    """Single command ``generate-report``; ensure --only is documented in --help output."""
    runner = CliRunner()
    r = runner.invoke(app, ["generate-report", "--help"])
    assert r.exit_code == 0
    out = r.stdout or ""
    assert "generate-report" in out.lower() or "report" in out.lower()
    assert "--only" in out or "only" in out.lower()
