"""
Subprocess e2e: real ``python -m fightsafe_ai.cli`` (no Ollama, no network).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.support.subprocess_env import REPO_ROOT, env_with_src_pythonpath


pytestmark = [pytest.mark.e2e]


def test_subprocess_fightsafe_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "fightsafe_ai.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
        check=False,
    )
    assert r.returncode == 0
    out = (r.stdout + r.stderr).lower()
    assert "fightsafe" in out
    assert "download" in out or "usage" in out or "command" in out


def test_subprocess_qa_and_generate_report_on_synthetic_run(tmp_path: Path) -> None:
    from tests.fixtures.mvp_runs import write_minimal_pipeline_run, write_mvp_qa_passing_run

    run = write_mvp_qa_passing_run(tmp_path / "ok")
    r1 = subprocess.run(  # noqa: S603 — trusted interpreter path from tests
        [sys.executable, "-m", "fightsafe_ai.cli", "qa", "-r", str(run)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
        check=False,
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr

    rpt = write_minimal_pipeline_run(tmp_path / "rep", include_mvp_artifacts=False)
    r2 = subprocess.run(  # noqa: S603 — trusted interpreter path from tests
        [sys.executable, "-m", "fightsafe_ai.cli", "generate-report", "-r", str(rpt)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
        check=False,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
