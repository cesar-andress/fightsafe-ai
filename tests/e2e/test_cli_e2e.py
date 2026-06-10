"""
Offline e2e smoke: invoke the installed CLI entrypoint (no network, no video files).

Run via ``make test-e2e`` or ``pytest tests/e2e``; not part of default ``testpaths``.
"""

from __future__ import annotations

import subprocess
import sys

from tests.support.subprocess_env import REPO_ROOT, env_with_src_pythonpath


def test_fightsafe_help_exits_zero() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "fightsafe_ai.cli", "--help"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
    )
    assert r.returncode == 0
    out = (r.stdout + r.stderr).lower()
    assert "fightsafe" in out
