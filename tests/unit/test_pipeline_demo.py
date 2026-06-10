"""Smoke: demo orchestrator is importable (full run needs video and GPU/CPU time)."""

from __future__ import annotations

from fightsafe_ai.pipeline.demo import run_e2e_demo


def test_run_e2e_demo_is_callable() -> None:
    assert callable(run_e2e_demo)
