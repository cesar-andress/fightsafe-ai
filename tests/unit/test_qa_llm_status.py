"""Tests for :mod:`fightsafe_ai.qa.llm_status` and LLM fields in :func:`run_quality_checks`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fightsafe_ai.qa import llm_status, metrics as qm
from fightsafe_ai.qa.validators import run_quality_checks, write_qa_report_json


def _minimal_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "frame_id": ["0", "1"],
            "timestamp": [0.0, 0.1],
            "risk_score": [0.2, 0.3],
            "risk_level": ["LOW", "LOW"],
        }
    ).to_csv(run / "risk_scores.csv", index=False)
    (run / "events.json").write_text(
        """
[{"event_id":0,"start_time":0.0,"end_time":0.1,"max_risk_score":0.2,"event_level":"MEDIUM"}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (run / "pose_keypoints.csv").write_text(
        "frame_id,keypoint_name,x,y\n0,n,0,0\n1,n,0,0\n", encoding="utf-8"
    )
    (run / "features.csv").write_text("frame_id\n0\n", encoding="utf-8")
    (run / "report.md").write_text("r", encoding="utf-8")
    (run / "output_overlay.mp4").write_bytes(b"ok")
    (run / "frames").mkdir()
    (run / "frames" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    return run


def test_load_ollama_enabled_in_default_config_false_on_ollama_config_error() -> None:
    # Use OSError, not ConfigurationError: isolated :mod:`tests.support.isolated`
    # re-exec may replace exception classes, but builtins stay identical.
    with patch(
        "fightsafe_ai.llm.ollama_client.load_ollama_config",
        side_effect=OSError("unreadable"),
    ):
        assert llm_status.load_ollama_enabled_in_default_config() is False


def test_build_llm_qa_metrics_warns_when_enabled_but_no_files(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    with patch.object(llm_status, "load_ollama_enabled_in_default_config", return_value=True):
        m, warns = llm_status.build_llm_qa_metrics(run, n_events=1)
    assert m["llm_enabled"] is True
    assert m["llm_used"] is False
    assert m["llm_success_rate"] is None
    assert any(n == "llm_explanations_missing" for n, _ in warns)


def test_build_llm_qa_metrics_success_rate_with_model_style_files(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    ex = run / "explanations"
    ex.mkdir()
    (ex / "event_0000.md").write_text(
        "## Explanation\n\nThis is a non-template paragraph.\n\n---\n",
        encoding="utf-8",
    )
    with patch.object(llm_status, "load_ollama_enabled_in_default_config", return_value=False):
        m, warns = llm_status.build_llm_qa_metrics(run, n_events=1)
    assert m["llm_used"] is True
    assert m["llm_success_rate"] == 1.0
    assert not any("template_fallback" in n for n, _ in warns)


def test_build_llm_qa_metrics_llm_error_from_state_file(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    (run / "llm_explanation_state.json").write_text(
        '{"llm_requested": true, "llm_available": false, '
        '"llm_fallback": true, "llm_error": "model failed to load"}',
        encoding="utf-8",
    )
    m, warns = llm_status.build_llm_qa_metrics(run, n_events=1)
    assert m["llm_error"] == "model failed to load"
    assert any(n == "llm_explanation_error" for n, _ in warns)


def test_build_llm_qa_metrics_template_fallback_warns(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    ex = run / "explanations"
    ex.mkdir()
    tpl = (
        "Event #0: the automated pipeline reported **max risk** (level **HIGH**), over **0.1 s to 0.2 s**"
        ". Heuristic **triggered rules** in context: r."
    )
    (ex / "event_0000.md").write_text(f"## Explanation\n\n{tpl}\n", encoding="utf-8")
    with patch.object(llm_status, "load_ollama_enabled_in_default_config", return_value=True):
        m, warns = llm_status.build_llm_qa_metrics(run, n_events=1)
    assert m["llm_success_rate"] == 0.0
    assert any(n == "llm_explanations_template_fallback" for n, w in warns)


def test_run_quality_checks_does_not_fail_on_llm(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    with patch.object(llm_status, "load_ollama_enabled_in_default_config", return_value=True):
        rep = run_quality_checks(run, require_frames=True)
    assert rep.passed is True
    assert rep.metrics.get("llm_enabled") is True
    assert rep.metrics.get("llm_success_rate") is None
    assert any("llm_explanations_missing" in r.name for r in rep.results)
    p = write_qa_report_json(run / "qa_report.json", rep)
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "pass"
    for k in ("llm_enabled", "llm_used", "llm_success_rate"):
        assert k in data["metrics"]
    for k in qm.QA_REPORT_METRIC_KEYS:
        assert k in data["metrics"]


def test_incomplete_explanation_file_count_warns(tmp_path: Path) -> None:
    run = _minimal_run_dir(tmp_path)
    (run / "events.json").write_text(
        '[{"event_id":0,"event_level":"MEDIUM"},{"event_id":1,"event_level":"HIGH"}]\n',
        encoding="utf-8",
    )
    ex = run / "explanations"
    ex.mkdir()
    (ex / "event_0000.md").write_text("## Explanation\n\nOnly one file.\n", encoding="utf-8")
    m, warns = llm_status.build_llm_qa_metrics(run, n_events=2)
    assert m["llm_used"] is True
    assert any(n == "llm_explanations_incomplete" for n, w in warns)
