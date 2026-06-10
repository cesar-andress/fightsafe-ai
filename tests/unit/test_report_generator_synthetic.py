"""Tests for clip-level Markdown report generation (isolated)."""

from __future__ import annotations

import importlib
from pathlib import Path

from tests.support.isolated import load_report_generator


def test_generate_clip_report_sections(tmp_path: Path) -> None:
    load_report_generator()
    rg = importlib.import_module("fightsafe_ai.llm.report_generator")
    out = tmp_path / "clip_report.md"
    p = rg.generate_clip_report(
        {
            "clip_id": "clip-001",
            "video_path": "/data/sessions/spar_A/cam1.mp4",
            "detected_events": [
                {
                    "event_id": 0,
                    "event_level": "HIGH",
                    "max_risk_score": 0.82,
                    "start_time": 12.0,
                    "end_time": 14.5,
                }
            ],
            "max_risk_score": 0.82,
            "triggered_rules_summary": ["fast_downward_motion", "high_instability"],
            "ollama_explanations": "Summarized narrative for documentation (not medical).",
        },
        out,
        llm_client=None,
    )
    assert p == out.resolve()
    text = out.read_text(encoding="utf-8")
    assert "# FightSafe AI Safety Report" in text
    assert "david.martinm@ucjc.edu" in text
    assert "# Combat sports safety review" in text
    assert "## 1. Clip summary" in text
    assert "## 2. Detected risk events" in text
    assert "## 3. Highest risk moment" in text
    assert "## 4. Triggered rules" in text
    assert "## 5. AI-generated explanation" in text
    assert "Summarized narrative" in text
    assert "## 6. Human review recommendation" in text
    assert "## 7. Safety disclaimer" in text
    assert "not a medical" in text.lower() or "not** a medical" in text.lower()


def test_generate_clip_report_with_mock_llm(tmp_path: Path) -> None:
    from unittest.mock import patch

    load_report_generator()
    rg = importlib.import_module("fightsafe_ai.llm.report_generator")
    base = importlib.import_module("fightsafe_ai.llm.base")
    cfg = importlib.import_module("fightsafe_ai.llm.config")
    oll = importlib.import_module("fightsafe_ai.llm.ollama_client")

    class _Stub(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            return "LLM bullet one. LLM bullet two."

    fake = cfg.LLMFileConfig(
        ollama=oll.OllamaClientConfig(enabled=True),
        explanations=cfg.ExplanationsConfig(),
    )
    p = tmp_path / "r2.md"
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=fake):
        rg.generate_clip_report(
            {
                "clip_id": "c2",
                "video_path": "v.mp4",
                "detected_events": 1,
                "max_risk_score": 0.1,
                "triggered_rules_summary": "none",
            },
            p,
            llm_client=_Stub(),
        )
    t = p.read_text(encoding="utf-8")
    assert "LLM bullet" in t
