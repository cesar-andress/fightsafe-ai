"""Unit tests for :mod:`fightsafe_ai.annotation.llm_assist` (mocked Ollama; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fixtures.mvp_runs import write_minimal_pipeline_run

from fightsafe_ai.annotation.llm_assist import (
    SUGGESTIONS_FORMAT_VERSION,
    _parse_json_object_from_llm_text,
    build_assist_bundle,
    run_pipeline_suggest_annotations,
    suggest_annotation_suggestions_ollama,
)
from fightsafe_ai.llm.ollama_client import OllamaClientConfig


pytestmark = pytest.mark.unit

_GOOD: str = json.dumps(
    {
        "suggestions": [
            {
                "event_id": 0,
                "start_time": 0.0,
                "end_time": 0.12,
                "suggested_event_type": "INSTABILITY",
                "confidence": 0.35,
                "rationale": "Heuristic HIGH window; not definitive.",
            }
        ]
    }
)


def test_parse_json_fenced() -> None:
    t = "```json\n" + _GOOD + "\n```"
    p = _parse_json_object_from_llm_text(t)
    assert p is not None
    assert isinstance(p, dict)
    assert len(p.get("suggestions", [])) == 1


def test_build_assist_bundle_minimal_run(tmp_path: Path) -> None:
    write_minimal_pipeline_run(tmp_path)
    b = build_assist_bundle(tmp_path, max_frames_per_event=1)
    assert b.get("candidates", [{}])[0].get("event_id") == 0
    assert "risk_signal_summary" in b["candidates"][0]


def test_suggest_ollama_mocked(tmp_path: Path) -> None:
    write_minimal_pipeline_run(tmp_path)
    b = build_assist_bundle(tmp_path)

    class _Gen:
        def generate(self, prompt: str) -> str:
            assert "candidates" in prompt or "run_dir" in prompt
            return _GOOD

    sugs, err = suggest_annotation_suggestions_ollama(
        b,
        ollama_config=OllamaClientConfig(enabled=True),
        ollama_client=_Gen(),
    )
    assert err is None
    assert len(sugs) == 1
    assert sugs[0]["requires_human_confirmation"] is True
    assert sugs[0]["suggested_event_type"] == "INSTABILITY"


def test_run_pipeline_no_ollama(tmp_path: Path) -> None:
    write_minimal_pipeline_run(tmp_path)
    o = tmp_path / "s.json"
    d = run_pipeline_suggest_annotations(
        tmp_path, o, use_ollama=False, use_vlm=False, max_frames_per_event=1
    )
    assert d["suggestions"] == []
    assert d.get("ollama_error") is None
    assert d["format_version"] == SUGGESTIONS_FORMAT_VERSION
    assert o.is_file()


def test_run_pipeline_ollama_disabled_with_no_client(tmp_path: Path) -> None:
    write_minimal_pipeline_run(tmp_path)
    llm_yaml = tmp_path / "llm_f.yaml"
    llm_yaml.write_text(
        "ollama:\n  enabled: false\n  base_url: http://localhost:11434\n  model: x\n"
        "  temperature: 0.1\n  timeout_seconds: 5\nexplanations: {}\n",
        encoding="utf-8",
    )
    o = tmp_path / "o.json"
    d = run_pipeline_suggest_annotations(
        tmp_path,
        o,
        use_ollama=True,
        use_vlm=False,
        max_frames_per_event=1,
        llm_config=llm_yaml,
    )
    assert d.get("suggestions", []) == []
    assert d.get("ollama_error") is not None
    assert d["ollama_used"] is False
