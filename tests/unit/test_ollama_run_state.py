"""Ollama run-level disable after model load / resource errors (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.ollama_client import is_ollama_model_load_or_resource_error
from fightsafe_ai.llm.risk_explainer import explain_risk_event
from fightsafe_ai.llm.run_state import LLMExplanationRunState
from fightsafe_ai.reports.summary import build_summary_dict


pytestmark = pytest.mark.unit


def test_is_load_resource_error_detects_500() -> None:
    assert is_ollama_model_load_or_resource_error(LLMError("Ollama HTTP 500: {}"))
    assert is_ollama_model_load_or_resource_error(LLMError("server OOM out of memory"))
    assert not is_ollama_model_load_or_resource_error(LLMError("Ollama HTTP 404: missing"))


def test_explain_stops_calling_llm_after_resource_error() -> None:
    n_calls = 0

    class _Bomb(BaseLLMClient):
        def generate(self, prompt: str) -> str:
            nonlocal n_calls
            n_calls += 1
            raise LLMError("Ollama HTTP 500: model loading")

    state = LLMExplanationRunState(llm_requested=True)
    c = _Bomb()
    t1 = explain_risk_event(
        {"event_id": 1, "max_risk_score": 0.5}, c, use_llm=True, run_state=state
    )
    t2 = explain_risk_event(
        {"event_id": 2, "max_risk_score": 0.5}, c, use_llm=True, run_state=state
    )
    assert n_calls == 1
    assert "Event #" in t1 and "Event #" in t2
    assert state.llm_disabled_for_run
    assert state.llm_error == "model failed to load"


def test_summary_merges_llm_state(tmp_path: Path) -> None:
    r = tmp_path / "r"
    r.mkdir()
    (r / "risk_scores.csv").write_text(
        "frame_id,timestamp,risk_score,risk_level\n0,0.0,0.1,LOW\n",
        encoding="utf-8",
    )
    (r / "events.json").write_text("[]", encoding="utf-8")
    (r / "llm_explanation_state.json").write_text(
        json.dumps(
            {
                "llm_requested": True,
                "llm_available": False,
                "llm_fallback": True,
                "llm_error": "model failed to load",
            }
        ),
        encoding="utf-8",
    )
    d = build_summary_dict(r)
    assert d["llm_requested"] is True
    assert d["llm_available"] is False
    assert d["llm_fallback"] is True
    assert d["llm_error"] == "model failed to load"
