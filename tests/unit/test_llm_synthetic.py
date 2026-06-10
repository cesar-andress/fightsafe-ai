"""Unit tests for optional Ollama LLM module (isolated, no Ollama daemon)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import patch

from tests.support.isolated import load_llm


def test_explain_risk_event_fallback_on_llm_error() -> None:
    load_llm()
    re = importlib.import_module("fightsafe_ai.llm.risk_explainer")
    base = importlib.import_module("fightsafe_ai.llm.base")
    ex = importlib.import_module("fightsafe_ai.exceptions")

    class _Bad(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            raise ex.LLMError("unavailable")

    data = {
        "event_id": 0,
        "start_time": 1.0,
        "end_time": 1.5,
        "max_risk_score": 0.9,
        "event_level": "CRITICAL",
        "triggered_rules": ["fast_downward_motion", "high_instability"],
    }
    out = re.explain_risk_event(data, _Bad())
    assert "decision-support" in out or "heuristic" in out.lower()
    assert "not a medical" in out.lower() or "medical diagnosis" in out.lower()
    assert "0.9" in out or "0.90" in out
    assert "1.000" in out or "1.0" in out
    assert "fast_downward" in out or "trigg" in out.lower()
    assert "review" in out.lower()


def test_fallback_risk_explanation_high_mandatory_review() -> None:
    load_llm()
    re = importlib.import_module("fightsafe_ai.llm.risk_explainer")
    text = re.fallback_risk_explanation(
        {
            "start_time": 0.0,
            "end_time": 0.1,
            "max_risk_score": 0.5,
            "event_level": "HIGH",
            "triggered_rules": ["large_torso_angle"],
        }
    )
    assert "recommended" in text.lower() or "review" in text.lower()
    assert "not a medical" in text.lower()


def test_ollama_client_generate_parses_response() -> None:
    load_llm()
    oc = importlib.import_module("fightsafe_ai.llm.ollama_client")
    cfg = oc.OllamaClient(
        base_url="http://127.0.0.1:99999",
        model="llama3.1",
        temperature=0.1,
        timeout=1.0,
    )
    body = json.dumps({"model": "llama3.1", "response": "OK", "done": True}).encode("utf-8")

    class _Resp:
        def read(self) -> bytes:
            return body

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    with patch("fightsafe_ai.llm.ollama_client.urlopen", return_value=_Resp()):
        assert cfg.generate("hi") == "OK"


def test_ollama_client_load_config(tmp_path: Path) -> None:
    load_llm()
    oc = importlib.import_module("fightsafe_ai.llm.ollama_client")
    p = tmp_path / "llm.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://localhost:11435'\n"
        "  model: test-model\n  temperature: 0.4\n  timeout_seconds: 30.0\n",
        encoding="utf-8",
    )
    c = oc.load_ollama_config(p)
    assert c.enabled is True
    assert c.base_url == "http://localhost:11435"
    assert c.model == "test-model"
    assert c.temperature == 0.4
    assert c.timeout == 30.0


def test_explain_risk_event_skips_network_when_use_llm_false() -> None:
    load_llm()
    re = importlib.import_module("fightsafe_ai.llm.risk_explainer")
    base = importlib.import_module("fightsafe_ai.llm.base")

    class _NeverCalled(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            raise AssertionError("LLM should not be called when use_llm=False")

    data = {
        "event_id": 1,
        "start_time": 0.0,
        "end_time": 0.5,
        "max_risk_score": 0.2,
        "event_level": "LOW",
        "triggered_rules": ["x"],
    }
    out = re.explain_risk_event(data, _NeverCalled(), use_llm=False)
    assert "max risk" in out.lower()
    assert "0.2" in out or "0.20" in out


def test_prompts_contain_risk_context() -> None:
    load_llm()
    pr = importlib.import_module("fightsafe_ai.llm.prompts")
    t = pr.explain_risk_event_prompt(
        {"start_time": 0.0, "end_time": 1.0, "max_risk_score": 0.3, "triggered_rules": ["a"]}
    )
    assert "0.0 s" in t or "video time" in t
    assert "a" in t
    c = pr.summarize_clip_prompt({"event_count": 1, "clip_id": "c1"})
    d = pr.build_clip_summary_prompt({"event_count": 1, "clip_id": "c1"})
    assert c == d
    assert "heuristic" in c.lower() or "algorithmic" in c.lower()
    a = pr.suggest_annotation_prompt({"event_id": 0})
    assert "annotation" in a.lower() or "annot" in a.lower()


def test_explain_event_uses_fallback_when_ollama_disabled() -> None:
    """Ollama off: ``explain_event`` is template-only; never calls the client."""
    load_llm()
    exp = importlib.import_module("fightsafe_ai.llm.explainer")
    base = importlib.import_module("fightsafe_ai.llm.base")
    cfg = importlib.import_module("fightsafe_ai.llm.config")
    oll = importlib.import_module("fightsafe_ai.llm.ollama_client")

    class _Never(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            raise AssertionError("LLM should not be called when Ollama is disabled")

    data = {
        "event_id": 0,
        "start_time": 0.0,
        "end_time": 0.1,
        "max_risk_score": 0.5,
        "event_level": "HIGH",
        "triggered_rules": ["r1"],
    }
    fake = cfg.LLMFileConfig(
        ollama=oll.OllamaClientConfig(enabled=False),
        explanations=cfg.ExplanationsConfig(),
    )
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=fake):
        out = exp.explain_event(data, _Never())
    assert "This is not a medical diagnosis" in out
    assert "0.0" in out


def test_explain_event_calls_llm_when_ollama_enabled() -> None:
    load_llm()
    exp = importlib.import_module("fightsafe_ai.llm.explainer")
    base = importlib.import_module("fightsafe_ai.llm.base")
    cfg = importlib.import_module("fightsafe_ai.llm.config")
    oll = importlib.import_module("fightsafe_ai.llm.ollama_client")

    class _Ok(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            assert "not a medical diagnosis" in prompt.lower() or "medical" in prompt.lower()
            return "  synthetic llm line  "

    data = {
        "event_id": 0,
        "start_time": 0.0,
        "end_time": 0.1,
        "max_risk_score": 0.3,
        "event_level": "MEDIUM",
        "triggered_rules": ["r1"],
    }
    fake = cfg.LLMFileConfig(
        ollama=oll.OllamaClientConfig(enabled=True),
        explanations=cfg.ExplanationsConfig(),
    )
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=fake):
        out = exp.explain_event(data, _Ok())
    assert out == "synthetic llm line"


def test_rule_based_clip_narrative_is_deterministic() -> None:
    from tests.support.isolated import load_report_generator

    load_report_generator()
    ren = importlib.import_module("fightsafe_ai.llm.report_enricher")
    t = ren.rule_based_clip_narrative({"clip_id": "x", "max_risk_score": 0.4, "event_count": 1})
    assert "not a medical diagnosis" in t.lower()
    assert "0.4" in t
    assert "x" in t


def test_enrich_clip_narrative_uses_llm_when_enabled() -> None:
    from tests.support.isolated import load_report_generator

    load_report_generator()
    ren = importlib.import_module("fightsafe_ai.llm.report_enricher")
    base = importlib.import_module("fightsafe_ai.llm.base")
    cfg = importlib.import_module("fightsafe_ai.llm.config")
    oll = importlib.import_module("fightsafe_ai.llm.ollama_client")

    class _Ok(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            return "clip summary from llm"

    fake = cfg.LLMFileConfig(
        ollama=oll.OllamaClientConfig(enabled=True),
        explanations=cfg.ExplanationsConfig(),
    )
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=fake):
        t = ren.enrich_clip_narrative({"clip_id": "c9", "event_count": 2}, _Ok())
    assert t == "clip summary from llm"

    ex = importlib.import_module("fightsafe_ai.exceptions")

    class _Bad(base.BaseLLMClient):  # type: ignore[misc, name-defined]
        def generate(self, prompt: str) -> str:
            raise ex.LLMError("down")

    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=fake):
        t2 = ren.enrich_clip_narrative({"clip_id": "c8"}, _Bad())
    assert "not a medical diagnosis" in t2.lower()
