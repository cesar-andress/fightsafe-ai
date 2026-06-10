"""
Unit tests for :mod:`fightsafe_ai.llm` (no Ollama daemon; fake :class:`BaseLLMClient`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.config import ExplanationsConfig, LLMFileConfig
from fightsafe_ai.llm.ollama_client import OllamaClientConfig
from fightsafe_ai.llm.risk_explainer import explain_risk_event, fallback_risk_explanation


# --- fake client (dependency injection) ---


@dataclass
class FakeLLMClient(BaseLLMClient):
    """In-memory client: records prompts, returns fixed text or raises."""

    response: str = "Synthetic model reply for human review. This is not a medical diagnosis."
    error: BaseException | None = None
    last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.error is not None:
            raise self.error
        return self.response


def _sample_event() -> dict[str, Any]:
    return {
        "event_id": 2,
        "start_time": 0.5,
        "end_time": 1.0,
        "max_risk_score": 0.88,
        "event_level": "CRITICAL",
        "triggered_rules": ["unstable_stance", "head_below_threshold"],
    }


# --- prompt generation ---


def test_explain_risk_event_prompt_includes_event_json_and_instructions() -> None:
    from fightsafe_ai.llm.prompts import explain_risk_event_prompt

    ex = ExplanationsConfig(
        include_triggered_rules=True,
        include_safety_disclaimer=True,
        recommend_human_review_threshold="HIGH",
    )
    ev = _sample_event()
    p = explain_risk_event_prompt(ev, ex)
    assert "Structured event (JSON)" in p
    assert str(ev["event_id"]) in p or "event_id" in p
    assert "CRITICAL" in p or "event_level" in p
    assert "This is not a medical diagnosis" in p
    assert "unstable_stance" in p or "triggered" in p.lower()


def test_build_risk_explanation_prompt_uses_explanations_from_file_when_mocked() -> None:
    from fightsafe_ai.llm import prompts

    ex = ExplanationsConfig(recommend_human_review_threshold="MEDIUM")
    with patch.object(prompts, "_explanations_from_file_or_default", return_value=ex):
        from fightsafe_ai.llm.prompts import build_risk_explanation_prompt

        p = build_risk_explanation_prompt(_sample_event())
    assert "combat-sports" in p.lower() or "analytics" in p.lower()
    assert "MEDIUM" in p or "mild" in p


def test_build_clip_summary_prompt_contains_clip_data() -> None:
    from fightsafe_ai.llm.prompts import build_clip_summary_prompt

    run_data = {
        "clip_id": "c1",
        "event_count": 3,
        "max_risk_score": 0.7,
        "notes": "synthetic",
    }
    p = build_clip_summary_prompt(run_data)
    assert "c1" in p
    assert "3" in p or "event" in p.lower()
    assert "heuristic" in p.lower() or "algorithmic" in p.lower()


# --- fallback behavior ---


def test_fallback_risk_explanation_includes_level_rules_and_disclaimer() -> None:
    text = fallback_risk_explanation(
        _sample_event(),
        ExplanationsConfig(
            include_triggered_rules=True,
            include_safety_disclaimer=True,
            recommend_human_review_threshold="HIGH",
        ),
    )
    assert "Event #2" in text
    assert "0.88" in text or "0.880" in text
    assert "unstable_stance" in text or "triggered" in text.lower()
    assert "not a medical" in text.lower() or "medical diagnosis" in text.lower()
    assert "review" in text.lower()


def test_explain_risk_event_skips_client_when_use_llm_false() -> None:
    fake = FakeLLMClient(response="SHOULD NOT APPEAR", error=RuntimeError("no network"))
    out = explain_risk_event(_sample_event(), fake, use_llm=False)
    assert "SHOULD NOT APPEAR" not in out
    assert "automated pipeline" in out.lower() or "max risk" in out.lower()
    assert fake.last_prompt is None


def test_explain_risk_event_uses_config_yaml_disabled_without_calling_model() -> None:
    from fightsafe_ai.llm import risk_explainer

    fake = FakeLLMClient(response="nope")
    with patch.object(
        risk_explainer,
        "_resolve_llm_enabled",
        return_value=False,
    ):
        out = explain_risk_event(_sample_event(), fake, use_llm=True)
    assert "nope" not in out
    assert "pipeline" in out.lower() or "Event #" in out


# --- success path: fake client + enabled LLM ---


def test_explain_risk_event_returns_model_text_when_enabled() -> None:
    from fightsafe_ai.llm import risk_explainer

    want = "Brief CRITICAL event summary. This is not a medical diagnosis."
    fake = FakeLLMClient(response=want)
    with patch.object(risk_explainer, "_resolve_llm_enabled", return_value=True):
        out = explain_risk_event(_sample_event(), fake, use_llm=True)
    assert out == want
    assert fake.last_prompt is not None
    assert "event_id" in (fake.last_prompt or "") or "CRITICAL" in (fake.last_prompt or "")


# --- error handling when client fails ---


def test_explain_risk_event_falls_back_on_llm_error() -> None:
    from fightsafe_ai.llm import risk_explainer

    fake = FakeLLMClient(error=LLMError("unavailable"))
    with patch.object(risk_explainer, "_resolve_llm_enabled", return_value=True):
        out = explain_risk_event(_sample_event(), fake, use_llm=True)
    assert "Event #" in out
    assert fake.last_prompt is not None


def test_explain_risk_event_falls_back_on_unexpected_exception() -> None:
    from fightsafe_ai.llm import risk_explainer

    fake = FakeLLMClient(error=ValueError("bad"))
    with patch.object(risk_explainer, "_resolve_llm_enabled", return_value=True):
        out = explain_risk_event(_sample_event(), fake, use_llm=True)
    assert "Event #" in out
    assert "0.88" in out or "0.880" in out


def test_explain_risk_event_falls_back_on_oserror() -> None:
    from fightsafe_ai.llm import risk_explainer

    fake = FakeLLMClient(error=OSError("network"))
    with patch.object(risk_explainer, "_resolve_llm_enabled", return_value=True):
        out = explain_risk_event(_sample_event(), fake, use_llm=True)
    assert "Event #" in out


# --- explainer.explain_event (higher-level) ---


def test_explain_event_uses_client_when_ollama_enabled_in_config() -> None:
    from fightsafe_ai.llm import explainer

    cfg = LLMFileConfig(
        ollama=OllamaClientConfig(enabled=True, model="fake"),
        explanations=ExplanationsConfig(),
    )
    want = "Narrative only. This is not a medical diagnosis."
    fake = FakeLLMClient(response=want)
    with (
        patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=cfg),
        patch("fightsafe_ai.llm.explainer.build_risk_explanation_prompt", return_value="PROMPT"),
    ):
        t = explainer.explain_event(_sample_event(), fake)
    assert t == want
    assert fake.last_prompt == "PROMPT"


def test_explain_event_falls_back_on_llm_error() -> None:
    from fightsafe_ai.llm import explainer

    p = _sample_event()
    cfg = LLMFileConfig(ollama=OllamaClientConfig(enabled=True), explanations=ExplanationsConfig())
    fake = FakeLLMClient(error=LLMError("down"))
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=cfg):
        t = explainer.explain_event(p, fake)
    assert "Event #" in t
    assert "not a medical" in t.lower() or "decision-support" in t.lower()


def test_explain_event_falls_back_on_load_config_error() -> None:
    """``load_llm_file_config`` can raise: template fallback (debug path)."""
    from fightsafe_ai.llm import explainer

    p = _sample_event()
    fake = FakeLLMClient(response="unused")
    with patch(
        "fightsafe_ai.llm.config.load_llm_file_config",
        side_effect=OSError("missing config"),
    ):
        t = explainer.explain_event(p, fake)
    assert "Event #" in t
    assert "not a medical" in t.lower() or "decision-support" in t.lower()


# --- explanation markdown formatting ---


def test_build_explanation_markdown_includes_header_and_disclaimer() -> None:
    from fightsafe_ai.llm.reporting import build_explanation_markdown

    body = "Rule-based one-liner. This is not a medical diagnosis."
    md = build_explanation_markdown(
        {**_sample_event(), "event_id": 2},
        body,
    )
    assert "# Risk event" in md
    assert "CRITICAL" in md
    assert body in md
    assert "not a medical" in md.lower() or "decision-support" in md.lower()


def test_write_explanation_markdown_uses_template_when_ollama_off(tmp_path: Path) -> None:
    """``write_explanation_markdown`` with ``use_ollama=False`` never calls the network."""
    from fightsafe_ai.llm.reporting import write_explanation_markdown

    out = tmp_path / "e.md"
    s = write_explanation_markdown(
        out,
        _sample_event(),
        use_ollama=False,
        llm_config=Path("/no/such/llm.yaml"),
    )
    assert "automated pipeline" in s.lower() or "Event #" in s
    t = out.read_text(encoding="utf-8")
    assert "# Risk event" in t
    assert "## Explanation" in t
    assert "not a medical" in t.lower() or "medical diagnosis" in t.lower()


# --- report enricher: inject fake client ---


def test_enrich_clip_narrative_uses_injected_client_when_config_enabled() -> None:
    from fightsafe_ai.llm.report_enricher import enrich_clip_narrative

    file_cfg = LLMFileConfig(
        ollama=replace(OllamaClientConfig(), enabled=True, model="x"),
        explanations=ExplanationsConfig(),
    )
    clip = {"clip_id": "z1", "event_count": 1, "source": "test"}
    client = FakeLLMClient(response="**Clip** summary. Not medical. Review recommended.")
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=file_cfg):
        t = enrich_clip_narrative(clip, client)
    assert "Clip" in t or "summary" in t.lower()
    assert client.last_prompt is not None
    assert "z1" in (client.last_prompt or "") or "z1" in t
