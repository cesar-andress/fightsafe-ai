"""
Unit tests for multi-signal Ollama explainability (no daemon; mock :class:`BaseLLMClient`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.hci.alerts import RefereeAlert, RefereeAlertLevel
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.config import ExplanationsConfig, LLMFileConfig
from fightsafe_ai.llm.ollama_client import OllamaClientConfig
from fightsafe_ai.llm.risk_explainer import fallback_multi_signal_explanation
from fightsafe_ai.risk.fusion import RiskDecision
from fightsafe_ai.risk.levels import RiskLevelName


@dataclass
class MockLLMClient(BaseLLMClient):
    response: str = "Multi-signal HITL paragraph. This is not a medical diagnosis."
    error: BaseException | None = None
    last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.error is not None:
            raise self.error
        return self.response


def _ctx() -> dict[str, Any]:
    rd = RiskDecision(
        timestamp=12.0,
        fighter_id="A",
        risk_score=0.86,
        risk_level=RiskLevelName.HIGH,
        triggered_signals=("unstable_stance",),
        explanation_facts=("fused from pose + inactivity heuristics",),
    )
    al = RefereeAlert(
        timestamp=12.0,
        fighter_id="A",
        alert_level=RefereeAlertLevel.WARNING,
        short_message="Possible balance loss — review",
        reason="unstable_stance",
        triggered_signals=("unstable_stance",),
        recommended_human_action="Check video; rules apply",
    )
    from fightsafe_ai.llm import explainer

    return explainer.multi_signal_context_to_dict(
        risk_decision=rd,
        referee_alert=al,
        detected_signals=["unstable_stance", "head_pose_anomaly"],
        signal_confidences={"unstable_stance": 0.71, "head_pose_anomaly": 0.42},
        time_range_start=10.0,
        time_range_end=14.5,
    )


def test_multi_signal_context_to_dict_shape() -> None:
    c = _ctx()
    assert c["risk_decision"]["risk_level"] == "HIGH"
    assert c["referee_alert"]["alert_level"] == "WARNING"
    assert "unstable_stance" in c["detected_signals"]
    assert c["signal_confidences"]["unstable_stance"] == 0.71
    assert c["time_range"] == {"start": 10.0, "end": 14.5}


def test_build_multi_signal_explanation_prompt_includes_json_and_hivocab() -> None:
    from fightsafe_ai.llm import prompts
    from fightsafe_ai.llm.prompts import build_multi_signal_explanation_prompt

    c = _ctx()
    with patch.object(
        prompts, "_explanations_from_file_or_default", return_value=ExplanationsConfig()
    ):
        p = build_multi_signal_explanation_prompt(c)
    assert "unstable_stance" in p
    assert "HIGH" in p or "risk_level" in p
    assert "10.0" in p or "14.5" in p
    assert "human" in p.lower() or "referee" in p.lower()
    assert "not" in p.lower()  # no diagnosis / HITL


def test_fallback_multi_signal_explanation_is_deterministic() -> None:
    a = fallback_multi_signal_explanation(_ctx())
    b = fallback_multi_signal_explanation(_ctx())
    assert a == b
    assert "heuristic" in a.lower() or "decision-support" in a.lower()
    assert "10.000" in a or "10.0" in a
    assert "referee" in a.lower() or "review" in a.lower() or "human" in a.lower()
    assert "not a medical" in a.lower() or "medical diagnosis" in a.lower()


def test_explain_multi_signal_falls_back_when_ollama_disabled() -> None:
    from fightsafe_ai.llm import explainer

    c = _ctx()
    cfg = LLMFileConfig(ollama=OllamaClientConfig(enabled=False), explanations=ExplanationsConfig())
    fake = MockLLMClient(response="This LLM line must not be used when Ollama is off.")
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=cfg):
        out = explainer.explain_multi_signal(c, fake)
    assert fake.response not in out
    assert fake.last_prompt is None
    assert out == fallback_multi_signal_explanation(c)


def test_explain_multi_signal_uses_client_when_enabled() -> None:
    from fightsafe_ai.llm import explainer

    c = _ctx()
    want = "Custom Ollama reply. This is not a medical diagnosis."
    cfg = LLMFileConfig(
        ollama=OllamaClientConfig(enabled=True, model="fake"), explanations=ExplanationsConfig()
    )
    fake = MockLLMClient(response=want)
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=cfg):
        out = explainer.explain_multi_signal(c, fake)
    assert out == want
    assert fake.last_prompt is not None
    assert "unstable_stance" in (fake.last_prompt or "") or "WARNING" in (fake.last_prompt or "")


def test_explain_multi_signal_falls_back_on_llm_error() -> None:
    from fightsafe_ai.llm import explainer

    c = _ctx()
    cfg = LLMFileConfig(ollama=OllamaClientConfig(enabled=True), explanations=ExplanationsConfig())
    fake = MockLLMClient(error=LLMError("unavailable"))
    with patch("fightsafe_ai.llm.config.load_llm_file_config", return_value=cfg):
        out = explainer.explain_multi_signal(c, fake)
    assert "heuristic" in out.lower() or "decision-support" in out.lower()
    assert out == fallback_multi_signal_explanation(c)
