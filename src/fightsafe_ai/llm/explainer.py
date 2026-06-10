"""
Single-event **explainability** via :class:`~fightsafe_ai.llm.base.BaseLLMClient` (e.g. Ollama).

**Not used for** risk detection, pose estimation, or feature computation — only post-hoc text
for human review. Fails gracefully when Ollama is down or disabled in :file:`configs/llm.yaml`.
"""

from __future__ import annotations

import logging
from typing import Any

from fightsafe_ai.exceptions import ConfigurationError, LLMError
from fightsafe_ai.hci.alerts import RefereeAlert
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.prompts import (
    build_multi_signal_explanation_prompt,
    build_risk_explanation_prompt,
)
from fightsafe_ai.llm.risk_explainer import (
    fallback_multi_signal_explanation,
    fallback_risk_explanation,
)
from fightsafe_ai.risk.fusion import RiskDecision


logger = logging.getLogger(__name__)


def explain_event(event: dict[str, Any], client: BaseLLMClient) -> str:
    """
    Return a concise explanation for one merged risk event (structured dict).

    Uses the configured LLM only when ``ollama.enabled`` is true in :file:`configs/llm.yaml`
    and generation succeeds; otherwise returns :func:`~fightsafe_ai.llm.risk_explainer.fallback_risk_explanation`.

    The prompt requires referencing triggered rules and timestamps where available, nudging
    human review for **HIGH** / **CRITICAL**, and stating that **this is not a medical diagnosis**.
    """
    from fightsafe_ai.llm.config import load_llm_file_config

    try:
        if not load_llm_file_config().ollama.enabled:
            return fallback_risk_explanation(event, None)
    except (ConfigurationError, OSError, ValueError) as e:
        logger.debug("No usable LLM config; template-only explanation: %s", e)
        return fallback_risk_explanation(event, None)

    prompt = build_risk_explanation_prompt(event)
    try:
        return client.generate(prompt).strip()
    except LLMError as e:
        logger.warning("explain_event: Ollama/LLM failed, using rule-based text: %s", e)
        return fallback_risk_explanation(event, None)
    except OSError as e:  # pragma: no cover
        logger.warning("explain_event: I/O error, using rule-based text: %s", e)
        return fallback_risk_explanation(event, None)
    except Exception as e:  # pragma: no cover
        logger.exception("explain_event: unexpected error, using rule-based text: %s", e)
        return fallback_risk_explanation(event, None)


def risk_decision_to_dict(d: RiskDecision) -> dict[str, Any]:
    return {
        "timestamp": d.timestamp,
        "fighter_id": d.fighter_id,
        "risk_score": d.risk_score,
        "risk_level": d.risk_level.value,
        "triggered_signals": list(d.triggered_signals),
        "explanation_facts": list(d.explanation_facts),
    }


def referee_alert_to_dict(a: RefereeAlert) -> dict[str, Any]:
    return {
        "timestamp": a.timestamp,
        "fighter_id": a.fighter_id,
        "alert_level": a.alert_level.value,
        "short_message": a.short_message,
        "reason": a.reason,
        "triggered_signals": list(a.triggered_signals),
        "recommended_human_action": a.recommended_human_action,
    }


def multi_signal_context_to_dict(
    *,
    risk_decision: RiskDecision | None = None,
    referee_alert: RefereeAlert | None = None,
    detected_signals: list[str] | None = None,
    signal_confidences: dict[str, float] | None = None,
    time_range_start: float | None = None,
    time_range_end: float | None = None,
) -> dict[str, Any]:
    """
    Build the plain ``dict`` expected by
    :func:`~fightsafe_ai.llm.prompts.build_multi_signal_explanation_prompt` and
    :func:`~fightsafe_ai.llm.risk_explainer.fallback_multi_signal_explanation`.
    """
    out: dict[str, Any] = {}
    if risk_decision is not None:
        out["risk_decision"] = risk_decision_to_dict(risk_decision)
    if referee_alert is not None:
        out["referee_alert"] = referee_alert_to_dict(referee_alert)
    if detected_signals is not None and len(detected_signals) > 0:
        out["detected_signals"] = list(detected_signals)
    if signal_confidences is not None and len(signal_confidences) > 0:
        out["signal_confidences"] = dict(signal_confidences)
    if time_range_start is not None and time_range_end is not None:
        out["time_range"] = {
            "start": float(time_range_start),
            "end": float(time_range_end),
        }
    return out


def explain_multi_signal(context: dict[str, Any], client: BaseLLMClient) -> str:
    """
    Concise HITL explanation for a fused + HCI + multi-signal row.

    Uses the LLM when ``ollama.enabled``; otherwise
    :func:`~fightsafe_ai.llm.risk_explainer.fallback_multi_signal_explanation`.
    """
    from fightsafe_ai.llm.config import load_llm_file_config

    try:
        if not load_llm_file_config().ollama.enabled:
            return fallback_multi_signal_explanation(context)
    except (ConfigurationError, OSError, ValueError) as e:
        logger.debug("explain_multi_signal: no usable LLM config; template-only: %s", e)
        return fallback_multi_signal_explanation(context)

    prompt = build_multi_signal_explanation_prompt(context)
    try:
        return client.generate(prompt).strip()
    except LLMError as e:
        logger.warning("explain_multi_signal: Ollama/LLM failed, using rule-based text: %s", e)
        return fallback_multi_signal_explanation(context)
    except OSError as e:  # pragma: no cover
        logger.warning("explain_multi_signal: I/O error, using rule-based text: %s", e)
        return fallback_multi_signal_explanation(context)
    except Exception as e:  # pragma: no cover
        logger.exception("explain_multi_signal: unexpected error, using rule-based text: %s", e)
        return fallback_multi_signal_explanation(context)


__all__ = [
    "explain_event",
    "explain_multi_signal",
    "multi_signal_context_to_dict",
    "referee_alert_to_dict",
    "risk_decision_to_dict",
]
