"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

LLM-backed narrative for **structured** risk events (Ollama optional).

Computer vision and the rule engine produce the numbers; this module only formats text
for human review. If the LLM is down, a deterministic fallback string is used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fightsafe_ai.exceptions import ConfigurationError, LLMError
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.ollama_client import is_ollama_model_load_or_resource_error


if TYPE_CHECKING:
    from fightsafe_ai.llm.run_state import LLMExplanationRunState
from fightsafe_ai.llm.config import (
    ExplanationsConfig,
    event_level_reaches_threshold,
    load_llm_file_config,
)
from fightsafe_ai.llm.prompts import explain_risk_event_prompt


logger = logging.getLogger(__name__)


def _format_rules_phrase(event_data: dict[str, Any]) -> str:
    rules = event_data.get("triggered_rules", event_data.get("rules"))
    if not rules:
        return "no specific triggered-rules list was attached to this event"
    if isinstance(rules, (list, tuple)):
        return ", ".join(str(r) for r in rules)
    return str(rules)


def _time_range_phrase(event_data: dict[str, Any]) -> str:
    t0 = event_data.get("start_time")
    t1 = event_data.get("end_time")
    if t0 is not None and t1 is not None:
        return f"{t0:.3f} s to {t1:.3f} s"
    return "an unknown time range (timestamps not provided)"


def _max_score_phrase(event_data: dict[str, Any]) -> str:
    s = event_data.get("max_risk_score")
    if s is None or (isinstance(s, float) and s != s):  # NaN
        return "N/A"
    try:
        return f"{float(s):.3f}"
    except (TypeError, ValueError):
        return str(s)


def _level_phrase(event_data: dict[str, Any]) -> str:
    lv = event_data.get("event_level") or event_data.get("risk_level")
    if lv is None:
        return "UNSPECIFIED"
    return str(lv).strip().upper()


def _resolve_explanations(ex: ExplanationsConfig | None) -> ExplanationsConfig:
    if ex is not None:
        return ex
    try:
        return load_llm_file_config().explanations
    except (ConfigurationError, OSError) as e:
        logger.debug("Explanations config unavailable, using defaults: %s", e)
        return ExplanationsConfig()


def _resolve_llm_enabled(override: bool | None) -> bool:
    if override is not None:
        return override
    try:
        return load_llm_file_config().ollama.enabled
    except Exception:
        return False


def fallback_risk_explanation(
    event_data: dict[str, Any], explanations: ExplanationsConfig | None = None
) -> str:
    """
    Non-LLM summary from structured fields (always available).

    Respects :class:`~fightsafe_ai.llm.config.ExplanationsConfig` when provided
    (or when :file:`configs/llm.yaml` can be read).
    """
    ex = _resolve_explanations(explanations)
    tr = _time_range_phrase(event_data)
    mx = _max_score_phrase(event_data)
    lvl = _level_phrase(event_data)
    eid = event_data.get("event_id", "—")
    base = f"Event #{eid}: the automated pipeline reported **max risk ≈ {mx}** (level **{lvl}**), over **{tr}**"
    if ex.include_triggered_rules:
        rules = _format_rules_phrase(event_data)
        base += f". Heuristic **triggered rules** in context: {rules}"
    base += "."
    if ex.include_safety_disclaimer:
        base += (
            " This text is **decision-support** from pose and rule outputs. "
            "This is not a medical diagnosis. It is not a substitute for professional judgment."
        )
    if event_level_reaches_threshold(lvl, ex.recommend_human_review_threshold):
        base += " **Human review** of the source video is **recommended** before any operational decision."
    else:
        base += " Visual spot-checks remain good practice for any highlighted segment."
    return base


def explain_risk_event(
    event_data: dict[str, Any],
    llm_client: BaseLLMClient,
    *,
    explanations: ExplanationsConfig | None = None,
    use_llm: bool | None = None,
    run_state: LLMExplanationRunState | None = None,
) -> str:
    """
    Produce a short natural-language explanation using ``llm_client`` when possible.

    If Ollama is disabled in :file:`configs/llm.yaml` (``ollama.enabled: false``), or
    ``use_llm=False``, returns :func:`fallback_risk_explanation` without calling the
    model. If generation fails, falls back the same way.

    When ``run_state`` is provided and a **model load/resource** error occurs, subsequent
    calls skip the LLM without extra logs (deterministic template only).
    """
    ex = _resolve_explanations(explanations)
    if not _resolve_llm_enabled(use_llm):
        return fallback_risk_explanation(event_data, ex)
    if run_state is not None and run_state.skip_llm_calls():
        return fallback_risk_explanation(event_data, ex)
    prompt = explain_risk_event_prompt(event_data, ex)
    try:
        return llm_client.generate(prompt)
    except LLMError as e:
        if run_state is not None and is_ollama_model_load_or_resource_error(e):
            run_state.record_resource_or_load_failure()
            return fallback_risk_explanation(event_data, ex)
        if run_state is not None:
            run_state.record_other_llm_failure()
        logger.warning("LLM explain_risk_event failed, using template fallback: %s", e)
        return fallback_risk_explanation(event_data, ex)
    except OSError as e:  # pragma: no cover
        if run_state is not None:
            run_state.record_other_llm_failure()
        logger.warning("LLM explain_risk_event I/O error, using template fallback: %s", e)
        return fallback_risk_explanation(event_data, ex)
    except Exception as e:
        if run_state is not None:
            run_state.record_other_llm_failure()
        logger.exception("Unexpected error from LLM client, using template fallback: %s", e)
        return fallback_risk_explanation(event_data, ex)


def fallback_multi_signal_explanation(context: dict[str, Any]) -> str:
    """
    Deterministic paragraph when the LLM is off or failed — mirrors the multi-signal **prompt** contract.

    Expects the same key shape as :func:`~fightsafe_ai.llm.prompts.build_multi_signal_explanation_prompt`.
    """
    rd: dict[str, Any] = context.get("risk_decision") or {}
    al: dict[str, Any] = context.get("referee_alert") or {}
    sigs: list[Any] = list(context.get("detected_signals") or [])
    conf: dict[str, Any] = dict(context.get("signal_confidences") or {})
    tr: dict[str, Any] = context.get("time_range") or {}

    chunks: list[str] = [
        "Heuristic multi-signal summary (decision-support only; not a medical finding, not an autonomous refereeing decision).",
    ]
    if rd:
        trig = ", ".join(str(x) for x in (rd.get("triggered_signals") or ()))
        chunks.append(
            f"Fused model suggests risk band **{rd.get('risk_level', '?')}** (score ≈ {rd.get('risk_score', 'n/a')}; "
            f"signals: {trig or '—'}). This is a heuristic label, not a certainty."
        )
    if al:
        amsg = str(al.get("short_message", ""))[:200]
        chunks.append(f"HCI layer suggests **{al.get('alert_level', '?')}** attention: {amsg}")
    if sigs:
        chunks.append("Reported label list includes: " + ", ".join(str(s) for s in sigs) + ".")
    if conf:
        cstr = ", ".join(f"{k}≈{float(v):.2f}" for k, v in list(conf.items())[:20])
        chunks.append(f"Best-effort confidences: {cstr}.")
    t0, t1 = tr.get("start"), tr.get("end")
    if t0 is not None and t1 is not None:
        try:
            chunks.append(
                f"Time span for review: {float(t0):.3f} s – {float(t1):.3f} s (source video time if aligned)."
            )
        except (TypeError, ValueError):
            chunks.append("Time span: (unparsed).")
    chunks.append(
        "What to review: confirm visually on the video, apply competition rules, and use human judgment — the system only highlights possibilities."
    )
    chunks.append("This is not a medical diagnosis.")
    return " ".join(chunks)


__all__ = [
    "explain_risk_event",
    "fallback_multi_signal_explanation",
    "fallback_risk_explanation",
]
