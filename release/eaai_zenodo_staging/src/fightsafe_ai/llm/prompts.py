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

Reusable prompt strings for Ollama (or any :class:`~fightsafe_ai.llm.base.BaseLLMClient`).

**Inputs** are plain dicts so they can be built from :func:`~fightsafe_ai.risk.events.frame_risk_to_events_list`
rows plus optional fields such as ``triggered_rules``.
"""

from __future__ import annotations

import json
from typing import Any

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.llm.config import ExplanationsConfig, load_llm_file_config


__all__ = [
    "build_clip_summary_prompt",
    "build_multi_signal_explanation_prompt",
    "build_risk_explanation_prompt",
    "explain_risk_event_prompt",
    "suggest_annotation_prompt",
    "summarize_clip_prompt",
]


def _explanations_from_file_or_default() -> ExplanationsConfig:
    try:
        return load_llm_file_config().explanations
    except (ConfigurationError, OSError, ValueError):
        return ExplanationsConfig()


def build_risk_explanation_prompt(event_data: dict[str, Any]) -> str:
    """
    Build the full user prompt for explaining one structured risk **event** (not detection).

    Loads explanation style from :file:`configs/llm.yaml` when present. The model must stay
    non-clinical: outputs assist human review only.
    """
    return explain_risk_event_prompt(event_data, _explanations_from_file_or_default())


def build_clip_summary_prompt(run_data: dict[str, Any]) -> str:
    """
    Build the prompt for a **run- or clip-level** summary from structured metadata only.

    ``run_data`` may include keys such as ``clip_id``, ``source``, ``event_count``, ``events``,
    ``max_risk_score``, ``notes`` — the same shape accepted by :func:`summarize_clip_prompt`.
    """
    return summarize_clip_prompt(run_data)


def _format_rules(data: dict[str, Any]) -> str:
    rules = data.get("triggered_rules")
    if rules is None and "rules" in data:
        rules = data["rules"]
    if not rules:
        return "(no triggered rules list provided — rely on event-level summary below)"
    if isinstance(rules, (list, tuple)):
        return ", ".join(str(r) for r in rules)
    return str(rules)


def _format_time_range(data: dict[str, Any]) -> str:
    t0 = data.get("start_time")
    t1 = data.get("end_time")
    if t0 is not None and t1 is not None:
        return f"{t0} s to {t1} s (video time)"
    return "(timestamp range not provided)"


def _format_risk_level(data: dict[str, Any]) -> str:
    """Best-effort risk / event level from common keys in event dicts or nested frames."""
    for k in ("event_level", "risk_level", "max_risk_level", "level"):
        v = data.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().upper()
    return ""


# Internal risk bands (interpretable rules) → referee **suggestion** vocabulary (HCI layer; not a command).
_REFEREE_SUGGESTION_BY_LEVEL: str = (
    "Use the following **mapping** for **suggested** referee **attention** (recommendation only, "
    "not a stoppage or score): **LOW** → **INFO**-style: routine monitoring, no special action. "
    "**MEDIUM** → **WATCH**: recommend sustained attention to the action. **HIGH** → **WARNING**: "
    "recommend that the official **prepare** to assess a possible **intervention** if their judgment agrees. "
    "**CRITICAL** maps to the HCI label **STOP_RECOMMENDED** (or legacy **STOP** in older docs): recommend "
    "**immediate human review**; this is **not** an automatic end to the match or a substitute for the referee."
)

# Multi-signal / fusion + HCI (aligned with :mod:`fightsafe_ai.hci` and :mod:`fightsafe_ai.risk.fusion`).
_REFEREE_MULTISIGNAL_VOCAB: str = (
    "Risk bands (LOW < MEDIUM < HIGH < CRITICAL) are fused from **heuristic** signals (pose, action, anomaly, "
    "inactivity, surrender *proxies*). The companion **referee alert** level uses **INFO / WATCH / WARNING / "
    "STOP_RECOMMENDED** — the latter means *immediate human attention to review the scene*, not a software "
    "order to stop the fight."
)

_MEDICAL_DISCLAIMER: str = "This is not a medical diagnosis."


def _event_json_for_prompt(event_data: dict[str, Any], ex: ExplanationsConfig) -> str:
    if ex.include_triggered_rules:
        payload: dict[str, Any] = dict(event_data)
    else:
        payload = {k: v for k, v in event_data.items() if k not in ("triggered_rules", "rules")}
    return json.dumps(payload, sort_keys=True, default=str, indent=2)


def explain_risk_event_prompt(
    event_data: dict[str, Any], explanations: ExplanationsConfig | None = None
) -> str:
    """
    Build a user prompt for explaining one merged risk event.

    ``event_data`` may include keys such as: ``event_id``, ``start_frame``,
    ``end_frame``, ``start_time``, ``end_time``, ``max_risk_score``,
    ``event_level`` / ``risk_level`` (LOW / MEDIUM / HIGH / CRITICAL), ``duration_seconds``,
    and ``triggered_rules`` (``list[str]``) if merged from per-frame data. The
    **prompt** asks the model to name the **risk** **level**, **justify** it with **rules**,
    **suggest** a **WATCH** / **WARNING** / **STOP**-style **recommendation**, and
    **always** include: "This is not a medical diagnosis."
    """
    ex = explanations or ExplanationsConfig()
    lvl = _format_risk_level(event_data)
    lines = [
        "You are a combat-sports **analytics assistant**. Your job is to explain",
        "structured, algorithmic **multi-level** risk **hints** to a human reviewer (e.g. a **referee** or **safety** analyst).",
        "You do **not** diagnose injury or medical conditions.",
        "",
        "Structured event (JSON) — use **event_level** / **risk_level** and **triggered_rules** to justify your text:",
        _event_json_for_prompt(event_data, ex),
        "",
        "Internal risk scale (ascending): LOW < MEDIUM < HIGH < CRITICAL.",
        _REFEREE_SUGGESTION_BY_LEVEL,
        "",
        "**Instructions** (follow in order):",
        "- Open with an explicit line of the form: **Risk level: [one of LOW, MEDIUM, HIGH, CRITICAL].**",
        (
            f"  Prefer the level from the JSON when present; it is: "
            f"{repr(lvl) if lvl else '(infer from event_level / risk_level / max_risk_score and context)'}."
        ),
        (
            "- In the next 2–3 sentences, **explain** **why** that level is plausible: tie your wording to the **triggered** "
            "**heuristic** **rule** **keys** and the **time** **range** when they appear in the JSON. "
            f"Time range for this event: **{_format_time_range(event_data)}**."
        ),
    ]
    if ex.include_triggered_rules:
        lines.append(
            f"  (Rule keys in data: {_format_rules(event_data)}. Paraphrase them in plain language.)"
        )
    lines += [
        "- Suggest one **referee-**oriented **action** using the **WATCH** / **WARNING** / **STOP** (and **INFO**-style for LOW) **vocabulary** from the **mapping** above, phrased as a **recommendation**, not an order. "
        'Example of tone for **HIGH** (illustration only, adapt to the actual case): "Risk level: HIGH. The athlete shows significant '
        "instability and reduced defensive posture. The referee should **prepare** for a possible **intervention** if their "
        'assessment agrees."',
        f'- End the paragraph with this **exact** sentence, on its own line: "{_MEDICAL_DISCLAIMER}"',
    ]
    if ex.include_safety_disclaimer:
        lines.append(
            "- State clearly (once) that the output is **decision-** support from **automated** heuristics, not an official call."
        )
    thr = ex.recommend_human_review_threshold.strip().upper()
    if thr in ("NONE", "OFF", ""):
        pass
    else:
        lines.append(
            f"- If the event is at or above **{thr}** on the **LOW** < **MEDIUM** < **HIGH** < **CRITICAL** **scale**, "
            f"recommend that a **human** **view** the **source** **video** before **drawing** **conclusions**."
        )
    lines += [
        "",
        "Write a **single** **concise** **paragraph** (4–7 sentences); you may use **one** line break immediately before the **required** disclaimer sentence.",
        "Respond with the explanation only, no JSON.",
    ]
    return "\n".join(lines)


def build_multi_signal_explanation_prompt(context: dict[str, Any]) -> str:
    """
    User prompt for narrating a **fused** risk result plus **HCI** alert and optional per-signal confidences.

    ``context`` should be built by :func:`~fightsafe_ai.llm.explainer.multi_signal_context_to_dict` or match its
    keys:

    - ``risk_decision`` — optional dict (``risk_level``, ``risk_score``, ``triggered_signals``, ``explanation_facts``, …)
    - ``referee_alert`` — optional dict (``alert_level``, ``short_message``, ``reason``, …)
    - ``detected_signals`` — optional list of short string labels (e.g. action/anomaly keys)
    - ``signal_confidences`` — optional ``dict[str, float]`` in ``[0,1]`` (best-effort)
    - ``time_range`` — optional ``{"start": float, "end": float}`` in seconds (video or segment time)

    The model must **not** give medical advice, claim certainty, or describe an autonomous stoppage.
    """
    ex = _explanations_from_file_or_default()
    raw = json.dumps(context, sort_keys=True, default=str, indent=2)
    tr = context.get("time_range") or {}
    t0, t1 = tr.get("start"), tr.get("end")
    tr_s = f"{t0} s to {t1} s" if t0 is not None and t1 is not None else "(timestamp range not set)"

    lines = [
        "You are a **combat-sports analytics assistant**. Explain **algorithmic, multi-signal** outputs to a **human** "
        "**referee** or **safety reviewer** (human-in-the-loop).",
        "",
        "You do **not** diagnose injury or medical conditions. You do **not** end a match or replace the referee. "
        "Avoid words that imply **certainty** or a **definitive** call (e.g. do not say the system *knows* an injury "
        "occurred, or that the fight *must* be stopped). Use hedged, review-oriented language.",
        "",
        "**Structured context (JSON)** — fusion risk, HCI alert, confidences, and time span:",
        raw,
        "",
        "Internal risk scale: LOW < MEDIUM < HIGH < CRITICAL.",
        _REFEREE_SUGGESTION_BY_LEVEL,
        _REFEREE_MULTISIGNAL_VOCAB,
        "",
        f"**Relevant time span** for this explanation (if any): {tr_s}.",
        "",
        "**Write a single concise section (4–6 short sentences, one short paragraph) that covers, in order:**",
        "1) **What** appears to have been *flagged* or *suggested* by the pipeline (signals, not ground truth).",
        "2) **Why** the **risk level** in the data is a reasonable *heuristic* label (tie to **triggered_signals** or **explanation_facts** if present).",
        "3) **Which** inputs likely **contributed** (name **signal_confidences** or **detected_signals** when helpful; if missing, say so briefly).",
        "4) **What** the **human referee** should **review** next (video, angle, rules) — as a **recommendation** only.",
        "",
    ]
    if ex.include_safety_disclaimer:
        lines.append(
            "Also state once that this is **decision-support** from heuristics and **not** an official ruling.",
        )
    lines += [
        "",
        f'End with this **exact** final sentence on its own line: "{_MEDICAL_DISCLAIMER}"',
        "Respond with the explanation text only, no JSON.",
    ]
    return "\n".join(lines)


def summarize_clip_prompt(clip_data: dict[str, Any]) -> str:
    """
    Summarize a clip (multiple events, metadata).

    Suggested keys: ``clip_id``, ``source``, ``time_range`` or ``start_time``/``end_time``,
    ``event_count``, ``events`` (list of small dicts), ``notes``.
    """
    raw = json.dumps(clip_data, sort_keys=True, default=str, indent=2)
    return f"""You are summarizing **algorithmic** combat-sports risk cues for researchers.

Clip metadata (JSON):
{raw}

Write 3–5 short bullet points. **Reference** **multi-level** **risk** where **events** include **event_level** / **risk_level** (LOW < MEDIUM < HIGH < CRITICAL). For each event or the worst case, name the **suggested** **referee** **attention** level (**WATCH** / **WARNING** / **STOP** per the project’s HCI **mapping** from internal bands, **recommendation**-only). Emphasize that the summary is from **pose** / **heuristic** **rules**, **not** **medical** **fact**. Suggest follow-up **human** **video** **review** when any event is **HIGH** or **CRITICAL** (or per metadata).

**Always** end with this **exact** sentence: "{_MEDICAL_DISCLAIMER}"
"""


def suggest_annotation_prompt(event_data: dict[str, Any]) -> str:
    """
    Suggest what an annotator might focus on (labels, time spans, free text).

    Typical keys overlap with ``explain_risk_event_prompt`` plus ``video_path`` (optional, redact PII)."""
    raw = json.dumps(event_data, sort_keys=True, default=str, indent=2)
    return f"""A risk event was flagged by **multi-level** **heuristic** rules. Suggest **annotation** tasks
for a human (what to look at, time spans, optional tags), without claiming **medical** **meaning**. If **event_level** or **risk_level** is in the JSON, **tie** **labels** to that **band** and to **WATCH** / **WARNING** / **STOP**-style **review** **(recommendation**-**only**).

Event (JSON):
{raw}

Output a short list (2–4 numbered items). Conclude with this **exact** sentence: "{_MEDICAL_DISCLAIMER}"
"""
