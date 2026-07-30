"""
Single-document Markdown **safety review** reports for one combat-sports clip.

Suitable for research documentation: non-clinical framing, explicit limitations, and
human-in-the-loop emphasis. This module does not call pose or risk models; it only
formats structured ``clip_data`` and optionally invokes an :class:`~fightsafe_ai.llm.base.BaseLLMClient`
to draft narrative when pre-computed explanations are absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from fightsafe_ai._metadata import (
    CLIP_REPORT_INTRO,
    REPORT_END_ATTRIBUTION,
    SAFETY_REPORT_MD_HEADER,
)
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.report_enricher import enrich_clip_narrative


logger = logging.getLogger(__name__)

ClipData = dict[str, Any]

REVIEW_SCORE_SUGGEST_HUMAN = 0.5


def _as_path_display(p: Any) -> str:
    if p is None:
        return "—"
    return str(Path(p).expanduser().resolve()) if p else "—"


def _format_events(detected: Any) -> str:
    if detected is None:
        return "*No event list was supplied.*"
    if isinstance(detected, int):
        return f"**Count (aggregated):** {detected} segment(s) flagged at elevated levels (per pipeline configuration)."
    if isinstance(detected, (list, tuple)):
        if not detected:
            return "*No events reported for this clip.*"
        lines: list[str] = []
        for i, ev in enumerate(detected):
            if isinstance(ev, dict):
                evs = json.dumps(ev, sort_keys=True, default=str, indent=2)
                lines.append(f"**Event {i}**\n\n```json\n{evs}\n```\n")
            else:
                lines.append(f"- {ev!s}")
        return "\n".join(lines)
    return f"```\n{detected!r}\n```"


def _pick_highest_risk_moment(clip: ClipData) -> str:
    h = clip.get("highest_risk_moment")
    if isinstance(h, dict) and h:
        parts: list[str] = []
        if h.get("time_s") is not None:
            parts.append(f"- **Time (s):** {h['time_s']}")
        if h.get("time_range_s") is not None:
            parts.append(f"- **Time range (s):** {h['time_range_s']}")
        if h.get("frame_id") is not None:
            parts.append(f"- **Frame id:** {h['frame_id']}")
        if h.get("event_id") is not None:
            parts.append(f"- **Event id:** {h['event_id']}")
        if h.get("risk_score") is not None:
            parts.append(f"- **Local risk score:** {h['risk_score']}")
        if h.get("notes"):
            parts.append(f"- **Notes:** {h['notes']}")
        if parts:
            return "\n".join(parts)
    # Derive a minimal note from events + global max
    evs = clip.get("detected_events")
    mx = clip.get("max_risk_score")
    if isinstance(evs, list) and evs:
        best: dict[str, Any] | None = None
        best_s = -1.0
        for e in evs:
            if not isinstance(e, dict):
                continue
            v_raw = e.get("max_risk_score", e.get("risk_score", -1))
            try:
                s = float(v_raw) if v_raw is not None else -1.0
            except (TypeError, ValueError):
                s = -1.0
            if s >= best_s:
                best_s, best = s, e
        if best is not None and best_s >= 0:
            t0, t1 = best.get("start_time"), best.get("end_time")
            tr = f"{t0} s–{t1} s" if t0 is not None and t1 is not None else "—"
            return (
                f"- **Inferred from events:** segment with maximum reported score in this list.\n"
                f"- **Approximate time range:** {tr}\n"
                f"- **Segment max risk score:** {best_s:.4f} (clip-wide maximum reported below: {mx!s})."
            )
    if mx is not None:
        try:
            fmx = float(mx)
            if fmx == fmx:
                return (
                    f"- **Clip-level maximum risk score (heuristic, 0–1):** {fmx:.4f}\n"
                    "- *Spatial or temporal localization was not provided; see event table above if available.*"
                )
        except (TypeError, ValueError):
            return f"- **Reported maximum risk:** {mx!s}"
    return "*Highest-risk localization was not provided; retain raw frame-level exports for post hoc inspection.*"


def _format_rules_summary(rules: Any) -> str:
    if rules is None:
        return "*No rule summary was supplied.*"
    if isinstance(rules, (list, tuple)):
        if not rules:
            return "*None (empty list).*"
        return "\n".join(f"- `{r!s}`" for r in rules)
    if isinstance(rules, dict):
        if not rules:
            return "*None (empty object).*"
        return "\n".join(f"- **{k}:** {v!s}" for k, v in rules.items())
    s = str(rules).strip()
    return s if s else "—"


def _resolve_ai_explanation(
    clip: ClipData,
    llm_client: BaseLLMClient | None,
) -> str:
    raw = clip.get("ollama_explanations") or clip.get("ai_explanations") or clip.get("explanations")
    if raw is not None and str(raw).strip() != "":
        if isinstance(raw, (list, tuple)):
            return "\n\n".join(str(x).strip() for x in raw if str(x).strip())
        return str(raw).strip()
    try:
        return enrich_clip_narrative(clip, llm_client)
    except Exception as e:  # pragma: no cover
        logger.warning("enrich_clip_narrative failed: %s", e)
        return "*Narrative section could not be built. Rely on tabular summaries and source CSVs.*"


def _human_recommendation(clip: ClipData) -> str:
    mx = clip.get("max_risk_score")
    high_level = False
    evs = clip.get("detected_events", [])
    if isinstance(evs, list):
        for e in evs:
            if isinstance(e, dict):
                lv = str(e.get("event_level", e.get("risk_level", ""))).upper()
                if lv in ("HIGH", "CRITICAL"):
                    high_level = True
                    break
    try:
        s = float(mx) if mx is not None else 0.0
        if s != s:  # NaN
            s = 0.0
    except (TypeError, ValueError):
        s = 0.0
    if high_level or s >= REVIEW_SCORE_SUGGEST_HUMAN:
        return (
            "**Recommended:** qualified reviewers should inspect the **source video** and, where applicable, "
            "independent performance records. Elevated algorithmic scores **do not** equal injury or medical harm; "
            "they only prioritize segments for *human* judgment and possible annotation or follow-up study design."
        )
    return (
        "**Suggested:** a brief **spot check** of the listed segments against the source recording remains good "
        "scientific practice. Low algorithmic concern does not certify absence of notable interactions in frame "
        "margins, occlusions, or outside the model’s design scope."
    )


def _safety_disclaimer() -> str:
    return (
        "This report is **not** a medical document, **not** a clinical decision support system, and **not** a "
        "substitute for qualified ringside or venue safety protocols. Outputs combine **heuristic** computer-vision "
        "features and optional language models; they are intended for **research, education, and triage of manual "
        "review** only. Always align conclusions with your institution’s ethical review, data use agreements, and "
        "sporting regulations."
    )


def _build_markdown(clip: ClipData, body_ai: str) -> str:
    clip_id = clip.get("clip_id", "—")
    vid = _as_path_display(clip.get("video_path"))
    title = f"Combat sports safety review — **clip `{clip_id}`**"
    summary_lines = [
        f"- **Clip identifier:** `{clip_id}`",
        f"- **Video path (or URI reference):** `{vid}`",
    ]
    for key in ("source", "sport", "session_id", "notes", "sampling_fps"):
        if key in clip and clip[key] is not None and str(clip[key]).strip() != "":
            summary_lines.append(f"- **{key.replace('_', ' ').title()}:** {clip[key]!s}")
    summary = "\n".join(summary_lines)
    out = f"""{SAFETY_REPORT_MD_HEADER}

# {title}

*{CLIP_REPORT_INTRO}*

---

## 1. Clip summary

{summary}

---

## 2. Detected risk events

{_format_events(clip.get("detected_events"))}

---

## 3. Highest risk moment

{_pick_highest_risk_moment(clip)}

---

## 4. Triggered rules

{_format_rules_summary(clip.get("triggered_rules_summary"))}

---

## 5. AI-generated explanation

{body_ai}

---

## 6. Human review recommendation

{_human_recommendation(clip)}

---

## 7. Safety disclaimer

{_safety_disclaimer()}

---
*{REPORT_END_ATTRIBUTION}*
"""
    return out


def generate_clip_report(
    clip_data: dict[str, Any],
    output_path: Path,
    llm_client: BaseLLMClient | None = None,
) -> Path:
    """
    Write a **Markdown** safety review for one clip.

    **Expected keys in** ``clip_data`` **(all optional except identifiers as needed for your study):**

    - ``clip_id`` — string or number label for the segment.
    - ``video_path`` — filesystem path or URI string (displayed as-is, expanded when path-like).
    - ``detected_events`` — list of event ``dict``s (e.g. from ``frame_risk_to_events_list``) **or** an integer count.
    - ``max_risk_score`` — global maximum heuristic score in ``[0, 1]`` (float).
    - ``triggered_rules_summary`` — human-readable or machine list of active rule names (str, list, or dict of counts).
    - ``highest_risk_moment`` *(optional)* — ``dict`` with e.g. ``time_s`` / ``time_range_s``, ``frame_id``,
      ``event_id``, ``risk_score``, ``notes``.
    - ``ollama_explanations`` *(optional)* — pre-baked narrative (``str`` or list of str). Also accepts
      ``ai_explanations`` or ``explanations`` for the same content.

    If no pre-baked text is present, :func:`~fightsafe_ai.llm.report_enricher.enrich_clip_narrative`
    fills the section (LLM when Ollama is enabled, else rule-based text).

    Parameters
    ----------
    clip_data
        Session metadata and analytics fields (see above).
    output_path
        Destination ``.md`` file; parent directories are created if missing.
    llm_client
        Optional :class:`~fightsafe_ai.llm.base.BaseLLMClient` (e.g. Ollama) to fill the AI section when
        no ``ollama_explanations`` are present.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    clip: ClipData = cast("ClipData", dict(clip_data))
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    body_ai = _resolve_ai_explanation(clip, llm_client)
    md = _build_markdown(clip, body_ai)
    out.write_text(md, encoding="utf-8")
    return out


__all__ = ["REVIEW_SCORE_SUGGEST_HUMAN", "generate_clip_report"]
