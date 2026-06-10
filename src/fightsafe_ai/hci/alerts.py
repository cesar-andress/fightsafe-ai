"""
Referee-facing alert model and frame-to-alert builder (decision-support only).

The system **never** issues a match stoppage or medical command. Wording is review-oriented;
see :mod:`fightsafe_ai.hci.referee_messages` for copy constraints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fightsafe_ai.risk.rules import COMBAT_MVP_INDICATOR_LABELS

from . import referee_messages as _rm


class RefereeAlertLevel(StrEnum):
    """
    How strongly the system suggests the referee *consider* attention.

    **Not** fight outcomes, automatic stoppages, or medical verdicts. ``STOP_RECOMMENDED`` means
    *immediate human review recommended* — not an instruction to end the contest.
    """

    INFO = "INFO"
    WATCH = "WATCH"
    WARNING = "WARNING"
    STOP_RECOMMENDED = "STOP_RECOMMENDED"


# Map interpretable risk bands to referee alert levels (one-to-one).
RISK_LEVEL_TO_ALERT: dict[str, RefereeAlertLevel] = {
    "LOW": RefereeAlertLevel.INFO,
    "MEDIUM": RefereeAlertLevel.WATCH,
    "HIGH": RefereeAlertLevel.WARNING,
    "CRITICAL": RefereeAlertLevel.STOP_RECOMMENDED,
}


@dataclass(frozen=True, slots=True)
class RefereeAlert:
    """
    One human-in-the-loop notification for a single frame or decision instant.

    **This is a decision-support system, not an automated referee.**
    """

    timestamp: float
    fighter_id: str
    alert_level: RefereeAlertLevel
    short_message: str
    reason: str
    triggered_signals: tuple[str, ...]
    recommended_human_action: str


# Backward-compatible name used in earlier releases.
Alert = RefereeAlert


def _parse_triggered(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return [str(x) for x in value]
    return [str(value)]


def _rules_to_reason(trigger_keys: list[str]) -> str:
    if not trigger_keys:
        return "No specific rule triggers above the reporting threshold; routine tracking."
    parts: list[str] = []
    for key in trigger_keys:
        label = COMBAT_MVP_INDICATOR_LABELS.get(key, key)
        parts.append(f"{key} — {label}")
    return " | ".join(parts)


def generate_referee_alert(frame_data: Mapping[str, Any]) -> RefereeAlert:
    """
    Build one :class:`RefereeAlert` from a per-frame record (e.g. row dict or ``Series``).

    Expected keys (best-effort):

    - ``risk_level`` — ``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL``
    - ``timestamp`` — float seconds
    - ``fighter_id`` — str (optional; default ``"0"``)
    - ``triggered_rules`` or ``triggered_signals`` — list of string keys
    - ``risk_score`` — optional float in ``[0, 1]`` for copy context
    - ``explanation_facts`` (optional) — if a sequence, first string can supplement *reason*

    **This is a decision-support system, not an automated referee.**
    """
    raw = str(frame_data.get("risk_level", "LOW")).upper()
    if raw not in RISK_LEVEL_TO_ALERT:
        raw = "LOW"
    alert_level = RISK_LEVEL_TO_ALERT[raw]

    ts_raw = frame_data.get("timestamp", float("nan"))
    try:
        timestamp = float(ts_raw)
    except (TypeError, ValueError):
        timestamp = float("nan")

    fighter_id = str(frame_data.get("fighter_id", "0") or "0")
    rules = _parse_triggered(
        frame_data.get("triggered_rules") or frame_data.get("triggered_signals")
    )
    ex = frame_data.get("explanation_facts")
    reason_override = str(frame_data.get("reason", "") or "").strip()
    if reason_override:
        reason = reason_override[:2000]
    elif isinstance(ex, (list, tuple)) and ex and str(ex[0]).strip():
        reason = str(ex[0])[:2000]
    else:
        reason = _rules_to_reason(rules)

    rs = frame_data.get("risk_score")
    risk_score: float | None
    try:
        risk_score = float(rs) if rs is not None and str(rs) != "" else None
    except (TypeError, ValueError):
        risk_score = None
    if risk_score is not None and not math.isfinite(risk_score):
        risk_score = None

    av = alert_level.value
    short = _rm.short_message_for_risk(av, raw, risk_score)
    action = _rm.recommended_action_for_risk(av)

    sigs = tuple(str(s) for s in rules) if rules else ("none_reported",)

    return RefereeAlert(
        timestamp=timestamp,
        fighter_id=fighter_id,
        alert_level=alert_level,
        short_message=short,
        reason=reason,
        triggered_signals=sigs,
        recommended_human_action=action,
    )
