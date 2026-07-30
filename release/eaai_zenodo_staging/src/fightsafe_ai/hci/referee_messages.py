"""
Templated **English** strings for human-in-the-loop referee UIs (recommendation-only copy).

**Never** imply an autonomous stoppage: avoid phrases like "stop the fight" as a system command.
Prefer **"human review recommended"** and **"immediate referee attention recommended"** at the
highest band (see :data:`_STOP_BAND_COPY`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping


# Broadcast-safe prefix (short for overlay)
_PREFIX = "[FS] "

# CRITICAL / STOP_RECOMMENDED tier: not an order to end the match
_STOP_BAND_COPY: str = (
    "Immediate referee attention recommended for human review — not a machine stoppage."
)
_HIGH_REVIEW: str = (
    "Higher risk band: closer observation; human review recommended if the scene looks unclear."
)
_MED_REVIEW: str = (
    "Sustained attention; confirm visually before acting; review recommended as needed."
)
_LOW_COPY: str = "Routine monitoring; no special system recommendation."

# Static lines by API string value of :class:`RefereeAlertLevel`
_MESSAGE_BY_VALUE: dict[str, str] = {
    "INFO": _LOW_COPY,
    "WATCH": _MED_REVIEW,
    "WARNING": _HIGH_REVIEW,
    "STOP": _STOP_BAND_COPY,  # legacy alias if ever stored
    "STOP_RECOMMENDED": _STOP_BAND_COPY,
}

_RECOMMENDED_ACTION: dict[str, str] = {
    "INFO": "Continue; glance at the scoreboard/overlay only as convenient.",
    "WATCH": "Look again within a few seconds; be ready to step in *if your rules/eyes agree*.",
    "WARNING": "Position for a possible break; use normal procedures — system suggests review only.",
    "STOP": "Pause routine checks; give this exchange immediate human focus (recommendation only).",
    "STOP_RECOMMENDED": "Direct attention here now; confirm before any stoppage — human review recommended.",
}


def message_for_level(level: object) -> str:
    """
    Return a static English line for an alert level (``str`` or :class:`enum.StrEnum` member).
    """
    s = str(getattr(level, "value", level)).strip().upper()
    return _PREFIX + _MESSAGE_BY_VALUE.get(s, _MED_REVIEW)


def short_message_for_risk(
    alert_value: str,
    risk_band: str,
    risk_score: float | None,
) -> str:
    """
    Short line for real-time display (ticker / corner overlay), under ~120 characters when possible.

    Parameters
    ----------
    alert_value
        :class:`RefereeAlertLevel` value string, e.g. ``"WATCH"``.
    risk_band
        Raw risk band ``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL``.
    risk_score
        Optional scalar in ``[0,1]`` for light context.
    """
    s = str(alert_value).strip().upper()
    rb = str(risk_band).upper()
    score_t = (
        f" s={float(risk_score):.2f}"
        if risk_score is not None and math.isfinite(float(risk_score))
        else ""
    )
    base = _MESSAGE_BY_VALUE.get(s, _MED_REVIEW)
    # Keep the overlay short: prefix + one clause
    line = f"{_PREFIX}{base} (band {rb}{score_t})"
    if len(line) > 118:
        line = line[:115] + "..."
    return line


def recommended_action_for_risk(alert: object) -> str:
    """
    What the **human** might do next, phrased as a recommendation (never a command to stop the match).
    """
    s = str(getattr(alert, "value", alert)).strip().upper()
    return _RECOMMENDED_ACTION.get(s, _RECOMMENDED_ACTION["WATCH"])


def short_message_for_risk_enum(
    alert: object,
    risk_band: str,
    risk_score: float | None,
) -> str:
    """Like :func:`short_message_for_risk` but accepts a :class:`~fightsafe_ai.hci.alerts.RefereeAlertLevel`."""
    av = str(getattr(alert, "value", alert))
    return short_message_for_risk(av, risk_band, risk_score)


def validate_no_forbidden_phrase(text: str) -> bool:
    """
    Return True if *text* avoids the forbidden autonomous phrase ``stop the fight`` (MVP test hook).

    English-only; the product copy in this module is hand-checked to stay in recommendation tone.
    """
    return "stop the fight" not in str(text).lower()


def template_payload(level_value: str) -> Mapping[str, str]:
    """Small dict for JSON/debug dumps (no I/O)."""
    s = str(level_value).strip().upper()
    return {
        "panel_line": message_for_level(s),
        "recommended_human_action": recommended_action_for_risk(s),
    }
