"""Copy / tone tests for :mod:`fightsafe_ai.hci.referee_messages` (recommendation-only)."""

from __future__ import annotations

import pytest

from fightsafe_ai.hci import (
    message_for_level,
    recommended_action_for_risk,
    short_message_for_risk,
    template_payload,
    validate_no_forbidden_phrase,
)
from fightsafe_ai.hci.alerts import RefereeAlertLevel


pytestmark = pytest.mark.unit

_LEVELS = (
    RefereeAlertLevel.INFO,
    RefereeAlertLevel.WATCH,
    RefereeAlertLevel.WARNING,
    RefereeAlertLevel.STOP_RECOMMENDED,
)


@pytest.mark.parametrize("level", _LEVELS)
def test_forbidden_phrase_absent_in_public_strings(level: RefereeAlertLevel) -> None:
    p = message_for_level(level)
    a = recommended_action_for_risk(level)
    t = short_message_for_risk(
        level.value, "HIGH" if level is not RefereeAlertLevel.INFO else "LOW", 0.5
    )
    o = template_payload(level.value)
    for blob in (p, a, t, o.get("panel_line", ""), o.get("recommended_human_action", "")):
        assert validate_no_forbidden_phrase(blob)
        assert "stop the fight" not in str(blob).lower()


def test_no_autonomous_stop_imperative_in_stop_band() -> None:
    t = message_for_level(RefereeAlertLevel.STOP_RECOMMENDED)
    assert "recommend" in t.lower() or "review" in t.lower() or "attention" in t.lower()
    # Must not read as a direct order to end the bout
    assert "not a machine" in t.lower() or "not" in t.lower() or "review" in t.lower()


def test_short_message_length_reasonable() -> None:
    s = short_message_for_risk("WARNING", "HIGH", 0.88)
    assert len(s) <= 120 or s.endswith("...")


def test_recommended_action_is_suggestion() -> None:
    for lev in _LEVELS:
        r = recommended_action_for_risk(lev)
        assert len(r) > 10
        # Never imply software stopped the match
        assert "stop the fight" not in r.lower()
