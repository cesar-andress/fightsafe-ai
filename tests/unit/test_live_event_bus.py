"""Unit tests for live EventBus and SafetyEvent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel


pytestmark = pytest.mark.unit


def _ev(
    ts: float,
    *,
    event_type: str = "test.kind",
    title: str = "t",
    score: float = 0.1,
    level: SafetyLevel = SafetyLevel.INFO,
) -> SafetyEvent:
    return SafetyEvent(
        event_type=event_type,
        category=EventCategory.UNKNOWN,
        start_time=ts,
        end_time=ts,
        level=level,
        score=score,
        title=title,
        description="d",
        explanation="because unit test",
        source="s",
    )


def test_add_event_and_recent_order() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.01)
    bus.add_event(_ev(1.0, title="a", event_type="a"))
    bus.add_event(_ev(2.0, title="b", event_type="b"))
    recent = bus.get_recent_events(limit=10)
    assert [e.title for e in recent] == ["a", "b"]
    assert bus.get_recent_events(limit=1)[0].title == "b"


def test_merge_same_event_type_extends_window() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=1.0)
    bus.add_event(_ev(0.0, event_type="risk.x", title="r"))
    bus.add_event(_ev(0.2, event_type="risk.x", title="r"))
    assert len(bus.all_events()) == 1
    e = bus.all_events()[0]
    assert e.start_time == 0.0
    assert e.end_time == 0.2
    assert e.last_seen_time == pytest.approx(0.2)
    assert e.duration == pytest.approx(0.2)


def test_cooldown_same_event_type_blocks_new_episode() -> None:
    bus = EventBus(cooldown_seconds=10.0, merge_gap_seconds=0.1)
    assert bus.add_event(_ev(0.0, event_type="same")) is True
    assert bus.add_event(_ev(5.0, event_type="same")) is False
    assert bus.add_event(_ev(11.0, event_type="same")) is True


def test_force_bypasses_cooldown() -> None:
    bus = EventBus(cooldown_seconds=100.0, merge_gap_seconds=0.05)
    bus.add_event(_ev(0.0, event_type="x"))
    assert bus.add_event(_ev(1.0, event_type="x"), force=True) is True
    assert len(bus.all_events()) == 2


def test_visible_events_respects_ttl() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.05, visual_expire_seconds=2.0)
    bus.add_event(_ev(10.0, event_type="a"))
    vis = bus.get_visible_events(now_seconds=11.0)
    assert len(vis) == 1
    vis2 = bus.get_visible_events(now_seconds=14.0)
    assert len(vis2) == 0


def test_export_json_csv_roundtrip(tmp_path: Path) -> None:
    bus = EventBus()
    bus.add_event(
        SafetyEvent(
            event_type="unit.probe",
            category=EventCategory.FALL,
            start_time=3.0,
            end_time=3.5,
            level=SafetyLevel.WARNING,
            score=0.4,
            title="w",
            description="desc",
            explanation="short why",
            source="src",
        )
    )
    jpath = tmp_path / "e.json"
    cpath = tmp_path / "e.csv"
    bus.export_json(jpath)
    bus.export_csv(cpath)
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["schema_version"] == 4
    assert len(data["events"]) == 1
    ev = data["events"][0]
    assert ev["event_id"]
    assert ev["is_finished"] is False
    assert ev["level"] == "WARNING"
    assert ev["category"] == "fall"
    assert ev["last_seen_time"] == pytest.approx(3.5)
    assert ev["duration"] == pytest.approx(0.5)
    assert ev["explanation"] == "short why"
    text = cpath.read_text(encoding="utf-8")
    assert "timestamp" in text and "category" in text and "duration" in text
    assert "start_time" in text and "end_time" in text and "event_type" in text
    assert "WARNING" in text


def test_get_recent_zero_limit() -> None:
    bus = EventBus()
    bus.add_event(_ev(0.0))
    assert bus.get_recent_events(limit=0) == []


def test_event_category_impact_token() -> None:
    assert EventCategory.IMPACT.value == "impact"
    assert EventCategory.STRIKE_IMPACT.value == "impact"
    # Aliased enum members (same value); mypy sees distinct literals for `==`.
    assert EventCategory.STRIKE_IMPACT == EventCategory.IMPACT  # type: ignore[comparison-overlap]


def test_visible_events_use_last_seen_ttl() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.05, visual_expire_seconds=2.0)
    bus.add_event(_ev(10.0, event_type="a"))
    e = bus.all_events()[0]
    e.last_seen_time = 12.0
    e.end_time = 12.0
    vis = bus.get_visible_events(now_seconds=13.0)
    assert len(vis) == 1
    vis2 = bus.get_visible_events(now_seconds=15.5)
    assert len(vis2) == 0


def test_get_active_vs_visible() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=1.0, visual_expire_seconds=10.0)
    bus.add_event(_ev(1.0))
    bus.tick(2.0)
    active = bus.get_active_events(limit=5)
    vis = bus.get_visible_events(now_seconds=2.0, limit=5)
    assert len(active) == 1
    assert active[0].event_type == vis[0].event_type
    assert not active[0].is_finished


def test_event_ids_and_merge_same_instance() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=1.0)
    bus.add_event(_ev(0.0, event_type="risk.x"))
    e0 = bus.all_events()[0]
    assert e0.event_id.startswith("evt-")
    bus.add_event(_ev(0.3, event_type="risk.x"))
    assert len(bus.all_events()) == 1
    assert bus.all_events()[0].event_id == e0.event_id
    assert len(bus.active_events) == 1


def test_tick_marks_finished_when_silent() -> None:
    bus = EventBus(
        cooldown_seconds=0.0,
        merge_gap_seconds=0.5,
        silence_close_seconds=1.0,
    )
    bus.add_event(_ev(0.0, event_type="z"))
    assert len(bus.get_active_events()) == 1
    bus.tick(2.0)
    assert len(bus.get_active_events()) == 0
    finished = bus.all_events()[0]
    assert finished.is_finished


def test_new_episode_after_gap_finishes_previous() -> None:
    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.3)
    bus.add_event(_ev(0.0, event_type="k"))
    first_id = bus.all_events()[0].event_id
    bus.add_event(_ev(5.0, event_type="k"))
    assert len(bus.all_events()) == 2
    prev = bus.all_events()[0]
    assert prev.event_id == first_id
    assert prev.is_finished
    assert not bus.all_events()[1].is_finished


def test_cooldown_override_per_event_type() -> None:
    bus = EventBus(
        cooldown_seconds=10.0,
        merge_gap_seconds=0.05,
        cooldown_by_event_type={"risk.x": 2.0},
    )
    assert bus.add_event(_ev(0.0, event_type="risk.x")) is True
    assert bus.add_event(_ev(1.0, event_type="risk.x")) is False
    assert bus.add_event(_ev(1.0, event_type="other_only")) is True
    assert bus.add_event(_ev(2.5, event_type="risk.x")) is True
