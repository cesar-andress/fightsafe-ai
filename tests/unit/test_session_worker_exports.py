"""Session artifact persistence (no video decode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.api.session_worker import persist_session_artifacts
from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel


pytestmark = pytest.mark.unit


def test_persist_session_artifacts_writes_json_csv_metadata(tmp_path: Path) -> None:
    bus = EventBus()
    bus.add_event(
        SafetyEvent(
            event_type="unit.persist",
            category=EventCategory.FALL,
            start_time=0.5,
            end_time=1.0,
            level=SafetyLevel.INFO,
            score=0.2,
            title="t",
            description="d",
            explanation="e",
            source="s",
        )
    )
    meta = {"video_path": str(tmp_path / "x.mp4"), "processed_frames": 42}
    se = tmp_path / "session_events.json"
    ej = tmp_path / "events.json"
    ec = tmp_path / "events.csv"
    sm = tmp_path / "session_metadata.json"
    persist_session_artifacts(
        bus,
        session_events_path=se,
        export_json_path=ej,
        export_csv_path=ec,
        session_metadata_path=sm,
        session_metadata=meta,
    )
    assert se.is_file() and ej.is_file() and ec.is_file() and sm.is_file()
    md = json.loads(sm.read_text(encoding="utf-8"))
    assert md["processed_frames"] == 42
    ev_doc = json.loads(ej.read_text(encoding="utf-8"))
    assert ev_doc["schema_version"] == 4
    assert len(ev_doc["events"]) == 1
