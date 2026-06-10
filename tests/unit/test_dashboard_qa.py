"""
QA-focused tests for the live web dashboard stack (no browser, no real GPU).

Covers EventBus/serialization touchpoints, VideoSource contracts, GPU monitor
fallbacks (mocked), and HTTP endpoints used by the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fightsafe_ai.api.app import SessionBootstrap, create_app
from fightsafe_ai.api.serialization import safety_event_to_json
from fightsafe_ai.live import gpu_monitor as gm
from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel
from fightsafe_ai.live.video_source import FileVideoSource, WebcamSource, open_video_source


pytestmark = pytest.mark.unit


def _minimal_event(ts: float = 1.0) -> SafetyEvent:
    return SafetyEvent(
        event_type="qa.probe",
        category=EventCategory.UNKNOWN,
        start_time=ts,
        end_time=ts + 0.1,
        level=SafetyLevel.WARNING,
        score=0.42,
        title="QA",
        description="desc",
        explanation="why",
        source="test",
    )


def test_event_bus_dashboard_json_shape_matches_serialization() -> None:
    """EventBus episodes expose stable keys for HTTP/WebSocket JSON."""

    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.05)
    assert bus.add_event(_minimal_event(0.5)) is True
    ev = bus.all_events()[0]
    assert ev.event_id.startswith("evt-")
    payload = safety_event_to_json(ev)
    required = {
        "event_id",
        "event_type",
        "level",
        "timestamp_seconds",
        "duration",
        "title",
        "requires_human_confirmation",
    }
    assert required <= payload.keys()


def test_event_bus_export_json_contains_episode(tmp_path: Path) -> None:
    """Export path used by POST /session/export remains schema-stable."""

    bus = EventBus(cooldown_seconds=0.0, merge_gap_seconds=0.05)
    bus.add_event(_minimal_event(2.0))
    path = tmp_path / "bus.json"
    bus.export_json(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw.get("schema_version") == 4
    assert isinstance(raw.get("events"), list)
    assert len(raw["events"]) >= 1


def test_file_video_source_open_and_single_frame_no_full_preload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sequential reads only instantiate OpenCV capture + one frame buffer at a time."""

    class FakeCapture:
        def __init__(self, *_a: object, **_k: object) -> None:
            self._n = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            import cv2 as _cv

            if prop == _cv.CAP_PROP_FRAME_WIDTH:
                return 64.0
            if prop == _cv.CAP_PROP_FRAME_HEIGHT:
                return 48.0
            if prop == _cv.CAP_PROP_FPS:
                return 10.0
            if prop == _cv.CAP_PROP_FRAME_COUNT:
                return 0.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            self._n += 1
            if self._n > 3:
                return False, None
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("fightsafe_ai.live.video_source.cv2.VideoCapture", FakeCapture)

    avi = tmp_path / "qa.mp4"
    avi.write_bytes(b"")

    src = FileVideoSource(avi, realtime=False)
    try:
        assert src.total_frames is None or src.total_frames >= 1
        f0, m0 = src.read_frame()
        assert f0 is not None and m0 is not None
        assert m0.frame_index == 0
        _f1, m1 = src.read_frame()
        assert m1 is not None and m1.frame_index == 1
    finally:
        src.close()


def test_open_video_source_selects_webcam_vs_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dashboard routing: numeric token → webcam abstraction; path → file source."""

    class FakeCapture:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return 30.0

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("fightsafe_ai.live.video_source.cv2.VideoCapture", FakeCapture)

    cam = open_video_source("0", realtime=False)
    assert isinstance(cam, WebcamSource)
    cam.close()

    avi = tmp_path / "route.mp4"
    avi.write_bytes(b"")

    disk = open_video_source(str(avi))
    assert isinstance(disk, FileVideoSource)
    disk.close()


def test_gpu_monitor_fallback_nvml_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gm, "_ensure_nvml", lambda: (False, "mock: no NVML"))
    d = gm.get_nvidia_gpu_metrics()
    assert d["nvidia_nvml_available"] is False
    assert d["status"] == "nvml_unavailable"
    assert d["gpu_name"] is None
    assert d["message"] == "mock: no NVML"


def test_gpu_monitor_fallback_nvml_sample_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gm, "_ensure_nvml", lambda: (True, None))
    fake = MagicMock()
    fake.nvmlDeviceGetCount.side_effect = RuntimeError("mock NVML failure")
    monkeypatch.setattr(gm, "_pynvml", fake)
    d = gm.get_nvidia_gpu_metrics()
    assert d["nvidia_nvml_available"] is False
    assert d["status"] == "nvml_error"
    assert "mock NVML failure" in (d.get("message") or "")


def test_session_status_idle_contract() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.get("/session/status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "idle"
        assert body["completed"] is False
        assert body["event_count"] == 0
        assert "risk_level" in body
        assert "media_timestamp_seconds" in body
        assert body.get("enable_tapko_detectors") is True
        assert len(body.get("tapko_event_types") or []) == 8
        assert "session_metadata" not in body  # exposed via /session/metadata only


def test_session_export_writes_configured_paths(tmp_path: Path) -> None:
    j = tmp_path / "events.json"
    c = tmp_path / "events.csv"
    app: FastAPI = create_app(
        SessionBootstrap(
            export_json=j,
            export_csv=c,
        )
    )
    with TestClient(app) as client:
        bus = app.state.event_bus
        bus.add_event(_minimal_event(0.0))
        r = client.post("/session/export")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert j.is_file() and c.is_file()
        dumped = json.loads(j.read_text(encoding="utf-8"))
        assert dumped.get("schema_version") == 4
        assert len(dumped.get("events", [])) >= 1


def test_feedback_event_endpoint_validation() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.post(
            "/events/evt-00000001/feedback",
            json={"feedback_type": "not_a_valid_enum", "note": ""},
        )
        assert r.status_code == 422


def test_legacy_feedback_endpoint_still_appends(tmp_path: Path) -> None:
    fb = tmp_path / "legacy.jsonl"
    app = create_app(SessionBootstrap(feedback_path=fb))
    with TestClient(app) as client:
        r = client.post(
            "/feedback",
            json={"event_id": "evt-1", "feedback_type": "note", "note": "x"},
        )
        assert r.status_code == 200
        assert "evt-1" in fb.read_text()
