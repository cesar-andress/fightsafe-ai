"""Dashboard FastAPI smoke tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fightsafe_ai.api.app import SessionBootstrap, create_app


pytestmark = pytest.mark.unit


def _write_minimal_avi(path: Path, *, n_frames: int = 4, fps: float = 20.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    h, w = 32, 48
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert out.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 15, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_health_and_events_empty() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
        r2 = client.get("/events")
        assert r2.status_code == 200
        assert r2.json() == []


def test_session_start_requires_source() -> None:
    app = create_app(SessionBootstrap(source=None))
    with TestClient(app) as client:
        r = client.post("/session/start")
        assert r.status_code == 400


def test_system_gpu_endpoint() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.get("/system/gpu")
        assert r.status_code == 200
        data = r.json()
        assert "cuda_available" in data
        assert "pose_backend" in data
        assert "pose_device" in data
        assert "nvidia_nvml_available" in data


def test_session_metadata_idle_shape() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.get("/session/metadata")
        assert r.status_code == 200
        data = r.json()
        assert "video_path" in data
        assert "duration_seconds" in data
        assert "processed_frames" in data
        assert data["video_path"] is None


def test_feedback_append(tmp_path: Path) -> None:
    fb = tmp_path / "fb.jsonl"
    app = create_app(SessionBootstrap(feedback_path=fb))
    with TestClient(app) as client:
        r = client.post(
            "/feedback",
            json={
                "event_id": "evt-00000001",
                "feedback_type": "correct",
                "note": "ok",
            },
        )
        assert r.status_code == 200
        assert fb.read_text().strip()


def test_event_feedback_jsonl(tmp_path: Path) -> None:
    from fightsafe_ai.live.event_bus import EventCategory, SafetyEvent, SafetyLevel

    fb = tmp_path / "fb.jsonl"
    app: FastAPI = create_app(SessionBootstrap(feedback_path=fb))
    with TestClient(app) as client:
        bus = app.state.event_bus
        bus.add_event(
            SafetyEvent(
                event_type="unit.fb",
                category=EventCategory.UNKNOWN,
                start_time=1.0,
                end_time=1.2,
                level=SafetyLevel.WARNING,
                score=0.5,
                title="t",
                description="d",
                explanation="e",
                source="s",
            )
        )
        eid = bus.all_events()[0].event_id
        r = client.post(
            f"/events/{eid}/feedback",
            json={"feedback_type": "needs_expert_review", "note": "check rule"},
        )
        assert r.status_code == 200
        line = json.loads(fb.read_text().strip().split("\n")[-1])
        assert line["event_id"] == eid
        assert line["feedback_type"] == "needs_expert_review"
        assert line["note"] == "check rule"
        assert "created_at" in line
        assert line["event_snapshot"]["event_id"] == eid
        assert line["event_timestamp"] == line["event_snapshot"]["timestamp_seconds"]


def test_event_feedback_unknown_returns_404() -> None:
    app = create_app(SessionBootstrap())
    with TestClient(app) as client:
        r = client.post(
            "/events/evt-99999999/feedback",
            json={"feedback_type": "correct"},
        )
        assert r.status_code == 404


def test_demo_session_writes_exports_and_metadata(tmp_path: Path) -> None:
    """End-of-session persistence without GPU or pose (demo_events)."""
    avi = tmp_path / "session.avi"
    _write_minimal_avi(avi, n_frames=3, fps=30.0)
    ej = tmp_path / "events.json"
    ec = tmp_path / "events.csv"
    se = tmp_path / "session_events.json"
    sm = tmp_path / "session_metadata.json"
    boot = SessionBootstrap(
        source=avi,
        demo_events=True,
        export_json=ej,
        export_csv=ec,
        session_events_json=se,
        session_metadata_json=sm,
    )
    app = create_app(boot)
    with TestClient(app) as client:
        assert client.post("/session/start").status_code == 200
        for _ in range(400):
            st = client.get("/session/status").json()
            if st.get("completed"):
                break
            time.sleep(0.02)
        else:
            pytest.fail("demo session did not complete")
        assert st.get("status") == "completed"

    assert ej.is_file() and ec.is_file() and se.is_file() and sm.is_file()
    meta = json.loads(sm.read_text(encoding="utf-8"))
    assert meta.get("processed_frames", 0) >= 1
    doc = json.loads(ej.read_text(encoding="utf-8"))
    assert doc.get("schema_version") == 4


def test_session_export_endpoint_writes_paths(tmp_path: Path) -> None:
    avi = tmp_path / "exp.avi"
    _write_minimal_avi(avi, n_frames=2, fps=60.0)
    ej = tmp_path / "out.json"
    ec = tmp_path / "out.csv"
    boot = SessionBootstrap(source=avi, demo_events=True, export_json=ej, export_csv=ec)
    app = create_app(boot)
    with TestClient(app) as client:
        client.post("/session/start")
        for _ in range(300):
            if client.get("/session/status").json().get("completed"):
                break
            time.sleep(0.02)
        r = client.post("/session/export")
        assert r.status_code == 200
        body = r.json()
        assert Path(body["json"]).resolve() == ej.resolve()
    assert ej.stat().st_size > 10
