"""FastAPI live API smoke tests (requires optional ``fastapi``)."""

from __future__ import annotations

import pytest


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from fightsafe_ai.api.app import app


pytestmark = pytest.mark.unit


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "websocket_clients" in data
        assert data["websocket_clients"] == 0


def test_events_returns_json_list() -> None:
    with TestClient(app) as client:
        r = client.get("/events")
        assert r.status_code == 200
        assert r.json() == []


def test_websocket_receives_payload_from_emit_queue() -> None:
    """Thread-safe queue is bridged to WebSocket (same path as live pipeline)."""
    import time

    payload = {"event_type": "unit.test", "level": "INFO", "source": "test"}
    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        assert client.get("/health").json()["websocket_clients"] == 1
        app.state.emit_queue.put(payload)  # type: ignore[attr-defined]
        time.sleep(0.15)
        assert ws.receive_json() == payload
