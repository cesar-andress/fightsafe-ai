"""Smoke tests for live_runner CLI (no GUI)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fightsafe_ai.live import live_runner
from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel


pytestmark = pytest.mark.unit


def test_main_missing_source_returns_one() -> None:
    assert live_runner.main(["--source", "/nonexistent/video_no_such_file.mp4"]) == 1


def test_main_rejects_non_positive_inference_fps() -> None:
    assert live_runner.main(["--source", "0", "--inference-fps", "0"]) == 1
    assert live_runner.main(["--source", "0", "--inference-fps", "-1"]) == 1


def test_main_accepts_webcam_index_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numeric --source skips file existence check and forwards index for WebcamSource."""

    called: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _stub(*args: object, **kwargs: object) -> None:
        called.append((args, kwargs))

    monkeypatch.setattr(live_runner, "run_live", _stub)
    assert live_runner.main(["--source", "0"]) == 0
    assert len(called) == 1
    assert called[0][0][0] == "0"


def test_draw_panel_dimensions_and_long_title() -> None:
    ev = SafetyEvent(
        event_type="test.ev",
        category=EventCategory.UNKNOWN,
        start_time=1.0,
        end_time=1.0,
        level=SafetyLevel.WARNING,
        score=0.5,
        title="x" * 50,
        description="d",
        explanation="unit test",
        source="s",
    )
    panel = live_runner._draw_panel(
        height=200,
        timestamp_seconds=1.25,
        playback_fps=30.0,
        display_fps=28.0,
        risk_level=SafetyLevel.HIGH,
        events=[ev],
        demo_mode=False,
    )
    assert panel.shape == (200, live_runner.PANEL_WIDTH, 3)


def test_demo_tick_updates_bus() -> None:
    bus = EventBus(cooldown_seconds=0.0)
    level = live_runner._demo_tick(bus, timestamp_seconds=0.5, frame_index=90)
    assert isinstance(level, SafetyLevel)
    assert len(bus.all_events()) >= 1


def _tiny_avi(path: Path, *, frames: int = 3) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 24))
    assert out.isOpened()
    for _ in range(frames):
        out.write(np.zeros((24, 32, 3), dtype=np.uint8))
    out.release()


def test_run_live_mock_gui_quit_exports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    avi = tmp_path / "live.avi"
    _tiny_avi(avi)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.namedWindow", lambda *a, **k: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.waitKey", lambda _=None: ord("q"))

    ej = tmp_path / "events.json"
    ec = tmp_path / "events.csv"
    live_runner.run_live(
        avi,
        realtime=False,
        export_json=ej,
        export_csv=ec,
        cooldown_seconds=0.0,
        demo_events=True,
    )
    payload = json.loads(ej.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and "events" in payload
    assert "timestamp" in ec.read_text(encoding="utf-8") and "category" in ec.read_text(
        encoding="utf-8"
    )


def test_run_live_reads_until_eof_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    avi = tmp_path / "live2.avi"
    _tiny_avi(avi, frames=4)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.namedWindow", lambda *a, **k: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("fightsafe_ai.live.live_runner.cv2.waitKey", lambda _=None: 0)

    ej = tmp_path / "out.json"
    ec = tmp_path / "out.csv"
    live_runner.run_live(
        avi,
        realtime=False,
        export_json=ej,
        export_csv=ec,
        cooldown_seconds=0.0,
        demo_events=True,
    )
    data = json.loads(ej.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and isinstance(data.get("events"), list)
