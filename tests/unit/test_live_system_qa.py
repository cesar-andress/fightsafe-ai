"""
QA coverage for the live stack: pipeline contract, EventBus timing, VideoSource with mocked OpenCV,
and session HTTP endpoints (no GPU, no real video assets).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fightsafe_ai.api.app import SessionBootstrap, create_app
from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel
from fightsafe_ai.live.live_pipeline import LiveFrameResult, LivePipeline, LivePipelineConfig
from fightsafe_ai.live.video_source import FileVideoSource, WebcamSource, open_video_source
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult


pytestmark = pytest.mark.unit

_CAP_PROP_DURATION = int(getattr(cv2, "CAP_PROP_DURATION", -1))


def _minimal_event(ts: float = 1.0) -> SafetyEvent:
    return SafetyEvent(
        event_type="test.signal",
        category=EventCategory.UNKNOWN,
        start_time=ts,
        end_time=ts,
        level=SafetyLevel.INFO,
        score=0.1,
        title="t",
        description="d",
        explanation="",
        source="unit",
    )


def test_live_pipeline_process_frame_result_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiveFrameResult exposes the documented keys after one inference step (pose mocked)."""

    kp = Keypoint("nose", 0.5, 0.5, 0.0, 1.0)

    def _predict(_frame: np.ndarray) -> PoseResult:
        return PoseResult(frame_id="", keypoints=[kp])

    fake_estimator = MagicMock()
    fake_estimator.predict = _predict

    monkeypatch.setattr(
        "fightsafe_ai.live.live_pipeline.create_runtime_pose_estimator",
        lambda *args, **kwargs: fake_estimator,
    )

    pipe = LivePipeline(
        LivePipelineConfig(
            video_fps=30.0,
            buffer_seconds=2.0,
            smooth_seconds=0.5,
            max_infer_hz=60.0,
        )
    )
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    out: LiveFrameResult = pipe.process_frame(frame, 0.0, frame_index=0)

    assert set(out.keys()) >= {
        "risk_level",
        "events",
        "pose",
        "risk_score",
        "smoothed_risk_score",
        "raw_risk_level",
    }
    assert isinstance(out["raw_risk_level"], str)
    assert isinstance(out["risk_score"], float)
    assert isinstance(out["smoothed_risk_score"], float)


def test_event_bus_tick_closes_open_episode_after_silence() -> None:
    bus = EventBus(
        cooldown_seconds=0.0,
        merge_gap_seconds=0.05,
        silence_close_seconds=0.5,
    )
    bus.add_event(_minimal_event(1.0))
    assert len(bus.active_events) == 1

    bus.tick(now_seconds=10.0)

    assert len(bus.active_events) == 0
    stored = bus.all_events()
    assert len(stored) == 1
    assert stored[0].is_finished is True


class _FakeVideoCapture:
    """Minimal cv2.VideoCapture stand-in (no hardware, no container files)."""

    def __init__(
        self,
        *,
        width: float = 64.0,
        height: float = 48.0,
        fps: float = 10.0,
        frame_count: float = 0.0,
        max_frames: int = 3,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_count = frame_count
        self._max_frames = max_frames
        self._reads = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if _CAP_PROP_DURATION >= 0 and prop == _CAP_PROP_DURATION:
            return 0.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._reads += 1
        if self._reads > self._max_frames:
            return False, None
        h = max(int(self._height), 1)
        w = max(int(self._width), 1)
        return True, np.zeros((h, w, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_file_video_source_mock_capture_reads_sequential_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "fightsafe_ai.live.video_source.cv2.VideoCapture",
        lambda *_a, **_k: _FakeVideoCapture(max_frames=3),
    )
    path = tmp_path / "placeholder.mp4"
    path.write_bytes(b"")

    src = FileVideoSource(path, realtime=False)
    try:
        assert src.width >= 1 and src.height >= 1
        f0, m0 = src.read_frame()
        assert f0 is not None and m0 is not None
        assert m0.frame_index == 0
        _f1, m1 = src.read_frame()
        assert m1 is not None and m1.frame_index == 1
        _f2, m2 = src.read_frame()
        assert m2 is not None
        end, _ = src.read_frame()
        assert end is None
    finally:
        src.close()


def test_open_video_source_webcam_vs_file_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "fightsafe_ai.live.video_source.cv2.VideoCapture",
        lambda *_a, **_k: _FakeVideoCapture(max_frames=99),
    )

    cam = open_video_source("0", realtime=False)
    assert isinstance(cam, WebcamSource)
    cam.close()

    disk_path = tmp_path / "clip.mp4"
    disk_path.write_bytes(b"")
    disk = open_video_source(str(disk_path))
    assert isinstance(disk, FileVideoSource)
    disk.close()


def test_session_export_empty_event_bus_writes_schema(tmp_path: Path) -> None:
    j = tmp_path / "e.json"
    c = tmp_path / "e.csv"
    app: FastAPI = create_app(SessionBootstrap(export_json=j, export_csv=c))
    with TestClient(app) as client:
        r = client.post("/session/export")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        dumped = json.loads(j.read_text(encoding="utf-8"))
        assert dumped.get("schema_version") == 4
        assert dumped.get("events") == []
        assert c.is_file()


def test_session_pause_resume_stop_when_idle() -> None:
    app: FastAPI = create_app(SessionBootstrap())
    with TestClient(app) as client:
        assert client.post("/session/pause").status_code == 200
        assert client.post("/session/resume").status_code == 200
        assert client.post("/session/stop").status_code == 200


def test_session_clear_resets_to_idle(tmp_path: Path) -> None:
    j = tmp_path / "e.json"
    c = tmp_path / "e.csv"
    app: FastAPI = create_app(SessionBootstrap(export_json=j, export_csv=c))
    with TestClient(app) as client:
        app.state.event_bus.add_event(_minimal_event(0.0))
        r = client.post("/session/clear")
        assert r.status_code == 200
        st = client.get("/session/status").json()
        assert st["status"] == "idle"
        assert st["event_count"] == 0


def test_session_status_includes_expected_keys() -> None:
    app: FastAPI = create_app(SessionBootstrap())
    with TestClient(app) as client:
        body = client.get("/session/status").json()
        for key in (
            "status",
            "completed",
            "event_count",
            "risk_level",
            "raw_risk_level",
            "frame_index",
            "fps",
            "media_timestamp_seconds",
        ):
            assert key in body
