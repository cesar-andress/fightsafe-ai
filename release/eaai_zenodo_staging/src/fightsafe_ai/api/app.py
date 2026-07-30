"""
FastAPI live dashboard: session control, MJPEG preview, WebSocket events, exports, feedback.

Run::

    python -m fightsafe_ai.api.app --source path/to/video.mp4 --realtime
    # open http://127.0.0.1:8000

The OpenCV :mod:`fightsafe_ai.live.live_runner` CLI remains the fallback for local preview.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from fightsafe_ai.api.serialization import safety_event_to_json
from fightsafe_ai.api.session_worker import run_session_worker
from fightsafe_ai.live.event_bus import EventBus, SafetyEvent
from fightsafe_ai.live.gpu_monitor import get_nvidia_gpu_metrics, shutdown_gpu_monitor
from fightsafe_ai.live.tapko_live_events import TAPKO_EVENT_TYPES


logger = logging.getLogger(__name__)


def _torch_cuda_available_safe() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _optional_fastapi():
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is required; install with: pip install 'fightsafe-ai[api]'"
        ) from exc
    return (
        FastAPI,
        HTTPException,
        WebSocket,
        WebSocketDisconnect,
        FileResponse,
        JSONResponse,
        StreamingResponse,
        StaticFiles,
    )


(
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    FileResponse,
    JSONResponse,
    StreamingResponse,
    StaticFiles,
) = _optional_fastapi()


DEFAULT_JSON = Path("outputs/live/events.json")
DEFAULT_CSV = Path("outputs/live/events.csv")
DEFAULT_FEEDBACK = Path("outputs/live/feedback.jsonl")
DEFAULT_SESSION_EVENTS = Path("outputs/live/session_events.json")
DEFAULT_SESSION_METADATA = Path("outputs/live/session_metadata.json")
STATIC_DIR = Path(__file__).resolve().parent / "static"

_SESSION_METADATA_TEMPLATE: dict[str, Any] = {
    "video_path": None,
    "duration_seconds": None,
    "fps": None,
    "width": None,
    "height": None,
    "total_frames": None,
    "processed_frames": None,
    "started_at": None,
    "ended_at": None,
    "progress_percent": None,
}


class FeedbackBody(BaseModel):
    event_id: str = Field(..., min_length=1)
    feedback_type: str = Field(..., min_length=1)
    note: str | None = None


EventFeedbackType = Literal[
    "correct",
    "false_positive",
    "missed_event",
    "wrong_subtype",
    "wrong_severity",
    "needs_expert_review",
    "needs_review",  # deprecated alias for older clients; stored as-is in JSONL
]


class EventFeedbackBody(BaseModel):
    feedback_type: EventFeedbackType
    note: str | None = Field(default=None, max_length=8000)


def _find_event_by_id(bus: EventBus, event_id: str) -> SafetyEvent | None:
    for ev in bus.all_events():
        if ev.event_id == event_id:
            return ev
    return None


def _append_feedback_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class SessionBootstrap:
    """CLI/session configuration (fixed for one server process unless restarted)."""

    source: Path | None = None
    realtime: bool = False
    demo_events: bool = False
    sensitivity: Literal["low", "medium", "high"] = "medium"
    debug_events: bool = False
    enable_strike_detector: bool = False
    strike_percentile: float = 85.0
    strike_merge_frames: int = 8
    enable_tapko_detectors: bool = True
    pose_backend: str = "torch"
    pose_device: str = "auto"
    pose_fp16: bool = False
    export_json: Path = field(default_factory=lambda: DEFAULT_JSON)
    export_csv: Path = field(default_factory=lambda: DEFAULT_CSV)
    feedback_path: Path = field(default_factory=lambda: DEFAULT_FEEDBACK)
    session_events_json: Path = field(default_factory=lambda: DEFAULT_SESSION_EVENTS)
    session_metadata_json: Path = field(default_factory=lambda: DEFAULT_SESSION_METADATA)


class WebSocketHub:
    __slots__ = ("_clients",)

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        try:
            self._clients.remove(websocket)
        except ValueError:
            pass

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


async def _emit_pump(
    *,
    emit_queue: queue.Queue[dict[str, Any]],
    hub: WebSocketHub,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        drained = False
        while True:
            try:
                payload = emit_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            await hub.broadcast_json(payload)
        if not drained:
            await asyncio.sleep(0.04)


@dataclass
class _SessionThreadResources:
    stop_event: threading.Event
    pause_event: threading.Event
    thread: threading.Thread | None = None


def _create_lifespan(bootstrap: SessionBootstrap):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus = EventBus(cooldown_seconds=2.0, visual_expire_seconds=30.0)
        hub = WebSocketHub()
        emit_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        pump_stop = asyncio.Event()
        metrics_lock = threading.Lock()
        metrics: dict[str, Any] = {
            "status": "idle",
            "completed": False,
            "frame_index": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "risk_level": "—",
            "raw_risk_level": "—",
            "error": None,
            "session_duration_s": None,
            "session_started_wall": None,
            "media_timestamp_seconds": None,
        }
        jpeg_lock = threading.Lock()
        latest_jpeg: list[bytes | None] = [None]
        session_res = _SessionThreadResources(
            stop_event=threading.Event(),
            pause_event=threading.Event(),
        )

        app.state.bootstrap = bootstrap
        app.state.event_bus = bus
        app.state.ws_hub = hub
        app.state.emit_queue = emit_queue
        app.state.metrics_lock = metrics_lock
        app.state.metrics = metrics
        app.state.jpeg_lock = jpeg_lock
        app.state.latest_jpeg = latest_jpeg
        app.state.session = session_res
        app.state.exports = {"json": bootstrap.export_json, "csv": bootstrap.export_csv}
        app.state.feedback_path = bootstrap.feedback_path
        app.state.session_metadata = {}
        app.state.db_engine = None
        app.state.db_session_factory = None
        app.state.db_recorder = None
        try:
            from fightsafe_ai.db.models import Base
            from fightsafe_ai.db.session import create_engine_and_sessionmaker

            eng, session_local = create_engine_and_sessionmaker()
            if eng is not None and session_local is not None:
                Base.metadata.create_all(eng)
                app.state.db_engine = eng
                app.state.db_session_factory = session_local
                logger.info("database enabled (ORM tables ensured)")
            else:
                logger.info("database disabled")
        except ImportError:
            logger.info("database disabled (optional sqlalchemy/psycopg not installed)")
        except Exception:
            logger.exception("database initialization failed; continuing without DB")

        pump_task = asyncio.create_task(_emit_pump(emit_queue=emit_queue, hub=hub, stop=pump_stop))

        yield

        pump_stop.set()
        session_res.stop_event.set()
        if session_res.thread is not None:
            session_res.thread.join(timeout=10.0)
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        shutdown_gpu_monitor()

    return lifespan


def create_app(bootstrap: SessionBootstrap | None = None) -> FastAPI:
    boot = bootstrap or SessionBootstrap()
    app = FastAPI(
        title="FightSafe Live Dashboard",
        version="0.2.0",
        lifespan=_create_lifespan(boot),
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    try:
        from fightsafe_ai.db.routes import register_db_routes

        register_db_routes(app)
    except ImportError:
        pass

    @app.get("/")
    def index() -> Any:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return JSONResponse(
                content={"detail": "Dashboard static files missing. Install package with static/."},
                status_code=503,
            )
        return FileResponse(index_path)

    @app.get("/health")
    def health() -> dict[str, str | int | float | bool | None]:
        hub: WebSocketHub = app.state.ws_hub
        m = _current_metrics(app)
        return {
            "status": "ok",
            "websocket_clients": hub.client_count,
            "session": m.get("status"),
        }

    @app.get("/system/gpu")
    def system_gpu() -> JSONResponse:
        boot: SessionBootstrap = app.state.bootstrap
        gpu = get_nvidia_gpu_metrics()
        return JSONResponse(
            content={
                **gpu,
                "cuda_available": _torch_cuda_available_safe(),
                "pose_backend": boot.pose_backend,
                "pose_device": boot.pose_device,
            }
        )

    @app.get(
        "/events",
        tags=["events"],
        summary="List recent safety episodes",
    )
    def list_events(limit: int = 2000) -> JSONResponse:
        """
        Returns JSON array of episodes. TapKO-aligned rows use ``event_type`` values listed in
        ``GET /session/status`` → ``tapko_event_types`` (``submission_signal.*``,
        ``extreme_vulnerability.*``). These are **candidate** cues — not official outcomes.
        """
        bus: EventBus = app.state.event_bus
        lim = max(1, min(limit, 20_000))
        rows = bus.all_events()[-lim:]
        payload = [safety_event_to_json(e) for e in rows]
        return JSONResponse(content=payload)

    @app.get("/session/status")
    def session_status() -> JSONResponse:
        bus: EventBus = app.state.event_bus
        m = _current_metrics(app)
        evs = bus.all_events()
        b: SessionBootstrap = app.state.bootstrap
        return JSONResponse(
            content={
                **m,
                "event_count": len(evs),
                "source": str(b.source) if b.source else None,
                "demo_events": bool(b.demo_events),
                "sensitivity": str(b.sensitivity),
                "debug_events": bool(b.debug_events),
                "enable_strike_detector": bool(b.enable_strike_detector),
                "strike_percentile": float(b.strike_percentile),
                "strike_merge_frames": int(b.strike_merge_frames),
                "enable_tapko_detectors": bool(b.enable_tapko_detectors),
                "tapko_event_types": list(TAPKO_EVENT_TYPES),
            }
        )

    @app.get("/session/metadata")
    def session_metadata() -> JSONResponse:
        with app.state.metrics_lock:
            raw = dict(app.state.session_metadata)
        merged = {**_SESSION_METADATA_TEMPLATE, **raw}
        return JSONResponse(content=merged)

    @app.post("/session/start")
    def session_start() -> JSONResponse:
        boot: SessionBootstrap = app.state.bootstrap
        if boot.source is None or not Path(boot.source).expanduser().is_file():
            raise HTTPException(
                status_code=400, detail="Invalid or missing --source for this server."
            )
        res: _SessionThreadResources = app.state.session
        if res.thread is not None and res.thread.is_alive():
            raise HTTPException(status_code=409, detail="Session already running.")
        # reset session state
        res.stop_event = threading.Event()
        res.pause_event = threading.Event()
        app.state.event_bus = EventBus(cooldown_seconds=2.0, visual_expire_seconds=30.0)
        session_md: dict[str, Any] = {}
        with app.state.metrics_lock:
            app.state.session_metadata = session_md
            app.state.metrics.clear()
            app.state.metrics.update(
                {
                    "status": "starting",
                    "completed": False,
                    "frame_index": 0,
                    "fps": 0.0,
                    "latency_ms": 0.0,
                    "risk_level": "—",
                    "raw_risk_level": "—",
                    "error": None,
                    "session_duration_s": None,
                    "session_started_wall": None,
                    "media_timestamp_seconds": None,
                }
            )
        with app.state.jpeg_lock:
            app.state.latest_jpeg[0] = None

        app.state.db_recorder = None
        eng = getattr(app.state, "db_engine", None)
        sf = getattr(app.state, "db_session_factory", None)
        rec = None
        if eng is not None and sf is not None:
            try:
                from fightsafe_ai.db.live_dashboard import create_live_run

                cfg = {
                    "sensitivity": boot.sensitivity,
                    "realtime": boot.realtime,
                    "demo_events": boot.demo_events,
                    "enable_strike_detector": boot.enable_strike_detector,
                    "strike_percentile": boot.strike_percentile,
                    "strike_merge_frames": boot.strike_merge_frames,
                    "enable_tapko_detectors": boot.enable_tapko_detectors,
                    "pose_backend": boot.pose_backend,
                    "pose_device": boot.pose_device,
                    "pose_fp16": boot.pose_fp16,
                }
                rec = create_live_run(
                    eng,
                    sf,
                    source=Path(boot.source).expanduser().resolve(),
                    bootstrap_config=cfg,
                    demo_events=bool(boot.demo_events),
                    realtime=bool(boot.realtime),
                )
                app.state.db_recorder = rec
            except Exception:
                logger.exception("Could not create DB run record")

        bus: EventBus = app.state.event_bus
        session_events_path = Path(boot.session_events_json).expanduser().resolve()
        export_json_path = Path(boot.export_json).expanduser().resolve()
        export_csv_path = Path(boot.export_csv).expanduser().resolve()
        session_metadata_path = Path(boot.session_metadata_json).expanduser().resolve()
        t = threading.Thread(
            target=run_session_worker,
            kwargs={
                "bus": bus,
                "video_path": Path(boot.source).expanduser().resolve(),
                "realtime": boot.realtime,
                "demo_events": boot.demo_events,
                "sensitivity": boot.sensitivity,
                "debug_events": boot.debug_events,
                "enable_strike_detector": boot.enable_strike_detector,
                "strike_percentile": boot.strike_percentile,
                "strike_merge_frames": boot.strike_merge_frames,
                "enable_tapko_detectors": boot.enable_tapko_detectors,
                "pose_backend": boot.pose_backend,
                "pose_device": boot.pose_device,
                "pose_fp16": boot.pose_fp16,
                "stop_event": res.stop_event,
                "pause_event": res.pause_event,
                "emit_queue": app.state.emit_queue,
                "metrics_lock": app.state.metrics_lock,
                "metrics": app.state.metrics,
                "jpeg_lock": app.state.jpeg_lock,
                "latest_jpeg": app.state.latest_jpeg,
                "session_metadata": session_md,
                "session_events_path": session_events_path,
                "export_json_path": export_json_path,
                "export_csv_path": export_csv_path,
                "session_metadata_path": session_metadata_path,
                "on_event_committed": rec.on_safety_event if rec is not None else None,
                "on_timeline_sample": rec.on_timeline_tick if rec is not None else None,
                "timeline_stride_frames": 6,
                "on_worker_finished": (
                    (
                        lambda: rec.finalize(
                            status="stopped" if res.stop_event.is_set() else "completed"
                        )
                    )
                    if rec is not None
                    else None
                ),
            },
            name="fightsafe-dashboard-session",
            daemon=True,
        )
        res.thread = t
        t.start()
        return JSONResponse(content={"ok": True, "message": "Session started"})

    @app.post("/session/pause")
    def session_pause() -> JSONResponse:
        res: _SessionThreadResources = app.state.session
        res.pause_event.set()
        return JSONResponse(content={"ok": True})

    @app.post("/session/resume")
    def session_resume() -> JSONResponse:
        res: _SessionThreadResources = app.state.session
        res.pause_event.clear()
        return JSONResponse(content={"ok": True})

    @app.post("/session/stop")
    def session_stop() -> JSONResponse:
        res: _SessionThreadResources = app.state.session
        res.stop_event.set()
        res.pause_event.clear()
        if res.thread is not None:
            res.thread.join(timeout=12.0)
        res.thread = None
        with app.state.metrics_lock:
            app.state.metrics["status"] = "stopped"
        app.state.db_recorder = None
        return JSONResponse(content={"ok": True})

    @app.post("/session/export")
    def session_export() -> JSONResponse:
        bus: EventBus = app.state.event_bus
        paths = app.state.exports
        pj = Path(paths["json"])
        pc = Path(paths["csv"])
        bus.export_json(pj)
        bus.export_csv(pc)
        dbrec = getattr(app.state, "db_recorder", None)
        if dbrec is not None:
            try:
                dbrec.on_export_paths(json_path=str(pj), csv_path=str(pc))
            except Exception:
                logger.exception("DB artifact record failed")
        return JSONResponse(content={"ok": True, "json": str(pj), "csv": str(pc)})

    @app.post("/session/clear")
    def session_clear() -> JSONResponse:
        res: _SessionThreadResources = app.state.session
        res.stop_event.set()
        if res.thread is not None:
            res.thread.join(timeout=12.0)
        res.thread = None
        res.stop_event = threading.Event()
        res.pause_event = threading.Event()
        app.state.event_bus = EventBus(cooldown_seconds=2.0, visual_expire_seconds=30.0)
        with app.state.metrics_lock:
            app.state.session_metadata = {}
            app.state.metrics.update(
                {
                    "status": "idle",
                    "completed": False,
                    "frame_index": 0,
                    "fps": 0.0,
                    "latency_ms": 0.0,
                    "risk_level": "—",
                    "raw_risk_level": "—",
                    "error": None,
                    "session_duration_s": None,
                    "media_timestamp_seconds": None,
                }
            )
        with app.state.jpeg_lock:
            app.state.latest_jpeg[0] = None
        return JSONResponse(content={"ok": True})

    @app.get("/video/stream")
    async def video_stream() -> StreamingResponse:
        boundary = "frame"

        async def gen():
            while True:
                chunk: bytes | None = None
                with app.state.jpeg_lock:
                    if app.state.latest_jpeg[0]:
                        chunk = app.state.latest_jpeg[0]
                if chunk:
                    yield (
                        b"--"
                        + boundary.encode()
                        + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                        + chunk
                        + b"\r\n"
                    )
                await asyncio.sleep(0.03)

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        hub: WebSocketHub = app.state.ws_hub
        await hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(websocket)

    @app.post("/feedback")
    def post_feedback(body: FeedbackBody) -> JSONResponse:
        """Legacy endpoint (minimal JSONL row). Prefer :http:post:`/events/{event_id}/feedback`."""

        path: Path = app.state.feedback_path
        line = {
            "timestamp": time.time(),
            "event_id": body.event_id,
            "feedback_type": body.feedback_type,
            "note": body.note,
        }
        _append_feedback_record(path, line)
        dbrec = getattr(app.state, "db_recorder", None)
        if dbrec is not None:
            try:
                dbrec.on_feedback(
                    external_event_id=body.event_id,
                    feedback_type=body.feedback_type,
                    note=body.note,
                    payload=line,
                )
            except Exception:
                logger.exception("DB feedback persist failed")
        return JSONResponse(content={"ok": True})

    @app.post("/events/{event_id}/feedback")
    def post_event_feedback(event_id: str, body: EventFeedbackBody) -> JSONResponse:
        bus: EventBus = app.state.event_bus
        ev = _find_event_by_id(bus, event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail=f"Unknown event_id: {event_id}")
        snap = safety_event_to_json(ev)
        record = {
            "event_id": event_id,
            "event_timestamp": snap.get("timestamp_seconds"),
            "feedback_type": body.feedback_type,
            "note": body.note,
            "created_at": datetime.now(UTC).isoformat(),
            "event_snapshot": snap,
        }
        path: Path = app.state.feedback_path
        _append_feedback_record(path, record)
        dbrec = getattr(app.state, "db_recorder", None)
        if dbrec is not None:
            try:
                dbrec.on_feedback(
                    external_event_id=event_id,
                    feedback_type=body.feedback_type,
                    note=body.note,
                    payload=record,
                )
            except Exception:
                logger.exception("DB feedback persist failed")
        return JSONResponse(content={"ok": True, "event_id": event_id})

    return app


def _current_metrics(app: FastAPI) -> dict[str, Any]:
    with app.state.metrics_lock:
        return dict(app.state.metrics)


def _bootstrap_from_env() -> SessionBootstrap:
    raw = (
        os.environ.get("FIGHTSAFE_SOURCE") or os.environ.get("FIGHTSAFE_LIVE_SOURCE") or ""
    ).strip()
    if not raw:
        return SessionBootstrap()
    p = Path(raw).expanduser().resolve()
    return SessionBootstrap(source=p if p.is_file() else None)


app = create_app(_bootstrap_from_env())


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="FightSafe live web dashboard")
    p.add_argument("--source", type=str, default=None, help="Video file path")
    p.add_argument("--realtime", action="store_true", help="Wall-clock pacing")
    p.add_argument("--demo-events", action="store_true", help="Synthetic events (no pose pipeline)")
    p.add_argument(
        "--sensitivity",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Live interpretable rules: medium=defaults from YAML; high=more sensitive; low=less",
    )
    p.add_argument(
        "--debug-events",
        action="store_true",
        help="Log each generated event to the server console (INFO).",
    )
    p.add_argument(
        "--enable-strike-detector",
        action="store_true",
        help="Run BoxingVI-style wrist-speed strike candidates on the live pose buffer (non-demo).",
    )
    p.add_argument(
        "--strike-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold for strike detector (default: 85, same as boxingvi_skeleton_runner).",
    )
    p.add_argument(
        "--strike-merge-frames",
        type=int,
        default=8,
        help="Merge strike segments when gap (frames) is at most this value (default: 8).",
    )
    p.add_argument(
        "--no-tapko-detectors",
        action="store_true",
        help="Disable TapKO pose detectors (submission_signal.* / extreme_vulnerability.* candidates).",
    )
    p.add_argument("--pose-backend", type=str, default="torch")
    p.add_argument("--pose-device", type=str, default="auto")
    p.add_argument("--pose-fp16", action="store_true")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--export-json",
        type=Path,
        default=DEFAULT_JSON,
        help="JSON export path",
    )
    p.add_argument(
        "--export-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV export path",
    )
    p.add_argument(
        "--session-events",
        type=Path,
        default=DEFAULT_SESSION_EVENTS,
        help="Periodic autosave JSON for current session events",
    )
    p.add_argument(
        "--session-metadata",
        type=Path,
        default=DEFAULT_SESSION_METADATA,
        help="Periodic autosave JSON for session progress / video metadata",
    )
    args = p.parse_args(argv)

    src = Path(args.source).expanduser().resolve() if args.source else None
    if src is not None and not src.is_file():
        print(f"Source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    bootstrap = SessionBootstrap(
        source=src,
        realtime=bool(args.realtime),
        demo_events=bool(args.demo_events),
        sensitivity=cast(Literal["low", "medium", "high"], str(args.sensitivity).lower()),  # noqa: TC006
        debug_events=bool(args.debug_events),
        enable_strike_detector=bool(args.enable_strike_detector),
        strike_percentile=float(args.strike_percentile),
        strike_merge_frames=int(args.strike_merge_frames),
        enable_tapko_detectors=not bool(args.no_tapko_detectors),
        pose_backend=args.pose_backend,
        pose_device=args.pose_device,
        pose_fp16=bool(args.pose_fp16),
        export_json=args.export_json,
        export_csv=args.export_csv,
        session_events_json=args.session_events,
        session_metadata_json=args.session_metadata,
    )

    dashboard = create_app(bootstrap)
    uvicorn.run(dashboard, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()


__all__ = [
    "EventFeedbackBody",
    "FeedbackBody",
    "SessionBootstrap",
    "WebSocketHub",
    "app",
    "create_app",
    "main",
]
