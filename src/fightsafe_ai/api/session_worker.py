"""
Background video session: decode frames, run :class:`~fightsafe_ai.live.live_pipeline.LivePipeline`
or demo mode, push JPEG previews and serialized events to async bridge queues.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import cv2

from fightsafe_ai.api.serialization import safety_event_to_json
from fightsafe_ai.live.event_bus import EventBus, SafetyLevel
from fightsafe_ai.live.live_overlay import draw_live_overlay
from fightsafe_ai.live.live_pipeline import LivePipeline, LivePipelineConfig
from fightsafe_ai.live.live_runner import _demo_tick, _smoothed_level_to_raw_label
from fightsafe_ai.live.performance import LivePerformanceMonitor
from fightsafe_ai.live.video_source import FileVideoSource, open_video_source


logger = logging.getLogger(__name__)

AUTOSAVE_INTERVAL_S = 5.0


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _session_progress(
    session_metadata: dict[str, Any],
    *,
    processed_frames: int,
    media_ts: float | None,
) -> float | None:
    tf = session_metadata.get("total_frames")
    if isinstance(tf, int) and tf > 0:
        return min(100.0, 100.0 * processed_frames / float(tf))
    dur = session_metadata.get("duration_seconds")
    if isinstance(dur, (int, float)) and float(dur) > 0 and media_ts is not None:
        return min(100.0, 100.0 * float(media_ts) / float(dur))
    return None


def _emit_after_add(
    bus: EventBus,
    emit_queue: queue.Queue[dict[str, Any]],
    ev: Any,
    *,
    debug_events: bool = False,
    on_event_committed: Callable[[Any], None] | None = None,
) -> None:
    if not bus.add_event(ev):
        return
    if debug_events:
        logger.info(
            "Dashboard event: %s",
            json.dumps(safety_event_to_json(ev), ensure_ascii=False, default=str),
        )
    for e in reversed(bus.all_events()):
        if e.event_type == ev.event_type:
            if on_event_committed is not None:
                try:
                    on_event_committed(e)
                except Exception:
                    logger.exception("on_event_committed callback failed")
            emit_queue.put({"type": "event", "event": safety_event_to_json(e)})
            return


def persist_session_artifacts(
    bus: EventBus,
    *,
    session_events_path: Path,
    export_json_path: Path | None,
    export_csv_path: Path | None,
    session_metadata_path: Path | None,
    session_metadata: dict[str, Any],
) -> None:
    """Write session_events snapshot, optional dashboard exports, and JSON metadata."""
    bus.export_json(session_events_path)
    if export_json_path is not None:
        ej = export_json_path.expanduser().resolve()
        ej.parent.mkdir(parents=True, exist_ok=True)
        bus.export_json(ej)
    if export_csv_path is not None:
        ec = export_csv_path.expanduser().resolve()
        ec.parent.mkdir(parents=True, exist_ok=True)
        bus.export_csv(ec)
    if session_metadata_path is not None:
        mp = session_metadata_path.expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(
            json.dumps(session_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def run_session_worker(
    *,
    bus: EventBus,
    video_path: Path,
    realtime: bool,
    demo_events: bool,
    sensitivity: str = "medium",
    debug_events: bool = False,
    enable_strike_detector: bool = False,
    strike_percentile: float = 85.0,
    strike_merge_frames: int = 8,
    enable_tapko_detectors: bool = True,
    pose_backend: str,
    pose_device: str,
    pose_fp16: bool,
    stop_event: threading.Event,
    pause_event: threading.Event,
    emit_queue: queue.Queue[dict[str, Any]],
    metrics_lock: threading.Lock,
    metrics: dict[str, Any],
    jpeg_lock: threading.Lock,
    latest_jpeg: list[bytes | None],
    session_metadata: dict[str, Any],
    session_events_path: Path,
    export_json_path: Path | None = None,
    export_csv_path: Path | None = None,
    session_metadata_path: Path | None = None,
    on_event_committed: Callable[[Any], None] | None = None,
    on_timeline_sample: Callable[..., None] | None = None,
    timeline_stride_frames: int = 6,
    on_worker_finished: Callable[[], None] | None = None,
) -> None:
    path = video_path.expanduser().resolve()
    if not path.is_file():
        logger.error("Session worker: file not found: %s", path)
        with metrics_lock:
            metrics["error"] = f"File not found: {path}"
            metrics["status"] = "error"
            session_metadata.update(
                {
                    "video_path": str(path),
                    "duration_seconds": None,
                    "fps": None,
                    "width": None,
                    "height": None,
                    "total_frames": None,
                    "processed_frames": 0,
                    "started_at": _iso_now(),
                    "ended_at": _iso_now(),
                    "progress_percent": None,
                }
            )
        return

    try:
        src = open_video_source(path, realtime=realtime)
    except OSError as exc:
        logger.exception("Session worker: open failed")
        with metrics_lock:
            metrics["error"] = str(exc)
            metrics["status"] = "error"
            session_metadata.update(
                {
                    "video_path": str(path),
                    "duration_seconds": None,
                    "fps": None,
                    "width": None,
                    "height": None,
                    "total_frames": None,
                    "processed_frames": 0,
                    "started_at": _iso_now(),
                    "ended_at": _iso_now(),
                    "progress_percent": None,
                }
            )
        return

    sens = str(sensitivity or "medium").strip().lower()
    if sens not in ("low", "medium", "high"):
        sens = "medium"

    pipeline: LivePipeline | None = None
    if not demo_events:
        pipeline = LivePipeline(
            LivePipelineConfig(
                pose_backend=pose_backend,
                pose_device=pose_device,
                pose_fp16=pose_fp16,
                video_fps=float(src.fps),
                live_sensitivity=cast(Literal["low", "medium", "high"], sens),  # noqa: TC006
                enable_strike_detector=bool(enable_strike_detector),
                strike_percentile=float(strike_percentile),
                strike_merge_frames=int(strike_merge_frames),
                enable_tapko_detectors=bool(enable_tapko_detectors),
            )
        )

    perf = LivePerformanceMonitor(target_fps=24.0, fps_recover=28.0, max_infer_stride=8)
    prev_t = time.perf_counter()
    ema_fps = float(src.fps or 30.0)
    session_t0 = time.perf_counter()
    last_demo_snap_t = 0.0
    processed_frames = 0
    last_media_ts: float | None = None
    last_autosave_m = time.monotonic()
    started_iso = _iso_now()

    def update_metrics(**kwargs: Any) -> None:
        with metrics_lock:
            metrics.update(kwargs)

    tf_hint: int | None = None
    dur_hint: float | None = None
    if isinstance(src, FileVideoSource):
        tf_hint = src.total_frames
        dur_hint = src.duration_seconds

    with metrics_lock:
        session_metadata.clear()
        session_metadata.update(
            {
                "video_path": str(path),
                "duration_seconds": dur_hint,
                "fps": float(src.fps),
                "width": int(src.width),
                "height": int(src.height),
                "total_frames": tf_hint,
                "processed_frames": 0,
                "started_at": started_iso,
                "ended_at": None,
                "progress_percent": None,
            }
        )

    def maybe_autosave() -> None:
        nonlocal last_autosave_m
        now_m = time.monotonic()
        if now_m - last_autosave_m < AUTOSAVE_INTERVAL_S:
            return
        last_autosave_m = now_m
        try:
            persist_session_artifacts(
                bus,
                session_events_path=session_events_path,
                export_json_path=export_json_path,
                export_csv_path=export_csv_path,
                session_metadata_path=session_metadata_path,
                session_metadata=session_metadata,
            )
        except Exception:
            logger.exception("Periodic session autosave failed")

    update_metrics(
        status="running",
        completed=False,
        error=None,
        frame_index=0,
        fps=0.0,
        latency_ms=0.0,
        risk_level="INFO",
        raw_risk_level="LOW",
        session_started_wall=time.time(),
    )

    try:
        while not stop_event.is_set():
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.04)
                with metrics_lock:
                    metrics["status"] = "paused"
            if stop_event.is_set():
                break
            with metrics_lock:
                if metrics.get("status") == "paused":
                    metrics["status"] = "running"

            t_frame = time.perf_counter()
            frame, meta = src.read_frame()
            if frame is None or meta is None:
                with metrics_lock:
                    metrics["status"] = "completed"
                    metrics["completed"] = True
                    metrics["session_duration_s"] = time.perf_counter() - session_t0
                    if (
                        session_metadata.get("duration_seconds") is None
                        and last_media_ts is not None
                    ):
                        session_metadata["duration_seconds"] = float(last_media_ts) + (
                            1.0 / max(float(src.fps), 1e-6)
                        )
                    if session_metadata.get("total_frames") is None and processed_frames > 0:
                        session_metadata["total_frames"] = int(processed_frames)
                    session_metadata["processed_frames"] = int(processed_frames)
                    session_metadata["progress_percent"] = 100.0
                    session_metadata["ended_at"] = _iso_now()
                try:
                    persist_session_artifacts(
                        bus,
                        session_events_path=session_events_path,
                        export_json_path=export_json_path,
                        export_csv_path=export_csv_path,
                        session_metadata_path=session_metadata_path,
                        session_metadata=session_metadata,
                    )
                except Exception:
                    logger.exception("Final session autosave failed")
                emit_queue.put({"type": "session", "status": "completed"})
                break

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-9)
            prev_t = now
            ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt)
            perf.tick_display_loop(dt_seconds=dt)

            t_proc = time.perf_counter()
            if demo_events:
                lvl = _demo_tick(
                    bus,
                    timestamp_seconds=meta.timestamp_seconds,
                    frame_index=meta.frame_index,
                    periodic_visible_high=True,
                    debug_events=debug_events,
                )
                raw_demo = _smoothed_level_to_raw_label(lvl)
                dangerous = lvl in (SafetyLevel.HIGH, SafetyLevel.CRITICAL) or raw_demo in (
                    "HIGH",
                    "CRITICAL",
                )
                vis = draw_live_overlay(
                    frame,
                    pose=None,
                    risk_level=lvl,
                    raw_risk_level=raw_demo,
                    triggered_rules=[],
                    dangerous=dangerous,
                )
                bus.tick(meta.timestamp_seconds)
                proc_ms = (time.perf_counter() - t_proc) * 1000.0
                lat_ms = proc_ms
                risk_s = lvl.value
                raw_s = raw_demo
                now_snap = time.perf_counter()
                if now_snap - last_demo_snap_t >= 0.12:
                    last_demo_snap_t = now_snap
                    chunk = bus.all_events()[-500:]
                    emit_queue.put(
                        {
                            "type": "events_snapshot",
                            "events": [safety_event_to_json(e) for e in chunk],
                        }
                    )
            else:
                assert pipeline is not None
                try:
                    out = pipeline.process_frame(
                        frame,
                        meta.timestamp_seconds,
                        frame_index=meta.frame_index,
                    )
                except Exception:
                    logger.exception("Session worker: pipeline frame failed")
                    maybe_autosave()
                    continue
                proc_ms = (time.perf_counter() - t_proc) * 1000.0
                perf.record_inference(
                    infer_seconds=proc_ms / 1000.0,
                    queue_to_done_seconds=proc_ms / 1000.0,
                )
                events = out["events"]
                pose = out["pose"]
                smoothed = out["risk_level"]
                raw_lvl = out["raw_risk_level"]
                trig = list(pipeline.last_triggered_rules)
                for ev in events:
                    _emit_after_add(
                        bus,
                        emit_queue,
                        ev,
                        debug_events=debug_events,
                        on_event_committed=on_event_committed,
                    )
                bus.tick(meta.timestamp_seconds)
                dangerous = smoothed in (SafetyLevel.HIGH, SafetyLevel.CRITICAL) or raw_lvl in (
                    "HIGH",
                    "CRITICAL",
                )
                vis = draw_live_overlay(
                    frame,
                    pose=pose,
                    risk_level=smoothed,
                    raw_risk_level=raw_lvl,
                    triggered_rules=trig,
                    dangerous=dangerous,
                )
                risk_s = smoothed.value
                raw_s = raw_lvl
                snap = perf.snapshot()
                lat_ms = snap.latency_ms

            ok, enc = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                buf = enc.tobytes()
                with jpeg_lock:
                    latest_jpeg[0] = buf

            perf.record_frame_processing(time.perf_counter() - t_frame)
            processed_frames += 1
            last_media_ts = float(meta.timestamp_seconds)
            prog = _session_progress(
                session_metadata,
                processed_frames=processed_frames,
                media_ts=last_media_ts,
            )
            with metrics_lock:
                session_metadata["processed_frames"] = int(processed_frames)
                session_metadata["progress_percent"] = prog
            update_metrics(
                frame_index=int(meta.frame_index),
                fps=float(ema_fps),
                latency_ms=float(lat_ms),
                risk_level=str(risk_s),
                raw_risk_level=str(raw_s),
                media_timestamp_seconds=float(meta.timestamp_seconds),
            )
            if (
                on_timeline_sample is not None
                and processed_frames % max(1, int(timeline_stride_frames)) == 0
            ):
                try:
                    on_timeline_sample(
                        frame_index=int(meta.frame_index),
                        timestamp=float(meta.timestamp_seconds),
                        risk_level=str(risk_s),
                        raw_risk_level=str(raw_s),
                        latency_ms=float(lat_ms),
                        fps=float(src.fps or 30.0),
                    )
                except Exception:
                    logger.exception("on_timeline_sample failed")
            maybe_autosave()
    finally:
        with metrics_lock:
            if session_metadata.get("ended_at") is None:
                session_metadata["ended_at"] = _iso_now()
            session_metadata["processed_frames"] = int(processed_frames)
            session_metadata["progress_percent"] = _session_progress(
                session_metadata,
                processed_frames=processed_frames,
                media_ts=last_media_ts,
            )
        try:
            persist_session_artifacts(
                bus,
                session_events_path=session_events_path,
                export_json_path=export_json_path,
                export_csv_path=export_csv_path,
                session_metadata_path=session_metadata_path,
                session_metadata=session_metadata,
            )
        except Exception:
            logger.exception("Session worker: flush on exit failed")
        src.close()
        logger.info("Session worker finished for %s", path)
        if on_worker_finished is not None:
            try:
                on_worker_finished()
            except Exception:
                logger.exception("on_worker_finished callback failed")


__all__ = ["persist_session_artifacts", "run_session_worker"]
