"""
CLI live playback: local file as a pseudo-stream with OpenCV preview and FightSafe events.

Default path runs pose → features → risk → anomaly via :class:`~fightsafe_ai.live.live_pipeline.LivePipeline`
in a background worker queue so decoding/rendering stays responsive. Use ``--demo-events`` for synthetic alerts only.

Requires an OpenCV build with GUI support (``opencv-python``); ``opencv-python-headless``
may not provide ``cv2.imshow``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from fightsafe_ai.api.serialization import safety_event_to_json
from fightsafe_ai.live.event_bus import EventBus, EventCategory, SafetyEvent, SafetyLevel
from fightsafe_ai.live.live_overlay import draw_live_overlay
from fightsafe_ai.live.live_pipeline import LivePipeline, LivePipelineConfig
from fightsafe_ai.live.performance import (
    LivePerformanceMonitor,
    PerformanceSnapshot,
    budget_infer_stride,
    merge_infer_strides,
)
from fightsafe_ai.live.video_source import VideoSource, open_video_source
from fightsafe_ai.pose.backends.constants import RUNTIME_BACKEND_CLI_CHOICES


logger = logging.getLogger(__name__)

DEFAULT_EXPORT_JSON = Path("outputs/live/events.json")
DEFAULT_EXPORT_CSV = Path("outputs/live/events.csv")
PANEL_WIDTH = 440
LINE_HEIGHT = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HEADER = 0.5
FONT_SCALE_BODY = 0.42
FONT_SCALE_EVENT = 0.4
EVENT_TITLE_MAX_CHARS = 46


def _level_rank(level: SafetyLevel) -> int:
    return {
        SafetyLevel.INFO: 0,
        SafetyLevel.WARNING: 1,
        SafetyLevel.HIGH: 2,
        SafetyLevel.CRITICAL: 3,
    }[level]


def _demo_tick(
    bus: EventBus,
    *,
    timestamp_seconds: float,
    frame_index: int,
    periodic_visible_high: bool = False,
    debug_events: bool = False,
) -> SafetyLevel:
    """
    Placeholder logic: emit occasional demo events; returns a coarse current risk level.

    With ``periodic_visible_high`` (dashboard demo mode), emits a **HIGH** synthetic alert about
    every **3 seconds** so operators can verify event plumbing quickly.
    """
    ts = timestamp_seconds
    level = SafetyLevel.INFO

    def _dbg(ev: SafetyEvent, ok: bool) -> None:
        if debug_events and ok:
            logger.info(
                "Demo event: %s",
                json.dumps(safety_event_to_json(ev), ensure_ascii=False, default=str),
            )

    if periodic_visible_high:
        if frame_index == 0:
            _demo_tick._pv_slot = -99999  # type: ignore[attr-defined]
        slot = math.floor(ts / 3.0)
        prev_slot = getattr(_demo_tick, "_pv_slot", -99999)
        if slot != prev_slot:
            _demo_tick._pv_slot = slot  # type: ignore[attr-defined]
            ev_hi = SafetyEvent(
                event_type="demo.interval_visible_high",
                category=EventCategory.STRIKE_IMPACT,
                start_time=ts,
                end_time=ts,
                level=SafetyLevel.HIGH,
                score=0.82,
                title="Demo alert (~3s interval)",
                description="Synthetic HIGH event for dashboard / export verification.",
                explanation="Enable demo mode to validate WebSocket and event list without pose.",
                source="demo.interval",
            )
            ok_hi = bus.add_event(ev_hi)
            _dbg(ev_hi, ok_hi)

    if frame_index % 45 == 0 and frame_index > 0:
        ev_hb = SafetyEvent(
            event_type="demo.heartbeat",
            category=EventCategory.UNKNOWN,
            start_time=ts,
            end_time=ts,
            level=SafetyLevel.INFO,
            score=0.15,
            title="Demo heartbeat",
            description=f"Frame {frame_index} — placeholder live signal.",
            explanation="Synthetic INFO pulse for panel/export checks.",
            source="demo",
        )
        _dbg(ev_hb, bus.add_event(ev_hb))
    phase = (timestamp_seconds % 30.0) / 30.0
    if phase > 0.55 and frame_index % 30 == 0:
        ev_em = SafetyEvent(
            event_type="demo.elevated_motion",
            category=EventCategory.IMBALANCE,
            start_time=ts,
            end_time=ts,
            level=SafetyLevel.WARNING,
            score=0.42 + 0.1 * math.sin(timestamp_seconds),
            title="Elevated motion (demo)",
            description="Synthetic alert for UI validation.",
            explanation="WARNING-tier demo with pseudo-periodic score.",
            source="demo.heuristic",
        )
        added = bus.add_event(ev_em)
        _dbg(ev_em, added)
        if added:
            level = SafetyLevel.WARNING
    if int(timestamp_seconds) % 40 == 0 and frame_index % 15 == 0 and timestamp_seconds >= 1.0:
        ev_rc = SafetyEvent(
            event_type="demo.review_candidate",
            category=EventCategory.STRIKE_IMPACT,
            start_time=ts,
            end_time=ts,
            level=SafetyLevel.HIGH,
            score=0.78,
            title="Review candidate (demo)",
            description="Would trigger human review in production.",
            explanation="HIGH-tier demo event (not real officiating output).",
            source="demo.policy",
        )
        added = bus.add_event(ev_rc)
        _dbg(ev_rc, added)
        if added:
            level = max(level, SafetyLevel.HIGH, key=_level_rank)
    if frame_index > 0 and frame_index % 400 == 0:
        ev_cp = SafetyEvent(
            event_type="demo.critical_placeholder",
            category=EventCategory.UNKNOWN,
            start_time=ts,
            end_time=ts,
            level=SafetyLevel.CRITICAL,
            score=0.95,
            title="Critical placeholder",
            description="Not real clinical/officiating risk.",
            explanation="Forced CRITICAL sample for UI/export validation.",
            source="demo",
        )
        ok_cp = bus.add_event(ev_cp, force=True)
        _dbg(ev_cp, ok_cp)
        level = SafetyLevel.CRITICAL

    visible = bus.get_visible_events(now_seconds=timestamp_seconds, limit=10)
    recent_levels = [e.level for e in visible]
    return max([level, *recent_levels], key=_level_rank)


def _panel_semantic_bgr(level: SafetyLevel) -> tuple[int, int, int]:
    """BGR colors for panel event rows and risk line (INFO gray … CRITICAL red)."""
    return {
        SafetyLevel.INFO: (200, 200, 200),
        SafetyLevel.WARNING: (0, 255, 255),
        SafetyLevel.HIGH: (0, 165, 255),
        SafetyLevel.CRITICAL: (80, 80, 255),
    }[level]


def _smoothed_level_to_raw_label(level: SafetyLevel) -> str:
    return {
        SafetyLevel.INFO: "LOW",
        SafetyLevel.WARNING: "MEDIUM",
        SafetyLevel.HIGH: "HIGH",
        SafetyLevel.CRITICAL: "CRITICAL",
    }[level]


def _format_event_line(ev: SafetyEvent, *, max_chars: int = EVENT_TITLE_MAX_CHARS) -> str:
    body = f"{ev.title} ({ev.duration:.1f}s)"
    if len(body) <= max_chars:
        return body
    budget = max(12, max_chars - 3)
    return body[:budget] + "…"


def _draw_panel(
    *,
    height: int,
    timestamp_seconds: float,
    playback_fps: float,
    display_fps: float,
    risk_level: SafetyLevel,
    events: list[SafetyEvent],
    demo_mode: bool,
    perf: PerformanceSnapshot | None = None,
    raw_risk_level: str | None = None,
    wall_time: float = 0.0,
    nominal_display_hz: float | None = None,
    inference_fps_cap: float | None = None,
) -> np.ndarray:
    """
    Right-hand column: telemetry + last N events (color by severity, CRITICAL blink outline).

    Left column in the window is the video with skeleton (drawn separately).
    """
    panel = np.full((height, PANEL_WIDTH, 3), (22, 22, 24), dtype=np.uint8)
    y = LINE_HEIGHT
    blink_critical = (int(wall_time * 3.2) % 2) == 0

    def line(
        text: str,
        color: tuple[int, int, int] = (235, 235, 235),
        *,
        scale: float = FONT_SCALE_BODY,
        thickness: int = 1,
    ) -> None:
        nonlocal y
        cv2.putText(panel, text, (10, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += LINE_HEIGHT

    line(
        "FightSafe Live — demo" if demo_mode else "FightSafe Live",
        (245, 245, 245),
        scale=FONT_SCALE_HEADER,
        thickness=1,
    )
    line("──────── Telemetry ────────", (110, 110, 115), scale=0.38)
    line(f"Timestamp (media)  {timestamp_seconds:10.3f} s", (210, 210, 215))
    line(f"Source FPS         {playback_fps:8.2f}", (200, 200, 205))
    if perf is None:
        line(f"Display FPS (UI)   {display_fps:8.2f}", (195, 195, 200))
        line("Latency            —", (150, 150, 155))
    else:
        lat_c = (160, 210, 255) if perf.below_fps_threshold else (190, 230, 190)
        line(f"Display FPS (EMA)  {perf.fps_ema:8.2f}  (tgt {perf.target_fps:.1f} Hz)", lat_c)
        line(f"Latency (e2e)      {perf.latency_ms:8.1f} ms", (200, 225, 245))
        line(f"Inference          {perf.infer_ms:8.1f} ms", (195, 220, 240))
        line(
            f"Frame / render     {perf.frame_processing_ms:5.1f} / {perf.render_ms:4.1f} ms",
            (170, 185, 195),
        )
        if inference_fps_cap is not None and nominal_display_hz is not None:
            line(
                f"Infer / display    {inference_fps_cap:5.1f} / {nominal_display_hz:5.1f} Hz (nominal)",
                (165, 195, 210),
            )
        if perf.infer_stride > 1:
            detail = ""
            if perf.stride_budget > 1 or perf.stride_adaptive > 1:
                detail = f"  (budget {perf.stride_budget} · adaptive {perf.stride_adaptive})"
            line(
                f"Infer stride       every {perf.infer_stride} frame(s){detail}",
                (120, 200, 140),
            )

    raw_lbl = (
        raw_risk_level if raw_risk_level is not None else _smoothed_level_to_raw_label(risk_level)
    )
    rl_color = _panel_semantic_bgr(risk_level)
    line(f"Risk (smoothed)    {risk_level.value}", rl_color, scale=FONT_SCALE_HEADER)
    line(f"Raw band           {raw_lbl}", (200, 200, 205))

    y += 4
    line("──────── Events (last 10) ──", (110, 110, 115), scale=0.38)
    show = events[:10]
    if not show:
        line("— none in TTL window —", (120, 120, 125))
    else:
        ev_y = y + LINE_HEIGHT - 6
        for ev in show:
            text = _format_event_line(ev)
            col = _panel_semantic_bgr(ev.level)
            if ev.level == SafetyLevel.CRITICAL and blink_critical:
                cv2.rectangle(
                    panel,
                    (4, ev_y - 16),
                    (PANEL_WIDTH - 4, ev_y + 6),
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            cv2.putText(panel, text, (10, ev_y), FONT, FONT_SCALE_EVENT, col, 1, cv2.LINE_AA)
            ev_y += LINE_HEIGHT

    cv2.putText(
        panel,
        "q — quit",
        (10, height - 12),
        FONT,
        0.38,
        (130, 130, 135),
        1,
        cv2.LINE_AA,
    )
    return panel


def _submit_latest_only(
    work_q: queue.Queue[tuple[np.ndarray, float, int, float] | None],
    job: tuple[np.ndarray, float, int, float],
) -> None:
    """Keep queue depth 1 so inference always catches up (drops stale frames)."""
    while True:
        try:
            work_q.put_nowait(job)
            return
        except queue.Full:
            try:
                work_q.get_nowait()
            except queue.Empty:
                pass


def run_live(
    source: str | Path,
    *,
    realtime: bool,
    export_json: Path,
    export_csv: Path,
    cooldown_seconds: float,
    demo_events: bool = False,
    max_infer_hz: float = 12.0,
    buffer_seconds: float = 1.5,
    smooth_seconds: float = 1.5,
    visual_expire_seconds: float = 14.0,
    fps_threshold: float = 18.0,
    fps_recover: float = 22.0,
    max_infer_stride: int = 8,
    stride_display_hz: float | None = None,
    inference_fps: float | None = None,
    pose_backend: str = "torch",
    pose_device: str = "auto",
    pose2d: str | None = None,
    onnx_model: Path | None = None,
    pose_fp16: bool = False,
) -> None:
    bus = EventBus(cooldown_seconds=cooldown_seconds, visual_expire_seconds=visual_expire_seconds)
    src: VideoSource = open_video_source(source, realtime=realtime)

    window = "FightSafe Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    ema_fps = 30.0
    perf = LivePerformanceMonitor(
        target_fps=fps_threshold,
        fps_recover=fps_recover,
        max_infer_stride=max_infer_stride,
    )

    if demo_events:
        current_risk = SafetyLevel.INFO
        try:
            while True:
                t_frame = time.perf_counter()
                frame, meta = src.read_frame()
                if frame is None or meta is None:
                    break

                now = time.perf_counter()
                dt = max(now - prev_time, 1e-9)
                prev_time = now
                ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt)
                perf.tick_display_loop(dt_seconds=dt)

                current_risk = _demo_tick(
                    bus, timestamp_seconds=meta.timestamp_seconds, frame_index=meta.frame_index
                )
                bus.tick(meta.timestamp_seconds)

                wall_clock = time.perf_counter()
                raw_demo = _smoothed_level_to_raw_label(current_risk)
                vis_left = draw_live_overlay(
                    frame,
                    pose=None,
                    risk_level=current_risk,
                    raw_risk_level=raw_demo,
                    triggered_rules=[],
                    dangerous=current_risk in (SafetyLevel.HIGH, SafetyLevel.CRITICAL),
                )
                t_draw = time.perf_counter()
                panel = _draw_panel(
                    height=vis_left.shape[0],
                    timestamp_seconds=meta.timestamp_seconds,
                    playback_fps=meta.fps,
                    display_fps=ema_fps,
                    risk_level=current_risk,
                    events=bus.get_visible_events(now_seconds=meta.timestamp_seconds, limit=10),
                    demo_mode=True,
                    perf=perf.snapshot(),
                    raw_risk_level=raw_demo,
                    wall_time=wall_clock,
                    nominal_display_hz=None,
                    inference_fps_cap=None,
                )
                combined = np.hstack([vis_left, panel])
                cv2.imshow(window, combined)
                perf.record_render(time.perf_counter() - t_draw)
                perf.record_frame_processing(time.perf_counter() - t_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            src.close()
            cv2.destroyAllWindows()
            bus.export_json(export_json)
            bus.export_csv(export_csv)
        return

    work_q: queue.Queue[tuple[np.ndarray, float, int, float] | None] = queue.Queue(maxsize=1)
    lock = threading.Lock()
    pending_events: list[SafetyEvent] = []
    snap_pose = None
    snap_smoothed = SafetyLevel.INFO
    snap_raw = "LOW"
    snap_score = 0.0
    snap_trig: list[str] = []
    stop_worker = threading.Event()

    def worker() -> None:
        nonlocal snap_pose, snap_smoothed, snap_raw, snap_score, snap_trig
        cap_infer_hz = float(max_infer_hz)
        if inference_fps is not None and inference_fps > 0:
            cap_infer_hz = min(cap_infer_hz, float(inference_fps))
        pipeline = LivePipeline(
            LivePipelineConfig(
                pose_backend=pose_backend,
                pose_device=pose_device,
                pose_fp16=pose_fp16,
                pose2d=pose2d,
                onnx_model=onnx_model,
                video_fps=float(src.fps),
                buffer_seconds=buffer_seconds,
                smooth_seconds=smooth_seconds,
                max_infer_hz=cap_infer_hz,
            )
        )
        while not stop_worker.is_set():
            try:
                item = work_q.get(timeout=0.08)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_bgr, ts, fidx, t_submit = item
            t_infer0 = time.perf_counter()
            try:
                out = pipeline.process_frame(frame_bgr, ts, frame_index=fidx)
                evs = out["events"]
            except Exception:
                logger.exception("Live inference failed")
                continue
            infer_s = time.perf_counter() - t_infer0
            perf.record_inference(
                infer_seconds=infer_s,
                queue_to_done_seconds=time.perf_counter() - t_submit,
            )
            with lock:
                snap_pose = out["pose"]
                snap_smoothed = out["risk_level"]
                snap_raw = out["raw_risk_level"]
                snap_score = out["risk_score"]
                snap_trig = list(pipeline.last_triggered_rules)
                pending_events.extend(evs)

    t = threading.Thread(target=worker, name="fightsafe-live-infer", daemon=True)
    t.start()

    try:
        while True:
            t_frame = time.perf_counter()
            frame, meta = src.read_frame()
            if frame is None or meta is None:
                break

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-9)
            prev_time = now
            ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt)
            perf.tick_display_loop(dt_seconds=dt)

            display_hz = (
                float(stride_display_hz)
                if stride_display_hz is not None
                else float(meta.fps or src.fps or 30.0)
            )
            budget_stride = budget_infer_stride(
                display_hz=display_hz,
                inference_fps=inference_fps,
                max_stride=max_infer_stride,
            )
            snap0 = perf.snapshot()
            stride = merge_infer_strides(
                budget_stride=budget_stride,
                adaptive_stride=snap0.stride_adaptive,
                max_stride=max_infer_stride,
            )
            perf_snap = replace(
                snap0,
                infer_stride=stride,
                stride_budget=budget_stride,
            )

            if meta.frame_index % stride == 0:
                _submit_latest_only(
                    work_q,
                    (frame.copy(), meta.timestamp_seconds, meta.frame_index, time.perf_counter()),
                )

            with lock:
                to_emit = pending_events
                pending_events = []
                pose = snap_pose
                smoothed = snap_smoothed
                raw_lvl = snap_raw
                trig = list(snap_trig)

            for ev in to_emit:
                bus.add_event(ev)
            bus.tick(meta.timestamp_seconds)

            dangerous = smoothed in (SafetyLevel.HIGH, SafetyLevel.CRITICAL) or raw_lvl in (
                "HIGH",
                "CRITICAL",
            )
            wall_clock = time.perf_counter()
            vis = draw_live_overlay(
                frame,
                pose=pose,
                risk_level=smoothed,
                raw_risk_level=raw_lvl,
                triggered_rules=trig,
                dangerous=dangerous,
            )
            t_draw = time.perf_counter()

            panel = _draw_panel(
                height=vis.shape[0],
                timestamp_seconds=meta.timestamp_seconds,
                playback_fps=meta.fps,
                display_fps=ema_fps,
                risk_level=smoothed,
                events=bus.get_visible_events(now_seconds=meta.timestamp_seconds, limit=10),
                demo_mode=False,
                perf=perf_snap,
                raw_risk_level=raw_lvl,
                wall_time=wall_clock,
                nominal_display_hz=display_hz if inference_fps is not None else None,
                inference_fps_cap=inference_fps,
            )
            combined = np.hstack([vis, panel])
            cv2.imshow(window, combined)
            perf.record_render(time.perf_counter() - t_draw)
            perf.record_frame_processing(time.perf_counter() - t_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_worker.set()
        try:
            while True:
                work_q.get_nowait()
        except queue.Empty:
            pass
        try:
            work_q.put_nowait(None)
        except queue.Full:
            pass
        t.join(timeout=8.0)

        src.close()
        cv2.destroyAllWindows()
        bus.export_json(export_json)
        bus.export_csv(export_csv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live-style playback from a local video file with FightSafe AI pipeline."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path, or a webcam index (e.g. 0 for default camera).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between frames to mimic live wall-clock rate.",
    )
    parser.add_argument(
        "--demo-events",
        action="store_true",
        help="Use synthetic demo alerts instead of pose/risk/anomaly pipeline (for UI tests).",
    )
    parser.add_argument(
        "--max-infer-hz",
        type=float,
        default=12.0,
        help="Hard ceiling on inference rate inside the pipeline; capped by --inference-fps when set.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=1.5,
        help="Temporal buffer length for features/risk/anomaly windows.",
    )
    parser.add_argument(
        "--smooth-seconds",
        type=float,
        default=1.5,
        help="Rolling window for smoothing mapped risk levels.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=DEFAULT_EXPORT_JSON,
        help=f"JSON export path (default: {DEFAULT_EXPORT_JSON})",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=DEFAULT_EXPORT_CSV,
        help=f"CSV export path (default: {DEFAULT_EXPORT_CSV})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between new episodes of the same event_type after a gap.",
    )
    parser.add_argument(
        "--visual-expire",
        type=float,
        default=14.0,
        help="Hide panel events whose end_time is older than this many seconds vs. playback clock.",
    )
    parser.add_argument(
        "--fps-threshold",
        type=float,
        default=18.0,
        help="If smoothed display FPS stays below this, increase adaptive inference stride. "
        "Use --target-fps to set this and the nominal display Hz for --inference-fps in one flag.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        metavar="HZ",
        help="Display FPS budget: when set, sets the adaptive-stride target (like --fps-threshold) and "
        "is the nominal display rate for the frame-skip ratio with --inference-fps. "
        "If omitted, ratio uses the source FPS.",
    )
    parser.add_argument(
        "--inference-fps",
        type=float,
        default=None,
        metavar="HZ",
        help="Target inference rate vs display: skip inference on intermediate frames and reuse the "
        "last pose/results (effective stride ≈ ceil(display_hz / this)). Display Hz is --target-fps "
        "when set, otherwise the source FPS.",
    )
    parser.add_argument(
        "--fps-recover",
        type=float,
        default=22.0,
        help="If smoothed display FPS rises above this, decrease inference stride toward 1.",
    )
    parser.add_argument(
        "--max-infer-stride",
        type=int,
        default=8,
        help="Maximum inference stride when FPS is low (process 1 frame every N displayed).",
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        choices=RUNTIME_BACKEND_CLI_CHOICES,
        default="torch",
        help="Runtime pose backend (Torch=RTMPose stack; onnx/tensorrt are partial/stub).",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="auto",
        help="Torch device hint (auto|cpu|cuda|cuda:0|mps); ignored for onnx/tensorrt stubs.",
    )
    parser.add_argument(
        "--pose2d",
        type=str,
        default=None,
        help="Optional MMPose pose2d model id for --pose-backend torch.",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional ONNX model path for --pose-backend onnx (decode still stub without wiring).",
    )
    parser.add_argument(
        "--pose-fp16",
        action="store_true",
        help="Torch AMP FP16 on CUDA for RTMPose; ONNX float16 inputs when using --pose-backend onnx.",
    )
    args = parser.parse_args(argv)

    src_token = args.source.strip()
    if src_token.isdigit():
        resolved_source: str | Path = src_token
    else:
        sp = Path(src_token).expanduser().resolve()
        if not sp.is_file():
            print(f"Source not found: {args.source}", file=sys.stderr)
            return 1
        resolved_source = sp

    if args.inference_fps is not None and args.inference_fps <= 0.0:
        print("--inference-fps must be positive", file=sys.stderr)
        return 1

    fps_budget = args.target_fps if args.target_fps is not None else args.fps_threshold

    try:
        run_live(
            resolved_source,
            realtime=args.realtime,
            export_json=args.export_json,
            export_csv=args.export_csv,
            cooldown_seconds=args.cooldown,
            demo_events=args.demo_events,
            max_infer_hz=args.max_infer_hz,
            buffer_seconds=args.buffer_seconds,
            smooth_seconds=args.smooth_seconds,
            visual_expire_seconds=args.visual_expire,
            fps_threshold=fps_budget,
            fps_recover=args.fps_recover,
            max_infer_stride=args.max_infer_stride,
            stride_display_hz=args.target_fps,
            inference_fps=args.inference_fps,
            pose_backend=args.pose_backend,
            pose_device=args.pose_device,
            pose2d=args.pose2d,
            onnx_model=args.onnx_model,
            pose_fp16=args.pose_fp16,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
