"""Thread-safe dashboard recording into the optional database (one session per call)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fightsafe_ai.api.serialization import safety_event_to_json
from fightsafe_ai.db import repositories
from fightsafe_ai.db.session import session_scope


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class LiveRunRecorder:
    """
    Persists live dashboard runs when a SQLAlchemy engine is configured.

    Maps ``SafetyEvent.event_id`` strings to database ``events.id`` for feedback FK linkage.
    """

    __slots__ = (
        "_engine",
        "_event_map",
        "_last_timeline_frame",
        "_run_id",
        "_session_factory",
        "_timeline_stride",
    )

    def __init__(
        self,
        engine: Engine,
        session_factory: sessionmaker[Session],
        *,
        run_id: int,
        timeline_stride_frames: int = 1,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._run_id = run_id
        self._event_map: dict[str, int] = {}
        self._timeline_stride = max(1, int(timeline_stride_frames))
        self._last_timeline_frame = -1

    @property
    def run_id(self) -> int:
        return self._run_id

    def remember_event_mapping(self, external_id: str, db_event_pk: int) -> None:
        if external_id:
            self._event_map[external_id] = db_event_pk

    def db_event_pk_for(self, external_event_id: str) -> int | None:
        return self._event_map.get(external_event_id)

    def on_safety_event(self, ev: Any) -> None:
        from fightsafe_ai.live.event_bus import SafetyEvent

        if not isinstance(ev, SafetyEvent):
            return
        snap = safety_event_to_json(ev)
        key = str(ev.event_id or "")
        try:
            with session_scope(self._session_factory) as sess:
                row = repositories.upsert_event_for_run(
                    sess,
                    self._run_id,
                    event_type=str(ev.event_type),
                    category=str(ev.category.value),
                    level=str(ev.level.value),
                    title=str(ev.title or "")[:10_000],
                    description=str(ev.description or "")[:50_000],
                    start_time=float(ev.start_time),
                    end_time=float(ev.end_time),
                    start_frame=None,
                    end_frame=None,
                    score=float(ev.score),
                    event_key=key or None,
                    payload_json=snap,
                )
                if key:
                    self.remember_event_mapping(key, int(row.id))
        except Exception:
            logger.exception("DB persist event failed")

    def on_timeline_tick(
        self,
        *,
        frame_index: int,
        timestamp: float,
        risk_level: str | None,
        raw_risk_level: str | None,
        latency_ms: float | None,
        fps: float | None,
    ) -> None:
        fi = int(frame_index)
        if fi % self._timeline_stride != 0 and fi != self._last_timeline_frame:
            return
        self._last_timeline_frame = fi
        rs: float | None = None
        rl = (risk_level or "").strip().upper()
        order = ("INFO", "WARNING", "HIGH", "CRITICAL")
        if rl in order:
            rs = order.index(rl) / 3.0
        payload = {
            "latency_ms": latency_ms,
            "raw_risk_level": raw_risk_level,
            "fps_assumed": fps,
        }
        try:
            with session_scope(self._session_factory) as sess:
                repositories.add_timeline_point(
                    sess,
                    self._run_id,
                    timestamp=float(timestamp),
                    frame_index=fi,
                    risk_score=rs,
                    risk_level=(risk_level or raw_risk_level),
                    payload_json=payload,
                )
        except Exception:
            logger.exception("DB timeline insert failed")

    def on_feedback(
        self,
        *,
        external_event_id: str,
        feedback_type: str,
        note: str | None,
        payload: dict[str, Any] | None,
    ) -> None:
        pk = self.db_event_pk_for(external_event_id)
        if pk is None:
            logger.warning("DB feedback skipped: unknown event_id %s", external_event_id)
            return
        try:
            with session_scope(self._session_factory) as sess:
                repositories.add_feedback(
                    sess,
                    event_db_id=pk,
                    feedback_type=feedback_type,
                    note=note,
                    payload_json=payload,
                )
        except Exception:
            logger.exception("DB feedback insert failed")

    def on_export_paths(self, *, json_path: str | None, csv_path: str | None) -> None:
        try:
            with session_scope(self._session_factory) as sess:
                if json_path:
                    repositories.add_artifact(
                        sess, self._run_id, artifact_type="events_json", path=json_path
                    )
                if csv_path:
                    repositories.add_artifact(
                        sess, self._run_id, artifact_type="events_csv", path=csv_path
                    )
        except Exception:
            logger.exception("DB artifact insert failed")

    def finalize(self, *, status: str = "completed") -> None:
        try:
            with session_scope(self._session_factory) as sess:
                repositories.finish_run(sess, self._run_id, status=status)
        except Exception:
            logger.exception("DB finalize run failed")


def create_live_run(
    engine: Engine,
    session_factory: sessionmaker[Session],
    *,
    source: Path,
    bootstrap_config: dict[str, Any],
    demo_events: bool,
    realtime: bool,
) -> LiveRunRecorder | None:
    """Insert project/video/run rows and return a recorder. Returns ``None`` on failure."""
    try:
        with session_scope(session_factory) as sess:
            proj = repositories.ensure_default_project(sess)
            # Metadata filled later by worker; probe file minimally
            dur: float | None = None
            tf: int | None = None
            fps_guess: float | None = None
            w: int | None = None
            h: int | None = None
            try:
                import cv2

                cap = cv2.VideoCapture(str(source.expanduser().resolve()))
                if cap.isOpened():
                    fc = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fpsi = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                    if fpsi > 1e-3 and fc > 0:
                        dur = fc / fpsi
                        tf = int(fc)
                    fps_guess = fpsi if fpsi > 1e-3 else None
                cap.release()
            except Exception:
                logger.debug("Video probe skipped for DB metadata", exc_info=True)

            vid = repositories.create_video_for_source(
                sess,
                project_id=int(proj.id),
                source_path=source,
                fps=fps_guess,
                width=w,
                height=h,
                duration_seconds=dur,
                frame_count=tf,
            )
            mode = "demo" if demo_events else ("realtime" if realtime else "playback")
            run = repositories.create_run(
                sess,
                project_id=int(proj.id),
                video_id=int(vid.id),
                mode=mode,
                status="running",
                config=bootstrap_config,
                output_dir=None,
            )
            repositories.set_run_parameters(sess, int(run.id), bootstrap_config)
        return LiveRunRecorder(engine, session_factory, run_id=int(run.id))
    except Exception:
        logger.exception("create_live_run failed")
        return None


__all__ = ["LiveRunRecorder", "create_live_run"]
