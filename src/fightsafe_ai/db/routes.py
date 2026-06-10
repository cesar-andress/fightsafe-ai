"""FastAPI router for optional PostgreSQL-backed resources."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fightsafe_ai.db import repositories
from fightsafe_ai.db.schemas import (
    EvaluationOut,
    EventOut,
    ProjectCreate,
    ProjectOut,
    RunOut,
    TimelinePointOut,
    VideoOut,
)


logger = logging.getLogger(__name__)


def _require_db(app: Any) -> tuple[Any, Any]:
    engine = getattr(app.state, "db_engine", None)
    session_factory = getattr(app.state, "db_session_factory", None)
    if engine is None or session_factory is None:
        return None, None
    return engine, session_factory


def register_db_routes(app: Any) -> None:
    """Attach routes to ``app`` (mutates in place). Idempotent if called twice."""
    if getattr(app.state, "_db_routes_registered", False):
        return

    try:
        from fastapi import Depends, HTTPException
        from sqlalchemy.orm import Session
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FastAPI and SQLAlchemy required for DB routes") from exc

    def get_sess() -> Session:
        _, session_factory = _require_db(app)
        if session_factory is None:
            raise HTTPException(status_code=503, detail="database disabled")
        s: Session = session_factory()
        try:
            yield s
        finally:
            s.close()

    db_session_dep = Annotated[Session, Depends(get_sess)]

    @app.get("/db/health")
    def db_health() -> dict[str, Any]:
        engine, _ = _require_db(app)
        if engine is None:
            return {"ok": False, "database": "disabled"}
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"ok": True, "database": "connected"}
        except Exception as exc:
            logger.warning("DB health check failed: %s", exc)
            return {"ok": False, "database": "error", "detail": str(exc)[:200]}

    @app.get("/projects", response_model=list[ProjectOut])
    def list_projects(sess: db_session_dep) -> list[Any]:
        return repositories.list_projects(sess)

    @app.post("/projects", response_model=ProjectOut)
    def create_project(body: ProjectCreate, sess: db_session_dep) -> Any:
        from fightsafe_ai.db import models

        p = models.Project(
            name=body.name.strip(),
            description=body.description,
            created_at=repositories.utc_now(),
        )
        sess.add(p)
        sess.commit()
        sess.refresh(p)
        return p

    @app.get("/videos", response_model=list[VideoOut])
    def list_videos(sess: db_session_dep, project_id: int | None = None) -> list[Any]:
        return repositories.list_videos(sess, project_id=project_id)

    @app.get("/runs", response_model=list[RunOut])
    def list_runs(sess: db_session_dep, limit: int = 200) -> list[Any]:
        return repositories.list_runs(sess, limit=min(limit, 500))

    @app.get("/runs/{run_id}", response_model=RunOut)
    def get_run(run_id: int, sess: db_session_dep) -> Any:
        r = repositories.get_run(sess, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail="run not found")
        return r

    @app.get("/runs/{run_id}/events", response_model=list[EventOut])
    def run_events(run_id: int, sess: db_session_dep) -> list[Any]:
        if repositories.get_run(sess, run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return repositories.list_events_for_run(sess, run_id)

    @app.get("/runs/{run_id}/timeline", response_model=list[TimelinePointOut])
    def run_timeline(run_id: int, sess: db_session_dep, limit: int = 50_000) -> list[Any]:
        if repositories.get_run(sess, run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return repositories.list_timeline_for_run(sess, run_id, limit=min(limit, 100_000))

    @app.get("/runs/{run_id}/evaluations", response_model=list[EvaluationOut])
    def run_evaluations(run_id: int, sess: db_session_dep) -> list[Any]:
        if repositories.get_run(sess, run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return repositories.list_evaluations_for_run(sess, run_id)

    app.state._db_routes_registered = True


__all__ = ["register_db_routes"]
