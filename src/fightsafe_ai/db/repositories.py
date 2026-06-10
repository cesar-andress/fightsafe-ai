"""Repository helpers for dashboard and evaluation persistence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from fightsafe_ai.db import models


def utc_now() -> datetime:
    return datetime.now(UTC)


def git_head_short(repo_root: Path | None = None) -> str | None:
    try:
        args = ["git", "rev-parse", "HEAD"]
        r = subprocess.run(  # noqa: S603
            args,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return r.stdout.strip()[:64]
    except (OSError, subprocess.TimeoutExpired):
        return None


def file_hash_partial(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if n >= max_bytes:
                break
    return h.hexdigest()


def ensure_default_project(sess: Session, name: str = "FightSafe Live") -> models.Project:
    q = select(models.Project).where(models.Project.name == name)
    row = sess.execute(q).scalar_one_or_none()
    if row is not None:
        return row
    p = models.Project(
        name=name, description="Auto-created for dashboard sessions", created_at=utc_now()
    )
    sess.add(p)
    sess.flush()
    return p


def create_video_for_source(
    sess: Session,
    *,
    project_id: int,
    source_path: Path,
    fps: float | None,
    width: int | None,
    height: int | None,
    duration_seconds: float | None,
    frame_count: int | None,
) -> models.Video:
    path = source_path.expanduser().resolve()
    v = models.Video(
        project_id=project_id,
        filename=path.name,
        path=str(path),
        source_type="file",
        fps=float(fps) if fps is not None else None,
        width=width,
        height=height,
        duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
        frame_count=frame_count,
        file_hash=file_hash_partial(path) if path.is_file() else None,
        created_at=utc_now(),
    )
    sess.add(v)
    sess.flush()
    return v


def create_run(
    sess: Session,
    *,
    project_id: int,
    video_id: int,
    mode: str,
    status: str,
    config: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> models.Run:
    r = models.Run(
        project_id=project_id,
        video_id=video_id,
        mode=mode,
        status=status,
        started_at=utc_now(),
        ended_at=None,
        git_commit=git_head_short(),
        config_json=config,
        output_dir=output_dir,
    )
    sess.add(r)
    sess.flush()
    return r


def set_run_parameters(sess: Session, run_id: int, params: dict[str, Any]) -> None:
    for k, v in params.items():
        key_s = str(k)[:128]
        val_s = json.dumps(v) if not isinstance(v, str) else v
        vt = type(v).__name__
        if isinstance(v, bool):
            vt = "bool"
        elif isinstance(v, int):
            vt = "int"
        elif isinstance(v, float):
            vt = "float"
        q = select(models.RunParameter).where(
            models.RunParameter.run_id == run_id,
            models.RunParameter.key == key_s,
        )
        existing = sess.execute(q).scalar_one_or_none()
        if existing is not None:
            existing.value = val_s[:50_000]
            existing.value_type = vt[:32]
        else:
            sess.add(
                models.RunParameter(
                    run_id=run_id,
                    key=key_s,
                    value=val_s[:50_000],
                    value_type=vt[:32],
                )
            )


def upsert_event_for_run(
    sess: Session,
    run_id: int,
    *,
    event_type: str,
    category: str,
    level: str,
    title: str,
    description: str,
    start_time: float | None,
    end_time: float | None,
    start_frame: int | None,
    end_frame: int | None,
    score: float | None,
    event_key: str | None,
    payload_json: dict[str, Any] | None,
) -> models.Event:
    if event_key:
        q = select(models.Event).where(
            models.Event.run_id == run_id,
            models.Event.event_key == event_key,
        )
        existing = sess.execute(q).scalar_one_or_none()
        if existing is not None:
            existing.event_type = event_type
            existing.category = category
            existing.level = level
            existing.title = title
            existing.description = description
            existing.start_time = start_time
            existing.end_time = end_time
            existing.start_frame = start_frame
            existing.end_frame = end_frame
            existing.score = score
            existing.payload_json = payload_json
            sess.flush()
            return existing

    ev = models.Event(
        run_id=run_id,
        event_type=event_type,
        category=category,
        level=level,
        title=title,
        description=description,
        start_time=start_time,
        end_time=end_time,
        start_frame=start_frame,
        end_frame=end_frame,
        score=score,
        event_key=event_key,
        payload_json=payload_json,
    )
    sess.add(ev)
    sess.flush()
    return ev


def add_timeline_point(
    sess: Session,
    run_id: int,
    *,
    timestamp: float,
    frame_index: int,
    risk_score: float | None,
    risk_level: str | None,
    payload_json: dict[str, Any] | None,
) -> None:
    sess.add(
        models.TimelinePoint(
            run_id=run_id,
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            risk_score=risk_score,
            risk_level=risk_level,
            payload_json=payload_json,
        )
    )


def add_feedback(
    sess: Session,
    *,
    event_db_id: int,
    feedback_type: str,
    note: str | None,
    payload_json: dict[str, Any] | None,
) -> models.Feedback:
    fb = models.Feedback(
        event_id=event_db_id,
        feedback_type=feedback_type,
        note=note,
        created_at=utc_now(),
        payload_json=payload_json,
    )
    sess.add(fb)
    sess.flush()
    return fb


def add_artifact(sess: Session, run_id: int, *, artifact_type: str, path: str) -> models.Artifact:
    a = models.Artifact(
        run_id=run_id,
        artifact_type=artifact_type,
        path=str(path),
        created_at=utc_now(),
    )
    sess.add(a)
    sess.flush()
    return a


def list_projects(sess: Session) -> list[models.Project]:
    return list(sess.execute(select(models.Project).order_by(models.Project.id)).scalars())


def list_videos(sess: Session, *, project_id: int | None = None) -> list[models.Video]:
    q = select(models.Video).order_by(models.Video.id.desc())
    if project_id is not None:
        q = q.where(models.Video.project_id == project_id)
    return list(sess.execute(q).scalars())


def list_runs(sess: Session, *, limit: int = 200) -> list[models.Run]:
    q = select(models.Run).order_by(models.Run.started_at.desc()).limit(limit)
    return list(sess.execute(q).scalars())


def get_run(sess: Session, run_id: int) -> models.Run | None:
    return sess.get(models.Run, run_id)


def list_events_for_run(sess: Session, run_id: int) -> list[models.Event]:
    q = (
        select(models.Event)
        .where(models.Event.run_id == run_id)
        .order_by(models.Event.start_time.asc().nullsfirst(), models.Event.id.asc())
    )
    return list(sess.execute(q).scalars())


def list_timeline_for_run(
    sess: Session, run_id: int, *, limit: int = 50_000
) -> list[models.TimelinePoint]:
    q = (
        select(models.TimelinePoint)
        .where(models.TimelinePoint.run_id == run_id)
        .order_by(models.TimelinePoint.frame_index.asc())
        .limit(limit)
    )
    return list(sess.execute(q).scalars())


def list_evaluations_for_run(sess: Session, run_id: int) -> list[models.Evaluation]:
    q = select(models.Evaluation).where(models.Evaluation.run_id == run_id)
    return list(sess.execute(q).scalars())


def finish_run(sess: Session, run_id: int, *, status: str = "completed") -> None:
    r = sess.get(models.Run, run_id)
    if r is None:
        return
    r.status = status
    r.ended_at = utc_now()


__all__ = [
    "add_artifact",
    "add_feedback",
    "add_timeline_point",
    "create_run",
    "create_video_for_source",
    "ensure_default_project",
    "finish_run",
    "get_run",
    "list_evaluations_for_run",
    "list_events_for_run",
    "list_projects",
    "list_runs",
    "list_timeline_for_run",
    "list_videos",
    "set_run_parameters",
    "upsert_event_for_run",
    "utc_now",
]
