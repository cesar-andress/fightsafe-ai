"""Pydantic schemas for DB-backed REST responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = None


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class VideoOut(BaseModel):
    id: int
    project_id: int
    filename: str
    path: str
    source_type: str
    fps: float | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    frame_count: int | None
    file_hash: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id: int
    project_id: int
    video_id: int
    mode: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    git_commit: str | None
    config_json: dict[str, Any] | None
    output_dir: str | None

    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: int
    run_id: int
    event_type: str
    category: str
    level: str
    title: str
    description: str
    start_time: float | None
    end_time: float | None
    start_frame: int | None
    end_frame: int | None
    score: float | None
    payload_json: dict[str, Any] | None
    event_key: str | None

    model_config = {"from_attributes": True}


class TimelinePointOut(BaseModel):
    id: int
    run_id: int
    timestamp: float
    frame_index: int
    risk_score: float | None
    risk_level: str | None
    payload_json: dict[str, Any] | None

    model_config = {"from_attributes": True}


class EvaluationOut(BaseModel):
    id: int
    run_id: int
    dataset: str
    dataset_video_id: str | None
    tp: int | None
    fp: int | None
    fn: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_latency: float | None
    tolerance_seconds: float | None
    config_json: dict[str, Any] | None

    model_config = {"from_attributes": True}


__all__ = [
    "EvaluationOut",
    "EventOut",
    "ProjectCreate",
    "ProjectOut",
    "RunOut",
    "TimelinePointOut",
    "VideoOut",
]
