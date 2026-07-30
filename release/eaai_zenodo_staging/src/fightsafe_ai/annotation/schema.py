"""
Pydantic schema for **manual** safety-event ground-truth labels (FightSafe evaluation).

``start_time`` / ``end_time`` are in **seconds** on the same time base as the video / pipeline
``timestamp`` in ``risk_scores.csv`` (typically relative to t=0 of the media used for a run).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


ANNOTATION_FORMAT_VERSION: str = "1.0"


class EventType(StrEnum):
    """Label vocabulary for hand-authored evaluation segments."""

    FALL = "FALL"
    KO = "KO"
    SURRENDER = "SURRENDER"
    INSTABILITY = "INSTABILITY"


class EventAnnotation(BaseModel):
    """
    A single, contiguous manual segment tagged with a high-level event type.
    Intervals are half-open in documentation only; the file stores inclusive ``[start, end]``.
    For metrics, we treat the segment as ``[start_time, end_time]`` closed, matching ``events.json``.
    """

    event_id: str | None = Field(
        default=None,
        description="Optional stable id for cross-referencing (not used by the matcher).",
    )
    start_time: float = Field(..., description="Segment start, seconds (>= 0, finite).")
    end_time: float = Field(..., description="Segment end, seconds (strictly > start_time).")
    event_type: EventType
    confidence: float | None = Field(
        default=None,
        description="Optional annotator confidence in [0, 1].",
        ge=0.0,
        le=1.0,
    )
    notes: str | None = Field(default=None, description="Optional free text.")

    model_config = {"extra": "forbid"}

    @field_validator("start_time", "end_time")
    @classmethod
    def _finite_time(cls, v: float) -> float:
        if v != v:  # NaN
            raise ValueError("time must be finite, not NaN")
        if abs(v) == float("inf"):
            raise ValueError("time must be finite, not inf")
        return v

    @model_validator(mode="after")
    def _order(self) -> EventAnnotation:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        return self


class AnnotationDocument(BaseModel):
    """
    On-disk file wrapper (e.g. ``annotations/demo_annotations.json``).
    The ``video`` field is a **reference string** (path, URI, or id); it is not validated to exist.
    """

    format_version: str = Field(
        default=ANNOTATION_FORMAT_VERSION,
        description="File format version for the FightSafe AI annotation spec.",
    )
    case_id: str | None = Field(
        default=None,
        description="Optional narrative case key (e.g. case_a_knockdown) for batch evaluation.",
    )
    source_reference: str | None = Field(
        default=None,
        description="Optional duplicate or alternate reference to the clip (URL, path); informational.",
    )
    clip_start_time: float | None = Field(
        default=None,
        description="Optional annotation-window start on the source timeline (seconds); informational.",
    )
    clip_end_time: float | None = Field(
        default=None,
        description="Optional annotation-window end on the source timeline (seconds); informational.",
    )
    video: str = Field(
        ...,
        min_length=1,
        description="Path or id of the source video; echo of CLI --video.",
    )
    time_unit: str = Field(
        default="seconds",
        description="Must be the string 'seconds' (same as pipeline timestamps).",
    )
    events: list[EventAnnotation] = Field(
        default_factory=list,
        description="Ordered list of hand-labelled event segments (may overlap; see validation).",
    )

    model_config = {"extra": "forbid"}

    @field_validator("format_version")
    @classmethod
    def _format_version(cls, v: str) -> str:
        if v != ANNOTATION_FORMAT_VERSION:
            raise ValueError(f"format_version must be {ANNOTATION_FORMAT_VERSION!r}, got {v!r}.")
        return v

    @field_validator("time_unit")
    @classmethod
    def _time_unit(cls, v: str) -> str:
        if v != "seconds":
            raise ValueError("time_unit must be 'seconds'.")
        return v

    @model_validator(mode="after")
    def _clip_bounds(self) -> AnnotationDocument:
        a, b = self.clip_start_time, self.clip_end_time
        if a is not None and b is not None:
            if b <= a:
                raise ValueError(
                    "clip_end_time must be greater than clip_start_time when both are set."
                )
            if a < 0:
                raise ValueError("clip_start_time must be non-negative when set.")
        return self


def parse_annotation_dict(data: Any) -> AnnotationDocument:
    """Parse a JSON-decoded dict (or list coerced) into a document."""
    if not isinstance(data, dict):
        raise TypeError("annotation file root must be a JSON object (dict).")
    return AnnotationDocument.model_validate(data)


__all__ = [
    "ANNOTATION_FORMAT_VERSION",
    "AnnotationDocument",
    "EventAnnotation",
    "EventType",
    "parse_annotation_dict",
]
