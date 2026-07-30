"""
TapKO **manual annotation** schema — submission signals & extreme vulnerability (paper2 track).

Times are **seconds**, non-negative, ``end_time > start_time``, consistent with pipeline timestamps.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


TAPKO_ANNOTATION_FORMAT_VERSION: str = "1.0"


class TapkoEventType(StrEnum):
    """Closed vocabulary for TapKO labels (dot-separated namespaces)."""

    SUBMISSION_HAND_TAP = "submission_signal.hand_tap"
    SUBMISSION_FOOT_TAP = "submission_signal.foot_tap"
    SUBMISSION_VERBAL_TAP = "submission_signal.verbal_tap"
    SUBMISSION_TECHNICAL_SUBMISSION_CANDIDATE = "submission_signal.technical_submission_candidate"
    VULN_KO_COLLAPSE = "extreme_vulnerability.ko_collapse"
    VULN_NO_INTELLIGENT_DEFENSE = "extreme_vulnerability.no_intelligent_defense"
    VULN_POST_IMPACT_INACTIVITY = "extreme_vulnerability.post_impact_inactivity"
    VULN_CHOKE_UNCONSCIOUSNESS_CANDIDATE = "extreme_vulnerability.choke_unconsciousness_candidate"
    NEG_HAND_POSTING = "negative.hand_posting"
    NEG_NORMAL_SCRAMBLE = "negative.normal_scramble"
    NEG_GRIP_FIGHTING = "negative.grip_fighting"
    NEG_CELEBRATION_SLAP = "negative.celebration_slap"
    NEG_FALL_WITHOUT_KO = "negative.fall_without_ko"


class Visibility(StrEnum):
    """Subjective visibility of the relevant body parts / action."""

    CLEAR = "clear"
    PARTIAL = "partial"
    POOR = "poor"
    UNKNOWN = "unknown"


class OcclusionLevel(StrEnum):
    """Approximate occlusion / crowd / cage obstruction."""

    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class TapkoAnnotationStatus(StrEnum):
    """Lifecycle state for an annotation document (dataset-wide QA gate)."""

    DRAFT_TRANSCRIPT_DERIVED = "draft_transcript_derived"
    DRAFT_VISUAL_REVIEW = "draft_visual_review"
    VISUALLY_CONFIRMED = "visually_confirmed"
    REJECTED = "rejected"


class TapkoAnnotation(BaseModel):
    """One contiguous labelled interval on a video."""

    video_id: str = Field(..., min_length=1, description="Stable id for the clip or session.")
    source_uri: str = Field(
        ...,
        min_length=1,
        description="URI or path-like reference to the media (file path, https URL, object store key).",
    )
    start_time: float = Field(..., description="Interval start, seconds (>= 0).")
    end_time: float = Field(..., description="Interval end, seconds (> start_time).")
    event_type: TapkoEventType
    visibility: Visibility
    occlusion_level: OcclusionLevel
    actor_id: str = Field(
        ...,
        min_length=1,
        description="Identifier for the athlete performing the action / subject of the label.",
    )
    target_id: str | None = Field(
        default=None,
        description="Optional opponent or interactee id (pair grappling / strikes).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Annotator confidence that this interval matches the event_type definition.",
    )
    notes: str | None = Field(
        default=None, description="Free-text rationale, edge cases, uncertainty."
    )
    rater_id: str = Field(..., min_length=1, description="Annotator or review batch id.")
    requires_audio: bool = Field(
        default=False,
        description="True if verbal tap or similar cues depend on synchronized audio review.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("start_time", "end_time")
    @classmethod
    def _finite_time(cls, v: float) -> float:
        if v != v or abs(v) == float("inf"):
            raise ValueError("time must be finite")
        return v

    @model_validator(mode="after")
    def _order(self) -> TapkoAnnotation:
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


def _canonical_root_projection(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only root keys accepted by :class:`TapkoAnnotationDocument` (drops stray legacy keys)."""

    out: dict[str, Any] = {
        "format_version": data.get("format_version", TAPKO_ANNOTATION_FORMAT_VERSION),
        "schema_id": data.get("schema_id", "fightsafe_ai.tapko_annotation"),
        "annotations": list(data.get("annotations") or []),
    }
    if "annotation_status" in data:
        out["annotation_status"] = data["annotation_status"]
    return out


def _legacy_bundle_to_canonical(data: dict[str, Any]) -> dict[str, Any]:
    """
    Pilot / dataset-bundle JSON: metadata at root and labelled intervals under ``events``.

    Maps each event to :class:`TapkoAnnotation` using root ``video_id`` / ``source_uri``
    when absent on the event. Document-level ``dataset_version``, ``fps``, etc. are
    folded into notes on the **first** interval to preserve provenance without duplicating.
    """

    video_id = str(data.get("video_id") or "").strip() or "unknown_video"
    source_uri = str(data.get("source_uri") or "").strip() or "unknown_source"
    rater_default = (
        str(data.get("rater_id") or data.get("annotator_id") or "legacy_import").strip()
        or "legacy_import"
    )

    meta_bits: list[str] = []
    if data.get("dataset_version") is not None:
        meta_bits.append(f"dataset_version={data['dataset_version']}")
    if data.get("annotation_status") is not None:
        meta_bits.append(f"annotation_status={data['annotation_status']}")
    if data.get("fps") is not None:
        meta_bits.append(f"fps={data['fps']}")
    if data.get("notes"):
        meta_bits.append(str(data["notes"]).strip())
    doc_meta = " | ".join(meta_bits) if meta_bits else ""

    raw_events = data.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("TapKO legacy bundle must have an 'events' array.")
    annotations: list[dict[str, Any]] = []
    for i, ev in enumerate(raw_events):
        if not isinstance(ev, dict):
            raise TypeError(
                f"TapKO legacy events[{i}] must be a JSON object, got {type(ev).__name__}."
            )
        for key in (
            "start_time",
            "end_time",
            "event_type",
            "visibility",
            "occlusion_level",
            "confidence",
        ):
            if key not in ev:
                raise ValueError(f"TapKO legacy events[{i}] missing required key {key!r}.")

        ev_notes = ev.get("notes")
        ev_id = ev.get("event_id")
        note_parts: list[str] = []
        if i == 0 and doc_meta:
            note_parts.append(doc_meta)
        if ev_notes is not None and str(ev_notes).strip():
            note_parts.append(str(ev_notes).strip())
        if ev_id is not None and str(ev_id).strip():
            note_parts.append(f"event_id={ev_id}")
        merged_notes = " | ".join(note_parts) if note_parts else None

        actor = str(ev.get("actor_id") or "unknown").strip() or "unknown"
        ann: dict[str, Any] = {
            "video_id": str(ev.get("video_id") or video_id).strip() or video_id,
            "source_uri": str(ev.get("source_uri") or source_uri).strip() or source_uri,
            "start_time": ev["start_time"],
            "end_time": ev["end_time"],
            "event_type": ev["event_type"],
            "visibility": ev["visibility"],
            "occlusion_level": ev["occlusion_level"],
            "actor_id": actor,
            "confidence": ev["confidence"],
            "rater_id": str(ev.get("rater_id") or rater_default).strip() or rater_default,
            "requires_audio": bool(ev.get("requires_audio", False)),
        }
        if "target_id" in ev:
            ann["target_id"] = ev.get("target_id")
        if merged_notes is not None:
            ann["notes"] = merged_notes

        annotations.append(ann)

    out_doc: dict[str, Any] = {
        "format_version": TAPKO_ANNOTATION_FORMAT_VERSION,
        "schema_id": "fightsafe_ai.tapko_annotation",
        "annotations": annotations,
    }
    if data.get("annotation_status") is not None:
        out_doc["annotation_status"] = data["annotation_status"]
    return out_doc


def _normalize_tapko_root_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Map known legacy shapes to the canonical document dict before Pydantic validation."""

    if "annotations" in data:
        return _canonical_root_projection(data)
    if "events" in data:
        return _legacy_bundle_to_canonical(data)
    return data


class TapkoAnnotationDocument(BaseModel):
    """
    Root JSON object for a TapKO annotation file (one file may hold many intervals).
    """

    format_version: str = Field(
        default=TAPKO_ANNOTATION_FORMAT_VERSION,
        description="TapKO schema version string.",
    )
    schema_id: str = Field(
        default="fightsafe_ai.tapko_annotation",
        description="Logical schema name for tooling.",
    )
    annotation_status: TapkoAnnotationStatus = Field(
        default=TapkoAnnotationStatus.DRAFT_VISUAL_REVIEW,
        description=(
            "QA gate for the document. Only visually_confirmed files qualify as "
            "final reference metrics (see docs)."
        ),
    )
    annotations: list[TapkoAnnotation] = Field(
        default_factory=list,
        description="Ordered list of TapKO interval labels.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("annotation_status", mode="before")
    @classmethod
    def _normalize_annotation_status(cls, v: Any) -> Any:
        """Map legacy pilot strings to current enum values."""
        if v is None or isinstance(v, TapkoAnnotationStatus):
            return v
        if not isinstance(v, str):
            return v
        s = v.strip()
        legacy = {
            "draft_needs_visual_confirmation": TapkoAnnotationStatus.DRAFT_VISUAL_REVIEW.value,
            "draft": TapkoAnnotationStatus.DRAFT_VISUAL_REVIEW.value,
        }
        return legacy.get(s, s)

    @field_validator("format_version")
    @classmethod
    def _format_version(cls, v: str) -> str:
        if v != TAPKO_ANNOTATION_FORMAT_VERSION:
            raise ValueError(
                f"format_version must be {TAPKO_ANNOTATION_FORMAT_VERSION!r}, got {v!r}."
            )
        return v


def parse_tapko_dict(data: Any) -> TapkoAnnotationDocument:
    """Validate a JSON-decoded dict as a :class:`TapkoAnnotationDocument`."""
    if not isinstance(data, dict):
        raise TypeError("TapKO annotation root must be a JSON object (dict).")
    normalized = _normalize_tapko_root_dict(data)
    return TapkoAnnotationDocument.model_validate(normalized)


def parse_tapko_json(text: str) -> TapkoAnnotationDocument:
    """Parse JSON text and validate."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise
    return parse_tapko_dict(raw)


def validate_tapko_json(text: str) -> TapkoAnnotationDocument:
    """Alias for :func:`parse_tapko_json` (explicit validation entry point)."""
    return parse_tapko_json(text)


def tapko_json_schema() -> dict[str, Any]:
    """Export JSON Schema (draft 2020-12) for tooling and editors."""
    return TapkoAnnotationDocument.model_json_schema()


# --- Example payloads (also documented in ``docs/tapko_annotation.md``) ---

EXAMPLE_DOCUMENT_MINIMAL: dict[str, Any] = {
    "format_version": TAPKO_ANNOTATION_FORMAT_VERSION,
    "schema_id": "fightsafe_ai.tapko_annotation",
    "annotation_status": TapkoAnnotationStatus.VISUALLY_CONFIRMED.value,
    "annotations": [
        {
            "video_id": "match_2026_03_clip_v3",
            "source_uri": "s3://bucket/fightsafe/clips/match_2026_03_v3.mp4",
            "start_time": 124.5,
            "end_time": 125.2,
            "event_type": "submission_signal.hand_tap",
            "visibility": "clear",
            "occlusion_level": "none",
            "actor_id": "athlete_blue_corner",
            "target_id": "athlete_red_corner",
            "confidence": 0.92,
            "notes": "Three palm strikes on mat; referee moves in same second.",
            "rater_id": "rater_07",
            "requires_audio": False,
        }
    ],
}

EXAMPLE_DOCUMENT_FULL: dict[str, Any] = {
    "format_version": TAPKO_ANNOTATION_FORMAT_VERSION,
    "schema_id": "fightsafe_ai.tapko_annotation",
    "annotation_status": TapkoAnnotationStatus.VISUALLY_CONFIRMED.value,
    "annotations": [
        {
            "video_id": "adcc_session_a_r3",
            "source_uri": "file:///data/adcc/session_a_round3.mp4",
            "start_time": 310.0,
            "end_time": 312.8,
            "event_type": "submission_signal.verbal_tap",
            "visibility": "partial",
            "occlusion_level": "light",
            "actor_id": "competitor_12",
            "target_id": "competitor_05",
            "confidence": 0.78,
            "notes": "Mouth visible; broadcast mix unclear — confirm against cage mic channel.",
            "rater_id": "rater_02",
            "requires_audio": True,
        },
        {
            "video_id": "adcc_session_a_r3",
            "source_uri": "file:///data/adcc/session_a_round3.mp4",
            "start_time": 450.0,
            "end_time": 453.5,
            "event_type": "negative.normal_scramble",
            "visibility": "clear",
            "occlusion_level": "none",
            "actor_id": "competitor_12",
            "confidence": 0.88,
            "notes": "Hard negative: rapid hand movements resemble tap but are framing during scramble.",
            "rater_id": "rater_02",
            "requires_audio": False,
        },
        {
            "video_id": "mma_card_xy_main",
            "source_uri": "https://example.org/replays/card_xy_main_cut.mp4",
            "start_time": 602.1,
            "end_time": 608.0,
            "event_type": "extreme_vulnerability.ko_collapse",
            "visibility": "poor",
            "occlusion_level": "moderate",
            "actor_id": "fighter_a",
            "target_id": None,
            "confidence": 0.65,
            "notes": "Collapse after strike sequence; cutaway partially obscures landing.",
            "rater_id": "rater_11",
            "requires_audio": False,
        },
    ],
}


__all__ = [
    "EXAMPLE_DOCUMENT_FULL",
    "EXAMPLE_DOCUMENT_MINIMAL",
    "TAPKO_ANNOTATION_FORMAT_VERSION",
    "OcclusionLevel",
    "TapkoAnnotation",
    "TapkoAnnotationDocument",
    "TapkoAnnotationStatus",
    "TapkoEventType",
    "Visibility",
    "parse_tapko_dict",
    "parse_tapko_json",
    "tapko_json_schema",
    "validate_tapko_json",
]
