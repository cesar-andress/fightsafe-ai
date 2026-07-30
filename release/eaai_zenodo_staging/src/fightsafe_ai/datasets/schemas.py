"""
Schema types for dataset **metadata** and for **per-sample** annotations (no I/O here).

All datasets are expected to be **obtained manually** by the researcher; FightSafe only
stores registry metadata, not bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Registry: dataset-level metadata (what appears in BUILTIN_REGISTRY)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """
    One entry in the dataset registry. Describes *where* a dataset fits and *how* to load
    it once on disk—**not** a download or license grant.

    Fields
    ------
    name
        Human-readable name.
    task_type
        High-level task, e.g. ``pose_estimation``, ``action_recognition``, ``risk_event``.
    license
        Short license summary; full text is always on the upstream source.
    source_url
        Official project / paper / catalog link (may be empty if private or TBD). **Not** fetched
        by FightSafe.
    annotation_format
        e.g. ``coco_person_json``, ``yolo_pose_labels``, ``fightsafe_csv``, ``custom``.
    supported_loader
        Module or convention name, e.g. ``coco_loader``, ``yolo_pose_loader``, ``manual_only``.
    notes
        Free text: splits, keypoint count, sport, and **how** to align with FightSafe pipelines.
    registry_key
        Optional stable id used as dict key; defaults to a slug derived from ``name`` if needed.
    local_root_hint
        Optional path the researcher may set in their own config (never committed).
    """

    name: str
    task_type: str
    license: str
    source_url: str
    annotation_format: str
    supported_loader: str
    notes: str = ""
    registry_key: str = ""
    local_root_hint: Path | None = None

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("DatasetMetadata.name must be non-empty.")


# Backward compatibility with earlier FightSafe code.
DatasetSpec = DatasetMetadata

# ---------------------------------------------------------------------------
# Sample-level: keypoints, actions, risk (for future dataloaders / eval)
# ---------------------------------------------------------------------------


class KeypointSource(StrEnum):
    """Provenance of keypoint columns (string values are stable for JSON)."""

    MEDIAPIPE_BLAZEPOSE_33 = "mediapipe_blazepose_33"
    COCO_17 = "coco_17"
    YOLO_POSE = "yolo_pose"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class FighterKeypointsSample:
    """
    A single time slice of pose for one fighter (normalized or pixel coords per project).

    Used for future Parquet/JSON loaders and for evaluation batching.
    """

    sample_id: str
    frame_index: int
    fighter_id: int
    keypoints: dict[str, tuple[float, float, float]]
    """Landmark name -> (x, y, visibility or confidence). Coordinates are project-defined."""
    keypoint_source: KeypointSource | str = KeypointSource.CUSTOM
    image_path: Path | None = None
    video_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionAnnotation:
    """
    A temporal **action** label (strike, block, clinch, etc.) on a clip or frame range.

    Does not encode scoring or legality; research / supervision only.
    """

    action_class: str
    t_start_s: float
    t_end_s: float
    confidence: float = 1.0
    source: Literal["human", "model", "weak"] = "human"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_end_s < self.t_start_s:
            raise ValueError("ActionAnnotation: t_end_s must be >= t_start_s.")


@dataclass(frozen=True, slots=True)
class RiskEventAnnotation:
    """
    A **reference** risk interval for calibration or evaluation (not live officiating output).

    Aligns with FightSafe *levels* as strings; timestamps are in seconds or frame indices
    via ``time_reference``.
    """

    risk_level: str
    t_start: float
    t_end: float
    time_reference: Literal["seconds", "frame_index"] = "seconds"
    reason_tags: tuple[str, ...] = ()
    review_priority: str = "unspecified"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError("RiskEventAnnotation: t_end must be >= t_start.")


__all__ = [
    "ActionAnnotation",
    "DatasetMetadata",
    "DatasetSpec",
    "FighterKeypointsSample",
    "KeypointSource",
    "RiskEventAnnotation",
]
