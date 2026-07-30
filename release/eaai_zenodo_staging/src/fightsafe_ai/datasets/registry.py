"""
In-memory **dataset registry**: metadata only (no downloads, no license grants).

Researchers must obtain each dataset from its **source** under the terms shown in
:attr:`DatasetMetadata.license`. URLs in the registry are for **documentation**;
FightSafe does not fetch them automatically.
"""

from __future__ import annotations

from fightsafe_ai.datasets.schemas import DatasetMetadata


# ---------------------------------------------------------------------------
# Curated entries (extend in your fork or at runtime via ``register_dataset``)
# ---------------------------------------------------------------------------

BUILTIN_REGISTRY: dict[str, DatasetMetadata] = {
    "synthetic_mvp": DatasetMetadata(
        name="FightSafe synthetic MVP fixtures",
        task_type="combat_safety_synthetic",
        license="MIT (project test data generated in-repo)",
        source_url="",
        annotation_format="fightsafe_csv",
        supported_loader="tests.fixtures",
        notes="Small CSV keypoint fixtures for unit tests; not a distributable research dataset.",
        registry_key="synthetic_mvp",
    ),
    "mma_fighter_pose_estimation": DatasetMetadata(
        name="MMA fighter pose estimation (family of public datasets)",
        task_type="pose_estimation",
        license="Varies by release; verify with each dataset’s terms before use.",
        source_url="",
        annotation_format="author_specific_keypoints",
        supported_loader="custom_lab_script",
        notes=(
            "Target category for MMA-style monocular pose. Pick a specific release, then set "
            "source_url and license from that release. Map keypoints to FightSafe "
            "BlazePose-33 or your own schema in a separate script."
        ),
        registry_key="mma_fighter_pose_estimation",
    ),
    "boxing_action_generic": DatasetMetadata(
        name="Boxing / combat action recognition datasets (generic)",
        task_type="action_recognition",
        license="Varies; many are academic-only with registration.",
        source_url="",
        annotation_format="video_clips_plus_labels",
        supported_loader="custom_or_pytorch_video",
        notes=(
            "Use for strike/block clip labels. Align time codes with your sampled FPS; "
            "FightSafe does not ship action labels—import as ``ActionAnnotation`` in your tooling."
        ),
        registry_key="boxing_action_generic",
    ),
    "coco_keypoints_person": DatasetMetadata(
        name="COCO person keypoints (2017-style)",
        task_type="pose_estimation",
        license="CC BY 4.0 (COCO; confirm on https://cocodataset.org )",
        source_url="https://cocodataset.org/",
        annotation_format="coco_person_json",
        supported_loader="coco_loader",
        notes=(
            "Standard 17 keypoints per person. Not combat-specific; useful for adapter tests. "
            "Download images/annotations manually; do not commit them to Git."
        ),
        registry_key="coco_keypoints_person",
    ),
    "yolo_pose_export": DatasetMetadata(
        name="Ultralytics YOLO-pose labels or exports",
        task_type="pose_estimation",
        license="Depends on base video + your model license; Ultralytics is AGPL-3.0.",
        source_url="https://docs.ultralytics.com/",
        annotation_format="yolo_pose_labels",
        supported_loader="yolo_pose_loader",
        notes=(
            "Expect local label files or exported tensors. COCO-17 topology is common; map to "
            "BlazePose-33 outside the core library if needed."
        ),
        registry_key="yolo_pose_export",
    ),
    "custom_safety_clips": DatasetMetadata(
        name="Custom manually annotated safety clips",
        task_type="risk_event",
        license="Set by your institution or data agreement; default is not redistributable.",
        source_url="",
        annotation_format="fightsafe_events_json_or_csv",
        supported_loader="manual_only",
        notes=(
            "In-house clips with ``RiskEventAnnotation``-style rows. Keep media and labels out "
            "of version control; store paths only in local ``.env`` or lab ``config``."
        ),
        registry_key="custom_safety_clips",
    ),
    "boxingvi_local": DatasetMetadata(
        name="BoxingVI-style local tree (annotations / skeleton / optional rgb)",
        task_type="pose_estimation",
        license="Follow the licence of the BoxingVI release you obtain; FightSafe does not redistribute it.",
        source_url="",
        annotation_format="excel_intervals_class_plus_npy_skeleton",
        supported_loader="datasets.boxingvi",
        notes=(
            "Expected layout: annotations/*.xlsx (Start Frame, End Frame, Class), skeleton/*.npy "
            "(2D keypoints), optional rgb/*.mp4. See BoxingVIDataset."
        ),
        registry_key="boxingvi_local",
    ),
}


def get_spec(name: str) -> DatasetMetadata | None:
    """Return metadata for ``name`` (the dict key), or ``None``."""
    return BUILTIN_REGISTRY.get(name)


def list_registry_keys() -> list[str]:
    """Stable keys for ``BUILTIN_REGISTRY`` (sorted)."""
    return sorted(BUILTIN_REGISTRY.keys())


def register_dataset(key: str, meta: DatasetMetadata) -> None:
    """
    Register or override a dataset at runtime (e.g. in a lab ``conftest`` or script).

    Does **not** persist to disk.
    """
    if not key or not key.strip():
        raise ValueError("register_dataset: key must be non-empty.")
    BUILTIN_REGISTRY[key] = meta
