"""Dataset metadata registry and local-file helpers (no auto-downloads)."""

from fightsafe_ai.datasets.boxingvi import (
    BoxingVIDataset,
    BoxingVIEvent,
    inspect_dataset,
    load_events_from_xlsx,
)
from fightsafe_ai.datasets.coco_loader import (
    list_image_ids,
    load_coco_annotations_dict,
    looks_like_coco_json,
    n_person_annotations,
)
from fightsafe_ai.datasets.registry import (
    BUILTIN_REGISTRY,
    get_spec,
    list_registry_keys,
    register_dataset,
)
from fightsafe_ai.datasets.schemas import (
    ActionAnnotation,
    DatasetMetadata,
    DatasetSpec,
    FighterKeypointsSample,
    KeypointSource,
    RiskEventAnnotation,
)
from fightsafe_ai.datasets.yolo_pose_loader import (
    YOLOKeypointFrame,
    as_plain_dict,
    describe_yolo_pose_label_convention,
    n_keypoints_to_blazepose_hint,
)


__all__ = [
    "BUILTIN_REGISTRY",
    "ActionAnnotation",
    "BoxingVIDataset",
    "BoxingVIEvent",
    "DatasetMetadata",
    "DatasetSpec",
    "FighterKeypointsSample",
    "KeypointSource",
    "RiskEventAnnotation",
    "YOLOKeypointFrame",
    "as_plain_dict",
    "describe_yolo_pose_label_convention",
    "get_spec",
    "inspect_dataset",
    "list_image_ids",
    "list_registry_keys",
    "load_coco_annotations_dict",
    "load_events_from_xlsx",
    "looks_like_coco_json",
    "n_keypoints_to_blazepose_hint",
    "n_person_annotations",
    "register_dataset",
]
