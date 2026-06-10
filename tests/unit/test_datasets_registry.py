"""Registry and schema tests (no network, no large files)."""

from __future__ import annotations

import pytest

from fightsafe_ai.datasets.coco_loader import looks_like_coco_json
from fightsafe_ai.datasets.registry import (
    BUILTIN_REGISTRY,
    get_spec,
    list_registry_keys,
    register_dataset,
)
from fightsafe_ai.datasets.schemas import (
    ActionAnnotation,
    DatasetMetadata,
    FighterKeypointsSample,
    KeypointSource,
    RiskEventAnnotation,
)
from fightsafe_ai.datasets.yolo_pose_loader import (
    YOLOKeypointFrame,
    as_plain_dict,
    describe_yolo_pose_label_convention,
)


pytestmark = pytest.mark.unit


def test_builtin_keys_include_expected() -> None:
    keys = list_registry_keys()
    assert "synthetic_mvp" in keys
    assert "coco_keypoints_person" in keys
    assert "yolo_pose_export" in keys
    assert "boxingvi_local" in keys


def test_get_spec_coco() -> None:
    m = get_spec("coco_keypoints_person")
    assert m is not None
    assert m.annotation_format == "coco_person_json"
    assert m.supported_loader == "coco_loader"


def test_register_dataset_runtime() -> None:
    k = "_test_ephemeral"
    try:
        register_dataset(
            k,
            DatasetMetadata(
                name="T",
                task_type="test",
                license="test",
                source_url="",
                annotation_format="none",
                supported_loader="none",
                notes="ephemeral",
            ),
        )
        assert get_spec(k) is not None
    finally:
        BUILTIN_REGISTRY.pop(k, None)


def test_fighter_keypoints_sample() -> None:
    s = FighterKeypointsSample(
        sample_id="a",
        frame_index=0,
        fighter_id=0,
        keypoints={"nose": (0.5, 0.1, 1.0)},
        keypoint_source=KeypointSource.MEDIAPIPE_BLAZEPOSE_33,
    )
    assert s.keypoints["nose"][0] == 0.5


def test_action_annotation_validates_time() -> None:
    with pytest.raises(ValueError):
        ActionAnnotation("punch", 1.0, 0.0)


def test_risk_event_validates_time() -> None:
    with pytest.raises(ValueError):
        RiskEventAnnotation("HIGH", 1.0, 0.0)


def test_yolo_keypoint_frame() -> None:
    rows = as_plain_dict([YOLOKeypointFrame(0, [(0.0, 0.0), (1.0, 1.0)])])
    assert rows[0]["kpts"][0]["x"] == 0.0
    assert "Ultralytics" in describe_yolo_pose_label_convention()


def test_looks_like_coco() -> None:
    assert not looks_like_coco_json({})
    assert looks_like_coco_json({"images": [], "annotations": []})
