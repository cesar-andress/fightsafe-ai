"""Unit tests for research framework pure helpers (no video, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.action.defense import guard_open_proxy
from fightsafe_ai.action.punch_kick import strike_energy_proxy
from fightsafe_ai.action.temporal_classifier import majority_vote
from fightsafe_ai.anomaly.fall_detector import fall_likelihood_from_y_coords
from fightsafe_ai.anomaly.inactivity_detector import inactivity_score
from fightsafe_ai.config.framework import (
    DEFAULT_FRAMEWORK,
    load_framework_config,
    pose_backend_name,
)
from fightsafe_ai.datasets.coco_loader import list_image_ids, load_coco_annotations_dict
from fightsafe_ai.datasets.registry import get_spec
from fightsafe_ai.evaluation.ablation import AblationRow, sort_rows_by_metric
from fightsafe_ai.evaluation.event_metrics import EventWindow, temporal_iou
from fightsafe_ai.evaluation.metrics import precision_recall_f1
from fightsafe_ai.risk.fusion import fuse_weighted_mean
from fightsafe_ai.risk.levels import RiskLevelName, parse_risk_level
from fightsafe_ai.tracking.identity import format_track_label
from fightsafe_ai.tracking.sports_tracker import SportsTracker


pytestmark = pytest.mark.unit


def test_fall_likelihood() -> None:
    assert fall_likelihood_from_y_coords(0.5, 0.5, ground_y=0.82) == 0.0
    assert fall_likelihood_from_y_coords(0.9, None, ground_y=0.82) > 0.0


def test_inactivity() -> None:
    assert inactivity_score([0.0, 0.0, 0.0], threshold=0.02) == 1.0
    assert inactivity_score([1.0, 1.0], threshold=0.02) == 0.0


def test_strike_and_guard() -> None:
    assert 0.0 <= strike_energy_proxy(1.0, 0.5) <= 1.0
    assert guard_open_proxy(0.3, 0.6) > 0.0


def test_majority_vote() -> None:
    assert majority_vote(["a", "a", "b"]) == "a"
    assert majority_vote([]) is None


def test_fusion() -> None:
    s = fuse_weighted_mean({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})
    assert abs(s - 0.5) < 1e-6


def test_risk_levels() -> None:
    assert parse_risk_level("HIGH") == RiskLevelName.HIGH
    assert parse_risk_level("nope") is None


def test_metrics() -> None:
    p, r, f1 = precision_recall_f1([1, 0, 1], [1, 1, 0], positive_label=1)
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0 and 0.0 <= f1 <= 1.0


def test_temporal_iou() -> None:
    a = EventWindow(0.0, 2.0)
    b = EventWindow(1.0, 3.0)
    # Overlap [1,2] length 1; union 3 -> IoU 1/3
    assert abs(temporal_iou(a, b) - 1.0 / 3.0) < 1e-6


def test_ablation_sort() -> None:
    rows = [
        AblationRow("a", metrics={"f1": 0.2}),
        AblationRow("b", metrics={"f1": 0.8}),
    ]
    s = sort_rows_by_metric(rows, "f1")
    assert s[0].name == "b"


def test_coco_loader(tmp_path: Path) -> None:
    p = tmp_path / "tiny.json"
    p.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}]}),
        encoding="utf-8",
    )
    d = load_coco_annotations_dict(p)
    assert list_image_ids(d) == [1]
    assert load_coco_annotations_dict(tmp_path / "missing.json") == {}


def test_dataset_registry() -> None:
    s = get_spec("synthetic_mvp")
    assert s is not None and s.task_type == "combat_safety_synthetic"
    assert s.supported_loader == "tests.fixtures"


def test_framework_defaults() -> None:
    assert DEFAULT_FRAMEWORK["pose"]["backend"] == "mediapipe"
    assert pose_backend_name({}) == "mediapipe"
    cfg = load_framework_config(path=Path("/nonexistent/does_not_exist.yaml"))
    assert cfg["pose"]["backend"] == "mediapipe"


def test_load_framework_config_merges_existing_yaml(tmp_path: Path) -> None:
    y = tmp_path / "fw.yaml"
    y.write_text(
        "pose:\n  backend: mock\ntracking:\n  enabled: true\n",
        encoding="utf-8",
    )
    merged = load_framework_config(path=y)
    assert merged["pose"]["backend"] == "mock"
    assert merged["tracking"]["enabled"] is True
    assert merged["anomaly"]["fall_detection"] is True
    assert merged["llm"]["explainability"] == "optional"


def test_pose_backend_name_branches() -> None:
    assert pose_backend_name({"pose": {"backend": "yolo"}}) == "yolo"
    assert pose_backend_name({"pose": {}}) == "mediapipe"
    assert pose_backend_name({"pose": "not-a-dict"}) == "mediapipe"
    assert pose_backend_name(None) == "mediapipe"


def test_pose_init_kwargs_for_backend() -> None:
    from fightsafe_ai.config.framework import pose_init_kwargs_for_backend

    fw = {
        "pose": {
            "device": "cuda:0",
            "yolo": {"model": "x.pt"},
            "rtmpose": {"pose2d": "rtmpose-s_xxx"},
        }
    }
    assert pose_init_kwargs_for_backend(fw, "yolo") == {"device": "cuda:0", "model_name": "x.pt"}
    assert pose_init_kwargs_for_backend(fw, "rtmpose") == {
        "device": "cuda:0",
        "pose2d": "rtmpose-s_xxx",
    }
    assert pose_init_kwargs_for_backend(fw, "mediapipe") == {}


def test_create_pose_yolo_routes_to_yolo_backend() -> None:
    from fightsafe_ai.pose.backends.yolo_pose_backend import YOLOPoseBackend
    from fightsafe_ai.pose.factory import create_pose_estimator

    be = create_pose_estimator("yolo_pose")
    assert isinstance(be, YOLOPoseBackend)


def test_tracking() -> None:
    tr = SportsTracker(iou_threshold=0.1)
    t1 = tr.update(0, [{"box_xyxy": [0, 0, 10, 10]}])
    t2 = tr.update(1, [{"box_xyxy": [0.5, 0.5, 10.5, 10.5]}])
    assert len(t1) == 1 and len(t2) == 1
    assert format_track_label(0) == "Fighter 1"
