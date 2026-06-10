"""Reusable in-memory and on-disk test data (no large assets, no network)."""

from tests.fixtures.mvp_runs import write_minimal_pipeline_run, write_mvp_qa_passing_run
from tests.fixtures.synthetic import (
    make_frame_risk_for_event_merge,
    make_interpretable_risk_feature_frame,
    make_keypoint_long_format_tiny,
    make_temporal_feature_input_small,
)
from tests.fixtures.synthetic_video import write_tiny_mp4


__all__ = [
    "make_frame_risk_for_event_merge",
    "make_interpretable_risk_feature_frame",
    "make_keypoint_long_format_tiny",
    "make_temporal_feature_input_small",
    "write_minimal_pipeline_run",
    "write_mvp_qa_passing_run",
    "write_tiny_mp4",
]
