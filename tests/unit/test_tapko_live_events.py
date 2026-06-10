"""TapKO → SafetyEvent bridge for live dashboard."""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.live.tapko_live_events import (
    TAPKO_EVENT_TYPES,
    tapko_detectors_to_safety_events,
    tapko_evidence_summary,
    tapko_family_subtype,
)


pytestmark = pytest.mark.unit


def test_tapko_event_types_registry_matches_schema_count() -> None:
    assert len(TAPKO_EVENT_TYPES) == 8
    assert "submission_signal.verbal_tap" in TAPKO_EVENT_TYPES
    assert "extreme_vulnerability.choke_unconsciousness_candidate" in TAPKO_EVENT_TYPES


def test_tapko_family_subtype() -> None:
    assert tapko_family_subtype("submission_signal.hand_tap") == ("submission_signal", "hand_tap")
    assert tapko_family_subtype("extreme_vulnerability.ko_collapse") == (
        "extreme_vulnerability",
        "ko_collapse",
    )


def test_tapko_evidence_summary_vulnerability() -> None:
    s = tapko_evidence_summary(
        {
            "head_drop_score": 0.4,
            "collapse_score": 0.9,
            "post_impact_context": True,
        },
        event_type="extreme_vulnerability.ko_collapse",
        candidate_level="moderate",
    )
    assert "candidate_level=moderate" in s
    assert "head_drop=" in s
    assert "post_impact_ctx=True" in s


_L_WR, _R_WR = 9, 10


def _base_pose_row(
    *, wrist_y: float, shoulder_y: float = 0.34, hip_y: float = 0.54, foot_y: float = 0.91
) -> np.ndarray:
    xy = np.zeros((17, 2), dtype=np.float64)
    x_center = 0.52
    for ix in range(17):
        xy[ix, 0] = x_center + (ix - 8) * 0.002
    xy[5, 1] = shoulder_y
    xy[6, 1] = shoulder_y
    xy[7, 1] = (shoulder_y + wrist_y) * 0.5
    xy[8, 1] = (shoulder_y + wrist_y) * 0.5
    xy[_L_WR, 1] = wrist_y
    xy[_R_WR, 1] = wrist_y
    xy[11, 1] = hip_y
    xy[12, 1] = hip_y
    xy[13, 1] = (hip_y + foot_y) * 0.5
    xy[14, 1] = (hip_y + foot_y) * 0.5
    xy[15, 1] = foot_y
    xy[16, 1] = foot_y
    return xy


def _inject_hand_spike(kp: np.ndarray, t0: int, t1: int, spike_y: float) -> None:
    for t in range(t0, min(t1 + 1, kp.shape[0])):
        alpha = (t - t0) / max(1, (t1 - t0))
        y = kp[t, _L_WR, 1] * (1 - alpha) + spike_y * alpha
        kp[t, _L_WR, 1] = y
        kp[t, _R_WR, 1] = y


def test_tapko_detectors_emit_evidence_summary_on_candidates() -> None:
    """Same geometry as tap unit test — expect at least one TapKO SafetyEvent with summary."""
    from fightsafe_ai.events.tap_detector import TapDetectorConfig, detect_tap_candidates

    fps = 30.0
    t_n = 120
    kp = np.stack([_base_pose_row(wrist_y=0.74) for _ in range(t_n)])
    _inject_hand_spike(kp, 22, 24, 0.93)
    _inject_hand_spike(kp, 52, 54, 0.92)
    cfg = TapDetectorConfig(
        min_score_emit=0.06,
        repetition_window_sec=2.0,
        min_hand_mat_band_ratio=0.48,
        peak_percentile_hand=82.0,
    )
    tap_ev = detect_tap_candidates(kp, fps, config=cfg)
    assert tap_ev, "fixture must emit tap candidates"

    times = [float(i) / fps for i in range(t_n)]
    dedup: set[tuple[str, float, float]] = set()
    rows = tapko_detectors_to_safety_events(
        stack_xy=kp,
        media_times=times,
        fps=fps,
        timestamp_seconds=float(times[-1]),
        dedup_sigs=dedup,
    )
    assert rows, "bridge must emit safety events when detectors fire"
    for se in rows:
        assert se.metadata is not None
        assert "evidence_summary" in se.metadata
        assert se.description == se.metadata["evidence_summary"]
        assert se.requires_human_confirmation is True
