"""
OpenCV overlay helpers for live preview (skeleton + risk HUD).

Reuses drawing primitives from :mod:`fightsafe_ai.visualization.overlay` where possible.
"""

from __future__ import annotations

import cv2
import numpy as np

from fightsafe_ai.live.event_bus import SafetyLevel
from fightsafe_ai.pose.blazepose import blazepose_index
from fightsafe_ai.pose.keypoints import PoseResult
from fightsafe_ai.visualization.overlay import (
    OverlayVizConfig,
    draw_elevated_risk_banner,
    draw_skeleton_bgr,
)


def pose_result_to_index_landmarks(pose: PoseResult) -> dict[int, tuple[float, float, float]]:
    """Map named BlazePose keypoints to indexed tuples for :func:`draw_skeleton_bgr`."""
    out: dict[int, tuple[float, float, float]] = {}
    for kp in pose.keypoints:
        ix = blazepose_index(kp.name)
        if ix is None:
            continue
        vis = float(kp.visibility) if kp.visibility is not None else 1.0
        out[ix] = (float(kp.x), float(kp.y), vis)
    return out


def draw_live_overlay(
    frame_bgr: np.ndarray,
    *,
    pose: PoseResult | None,
    risk_level: SafetyLevel,
    raw_risk_level: str,
    triggered_rules: list[str],
    dangerous: bool,
    cfg: OverlayVizConfig | None = None,
) -> np.ndarray:
    """
    Draw skeleton, optional elevated-risk banner, red danger tint, and a compact HUD strip.

    Mutates a **copy** of ``frame_bgr`` and returns it.
    """
    cfg = cfg or OverlayVizConfig()
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if pose and pose.keypoints:
        lm_idx = pose_result_to_index_landmarks(pose)
        draw_skeleton_bgr(out, lm_idx, w, h, cfg)

    tier_banner = None
    if raw_risk_level in ("HIGH", "CRITICAL"):
        tier_banner = raw_risk_level
    elif risk_level in (SafetyLevel.HIGH, SafetyLevel.CRITICAL):
        tier_banner = risk_level.value

    y_off = 0
    if tier_banner is not None:
        y_off = draw_elevated_risk_banner(out, tier_banner, cfg)

    hud = f"{raw_risk_level} | rules={len(triggered_rules)}"
    if triggered_rules[:3]:
        hud += " | " + ",".join(triggered_rules[:3])
    hx = hud[: min(120, len(hud))]
    org = (8, max(28, y_off + 8))
    cv2.putText(out, hx, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hx, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if dangerous:
        critical = risk_level == SafetyLevel.CRITICAL or raw_risk_level == "CRITICAL"
        _apply_danger_tint(out, strength=0.34 if critical else 0.22)

    return out


def _apply_danger_tint(frame: np.ndarray, *, strength: float) -> None:
    overlay = frame.copy()
    overlay[:] = (55, 55, 220)
    cv2.addWeighted(overlay, strength, frame, 1.0 - strength, 0, dst=frame)


__all__ = ["draw_live_overlay", "pose_result_to_index_landmarks"]
