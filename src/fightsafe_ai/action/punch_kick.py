"""
Heuristic punch / kick **activity** from wrist, ankle, shoulder, and hip speeds (MVP).

Training-free; not a competition strike counter. Replaceable by learned action models
while keeping :class:`~fightsafe_ai.action.base.ActionSignal` outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


XY = tuple[float, float]


@dataclass(frozen=True, slots=True)
class LimbMotionFeatures:
    """Per-frame motion magnitudes (norm of landmark displacement / dt) in the same space as keypoints."""

    max_wrist_speed: float
    max_ankle_speed: float
    shoulder_center_speed: float
    hip_center_speed: float
    left_wrist_speed: float
    right_wrist_speed: float
    left_ankle_speed: float
    right_ankle_speed: float

    @property
    def max_limb_speed(self) -> float:
        return max(
            self.left_wrist_speed,
            self.right_wrist_speed,
            self.left_ankle_speed,
            self.right_ankle_speed,
        )


def _pt(
    m: Mapping[str, XY],
    name: str,
) -> XY | None:
    p = m.get(name)
    if p is None:
        return None
    return (float(p[0]), float(p[1]))


def _vel(a: XY | None, b: XY | None, dt: float) -> float:
    if a is None or b is None or dt <= 0.0:
        return 0.0
    d = (b[0] - a[0], b[1] - a[1])
    return float(np.hypot(d[0], d[1]) / dt)


def _mid(a: XY | None, b: XY | None) -> XY | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def body_scale(landmarks: Mapping[str, XY]) -> float:
    """Shoulder line length as a scene scale (relative coords)."""
    ls, rs = _pt(landmarks, "left_shoulder"), _pt(landmarks, "right_shoulder")
    if ls is None or rs is None:
        return 0.1
    w = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    return max(w, 1e-5)


def limb_motion_features(
    landmarks_prev: Mapping[str, XY] | None,
    landmarks_cur: Mapping[str, XY],
    dt: float,
) -> LimbMotionFeatures:
    """
    Require ``landmarks_prev`` and positive ``dt`` to get non-zero speeds; else everything is 0.
    """
    if landmarks_prev is None or dt <= 0.0:
        return LimbMotionFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def v(a: str, b: str) -> float:
        return _vel(_pt(landmarks_prev, a), _pt(landmarks_cur, b), dt)

    lws, rws = v("left_wrist", "left_wrist"), v("right_wrist", "right_wrist")
    las, ras = v("left_ankle", "left_ankle"), v("right_ankle", "right_ankle")
    sm0 = _mid(_pt(landmarks_prev, "left_shoulder"), _pt(landmarks_prev, "right_shoulder"))
    sm1 = _mid(_pt(landmarks_cur, "left_shoulder"), _pt(landmarks_cur, "right_shoulder"))
    hm0 = _mid(_pt(landmarks_prev, "left_hip"), _pt(landmarks_prev, "right_hip"))
    hm1 = _mid(_pt(landmarks_cur, "left_hip"), _pt(landmarks_cur, "right_hip"))
    scs = _vel(sm0, sm1, dt)
    hps = _vel(hm0, hm1, dt)
    mws = max(lws, rws)
    mas = max(las, ras)
    return LimbMotionFeatures(
        max_wrist_speed=mws,
        max_ankle_speed=mas,
        shoulder_center_speed=scs,
        hip_center_speed=hps,
        left_wrist_speed=lws,
        right_wrist_speed=rws,
        left_ankle_speed=las,
        right_ankle_speed=ras,
    )


def strike_energy_proxy(wrist_speed: float, hip_rotation_speed: float) -> float:
    """
    Return a [0,1] score combining end-effector speed and torso rotation (unscaled proxy).

    Parameters
    ----------
    wrist_speed
        Norm of wrist motion in relative screen units per second (or arbitrary scale).
    hip_rotation_speed
        Magnitude of hip / pelvis angular proxy in compatible units.
    """
    a = max(0.0, float(wrist_speed))
    b = max(0.0, float(hip_rotation_speed))
    raw = 0.65 * a + 0.35 * b
    return float(np.clip(raw / (1.0 + raw), 0.0, 1.0))


def punch_activity_confidence(
    features: LimbMotionFeatures,
    scale: float,
    *,
    vel_over_scale_threshold: float = 2.5,
) -> float:
    """
    Wrist-dominant burst relative to body scale, with some shoulder involvement (chain).
    """
    s = max(float(scale), 1e-5)
    rel_wrist = features.max_wrist_speed / s
    rel_shoulder = features.shoulder_center_speed / s
    raw = 0.72 * rel_wrist + 0.28 * min(rel_wrist, rel_shoulder * 1.2)
    out = (raw - vel_over_scale_threshold * 0.25) / (vel_over_scale_threshold * 0.75)
    return float(np.clip(out, 0.0, 1.0))


def kick_activity_confidence(
    features: LimbMotionFeatures,
    scale: float,
    *,
    vel_over_scale_threshold: float = 2.0,
) -> float:
    """
    Ankle-dominant burst; hip line motion supports a kick (vs. pure hand flail).
    """
    s = max(float(scale), 1e-5)
    rel_ankle = features.max_ankle_speed / s
    rel_hip = features.hip_center_speed / s
    raw = 0.55 * rel_ankle + 0.45 * min(rel_ankle, rel_hip * 1.3)
    out = (raw - vel_over_scale_threshold * 0.2) / (vel_over_scale_threshold * 0.8)
    return float(np.clip(out, 0.0, 1.0))
