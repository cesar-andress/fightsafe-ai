"""
Defensive posture heuristics: low guard, turned-back proxy, and composite **defensive incapacity**.

``DEFENSIVE_INCAPACITY`` here means *exposed static guard* (hands down, little parry motion) — not
a medical assessment. Replace with learned classifiers or fusion with risk as needed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fightsafe_ai.action.punch_kick import LimbMotionFeatures


XY = tuple[float, float]


def guard_open_proxy(shoulder_y: float, wrist_y: float) -> float:
    """
    Return a [0,1] score where **higher** means more exposed high line (wrists well below shoulders).

    Pure geometry on normalized y (image coordinates, **increasing** downward in many CV setups).
    """
    s, w = float(shoulder_y), float(wrist_y)
    gap = w - s
    if gap <= 0.0:
        return 0.0
    return float(min(1.0, gap * 4.0))


def _get(m: Mapping[str, XY], name: str) -> XY | None:
    p = m.get(name)
    if p is None:
        return None
    return (float(p[0]), float(p[1]))


def low_guard_confidence(landmarks: Mapping[str, XY], *, margin_y: float = 0.04) -> float:
    """
    Hands well below a head proxy (nose), symmetric check on both sides.

    ``y`` increases downward: large ``wrist_y - nose_y`` → hands dropped (bad guard in upright stance).
    """
    nose = _get(landmarks, "nose")
    l_w = _get(landmarks, "left_wrist")
    r_w = _get(landmarks, "right_wrist")
    if nose is None or (l_w is None and r_w is None):
        return 0.0
    ny = nose[1]
    scores: list[float] = []
    for w in (l_w, r_w):
        if w is None:
            continue
        # wrist clearly below nose -> positive gap
        g = w[1] - ny
        if g <= margin_y:
            scores.append(0.0)
        else:
            scores.append(float(np.clip((g - margin_y) / 0.12, 0.0, 1.0)))
    if not scores:
        return 0.0
    return float(max(scores))


def turned_back_confidence(landmarks: Mapping[str, XY], *, min_hip_width: float = 0.02) -> float:
    """
    Frontal view has **wider** projected shoulder line than a profile; ratio shoulder/hip
    width drops when the upper body is heavily rotated in the image.
    """
    ls, rs = _get(landmarks, "left_shoulder"), _get(landmarks, "right_shoulder")
    lh, rh = _get(landmarks, "left_hip"), _get(landmarks, "right_hip")
    if ls is None or rs is None or lh is None or rh is None:
        return 0.0
    sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    hw = float(np.hypot(rh[0] - lh[0], rh[1] - lh[1]))
    if hw < min_hip_width:
        return 0.0
    r = sw / max(hw, 1e-5)
    # Typical front: r ~0.8-1.2; strong profile: r can drop
    if r > 0.75:
        return 0.0
    return float(np.clip((0.75 - r) / 0.35, 0.0, 1.0))


def defensive_incapacity_confidence(
    low_guard: float,
    features: LimbMotionFeatures,
    *,
    static_speed_threshold: float = 0.45,
) -> float:
    """
    Exposed (high low_guard) and little end-effector motion: little effective parry or shell.

    Fires when the athlete is *static* at low guard; fast limb motion is assumed to be offense or
    evasion so we suppress this signal.
    """
    if low_guard < 0.42:
        return 0.0
    s = max(0.0, float(features.max_limb_speed))
    if s > static_speed_threshold * 1.4:
        return 0.0
    # High low_guard + low speed => high
    staticness = 1.0 - min(1.0, s / max(static_speed_threshold, 1e-5))
    return float(np.clip(low_guard * (0.35 + 0.65 * staticness), 0.0, 1.0))
