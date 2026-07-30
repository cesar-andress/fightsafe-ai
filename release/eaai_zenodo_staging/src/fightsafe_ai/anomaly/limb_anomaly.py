"""
Limb / joint anomaly heuristics: tabular columns (legacy) + time-series :class:`LimbAnomalyDetector`.

**Not** a clinical exam. 2D joint angles differ from goniometry; asymmetry can be stance,
injury, or pose-estimation bias. Do not infer injury type or severity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fightsafe_ai.anomaly.base import AnomalySignal, AnomalyType, BaseTimeSeriesAnomalyDetector
from fightsafe_ai.features.anomaly import (
    COL_ANOMALY_SCORE,
    COL_ANOMALY_TYPE,
    add_limb_anomaly_columns,
)
from fightsafe_ai.features.biomechanics import knee_flexion_deg


@dataclass
class LimbAnomalyDetectorConfig:
    knee_stress_threshold: float = 0.55
    asym_threshold: float = 0.5
    collapse_threshold: float = 0.5
    min_confidence: float = 0.33
    fps_assumed: float = 30.0


def _pt(f: dict[str, tuple[float, float]], k: str) -> tuple[float, float] | None:
    p = f.get(k)
    return (float(p[0]), float(p[1])) if p is not None else None


def _knee_flexes(f: dict[str, tuple[float, float]]) -> tuple[float, float]:
    kfl = knee_flexion_deg(_pt(f, "left_hip"), _pt(f, "left_knee"), _pt(f, "left_ankle"))
    kfr = knee_flexion_deg(_pt(f, "right_hip"), _pt(f, "right_knee"), _pt(f, "right_ankle"))
    if not (np.isfinite(kfl) and np.isfinite(kfr)):
        return (float("nan"), float("nan"))
    return (float(kfl), float(kfr))


def _elbow_flexes(f: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Reuses knee flexion chain geometry (2D) at elbow — same API constraints."""
    e_l = knee_flexion_deg(_pt(f, "left_shoulder"), _pt(f, "left_elbow"), _pt(f, "left_wrist"))
    e_r = knee_flexion_deg(_pt(f, "right_shoulder"), _pt(f, "right_elbow"), _pt(f, "right_wrist"))
    if not (np.isfinite(e_l) and np.isfinite(e_r)):
        return (float("nan"), float("nan"))
    return (float(e_l), float(e_r))


def _knee_bend_stress_01(deg: float) -> float:
    if not np.isfinite(deg):
        return 0.0
    d = float(deg)
    over_straight = float(np.clip((5.0 - d) / 5.0, 0.0, 1.0))
    deep = float(np.clip((d - 100.0) / 45.0, 0.0, 1.0))
    return float(max(over_straight, deep))


def _bilateral_knee_asym_01(deg_l: float, deg_r: float) -> float:
    if not (np.isfinite(deg_l) and np.isfinite(deg_r)):
        return 0.0
    return float(np.clip(abs(deg_l - deg_r) / 40.0, 0.0, 1.0))


class LimbAnomalyDetector(BaseTimeSeriesAnomalyDetector):
    """
    Highlights unusual joint configuration, leg asymmetry, and sudden 2D *support* change.

    Mirrors logic used in :func:`add_limb_anomaly_columns` but returns structured signals for the
    end of the window. Does **not** add columns to a DataFrame.
    """

    def __init__(self, config: LimbAnomalyDetectorConfig | None = None) -> None:
        self.config = config or LimbAnomalyDetectorConfig()

    def analyze(
        self,
        times: list[float],
        frames: list[dict[str, tuple[float, float]]],
        fighter_id: str,
    ) -> list[AnomalySignal]:
        cfg = self.config
        if len(frames) < 2 or len(frames) != len(times):
            return []
        t_end = float(times[-1])
        f0, f1 = frames[-2], frames[-1]
        kl0, kr0 = _knee_flexes(f0)
        kl1, kr1 = _knee_flexes(f1)
        _ = _elbow_flexes(f0)
        el1, er1 = _elbow_flexes(f1)

        s_k1 = max(_knee_bend_stress_01(kl1), _knee_bend_stress_01(kr1))
        s_e1 = max(_knee_bend_stress_01(el1), _knee_bend_stress_01(er1))
        s_joint = max(s_k1, s_e1)
        asym1 = _bilateral_knee_asym_01(kl1, kr1)

        ay0 = [p for a in ("left_ankle", "right_ankle") if (p := _pt(f0, a)) is not None]
        ay1 = [p for a in ("left_ankle", "right_ankle") if (p := _pt(f1, a)) is not None]
        aymin0 = max(p[1] for p in ay0) if ay0 else float("nan")
        aymin1 = max(p[1] for p in ay1) if ay1 else float("nan")
        jchange = 0.0
        if np.isfinite(kl0) and np.isfinite(kl1):
            jchange = max(jchange, abs(kl1 - kl0))
        if np.isfinite(kr0) and np.isfinite(kr1):
            jchange = max(jchange, abs(kr1 - kr0))
        drop = 0.0
        if np.isfinite(aymin0) and np.isfinite(aymin1) and aymin1 >= aymin0:
            drop = float(
                np.clip(
                    (aymin1 - aymin0) * max(cfg.fps_assumed, 1e-6) / 0.5,
                    0.0,
                    1.0,
                )
            )
        c_col = float(np.clip(0.55 * min(1.0, jchange / 25.0) + 0.45 * drop, 0.0, 1.0))

        out: list[AnomalySignal] = []
        cj = float(np.clip((s_joint - 0.3) / 0.5, 0.0, 1.0))
        if cj >= cfg.min_confidence and s_joint >= cfg.knee_stress_threshold * 0.88:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.LIMB_JOINT_ANGULAR_ANOMALY,
                    cj,
                    {
                        "knee_stress_max": float(s_k1),
                        "elbow_stress_max": float(s_e1),
                    },
                )
            )
        if asym1 >= cfg.asym_threshold and asym1 * 0.95 >= cfg.min_confidence:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.LIMB_BILATERAL_ASYMMETRY,
                    min(1.0, float(asym1)),
                    {
                        "knee_flexion_left_deg": float(kl1) if np.isfinite(kl1) else -1.0,
                        "knee_flexion_right_deg": float(kr1) if np.isfinite(kr1) else -1.0,
                    },
                )
            )
        if c_col >= cfg.collapse_threshold and c_col >= cfg.min_confidence * 0.9:
            out.append(
                AnomalySignal(
                    t_end,
                    fighter_id,
                    AnomalyType.LIMB_SUDDEN_SUPPORT_LOSS,
                    c_col,
                    {
                        "knee_delta_max": float(jchange),
                        "ankle_y_drop_01": float(drop),
                    },
                )
            )
        return out


__all__ = [
    "COL_ANOMALY_SCORE",
    "COL_ANOMALY_TYPE",
    "LimbAnomalyDetector",
    "LimbAnomalyDetectorConfig",
    "add_limb_anomaly_columns",
]
