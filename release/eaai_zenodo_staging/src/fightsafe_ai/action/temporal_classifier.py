"""Temporal helpers and the MVP heuristic action pipeline (pose → :class:`ActionSignal`)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from fightsafe_ai.action.base import (
    ActionSignal,
    ActionType,
    BaseActionSignalEmitter,
)
from fightsafe_ai.action.defense import (
    defensive_incapacity_confidence,
    low_guard_confidence,
    turned_back_confidence,
)
from fightsafe_ai.action.punch_kick import (
    body_scale,
    kick_activity_confidence,
    limb_motion_features,
    punch_activity_confidence,
)


def majority_vote[T](items: list[T]) -> T | None:
    """Return the most common item, or ``None`` if empty."""
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def confidence_weighted_mean(values: list[float], weights: list[float]) -> float:
    """Return weighted mean; falls back to simple mean on length mismatch or zero weight."""
    if not values:
        return 0.0
    if len(values) != len(weights) or sum(weights) <= 0.0:
        return float(sum(values) / len(values))
    s = sum(v * w for v, w in zip(values, weights, strict=True))
    return float(s / sum(weights))


@dataclass
class HeuristicMVPConfig:
    """Tunable bounds for the velocity / posture heuristics (MVP, not a tuned sports model)."""

    min_confidence: float = 0.3
    punch_vel_over_scale: float = 2.5
    kick_vel_over_scale: float = 2.0


def _evidence(motion: str, **kv: float) -> dict[str, float | str]:
    d: dict[str, float | str] = {"detector": motion}
    for k, v in kv.items():
        d[k] = float(v)
    return d


@dataclass
class HeuristicMVPActionDetector(BaseActionSignalEmitter):
    """
    IoU/velocity MVP: one list of :class:`ActionSignal` per frame, independent of risk scoring.

    Swappable for learned temporal models; interface matches :class:`BaseActionSignalEmitter`.
    """

    config: HeuristicMVPConfig = field(default_factory=HeuristicMVPConfig)

    def process_frame(
        self,
        timestamp: float,
        fighter_id: str,
        current_landmarks: dict[str, tuple[float, float]],
        previous_landmarks: dict[str, tuple[float, float]] | None,
        dt: float,
    ) -> list[ActionSignal]:
        c = self.config
        out: list[ActionSignal] = []
        fe = limb_motion_features(previous_landmarks, current_landmarks, dt)
        sc = body_scale(current_landmarks)

        p = punch_activity_confidence(
            fe,
            sc,
            vel_over_scale_threshold=c.punch_vel_over_scale,
        )
        kc = kick_activity_confidence(
            fe,
            sc,
            vel_over_scale_threshold=c.kick_vel_over_scale,
        )
        # If both pass, keep the more salient one to reduce duplicate *activity* flags (MVP only).
        if p >= c.min_confidence and kc >= c.min_confidence:
            if kc > p + 0.02 and fe.max_ankle_speed + 0.01 >= fe.max_wrist_speed:
                p = 0.0
            elif p >= kc:
                kc = 0.0
        if p >= c.min_confidence:
            out.append(
                ActionSignal(
                    float(timestamp),
                    str(fighter_id),
                    ActionType.PUNCH_ACTIVITY,
                    min(1.0, p),
                    _evidence(
                        "punch_mvp",
                        max_wrist_speed=fe.max_wrist_speed,
                        shoulder_speed=fe.shoulder_center_speed,
                        scale=sc,
                    ),
                )
            )
        if kc >= c.min_confidence:
            out.append(
                ActionSignal(
                    float(timestamp),
                    str(fighter_id),
                    ActionType.KICK_ACTIVITY,
                    min(1.0, kc),
                    _evidence(
                        "kick_mvp",
                        max_ankle_speed=fe.max_ankle_speed,
                        hip_speed=fe.hip_center_speed,
                        scale=sc,
                    ),
                )
            )

        # Posture: even without motion history
        low = low_guard_confidence(current_landmarks)
        if low >= c.min_confidence:
            out.append(
                ActionSignal(
                    float(timestamp),
                    str(fighter_id),
                    ActionType.LOW_GUARD,
                    min(1.0, low),
                    _evidence("low_guard", low_guard=low),
                )
            )
        back = turned_back_confidence(current_landmarks)
        if back >= c.min_confidence:
            out.append(
                ActionSignal(
                    float(timestamp),
                    str(fighter_id),
                    ActionType.TURNED_BACK,
                    min(1.0, back),
                    _evidence("turned_back", turned_back=back),
                )
            )
        dfi = defensive_incapacity_confidence(low, fe)
        if dfi >= c.min_confidence:
            out.append(
                ActionSignal(
                    float(timestamp),
                    str(fighter_id),
                    ActionType.DEFENSIVE_INCAPACITY,
                    min(1.0, dfi),
                    _evidence(
                        "defense_composite",
                        defensive_incapacity=dfi,
                        low_guard=low,
                        max_limb=fe.max_limb_speed,
                    ),
                )
            )
        return out


def run_sequence_mvp(
    times: list[float],
    frames: list[dict[str, tuple[float, float]]],
    fighter_id: str = "0",
    *,
    detector: HeuristicMVPActionDetector | None = None,
) -> list[ActionSignal]:
    """
    Consecutive landmark maps; ``dt`` is inferred from time stamps (or 1/30s if single step).

    Empty ``frames`` → no outputs.
    """
    if not frames or not times or len(frames) != len(times):
        return []
    det = detector or HeuristicMVPActionDetector()
    acc: list[ActionSignal] = []
    for i, (t, m) in enumerate(zip(times, frames, strict=True)):
        prev = frames[i - 1] if i > 0 else None
        if i == 0:
            dt = max(1.0 / 30.0, 1e-3)
        else:
            dt = max(1.0 / 30.0, float(times[i] - times[i - 1]))
        acc.extend(
            det.process_frame(
                t,
                fighter_id,
                m,
                prev,
                dt,
            )
        )
    return acc
