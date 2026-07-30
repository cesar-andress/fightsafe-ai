"""
Adapters from MVP feature rows / rule components to :class:`~fightsafe_ai.risk.formal_model.RiskSignal`.

Does **not** replace :func:`~fightsafe_ai.risk.scorer.compute_interpretable_risk`; it exposes the same
physics through the formal fusion datatype for experiments and audits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from fightsafe_ai.risk.formal_model import RiskFusionConfig, RiskSignal, SignalGroup
from fightsafe_ai.risk.rules import ALL_RULE_NAMES, InterpretableRiskConfig, build_rule_components
from fightsafe_ai.risk.surrender import COL_SURRENDER_CONFIDENCE, SURRENDER_RULE_KEY


RULE_GROUP: dict[str, SignalGroup] = {
    "fast_downward_motion": "biomechanics",
    "large_torso_angle": "biomechanics",
    "prolonged_low_posture": "posture",
    "high_instability": "biomechanics",
    "post_fall_low_movement": "inactivity",
    "low_guard": "posture",
    "facing_away": "posture",
    "reaction_delay_after_impact": "action",
    "loss_of_control": "action",
    "clear_danger_fall": "anomaly",
    "intervention_urgent": "anomaly",
    "high_risk_guard_strike": "action",
    "limb_anomaly": "anomaly",
    SURRENDER_RULE_KEY: "surrender",
    "vlm_review_hint": "vlm",
}


def signals_from_feature_row(
    row: Mapping[str, Any],
    *,
    interpretable_config: InterpretableRiskConfig,
    fusion_config: RiskFusionConfig,
    surrender_confidence: float | None = None,
    vlm_review_score: float | None = None,
) -> list[RiskSignal]:
    """
    Map one feature/risk row into formal :class:`RiskSignal` instances.

    Uses :func:`~fightsafe_ai.risk.rules.build_rule_components` on a one-row DataFrame so behavior
    tracks the MVP scorer. Optional surrender/VLM confidences come from explicit kwargs or row keys.
    """
    df = pd.DataFrame([dict(row)])
    comp, _active = build_rule_components(df, interpretable_config)
    signals: list[RiskSignal] = []

    for name in ALL_RULE_NAMES:
        val = float(comp[name][0])
        grp = RULE_GROUP.get(name, "biomechanics")
        w = float(fusion_config.signal_weights.get(name, 0.0))
        signals.append(
            RiskSignal(
                name=name,
                group=grp,
                confidence=val,
                weight=w,
                polarity="risk_increasing",
                evidence={"source": "build_rule_components"},
            )
        )

    sur = surrender_confidence
    if sur is None and COL_SURRENDER_CONFIDENCE in df.columns:
        sur = float(df[COL_SURRENDER_CONFIDENCE].iloc[0])
    if sur is None and "surrender_confidence" in df.columns:
        sur = float(df["surrender_confidence"].iloc[0])

    if sur is not None:
        signals.append(
            RiskSignal(
                name=SURRENDER_RULE_KEY,
                group="surrender",
                confidence=max(0.0, min(1.0, float(sur))),
                weight=float(fusion_config.signal_weights.get(SURRENDER_RULE_KEY, 0.06)),
                polarity="risk_increasing",
                evidence={"source": "surrender_heuristic"},
            )
        )

    vlm = vlm_review_score
    if vlm is None and "vlm_review_score" in df.columns:
        vlm = float(df["vlm_review_score"].iloc[0])
    if vlm is not None:
        signals.append(
            RiskSignal(
                name="vlm_review_hint",
                group="vlm",
                confidence=max(0.0, min(1.0, float(vlm))),
                weight=float(fusion_config.signal_weights.get("vlm_review_hint", 0.0)),
                polarity="risk_increasing",
                evidence={"source": "vlm_optional"},
            )
        )

    return signals
