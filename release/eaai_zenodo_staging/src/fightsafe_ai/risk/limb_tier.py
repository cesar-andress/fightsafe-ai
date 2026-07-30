"""
Post-hoc **tier nudges** for limb anomaly (adds to weighted interpretable risk).

**Not** a medical or clinical step; see :mod:`fightsafe_ai.features.anomaly` for the
heuristic feature definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fightsafe_ai.risk.rules import RULE_LIMB_ANOMALY, InterpretableAggregationConfig


# Same semantics as in ``features.anomaly`` (MVP; tunable in code before YAML).
COL_ANOMALY_SCORE: str = "anomaly_score"
LIMB_TIER_HIGH: float = 0.55
LIMB_TIER_CRITICAL: float = 0.80


def _level_index(level: str) -> int:
    o = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    return o.get(str(level), 0)


def apply_limb_anomaly_tier_overrides(
    df: pd.DataFrame,
    agg: InterpretableAggregationConfig,
    *,
    col_score: str = "risk_score",
    col_level: str = "risk_level",
    col_triggered: str = "triggered_rules",
) -> pd.DataFrame:
    """
    After the weighted :func:`fightsafe_ai.risk.scorer.compute_interpretable_risk` pass,
    optionally raise **risk_level** to HIGH or **CRITICAL** when ``anomaly_score`` is high.

    **MVP, not clinically validated.**
    """
    if df.empty or COL_ANOMALY_SCORE not in df.columns:
        return df
    h_min = float(agg.level_high_min)
    cmin = float(agg.level_critical_min)
    an = np.nan_to_num(df[COL_ANOMALY_SCORE].to_numpy(dtype=float, copy=False), nan=0.0)
    scores = np.nan_to_num(
        df[col_score].to_numpy(dtype=float, copy=False), nan=0.0, posinf=1.0, neginf=0.0
    )
    levels: list[str] = list(df[col_level].astype(str))
    raw_trig = df[col_triggered].tolist()
    trig: list[list[str]] = [
        [str(x) for x in row] if isinstance(row, (list, tuple)) else [] for row in raw_trig
    ]
    n = len(df)
    for i in range(n):
        a = float(an[i])
        if a < LIMB_TIER_HIGH:
            continue
        tlist = list(trig[i])
        if RULE_LIMB_ANOMALY not in tlist:
            tlist.append(RULE_LIMB_ANOMALY)
        trig[i] = tlist
        if a >= LIMB_TIER_CRITICAL:
            levels[i] = "CRITICAL"
            scores[i] = max(scores[i], cmin, min(0.99, cmin + 0.1))
        else:
            cur = str(levels[i])
            if _level_index(cur) < _level_index("HIGH"):
                levels[i] = "HIGH"
                scores[i] = max(scores[i], h_min, min(0.99, h_min + 0.02))
            else:
                scores[i] = max(scores[i], h_min, min(0.99, h_min + 0.01))
    out = df.copy()
    out[col_score] = np.clip(scores, 0.0, 1.0)
    out[col_level] = levels
    out[col_triggered] = trig
    return out
