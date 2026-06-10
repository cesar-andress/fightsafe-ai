"""
JSON/CSV helpers for risk tables and run artifacts (shared by :mod:`fightsafe_ai.pipeline.mvp` and steps).
"""

from __future__ import annotations

import json
import math
import numbers
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


COL_RISK_SCORE = "risk_score"
COL_RISK_LEVEL = "risk_level"
COL_TRIGGERED = "triggered_rules"


def _is_empty_cell(x: Any) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or np.isnan(x)))


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas scalars and NaN into JSON-friendly values.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return sanitize_for_json(obj.item())
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def risk_scores_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list columns (e.g. ``triggered_rules``) for stable CSV output."""
    out = df.copy()
    if COL_TRIGGERED in out.columns:
        out[COL_TRIGGERED] = out[COL_TRIGGERED].apply(
            lambda x: (
                json.dumps(list(x))
                if isinstance(x, (list, tuple))
                else ("" if _is_empty_cell(x) else x)
            )
        )
    return out


__all__ = [
    "COL_RISK_LEVEL",
    "COL_RISK_SCORE",
    "COL_TRIGGERED",
    "risk_scores_dataframe_for_csv",
    "sanitize_for_json",
]
