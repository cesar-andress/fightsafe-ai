"""
Deterministic, testable run-level metrics and analytic warnings for QA.

All helpers are pure (no I/O) except :func:`count_frame_images` which is re-exported
for convenience. Callers in :func:`fightsafe_ai.qa.validators.run_quality_checks`
aggregate filesystem reads; unit tests can pass in-memory
:class:`pandas.DataFrame` samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from fightsafe_ai.qa.dataset_checks import count_frame_images


METRIC_KEYS: Final[tuple[str, ...]] = (
    "total_frames",
    "frames_with_pose",
    "pose_coverage_percent",
    "average_risk_score",
    "max_risk_score",
    "number_of_events",
    "duration_seconds",
)

# Extra keys written by :func:`fightsafe_ai.qa.llm_status.build_llm_qa_metrics` into
# ``qa_report.json`` / :class:`QualityReport.metrics` (``merge_metrics`` / defaults).
LLM_QA_METRIC_KEYS: Final[tuple[str, ...]] = (
    "llm_enabled",
    "llm_used",
    "llm_success_rate",
)

# All flat keys present in a complete QA report ``metrics`` object (core + LLM).
QA_REPORT_METRIC_KEYS: Final[tuple[str, ...]] = METRIC_KEYS + LLM_QA_METRIC_KEYS


# Match :mod:`fightsafe_ai.reports.summary` semantics for a comparable duration.
def duration_seconds_from_risk(risk_df: pd.DataFrame | None) -> float:
    """``max(timestamp) - min(timestamp)`` when ``timestamp`` is present, else 0.0."""
    if risk_df is None or "timestamp" not in risk_df.columns or len(risk_df) == 0:
        return 0.0
    t = pd.to_numeric(risk_df["timestamp"], errors="coerce").dropna()
    if len(t) < 1:
        return 0.0
    if len(t) < 2:
        return 0.0
    return float(t.max() - t.min())


def count_total_frames(frames_dir: Path, risk_df: pd.DataFrame | None) -> int:
    """
    Count extracted frame images; if there are none, use unique ``frame_id`` in risk,
    or row count as last resort.
    """
    n_img = int(count_frame_images(frames_dir))
    if n_img > 0:
        return n_img
    if risk_df is not None and len(risk_df) and "frame_id" in risk_df.columns:
        return int(risk_df["frame_id"].astype(str).nunique())
    if risk_df is not None:
        return len(risk_df)
    return 0


def count_frames_with_pose(pose_df: pd.DataFrame | None) -> int:
    if pose_df is None or "frame_id" not in pose_df.columns or len(pose_df) == 0:
        return 0
    return int(pose_df["frame_id"].astype(str).nunique())


def average_risk_score(risk_df: pd.DataFrame | None) -> float | None:
    if risk_df is None or "risk_score" not in risk_df.columns or len(risk_df) == 0:
        return None
    s = pd.to_numeric(risk_df["risk_score"], errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.mean())


def max_risk_score(risk_df: pd.DataFrame | None) -> float | None:
    if risk_df is None or "risk_score" not in risk_df.columns or len(risk_df) == 0:
        return None
    s = pd.to_numeric(risk_df["risk_score"], errors="coerce").dropna()
    if len(s) == 0:
        return None
    m = float(s.max())
    return m if np.isfinite(m) else None


def is_constant_risk_score(risk_df: pd.DataFrame | None) -> bool:
    if risk_df is None or "risk_score" not in risk_df.columns or len(risk_df) < 2:
        return False
    s = pd.to_numeric(risk_df["risk_score"], errors="coerce")
    s = s.dropna()
    if len(s) < 2:
        return False
    u = s.nunique(dropna=True)
    return u <= 1


def build_run_metrics(
    *,
    frames_dir: Path,
    risk_df: pd.DataFrame | None,
    pose_df: pd.DataFrame | None,
    number_of_events: int,
    pose_coverage_percent: float | None,
) -> dict[str, Any]:
    """
    Canonical metric names for ``qa_report.json`` ``"metrics"`` (flat scalars, JSON-safe).
    """
    m: dict[str, Any] = {
        "total_frames": int(count_total_frames(frames_dir, risk_df)),
        "frames_with_pose": int(count_frames_with_pose(pose_df)),
        "pose_coverage_percent": None
        if pose_coverage_percent is None
        else float(pose_coverage_percent),
        "average_risk_score": average_risk_score(risk_df),
        "max_risk_score": max_risk_score(risk_df),
        "number_of_events": int(max(0, number_of_events)),
        "duration_seconds": float(duration_seconds_from_risk(risk_df)),
    }
    pcp0 = m.get("pose_coverage_percent")
    if pcp0 is not None and isinstance(pcp0, (int, float, np.floating)):
        m["pose_coverage_percent"] = round(float(pcp0), 6)
    m["duration_seconds"] = round(float(m["duration_seconds"]), 6)
    for v in "average_risk_score", "max_risk_score":
        if m.get(v) is not None and isinstance(m[v], (int, float, np.floating)):
            m[v] = round(float(m[v]), 8)
    return m


def default_metrics_block() -> dict[str, Any]:
    """All metric keys with ``null`` (JSON ``null``) for schema-stable reports."""
    d = dict.fromkeys(METRIC_KEYS)
    d.update(dict.fromkeys(LLM_QA_METRIC_KEYS))
    return d


def merge_metrics(
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**default_metrics_block(), **(current or {})}


__all__ = [
    "LLM_QA_METRIC_KEYS",
    "METRIC_KEYS",
    "QA_REPORT_METRIC_KEYS",
    "average_risk_score",
    "build_run_metrics",
    "count_frame_images",
    "count_frames_with_pose",
    "count_total_frames",
    "default_metrics_block",
    "duration_seconds_from_risk",
    "is_constant_risk_score",
    "max_risk_score",
    "merge_metrics",
]
