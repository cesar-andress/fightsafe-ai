"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Aggregate frame-level risk labels into time-bounded **risk events**.

This is a lightweight post-processing step for analytics and visualization; it does
not itself detect medical conditions (see :mod:`fightsafe_ai.risk.rules` disclaimer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd


COL_FRAME_ID: Final = "frame_id"
COL_TIMESTAMP: Final = "timestamp"
COL_RISK_SCORE: Final = "risk_score"
COL_RISK_LEVEL: Final = "risk_level"

# Output columns
COL_EVENT_ID: Final = "event_id"
COL_START_FRAME: Final = "start_frame"
COL_END_FRAME: Final = "end_frame"
COL_START_TIME: Final = "start_time"
COL_END_TIME: Final = "end_time"
COL_MAX_RISK_SCORE: Final = "max_risk_score"
COL_EVENT_LEVEL: Final = "event_level"
COL_DURATION_SECONDS: Final = "duration_seconds"

# When timestamps collapse to a single value (e.g. one high-risk row) and FPS is unknown
DEFAULT_MVP_FRAME_DURATION_SECONDS: Final = 0.1

_TIME_ORDER_EPS: Final = 1e-12


def _norm_level(x: Any) -> str:
    return str(x).strip().upper()


@dataclass(frozen=True)
class RiskEventExtractionConfig:
    """
    Parameters for :func:`frame_risk_to_events`.

    Parameters
    ----------
    merge_gap_frames
        Merge two candidate regions if the number of **non-event** rows between
        the end of the first and the start of the second is **strictly less than**
        this value. For example, ``2`` merges when there is 0 or 1 separator row
        between two HIGH/CRITICAL runs (assuming one table row per frame, ordered in time).
    min_duration_seconds
        Drop events whose ``(end_time - start_time)`` is **strictly less than** this.
    event_risk_levels
        Frame is part of a candidate run if its ``risk_level`` (case-insensitive)
        is in this set. Default: HIGH and CRITICAL only.
    fps
        If set (positive), frame spacing is ``1.0 / fps`` seconds when adjusting
        zero-width intervals. When omitted, spacing is inferred from consecutive
        ``timestamp`` deltas in the input table when possible.
    default_frame_duration_seconds
        If ``fps`` is unset and timestamps do not allow inference (e.g. a single
        row), use this duration for extending single-frame events (default ``0.1``
        matches common 10 FPS MVP extraction).
    """

    merge_gap_frames: int = 2
    min_duration_seconds: float = 0.0
    event_risk_levels: frozenset[str] = frozenset({"HIGH", "CRITICAL"})
    fps: float | None = None
    default_frame_duration_seconds: float = DEFAULT_MVP_FRAME_DURATION_SECONDS

    def __post_init__(self) -> None:
        if self.merge_gap_frames < 0:
            raise ValueError("merge_gap_frames must be >= 0.")
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be >= 0.")
        if not self.event_risk_levels:
            raise ValueError("event_risk_levels must be non-empty.")
        if self.fps is not None and float(self.fps) <= 0:
            raise ValueError("fps must be positive when set.")
        if float(self.default_frame_duration_seconds) <= 0:
            raise ValueError("default_frame_duration_seconds must be positive.")


def _required_columns(df: pd.DataFrame) -> None:
    need = {COL_FRAME_ID, COL_TIMESTAMP, COL_RISK_SCORE, COL_RISK_LEVEL}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"DataFrame must include columns {sorted(need)}; missing {miss}.")


def _sort_time_ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_numeric(out[COL_TIMESTAMP], errors="coerce")
    out = out.assign(_ts=ts, _frame=out[COL_FRAME_ID].astype(str))
    out = out.sort_values(by=["_ts", "_frame"], kind="mergesort")
    return out.drop(columns=["_ts", "_frame"])


def _consecutive_runs(
    is_event: np.ndarray,
) -> list[tuple[int, int]]:
    """Inclusive iloc (start, end) indices of True runs."""
    n = len(is_event)
    if n == 0:
        return []
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not is_event[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_event[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _merge_runs(runs: list[tuple[int, int]], merge_gap_frames: int) -> list[tuple[int, int]]:
    """Merge runs when the number of non-event rows between them is < merge_gap_frames."""
    if not runs:
        return runs
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        gap = s - cur_e - 1
        if gap < merge_gap_frames:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _infer_median_frame_delta_seconds(work: pd.DataFrame) -> float | None:
    """Median positive delta between consecutive ``timestamp`` values, or ``None``."""
    ts = pd.to_numeric(work[COL_TIMESTAMP], errors="coerce")
    d = ts.diff().dropna()
    d = d[(d > _TIME_ORDER_EPS) & d.notna()]
    if len(d) == 0:
        return None
    med = float(d.median())
    if not (med > _TIME_ORDER_EPS) or med > 1e6:
        return None
    return med


def _resolve_frame_duration_seconds(work: pd.DataFrame, cfg: RiskEventExtractionConfig) -> float:
    if cfg.fps is not None and float(cfg.fps) > 0:
        return 1.0 / float(cfg.fps)
    inferred = _infer_median_frame_delta_seconds(work)
    if inferred is not None:
        return inferred
    return float(cfg.default_frame_duration_seconds)


def _finalize_event_end_time(
    t0: float,
    t1: float,
    n_rows: int,
    frame_duration: float,
) -> tuple[float, float, float]:
    """
    Enforce ``end_time > start_time`` with a **positive** duration.

    * One row: ``end = start + frame_duration``.
    * Several rows with ``t1 > t0``: ``end = max(t1, t0 + frame_duration)`` (span
      covers the last sample; still strictly after start).
    * Several rows with collapsed timestamps (``t1`` ≈ ``t0``): span
      ``n_rows * frame_duration`` from ``t0``.
    """
    fd = max(float(frame_duration), 1e-12)
    if n_rows <= 0:
        end = t0 + fd
        return t0, end, end - t0
    if n_rows == 1:
        end = t0 + fd
        return t0, end, end - t0
    if t1 > t0 + _TIME_ORDER_EPS:
        end = max(t1, t0 + fd)
        return t0, end, end - t0
    end = t0 + n_rows * fd
    return t0, end, end - t0


def _event_rows(
    work: pd.DataFrame,
    s: int,
    e: int,
    frame_duration: float,
) -> dict[str, Any]:
    sl = work.iloc[s : e + 1]
    tss = pd.to_numeric(sl[COL_TIMESTAMP], errors="coerce")
    t0 = float(tss.iloc[0])
    t1 = float(tss.iloc[-1])
    n_rows = len(sl)
    t_start, t_end, duration = _finalize_event_end_time(t0, t1, n_rows, frame_duration)
    scores = pd.to_numeric(sl[COL_RISK_SCORE], errors="coerce")
    max_s = float(np.nanmax(scores.to_numpy())) if len(scores) else float("nan")
    levels = [_norm_level(x) for x in sl[COL_RISK_LEVEL].tolist()]
    ev_l = _segment_event_level(levels)
    return {
        COL_START_FRAME: sl[COL_FRAME_ID].iloc[0],
        COL_END_FRAME: sl[COL_FRAME_ID].iloc[-1],
        COL_START_TIME: t_start,
        COL_END_TIME: t_end,
        COL_MAX_RISK_SCORE: max_s,
        COL_EVENT_LEVEL: ev_l,
        "_duration": duration,
    }


def _segment_event_level(levels: list[str]) -> str:
    """CRITICAL if any frame in the segment is CRITICAL, else HIGH."""
    if "CRITICAL" in levels:
        return "CRITICAL"
    return "HIGH"


def frame_risk_to_events(
    frame_df: pd.DataFrame,
    config: RiskEventExtractionConfig | None = None,
) -> pd.DataFrame:
    """
    Convert per-frame risk rows into merged, filtered **risk events**.

    Expects one row per frame, ordered in time (will sort by ``timestamp`` then
    ``frame_id``). **HIGH** and **CRITICAL** rows (configurable) form candidate
    segments; adjacent segments are merged if the inter-run gap (in number of
    table rows) is **strictly less than** ``config.merge_gap_frames``.

    Parameters
    ----------
    frame_df
        Columns: ``frame_id``, ``timestamp``, ``risk_score``, ``risk_level``.
    config
        Gap merge and minimum duration rules.

    Returns
    -------
    pd.DataFrame
        One row per event: ``event_id``, ``start_frame``, ``end_frame``,
        ``start_time``, ``end_time``, ``max_risk_score``, ``event_level``,
        ``duration_seconds`` (``end_time - start_time``, **strictly positive**;
        single-frame events are extended by one frame duration so
        ``start_time < end_time`` holds for quality-assurance checks).

    See Also
    --------
    :func:`frame_risk_to_events_list` : same data as ``list[dict]``.
    """
    cfg = config or RiskEventExtractionConfig()
    if frame_df is None or len(frame_df) == 0:
        return _empty_events_df()

    _required_columns(frame_df)
    work = _sort_time_ordered(frame_df).reset_index(drop=True)
    frame_duration = _resolve_frame_duration_seconds(work, cfg)

    levels = {_norm_level(x) for x in cfg.event_risk_levels}
    is_event = work[COL_RISK_LEVEL].map(_norm_level).isin(levels).to_numpy()
    runs = _consecutive_runs(is_event)
    runs = _merge_runs(runs, cfg.merge_gap_frames)

    records: list[dict[str, Any]] = []
    eid = 0
    for s, e in runs:
        rowd = _event_rows(work, s, e, frame_duration)
        d = float(rowd.pop("_duration"))
        if d < float(cfg.min_duration_seconds):
            continue
        rec = {
            COL_EVENT_ID: eid,
            COL_START_FRAME: rowd[COL_START_FRAME],
            COL_END_FRAME: rowd[COL_END_FRAME],
            COL_START_TIME: rowd[COL_START_TIME],
            COL_END_TIME: rowd[COL_END_TIME],
            COL_MAX_RISK_SCORE: rowd[COL_MAX_RISK_SCORE],
            COL_EVENT_LEVEL: rowd[COL_EVENT_LEVEL],
            COL_DURATION_SECONDS: d,
        }
        records.append(rec)
        eid += 1

    if not records:
        return _empty_events_df()

    return pd.DataFrame.from_records(
        records,
        columns=[
            COL_EVENT_ID,
            COL_START_FRAME,
            COL_END_FRAME,
            COL_START_TIME,
            COL_END_TIME,
            COL_MAX_RISK_SCORE,
            COL_EVENT_LEVEL,
            COL_DURATION_SECONDS,
        ],
    )


def frame_risk_to_events_list(
    frame_df: pd.DataFrame,
    config: RiskEventExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Like :func:`frame_risk_to_events` but returns a list of plain dicts.
    """
    out = frame_risk_to_events(frame_df, config=config)
    if len(out) == 0:
        return []
    return cast("list[dict[str, Any]]", out.to_dict(orient="records"))


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_EVENT_ID,
            COL_START_FRAME,
            COL_END_FRAME,
            COL_START_TIME,
            COL_END_TIME,
            COL_MAX_RISK_SCORE,
            COL_EVENT_LEVEL,
            COL_DURATION_SECONDS,
        ],
    )
