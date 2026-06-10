"""
Static matplotlib figures from pipeline run artifacts (no display required).

The figures use a restrained, print-oriented palette: gray bands for risk
levels, a dark line for the risk score, and small markers for **detected**
events. Matplotlib only (``Agg`` backend).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.patches import Patch

from fightsafe_ai.pipeline.artifact_io import COL_RISK_LEVEL, COL_RISK_SCORE


logger = logging.getLogger(__name__)

# Muted academic palette (no bright UI / traffic-light colors)
_COLOR_LINE = "#1a1a1a"
_COLOR_GRID = "#d0d0d0"
_COLOR_HIGH = "#6e6e6e"
_COLOR_CRITICAL = "#2c2c2c"
_COLOR_MARKER = "#0d0d0d"
_COLOR_TIER_MED = "#9a9a9a"
_COLOR_TIER_LOW = "#b8b8b8"
_COLOR_TIER_UNKNOWN = "#8a8a8a"

_LEVEL_FILL: dict[str, str] = {
    "CRITICAL": _COLOR_CRITICAL,
    "HIGH": _COLOR_HIGH,
}

_ACADEMIC_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "sans-serif",
    ],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _COLOR_GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.framealpha": 0.95,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.1,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

_LEVEL_TONE_EVENT: dict[str, str] = {
    "CRITICAL": _COLOR_CRITICAL,
    "HIGH": _COLOR_HIGH,
    "MEDIUM": _COLOR_TIER_MED,
    "LOW": _COLOR_TIER_LOW,
}
_LEGEND_ORDER: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


def _load_risk_frame(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "risk_scores.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)


def _load_events_list(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "events.json"
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
    elif "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    else:
        t = np.arange(len(df), dtype=float)
        logger.info("No timestamp column; using frame index for x-axis.")
    return t


def _time_axis_epsilon(t: np.ndarray) -> float:
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.02
    t_sorted = np.sort(np.unique(t))
    d = np.diff(t_sorted)
    d = d[d > 0]
    if d.size:
        return float(0.25 * np.median(d))
    return 0.02


def _norm_risk_level_column(df: pd.DataFrame) -> np.ndarray:
    c = "risk_level"
    if c not in df.columns and COL_RISK_LEVEL in df.columns:
        c = COL_RISK_LEVEL
    if c not in df.columns:
        return np.array([""] * len(df), dtype=object)
    return df[c].astype(str).str.strip().str.upper().to_numpy()


def _consecutive_risk_spans(
    t: np.ndarray, levels: np.ndarray, want: frozenset[str]
) -> list[tuple[float, float, str]]:
    n = len(t)
    if n == 0:
        return []
    spans: list[tuple[float, float, str]] = []
    i = 0
    while i < n:
        if levels[i] not in want:
            i += 1
            continue
        lv = str(levels[i])
        j = i
        while j + 1 < n and str(levels[j + 1]) == lv and levels[j + 1] in want:
            j += 1
        t0, t1 = float(t[i]), float(t[j])
        spans.append((t0, t1, lv))
        i = j + 1
    return spans


def _span_limits(t0: float, t1: float, eps: float) -> tuple[float, float]:
    if t1 < t0:
        t0, t1 = t1, t0
    if abs(t1 - t0) < 1e-9:
        return t0 - eps, t0 + eps
    return t0, t1


def _resample_risk(t: np.ndarray, y: np.ndarray, tq: float) -> float:
    if len(t) == 0:
        return 0.0
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size == 0:
        return 0.0
    if tq <= float(np.min(t)):
        return float(y[np.argmin(t)])
    if tq >= float(np.max(t)):
        return float(y[np.argmax(t)])
    j = int(np.searchsorted(t, tq, side="right") - 1)
    j = int(np.clip(j, 0, len(t) - 2))
    t0, t1 = t[j], t[j + 1]
    w = (tq - t0) / (t1 - t0) if t1 > t0 else 0.0
    return float((1.0 - w) * y[j] + w * y[j + 1])


def _event_span(ev: dict[str, Any]) -> tuple[float, float] | None:
    t0 = ev.get("start_time", ev.get("startTime"))
    t1 = ev.get("end_time", ev.get("endTime"))
    if t0 is None or t1 is None:
        return None
    try:
        a, b = float(t0), float(t1)
    except (TypeError, ValueError):
        return None
    if b < a:
        a, b = b, a
    return a, b


def _level_event(ev: dict[str, Any]) -> str:
    return str(ev.get("event_level", ev.get("eventLevel", ""))).strip().upper()


def _event_label(i: int, ev: dict[str, Any]) -> str:
    eid = ev.get("event_id", i)
    return f"Event {eid}"


def plot_risk_timeline(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Plot ``risk_score`` versus time, shade **HIGH** / **CRITICAL** time ranges, and
    mark each detected event from ``events.json`` (if present) at the segment midpoint.

    Shading follows **per-frame** ``risk_level`` in ``risk_scores.csv`` when that column
    has HIGH/CRITICAL values; otherwise shading uses event time bounds in ``events.json``.

    The default output file is ``<run_dir>/risk_timeline.png`` when ``output_path`` is omitted.
    """
    run_dir = run_dir.expanduser().resolve()
    out = (
        output_path.expanduser().resolve()
        if output_path is not None
        else run_dir / "risk_timeline.png"
    )
    score_col = COL_RISK_SCORE
    df = _load_risk_frame(run_dir)
    if score_col not in df.columns and "risk_score" in df.columns:
        score_col = "risk_score"
    if score_col not in df.columns:
        raise ValueError("risk_scores.csv must contain a 'risk_score' column.")

    t_all = _time_seconds(df)
    y = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    order = np.argsort(t_all, kind="mergesort")
    t, y = t_all[order], y[order]
    df_s = df.iloc[order].reset_index(drop=True)
    levels = _norm_risk_level_column(df_s)

    events = _load_events_list(run_dir)
    eps = _time_axis_epsilon(t)
    want = frozenset({"HIGH", "CRITICAL"})

    t_min = float(np.nanmin(t)) if len(t) else 0.0
    t_max = float(np.nanmax(t)) if len(t) else 0.0

    use_frame = bool(np.any(np.isin(levels, list(want))))

    with rc_context(_ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
        if use_frame:
            for t0, t1, lv in _consecutive_risk_spans(t, levels, want):
                a, b = _span_limits(t0, t1, eps)
                col = _LEVEL_FILL.get(lv, _COLOR_HIGH)
                ax.axvspan(a, b, color=col, alpha=0.17, zorder=0)
        else:
            for ev in events:
                lev = _level_event(ev)
                if lev not in want:
                    continue
                sp = _event_span(ev)
                if sp is None:
                    continue
                a, b = sp
                if b < t_min - 1e-9 or a > t_max + 1e-9:
                    continue
                col = _COLOR_CRITICAL if lev == "CRITICAL" else _COLOR_HIGH
                ax.axvspan(a, b, color=col, alpha=0.17, zorder=0)

        (line,) = ax.plot(t, y, color=_COLOR_LINE, zorder=3, label="Risk score")

        for ev in events:
            sp = _event_span(ev)
            if sp is None:
                continue
            t0, t1 = sp
            tm = 0.5 * (t0 + t1)
            if tm < t_min - 1e-9 or tm > t_max + 1e-9:
                continue
            ym = _resample_risk(t, y, tm)
            ax.plot(
                tm,
                ym,
                marker="o",
                mfc=_COLOR_MARKER,
                mec="0.99",
                mew=0.5,
                ms=4.2,
                zorder=4,
                clip_on=True,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Risk score")
        ax.set_title("Risk score and elevated regions (detected events marked)")

        p_hi = Patch(
            facecolor=_COLOR_HIGH,
            alpha=0.35,
            edgecolor="none",
            label="HIGH region",
        )
        p_cr = Patch(
            facecolor=_COLOR_CRITICAL,
            alpha=0.4,
            edgecolor="none",
            label="CRITICAL region",
        )
        m_ev = ax.plot(
            [],
            [],
            "o",
            mfc=_COLOR_MARKER,
            mec="0.99",
            ms=4.2,
            linestyle="none",
            label="Event (list)",
        )[0]
        ax.legend(handles=[line, p_hi, p_cr, m_ev], loc="best", fontsize=8)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="png", facecolor="white")
        plt.close(fig)
    return out


def plot_events_timeline(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """
    One horizontal track per line in ``events.json``: bar from ``start_time`` to ``end_time``,
    tone by event level. Time axis is aligned to ``risk_scores.csv`` if available.

    Default file: ``<run_dir>/events_timeline.png``.
    """
    run_dir = run_dir.expanduser().resolve()
    out = (
        output_path.expanduser().resolve()
        if output_path is not None
        else run_dir / "events_timeline.png"
    )
    events = _load_events_list(run_dir)

    t_lo, t_hi = 0.0, 1.0
    rpath = run_dir / "risk_scores.csv"
    if rpath.is_file():
        try:
            df = _load_risk_frame(run_dir)
            tt = _time_seconds(df)
            if len(tt):
                t_lo = float(np.nanmin(tt))
                t_hi = float(np.nanmax(tt))
        except (OSError, ValueError) as e:
            logger.debug("Axis limits from risk_scores.csv: %s", e)

    for ev in events:
        sp = _event_span(ev)
        if sp is not None:
            t_lo = min(t_lo, sp[0])
            t_hi = max(t_hi, sp[1])
    if t_hi <= t_lo:
        t_hi = t_lo + 1.0

    h = max(2.4, 0.42 * max(len(events), 1) + 0.9)

    with rc_context(_ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(9, h), layout="constrained")
        if not events:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Events")
            ax.set_title("Events timeline (empty)")
            ax.text(
                0.5,
                0.5,
                "No events in events.json",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#5a5a5a",
            )
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        else:
            n = len(events)
            yc = np.arange(n, dtype=float)[::-1]
            for i, ev in enumerate(events):
                y0 = float(yc[i])
                sp = _event_span(ev)
                lv = _level_event(ev)
                col = _LEVEL_TONE_EVENT.get(lv, _COLOR_TIER_UNKNOWN)
                if sp is not None:
                    a, b = sp
                    ax.broken_barh(
                        [(a, max(b - a, 1e-6))],
                        (y0 - 0.38, 0.76),
                        facecolors=col,
                        edgecolors="#4a4a4a",
                        linewidth=0.45,
                    )
                else:
                    ax.text(
                        t_lo + 0.01 * (t_hi - t_lo),
                        y0,
                        "missing start/end",
                        va="center",
                        fontsize=8,
                        color="#333",
                    )
            ax.set_yticks(yc)
            ax.set_yticklabels([_event_label(i, ev) for i, ev in enumerate(events)], fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Reported event")
            ax.set_title("Events vs. time (by listed level)")

            present: set[str] = set()
            for ev in events:
                u = _level_event(ev)
                if u in _LEVEL_TONE_EVENT:
                    present.add(u)
                else:
                    present.add("OTHER")
            leg: list[Patch] = []
            for key in _LEGEND_ORDER:
                if key in present:
                    leg.append(
                        Patch(
                            facecolor=_LEVEL_TONE_EVENT.get(key, _COLOR_TIER_UNKNOWN),
                            edgecolor="#555",
                            linewidth=0.3,
                            label=key,
                        )
                    )
            if "OTHER" in present:
                leg.append(
                    Patch(
                        facecolor=_COLOR_TIER_UNKNOWN,
                        edgecolor="#555",
                        linewidth=0.3,
                        label="Other / unknown",
                    )
                )
            if leg:
                ax.legend(handles=leg, loc="lower right", fontsize=8)

        ax.set_xlim(t_lo, t_hi)
        if events:
            ax.margins(y=0.1)
        ax.grid(True, axis="x", alpha=0.75, linestyle="--", linewidth=0.55, color=_COLOR_GRID)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="png", facecolor="white")
        plt.close(fig)
    return out


def plot_event_timeline(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Alias for :func:`plot_events_timeline` (same file names and layout)."""
    return plot_events_timeline(run_dir, output_path)


def plot_pose_coverage(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Binary "pose available" signal vs. time, aligned to ``risk_scores.csv``.

    A frame is **covered** if its ``frame_id`` (from the risk table) appears in
    ``pose_keypoints.csv``. Default output: ``<run_dir>/pose_coverage.png`` when
    ``output_path`` is omitted.
    """
    run_dir = run_dir.expanduser().resolve()
    out = (
        output_path.expanduser().resolve()
        if output_path is not None
        else run_dir / "pose_coverage.png"
    )
    ppath = run_dir / "pose_keypoints.csv"
    rpath = run_dir / "risk_scores.csv"
    if not ppath.is_file():
        raise FileNotFoundError(f"Missing {ppath}")
    if not rpath.is_file():
        raise FileNotFoundError(f"Missing {rpath} (need time + frame_id alignment).")
    try:
        pdf = pd.read_csv(ppath, usecols=["frame_id"])
    except (ValueError, OSError, pd.errors.ParserError):
        pdf = pd.read_csv(ppath)
    if "frame_id" not in pdf.columns:
        raise ValueError("pose_keypoints.csv must include a 'frame_id' column.")
    pose_ids = set(pdf["frame_id"].astype(str).unique().tolist())
    rdf = _load_risk_frame(run_dir)
    t = _time_seconds(rdf)
    if "frame_id" in rdf.columns:
        fid = rdf["frame_id"].astype(str).to_numpy()
    else:
        fid = np.array([f"row_{i}" for i in range(len(rdf))], dtype=object)
    y = np.array([1.0 if str(f) in pose_ids else 0.0 for f in fid], dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]
    covered = int(np.nansum(y))
    n = len(y)

    with rc_context(_ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(9, 2.4), layout="constrained")
        ax.fill_between(
            t,
            0.0,
            y,
            step="post",
            color=_COLOR_TIER_MED,
            alpha=0.32,
            linewidth=0.0,
        )
        t_ok = t[np.isfinite(t)]
        if t_ok.size:
            ax.set_xlim(float(np.min(t_ok)), float(np.max(t_ok)))
        else:
            ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 1.0], ["no", "yes"], fontsize=8)
        ax.set_ylabel("Pose in table")
        ax.set_xlabel("Time (s)")
        ax.set_title("Per-frame pose coverage (intersection of risk and pose keypoint tables)")
        if n:
            pct = 100.0 * (covered / n)
            ann = f"With pose: {covered} / {n} frames  ({pct:.1f}%)"
        else:
            ann = "—"
        ax.text(
            0.99,
            0.1,
            ann,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4a4a4a",
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="png", facecolor="white")
        plt.close(fig)
    return out


__all__ = [
    "plot_event_timeline",
    "plot_events_timeline",
    "plot_pose_coverage",
    "plot_risk_timeline",
]
