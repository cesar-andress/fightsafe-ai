"""
Machine-readable run summary (JSON) from pipeline artifacts in a run directory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fightsafe_ai._metadata import DECISION_SUPPORT_SCOPE, REPORT_ATTRIBUTION_JSON


logger = logging.getLogger(__name__)

EVENT_LEVEL_ORDER: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


def load_risk_dataframe(risk_path: Path) -> pd.DataFrame | None:
    if not risk_path.is_file():
        return None
    try:
        return pd.read_csv(risk_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.error("Could not read risk_scores.csv: %s", exc)
        return None


def load_events_list(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.is_file():
        return []
    try:
        raw = json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read events.json: %s", exc)
        return []
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def load_qa_dict(qa_path: Path) -> dict[str, Any] | None:
    if not qa_path.is_file():
        return None
    try:
        raw = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read qa_report.json: %s", exc)
        return None
    return raw if isinstance(raw, dict) else None


def infer_input_video_path(report_md_path: Path) -> str | None:
    if not report_md_path.is_file():
        return None
    try:
        text = report_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("No report.md: %s", exc)
        return None
    m = re.search(r"`([^`]*\.(?:mp4|mov|mkv|avi|webm))`", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(
        r"(?:input|source|video)[^\n`]{0,80}`([^`]+)`",
        text,
        re.IGNORECASE,
    )
    if m2 and "." in m2.group(1):
        return m2.group(1).strip()
    return None


def _duration_from_risk_df(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) == 0:
        return 0.0
    t = pd.to_numeric(df["timestamp"], errors="coerce")
    t = t.dropna()
    if len(t) == 0:
        return 0.0
    return float(t.max() - t.min())


def _max_risk(df: pd.DataFrame) -> float | None:
    if "risk_score" not in df.columns or len(df) == 0:
        return None
    s = pd.to_numeric(df["risk_score"], errors="coerce")
    if s.isna().all():
        return None
    return float(s.max())


def _highest_event_level(events: list[dict[str, Any]]) -> str | None:
    if not events:
        return None
    best: str | None = None
    best_n = 0
    for ev in events:
        raw = str(ev.get("event_level", "")).strip().upper()
        n = EVENT_LEVEL_ORDER.get(raw, 0)
        if n > best_n:
            best_n = n
            best = raw
    return best


def _merge_llm_explanation_fields(run_dir: Path, payload: dict[str, Any]) -> None:
    """Attach ``llm_*`` keys from ``llm_explanation_state.json`` when present."""
    p = run_dir / "llm_explanation_state.json"
    if not p.is_file():
        return
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return
    if not isinstance(raw, dict):
        return
    for key in ("llm_requested", "llm_available", "llm_fallback", "llm_error"):
        if key in raw:
            payload[key] = raw[key]


def build_summary_dict(run_dir: Path) -> dict[str, Any]:
    """
    Build the canonical summary payload (without writing to disk).

    Fields align with :func:`generate_summary_json` output.
    """
    run_dir = run_dir.expanduser().resolve()
    risk_path = run_dir / "risk_scores.csv"
    events_path = run_dir / "events.json"
    qa_path = run_dir / "qa_report.json"

    risk_df = load_risk_dataframe(risk_path)
    events = load_events_list(events_path)
    qa = load_qa_dict(qa_path)

    total_frames = 0
    if risk_df is not None and len(risk_df) > 0:
        if "frame_id" in risk_df.columns:
            total_frames = int(risk_df["frame_id"].astype(str).nunique())
        else:
            total_frames = len(risk_df)
    duration_seconds = _duration_from_risk_df(risk_df) if risk_df is not None else 0.0
    max_risk = _max_risk(risk_df) if risk_df is not None else None
    hel = _highest_event_level(events)

    qa_status = "unknown"
    if qa is not None:
        if qa.get("passed") is True:
            qa_status = "pass"
        elif qa.get("passed") is False:
            qa_status = "fail"
        else:
            qa_status = "unknown"

    payload = {
        "summary_schema": "1.0",
        "decision_support_scope": DECISION_SUPPORT_SCOPE,
        "attribution": dict(REPORT_ATTRIBUTION_JSON),
        "clip_id": run_dir.name,
        "total_frames": int(total_frames),
        "duration_seconds": round(float(duration_seconds), 6),
        "max_risk_score": (
            round(float(max_risk), 6) if max_risk is not None and np.isfinite(max_risk) else None
        ),
        "number_of_events": len(events),
        "highest_event_level": hel,
        "qa_status": qa_status,
    }
    _merge_llm_explanation_fields(run_dir, payload)
    return payload


def generate_summary_json(run_dir: Path, output_path: Path) -> Path:
    """
    Write a JSON file with run-level metrics: clip id, frame count, duration, max risk, events, QA.
    """
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_summary_dict(run_dir)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "build_summary_dict",
    "generate_summary_json",
    "infer_input_video_path",
    "load_events_list",
    "load_qa_dict",
    "load_risk_dataframe",
]
