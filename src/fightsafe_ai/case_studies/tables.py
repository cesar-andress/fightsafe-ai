"""LaTeX / {CSV} tables for case-study runs."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _escape_latex(s: str) -> str:
    t = (s or "").replace("\n", " ").replace("\\", r"\textbackslash{}")
    for a, b in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        t = t.replace(a, b)
    return t


def _parse_triggered(v: Any) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        t = v.strip()
        if t.startswith("[") and t.endswith("]"):
            try:
                p = ast.literal_eval(t)
                if isinstance(p, (list, tuple)):
                    return [str(x) for x in p]
            except (SyntaxError, ValueError, TypeError):
                return [t] if t else []
        return [t] if t else []
    return [str(v)]


def _rules_in_window(risk_df: pd.DataFrame, start: float, end: float, *, n_max: int = 12) -> str:
    if risk_df is None or len(risk_df) == 0 or "timestamp" not in risk_df.columns:
        return "—"
    ts = pd.to_numeric(risk_df["timestamp"], errors="coerce")
    w = risk_df.loc[(ts >= start) & (ts <= end)]
    if "triggered_rules" not in w.columns:
        return "—"
    u: set[str] = set()
    for v in w["triggered_rules"].tolist():
        u.update(_parse_triggered(v))
    if not u:
        return "—"
    out = sorted(u)[:n_max]
    s = ", ".join(out)
    if len(u) > n_max:
        s += ",…"
    return s


def _highest_risk_level(risk_df: pd.DataFrame | None) -> str:
    if risk_df is None or len(risk_df) == 0 or "risk_level" not in risk_df.columns:
        return "—"
    order = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    seen = {str(x).upper().split()[0] for x in risk_df["risk_level"].astype(str) if str(x).strip()}
    for o in order:
        if o in seen:
            return o
    return "—"


def load_events_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def load_qa_metrics(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "qa_report.json"
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(d, dict) and "metrics" in d and isinstance(d["metrics"], dict):
        return dict(d["metrics"])
    if isinstance(d, dict):
        m = d.get("metrics", {})
        return m if isinstance(m, dict) else {}
    return {}


def build_events_table_tex(case_run_dir: Path, risk_path: Path | None = None) -> str:
    """``booktabs``-style tabular (no table env) for a single case."""
    case_run_dir = case_run_dir.expanduser().resolve()
    evp = case_run_dir / "events.json"
    rp = risk_path or (case_run_dir / "risk_scores.csv")
    evs = load_events_list(evp)
    rdf: pd.DataFrame | None
    rdf = pd.read_csv(rp) if rp.is_file() else None
    if not evs:
        return (
            "\\begin{tabular}{@{}rrrrp{0.2\\linewidth}lp{0.3\\linewidth}@{}}\n"
            "\\toprule\n"
            "ID & Start (s) & End (s) & Duration (s) & Max risk & Level & Triggers/notes \\\\\n"
            "\\midrule\n"
            "\\multicolumn{7}{@{}l@{}}{\\textit{No high/critical event segments.}} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
    lines: list[str] = [
        "\\begin{tabular}{@{}rrrrp{0.15\\linewidth}lp{0.32\\linewidth}@{}}",
        "\\toprule",
        "ID & Start (s) & End (s) & Dur (s) & Max & Level & Triggers \\\\\n\\midrule",
    ]
    for e in evs:
        eid = e.get("event_id", 0)
        st = float(e.get("start_time", 0.0))
        et_ = float(e.get("end_time", 0.0))
        dur = float(et_ - st)
        mrs = float(e.get("max_risk_score", 0.0)) if e.get("max_risk_score") is not None else 0.0
        lv = str(e.get("event_level", "—"))
        trig = _rules_in_window(rdf, st, et_) if rdf is not None else "—"
        lines.append(
            f"{eid} & {st:.3f} & {et_:.3f} & {dur:.3f} & {mrs:.3f} & {lv} & {_escape_latex(trig)[:200]} \\\\"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    return "\n".join(lines)


def _row_for_summary(
    case_meta: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    m = load_qa_metrics(run_dir)
    rpath = run_dir / "risk_scores.csv"
    rdf = pd.read_csv(rpath) if rpath.is_file() else None
    evs = load_events_list(run_dir / "events.json")
    pcp = m.get("pose_coverage_percent")
    pcp = float(pcp) if pcp is not None else None
    return {
        "case_id": str(case_meta.get("case_id", "")),
        "focus": str(case_meta.get("expected_focus", "")),
        "total_frames": m.get("total_frames", ""),
        "pose_frames": m.get("frames_with_pose", ""),
        "pose_coverage_percent": "" if pcp is None else round(pcp, 2),
        "predicted_events": len(evs),
        "max_risk_score": m.get("max_risk_score", ""),
        "highest_risk_level": _highest_risk_level(rdf) if rdf is not None else "—",
        "notes": str(case_meta.get("notes", "")),
    }


def write_global_summaries(
    out_base: Path, rows: list[tuple[dict[str, Any], Path]]
) -> tuple[Path, Path]:
    """``rows``: ``(case yaml dict, run_dir)`` per successful case. Writes {CSV} + {TeX} table body."""
    out_base = out_base.expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    p_csv = out_base / "case_study_summary.csv"
    data = [_row_for_summary(c, d) for c, d in rows]
    fieldnames = (
        list(data[0].keys())
        if data
        else [
            "case_id",
            "focus",
            "total_frames",
            "pose_frames",
            "pose_coverage_percent",
            "predicted_events",
            "max_risk_score",
            "highest_risk_level",
            "notes",
        ]
    )
    with p_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in data:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tex = out_base / "case_study_summary.tex"
    tlines: list[str] = [
        "% Auto-generated; do not commit (under runs/; gitignored).",
        r"\begin{tabular}{@{}lrrrrlrl@{}}",
        r"\toprule",
        r"Case & Frames & Pose & Cov.\% & Events & MaxR & Lvl & Notes \\",
        r"\midrule",
    ]
    for r in data:
        notes = _escape_latex(str(r.get("notes", "")))[:60]
        tlines.append(
            f"{_escape_latex(str(r.get('case_id', '')))} & "
            f"{r.get('total_frames', '')} & {r.get('pose_frames', '')} & "
            f"{r.get('pose_coverage_percent', '')} & {r.get('predicted_events', '')} & "
            f"{r.get('max_risk_score', '')} & {r.get('highest_risk_level', '—')} & {notes} \\\\"
        )
    tlines.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    tex.write_text("\n".join(tlines), encoding="utf-8")
    return p_csv, tex
