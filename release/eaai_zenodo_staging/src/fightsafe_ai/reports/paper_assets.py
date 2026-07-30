"""
Build LaTeX table fragments and run-level stats from a pipeline run directory
(``runs/<name>/``) for use with ``\\input{}`` in the living paper.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fightsafe_ai.pipeline.output_paths import paths_for_run_root
from fightsafe_ai.qa import dataset_checks
from fightsafe_ai.reports.summary import load_events_list


def latex_escape(text: str) -> str:
    r"""Escape user-facing strings for LaTeX (table cells, ``\texttt{...}``; not for ``\path{...}``)."""
    s = str(text)
    s = s.replace(r"\\", r"\xDUMMYBSL\x")
    s = s.replace(r"&", r"\&")
    s = s.replace(r"%", r"\%")
    s = s.replace(r"#", r"\#")
    s = s.replace(r"$", r"\$")
    s = s.replace(r"_", r"\_")
    s = s.replace(r"{", r"\{")
    s = s.replace(r"}", r"\}")
    s = s.replace(r"^", r"\textasciicircum{}")
    s = s.replace(r"~", r"\textasciitilde{}")
    s = s.replace(r"\xDUMMYBSL\x", r"{\textbackslash}")
    return s


def _fmt_int_latex(n: int) -> str:
    s = f"{n:,}"
    return s.replace(",", r"\,")


def _label_safe_tag(tag: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9-]", "-", str(tag).strip() or "run")
    t = t.strip("-") or "run"
    return t


def _infer_extraction_fps(risk_df: pd.DataFrame) -> str:
    if len(risk_df) < 2 or "timestamp" not in risk_df.columns:
        return "—"
    t = pd.to_numeric(risk_df["timestamp"], errors="coerce")
    d = t.diff()
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return "—"
    med = float(np.median(d))
    if med <= 0 or not np.isfinite(med):
        return "—"
    v = 1.0 / med
    fps_s = f"{v:.2f}"
    fps_s = fps_s.rstrip("0").rstrip(".")
    return fps_s if fps_s else f"{v:.0f}"


def compute_paper_run_metrics(run_dir: Path, run_path_display: str | None = None) -> dict[str, Any]:
    """
    Aggregate engine metrics, pose coverage, and file tallies for ``<tag>_summary.tex`` tables.
    """
    p = run_dir.expanduser().resolve()
    paths = paths_for_run_root(p)
    n_risk = 0
    if paths.risk_scores_csv.is_file():
        try:
            n_risk = len(pd.read_csv(paths.risk_scores_csv))
        except (OSError, ValueError, pd.errors.ParserError):
            n_risk = 0
    n_frames_img = dataset_checks.count_frame_images(paths.frames_dir)
    n_total = n_risk if n_risk > 0 else n_frames_img
    n_pose_rows = 0
    n_unique_pose = 0
    pose_path = paths.pose_keypoints_csv
    if pose_path.is_file():
        try:
            pose_df = pd.read_csv(pose_path)
        except (OSError, ValueError, pd.errors.ParserError):
            pose_df = None
        if pose_df is not None and "frame_id" in pose_df.columns and len(pose_df) > 0:
            n_pose_rows = len(pose_df)
            n_unique_pose = int(pose_df["frame_id"].astype(str).nunique())
    if n_total == 0 and n_unique_pose > 0:
        n_total = n_unique_pose
    n_overlay = n_risk if n_risk else n_total

    n_feat = 0
    if paths.features_csv.is_file():
        try:
            n_feat = len(pd.read_csv(paths.features_csv))
        except (OSError, ValueError, pd.errors.ParserError):
            n_feat = 0

    numer: int = 0
    denom: int = 1
    cov: float | None = None
    if n_risk > 0:
        numer = n_unique_pose
        denom = n_risk
        cov = (float(numer) / float(denom)) if denom else None
    elif n_frames_img > 0 and pose_path.is_file():
        pct, meta = dataset_checks.pose_coverage_metrics(paths.frames_dir, pose_path)
        if pct is not None and math.isfinite(pct):
            numer = int(meta.get("n_unique_pose_frame_ids", n_unique_pose))
            denom = int(meta.get("n_frame_image_files", n_frames_img) or 1)
        else:
            numer, denom = n_unique_pose, max(n_frames_img, 1)
        cov = float(numer) / float(denom) if denom else None
    else:
        denom = max(n_total, 1)
        numer = n_unique_pose
        cov = (float(numer) / float(denom)) if denom else None

    metric_note: str | None = None
    if n_risk > 0 and n_unique_pose > n_risk:
        metric_note = "More unique pose frame IDs than risk rows; check alignment before interpreting coverage."

    events = load_events_list(paths.events_json)
    ex_fps = "—"
    if paths.risk_scores_csv.is_file():
        try:
            risk = pd.read_csv(paths.risk_scores_csv)
        except (OSError, ValueError, pd.errors.ParserError):
            risk = None
        if risk is not None and len(risk) > 0:
            ex_fps = _infer_extraction_fps(risk)
    else:
        risk = None

    ok_artifacts = bool(paths.pose_keypoints_csv.is_file() and paths.risk_scores_csv.is_file())
    rdisplay = run_path_display or (str(p).replace(p.anchor, "").lstrip("/") or str(p))
    return {
        "run_root_display": rdisplay,
        "n_total_video_frames": n_total,
        "n_overlay_output_frames": n_overlay,
        "n_frame_image_files": n_frames_img,
        "n_pose_keypoint_rows": n_pose_rows,
        "n_frames_with_pose": n_unique_pose,
        "n_risk_rows": n_risk,
        "n_feature_rows": n_feat,
        "has_features": n_feat > 0,
        "pose_coverage_ratio": cov,
        "pose_coverage_numer": numer,
        "pose_coverage_denom": denom,
        "pose_coverage_pct": 100.0 * cov if cov is not None and math.isfinite(cov) else None,
        "extraction_fps": ex_fps,
        "overlay_fps": ex_fps,
        "n_events": len(events),
        "run_status": "Success" if ok_artifacts else "Partial",
        "metric_note": metric_note,
    }


def _cov_cell(m: dict[str, Any]) -> str:
    n = m.get("pose_coverage_numer", 0)
    d = m.get("pose_coverage_denom", 0)
    if isinstance(n, (int, float)) and isinstance(d, (int, float)) and int(d) > 0:
        frac = f"{_fmt_int_latex(int(n))}/{_fmt_int_latex(int(d))}"
    else:
        frac = "—"
    pct = m.get("pose_coverage_pct")
    if isinstance(pct, (int, float, np.floating)) and pct is not None and math.isfinite(float(pct)):
        p2 = min(float(pct), 100.0) if float(pct) > 100.0 else float(pct)
        return f"$ {frac} $ ($\\sim${p2:.1f}\\%)" if frac != "—" else f"$\\sim${p2:.1f}\\%"
    r0 = m.get("pose_coverage_ratio")
    if isinstance(r0, (int, float, np.floating)) and r0 is not None and math.isfinite(float(r0)):
        r0f = float(r0)
        if 0.0 < r0f <= 1.0:
            return f"$ {frac} $ (share $\\approx$ {r0f * 100:.1f}\\%)"
    if frac != "—":
        return f"$ {frac} $"
    return "—"


def write_summary_tex(output_path: Path, *, tag: str, m: dict[str, Any]) -> Path:
    r_display = str(m.get("run_root_display", ""))
    ccap = rf"\caption{{FightSafe run: engineering summary (root \path{{{r_display}}}).}}"
    ltag = _label_safe_tag(tag)
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        ccap,
        rf"\label{{tab:paper-{ltag}-summary}}",
        r"\begin{tabular}{@{}p{0.24\linewidth}p{0.20\linewidth}p{0.48\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} & \textbf{Interpretation} \\",
        r"\midrule",
    ]
    ef = str(m.get("extraction_fps", "—"))
    lines.append(f"Extraction {{FPS}} & {latex_escape(ef)} & Sampling rate for stored frames. \\\\")
    lines.append(
        f"Total video frames & {_fmt_int_latex(int(m.get('n_total_video_frames', 0) or 0))} & "
        "Frames along the output / overlay timeline. \\\\"
    )
    lines.append(
        f"Pose {{CSV}} row count (keypoints) & "
        f"{_fmt_int_latex(int(m.get('n_pose_keypoint_rows', 0) or 0))} & "
        "Long-format keypoint table. \\\\"
    )
    lines.append(
        f"Frames with pose available & "
        f"{_fmt_int_latex(int(m.get('n_frames_with_pose', 0) or 0))} & "
        "Frame IDs present in the pose table. \\\\"
    )
    lines.append(
        f"Overlay output frames & "
        f"{_fmt_int_latex(int(m.get('n_overlay_output_frames', 0) or 0))} & "
        "One overlay output row per risk timeline step when the pipeline wrote both. \\\\"
    )
    ofps = str(m.get("overlay_fps", "—"))
    lines.append(
        f"Overlay {{FPS}} & {latex_escape(ofps)} & Matches regular sampling when timestamps are uniform. \\\\"
    )
    ccell = _cov_cell(m)
    itxt = "Share of timeline frames for which a pose was exported"
    nnote = m.get("metric_note")
    if nnote:
        itxt = f"{itxt}. {nnote}"
    lines.append(f"Pose coverage & {ccell} & {latex_escape(itxt)}. \\\\")
    lines.append(r"Pose backend & {MediaPipe} & Default {MVP} stack. \\")
    rs = m.get("run_status", "—")
    lines.append(
        f"Run status & {latex_escape(str(rs))} & " r"Artefact checks for the run directory. \\"
    )
    if m.get("has_features"):
        lines.append(
            f"Feature table rows & {_fmt_int_latex(int(m.get('n_feature_rows', 0) or 0))} & "
            r"Biomechanics and temporal features (\path{features.csv}). \\"
        )
    lines.append(
        f"Detected events & {_fmt_int_latex(int(m.get('n_events', 0) or 0))} & "
        r"High-risk or review segments in \path{events.json}. \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_artifacts_tex(output_path: Path, *, tag: str, m: dict[str, Any]) -> Path:
    rbase = str(m.get("run_root_display", "runs/…"))
    ccap = rf"\caption{{Primary run artefacts (paths under \path{{{rbase}}}).}}"
    ltag = _label_safe_tag(tag)
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        ccap,
        rf"\label{{tab:paper-{ltag}-artifacts}}",
        r"\begin{tabular}{@{}p{0.28\linewidth}p{0.34\linewidth}p{0.30\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Artefact} & \textbf{Path (run root relative)} & \textbf{Role} \\",
        r"\midrule",
        r"Sampled frames & \path{frames/} & {JPEG} inputs to pose. \\",
        r"Keypoints {CSV} & \path{pose_keypoints.csv} & Per-landmark table. \\",
    ]
    if m.get("has_features"):
        lines.append(
            r"Feature table & \path{features.csv} & Biomechanics and temporal features. \\"
        )
    else:
        lines.append(
            r"Feature table & \path{features.csv} (if present) & "
            "Written when the feature step ran. \\"
        )
    lines.extend(
        [
            r"Risk table & \path{risk_scores.csv} & {MVP} risk score and level over time. \\",
            r"Events & \path{events.json} & Detected review intervals. \\",
            r"Overlay video & \path{output_overlay.mp4} & {RGB} + cues for triage. \\",
            r"Narrative report & \path{report.md} & {Markdown} summary. \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_events_tex(
    output_path: Path,
    *,
    tag: str,
    run_dir: Path,
) -> Path:
    """``demo_events.tex`` from :func:`fightsafe_ai.reports.summary.load_events_list`."""
    p = run_dir.expanduser().resolve()
    paths = paths_for_run_root(p)
    evs = load_events_list(paths.events_json)
    ltag = _label_safe_tag(tag)
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detected high-risk or review events (\path{events.json}).}",
        rf"\label{{tab:paper-{ltag}-events}}",
        r"\begin{tabular}{@{}rllrrr@{}}",
        r"\toprule",
        r"\textbf{Id} & \textbf{Level} & \textbf{Type} & \textbf{Start (s)} & \textbf{End (s)} & \textbf{Max risk} \\",
        r"\midrule",
    ]
    if not evs:
        lines.append(r"\multicolumn{6}{@{}l@{}}{No events in \path{events.json}.} \\")

    def _fnum(x: Any) -> str:
        if x is None:
            return "—"
        try:
            v = float(x)
        except (TypeError, ValueError):
            return latex_escape(str(x))
        if not math.isfinite(v):
            return "—"
        s0 = f"{v:.3f}"
        s0 = s0.rstrip("0").rstrip(".")
        return s0 if s0 else f"{v:.0f}"

    for i, ev in enumerate(evs):
        eid = ev.get("event_id", i)
        lv = str(ev.get("event_level", ev.get("eventLevel", "—"))).strip()
        et = str(ev.get("event_type", ev.get("eventType", "—"))).strip()
        t0 = ev.get("start_time", ev.get("startTime"))
        t1 = ev.get("end_time", ev.get("endTime"))
        mx = ev.get("max_risk_score", ev.get("maxRiskScore"))
        t0s, t1s, mxs = _fnum(t0), _fnum(t1), _fnum(mx)
        lines.append(
            f"{latex_escape(str(eid))} & {latex_escape(lv)} & {latex_escape(et)} & "
            f"{t0s} & {t1s} & {mxs} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
