"""
Self-contained HTML reports (no third-party web assets) for a pipeline run directory.
"""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fightsafe_ai._metadata import AUTHORS_BIBLIOGRAPHIC, REPORT_END_ATTRIBUTION
from fightsafe_ai.reports.markdown import build_ai_explanations_html_fragment
from fightsafe_ai.reports.summary import (
    build_summary_dict,
    infer_input_video_path,
    load_events_list,
    load_qa_dict,
    load_risk_dataframe,
)


logger = logging.getLogger(__name__)

_DISCLAIMER_TEXT = (
    "This system provides decision-support only and does not perform medical diagnosis."
)

_TIMELINE_CANDIDATES: tuple[str, ...] = (
    "risk_timeline.png",
    "plots/risk_timeline.png",
)


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _rel_asset(run_dir: Path, html_path: Path, *parts: str) -> str | None:
    p = run_dir.joinpath(*parts) if parts else run_dir
    if not p.is_file():
        return None
    try:
        return Path(os.path.relpath(p.resolve(), html_path.parent.resolve())).as_posix()
    except ValueError:
        return p.name


def _risk_statistics(df: pd.DataFrame | None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "n_frames": 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "level_counts": {},
    }
    if df is None or "risk_score" not in df.columns or len(df) == 0:
        return d
    s = pd.to_numeric(df["risk_score"], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        d["n_frames"] = len(df)
        return d
    d["n_frames"] = len(s)
    d["min"] = float(s.min())
    d["max"] = float(s.max())
    d["mean"] = float(s.mean())
    d["std"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
    if "risk_level" in df.columns:
        d["level_counts"] = (
            df["risk_level"].astype(str).str.strip().str.upper().value_counts().to_dict()
        )
    return d


def _highest_risk_moment(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or "risk_score" not in df.columns or len(df) == 0:
        return None
    s = pd.to_numeric(df["risk_score"], errors="coerce")
    if s.isna().all():
        return None
    s_arr = np.asarray(s, dtype=float)
    j = int(s_arr.argmax()) if len(s_arr) else 0
    row = df.iloc[j]
    out: dict[str, Any] = {
        "row_index": j,
        "risk_score": float(s.iloc[j]) if pd.notna(s.iloc[j]) else None,
    }
    for c in ("frame_id", "timestamp", "time", "risk_level"):
        if c in row.index and pd.notna(row[c]):
            out[c] = row[c]
    return out


def _events_table_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return "<p class='empty'><em>No events in <code>events.json</code>.</em></p>"
    keys = (
        "event_id",
        "event_level",
        "start_time",
        "end_time",
        "max_risk_score",
        "duration_seconds",
    )
    header = "<thead><tr>" + "".join(f"<th>{_esc(k)}</th>" for k in keys) + "</tr></thead>"
    rows: list[str] = []
    for ev in events:
        tds = "".join(f"<td>{_esc(ev.get(k, '—'))}</td>" for k in keys)
        rows.append(f"<tr>{tds}</tr>")
    return f"<table class='data'>{header}<tbody>{''.join(rows)}</tbody></table>"


def _qa_block_html(qa: dict[str, Any] | None) -> str:
    if not qa:
        return "<p class='empty'><em>No <code>qa_report.json</code> in this run directory.</em></p>"
    st = "pass" if qa.get("passed") is True else "fail" if qa.get("passed") is False else "unknown"
    line = (
        f"<p><strong>Overall:</strong> <span class='qa q-{st}'>{_esc(st)}</span> · "
        f"total checks: {_esc(qa.get('total_checks', '—'))} · "
        f"failed: {_esc(qa.get('failed_checks', '—'))}"
    )
    warns = qa.get("warnings")
    if isinstance(warns, list) and warns:
        wul = "<ul class='warn'>" + "".join(f"<li>{_esc(x)}</li>" for x in warns) + "</ul>"
        line += f" · Warnings: {wul}</p>"
    else:
        line += "</p>"

    res = qa.get("results")
    if not isinstance(res, list) or not res:
        return line
    h = "<thead><tr><th>Check</th><th>Status</th><th>Message</th></tr></thead>"
    trs: list[str] = []
    for it in res:
        if not isinstance(it, dict):
            continue
        raw_st = str(it.get("status", "—"))
        scls = f"st-{raw_st}" if raw_st in ("pass", "fail", "warn", "skip") else "st-oth"
        trs.append(
            "<tr>"
            f"<td>{_esc(it.get('name', '—'))}</td>"
            f"<td class='{scls}'>{_esc(raw_st)}</td>"
            f"<td class='msg'>{_esc(it.get('message', '—'))}</td>"
            "</tr>"
        )
    if not trs:
        return line
    return f"{line}<table class='data'>{h}<tbody>{''.join(trs)}</tbody></table>"


def _stats_list_html(st: dict[str, Any]) -> str:
    bits = [f"<li>Frames in risk table: <strong>{_esc(st.get('n_frames', 0))}</strong></li>"]
    for k, label in (("min", "Min"), ("max", "Max"), ("mean", "Mean"), ("std", "St. dev.")):
        v = st.get(k)
        if v is not None and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            bits.append(
                f"<li>{_esc(label)} risk: <code>{_esc(round(float(v), 6) if k != 'std' else float(v))}</code></li>"
            )
    lc = st.get("level_counts")
    if isinstance(lc, dict) and lc:
        inner = ", ".join(f"{_esc(k)}: {int(v)}" for k, v in sorted(lc.items(), key=str))
        bits.append(f"<li>Frame counts by <code>risk_level</code>: {inner}</li>")
    return "<ul class='k'>" + "".join(bits) + "</ul>"


def _highest_html(h: dict[str, Any] | None) -> str:
    if not h or h.get("risk_score") is None:
        return (
            "<p class='empty'><em>Could not determine a single highest-risk frame "
            "from <code>risk_scores.csv</code>.</em></p>"
        )
    parts = [f"risk score <code>{_esc(h['risk_score'])}</code>"]
    if "timestamp" in h:
        parts.append(f"time (s) <code>{_esc(h['timestamp'])}</code>")
    elif "time" in h:
        parts.append(f"time <code>{_esc(h['time'])}</code>")
    if "frame_id" in h:
        parts.append(f"<code>frame_id</code> {_esc(h['frame_id'])}")
    if "risk_level" in h:
        parts.append(f"level {_esc(h['risk_level'])}")
    return f"<p>{' · '.join(parts)}</p>"


def generate_html_report(run_dir: Path, output_path: Path) -> Path:
    """
    Write a self-contained HTML page: run summary, input video name (when present),
    risk statistics, event table, optional AI/rule-based explanation block, highest-risk
    moment, QA, embedded risk timeline (if found), and link to ``output_overlay.mp4`` when present.

    All asset links are relative to ``output_path``'s parent. No external
    JavaScript, fonts, or stylesheets. Includes a decision-support disclaimer in the
    page body and a short author line in the footer.
    """
    run_path = run_dir.expanduser().resolve()
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sm = build_summary_dict(run_path)
    events = load_events_list(run_path / "events.json")
    risk = load_risk_dataframe(run_path / "risk_scores.csv")
    qa = load_qa_dict(run_path / "qa_report.json")
    rstats = _risk_statistics(risk)
    peak = _highest_risk_moment(risk)
    video_guess = infer_input_video_path(run_path / "report.md") or ""
    if not (video_guess and str(video_guess).strip()):
        video_guess = "— (see run folder or re-run pipeline with a named input file)"

    img_href: str | None = None
    for cand in _TIMELINE_CANDIDATES:
        img_href = _rel_asset(run_path, out, *Path(cand).parts)
        if img_href:
            break
    video_href = _rel_asset(run_path, out, "output_overlay.mp4")

    title = f"Report — {_esc(sm.get('clip_id', run_path.name))}"
    disc = f'<p class="disclaimer">{_esc(_DISCLAIMER_TEXT)}</p>'

    fig = ""
    if img_href:
        fig = f"""
  <section class="fig" id="timeline">
    <h2>Risk timeline (embedded)</h2>
    <p class="figcap"><code>risk_timeline.png</code> (relative to this file)</p>
    <p><img src="{_esc(img_href)}" alt="Risk score over time" loading="lazy" width="100%" /></p>
  </section>"""
    else:
        fig = """
  <section class="fig" id="timeline">
    <h2>Risk timeline</h2>
    <p class="empty"><em>No <code>risk_timeline.png</code> under the run (generate with the visualization step).</em></p>
  </section>"""

    if video_href:
        vblock = f'<p><a href="{_esc(video_href)}">Open <code>output_overlay.mp4</code> (skeleton + risk overlay)</a></p>'
    else:
        vblock = "<p class='empty'><em>Overlay <code>output_overlay.mp4</code> not found in the run.</em></p>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.55; max-width: 50rem; margin: 0 auto; padding: 1.5rem 1.25rem 2.5rem;
      color: #1a1a1a; background: #fff; }}
    h1 {{ font-size: 1.45rem; font-weight: 600; margin: 0 0 0.5rem; }}
    h2 {{ font-size: 1.05rem; font-weight: 600; margin: 1.6rem 0 0.5rem; color: #222; }}
    .disclaimer {{ background: #f3f3f3; border-left: 3px solid #333; padding: 0.65rem 0.9rem; margin: 0.5rem 0 1.25rem; font-size: 0.9rem; }}
    .meta {{ color: #444; font-size: 0.9rem; margin: 0 0 0.4rem; }}
    code {{ background: #f0f0f0; padding: 0.1em 0.35em; border-radius: 2px; font-size: 0.88em; }}
    p {{ margin: 0.4rem 0; }}
    section {{ margin: 0.2rem 0; }}
    table.data {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 0.5rem 0; }}
    table.data th, table.data td {{ border: 1px solid #c8c8c8; padding: 0.35rem 0.5rem; text-align: left; }}
    table.data thead {{ background: #f5f5f5; font-weight: 600; }}
    .sum th {{ width: 14rem; font-weight: 500; color: #333; }}
    ul.k {{ margin: 0.35rem 0; padding-left: 1.2rem; }}
    .empty {{ color: #666; font-style: italic; }}
    .fig img {{ max-width: 100%; height: auto; display: block; border: 1px solid #ccc; background: #fafafa; }}
    .figcap {{ color: #666; font-size: 0.85rem; margin: 0.15rem 0; }}
    .q-pass, span.q-pass {{ color: #1a3d1a; font-weight: 500; }}
    .q-fail, span.q-fail {{ color: #6a1a1a; font-weight: 500; }}
    .q-unknown, span.q-unknown {{ color: #444; }}
    td.st-pass {{ color: #1a3d1a; font-weight: 500; }}
    td.st-fail {{ color: #6a1a1a; font-weight: 500; }}
    td.st-warn {{ color: #5a4a00; font-weight: 500; }}
    td.msg {{ max-width: 32rem; }}
    ul.warn {{ font-size: 0.9rem; color: #444; }}
    .footer {{ margin-top: 2rem; font-size: 0.8rem; color: #666; border-top: 1px solid #e2e2e2; padding-top: 0.9rem; }}
    .ai-disclaimer {{ background: #f7f2e6; border-left: 3px solid #8a6d3b; padding: 0.6rem 0.85rem; font-size: 0.88rem; margin: 0.5rem 0 1rem; }}
    .ai-event {{ margin: 1rem 0 1.25rem; padding: 0.5rem 0; border-top: 1px solid #e5e5e5; }}
    .ai-event:first-of-type {{ border-top: none; }}
    h4 .src-tag {{ font-size: 0.78rem; font-weight: 500; color: #555; }}
    ul.ai-bullets {{ margin: 0.4rem 0 0.6rem; padding-left: 1.2rem; }}
    ul.ai-bullets li {{ margin: 0.25rem 0; }}
    .ai-narrative {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5rem 0.75rem; font-size: 0.9rem; margin-top: 0.5rem; }}
    .ai-narrative p {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  {disc}
  <h1>Pipeline run report</h1>
  <p class="meta">Run directory: <code>{_esc(str(run_path))}</code></p>
  <section>
    <h2>Run summary</h2>
    <table class="data sum">
      <tr><th>Run id / clip</th><td><code>{_esc(sm.get("clip_id", ""))}</code></td></tr>
      <tr><th>Input video (from <code>report.md</code> if present)</th><td>{_esc(str(video_guess))}</td></tr>
      <tr><th>Total frames (approx.)</th><td>{_esc(sm.get("total_frames", 0))}</td></tr>
      <tr><th>Time span (s) from <code>risk_scores.csv</code></th><td>{_esc(sm.get("duration_seconds", 0.0))}</td></tr>
      <tr><th>Event count</th><td>{_esc(sm.get("number_of_events", 0))}</td></tr>
      <tr><th>Highest event level (from <code>events.json</code>)</th><td>{_esc(sm.get("highest_event_level", "—"))}</td></tr>
      <tr><th>QA (summary only)</th><td><code>{_esc(sm.get("qa_status", "unknown"))}</code></td></tr>
    </table>
  </section>
  <section>
    <h2>Risk statistics (from <code>risk_scores.csv</code>)</h2>
    {_stats_list_html(rstats)}
  </section>
  <section>
    <h2>Detected events</h2>
    {_events_table_html(events)}
  </section>
  <section id="ai-explanations">
    <h2>AI Explanation (Optional)</h2>
    {build_ai_explanations_html_fragment(run_path, events)}
  </section>
  <section>
    <h2>Highest risk moment</h2>
    {_highest_html(peak if isinstance(peak, dict) else None)}
  </section>
  <section>
    <h2>QA results</h2>
    {_qa_block_html(qa)}
  </section>
  <section>
    <h2>Overlay video</h2>
    {vblock}
  </section>
  {fig}
  <p class="footer">
    {_esc(AUTHORS_BIBLIOGRAPHIC)} ·
    <span>Not a medical device.</span>
    {_esc(REPORT_END_ATTRIBUTION)}
  </p>
</body>
</html>"""

    out.write_text(doc, encoding="utf-8")
    logger.info("Wrote HTML report: %s", out)
    return out


__all__ = ["generate_html_report"]
