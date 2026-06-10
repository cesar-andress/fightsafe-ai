"""
Academic Markdown report for FightSafe AI MVP run artifacts.

Consumes ``events.json``, ``risk_scores.csv``, and metadata paths; optionally references
Ollama-generated explanation files or inline narrative text.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fightsafe_ai._metadata import REPORT_END_ATTRIBUTION, SAFETY_REPORT_MD_HEADER
from fightsafe_ai.risk.rules import COMBAT_MVP_INDICATOR_LABELS


def _path_rel(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path)


def _parse_triggered_cell(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and (np.isnan(cell) or pd.isna(cell))):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                return [s]
        return [s]
    return [str(cell)]


def _load_events(path: Path) -> list[dict[str, Any]]:
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


def _aggregate_triggered_rules(df: pd.DataFrame, col: str = "triggered_rules") -> Counter[str]:
    counts: Counter[str] = Counter()
    if col not in df.columns or len(df) == 0:
        return counts
    for cell in df[col]:
        for name in _parse_triggered_cell(cell):
            if name:
                counts[name] += 1
    return counts


def _highest_risk_row(df: pd.DataFrame) -> pd.Series | None:
    if len(df) == 0 or "risk_score" not in df.columns:
        return None
    s = pd.to_numeric(df["risk_score"], errors="coerce")
    if s.isna().all():
        return None
    idx = int(s.idxmax())
    row = df.loc[idx]
    if not isinstance(row, pd.Series):
        return None
    return row


@dataclass(frozen=True)
class MVPReportConfig:
    """Inputs for :func:`generate_mvp_report_markdown`."""

    video_path: Path
    output_root: Path
    events_path: Path
    risk_scores_path: Path
    rules_config_path: Path | None = None
    sampling_fps: float | None = None
    n_sampled_frames: int | None = None
    optional_ollama_narrative: str = ""
    explanations_dir: Path | None = None


def generate_mvp_report_markdown(cfg: MVPReportConfig) -> str:
    """
    Build a structured academic report (Markdown) for an MVP pipeline run.

    Sections: demo clip, pipeline outputs, detected events, highest-risk moment,
    triggered-rule statistics, human-review guidance, limitations, safety disclaimer.
    """
    video_path = cfg.video_path.expanduser().resolve()
    output_root = cfg.output_root.expanduser().resolve()
    events = _load_events(cfg.events_path.expanduser().resolve())

    risk_path = cfg.risk_scores_path.expanduser().resolve()
    if risk_path.is_file():
        rdf = pd.read_csv(risk_path)
    else:
        rdf = pd.DataFrame()

    lines: list[str] = [
        SAFETY_REPORT_MD_HEADER.rstrip(),
        "",
        f"*Generated (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}*",
        "",
        "## 1. Demo clip",
        "",
    ]

    lines.append(
        f"The analysis referent is the video file **`{video_path.name}`** (path: `{video_path}`). "
    )
    if cfg.sampling_fps is not None and cfg.n_sampled_frames is not None:
        lines.append(
            f"Frames were sampled for processing at **{cfg.sampling_fps:g} Hz**, yielding "
            f"**{cfg.n_sampled_frames}** extracted frame(s) for pose and risk estimation. "
        )
    lines.append(
        "This segment should be understood as a **laboratory or research clip**, not as "
        "a complete match record."
    )
    lines.append("")

    lines.extend(
        [
            "## 2. Pipeline outputs",
            "",
            f"The following artifacts were produced under the run directory `{output_root}`:",
            "",
            "| Relative path | Description |",
            "| --- | --- |",
            "| `frames/` | Sampled JPEG frames from the input video |",
            "| `pose_keypoints.csv` | Consolidated MediaPipe pose landmarks |",
            "| `features.csv` | Biomechanical and temporal feature table |",
            "| `risk_scores.csv` | Per-frame interpretable risk scores, levels, and rule flags |",
            "| `events.json` | Merged time intervals rated HIGH or CRITICAL (heuristic) |",
            "| `output_overlay.mp4` | Overlay visualization (pose and risk readout) |",
            "| `report.md` | This document |",
            "",
        ]
    )
    if cfg.rules_config_path and Path(cfg.rules_config_path).is_file():
        lines.append(
            f"**Risk configuration:** `configs/risk_rules.yaml` was applied (resolved path: "
            f"`{Path(cfg.rules_config_path).resolve()}`). "
        )
    else:
        lines.append(
            "**Risk configuration:** project defaults or in-memory fallbacks were used "
            "(no project `configs/risk_rules.yaml` found at run time, or an explicit file "
            "was not provided). "
        )
    lines.append("")

    lines.extend(["## 3. Detected risk events", ""])
    if not events:
        lines.append(
            "No **HIGH** or **CRITICAL** event segments were recorded in `events.json` under "
            "the current policy and merge settings. This does not imply the absence of risk "
            "factors in the raw footage, only that the automated segment detector did not "
            "classify a sustained high-risk interval."
        )
    else:
        lines.append(
            "The table below lists merged segments in which the frame-level risk level was "
            "classified as **HIGH** or **CRITICAL** (see `events.json` for the authoritative "
            "record). Times are expressed in **seconds** relative to the sampled sequence."
        )
        lines.extend(
            [
                "",
                "| `event_id` | Level | Start time (s) | End time (s) | max `risk_score` |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for ev in events:
            eid = ev.get("event_id", "—")
            lev = ev.get("event_level", "—")
            t0 = ev.get("start_time", "—")
            t1 = ev.get("end_time", "—")
            mx = ev.get("max_risk_score", "—")
            if isinstance(t0, (int, float)):
                t0 = f"{float(t0):.4f}"
            if isinstance(t1, (int, float)):
                t1 = f"{float(t1):.4f}"
            if isinstance(mx, (int, float, np.floating)):
                mx = f"{float(mx):.4f}"
            lines.append(f"| {eid} | {lev} | {t0} | {t1} | {mx} |")
    lines.append("")

    lines.extend(["## 4. Highest risk moment", ""])
    row = _highest_risk_row(rdf)
    if row is None:
        lines.append(
            "No frame-wise risk data were available, or the `risk_score` column was empty; "
            "a peak-instant summary cannot be reported."
        )
    else:
        sc = float(pd.to_numeric(row.get("risk_score", np.nan), errors="coerce"))
        level = row.get("risk_level", "—")
        ts = row.get("timestamp", None)
        fid = row.get("frame_id", "—")
        fidx = row.get("frame_index", "—")
        ts_s = f"{float(ts):.4f} s" if ts is not None and pd.notna(ts) else "—"
        lines.append(
            f"Across sampled frames, the **maximum** scalar risk value **"
            f"{sc:.4f}** (level **{level}**) occurred at an annotated time of **{ts_s}** "
            f"(row key `frame_id` = `{fid}`; `frame_index` = `{fidx}`). "
            "This instant should be **cross-checked against the overlay video and source "
            "clip**; it is a local maximum of a hand-tuned heuristic score, not a label for "
            "a specific medical or disciplinary outcome."
        )
    lines.append("")

    lines.extend(["## 5. Triggered rules", ""])
    counts = _aggregate_triggered_rules(rdf, "triggered_rules")
    if not counts:
        lines.append(
            "No per-frame `triggered_rules` field could be aggregated (missing column, empty "
            "table, or no component above the configured `trigger_epsilon` threshold in "
            "`configs/risk_rules.yaml`)."
        )
    else:
        lines.append(
            "The following table summarizes how often each **interpretable rule key** "
            "appeared in `triggered_rules` across frames (a rule may be listed in multiple "
            "frames). The short labels paraphrase the design intent; authoritative "
            "definitions and thresholds remain in the YAML policy file."
        )
        lines.extend(
            [
                "",
                "| Rule key | Indicative label (informative) | Frame count |",
                "| --- | --- | ---: |",
            ]
        )
        for key, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            label = COMBAT_MVP_INDICATOR_LABELS.get(key, "—")
            lines.append(f"| `{key}` | {label} | {c} |")
    lines.append("")

    lines.extend(["## 6. Human review recommendation", ""])
    lines.append(
        "**It is recommended that a qualified human reviewer** (e.g. designated official, "
        "safety staff, or researcher) examine the list of events, the `output_overlay.mp4` "
        "rendering, and the source video **before** drawing operational conclusions. Use "
        "this report to **triangulate** between numerical summaries and visual evidence. "
    )
    lines.append(
        "Automated risk scores and segment boundaries are **auxiliary**; they do not "
        "constitute a determination of fact regarding fouls, stoppages, or medical condition."
    )
    if (cfg.optional_ollama_narrative or "").strip():
        lines.extend(
            [
                "",
                "### Optional model-generated narrative",
                "",
                f"{(cfg.optional_ollama_narrative or '').strip()}",
            ]
        )
    ex_dir = cfg.explanations_dir
    if ex_dir is not None:
        ex_dir = ex_dir.expanduser().resolve()
        if ex_dir.is_dir():
            mds = sorted(ex_dir.glob("event_*.md"))
            if mds:
                lines.extend(
                    [
                        "",
                        "### Per-event LLM- or template-based explanations",
                        "",
                        "Supplementary narrative files (not verified as factual) were written under `"
                        f"{_path_rel(ex_dir, output_root)}`:",
                        "",
                    ]
                )
                for p in mds:
                    rel = p.name
                    try:
                        rel = str(p.relative_to(output_root))
                    except ValueError:
                        rel = str(p)
                    lines.append(f"- `{rel}`")
    lines.append("")

    lines.extend(
        [
            "## 7. Limitations",
            "",
            "- **Heuristic model:** The scalar `risk_score` and discrete `risk_level` are produced "
            "by a **transparent, tunable** combination of hand-crafted rules; they are not "
            "end-to-end machine-learned on injury outcomes.",
            "- **Pose and sampling:** Scores depend on **MediaPipe** tracking quality, lighting, "
            "occlusion, and the chosen **sampling** rate. Dense temporal phenomena between "
            "sampled instants are not fully captured.",
            "- **Policy dependence:** All thresholds and weights in `interpretable_aggregation` "
            "influence outcomes; re-tuning is expected when changing discipline or camera setup.",
            "- **Scope:** This tool supports **inspection and research workflows**; it does not "
            "replace event documentation required by a sanctioning body.",
            "",
        ]
    )

    lines.extend(
        [
            "## 8. Safety disclaimer",
            "",
            "FightSafe AI and this report are provided for **decision support in human review**, "
            "**training**, and **academic** contexts only. The system is **not** a medical device, "
            "does **not** diagnose or predict injury, acute neurological compromise, or any other "
            "medical condition, and must not be used as a substitute for appropriate medical or "
            "safety professionals or governing rules. The authors and affiliated institution "
            "disclaim liability for any operational or health-related actions taken on the basis "
            "of these automated outputs; responsibility remains with the user organization.",
            "",
            REPORT_END_ATTRIBUTION,
            "",
        ]
    )
    return "\n".join(lines)


def write_mvp_report(
    report_path: Path,
    cfg: MVPReportConfig,
) -> Path:
    """Render :func:`generate_mvp_report_markdown` to ``report_path`` and return the path."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(generate_mvp_report_markdown(cfg), encoding="utf-8")
    return report_path


__all__ = [
    "MVPReportConfig",
    "generate_mvp_report_markdown",
    "write_mvp_report",
]
