"""
Behavior-level **ablation** over the formal risk fusion (not semantic validation).

These metrics describe how **candidate** risk traces change when signal groups or interaction
rules are toggled—**not** accuracy, precision, or clinical correctness.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

import matplotlib.pyplot as plt
import pandas as pd

from fightsafe_ai.risk.adapters import signals_from_feature_row
from fightsafe_ai.risk.events import (
    COL_FRAME_ID,
    COL_TIMESTAMP as EV_COL_TIMESTAMP,
    RiskEventExtractionConfig,
    frame_risk_to_events_list,
)
from fightsafe_ai.risk.formal_model import (
    RiskFusionConfig,
    RiskSignal,
    compute_fused_risk_score,
    load_risk_fusion_config,
    map_score_to_levels,
)
from fightsafe_ai.risk.rules import load_interpretable_risk_config
from fightsafe_ai.risk.scorer import COL_RISK_LEVEL, COL_RISK_SCORE


ABLATION_MODES: Final[tuple[str, ...]] = (
    "biomechanics_only",
    "posture_only",
    "anomaly_only",
    "biomechanics_plus_posture",
    "full_fusion",
    "full_fusion_without_interactions",
    "full_fusion_with_surrender_disabled",
    "full_fusion_with_limb_anomaly_disabled",
)

_GROUP_FILTERS: dict[str, frozenset[str] | None] = {
    "biomechanics_only": frozenset({"biomechanics"}),
    "posture_only": frozenset({"posture"}),
    "anomaly_only": frozenset({"anomaly"}),
    "biomechanics_plus_posture": frozenset({"biomechanics", "posture"}),
    "full_fusion": None,
    "full_fusion_without_interactions": None,
    "full_fusion_with_surrender_disabled": None,
    "full_fusion_with_limb_anomaly_disabled": None,
}


def _resolve_features_csv(path: Path) -> Path:
    if path.is_dir():
        cand = path / "features.csv"
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"No features.csv under run directory: {path}")
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    raise FileNotFoundError(f"Expected directory with features.csv or a CSV file: {path}")


def _default_rules_yaml() -> Path | None:
    """Project ``configs/risk_rules.yaml`` when present (same default spirit as the MVP scorer)."""
    p = Path(__file__).resolve().parents[3] / "configs" / "risk_rules.yaml"
    return p if p.is_file() else None


def _read_fps(run_dir: Path | None, fps_override: float | None) -> float:
    if fps_override is not None and fps_override > 0:
        return float(fps_override)
    if run_dir is not None:
        qa = run_dir / "qa_report.json"
        if qa.is_file():
            try:
                data = json.loads(qa.read_text(encoding="utf-8"))
                m = data.get("metrics") or {}
                for key in ("extraction_fps", "fps", "overlay_fps"):
                    v = m.get(key)
                    if v is not None and float(v) > 0:
                        return float(v)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    return 10.0


def filter_signals_for_ablation(signals: list[RiskSignal], mode: str) -> list[RiskSignal]:
    """Drop signals by group or name according to ablation mode."""
    if mode == "full_fusion_with_surrender_disabled":
        return [s for s in signals if s.name != "surrender_gesture"]
    if mode == "full_fusion_with_limb_anomaly_disabled":
        return [s for s in signals if s.name != "limb_anomaly"]
    allowed_groups = _GROUP_FILTERS.get(mode)
    if allowed_groups is None:
        return list(signals)
    return [s for s in signals if s.group in allowed_groups]


def config_for_ablation_mode(base: RiskFusionConfig, mode: str) -> RiskFusionConfig:
    if mode == "full_fusion_without_interactions":
        return replace(base, interaction_rules=())
    return base


def formal_risk_timeseries(
    features_csv: Path,
    *,
    fusion_yaml: Path | None,
    rules_yaml: Path | None,
    mode: str,
    fps: float,
) -> pd.DataFrame:
    feat_path = _resolve_features_csv(features_csv)
    df = pd.read_csv(feat_path)
    if df.empty:
        return pd.DataFrame(
            columns=[COL_FRAME_ID, EV_COL_TIMESTAMP, COL_RISK_SCORE, COL_RISK_LEVEL],
        )

    rpath = rules_yaml if rules_yaml is not None else _default_rules_yaml()
    icfg = load_interpretable_risk_config(rpath)
    fcfg = load_risk_fusion_config(fusion_yaml)
    fcfg_mode = config_for_ablation_mode(fcfg, mode)

    scores: list[float] = []
    levels: list[str] = []
    ts_list: list[float] = []
    for i in range(len(df)):
        row = cast("Mapping[str, Any]", df.iloc[i].to_dict())
        sig = signals_from_feature_row(row, interpretable_config=icfg, fusion_config=fcfg_mode)
        sig = filter_signals_for_ablation(sig, mode)
        rs = compute_fused_risk_score(sig, fcfg_mode)
        scores.append(rs)
        levels.append(map_score_to_levels(rs, fcfg_mode))
        ts_list.append(float(i / fps) if fps > 0 else float(i))

    out = pd.DataFrame(
        {
            COL_FRAME_ID: range(len(df)),
            EV_COL_TIMESTAMP: ts_list,
            COL_RISK_SCORE: scores,
            COL_RISK_LEVEL: levels,
        }
    )
    return out


def _behavior_metrics(risk_df: pd.DataFrame, fps: float) -> dict[str, Any]:
    if risk_df.empty or COL_RISK_SCORE not in risk_df.columns:
        return {
            "max_risk_score": 0.0,
            "mean_risk_score": 0.0,
            "number_of_candidate_events": 0,
            "risk_event_density": 0.0,
            "percentage_high_or_critical_frames": 0.0,
            "earliest_high_or_critical_timestamp": None,
        }

    rs = risk_df[COL_RISK_SCORE].astype(float)
    mx = float(rs.max())
    mean = float(rs.mean())
    hi = risk_df[COL_RISK_LEVEL].astype(str).str.upper().isin({"HIGH", "CRITICAL"})
    pct_hi = float(hi.mean() * 100.0) if len(risk_df) else 0.0
    earliest: float | None = None
    if hi.any() and EV_COL_TIMESTAMP in risk_df.columns:
        sub = risk_df.loc[hi, EV_COL_TIMESTAMP]
        earliest = float(sub.min()) if len(sub) else None

    ev_cfg = RiskEventExtractionConfig(fps=float(fps) if fps > 0 else None)
    events = frame_risk_to_events_list(risk_df, config=ev_cfg)
    n_ev = len(events)
    dur = float(len(risk_df) / fps) if fps > 0 else float(len(risk_df))
    density = float(n_ev / dur) if dur > 0 else 0.0

    return {
        "max_risk_score": mx,
        "mean_risk_score": mean,
        "number_of_candidate_events": n_ev,
        "risk_event_density": density,
        "percentage_high_or_critical_frames": pct_hi,
        "earliest_high_or_critical_timestamp": earliest,
    }


def run_risk_ablation(
    risk_scores_or_features_path: Path,
    output_dir: Path,
    configs: list[str] | None = None,
    *,
    fusion_yaml: Path | None = None,
    rules_yaml: Path | None = None,
    fps: float | None = None,
) -> Path:
    """
    Run selected ablation modes; write CSV, TeX summary, and optional timeline plot.

    ``risk_scores_or_features_path`` may be a **run directory** (must contain ``features.csv``)
    or a path to ``features.csv``.
    """
    root = risk_scores_or_features_path.expanduser().resolve()
    feat_file = _resolve_features_csv(root)
    run_dir = feat_file.parent if feat_file.parent.is_dir() else root.parent
    fps_val = _read_fps(run_dir if root.is_dir() or feat_file.parent.is_dir() else None, fps)

    modes = list(configs) if configs else list(ABLATION_MODES)
    for m in modes:
        if m not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {m!r}. Expected one of {ABLATION_MODES}.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    series_map: dict[str, pd.DataFrame] = {}

    for mode in modes:
        ts = formal_risk_timeseries(
            feat_file,
            fusion_yaml=fusion_yaml,
            rules_yaml=rules_yaml,
            mode=mode,
            fps=fps_val,
        )
        series_map[mode] = ts
        met = _behavior_metrics(ts, fps_val)
        row = {"ablation_mode": mode, **met}
        rows.append(row)
        ts.to_csv(output_dir / f"risk_series_{mode}.csv", index=False)

    summary = pd.DataFrame(rows)
    csv_path = output_dir / "ablation_results.csv"
    summary.to_csv(csv_path, index=False)

    tex_path = output_dir / "ablation_results.tex"
    tex_path.write_text(_latex_table(summary), encoding="utf-8")

    _plot_timelines(series_map, output_dir / "ablation_risk_timeline.png", fps_val)

    return output_dir


def _latex_table(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "ablation_mode"]
    header = " & ".join(c.replace("_", r"\_") for c in ["mode", *cols]) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        mode = str(row["ablation_mode"]).replace("_", r"\_")
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
                cells.append("---")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rest = " & ".join(cells)
        lines.append(f"{mode} & {rest} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _plot_timelines(series_map: Mapping[str, pd.DataFrame], out_path: Path, fps: float) -> None:
    if not series_map:
        return
    plt.figure(figsize=(10, 5))
    for mode, df in series_map.items():
        if df.empty or EV_COL_TIMESTAMP not in df.columns:
            continue
        plt.plot(
            df[EV_COL_TIMESTAMP],
            df[COL_RISK_SCORE],
            label=mode[:28],
            linewidth=1.2,
            alpha=0.85,
        )
    plt.xlabel("time (s)")
    plt.ylabel("fused risk score (behavior metric)")
    plt.title("Formal fusion ablation — not validation accuracy")
    plt.legend(fontsize=6, loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_risk_ablation_all(
    base_dir: Path,
    output_dir: Path,
    *,
    fusion_yaml: Path | None = None,
    rules_yaml: Path | None = None,
    fps: float | None = None,
    configs: list[str] | None = None,
) -> Path:
    """Run ablation for each subdirectory of ``base_dir`` that contains ``features.csv``."""
    base_dir = base_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[pd.DataFrame] = []

    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "features.csv").is_file():
            continue
        out_sub = output_dir / sub.name
        run_risk_ablation(
            sub,
            out_sub,
            configs=configs,
            fusion_yaml=fusion_yaml,
            rules_yaml=rules_yaml,
            fps=fps,
        )
        summ = pd.read_csv(out_sub / "ablation_results.csv")
        summ.insert(0, "run_id", sub.name)
        summary_rows.append(summ)

    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(
            output_dir / "ablation_all_runs.csv", index=False
        )
    return output_dir


def iter_ablation_modes() -> Iterable[str]:
    return iter(ABLATION_MODES)
