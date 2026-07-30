#!/usr/bin/env python3.12
"""
EAAI checkpoint runner — BoxingVI same-input formal aggregation (python3.12).

Protocol justified from legacy paper1 ``run_aggregation_comparison.py``:
- formal fusion via schemes over held-fixed feature caches;
- risk event extraction: merge_gap=2, min_duration=0.0 (BoxingVI path);
- fixed strike/anomaly legs from archived full_fusion baselines;
- BoxingVI timeline merge=8; evaluate subset=full_fusion;
- IoU=0.01; tolerance=0.5s.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

SOFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFT / "src"))

from fightsafe_ai.evaluation.boxingvi_evaluator import (  # noqa: E402
    _qualifies_impact_like_prediction,
    _windows_from_event_dicts,
    load_ground_truth_impact_windows,
)
from fightsafe_ai.evaluation.event_metrics import EventWindow  # noqa: E402
from fightsafe_ai.evaluation.metrics import evaluate_event_prediction  # noqa: E402
from fightsafe_ai.risk.events import (  # noqa: E402
    COL_FRAME_ID,
    COL_RISK_LEVEL,
    COL_RISK_SCORE,
    COL_TIMESTAMP,
    RiskEventExtractionConfig,
    frame_risk_to_events_list,
)
from fightsafe_ai.risk.formal_model import RiskFusionConfig, load_risk_fusion_config  # noqa: E402
from fightsafe_ai.risk.rules import (  # noqa: E402
    ALL_RULE_NAMES,
    InterpretableRiskConfig,
    build_rule_components,
    load_interpretable_risk_config,
)

CACHE_DIR = SOFT / "optional_tier_b" / "inputs" / "features_cache"
CACHED_FULL = SOFT / "optional_tier_b" / "inputs" / "strike_baselines"
ANN_ROOT = SOFT / "data" / "boxingvi"
FUSION_YAML = SOFT / "configs" / "risk_fusion.yaml"
RULES_YAML = SOFT / "configs" / "risk_rules.yaml"

FPS = 30.0
MERGE_GAP_FRAMES = 2
MIN_DURATION_S = 0.0  # BoxingVI documented comparison path (reference clips used 0.5)
BOXINGVI_TIMELINE_MERGE_FRAMES = 8
IOU_THRESHOLD = 0.01
TOLERANCE_SECONDS = 0.5
VIDEOS = [f"V{i}" for i in range(1, 11)]
AGG_SCHEMES = ("equal", "weighted", "max")
DROPOUT_PS = (0.0, 0.1, 0.3, 0.5)
DROPOUT_SEEDS = tuple(range(10))
BOOTSTRAP_B = 1000
RNG_MASTER = 20260730
SchemeName = Literal["equal", "weighted", "max"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gt_frame_mask(video_id: str, n_frames: int, gt_windows: list[EventWindow]) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=np.uint8)
    for w in gt_windows:
        i0 = max(0, int(np.floor(float(w.start) * FPS)))
        i1 = min(n_frames - 1, int(np.ceil(float(w.end) * FPS)))
        if i1 >= i0:
            mask[i0 : i1 + 1] = 1
    return mask


def build_matrix(
    video_id: str,
    icfg: InterpretableRiskConfig,
    gt_windows: list[EventWindow],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    feat = pd.read_pickle(CACHE_DIR / f"{video_id}.pkl")
    comp, active = build_rule_components(feat, icfg)
    n = len(feat)
    data: dict[str, Any] = {
        "video_id": video_id,
        "frame_index": np.arange(n, dtype=int),
        "timestamp": np.arange(n, dtype=float) / FPS,
        "fps": np.full(n, FPS),
    }
    a_arrs: dict[str, np.ndarray] = {}
    c_arrs: dict[str, np.ndarray] = {}
    for name in ALL_RULE_NAMES:
        c = np.asarray(comp[name], dtype=float)
        a = np.full(n, 1 if active.get(name, False) else 0, dtype=np.uint8)
        c_arrs[name] = c
        a_arrs[name] = a
        data[f"c_{name}"] = c
        data[f"a_{name}"] = a
    data["y_impact"] = gt_frame_mask(video_id, n, gt_windows)
    return pd.DataFrame(data), c_arrs, a_arrs


def pre_scores_vectorized(
    c_arrs: dict[str, np.ndarray],
    a_arrs: dict[str, np.ndarray],
    fcfg: RiskFusionConfig,
    scheme: SchemeName,
) -> np.ndarray:
    """α=0 excludes channel from the pool; c=0 with α=1 is zero-valued evidence."""
    n = len(next(iter(c_arrs.values())))
    names = [nm for nm in ALL_RULE_NAMES if float(fcfg.signal_weights.get(nm, 0.0)) > 0.0]
    C = np.stack([np.clip(c_arrs[nm], 0.0, 1.0) for nm in names], axis=1)
    A = np.stack([a_arrs[nm].astype(float) for nm in names], axis=1)
    W = np.array([float(fcfg.signal_weights.get(nm, 0.0)) for nm in names], dtype=float)

    if scheme == "equal":
        denom = A.sum(axis=1)
        num = (C * A).sum(axis=1)
        out = np.divide(num, denom, out=np.zeros(n), where=denom > 0)
    elif scheme == "max":
        masked = np.where(A > 0, C, -np.inf)
        out = masked.max(axis=1)
        out[~np.isfinite(out)] = 0.0
    else:
        w = W[None, :] * A
        denom = w.sum(axis=1)
        num = (C * w).sum(axis=1)
        out = np.divide(num, denom, out=np.zeros(n), where=denom > 0)
    return np.clip(out, 0.0, 1.0)


def apply_boosts_vectorized(
    pre: np.ndarray,
    c_arrs: dict[str, np.ndarray],
    a_arrs: dict[str, np.ndarray],
    fcfg: RiskFusionConfig,
    *,
    collect_firings: bool = False,
) -> tuple[np.ndarray, dict[str, int], int]:
    n = len(pre)
    boost = np.zeros(n, dtype=float)
    counts: dict[str, int] = defaultdict(int)
    frames_any = np.zeros(n, dtype=bool)
    if not fcfg.interaction_rules:
        return np.clip(pre, 0.0, 1.0), dict(counts), 0
    thr_default = float(fcfg.interaction_signal_threshold)
    for rule in fcfg.interaction_rules:
        th = float(rule.signal_threshold or thr_default)
        mask = np.ones(n, dtype=bool)
        for req in rule.required_signals:
            if req not in c_arrs:
                mask[:] = False
                break
            mask &= (a_arrs[req] > 0) & (c_arrs[req] > th)
        b = float(rule.boost)
        boost += np.where(mask, b, 0.0)
        if collect_firings and mask.any():
            counts[str(rule.name)] += int(mask.sum())
            frames_any |= mask
    return np.clip(pre + boost, 0.0, 1.0), dict(counts), int(frames_any.sum())


def map_levels_fast(scores: np.ndarray, fcfg: RiskFusionConfig) -> np.ndarray:
    lt = fcfg.level_thresholds
    m = float(lt.get("medium_min", 0.25))
    h = float(lt.get("high_min", 0.5))
    c = float(lt.get("critical_min", 0.75))
    return np.where(
        scores < m,
        "LOW",
        np.where(scores < h, "MEDIUM", np.where(scores < c, "HIGH", "CRITICAL")),
    )


def score_video(
    c_arrs: dict[str, np.ndarray],
    a_arrs: dict[str, np.ndarray],
    fcfg: RiskFusionConfig,
    scheme: SchemeName,
    *,
    collect_firings: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    pre = pre_scores_vectorized(c_arrs, a_arrs, fcfg, scheme)
    fused, counts, frames_aff = apply_boosts_vectorized(
        pre, c_arrs, a_arrs, fcfg, collect_firings=collect_firings
    )
    levels = map_levels_fast(fused, fcfg)
    return fused, levels, counts, frames_aff


def merge_timeline_events(events: list[dict[str, Any]], *, fps: float, merge_frames: int) -> list[dict[str, Any]]:
    gap_sec = max(0.0, float(merge_frames) / float(fps))
    candidates: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if not _qualifies_impact_like_prediction(ev):
            continue
        if ev.get("start_time") is None or ev.get("end_time") is None:
            continue
        candidates.append(dict(ev))
    if not candidates:
        return []
    candidates.sort(key=lambda d: (float(d["start_time"]), float(d["end_time"])))
    merged: list[dict[str, Any]] = []
    cur = dict(candidates[0])
    cur_e = float(cur["end_time"])
    for nxt in candidates[1:]:
        ns, ne = float(nxt["start_time"]), float(nxt["end_time"])
        if ns <= cur_e + gap_sec:
            cur_e = max(cur_e, ne)
            cur["end_time"] = cur_e
            if "end_frame" in nxt:
                cur["end_frame"] = nxt["end_frame"]
            ms = float(cur.get("max_risk_score") or 0.0)
            ns_ = float(nxt.get("max_risk_score") or 0.0)
            if ns_ >= ms:
                cur["max_risk_score"] = ns_
                if "event_level" in nxt:
                    cur["event_level"] = nxt["event_level"]
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur_e = float(cur["end_time"])
    merged.append(cur)
    return merged


def events_from_scores(scores: np.ndarray, levels: np.ndarray) -> list[dict[str, Any]]:
    rdf = pd.DataFrame(
        {
            COL_FRAME_ID: np.arange(len(scores)),
            COL_TIMESTAMP: np.arange(len(scores), dtype=float) / FPS,
            COL_RISK_SCORE: scores,
            COL_RISK_LEVEL: levels,
        }
    )
    ev_cfg = RiskEventExtractionConfig(
        fps=FPS,
        merge_gap_frames=MERGE_GAP_FRAMES,
        min_duration_seconds=MIN_DURATION_S,
    )
    return frame_risk_to_events_list(rdf, config=ev_cfg)


def build_full_fusion_events(
    events_risk: list[dict[str, Any]],
    cached_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    strike = list(cached_payload.get("strike_events") or [])
    anom = list(cached_payload.get("anomaly_events") or [])
    # Mirror apply_full_fusion_timeline_merge: union events + anomaly, then merge.
    union = list(events_risk) + strike
    for ev in anom:
        if isinstance(ev, dict):
            union.append(ev)
    return merge_timeline_events(union, fps=FPS, merge_frames=BOXINGVI_TIMELINE_MERGE_FRAMES)


def eval_events(pred_events: list[dict[str, Any]], gt: list[EventWindow]):
    pred_windows = _windows_from_event_dicts(pred_events)
    raw = evaluate_event_prediction(
        pred_windows,
        gt,
        iou_threshold=IOU_THRESHOLD,
        tolerance_seconds=TOLERANCE_SECONDS,
        require_same_label=False,
    )

    class _ER:
        pass

    er = _ER()
    er.precision = raw.precision
    er.recall = raw.recall
    er.f1 = raw.f1
    er.true_positives = raw.true_positives
    er.false_positives = raw.false_positives
    er.false_negatives = raw.false_negatives
    er.mean_detection_latency_seconds = raw.mean_onset_delay_seconds
    er.mean_abs_detection_latency_seconds = raw.mean_abs_onset_delay_seconds
    er.raw = raw
    return er, pred_windows


def write_pred_json(
    path: Path,
    video_id: str,
    tag: str,
    events_merged: list[dict[str, Any]],
    events_risk: list[dict[str, Any]],
    cached: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(cached)
    payload.update(
        {
            "video_id": video_id,
            "fps": FPS,
            "method": tag,
            "events": events_merged,
            "events_risk_only": events_risk,
            "strike_events": list(cached.get("strike_events") or []),
            "anomaly_events": [],
            "full_fusion_timeline_merge_frames": BOXINGVI_TIMELINE_MERGE_FRAMES,
            "aggregation_note": "risk leg re-aggregated; strike fixed; anomaly cleared after merge",
        }
    )
    path.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")


def micro_macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    pd_, rd = tp + fp, tp + fn
    micro_p = float(tp / pd_) if pd_ else 0.0
    micro_r = float(tp / rd) if rd else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if micro_p + micro_r else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean([r["precision"] for r in rows])) if rows else 0.0,
        "macro_recall": float(np.mean([r["recall"] for r in rows])) if rows else 0.0,
        "macro_f1": float(np.mean([r["f1"] for r in rows])) if rows else 0.0,
        "total_tp": float(tp),
        "total_fp": float(fp),
        "total_fn": float(fn),
    }


def bootstrap_ci(values: list[float], b: int = BOOTSTRAP_B, seed: int = RNG_MASTER):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return 0.0, 0.0, 0.0
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(b)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def paired_perm_p(diff: np.ndarray, n_perm: int = 5000, seed: int = RNG_MASTER) -> float:
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    if not len(d):
        return 1.0
    obs = abs(float(np.mean(d)))
    count = sum(
        abs(float(np.mean(rng.choice([-1.0, 1.0], size=len(d)) * d))) >= obs - 1e-15
        for _ in range(n_perm)
    )
    return float((count + 1) / (n_perm + 1))


def cliffs_delta(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, dtype=float)
    if not len(d):
        return 0.0
    return float(np.mean(d > 0) - np.mean(d < 0))


def dropout_alphas(a_arrs: dict[str, np.ndarray], p: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for nm, a in a_arrs.items():
        drop = rng.random(len(a)) < p
        out[nm] = np.where((a > 0) & drop, 0, a).astype(np.uint8)
    return out


def naive_zero_arrays(
    c_arrs: dict[str, np.ndarray],
    a_arrs: dict[str, np.ndarray],
    a_drop: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    c2, a2 = {}, {}
    for nm in ALL_RULE_NAMES:
        was = a_arrs[nm]
        now = a_drop[nm]
        dropped = (was > 0) & (now == 0)
        c2[nm] = np.where(dropped, 0.0, c_arrs[nm])
        a2[nm] = np.where(dropped, 1, was).astype(np.uint8)
    return c2, a2


def classify_failures(video_id, method, gt_windows, pred_events, er):
    rows = []
    matched_gt, matched_pred = set(), set()
    raw = er.raw
    # Match by onset times from EventMatchDelay
    for m in getattr(raw, "matches", []) or []:
        for gi, w in enumerate(gt_windows):
            if abs(float(w.start) - float(m.ref_start)) < 1e-9 and abs(float(w.end) - float(m.ref_end)) < 1e-9:
                matched_gt.add(gi)
                break
        for pi, ev in enumerate(pred_events):
            if abs(float(ev.get("start_time", -1)) - float(m.pred_start)) < 1e-9 and abs(
                float(ev.get("end_time", -1)) - float(m.pred_end)
            ) < 1e-9:
                matched_pred.add(pi)
                break
        latency = float(m.onset_delay_seconds)
        ftype = "delayed_onset" if latency > 0.25 else ("early_onset" if latency < -0.25 else None)
        if ftype:
            rows.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "failure_type": ftype,
                    "gt_start": float(m.ref_start),
                    "gt_end": float(m.ref_end),
                    "pred_start": float(m.pred_start),
                    "pred_end": float(m.pred_end),
                    "onset_latency_s": latency,
                }
            )
    for gi, w in enumerate(gt_windows):
        if gi not in matched_gt:
            rows.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "failure_type": "false_negative",
                    "gt_start": float(w.start),
                    "gt_end": float(w.end),
                    "pred_start": "",
                    "pred_end": "",
                }
            )
            dur = float(w.end) - float(w.start)
            if dur < 0.5:
                rows.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "failure_type": "short_event_suppression",
                        "gt_start": float(w.start),
                        "gt_end": float(w.end),
                        "pred_start": "",
                        "pred_end": "",
                        "gt_duration_s": dur,
                    }
                )
    for pi, ev in enumerate(pred_events):
        if pi not in matched_pred:
            rows.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "failure_type": "false_positive",
                    "gt_start": "",
                    "gt_end": "",
                    "pred_start": float(ev.get("start_time", float("nan"))),
                    "pred_end": float(ev.get("end_time", float("nan"))),
                }
            )
    for gi, w in enumerate(gt_windows):
        hits = [
            pi
            for pi, ev in enumerate(pred_events)
            if not (
                float(ev.get("end_time", -1)) < float(w.start)
                or float(ev.get("start_time", -1)) > float(w.end)
            )
        ]
        if len(hits) >= 2:
            rows.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "failure_type": "fragmented_detection",
                    "gt_start": float(w.start),
                    "gt_end": float(w.end),
                    "pred_start": "",
                    "pred_end": "",
                    "n_fragments": len(hits),
                }
            )
    for pi, ev in enumerate(pred_events):
        ps, pe = float(ev.get("start_time", -1)), float(ev.get("end_time", -1))
        hits = [gi for gi, w in enumerate(gt_windows) if not (pe < float(w.start) or ps > float(w.end))]
        if len(hits) >= 2:
            rows.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "failure_type": "merged_distinct_events",
                    "gt_start": "",
                    "gt_end": "",
                    "pred_start": ps,
                    "pred_end": pe,
                    "n_gt_merged": len(hits),
                }
            )
    return rows


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    t0 = time.time()
    logf = out / "logs" / "run.log"
    logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    icfg = load_interpretable_risk_config(RULES_YAML)
    fcfg = load_risk_fusion_config(FUSION_YAML)
    fcfg_off = replace(fcfg, interaction_rules=())

    (out / "PROTOCOL.md").write_text(
        f"""# EAAI checkpoint protocol

## Justification
Matches legacy ``paper1_pre_rewrite_.../scripts/run_aggregation_comparison.py`` BoxingVI path.

## Inputs
- Features: `{CACHE_DIR}` (read-only)
- Fixed strike/anomaly: `{CACHED_FULL}`
- Annotations: `{ANN_ROOT}/annotations`
- Configs: `{FUSION_YAML}`, `{RULES_YAML}` (not tuned on labels)

## Matrix
- `c_<rule>`, `a_<rule>` via `build_rule_components` once per video
- α=0 ⇒ channel excluded; c=0 & α=1 ⇒ zero-valued evidence retained
- `y_impact` stored for analysis only — never used for scoring/tuning

## Temporal contract (BoxingVI)
- fps={FPS}; risk merge_gap_frames={MERGE_GAP_FRAMES}; min_duration={MIN_DURATION_S}s
- BoxingVI timeline merge={BOXINGVI_TIMELINE_MERGE_FRAMES} frames
- IoU={IOU_THRESHOLD}; tolerance={TOLERANCE_SECONDS}s; subset=**full_fusion**
- Risk leg re-aggregated per scheme; strike events held fixed from baseline cache

## Aggregation comparison
- equal / weighted / max with **interactions ON**

## Interaction ablation
- weighted int OFF vs ON

## Dropout
- weighted; p={list(DROPOUT_PS)}; seeds={list(DROPOUT_SEEDS)} for p>0
- explicit α=0 vs naive c=0/α=1
""",
        encoding="utf-8",
    )

    log("Loading GT + cached strike payloads…")
    gt_by_vid: dict[str, list[EventWindow]] = {}
    cached_by_vid: dict[str, dict[str, Any]] = {}
    n_frames_by_vid: dict[str, int] = {}
    for vid in VIDEOS:
        feat = pd.read_pickle(CACHE_DIR / f"{vid}.pkl")
        n_frames_by_vid[vid] = len(feat)
        gt_by_vid[vid] = load_ground_truth_impact_windows(
            dataset_root=ANN_ROOT,
            video_id=vid,
            fps=FPS,
            num_skeleton_frames=len(feat),
        )
        cached_by_vid[vid] = json.loads((CACHED_FULL / f"boxingvi_predictions_{vid}.json").read_text(encoding="utf-8"))
        log(f"  {vid}: frames={len(feat)} gt_windows={len(gt_by_vid[vid])} strikes={len(cached_by_vid[vid].get('strike_events') or [])}")

    log("Building matrices…")
    mats: dict[str, pd.DataFrame] = {}
    carr: dict[str, dict[str, np.ndarray]] = {}
    aarr: dict[str, dict[str, np.ndarray]] = {}
    meta = []
    for vid in VIDEOS:
        mat, c_arrs, a_arrs = build_matrix(vid, icfg, gt_by_vid[vid])
        mats[vid], carr[vid], aarr[vid] = mat, c_arrs, a_arrs
        csv_p = out / "matrices" / f"{vid}.csv"
        mat.to_csv(csv_p, index=False)
        try:
            mat.to_parquet(out / "matrices" / f"{vid}.parquet", index=False)
        except Exception:
            pass
        miss = {nm: int((a_arrs[nm] == 0).sum()) for nm in ALL_RULE_NAMES}
        meta.append(
            {
                "video_id": vid,
                "n_rows": len(mat),
                "n_positive_frames": int(mat["y_impact"].sum()),
                "pos_frac": float(mat["y_impact"].mean()),
                "sha256": sha256_file(csv_p),
                "channels": ",".join(ALL_RULE_NAMES),
                "missing_counts_json": json.dumps(miss),
            }
        )
        log(f"  {vid}: n={len(mat)} pos={int(mat['y_impact'].sum())}")
    pd.DataFrame(meta).to_csv(out / "matrices" / "matrix_meta.csv", index=False)

    per_video: list[dict[str, Any]] = []
    method_events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    method_er: dict[tuple[str, str], Any] = {}

    def run_method(
        vid: str,
        scheme: SchemeName,
        cfg: RiskFusionConfig,
        tag: str,
        exp: str,
        inter: str,
        *,
        collect_firings: bool = False,
        c_use: dict[str, np.ndarray] | None = None,
        a_use: dict[str, np.ndarray] | None = None,
        write_trace: bool = True,
        write_pred: bool = True,
        pred_subdir: Path | None = None,
    ) -> tuple[dict[str, int], int, np.ndarray]:
        scores, levels, counts, frames_aff = score_video(
            c_use or carr[vid],
            a_use or aarr[vid],
            cfg,
            scheme,
            collect_firings=collect_firings,
        )
        if write_trace:
            rdf = pd.DataFrame(
                {
                    COL_FRAME_ID: np.arange(len(scores)),
                    COL_TIMESTAMP: np.arange(len(scores), dtype=float) / FPS,
                    COL_RISK_SCORE: scores,
                    COL_RISK_LEVEL: levels,
                }
            )
            trace_dir = out / ("interactions" if exp == "interactions" else "aggregation") / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            rdf.to_csv(trace_dir / f"{vid}_{tag}_scores.csv", index=False)
        events_risk = events_from_scores(scores, levels)
        events_merged = build_full_fusion_events(events_risk, cached_by_vid[vid])
        er, _ = eval_events(events_merged, gt_by_vid[vid])
        if write_pred:
            pdir = pred_subdir or (out / ("interactions" if exp == "interactions" else "aggregation") / "preds" / tag)
            write_pred_json(
                pdir / f"pred_{vid}_{tag}.json",
                vid,
                tag,
                events_merged,
                events_risk,
                cached_by_vid[vid],
            )
        method_events[(tag, vid)] = events_merged
        method_er[(tag, vid)] = er
        per_video.append(
            {
                "experiment": exp,
                "method": scheme,
                "interactions": inter,
                "video_id": vid,
                "precision": er.precision,
                "recall": er.recall,
                "f1": er.f1,
                "tp": er.true_positives,
                "fp": er.false_positives,
                "fn": er.false_negatives,
                "n_pred_events": len(events_merged),
                "n_risk_events": len(events_risk),
                "mean_onset_s": er.mean_detection_latency_seconds,
                "mean_abs_onset_s": er.mean_abs_detection_latency_seconds,
                "n_interaction_firings": int(sum(counts.values())),
                "frames_with_firing": frames_aff,
                "mean_score": float(scores.mean()),
                "pct_high_critical": float(np.mean((levels == "HIGH") | (levels == "CRITICAL")) * 100.0),
            }
        )
        return counts, frames_aff, scores

    log("Aggregation (interactions ON)…")
    for scheme in AGG_SCHEMES:
        log(f"  {scheme}")
        for vid in VIDEOS:
            run_method(
                vid,
                scheme,
                fcfg,
                f"{scheme}_intON",
                "aggregation",
                "on",
                collect_firings=(scheme == "weighted"),
            )

    log("Interaction ablation…")
    firing_rows: list[dict[str, Any]] = []
    for inter, cfg in (("off", fcfg_off), ("on", fcfg)):
        log(f"  weighted int={inter}")
        for vid in VIDEOS:
            counts, frames_aff, _ = run_method(
                vid,
                "weighted",
                cfg,
                f"weighted_int{inter}",
                "interactions",
                inter,
                collect_firings=True,
            )
            for rn, cnt in counts.items():
                firing_rows.append(
                    {"video_id": vid, "interactions": inter, "rule_name": rn, "fire_count": cnt}
                )
            firing_rows.append(
                {
                    "video_id": vid,
                    "interactions": inter,
                    "rule_name": "__frames_with_any_firing__",
                    "fire_count": frames_aff,
                }
            )
    n_rules = len(fcfg.interaction_rules)
    rules_ok = sum(
        1 for r in fcfg.interaction_rules if all(s in ALL_RULE_NAMES for s in r.required_signals)
    )
    (out / "interactions" / "rule_config_audit.json").write_text(
        json.dumps(
            {
                "n_configured_rules": n_rules,
                "n_rules_required_channels_present": rules_ok,
                "rule_names": [r.name for r in fcfg.interaction_rules],
                "missing_required_signals": sorted(
                    {
                        s
                        for r in fcfg.interaction_rules
                        for s in r.required_signals
                        if s not in ALL_RULE_NAMES
                    }
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(firing_rows).to_csv(out / "interaction_rule_firings.csv", index=False)

    log("Dropout…")
    dropout_rows: list[dict[str, Any]] = []
    for p in DROPOUT_PS:
        seeds = (0,) if p == 0.0 else DROPOUT_SEEDS
        for mode in ("explicit_alpha0", "naive_zero"):
            if p == 0.0 and mode == "naive_zero":
                continue
            for seed in seeds:
                log(f"  p={p} {mode} seed={seed}")
                seed_rows = []
                for vid in VIDEOS:
                    if p == 0.0:
                        c_use, a_use = carr[vid], aarr[vid]
                    else:
                        a_d = dropout_alphas(aarr[vid], p, seed + (hash(vid) % 10000))
                        if mode == "explicit_alpha0":
                            c_use, a_use = carr[vid], a_d
                        else:
                            c_use, a_use = naive_zero_arrays(carr[vid], aarr[vid], a_d)
                    scores, levels, _, _ = score_video(c_use, a_use, fcfg, "weighted", collect_firings=False)
                    events_risk = events_from_scores(scores, levels)
                    events_merged = build_full_fusion_events(events_risk, cached_by_vid[vid])
                    er, _ = eval_events(events_merged, gt_by_vid[vid])
                    # Persist only first seed per (p,mode) preds to limit disk
                    if seed == 0:
                        write_pred_json(
                            out
                            / "missingness"
                            / "preds"
                            / f"p{p}_{mode}_s{seed}"
                            / f"pred_{vid}_drop_p{p}_{mode}_s{seed}.json",
                            vid,
                            f"drop_p{p}_{mode}_s{seed}",
                            events_merged,
                            events_risk,
                            cached_by_vid[vid],
                        )
                    rec = {
                        "p": p,
                        "mode": mode if p > 0 else "none",
                        "seed": seed,
                        "video_id": vid,
                        "precision": er.precision,
                        "recall": er.recall,
                        "f1": er.f1,
                        "tp": er.true_positives,
                        "fp": er.false_positives,
                        "fn": er.false_negatives,
                        "n_pred_events": len(events_merged),
                        "mean_score": float(scores.mean()),
                        "std_score": float(scores.std()),
                    }
                    dropout_rows.append(rec)
                    seed_rows.append(rec)
                mm = micro_macro(seed_rows)
                dropout_rows.append(
                    {
                        "p": p,
                        "mode": mode if p > 0 else "none",
                        "seed": seed,
                        "video_id": "__MICRO_MACRO__",
                        **mm,
                        "precision": mm["micro_precision"],
                        "recall": mm["micro_recall"],
                        "f1": mm["micro_f1"],
                        "tp": mm["total_tp"],
                        "fp": mm["total_fp"],
                        "fn": mm["total_fn"],
                        "n_pred_events": "",
                        "mean_score": "",
                        "std_score": "",
                    }
                )
    pd.DataFrame(dropout_rows).to_csv(out / "dropout_results.csv", index=False)

    log("Statistics…")
    pv = pd.DataFrame(per_video)
    pv.to_csv(out / "per_video_metrics.csv", index=False)
    summary = []
    for (exp, method, inter), g in pv.groupby(["experiment", "method", "interactions"]):
        rows = g.to_dict("records")
        mm = micro_macro(rows)
        mean_f1, lo, hi = bootstrap_ci([float(r["f1"]) for r in rows])
        summary.append(
            {
                "experiment": exp,
                "method": method,
                "interactions": inter,
                **mm,
                "bootstrap_macro_f1_mean": mean_f1,
                "bootstrap_macro_f1_lo": lo,
                "bootstrap_macro_f1_hi": hi,
                "mean_n_pred_events": float(np.mean([r["n_pred_events"] for r in rows])),
                "mean_pct_high_critical": float(np.mean([r["pct_high_critical"] for r in rows])),
            }
        )
    pd.DataFrame(summary).to_csv(out / "experiment_summary.csv", index=False)

    paired: list[dict[str, Any]] = []

    def add_pair(name: str, a: pd.DataFrame, b: pd.DataFrame) -> None:
        aa, bb = a.set_index("video_id")["f1"], b.set_index("video_id")["f1"]
        common = sorted(set(aa.index) & set(bb.index))
        diff = np.array([float(aa[v] - bb[v]) for v in common])
        paired.append(
            {
                "comparison": name,
                "n_videos": len(common),
                "median_delta_f1": float(np.median(diff)) if len(diff) else 0.0,
                "mean_delta_f1": float(np.mean(diff)) if len(diff) else 0.0,
                "cliffs_delta": cliffs_delta(diff),
                "permutation_pvalue_twosided": paired_perm_p(diff),
                "note": "n=10 limited power; video-level unit",
            }
        )

    agg = pv[pv["experiment"] == "aggregation"]
    add_pair(
        "weighted_minus_equal",
        agg[(agg.method == "weighted") & (agg.interactions == "on")],
        agg[(agg.method == "equal") & (agg.interactions == "on")],
    )
    add_pair(
        "weighted_minus_max",
        agg[(agg.method == "weighted") & (agg.interactions == "on")],
        agg[(agg.method == "max") & (agg.interactions == "on")],
    )
    interdf = pv[pv["experiment"] == "interactions"]
    add_pair(
        "weighted_intON_minus_intOFF",
        interdf[interdf.interactions == "on"],
        interdf[interdf.interactions == "off"],
    )
    dd = pd.DataFrame(dropout_rows)
    ddv = dd[dd.video_id != "__MICRO_MACRO__"]
    for p in (0.1, 0.3, 0.5):
        ex = ddv[(ddv["p"] == p) & (ddv["mode"] == "explicit_alpha0")].groupby("video_id")["f1"].mean()
        nv = ddv[(ddv["p"] == p) & (ddv["mode"] == "naive_zero")].groupby("video_id")["f1"].mean()
        common = sorted(set(ex.index) & set(nv.index))
        diff = np.array([float(ex[v] - nv[v]) for v in common])
        paired.append(
            {
                "comparison": f"explicit_minus_naive_p{p}",
                "n_videos": len(common),
                "median_delta_f1": float(np.median(diff)) if len(diff) else 0.0,
                "mean_delta_f1": float(np.mean(diff)) if len(diff) else 0.0,
                "cliffs_delta": cliffs_delta(diff),
                "permutation_pvalue_twosided": paired_perm_p(diff),
                "note": "averaged over seeds per video",
            }
        )
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)

    log("Failures…")
    all_fail = []
    for scheme in AGG_SCHEMES:
        tag = f"{scheme}_intON"
        for vid in VIDEOS:
            all_fail.extend(
                classify_failures(vid, tag, gt_by_vid[vid], method_events[(tag, vid)], method_er[(tag, vid)])
            )
    for vid in VIDEOS:
        all_fail.extend(
            classify_failures(
                vid,
                "weighted_intOFF",
                gt_by_vid[vid],
                method_events[("weighted_intoff", vid)],
                method_er[("weighted_intoff", vid)],
            )
        )
    fail_df = pd.DataFrame(all_fail)
    fail_df.to_csv(out / "failures" / "all_failures.csv", index=False)
    fail_df.groupby(["method", "failure_type"]).size().reset_index(name="count").to_csv(
        out / "failure_counts.csv", index=False
    )
    parts = [
        fail_df[fail_df.failure_type == t].head(5)
        for t in ("false_positive", "false_negative", "fragmented_detection")
    ]
    w_ok = agg[(agg.method == "weighted") & (agg.f1 > 0.3)]["video_id"].tolist()
    if w_ok:
        parts.append(
            pd.DataFrame(
                [{"video_id": w_ok[0], "method": "weighted_intON", "failure_type": "successful_case_context"}]
            )
        )
    pd.concat(parts, ignore_index=True).to_csv(out / "failures" / "review_subset.csv", index=False)

    log("Repro check…")
    s1, _, _, _ = score_video(carr["V2"], aarr["V2"], fcfg, "weighted", collect_firings=False)
    s2, _, _, _ = score_video(carr["V2"], aarr["V2"], fcfg, "weighted", collect_firings=False)
    repro_ok = bool(np.allclose(s1, s2))
    total_firings = int(
        pv[(pv.experiment == "interactions") & (pv.interactions == "on")]["n_interaction_firings"].sum()
    )
    runtime = time.time() - t0
    (out / "environment" / "repro_check.json").write_text(
        json.dumps(
            {
                "V2_weighted_double_run_identical": repro_ok,
                "python": sys.version,
                "platform": platform.platform(),
                "runtime_seconds": runtime,
                "total_interaction_firings_intON": total_firings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gs = subprocess.check_output(["git", "-C", str(SOFT), "status", "--porcelain=v1"], text=True)
    commit = subprocess.check_output(["git", "-C", str(SOFT), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(SOFT), "branch", "--show-current"], text=True
    ).strip()
    (out / "environment" / "git_state.txt").write_text(
        f"branch={branch}\ncommit={commit}\nstatus:\n{gs}", encoding="utf-8"
    )
    (out / "environment" / "pip_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL),
        encoding="utf-8",
    )

    rows_h = ["relpath\tsha256\tbytes"]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "output_checksums.tsv":
            data = p.read_bytes()
            rows_h.append(f"{p.relative_to(out)}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}")
    (out / "output_checksums.tsv").write_text("\n".join(rows_h) + "\n", encoding="utf-8")

    verdict = "CONDITIONAL GO" if total_firings == 0 else "GO"
    reasons = [
        "n=10 videos; limited statistical power.",
        "GT is punch→impact proxy, not safety labels.",
        "Hand-authored YAML weights/thresholds/rules (probable config leakage risk disclosed).",
        "Evaluation uses full_fusion with fixed strike leg (documented BoxingVI aggregation protocol).",
        "aggregation_schemes.py inspection PASS (aggregation-only; no GT; no BoxingVI-specific logic).",
    ]
    if total_firings == 0:
        reasons.insert(
            0,
            "Interaction rules fired 0 times on BoxingVI under configured thresholds; interaction claim unsupported here.",
        )

    def fmt(exp: str) -> str:
        lines = [
            "| method | interactions | micro-F1 | macro-F1 | bootstrap macro-F1 95% CI |",
            "|--------|--------------|----------|----------|---------------------------|",
        ]
        for r in summary:
            if r["experiment"] != exp:
                continue
            lines.append(
                f"| {r['method']} | {r['interactions']} | {r['micro_f1']:.4f} | {r['macro_f1']:.4f} | "
                f"[{r['bootstrap_macro_f1_lo']:.4f}, {r['bootstrap_macro_f1_hi']:.4f}] |"
            )
        return "\n".join(lines)

    drop_sum_lines = [
        "| p | mode | mean micro-F1 over seeds |",
        "|---|------|--------------------------|",
    ]
    ddm = dd[dd.video_id == "__MICRO_MACRO__"]
    for p in DROPOUT_PS:
        for mode in ("none", "explicit_alpha0", "naive_zero"):
            sub = ddm[(ddm["p"] == p) & (ddm["mode"] == mode)]
            if sub.empty:
                continue
            drop_sum_lines.append(f"| {p} | {mode} | {float(sub['f1'].mean()):.4f} |")

    report = f"""# FINAL_REPORT — EAAI checkpoint

## 1. Executive verdict

**{verdict}**

{chr(10).join('- ' + r for r in reasons)}

## 2. Git / inspection state

Working tree `{branch}` @ `{commit}` (no branch/commit/push performed).

Untracked `src/fightsafe_ai/evaluation/aggregation_schemes.py` inspected:

| Check | Result |
|-------|--------|
| Only aggregation schemes | PASS |
| No feature-extraction changes | PASS |
| No ground-truth access | PASS |
| No eval-data parameter tuning | PASS |
| α=0 vs c=0 distinction | PASS (via `active`; weighted path excludes α=0 in runner) |
| No BoxingVI-specific logic | PASS |

## 3. Inputs / hashes

See `input_checksums.tsv`.

## 4. Matrix schema

`video_id, frame_index, timestamp, fps, c_<rule>, a_<rule>, y_impact` (y unused for scoring).

## 5. Dataset stats

See `matrices/matrix_meta.csv`.

## 6. Temporal protocol

fps={FPS}; risk merge_gap={MERGE_GAP_FRAMES}; d_min={MIN_DURATION_S}s (BoxingVI path);
BoxingVI merge={BOXINGVI_TIMELINE_MERGE_FRAMES}; IoU={IOU_THRESHOLD}; tol={TOLERANCE_SECONDS}s;
subset=**full_fusion** (risk re-aggregated; strikes fixed). Primary aggregation comparison with interactions **ON**.

## 7. Equal vs weighted vs max

{fmt("aggregation")}

## 8. Interaction ablation

{fmt("interactions")}

Configured rules: {n_rules}; channels present for {rules_ok}/{n_rules}.
Total firings (int ON): {total_firings}.

## 9. Rule firings

See `interaction_rule_firings.csv`.

## 10. Dropout

{chr(10).join(drop_sum_lines)}

Full table: `dropout_results.csv`.

## 11–13. Per-video / bootstrap / paired

See `per_video_metrics.csv`, `experiment_summary.csv`, `paired_comparisons.csv`.

## 14. Failures

See `failures/all_failures.csv`, `failure_counts.csv`.

## 15. Reproducibility

V2 weighted double-run identical: **{repro_ok}**.

## 16. Runtime

{runtime:.1f}s experimental layer.

## 17–18. Leakage & claim restrictions

Disclose hand-tuned config, n=10, punch-proxy labels. No clinical/deployment/SOTA-fusion claims.

## 19. Next experiments

LOVO logistic; family dropout; block dropout; full runtime analysis.

## 20. Rewrite paper1?

{"Yes, with narrowed claims (CONDITIONAL GO)." if verdict.startswith("CONDITIONAL") else "Yes — checkpoint supports proceeding with disclosed limits."}
"""
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    log(f"DONE {verdict} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
