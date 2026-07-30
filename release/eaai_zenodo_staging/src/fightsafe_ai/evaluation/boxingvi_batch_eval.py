"""
Batch BoxingVI evaluation: skeleton runner + event metrics for many videos.

Default: one **full_fusion** pass writes per-video ``boxingvi_predictions_<id>.json``,
``boxingvi_results_<id>.csv``, plus aggregate ``boxingvi_results_all.{csv,tex}`` under
``--output-dir``. Existing per-video outputs are skipped unless ``--force``.

Optional ``--compare-baselines`` runs multiple detector configurations under
``baselines/<name>/`` and writes ``baseline_comparison.{csv,tex}``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fightsafe_ai.evaluation.boxingvi_evaluator import (
    BoxingVIEvalResult,
    evaluate_boxingvi_video,
    write_boxingvi_results_csv,
    write_boxingvi_results_tex,
)
from fightsafe_ai.evaluation.boxingvi_skeleton_runner import run_boxingvi_skeleton_evaluation
from fightsafe_ai.evaluation.metrics import EventEvaluationResult


logger = logging.getLogger(__name__)


@dataclass
class VideoBatchRow:
    video_id: str
    status: str
    tp: int | None
    fp: int | None
    fn: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_latency_seconds: float | None
    error: str | None = None


def _read_cached_boxingvi_result(csv_path: Path, video_id: str) -> BoxingVIEvalResult | None:
    """Rebuild :class:`BoxingVIEvalResult` from ``write_boxingvi_results_csv`` output."""
    try:
        with Path(csv_path).open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)
        if not row:
            return None
        raw = EventEvaluationResult(
            n_predicted=int(row["n_predicted_qualified"]),
            n_ground_truth=int(row["n_ground_truth_punches"]),
            true_positives=int(row["true_positives"]),
            false_positives=int(row["false_positives"]),
            false_negatives=int(row["false_negatives"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1"]),
            iou_threshold=float(row["iou_threshold"]),
            tolerance_seconds=float(row["tolerance_seconds"]),
            require_same_label=False,
            mean_onset_delay_seconds=float(row["mean_detection_latency_seconds"]),
            mean_abs_onset_delay_seconds=float(row["mean_abs_detection_latency_seconds"]),
            matches=[],
            iou_by_match=[],
        )
        return BoxingVIEvalResult(
            video_id=str(video_id).strip(),
            n_ground_truth_punches=int(row["n_ground_truth_punches"]),
            n_predicted_qualified=int(row["n_predicted_qualified"]),
            true_positives=raw.true_positives,
            false_positives=raw.false_positives,
            false_negatives=raw.false_negatives,
            precision=raw.precision,
            recall=raw.recall,
            f1=raw.f1,
            mean_detection_latency_seconds=raw.mean_onset_delay_seconds,
            mean_abs_detection_latency_seconds=raw.mean_abs_onset_delay_seconds,
            iou_threshold=raw.iou_threshold,
            tolerance_seconds=raw.tolerance_seconds,
            annotation_path=str(row.get("annotation_path", "")),
            predictions_path=str(row.get("predictions_path", "")),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Cache read failed for %s: %s", csv_path, exc)
        return None


def _safe_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}".rstrip("0").rstrip(".")


AGG_TABLE_FIELDS = [
    "video_id",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "mean_latency",
    "status",
]


def _write_aggregate_table_csv(
    path: Path, rows: list[VideoBatchRow], summary: dict[str, Any]
) -> None:
    """Per-video rows plus ``__micro__`` (pooled counts + micro P/R/F1) and ``__macro__`` (macro P/R/F1, mean latency)."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AGG_TABLE_FIELDS)
        w.writeheader()
        for r in rows:
            has_metrics = r.status in {"OK", "CACHED"} and r.tp is not None
            w.writerow(
                {
                    "video_id": r.video_id,
                    "TP": "" if r.tp is None else r.tp,
                    "FP": "" if r.fp is None else r.fp,
                    "FN": "" if r.fn is None else r.fn,
                    "precision": _safe_float(r.precision) if has_metrics else "",
                    "recall": _safe_float(r.recall) if has_metrics else "",
                    "F1": _safe_float(r.f1) if has_metrics else "",
                    "mean_latency": _safe_float(r.mean_latency_seconds) if has_metrics else "",
                    "status": r.status,
                }
            )
        w.writerow(
            {
                "video_id": "__micro__",
                "TP": summary["micro_tp"],
                "FP": summary["micro_fp"],
                "FN": summary["micro_fn"],
                "precision": _safe_float(summary.get("micro_precision")),
                "recall": _safe_float(summary.get("micro_recall")),
                "F1": _safe_float(summary.get("micro_f1")),
                "mean_latency": "",
                "status": "OK",
            }
        )
        w.writerow(
            {
                "video_id": "__macro__",
                "TP": "",
                "FP": "",
                "FN": "",
                "precision": _safe_float(summary.get("macro_precision")),
                "recall": _safe_float(summary.get("macro_recall")),
                "F1": _safe_float(summary.get("macro_f1")),
                "mean_latency": _safe_float(summary.get("mean_latency_mean")),
                "status": "OK",
            }
        )


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _write_aggregate_table_tex(
    path: Path, rows: list[VideoBatchRow], summary: dict[str, Any]
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by fightsafe_ai.evaluation.boxingvi_batch_eval",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{BoxingVI batch evaluation (per-video and pooled metrics).}",
        "\\label{tab:boxingvi-batch-eval}",
        "\\begin{tabular}{@{}lcccrrrrr@{}}",
        "\\toprule",
        "\\textbf{video} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & "
        "\\textbf{P} & \\textbf{R} & \\textbf{F1} & "
        "\\textbf{mean $\\Delta$t} & \\textbf{status} \\\\",
        "\\midrule",
    ]
    for r in rows:
        st = _tex_escape(r.status)
        if r.status in {"OK", "CACHED"} and r.tp is not None:
            lines.append(
                f"{_tex_escape(r.video_id)} & {r.tp} & {r.fp} & {r.fn} & "
                f"{_safe_float(r.precision)} & {_safe_float(r.recall)} & {_safe_float(r.f1)} & "
                f"{_safe_float(r.mean_latency_seconds)} & {st} \\\\"
            )
        else:
            err = _tex_escape((r.error or "")[:32])
            lines.append(
                f"{_tex_escape(r.video_id)} & --- & --- & --- & --- & --- & --- & "
                f"\\footnotesize {err} & {st} \\\\"
            )
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{{_tex_escape('__micro__')}}} & {summary['micro_tp']} & {summary['micro_fp']} & {summary['micro_fn']} & "
        f"{_safe_float(summary.get('micro_precision'))} & "
        f"{_safe_float(summary.get('micro_recall'))} & "
        f"{_safe_float(summary.get('micro_f1'))} &  & OK \\\\"
    )
    lines.append(
        f"\\textbf{{{_tex_escape('__macro__')}}} &  &  &  & {_safe_float(summary.get('macro_precision'))} & "
        f"{_safe_float(summary.get('macro_recall'))} & {_safe_float(summary.get('macro_f1'))} & "
        f"{_safe_float(summary.get('mean_latency_mean'))} & OK \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_micro_macro(ok_results: list[BoxingVIEvalResult]) -> dict[str, Any]:
    micro_tp = sum(r.true_positives for r in ok_results)
    micro_fp = sum(r.false_positives for r in ok_results)
    micro_fn = sum(r.false_negatives for r in ok_results)
    p_d = micro_tp + micro_fp
    r_d = micro_tp + micro_fn
    micro_p = float(micro_tp / p_d) if p_d > 0 else 0.0
    micro_r = float(micro_tp / r_d) if r_d > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if micro_p + micro_r > 0 else 0.0
    f1s = [r.f1 for r in ok_results]
    precs = [r.precision for r in ok_results]
    recs = [r.recall for r in ok_results]
    lats = [r.mean_detection_latency_seconds for r in ok_results]
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    macro_p = float(sum(precs) / len(precs)) if precs else 0.0
    macro_r = float(sum(recs) / len(recs)) if recs else 0.0
    mean_latency_mean = float(sum(lats) / len(lats)) if lats else 0.0
    return {
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "mean_latency_mean": mean_latency_mean,
    }


def _write_baseline_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "TP",
        "FP",
        "FN",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_latency",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_baseline_comparison_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated - BoxingVI baseline comparison",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{BoxingVI baselines: pooled metrics (micro / macro).}",
        "\\label{tab:boxingvi-baselines}",
        "\\begin{tabular}{@{}lcccccccccc@{}}",
        "\\toprule",
        "\\textbf{baseline} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & "
        "$\\mu$P & $\\mu$R & $\\mu$F1 & MP & MR & MF1 & $\\overline{\\Delta t}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(str(row['baseline']))} & {row['TP']} & {row['FP']} & {row['FN']} & "
            f"{row['micro_precision']:.4f} & {row['micro_recall']:.4f} & {row['micro_f1']:.4f} & "
            f"{row['macro_precision']:.4f} & {row['macro_recall']:.4f} & {row['macro_f1']:.4f} & "
            f"{row['mean_latency']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch(
    *,
    dataset_root: Path,
    video_ids: list[str],
    fps: float,
    output_dir: Path,
    tolerance_seconds: float = 0.5,
    iou_threshold: float = 0.01,
    strike_percentile: float = 85.0,
    strike_merge_frames: int = 8,
    enable_strike_detector: bool = True,
    prediction_subset: str = "full_fusion",
    rolling_window: int = 5,
    min_valid_keypoints: int = 4,
    infer_skeleton_frames: bool = True,
    num_skeleton_frames: int | None = None,
    frame_offset: int | None = None,
    no_frame_offset: bool = False,
    overwrite: bool = False,
) -> tuple[list[VideoBatchRow], dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    batch_rows: list[VideoBatchRow] = []
    ok_eval_results: list[BoxingVIEvalResult] = []

    for vid in video_ids:
        stem = str(vid).strip()
        if not stem:
            continue
        err_msg: str | None = None
        pred_path = out / f"boxingvi_predictions_{stem}.json"
        per_csv = out / f"boxingvi_results_{stem}.csv"
        per_tex = out / f"boxingvi_results_{stem}.tex"

        if not overwrite and per_csv.is_file():
            cached = _read_cached_boxingvi_result(per_csv, stem)
            if cached is not None:
                ok_eval_results.append(cached)
                batch_rows.append(
                    VideoBatchRow(
                        video_id=stem,
                        status="CACHED",
                        tp=cached.true_positives,
                        fp=cached.false_positives,
                        fn=cached.false_negatives,
                        precision=cached.precision,
                        recall=cached.recall,
                        f1=cached.f1,
                        mean_latency_seconds=cached.mean_detection_latency_seconds,
                        error=None,
                    )
                )
                logger.info(
                    "SKIP %s (boxingvi_results_%s.csv exists; use --force to recompute)",
                    stem,
                    stem,
                )
                continue

        try:
            run_skeleton = bool(overwrite) or not pred_path.is_file()
            if run_skeleton:
                run_boxingvi_skeleton_evaluation(
                    dataset_root=root,
                    video_id=stem,
                    fps=float(fps),
                    rolling_window=int(rolling_window),
                    min_valid_keypoints=int(min_valid_keypoints),
                    output_dir=out,
                    rules_yaml=None,
                    enable_strike_detector=bool(enable_strike_detector),
                    strike_percentile=float(strike_percentile),
                    strike_merge_frames=int(strike_merge_frames),
                )
            if not pred_path.is_file():
                raise FileNotFoundError(f"missing predictions after runner: {pred_path}")

            res = evaluate_boxingvi_video(
                dataset_root=root,
                video_id=stem,
                predictions_path=pred_path,
                fps=float(fps),
                tolerance_seconds=float(tolerance_seconds),
                iou_threshold=float(iou_threshold),
                annotation_path=None,
                num_skeleton_frames=num_skeleton_frames,
                infer_skeleton_frames=infer_skeleton_frames,
                frame_offset=frame_offset,
                no_frame_offset=no_frame_offset,
                prediction_subset=prediction_subset,
            )
            ok_eval_results.append(res)

            if overwrite or not per_csv.is_file():
                write_boxingvi_results_csv(per_csv, res, append=False)
            if overwrite or not per_tex.is_file():
                write_boxingvi_results_tex(per_tex, res)

            batch_rows.append(
                VideoBatchRow(
                    video_id=stem,
                    status="OK",
                    tp=res.true_positives,
                    fp=res.false_positives,
                    fn=res.false_negatives,
                    precision=res.precision,
                    recall=res.recall,
                    f1=res.f1,
                    mean_latency_seconds=res.mean_detection_latency_seconds,
                    error=None,
                )
            )
            logger.info(
                "OK %s TP=%s FP=%s FN=%s F1=%.4f",
                stem,
                res.true_positives,
                res.false_positives,
                res.false_negatives,
                res.f1,
            )
        except Exception as exc:
            err_msg = str(exc)
            logger.exception("FAILED %s: %s", stem, exc)
            batch_rows.append(
                VideoBatchRow(
                    video_id=stem,
                    status="FAILED",
                    tp=None,
                    fp=None,
                    fn=None,
                    precision=None,
                    recall=None,
                    f1=None,
                    mean_latency_seconds=None,
                    error=err_msg,
                )
            )

    summary = _compute_micro_macro(ok_eval_results)
    return batch_rows, summary


def _log_final_summary(
    *,
    rows: list[VideoBatchRow],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    n_ok = sum(1 for r in rows if r.status == "OK")
    n_cached = sum(1 for r in rows if r.status == "CACHED")
    n_fail = sum(1 for r in rows if r.status == "FAILED")
    logger.info(
        "=== BoxingVI batch summary (%s) ===",
        output_dir,
    )
    logger.info(
        "Videos: %s OK (fresh), %s CACHED (skipped), %s FAILED",
        n_ok,
        n_cached,
        n_fail,
    )
    logger.info(
        "Totals  TP=%s FP=%s FN=%s",
        summary["micro_tp"],
        summary["micro_fp"],
        summary["micro_fn"],
    )
    logger.info(
        "Micro   precision=%.6f recall=%.6f F1=%.6f",
        float(summary["micro_precision"]),
        float(summary["micro_recall"]),
        float(summary["micro_f1"]),
    )
    logger.info(
        "Macro   precision=%.6f recall=%.6f F1=%.6f (mean over successful videos)",
        float(summary["macro_precision"]),
        float(summary["macro_recall"]),
        float(summary["macro_f1"]),
    )
    logger.info(
        "Mean detection latency (avg over successful videos): %.6fs",
        float(summary["mean_latency_mean"]),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch BoxingVI: skeleton runner + punch-vs-impact evaluation for many videos.",
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument(
        "--video-ids",
        nargs="+",
        required=True,
        help="Video stems e.g. V1 V2 ... V10",
    )
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--strike-percentile", type=float, default=85.0)
    p.add_argument("--strike-merge-frames", type=int, default=8)
    p.add_argument("--tolerance-seconds", type=float, default=0.5)
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.01,
        help="Min temporal IoU on dilated windows (default: 0.01)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/boxingvi_batch"),
        help="Directory for per-video outputs and boxingvi_results_all.{csv,tex}",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing boxingvi_predictions_* / boxingvi_results_* files",
    )
    p.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Run multiple detector configs under baselines/<name>/ plus baseline_comparison.{csv,tex}",
    )
    p.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Biomechanical rolling window (skeleton runner)",
    )
    p.add_argument(
        "--min-valid-keypoints",
        type=int,
        default=4,
        help="Minimum valid joints per frame (skeleton runner)",
    )
    p.add_argument(
        "--no-infer-skeleton",
        action="store_true",
        help="Do not infer skeleton frame count from .npy for GT alignment",
    )
    p.add_argument(
        "--num-skeleton-frames",
        type=int,
        default=None,
        metavar="T",
        help="Override skeleton length for all videos (optional)",
    )
    off = p.add_mutually_exclusive_group()
    off.add_argument(
        "--no-frame-offset",
        action="store_true",
        help="GT frame indices as-is (no auto offset)",
    )
    off.add_argument(
        "--frame-offset",
        type=int,
        default=None,
        metavar="N",
        help="Explicit annotation frame offset (else auto when possible)",
    )
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="[compare-baselines] Only run full_fusion baseline",
    )
    p.add_argument(
        "--velocity-percentile",
        type=float,
        default=95.0,
        help="[compare-baselines] Strike percentile for velocity_threshold baseline (default: 95)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    common_kw: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "video_ids": list(args.video_ids),
        "fps": float(args.fps),
        "tolerance_seconds": float(args.tolerance_seconds),
        "iou_threshold": float(args.iou_threshold),
        "strike_percentile": float(args.strike_percentile),
        "strike_merge_frames": int(args.strike_merge_frames),
        "rolling_window": int(args.rolling_window),
        "min_valid_keypoints": int(args.min_valid_keypoints),
        "infer_skeleton_frames": not bool(args.no_infer_skeleton),
        "num_skeleton_frames": args.num_skeleton_frames,
        "frame_offset": args.frame_offset,
        "no_frame_offset": bool(args.no_frame_offset),
        "overwrite": bool(args.force),
    }

    if not args.compare_baselines:
        rows, summary = run_batch(
            output_dir=out_dir,
            enable_strike_detector=True,
            prediction_subset="full_fusion",
            **common_kw,
        )
        _write_aggregate_table_csv(out_dir / "boxingvi_results_all.csv", rows, summary)
        _write_aggregate_table_tex(out_dir / "boxingvi_results_all.tex", rows, summary)
        _log_final_summary(rows=rows, summary=summary, output_dir=out_dir)
        n_fail = sum(1 for r in rows if r.status == "FAILED")
        logger.info(
            "Wrote %s and %s",
            out_dir / "boxingvi_results_all.csv",
            out_dir / "boxingvi_results_all.tex",
        )
        return 1 if n_fail else 0

    pct_default = float(args.strike_percentile)
    vel_pct = float(args.velocity_percentile)
    if args.skip_baselines:
        baseline_specs = [("full_fusion", True, pct_default, "full_fusion")]
    else:
        baseline_specs = [
            ("full_fusion", True, pct_default, "full_fusion"),
            ("risk_only", False, pct_default, "risk_only"),
            ("strike_detector", True, pct_default, "strike_only"),
            ("velocity_threshold", True, vel_pct, "strike_only"),
        ]

    comparison: list[dict[str, Any]] = []
    n_fail_total = 0

    # common_kw already has strike_percentile; baselines use per-row strike_pct (e.g. velocity uses --velocity-percentile).
    batch_kw = {k: v for k, v in common_kw.items() if k != "strike_percentile"}
    for name, en_strike, strike_pct, psub in baseline_specs:
        sub_out = out_dir / "baselines" / name
        rows, summary = run_batch(
            output_dir=sub_out,
            enable_strike_detector=en_strike,
            strike_percentile=strike_pct,
            prediction_subset=psub,
            **batch_kw,
        )
        _write_aggregate_table_csv(sub_out / "boxingvi_results_all.csv", rows, summary)
        _write_aggregate_table_tex(sub_out / "boxingvi_results_all.tex", rows, summary)
        _log_final_summary(rows=rows, summary=summary, output_dir=sub_out)
        n_fail_total += sum(1 for r in rows if r.status == "FAILED")
        comparison.append(
            {
                "baseline": name,
                "TP": summary["micro_tp"],
                "FP": summary["micro_fp"],
                "FN": summary["micro_fn"],
                "micro_precision": summary["micro_precision"],
                "micro_recall": summary["micro_recall"],
                "micro_f1": summary["micro_f1"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "mean_latency": summary["mean_latency_mean"],
            }
        )

    _write_baseline_comparison_csv(out_dir / "baseline_comparison.csv", comparison)
    _write_baseline_comparison_tex(out_dir / "baseline_comparison.tex", comparison)

    ff = out_dir / "baselines" / "full_fusion"
    if ff.is_dir():
        for fn in ("boxingvi_results_all.csv", "boxingvi_results_all.tex"):
            src = ff / fn
            if src.is_file():
                shutil.copy2(src, out_dir / fn)

    logger.info(
        "compare-baselines: wrote baseline_comparison + copied full_fusion boxingvi_results_all to %s. "
        "Total FAILED rows (all baselines): %s",
        out_dir,
        n_fail_total,
    )
    return 1 if n_fail_total else 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "VideoBatchRow",
    "main",
    "run_batch",
]
