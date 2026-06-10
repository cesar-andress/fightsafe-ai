"""
Pipeline output validation for FightSafe AI (artifacts, schema, invariants, coverage).

**Orchestration:** :func:`run_quality_checks` aggregates all checks and optional metrics
(``max_risk_score``, ``n_events``, ``pose_coverage_percent``, …). **Dataclasses**
:data:`QualityCheckResult` and :data:`QualityReport` describe the outcome.

**End-to-end success (guidance).** A run should be considered successful when the
**expected output files** exist (e.g. ``pose_keypoints.csv``, ``risk_scores.csv``,
``qa_report.json``) and, when you run the QA pass, the report’s ``passed`` is true
(or you accept documented ``warn`` states for your use case). **Do not** infer
success or failure from **TensorFlow Lite** / **MediaPipe** **stderr** messages
alone (XNNPACK delegate, Feedback Manager, NORM_RECT); those are third-party
and often appear in **CPU** mode. See ``docs/troubleshooting.md``.

**Low-level helpers** (per-file and per-invariant) are public for testing and reuse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from fightsafe_ai.qa import llm_status as qa_llm_status, metrics as qa_metrics


logger = logging.getLogger(__name__)

Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class QualityCheckResult:
    """
    One atomic QA outcome.

    name
        Stable identifier for the check.
    status
        ``"pass"`` = OK; ``"warn"`` = non-blocking issue; ``"fail"`` = validation error.
    message
        Human-readable one-line summary.
    details
        Optional technical detail (paths, row counts, exception text).
    """

    name: str
    status: Status
    message: str
    details: str = ""


@dataclass(frozen=True)
class QualityReport:
    """
    Summary of a :func:`run_quality_checks` run.

    run_dir
        Assessed run directory.
    passed
        ``True`` if no check has status ``"fail"`` (warnings do not fail the run).
    total_checks
        Number of atomic :class:`QualityCheckResult` records.
    failed_checks
        Count of results with status ``"fail"``.
    warnings
        Messages for checks with status ``"warn"`` (one entry per such check’s message).
    metrics
        Computed values (e.g. ``max_risk_score``, ``n_events``, ``pose_coverage_percent``).
    results
        Full per-check detail.
    """

    run_dir: Path
    passed: bool
    total_checks: int
    failed_checks: int
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    results: list[QualityCheckResult] = field(default_factory=list)


# --- Column sets -----------------------------------------------------------------

VALID_RISK_LEVELS: Final[frozenset[str]] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

RISK_SCORES_REQUIRED_COLS: Final[Sequence[str]] = (
    "frame_id",
    "timestamp",
    "risk_score",
    "risk_level",
)

FEATURES_REQUIRED_COLS: Final[Sequence[str]] = ("frame_id",)

POSE_KEYPOINTS_REQUIRED_COLS: Final[Sequence[str]] = (
    "frame_id",
    "keypoint_name",
    "x",
    "y",
)


def _result(
    name: str,
    status: Status,
    message: str,
    details: str = "",
) -> QualityCheckResult:
    r = QualityCheckResult(name=name, status=status, message=message, details=details)
    if status == "pass":
        logger.info("QA %s: pass — %s", name, message)
    elif status == "warn":
        logger.warning("QA %s: warn — %s", name, message)
    else:
        logger.error("QA %s: fail — %s", name, message)
    if details:
        logger.debug("QA %s: %s", name, details)
    return r


def _missing(cols: Sequence[str], present: set[str]) -> list[str]:
    return [c for c in cols if c not in present]


def _aggregate(
    run_dir: Path, results: list[QualityCheckResult], metrics: dict[str, Any]
) -> QualityReport:
    failed = sum(1 for r in results if r.status == "fail")
    warn_msgs = [r.message for r in results if r.status == "warn"]
    return QualityReport(
        run_dir=run_dir,
        passed=(failed == 0),
        total_checks=len(results),
        failed_checks=failed,
        warnings=warn_msgs,
        metrics=metrics,
        results=results,
    )


def _analytic_metric_checks(
    m: dict[str, Any],
    risk_df: pd.DataFrame | None,
    events_json_path: Path,
    n_events: int,
) -> list[QualityCheckResult]:
    """Warnings for empty events (with risk data) and constant risk score."""
    out: list[QualityCheckResult] = []
    has_events_file = events_json_path.is_file()
    if has_events_file and n_events == 0 and risk_df is not None and len(risk_df) > 0:
        out.append(
            _result(
                "metric_no_events",
                "warn",
                "No events detected in events.json while risk data is present.",
            )
        )
    if qa_metrics.is_constant_risk_score(risk_df):
        v = m.get("average_risk_score")
        out.append(
            _result(
                "metric_constant_risk",
                "warn",
                f"Constant risk score across all sampled rows (value ≈ {v!s}).",
            )
        )
    return out


def run_quality_checks(
    run_dir: Path,
    *,
    require_frames: bool = True,
) -> QualityReport:
    """
    Run all QA checks for a pipeline run directory.

    Validates, where applicable: required file layout, readable CSVs, required columns,
    monotonic ``timestamp`` in ``risk_scores.csv``, ``risk_score`` in ``[0,1]``,
    ``risk_level`` in the allowed set, event time order in ``events.json``,
    non-empty overlay video, non-empty report, and pose image coverage.

    **Metrics** (see :mod:`fightsafe_ai.qa.metrics`) always merge canonical keys:
    ``total_frames``, ``frames_with_pose``, ``pose_coverage_percent``,
    ``average_risk_score``, ``max_risk_score``, ``number_of_events``, ``duration_seconds``,
    plus :mod:`fightsafe_ai.qa.llm_status` fields ``llm_enabled``, ``llm_used``,
    ``llm_success_rate`` (Ollama need not be running). LLM-related issues are **warn**-only;
    they never set ``passed`` to ``False``.

    Legacy fields such as ``n_risk_rows``, ``n_events``, and nested ``pose_coverage`` are also set.
    """
    from fightsafe_ai.qa import dataset_checks

    run_path = run_dir.expanduser().resolve()
    metrics: dict[str, Any] = {"run_id": run_path.name, "run_dir": str(run_path)}
    out: list[QualityCheckResult] = []
    dfr: pd.DataFrame | None = None
    dfp: pd.DataFrame | None = None
    n_events: int = 0

    out.append(check_run_directory_exists(run_path))
    if out[-1].status == "fail":
        return _aggregate(run_path, out, metrics)

    out.extend(check_required_artifacts(run_path, require_frames=require_frames))

    if (run_path / "risk_scores.csv").is_file():
        chain, dfr = validate_risk_artifact(run_path / "risk_scores.csv")
        out.extend(chain)
        if dfr is not None:
            metrics["n_risk_rows"] = len(dfr)
    else:
        logger.info("QA: risk_scores.csv missing; skipping risk content checks")

    if (run_path / "features.csv").is_file():
        out.extend(validate_feature_artifact(run_path / "features.csv"))
        r2, dff = try_read_csv(run_path / "features.csv", "features")
        if r2.status == "pass" and dff is not None:
            metrics["n_feature_rows"] = len(dff)

    if (run_path / "pose_keypoints.csv").is_file():
        out.extend(validate_pose_artifact(run_path / "pose_keypoints.csv"))
        r3, dfp = try_read_csv(run_path / "pose_keypoints.csv", "pose")
        if r3.status == "pass" and dfp is not None:
            metrics["n_pose_keypoint_rows"] = len(dfp)
    else:
        dfp = None

    if (run_path / "events.json").is_file():
        ev_r, evs = load_events_list_from_json(run_path / "events.json")
        out.append(ev_r)
        if evs is not None:
            out.extend(check_event_time_order(evs))
            n_events = len(evs)
            metrics["n_events"] = n_events
        else:
            n_events = 0
    else:
        logger.info("QA: events.json missing; skipping event time checks")
        metrics["n_events"] = 0
        n_events = 0

    if (run_path / "report.md").is_file():
        out.append(check_report_md(run_path / "report.md"))
        metrics["report_size_bytes"] = int((run_path / "report.md").stat().st_size)
    else:
        logger.info("QA: report.md missing; file check already in required_artifacts")

    if (run_path / "output_overlay.mp4").is_file():
        out.append(check_video_file(run_path / "output_overlay.mp4"))
        metrics["output_overlay_bytes"] = int((run_path / "output_overlay.mp4").stat().st_size)

    frames_p = run_path / "frames"
    pose_p = run_path / "pose_keypoints.csv"
    cov_pct, pmeta = dataset_checks.pose_coverage_metrics(frames_p, pose_p)
    metrics["pose_coverage"] = pmeta
    if cov_pct is not None:
        metrics["pose_coverage_percent"] = float(cov_pct)
    out.extend(dataset_checks.coverage_to_results(cov_pct, pmeta))

    std = qa_metrics.build_run_metrics(
        frames_dir=frames_p,
        risk_df=dfr,
        pose_df=dfp,
        number_of_events=n_events,
        pose_coverage_percent=cov_pct,
    )
    metrics.update(std)
    metrics["n_events"] = n_events
    if std.get("number_of_events") is not None:
        metrics["number_of_events"] = int(std["number_of_events"])
    out.extend(_analytic_metric_checks(std, dfr, run_path / "events.json", n_events))

    llm_m, llm_warns = qa_llm_status.build_llm_qa_metrics(run_path, n_events)
    metrics.update(llm_m)
    for wname, wmsg in llm_warns:
        out.append(_result(wname, "warn", wmsg))

    return _aggregate(run_path, out, metrics)


def quality_report_to_dict(report: QualityReport) -> dict[str, Any]:
    """Serialize a :class:`QualityReport` to JSON-compatible dict (``qa_report.json``)."""
    return {
        "status": "pass" if report.passed else "fail",
        "passed": report.passed,
        "total_checks": report.total_checks,
        "failed_checks": report.failed_checks,
        "warning_count": len(report.warnings),
        "warnings": list(report.warnings),
        "metrics": qa_metrics.merge_metrics(dict(report.metrics)),
        "run_dir": str(report.run_dir),
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "details": r.details,
            }
            for r in report.results
        ],
    }


def write_qa_report_json(
    output_path: Path,
    report: QualityReport,
) -> Path:
    """
    Write ``<run_dir>/qa_report.json`` (e.g. ``runs/demo/qa_report.json``).

    Parent directories are created if missing.
    """
    out = output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = quality_report_to_dict(report)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# --- Per-file / invariant checks -------------------------------------------------


def check_run_directory_exists(run_dir: Path) -> QualityCheckResult:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        return _result("run_directory", "fail", f"Not a directory: {run_dir}")
    return _result("run_directory", "pass", f"Run directory: {run_dir}")


def check_required_artifacts(
    run_dir: Path,
    *,
    require_frames: bool = True,
) -> list[QualityCheckResult]:
    """Check presence of expected MVP file names (and optional ``frames/``)."""
    run_dir = run_dir.resolve()
    out: list[QualityCheckResult] = []
    files: list[tuple[str, Path]] = [
        ("file_pose_keypoints_csv", run_dir / "pose_keypoints.csv"),
        ("file_features_csv", run_dir / "features.csv"),
        ("file_risk_scores_csv", run_dir / "risk_scores.csv"),
        ("file_events_json", run_dir / "events.json"),
        ("file_output_overlay_mp4", run_dir / "output_overlay.mp4"),
        ("file_report_md", run_dir / "report.md"),
    ]
    for name, p in files:
        if p.is_file():
            out.append(_result(name, "pass", f"Found: {p.name}"))
        else:
            out.append(_result(name, "fail", f"Missing or not a file: {p}"))

    frames = run_dir / "frames"
    if require_frames:
        if frames.is_dir():
            out.append(_result("dir_frames", "pass", f"Found: {frames}"))
        else:
            out.append(_result("dir_frames", "fail", f"Expected directory missing: {frames}"))
    elif not frames.is_dir():
        out.append(
            _result("dir_frames", "warn", f"Optional frames/ not present: {frames}"),
        )
    return out


def try_read_csv(path: Path, label: str) -> tuple[QualityCheckResult, pd.DataFrame | None]:
    path = path.expanduser().resolve()
    if not path.is_file():
        return _result(f"csv_read_{label}", "fail", f"File missing: {path}"), None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return (
            _result(
                f"csv_read_{label}",
                "fail",
                f"Parser error: {path}",
                details=repr(exc),
            ),
            None,
        )
    return _result(
        f"csv_read_{label}",
        "pass",
        f"Read {len(df)} row(s) from {path.name}",
    ), df


def validate_risk_artifact(risk_path: Path) -> tuple[list[QualityCheckResult], pd.DataFrame | None]:
    r, df = try_read_csv(risk_path, "risk_scores")
    if r.status == "fail" or df is None:
        return [r], None
    cols = {str(c) for c in df.columns}
    miss = _missing(RISK_SCORES_REQUIRED_COLS, cols)
    if miss:
        return [r, _result("risk_scores_columns", "fail", f"Missing columns: {miss}")], df
    chain = [r, _result("risk_scores_columns", "pass", "All required risk columns present")]
    chain.extend(check_risk_score_range(df))
    chain.extend(check_risk_level_values(df))
    chain.extend(check_monotonic_timestamps(df))
    return chain, df


def validate_feature_artifact(feat_path: Path) -> list[QualityCheckResult]:
    r, df = try_read_csv(feat_path, "features")
    if r.status == "fail" or df is None:
        return [r]
    miss = _missing(FEATURES_REQUIRED_COLS, {str(c) for c in df.columns})
    if miss:
        return [r, _result("features_columns", "fail", f"Missing: {miss}")]
    return [r, _result("features_columns", "pass", "All required feature columns present")]


def validate_pose_artifact(pose_path: Path) -> list[QualityCheckResult]:
    r, df = try_read_csv(pose_path, "pose")
    if r.status == "fail" or df is None:
        return [r]
    miss = _missing(POSE_KEYPOINTS_REQUIRED_COLS, {str(c) for c in df.columns})
    if miss:
        return [r, _result("pose_columns", "fail", f"Missing: {miss}")]
    return [r, _result("pose_columns", "pass", "All required pose columns present")]


def check_risk_score_range(df: pd.DataFrame) -> list[QualityCheckResult]:
    if "risk_score" not in df.columns or len(df) == 0:
        return [
            _result(
                "risk_score_range",
                "warn",
                "No rows or no risk_score; skip [0,1] check",
            )
        ]
    s = pd.to_numeric(df["risk_score"], errors="coerce")
    oob = s[(s < 0.0) | (s > 1.0) | s.isna()]
    n = int(oob.count())
    if n:
        return [_result("risk_score_range", "fail", f"{n} value(s) outside [0,1] or non-numeric")]
    return [_result("risk_score_range", "pass", "All risk_score in [0,1]")]


def check_risk_level_values(df: pd.DataFrame) -> list[QualityCheckResult]:
    if "risk_level" not in df.columns or len(df) == 0:
        return [
            _result("risk_level_values", "warn", "No risk_level; skip value check"),
        ]
    bad: list[str] = []
    for v in df["risk_level"].dropna().astype(str).unique().tolist():
        if str(v).strip().upper() not in VALID_RISK_LEVELS:
            bad.append(str(v))
    if bad:
        return [
            _result("risk_level_values", "fail", f"Invalid level(s): {bad[:8]}"),
        ]
    return [_result("risk_level_values", "pass", "risk_level in valid set")]


def check_monotonic_timestamps(df: pd.DataFrame) -> list[QualityCheckResult]:
    if "timestamp" not in df.columns or len(df) < 2:
        return [
            _result(
                "monotonic_timestamps",
                "warn",
                "Skip monotonicity: <2 rows or no timestamp",
            )
        ]
    t = pd.to_numeric(df["timestamp"], errors="coerce")
    if t.isna().all():
        return [_result("monotonic_timestamps", "fail", "All timestamps non-numeric")]
    d = t.diff().iloc[1:]
    if (d < -1e-9).any():
        n = int((d < -1e-9).sum())
        return [
            _result("monotonic_timestamps", "fail", f"Non-increasing in {n} place(s)"),
        ]
    return [_result("monotonic_timestamps", "pass", "Timestamps non-decreasing")]


def check_event_time_order(events: list[dict[str, Any]]) -> list[QualityCheckResult]:
    if not events:
        return [_result("events_time_order", "pass", "No events (empty list)")]

    def _parse(ev: dict[str, Any]) -> tuple[str | int | None, str | int | None]:
        t0 = ev.get("start_time", ev.get("startTime"))
        t1 = ev.get("end_time", ev.get("endTime"))
        if t0 is None or t1 is None:
            return None, None
        return t0, t1

    bad: list[str] = []
    for i, ev in enumerate(events):
        t0, t1 = _parse(ev)
        if t0 is None and t1 is None:
            bad.append(f"event[{i}]: no times")
            continue
        if t0 is None or t1 is None:
            bad.append(f"event[{i}]: partial times")
            continue
        try:
            a, b = float(t0), float(t1)
        except (TypeError, ValueError):
            bad.append(f"event[{i}]: not numeric")
            continue
        if a >= b:
            bad.append(f"event[{i}]: {a} >= {b}")
    if bad:
        return [
            _result(
                "events_time_order",
                "fail",
                "start_time < end_time not satisfied: " + "; ".join(bad[:6]),
            )
        ]
    return [_result("events_time_order", "pass", "All events have start_time < end_time")]


def load_events_list_from_json(
    path: Path,
) -> tuple[QualityCheckResult, list[dict[str, Any]] | None]:
    path = path.expanduser().resolve()
    if not path.is_file():
        return _result("events_json", "fail", f"Missing: {path}"), None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            _result("events_json", "fail", f"Invalid JSON: {path}", details=repr(exc)),
            None,
        )
    if raw is None:
        return _result("events_json", "warn", "JSON is null"), None
    if isinstance(raw, list):
        evs = [x for x in raw if isinstance(x, dict)]
        return _result("events_json", "pass", f"Array, {len(evs)} object(s)"), evs
    if isinstance(raw, dict):
        return _result("events_json", "pass", "Root object, length 1"), [raw]
    return _result("events_json", "fail", "Root must be object or array"), None


# Backward-compatible alias
def load_events_list(
    path: Path,
) -> tuple[QualityCheckResult, list[dict[str, Any]] | None]:
    return load_events_list_from_json(path)


def check_report_md(path: Path) -> QualityCheckResult:
    if not path.is_file():
        return _result("report_md_content", "fail", f"Missing: {path}")
    if path.stat().st_size <= 0:
        return _result("report_md_content", "fail", f"Empty file: {path}")
    return _result("report_md_content", "pass", f"Non-empty: {path.stat().st_size} bytes")


def check_video_file(video_path: Path) -> QualityCheckResult:
    p = video_path.expanduser().resolve()
    if not p.is_file():
        return _result("output_video", "fail", f"Missing: {p}")
    sz = int(p.stat().st_size)
    if sz <= 0:
        return _result("output_video", "fail", f"0-byte file: {p}")
    return _result("output_video", "pass", f"size={sz} bytes")


__all__ = [
    "FEATURES_REQUIRED_COLS",
    "POSE_KEYPOINTS_REQUIRED_COLS",
    "RISK_SCORES_REQUIRED_COLS",
    "VALID_RISK_LEVELS",
    "QualityCheckResult",
    "QualityReport",
    "Status",
    "check_event_time_order",
    "check_monotonic_timestamps",
    "check_report_md",
    "check_required_artifacts",
    "check_risk_level_values",
    "check_risk_score_range",
    "check_run_directory_exists",
    "check_video_file",
    "load_events_list",
    "load_events_list_from_json",
    "quality_report_to_dict",
    "run_quality_checks",
    "try_read_csv",
    "validate_feature_artifact",
    "validate_pose_artifact",
    "validate_risk_artifact",
    "write_qa_report_json",
]
