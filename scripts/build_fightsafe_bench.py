#!/usr/bin/env python3
"""
Build FightSafe-Bench exports from student annotation spreadsheets.

Reads ``.xlsx`` files (``Annotations`` sheet) matching
``paper3/FightSafeBench_Annotation_Template.xlsx``, validates rows, merges,
assigns unique ``event_id`` values, and writes:

- ``data/FightSafeBench/events.csv``
- ``data/FightSafeBench/dataset_statistics.json``
- ``../sports/dataset_summary.md`` (or ``--summary-path``)

Example::

    python scripts/build_fightsafe_bench.py
    python scripts/build_fightsafe_bench.py \\
        --input-dir data/FightSafeBench/annotations \\
        --output-dir data/FightSafeBench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "FightSafeBench" / "annotations"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "FightSafeBench"
DEFAULT_SUMMARY_PATH = _REPO_ROOT / ".." / "sports" / "dataset_summary.md"
ANNOTATIONS_SHEET = "Annotations"

REQUIRED_COLUMNS = (
    "alumno_id",
    "video_url",
    "video_id",
    "start_time",
    "end_time",
    "event_class",
    "visibility_tier",
    "confidence",
    "comments",
)

COLUMN_ALIASES = {
    "annotator_id": "alumno_id",
}

VALID_EVENT_CLASSES = frozenset(
    {
        "hand_tap",
        "foot_tap",
        "verbal_submission",
        "knockdown_proxy",
        "defensive_breakdown",
        "post_impact_inactivity",
        "hard_negative_defensive_shell",
        "hard_negative_grappling_transition",
        "hard_negative_instructional_pause",
        "hard_negative_recovery_after_impact",
        "hard_negative_ambiguous_motion",
    }
)

VALID_VISIBILITY_TIERS = frozenset({"V0", "V1", "V2", "VX"})
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

EXAMPLE_COMMENT_PREFIXES = ("ejemplo", "example row", "delete or overwrite", "borrar o sustituir")


@dataclass
class ValidationIssue:
    source_file: str
    row_number: int
    message: str


@dataclass
class BuildResult:
    frames: list[pd.DataFrame] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    skipped_example_rows: int = 0
    skipped_empty_rows: int = 0


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Reading .xlsx annotations requires openpyxl. Install with: pip install openpyxl"
        ) from exc


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return bool(isinstance(value, str) and not value.strip())


def _normalize_str(value: object) -> str:
    if _is_blank(value):
        return ""
    return str(value).strip()


def _parse_time(value: object) -> float | None:
    if _is_blank(value):
        return None
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    if t != t or abs(t) == float("inf"):
        return None
    return t


def _is_example_row(row: pd.Series) -> bool:
    comments = _normalize_str(row.get("comments", "")).lower()
    if not comments:
        return False
    return any(comments.startswith(prefix) for prefix in EXAMPLE_COMMENT_PREFIXES)


def discover_spreadsheets(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    files = sorted(input_dir.glob("*.xlsx"))
    return [p for p in files if "template" not in p.name.lower()]


def load_annotation_spreadsheet(path: Path) -> pd.DataFrame:
    _require_openpyxl()
    raw = pd.read_excel(path, sheet_name=ANNOTATIONS_SHEET, engine="openpyxl")
    raw.columns = [_normalize_str(c).lower() for c in raw.columns]
    rename = {old: new for old, new in COLUMN_ALIASES.items() if old in raw.columns}
    if rename:
        raw = raw.rename(columns=rename)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return raw[list(REQUIRED_COLUMNS)].copy()


def validate_row(
    row: pd.Series,
    *,
    source_file: str,
    row_number: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    alumno_id = _normalize_str(row["alumno_id"])
    video_id = _normalize_str(row["video_id"])
    event_class = _normalize_str(row["event_class"])
    visibility_tier = _normalize_str(row["visibility_tier"])
    confidence = _normalize_str(row["confidence"]).lower()

    if not alumno_id:
        issues.append(ValidationIssue(source_file, row_number, "alumno_id is required"))
    if not video_id:
        issues.append(ValidationIssue(source_file, row_number, "video_id is required"))

    start = _parse_time(row["start_time"])
    end = _parse_time(row["end_time"])
    if start is None:
        issues.append(ValidationIssue(source_file, row_number, "start_time must be a finite number"))
    elif start < 0:
        issues.append(ValidationIssue(source_file, row_number, "start_time must be >= 0"))
    if end is None:
        issues.append(ValidationIssue(source_file, row_number, "end_time must be a finite number"))
    elif end <= 0:
        issues.append(ValidationIssue(source_file, row_number, "end_time must be > 0"))
    if start is not None and end is not None and start >= end:
        issues.append(
            ValidationIssue(
                source_file,
                row_number,
                f"start_time ({start}) must be < end_time ({end})",
            )
        )

    if not event_class:
        issues.append(ValidationIssue(source_file, row_number, "event_class is required"))
    elif event_class not in VALID_EVENT_CLASSES:
        issues.append(
            ValidationIssue(
                source_file,
                row_number,
                f"invalid event_class {event_class!r}; allowed: {sorted(VALID_EVENT_CLASSES)}",
            )
        )

    if not visibility_tier:
        issues.append(ValidationIssue(source_file, row_number, "visibility_tier is required"))
    elif visibility_tier not in VALID_VISIBILITY_TIERS:
        issues.append(
            ValidationIssue(
                source_file,
                row_number,
                f"invalid visibility_tier {visibility_tier!r}; allowed: {sorted(VALID_VISIBILITY_TIERS)}",
            )
        )

    if not confidence:
        issues.append(ValidationIssue(source_file, row_number, "confidence is required"))
    elif confidence not in VALID_CONFIDENCE:
        issues.append(
            ValidationIssue(
                source_file,
                row_number,
                f"invalid confidence {confidence!r}; allowed: {sorted(VALID_CONFIDENCE)}",
            )
        )

    return issues


def load_all_annotations(input_dir: Path) -> BuildResult:
    result = BuildResult()
    paths = discover_spreadsheets(input_dir)
    if not paths:
        print(f"No annotation spreadsheets found in {input_dir} (excluding *Template*).")
        return result

    for path in paths:
        try:
            df = load_annotation_spreadsheet(path)
        except ValueError as exc:
            result.issues.append(
                ValidationIssue(path.name, 0, str(exc)),
            )
            continue

        kept_rows: list[dict[str, object]] = []
        for idx, row in df.iterrows():
            excel_row = int(idx) + 2  # header + 1-based sheet row
            if all(_is_blank(row[c]) for c in REQUIRED_COLUMNS):
                result.skipped_empty_rows += 1
                continue
            if _is_example_row(row):
                result.skipped_example_rows += 1
                continue

            row_issues = validate_row(row, source_file=path.name, row_number=excel_row)
            result.issues.extend(row_issues)
            if row_issues:
                continue

            kept_rows.append(
                {
                    "alumno_id": _normalize_str(row["alumno_id"]),
                    "video_url": _normalize_str(row["video_url"]),
                    "video_id": _normalize_str(row["video_id"]),
                    "start_time": float(_parse_time(row["start_time"]) or 0.0),
                    "end_time": float(_parse_time(row["end_time"]) or 0.0),
                    "event_class": _normalize_str(row["event_class"]),
                    "visibility_tier": _normalize_str(row["visibility_tier"]),
                    "confidence": _normalize_str(row["confidence"]).lower(),
                    "comments": _normalize_str(row["comments"]),
                    "source_file": path.name,
                }
            )

        if kept_rows:
            result.frames.append(pd.DataFrame(kept_rows))
            result.source_files.append(path.name)

    return result


def assign_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["event_id"] = pd.Series(dtype=str)
        return df

    out = df.sort_values(["video_id", "start_time", "end_time"], kind="stable").copy()
    counters: Counter[str] = Counter()
    event_ids: list[str] = []
    for video_id in out["video_id"]:
        counters[video_id] += 1
        event_ids.append(f"{video_id}_e{counters[video_id]:04d}")
    out["event_id"] = event_ids

    columns = [
        "event_id",
        "alumno_id",
        "video_url",
        "video_id",
        "start_time",
        "end_time",
        "event_class",
        "visibility_tier",
        "confidence",
        "comments",
        "source_file",
    ]
    return out[columns]


def build_statistics(df: pd.DataFrame, *, source_files: list[str]) -> dict[str, object]:
    event_class_counts = dict(sorted(Counter(df["event_class"]).items())) if not df.empty else {}
    visibility_distribution = (
        dict(sorted(Counter(df["visibility_tier"]).items())) if not df.empty else {}
    )
    confidence_distribution = (
        dict(sorted(Counter(df["confidence"]).items())) if not df.empty else {}
    )
    annotator_counts = dict(sorted(Counter(df["alumno_id"]).items())) if not df.empty else {}

    videos = int(df["video_id"].nunique()) if not df.empty else 0
    events = len(df)

    per_video: dict[str, int] = {}
    if not df.empty:
        per_video = {
            str(k): int(v)
            for k, v in sorted(df.groupby("video_id").size().items(), key=lambda x: x[0])
        }

    return {
        "benchmark": "FightSafe-Bench",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_spreadsheets": source_files,
        "videos": videos,
        "events": events,
        "events_per_video": per_video,
        "event_class_counts": event_class_counts,
        "visibility_distribution": visibility_distribution,
        "confidence_distribution": confidence_distribution,
        "annotator_counts": annotator_counts,
        "validation_schema": {
            "event_class": sorted(VALID_EVENT_CLASSES),
            "visibility_tier": sorted(VALID_VISIBILITY_TIERS),
            "confidence": sorted(VALID_CONFIDENCE),
        },
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def write_dataset_summary(
    stats: dict[str, object],
    *,
    issues: list[ValidationIssue],
    skipped_example_rows: int,
    skipped_empty_rows: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# FightSafe-Bench — Dataset Summary",
        "",
        f"_Generated by `scripts/build_fightsafe_bench.py` at {stats['generated_at_utc']}._",
        "",
        "## Overview",
        "",
        f"- **Videos:** {stats['videos']}",
        f"- **Events:** {stats['events']}",
        f"- **Source spreadsheets:** {len(stats['source_spreadsheets'])}",
        "",
    ]

    sources = stats["source_spreadsheets"]
    if sources:
        lines.append("### Source files")
        lines.append("")
        for name in sources:
            lines.append(f"- `{name}`")
        lines.append("")

    if skipped_example_rows or skipped_empty_rows:
        lines.extend(
            [
                "### Rows skipped during ingest",
                "",
                f"- Example/template rows: {skipped_example_rows}",
                f"- Empty rows: {skipped_empty_rows}",
                "",
            ]
        )

    lines.extend(["## Events per video", ""])
    per_video = stats.get("events_per_video") or {}
    if per_video:
        rows = [[vid, str(count)] for vid, count in per_video.items()]
        lines.append(_markdown_table(["video_id", "events"], rows))
    else:
        lines.append("_No adjudicated events in current spreadsheets._\n")

    lines.extend(["## Event class counts", ""])
    ecc = stats.get("event_class_counts") or {}
    if ecc:
        rows = [[cls, str(n)] for cls, n in ecc.items()]
        lines.append(_markdown_table(["event_class", "count"], rows))
    else:
        lines.append("_No events._\n")

    lines.extend(["## Visibility distribution", ""])
    vis = stats.get("visibility_distribution") or {}
    if vis:
        rows = [[tier, str(n)] for tier, n in vis.items()]
        lines.append(_markdown_table(["visibility_tier", "count"], rows))
    else:
        lines.append("_No events._\n")

    lines.extend(["## Confidence distribution", ""])
    conf = stats.get("confidence_distribution") or {}
    if conf:
        rows = [[level, str(n)] for level, n in conf.items()]
        lines.append(_markdown_table(["confidence", "count"], rows))
    else:
        lines.append("_No events._\n")

    lines.extend(["## Annotator counts", ""])
    ann = stats.get("annotator_counts") or {}
    if ann:
        rows = [[aid, str(n)] for aid, n in ann.items()]
        lines.append(_markdown_table(["alumno_id", "intervals"], rows))
    else:
        lines.append("_No events._\n")

    if issues:
        lines.extend(["## Validation issues", ""])
        rows = [[i.source_file, str(i.row_number), i.message] for i in issues]
        lines.append(_markdown_table(["source_file", "row", "message"], rows))

    lines.extend(
        [
            "## Export paths",
            "",
            "- `data/FightSafeBench/events.csv`",
            "- `data/FightSafeBench/dataset_statistics.json`",
            "",
            "See also: `../sports/FightSafeBench_Annotation_Guide_v1.md`, "
            "`../sports/fightsafe_benchmark_spec.md`.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_outputs(
    df: pd.DataFrame,
    stats: dict[str, object],
    *,
    output_dir: Path,
    summary_path: Path,
    issues: list[ValidationIssue],
    skipped_example_rows: int,
    skipped_empty_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "events.csv"
    df.to_csv(events_path, index=False)

    stats_path = output_dir / "dataset_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_dataset_summary(
        stats,
        issues=issues,
        skipped_example_rows=skipped_example_rows,
        skipped_empty_rows=skipped_empty_rows,
        output_path=summary_path,
    )

    print(f"Wrote {events_path.relative_to(_REPO_ROOT)} ({len(df)} events)")
    print(f"Wrote {stats_path.relative_to(_REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(_REPO_ROOT)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FightSafe-Bench annotation spreadsheets into benchmark exports.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with .xlsx annotation files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV/JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Markdown summary path (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Exit with code 1 when no valid events are exported.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir if args.input_dir.is_absolute() else _REPO_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else _REPO_ROOT / args.output_dir
    summary_path = (
        args.summary_path if args.summary_path.is_absolute() else _REPO_ROOT / args.summary_path
    )

    result = load_all_annotations(input_dir)

    if result.issues:
        print(f"Validation failed with {len(result.issues)} issue(s):", file=sys.stderr)
        for issue in result.issues:
            loc = f"{issue.source_file}:{issue.row_number}" if issue.row_number else issue.source_file
            print(f"  [{loc}] {issue.message}", file=sys.stderr)
        return 1

    merged = (
        pd.concat(result.frames, ignore_index=True)
        if result.frames
        else pd.DataFrame(columns=[*REQUIRED_COLUMNS, "source_file"])
    )
    events = assign_event_ids(merged)
    stats = build_statistics(events, source_files=result.source_files)

    export_outputs(
        events,
        stats,
        output_dir=output_dir,
        summary_path=summary_path,
        issues=result.issues,
        skipped_example_rows=result.skipped_example_rows,
        skipped_empty_rows=result.skipped_empty_rows,
    )

    if args.fail_on_empty and events.empty:
        print("No events exported.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
