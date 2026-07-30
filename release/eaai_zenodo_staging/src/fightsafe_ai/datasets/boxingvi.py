"""
BoxingVI-style local layout under a configurable root (no automatic download).

Expected root (e.g. ``data/boxingvi/``)::

    annotations/   V1.xlsx, V2.xlsx, ...  (flexible column names; multi-sheet)
    skeleton/      V1.npy, V2.npy, ...   (AlphaPose / COCO-style 2D keypoints)
    rgb/           V1.mp4, ...            (optional)
    metadata/      Meta_data.ods, ...     (optional; not read here)

Reading ``.xlsx`` needs ``openpyxl`` (``pip install openpyxl``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BoxingVIEvent:
    """One interval label from a BoxingVI-style Excel sheet."""

    start_frame: int
    end_frame: int
    class_name: str


@dataclass
class AnnotationInspectReport:
    """Per-file summary for ``--inspect`` (does not raise on parse failure)."""

    path: Path
    ok: bool
    status: str
    sheets_used: list[str] = field(default_factory=list)
    raw_shape: tuple[int, int] | None = None
    columns_detected: tuple[str, str, str] | None = None
    valid_event_count: int = 0
    first_event: BoxingVIEvent | None = None
    error: str | None = None
    inspect_debug: str | None = None


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading BoxingVI .xlsx annotations requires the 'openpyxl' package. "
            "Install with: pip install openpyxl"
        ) from exc


def _slug_key(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _is_ignored_column(name: Any) -> bool:
    return str(name).strip().lower().startswith("unnamed")


# Punch labels (BoxingVI-style); used for heuristic column detection.
_PUNCH_EXACT: frozenset[str] = frozenset(
    {
        "Jab",
        "Cross",
        "Lead Hook",
        "Rear Hook",
        "Lead Uppercut",
        "Rear Uppercut",
    }
)
_PUNCH_SUBSTRINGS: tuple[str, ...] = (
    "jab",
    "cross",
    "hook",
    "uppercut",
    "lead",
    "rear",
)


def _is_punch_like_label(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    if not s:
        return False
    if s in _PUNCH_EXACT:
        return True
    sl = s.lower()
    return any(p in sl for p in _PUNCH_SUBSTRINGS)


def _header_cell_to_str(val: Any, idx: int) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"Unnamed: {idx}"
    t = str(val).strip()
    return t if t and t.lower() != "nan" else f"Unnamed: {idx}"


def _make_unique_column_labels(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = n.strip() or "col"
        k = _slug_key(base)
        if not k:
            base = "col"
            k = "col"
        if k in seen:
            seen[k] += 1
            out.append(f"{base}_{seen[k]}")
        else:
            seen[k] = 0
            out.append(base)
    return out


def _row_looks_like_header(cells: list[Any]) -> bool:
    """True if row looks like a header row (start/end/class tokens)."""
    parts: list[str] = []
    for v in cells:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = str(v).strip()
        if not t or t.lower() == "nan":
            continue
        parts.append(_slug_key(t))
    joined = " ".join(parts)
    if not joined:
        return False
    hints = (
        "start_frame",
        "start_frame",
        "starting_frame",
        "begin_frame",
        "end_frame",
        "ending_frame",
        "finish_frame",
        "class",
        "label",
        "type",
        "category",
        "action",
        "punch",
    )
    return any(h in joined for h in hints)


def _apply_header_row(raw: pd.DataFrame, header_idx: int) -> pd.DataFrame:
    """Use ``raw.iloc[header_idx]`` as column labels; body is rows below."""
    ncols = int(raw.shape[1])
    hdr_cells = [raw.iat[header_idx, j] for j in range(ncols)]
    labels = [_header_cell_to_str(hdr_cells[j], j) for j in range(ncols)]
    labels = _make_unique_column_labels(labels)
    body = raw.iloc[header_idx + 1 :].copy()
    body.columns = labels[: body.shape[1]]
    return body


def _score_frame_label_triple(
    df: pd.DataFrame,
    triple: tuple[int, int, int],
    *,
    require_punch_label: bool,
) -> int:
    """Count rows that look like (start_frame, end_frame, punch_label)."""
    i, j, k = triple
    if max(i, j, k) >= df.shape[1]:
        return 0
    ok = 0
    for _, row in df.iterrows():
        try:
            a = row.iloc[i]
            b = row.iloc[j]
            c = row.iloc[k]
            _normalize_event_frame(a)
            _normalize_event_frame(b)
            if require_punch_label and not _is_punch_like_label(c):
                continue
            ok += 1
        except (ValueError, IndexError, TypeError):
            continue
    return ok


def _iter_heuristic_triples(ncols: int) -> list[tuple[int, int, int]]:
    """Column index triples for spacer layouts (5-6 cols) and contiguous 3-col."""
    trips: list[tuple[int, int, int]] = []
    if ncols >= 3:
        trips.append((0, 1, 2))
    if ncols >= 5:
        trips.extend(
            [
                (0, 2, 4),
                (0, 1, 4),
                (0, 2, 3),
                (0, 3, 4),
                (0, 1, 2),
            ]
        )
    if ncols >= 6:
        trips.extend([(0, 2, 5), (0, 3, 5), (0, 1, 5)])
    # Dedupe preserving order
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for t in trips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _best_heuristic_triple(
    df: pd.DataFrame,
    *,
    require_punch_label: bool,
    min_valid_rows: int,
) -> tuple[int, int, int] | None:
    ncols = int(df.shape[1])
    if ncols < 3:
        return None
    best_t: tuple[int, int, int] | None = None
    best_score = 0
    for trip in _iter_heuristic_triples(ncols):
        sc = _score_frame_label_triple(df, trip, require_punch_label=require_punch_label)
        if sc > best_score:
            best_score = sc
            best_t = trip
    if best_t is None or best_score < min_valid_rows:
        return None
    return best_t


def _dataframe_from_triple(
    df: pd.DataFrame,
    triple: tuple[int, int, int],
) -> tuple[pd.DataFrame, str, str, str, tuple[str, str, str]]:
    i, j, k = triple
    trial = df.iloc[:, [i, j, k]].copy()
    trial.columns = ["c_start", "c_end", "c_cls"]
    disp = (f"col{i}", f"col{j}", f"col{k}")
    return trial, "c_start", "c_end", "c_cls", disp


def _build_col_map(df: pd.DataFrame) -> dict[str, str]:
    col_map: dict[str, str] = {}
    for c in df.columns:
        if _is_ignored_column(c):
            continue
        key = _slug_key(str(c))
        if not key:
            continue
        if key not in col_map:
            col_map[key] = str(c)
    return col_map


def _pick_column(col_map: dict[str, str], *aliases: str) -> str | None:
    for a in aliases:
        k = _slug_key(a)
        if k in col_map:
            return col_map[k]
    return None


def _is_positional_style_column_name(c: Any) -> bool:
    """True if column label looks like a bare index / ``Unnamed`` placeholder."""
    if isinstance(c, (int, np.integer)):
        return True
    s = str(c).strip()
    if not s:
        return True
    if s.lower().startswith("unnamed"):
        return True
    return bool(s.isdigit())


def _should_skip_naive_first_three_non_named(df: pd.DataFrame) -> bool:
    """5+ column grids with only positional names need spacer heuristics (V9/V10-style)."""
    if df.shape[1] < 5:
        return False
    return all(_is_positional_style_column_name(c) for c in df.columns)


def _logical_columns_from_merged_header(name: str) -> bool:
    """True if header looks like concatenated start/end/class (e.g. ``start_frameend_frametype``)."""
    s = _slug_key(name)
    if len(s) < 12:
        return False
    for sep in ("ending_frame", "end_frame", "endframe"):
        if sep not in s:
            continue
        i = s.index(sep)
        left = s[:i]
        right = s[i + len(sep) :].strip("_")
        if not left.startswith("start") and "start" not in left:
            continue
        if right in ("type", "class", "label") or any(
            x in right for x in ("type", "class", "label")
        ):
            return True
    return False


def _split_triple_series(series: pd.Series) -> pd.DataFrame | None:
    """Split ``\"10 20 jab\"`` / comma / tab separated triples per row."""
    rows: list[list[Any]] = []
    saw_short = False
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = str(v).strip()
        if not t:
            continue
        parts = re.split(r"[\t,;]+|\s+", t)
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            rows.append([parts[0], parts[1], parts[2]])
        elif len(parts) == 1:
            saw_short = True
    if saw_short and not rows:
        return None
    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["c_start", "c_end", "c_cls"])
    return out


def _normalize_event_frame(row: Any) -> int:
    if row is None or (isinstance(row, float) and np.isnan(row)):
        raise ValueError("invalid frame")
    if isinstance(row, str):
        s = row.strip()
        if not s:
            raise ValueError("invalid frame")
        return int(float(s))
    if isinstance(row, (np.floating, float)) and not np.isnan(row):
        return int(row)
    return int(row)


def _build_events(
    df: pd.DataFrame,
    c_start: str,
    c_end: str,
    c_cls: str,
    *,
    skip_invalid_rows: bool,
) -> list[BoxingVIEvent]:
    events: list[BoxingVIEvent] = []
    for _, row in df.iterrows():
        try:
            sf = _normalize_event_frame(row[c_start])
            ef = _normalize_event_frame(row[c_end])
            raw_cls = row[c_cls]
            if raw_cls is None or (isinstance(raw_cls, float) and np.isnan(raw_cls)):
                cls_s = ""
            else:
                cls_s = str(raw_cls).strip()
            events.append(BoxingVIEvent(start_frame=sf, end_frame=ef, class_name=cls_s))
        except (ValueError, KeyError, TypeError):
            if skip_invalid_rows:
                continue
            raise
    return events


def _resolve_columns_and_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str, tuple[str, str, str]] | None:
    """
    Return working dataframe, column keys for access, and display names for reporting.

    Supports::
        Start_Frame, Ending_Frame, Class
        Start Frame, End Frame, Type / Label
        StartFrame, EndFrame, Label
    """
    cmap = _build_col_map(df)
    c_start = _pick_column(
        cmap,
        "start_frame",
        "startframe",
        "start",
        "start_frame_number",
        "begin_frame",
        "beginframe",
    )
    c_end = _pick_column(
        cmap,
        "end_frame",
        "ending_frame",
        "endframe",
        "end",
        "ending",
        "finish_frame",
    )
    c_cls = _pick_column(
        cmap,
        "class",
        "label",
        "type",
        "punch_class",
        "action",
        "category",
    )
    if c_start and c_end and c_cls:
        disp = (str(c_start), str(c_end), str(c_cls))
        return df, c_start, c_end, c_cls, disp

    # Pattern C: 5-6 columns with spacers; prefer scored triples before naive first-three.
    triple_punch = _best_heuristic_triple(
        df,
        require_punch_label=True,
        min_valid_rows=1,
    )
    if triple_punch is not None:
        return _dataframe_from_triple(df, triple_punch)

    usable = [c for c in df.columns if not _is_ignored_column(c)]
    if len(usable) >= 3 and not _should_skip_naive_first_three_non_named(df):
        a, b, cl = usable[0], usable[1], usable[2]
        trial = df[[a, b, cl]].copy()
        trial.columns = ["c_start", "c_end", "c_cls"]
        ok = 0
        for _, row in trial.iterrows():
            try:
                _normalize_event_frame(row["c_start"])
                _normalize_event_frame(row["c_end"])
                ok += 1
            except ValueError:
                continue
        if ok > 0:
            disp = (str(a), str(b), str(cl))
            return trial, "c_start", "c_end", "c_cls", disp

    if len(usable) == 1 and len(df.columns) >= 3:
        merged = str(usable[0])
        if _logical_columns_from_merged_header(merged):
            cols = list(df.columns)
            a, b, cl = cols[0], cols[1], cols[2]
            trial = df[[a, b, cl]].copy()
            trial.columns = ["c_start", "c_end", "c_cls"]
            ok = 0
            for _, row in trial.iterrows():
                try:
                    _normalize_event_frame(row["c_start"])
                    _normalize_event_frame(row["c_end"])
                    ok += 1
                except ValueError:
                    continue
            if ok > 0:
                disp = ("start_frame", "end_frame", "class")
                return trial, "c_start", "c_end", "c_cls", disp

    if len(usable) == 1:
        merged = str(usable[0])
        if _logical_columns_from_merged_header(merged):
            split = _split_triple_series(df[merged])
            if split is not None:
                disp = ("start_frame", "end_frame", "class")
                return split, "c_start", "c_end", "c_cls", disp

    # No header / non-punch labels: any three columns with two numeric frame columns.
    # First 3 columns as start/end/label (no header, or header failed); allow any label.
    if df.shape[1] >= 3:
        trip2 = _best_heuristic_triple(
            df,
            require_punch_label=False,
            min_valid_rows=1,
        )
        if trip2 is not None:
            return _dataframe_from_triple(df, trip2)

    return None


def _rows_to_events_impl(
    df: pd.DataFrame,
    *,
    skip_invalid_rows: bool,
) -> tuple[list[BoxingVIEvent], tuple[str, str, str], tuple[int, int]]:
    resolved = _resolve_columns_and_frame(df)
    if resolved is None:
        normalized_present = sorted(
            k for k in (_slug_key(str(c)) for c in df.columns if not _is_ignored_column(c)) if k
        )
        raise ValueError(
            "Annotation sheet must contain recognizable columns for start frame, end frame, "
            "and class/label/type. "
            f"Normalized column keys (excluding Unnamed* placeholders): {normalized_present}."
        )
    work, ks, ke, kc, disp = resolved
    ev = _build_events(work, ks, ke, kc, skip_invalid_rows=skip_invalid_rows)
    raw_shape = (int(work.shape[0]), int(work.shape[1]))
    return ev, disp, raw_shape


def _rows_to_events(df: pd.DataFrame) -> list[BoxingVIEvent]:
    """Strict row parsing (raises on missing columns); used by unit tests."""
    ev, _, _ = _rows_to_events_impl(df, skip_invalid_rows=False)
    return ev


def _try_parse_dataframe(
    df: pd.DataFrame,
    *,
    skip_invalid_rows: bool = True,
) -> tuple[list[BoxingVIEvent], tuple[str, str, str], tuple[int, int]] | None:
    if df.empty or df.dropna(how="all").empty:
        return None
    try:
        return _rows_to_events_impl(df, skip_invalid_rows=skip_invalid_rows)
    except ValueError:
        return None


def _load_events_from_sheet(
    path: Path,
    sheet_name: str,
    *,
    skip_invalid_rows: bool = True,
) -> tuple[list[BoxingVIEvent], tuple[str, str, str], int, tuple[int, int]]:
    """Parse one sheet; tries ``header=0``, ``header=None`` grid, header rows 0..9, then ``header`` 1..15."""

    best: tuple[list[BoxingVIEvent], tuple[str, str, str], int, tuple[int, int]] | None = None

    def consider(
        parsed: tuple[list[BoxingVIEvent], tuple[str, str, str], tuple[int, int]] | None,
        header_row: int,
    ) -> None:
        nonlocal best
        if parsed is None:
            return
        ev, cols, raw_shape = parsed
        if not ev:
            return
        if best is None or len(ev) > len(best[0]):
            best = (ev, cols, header_row, raw_shape)

    # ``pd.read_excel`` with ``header=0`` (explicit first-row header).
    try:
        df0 = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=0,
            engine="openpyxl",
        )
        consider(_try_parse_dataframe(df0, skip_invalid_rows=skip_invalid_rows), 0)
    except (ValueError, IndexError, OSError):
        pass

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")

    # Full grid as data (Pattern B: first row is already start/end/label).
    consider(_try_parse_dataframe(raw, skip_invalid_rows=skip_invalid_rows), -1)

    # First 10 rows may each be the real header row (Pattern: header not on line 1).
    max_hdr_scan = min(10, len(raw))
    for i in range(max_hdr_scan):
        try:
            body = _apply_header_row(raw, i).dropna(how="all")
        except (ValueError, TypeError, IndexError):
            continue
        if body.empty:
            continue
        # Prefer rows that look header-like when multiple parses tie (optional fast-path skip).
        parsed_i = _try_parse_dataframe(body, skip_invalid_rows=skip_invalid_rows)
        consider(parsed_i, i)

    # Broader ``header=`` sweep for odd layouts (merged cells, extra title rows).
    for header_row in range(1, 16):
        if header_row == 0:
            continue
        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row,
                engine="openpyxl",
            )
        except (ValueError, IndexError, OSError):
            continue
        consider(_try_parse_dataframe(df, skip_invalid_rows=skip_invalid_rows), header_row)

    # Legacy: rebuild from raw with explicit header list (no ``_make_unique`` — rare paths).
    max_scan = min(25, len(raw))
    for i in range(max_scan):
        body = raw.iloc[i + 1 :]
        if body.empty:
            continue
        body = body.dropna(how="all")
        if body.empty:
            continue
        hdr = raw.iloc[i].tolist()
        try:
            body2 = body.copy()
            body2.columns = hdr
        except (ValueError, TypeError, IndexError):
            continue
        consider(_try_parse_dataframe(body2, skip_invalid_rows=skip_invalid_rows), i)

    if best is None:
        raise ValueError(f"Could not parse sheet {sheet_name!r} in {path}")
    ev, cols, hdr_row, raw_shape = best
    return ev, cols, hdr_row, raw_shape


def load_events_from_xlsx(path: Path, *, skip_invalid_rows: bool = True) -> list[BoxingVIEvent]:
    """Load and merge events from **all** worksheets in the workbook."""
    _require_openpyxl()
    path = path.expanduser().resolve()
    xf = pd.ExcelFile(path, engine="openpyxl")
    all_e: list[BoxingVIEvent] = []
    last_err: Exception | None = None
    for raw_sheet in xf.sheet_names:
        sheet_key: str = str(raw_sheet)
        try:
            ev, _cols, _hr, _shp = _load_events_from_sheet(
                path, sheet_key, skip_invalid_rows=skip_invalid_rows
            )
            all_e.extend(ev)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            last_err = exc
            continue
    if not all_e:
        if xf.sheet_names:
            try:
                df0 = pd.read_excel(
                    path,
                    sheet_name=xf.sheet_names[0],
                    header=0,
                    engine="openpyxl",
                )
                if len(df0) == 0:
                    return []
            except (OSError, ValueError, KeyError):
                pass
        if last_err is not None:
            raise last_err
        raise ValueError(f"No parseable events in {path}")
    return sorted(all_e, key=lambda e: (e.start_frame, e.end_frame))


def _format_inspect_failure_debug(path: Path, xf: pd.ExcelFile) -> str:
    """Human-readable preview for FAILED inspect (sheet names, columns, first rows)."""
    lines: list[str] = []
    lines.append(f"sheet_names: {list(xf.sheet_names)}")
    for sn in xf.sheet_names:
        try:
            raw = pd.read_excel(
                path,
                sheet_name=sn,
                header=None,
                engine="openpyxl",
                nrows=5,
            )
        except Exception as exc:
            lines.append(f"\n[{sn!r}] read error: {exc}")
            continue
        lines.append(f"\n[{sn!r}] raw_columns (header=None): {list(raw.columns)}")
        lines.append(f"[{sn!r}] raw_shape (preview): {raw.shape}")
        lines.append(raw.head(5).to_string(index=False))
    return "\n".join(lines)


def _fallback_raw_shape(path: Path) -> tuple[int, int] | None:
    """Best-effort grid shape for FAILED reports (first sheet, no header)."""
    try:
        xf = pd.ExcelFile(path, engine="openpyxl")
        if not xf.sheet_names:
            return None
        raw = pd.read_excel(path, sheet_name=xf.sheet_names[0], header=None, engine="openpyxl")
        return (int(raw.shape[0]), int(raw.shape[1]))
    except Exception:
        return None


def inspect_annotation_xlsx(path: Path) -> AnnotationInspectReport:
    """Parse one workbook for reporting; returns ``ok=False`` on failure (no raise)."""
    path = path.expanduser().resolve()
    rep = AnnotationInspectReport(path=path, ok=False, status="FAILED")
    try:
        _require_openpyxl()
    except ImportError as exc:
        rep.error = str(exc)
        return rep

    if not path.is_file():
        rep.error = "file not found"
        return rep

    try:
        xf = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        rep.error = f"open excel: {exc}"
        return rep

    all_events: list[BoxingVIEvent] = []
    per_sheet: list[tuple[str, int, tuple[int, int], tuple[str, str, str], int]] = []

    for sheet_idx, raw_sheet in enumerate(xf.sheet_names):
        sheet_key: str = str(raw_sheet)
        try:
            ev, cols, _hr, shp = _load_events_from_sheet(path, sheet_key, skip_invalid_rows=True)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        if ev:
            per_sheet.append((sheet_key, len(ev), shp, cols, sheet_idx))
            all_events.extend(ev)

    if not all_events:
        rep.error = "no sheet produced valid events"
        rep.raw_shape = _fallback_raw_shape(path)
        rep.inspect_debug = _format_inspect_failure_debug(path, xf)
        return rep

    dom = max(per_sheet, key=lambda t: (t[1], -t[4]))
    rep.raw_shape = dom[2]
    rep.columns_detected = dom[3]

    all_sorted = sorted(all_events, key=lambda e: (e.start_frame, e.end_frame))
    rep.ok = True
    rep.status = "OK"
    rep.sheets_used = [t[0] for t in per_sheet]
    rep.valid_event_count = len(all_sorted)
    rep.first_event = all_sorted[0]
    return rep


def inspect_dataset(root: Path) -> list[AnnotationInspectReport]:
    """Inspect ``annotations/V*.xlsx`` under ``root``."""
    root = Path(root).expanduser().resolve()
    ann = root / "annotations"
    reports: list[AnnotationInspectReport] = []
    if not ann.is_dir():
        return [
            AnnotationInspectReport(
                path=ann,
                ok=False,
                status="FAILED",
                error="annotations directory missing",
            )
        ]

    def _sort_key(p: Path) -> tuple[int, str]:
        m = re.match(r"^V(\d+)$", p.stem, re.I)
        if m:
            return (int(m.group(1)), p.stem)
        return (9999, p.stem)

    files = sorted(ann.glob("V*.xlsx"), key=_sort_key)
    for p in files:
        try:
            reports.append(inspect_annotation_xlsx(p))
        except Exception as exc:
            reports.append(
                AnnotationInspectReport(
                    path=p,
                    ok=False,
                    status="FAILED",
                    error=str(exc),
                )
            )
    return reports


class BoxingVIDataset:
    """Access BoxingVI-style files under a single root directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def annotations_dir(self) -> Path:
        return self._root / "annotations"

    @property
    def skeleton_dir(self) -> Path:
        return self._root / "skeleton"

    @property
    def rgb_dir(self) -> Path:
        return self._root / "rgb"

    @property
    def metadata_dir(self) -> Path:
        return self._root / "metadata"

    def list_video_ids(self) -> list[str]:
        ids: set[str] = set()
        if self.annotations_dir.is_dir():
            for p in self.annotations_dir.glob("*.xlsx"):
                ids.add(p.stem)
        if self.skeleton_dir.is_dir():
            for p in self.skeleton_dir.glob("*.npy"):
                ids.add(p.stem)
        return sorted(ids)

    def get_video_path(self, video_id: str) -> Path:
        name = str(video_id).strip()
        if not name:
            raise ValueError("video_id must be non-empty")
        return self.rgb_dir / f"{name}.mp4"

    def _annotation_path(self, video_id: str) -> Path:
        return self.annotations_dir / f"{video_id}.xlsx"

    def _skeleton_path(self, video_id: str) -> Path:
        return self.skeleton_dir / f"{video_id}.npy"

    def load_annotations(self, video_id: str) -> list[BoxingVIEvent]:
        path = self._annotation_path(video_id)
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return load_events_from_xlsx(path, skip_invalid_rows=True)

    def load_skeleton(self, video_id: str) -> np.ndarray:
        path = self._skeleton_path(video_id)
        if not path.is_file():
            raise FileNotFoundError(f"Skeleton file not found: {path}")
        return cast("np.ndarray", np.load(path, allow_pickle=False))


def _print_inspect_table(reports: list[AnnotationInspectReport]) -> None:
    """Aligned columns for ``--inspect`` (stdout-friendly)."""
    headers = [
        "video_id",
        "status",
        "sheet_used",
        "raw_shape",
        "detected_columns",
        "valid_event_count",
        "first_event",
        "error",
    ]
    rows: list[list[str]] = []
    for rep in reports:
        vid = rep.path.stem if rep.path.is_file() else str(rep.path)
        sheet_used = ", ".join(rep.sheets_used)
        rs = ""
        if rep.raw_shape is not None:
            rs = f"({rep.raw_shape[0]}, {rep.raw_shape[1]})"
        dc = ""
        if rep.columns_detected:
            dc = "[" + ", ".join(rep.columns_detected) + "]"
        vc = str(rep.valid_event_count)
        fe = ""
        if rep.first_event:
            e = rep.first_event
            fe = f"({e.start_frame}, {e.end_frame}, {e.class_name!r})"
        err = (rep.error or "").replace("\n", " ").strip()
        if len(err) > 100:
            err = err[:97] + "..."
        rows.append([vid, rep.status, sheet_used, rs, dc, vc, fe, err])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = ["-" * widths[i] for i in range(len(headers))]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt(headers))
    print(fmt(sep))
    for row in rows:
        print(fmt(row))

    for rep in reports:
        if rep.ok or not rep.inspect_debug:
            continue
        print()
        print(f"--- inspect debug: {rep.path.name} ---")
        print(rep.inspect_debug)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="BoxingVI local dataset helpers")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root (expects annotations/V*.xlsx)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print per-V*.xlsx parse status, sheets, columns, and event counts",
    )
    args = p.parse_args(argv)
    if args.inspect:
        reports = inspect_dataset(args.dataset_root)
        _print_inspect_table(reports)
        any_fail = any(not r.ok for r in reports)
        sys.exit(1 if any_fail else 0)
    p.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "AnnotationInspectReport",
    "BoxingVIDataset",
    "BoxingVIEvent",
    "inspect_annotation_xlsx",
    "inspect_dataset",
    "load_events_from_xlsx",
]
