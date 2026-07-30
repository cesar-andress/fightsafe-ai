"""Load keypoint CSV files produced by pose export (per-frame legacy or consolidated)."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.pose.blazepose import blazepose_index
from fightsafe_ai.utils.sorting import natural_sort_paths, natural_sort_strings


logger = logging.getLogger(__name__)

# (x, y) normalized, [0,1]
LandmarkMap = dict[str, tuple[float, float]]
# index -> (x, y, visibility)
IndexedLandmarks = dict[int, tuple[float, float, float]]


def load_keypoint_csv(path: Path) -> LandmarkMap | None:
    """
    Return ``landmark_name -> (x, y)`` or ``None`` if the file is empty or unreadable.

    Expects legacy columns ``landmark``, ``x``, ``y``.
    """
    points: LandmarkMap = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    if not rows:
        return None
    for row in rows:
        name = row.get("landmark") or row.get("keypoint_name")
        if not name:
            continue
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, ValueError):
            continue
        points[name] = (x, y)
    return points if points else None


def load_keypoint_csv_indexed(path: Path) -> IndexedLandmarks:
    """Map BlazePose index 0..32 -> (x, y, visibility) for drawing / geometry (legacy CSV)."""
    out: IndexedLandmarks = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("landmark") or row.get("keypoint_name")
                if not name:
                    continue
                idx = blazepose_index(name)
                if idx is None:
                    continue
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                    vis = float(row.get("visibility", "1"))
                except (KeyError, ValueError, AttributeError):
                    continue
                out[idx] = (x, y, vis)
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
    return out


def load_landmark_maps_ordered(
    source: Path,
    glob_pattern: str = "*.csv",
) -> list[tuple[str, LandmarkMap | None]]:
    """
    Load ordered ``(label, landmark_map)`` for feature extraction.

    * **Directory:** one legacy CSV per frame (``glob_pattern``), sorted naturally by filename.
    * **Single CSV:** consolidated ``frame_id``, ``keypoint_name``, … rows (MediaPipe export).

    ``label`` is the CSV filename or ``frame_id`` string used in ``source_csv`` column.
    """
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise VideoIOError(f"Keypoints file must be a .csv file, got: {source}")
        return _ordered_maps_from_consolidated_csv(source)
    if source.is_dir():
        paths = natural_sort_paths([p for p in source.glob(glob_pattern) if p.is_file()])
        return [(p.name, load_keypoint_csv(p)) for p in paths]

    raise VideoIOError(f"Keypoints source must be a directory or a .csv file, got: {source}")


def _ordered_maps_from_consolidated_csv(path: Path) -> list[tuple[str, LandmarkMap | None]]:
    """Group consolidated rows by ``frame_id``; sort frame IDs naturally."""
    by_frame: defaultdict[str, LandmarkMap] = defaultdict(dict)
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"frame_id", "keypoint_name", "x", "y"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise VideoIOError(
                    f"Consolidated CSV missing columns {required}: {path}",
                )
            for row in reader:
                fid = row.get("frame_id")
                name = row.get("keypoint_name")
                if not fid or not name:
                    continue
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                except (KeyError, ValueError):
                    continue
                by_frame[fid][name] = (x, y)
    except OSError as exc:
        raise VideoIOError(f"Could not read consolidated keypoints: {path}") from exc

    ordered_ids = natural_sort_strings(list(by_frame.keys()))
    return [(fid, by_frame[fid] if by_frame[fid] else None) for fid in ordered_ids]


def load_indexed_sequence(source: Path, glob_pattern: str = "*.csv") -> list[IndexedLandmarks]:
    """
    Ordered list of per-frame indexed landmarks for visualization.

    Supports the same ``source`` modes as :func:`load_landmark_maps_ordered`.
    """
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise VideoIOError(f"Keypoints file must be a .csv file, got: {source}")
        return _indexed_sequence_from_consolidated(source)
    if source.is_dir():
        paths = natural_sort_paths([p for p in source.glob(glob_pattern) if p.is_file()])
        return [load_keypoint_csv_indexed(p) for p in paths]

    raise VideoIOError(f"Keypoints source must be a directory or a .csv file, got: {source}")


def _indexed_sequence_from_consolidated(path: Path) -> list[IndexedLandmarks]:
    """Group consolidated CSV rows by ``frame_id`` and build indexed landmark dicts."""
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            for row in reader:
                fid = row.get("frame_id")
                if not fid:
                    continue
                groups[fid].append(row)
    except OSError as exc:
        raise VideoIOError(f"Could not read consolidated keypoints: {path}") from exc

    ordered_ids = natural_sort_strings(list(groups.keys()))
    result: list[IndexedLandmarks] = []
    for fid in ordered_ids:
        result.append(_rows_to_indexed(groups[fid]))
    return result


def _rows_to_indexed(rows: list[dict[str, str]]) -> IndexedLandmarks:
    out: IndexedLandmarks = {}
    for row in rows:
        name = row.get("keypoint_name")
        if not name:
            continue
        idx = blazepose_index(name)
        if idx is None:
            continue
        try:
            x = float(row["x"])
            y = float(row["y"])
            vis = float(row.get("visibility", "1"))
        except (KeyError, ValueError, AttributeError):
            continue
        out[idx] = (x, y, vis)
    return out
