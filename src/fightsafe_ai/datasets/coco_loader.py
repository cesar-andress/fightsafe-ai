"""
**Local-file** helpers for COCO-style **person keypoint** JSON (no download).

`COCO <https://cocodataset.org/>`_ annotations are **not** included in this repository.
Obtain ``instances_person_keypoints_*.json`` and images under the COCO license, then
point :func:`load_coco_annotations_dict` at a file on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_coco_annotations_dict(path: Path) -> dict[str, Any]:
    """
    Parse a COCO-format JSON file from ``path`` (must exist locally).

    **Does not** download anything. Returns ``{}`` if the file is missing or not an object.
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def list_image_ids(coco: dict[str, Any]) -> list[int]:
    """Return numeric ``id`` from each element of ``images`` (empty if absent)."""
    out: list[int] = []
    for im in coco.get("images", []):
        if not isinstance(im, dict):
            continue
        if "id" in im and isinstance(im["id"], (int, float)):
            out.append(int(im["id"]))
    return out


def n_person_annotations(coco: dict[str, Any]) -> int:
    """Count list elements under ``annotations`` (0 if missing)."""
    anns = coco.get("annotations", [])
    if not isinstance(anns, list):
        return 0
    return len(anns)


def looks_like_coco_json(data: dict[str, Any]) -> bool:
    """Heuristic: whether a parsed JSON object looks like a COCO annotation file."""
    if not data:
        return False
    need = ("images", "annotations")
    return all(k in data for k in need) and isinstance(data.get("images"), list)
