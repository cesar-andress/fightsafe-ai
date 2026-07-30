"""
Load and save :class:`AnnotationDocument` to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fightsafe_ai.annotation.schema import ANNOTATION_FORMAT_VERSION, AnnotationDocument


_DEFAULT_INDENT: int = 2


def new_empty_template(*, video: str) -> AnnotationDocument:
    """
    An empty, valid document for manual population (e.g. output of
    ``fightsafe annotate-template``).
    """
    return AnnotationDocument(
        format_version=ANNOTATION_FORMAT_VERSION,
        video=video.strip() or "unknown",
        time_unit="seconds",
        events=[],
    )


def _read_json_object(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def load_annotation_file(path: Path) -> AnnotationDocument:
    """
    Read ``.json`` from ``path`` and return a validated :class:`AnnotationDocument`.
    See :func:`fightsafe_ai.annotation.validator.validate_annotation_document` for
    file-level cross-checks after load.
    """
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Annotation file not found: {p}")
    data = _read_json_object(p)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    return AnnotationDocument.model_validate(data)


def save_annotation_file(
    path: Path,
    doc: AnnotationDocument,
    *,
    indent: int = _DEFAULT_INDENT,
) -> None:
    """Write ``doc`` to ``path`` (UTF-8) with a stable, human-editable layout."""
    p = path.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    s = doc.model_dump_json(indent=indent) if indent else doc.model_dump_json()
    if not s.endswith("\n"):
        s += "\n"
    p.write_text(s, encoding="utf-8")


__all__ = [
    "load_annotation_file",
    "new_empty_template",
    "save_annotation_file",
]
