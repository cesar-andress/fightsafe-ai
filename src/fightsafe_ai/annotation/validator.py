"""
Cross-field validation and human-readable error lists for evaluation annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fightsafe_ai.annotation.loader import load_annotation_file
from fightsafe_ai.annotation.schema import (
    AnnotationDocument,
    EventAnnotation,
    EventType,
)


def validate_event_type(value: str) -> bool:
    return value in _EVENT_VALUES


def list_event_type_values() -> list[str]:
    return sorted(_EVENT_VALUES)


def validate_event_annotation(e: EventAnnotation) -> list[str]:
    """
    Pydantic already enforces the model; this layer adds explicit messages for
    any extra checks (e.g. optional overlap warnings could go here in future).
    """
    return []


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = "/".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        out.append(f"{loc or 'root'}: {msg}")
    return out


def validate_annotation_file(path: Path) -> list[str]:
    """
    Load and validate a JSON file. Returns an empty list if the file is a valid
    :class:`AnnotationDocument`; otherwise returns one or more human-readable error lines
    (parse errors, JSON errors, or Pydantic :class:`ValidationError` details).
    """
    p = path.expanduser().resolve()
    try:
        if not p.is_file():
            return [f"Not a file: {p}"]
    except OSError as exc:
        return [str(exc)]
    try:
        load_annotation_file(p)
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]
    except OSError as exc:
        return [f"Read error: {exc}"]
    except ValidationError as exc:
        return _format_pydantic_errors(exc)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def is_valid_annotation_file(path: Path) -> bool:
    return len(validate_annotation_file(path)) == 0


def is_valid_document(doc: AnnotationDocument) -> bool:
    return len(validate_annotation_document(doc)) == 0


def validate_annotation_document(_doc: AnnotationDocument) -> list[str]:
    """
    Extra checks on an already-parsed document (Pydantic already enforces
    time ranges and enums). Returns no errors today; reserved for future rules
    (e.g. no duplicate ids).
    """
    return []


_EVENT_VALUES: frozenset[str] = frozenset(x.value for x in EventType)


# --- Optional pair-wise overlap (informational; not a hard error) -----------------


def format_overlap_warnings(
    doc: AnnotationDocument, *, jaccard_threshold: float = 0.0
) -> list[str]:
    """
    Return warning strings for same-type intervals that share time (Jaccard overlap
    of intervals on the line). `jaccard_threshold` 0.0 => any non-zero shared duration.
    For strict evaluation you may de-duplicate; by default this does **not** block validation.
    """
    warn: list[str] = []
    n = len(doc.events)
    for a in range(n):
        for b in range(a + 1, n):
            e1, e2 = doc.events[a], doc.events[b]
            if e1.event_type != e2.event_type:
                continue
            t0_1, t1_1 = e1.start_time, e1.end_time
            t0_2, t1_2 = e2.start_time, e2.end_time
            o0 = max(t0_1, t0_2)
            o1 = min(t1_1, t1_2)
            if o1 > o0:
                inter = o1 - o0
                union = max(t1_1, t1_2) - min(t0_1, t0_2)
                jac = inter / union if union > 0 else 0.0
                if jac > jaccard_threshold:
                    warn.append(
                        f"Overlap (same {e1.event_type.value}): events[{a}] vs events[{b}] "
                        f" share ~{inter:.3f}s (Jaccard ~{jac:.2f} on their union window)."
                    )
    return warn


__all__ = [
    "format_overlap_warnings",
    "is_valid_annotation_file",
    "is_valid_document",
    "list_event_type_values",
    "validate_annotation_document",
    "validate_annotation_file",
    "validate_event_annotation",
    "validate_event_type",
]
