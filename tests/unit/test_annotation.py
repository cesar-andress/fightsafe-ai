"""Tests for manual evaluation annotation schema, loader, and validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fightsafe_ai.annotation.loader import (
    load_annotation_file,
    new_empty_template,
    save_annotation_file,
)
from fightsafe_ai.annotation.schema import (
    ANNOTATION_FORMAT_VERSION,
    EventType,
    parse_annotation_dict,
)
from fightsafe_ai.annotation.validator import (
    format_overlap_warnings,
    is_valid_annotation_file,
    validate_annotation_file,
    validate_event_type,
)


def test_new_empty_template_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    d = new_empty_template(video="clips/x.mp4")
    assert d.events == []
    assert d.video == "clips/x.mp4"
    assert d.time_unit == "seconds"
    save_annotation_file(p, d)
    d2 = load_annotation_file(p)
    assert d2.model_dump() == d.model_dump()


def test_event_validation() -> None:
    from fightsafe_ai.annotation.schema import EventAnnotation

    e = EventAnnotation(start_time=0.0, end_time=1.0, event_type=EventType.KO, confidence=0.5)
    assert e.event_type == EventType.KO
    with pytest.raises(ValidationError):
        EventAnnotation(start_time=1.0, end_time=1.0, event_type=EventType.FALL)
    with pytest.raises(ValidationError):
        EventAnnotation(start_time=2.0, end_time=1.0, event_type=EventType.FALL)
    with pytest.raises(ValidationError):
        EventAnnotation(start_time=-0.1, end_time=1.0, event_type=EventType.FALL)


def test_parse_and_validate_file(tmp_path: Path) -> None:
    good = {
        "format_version": ANNOTATION_FORMAT_VERSION,
        "video": "v.mp4",
        "time_unit": "seconds",
        "events": [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "event_type": "SURRENDER",
            }
        ],
    }
    doc = parse_annotation_dict(good)
    assert doc.events[0].event_type == EventType.SURRENDER

    p = tmp_path / "g.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    assert is_valid_annotation_file(p)
    assert validate_annotation_file(p) == []


def test_invalid_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        '{"format_version": "0.0", "video": "a", "time_unit": "seconds", "events": []}',
        encoding="utf-8",
    )
    errs = validate_annotation_file(p)
    assert errs
    assert not is_valid_annotation_file(p)


def test_overlap_warning() -> None:
    from fightsafe_ai.annotation.schema import AnnotationDocument, EventAnnotation

    d = AnnotationDocument(
        video="a",
        events=[
            EventAnnotation(start_time=0.0, end_time=2.0, event_type=EventType.FALL),
            EventAnnotation(start_time=1.0, end_time=3.0, event_type=EventType.FALL),
        ],
    )
    w = format_overlap_warnings(d)
    assert w


def test_validate_event_type() -> None:
    assert validate_event_type("KO")
    assert not validate_event_type("NOT_A_TYPE")
