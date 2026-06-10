"""Tests for clip timecode parsing (isolated :mod:`fightsafe_ai.video.cutter` import)."""

import pytest
from tests.support.isolated import load_cutter


_c = load_cutter()
parse_timecode = _c.parse_timecode


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("83.5", 83.5),
        ("120", 120.0),
        ("01:23", 83.0),
        ("00:01:23", 83.0),
        ("00:01:23.5", 83.5),
        ("1:02:03", 3723.0),
    ],
)
def test_parse_timecode(raw: str, expected: float) -> None:
    assert parse_timecode(raw) == pytest.approx(expected)


def test_parse_timecode_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_timecode("")
