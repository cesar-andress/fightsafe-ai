""":func:`_parse_print_filepath` behavior for yt-dlp stdout edge cases."""

from __future__ import annotations

import pytest

from fightsafe_ai.video.downloader import _parse_print_filepath


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("  \n  ", None),
        ("/var/media/expected_path/a.mp4", "/var/media/expected_path/a.mp4"),
        ("line1\nline2\n/path/final.mp4", "/path/final.mp4"),
        ("NA", None),
        ("na", None),
        ("N/A", None),
    ],
)
def test_parse_print_filepath(raw: str, expected: str | None) -> None:
    assert _parse_print_filepath(raw) == expected
