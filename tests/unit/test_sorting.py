"""Tests for natural sort helpers (isolated import)."""

from pathlib import Path

from tests.support.isolated import load_utils_sorting


_s = load_utils_sorting()
natural_sort_paths = _s.natural_sort_paths


def test_natural_sort_paths_orders_numeric_stems() -> None:
    paths = [
        Path("frame_10.jpg"),
        Path("frame_2.jpg"),
        Path("frame_1.jpg"),
    ]
    sorted_paths = natural_sort_paths(paths)
    assert [p.name for p in sorted_paths] == ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"]
