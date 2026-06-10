"""``natural_sort_strings`` (lexicographic + embedded digit runs)."""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.utils.sorting import natural_sort_paths, natural_sort_strings


def test_natural_sort_strings_frame_ids() -> None:
    ids = ["frame_10", "frame_2", "frame_1a", "frame_1b"]
    assert natural_sort_strings(ids) == [
        "frame_1a",
        "frame_1b",
        "frame_2",
        "frame_10",
    ]


def test_natural_sort_paths_tie_breaker() -> None:
    paths = [Path("z/f10.jpg"), Path("z/f2.jpg"), Path("z/f1.jpg")]
    out = natural_sort_paths(paths)
    assert [p.name for p in out] == ["f1.jpg", "f2.jpg", "f10.jpg"]


def test_natural_sort_empty() -> None:
    assert natural_sort_strings([]) == []
    assert natural_sort_paths([]) == []
