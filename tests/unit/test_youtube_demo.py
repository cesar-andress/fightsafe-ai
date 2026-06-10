"""Orchestration for :func:`fightsafe_ai.pipeline.youtube_demo.run_demo_youtube` (no network / YouTube)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_video_id_hint_for_url_matches_youtube() -> None:
    from fightsafe_ai.pipeline.youtube_demo import video_id_hint_for_url

    assert (
        video_id_hint_for_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share")
        == "dQw4w9WgXcQ"
    )
    assert video_id_hint_for_url("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id_hint_for_url("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id_hint_for_url("https://example.com/page") == "video"


def test_run_demo_youtube_download_cut_pipeline_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fightsafe_ai.pipeline import youtube_demo as mod

    root = tmp_path / "run"
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    downloaded = source / "dQw4w9WgXcQ.mp4"
    downloaded.write_bytes(b"downloaded")

    def fake_download(url: str, output_dir: Path, filename: str | None = None) -> Path:
        assert "youtu.be" in url or "youtube" in url
        assert output_dir == source
        assert filename and filename.endswith(".mp4")
        return downloaded

    def fake_cut(input_video: Path, start_time: str, end_time: str, output_path: Path) -> Path:
        assert input_video == downloaded
        assert start_time == "0"
        assert end_time == "1"
        assert output_path == root / "input_clip.mp4"
        output_path.write_bytes(b"clip")
        return output_path

    def fake_e2e(
        video: Path,
        output_root: Path,
        **kwargs: object,
    ) -> tuple[object, bool, object]:
        assert video == root / "input_clip.mp4"
        assert output_root == root
        p = SimpleNamespace(root=output_root)
        return p, True, SimpleNamespace(passed=True)

    monkeypatch.setattr(mod, "download_video", fake_download)
    monkeypatch.setattr(mod, "cut_clip", fake_cut)
    monkeypatch.setattr(mod, "run_e2e_demo", fake_e2e)

    out_paths, ok, rep = mod.run_demo_youtube(
        "https://youtu.be/dQw4w9WgXcQ", "0", "1", root, download_filename=None
    )
    assert out_paths.root == root
    assert ok is True
    assert rep.passed is True
    assert (root / "source" / "dQw4w9WgXcQ.mp4").is_file()
    assert (root / "input_clip.mp4").read_bytes() == b"clip"


def test_run_demo_youtube_rejects_path_outside_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fightsafe_ai.pipeline import youtube_demo as mod

    root = tmp_path / "out"

    def evil_download(url: str, output_dir: Path, filename: str | None = None) -> Path:
        return Path("/this/should/never/happen/clip.mp4")

    monkeypatch.setattr(mod, "download_video", evil_download)

    # Do not use pytest.raises(VideoDownloadError) with a module-level import: other
    # tests (e.g. importlib-based isolated loads) can register a second
    # fightsafe_ai.exceptions, so class identity can differ.
    with pytest.raises(Exception) as c:
        mod.run_demo_youtube("https://example.com", "0", "1", root, download_filename="x.mp4")
    e = c.value
    assert (
        e.__class__.__name__ == "VideoDownloadError"
        and e.__class__.__module__ == "fightsafe_ai.exceptions"
    )
    assert "Download path is outside" in str(e)
    assert "run directory" in str(e)


def test_invalid_download_name_is_rejected(tmp_path: Path) -> None:
    from fightsafe_ai.pipeline import youtube_demo as mod

    with pytest.raises(ValueError, match="Invalid download filename"):
        mod.run_demo_youtube(
            "https://youtu.be/x",
            "0",
            "1",
            tmp_path,
            download_filename="bad/name.mp4",
        )
