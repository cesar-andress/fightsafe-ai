"""
Mock-based tests for I/O wrappers (subprocess, OpenCV, ffmpeg) without real network or codecs.

Covers: downloader, clip cutter, frame extraction, overlay render core, MVP pipeline entry.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.exceptions import VideoDownloadError, VideoIOError
from fightsafe_ai.pipeline.output_paths import paths_for_run_root


def test_download_video_uses_ytdlp_stdout_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fightsafe_ai.video import downloader

    out_file = tmp_path / "down.mp4"
    out_file.write_bytes(b"fake")
    m_sh = SimpleNamespace(which=lambda *_a, **_k: "/usr/bin/yt-dlp")
    monkeypatch.setattr(downloader, "shutil", m_sh)

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        assert "yt-dlp" in cmd[0] or cmd[0].endswith("yt-dlp")
        return SimpleNamespace(
            returncode=0,
            stdout=f"{out_file.resolve()!s}\n",
            stderr="",
        )

    monkeypatch.setattr(downloader, "subprocess", SimpleNamespace(run=fake_run))

    result = downloader.download_video("https://example.invalid/watch", tmp_path, filename="x.mp4")

    assert result == out_file.resolve()


def test_download_video_errors_when_ytdlp_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fightsafe_ai.video import downloader

    m_sh = SimpleNamespace(which=lambda *_a, **_k: None)
    monkeypatch.setattr(downloader, "shutil", m_sh)
    with pytest.raises(VideoDownloadError) as err:
        downloader.download_video("https://x", tmp_path)
    assert "yt-dlp" in str(err.value).lower()


def test_cut_clip_runs_ffmpeg_and_returns_output(tmp_path: Path) -> None:
    from fightsafe_ai.video.cutter import cut_clip

    src = tmp_path / "in.mp4"
    src.write_bytes(b"header")
    dst = tmp_path / "out.mp4"

    def fake_run(out_stream: Any, **kwargs: Any) -> None:
        dst.write_bytes(b"ok")

    with patch("fightsafe_ai.video.cutter.ffmpeg.run", side_effect=fake_run):
        p = cut_clip(src, "0", "1", dst)

    assert p == dst.resolve() and dst.is_file()


def test_extract_frames_mocks_opencv(tmp_path: Path) -> None:
    from fightsafe_ai.video import frame_extractor

    vid = tmp_path / "a.mp4"
    vid.write_bytes(b"1")
    out = tmp_path / "frames"
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = [30.0]  # FPS
    cap.read.side_effect = [(True, frame)] * 6 + [(False, None)]

    with (
        patch("fightsafe_ai.video.frame_extractor.cv2.VideoCapture", return_value=cap),
        patch("fightsafe_ai.video.frame_extractor.cv2.imwrite", return_value=True),
    ):
        paths = frame_extractor.extract_frames(vid, out, fps=10)

    assert len(paths) >= 1
    assert all(p.suffix == ".jpg" for p in paths)
    cap.release.assert_called_once()


def test_render_core_mocks_opencv_no_skeleton(tmp_path: Path) -> None:
    from fightsafe_ai.visualization import overlay

    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    outp = tmp_path / "o.mp4"
    risk = pd.DataFrame({"risk_score": [0.1, 0.2], "risk_level": ["LOW", "MEDIUM"]})

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = [64.0, 48.0, 25.0]  # w, h, fps order per OpenCV; code uses three gets
    frm = np.zeros((48, 64, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frm), (True, frm), (False, None)]

    writer = MagicMock()
    writer.isOpened.return_value = True

    with (
        patch("fightsafe_ai.visualization.overlay.cv2.VideoCapture", return_value=cap),
        patch("fightsafe_ai.visualization.overlay.cv2.VideoWriter", return_value=writer),
    ):
        n, fps = overlay._render_core(
            vid,
            [],
            risk,
            outp,
            overlay.OverlayVizConfig(),
        )

    assert n == 2
    assert fps == 25.0
    cap.release.assert_called()
    writer.release.assert_called()


def test_run_mvp_pipeline_delegates_to_run_pipeline(tmp_path: Path) -> None:
    from fightsafe_ai.pipeline import mvp

    video = tmp_path / "in.mp4"
    video.write_bytes(b"0")
    root = tmp_path / "run1"
    paths = paths_for_run_root(root)

    def _fake_run(
        *args: object,
        **kwargs: object,
    ) -> Any:
        return SimpleNamespace(paths=paths)

    with patch.object(mvp, "run_pipeline", side_effect=_fake_run) as p_run:
        out = mvp.run_mvp_pipeline(video, root, fps=5, rolling_window=2)

    p_run.assert_called_once()
    assert out is paths
    assert out.report_md == root / "report.md"


def test_download_subprocess_nonzero_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fightsafe_ai.video import downloader

    m_sh = SimpleNamespace(which=lambda *_a, **_k: "/bin/yt-dlp")
    monkeypatch.setattr(downloader, "shutil", m_sh)
    monkeypatch.setattr(
        downloader,
        "subprocess",
        SimpleNamespace(
            run=lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="fail")
        ),
    )
    with pytest.raises(VideoDownloadError) as err:
        downloader.download_video("https://u", tmp_path, filename="z.mp4")
    assert "failed" in str(err.value).lower()


def test_read_risk_csv_sorts_by_frame_index(tmp_path: Path) -> None:
    from fightsafe_ai.visualization import overlay

    p = tmp_path / "r.csv"
    p.write_text(
        "frame_index,risk_score,risk_level,near_ground,risk_flag\n"
        "1,0.5,HIGH,True,1\n"
        "0,0.1,LOW,False,0\n",
        encoding="utf-8",
    )
    df = overlay.read_risk_csv(p)
    assert list(df["frame_index"]) == [0, 1]
    assert df["near_ground"].dtype == object or str(df["near_ground"].iloc[0]) in ("False", "0")


def test_risk_values_for_frame_critical_banner() -> None:
    from fightsafe_ai.visualization import overlay

    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "risk_score": [0.0, 0.99],
            "risk_level": ["LOW", "CRITICAL"],
        }
    )
    sc, _lv, ban, _f = overlay.risk_values_for_frame(df, 1)
    assert ban == "CRITICAL" and sc > 0.5


def test_load_pose_indexed_sequence_missing_returns_empty() -> None:
    from fightsafe_ai.visualization import overlay

    assert overlay.load_pose_indexed_sequence(Path("/this/path/does/not/exist.csv")) == []


def test_viz_dict_to_config_sets_bgr() -> None:
    from fightsafe_ai.visualization import overlay

    cfg = overlay._viz_dict_to_config(
        {
            "skeleton": {
                "line_bgr": (10, 20, 30),
                "joint_bgr": (1, 2, 3),
                "line_thickness": 4,
                "joint_radius": 2,
            },
            "risk": {"border_thickness": 2, "overlay_strength": 0.1},
        }
    )
    assert cfg.skeleton_line_bgr == (10, 20, 30)
    assert cfg.skeleton_line_thickness == 4
    assert cfg.skeleton_joint_radius == 2


def test_frame_extractor_file_not_found_and_invalid_fps(
    tmp_path: Path,
) -> None:
    from fightsafe_ai.video import frame_extractor

    with pytest.raises(VideoIOError) as err1:
        frame_extractor.extract_frames(tmp_path / "missing.mp4", tmp_path, fps=2)
    assert "video" in str(err1.value).lower() and "not" in str(err1.value).lower()
    p = tmp_path / "a.mp4"
    p.write_bytes(b"x")
    with pytest.raises(ValueError) as err2:
        frame_extractor.extract_frames(p, tmp_path / "o", fps=0)
    assert "fps" in str(err2.value).lower()
