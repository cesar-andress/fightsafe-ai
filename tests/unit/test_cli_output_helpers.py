"""Coverage for :mod:`fightsafe_ai.cli` path/summary helpers (no subprocess video)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fightsafe_ai.cli import _display_path, _print_demo_completed_outputs


pytestmark = pytest.mark.unit


def test_display_path_relative_under_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "runs" / "demo").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _display_path(tmp_path / "runs" / "demo") == "runs/demo"


def test_display_path_falls_back_to_str_when_not_under_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = _display_path(Path("/") / "nonexistent" / "absolute" / "branch")
    assert out.startswith("/")


@patch("fightsafe_ai.cli._ok")
def test_print_demo_completed_outputs_calls_ok(
    mock_ok: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs" / "demo"
    root.mkdir(parents=True)
    _print_demo_completed_outputs(root, heading="Test done.")
    lines = [c[0][0] for c in mock_ok.call_args_list]
    assert "Test done." in lines
    assert any("Overlay video:" in str(x) for x in lines)
    assert any("Run directory:" in str(x) for x in lines)
