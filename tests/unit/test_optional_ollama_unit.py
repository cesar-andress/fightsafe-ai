"""Unit tests for :mod:`fightsafe_ai.pipeline.optional_ollama` (mocked I/O, no live Ollama)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fightsafe_ai.pipeline.optional_ollama import try_optional_ollama_narrative


pytestmark = pytest.mark.unit


def test_try_optional_ollama_zero_events() -> None:
    assert try_optional_ollama_narrative(0, None) == ""


def test_try_optional_ollama_config_load_fails(tmp_path: Path) -> None:
    p = tmp_path / "nope.yaml"
    p.write_text("not: valid: yaml: [[", encoding="utf-8")
    assert try_optional_ollama_narrative(1, p) == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_disabled_no_network(load_cfg: MagicMock, tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("ollama:\n  enabled: false\n  base_url: 'http://127.0.0.1:1'\n", encoding="utf-8")
    load_cfg.return_value = SimpleNamespace(enabled=False, base_url="http://127.0.0.1:1", model="m")
    with patch("urllib.request.urlopen") as uo:
        out = try_optional_ollama_narrative(2, p, force_enabled=False)
    uo.assert_not_called()
    assert out == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_non_http_base_returns_empty(
    load_cfg: MagicMock, tmp_path: Path
) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("ollama:\n  enabled: true\n  base_url: 'ftp://x'\n  model: m\n", encoding="utf-8")
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="ftp://x", model="m", temperature=0.2
    )
    with patch("urllib.request.urlopen") as uo:
        out = try_optional_ollama_narrative(1, p, force_enabled=True)
    uo.assert_not_called()
    assert out == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_tags_unreachable(load_cfg: MagicMock, tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        assert try_optional_ollama_narrative(1, p, force_enabled=True) == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_tags_non_200_returns_empty(
    load_cfg: MagicMock, tmp_path: Path
) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )

    class _Bad:
        status = 500

        def __enter__(self) -> _Bad:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    with patch("urllib.request.urlopen", side_effect=[_Bad()]):
        assert try_optional_ollama_narrative(1, p, force_enabled=True) == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_happy_path(load_cfg: MagicMock, tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )

    class _Ctx:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._b = body

        def read(self) -> bytes:
            return self._b

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    tags = _Ctx(200, b"{}")
    gen = _Ctx(200, json.dumps({"response": "One line summary."}).encode("utf-8"))
    with patch("urllib.request.urlopen", side_effect=[tags, gen]):
        out = try_optional_ollama_narrative(2, p, force_enabled=True)
    assert "summary" in out.lower() or "line" in out.lower()


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_generate_non_200(load_cfg: MagicMock, tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )

    class _Ok:
        def __init__(self, st: int) -> None:
            self.status = st

        def __enter__(self) -> _Ok:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    with patch("urllib.request.urlopen", side_effect=[_Ok(200), _Ok(500)]):
        assert try_optional_ollama_narrative(1, p, force_enabled=True) == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_generate_json_keyerror_branch(
    load_cfg: MagicMock, tmp_path: Path
) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )

    class _Ctx:
        def __init__(self, st: int, b: bytes) -> None:
            self.status = st
            self._b = b

        def read(self) -> bytes:
            return self._b

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    with patch("urllib.request.urlopen", side_effect=[_Ctx(200, b"{}"), _Ctx(200, b"[]")]):
        assert try_optional_ollama_narrative(1, p, force_enabled=True) == ""


@patch("fightsafe_ai.llm.ollama_client.load_ollama_config")
def test_try_optional_ollama_empty_response_text(load_cfg: MagicMock, tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "ollama:\n  enabled: true\n  base_url: 'http://127.0.0.1:9'\n  model: m\n", encoding="utf-8"
    )
    load_cfg.return_value = SimpleNamespace(
        enabled=True, base_url="http://127.0.0.1:9", model="m", temperature=0.2
    )

    class _Ctx:
        def __init__(self, st: int, b: bytes) -> None:
            self.status = st
            self._b = b

        def read(self) -> bytes:
            return self._b

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    with patch(
        "urllib.request.urlopen",
        side_effect=[_Ctx(200, b"{}"), _Ctx(200, b'{"response":""}')],
    ):
        assert try_optional_ollama_narrative(1, p, force_enabled=True) == ""
