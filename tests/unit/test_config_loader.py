"""Tests for :mod:`fightsafe_ai.config.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from fightsafe_ai.config.loader import load_default_paths, load_yaml_file, merge_dicts
from fightsafe_ai.exceptions import ConfigurationError


def test_load_yaml_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.yaml"
    p.write_text("a: 1\nb:\n  c: x\n", encoding="utf-8")
    d = load_yaml_file(p)
    assert d == {"a": 1, "b": {"c": "x"}}


def test_load_yaml_file_missing() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_yaml_file(Path("/no/such/config.yaml"))


def test_load_yaml_file_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "b.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Expected mapping"):
        load_yaml_file(p)


def test_load_yaml_file_empty_file_returns_empty_mapping(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml_file(p) == {}


def test_merge_dicts() -> None:
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}


def test_load_default_paths(tmp_path: Path) -> None:
    d1 = tmp_path / "d.yaml"
    d1.write_text("k: 1\n", encoding="utf-8")
    d2 = tmp_path / "r.yaml"
    d2.write_text("rules: []", encoding="utf-8")
    a, r = load_default_paths(d1, d2)
    assert a == {"k": 1} and "rules" in r


def test_load_default_paths_risk_none(tmp_path: Path) -> None:
    d1 = tmp_path / "a.yaml"
    d1.write_text("x: y\n", encoding="utf-8")
    app, risk = load_default_paths(d1, None)
    assert app == {"x": "y"} and risk == {}
