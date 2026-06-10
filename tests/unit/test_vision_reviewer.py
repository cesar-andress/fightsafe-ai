"""Unit tests for :mod:`fightsafe_ai.llm.vision_reviewer` (mocked Ollama; no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fightsafe_ai.llm.ollama_client import OllamaClientConfig
from fightsafe_ai.llm.vision_reviewer import _RESULT_KEYS, _extract_json_object, review_event_frames


pytestmark = pytest.mark.unit


def _ok_json() -> str:
    return json.dumps(
        {
            "visible_fighters": 2,
            "apparent_posture_issue": "unclear stance at frame edge",
            "possible_surrender_gesture": "unclear",
            "possible_loss_of_control": "unclear",
            "uncertainty_notes": "Partial view.",
            "human_review_recommendation": "Optional re-watch of the segment.",
        }
    )


def test_extract_json_object_strips_fence() -> None:
    raw = "```json\n" + _ok_json() + "\n```"
    o = _extract_json_object(raw)
    assert o is not None
    assert o.get("visible_fighters") == 2


def test_review_event_frames_disabled_returns_fallback() -> None:
    cfg = OllamaClientConfig(enabled=False, enable_vlm_review=True)
    out = review_event_frames([Path("nope.png")], {"x": 1}, ollama_config=cfg)
    assert set(out.keys()) == set(_RESULT_KEYS)
    assert (
        "disabled" in str(out["uncertainty_notes"]).lower()
        or "llm.yaml" in out["uncertainty_notes"]
    )


def test_review_event_frames_vlm_off_returns_fallback() -> None:
    cfg = OllamaClientConfig(enabled=True, enable_vlm_review=False)
    out = review_event_frames([], {}, ollama_config=cfg)
    assert (
        "enable" in out["uncertainty_notes"].lower()
        or "disabled" in out["uncertainty_notes"].lower()
    )


def test_review_event_frames_no_files_returns_fallback(tmp_path: Path) -> None:
    cfg = OllamaClientConfig(enabled=True, enable_vlm_review=True, max_frames_per_event=2)
    out = review_event_frames([], {"t": 0.0}, ollama_config=cfg)
    assert "No existing" in out["uncertainty_notes"] or "no" in out["uncertainty_notes"].lower()


def test_review_event_frames_mock_client(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    p1.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    p2.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg2")

    class _Stub:
        def chat(
            self,
            messages: list[dict[str, Any]],
            *,
            model: str | None = None,
        ) -> str:
            assert model == "qwen2.5vl:7b"
            assert any("images" in m for m in messages if m.get("role") == "user")
            return _ok_json()

    cfg = OllamaClientConfig(
        enabled=True,
        enable_vlm_review=True,
        max_frames_per_event=4,
        vision_model="qwen2.5vl:7b",
    )
    out = review_event_frames(
        [p1, p2], {"event": "test"}, model="qwen2.5vl:7b", ollama_config=cfg, ollama_client=_Stub()
    )
    assert out["visible_fighters"] == 2
    assert out["possible_surrender_gesture"] == "unclear"


def test_review_event_frames_malformed_json_fallback(tmp_path: Path) -> None:
    p = tmp_path / "c.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")

    class _Bad:
        def chat(
            self,
            messages: list[dict[str, Any]],
            *,
            model: str | None = None,
        ) -> str:
            return "not json at all {["

    cfg = OllamaClientConfig(enabled=True, enable_vlm_review=True, max_frames_per_event=2)
    out = review_event_frames([p], {}, ollama_config=cfg, ollama_client=_Bad())
    assert (
        "parseable" in out["uncertainty_notes"].lower()
        or "json" in out["uncertainty_notes"].lower()
    )


def test_max_frames_zero_short_circuits() -> None:
    cfg = OllamaClientConfig(enabled=True, enable_vlm_review=True, max_frames_per_event=0)
    p = Path("x")
    out = review_event_frames([p], {}, ollama_config=cfg)
    assert "0" in out["uncertainty_notes"] or "max_frames" in out["uncertainty_notes"].lower()
