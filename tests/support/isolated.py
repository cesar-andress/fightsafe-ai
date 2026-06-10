"""
Load FightSafe submodules without importing ``fightsafe_ai`` package ``__init__``
(avoid optional heavy deps like MediaPipe during collection).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
SRC = _ROOT / "src"


def _ensure_pkg(name: str, path: Path) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m


def _exec(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exceptions() -> Any:
    """
    Load :mod:`fightsafe_ai.exceptions` once. If it is already imported (e.g. by
    the real package in tests), reuse it so exception classes keep a single
    identity across the suite. Re-execing would break ``isinstance`` and
    ``except`` matching against ``from fightsafe_ai.exceptions import ...`` in
    other modules.
    """
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    name = "fightsafe_ai.exceptions"
    if name in sys.modules:
        return sys.modules[name]
    return _exec(name, SRC / "fightsafe_ai" / "exceptions.py")


def load_risk_models() -> Any:
    load_exceptions()
    _ensure_pkg("fightsafe_ai.risk", SRC / "fightsafe_ai" / "risk")
    return _exec("fightsafe_ai.risk.models", SRC / "fightsafe_ai" / "risk" / "models.py")


def load_risk_engine() -> Any:
    """``detect_risk_events``, ``RiskRuleParams`` via ``models`` submodule."""
    load_risk_models()
    return _exec("fightsafe_ai.risk.engine", SRC / "fightsafe_ai" / "risk" / "engine.py")


def load_risk_rules() -> Any:
    load_exceptions()
    _ensure_pkg("fightsafe_ai.risk", SRC / "fightsafe_ai" / "risk")
    return _exec("fightsafe_ai.risk.rules", SRC / "fightsafe_ai" / "risk" / "rules.py")


def load_risk_scorer() -> Any:
    load_risk_rules()
    _exec("fightsafe_ai.risk.limb_tier", SRC / "fightsafe_ai" / "risk" / "limb_tier.py")
    return _exec("fightsafe_ai.risk.scorer", SRC / "fightsafe_ai" / "risk" / "scorer.py")


def _stub_ffmpeg_module() -> None:
    if "ffmpeg" in sys.modules:
        return
    ff = types.ModuleType("ffmpeg")

    class _FfmpegError(Exception):
        pass

    ff.Error = _FfmpegError  # type: ignore[attr-defined]
    sys.modules["ffmpeg"] = ff


def load_cutter() -> Any:
    """``parse_timecode`` / ``cut_clip`` without requiring a working ``ffmpeg`` CLI in tests."""
    _stub_ffmpeg_module()
    load_exceptions()
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.video", SRC / "fightsafe_ai" / "video")
    return _exec("fightsafe_ai.video.cutter", SRC / "fightsafe_ai" / "video" / "cutter.py")


def load_risk_events() -> Any:
    """``frame_risk_to_events`` (dataclasses need parent package names in ``sys.modules``)."""
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.risk", SRC / "fightsafe_ai" / "risk")
    return _exec("fightsafe_ai.risk.events", SRC / "fightsafe_ai" / "risk" / "events.py")


def load_utils_sorting() -> Any:
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.utils", SRC / "fightsafe_ai" / "utils")
    return _exec("fightsafe_ai.utils.sorting", SRC / "fightsafe_ai" / "utils" / "sorting.py")


def load_biomechanics() -> Any:
    """
    Load :mod:`fightsafe_ai.features.biomechanics` without ``fightsafe_ai`` package ``__init__``.

    Uses the real :mod:`fightsafe_ai.keypoints.io` (MediaPipe is lazy) so
    ``sys.modules`` is not left with a stub that lacks ``load_indexed_sequence``.
    """
    load_exceptions()
    load_utils_sorting()
    _ensure_pkg("fightsafe_ai.keypoints", SRC / "fightsafe_ai" / "keypoints")
    _exec("fightsafe_ai.keypoints.io", SRC / "fightsafe_ai" / "keypoints" / "io.py")
    _ensure_pkg("fightsafe_ai.features", SRC / "fightsafe_ai" / "features")
    return _exec(
        "fightsafe_ai.features.biomechanics",
        SRC / "fightsafe_ai" / "features" / "biomechanics.py",
    )


def load_temporal() -> Any:
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.features", SRC / "fightsafe_ai" / "features")
    return _exec(
        "fightsafe_ai.features.temporal", SRC / "fightsafe_ai" / "features" / "temporal.py"
    )


def load_llm() -> Any:
    """
    ``fightsafe_ai.llm`` (Ollama client, prompts, risk explainer) without loading
    the full ``fightsafe_ai`` package ``__init__``.
    """
    load_exceptions()
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.llm", SRC / "fightsafe_ai" / "llm")
    _exec("fightsafe_ai.llm.base", SRC / "fightsafe_ai" / "llm" / "base.py")
    _exec("fightsafe_ai.llm.ollama_client", SRC / "fightsafe_ai" / "llm" / "ollama_client.py")
    _exec("fightsafe_ai.llm.config", SRC / "fightsafe_ai" / "llm" / "config.py")
    _exec("fightsafe_ai.llm.prompts", SRC / "fightsafe_ai" / "llm" / "prompts.py")
    re = _exec(
        "fightsafe_ai.llm.risk_explainer", SRC / "fightsafe_ai" / "llm" / "risk_explainer.py"
    )
    exm = _exec("fightsafe_ai.llm.explainer", SRC / "fightsafe_ai" / "llm" / "explainer.py")
    # So ``patch("fightsafe_ai.llm...")`` and importlib can walk package attributes.
    r = sys.modules["fightsafe_ai"]
    llm = sys.modules["fightsafe_ai.llm"]
    r.llm = llm  # type: ignore[attr-defined]
    llm.base = sys.modules["fightsafe_ai.llm.base"]  # type: ignore[attr-defined]
    llm.ollama_client = sys.modules["fightsafe_ai.llm.ollama_client"]  # type: ignore[attr-defined]
    llm.config = sys.modules["fightsafe_ai.llm.config"]  # type: ignore[attr-defined]
    llm.prompts = sys.modules["fightsafe_ai.llm.prompts"]  # type: ignore[attr-defined]
    llm.risk_explainer = re  # type: ignore[attr-defined]
    llm.explainer = exm  # type: ignore[attr-defined]
    return re


def load_report_generator() -> Any:
    """
    :mod:`fightsafe_ai.llm.report_generator` (depends on ``llm`` prompts, which need ``llm.config``).
    """
    load_exceptions()
    _ensure_pkg("fightsafe_ai", SRC / "fightsafe_ai")
    _ensure_pkg("fightsafe_ai.llm", SRC / "fightsafe_ai" / "llm")
    _exec("fightsafe_ai.llm.base", SRC / "fightsafe_ai" / "llm" / "base.py")
    _exec("fightsafe_ai.llm.ollama_client", SRC / "fightsafe_ai" / "llm" / "ollama_client.py")
    _exec("fightsafe_ai.llm.config", SRC / "fightsafe_ai" / "llm" / "config.py")
    _exec("fightsafe_ai.llm.prompts", SRC / "fightsafe_ai" / "llm" / "prompts.py")
    _exec("fightsafe_ai.llm.risk_explainer", SRC / "fightsafe_ai" / "llm" / "risk_explainer.py")
    _exec("fightsafe_ai.llm.report_enricher", SRC / "fightsafe_ai" / "llm" / "report_enricher.py")
    rmod = _exec(
        "fightsafe_ai.llm.report_generator", SRC / "fightsafe_ai" / "llm" / "report_generator.py"
    )
    r = sys.modules["fightsafe_ai"]
    llm = sys.modules["fightsafe_ai.llm"]
    r.llm = llm  # type: ignore[attr-defined]
    llm.base = sys.modules["fightsafe_ai.llm.base"]  # type: ignore[attr-defined]
    llm.ollama_client = sys.modules["fightsafe_ai.llm.ollama_client"]  # type: ignore[attr-defined]
    llm.config = sys.modules["fightsafe_ai.llm.config"]  # type: ignore[attr-defined]
    llm.prompts = sys.modules["fightsafe_ai.llm.prompts"]  # type: ignore[attr-defined]
    llm.report_generator = rmod  # type: ignore[attr-defined]
    return rmod
