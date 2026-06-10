"""
Ollama-assisted **drafts** for manual safety-event labels.

Output is *never* ground truth: every suggestion is tagged with
``requires_human_confirmation: true`` and must be confirmed by a human in a separate
:file:`AnnotationDocument` (see :mod:`fightsafe_ai.annotation.schema`).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Final, Protocol, cast

import pandas as pd

from fightsafe_ai.annotation.schema import ANNOTATION_FORMAT_VERSION, EventType
from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.llm.ollama_client import OllamaClient, OllamaClientConfig, load_ollama_config


logger = logging.getLogger(__name__)

SUGGESTIONS_FORMAT_VERSION: Final[str] = "1.0"

# Fixed disclaimer stored in the JSON file (machine-readable, not legal advice text).
SUGGESTIONS_NOT_GROUND_TRUTH: Final[str] = (
    "These rows are **drafts** for human-in-the-loop annotation. They are not ground truth, "
    "not medical or officiating fact, and must be reviewed before use in "
    f"{ANNOTATION_FORMAT_VERSION!r} annotation files."
)

_EVENT_TYPES: Final[frozenset[str]] = frozenset(e.value for e in EventType) | {"UNCLEAR"}


def _parse_json_object_from_llm_text(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t2 = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
        t2 = t2.rsplit("```", 1)[0].strip()
        t = t2
    try:
        o = json.loads(t)
        if isinstance(o, dict):
            return o
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            o2 = json.loads(t[i : j + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(o2, dict):
            return o2
    return None


def load_event_candidates(path: Path) -> list[dict[str, Any]]:
    """Load pipeline ``events.json`` (list of event dicts)."""
    p = path.expanduser().resolve()
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def load_risk_scores(path: Path) -> pd.DataFrame | None:
    p = path.expanduser().resolve()
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, ValueError) as e:
        logger.warning("llm_assist: could not read risk_scores.csv: %s", e)
        return None


def _window_rows(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.iloc[0:0]
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    sub = df.loc[(ts >= start) & (ts <= end)].copy()
    return sub


def risk_signal_summary(event: dict[str, Any], risk_df: pd.DataFrame | None) -> dict[str, Any]:
    """
    Condensed, numeric summaries over ``[start_time, end_time]`` in ``risk_scores.csv``.
    """
    if risk_df is None or len(risk_df) == 0:
        return {
            "frames_in_window": 0,
            "note": "no risk_scores.csv or empty table",
        }
    st = float(event.get("start_time", 0.0))
    et = float(event.get("end_time", 0.0))
    w = _window_rows(risk_df, st, et)
    n = len(w)
    out: dict[str, Any] = {
        "frames_in_window": n,
        "window_start_s": st,
        "window_end_s": et,
    }
    if n == 0:
        return out
    for col, label in (("risk_score", "max_risk_score"), ("risk_level", "level_counts")):
        if col in w.columns:
            if col == "risk_level":
                vc = w[col].astype(str).str.upper().value_counts()
                out[label] = {str(k): int(v) for k, v in vc.items()}
            else:
                s = pd.to_numeric(w[col], errors="coerce")
                out["max_risk_score"] = float(s.max()) if s.notna().any() else None
                out["mean_risk_score"] = float(s.mean()) if s.notna().any() else None
    if "max_risk_score" in event:
        out["event_json_max_risk"] = event.get("max_risk_score")
    if "event_level" in event:
        out["event_json_level"] = event.get("event_level")
    return out


def _jpg_paths_sorted(frames_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not frames_dir.is_dir():
        return []
    out: list[Path] = []
    for p in frames_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def _paths_for_event_frames(
    frames_dir: Path,
    event: dict[str, Any],
    risk_df: pd.DataFrame | None,
    *,
    run_root: Path,
    max_n: int,
) -> list[str]:
    if max_n <= 0 or not frames_dir.is_dir():
        return []
    st = float(event.get("start_time", 0.0))
    et = float(event.get("end_time", 0.0))
    fids: list[str] = []
    if risk_df is not None and "timestamp" in risk_df.columns and "frame_id" in risk_df.columns:
        w = _window_rows(risk_df, st, et)
        for fid in w["frame_id"].astype(str).tolist():
            if fid not in fids:
                fids.append(fid)
    rel: list[str] = []
    all_j = _jpg_paths_sorted(frames_dir)
    for fid in fids[: max_n * 3]:
        hit = [p for p in all_j if fid in p.name or fid in str(p)]
        for p in hit:
            s = str(p.resolve().relative_to(run_root.resolve()))
            if s not in rel:
                rel.append(s)
        if len(rel) >= max_n:
            return rel[:max_n]
    for p in all_j:
        s = str(p.resolve().relative_to(run_root.resolve()))
        if s not in rel:
            rel.append(s)
        if len(rel) >= max_n:
            return rel
    return rel


class _GenerateClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def build_assist_bundle(
    run_dir: Path,
    *,
    max_frames_per_event: int = 3,
    frames_subdir: str = "frames",
) -> dict[str, Any]:
    """
    Gather ``events.json`` event candidates, per-event risk summaries, and selected
    on-disk frame paths (relative to ``run_dir``) for an LLM prompt.
    """
    r = run_dir.expanduser().resolve()
    ev_path = r / "events.json"
    events = load_event_candidates(ev_path)
    rdf = load_risk_scores(r / "risk_scores.csv")
    frames_dir = r / frames_subdir
    out_events: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        eid = ev.get("event_id", i)
        try:
            eid_i = int(eid) if eid is not None else i
        except (TypeError, ValueError):
            eid_i = i
        ecopy = {**ev, "event_id": eid_i}
        fpaths = _paths_for_event_frames(
            frames_dir, ecopy, rdf, run_root=r, max_n=max_frames_per_event
        )
        out_events.append(
            {
                "event_id": eid_i,
                "event": ecopy,
                "risk_signal_summary": risk_signal_summary(ecopy, rdf),
                "selected_frame_paths_relative": fpaths,
            }
        )
    return {
        "run_dir": str(r),
        "events_path": str(ev_path),
        "candidates": out_events,
    }


def _build_generate_prompt(bundle: dict[str, Any]) -> str:
    ctx = {
        "run_dir": bundle.get("run_dir"),
        "candidates": bundle.get("candidates", []),
        "optional_vlm_reviews": bundle.get("optional_vlm_reviews", []),
    }
    return (
        "You are helping a sports-safety *annotation* workflow. The user will copy your "
        "suggestions into a *manual* editor — your labels are NOT final.\n"
        "Rules: (1) no medical or clinical diagnosis; (2) never claim your label matches an "
        "official ruling; (3) respect that deterministic risk scores in the data are a separate system — "
        "suggest a human-readable event type and rationale for **annotation speed-up only**.\n"
        "Propose one row per *candidate* event interval, or suggest splitting if clearly wrong, "
        "but prefer one suggestion per candidate. Use the exact JSON object shape in the user message; "
        "suggested_event_type must be one of: FALL, KO, SURRENDER, INSTABILITY, UNCLEAR.\n"
        "Every suggestion MUST include requires_human_confirmation: true.\n\n"
        f"Data (JSON): {json.dumps(ctx, ensure_ascii=False)[:20000]}"
    )


_SYSTEM_PROMPT: Final[str] = (
    'Return **only** a single JSON object with a key "suggestions" whose value is a JSON array. '
    "Each element: start_time, end_time, suggested_event_type, confidence (0-1), rationale (string), "
    "requires_human_boolean (bool, must be true), event_id (integer, matching input candidate if possible). "
    "Use the key name requires_human_confirmation with boolean true, not a typo. No markdown fences."
)
# Ollama sometimes mangles key names; we accept a few and normalize in _normalize_suggestion.


def _coerce_suggestion(o: Any, fallback_event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(o, dict):
        return None
    st = o.get("start_time", fallback_event.get("start_time") if fallback_event else None)
    et = o.get("end_time", fallback_event.get("end_time") if fallback_event else None)
    if st is None or et is None:
        return None
    try:
        stf, etf = float(st), float(et)
    except (TypeError, ValueError):
        return None
    if etf <= stf or stf < 0 or etf < 0:
        return None
    raw = str(o.get("suggested_event_type", "UNCLEAR")).strip().upper()
    if raw not in _EVENT_TYPES:
        raw = "UNCLEAR"
    conf: float
    try:
        conf = float(o.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    rationale = str(o.get("rationale", o.get("reasoning", "No rationale provided.")) or "").strip()[
        :2000
    ]
    return {
        "start_time": stf,
        "end_time": etf,
        "suggested_event_type": raw,
        "confidence": conf,
        "rationale": rationale,
        "requires_human_confirmation": True,
    }


def _normalize_suggestion_list(
    data: dict[str, Any] | None, events_by_id: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    if not data or "suggestions" not in data:
        return []
    raw = data.get("suggestions")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        eid: int | None = None
        ev = o.get("event_id")
        if ev is not None and not isinstance(ev, bool):
            try:
                eid = int(float(ev))
            except (TypeError, ValueError):
                eid = None
        fb: dict[str, Any] | None
        if eid is not None and eid in events_by_id:
            fb = events_by_id[eid]
        else:
            fb = None
        s = _coerce_suggestion(o, fb)
        if s and eid is not None:
            s = {**s, "event_id": eid}
        if s and eid is None and fb is not None and "event_id" in fb:
            s = {**s, "event_id": int(fb["event_id"])}
        if s is not None:
            out.append(s)
    return out


def _events_by_id_from_bundle(bundle: dict[str, Any]) -> dict[int, dict[str, Any]]:
    d: dict[int, dict[str, Any]] = {}
    for c in bundle.get("candidates", []):
        if not isinstance(c, dict):
            continue
        eid = c.get("event_id", 0)
        try:
            eid_i = int(eid)
        except (TypeError, ValueError):
            eid_i = 0
        d[eid_i] = cast("dict[str, Any]", c.get("event", {}))
    return d


def run_ollama_suggestion_json(
    bundle: dict[str, Any],
    client: _GenerateClient,
) -> dict[str, Any]:
    """One ``/api/generate`` call. Returns a parsed object or ``{}`` on failure."""
    prompt = _build_generate_prompt(bundle)
    text = f"{_SYSTEM_PROMPT}\n\nUser:\n{prompt}\n"
    out = client.generate(text)
    parsed = _parse_json_object_from_llm_text(out)
    if isinstance(parsed, dict):
        return parsed
    return {}


def collect_vlm_reviews_for_run(
    bundle: dict[str, Any],
    run_root: Path,
    *,
    ollama_config: OllamaClientConfig,
) -> list[dict[str, Any]]:
    """
    Call the optional VLM (if enabled in config) for each candidate's selected frames; returns
    ``[{\"event_id\": n, \"vlm\": { ... } }, ...]``. On disabled VLM, each ``vlm`` is the
    deterministic placeholder from :mod:`fightsafe_ai.llm.vision_reviewer`.
    """
    from fightsafe_ai.llm.vision_reviewer import review_event_frames

    out: list[dict[str, Any]] = []
    for c in bundle.get("candidates", []):
        if not isinstance(c, dict):
            continue
        try:
            eid = int(c.get("event_id", 0))
        except (TypeError, ValueError):
            eid = 0
        rels = c.get("selected_frame_paths_relative") or []
        paths = [run_root / str(x) for x in rels]
        ev = cast("dict[str, Any]", c.get("event", {}))
        vlm = review_event_frames(paths, ev, ollama_config=ollama_config)
        out.append({"event_id": eid, "vlm": vlm})
    return out


def merge_vlm_reviews_into_bundle(
    bundle: dict[str, Any], vlm_reviews: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """
    Attaches ``optional_vlm_reviews`` for prompt building. Each item should be
    ``{ "event_id": int, "vlm": { ... } }`` (e.g. output of :func:`~fightsafe_ai.llm.vision_reviewer.review_event_frames`).
    """
    b2 = {**bundle}
    b2["optional_vlm_reviews"] = vlm_reviews or []
    return b2


def _vlm_reviews_to_prompt_list(vlm: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return list(vlm or [])


def suggest_annotation_suggestions_ollama(
    bundle: dict[str, Any],
    *,
    ollama_config: OllamaClientConfig | None = None,
    ollama_client: OllamaClient | _GenerateClient | None = None,
    vlm_reviews: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Return ``(suggestions, error_or_none)``. Suggestions are normalized dicts; on failure, empty
    list and a short error string.
    """
    vlm = _vlm_reviews_to_prompt_list(vlm_reviews)
    cfg = ollama_config or load_ollama_config()
    b_for_prompt: dict[str, Any] = merge_vlm_reviews_into_bundle(bundle, vlm)
    e_map = _events_by_id_from_bundle(b_for_prompt)
    if not b_for_prompt.get("candidates"):
        return (
            [],
            "No event candidates; nothing to suggest.",
        )
    client = ollama_client
    if client is None and cfg.enabled:
        client = OllamaClient(cfg)
    if client is None:
        return (
            [],
            "Ollama not configured or not enabled; set ollama.enabled in configs/llm.yaml and pass a client or --use-ollama.",
        )
    try:
        r = run_ollama_suggestion_json(b_for_prompt, client)
    except (LLMError, OSError, TypeError) as e:
        logger.warning("suggest-annotations: Ollama call failed: %s", e)
        return (
            [],
            str(e),
        )
    data: dict[str, Any] = r
    if (
        isinstance(data, dict)
        and "suggestions" not in data
        and isinstance(data.get("suggestion"), list)
    ):
        data = {"suggestions": data.get("suggestion")}
    suggestions = _normalize_suggestion_list(data, e_map)
    return (suggestions, None)


def write_annotation_suggestions(
    out_path: Path,
    document: dict[str, Any],
) -> Path:
    p = out_path.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def build_suggestions_document(
    bundle: dict[str, Any],
    suggestions: list[dict[str, Any]],
    *,
    ollama_used: bool,
    ollama_error: str | None = None,
    vlm_reviews: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "format_version": SUGGESTIONS_FORMAT_VERSION,
        "source_run": bundle.get("run_dir", ""),
        "disclaimer": SUGGESTIONS_NOT_GROUND_TRUTH,
        "not_ground_truth": True,
        "requires_human_merge": True,
        "ollama_used": ollama_used,
        "suggestions": suggestions,
    }
    if ollama_error is not None:
        d["ollama_error"] = ollama_error
    if vlm_reviews is not None:
        d["optional_vlm_reviews"] = vlm_reviews
    return d


def run_pipeline_suggest_annotations(
    run_dir: Path,
    out_path: Path | None = None,
    *,
    use_ollama: bool = False,
    use_vlm: bool = False,
    ollama_client: OllamaClient | _GenerateClient | None = None,
    llm_config: Path | None = None,
    vlm_reviews: list[dict[str, Any]] | None = None,
    max_frames_per_event: int = 3,
) -> dict[str, Any]:
    """
    All-in-one: read run, build bundle, optional VLM text, optionally call Ollama ``generate``,
    write ``annotation_suggestions.json`` and return the document dict.
    """
    r = run_dir.expanduser().resolve()
    out = (out_path or (r / "annotation_suggestions.json")).expanduser().resolve()
    cfg: OllamaClientConfig
    if llm_config and llm_config.is_file():
        from fightsafe_ai.llm.config import load_llm_file_config

        cfg = load_llm_file_config(llm_config).ollama
    else:
        cfg = load_ollama_config()
    bundle = build_assist_bundle(r, max_frames_per_event=max_frames_per_event)
    vlm_merged: list[dict[str, Any]] | None
    vlm_merged = vlm_reviews
    if use_vlm and vlm_merged is None:
        vlm_merged = collect_vlm_reviews_for_run(bundle, r, ollama_config=cfg)
    vlm_for_doc: list[dict[str, Any]] | None = vlm_merged if use_vlm else None
    ollama_used: bool
    suggestions: list[dict[str, Any]] = []
    err: str | None = None
    if use_ollama:
        if not cfg.enabled and ollama_client is None:
            ollama_used = False
            err = "Ollama is disabled in configs/llm.yaml; enable ollama.enabled or inject ollama_client in tests."
        else:
            suggestions, err = suggest_annotation_suggestions_ollama(
                bundle,
                ollama_config=cfg,
                ollama_client=ollama_client,
                vlm_reviews=vlm_merged,
            )
            ollama_used = err is None
    else:
        ollama_used = False
        err = None
    doc = build_suggestions_document(
        bundle,
        suggestions,
        ollama_used=ollama_used,
        ollama_error=err,
        vlm_reviews=vlm_for_doc,
    )
    write_annotation_suggestions(out, doc)
    return doc


__all__ = [
    "SUGGESTIONS_FORMAT_VERSION",
    "SUGGESTIONS_NOT_GROUND_TRUTH",
    "build_assist_bundle",
    "build_suggestions_document",
    "collect_vlm_reviews_for_run",
    "load_event_candidates",
    "load_risk_scores",
    "merge_vlm_reviews_into_bundle",
    "risk_signal_summary",
    "run_ollama_suggestion_json",
    "run_pipeline_suggest_annotations",
    "suggest_annotation_suggestions_ollama",
    "write_annotation_suggestions",
]
