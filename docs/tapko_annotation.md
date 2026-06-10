# TapKO annotation schema

Manual labels for the **FightSafe-TapKO** track: submission-like signals and extreme-vulnerability cues in combat-sports video.

**Machine-readable spec:** `src/fightsafe_ai/annotation/tapko_schema.py` (Pydantic v2).

## Principles

- **Not official outcomes:** intervals describe **annotator judgments** on observable cues, not commission rulings.
- **Human-in-the-loop:** predictions from code are evaluated against these labels; labels themselves may be reviewed in second-pass QA.
- **Times:** `start_time` and `end_time` are **seconds**, `≥ 0`, with **`end_time > start_time`**, on the same timeline as evaluation exports (typically media-relative).

## Root JSON object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format_version` | string | yes | Must be `"1.0"`. |
| `schema_id` | string | yes | Logical id: `fightsafe_ai.tapko_annotation`. |
| `annotation_status` | string | no (default) | Dataset-wide QA gate; default `draft_visual_review` if omitted. See **Annotation status** below. |
| `annotations` | array | yes | List of annotation objects (may be empty). |

### Annotation status (`annotation_status`)

Closed vocabulary on the **document root** (same status applies to all intervals in the file for exports that use a single JSON):

| Value | Meaning |
|-------|---------|
| `draft_transcript_derived` | Intervals seeded from transcripts or tooling; **not** visually verified. |
| `draft_visual_review` | Under annotator review; may omit field to accept this default. |
| `visually_confirmed` | Intervals have passed visual confirmation per protocol—**only this status** qualifies the file as reference-grade for **final reported TapKO metrics** (tables, papers, headline precision/recall). |
| `rejected` | Document discarded or superseded; retain only for audit trails. |

**Reporting rule:** treat precision, recall, F-scores, and related headline numbers as **final reported results** only when `annotation_status` is **`visually_confirmed`**. For any other status (including the default), evaluation may still run for debugging or diagnostics; the TapKO evaluator prints a stderr warning that metrics are **diagnostic only**.

Pilot or transcript-derived bundles should use `draft_transcript_derived` or `draft_visual_review` until visual QA completes.

## Annotation object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_id` | string | yes | Stable identifier for the clip or session. |
| `source_uri` | string | yes | URI or path-like reference to media (file path, `https://`, bucket key, etc.). |
| `start_time` | number | yes | Interval start (seconds). |
| `end_time` | number | yes | Interval end (seconds). |
| `event_type` | string | yes | One of the **TapKO event types** below. |
| `visibility` | string | yes | `clear` \| `partial` \| `poor` \| `unknown` — quality of view of the relevant action. |
| `occlusion_level` | string | yes | `none` \| `light` \| `moderate` \| `heavy` \| `unknown`. |
| `actor_id` | string | yes | Subject of the label (athlete id in your protocol). |
| `target_id` | string \| null | no | Optional opponent / interactee id. |
| `confidence` | number | yes | Annotator confidence in \([0, 1]\). |
| `notes` | string \| null | no | Free text (ambiguity, replay angle, disagreement notes). |
| `rater_id` | string | yes | Annotator or batch id. |
| `requires_audio` | boolean | yes | `true` if the label depends on synchronized audio (e.g. verbal tap). Default `false`. |

Unknown fields are **rejected** (`extra: forbid`).

## Event types (`event_type`)

### Submission signals (`submission_signal.*`)

| Value | Meaning (annotation intent) |
|-------|-----------------------------|
| `submission_signal.hand_tap` | Physical surrender via hand tapping (mat/body/opponent). |
| `submission_signal.foot_tap` | Physical surrender via foot/leg when hands unavailable. |
| `submission_signal.verbal_tap` | Surrender expressed verbally (often **requires_audio**). |
| `submission_signal.technical_submission_candidate` | Visual proxies for referee/medical stoppage during submission **without equating to clinical diagnosis**. |

### Extreme vulnerability (`extreme_vulnerability.*`)

| Value | Meaning |
|-------|---------|
| `extreme_vulnerability.ko_collapse` | Knockdown / collapse / KO–TKO **visual proxy**. |
| `extreme_vulnerability.no_intelligent_defense` | Prolonged inability to defend (high ambiguity — use sparingly). |
| `extreme_vulnerability.post_impact_inactivity` | Extended immobility or disorganized posture after impact or submission entries. |
| `extreme_vulnerability.choke_unconsciousness_candidate` | Visual/audio proxy only; **not** a medical unconsciousness verdict. |

### Negatives (`negative.*`)

Hard negatives and confounders for precision-oriented training/eval.

| Value | Meaning |
|-------|---------|
| `negative.hand_posting` | Hand on mat/body for balance or posting, not a tap. |
| `negative.normal_scramble` | Busy hands/feet during scramble without submission intent. |
| `negative.grip_fighting` | Grip exchanges without tap-like cessation. |
| `negative.celebration_slap` | Non-match-ending slaps (celebration, corner). |
| `negative.fall_without_ko` | Impact or knockdown without KO-like outcome. |

## JSON validation (Python)

```python
from fightsafe_ai.annotation.tapko_schema import (
    parse_tapko_json,
    tapko_json_schema,
    EXAMPLE_DOCUMENT_MINIMAL,
)

doc = parse_tapko_json(open("tapko_labels.json", encoding="utf-8").read())
assert doc.annotations[0].event_type.value.startswith("submission_signal.")

# JSON Schema for editors / CI
schema = tapko_json_schema()
```

Round-trip check on bundled examples:

```python
from fightsafe_ai.annotation.tapko_schema import (
    EXAMPLE_DOCUMENT_MINIMAL,
    EXAMPLE_DOCUMENT_FULL,
    parse_tapko_dict,
)

parse_tapko_dict(EXAMPLE_DOCUMENT_MINIMAL)
parse_tapko_dict(EXAMPLE_DOCUMENT_FULL)
```

## Minimal example

```json
{
  "format_version": "1.0",
  "schema_id": "fightsafe_ai.tapko_annotation",
  "annotation_status": "visually_confirmed",
  "annotations": [
    {
      "video_id": "match_2026_03_clip_v3",
      "source_uri": "s3://bucket/fightsafe/clips/match_2026_03_v3.mp4",
      "start_time": 124.5,
      "end_time": 125.2,
      "event_type": "submission_signal.hand_tap",
      "visibility": "clear",
      "occlusion_level": "none",
      "actor_id": "athlete_blue_corner",
      "target_id": "athlete_red_corner",
      "confidence": 0.92,
      "notes": "Three palm strikes on mat; referee moves in same second.",
      "rater_id": "rater_07",
      "requires_audio": false
    }
  ]
}
```

## Full example (mixed labels)

See `EXAMPLE_DOCUMENT_FULL` in `tapko_schema.py` — includes verbal tap (`requires_audio: true`), a negative scramble, and a KO-collapse proxy with poor visibility.

## Versioning

Bump **`format_version`** in `tapko_schema.py` when breaking JSON shape or enum membership changes; keep migration notes in this file.
