# Manual annotations for evaluation

FightSafe AI can be scored **offline** against **human ground truth**: time intervals in a clip where a specific **safety-related event** occurred. This document describes the JSON format, the workflow, and the limits of hand labelling.

## Why manual labels?

The pipeline’s `events.json` is **model output**, not a gold standard. For research you need a separate, versioned file of **intended** segments (e.g. for precision/recall or temporal IoU in `fightsafe_ai.evaluation`).

## File format

- **Path (example):** `annotations/demo_annotations.json` (any path; keep under `annotations/` or your study folder).
- **Root object fields:**

| Field            | Type   | Description |
|------------------|--------|-------------|
| `format_version` | string | Must be `"1.0"`. |
| `video`          | string | Reference to the clip: path, URI, or id. Echo of `fightsafe annotate-template --video`. |
| `time_unit`      | string | Must be `"seconds"`. |
| `events`         | array  | List of event objects (see below). |

**Event object** (`EventAnnotation`):

| Field         | Type   | Required | Description |
|---------------|--------|----------|-------------|
| `start_time`  | number | yes      | Start in **seconds** (≥ 0, finite). |
| `end_time`    | number | yes      | End in seconds (**strictly greater** than `start_time`). |
| `event_type`  | string | yes      | One of: `FALL`, `KO`, `SURRENDER`, `INSTABILITY`. |
| `confidence`  | number | no       | Optional annotator self-confidence in **\[0, 1\]**. |
| `notes`       | string | no       | Free text (e.g. “partially occluded”). |

Example (one region marked):

```json
{
  "format_version": "1.0",
  "video": "data/clips/match_A_round_1.mp4",
  "time_unit": "seconds",
  "events": [
    {
      "start_time": 42.0,
      "end_time": 45.2,
      "event_type": "FALL",
      "confidence": 0.9,
      "notes": "Clear knockdown, both feet off canvas"
    }
  ]
}
```

## Workflow

1. **Create a template** (empty `events` list) tied to your video:
   ```bash
   fightsafe annotate-template --video data/clips/my_clip.mp4 -o annotations/demo_annotations.json
   ```
2. **Edit the JSON** in a text editor or small script. Add one object per segment. Use the **same time base** you will use for the pipeline (usually seconds from the start of the same clip the run used).
3. **Validate** before scoring:
   ```bash
   fightsafe validate-annotations annotations/demo_annotations.json
   ```
4. **Use in evaluation** by loading the document in Python:
   ```python
   from fightsafe_ai.annotation import load_annotation_file
   doc = load_annotation_file("annotations/demo_annotations.json")
   for e in doc.events:
       print(e.start_time, e.end_time, e.event_type)
   ```

## How to define events

- **One row per segment** of interest. Instants can be approximated with a very short \([t, t+\varepsilon]\) window (document the choice in `notes` if needed).
- **Vocabulary** is fixed to the four `event_type` values so reports stay comparable. Map your protocol to them explicitly (e.g. “TKO stoppage” → `KO` for your study, and record the mapping in the paper or protocol).
- **Time alignment:** Prefer the same clip and trim as the automated run. If the run used `risk_scores.csv` with `timestamp` in seconds from t=0 of that clip, use the same reference for `start_time` / `end_time`.
- **Overlap:** The schema allows overlapping intervals. The validator may print **notes** (not errors) if two **same-type** segments overlap; you can merge or split them for stricter studies.

## Limitations of manual annotation

- **Subjectivity:** “Fall” vs “instability” and confidence in broadcast footage are not objective; use multiple raters and agreement metrics if you need rigour.
- **Occlusion and camera:** Labels reflect what the annotator could see, not a full 3D truth.
- **Not a medical or officiating record:** These files are **research aids** for system evaluation, not a medical diagnosis or a formal contest decision.
- **No automatic sync with `events.json`:** You must align ground truth to model output in code (same video, same clock).

## API reference (library)

| Module | Role |
|--------|------|
| `fightsafe_ai.annotation.schema` | Pydantic models: `EventAnnotation`, `AnnotationDocument`, `EventType` |
| `fightsafe_ai.annotation.loader` | `new_empty_template`, `load_annotation_file`, `save_annotation_file` |
| `fightsafe_ai.annotation.validator` | `validate_annotation_file`, `format_overlap_warnings` |
| `fightsafe_ai.annotation.llm_assist` | `run_pipeline_suggest_annotations` (Ollama draft labels; not ground truth) |

## Comparing to pipeline output

After filling labels, compare your file to a run’s `events.json` with the evaluation driver and metrics in [`docs/evaluation.md`](evaluation.md) (command: `fightsafe evaluate --predicted ... --ground-truth ...`).

## LLM-assisted drafts (Ollama; not ground truth)

The `fightsafe suggest-annotations` command writes **`annotation_suggestions.json`** in a **completed pipeline run** directory. It is a **separate** artefact from the official `AnnotationDocument` used in evaluation. Purpose: speed up *manual* labelling by offering draft intervals and proposed `event_type` labels based on `events.json`, per-event `risk_scores.csv` summaries, optional on-disk frame paths under `frames/`, and (if you enable it) **optional** VLM text from the same Ollama stack (see `configs/llm.yaml`).

| Rule | |
|------|---|
| **Human confirmation** | **Mandatory** — copy accepted rows into your annotation file only after you agree; never treat the file as auto-ground-truth. |
| **Not authoritative** | Suggested labels and rationales are **not** a substitute for watching the source video or following your study protocol. |
| **Ollama** | Set `ollama.enabled: true` in `configs/llm.yaml` when you want a local model to generate the JSON; without `--use-ollama` the command still writes a file with empty `suggestions` and run context. |
| **Optional VLM** | With `--use-vlm`, the optional vision review text is attached and may be passed into the text model’s context (`enable_vlm_review` in YAML). VLM output remains interpretability-only (see the vision-language event review design in the paper). |

Example:

```bash
fightsafe suggest-annotations --run runs/demo/ --use-ollama
# with optional VLM text in the text prompt
fightsafe suggest-annotations --run runs/demo/ --use-ollama --use-vlm
```

Output is versioned with `format_version: "1.0"` in `annotation_suggestions.json`. The Python entry point is `fightsafe_ai.annotation.llm_assist.run_pipeline_suggest_annotations`.

## Versioning

`format_version: "1.0"` is the only supported value in this release. A future version may add fields; old files will remain loadable with explicit version checks when introduced.
