# Datasets in FightSafe AI

FightSafe AI ships **metadata only**: a registry describing kinds of combat-sports and
pose-related datasets you may use with the tooling. **No dataset bytes are downloaded,
installed, or committed** by this repository.

Researchers must **manually download** each corpus from its official source and comply with
that source’s **license and terms** (academic use, redistribution, attribution, etc.).

---

## Supported formats (at a glance)

| Format | Typical use | Loader module | Auto-download |
|--------|-------------|---------------|---------------|
| **FightSafe CSV** | Long keypoints (`frame_id`, `keypoint_name`, `x`, `y`, …) from the MVP pipeline | Core pipeline / `keypoints.io` | No |
| **COCO person keypoints** | `instances_person_keypoints_*.json` + images | `fightsafe_ai.datasets.coco_loader` | No |
| **YOLO-pose / Ultralytics** | Label files or exported tensors from YOLO-pose training | `fightsafe_ai.datasets.yolo_pose_loader` (helpers; add your reader) | No |
| **Custom safety clips** | In-house video + event / risk annotations | Your lab script + `schemas.RiskEventAnnotation` | No |
| **MMA / boxing public sets** | Varies by paper; often custom JSON or video + labels | **Custom** (document in registry `notes`) | No |

The registry entry field `supported_loader` points to the **intended** Python module or
`manual_only` when you must write an adapter.

---

## Registry (`fightsafe_ai.datasets.registry`)

Each entry is a :class:`fightsafe_ai.datasets.schemas.DatasetMetadata` instance with:

- **name** — Human-readable title.
- **task_type** — e.g. `pose_estimation`, `action_recognition`, `risk_event`.
- **license** — Short summary; **the legal text is always on the upstream site**.
- **source_url** — Link to the official page or paper (may be empty for private/TBD data).
- **annotation_format** — e.g. `coco_person_json`, `yolo_pose_labels`, `fightsafe_csv`.
- **supported_loader** — e.g. `coco_loader`, `yolo_pose_loader`, `manual_only`.
- **notes** — Splits, keypoint count, sport, and how to map to FightSafe (e.g. BlazePose-33).

**Browse in code:** `BUILTIN_REGISTRY` in `src/fightsafe_ai/datasets/registry.py`.

**Query:** `get_spec("registry_key")` or `list_registry_keys()`.

**Runtime extension (non-persistent):** `register_dataset("my_key", DatasetMetadata(...))` for
local experiments or tests.

---

## How to add a dataset

1. **Obtain the data** from the author or official host under their license. Do not commit
   videos, images, or large JSON to Git.
2. **Add or update** a `DatasetMetadata` entry in `BUILTIN_REGISTRY` (or call
   `register_dataset` from your private config) with honest `license`, `source_url`, and
   `annotation_format`.
3. **Write a small loader** (in your repo or a private package) that reads **local paths**
   and returns rows matching `FighterKeypointsSample`, `ActionAnnotation`, or
   `RiskEventAnnotation` as needed.
4. **Map keypoints** if the upstream schema differs (e.g. COCO-17 → BlazePose-33) in that
   adapter; the core library does not assume a single global mapping.

---

## Licensing rules

- The **registry text** is documentation, not a grant of rights. You must read and follow
  each dataset’s **original** license and any click-through or registration requirements.
- **COCO** is widely used under its terms (see [COCO](https://cocodataset.org/)); do not
  redistribute the full image set from this project.
- **Ultralytics / YOLO** tooling is subject to [AGPL-3.0](https://github.com/ultralytics/ultralytics)
  for the `ultralytics` package; your **videos and labels** have separate terms.
- **In-house** safety clips: default to **not redistributable**; use NDAs and institutional
  policy as required.

When in doubt, keep data **local** and document access in your lab’s README, not in this repo.

---

## Why datasets are not in Git

- **Size** — Video and high-resolution image corpora are too large for normal version control.
- **License** — Many datasets forbid public redistribution or require approval.
- **Liability** — Raw fight footage and annotations may be sensitive; keep them on secure
  storage and reference paths only (e.g. environment variables, local `config.yaml`).

The FightSafe repository intentionally contains **schemas, registry metadata, and small
test fixtures** (e.g. synthetic CSVs under `tests/`) only.

---

## Schema types (for adapters)

Defined in `fightsafe_ai.datasets.schemas`:

- `FighterKeypointsSample` — one frame’s keypoints for one fighter.
- `ActionAnnotation` — a temporal action label.
- `RiskEventAnnotation` — a reference risk interval for evaluation (not live officiating).

Use these for type clarity and future Parquet/JSON export; they do not pull in heavy deps.

---

## TapKO labels and collection policy

The **TapKO** track uses its **own** annotation JSON schema (`fightsafe_ai.annotation.tapko_schema`; human-readable guide: [`tapko_annotation.md`](tapko_annotation.md)). That schema covers **submission_signal.*** and **extreme_vulnerability.*** event types (plus optional **negative.*** hard negatives), independent of the legacy MVP annotation template in [`annotation.md`](annotation.md).

**Policy (practical).**

- **No dataset bytes in this repo** — same rule as above: clips, frames, and label files stay **local** or on governed storage; reference paths in experiment configs only.
- **Plan and protocol** — clip selection, consent, and multi-rater workflow are described in [`tapko_dataset_plan.md`](tapko_dataset_plan.md). Treat that file as the **lab checklist**, not a guarantee that a public corpus exists.
- **Licensing and ethics** — combat footage is sensitive. Obtain **rights and consent** for recording, reuse, and publication of metrics; redact or withhold identifiers per institutional policy.
- **Evaluation hooks** — once labels exist locally, run `fightsafe tapko-evaluate` (see [`evaluation.md`](evaluation.md)) against `tapko_predictions.json` from `fightsafe tapko-detect`; align **`video_id`** strings between annotations and predictions.

Do **not** assume TapKO-labeled data ships with the package; registry entries may reference future corpora in `BUILTIN_REGISTRY` with honest `notes` and `manual_only` loaders until adapters exist.
