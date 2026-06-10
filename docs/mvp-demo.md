# FightSafe AI — First MVP demo

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:** Camilo José Cela University (UCJC), Madrid, Spain

---

FightSafe AI is a **research software artifact** for traceability and auditability, not a certified medical or officiating system; outputs are **event-level safety alert exports** and **evaluator bookkeeping** under protocol defaults—not claims of operator benefit.

This guide shows how to run the **end-to-end MVP pipeline** (frames → pose → features → interpretable risk → **event-level safety alerts** / overlay video → Markdown report) on a **local combat sports video clip**.

---

## Required Python version

The project targets **Python 3.12** (see `requires-python` in `pyproject.toml`: `>=3.12,<3.13`). Use a 3.12.x interpreter for a supported environment.

---

## Installation steps

1. **Clone the repository** and change into the project root (where `pyproject.toml` lives).

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode** (dev dependencies optional):
   ```bash
   pip install -e ".[dev]"
   ```
   Or, without dev tools:
   ```bash
   pip install -e .
   ```

4. **Verify the CLI** (after install, the `fightsafe` console script is available). The demo below uses `python -m fightsafe_ai.cli` so you do not rely on `PATH` for the script entry point.

**Dependencies** include OpenCV, MediaPipe, NumPy, pandas, and others listed in `pyproject.toml`. No network access is required to *run* the pipeline after install (optional Ollama features are off by default).

---

## Where to place a demo video

- Put your input file at **`data/clips/demo.mp4`** (or pass any other path to `--video`).
- The `data/clips/` directory is intended for **local** clips; the repository is set up so **clip contents are not committed** to Git (see “Git and local artifacts” below).

## Internet / YouTube sources and codecs

Clips from **`yt-dlp`**, **YouTube**, or other web sources are often **AV1** / **VP9** / **HEVC**. **OpenCV** (used to sample frames) may return **no frames** on some systems. **Re-encode to H.264** before `run-pipeline` if you see “no frames extracted” or similar. Step-by-step commands and a short-clip example are in **[`docs/internet-video-codecs.md`](internet-video-codecs.md)**. Respect source **terms of use** and **copyright** for any download.

If `data/clips/` does not exist, create it before copying your file.

---

## Exact command to run

From the **repository root**, with your virtual environment activated and dependencies installed:

```bash
python -m fightsafe_ai.cli run-pipeline \
  --video data/clips/demo.mp4 \
  --output runs/demo/
```

**Notes:**

- `--output` is the **run directory**; it is created or reused and will be filled with artifacts.
- To point at a project `configs/risk_rules.yaml`, run the command from the repo root, or pass `--rules /path/to/risk_rules.yaml`.
- Optional: `--report-ollama` adds a short narrative to `report.md` **only** if a local Ollama server is reachable; the MVP works **without** Ollama.
- Optional: `--explain-events` writes per-event Markdown under `explanations/` (template or Ollama depending on flags).

If you prefer the installed script name: `fightsafe run-pipeline --video ... --output ...` (same options).

---

## Expected outputs

After a successful run, **under `runs/demo/`** (or your chosen `--output` path) you should see:

| Path | Description |
|------|-------------|
| `frames/` | JPEG frames sampled from the input video (sampling rate set by `--fps`, default 10) |
| `pose_keypoints.csv` | MediaPipe pose export (long format) |
| `features.csv` | Biomechanical and temporal features used for scoring |
| `risk_scores.csv` | Per-frame interpretable risk: `timestamp`, `risk_score`, `risk_level`, `triggered_rules`, `frame_id`, `frame_index`, etc. |
| `events.json` | Merged segments where level is **HIGH** or **CRITICAL** (heuristic) |
| `output_overlay.mp4` | Video built from the sampled frames with pose + on-screen risk HUD |
| `report.md` | Human-readable summary, artifact list, and disclaimer |

A temporary stitched preview file may be used internally during rendering and is not part of the stable artifact list.

---

## How to interpret risk scores

- **`risk_score`** (0.0–1.0): weighted combination of **transparent rule components** defined in `configs/risk_rules.yaml` (e.g. fast downward motion, large torso angle, prolonged low posture, high instability, post-fall low movement). Weights and thresholds are **tunable**; they are heuristics, not learned “AI mystery scores.”
- **`risk_level`**: **LOW**, **MEDIUM**, **HIGH**, or **CRITICAL** — derived from `risk_score` using YAML cutoffs (`level_medium_min`, `level_high_min`, `level_critical_min` under `interpretable_aggregation`).
- **`triggered_rules`**: list of rule **keys** whose per-frame component exceeded `trigger_epsilon` — use this to see *which* signals contributed, for explainability.
- **`timestamp`**: time in **seconds** along the **sampled** frame sequence (index / `fps` after natural sort of `frame_id`).

**Important:** this is **decision-support for human review**, not an automated call on fouls, stoppage, or medical state.

---

## Limitations

- Scores are **heuristic** and **domain- and data-dependent**; validate on your own league’s footage before any operational use.
- Pose and features depend on **MediaPipe** visibility, lighting, and occlusions; poor tracking can distort geometry and risk.
- Sampling with `--fps` does **not** use every source-video frame: alignment uses a stitched video from the extracted frames, not necessarily the full original timebase frame-by-frame.
- **HIGH/CRITICAL** in `events.json` are **post-processed segment labels** for analytics and visualization, not a substitute for an official’s judgment.

---

## Safety disclaimer

FightSafe AI outputs (scores, levels, events, and overlays) are for **research, training, and engineering** workflows. The system is **not** a medical device, does **not** diagnose injury, concussion, or any medical condition, and is **not** a substitute for qualified **medical** staff, **safety** protocols, or **governing-body** rules. Any action in a real contest remains the responsibility of the **human officials and organizers**.

---

## Git and local artifacts

**Do not commit** raw or processed **videos** under `data/clips/`, or **run outputs** under `runs/`, to Git. The project `.gitignore` is intended to exclude typical paths such as `data/clips/*` and `runs/*` (with exceptions like `.gitkeep` only where used). Keep demo videos and generated files **local** or in private storage; share only what your policy allows.

For reproducibility, version **code** and **config** (e.g. `configs/risk_rules.yaml`); keep large binaries and per-run exports **out of** the repository.

---

## See also

- `configs/risk_rules.yaml` — rule weights, thresholds, and level bands for the interpretable combat MVP
- `docs/architecture.md` — system structure and design intent
