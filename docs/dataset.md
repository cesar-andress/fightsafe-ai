# FightSafe AI

**Authors:**
David Martin Moncunill (david.martinm@ucjc.edu)
César Andrés Sánchez (cesar.andress@ucjc.edu)

**Affiliation:**
Camilo José Cela University (UCJC)
Madrid, Spain

---

# Dataset guidelines

FightSafe AI does **not** ship training data. Teams should curate their own video datasets under agreed ethics and consent policies.

## Recommended metadata (per clip)

- **Source**: event id, date, camera id (if multi-view).
- **Sport / ruleset**: e.g. boxing, MMA, muay thai.
- **Participant**: anonymized id; age band if policy allows.
- **Labeling**: optional expert annotations for falls, stoppages, medical timeouts—used for evaluation, not required for heuristic MVP.

## Storage layout (example)

```text
data/
  raw/           # original videos (gitignored)
  derived/
    frames/      # extracted frames
    keypoints/   # CSV per frame
  labels/        # optional JSON/CSV annotations
```

## Privacy

- Strip audio unless needed and permitted.
- Avoid storing identifiable imagery in public forks; use `.gitignore` for `data/raw/`.

## Benchmarking

When publishing results, report:

- Frame rate used for extraction and pose.
- Camera angle (side, front, elevated).
- Dataset size and label definition for “risk” events.
