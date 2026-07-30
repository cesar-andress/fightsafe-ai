# Data and licensing notice

## Software
Source code and configs in this package are released under the MIT License (see `LICENSE`).

## BoxingVI punch-interval annotations
Spreadsheets under `annotations/boxingvi/` are research copies of BoxingVI punch-interval labels
(V1–V10) used as an **impact/punch proxy** in the paper. Cite Kumar et al. (BoxingVI; arXiv:2511.16524).
Upstream BoxingVI terms continue to apply.

## Not redistributed
- Raw BoxingVI videos / frames
- BoxingVI skeleton keypoints (`data/boxingvi/skeleton/`)
- Derived `features_cache/*.pkl` (Tier B) — **PENDING INSTITUTIONAL APPROVAL**
- MediaPipe / TensorFlow / OpenCV binaries (install via pip)

## Strike baselines
JSON strike/anomaly baselines under `optional_tier_b/inputs/strike_baselines/` are derived pipeline
outputs from this software. Status: **APPROVED FOR RELEASE** as research artefacts of this work,
but they are useful only together with Tier B feature caches.

## What Tier A can do without restricted data
Regenerate and verify manuscript tables and figures from frozen canonical CSVs under
`results/run_20260730_005150/`.
