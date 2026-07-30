# Data and licensing notice

## Software
Source code and configs in this repository are released under the MIT License (see `LICENSE`).

## BoxingVI punch-interval annotations
Spreadsheets under `annotations/boxingvi/` are research copies of BoxingVI punch-interval labels
(V1–V10) used as an impact/punch proxy in the paper. Cite Kumar et al. (BoxingVI; arXiv:2511.16524).
Upstream BoxingVI terms continue to apply.

## Not redistributed
- Raw BoxingVI videos / frames
- BoxingVI skeleton keypoints
- Derived `features_cache/*.pkl` (optional Tier B inputs)
- MediaPipe / TensorFlow / OpenCV binaries (install via pip)

## Strike baselines
JSON strike/anomaly baselines under `optional_tier_b/inputs/strike_baselines/` are derived pipeline
outputs from this software and are included for interpretation of the combined timeline.
They are useful together with Tier B feature caches when those become available under appropriate clearance.

## Tier A without restricted data
Regenerate and verify manuscript tables and figures from frozen canonical CSVs under
`canonical_results/run_20260730_005150/`.
