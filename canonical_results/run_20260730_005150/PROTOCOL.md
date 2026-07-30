# EAAI checkpoint protocol

## Justification
Matches legacy ``paper1_pre_rewrite_.../scripts/run_aggregation_comparison.py`` BoxingVI path.

## Inputs
- Features: `optional_tier_b/inputs/features_cache` (read-only)
- Fixed strike/anomaly: `optional_tier_b/inputs/strike_baselines`
- Annotations: `annotations/boxingvi`
- Configs: `configs/risk_fusion.yaml`, `configs/risk_rules.yaml` (not tuned on labels)

## Matrix
- `c_<rule>`, `a_<rule>` via `build_rule_components` once per video
- α=0 ⇒ channel excluded; c=0 & α=1 ⇒ zero-valued evidence retained
- `y_impact` stored for analysis only — never used for scoring/tuning

## Temporal contract (BoxingVI)
- fps=30.0; risk merge_gap_frames=2; min_duration=0.0s
- BoxingVI timeline merge=8 frames
- IoU=0.01; tolerance=0.5s; subset=**full_fusion**
- Risk leg re-aggregated per scheme; strike events held fixed from baseline cache

## Aggregation comparison
- equal / weighted / max with **interactions ON**

## Interaction ablation
- weighted int OFF vs ON

## Dropout
- weighted; p=[0.0, 0.1, 0.3, 0.5]; seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for p>0
- explicit α=0 vs naive c=0/α=1
