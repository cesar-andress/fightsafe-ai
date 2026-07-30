# FINAL_REPORT — EAAI checkpoint

## 1. Executive verdict

**GO**

- n=10 videos; limited statistical power.
- GT is punch→impact proxy, not safety labels.
- Hand-authored YAML weights/thresholds/rules (probable config leakage risk disclosed).
- Evaluation uses full_fusion with fixed strike leg (documented BoxingVI aggregation protocol).
- aggregation_schemes.py inspection PASS (aggregation-only; no GT; no BoxingVI-specific logic).

## 2. Git / inspection state

Working tree `main` @ `6c1c3e860d8623a0beaa40b8d7c8495c07037f48` (no branch/commit/push performed).

Untracked `src/fightsafe_ai/evaluation/aggregation_schemes.py` inspected:

| Check | Result |
|-------|--------|
| Only aggregation schemes | PASS |
| No feature-extraction changes | PASS |
| No ground-truth access | PASS |
| No eval-data parameter tuning | PASS |
| α=0 vs c=0 distinction | PASS (via `active`; weighted path excludes α=0 in runner) |
| No BoxingVI-specific logic | PASS |

## 3. Inputs / hashes

See `input_checksums.tsv`.

## 4. Matrix schema

`video_id, frame_index, timestamp, fps, c_<rule>, a_<rule>, y_impact` (y unused for scoring).

## 5. Dataset stats

See `matrices/matrix_meta.csv`.

## 6. Temporal protocol

fps=30.0; risk merge_gap=2; d_min=0.0s (BoxingVI path);
BoxingVI merge=8; IoU=0.01; tol=0.5s;
subset=**full_fusion** (risk re-aggregated; strikes fixed). Primary aggregation comparison with interactions **ON**.

## 7. Equal vs weighted vs max

| method | interactions | micro-F1 | macro-F1 | bootstrap macro-F1 95% CI |
|--------|--------------|----------|----------|---------------------------|
| equal | on | 0.5022 | 0.5546 | [0.4714, 0.6670] |
| max | on | 0.1777 | 0.2348 | [0.1337, 0.3586] |
| weighted | on | 0.5025 | 0.5673 | [0.4922, 0.6739] |

## 8. Interaction ablation

| method | interactions | micro-F1 | macro-F1 | bootstrap macro-F1 95% CI |
|--------|--------------|----------|----------|---------------------------|
| weighted | off | 0.4668 | 0.4441 | [0.3568, 0.5200] |
| weighted | on | 0.5025 | 0.5673 | [0.4922, 0.6739] |

Configured rules: 6; channels present for 5/6.
Total firings (int ON): 34329.

## 9. Rule firings

See `interaction_rule_firings.csv`.

## 10. Dropout

| p | mode | mean micro-F1 over seeds |
|---|------|--------------------------|
| 0.0 | none | 0.5025 |
| 0.1 | explicit_alpha0 | 0.5057 |
| 0.1 | naive_zero | 0.4986 |
| 0.3 | explicit_alpha0 | 0.5074 |
| 0.3 | naive_zero | 0.4830 |
| 0.5 | explicit_alpha0 | 0.5090 |
| 0.5 | naive_zero | 0.4709 |

Full table: `dropout_results.csv`.

## 11–13. Per-video / bootstrap / paired

See `per_video_metrics.csv`, `experiment_summary.csv`, `paired_comparisons.csv`.

## 14. Failures

See `failures/all_failures.csv`, `failure_counts.csv`.

## 15. Reproducibility

V2 weighted double-run identical: **True**.

## 16. Runtime

81.3s experimental layer.

## 17–18. Leakage & claim restrictions

Disclose hand-tuned config, n=10, punch-proxy labels. No clinical/deployment/SOTA-fusion claims.

## 19. Next experiments

LOVO logistic; family dropout; block dropout; full runtime analysis.

## 20. Rewrite paper1?

Yes — checkpoint supports proceeding with disclosed limits.
