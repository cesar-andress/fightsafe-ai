# Reference reproduction snapshots

Small CSV/JSON files shipped with the repository so verification scripts can run **without** large videos or BoxingVI skeleton keypoints.

## fusion2026

`fusion2026/reference/boxingvi/`

| File | Description |
|------|-------------|
| `boxingvi_results_all.csv` | Pooled full-fusion batch metrics |
| `baseline_comparison.csv` | Four-way baseline comparison |
| `boxingvi_results_V*.csv` | Per-stem aggregates |
| `sweep_summary.csv` | Strike-percentile sweep summary |
| `sweeps/p{85,90,95,97}_boxingvi_results_all.csv` | Per-percentile pooled exports |

These document software exports used during manuscript development. They do not replace re-running `make fusion-all` when `data/boxingvi/skeleton/` is available.

## jss2026 (traceability stress test)

`eswa2026/reference/` — internal folder name at artifact creation (v0.1.3).

| File | Description |
|------|-------------|
| `tapko_predictions.json` | Reference detector export (`jedi_submissions`) |
| `tapko_results.csv` | Reference evaluator metrics |
| `tapko_error_analysis.md` | Error category counts |

Use with `REPRO_USE_REFERENCE=1 bash scripts/reproduce_eswa2026.sh` when the source video is not downloaded.

## Regenerating references

After a verified full reproduction run, maintainers may refresh these files:

```bash
cp outputs/tapko/jedi_submissions_eval/tapko_results.csv data/repro/eswa2026/reference/
cp outputs/evaluation/boxingvi_batch/boxingvi_results_all.csv data/repro/fusion2026/reference/boxingvi/
```

Do not commit large prediction JSON/CSV dumps from BoxingVI batch runs.
