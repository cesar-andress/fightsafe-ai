# Scripts

Reproduction and dataset utilities for FightSafe AI.

## Reproduction (companion manuscripts)

| Script | Manuscript | Description |
|--------|------------|-------------|
| `reproduce_fusion2026.sh` | `../fusion2026/` | BoxingVI batch eval, LaTeX assets, PDF |
| `reproduce_sinica2026.sh` | `../sinica2026/` | TapKO `jedi_submissions` pilot |
| `reproduce_sports.sh` | `../sports/` | FightSafe-Bench CSV/JSON exports |
| `reproduce_all.sh` | all three | Best-effort; continues on missing data |

Makefile equivalents: `make reproduce-fusion`, `make reproduce-sinica`, `make reproduce-sports`, `make reproduce-all`.

## Asset generation (fusion2026)

```bash
python scripts/generate_paper_assets.py --paper-dir ../fusion2026
```

Delegates to `tools/generate_paper_assets.py` and related helpers. Requires ablation exports under `runs/case_studies/ablation_summary/` when regenerating ablation tables.

## Benchmark dataset (sports)

```bash
python scripts/build_fightsafe_bench.py
python scripts/extract_benchmark_features.py
```

## ONNX export (optional pose backend)

```bash
fightsafe export-rtmpose-onnx --config <config.py> --checkpoint <weights.pth> -o model.onnx
```

See `docs/architecture.md` for pose backend options.
