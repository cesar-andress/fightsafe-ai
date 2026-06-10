#!/usr/bin/env bash
# Reproduce sports / FightSafe-Bench dataset exports and feature extraction.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPORTS_DIR="${SPORTS_DIR:-$ROOT/../sports}"
cd "$ROOT"

echo "== FightSafe AI: reproduce sports / FightSafe-Bench =="

python3.12 scripts/build_fightsafe_bench.py \
  --input-dir data/FightSafeBench/annotations \
  --output-dir data/FightSafeBench \
  --summary-path "$SPORTS_DIR/dataset_summary.md"

if [[ -f outputs/tapko/jedi_submissions/pose_keypoints.csv ]]; then
  python3.12 scripts/extract_benchmark_features.py
else
  echo "NOTE: outputs/tapko/jedi_submissions/pose_keypoints.csv not found."
  echo "Run 'make reproduce-sinica' first for feature extraction, or pass --pose-csv manually."
fi

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$SPORTS_DIR/main.tex" ]]; then
  (cd "$SPORTS_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  echo "PDF: $SPORTS_DIR/main.pdf"
fi

echo "Done."
