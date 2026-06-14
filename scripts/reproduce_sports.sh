#!/usr/bin/env bash
# Reproduce sports / FightSafe-Bench dataset exports and feature extraction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

echo "== FightSafe AI: reproduce sports / FightSafe-Bench =="
echo "Manuscript dir: $SPORTS_DIR"
echo "Repro outputs:  $SPORTS_REPRO_DIR"

fs_mkdir_outputs

"$PYTHON" scripts/build_fightsafe_bench.py \
  --input-dir "$REPO_ROOT/data/FightSafeBench/annotations" \
  --output-dir "$REPO_ROOT/data/FightSafeBench" \
  --summary-path "$SPORTS_DIR/dataset_summary.md"

cp -f "$REPO_ROOT/data/FightSafeBench/events.csv" "$SPORTS_REPRO_DIR/" 2>/dev/null || true
cp -f "$REPO_ROOT/data/FightSafeBench/dataset_statistics.json" "$SPORTS_REPRO_DIR/" 2>/dev/null || true

POSE_CSV="${BENCHMARK_POSE_CSV:-$REPO_ROOT/outputs/tapko/jedi_submissions/pose_keypoints.csv}"
if [[ -f "$POSE_CSV" ]]; then
  "$PYTHON" scripts/extract_benchmark_features.py --pose-csv "$POSE_CSV" \
    --output "$SPORTS_REPRO_DIR/benchmark_features.csv"
else
  echo "NOTE: pose_keypoints.csv not found ($POSE_CSV)."
  echo "      Run reproduce_iswa2026.sh first, or set BENCHMARK_POSE_CSV=..."
fi

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$SPORTS_DIR/main.tex" ]]; then
  (cd "$SPORTS_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  cp -f "$SPORTS_DIR/main.pdf" "$SPORTS_REPRO_DIR/main.pdf" 2>/dev/null || true
  echo "PDF: $SPORTS_DIR/main.pdf (skeleton manuscript; result tables are TBD)"
fi

"$PYTHON" scripts/verify_paper_outputs.py --paper sports
echo "Done."
