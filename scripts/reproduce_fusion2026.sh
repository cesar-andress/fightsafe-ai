#!/usr/bin/env bash
# Reproduce fusion2026 (Information Fusion) experiment assets and PDF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

echo "== FightSafe AI: reproduce fusion2026 =="
echo "Repository root: $REPO_ROOT"
echo "Manuscript dir:  $FUSION_DIR"
echo "Repro outputs:   $FUSION_REPRO_DIR"

fs_require_file "$FUSION_DIR/main.tex" "Place fusion2026 next to this repository or set FUSION_DIR."
fs_mkdir_outputs

# --- Stage A: ablation tables + figures (always feasible; bundled ablation CSV) ---
"$PYTHON" scripts/export_fusion2026_assets.py \
  --fusion-dir "$FUSION_DIR" \
  --output-dir "$FUSION_REPRO_DIR"

# --- Stage B: BoxingVI batch evaluation (requires skeleton keypoints) ---
if [[ -d "$REPO_ROOT/data/boxingvi/skeleton" ]]; then
  echo "BoxingVI skeleton found — running full batch pipeline."
  make fusion-all FUSION_DIR="$FUSION_DIR" OUTPUT_DIR="$REPO_ROOT/outputs/evaluation/boxingvi_batch" || {
    echo "WARNING: fusion-all failed; continuing with reference snapshots."
  }
else
  echo "NOTE: data/boxingvi/skeleton/ not present."
  echo "      Using reference CSV snapshots from data/repro/fusion2026/reference/boxingvi/."
  echo "      See docs/REPRODUCIBILITY.md for download instructions."
  mkdir -p "$REPO_ROOT/outputs/evaluation/boxingvi_batch"
  cp -f "$REPRO_DATA_ROOT/fusion2026/reference/boxingvi/boxingvi_results_all.csv" \
    "$REPO_ROOT/outputs/evaluation/boxingvi_batch/" 2>/dev/null || true
  cp -f "$REPRO_DATA_ROOT/fusion2026/reference/boxingvi/baseline_comparison.csv" \
    "$REPO_ROOT/outputs/evaluation/boxingvi_batch/" 2>/dev/null || true
fi

# --- Stage C: compile PDF ---
if command -v pdflatex >/dev/null 2>&1; then
  make fusion-pdf FUSION_DIR="$FUSION_DIR"
  if [[ -f "$FUSION_DIR/main.pdf" ]]; then
    cp -f "$FUSION_DIR/main.pdf" "$FUSION_REPRO_DIR/main.pdf"
    echo "PDF: $FUSION_DIR/main.pdf (copy: $FUSION_REPRO_DIR/main.pdf)"
  fi
else
  echo "pdflatex not found — skipping PDF build."
fi

"$PYTHON" scripts/verify_paper_outputs.py --paper fusion
echo "Done."
