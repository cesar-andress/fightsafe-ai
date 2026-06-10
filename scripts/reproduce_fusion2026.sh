#!/usr/bin/env bash
# Reproduce fusion2026 (Information Fusion) experiment assets and PDF.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUSION_DIR="${FUSION_DIR:-$ROOT/../fusion2026}"
cd "$ROOT"

echo "== FightSafe AI: reproduce fusion2026 =="
echo "Repository root: $ROOT"
echo "Manuscript dir:  $FUSION_DIR"

if [[ ! -f "$FUSION_DIR/main.tex" ]]; then
  echo "ERROR: $FUSION_DIR/main.tex not found."
  echo "Clone or place the fusion2026 manuscript next to this repository."
  exit 1
fi

if [[ ! -d data/boxingvi/skeleton ]]; then
  echo "WARNING: data/boxingvi/skeleton/ missing — BoxingVI batch eval will be skipped."
  echo "See data/README.md for download instructions."
  echo "Continuing with asset regeneration from existing exports (if any)..."
  make fusion-assets FUSION_DIR="$FUSION_DIR" || true
else
  make fusion-all FUSION_DIR="$FUSION_DIR"
fi

if command -v pdflatex >/dev/null 2>&1; then
  make fusion-pdf FUSION_DIR="$FUSION_DIR"
  echo "PDF: $FUSION_DIR/main.pdf"
else
  echo "pdflatex not found — skipping PDF build. Install TeX Live and run: make fusion-pdf"
fi

if [[ -f "$FUSION_DIR/scripts/regenerate_figures.py" ]]; then
  python3.12 "$FUSION_DIR/scripts/regenerate_figures.py" || echo "Figure regeneration skipped (optional)."
fi

echo "Done."
