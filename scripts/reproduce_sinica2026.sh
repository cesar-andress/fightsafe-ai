#!/usr/bin/env bash
# Reproduce sinica2026 TapKO workflow demonstration (jedi_submissions pilot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

DETECT_DIR="${SINICA_DETECT_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions}"
EVAL_DIR="${SINICA_EVAL_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions_eval}"
REFERENCE_DIR="$REPRO_DATA_ROOT/sinica2026/reference"
USE_REFERENCE="${REPRO_USE_REFERENCE:-0}"

echo "== FightSafe AI: reproduce sinica2026 TapKO pilot =="
echo "Manuscript dir: $SINICA_DIR"
echo "Repro outputs: $SINICA_REPRO_DIR"

fs_require_file "$TAPKO_ANNOTATIONS"
fs_mkdir_outputs
mkdir -p "$DETECT_DIR" "$EVAL_DIR" "$SINICA_REPRO_DIR/tables"

if [[ "$USE_REFERENCE" == "1" ]]; then
  echo "REPRO_USE_REFERENCE=1 — using bundled reference predictions (no video required)."
  fs_require_file "$REFERENCE_DIR/tapko_predictions.json"
  cp -f "$REFERENCE_DIR/tapko_predictions.json" "$DETECT_DIR/"
  [[ -f "$REFERENCE_DIR/tapko_manifest.json" ]] && cp -f "$REFERENCE_DIR/tapko_manifest.json" "$DETECT_DIR/"
  cp -f "$REFERENCE_DIR/tapko_results.csv" "$EVAL_DIR/" 2>/dev/null || true
else
  fs_require_file "$TAPKO_VIDEO" "Download video per data/README.md or set TAPKO_VIDEO=... or REPRO_USE_REFERENCE=1"
  fs_require_cmd fightsafe

  fightsafe tapko-validate-annotations --annotations "$TAPKO_ANNOTATIONS"
  fightsafe tapko-detect \
    --source "$TAPKO_VIDEO" \
    --output-dir "$DETECT_DIR" \
    --fps 30 \
    --pose-backend mediapipe
  fightsafe tapko-evaluate \
    --annotations "$TAPKO_ANNOTATIONS" \
    --predictions "$DETECT_DIR/tapko_predictions.json" \
    --output-dir "$EVAL_DIR" \
    --tolerance-seconds 0.5 \
    --match-mode family
fi

# If evaluate was skipped (reference mode), re-run evaluate from reference predictions when possible
if [[ ! -f "$EVAL_DIR/tapko_results.csv" && -f "$DETECT_DIR/tapko_predictions.json" ]]; then
  fightsafe tapko-evaluate \
    --annotations "$TAPKO_ANNOTATIONS" \
    --predictions "$DETECT_DIR/tapko_predictions.json" \
    --output-dir "$EVAL_DIR" \
    --tolerance-seconds 0.5 \
    --match-mode family
fi

"$PYTHON" scripts/export_sinica2026_tables.py \
  --results-csv "$EVAL_DIR/tapko_results.csv" \
  --predictions-json "$DETECT_DIR/tapko_predictions.json" \
  --output-dir "$SINICA_REPRO_DIR/tables" \
  --install \
  --sinica-dir "$SINICA_DIR"

cp -f "$EVAL_DIR/tapko_results.csv" "$SINICA_REPRO_DIR/" 2>/dev/null || true
cp -f "$EVAL_DIR/tapko_error_analysis.md" "$SINICA_REPRO_DIR/" 2>/dev/null || true

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$SINICA_DIR/main.tex" ]]; then
  (cd "$SINICA_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  cp -f "$SINICA_DIR/main.pdf" "$SINICA_REPRO_DIR/main.pdf" 2>/dev/null || true
  echo "PDF: $SINICA_DIR/main.pdf"
fi

"$PYTHON" scripts/verify_paper_outputs.py --paper sinica
echo "Done."
