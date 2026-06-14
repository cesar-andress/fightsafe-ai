#!/usr/bin/env bash
# Reproduce iswa2026 TapKO workflow demonstration (jedi_submissions pilot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

DETECT_DIR="${ISWA_DETECT_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions}"
EVAL_DIR="${ISWA_EVAL_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions_eval}"
REFERENCE_DIR="$REPRO_DATA_ROOT/iswa2026/reference"
USE_REFERENCE="${REPRO_USE_REFERENCE:-0}"

echo "== FightSafe AI: reproduce iswa2026 TapKO pilot =="
echo "Manuscript dir: $ISWA_DIR"
echo "Repro outputs: $ISWA_REPRO_DIR"

fs_require_file "$TAPKO_ANNOTATIONS"
fs_mkdir_outputs
mkdir -p "$DETECT_DIR" "$EVAL_DIR" "$ISWA_REPRO_DIR/tables"

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

"$PYTHON" scripts/export_iswa2026_tables.py \
  --results-csv "$EVAL_DIR/tapko_results.csv" \
  --predictions-json "$DETECT_DIR/tapko_predictions.json" \
  --output-dir "$ISWA_REPRO_DIR/tables" \
  --install \
  --iswa-dir "$ISWA_DIR"

cp -f "$EVAL_DIR/tapko_results.csv" "$ISWA_REPRO_DIR/" 2>/dev/null || true
cp -f "$EVAL_DIR/tapko_error_analysis.md" "$ISWA_REPRO_DIR/" 2>/dev/null || true

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$ISWA_DIR/main.tex" ]]; then
  (cd "$ISWA_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  cp -f "$ISWA_DIR/main.pdf" "$ISWA_REPRO_DIR/main.pdf" 2>/dev/null || true
  echo "PDF: $ISWA_DIR/main.pdf"
fi

"$PYTHON" scripts/verify_paper_outputs.py --paper iswa
echo "Done."
