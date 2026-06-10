#!/usr/bin/env bash
# Reproduce eswa2026 TapKO workflow demonstration (jedi_submissions pilot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

DETECT_DIR="${ESWA_DETECT_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions}"
EVAL_DIR="${ESWA_EVAL_DIR:-$REPO_ROOT/outputs/tapko/jedi_submissions_eval}"
REFERENCE_DIR="$REPRO_DATA_ROOT/eswa2026/reference"
USE_REFERENCE="${REPRO_USE_REFERENCE:-0}"

echo "== FightSafe AI: reproduce eswa2026 TapKO pilot =="
echo "Manuscript dir: $ESWA_DIR"
echo "Repro outputs: $ESWA_REPRO_DIR"

fs_require_file "$TAPKO_ANNOTATIONS"
fs_mkdir_outputs
mkdir -p "$DETECT_DIR" "$EVAL_DIR" "$ESWA_REPRO_DIR/tables"

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

"$PYTHON" scripts/export_eswa2026_tables.py \
  --results-csv "$EVAL_DIR/tapko_results.csv" \
  --predictions-json "$DETECT_DIR/tapko_predictions.json" \
  --output-dir "$ESWA_REPRO_DIR/tables" \
  --install \
  --eswa-dir "$ESWA_DIR"

cp -f "$EVAL_DIR/tapko_results.csv" "$ESWA_REPRO_DIR/" 2>/dev/null || true
cp -f "$EVAL_DIR/tapko_error_analysis.md" "$ESWA_REPRO_DIR/" 2>/dev/null || true

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$ESWA_DIR/main.tex" ]]; then
  (cd "$ESWA_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  cp -f "$ESWA_DIR/main.pdf" "$ESWA_REPRO_DIR/main.pdf" 2>/dev/null || true
  echo "PDF: $ESWA_DIR/main.pdf"
fi

"$PYTHON" scripts/verify_paper_outputs.py --paper eswa
echo "Done."
