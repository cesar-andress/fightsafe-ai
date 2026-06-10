#!/usr/bin/env bash
# Reproduce sinica2026 TapKO workflow demonstration (jedi_submissions pilot).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINICA_DIR="${SINICA_DIR:-$ROOT/../sinica2026}"
VIDEO="${TAPKO_VIDEO:-$ROOT/data/tapko/videos/jedi_submissions.mp4}"
ANNOTATIONS="$ROOT/data/tapko/annotations/jedi_submissions.json"
DETECT_DIR="$ROOT/outputs/tapko/jedi_submissions"
EVAL_DIR="$ROOT/outputs/tapko/jedi_submissions_eval"

cd "$ROOT"

echo "== FightSafe AI: reproduce sinica2026 TapKO pilot =="

if [[ ! -f "$ANNOTATIONS" ]]; then
  echo "ERROR: Missing annotations: $ANNOTATIONS"
  exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: Missing video: $VIDEO"
  echo "Download per data/README.md (yt-dlp) or set TAPKO_VIDEO=/path/to/clip.mp4"
  exit 1
fi

fightsafe tapko-validate-annotations --annotations "$ANNOTATIONS"

fightsafe tapko-detect \
  --source "$VIDEO" \
  --output-dir "$DETECT_DIR" \
  --fps 30 \
  --pose-backend mediapipe

fightsafe tapko-evaluate \
  --annotations "$ANNOTATIONS" \
  --predictions "$DETECT_DIR/tapko_predictions.json" \
  --output-dir "$EVAL_DIR" \
  --tolerance-seconds 0.5 \
  --match-mode family

echo "Evaluator outputs: $EVAL_DIR"
echo "To compile sinica2026 PDF:"
echo "  cd $SINICA_DIR && pdflatex main && bibtex main && pdflatex main && pdflatex main"

if command -v pdflatex >/dev/null 2>&1 && [[ -f "$SINICA_DIR/main.tex" ]]; then
  (cd "$SINICA_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
  echo "PDF: $SINICA_DIR/main.pdf"
fi

echo "Done."
