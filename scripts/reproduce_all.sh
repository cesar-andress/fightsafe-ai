#!/usr/bin/env bash
# Best-effort reproduction of all three companion manuscripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$REPO_ROOT"

echo "== FightSafe AI: reproduce all companion experiments =="

failures=0
run_step() {
  local name="$1"
  shift
  echo ""
  echo "--- $name ---"
  if "$@"; then
    echo "OK: $name"
  else
    echo "FAILED: $name"
    failures=$((failures + 1))
  fi
}

# sinica first when video missing: reference mode for table sync
if [[ ! -f "$TAPKO_VIDEO" && "${REPRO_USE_REFERENCE:-1}" != "0" ]]; then
  export REPRO_USE_REFERENCE=1
fi

run_step "fusion2026" bash "$SCRIPT_DIR/reproduce_fusion2026.sh"
run_step "sinica2026" bash "$SCRIPT_DIR/reproduce_sinica2026.sh"
run_step "sports" bash "$SCRIPT_DIR/reproduce_sports.sh"

echo ""
if [[ "$failures" -eq 0 ]]; then
  echo "All reproduction steps completed."
  "$PYTHON" scripts/verify_paper_outputs.py --paper all
else
  echo "$failures step(s) failed — see docs/REPRODUCIBILITY.md"
  exit 1
fi
