#!/usr/bin/env bash
# Best-effort reproduction of all three companion manuscripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
    echo "FAILED (continuing): $name"
    failures=$((failures + 1))
  fi
}

run_step "fusion2026" bash scripts/reproduce_fusion2026.sh
run_step "sinica2026" bash scripts/reproduce_sinica2026.sh
run_step "sports" bash scripts/reproduce_sports.sh

echo ""
if [[ "$failures" -eq 0 ]]; then
  echo "All reproduction steps completed."
else
  echo "$failures step(s) failed — often due to missing large datasets. See data/README.md."
  exit 1
fi
