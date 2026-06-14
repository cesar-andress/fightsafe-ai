# Shared path and environment defaults for reproduction scripts.
# Source from other scripts:  source "$(dirname "$0")/lib/common.sh"

fs_require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd" >&2
    return 1
  fi
}

fs_require_file() {
  local path="$1"
  local msg="${2:-Missing required file: $path}"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $msg" >&2
    return 1
  fi
}

fs_mkdir_outputs() {
  mkdir -p "$FUSION_REPRO_DIR" "$ISWA_REPRO_DIR" "$SPORTS_REPRO_DIR"
}

if [[ -n "${FIGHTSAFE_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
export FIGHTSAFE_COMMON_LOADED=1

# Repository root (fightsafe-ai package root)
if [[ -z "${REPO_ROOT:-}" ]]; then
  _common_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$_common_lib/../.." && pwd)"
fi
export REPO_ROOT

# Companion manuscript directories (monorepo layout)
export FUSION_DIR="${FUSION_DIR:-$REPO_ROOT/../fusion2026}"
export ISWA_DIR="${ISWA_DIR:-$REPO_ROOT/../iswa2026}"
export SPORTS_DIR="${SPORTS_DIR:-$REPO_ROOT/../sports}"

# Reproducibility output roots (generated artefacts; gitignored)
export REPRO_OUTPUT_ROOT="${REPRO_OUTPUT_ROOT:-$REPO_ROOT/outputs/repro}"
export FUSION_REPRO_DIR="${FUSION_REPRO_DIR:-$REPRO_OUTPUT_ROOT/fusion2026}"
export ISWA_REPRO_DIR="${ISWA_REPRO_DIR:-$REPRO_OUTPUT_ROOT/iswa2026}"
export SPORTS_REPRO_DIR="${SPORTS_REPRO_DIR:-$REPRO_OUTPUT_ROOT/sports}"

# Reference snapshots shipped in Git (verification without large media)
export REPRO_DATA_ROOT="${REPRO_DATA_ROOT:-$REPO_ROOT/data/repro}"

export PYTHON="${PYTHON:-python3.12}"
export TAPKO_VIDEO="${TAPKO_VIDEO:-$REPO_ROOT/data/tapko/videos/jedi_submissions.mp4}"
export TAPKO_ANNOTATIONS="${TAPKO_ANNOTATIONS:-$REPO_ROOT/data/tapko/annotations/jedi_submissions.json}"
