set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_FILE="${PLAN_FILE:-$ROOT_DIR/plan.md}"
REVIEW_FILE="${REVIEW_FILE:-$ROOT_DIR/review.md}"
PROGRESS_FILE="${PROGRESS_FILE:-$ROOT_DIR/progress.md}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/artifacts}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

run_builder() {
  echo "TODO: configure builder CLI" >&2
  return 1
}

run_reviewer() {
  printf '%s\n' '{"ready":false,"summary":"TODO: configure reviewer CLI","missing":["reviewer command not configured"],"next_action":"Configure run_reviewer()."}'
}

mkdir -p "$ARTIFACT_DIR"
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  attempt_dir="$ARTIFACT_DIR/attempt-$(printf '%02d' "$attempt")"
  mkdir -p "$attempt_dir"
  run_builder >"$attempt_dir/builder.log" 2>&1 || true
  run_reviewer >"$attempt_dir/reviewer.json"
  cat "$attempt_dir/reviewer.json"
  attempt=$((attempt + 1))
done
