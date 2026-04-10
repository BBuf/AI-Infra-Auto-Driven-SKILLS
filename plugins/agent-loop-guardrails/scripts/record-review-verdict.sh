#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PROJECT_ROOT="$(pwd)"
JSON_INPUT=""

usage() {
    cat <<'EOF'
Record the latest reviewer verdict for the active agent loop.

Usage:
  echo '{"ready":false,"summary":"...","missing":["..."],"next_action":"..."}' | record-review-verdict.sh [--root /path]
  record-review-verdict.sh --root /path --json '{"ready":true,"summary":"done","missing":[],"next_action":""}'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            PROJECT_ROOT="${2:-}"
            shift 2
            ;;
        --json)
            JSON_INPUT="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$JSON_INPUT" ]]; then
    JSON_INPUT="$(cat)"
fi

[[ -n "$JSON_INPUT" ]] || { echo "No JSON verdict provided" >&2; exit 1; }

LOOP_DIR="$PROJECT_ROOT/.agent-loop"
STATE_FILE="$LOOP_DIR/state.json"
VERDICT_FILE="$LOOP_DIR/reviewer.latest.json"

[[ -f "$STATE_FILE" ]] || { echo "Missing state file: $STATE_FILE" >&2; exit 1; }

python3 - "$STATE_FILE" "$VERDICT_FILE" "$JSON_INPUT" <<'PY'
import json, sys
state_path, verdict_path, raw = sys.argv[1], sys.argv[2], sys.argv[3]

verdict = json.loads(raw)
required = ["ready", "summary", "missing", "next_action"]
missing = [k for k in required if k not in verdict]
if missing:
    raise SystemExit(f"Missing verdict keys: {', '.join(missing)}")
if not isinstance(verdict["ready"], bool):
    raise SystemExit("verdict.ready must be boolean")
if not isinstance(verdict["missing"], list):
    raise SystemExit("verdict.missing must be an array")

with open(verdict_path, "w", encoding="utf-8") as f:
    json.dump(verdict, f, indent=2)
    f.write("\n")

with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)
state["last_review_ready"] = verdict["ready"]
state["last_review_summary"] = verdict["summary"]
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
    f.write("\n")
PY

echo "Recorded reviewer verdict at $VERDICT_FILE"
