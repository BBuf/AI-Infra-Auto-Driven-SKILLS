#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(pwd)"
NEW_STATUS="closed"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            PROJECT_ROOT="${2:-}"
            shift 2
            ;;
        --status)
            NEW_STATUS="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

STATE_FILE="$PROJECT_ROOT/.agent-loop/state.json"
[[ -f "$STATE_FILE" ]] || { echo "Missing state file: $STATE_FILE" >&2; exit 1; }

python3 - "$STATE_FILE" "$NEW_STATUS" <<'PY'
import json, sys, datetime
state_path, new_status = sys.argv[1], sys.argv[2]
with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)
state["status"] = new_status
state["closed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
    f.write("\n")
PY

echo "Updated loop status to: $NEW_STATUS"
