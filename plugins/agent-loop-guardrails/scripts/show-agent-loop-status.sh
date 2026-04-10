#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            PROJECT_ROOT="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

LOOP_DIR="$PROJECT_ROOT/.agent-loop"
STATE_FILE="$LOOP_DIR/state.json"
VERDICT_FILE="$LOOP_DIR/reviewer.latest.json"

[[ -f "$STATE_FILE" ]] || { echo "No active loop state at $STATE_FILE" >&2; exit 1; }

echo "=== state.json ==="
cat "$STATE_FILE"
echo
if [[ -f "$VERDICT_FILE" ]]; then
    echo "=== reviewer.latest.json ==="
    cat "$VERDICT_FILE"
else
    echo "Missing reviewer verdict: $VERDICT_FILE"
fi
