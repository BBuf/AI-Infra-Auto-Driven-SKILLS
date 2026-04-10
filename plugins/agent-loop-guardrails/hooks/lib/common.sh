#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"

LOOP_DIR_NAME=".agent-loop"
STATE_FILE_NAME="state.json"
VERDICT_FILE_NAME="reviewer.latest.json"
PLAN_FILE_NAME="plan.md"
REVIEW_FILE_NAME="review.md"
PROGRESS_FILE_NAME="progress.md"

loop_dir() {
    echo "$PROJECT_ROOT/$LOOP_DIR_NAME"
}

state_file() {
    echo "$(loop_dir)/$STATE_FILE_NAME"
}

verdict_file() {
    echo "$(loop_dir)/$VERDICT_FILE_NAME"
}

plan_file() {
    echo "$(loop_dir)/$PLAN_FILE_NAME"
}

review_file() {
    echo "$(loop_dir)/$REVIEW_FILE_NAME"
}

progress_file() {
    echo "$(loop_dir)/$PROGRESS_FILE_NAME"
}

normalize_path() {
    local p="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$p" <<'PY'
import os, sys
print(os.path.realpath(os.path.expanduser(sys.argv[1])))
PY
    else
        echo "$p"
    fi
}

state_is_active() {
    local sf
    sf="$(state_file)"
    [[ -f "$sf" ]] || return 1
    jq -e '.status == "active"' "$sf" >/dev/null 2>&1
}

validate_hook_input() {
    local input="$1"
    jq -e '.' >/dev/null 2>&1 <<<"$input"
}

tool_name() {
    local input="$1"
    jq -r '.tool_name // ""' <<<"$input"
}

tool_field() {
    local input="$1"
    local field="$2"
    jq -r --arg field "$field" '.tool_input[$field] // ""' <<<"$input"
}

is_protected_path() {
    local candidate="$1"
    local resolved
    resolved="$(normalize_path "$candidate")"
    [[ "$resolved" == "$(normalize_path "$(state_file)")" ]] || [[ "$resolved" == "$(normalize_path "$(verdict_file)")" ]]
}

protected_path_message() {
    cat >&2 <<'EOF'
Agent Loop Guardrails blocked this change because it targets a protected loop file.
Use the plugin helper scripts instead:
- init-agent-loop.sh
- record-review-verdict.sh
- close-agent-loop.sh
EOF
}

stop_block_message() {
    local reason="$1"
    cat >&2 <<EOF
Agent Loop Guardrails blocked stop because the active loop is not ready.
$reason
Review the loop state under .agent-loop/ and record a reviewer verdict before stopping.
EOF
}
