#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

state_is_active || exit 0

if [[ ! -f "$(verdict_file)" ]]; then
    stop_block_message "Missing reviewer verdict file: $(verdict_file)"
    exit 2
fi

if ! jq -e '.' "$(verdict_file)" >/dev/null 2>&1; then
    stop_block_message "Reviewer verdict is not valid JSON."
    exit 2
fi

READY="$(jq -r '.ready // false' "$(verdict_file)")"
if [[ "$READY" != "true" ]]; then
    SUMMARY="$(jq -r '.summary // "Reviewer has not approved the work yet."' "$(verdict_file)")"
    NEXT_ACTION="$(jq -r '.next_action // ""' "$(verdict_file)")"
    MISSING="$(jq -r '(.missing // []) | join(", ")' "$(verdict_file)")"
    REASON=$'Reviewer says not ready.\n'"summary: $SUMMARY"
    if [[ -n "$MISSING" ]]; then
        REASON+=$'\n'"missing: $MISSING"
    fi
    if [[ -n "$NEXT_ACTION" ]]; then
        REASON+=$'\n'"next_action: $NEXT_ACTION"
    fi
    stop_block_message "$REASON"
    exit 2
fi

exit 0
