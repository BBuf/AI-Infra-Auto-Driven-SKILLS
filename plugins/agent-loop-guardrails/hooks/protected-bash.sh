#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

HOOK_INPUT="$(cat)"
validate_hook_input "$HOOK_INPUT" || exit 1

[[ "$(tool_name "$HOOK_INPUT")" == "Bash" ]] || exit 0
state_is_active || exit 0

COMMAND="$(tool_field "$HOOK_INPUT" "command")"
[[ -n "$COMMAND" ]] || exit 1

ALLOWLIST_REGEX='(record-review-verdict\.sh|init-agent-loop\.sh|close-agent-loop\.sh)'
PROTECTED_REGEX='(\.agent-loop/(state\.json|reviewer\.latest\.json)|state\.json|reviewer\.latest\.json)'

if echo "$COMMAND" | grep -Eq "$PROTECTED_REGEX" && ! echo "$COMMAND" | grep -Eq "$ALLOWLIST_REGEX"; then
    protected_path_message
    exit 2
fi

if echo "$COMMAND" | grep -Eq '(^|[[:space:]])rm([[:space:]].*)?(\.agent-loop([[:space:]/]|$))'; then
    protected_path_message
    exit 2
fi

exit 0
