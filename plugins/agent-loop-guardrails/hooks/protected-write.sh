#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

HOOK_INPUT="$(cat)"
validate_hook_input "$HOOK_INPUT" || exit 1

[[ "$(tool_name "$HOOK_INPUT")" == "Write" ]] || exit 0
state_is_active || exit 0

FILE_PATH="$(tool_field "$HOOK_INPUT" "file_path")"
[[ -n "$FILE_PATH" ]] || exit 1

if is_protected_path "$FILE_PATH"; then
    protected_path_message
    exit 2
fi

exit 0
