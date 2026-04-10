#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="$(pwd)"
GOAL=""
MAX_ATTEMPTS=6

usage() {
    cat <<'EOF'
Initialize a guarded agent loop in the target project.

Usage:
  init-agent-loop.sh --goal "task goal" [--root /path/to/project] [--max-attempts N]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --goal)
            GOAL="${2:-}"
            shift 2
            ;;
        --root)
            PROJECT_ROOT="${2:-}"
            shift 2
            ;;
        --max-attempts)
            MAX_ATTEMPTS="${2:-}"
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

[[ -n "$GOAL" ]] || { echo "--goal is required" >&2; exit 1; }

LOOP_DIR="$PROJECT_ROOT/.agent-loop"
mkdir -p "$LOOP_DIR/artifacts"

cp "$PLUGIN_ROOT/assets/templates/plan.md" "$LOOP_DIR/plan.md"
cp "$PLUGIN_ROOT/assets/templates/review.md" "$LOOP_DIR/review.md"
cp "$PLUGIN_ROOT/assets/templates/progress.md" "$LOOP_DIR/progress.md"
cp "$PLUGIN_ROOT/assets/templates/loop.sh" "$LOOP_DIR/loop.sh"
chmod +x "$LOOP_DIR/loop.sh"

python3 - "$LOOP_DIR" "$GOAL" "$MAX_ATTEMPTS" <<'PY'
import json, os, sys, datetime
loop_dir, goal, max_attempts = sys.argv[1], sys.argv[2], int(sys.argv[3])
state = {
    "version": 1,
    "status": "active",
    "goal": goal,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "max_attempts": max_attempts,
    "require_reviewer_ready": True,
    "plan_file": ".agent-loop/plan.md",
    "review_file": ".agent-loop/review.md",
    "progress_file": ".agent-loop/progress.md",
    "verdict_file": ".agent-loop/reviewer.latest.json"
}
with open(os.path.join(loop_dir, "state.json"), "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
    f.write("\n")
PY

cat > "$LOOP_DIR/reviewer.latest.json" <<'EOF'
{
  "ready": false,
  "summary": "Loop initialized. No reviewer verdict has been recorded yet.",
  "missing": [
    "First reviewer verdict"
  ],
  "next_action": "Fill plan.md and review.md, run the loop, then record a reviewer verdict."
}
EOF

echo "Initialized guarded loop at: $LOOP_DIR"
echo "Next steps:"
echo "1. Fill $LOOP_DIR/plan.md"
echo "2. Fill $LOOP_DIR/review.md"
echo "3. Run the builder/reviewer workflow"
echo "4. Record verdicts with $PLUGIN_ROOT/scripts/record-review-verdict.sh"
