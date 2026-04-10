---
name: agent-loop-governed
description: "Use when Codex should initialize or operate a guarded agent loop inside the current project. This skill assumes the Agent Loop Guardrails plugin is installed and uses `.agent-loop/plan.md`, `.agent-loop/review.md`, `.agent-loop/progress.md`, reviewer verdicts, and plugin hooks to prevent premature stop."
---

# Agent Loop Governed

## Overview

Use this skill when you want a task to run under plugin-enforced loop guardrails instead of relying on prompt discipline alone.
The plugin becomes active when the project contains an active `.agent-loop/state.json`.

## Main Flow

### 1. Initialize the loop

Run [../../scripts/init-agent-loop.sh](../../scripts/init-agent-loop.sh):

```bash
bash ../../scripts/init-agent-loop.sh --goal "replace me with the concrete task goal"
```

This creates:

- `.agent-loop/plan.md`
- `.agent-loop/review.md`
- `.agent-loop/progress.md`
- `.agent-loop/loop.sh`
- `.agent-loop/state.json`
- `.agent-loop/reviewer.latest.json`

### 2. Fill plan and review before asking the builder to run

The plugin is strongest when:

- `plan.md` contains exact scope, commands, thresholds, and stop conditions
- `review.md` contains strict acceptance checks and a JSON output contract

You can reuse the starter templates that ship with the plugin:

- [../../assets/templates/plan.md](../../assets/templates/plan.md)
- [../../assets/templates/review.md](../../assets/templates/review.md)
- [../../assets/templates/progress.md](../../assets/templates/progress.md)
- [../../assets/templates/loop.sh](../../assets/templates/loop.sh)

### 3. Run the task

The builder can read `.agent-loop/plan.md` and update `.agent-loop/progress.md`.
The reviewer should judge against `.agent-loop/review.md`.

### 4. Record the reviewer verdict through the plugin helper

Do not directly edit `.agent-loop/state.json`.
Record the reviewer verdict with [../../scripts/record-review-verdict.sh](../../scripts/record-review-verdict.sh):

```bash
echo '{"ready":false,"summary":"tests still fail","missing":["pytest -q"],"next_action":"fix the failing tests"}' \
  | bash ../../scripts/record-review-verdict.sh
```

When the reviewer returns `"ready": true`, the stop hook will allow Codex to finish.

### 5. Close the loop when done

Optionally mark the loop closed:

```bash
bash ../../scripts/close-agent-loop.sh --status complete
```

## What The Plugin Enforces

- direct `Write` or `Edit` operations to `.agent-loop/state.json` are blocked
- direct `Write` or `Edit` operations to `.agent-loop/reviewer.latest.json` are blocked
- shell commands that try to mutate those protected files are blocked unless they use the plugin helper scripts
- `Stop` is blocked until `.agent-loop/reviewer.latest.json` exists and says `"ready": true`

## Practical Advice

- keep the loop files under `.agent-loop/` so the hook surface stays narrow
- store artifacts under `.agent-loop/artifacts/`
- use bounded loops for engineering tasks
- let the reviewer own the acceptance threshold, not the builder
