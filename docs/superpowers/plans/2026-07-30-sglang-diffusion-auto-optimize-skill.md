# SGLang Diffusion Auto-Optimize Skill Implementation Plan

> **For Codex:** Implement task-by-task with tests first where practical. Do not
> weaken the Sol correctness contract to simplify the launcher or progress UI.

**Goal:** Add an installable conversational Skill, one-shot durable launch,
user-command-derived baseline execution, exact token telemetry, and persistent
technique/end-to-end progress reporting to `sgl-engine-sglang-diffusion`.

**Architecture:** The Skill resolves remote machine/container policy and invokes
the Python controller on that machine. A new strict launch request and benchmark
command template feed the existing campaign goal and driver. Telemetry and
progress are projections of durable agent streams, SQLite events, and verified
deliveries.

**Tech Stack:** Python 3.11, Pydantic v2, argparse, SQLite, JSONL/YAML, pytest,
Codex/Claude-compatible `SKILL.md`.

---

### Task 1: Add launch-request and command-template models

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/request.py`
- Create: `sgl-engine-sglang-diffusion/schemas/launch-request.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_request.py`

Implement strict models for launch inputs, normalized argv/env/cwd, parser
adapter, token budget, and idempotency key. Parse leading environment
assignments without a shell; reject operators, substitutions, expansions,
redirects, and ambiguous workload flags. Extract the native SGLang Diffusion
workload and write a frozen command-template receipt.

### Task 2: Make the benchmark driver derive every run from the frozen command

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/driver.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/baseline.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_driver.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`

Load `BASELINE-COMMAND.json` when present, relocate the checkout and output
paths, preserve user flags, and inject activation/profile arguments. Continue
supporting legacy goal-only campaigns. Verify every authoritative command
receipt contains the template digest.

### Task 3: Add exact Agent token telemetry

**Files:**
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/telemetry.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_telemetry.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`

Recognize Codex commands, add `--json` exactly once, retain raw JSONL output,
normalize terminal usage events, and append deduplicated records. Pass role,
epoch, technique, and invocation ID through every executor and Master launch.
Represent unsupported runtimes explicitly as unavailable.

### Task 4: Build the progress projection and terminal renderer

**Files:**
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/cli.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_progress.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_cli.py`

Project campaign state, attempts, verified isolated speedups, integrated stack
speedup, current work, token totals, and artifact paths. Atomically write
`PROGRESS.json`. Implement static, watch, and JSON output; keep bars bounded and
distinguish search consumption from performance progress.

### Task 5: Add one-shot idempotent launch and detached ownership

**Files:**
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/launcher.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/cli.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/watchdog.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/config.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_launcher.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_watchdog.py`

Convert a launch request into the frozen goal and command template, index the
request idempotency key, and start a campaign-owned detached watchdog. Return
campaign/status/progress commands and PIDs in a durable launch receipt. Do not
embed SSH behavior in the package.

### Task 6: Create and validate the conversational Skill

**Files:**
- Create: `skills/sglang-diffusion-auto-optimize/SKILL.md`
- Create: `skills/sglang-diffusion-auto-optimize/agents/openai.yaml`
- Create: `skills/sglang-diffusion-auto-optimize/references/request-template.yaml`
- Create: `skills/sglang-diffusion-auto-optimize/references/progress-contract.md`
- Create: `skills/sglang-diffusion-auto-optimize/references/remote-ownership.md`
- Modify: `README.md`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Test: `tests/test_plugin_metadata.py`

Use the skill-creator structure and validate it with `quick_validate.py`. Make
natural language the primary path, document machine-skill selection and remote
bootstrap, and keep direct CLI use as a troubleshooting/reproducibility path.

### Task 7: Complete end-to-end and regression verification

**Files:**
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Modify: `sgl-engine-sglang-diffusion/README.md`

Exercise a fake detached launch through baseline, token records, isolated
candidate metrics, integrated speedup, restart, and terminal packaging. Then run
all package and root suites, lint/format checks, skill validation, and a fresh
virtual-environment install. Inspect the final diff for secret/private-host
leaks and update the existing draft PR.
