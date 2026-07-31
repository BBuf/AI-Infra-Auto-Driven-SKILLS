# SGLang Diffusion Single-Agent Flow Implementation Plan

> **For agentic workers:** Execute this plan inline in the current root-agent
> conversation. Do not spawn subagents, nested Codex/Claude sessions, or any
> other AI subprocess.

**Goal:** Replace the Master/executor hierarchy with a serial work-order
protocol owned by the current interactive root agent, while retaining the
Python controller as a deterministic evidence and state-management tool.

**Architecture:** The root agent chooses and implements one hypothesis at a
time. The CLI locks sources, records the baseline/profile, creates one exclusive
worktree, verifies submitted artifacts, integrates only verified
latency-positive candidates, and reports the next legal action. The CLI and
watchdog may launch benchmark and validation programs but may never launch an
AI command.

**Tech stack:** Python 3.11+, Pydantic v2, SQLite, Typer, pytest, Ruff, Git
worktrees.

---

## Task 1: Add the interactive single-agent campaign contract

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/request.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_config.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_request.py`

### Step 1: Write failing model and normalization tests

Add tests asserting that:

```python
assert goal.execution_mode == "interactive_single_agent"
assert not hasattr(goal, "agent")
assert CampaignStatus.AWAITING_AGENT.value == "AWAITING_AGENT"
```

Cover both a request without `agent` and a legacy request containing:

```yaml
agent:
  command: [codex, exec]
  model: gpt-test
```

The legacy field must parse for compatibility but must be omitted from the
normalized `CampaignGoal`.

Run:

```bash
cd sgl-engine-sglang-diffusion
pytest -q tests/test_config.py tests/test_request.py
```

Expected: FAIL because the status and execution-mode contract do not exist.

### Step 2: Implement the contract

In `models.py`:

```python
class CampaignStatus(StrEnum):
    ...
    AWAITING_AGENT = "AWAITING_AGENT"


class CampaignGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    execution_mode: Literal["interactive_single_agent"] = (
        "interactive_single_agent"
    )
    # Existing non-agent campaign fields remain unchanged.
```

Remove `agent` from `CampaignGoal`. Keep `AgentSpec` only in the request
compatibility layer until the request migration is complete.

In `request.py`, make `LaunchRequest.agent` optional and deprecated:

```python
agent: AgentSpec | None = Field(
    default=None,
    deprecated="ignored; campaigns use the current interactive root agent",
)
```

Normalize every accepted request to `schema_version=2` and
`execution_mode="interactive_single_agent"` without copying the legacy agent
command into the frozen goal.

### Step 3: Run focused tests

```bash
pytest -q tests/test_config.py tests/test_request.py
```

Expected: PASS.

### Step 4: Commit

```bash
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py \
  sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/request.py \
  sgl-engine-sglang-diffusion/tests/test_config.py \
  sgl-engine-sglang-diffusion/tests/test_request.py
git commit -m "refactor: define interactive single-agent campaigns"
```

## Task 2: Make `AWAITING_AGENT` a safe controller boundary

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/state.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/controller.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/watchdog.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_state.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_controller.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_watchdog.py`

### Step 1: Write failing transition and stop-boundary tests

Cover this path:

```text
NEW -> BASELINE_LOCKED -> PROFILED -> AWAITING_AGENT
AWAITING_AGENT -> SEARCHING -> INTEGRATING -> FINAL_VERIFYING
FINAL_VERIFYING -> AWAITING_AGENT
```

Assert that:

- `run_until_wait()` returns immediately at `AWAITING_AGENT`;
- the watchdog exits successfully at `AWAITING_AGENT`;
- the controller never increments an epoch or creates search work by itself;
- recoverable states resume only to their recorded prior state.

Run:

```bash
pytest -q tests/test_state.py tests/test_controller.py tests/test_watchdog.py
```

Expected: FAIL on missing transitions and old executor hooks.

### Step 2: Update the state graph

Use these forward transitions:

```python
CampaignStatus.PROFILED: frozenset({CampaignStatus.AWAITING_AGENT}),
CampaignStatus.AWAITING_AGENT: frozenset(
    {
        CampaignStatus.SEARCHING,
        CampaignStatus.SEARCH_SPACE_EXHAUSTED,
    }
),
CampaignStatus.SEARCHING: frozenset(
    {
        CampaignStatus.AWAITING_AGENT,
        CampaignStatus.INTEGRATING,
        CampaignStatus.SEARCH_SPACE_EXHAUSTED,
    }
),
CampaignStatus.INTEGRATING: frozenset(
    {
        CampaignStatus.AWAITING_AGENT,
        CampaignStatus.FINAL_VERIFYING,
    }
),
CampaignStatus.FINAL_VERIFYING: frozenset(
    {
        CampaignStatus.AWAITING_AGENT,
        CampaignStatus.TARGET_REACHED,
        CampaignStatus.SEARCH_SPACE_EXHAUSTED,
        CampaignStatus.UNREACHABLE_CERTIFIED,
    }
),
```

`AWAITING_AGENT` is active but is also an explicit controller wait boundary.

### Step 3: Replace executor hooks with deterministic hooks

Change `CampaignHooks` to expose:

```python
def enter_agent_wait(self, epoch: int) -> StepResult: ...
def verify_submitted_delivery(self, epoch: int) -> StepResult: ...
```

Map `PROFILED` to `enter_agent_wait`. Map `SEARCHING` to
`verify_submitted_delivery`. Remove repeated generic failure-signature campaign
termination from `_normalize_result`; a rejected candidate returns to
`AWAITING_AGENT`.

### Step 4: Stop the watchdog at the interactive boundary

Treat `AWAITING_AGENT` like a successful yielded state in `run_forever()`.
Do not poll, sleep, or restart the controller there.

### Step 5: Run focused tests and commit

```bash
pytest -q tests/test_state.py tests/test_controller.py tests/test_watchdog.py
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/{state.py,controller.py,watchdog.py} \
  sgl-engine-sglang-diffusion/tests/{test_state.py,test_controller.py,test_watchdog.py}
git commit -m "refactor: add the interactive agent wait boundary"
```

## Task 3: Add the exclusive work-order protocol

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/work_orders.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/state.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_work_orders.py`

### Step 1: Write failing protocol tests

Test these cases:

1. `claim("torch_compile")` creates one worktree and one
   `AGENT-WORK.json`, moves `AWAITING_AGENT -> SEARCHING`, and increments the
   epoch exactly once.
2. A second claim while work is active fails without creating a worktree.
3. An unrouted or closed technique fails closed.
4. `skip(..., classification="unsupported")` records a disposition and does
   not consume a scientific round.
5. `skip(..., classification="blocked")` keeps the technique recoverable.
6. Only a submitted, measured candidate increments a technique round.
7. Exhaustion is emitted only when every suggestion is explicitly closed or
   has consumed its scientific budget.

Run:

```bash
pytest -q tests/test_work_orders.py
```

Expected: FAIL because the manager and schemas do not exist.

### Step 2: Add strict durable schemas

Define:

```python
class AgentWorkOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    campaign_id: str
    epoch: int
    technique: str
    worktree: Path
    delivery_path: Path
    source_lock_sha256: str
    baseline_sha256: str
    profile_sha256: str
    technique_contract_sha256: str
    scientific_rounds_used: int
    scientific_rounds_remaining: int


class TechniqueDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technique: str
    classification: Literal["unsupported", "no_gain", "blocked"]
    reason: str
    closed: bool
```

### Step 3: Implement `WorkOrderManager`

The manager must:

- acquire one campaign-scoped lock before inspection or mutation;
- derive every path beneath the campaign directory;
- create a Git worktree from the locked SGLang source commit;
- atomically write `AGENT-WORK.json`;
- record `work_claimed`, `candidate_submitted`, `work_rejected`,
  `work_accepted`, and `technique_skipped` events;
- derive round use only from `candidate_submitted`;
- keep one active work order;
- delete no user worktree or delivery automatically.

Return a machine-readable `work()` payload with `legal_actions`.

### Step 4: Run tests and commit

```bash
pytest -q tests/test_work_orders.py
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/{models.py,state.py,work_orders.py} \
  sgl-engine-sglang-diffusion/tests/test_work_orders.py
git commit -m "feat: add serial root-agent work orders"
```

## Task 4: Replace AI review subprocesses with bound same-agent evidence

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/review.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_review.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

### Step 1: Write failing provenance tests

Require an `AGENT-REVIEW.json` beside the delivery:

```json
{
  "schema_version": 1,
  "producer": "interactive-root-agent",
  "campaign_id": "campaign",
  "epoch": 1,
  "technique": "torch_compile",
  "baseline_commit": "<sha>",
  "candidate_commit": "<sha>",
  "diff_sha256": "<sha256>",
  "method_argument_sha256": "<sha256>",
  "method_equivalent": true,
  "visual_review": {
    "required": false,
    "accepted": true,
    "prompt_count": 0,
    "artifact_sha256": []
  },
  "findings": []
}
```

Test rejection of a stale diff hash, wrong campaign/epoch/technique, missing
method argument, a quality-changing candidate without five reviewed prompts,
and any producer that claims an independent Master/reviewer.

### Step 2: Implement deterministic review validation

`SameAgentReviewValidator` recomputes:

- the baseline-to-candidate Git diff digest;
- the frozen method-argument digest;
- the referenced visual-artifact digests;
- campaign, epoch, technique, source commit, and candidate commit bindings.

It validates the root agent's explicit verdict but does not represent it as an
independent review.

### Step 3: Refactor verification

Remove the runtime AI invocations from method and visual assessment.
`DeliveryVerifier` must:

- call `SameAgentReviewValidator`;
- recompute latency, speedup, backend, fallback, engagement, and run hashes;
- recompute deterministic LPIPS/structural metrics for quality-changing work;
- compare the root-agent visual verdict with its bound artifacts;
- emit terminology `same_agent_method_review` and
  `same_agent_visual_review`.

The verifier must never accept an agent command or call `subprocess` with
Codex/Claude.

### Step 4: Run tests and commit

```bash
pytest -q tests/test_review.py tests/test_verifier.py
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/{review.py,verifier.py,runtime.py} \
  sgl-engine-sglang-diffusion/tests/{test_review.py,test_verifier.py}
git commit -m "refactor: bind verification to same-agent reviews"
```

## Task 5: Make runtime search serial and integrate a selected subset

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/router.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_integrator.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_router.py`

### Step 1: Write failing serial-runtime tests

Assert that:

- setup ends at `AWAITING_AGENT` without any agent executable in the process
  log;
- one submitted delivery is verified at a time;
- a rejected candidate returns to `AWAITING_AGENT`;
- an infrastructure failure is classified separately and consumes no round;
- a repeated candidate failure rejects only that hypothesis;
- one verified latency-positive technique can enter integration without all
  suggested techniques delivering;
- memory-only or latency-regressing frontier points are not added to the
  latency integration subset;
- unsupported hardware suggestions are filtered before claim.

### Step 2: Remove executor orchestration from `FileCampaignHooks`

Delete construction and use of:

- `AgentRunner`;
- `ExecutorManager`;
- executor handles, leases, prompts, receipts, and resume prompts;
- independent Master and visual-review runners.

Implement:

```python
def enter_agent_wait(self, epoch: int) -> StepResult:
    return StepResult(CampaignStatus.AWAITING_AGENT)


def verify_submitted_delivery(self, epoch: int) -> StepResult:
    # Resolve the sole active work order and durable delivery.
    # Return AWAITING_AGENT on a reviewed rejection.
    # Return INTEGRATING only for an accepted latency-positive point.
```

### Step 3: Select the integration subset

Before calling `IntegrationManager`, filter verified candidates to:

```python
candidate.verified_speedup > 1.0 and candidate.correctness_accepted
```

Use compatibility validation only across that selected set. Do not require
`set(verified_techniques) == set(routed_techniques)`.

After a below-target final verification, transition to `AWAITING_AGENT` so the
root agent chooses the next hypothesis.

### Step 4: Filter known hardware mismatches

Route records become suggestions. Filter NVFP4 on pre-Blackwell hardware and
other capability mismatches discoverable from the locked machine manifest.
Do not automatically route every quality-changing method merely because a
quality gate exists.

### Step 5: Run tests and commit

```bash
pytest -q tests/test_runtime_e2e.py tests/test_integrator.py tests/test_router.py
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/{runtime.py,integrator.py,router.py} \
  sgl-engine-sglang-diffusion/tests/{test_runtime_e2e.py,test_integrator.py,test_router.py}
git commit -m "refactor: run diffusion search serially"
```

## Task 6: Expose `work`, `claim`, `submit`, and `skip` commands

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/cli.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/launcher.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_cli.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_launcher.py`

### Step 1: Write failing CLI tests

Cover:

```bash
sgl-diffusion-engine work --campaign <path> --json
sgl-diffusion-engine claim --campaign <path> --technique torch_compile
sgl-diffusion-engine submit --campaign <path> --delivery <path>
sgl-diffusion-engine skip --campaign <path> --technique quantization \
  --classification unsupported --reason "H100 has no NVFP4"
```

Every JSON response must contain `campaign_id`, `status`, `legal_actions`, and
the relevant work-order or verification result. Invalid state/action
combinations must exit nonzero without partial mutation.

### Step 2: Add command handlers

`launch` keeps deterministic watchdog setup. It reports the future `work`
command in its payload. `claim`, `submit`, and `skip` use `WorkOrderManager`
under the campaign lock. `submit` validates that the delivery belongs to the
active work order before asking the controller to resume deterministic
verification.

Do not add an `--agent`, `--model`, or nested-session option.

### Step 3: Run tests and commit

```bash
pytest -q tests/test_cli.py tests/test_launcher.py
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/{cli.py,launcher.py,runtime.py} \
  sgl-engine-sglang-diffusion/tests/{test_cli.py,test_launcher.py}
git commit -m "feat: expose the root-agent campaign protocol"
```

## Task 7: Remove dormant multi-agent execution code

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/redaction.py`
- Modify: all imports of redaction helpers
- Delete: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Delete: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Delete: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/prompts/master.md`
- Delete: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/prompts/executor.md`
- Delete: `sgl-engine-sglang-diffusion/tests/test_agents.py`
- Delete: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_redaction.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_no_ai_subprocess.py`

### Step 1: Preserve only non-agent redaction utilities

Move `redact_argv`, `redact_environment`, and their constants to
`redaction.py`. Update launcher, verifier, and runtime imports.

### Step 2: Add a source-level execution-boundary test

Walk the installed package and assert that campaign execution code contains no:

```text
AgentRunner
ExecutorManager
codex exec
claude
subprocess.Popen(...agent...)
```

The test may allow documentation describing prohibited commands, but not
executable Python paths.

### Step 3: Delete orchestration-only modules and prompts

Remove the files only after `rg` confirms no non-test imports remain:

```bash
rg -n "agents|orchestration|master.md|executor.md" \
  sgl-engine-sglang-diffusion/src \
  sgl-engine-sglang-diffusion/tests
```

### Step 4: Run tests and commit

```bash
pytest -q tests/test_redaction.py tests/test_no_ai_subprocess.py
git add -A sgl-engine-sglang-diffusion
git commit -m "refactor: remove nested agent execution"
```

## Task 8: Rewrite progress and skill instructions around one owner

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_progress.py`
- Modify: `sglang-diffusion-auto-optimize/SKILL.md`
- Modify: relevant files under `sglang-diffusion-auto-optimize/references/`
- Modify: repository README/catalog entries that describe the old hierarchy

### Step 1: Write failing progress tests

Assert that progress exposes:

- `execution_mode: interactive_single_agent`;
- current work order and legal next command;
- submitted scientific rounds rather than executor attempts;
- explicit dispositions and failure classifications;
- no Master/executor/reviewer role or per-agent token projection.

### Step 2: Update the renderer

Render `AWAITING_AGENT` as:

```text
waiting for the current root agent to claim or complete one work order
```

Replace executor lanes with technique suggestions, dispositions, active
work-order state, and verified frontier entries.

### Step 3: Rewrite the skill workflow

The skill must explicitly say:

- use only the current conversation's root agent;
- never call `spawn_agent`;
- never launch `codex exec`, Claude, or an AI reviewer;
- claim and finish one technique at a time;
- run one candidate GPU job at a time;
- write `AGENT-REVIEW.json` before `submit`;
- classify infrastructure separately from scientific results;
- integrate only verified latency-positive candidates;
- continue until the target or a reviewed terminal boundary.

Remove all instructions for Master prompts, executor prompts, executor polling,
multi-lane fan-out, and independent AI review claims.

### Step 4: Scan documentation and run focused tests

```bash
rg -n "Master agent|executor agent|sub-agent|independent AI|codex exec|Claude" \
  sglang-diffusion-auto-optimize \
  sgl-engine-sglang-diffusion \
  README.md
pytest -q tests/test_progress.py
```

Every remaining match must either describe the prohibition or historical
migration, not an executable workflow.

### Step 5: Commit

```bash
git add sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py \
  sgl-engine-sglang-diffusion/tests/test_progress.py \
  sglang-diffusion-auto-optimize README.md
git commit -m "docs: teach the diffusion flow to use one root agent"
```

## Task 9: Migrate compatibility and run the full validation matrix

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Modify: package schemas/examples as discovered by validation

### Step 1: Add fail-closed legacy campaign handling

An old campaign whose frozen goal or state contains executor handles must not
silently resume. Return a structured migration error:

```text
legacy multi-agent campaign cannot resume in interactive_single_agent mode;
create a new campaign from the frozen request/source lock
```

A legacy launch request may be normalized into a new v2 single-agent campaign,
but no saved executor PID, lease, or prompt may be adopted.

### Step 2: Run unit and integration tests

```bash
cd sgl-engine-sglang-diffusion
pytest -q
ruff check .
ruff format --check .
python -m build
```

Expected: all pass.

### Step 3: Run repository validation and safety scans

From the repository root:

```bash
git diff --check origin/main...
rg -n "AgentRunner|ExecutorManager|start_search_epoch|poll_and_verify_executors" \
  sgl-engine-sglang-diffusion/src
rg -n "codex exec|claude" sgl-engine-sglang-diffusion/src
```

Expected: no execution-path matches.

### Step 4: Exercise the CLI smoke path

Use a fixture/fake benchmark request to verify:

```text
launch -> AWAITING_AGENT
work -> legal claim
claim -> SEARCHING
submit rejected -> AWAITING_AGENT
claim -> SEARCHING
submit accepted -> INTEGRATING -> FINAL_VERIFYING
final below target -> AWAITING_AGENT
```

Record the smoke output in the PR test plan.

### Step 5: Commit any final migration fixes

```bash
git add -A
git commit -m "test: validate the single-agent diffusion flow"
```

## Task 10: Review, publish, and open a draft PR

### Step 1: Review the complete branch

```bash
git status --short
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
```

Check specifically for:

- accidental unrelated edits;
- stale Master/executor terminology;
- any reachable AI subprocess;
- parallel GPU candidate paths;
- automatic campaign exhaustion from generic failures;
- all-route integration barriers;
- secrets, local paths, or generated campaign artifacts.

### Step 2: Push the explicit branch

```bash
git push -u origin agent/single-agent-diffusion-flow
```

### Step 3: Open a draft PR

Create a draft PR into `main` with:

- the motivation and the Sol comparison finding;
- the single-agent execution contract;
- Controller responsibilities and non-responsibilities;
- compatibility/migration behavior;
- exact test commands and results;
- explicit statement that no AI subprocess or parallel candidate lane remains.

### Step 4: Inspect the published PR

Verify the base, head, title, draft state, changed files, and checks. Return the
PR URL and any outstanding external check status to the user.
