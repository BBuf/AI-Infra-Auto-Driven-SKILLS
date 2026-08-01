# Diffusion Optimization Loop Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make unattended diffusion optimization resume one Codex session, preflight exact evidence contracts, recover complete KDA knowledge, bound broken retries, and reach high-leverage lanes without weakening verification.

**Architecture:** Keep `sgl-diffusion-engine` as the sole state machine. Add transactional source/knowledge validation, a controller-generated delivery contract consumed by a shared static verifier, durable Codex thread IDs, authenticated-measurement accounting, epoch-local lane deferral, and target-aware routing.

**Tech Stack:** Python 3.11, Pydantic v2, JSON/JSONL, Git worktrees and submodules, Codex CLI, pytest, ruff.

---

### Task 1: Preserve the campaign fixes already proven necessary

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/request.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_request.py`

- [ ] **Step 1: Apply the four isolated campaign fixes**

Cherry-pick commits `ddb4832`, `a17be8f`, `e28a968`, and `7ca5a07` from the
existing controller-fix branch. They provide process-local integration and
Executor identities, writable non-interactive Codex execution, and complete
H3/backend/batch/topology workload freezing.

- [ ] **Step 2: Run their focused tests**

Run:

```bash
pytest -q tests/test_integration_flow.py tests/test_orchestration.py tests/test_request.py
```

Expected: all tests pass with global Git configuration disabled in the identity
regressions.

### Task 2: Make KDA source material complete before snapshotting

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/sources.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/knowledge.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_sources.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_knowledge.py`

- [ ] **Step 1: Write failing gitlink and snapshot tests**

Create a local parent repository with a pinned local submodule. Assert that
`create_worktree(..., initialize_submodules=True, required_submodules=(...))`
checks out the gitlink commit and rejects a missing required path. Add snapshot
tests that reject zero entries, a missing required prefix, a missing copied
reference, and a changed reference digest.

- [ ] **Step 2: Implement explicit submodule materialization**

Add a method equivalent to:

```python
def ensure_submodules(self, worktree: Path, required: Sequence[str]) -> None:
    run(["git", "submodule", "sync", "--recursive"], cwd=worktree)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=worktree)
    for relative in required:
        expected = gitlink_commit(worktree, relative)
        actual = run(["git", "rev-parse", "HEAD"], cwd=worktree / relative)
        if actual.stdout.strip() != expected:
            raise RuntimeError("required submodule is not at its pinned gitlink")
```

Validate safe relative paths before invoking Git.

- [ ] **Step 3: Build and validate snapshots transactionally**

Build references and `index.json` in a sibling staging directory. Require at
least one entry and each configured KDA prefix. Verify each copied file's
`reference_sha256` before publishing or reusing the snapshot. Runtime preserves
an invalid existing output under `knowledge/rejected/` and rebuilds it.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_sources.py tests/test_knowledge.py
```

Expected: complete pinned submodules and snapshot reuse pass; incomplete
material fails closed or is recovered by runtime.

### Task 3: Generate an executable delivery contract and shared preflight

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/delivery_contract.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/cli.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_cli.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

- [ ] **Step 1: Write contract-generation failures**

Assert that the contract contains `Delivery.model_json_schema()`, the active
technique evidence schema, exact performance/command requirements, baseline and
profile hashes, and only pinned KernelWiki reference paths plus matching
digests. An empty KernelWiki set must prevent a kernel Executor from spawning.

- [ ] **Step 2: Add static verifier mode**

Extend `DeliveryVerifier.verify` with `independent_gates: bool = True`. Static
mode runs every schema, containment, command, benchmark, performance, source,
engagement, equivalence, and technique-evidence check, but skips the external
method auditor and quality evaluator. Authoritative calls keep the default.

- [ ] **Step 3: Add the CLI preflight**

Implement:

```text
sgl-diffusion-engine preflight-delivery \
  --campaign <campaign> --technique <lane> \
  --executor-worktree <worktree> --delivery <DELIVERY.json>
```

Print JSON findings and return `0` only when static verification accepts the
bundle.

- [ ] **Step 4: Inject the controller-owned contract**

Write `DELIVERY-CONTRACT.json` under the Executor root before launch and add it
as a hashed prompt section before search state. Include the exact preflight
argv. Keep the file outside the candidate worktree so it cannot enter the
candidate commit accidentally.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/test_cli.py tests/test_orchestration.py tests/test_verifier.py
```

Expected: a valid fixture passes preflight and authoritative verification;
invalid KernelWiki paths fail identically in both modes.

### Task 4: Resume the same Codex thread

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_telemetry.py`

- [ ] **Step 1: Write fake Codex JSONL tests**

Emit:

```json
{"type":"thread.started","thread_id":"67e55044-10b1-426f-9247-bb680e5fe0c8"}
```

Assert persistence to `SESSION.json`, rejection of conflicting IDs, and argv
construction as `codex exec resume <id> <prompt>` on attempt two.

- [ ] **Step 2: Implement thread extraction and durable identity**

Parse only `thread.started` JSON objects from the attempt stdout. Persist the
thread ID atomically with the Executor identity. Do not use `--last`, because
another campaign may have started a newer thread.

- [ ] **Step 3: Use compact feedback turns**

For Codex resume, write a prompt containing the new finding set, delivery path,
and preflight command. Do not concatenate the base prompt or earlier feedback.
Retain full replay for non-Codex commands that cannot resume sessions.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_orchestration.py tests/test_telemetry.py
```

Expected: the first and second invocation share one thread ID and token ledgers
continue to record each turn separately.

### Task 5: Count real measurements and bound broken retries

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_runtime_scheduler.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_progress.py`

- [ ] **Step 1: Write authenticated malformed-delivery tests**

Start with a valid full run and break only `KERNEL-EVIDENCE.json`. Assert that
verification rejects the candidate but exposes one authenticated measurement.
Break the frozen command or benchmark count and assert that no authenticated
measurement is exposed.

- [ ] **Step 2: Record measurement identity independently of acceptance**

Add an `AuthenticatedMeasurement` result containing candidate ID, run path,
mean, workload total, request count, peak memory, and speedup. Record its round
once even when later evidence or audit findings reject the candidate.

- [ ] **Step 3: Add epoch-local lane deferral**

Count prior `executor_resumed` events by failure signature. On the third same
signature or sixth total attempt, write `search/<epoch>/DEFERRED-LANES.json`,
clear the active lane, and select the next unresolved non-deferred route.
Deferred lanes never become coverage dispositions.

- [ ] **Step 4: Make progress reflect durable scheduler state**

Read `EXECUTORS.json`, `TECHNIQUE-DISPOSITIONS.json`, and
`DEFERRED-LANES.json`. Mark only `active_technique` as `running`; use
`attempted`, `deferred`, and `dispositioned` for the rest.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/test_verifier.py tests/test_runtime_scheduler.py tests/test_progress.py
```

Expected: evidence repair consumes the real GPU round once, repeated signatures
advance the lane queue, and progress has exactly one running row.

### Task 6: Route large targets toward high-leverage methods

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/profiler.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/techniques/kernel.md`
- Modify: `skills/sglang-diffusion-auto-optimize/SKILL.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/progress-contract.md`
- Test: `sgl-engine-sglang-diffusion/tests/test_profiler.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_runtime_scheduler.py`

- [ ] **Step 1: Write target-aware routing tests**

Assert the original `residency, kernel` order for lossless-only campaigns and
modest targets. For `allow_quality_gated=true` and target at least `3.0`, assert
`residency, cache, pisa, quantization, token_pruning, kernel`, plus a recorded
`large-gap-quality-first-v1` policy.

- [ ] **Step 2: Pass target and current gap into routing/search state**

Add target speedup to `TechniqueRouter.route`. Include target, best verified
speedup, remaining multiplier, and route policy in `ROUTES.json` and each
Executor's search state.

- [ ] **Step 3: Require screening and production wiring**

Update the kernel contract and skill so a candidate first runs a targeted
correctness/microbenchmark screen. Existing flags may be controls, but a patch
candidate must change production-consumed source and prove positive engagement;
an inert JSON file cannot stand in for the activation that produced a result.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_profiler.py tests/test_runtime_scheduler.py
```

Expected: a 5x quality-gated campaign reaches cache before kernel while all
required lanes remain scheduled.

### Task 7: Validate and publish

**Files:**

- Validate: `skills/sglang-diffusion-auto-optimize/`
- Validate: `sgl-engine-sglang-diffusion/`

- [ ] **Step 1: Run formatting and the complete suite**

Run:

```bash
cd sgl-engine-sglang-diffusion
python -m ruff check src tests
pytest -q
```

Expected: no lint errors and the full test suite passes.

- [ ] **Step 2: Validate the updated skill**

Run:

```bash
python /Users/bbuf/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/sglang-diffusion-auto-optimize
```

Expected: `Skill is valid!`

- [ ] **Step 3: Review, commit, push, and open a draft PR**

Stage only the files listed by this plan, commit with a terse reliability-fix
message, push `agent/repair-diffusion-optimization-loop`, and create a draft PR
against `BBuf/AI-Infra-Auto-Driven-SKILLS:main`. The PR body must describe root
causes, behavior changes, compatibility, and exact validation commands.
