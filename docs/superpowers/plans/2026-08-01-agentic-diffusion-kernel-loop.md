# Agentic Diffusion Kernel Optimization Loop Implementation Plan

> **For Codex:** Execute this plan task by task without delegating the outer
> campaign implementation. Keep `sgl-diffusion-engine` as the sole state owner.

**Goal:** Restore unattended Executor/Master operation and make its kernel
optimization loop profile-driven, GPU-serial, compositional, and fail-closed at
the final multimodal quality gate.

**Architecture:** Restore the reviewed pre-PR81 agent runtime as the starting
point, then harden its controller invariants. Agents work in isolated
worktrees, while the outer runtime alone freezes baseline/topology, grants GPU
measurement work, verifies evidence, counts scientific rounds, composes
candidates, and decides terminal state. Pinned Sol-Engine components and the
three KDA-Pilot skills are internal dependencies, never an alternate campaign
driver.

**Tech Stack:** Python 3.11, Pydantic v2, JSON Schema artifacts, subprocess argv
execution, Git worktrees, gzip/JSON trace parsing, pytest, ruff.

---

## Task 1: Restore the reviewed agentic foundation

**Files:**

- Restore changes removed by commit `2e17b7a` under
  `sgl-engine-sglang-diffusion/` and
  `skills/sglang-diffusion-auto-optimize/`.
- Preserve `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/patcher.py`
  and the model-root placement behavior already present on current `main`.

**Step 1: Mechanically restore the prior foundation**

Run:

```bash
git revert --no-commit 2e17b7a
```

Resolve any conflict by retaining the newer placement correction and the
agentic runtime side for files intentionally converted to manual work orders.

**Step 2: Verify the architecture boundary**

Run:

```bash
rg -n "AgentRunner|ExecutorManager|Master" \
  sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion
rg -n "work/claim|AWAITING_AGENT|no nested" sgl-engine-sglang-diffusion
```

Expected: agent runtime symbols are restored and the manual-only/no-agent
guard is gone. No command invokes Sol-Engine's full campaign entry point.

**Step 3: Commit the mechanical restore separately**

```bash
git add README.md sgl-engine-sglang-diffusion \
  skills/sglang-diffusion-auto-optimize
git commit -m "refactor: restore agentic diffusion optimization flow"
```

## Task 2: Freeze the exact baseline and parallel topology

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/request.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/baseline.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/schemas/baseline.schema.json`
- Modify: `sgl-engine-sglang-diffusion/schemas/launch-request.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_request.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

**Step 1: Write failing invariant tests**

Add tests proving that launch-request normalization keeps the supplied argv
byte-for-byte by element, never adds `--performance-mode speed`, records a
stable argv digest, extracts a stable parallel-topology projection, and rejects
a candidate receipt that changes that projection.

**Step 2: Run the focused tests and observe failure**

```bash
cd sgl-engine-sglang-diffusion
pytest -q tests/test_request.py tests/test_verifier.py
```

**Step 3: Implement frozen invocation metadata**

Add a Pydantic record containing `argv`, `argv_sha256`, and explicitly selected
parallel flags. Populate it once from the launch request, persist it with the
baseline, and compare it against all full-workload receipts. Profiling-only
instrumentation belongs in a separate field and does not mutate the frozen
argv.

**Step 4: Re-run focused tests**

Expected: all request and verifier tests pass.

## Task 3: Parse real traces and fail closed on empty profile evidence

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/profiler.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/schemas/profile-digest.schema.json`
- Create: `sgl-engine-sglang-diffusion/schemas/profile-inventory.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_profiler.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`

**Step 1: Add compressed-trace fixtures in test code**

Generate a small Perfetto/Chrome trace with complete CUDA kernel, collective,
copy/layout, and CPU-range events. Assert exact call counts, duration totals,
shares, categories, source hash, and parser version. Add corrupt, unsupported,
and event-empty cases that must raise `ProfileEvidenceError`.

**Step 2: Run the tests and observe the missing extraction**

```bash
pytest -q tests/test_profiler.py tests/test_orchestration.py -k profile
```

**Step 3: Implement capture inventory and streaming extraction**

Open `.json`, `.trace.json`, and `.trace.json.gz` safely; parse `traceEvents`,
retain complete positive-duration events, normalize microseconds to
milliseconds, categorize hotspots, and write an inventory of source path,
size, hash, parser, and event count. Sidecars may augment, but never replace,
the raw trace result.

**Step 4: Enforce non-empty routing inputs**

Reject cached digests whose source hash no longer matches, whose stage table is
empty, or whose hotspot list is empty/non-finite. Keep the state in profiling
for a bounded retry and produce an explicit terminal reason if capture cannot
be repaired.

**Step 5: Run focused tests**

Expected: raw compressed traces route correctly; broken extraction cannot
reach search.

## Task 4: Serialize Executors and count scientific rounds correctly

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py`
- Create: `sgl-engine-sglang-diffusion/schemas/technique-disposition.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_progress.py`

**Step 1: Write scheduler and round-accounting tests**

Assert that one search epoch has at most one active Executor, a second
technique starts only after the first candidate/disposition is processed,
spawn/resume/crash events do not count as rounds, and only a verified complete
full-workload receipt appends `scientific_round_completed`.

Add the critical regression test: a valid but slower kernel candidate resumes
the kernel lane; exhausting that lane starts the next technique rather than
terminating the campaign.

**Step 2: Run focused tests and observe failure**

```bash
pytest -q tests/test_orchestration.py tests/test_progress.py
```

**Step 3: Implement a deterministic serial technique queue**

Persist the ordered queue and current lane. Spawn only the current Executor.
After its accepted candidate or validated disposition, advance the queue.
Keep technique-local attempt, failure, and round budgets; calculate global
search exhaustion only when every required lane has a disposition.

**Step 4: Add authenticated round events**

Append a round record after command receipt, correctness, latency, and baseline
comparability pass structural verification. Record outcome `improved`,
`regressed`, or `equal`; only the first remains a composition candidate.

**Step 5: Re-run focused tests**

Expected: scheduling and lifecycle tests pass without concurrent agents.

## Task 5: Require coverage-complete lane dispositions

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/techniques.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/prompts/executor.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/kernel.md`
- Test: `sgl-engine-sglang-diffusion/tests/test_techniques.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

**Step 1: Write failing coverage tests**

Require every configured kernel family to have one of `measured`,
`inapplicable`, or `blocked`, with evidence. Reject a disposition that closes
the lane after one miss or uses a generic reason for multiple unexamined
families.

**Step 2: Implement the disposition model and verifier**

Define coverage item IDs in the technique contract. Validate uniqueness,
allowed status transitions, artifact references, authentic scientific round
IDs for measured items, and exact required-set coverage.

**Step 3: Update the Executor contract**

Tell each invocation to deliver one candidate or one complete disposition. A
single rejected hypothesis is feedback for another round, not permission to
declare a lane exhausted.

**Step 4: Run focused tests**

```bash
pytest -q tests/test_techniques.py tests/test_verifier.py -k "coverage or disposition"
```

## Task 6: Enforce all three KDA-Pilot kernel evidence paths

**Files:**

- Modify: `sgl-engine-sglang-diffusion/knowledge/registry.toml`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/knowledge.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/prompts/executor.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/kernel.md`
- Create: `sgl-engine-sglang-diffusion/schemas/kernel-evidence.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_knowledge.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

**Step 1: Write failing kernel evidence tests**

Reject kernel deliveries that omit KernelWiki queries/pages, NCU before/after
metadata or an evidence-backed non-applicability statement, and the
warp-specialization applicability audit. For a CUDA/CuTe warp-specialized
candidate, additionally reject missing timeline and reconciliation reports.

**Step 2: Pin the actual skill resources**

Expand the knowledge lock beyond the three `SKILL.md` files to the required
KernelWiki query/index resources, NCU reference/helpers, and
warp-specialization timeline instructions/helpers. Record every consumed file
hash in the campaign knowledge manifest.

**Step 3: Implement `KERNEL-EVIDENCE.json` verification**

Bind evidence to profile digest hash, hotspot, candidate family, source
revision, report hashes, metric values, correctness shapes, microbenchmark,
and full-workload round. Require all referenced files to stay within the locked
knowledge tree, campaign, or assigned worktree.

**Step 4: Re-run focused tests**

```bash
pytest -q tests/test_knowledge.py tests/test_verifier.py -k "kernel or knowledge"
```

## Task 7: Preserve and compose every positive candidate

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Create: `sgl-engine-sglang-diffusion/schemas/candidate-registry.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`

**Step 1: Write the combination regression tests**

Create two candidates that improve the baseline by less than the target
individually but meet the target after combination. Assert both are retained,
applied in stable order, and remeasured as one integrated revision. Add an
interaction regression test that excludes the bad combination while preserving
the individual candidates and resumes the affected lane.

**Step 2: Run the tests and observe current all-or-nothing behavior**

```bash
pytest -q tests/test_orchestration.py tests/test_integration_flow.py -k \
  "compose or candidate_registry"
```

**Step 3: Implement the append-only candidate registry**

Store independent positive points by delivery and patch hash. Remove the
requirement that every route provide a candidate. Integrate verified positives
incrementally; record apply conflicts and measured interactions separately.

**Step 4: Re-run focused tests**

Expected: sub-target wins compose; one failed optimization never closes another
lane or erases another win.

## Task 8: Replace the incomplete completion check with a fail-closed media gate

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/quality.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/prompts/master.md`
- Create: `sgl-engine-sglang-diffusion/schemas/quality-evidence.schema.json`
- Test: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

**Step 1: Write one missing-section test per gate**

Start from a complete five-prompt record, then remove LPIPS, VBench, audio,
AV-sync, media contract, independent visual review, producer identity, receipt,
or an artifact hash. Each mutation must block `COMPLETED`. Add non-finite,
threshold failure, stale revision, and four-prompt cases.

**Step 2: Implement structured independent evidence**

Require Master-owned command receipts and per-prompt records. Recompute summary
threshold decisions from leaf values. Validate stream facts and hashes rather
than trusting a single `pass` boolean.

**Step 3: Gate the terminal transition**

Allow `COMPLETED` only when integrated target comparison and every quality
section pass for the same integrated revision. Missing evaluator capability is
a blocked/unreachable reason, never success.

**Step 4: Run focused tests**

```bash
pytest -q tests/test_verifier.py tests/test_orchestration.py -k quality
```

## Task 9: Update skill, examples, schemas, and operator documentation

**Files:**

- Modify: `sgl-engine-sglang-diffusion/README.md`
- Modify: `sgl-engine-sglang-diffusion/examples/goal.yaml`
- Modify: `skills/sglang-diffusion-auto-optimize/SKILL.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/progress-contract.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/remote-ownership.md`
- Modify: root `README.md` where discovery commands are listed.
- Test: `sgl-engine-sglang-diffusion/tests/test_cli.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_launcher.py`

Document that the user selects and freezes parallelism, the discovery example
does not contain `--performance-mode speed`, the controller launches Codex
Executor/Master agents, Sol components are permitted under the ownership
boundary, measurements are serial, and all three KDA evidence paths plus the
full final media gate are mandatory.

Run:

```bash
pytest -q tests/test_cli.py tests/test_launcher.py
rg -n -- "--performance-mode speed" README.md examples \
  ../skills/sglang-diffusion-auto-optimize
```

Expected: the forbidden example flag is absent unless explicitly presented as
a user-supplied counterexample.

## Task 10: Full validation and focused commits

**Step 1: Run package tests**

```bash
cd sgl-engine-sglang-diffusion
python -m pytest -q
```

**Step 2: Run static and packaging checks**

```bash
python -m compileall -q src tests
python -m pip wheel . --no-deps -w /tmp/sgl-diffusion-engine-wheel
ruff check src tests
```

If `ruff` is not installed, run the repository's available pre-commit checks
and record that boundary in the pull request.

**Step 3: Audit invariants and the staged diff**

```bash
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
rg -n "Sol-Engine|sol-engine" sgl-engine-sglang-diffusion \
  skills/sglang-diffusion-auto-optimize
rg -n -- "--performance-mode speed" sgl-engine-sglang-diffusion \
  skills/sglang-diffusion-auto-optimize
git status --short
```

Confirm every Sol reference is a component/contract dependency, no full Sol
campaign command exists, no generated cache or secret is staged, and unrelated
repository files remain untouched.

**Step 4: Commit coherent implementation groups**

Use focused messages such as:

```bash
git commit -m "fix: require real profile evidence for diffusion search"
git commit -m "fix: serialize agentic optimization rounds"
git commit -m "feat: enforce kernel and media evidence gates"
git commit -m "docs: document compositional agentic optimization"
```

## Task 11: Push and open a draft pull request

**Step 1: Reconfirm GitHub authentication and scope**

```bash
gh auth status
git status --short --branch
git log --oneline origin/main..HEAD
```

**Step 2: Push the focused branch**

```bash
git push -u origin agent/diffusion-agentic-kernel-loop
```

**Step 3: Open a draft PR against `main`**

The title should describe the agentic kernel-loop repair. The body must include
the ownership boundary, the six corrected failure modes, exact validation
commands/results, and any hardware-backed validation not performed locally.
Open as draft so the user can inspect the controller and evidence contracts
before merge.
