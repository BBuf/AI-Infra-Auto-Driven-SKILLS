---
name: sglang-diffusion-auto-optimize
description: Autonomously optimize one SGLang Diffusion model on a named GPU machine from a frozen baseline command until a measured end-to-end target or reviewed terminal boundary. Use when the current interactive root agent should directly own profiling, serial SGLang or kernel changes, self-contained verification, integration, progress, recovery, and patch packaging without spawning subagents or nested AI processes.
---

# SGLang Diffusion Auto-Optimize

## Contract

Own one campaign in the current conversation. Require only:

- target machine or SSH alias;
- model ID or checkpoint;
- exact native SGLang Diffusion baseline command; and
- target measured end-to-end speedup.

Use exactly one AI owner: the current root agent. Never call `spawn_agent`,
launch `codex exec` or Claude, create an AI reviewer, or delegate a technique
to another conversation. Never run candidate GPU measurements in parallel.

The Python controller is a deterministic evidence tool. It may lock sources,
run benchmarks and profilers, create one worktree, verify artifacts, integrate
candidates, and package a patch. It must not choose hypotheses or launch AI.

Keep the bundled correctness and quality rules binding. Additional SGLang,
FastVideo, KDA-Pilot, KernelWiki, profiler, or model-history evidence may
suggest a hypothesis but may not relax a gate.

Use the complete bundled method and candidate space. Distinguish a documented
or referenced method from one that is SGLang-adapted or end-to-end validated.

## Read Before Acting

Read:

- [references/remote-ownership.md](references/remote-ownership.md);
- [references/progress-contract.md](references/progress-contract.md); and
- [references/work-order-protocol.md](references/work-order-protocol.md); and
- [references/search-space-and-knowledge.md](references/search-space-and-knowledge.md).

Start from
[references/request-template.yaml](references/request-template.yaml).

Read the installed skill whose name exactly matches the requested machine when
one exists. Follow its access, container, repository, GPU-allocation, and
cleanup rules. Never invent an endpoint.

Read `sglang-diffusion-benchmark-profile`,
`sglang-diffusion-performance`, and `model-pr-history-knowledge` when
available. Read kernel-specific skills only after evidence selects a kernel
hypothesis.

## Phase 1: Resolve And Freeze

Connect through the matching host skill. Determine:

- persistent campaign and tool roots;
- the SGLang checkout, origin, and fetched full commit;
- the fixed GPU IDs and their UUIDs;
- the active Python, CUDA, and PyTorch environment; and
- whether another user owns a process on the fixed GPUs.

Preserve dirty user work. Do not reset, clean, kill, or overwrite it. The
controller creates detached campaign worktrees from its bare cache.

Validate paths, imports, benchmark help, and the five-prompt file without
running a second full baseline. The controller owns the one authoritative
baseline.

## Phase 2: Install The Evidence Tool

Prefer the package beside this skill:

```text
<repository-root>/sgl-engine-sglang-diffusion
```

Install it into an isolated environment on the target machine:

```bash
python3 -m venv <persistent-tool-root>/.venv
<persistent-tool-root>/.venv/bin/python -m pip install \
  <repository-root>/sgl-engine-sglang-diffusion
```

If only the skill was installed, use the package from the same immutable
repository revision. Do not silently use another branch.

## Phase 3: Launch Deterministic Setup

Write the request under a campaign-owned directory. Preserve the exact model,
workload flags, ordering, non-secret environment, absolute checkout, source
origin, and fetched commit. Do not add an `agent` command.

Launch:

```bash
sgl-diffusion-engine launch \
  --request <absolute-request.yaml> \
  --detach
```

The watchdog may run source locking, baseline, and profiling. It stops at
`AWAITING_AGENT`. That status is not terminal: it means this root agent must
choose the next action.

Record the campaign ID/path and returned `work_command`. Reuse an idempotent
campaign instead of launching a duplicate.

## Phase 4: Run The Serial Search Loop

Repeat this loop until a terminal state:

1. Read current evidence:

   ```bash
   sgl-diffusion-engine work --campaign <campaign> --json
   ```

2. Inspect the frozen baseline, profile, prior failures, verified frontier,
   hardware capability, technique dispositions, bound `SEARCH-SPACE.json`, and
   `KNOWLEDGE.json`.

3. Read the routed family's complete method/candidate projection. Query the
   SGLang, KDA-Pilot, KernelWiki, NCU, and FastVideo snapshots using the
   knowledge protocol. Choose one evidence-backed hypothesis. Routes are
   suggestions, not mandatory lanes. Exclude known-inapplicable methods; for
   example, do not attempt NVFP4 on Hopper.

4. Claim exactly one technique:

   ```bash
   sgl-diffusion-engine claim \
     --campaign <campaign> \
     --technique <technique>
   ```

5. Work directly in the returned `worktree`. Before every candidate run,
   verify the fixed GPU UUIDs, ownership, memory, utilization, and rendezvous
   port. Treat resource contention as `WAITING_RESOURCE`, not a failed
   hypothesis.

6. Implement one hypothesis, commit it, and run one complete frozen workload.
   Preserve the timing scope, prompt set, seed, shape, steps, guidance, dtype,
   GPU set, native backend, command receipt, engagement receipt, source hashes,
   and real media.

   Cite exact snapshot source, commit, relative path, and raw SHA-256 values in
   the implementation manifest's nonempty `knowledge_origin`.

7. Review the actual diff and evidence yourself. Write the required
   `AGENT-REVIEW.json` and, for quality-changing work, review all five prompt
   outputs with built-in vision. Do not claim independent review.

8. Submit:

   ```bash
   sgl-diffusion-engine submit \
     --campaign <campaign> \
     --delivery <worktree>/DELIVERY.json
   ```

9. Read the verifier findings. A rejection returns to `AWAITING_AGENT`; it
   rejects that candidate, not the campaign. Choose a changed hypothesis and
   claim a new work order.

The verifier recomputes latency, speedup, hashes, command equivalence,
backend/fallback state, engagement, LPIPS, and review bindings. Never edit
evidence after `submit`.

## Scientific Round And Skip Rules

Count one scientific round only after one distinct candidate completes a full
frozen-workload measurement and is explicitly submitted.

Do not count:

- GPU or port contention;
- disconnects or root-agent interruption;
- dependency failure found in preflight;
- launch failure before model execution; or
- malformed metadata that references no measured run.

A repeated failure signature closes only that hypothesis.

Use `skip` only after reviewing evidence:

```bash
sgl-diffusion-engine skip \
  --campaign <campaign> \
  --technique <technique> \
  --classification <unsupported|no_gain|blocked> \
  --reason <specific-evidence-backed-reason>
```

`unsupported` and `no_gain` close the technique. `blocked` remains recoverable.
Closing a technique that already has a verified candidate excludes that
candidate and triggers a serial remeasurement of the remaining verified
subset; this selection-only epoch consumes no scientific round.
`SEARCH_SPACE_EXHAUSTED` is valid only after every suggested technique is
explicitly closed or has consumed its scientific budget and the best verified
integrated subset remains below target. Before closing a technique, satisfy its
full bundled candidate-coverage requirements; a PISA-only sparse search or a
three-family-only cache search is incomplete.

## Integration And Progress

The tool integrates only verified latency-positive candidates. It does not
wait for every suggested technique and never adds isolated speedups.
Composition conflicts return to this root agent for a changed candidate.

Use:

```bash
sgl-diffusion-engine progress --campaign <campaign> --json
```

Report baseline and integrated end-to-end latency, best isolated speedup,
gate/findings, integrated stack speedup, epoch, scientific rounds, active work
order, and disposition. CLI token accounting is unavailable because the AI
owner is this conversation, not a spawned process.

Send concise updates at meaningful transitions and continue acting. Do not
stop merely because the tool yielded at `AWAITING_AGENT`.

## Finish

Finish only at:

- `TARGET_REACHED`;
- `SEARCH_SPACE_EXHAUSTED`; or
- `UNREACHABLE_CERTIFIED`.

Do not call a plateau impossible. `UNREACHABLE_CERTIFIED` requires a
deterministically checkable lower-bound certificate.

For `TARGET_REACHED`, report the locked commit and workload, baseline/final
latency, measured speedup, selected techniques, verification result, patch
path and SHA-256, application command, and GPU revalidation command.

Leave campaign evidence and detached worktrees available for audit. Clean only
campaign-owned processes and temporary paths.
