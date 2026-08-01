# Diffusion Optimization Loop Reliability Design

## Purpose

Repair the unattended `sgl-diffusion-engine` loop after the MiniMax-H3 campaign
showed that valid GPU work could be hidden behind repeated delivery-protocol
failures. The controller must preserve strict verification while making every
contract executable, resuming the same Codex thread, bounding repeated
infrastructure failures, and allowing high-leverage lanes to proceed.

reviews SGLang changes; `sglang-diffusion-auto-optimize` owns this optimization
flow. The implementation and user-facing changes therefore belong to the
diffusion skill and its adjacent Python controller.

## Success Criteria

1. KDA-Pilot worktrees initialize the exact pinned submodule commits for
   KernelWiki, NCU reporting, and warp-specialization reporting before the
   knowledge snapshot is built.
2. A knowledge snapshot is non-empty, satisfies required path families, and
   revalidates every copied file and digest when reused. An incomplete cached
   snapshot is preserved for audit and rebuilt.
3. Every Executor receives a controller-generated delivery contract containing
   exact schemas, required artifacts, profile and baseline bindings, and valid
   pinned KernelWiki citation paths and hashes.
4. Executors can run the same structural verifier locally before returning a
   delivery. Independent method and quality gates remain controller-owned and
   run only during authoritative verification.
5. `codex exec` attempts persist the `thread.started` ID. Feedback uses
   `codex exec resume <thread-id>` with a compact turn prompt rather than
   starting another conversation with the complete original prompt.
6. Candidate worktrees receive process-local Git identities and a writable
   non-interactive Codex sandbox without mutating global configuration.
7. The exact H3/backend/batch/shape/parallel workload is frozen, and
   integration cherry-picks have a process-local committer identity.
8. A structurally flawed delivery that nevertheless contains an authenticated
   full frozen-workload measurement consumes one scientific round. Duplicate
   measurements remain idempotent.
9. The same verifier failure may be resumed at most three times and one
   Executor generation at most six times. A repeatedly broken lane is deferred
   for the epoch so other lanes run; it is not falsely declared exhausted.
10. For large quality-gated targets, routing begins with residency, then moves
    through high-leverage algorithmic lanes before the kernel tail. Low-impact
    candidates are screened with targeted tests before a costly five-prompt
    run.
11. Progress reports only the manifest's active lane as running, and exposes
    deferred and dispositioned lanes accurately.

## Approaches Considered

### 1. Controller-owned contracts and bounded recovery

Generate the contract from the same Pydantic models and pinned campaign
artifacts used by the verifier. Add a static preflight command, persist Codex
thread identity, and defer a lane after bounded repeated failures.

This is the selected approach. It removes duplicated human interpretation
without weakening scientific or quality gates, and it can be introduced
without replacing the existing campaign state machine.

### 2. Prompt-only clarification

Add more examples and warnings to `executor.md`. This is small but leaves
schema drift, invalid knowledge paths, fresh Codex conversations, and infinite
retry behavior intact. The failed campaign already demonstrated that prose is
not a reliable protocol boundary.

### 3. Move all candidate execution out of Executors

Make Executors submit only source patches and have the outer controller create
all run artifacts. This provides the strongest ownership boundary, but it is a
larger protocol migration that would invalidate current delivery producers.
The selected design first makes the existing boundary deterministic and
machine-checkable.

## Architecture

```text
pinned source locks
  -> submodule-complete immutable worktrees
  -> validated knowledge snapshots
  -> Controller delivery contract
  -> Codex Executor thread (initial turn)
       -> targeted screen
       -> full frozen workload
       -> static preflight
  -> same Codex thread (feedback turns)
  -> authoritative verifier + independent gates
  -> candidate registry / next gap-aware lane
```

The Controller remains the sole campaign owner. The static preflight shares
schema, path, command, performance, kernel/residency evidence, and source-diff
checks with the authoritative verifier. It deliberately skips independent
method review and quality evaluation, so an Executor cannot self-approve.

## Source and Knowledge Integrity

`SourceManager.create_worktree` gains an explicit submodule mode. Runtime uses
it only for KDA-Pilot and requires these paths:

- `external/KernelWiki`;
- `external/ncu-report-skill`;
- `external/warp-specialization-report-skill`.

Each required path must be a gitlink at the locked parent commit, must be
checked out at that gitlink commit, and must contain files. Knowledge sync is
transactional: build in a staging directory, validate required prefixes and
all reference hashes, then publish the index. Reuse repeats the validation.
Invalid legacy output is renamed under a campaign-owned rejected directory
before reconstruction, preserving audit evidence.

## Delivery Contract and Preflight

Before spawning an Executor, the runtime writes `DELIVERY-CONTRACT.json`
outside the candidate worktree. It contains:

- JSON schemas generated from the current delivery and technique evidence
  models;
- exact required artifact names and performance fields;
- frozen baseline identity, timing scope, request count, command-template hash,
  profile hash, and GPU inventory binding;
- valid pinned KernelWiki reference paths with their reference digests;
- the exact static-preflight argv for the assigned worktree and delivery.

The assembled prompt includes this contract as a hashed, higher-precedence
controller section. `sgl-diffusion-engine preflight-delivery` loads the same
campaign baseline, registry, command template, and verifier implementation.
It emits structured findings and exits non-zero until all static checks pass.
The authoritative verifier subsequently repeats those checks and adds the
independent method or quality gate.

## Codex Session Continuity

For a Codex command, the first JSONL stream must contain exactly one
`thread.started` event with a non-empty thread ID. The manager persists it in
`SESSION.json`, includes it in attempt manifests, and rejects conflicting IDs.

Attempt one uses `codex exec`. Later attempts use:

```text
codex exec resume <thread-id> <compact-feedback-prompt>
```

The compact prompt contains only the new verifier findings, the durable
delivery path, and the preflight command. The original contract and earlier
reasoning stay in the Codex thread and in immutable campaign artifacts. Generic
non-Codex agent commands retain the legacy full-prompt replay behavior.

## Measurement Accounting and Recovery

Verification results distinguish:

- a fully accepted frontier point;
- an authenticated full-workload measurement with later evidence/audit
  findings; and
- a pre-measurement or incomparable failure.

The second category consumes a scientific round because the expensive frozen
workload really ran. It never enters the candidate registry until every gate
passes. Round identity binds technique, candidate, run directory, and normalized
performance digest so a repaired resubmission cannot count twice.

Repeated failure signatures are bounded per Executor. At the third occurrence,
or the sixth total attempt, the lane is recorded in the epoch's deferral ledger
and the scheduler advances to another unresolved lane. Deferred lanes are not
coverage dispositions and cannot justify `SEARCH_SPACE_EXHAUSTED`. A later
epoch gets a fresh Executor/thread; if no productive lane remains, the
controller reports a precise infrastructure block instead of looping.

## Gap-Aware Scheduling

For lossless-only or modest targets, keep `residency -> kernel`. When quality
gating is allowed and the target is at least 3x, route:

```text
residency -> cache -> pisa -> quantization -> token_pruning -> kernel
```

This preserves an early lossless residency result while preventing a marginal
kernel/configuration candidate from blocking algorithmic methods capable of
closing a large target gap. The route manifest records the selected policy and
target. Executor search state includes the target, current verified best, and
remaining multiplier. Technique prompts require a targeted screen before a
full workload and require production-wired source changes rather than inert
evidence/config files.

## Testing

Tests use local Git repositories and fake Codex JSONL streams. They cover
gitlink checkout validation, stale snapshot rejection/rebuild, exact
KernelWiki citation contracts, static preflight parity, thread ID persistence
and resume argv, compact prompts, authenticated malformed-delivery round
accounting, duplicate suppression, bounded deferral, high-target route order,
accurate progress state, workload freezing, writable agent execution, and Git
identity isolation. The full controller suite and skill validator must pass
before publication.
