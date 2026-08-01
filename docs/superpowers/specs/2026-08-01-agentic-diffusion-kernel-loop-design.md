# Agentic Diffusion Kernel Optimization Loop Design

## Purpose

Repair `sgl-diffusion-engine` so that an unattended optimization campaign can
actively inspect, generate, edit, profile, and compose kernel changes without
delegating campaign ownership to Sol-Engine. The controller must preserve a
fair user-selected baseline, turn real profiler evidence into kernel work,
continue after individual hypotheses fail, and refuse to report completion
until the combined patch passes the full performance and media-quality gate.

This design corrects the assumptions that caused the MiniMax-H3 campaign to
miss known optimization space: the discovery baseline must not inject
`--performance-mode speed`, an empty profile is not usable evidence, an
individual kernel miss does not exhaust the kernel lane, and LPIPS plus a
visual pass is not a sufficient completion gate for video with audio.

## Success Criteria

The work is complete when:

1. `sgl-diffusion-engine` is the only outer campaign state machine and it
   launches resumable Executor and Master agents itself.
2. The engine may import, pin, and reuse Sol-Engine contracts, evaluators,
   prompts, and kernel tooling, but never launches Sol-Engine's complete
   campaign flow.
3. The exact baseline argv selected by the user or launcher is frozen and
   hashed. The controller never silently adds `--performance-mode speed` or
   changes tensor, context, data, or Ulysses parallel topology afterward.
4. GPU-backed baseline, profile, candidate, integration, and final verification
   runs are serialized by the outer controller even when CPU-only reasoning or
   source inspection can run independently.
5. A usable profile digest is derived from the actual trace, contains non-empty
   stage and hotspot evidence, and fails closed if trace discovery or parsing
   is broken.
6. Kernel work records use of KernelWiki, Nsight Compute evidence, and the
   warp-specialization analysis skill where applicable. Unsupported or
   inapplicable tools require an explicit evidence-backed disposition.
7. Scientific rounds count complete workload measurements, not agent process
   launches. A failed hypothesis feeds the next round and cannot close an
   entire technique lane by itself.
8. Independently verified, latency-positive candidates remain available for
   composition. Final performance is measured on the combined patch rather
   than inferred from isolated gains.
9. Final success requires the target latency and fail-closed LPIPS, VBench,
   audio, AV-sync, and media-contract evidence. Missing evidence cannot be
   interpreted as a pass.
10. Unit and end-to-end controller tests cover the failure modes above, and the
    documentation describes the exact unattended command and evidence outputs.

## Approaches Considered

### 1. Restore the reviewed agentic foundation and harden its invariants

Restore the Executor/Master orchestration that existed before the serial
single-agent refactor, retain the newer placement and packaging corrections on
`main`, then repair its profiler, scheduler, round accounting, integration,
kernel evidence, and quality gate.

This is the chosen approach. The repository already contains reviewed agent
process isolation, prompt contracts, durable handles, Sol component adapters,
and watchdog recovery in that foundation. Reusing it avoids inventing a second
agent protocol while leaving the new engine in control of every state
transition.

### 2. Add an agent adapter to the current manual work-order flow

Keep the current `work/claim/submit` architecture and add a new process that
drives those commands on behalf of a Codex agent. This produces a smaller
conceptual change, but duplicates lifecycle, lease, redaction, and resume logic
already implemented in the prior agentic runtime. It also leaves two competing
protocols for what is logically one campaign.

### 3. Invoke Sol-Engine's full optimization campaign

Hand the request and GPU allocation to Sol-Engine and import its final patch.
This would expose the broadest existing feature set quickly, but violates the
approved ownership boundary: state, scheduling, integration, quality, and
terminal decisions would no longer belong to `sgl-diffusion-engine`.

## Ownership Boundary

```text
sgl-diffusion-engine (sole outer controller)
  |
  +-- freezes request, source locks, baseline argv, topology, and GPU lease
  +-- captures and validates profile evidence
  +-- schedules one GPU-active Executor at a time
  +-- verifies candidate receipts and accumulates positive patches
  +-- asks Master to review evidence and combined quality
  +-- integrates, remeasures, resumes, or terminates
  |
  +-- reusable internals
       +-- Sol contracts/evaluators/kernel utilities (pinned components)
       +-- KDA-Pilot knowledge and profiling skills (pinned evidence sources)
       +-- Codex Executor/Master subprocesses (bounded workers)
```

Executor and Master agents may reason, edit their isolated worktrees, and run
the commands authorized by their contracts. They cannot directly transition
campaign state, declare final success, alter the frozen baseline, or bypass the
controller's GPU measurement lease.

## Baseline and Measurement Invariants

The launch request owns the discovery baseline. `baseline.argv` is normalized
as an argv array, stored verbatim in `REQUEST.json` and `BASELINE.json`, and
hashed before the first run. The engine rejects any later measurement whose
argv digest or parallel-topology projection differs, except for explicitly
declared profiling instrumentation and candidate source/feature toggles.

The launcher and controller contain no policy that adds
`--performance-mode speed`. A caller may include that option deliberately, but
its presence then becomes part of the frozen baseline. For MiniMax-H3 discovery
the caller supplies the normal non-performance-mode command.

A campaign-wide GPU lease covers all commands that can execute the workload:
baseline, profiling, microbenchmark/NCU collection, candidate end-to-end
measurement, combined integration measurement, and final quality generation.
Only the outer controller grants that lease. CPU-only source analysis may be
prepared ahead of the next measurement, but the initial implementation keeps
one Executor process active at a time so an agent cannot accidentally run an
unmediated competing benchmark.

## Profile Evidence Pipeline

Profiling has three explicit stages:

1. **Capture:** execute the frozen profile argv and inventory every trace and
   profiler artifact with hashes and sizes.
2. **Extract:** parse supported `.json`, `.trace.json`, and
   `.trace.json.gz` traces directly. Aggregate complete events by stage,
   operator/kernel name, category, call count, total time, and share of traced
   device time. Sidecar summaries are supplemental rather than required.
3. **Validate and route:** require at least one timed stage and one hotspot with
   finite positive duration. A missing, corrupt, empty, or unsupported trace
   transitions to a recoverable profile failure; the controller retries or
   terminates with an explicit certificate instead of routing from E2E time.

The digest records its source artifact hashes and parser version. Executor
deliveries bind to that digest hash, preventing a stale or hand-written empty
summary from driving kernel decisions.

## Agent and Scientific-Round Protocol

Each technique has one durable Executor identity per epoch. The scheduler
activates techniques serially in a deterministic order derived from the
profile. An Executor invocation produces exactly one of:

- a measured candidate delivery with patch, full workload receipt, correctness
  evidence, and technique-specific evidence;
- a retryable failure report that preserves the lane and feeds a concrete
  verifier finding into the next invocation; or
- a lane disposition whose coverage ledger proves that every required family
  was tried, shown inapplicable, or blocked by a reproducible constraint.

A scientific round is recorded only after the controller authenticates a
complete full-workload candidate receipt. Agent starts, crashes, malformed
deliveries, source-reading attempts, and microbenchmarks do not consume a
scientific round.

Technique budgets are independent. Exhausting or dispositioning one lane
moves to the next lane; it does not close the campaign. Search ends only after
all required lanes have valid dispositions, the combined target and final gate
pass, or a genuine global terminal condition is recorded.

## Kernel Optimization Contract

The kernel Executor is expected to write production kernel code when profiler
evidence supports it. Its coverage ledger includes, at minimum:

- layout/copy removal and launch elimination;
- operator fusion, including normalization, activation, and projection
  patterns;
- attention backend or shape-specialized attention changes;
- GEMM, quantization, and epilogue specialization;
- communication layout, collective fusion, or overlap;
- custom Triton, CUDA/CuTe, or reused upstream kernels.

The required `KERNEL-EVIDENCE.json` binds every kernel candidate or lane
disposition to:

1. the profile digest and exact hotspot/family under investigation;
2. KernelWiki queries and the pinned pages or upstream source evidence used;
3. before/after Nsight Compute report metadata and a metrics digest for an
   implemented custom kernel, or an explicit applicability reason when NCU
   cannot produce meaningful evidence;
4. a warp-specialization applicability audit; CUDA/CuTe warp-specialized
   candidates additionally require the timeline report and prediction versus
   measurement reconciliation from the warp-specialization skill;
5. correctness and shape coverage, microbenchmark data, full-workload latency,
   and patch provenance.

KernelWiki supplies prior art, Nsight Compute identifies the limiting resource,
and the warp-specialization report explains scheduling when that design is
actually present. None substitutes for the full end-to-end measurement.

## Candidate Composition

The candidate registry is append-only and keeps all independently verified
latency-positive points, including wins below the overall target. Integration
starts from the frozen baseline and composes candidates in a stable order.
After each patch is applied, the controller runs correctness and the full
workload; incompatible or regressive combinations are recorded and excluded
without deleting the underlying individual win.

The integrated patch is the unit used for the target comparison. Isolated
speedups are evidence for selection, not proof that their product will be
achieved. When interactions erase a gain, the Master receives the combined
profile and conflict evidence and may request an adapted candidate in the
affected lane.

## Quality and Completion Gate

Final verification uses fresh outputs produced by the integrated revision for
all five validation prompts. `QUALITY-EVIDENCE.json` is owned by the Master and
contains hashed, tool-attributed records for:

- LPIPS per prompt, mean, maximum, and configured threshold;
- VBench per-prompt dimensions and aggregate result;
- audio presence, duration, sample rate, channels, silence/clipping and the
  configured audio-quality checks;
- AV synchronization per prompt and maximum drift threshold;
- media contract facts: container, video/audio codecs, dimensions, frame rate,
  frame count, durations, and stream presence;
- an independent visual review verdict and evidence references.

Schemas require exactly five prompt results and explicit pass/fail status for
every required section. The controller verifies artifact existence, hashes,
producer identity, command receipts, and thresholds. Missing, non-finite,
executor-authored, or stale fields fail closed. Completion requires both the
integrated performance target and this full quality record; otherwise the
controller resumes the responsible lane or produces an unreachable
certificate.

## State and Artifact Changes

The restored state machine retains durable campaign artifacts and adds:

- `PROFILE-INVENTORY.json` for raw trace provenance;
- `TECHNIQUE-DISPOSITIONS.json` for per-lane terminal evidence;
- `SCIENTIFIC-ROUNDS.jsonl` for authenticated workload measurements;
- `KERNEL-EVIDENCE.json` inside kernel deliveries;
- `CANDIDATE-REGISTRY.json` for append-only positive points;
- `COMPOSITION-RESULTS.jsonl` for interaction outcomes;
- `QUALITY-EVIDENCE.json` for the final fail-closed media gate.

All JSON writes remain atomic. Agent-originated paths are contained within the
campaign or assigned worktree, argv is represented as arrays, logs are
redacted, and terminal artifacts include the exact reasons a target was met or
proved unreachable.

## Failure and Recovery

- Profile capture or extraction failures retry from `PROFILING`; an empty
  digest is never cached as valid.
- Agent crashes preserve the worktree and resume token; they do not consume a
  scientific round.
- Invalid delivery evidence resumes the same lane with verifier findings.
- A slower but valid candidate counts as one round and updates the lane's
  hypothesis history; it does not close the lane.
- Technique budget exhaustion writes a per-lane disposition and continues with
  remaining techniques.
- Candidate interaction regressions trigger composition rollback to the last
  verified integrated revision and keep searching.
- Missing quality evidence returns to final verification or search; it cannot
  create `COMPLETED`.

## Validation Strategy

Tests will use deterministic fake workload, profiler, NCU, quality, and agent
commands. They will prove:

1. baseline argv and topology are frozen without injected flags;
2. compressed trace extraction produces real hotspots and corrupt or empty
   traces fail closed;
3. only one GPU-capable Executor is active and measurements are serialized;
4. process launches do not consume rounds, while authenticated candidate runs
   do;
5. one failed kernel candidate resumes the kernel lane and a disposed lane does
   not prevent other techniques from running;
6. multiple sub-target wins compose and are remeasured;
7. kernel deliveries without the three-part KDA evidence contract are rejected;
8. missing VBench, audio, AV-sync, media-contract, or five-prompt evidence
   prevents completion;
9. valid combined performance and complete independent quality evidence create
   the only successful terminal state;
10. watchdog recovery preserves these invariants after interruption.

## Commit and Pull Request Strategy

Use focused commits for the approved design/plan, restored agentic foundation,
profile and scheduling fixes, kernel/quality evidence gates, and documentation
plus validation. Push `agent/diffusion-agentic-kernel-loop` and open a draft
pull request against `main` with the exact checks and known runtime validation
boundary recorded in the body.
