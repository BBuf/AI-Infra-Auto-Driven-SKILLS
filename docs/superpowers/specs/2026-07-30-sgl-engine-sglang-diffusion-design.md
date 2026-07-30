# SGL-Engine for SGLang Diffusion Design

**Date:** 2026-07-30

**Status:** Approved design

## 1. Summary

Build an executable optimization engine under
`sgl-engine-sglang-diffusion/` in
`BBuf/AI-Infra-Auto-Driven-SKILLS`. A user supplies a diffusion model,
hardware target, frozen workload, and end-to-end speedup target. The engine
then:

1. locks the latest SGLang `main` commit visible at campaign start;
2. measures one frozen baseline;
3. profiles and searches Sol-Engine-compatible optimization techniques;
4. expands candidate generation with KDA-Pilot, SGLang Diffusion, and
   FastVideo knowledge;
5. independently verifies every candidate using Sol-Engine correctness and
   quality semantics;
6. integrates compatible candidates and repeats the search when the target is
   not reached; and
7. emits a cleanly applicable, independently verified SGLang patch bundle.

The primary deliverable is `sglang.patch`. Applying it to the locked SGLang
base commit adds a model-specific `--agent-optimization` inference profile and
the code needed to realize the measured acceleration. The inference process
does not run an optimization agent; it only selects an already generated and
validated profile.

## 2. Goals

- Reproduce Sol-Engine's master/executor/verifier/integrator workflow for
  SGLang Diffusion.
- Preserve Sol-Engine's technique scopes, frozen-baseline discipline,
  candidate contract, correctness modes, quality gates, provenance checks,
  independent verification, and integrated revalidation.
- Add SGLang-native optimization and code-placement rules.
- Add KDA-Pilot kernel skills and prior diffusion kernel evidence.
- Automatically extract optimization knowledge from locked SGLang and
  FastVideo source revisions.
- Survive agent, controller, scheduler, SSH, and GPU interruptions without
  losing campaign state.
- Continue through new search epochs until the target is reached, the
  registered search space is exhausted, or a scoped lower-bound certificate
  proves the target impossible.
- Keep generated, model-specific kernels isolated under an SGLang
  `kernels/agent` namespace.
- Produce a patch whose base, source files, performance evidence, quality
  evidence, and activation behavior are auditable.

## 3. Non-goals

- The first pull request does not directly modify the SGLang repository.
- The first pull request does not provide a hosted scheduler, database
  service, web UI, or multi-user control plane.
- The runtime `--agent-optimization` flag does not call an LLM or edit code.
- An arbitrary requested speedup is not guaranteed to be attainable.
- A search plateau is not described as a mathematical impossibility.
- External knowledge is not allowed to weaken or replace Sol-Engine
  correctness and quality gates.
- The engine does not silently rebase a running campaign onto a newer SGLang
  revision.
- The engine does not embed model weights in a source patch.

## 4. Design Choice

Use a hybrid architecture:

- a deterministic Python control plane owns source locks, state transitions,
  leases, command idempotency, artifacts, benchmark calculations, schema
  validation, patch generation, and process recovery;
- coding agents own open-ended source analysis, hypothesis formation, isolated
  code changes, method-equivalence arguments, and Sol-Engine's built-in
  multimodal visual review; and
- a separate master verifier rechecks executor deliveries before integration.

This retains Sol-Engine's agent-native search while preventing an agent from
declaring its own performance, provenance, or delivery valid.

A prompt-only clone was rejected because durable recovery and anti-fabrication
would remain dependent on one conversation. A hosted orchestration service was
deferred because it is unnecessary for the first executable campaign.

## 5. Upstream Authorities and Precedence

The initial Sol-Engine semantic snapshot is based on:

- repository: `NVlabs/Sana`
- branch: `sol-engine`
- reviewed commit:
  `cee25847afdd34bc656abcca126262200b088dc8`

The implementation must record this provenance. It should adapt the contracts
to SGLang rather than copy large upstream documents or source files verbatim.
Source links, commit hashes, file hashes, and applicable third-party notices
must remain available.

When Sol-Engine sources disagree, precedence is:

1. `orchestration/prompts/loop_and_gate_contract.md`;
2. `orchestration/prompts/master.md`;
3. the applicable `workflow/*/nodes/codex_executor/*_scope.md`;
4. deterministic delivery verification code and schemas;
5. technique and site documentation; and
6. paper prose.

The latest lightweight orchestration contract is authoritative because it
defines the current lossless/quality-gated split more precisely than older
site documentation.

Every campaign independently locks:

- SGLang `main`;
- FastVideo `main`;
- KDA-Pilot `main`; and
- the configured Sol-Engine reference.

No source revision changes during a campaign.

## 6. Contract and Knowledge Precedence

Every executor prompt is assembled in this order:

1. Sol-Engine correctness contract;
2. Sol-Engine technique scope;
3. SGLang integration and code-placement rules;
4. SGLang, KDA-Pilot, and FastVideo auxiliary knowledge;
5. current profile and durable experiment history; and
6. agent-generated reasoning.

Higher layers cannot be overridden by lower layers. Auxiliary knowledge can
produce a new candidate but cannot change its correctness mode, frozen
workload, quality gate, or evidence requirements.

## 7. High-level Architecture

```text
optimization_goal.yaml
        |
        v
Source Freezer and Knowledge Sync
        |
        v
Baseline Runner and Profiler
        |
        v
Technique Router
        |
        +--> isolated kernel executor
        +--> isolated cache executor
        +--> isolated PISA executor
        +--> isolated topology executor
        +--> isolated quantization executor
        +--> isolated token-pruning executor
        +--> isolated SGLang extension lanes
        |
        v
Deterministic Delivery Verifier
        |
        v
Master semantic and visual verification
        |
        v
Integrator and full-workload revalidation
        |
        +--> target reached --> patch packager
        |
        +--> target not reached --> reprofile --> next epoch
```

### 7.1 Components

`GoalLoader`
: Validates the model, hardware, workload, approximation policy, target
  speedup, source repositories, artifact root, and optional budget.

`SourceFreezer`
: Fetches configured repositories, resolves immutable commits, and creates a
  clean SGLang baseline worktree plus isolated executor worktrees.

`KnowledgeSync`
: Extracts allowlisted optimization knowledge and writes a source-indexed,
  immutable campaign snapshot.

`BaselineRunner`
: Materializes the frozen evaluation contract and runs the baseline once.

`Profiler`
: Collects SGLang performance dumps and torch-profiler traces and maps stage,
  layer, operator, and kernel hotspots.

`TechniqueRouter`
: Performs an existing-fast-path audit, selects applicable Sol-Engine and
  extension executors, and avoids methods excluded by the goal.

`CampaignController`
: Owns transactional state, epochs, leases, retries, failure signatures, and
  target decisions.

`AgentRunner`
: Provides a configurable command/model adapter for the coding-agent runtime.
  No correctness rule depends on a particular model name.

`DeliveryVerifier`
: Validates delivery schema, real-run provenance, frozen timing scope,
  performance calculations, engagement, lossless structure, and lossy quality
  artifacts without trusting executor-reported numbers.

`MasterVerifier`
: Reads the actual code, independently reasons about method semantics, checks
  authenticity with built-in multimodal vision, and resumes rejected
  executors with exact feedback.

`Integrator`
: Composes compatible verified frontier points, launches the combined
  full-workload run, and repeats all applicable Sol-Engine gates.

`PatchPackager`
: Generates a binary-safe full-index patch and verifies it in a new worktree.

`Watchdog`
: Restarts the controller and reclaims expired executor leases. It cannot
  accept experiments or change scientific state.

## 8. Campaign Input

An input file has this conceptual shape:

```yaml
schema_version: 1
model:
  id: Wan-AI/Wan2.2-T2V-A14B-Diffusers
hardware:
  environment: b200
  gpu_count: 1
workload:
  prompts: validation-prompts.txt
  prompt_count: 5
  seed_policy: fixed
  height: 720
  width: 1280
  frames: 81
  fps: 16
  steps: 50
  guidance: 5.0
  dtype: bfloat16
  timing_scope: load_excluded_end_to_end
goal:
  target_speedup: 2.0
  allow_quality_gated: true
source:
  sglang_repo: https://github.com/sgl-project/sglang.git
  sglang_ref: main
agent:
  command: codex
  model: configurable
```

The exact schema will be defined in the implementation plan. Once the baseline
exists, fields that affect the evaluation contract become immutable.

If no validation prompt file exists, the controller may materialize five
prompts from a checked-in SGLang benchmark preset or an explicitly supplied
production workload. It then records that file as the campaign-local canonical
validation set. It cannot replace those prompts after baseline creation.

## 9. Frozen Baseline

The baseline is measured exactly once and records:

- model/checkpoint identity and revision;
- SGLang source commit and dirty-state check;
- hardware, driver, CUDA, library, and dependency information;
- prompt text and order;
- seed policy;
- scheduler, steps, guidance, flow shift, and motion settings;
- resolution, frame count, duration, and FPS;
- precision and quantization;
- GPU count and topology;
- timing scope and warmup policy;
- end-to-end and stage timing;
- peak memory;
- output video or image artifacts and aligned frames; and
- run and source provenance.

Executors receive the frozen baseline but cannot recreate or modify it. Every
authoritative speedup is recomputed as:

```text
frozen_baseline_total_s / candidate_benchmark_total_s
```

The timing scopes must match exactly. Microbenchmarks can screen a hypothesis
but cannot populate a frontier.

## 10. Sol-Engine Technique Registry

### 10.1 Compatibility pass

| Technique | Correctness | Search scope | Per-session rounds |
| --- | --- | --- | ---: |
| `kernel` | lossless | kernel/backend/fusion/compile optimization | 40 |
| `cache` | quality-gated | TeaCache, EasyCache, TaylorSeer matched-time frontier | 20 |
| `pisa` | quality-gated | PISA exact critical blocks plus approximate remainder | 20 |
| `topology` | lossless | distributed partitioning, placement, communication, scheduling | 20 |
| `quantization` | quality-gated | Sol-Engine quantization families | 20 |
| `token_pruning` | quality-gated | Sol-Engine pruning and token-merging families | 20 |

The first four entries inherit the lightweight Sol-Engine orchestration. The
last two exist in Sol-Engine's paper and technique documentation but not as
standalone lightweight executors. This project adds their lightweight
executors using the same loop, delivery, provenance, visual, and quality
contracts. Their 20-round default is an SGLang adaptation and must not be
misrepresented as an upstream lightweight setting.

### 10.2 SGLang extension pass

The compatibility pass executes the original technique boundaries before
SGLang-specific scope expansion. Additional method families run in separately
labeled extension lanes. Examples include:

- Cache-DiT and other SGLang cache integrations;
- Breakable CUDA Graph, dual-stream, and compiler variants;
- non-PISA sparse/video attention backends;
- SGLang ModelOpt, Nunchaku, MXFP4, and checkpoint integrations;
- VAE, text-encoder, decoder, offload, and residency improvements;
- SGLang diffusion JIT, Triton, CuTeDSL, FlyDSL, and AOT kernels;
- FastVideo FA4, VSA, V-MoBA, SLA, CFG-gating, and TurboDiffusion ideas; and
- KDA-Pilot shape-specialized diffusion kernels.

An extension lane inherits the Sol-Engine correctness mode appropriate for the
actual method. It cannot hide inside a compatibility frontier. For example,
Cache-DiT is not added to Sol-Engine's closed three-family cache comparison; it
runs as `sglang-cache-extension` under the same quality-gated semantics.

Every candidate records its knowledge origin, repository, commit, and source
paths.

## 11. Sol-Engine Correctness and Quality

This project does not add an alternative quality system. Sol-Engine semantics
are the acceptance authority.

### 11.1 Common requirements

Every retained candidate must:

- edit only its isolated worktree;
- preserve the frozen workload and timing scope;
- have an explicit OFF guard that restores the source-current path;
- prove OFF identity;
- run the complete frozen workload on real target hardware;
- preserve durable output, frame, benchmark, manifest, assessment, and
  provenance artifacts;
- demonstrate real technique engagement and report fallback counters;
- improve latency or establish a non-dominated memory point;
- report exact reproduction commands and source hashes; and
- produce a schema-valid `DELIVERY.json`.

The master independently recomputes performance and verifies provenance. It
rejects fabricated, mismatched, relabeled, no-op, or configuration-only
deliveries and resumes the executor with exact problems.

### 11.2 Lossless techniques

Kernel and topology candidates are mathematically or algorithmically
lossless. Correctness is a property of the method and actual code, not an
output-difference test.

The verifier and master must not use:

- bit identity;
- latent or tensor differences;
- floating-point tolerances;
- LPIPS;
- PSNR; or
- visual-quality similarity

as a rejection gate for lossless candidates.

A lossless candidate must:

- compute the same algorithm;
- preserve global logical denoising-step and DiT/model-call counts;
- introduce no approximation;
- introduce no step skipping;
- introduce no sparsity;
- introduce no sub-16-bit quantization;
- introduce no rank reduction; and
- preserve logical model work.

Output frames are inspected only for authenticity. Numeric movement caused by
fusion or floating-point reduction order is not itself a defect.

Topology candidates additionally require durable preflight, topology
manifest, trace, and equivalence artifacts. The master audits world size,
active ranks, rank map, process groups, token/head/expert/parameter/CFG
coverage, collective ordering, all-rank participation, source hashes, and
absence of silent fallback.

### 11.3 Quality-gated techniques

Cache, PISA/sparse attention, quantization, and token pruning use:

- fixed baseline-aligned frames;
- aligned LPIPS;
- built-in agent multimodal review with no external Gemini or vision API;
- Sol-Engine's visual artifact rubric;
- prompt-level evidence rather than aggregate-only reporting;
- technique engagement evidence; and
- full end-to-end performance in the frozen timing scope.

The visual rubric covers snow/speckle, blur, mosaic and patch boundaries,
banding, ghosting, melting, temporal flicker, coherence, motion, and new
artifacts. A candidate does not pass merely because LPIPS is low.

The cache compatibility executor preserves Sol-Engine's matched-time
comparison of TeaCache, EasyCache, and TaylorSeer. Operating points are
quality-ranked only at comparable measured time ratios, and noisy or close
points use repeated full runs and their median.

### 11.4 Integration

The integrator can use only independently verified frontier points. It must
relaunch the composed recipe and recheck:

- authoritative speedup against the same frozen baseline;
- timing-scope identity;
- source and run provenance;
- authenticity and engagement of every selected technique;
- lossless method correctness;
- topology compatibility after composition; and
- LPIPS and multimodal visual quality when any selected technique is lossy.

An all-lossless recipe must not add an output-difference or visual-quality
gate.

## 12. Knowledge Sources

### 12.1 KDA-Pilot

The kernel executor can use:

- `KernelWiki`;
- `ncu-report-skill`;
- `warp-specialization-report-skill`;
- diffusion kernel rules;
- diffusion correctness and benchmark-shape contracts;
- existing B200 and H200 diffusion kernel campaigns; and
- profile and implementation evidence for QKNorm+RoPE, normalization,
  GroupNorm+SiLU, rotary embedding, scale/shift, residual gates, causal Conv3D,
  attention copy/concatenation, FA4, and related fusions.

These inputs improve profiling, kernel design, hardware reasoning, NCU
collection, and source implementation. They do not replace the Sol-Engine
lossless gate.

### 12.2 SGLang Diffusion

The campaign indexes the locked SGLang revision, including:

- `sglang-diffusion-benchmark-profile`;
- `sglang-diffusion-performance`;
- `sglang-diffusion-modelopt-quant`;
- JIT and AOT kernel authoring guidance;
- benchmark and profiler utilities;
- attention backends and selectors;
- TeaCache and Cache-DiT;
- ModelOpt FP8/NVFP4, Nunchaku, MXFP4, and weight-only paths;
- `torch.compile`, BCG, topology, and offload controls;
- existing diffusion JIT, Triton, CuTeDSL, FlyDSL, and AOT operators; and
- associated unit tests, microbenchmarks, and server tests.

Every kernel round first runs an existing-fast-path audit. A new operator
requires profile evidence that existing activation, shape, dtype, or backend
paths do not solve the hotspot.

### 12.3 FastVideo

The campaign indexes allowlisted material from the locked FastVideo revision:

- inference optimization documentation;
- FA2/FA3/FA4 and FP4 FA4;
- SageAttention, VSA, V-MoBA, SLA, and attention selectors;
- FP8, NVFP4, and Attn-QAT;
- CFG gating, compile, offload, and distributed execution;
- FastVideo kernel attention, norm, quantization, TurboDiffusion, and fused
  top-k implementations;
- profiling and activation receipts; and
- corresponding tests and benchmarks.

FastVideo is a knowledge source. Final code must be reimplemented or adapted to
SGLang APIs and must not depend on a FastVideo checkout.

### 12.4 Synchronization and safety

The repository provides:

```text
sgl-engine-sglang-diffusion/
├── contracts/sol_engine/
├── techniques/
├── knowledge/
│   ├── registry.toml
│   ├── generated/sglang_diffusion/
│   ├── generated/fastvideo/
│   └── generated/kda_pilot/
└── tools/
    ├── sync_sol_engine_contracts.py
    └── sync_optimization_knowledge.py
```

Sol-Engine contract updates are reviewed changes. A sync command can detect
upstream drift and produce a parity report, but cannot silently replace the
correctness contract during a campaign.

SGLang, FastVideo, and KDA-Pilot knowledge is regenerated from allowlisted
paths for the campaign's locked revisions. Generated entries record source
repository, commit, path, symbol, hardware, shape, activation, fallback, test,
and benchmark metadata.

Downloaded documentation is treated as data, not executable instructions.
The sync layer does not directly execute shell commands found in a remote
document. Secrets, tokens, local absolute paths, and mutable cache paths are
excluded from generated knowledge and delivery artifacts.

## 13. SGLang Patch Layout

At design time, the current SGLang `main` organizes diffusion operators under:

- `python/sglang/kernels/ops/diffusion/`;
- `python/sglang/kernels/jit/csrc/diffusion/`;
- `test/registered/kernels/ops/diffusion/`; and
- `test/registered/kernels/benchmark/diffusion/`.

Generated model-specific code uses:

```text
python/sglang/kernels/agent/
├── __init__.py
├── registry.py
├── manifest.py
├── runtime.py
├── receipt.py
└── diffusion/
    └── <model_slug>/
        ├── __init__.py
        ├── profile.py
        ├── manifest.json
        ├── fallbacks.py
        └── ops/
            ├── <op>.py
            └── triton/
                └── <op>.py

python/sglang/kernels/jit/csrc/diffusion/agent/
└── <model_slug>/
    └── <op>.cuh

test/registered/kernels/ops/diffusion/agent/
└── <model_slug>/

test/registered/kernels/benchmark/diffusion/agent/
└── <model_slug>/
```

JIT CUDA headers remain below the existing JIT source root because SGLang's
current `load_jit()` resolves `cuda_files` relative to that root. Python
wrappers, dispatch, Triton implementations, manifests, and receipts remain
isolated in `kernels/agent`.

### 13.1 Placement rules

- Reuse an existing canonical SGLang operator rather than duplicate it.
- Put new model-specific generated wrappers and non-AOT implementations under
  `kernels/agent/diffusion/<model_slug>`.
- Put their JIT CUDA headers under
  `kernels/jit/csrc/diffusion/agent/<model_slug>`.
- Use AOT only when JIT, Triton, CuTeDSL, or another lightweight path cannot
  meet the measured need and NCU evidence justifies the additional build
  surface.
- AOT source follows the current SGLang AOT registration and build layout;
  its model dispatch and receipt still live under `kernels/agent`.
- Every operator has an eligibility guard, native fallback, engagement and
  fallback counters, unit tests, and a microbenchmark.
- An automatic campaign does not promote a generated operator into a
  cross-model canonical namespace. A later human-reviewed SGLang pull request
  may perform that refactor.

## 14. Runtime Activation Protocol

The generated patch adds:

```text
--agent-optimization off
--agent-optimization auto
--agent-optimization <profile-id>
```

`off`
: Default. Uses the original SGLang behavior.

`auto`
: Resolves a built-in profile from model identity, hardware, precision, and
  request shape. A nonmatching request uses the native fallback.

`<profile-id>`
: Selects an exact profile. A model, hardware, dependency, dtype, or shape
  mismatch fails closed instead of silently running a different path.

The flag does not contact an agent.

### 14.1 Profile manifest

A generated profile declares:

- profile and campaign IDs;
- accepted model IDs and architectures;
- locked SGLang base commit;
- hardware vendor, compute capabilities, and GPU count;
- dtype and workload shape constraints;
- selected technique activations;
- exact server arguments;
- fallback policy;
- source hashes; and
- integrated delivery evidence hash and measured speedup.

Manifests cannot contain a dynamic Python import path, remote executable, or
arbitrary shell command.

### 14.2 Argument precedence

Profile resolution happens before existing performance auto-tuning can alter
the verified configuration:

1. parse the requested agent optimization;
2. resolve model and hardware identity;
3. load and validate the built-in profile;
4. reject conflicting explicitly supplied arguments;
5. apply profile values to unspecified fields;
6. force `performance_mode=manual`;
7. run normal SGLang validation and platform setup;
8. initialize registered dispatch; and
9. recheck request-shape guards at execution time.

Explicit conflicts fail with a diagnostic rather than silently overriding the
user or invalidating the verified recipe.

### 14.3 Engagement receipt

Every optimized inference run emits a structured receipt containing:

- selected profile;
- model, hardware, dependency, and workload matches;
- requested techniques;
- per-technique and per-operator engagement counts;
- fallback counts and reasons; and
- implementation source hashes.

An enabled flag with zero engagement is a no-op and cannot satisfy
Sol-Engine's authenticity gate.

## 15. Patch Bundle

The final directory is:

```text
deliverables/<campaign_id>/
├── sglang.patch
├── manifest.json
├── SHA256SUMS
├── apply_and_verify.sh
├── README.md
└── evidence/
    ├── BASELINE.json
    ├── INTEGRATED-DELIVERY.json
    ├── source-locks.json
    ├── quality/
    ├── profiles/
    └── receipts/
```

`sglang.patch` is generated as a full-index, binary-safe diff from the locked
base to the verified integration commit.

Clean-room validation creates a new worktree, checks out the exact base,
performs `git apply --check`, applies the patch, runs build/import checks,
operator tests, the frozen end-to-end workload, and the complete applicable
Sol-Engine gate.

`apply_and_verify.sh` refuses a base-commit mismatch. It does not automatically
resolve conflicts or silently use a three-way merge. A newer SGLang base
requires a patch-rebase campaign and revalidation.

### 15.1 Quantized weights

Some quantization candidates require derived weights that a source patch cannot
contain.

- Online quantization and standard publicly available checkpoints can
  participate in a patch-only delivery.
- A derived checkpoint can participate only when uploaded to a user-configured
  artifact store and locked by immutable URI, revision, size, and SHA-256.
- A candidate whose required weights are unavailable remains experimental
  frontier evidence and cannot claim patch-application acceleration.
- The final integrated frontier must expose at least one runnable path whose
  complete dependencies are declared and obtainable.

## 16. Persistent State and Execution

Each campaign stores:

```text
runs/<campaign_id>/
├── goal.lock.yaml
├── state.sqlite
├── events.jsonl
├── source-locks.json
├── baseline/
├── profiles/
├── knowledge/
├── hypotheses/
├── candidates/
├── integration/
├── failures/
├── leases/
└── deliverables/
```

SQLite owns transactional current state. The JSONL event log is append-only and
supports audit and recovery. Artifacts are written to temporary paths and
atomically renamed before state transitions reference them.

Commands and candidate attempts have idempotency keys. Executors acquire
time-limited leases. An expired lease returns work to the queue without
accepting partial evidence.

## 17. Search Epochs and Termination

Sol-Engine's per-executor round budgets remain intact. Campaign persistence is
implemented through epochs rather than an unbounded executor session.

An epoch:

1. profiles the current best verified integration;
2. runs applicable compatibility executors;
3. runs applicable extension lanes;
4. independently verifies deliveries;
5. integrates compatible frontier points; and
6. performs a full frozen-workload gate.

If the target is not reached, a new epoch can start only when there is:

- a new measured bottleneck;
- a hotspot changed by integration;
- newly synchronized knowledge;
- a previously untested admissible hypothesis; or
- a compatibility problem that requires a revised candidate.

The global failure ledger prevents repeated failure signatures across epochs.

Terminal states are:

`TARGET_REACHED`
: The clean-room integrated patch passes all applicable Sol-Engine gates and
  its authoritative integrated speedup meets or exceeds the requested target.

`UNREACHABLE_CERTIFIED`
: A machine-readable certificate shows that the target violates a lower bound
  for the frozen workload, hardware envelope, and allowed method set. A generic
  search plateau cannot produce this state.

`SEARCH_SPACE_EXHAUSTED`
: No new admissible hypothesis remains in the registered and locked knowledge
  space, but the system cannot prove that an unknown algorithm would fail.

`CANCELLED`
: Explicit user cancellation.

Recoverable nonterminal states include `WAITING_RESOURCE`, `INFRA_BLOCKED`, and
`PAUSED_BUDGET`.

## 18. Error Handling

| Failure | Required behavior |
| --- | --- |
| Agent process exits | Reclaim the expired lease and resume the executor |
| GPU, SSH, or scheduler outage | Enter `WAITING_RESOURCE`; do not consume an optimization round |
| OOM | Record a hardware failure signature; allow a distinct memory hypothesis |
| Candidate build or test fails | Preserve logs and consume that candidate round |
| Quality gate fails | Reject the candidate and retain prompt-level evidence |
| Delivery is missing or fabricated | Reject and resume the executor with exact issues |
| Requested optimization has zero engagement | Reject as a no-op |
| Integration conflict | Return to the relevant executor for a compatible implementation |
| Patch fails against the locked base | Fail delivery; do not publish the patch |
| Upstream advances during a campaign | Keep the campaign locked; use the new commit only in a new campaign |
| Knowledge source path is stale | Mark stale and resync; do not invent a replacement command |
| Failure signature repeats | Skip the duplicate and preserve the prior evidence |
| Budget is exhausted | Enter `PAUSED_BUDGET`, not a completed or unreachable state |

The watchdog can restart processes and reclaim leases. It cannot modify
baseline, frontier, quality, or terminal decisions.

## 19. Testing Strategy

### 19.1 Sol-Engine parity

Tests must prove:

- all six technique identities exist;
- compatibility scopes and correctness modes are preserved;
- lossless verification never invokes LPIPS or output-difference gates;
- quality-gated verification requires LPIPS, visual verdict, and engagement;
- the frozen baseline cannot be refreshed;
- authoritative speedup is independently recomputed;
- the master must independently verify before integration;
- Sol-Engine source commit and contract hashes are traceable; and
- upstream contract drift is reported rather than silently accepted.

### 19.2 State machine

Tests cover:

- crash recovery;
- lease expiry;
- command idempotency;
- candidate acceptance and rejection;
- executor resumption with feedback;
- epoch advancement;
- global failure-signature deduplication;
- budget pause and resume; and
- prevention of `INFRA_BLOCKED` becoming an unreachable decision.

### 19.3 Knowledge synchronization

Tests cover:

- repository and path allowlists;
- source SHA and path preservation;
- malformed or stale sources;
- downloaded commands treated as non-executable data;
- contract precedence over auxiliary knowledge;
- secret and absolute-path sanitization; and
- deterministic generated indexes for a fixed source lock.

### 19.4 Patch generation

Using a temporary fake SGLang repository, tests cover:

- full-index patch generation;
- `git apply --check`;
- base-commit enforcement;
- source-hash verification;
- expected `kernels/agent` placement;
- JIT source placement;
- forbidden edits to baseline or shared checkouts;
- rejection of local absolute paths and credentials; and
- clean-room application.

### 19.5 Mocked orchestration

A CPU-only mocked end-to-end test:

1. creates a fake baseline;
2. launches a fake executor;
3. receives a fabricated speedup and rejects it;
4. resumes the executor;
5. receives a valid second delivery;
6. independently verifies it;
7. integrates it;
8. restarts the controller mid-run;
9. resumes without duplicate work; and
10. produces a valid patch bundle.

### 19.6 GPU release validation

A real GPU release campaign uses a small supported SGLang Diffusion model and
completes:

- one frozen baseline;
- one profiled hotspot;
- one kernel candidate;
- one Sol-Engine delivery and independent verification;
- one integrated run;
- one clean-room patch application; and
- one end-to-end revalidation.

This is release evidence, not a normal pull-request CI job.

## 20. Initial Repository Layout

The first implementation plan should materialize focused modules similar to:

```text
sgl-engine-sglang-diffusion/
├── README.md
├── pyproject.toml
├── contracts/
├── techniques/
├── knowledge/
├── schemas/
├── prompts/
├── src/sgl_engine_sglang_diffusion/
│   ├── cli.py
│   ├── config.py
│   ├── state.py
│   ├── sources.py
│   ├── knowledge.py
│   ├── controller.py
│   ├── agents.py
│   ├── baseline.py
│   ├── profiler.py
│   ├── verifier.py
│   ├── integrator.py
│   ├── patcher.py
│   └── watchdog.py
├── tools/
├── examples/
└── tests/
```

The implementation plan may refine filenames to follow repository conventions,
but it must preserve these responsibility boundaries.

## 21. First Pull Request Acceptance Criteria

The first pull request is complete when it contains:

- an installable Python package and CLI;
- goal, technique, candidate, delivery, integrated-delivery, profile, receipt,
  and source-lock schemas;
- the Sol-Engine-compatible technique and correctness contract layer;
- deterministic persistent state and recovery;
- isolated SGLang worktree management;
- allowlisted knowledge synchronization and provenance;
- configurable agent runner interfaces;
- executor spawn, poll, resume, and delivery validation;
- master/integrator prompt and artifact contracts;
- patch generation and clean-room application checks;
- CPU-only unit and mocked orchestration tests;
- documentation for starting, resuming, inspecting, and packaging a campaign;
  and
- a documented GPU release-validation procedure.

The first pull request does not need to demonstrate a universal 2x model
speedup. It must demonstrate that the engine can run the complete bounded
workflow, persist its evidence, reject invalid results, resume, and produce a
verified SGLang patch from a valid campaign.
