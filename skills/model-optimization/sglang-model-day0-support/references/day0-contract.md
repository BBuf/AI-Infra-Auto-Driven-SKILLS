# SGLang Model Day-0 Contract

## Contents

- [Required Bundle](#required-bundle)
- [Capability Taxonomy](#capability-taxonomy)
- [Evidence Classes](#evidence-classes)
- [Seven Gates](#seven-gates)
- [PR DAG Rules](#pr-dag-rules)
- [Risk-Pair Selection](#risk-pair-selection)
- [Completion Rules](#completion-rules)

## Required Bundle

A Day-0 program owns eight files:

| File | Purpose |
|---|---|
| `scope-contract.md` | Lock the release target, required lanes, exclusions, and definitions of success. |
| `architecture-gap-map.md` | Map model contracts to existing SGLang paths, missing work, fallbacks, and evidence classes. |
| `pr-dag.md` | Divide implementation into reviewable PRs with dependencies and merge gates. |
| `validation-matrix.md` | Define baseline lanes and selected high-risk feature intersections. |
| `release-lock.md` | Pin source, weights, processors, dependencies, images, and documentation revisions. |
| `pr-body.md` | Present public motivation, implementation, validation, limitations, and evidence. |
| `follow-up-ledger.md` | Separate later fixes, performance work, platform extensions, experiments, and reverts. |
| `sanitization-report.md` | Record public-evidence checks and an uncommitted denylist scan. |

Keep commands and results beside the active model-support work. Do not store
secrets or private artifact identifiers in the public bundle.

## Capability Taxonomy

### Source and configuration

- architecture and `model_type` detection;
- configuration defaults, nested sub-configs, and remote-code requirements;
- model/tokenizer/processor revisions and generation configuration;
- base, instruct, multimodal, and draft checkpoint variants.

### Loading and quantization

- weight names, stacked projections, expert mapping, PP filtering, and tied
  weights;
- native checkpoint dtype and quantization metadata;
- post-load transforms, scale layouts, padding, and architecture-gated
  backends;
- local, Hugging Face, object-store, and streaming loaders;
- target and speculative-draft paths independently.

### Model execution

- embeddings, normalization, dense MLP, MoE routing/activation/experts, and LM
  head;
- attention types: full, MLA, sliding window, linear/recurrent, sparse/indexed,
  compressed, and mixtures;
- position encoding, attention residuals, shared experts, and auxiliary heads;
- eager reference path before specialized kernels.

### State and memory

- KV, recurrent, convolution, compression, draft, and multimodal state;
- item size, dtype, stride, alignment, padding slots, and logical ownership;
- prefix cache snapshots, restore, eviction, locking, and session release;
- graph padding, accepted-token commit/rollback, and pool capacity solve;
- L1/L2/L3 or external-cache transfer semantics.

### Protocol

- chat encoding and roles;
- reasoning sections and effort controls;
- tool definitions, calls, arguments, and termination;
- structured output and grammar interaction;
- streaming marker fragmentation and end-of-stream holdback;
- invalid input and HTTP error mapping.

### Speculative decoding

- draft model detection/loading and attention backend;
- target hidden-state contract;
- proposal, verify, sampling, and accepted-length accounting;
- target and draft state commit/rollback;
- graph capture, padding, remote storage, PD, cache, and parallel composition.

### Multimodal

- media parsing and input validation;
- reference-equivalent resize, color, normalization, patchification, and token
  expansion;
- encoder and projector tensor contracts;
- processor-to-scheduler and encoder-to-language feature transport;
- encoder DP, EPD, PD-prefill, caching, shape bucketing, and graph fallback.

### Parallelism and serving topology

- TP, DP attention, EP, expert load balancing, CP/DCP, and PP;
- unified, PD, and EPD roles;
- heterogeneous prefill/decode layouts;
- collectives, symmetric-memory assumptions, rank-idle behavior, and topology
  defaults;
- per-rank logical-to-physical state mapping.

### Graphs, overlap, and kernels

- eager fallback and `covered()` envelope;
- capture/replay buffer refresh and stable stream/event topology;
- alternative-stream tensor lifetime and joins;
- PDL/barrier generation and producer visibility;
- shape, dtype, architecture, alignment, and numerical contracts;
- microbenchmark, dispatcher, model, and end-to-end evidence.

### Platform and release

- NVIDIA, AMD, NPU, XPU, CPU, and architecture-specific dependencies;
- source branch or release tag, Python wheels, compiled extensions, and images;
- cookbook commands and verified markers;
- unit, kernel, model, accuracy, performance, and nightly CI;
- public issue/PR ownership and post-release response.

## Evidence Classes

Assign exactly one class to each item:

| Class | Definition | Public treatment |
|---|---|---|
| `day0-required` | Needed for a claimed load, correctness, protocol, topology, hardware, or release lane. | Must pass or narrow the Day-0 claim. |
| `post-day0-fix` | Repairs a previously claimed behavior after the release cut. | Link the affected original claim and reopen its gate. |
| `performance-only` | Improves an already correct supported path. | Keep outside the correctness spine; preserve benchmark scope. |
| `experiment-or-revert` | Open experiment, closed-unmerged work, reverted code, or unreachable path. | Retain as a negative lesson; never present as shipped. |

## Seven Gates

### 1. Source gate

Pin immutable model, configuration, tokenizer/processor, SGLang, dependency,
and image revisions. Record missing public artifacts as blockers.

### 2. Load gate

Validate architecture detection, weight mapping, quantization post-processing,
memory allocation, and deterministic short generation on the eager fallback.

### 3. Protocol gate

Validate chat encoding, reasoning, tools, structured output, streaming splits,
stop conditions, invalid inputs, and public API fields.

### 4. State gate

Validate state sizes/layouts/dtypes, cache snapshot/restore/eviction, graph
padding, accepted-state commit/rollback, session release, and fallback parity.

### 5. Topology gate

Validate every required role and selected risk pair. Check rank-idle paths,
logical state ownership, collective symmetry, and heterogeneous transfer
layouts.

### 6. Quality/performance gate

Validate accuracy in band, memory capacity, representative latency/throughput,
and evidence that intended fast paths engage. Keep quality and performance
results bound to their exact revision, hardware, and workload.

### 7. Release gate

Lock public source, images, dependencies, cookbook, CI, limitations, support
owners, and the post-Day-0 ledger. Re-run the public-evidence and sanitization
checks on the final diff.

## PR DAG Rules

- Put reusable infrastructure before model wiring.
- Land eager correctness before optional fast paths.
- Isolate platform-specific dependencies and kernels where review permits.
- Give every fast path a fallback, envelope, dispatcher test, and model-level
  test.
- Keep documentation commands tied to the exact public source they install.
- Use an umbrella PR for integration visibility, not as a substitute for
  reviewable component boundaries.
- For a development-branch rebase, record each non-mechanical conflict,
  resolution, owner, and validation.
- Backfill missing commits and tests explicitly after merge; do not silently
  amend the historical Day-0 claim.

## Risk-Pair Selection

Add a pair when both features change at least one of:

- token count or padding;
- state layout, dtype, ownership, or lifetime;
- graph buffers, streams, events, or capture control flow;
- collective membership or allocation order;
- prefill/decode or encoder/language transfer;
- parser markers, stop conditions, or output fields.

Cover every required feature once, then prioritize pairs with shared state.
Document unsupported pairs with a fail-fast check and public limitation.

## Completion Rules

- Startup is evidence only for the load gate.
- A kernel microbenchmark is not model correctness or end-to-end performance.
- Equal allocation sizes do not prove equal logical addresses across ranks.
- A public recipe is unverified until its exact command and artifacts pass.
- An open PR is candidate evidence, not shipped support.
- A reverted/default-disabled path remains an experiment unless a reachable
  public dispatcher proves otherwise.
- A release is complete only when all eight files pass validation and every
  claimed lane closes all seven gates.
