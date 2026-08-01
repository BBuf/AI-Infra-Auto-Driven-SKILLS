# Profile-Driven Residency and Historical Optimization Rules

## Goal

Extend `sgl-diffusion-engine` so that high-memory GPU residency choices and
reusable SGLang Diffusion optimization patterns are searched deliberately,
measured on the frozen workload, composed with other wins, and never accepted
from a copied heuristic alone.

## Evidence base

The design was derived from a complete subject-level scan of the 1,001
PR-numbered commits touching `python/sglang/multimodal_gen` in SGLang main at
`fd96a35fb087a30e55a17bca28b198028b689f5f`. A semantic audit identified 365
unique PRs in at least one performance family: residency/offload,
compile/graph/warmup, kernel/fusion/layout, parallel communication, VAE/decode,
exact reuse, runtime/I/O, quantization, or sparse approximation. The checked-in
rule catalog will cite only representative PRs whose source diffs were manually
reviewed; the broader scan establishes coverage, not an unverified speed claim.

The key residency lesson is conditional. More VRAM can remove repeated H2D
traffic, but disabling every offload flag is unsafe. The decision depends on
selected-GPU minimum free memory, steady-state peak, transient compile peak,
component sizes, copy/compute overlap, model stages, and conflicts with FSDP or
Cache-DiT. Partial DiT layer residency is a distinct candidate between pure
streaming and full residency.

## Architecture

### Structured historical rule catalog

Add a checked-in TOML catalog with stable rule IDs. Each rule records:

- owner technique and correctness mode;
- profile and memory signals that justify trying it;
- candidate actions and required measurements;
- incompatibilities and routing boundaries;
- representative SGLang PR URL and merge commit; and
- validation notes that prevent copying model-specific thresholds blindly.

A small parser validates schema version, unique rule IDs, known techniques,
full commit hashes, GitHub PR URLs, nonempty triggers/actions/evidence, and
correctness agreement with the technique registry. The controller injects only
the rules owned by the active technique into the Executor prompt, with a digest
of the catalog. These rules are hypothesis generators below the frozen
correctness and technique contracts.

### Residency lane

Add `residency` as a default lossless technique before kernel search. Its
coverage IDs are:

1. `component-residency` — text/image encoders, VAE, vocoder, bridges, and DiT;
2. `partial-dit-residency` — resident leading-layer subsets while streaming the
   tail;
3. `layerwise-prefetch` — prefetch distance and copy/compute overlap;
4. `compile-time-residency` — transient compile offload versus steady-state
   placement; and
5. `load-order-lifetime` — memory-aware loading, release points, and stage
   lifetime.

Preflight and unsupported combinations consume no scientific round. A candidate
round still requires a complete frozen-workload measurement. Positive
residency candidates enter the existing append-only candidate registry and are
combined with compatible kernel, cache, sparse-attention, quantization, and
pruning wins.

### Residency evidence

Every residency delivery includes `RESIDENCY-EVIDENCE.json`, bound to the
candidate, run, profile digest, and frozen GPU set. It records:

- GPU total memory, minimum free memory before the run, baseline/candidate peak,
  and configured safety margin;
- component strategy and estimated/resolved memory footprint;
- DiT layer count, resident-layer count, and prefetch depth when applicable;
- baseline/candidate H2D copy count and duration;
- compile and steady-state placement separately;
- incompatibility checks for FSDP, Cache-DiT, explicit user placement, and
  unsupported components;
- engagement counters and complete-run performance artifact; and
- an equivalence argument proving unchanged logical work.

The verifier fails closed on missing, stale, contradictory, negative-headroom,
or non-engaged evidence. A hard-coded VRAM threshold without a measured frozen
workload cannot pass.

### Broader lossless E2E kernel coverage

The kernel lane expands from only the repeated DiT path to every hot region
inside the frozen load-excluded E2E scope. New required coverage IDs are:

- `compile-graph-warmup` for regional compile, graph capture, persistent
  compiler caches, and graph-break reduction;
- `vae-decode-postprocess` for VAE layouts, exact decode parallelism, halo/copy
  removal, and media finalization fast paths; and
- `scheduler-precompute-sync` for invariant precomputation, exact K/V reuse,
  host synchronization removal, and inference-mode/runtime overhead.

Precision changes such as FP32-to-BF16 VAE decode remain in the quality-gated
quantization lane. Progressive resolution, approximate cache, sparse attention,
and reduced work remain in their existing quality-gated lanes. Parallel degree
changes remain forbidden after the baseline topology is frozen.

## Data flow

1. Freeze source, workload, parallel topology, GPU identity, and baseline.
2. Capture and validate the raw trace.
3. Route `residency` and `kernel`, plus existing applicable lanes.
4. Load and validate the historical rule catalog.
5. Inject only active-lane rules into the Executor prompt.
6. Run preflight without consuming a round.
7. Implement one candidate, emit lane-specific evidence, and run the full
   frozen workload.
8. Verify evidence and register positive candidates.
9. Compose compatible candidates and remeasure the integrated stack.
10. Apply the existing independent final quality and delivery gates.

## Failure and interaction rules

- No universal `GPU memory >= X` rule is accepted. Historical thresholds seed
  a measured candidate only.
- `partial-dit-residency` requires layerwise offload; raising prefetch depth is
  not equivalent because prefetched layers may be retransferred every step.
- Cache-DiT and DiT layerwise offload are incompatible unless the frozen source
  provides and verifies a compatible path.
- FSDP and DiT layerwise offload are treated as mutually exclusive unless the
  source-current implementation proves otherwise.
- Compile-time offload is transient and must restore the requested steady-state
  placement before scored inference.
- Explicit user memory flags are part of the baseline but may be changed only
  by a declared residency candidate; user workload and parallel topology stay
  frozen.
- One rejected residency hypothesis does not close the lane. A complete
  disposition must cover every residency coverage ID.

## Historical audit artifact

Add a manual PR-diff dossier for each PR cited by the rule catalog. Every card
records the PR link/state/merge time, diff size, motivation, concrete files and
symbols, a short real diff excerpt, and validation implications. This is the
human-auditable source for the compact TOML rules.

## Tests

- catalog accepts complete rules and rejects duplicate IDs, unknown techniques,
  malformed PR references, missing triggers, and correctness drift;
- router includes residency without changing frozen parallel topology;
- prompts include only rules owned by the active lane and bind the catalog hash;
- residency deliveries require valid profile-bound evidence and measured
  headroom/copy/engagement fields;
- malformed evidence and copied thresholds fail without consuming a scientific
  round;
- registry coverage/default order and progress budgets include residency;
- expanded kernel coverage cannot be closed by a compile-only attempt; and
- the complete existing test, lint, compile, schema, and wheel suite remains
  green.

## Delivery

Publish the implementation as a draft PR, wait for CI, merge it using the
repository's squash convention, reinstall the merged
`sglang-diffusion-auto-optimize` skill locally, and verify the installed files
match the merge commit. Separately update `/Users/bbuf/工作目录/Common/prompt.md`
so every target, explanation, request value, idempotency key, and completion
condition uses 5.00x instead of 3.00x.
