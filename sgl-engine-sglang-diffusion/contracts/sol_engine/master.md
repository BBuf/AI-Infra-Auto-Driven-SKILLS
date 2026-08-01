# Master verification and integration contract

The master coordinates exactly the technique executors selected for a campaign.
It does not modify their isolated worktrees or trust their self-reported
delivery facts.

## Persistent protocol

1. Spawn or resume one executor for each selected registered technique.
2. Poll until each executor produces a schema-valid `DELIVERY.json`.
3. Independently verify every delivery's real run directories, source
   provenance, timing scope, benchmark values, activation, engagement, OFF
   identity, and durable artifacts.
4. Apply the correctness branch declared in the technique registry.
5. Return rejected deliveries to the same executor with specific, actionable
   findings, then repeat verification.
6. Compose only independently verified frontier points.
7. Run the composed SGLang candidate against the same frozen baseline and
   reapply every applicable gate.
8. Write `INTEGRATED-DELIVERY.json` only from independently verified measured
   facts.

Restarting the master must resume existing executor state rather than
double-spawning work or refreshing the baseline.

## Independent performance and authenticity

Authoritative speedup is recomputed as:

```text
frozen baseline total_s / candidate benchmark total_s
```

Both values must cover the identical frozen timing scope. Projected,
microbenchmark-derived, configuration-only, fabricated, relabeled,
baseline-resubmitted, fallback, or otherwise mismatched results are rejected.
The master checks the actual candidate code, implementation manifest, source
hashes, run provenance, output frames, and engagement receipt.

## Correctness branch

For `residency`, `kernel`, and the optional legacy `topology` lane, correctness
is mathematical and algorithmic. The
master audits the actual method and code and accepts only the same global
algorithm with unchanged logical denoising-step and DiT/model-call counts and
no approximation, step skip, sparsity, sub-16-bit quantization, rank reduction,
or changed logical work. It must not compute or gate on bit identity, tensor or
latent differences, floating-point tolerances, LPIPS, PSNR, or visual
similarity. Frames establish authenticity only.

Residency verification owns declared component placement, transfer scheduling,
measured memory headroom, and compile-versus-steady-state restoration while
preserving the baseline GPU set and parallel degrees. Kernel verification owns local operator, backend, fusion, compilation, exact
invariant preparation, and layout semantics while preserving the frozen
distributed topology. Topology verification owns world size, active ranks,
rank map, process groups, placement, token/head/expert/parameter/CFG coverage,
collective ordering, all-rank participation, and absence of silent fallback.

For `cache`, `pisa`, `quantization`, and `token_pruning`, the master requires
aligned LPIPS, prompt-level evidence, positive technique engagement, and its own
built-in multimodal review of candidate frames beside the frozen baseline.
The review covers authenticity, snow or speckle, blur, mosaic or patch
boundaries, banding, ghosting, melting, temporal flicker, coherence, motion, and
new artifacts. A low aggregate LPIPS value alone cannot pass the candidate, and
no external Gemini or vision API may supply the verdict.

## Composition

The baseline topology is frozen in the default flow. Topology, only when an
explicit legacy flow selected it before baseline freeze, is the distributed substrate and is applied before
local techniques. The master re-audits local shape, process-group, dispatch,
and fallback assumptions under that topology. Every selected technique must
show real engagement in the integrated run. Any quality-gated component makes
the complete composition quality-gated; an all-lossless composition remains
lossless and is never subjected to an output-difference or quality-similarity
gate.
