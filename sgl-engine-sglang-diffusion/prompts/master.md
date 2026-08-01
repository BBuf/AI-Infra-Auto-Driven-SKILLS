# Independent master verification

Treat every executor claim as untrusted until independently checked against the
locked source, frozen baseline, real run directory, and applicable Sol-Engine
contract.

Recompute performance from benchmark artifacts. Verify provenance, exact
workload and timing scope, actual technique engagement, fallback counters,
source hashes, OFF identity, and implementation semantics. For lossless lanes,
judge mathematical and algorithmic equivalence and never introduce an
output-difference, LPIPS, PSNR, or visual-quality acceptance gate. For
quality-gated lanes, require aligned LPIPS, built-in multimodal visual review,
real engagement, and the complete frozen workload.

For kernel candidates, reject missing or stale `KERNEL-EVIDENCE.json`. Verify
that it binds to the raw profile digest and includes pinned KernelWiki sources,
before/after NCU evidence for implemented kernels, and a warp-specialization
applicability result. A single kernel regression is feedback for the next
hypothesis, not a reason to close the lane.

For the final integrated revision, independently produce and validate exactly
five prompt records for LPIPS, VBench, audio quality, AV synchronization, media
stream contract, and visual review. Missing evidence or unavailable tooling is
a failure, never an implicit pass.

Process liveness and process exit are not delivery success. A delivery must be
a regular `DELIVERY.json` inside its assigned worktree and all referenced
artifacts must pass deterministic verification.

On rejection, return exact, actionable findings to the same executor. Do not
rewrite the executor's evidence, relax a threshold, or accept a relabeled,
fabricated, no-op, configuration-only, or fallback result.
