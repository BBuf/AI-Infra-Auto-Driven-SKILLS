# Independent master verification

Treat every executor claim as untrusted until independently checked against the
locked source, frozen baseline, real run directory, and applicable Sol-Engine
contract.

Independently derive `mean_e2e_s = workload_total_s / request_count` from the
raw benchmark. Require exactly five successful requests and zero failed
requests, cross-check any benchmark-reported per-request latency, and compute
speedup only as frozen baseline mean divided by candidate mean. Reject legacy
or ambiguous `total_s`, `baseline_total_s`, and `candidate_total_s` fields in
new artifacts.

Recompute performance from benchmark artifacts. Verify provenance, exact
workload and timing scope, actual technique engagement, fallback counters,
source hashes, OFF identity, and implementation semantics. For lossless lanes,
judge mathematical and algorithmic equivalence and never introduce an
output-difference, LPIPS, PSNR, or visual-quality acceptance gate. For
quality-gated lanes, require aligned LPIPS, built-in multimodal visual review,
real engagement, and the complete frozen workload.

For residency candidates, reject missing or stale
`RESIDENCY-EVIDENCE.json`. Verify its profile digest, frozen GPU UUID set,
per-GPU measured free/peak/safety memory, component and DiT-layer strategy,
H2D measurements, compile/steady-state placement, conflict checks, positive
engagement, and hashed full-run/equivalence artifacts. A copied total-VRAM
threshold is not measured headroom.

For kernel candidates, reject missing or stale `KERNEL-EVIDENCE.json`. Verify
that it binds to the raw profile digest and includes pinned KernelWiki sources,
before/after NCU evidence for implemented kernels, and a warp-specialization
applicability result. A single kernel regression is feedback for the next
hypothesis, not a reason to close the lane. Kernel coverage includes compiler
and graph warmup, VAE/decode/output, scheduler/precompute/synchronization, and
the repeated DiT path; a compile-only experiment cannot close it.

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
