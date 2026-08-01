# Executor runtime contract

The initial assembled `goal.md` and controller-owned `DELIVERY-CONTRACT.json`
are binding. Read them in numbered precedence order. A lower-precedence
knowledge excerpt cannot override the Sol-Engine correctness contract or the
selected technique scope. On resume, the controller continues this same Codex
thread with only the new findings; do not expect the complete goal to be
replayed.

Work only in the assigned detached SGLang worktree. Preserve the frozen
baseline, workload, timing scope, real-run artifacts, source hashes, activation
evidence, and OFF identity. One round is one hypothesis, one candidate, one
real run, and one applicable gate.

The authoritative performance metric is always the arithmetic mean E2E time
per successful request over the exact five-request workload. Preserve all
three explicit values in every result: `mean_e2e_s`, `workload_total_s`, and
`request_count: 5`. Compute speedup from baseline mean divided by candidate
mean. Never optimize against, compare, or report the five-request total as if
it were a single-request latency.

The active-lane historical rule section contains manually diff-reviewed SGLang
PR patterns. Use it to generate hypotheses only. Do not copy its GPU threshold,
layer count, rank count, shape, or speed claim as applicability or acceptance
evidence.

The controller, not this process, owns campaign state and GPU scheduling. Do
not launch Sol-Engine's complete campaign flow. You may import or adapt pinned
Sol components and you are expected to write or reuse production kernel code
when the profile supports it.

For the `kernel` lane, use all three pinned KDA-Pilot evidence paths:

1. query KernelWiki for current-shape prior art and cite the exact locked pages;
2. collect and compare Nsight Compute evidence for every implemented Triton,
   CUDA/CuTe, or reused upstream kernel;
3. perform the warp-specialization applicability audit and, when applicable,
   produce its timeline plus prediction/measurement reconciliation.

Write `KERNEL-EVIDENCE.json` and include it in delivery artifacts. An explicit
non-applicability reason is allowed only where the evidence schema permits it.
The three skills complement one another and none replaces the full frozen
workload run.
Use metrics supported by the frozen GPU architecture: query the installed NCU
metric set on Hopper rather than copying Blackwell-only metric names.
Choose KernelWiki citations only from
`DELIVERY-CONTRACT.json.pinned_kernelwiki_sources`; copy both its absolute
reference path and exact digest. A run-local summary is not a pinned citation.

For the `residency` lane, match every measured GPU UUID and total-memory value
to the controller-owned `GPU-INVENTORY.json`, then begin with measured
total/minimum-free memory on each
frozen GPU, component footprints, compile and steady-state peaks, H2D traffic,
and source-current incompatibilities. Search auxiliary-component residency,
partial DiT residency, prefetch, transient compile placement, and load order as
separate candidates. Preserve selected GPU UUIDs and every parallel degree.
Write `RESIDENCY-EVIDENCE.json`; high total VRAM alone cannot pass it.

For `kernel`, search the complete load-excluded E2E profile, including regional
compile/graph/warmup, VAE decode and output finalization, scheduler/exact reuse
and synchronization, distributed layout/collective implementation under the
frozen topology, and repeated DiT kernels. Precision or approximate work stays
in its quality-gated lane.

Before a costly full workload, run a targeted correctness/microbenchmark screen
and materialize the exact activation and production source diff. An existing
flag may be used as a control, but an inert JSON/evidence file is not a patch
for behavior activated by pre-existing flags.

Do not treat process completion, a benchmark log, a claimed speedup, or the
mere existence of output media as acceptance. Write the required
`DELIVERY.json` only after its referenced durable artifacts exist. The
deterministic verifier and master independently validate the delivery. Before
exiting, run the exact `preflight_argv` from `DELIVERY-CONTRACT.json` and repair
every static finding. Passing preflight does not replace the independent gate.

If resumed, address the exact numbered master feedback in the existing Codex
thread without discarding the worktree or search ledger. Never weaken a gate to
make a candidate pass.

A rejected or slower hypothesis closes only that hypothesis. Continue through
the technique's required coverage IDs until a measured candidate is delivered,
the scientific-round budget is genuinely consumed, or `DISPOSITION.json`
contains a complete evidence-backed coverage ledger. Process launches,
dependency preflight, pre-measurement malformed output, and microbenchmarks are
not scientific rounds. A malformed evidence bundle that already contains an
authenticated complete frozen-workload measurement does consume one round;
repairing and resubmitting that same run never consumes it twice.
