# Executor runtime contract

The assembled `goal.md` is the binding prompt. Read it in numbered precedence
order. A lower-precedence knowledge excerpt cannot override the Sol-Engine
correctness contract or the selected technique scope.

Work only in the assigned detached SGLang worktree. Preserve the frozen
baseline, workload, timing scope, real-run artifacts, source hashes, activation
evidence, and OFF identity. One round is one hypothesis, one candidate, one
real run, and one applicable gate.

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

Do not treat process completion, a benchmark log, a claimed speedup, or the
mere existence of output media as acceptance. Write the required
`DELIVERY.json` only after its referenced durable artifacts exist. The
deterministic verifier and master independently validate the delivery.

If resumed, address the exact numbered master feedback without discarding the
existing worktree or search ledger. Never weaken a gate to make a candidate
pass.

A rejected or slower hypothesis closes only that hypothesis. Continue through
the technique's required coverage IDs until a measured candidate is delivered,
the scientific-round budget is genuinely consumed, or `DISPOSITION.json`
contains a complete evidence-backed coverage ledger. Process launches,
dependency preflight, malformed output, and microbenchmarks are not scientific
rounds.
