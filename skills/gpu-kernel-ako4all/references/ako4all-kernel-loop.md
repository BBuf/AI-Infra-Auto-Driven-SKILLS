# AKO4ALL Kernel Loop

This reference adapts the AKO4ALL framework from the SGLang diffusion AKO4ALL skill for general AI-infra GPU kernel work.

## Minimum AKO Layout

Inside `AKO4ALL/<task>/`, prefer:

- `input/reference.*`
- `input/<kernel>.*`
- `solution/<kernel>.*`
- `bench/bench_<kernel>.*`
- `context/<kernel>_notes.md`
- `ITERATIONS.md`

Use the target stack's natural extension: `.py` for Triton or CuTe DSL, `.cu`/`.cc`/`.h` for CUDA or CUTLASS, plus any build scripts needed by the microbench.

Bootstrap from [`../templates/`](../templates/README.md):

- copy `templates/ITERATIONS.md` into `<task>/ITERATIONS.md`
- copy `templates/kernel_notes.md` into `<task>/context/<kernel>_notes.md`
- copy `templates/bench_kernel.py` into `<task>/bench/bench_<kernel>.py` for Triton / CuTe DSL
- copy `templates/bench_kernel.cu` into `<task>/bench/bench_<kernel>.cu` for CUDA / CUTLASS

## Baseline Checklist

- Reproduce the current target-repo implementation in AKO first.
- Confirm the AKO baseline matches the trusted reference.
- Run the microbench before making edits.
- Capture one representative `ncu` baseline for the hottest meaningful shape.
- Use `nsys` when the suspicion is launch overhead, stream gaps, or end-to-end scheduling.
- Record the plain-language bottleneck: registers, occupancy, instruction count, DRAM, L2, shared-memory bank conflicts, launch overhead, synchronization, or layout conversion.

## Iteration Discipline

- One optimization idea per iteration.
- Re-run correctness before timing.
- Re-run the benchmark after every code change.
- Log the hypothesis, command, result, and next decision in `ITERATIONS.md`.
- Keep the current best candidate easy to identify.

Stop a direction early when:

- 3 consecutive iterations do not beat the best runtime
- correctness becomes fragile or shape-specific
- the win depends on unrepresentative shapes
- the microbench improves but the real operator or model path does not

## Validation Gate

Before calling a kernel done, validate all of:

- syntax, import, build, or compile check
- focused correctness test
- kernel or operator benchmark
- old implementation vs new implementation when this is a rewrite
- model-level or service-level benchmark when the kernel is on a production inference path

## Artifact Checklist

Keep these artifacts with the final notes:

- microbench table
- operator or model benchmark table
- `ncu` before/after summary for the representative shape
- `nsys` summary if launch gaps or overlap were involved
- one short explanation of why the kernel got faster or why a direction failed
- source reference list for any copied or adapted technique
