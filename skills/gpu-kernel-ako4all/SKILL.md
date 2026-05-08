---
name: gpu-kernel-ako4all
description: Use when developing, optimizing, debugging, or porting AI-infra GPU kernels through an AKO4ALL-centered loop, including Triton, CUDA C++/PTX, CUTLASS/CuTe C++, and CuTe DSL kernels; also use when setting up a sibling AKO4ALL repo, creating microbench harnesses, profiling with nsys/ncu, and validating kernel changes against real operator or model benchmarks. Do not trigger on simple Triton or CUDA API lookups; this skill is for full optimization or rewrite tasks where AKO discipline pays off.
disable-model-invocation: true
---

# GPU Kernel AKO4ALL

Use this skill to run a disciplined GPU-kernel optimization loop with AKO4ALL as the outer framework and stack-specific bundled references for Triton, CUDA C++/PTX, CUTLASS/CuTe C++, and CuTe DSL.

This is a derivative synthesis, not original material. Every upstream skill or document referenced is **copied into this skill** under `references/` and `templates/`. Do not go to `.claude/skills`, `.copilot/skills`, temporary clones, or source repositories to read the upstream skills. Read the bundled materials. Preserve the attribution in [references/source-attribution.md](references/source-attribution.md) when copying, publishing, or adapting this skill.

## Use This Skill When

- optimizing an existing kernel after a real hotspot has been proven
- writing a new Triton, CUDA C++/PTX, CUTLASS/CuTe C++, or CuTe DSL kernel for an AI-infra path
- creating an AKO4ALL microbench harness for a kernel family
- interpreting `nsys` or `ncu` results before changing tiling, memory movement, pipeline, or epilogue structure
- porting a kernel between implementation styles while preserving correctness and performance evidence

Do not start here from a vague "make it faster" request. First establish the target kernel, shape family, dtype/layout contract, hardware, and baseline runtime.

## Mandatory Reference Gate

Read the AKO loop reference before any implementation:

- [references/ako4all-kernel-loop.md](references/ako4all-kernel-loop.md)

Then read every implementation reference that matches the task. The top-level `*-reference.md` files are compact routing guides; the stack-specific subdirectories under `references/` contain the deeper bundled material.

| Kernel work | Required bundled references |
| --- | --- |
| Triton kernel or launcher | [`references/triton-kernel-reference.md`](references/triton-kernel-reference.md), [`references/triton/triton-overview.md`](references/triton/triton-overview.md), and the matching `references/triton/triton-*.md` file(s) for the specific pattern (FlashAttention, persistent matmul, fused norm, quantized GEMM, etc.) |
| CUDA C++ or PTX kernel | [`references/cuda-cpp-kernel-reference.md`](references/cuda-cpp-kernel-reference.md), [`references/cuda-cpp/cuda-cpp-overview.md`](references/cuda-cpp/cuda-cpp-overview.md), and (only on demand) narrow targets inside [`references/cuda-cpp/vendored-docs/`](references/cuda-cpp/vendored-docs/README.md) |
| CUTLASS or CuTe C++ kernel | [`references/cutlass-cpp-kernel-reference.md`](references/cutlass-cpp-kernel-reference.md) and [`references/cutlass-cpp/cutlass-cpp-overview.md`](references/cutlass-cpp/cutlass-cpp-overview.md) |
| CuTe DSL Python kernel | [`references/cute-dsl-kernel-reference.md`](references/cute-dsl-kernel-reference.md), [`references/cute-dsl/cute-dsl-overview.md`](references/cute-dsl/cute-dsl-overview.md), and the matching `references/cute-dsl/cute*.md` / `intro.md` / `pipeline.md` / `utils*.md` API snapshots |
| Profiling, debugging, correctness failure, or perf claim | [`references/profiling-debugging-reference.md`](references/profiling-debugging-reference.md); only descend into [`references/cuda-cpp/vendored-docs/ncu-docs/`](references/cuda-cpp/vendored-docs/README.md) or `nsys-docs/` when narrowing to a specific section / metric / counter |
| Architecture-specific tuning (sm89 / sm90) | [`references/nvidia-architecture-reference.md`](references/nvidia-architecture-reference.md) plus [`references/architectures/sm89-optimization-guide.md`](references/architectures/sm89-optimization-guide.md) or [`sm90-optimization-guide.md`](references/architectures/sm90-optimization-guide.md) |
| Architecture-specific tuning (sm100 / sm103 / sm120) | [`references/nvidia-architecture-reference.md`](references/nvidia-architecture-reference.md) plus the **stack-specific** `references/<stack>/sm{100,103,120}-optimization-guide.md` matching your lane (CUDA / CUTLASS / CuTe DSL) |
| Generic kernel template families | [`references/kernel-templates.md`](references/kernel-templates.md) |
| Compute-sanitizer / cuda-gdb / build flag troubleshooting | [`references/troubleshooting.md`](references/troubleshooting.md) |

For mixed implementations, read all applicable references. A CUDA wrapper around a CUTLASS kernel needs the CUDA, CUTLASS, profiling, and architecture references.

## AKO4ALL Preflight

This skill assumes a sibling layout:

```text
<base-dir>/
├── <target-repo>/
└── AKO4ALL/
```

Before any AKO work:

1. Run `scripts/ensure_ako4all_clean.sh [base-dir]`.
2. Let the script clone `AKO4ALL/` when missing (override the upstream URL with `AKO4ALL_UPSTREAM_URL` if your environment uses a fork or mirror).
3. Continue only when the AKO4ALL worktree is clean and exactly synced to its upstream default branch.
4. If the script reports divergence, local commits, tracked changes, or untracked files, stop and clean or reclone AKO4ALL before using it as an optimization harness.

## Harness Templates

Bootstrap a new AKO task by copying [`templates/`](templates/README.md) into `AKO4ALL/<task>/`:

- `templates/ITERATIONS.md` → `<task>/ITERATIONS.md`
- `templates/kernel_notes.md` → `<task>/context/<kernel>_notes.md`
- `templates/bench_kernel.py` → `<task>/bench/bench_<kernel>.py` (Triton or CuTe DSL)
- `templates/bench_kernel.cu` → `<task>/bench/bench_<kernel>.cu` (CUDA C++ or CUTLASS/CuTe C++)

These are scaffolds, not auto-discovered tools — fill in the kernel-specific bits before the first run.

## Workflow

### 1. Scope the Kernel

- Identify the exact kernel entry point, launcher, and runtime call sites.
- Record shape family, dtype, memory layout, target GPU architecture, expected alignment, and whether the path is latency-sensitive or throughput-sensitive (write these into `<task>/context/<kernel>_notes.md`).
- Keep the original implementation or a PyTorch eager equivalent as the correctness reference.
- Decide whether the bottleneck is compute, memory, launch overhead, synchronization, or layout conversion before changing code.

### 2. Pick the Implementation Lane

- **Triton** — Python-integrated, shape-specialized, fusion-heavy, or best expressed with block programs and `tl.dot`.
- **CUDA C++/PTX** — low-level memory control, runtime / driver integration, inline PTX, special synchronization, or direct CUDA debugging.
- **CUTLASS/CuTe C++** — GEMM-like, tensor-core-heavy, epilogue-rich, or wants to reuse CUTLASS collectives, schedules, or pipeline templates.
- **CuTe DSL** — Python-authored CuTe layouts, JIT/AOT behavior, or generated kernels are the intended implementation surface.

If the first lane fails to transfer to the real workload, keep the AKO record and switch lanes intentionally instead of layering more speculative changes.

### 3. Bootstrap the AKO Harness

Inside the clean `AKO4ALL` repo, create a custom harness instead of relying on stock benchmark tasks. Copy `templates/` (see above) and prefer this layout:

- `input/reference.*` — trusted reference
- `input/<kernel>.*` — current baseline (copied from target repo)
- `solution/<kernel>.*` — candidate(s)
- `bench/bench_<kernel>.*` — per-shape timing + correctness
- `context/<kernel>_notes.md` — contract, profiler findings, failed attempts
- `ITERATIONS.md` — hypothesis / change / result / decision per iteration

The benchmark must cover representative production shapes, tail shapes, dtype variants, and the fastest known baseline.

### 4. Establish the Baseline

- Run the AKO microbench before editing.
- Capture one representative `ncu` report for the hot shape.
- Use `nsys` when launch overhead, CPU/GPU gaps, stream overlap, memory copies, or many small kernels may matter.
- Record the baseline in `ITERATIONS.md` with device, shapes, dtype, warmup, iterations, median (p50), and p95 when latency matters.

### 5. Iterate

- Change one idea at a time.
- Re-run correctness before performance after every code change.
- Re-run the microbench after every accepted change.
- Update `ITERATIONS.md` with the hypothesis, exact result, and decision.
- After 3 consecutive no-improvement iterations, rerun `ncu`, reread the iteration log, and change direction (or switch lane).

### 6. Port Back

- Port only the winning candidate into the target repo.
- Keep integration separate from kernel logic where possible.
- Preserve the old path long enough to compare accuracy and performance for rewrites or migrations.
- Keep all shape, dtype, layout, alignment, and hardware assumptions explicit in code or tests.

### 7. Validate and Hand Off

Minimum validation:

- syntax / import / build check for the modified path
- focused correctness test against the trusted reference
- kernel or operator benchmark against the previous implementation
- `ncu` or `nsys` evidence when claiming a performance root cause

For model-serving or diffusion paths, also run the smallest real model-level benchmark that proves the kernel win transfers beyond the microbench. Hand off as follows:

- **LLM serving end-to-end**: pair with `llm-serving-auto-benchmark` (per-framework deployment search) and `sglang-sota-performance` (model-level SOTA loop).
- **Trace / profiler analysis**: pair with `llm-torch-profiler-analysis` to convert a torch-profiler trace into a kernel / overlap / fusion table.
- **SGLang diffusion paths**: pair with `sglang-diffusion-ako4all-kernel` for diffusion-specific validation (denoise step accuracy, scheduler interaction). This skill stays the home for cross-stack AKO discipline; that skill carries the diffusion-specific gates.

## Output Expectations

When finishing work under this skill, report:

- which required references were read (file paths inside this skill)
- implementation lane and files touched in the target repo
- correctness comparison and tolerances against the trusted reference (and against the previous operator if this is a rewrite)
- baseline vs optimized benchmark table copied or summarized from `ITERATIONS.md`
- path to (or a short excerpt from) the final `ITERATIONS.md` so the iteration history is reviewable
- profiling explanation for the speedup or regression (`ncu` summary delta and/or `nsys` summary delta)
- remaining risks, especially untested GPU architectures, layouts, dtype variants, or tail shapes

## Operating Rules

- AKO4ALL cleanliness is a gate, not a suggestion.
- Profiling evidence drives optimization direction; intuition does not.
- One optimization idea per iteration.
- Failed attempts are useful artifacts; record them in `ITERATIONS.md`.
- Do not keep an AKO-only win that fails to improve the real operator or model workload.
- Vendored docs under `references/cuda-cpp/vendored-docs/` are opt-in: open one file at a time when narrowing to a specific instruction, API, or metric — never load the whole mirror.
