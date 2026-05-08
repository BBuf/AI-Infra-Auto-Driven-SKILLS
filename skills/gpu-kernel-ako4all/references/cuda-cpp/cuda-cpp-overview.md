# CUDA C++ Kernel Overview (Vendored Reference)

> **Adapter note for `gpu-kernel-ako4all`**: This file was copied from
> `https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cuda-cpp-kernel`
> (`SKILL.md`). It is kept here as a **document, not an invocable skill**. The
> original frontmatter (`name`, `description`, `argument-hint`,
> `user-invocable: true`) has been removed so no skill loader treats this as a
> standalone skill.
>
> Path mapping for the bundled materials it references:
>
> | Original (cache-dit) path           | Path inside this skill                         |
> | ----------------------------------- | ---------------------------------------------- |
> | `references/ptx-docs/`              | `references/cuda-cpp/vendored-docs/ptx-docs/`  |
> | `references/ptx-simple/`            | `references/cuda-cpp/vendored-docs/ptx-simple/`|
> | `references/cuda-runtime-docs/`     | `references/cuda-cpp/vendored-docs/cuda-runtime-docs/` |
> | `references/cuda-driver-docs/`      | `references/cuda-cpp/vendored-docs/cuda-driver-docs/`  |
> | `references/cuda-guide/`            | `references/cuda-cpp/vendored-docs/cuda-guide/`        |
> | `references/best-practices-guide/`  | `references/cuda-cpp/vendored-docs/best-practices-guide/` |
> | `references/ncu-docs/`              | `references/cuda-cpp/vendored-docs/ncu-docs/`  |
> | `references/nsys-docs/`             | `references/cuda-cpp/vendored-docs/nsys-docs/` |
> | `references/debugging-tools.md`     | `references/cuda-cpp/vendored-docs/debugging-tools.md` |
> | `references/performance-traps.md`   | `references/cuda-cpp/vendored-docs/performance-traps.md` |
> | top-level `smXX-optimization-guide.md` (sm89/sm90) | `references/architectures/smXX-optimization-guide.md` |
> | top-level `smXX-optimization-guide.md` (sm100/103/120) | `references/cuda-cpp/smXX-optimization-guide.md` (kept stack-specific) |
> | top-level `kernel-templates.md`     | `references/kernel-templates.md`               |
> | top-level `troubleshooting.md`      | `references/troubleshooting.md`                |
>
> Treat the vendored doc tree as **opt-in**: only open a file when narrowing to
> a specific PTX instruction, CUDA API, or NCU/NSYS section. Do not load the
> entire mirror into context.

# CUDA C++ and PTX Kernel Development

## Goal

Use the bundled CUDA, PTX, and profiling references in this directory to implement, debug, and optimize CUDA kernels without relying on agent-specific install paths or ad-hoc web searches.

## Bundled Reference Map (after path mapping above)

Vendored mirror, opt-in only:

- `vendored-docs/ptx-docs/` — full PTX ISA reference
- `vendored-docs/ptx-simple/` — condensed PTX quick reference
- `vendored-docs/cuda-runtime-docs/` — CUDA Runtime API reference
- `vendored-docs/cuda-driver-docs/` — CUDA Driver API reference
- `vendored-docs/cuda-guide/` — CUDA Programming Guide
- `vendored-docs/best-practices-guide/` — CUDA C++ Best Practices Guide
- `vendored-docs/ncu-docs/` — Nsight Compute docs
- `vendored-docs/nsys-docs/` — Nsight Systems docs
- `vendored-docs/debugging-tools.md` — debugging workflow notes
- `vendored-docs/performance-traps.md` — common optimization traps

Always-load architecture / template / troubleshooting files:

- `../architectures/sm89-optimization-guide.md` — Ada
- `../architectures/sm90-optimization-guide.md` — Hopper
- `sm100-optimization-guide.md` — Blackwell datacenter (CUDA-specific)
- `sm103-optimization-guide.md` — Blackwell Ultra (CUDA-specific)
- `sm120-optimization-guide.md` — Blackwell desktop (CUDA-specific)
- `../kernel-templates.md` — kernel template families
- `../troubleshooting.md` — compute-sanitizer / debugger / NCU troubleshooting

## How to Search the Bundle

Prefer narrow text search over loading large reference files into context.

1. Identify the exact instruction, API, metric, or concept.
2. Search the narrowest relevant subdirectory first.
3. Read only the matching file or a short relevant range.
4. Translate the documentation into the specific kernel or operator constraint you are implementing.

Typical search targets:

- PTX instruction syntax: `vendored-docs/ptx-docs/9-instruction-set/`
- Quick PTX lookup: `vendored-docs/ptx-simple/`
- CUDA Runtime APIs: `vendored-docs/cuda-runtime-docs/modules/`
- CUDA Driver APIs: `vendored-docs/cuda-driver-docs/modules/`
- programming-model behavior: `vendored-docs/cuda-guide/`
- profiling metrics and sections: `vendored-docs/ncu-docs/ProfilingGuide.md`
- timeline and launch behavior: `vendored-docs/nsys-docs/UserGuide.md`

## Architecture-Specific Profiling Workflow

Interpret `nsys`/`ncu` results in the context of the target architecture rather than treating all GPUs the same.

1. **sm89 (Ada)**: focus on memory throughput, L2 hit rate, kernel fusion opportunity, and the lack of TMA or cluster features.
2. **sm90 (Hopper)**: focus on TMA overlap, warpgroup behavior, shared-memory staging, and whether the timeline shows good load/compute overlap.
3. **sm100 / sm103 (Blackwell datacenter)**: focus on `tcgen05` usage, TMEM behavior, TMA v2 overlap, cluster behavior, and whether the kernel actually benefits from datacenter features.
4. **sm120 (Blackwell desktop)**: treat closer to Ada than datacenter Blackwell — watch memory throughput, L2 hit rate, shared-memory limits, lack of TMEM or cluster, and decide explicitly between TMA or `cp.async` for staging.

Recommended order:

1. Read the matching `smXX-optimization-guide.md` file.
2. Use `nsys` to identify launch gaps, overlap, copy/compute concurrency, end-to-end bottlenecks.
3. Use `ncu` to inspect architecture-specific limits.
4. Compare observations against the architecture guide before changing tile shapes, pipelines, or memory movement.

## Implementation Checklist

Before changing code, answer:

1. Exact shape, dtype, and layout contract?
2. Architectural assumptions: SM target, shared-memory budget, alignment, Tensor Core mode?
3. Is the bottleneck compute, memory, launch overhead, or synchronization?
4. Which CUDA Runtime, Driver, or PTX rules must be preserved?
5. What verification proves correctness and a sufficient performance gain?

## Debugging Workflow

1. Reproduce with the smallest failing input.
2. Confirm the failure mode: wrong value, launch error, illegal memory access, race, hang, or perf regression.
3. For shared memory / `cp.async` / async pipelines, treat data-sync bugs as the first hypothesis. If only some shapes or stage counts fail, inspect barriers, slot reuse, and partial-tile predicates before suspecting the math.
4. Use `compute-sanitizer` or `cuda-gdb` for correctness.
5. Use `nsys` first for end-to-end bottlenecks, then `ncu` for per-kernel root cause.
6. Re-run focused correctness after every change before broader benchmarks.

## Performance Workflow

Never optimize by intuition alone.

1. Establish a baseline wall-clock.
2. Use `nsys` to see where time is spent.
3. Use `ncu` to explain why a kernel is slow.
4. Change one dimension at a time: tile shape, memory movement, sync, vectorization, or epilogue.
5. Re-measure and compare against the previous version.

If the result differs across GPU generations, consult the matching `smXX-optimization-guide.md` before generalizing.

## Validation Requirements

Minimum:

1. Add/update unit tests for correctness.
2. Compare numerical accuracy against a PyTorch / trusted eager reference.
3. Compare performance against that baseline when the task claims a perf benefit or replaces a baseline path.
4. Record exact benchmark setup: shapes, dtypes, device, warmup, iterations, timing method.

Additional for rewrites:

1. Compare new vs pre-rewrite operator on both accuracy and performance.
2. Treat the PyTorch baseline and the pre-rewrite operator as separate comparison targets.
