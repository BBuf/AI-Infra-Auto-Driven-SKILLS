# CuTe DSL Kernel Overview (Vendored Reference)

> **Adapter note for `gpu-kernel-ako4all`**: This file was copied from
> `https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cute-dsl-kernel`
> (`SKILL.md`). It is kept here as a **document, not an invocable skill**. The
> original frontmatter (`name`, `description`, `argument-hint`,
> `user-invocable: true`) has been removed so no skill loader treats this as a
> standalone skill.
>
> Path mapping for files this overview originally referenced:
>
> | Original (cache-dit) path                                | Path inside this skill                                |
> | -------------------------------------------------------- | ----------------------------------------------------- |
> | top-level `sm89-optimization-guide.md` / `sm90-optimization-guide.md` | `../architectures/sm89-optimization-guide.md` / `sm90-optimization-guide.md` |
> | top-level `sm100/103/120-optimization-guide.md`          | sibling `sm100/103/120-optimization-guide.md` (kept stack-specific) |
> | top-level `troubleshooting.md`                           | `../troubleshooting.md`                               |
> | sibling `cute.md`, `cute_runtime.md`, `utils.md`, `intro.md`, `pipeline.md`, etc. | same — these CuTe DSL API snapshots are bundled in this directory |
>
> The original document also referenced workspace paths under
> `vipshop/cutlass/python/CuTeDSL/`, `vipshop/cutlass/examples/python/CuTeDSL/`,
> `vipshop/cutlass/python/pycute/`, `vipshop/cutlass/include/cute/`, and
> `vipshop/cutlass/media/docs/pythonDSL/`. **No CUTLASS source tree is bundled
> in this skill.** When CuTe DSL examples or the CUTLASS Python DSL conceptual
> docs are needed beyond what's in the API snapshots here, point at a separate
> CUTLASS checkout under the user's workspace (e.g. a sibling `<base-dir>/cutlass/`
> clone of `https://github.com/NVIDIA/cutlass`) — not at any path inside this
> skill.

# Write a CuTe DSL GPU Kernel

## Goal

Use the bundled CuTe DSL API snapshots in this directory (and an optional external CUTLASS checkout) to design, implement, debug, and integrate CuTe DSL GPU kernels in a way that is reusable across projects.

## Core Rule

**Read the relevant API reference files before writing kernel code.** Do not guess CuTe DSL APIs or architecture helpers from memory when the bundled docs (or an external CUTLASS Python DSL checkout) can answer the question precisely.

## Bundled API Reference Map (sibling files in this directory)

Core API:

- `cute.md` — core CuTe DSL types and tensor / layout operations
- `cute_runtime.md` — runtime helpers and data interop (DLPack, runtime tensors, fake tensors, dynamic shapes / layouts)
- `utils.md` — helper utilities, hardware info, persistent schedulers, tensor-map helpers, SM90/SM100 helper functions

Architecture-specific:

- `cute_arch.md` — low-level architecture primitives (thread/block IDs, barriers, atomics, SMEM/TMEM allocation)
- `cute_nvgpu.md` — architecture API index
- `cute_nvgpu_warp.md` — warp-level APIs for SM80–SM89
- `cute_nvgpu_warpgroup.md` — warpgroup APIs for SM90
- `cute_nvgpu_tcgen05.md` — tcgen05, TMEM, block-scaled MMA, operand source, major-mode enums for SM100+
- `cute_nvgpu_cpasync.md` — TMA and `cp.async` copy atoms
- `cute_nvgpu_common.md` — common nvgpu helpers
- `utils_sm90.md` / `utils_sm100.md` — architecture-specific helper utilities

Concept and workflow:

- `intro.md` — decorators (`@jit`, `@kernel`), JIT/kernel calling conventions, hybrid DSL compilation, meta-stage vs object-stage, control-flow behavior
- `pipeline.md` — mbarriers, named barriers, producer/consumer pipeline abstractions

CUDA architecture / profiling:

- `../architectures/sm89-optimization-guide.md` — Ada
- `../architectures/sm90-optimization-guide.md` — Hopper
- `sm100-optimization-guide.md` — Blackwell datacenter (CuTe DSL–specific)
- `sm103-optimization-guide.md` — Blackwell Ultra (CuTe DSL–specific)
- `sm120-optimization-guide.md` — Blackwell desktop (CuTe DSL–specific)
- `../troubleshooting.md` — debugging / profiling troubleshooting

## External CUTLASS Checkout (optional, not bundled)

When the bundled API snapshots are too terse for workflow / compilation / debugging / integration questions, or when you need source examples beyond the snapshots, use an external CUTLASS checkout:

```text
<base-dir>/
├── AKO4ALL/
├── cutlass/                # external clone, e.g. NVIDIA/cutlass
└── <target-repo>/
```

Useful sub-locations there:

- `cutlass/python/CuTeDSL/` — CuTe DSL implementation sources
- `cutlass/examples/python/CuTeDSL/` — CuTe DSL examples by architecture and topic
- `cutlass/python/pycute/` — pycute helpers and layout utilities
- `cutlass/include/cute/` — CuTe C++ headers for semantic grounding
- `cutlass/media/docs/pythonDSL/` — overview, quick start, functionality, limitations, FAQs, control flow, dynamic layout, JIT arg generation, JIT caching, JIT compilation options, framework integration, AOT compilation, debugging, autotuning_gemm

Cite these as workspace-relative paths (e.g. `cutlass/python/CuTeDSL/`), not as paths inside this skill.

## CuTe DSL Mental Model

- `@jit` defines host-side JIT functions; `@kernel` defines GPU kernels.
- Python code runs in a meta-stage to build IR; generated code runs later on the GPU.
- Use Python `print()` for compile-time inspection; `cute.printf()` for runtime GPU-side inspection.
- Use `cutlass.Constexpr` to specialize at JIT time; use dynamic parameters when one compiled kernel should handle varying inputs.

## Control Flow

- Python `range` and `cutlass.range` lower to IR loops.
- `cutlass.range_constexpr` unrolls at compile time.
- Do not pass IR values into native Python control flow that expects concrete values.
- For data-dependent runtime conditions, use DSL-supported control flow and tensor operations.

## Implementation Workflow

Before writing code, answer:

1. Input/output shapes, dtypes, memory spaces, layouts, strides, dynamic dimensions?
2. Target GPU architecture?
3. New kernel or rewrite of an existing operator?
4. Closest CuTe DSL example or CUTLASS source pattern?
5. Kernel structure: elementwise, reduction, tiled GEMM, fused epilogue, persistent, or pipeline?

Then:

1. Read the relevant bundled API docs (sibling `cute*.md`, `intro.md`, `pipeline.md`, `utils*.md`).
2. If conceptual / workflow / debugging / AOT / integration / limitations questions remain, consult the optional external CUTLASS Python DSL docs.
3. Read the closest source example in the external CUTLASS checkout's `examples/python/CuTeDSL/`.
4. Decide kernel structure and write the kernel and launch path.
5. Convert framework tensors with `from_dlpack` or the project's adapter.
6. Mark dynamic layout or compact dynamic shapes explicitly when needed.
7. Use architecture APIs for thread/block IDs, barriers, SMEM, TMEM, `cp.async`, or TMA.
8. Run correctness before tuning.

## Pipeline and Synchronization

- Use `PipelineAsync` or specialized pipeline classes for producer/consumer staging.
- Initialize mbarriers and named barriers carefully.
- Match producer and consumer groups to the actually participating threads.
- Use tail cleanup so mbarrier state is not left dangling after kernel exit.
- If errors occur only for some stage counts or tail shapes, inspect barrier phase, stage reuse, and partial-tile predicates first.

## Architecture-Specific Profiling

1. **sm89 / sm120**: prioritize memory throughput, L2 hit rate, occupancy, and fusion opportunity; these targets do not have TMA, TMEM, or cluster features.
2. **sm90**: inspect whether TMA-style overlap, warpgroup execution, and shared-memory staging are actually visible in the timeline and counters.
3. **sm100 / sm103**: inspect whether tcgen05 (or WGMMA), TMEM, TMA v2, and cluster-capable execution are being used effectively.

Profiling order:

1. Read the relevant `smXX-optimization-guide.md`.
2. Use `nsys` to identify launch gaps, missing overlap, copy/compute imbalance, end-to-end bottlenecks.
3. Use `ncu` to inspect occupancy, memory throughput, L2 hit rate, register pressure, shared-memory pressure, tensor-core utilization, stall reasons.
4. Only then decide whether to change tiling, pipelining, copy strategy, or fusion structure.

## Debugging Workflow

1. Use compile-time inspection (Python `print`) for layouts, tiling, static shapes.
2. Use runtime printing (`cute.printf`) sparingly for GPU-side debugging.
3. Save PTX or IR when codegen is suspect.
4. Reduce to the smallest shape that reproduces the failure.
5. For shared memory, `cp.async`, pipeline stages, or asynchronous movement, treat synchronization as a primary suspect. When only specific shapes or pipeline configurations produce bad outputs, first inspect barrier placement, shared-stage reuse, and predicate coverage on partial-tile loads/stores.
6. Once correctness is stable, profile before tuning.

## Validation Requirements

Minimum:

1. Add/update unit tests.
2. Compare numerical accuracy against a PyTorch / trusted eager reference.
3. Compare performance against that baseline when the kernel is meant to replace or outperform it.
4. Record benchmark setup clearly.

Additional for rewrites or migrations:

1. Compare the new kernel against the pre-rewrite implementation on both accuracy and performance.
2. Treat the PyTorch baseline and the previous implementation as separate validation targets.
3. Explain any remaining gap rather than masking it with one favorable benchmark.
