# CuTe DSL Kernel Reference

Based on https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cute-dsl-kernel.

Read this file before authoring, porting, debugging, or optimizing CuTe DSL Python kernels under `gpu-kernel-ako4all`.

## Core Rule

Do not guess CuTe DSL APIs from memory. Read the relevant API snapshot or source example before writing kernel code.

## Read-First Map

Core API:

- `cute.md` for layouts, tensors, `TiledMma`, `TiledCopy`, `TensorSSA`, `copy`, `gemm`, tiling, and layout algebra
- `cute_runtime.md` for DLPack conversion, runtime tensors, fake tensors, stream placeholders, dynamic shapes, and dynamic layouts
- `utils.md` for hardware info, persistent schedulers, tensor map helpers, shared-memory layouts, and SM90/SM100 helper functions

Architecture APIs:

- `cute_arch.md` for thread/block IDs, barriers, atomics, SMEM allocation, and TMEM allocation
- `cute_nvgpu_cpasync.md` for TMA and cp.async copy atoms
- `cute_nvgpu_warp.md` for warp-level APIs on SM80-SM89
- `cute_nvgpu_warpgroup.md` for SM90 warpgroup APIs
- `cute_nvgpu_tcgen05.md` for SM100+ tcgen05, TMEM loads/stores, block-scaled MMA, operand source, and major-mode enums
- `utils_sm90.md` and `utils_sm100.md` for architecture-specific shared-memory and MMA helper utilities

Concept and workflow:

- `intro.md` for decorators, JIT/kernel calling conventions, hybrid DSL compilation, meta-stage vs object-stage, and control-flow behavior
- `pipeline.md` for mbarriers, named barriers, and producer/consumer pipeline abstractions

## CuTe DSL Mental Model

- `@jit` defines host-side JIT functions.
- `@kernel` defines GPU kernels.
- Python code runs in a meta-stage to build IR; generated code runs later on the GPU.
- Use Python `print()` for compile-time inspection.
- Use `cute.printf()` for runtime GPU-side inspection.
- Use `cutlass.Constexpr` values to specialize code at JIT time.
- Use dynamic parameters when one compiled kernel should handle varying inputs.

## Control Flow

- Python `range` and `cutlass.range` lower to IR loops.
- `cutlass.range_constexpr` unrolls at compile time.
- Do not pass IR values into native Python control flow that expects concrete values.
- For data-dependent runtime conditions, use DSL-supported control flow and tensor operations.

## Implementation Workflow

Before writing code:

1. Write down tensor shapes, dtypes, memory spaces, layouts, strides, and dynamic dimensions.
2. Identify the target GPU architecture.
3. Decide whether this is a new kernel or a rewrite of an existing operator.
4. Find the closest CuTe DSL example or CUTLASS source pattern.
5. Choose kernel structure: elementwise, reduction, tiled GEMM, fused epilogue, persistent schedule, or pipeline.

Then:

1. Convert framework tensors with `from_dlpack` or the repository's established adapter.
2. Mark dynamic layout or compact dynamic shapes explicitly when needed.
3. Build layouts and tensors deliberately.
4. Use architecture APIs for thread/block IDs, barriers, SMEM, TMEM, cp.async, or TMA.
5. Keep integration separate from kernel definition where practical.
6. Run correctness before tuning.

## Pipeline and Synchronization

- Use `PipelineAsync` or specialized pipeline classes for producer/consumer staging.
- Initialize mbarriers and named barriers carefully.
- Match producer and consumer groups to the actual participating threads.
- Use tail cleanup where required so mbarrier state is not left dangling after kernel exit.
- If errors occur only for some stage counts or tail shapes, inspect barrier phase, stage reuse, and partial-tile predicates first.

## Architecture Notes

- SM89 and SM120 behave closer to warp-level and register-fragment designs.
- SM90 introduces warpgroup/TMA-style patterns.
- SM100/SM103 introduce tcgen05, TMEM, block-scaled MMA paths, TMA v2, and cluster-capable execution.
- Read [nvidia-architecture-reference.md](nvidia-architecture-reference.md) before changing tiling, stage count, or memory movement.

## Debugging

- Use Python `print()` for layout, shape, and compile-time decisions.
- Use `cute.printf()` sparingly for runtime GPU values.
- Save generated IR, PTX, or build artifacts when codegen is suspect.
- Reduce to the smallest shape that reproduces the failure.
- Compare generated behavior against a simple eager reference before tuning.

## Validation

- Add or update unit tests.
- Compare against PyTorch or another eager reference.
- For rewrites, compare against the old implementation separately from PyTorch.
- Benchmark with the same shape and dtype contract as production.
- Report which API docs and source examples were used.
