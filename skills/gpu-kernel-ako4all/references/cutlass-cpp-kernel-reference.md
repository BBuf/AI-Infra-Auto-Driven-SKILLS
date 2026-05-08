# CUTLASS and CuTe C++ Kernel Reference

Based on https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cutlass-cpp-kernel.

Read this file before writing, porting, debugging, reviewing, or optimizing CUTLASS or CuTe C++ kernels under `gpu-kernel-ako4all`.

## Scope

Use CUTLASS/CuTe C++ when the task involves:

- GEMM, grouped GEMM, sparse GEMM, MoE, FP8, FP4, or tensor-core-heavy kernels
- CUTLASS collectives, mainloops, epilogues, schedules, or stage count choices
- CuTe C++ layout algebra, tensor partitioning, swizzles, MMA atoms, or copy atoms
- Hopper or Blackwell pipelines, TMA, WGMMA, tcgen05, TMEM, cluster execution, or fused epilogues
- reviewing a rewrite from handwritten CUDA to CUTLASS or between CUTLASS designs

For generic CUDA debugging, also read [cuda-cpp-kernel-reference.md](cuda-cpp-kernel-reference.md).

## Source Navigation Order

Prefer targeted source study:

1. runnable examples closest to the target operation
2. `include/cutlass/gemm/collective/` for mainloop and collective-builder patterns
3. `include/cutlass/epilogue/` for output transforms and fusion
4. `include/cutlass/pipeline/` for async stage and producer/consumer structure
5. `include/cute/` for layout, tensor, atom, copy, and swizzle semantics

When adapting from a workspace CUTLASS checkout, cite workspace-relative paths in notes, not agent-local absolute paths.

## Design Questions

Before editing:

- Which existing example is the closest semantic starting point?
- Which layout, copy atom, and MMA atom define data movement?
- Which collective, kernel schedule, epilogue schedule, stage count, and cluster shape matter?
- What public operator contract must stay stable?
- Which shapes and layouts will catch the likely template or alignment bugs?

## Performance Workflow

- Read the target SM architecture reference first.
- Use `nsys` for launch gaps, poor overlap, memory copies, and fusion opportunity.
- Use `ncu` for occupancy, register pressure, shared-memory pressure, L2 hit rate, tensor-core utilization, and stall reasons.
- Only then change tile shapes, stage count, epilogue schedule, kernel schedule, or data movement.

## Rewrite Rules

- Preserve the old implementation until parity is proven.
- Treat PyTorch baseline and previous operator implementation as separate validation targets.
- Preserve shape, dtype, alignment, layout, accumulation, and epilogue semantics explicitly.
- If a rewrite changes schedules, stages, or layouts, isolate whether it improved throughput, latency, or only a narrow shape.

## Synchronization Audit

If a CUTLASS or CuTe C++ kernel uses shared memory, TMA-like movement, async-copy pipelines, or multi-stage buffering:

- inspect barrier placement
- inspect stage-slot reuse
- inspect predicate guards for partial tiles
- inspect whether producer and consumer roles can observe stale data

Shape-specific failures are often synchronization or predicate bugs, not template magic.

## Output Evidence

For final notes, include:

- source example or header pattern used as reference
- important layout, stage, schedule, and epilogue choices
- correctness result against reference and previous implementation
- benchmark table
- profiling explanation tied to architecture-specific metrics
