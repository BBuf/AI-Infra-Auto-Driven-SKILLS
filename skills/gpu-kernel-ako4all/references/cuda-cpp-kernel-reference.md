# CUDA C++ and PTX Kernel Reference

Based on https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cuda-cpp-kernel.

Read this file before writing, debugging, reviewing, or optimizing CUDA C++ or PTX kernels under `gpu-kernel-ako4all`.

## Scope

Use CUDA C++/PTX when the task needs:

- handwritten kernels or host launch code
- CUDA Runtime or Driver API behavior
- inline PTX or instruction-level reasoning
- shared memory, bank conflicts, occupancy, async copy, TMA, WGMMA, tcgen05, or CUDA graphs
- compute-sanitizer, cuda-gdb, cuobjdump, `nsys`, or `ncu`

For CUTLASS template design, read the CUTLASS reference too. For CuTe DSL authoring, read the CuTe DSL reference too.

## Implementation Checklist

Before editing code, answer:

1. What are the exact shapes, dtypes, layouts, strides, and alignment assumptions?
2. What target SM architecture and shared-memory budget are expected?
3. Is the bottleneck compute, memory, launch overhead, synchronization, or layout conversion?
4. Which Runtime, Driver, or PTX rule must be preserved?
5. What will prove correctness and performance against both PyTorch/reference and the previous implementation?

## Kernel Template Families

Use these starting structures:

- elementwise: one thread per element, coalesced vectorized loads/stores when aligned
- row-wise reduction: one block per row, warp shuffles plus shared memory for cross-warp reduction
- tiled matrix or attention-like kernel: cooperative global-to-shared staging, register accumulation, explicit barriers
- two-pass reduction: per-block partials followed by final reduction
- stream overlap: pinned host memory plus `cudaMemcpyAsync` across multiple streams

Always add explicit error handling around launches and async Runtime calls.

## Memory and Synchronization Rules

- Coalesce global memory accesses and align to 128-byte boundaries when possible.
- Use vectorized access (`float4`, `__half2`, `__nv_bfloat162`) when alignment and tails permit.
- Use shared-memory padding such as `[32][33]` for transpose-like access patterns that would bank-conflict.
- Audit every shared-memory producer/consumer path for barriers.
- If failures depend on stage count or partial tile shape, inspect predicate coverage and shared-memory slot reuse before blaming math.
- Use FP32 accumulators for reductions and mixed-precision math.

## Runtime vs Driver API

Use Runtime API (`cudaXxx`) for most application kernels:

- device properties
- streams and events
- memory allocation
- CUDA graphs
- occupancy helpers

Use Driver API (`cuXxx`) when you need:

- explicit context management
- loading PTX or CUBIN modules at runtime
- virtual memory APIs
- tensor maps or low-level features not exposed cleanly through Runtime API

## PTX Reference Workflow

Use PTX docs when:

- inspecting generated code with `cuobjdump -ptx`
- writing inline PTX
- checking WGMMA, WMMA, TMA, mbarrier, tensor memory, or swizzle semantics
- validating register fragment layouts

Practical commands:

```bash
nvcc -ptx kernel.cu -o kernel.ptx
cuobjdump -ptx ./program > extracted.ptx
cuobjdump -sass ./program > extracted.sass
cuobjdump -res-usage ./program
```

## Debug Build Flags

Use debug flags for correctness failures:

```bash
nvcc -G -lineinfo -Xcompiler -rdynamic -O0 kernel.cu
```

Use release-with-symbols for profiling:

```bash
nvcc -O3 -lineinfo --ptxas-options=-v kernel.cu
```

## Validation Requirements

- correctness against PyTorch or a trusted eager reference
- old operator vs new operator comparison for rewrites
- performance benchmark with warmup, iterations, device, shape, dtype, and timing method
- sanitizer or debugger evidence for memory, race, or synchronization fixes
- `ncu` evidence for optimization claims

## Common Pitfalls

- using 1024-thread blocks on architectures where they reduce occupancy
- adding async copy without enough independent compute to hide latency
- assuming a paper-scale optimization transfers to a small production shape
- measuring pageable host-memory transfers while expecting overlap
- changing multiple dimensions at once and losing causality
