# CUTLASS / CuTe C++ Kernel Overview (Vendored Reference)

> **Adapter note for `gpu-kernel-ako4all`**: This file was copied from
> `https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cutlass-cpp-kernel`
> (`SKILL.md`). It is kept here as a **document, not an invocable skill**. The
> original frontmatter (`name`, `description`, `argument-hint`,
> `user-invocable: true`) has been removed so no skill loader treats this as a
> standalone skill.
>
> Path mapping for files this overview originally referenced:
>
> | Original (cache-dit) path                                       | Path inside this skill                                |
> | --------------------------------------------------------------- | ----------------------------------------------------- |
> | top-level `sm89-optimization-guide.md` / `sm90-optimization-guide.md` | `../architectures/sm89-optimization-guide.md` / `sm90-optimization-guide.md` |
> | top-level `sm100/103/120-optimization-guide.md`                 | sibling `sm100/103/120-optimization-guide.md` (kept stack-specific) |
> | top-level `troubleshooting.md`                                  | `../troubleshooting.md`                               |
>
> The original document referenced workspace paths such as
> `vipshop/cutlass/include/cutlass/...`, `vipshop/cutlass/examples/...`, and
> `vipshop/cutlass/include/cute/...`. **No CUTLASS source tree is bundled in
> this skill.** When CUTLASS source study is needed, point at a separate
> CUTLASS checkout under the user's workspace (e.g. a sibling
> `<base-dir>/cutlass/` clone of `https://github.com/NVIDIA/cutlass`) — not at
> any path inside this skill.

# CUTLASS and CuTe C++ Kernel Development

## Goal

Use a **separate, user-controlled CUTLASS checkout** to understand, implement, and optimize CUTLASS- or CuTe-based C++ kernels. This document describes the navigation and review patterns; the architecture/profiling references travel with this skill, but the CUTLASS sources do not.

## Bundled Reference Map (after path mapping above)

- `../architectures/sm89-optimization-guide.md` — Ada
- `../architectures/sm90-optimization-guide.md` — Hopper
- `sm100-optimization-guide.md` — Blackwell datacenter (CUTLASS-specific)
- `sm103-optimization-guide.md` — Blackwell Ultra (CUTLASS-specific)
- `sm120-optimization-guide.md` — Blackwell desktop (CUTLASS-specific)
- `../troubleshooting.md` — debugging / profiling troubleshooting

## External CUTLASS Checkout (not bundled)

Recommended layout when CUTLASS source study is needed:

```text
<base-dir>/
├── AKO4ALL/
├── cutlass/                # external clone, e.g. NVIDIA/cutlass
└── <target-repo>/
```

Use **workspace-relative** citations in your notes, for example:

- `cutlass/include/cutlass/gemm/collective/`
- `cutlass/include/cute/layout.hpp`
- `cutlass/examples/49_hopper_gemm_with_collective_builder/`

Use absolute shell paths only inside literal command examples.

## Source Navigation Order

Prefer targeted source study over reading large header trees end-to-end:

1. `examples/` for a runnable pattern close to your target.
2. `include/cutlass/gemm/collective/` for collective-builder and mainloop config.
3. `include/cutlass/epilogue/` for fusion and output transforms.
4. `include/cutlass/pipeline/` for stage and async-copy structure.
5. `include/cute/` for layout algebra, tensor partitioning, swizzle, and atom semantics.

## Architecture-Specific Profiling

When using this overview for optimization or rewrites, do not read `nsys` or `ncu` output in isolation:

1. **sm89 / sm120**: prioritize memory throughput, L2 hit rate, occupancy, and the cost of not having cluster or TMEM-backed datacenter features; on `sm120`, decide explicitly whether TMA or `cp.async` is the better staging path.
2. **sm90**: verify the design exploits Hopper-specific staging and overlap.
3. **sm100 / sm103**: verify the kernel structure aligns with `tcgen05`, TMEM, TMA v2, and cluster-capable execution rather than only recompiling an older design.

Workflow:

1. Read the relevant `smXX-optimization-guide.md` first.
2. Use `nsys` for launch gaps, poor overlap, fusion opportunity at the operator level.
3. Use `ncu` for occupancy, memory throughput, L2 reuse, register pressure, shared-memory pressure, or architecture-specific features.
4. Only then change tile shapes, stage count, epilogue schedule, kernel schedule, or data-movement strategy.

## Implementation Workflow

Before editing code, answer:

1. Which existing CUTLASS example is the closest semantic starting point?
2. Which layout, copy, and MMA atoms define the kernel's data movement?
3. Which collective, schedule, and stage decisions matter for the target architecture?
4. What public operator contract or wrapper must remain stable?
5. What tests and benchmarks will prove the rewrite is valid?

If the kernel uses shared memory, async-copy pipelines, TMA-like staging, or multi-stage buffering, explicitly audit synchronization before blaming layout algebra or MMA semantics. When only some shapes, stage counts, or schedule variants fail, check barrier placement, stage-slot reuse, and predicate guards for partial tiles before assuming the math is wrong.

## Rewrite Guidance

1. Preserve the operator contract first.
2. Keep shape, dtype, alignment, and epilogue semantics explicit.
3. Verify the new implementation against the original operator before claiming success.
4. Only optimize after correctness and parity are established.

If the target implementation is CuTe DSL Python rather than C++ templates, use this overview for source study and switch to the CuTe DSL overview for authoring.

## Validation Requirements

Minimum:

1. Add/update unit tests.
2. Compare numerical accuracy against a PyTorch / trusted eager reference.
3. Compare performance against that baseline when the work replaces or claims to improve a baseline path.
4. Record benchmark setup clearly.

Additional for rewrites or ports:

1. Compare new vs pre-rewrite operator on both accuracy and performance.
2. Treat "PyTorch baseline" and "previous operator implementation" as separate comparisons.
3. If a rewrite changes schedules, stages, or layouts, isolate whether it improved throughput, latency, or both.
