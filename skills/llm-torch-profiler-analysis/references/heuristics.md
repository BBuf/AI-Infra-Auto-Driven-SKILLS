# Overlap Heuristics

This analyzer is intentionally conservative.

## Establish dispatch and numerical contracts

Apply these principles to every model and backend. Use the
[fusion catalog](fuse-overlap-catalog.md) for concrete source examples and the
[overlap catalog](overlap-catalog.md) for scheduling precedents.

1. Record checkpoint, framework and kernel-package revisions; target versus
   draft; actual kernel rows, graph padding, phase, TP/EP/DP and dtype/layout.
   Request concurrency is not kernel M during speculative verification.
   `TP-0` identifies a rank, not the world size. Compare matching conditions.
2. Prove the caller selected the candidate in each intended phase and rank.
   Kernel existence, a default-on flag or an isolated microbenchmark does not
   establish serving dispatch. Short eager traces supply call-site evidence;
   warmed graph traces supply runtime timing. Match signatures and shapes.
3. Follow both sides of a proposed fusion. Shared and routed experts can have
   different activation, quantization and output contracts. Preserve clamp
   order, BF16 rounding points, scale rounding/swizzle, padding, strides and
   consumer format. Returning quantized output can still materialize BF16 too.
4. Separate logical selection, page transformation and expert routing.
   Attention token top-k is not MoE top-k. Check valid lengths, `-1` padding,
   ties, signed zeros, nonfinite inputs and capture-buffer lifetime. Slot
   permutation and a different tied selection set are different questions.
5. Treat PDL as dependency scheduling. Consumer duration can include the
   in-kernel wait and resource contention. Inspect the launch flag and GDC
   operations; compare PDL on/off at fixed warp count before sweeping warps.
   A bimodal duration or tighter stream gaps alone cannot identify its cause.
6. Evaluate split-K, extra streams and larger epilogues at the layer join.
   More CTAs may compete with statistics or experts; fewer launches may leave
   the same exposed dependency. Rotate weights or reproduce cache state in
   microbenchmarks, then measure all-rank joins and unprofiled serving repeats.
7. Preserve intermediate floating-point semantics when folding RoPE, norms,
   residual mixes and quantization. Equivalent algebra can round differently
   before FMA or BF16 conversion. Test the original rounding contract, integer
   indices and graph replay, then use the
   [paired validation protocol](../../llm-serving-auto-benchmark/references/paired-validation.md).
   Task accuracy cannot validate an unused kernel; nonsignificance is not
   evidence of equivalence. Attribute a score change only after an ablation.
8. Revalidate layer anchors after fusion or speculation changes launch counts.
   Generic names such as `_combine` need caller evidence. Synthetic display
   lanes preserve timestamps and do not represent runtime stream changes.
   Use [layer tracking](../../torch-profiler-layer-track/SKILL.md) with a verified
   layer count and repeated boundary pattern.

KV capacity needs the same producer/consumer discipline: inspect payload and
scale bytes, padded page strides, compression ratios, source-layer sharing and
auxiliary state. The catalog's [paged format contract](fuse-overlap-catalog.md#paged-kv-storage-contract)
shows why bytes per stored token alone cannot determine whole-model capacity.

## What Comes From Which Trace

### Mapping trace

Used for:

- `kernel -> cpu_op -> python scope`
- launch-site call chains

This trace should be easier to read, even if it is not the exact final serving schedule.

### Formal trace

Used for:

- hidden ratio
- exclusive ratio
- overlap headroom
- ASCII timelines

This trace should reflect the real serving shape.

## What It Treats As Hidden

A kernel is treated as hidden for a segment if:

- it is active during that segment
- at least one kernel on a different stream is also active

If the overlapping kernel is compute-like, the analyzer separately records that it is hidden under compute.

## Category Heuristics

The analyzer classifies kernels by name:

- `compute`: GEMM, attention, cutlass, cublas, Triton matmul-like kernels
- `communication`: NCCL, all-reduce, reduce-scatter, all-gather, DeepEP dispatch/combine
- `elementwise`: sigmoid, top-k, gate, rmsnorm, layernorm, rope, casts
- `memory`: memcpy, memset, fill, copy
- `other`: everything else

These categories are for prioritization only.

## How To Read The Action Table

The overlap-opportunity table is intentionally not a full kernel dump.

It only keeps rows that already have an action-oriented label:

- `headroom`
- `low-roi-hidden`

It also prunes very small `headroom` rows after prioritization.

- if a `headroom` row would end up as `P5` because it is below the default `1%` share bar, it is omitted from the table
- `low-roi-hidden` rows can still remain even when they are small, because they are useful as "do not chase this first" signals

### `headroom`

Interpretation:

- the kernel still spends meaningful time exposed in the formal trace
- the mapped Python scope is a good place to inspect scheduling or fusion opportunities
- the dependency signal should still be checked before treating it as a serious overlap candidate

### `low-roi-hidden`

Interpretation:

- the kernel is already mostly hidden by another stream
- optimizing it in isolation is less likely to move end-to-end latency
- focus on fusion, launch reduction, or the surrounding schedule instead

## Dependency Signal

The table includes a dependency-oriented adjacency signal from the formal trace.

It is built from the nearest previous and next kernels on the same stream plus the mapping-trace source attribution.

Communication kernels are treated more conservatively than before:

- if a tight adjacent kernel looks like a likely producer or consumer, the table will raise the dependency risk even when the Python scope names differ
- this avoids over-claiming that an all-reduce-like kernel is a clean overlap candidate just because its neighbors map to different functions

Typical labels:

- `serial risk low`: adjacent kernels do not look like a tight same-code serial chain
- `prev-side serial risk`: the previous adjacent kernel looks tightly tied to the same code path
- `next-side serial risk`: the next adjacent kernel looks tightly tied to the same code path
- `both-side serial risk`: both sides look like a tight serial chain
- `adjacency unclear`: the timing is tight but source attribution is too weak to trust a stronger claim

Treat this as a strong heuristic, not proof of dataflow.

The readable table compresses those into shorter labels:

- `low`
- `high`
- `unclear`

The recommendation labels are also intentionally short:

- `try overlap`
- `try fusion`
- `check deps`
- `skip overlap`
- `manual check`
- `observe later`

## Important Limits

- A trace shows what overlapped, not what could legally overlap.
- Two kernels on different streams do not prove they are dependency-free.
- A mapped Python scope is a launch-site clue, not the only relevant code location.
- A hidden kernel can still matter if it changes occupancy, launch count, or surrounding schedule.

## GPU idle gaps and launch overhead

Use the [launch-overhead families](overlap-catalog.md) before interpreting an
empty GPU interval as an unfused kernel. Scheduler dispatch, phase graph
coverage, metadata glue graphs, replay-stream joins and CPU frontend work can
all expose gaps. Check worker step annotations and CUPTI completeness first.
PDL kernel time includes dependency waits; overlap duration alone does not
establish removable latency. FlashInfer main at 0.7.1 is newer than the
0.7.0.post1 framework pins: source existence does not prove installed dispatch.
