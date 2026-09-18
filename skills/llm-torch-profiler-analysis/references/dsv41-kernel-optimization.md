# DeepSeek-V4.1 profiling and optimization lessons

Use for GB300/Blackwell DSV4.1 target verify, draft and medium-batch decode.
These are shape- and revision-specific decisions, not universal CUDA rules.
Source inspection is dated **2026-09-18**; see the
[source contracts](../../../docs/upstream-source-contracts.md) for exact upstream
SHAs. This refresh ran offline tooling against saved traces; it did not rerun
GPU performance or GSM8K/AIME.

## Establish the experiment before interpreting a kernel

Record model/checkpoint revision, target vs draft, request concurrency, actual
kernel M, graph padding, input/output lengths, TP/EP/DP, quant backend and every
candidate SHA. In DSPARK, request BS=1 with block size 5 can mean six target
verify rows. A six-CTA kernel is not necessarily six requests. `TP-0` is rank
zero, not proof of TP world size or EP=1. The historical small-batch campaign
used TP4/EP1; the medium-batch curves also used EP4 and are not interchangeable.

Capture a short eager/mapping trace for source attribution and a warmed graph
trace for runtime timing. Match exact kernel/template, shapes and call site;
an eager call chain cannot by itself prove the compiled graph's dispatch.
Prove the candidate kernel ran in each intended phase/rank before reporting
performance or task accuracy as validation of that kernel.

## Kernel-to-model map

| Trace clue | Model work | Verification and optimization boundary |
|---|---|---|
| `partial_gluon`, `_combine` near WO-A | Sparse attention partial results and split reduction | Follow [Gluon attention](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py); `_combine` alone is too generic to assign. It is not automatically MoE communication. |
| `_wo_a_partial`, `_wo_a_reduce`, `_wo_a_reduce_quant`, `wo_a_mega` family | Grouped attention output projection, sometimes quantization for WO-B | Read [WO-A implementation](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/ops/attention/dsv4/wo_a_bf16.py) and the caller. A generic following CUTLASS kernel is WO-B only after checking arguments/call site. |
| `_router_triton_kernel` | Scoring, bias, top-k, normalization and output formatting after gate logits | [Router source](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/ops/moe/moe_fused_gate.py); distinguish this from the gate GEMM. Check packed IDs, live-row padding, VL token-dependent bias and epsilon. |
| `_hc_mix_stats_partial_kernel`, `_hc_mix_reduce_sinkhorn_kernel` | mHC coefficient statistics and Sinkhorn normalization | [mHC source](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/ops/layernorm/mhc.py); separate the coefficient stream from the main attention/expert stream and identify the joining wait. |
| `silu_mul_clamp_kernel`, quant, two GEMMs | Often the shared-expert gate/up → clamped SwiGLU → down chain | Verify the shared-expert caller and quant type, rather than borrowing the routed-expert kernel solely because names match. |
| `moe_finalize_all_reduce` | Routed expert weighting/sum, shared contribution, TP reduction, optional mHC epilogue | [Collective implementation](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh) and [model-side context](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/srt/layers/moe/mhc_post_fusion.py); the same family can have different epilogues. |

## PDL and the bimodal small router

For the inspected Triton router, 384 experts pad to `BLOCK_N=512`,
`BLOCK_M=1`, `num_warps=1`; grid has M programs, each with 32 threads.
Expert-axis reductions/top-k operate within a warp. Wider rows can use four
warps; changing this is not equivalent to disabling PDL.

PDL can start the consumer before preceding work completes. The consumer's
`gdc_wait` happens before dependent loads, so a measured kernel duration can
include the dependency wait and resource contention. A 3–4 µs / 13–16 µs
histogram alone cannot distinguish waiting, scheduling and arithmetic.
Fewer than four warps is a hypothesis to test, not a documented universal
PDL failure threshold. Six CTAs can still wait; low CTA count does not prove
that concurrent work cannot delay them.

[#39941](https://github.com/sgl-project/sglang/pull/39941) disabled both the
PDL launch flag and in-kernel PDL operations for SM103, sqrtsoftplus,
384 experts, top-6, `0 < M <= 8`, retaining ordinary stream dependencies.
It changed scheduling rather than router arithmetic. As of this audit the PR
is **closed, folded into open #39704**, not independently merged into main.
The inspected main/dsv4.1 snapshots must not be assumed to contain this guard.

Validate with graph replay: PDL on/off at fixed warps, then a separate warp
count sweep; isolated router versus real mHC/shared-expert overlap; all-rank
join and iteration timings; unprofiled serving repeats. Inspect Nsight Systems
for concurrency and Nsight Compute for the isolated kernel's resources/stalls.
NCU replay can serialize concurrent launches and change cache state, so it
cannot alone establish the in-service cause of the long tail.

## Shared-expert GEMMs and activation + quantization

For the historical TP4 shape, gate/up has input H=5120 and local output
`2*2304/4=1152`. Nine CTAs are consistent with nine N tiles of width 128 and
no K split, but confirm the launch/template; a persistent grid or swapped
operands changes the inference. The down projection has a different K/N
shape and must be tuned independently. Rotate layer weights or otherwise
reproduce cache state; reusing one small tensor can flatter a microbenchmark.

[MXFP8 cluster split-K](https://github.com/flashinfer-ai/flashinfer/blob/eb5f05be1f8e3ef8aa017a66dcbd5d80119b1095/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py)
provides two/four K slices and DSM reduction for supported small-M R128C4
layouts. Compare its actual selected tactic with the serving GEMM. Enabling
an available kernel family is not evidence of an improvement; extra CTAs can
compete with mHC and routed MoE. Measure the layer join and serving latency.

Fusing SiLU/clamp/mul with quant can remove a launch and BF16 write/read even
when much of that work is usually overlapped. First distinguish shared and
routed experts: [EP kernels](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/kernels/ops/moe/ep_moe_kernels.py)
already include fused masked activation/post-quant paths. Reuse requires the
same clamp order, BF16 rounding point, quant group size, UE8M0 scale rounding,
scale swizzle, padding and consumer layout. MXFP8 and NVFP4 paths are not
interchangeable. If downstream work is waiting for mHC instead, an isolated
activation win need not reduce wall time.

## WO-A, mHC and metadata opportunities

- **WO-A + quant:** the pinned V4.1 caller selects specialized BF16 projection
  paths for exact TP4 shapes `[T,2,4096] @ [2,1024,4096]`. The small target path
  can return `Mxfp8SwizzledInput` to avoid quantizing again at WO-B. Check dtype,
  strides, phase and token count at the caller, including draft warmup/capture.
- **Inverse RoPE + WO-A + quant:** open
  [#39957](https://github.com/sgl-project/sglang/pull/39957), inspected head
  `e919e8be784473d2fb7ea10d10f45c23af658f68`, uses 32 output-column clusters
  and eight K-slice CTAs per cluster, folding the RoPE window into its K tile
  and reducing FP32 partials before the output epilogue. Treat this as PR-head
  evidence, not mainline. Check int32/int64 positions, strided inputs, padded
  scales and graph buffer reuse, plus changes to its benchmark's cache flush.
- **mHC boundaries + all-reduce:** merged
  [#39370](https://github.com/sgl-project/sglang/pull/39370) is the starting
  point; open [#39704](https://github.com/sgl-project/sglang/pull/39704), head
  `46a204a5eca58ec5aba316f9c538a871ddf5be69`, extends medium-batch post/combine,
  collective epilogues and prefill overlap, and incorporates the PDL guard.
  Preserve communicator buffer/counter bounds, graph replay ordering, and
  BF16 rounding boundaries. Cross-layer cached input must be invalidated when
  Engram or row selection changes it. Compare AR-fused and non-AR-fused paths;
  a larger fusion can raise register pressure or move a wait onto the critical path.
- **Metadata:** page tables, scheduler metadata, KV commit, sharded draft
  sampling and argmax can cost more launches than their arithmetic suggests.
  Keep integer indices/ties exact, validate live versus padded rows, capture
  reuse, and batch boundaries. Removing metadata launches is distinct from
  changing attention arithmetic or replacing collectives with symmetric memory.

## Why numerical validation must follow dispatch

A fused inverse-RoPE implementation can change `y*cos - x*sin` by changing
which multiply is rounded before FMA:

```text
fma(y, cos, -round_fp32(x*sin))
fma(-x, sin, round_fp32(y*cos))
```

Both express the same real-number formula but can differ near cancellation,
then cross a BF16/quantization boundary. Test the historical rounding contract,
not merely agreement with an algebraically equivalent float reference. Do not
attribute a GSM8K score change to this mechanism without a controlled ablation.

[BBuf/sglang#11](https://github.com/BBuf/sglang/pull/11), inspected head
`fa76ccc45fd305cb2c46e762a4114bd1f2387057`, illustrates the dispatch trap:
the replacement C++ router does not cover all model extras, so an end-to-end
run can keep using Triton while its standalone benchmark uses C++. Validate
selected expert sets and dense-by-expert weighted outputs, ties, BF16 logits,
zero batch, padding and nonfinite behavior separately. Top-k slot permutation
and choosing a different tied expert set are different correctness questions.
Random finite-input agreement alone is insufficient.

Use the [paired validation protocol](../../llm-serving-auto-benchmark/references/paired-validation.md):
real acceptance for accuracy, exact matched questions and prompts, truncations
kept in the denominator, paired wins/losses, and profiler evidence that the
candidate actually executed. Similar task scores cannot validate an unused
kernel; a nonsignificant difference is not proof of equivalence.

## Interpret traces without inventing idle time or speedups

A CPU `cudaGraphLaunch` bar is host API duration, not a GPU busy interval.
Check every stream on the selected device, missing/dropped activity, other
ranks, warmup and profiler overhead before calling a blank region idle.
Compute the union of GPU intervals for busy time; summed concurrent kernel
durations can exceed wall time. Report the exposed tail at a producer/consumer
join rather than summing alleged removable kernels.

Do not automatically reuse the old two-`mhc_post_tilelang`-per-layer anchor
for V4.1: fusion can remove those launches and draft passes can add matches.
The `dsv41` pipeline profile therefore requires a verified once-per-layer
anchor. For presentation, use
[torch-profiler-layer-track](../../torch-profiler-layer-track/SKILL.md): preserve
timestamps, durations and original stream metadata. Its compact lanes are
synthetic display tracks, not newly created or removed CUDA streams.
