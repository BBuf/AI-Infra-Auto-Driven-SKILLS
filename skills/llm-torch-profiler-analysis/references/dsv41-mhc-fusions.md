# DSV4.1 mHC fusion boundaries

Source inspection: **2026-09-22**, SGLang main
[`771c9d782d9e`](https://github.com/sgl-project/sglang/commit/771c9d782d9ecf0324e70b7f5a08c32644d652c5).
Use this to interpret single-pass mHC traces and to decide where another fusion
can remove exposed work. Eligibility below describes the inspected caller,
not every kernel's standalone capability. No new GPU measurements are included.

## What single-pass overlaps

The main branch combines four residual streams with the previous sublayer's
pre coefficients, applies RMSNorm, then runs attention or MoE. In parallel,
the statistics branch derives this sublayer's post/comb coefficients and the
next sublayer's pre coefficients from the current residual. The branches must
join before post mixing consumes those coefficients.

In the common split-statistics path, `_hc_mix_stats_partial_kernel` produces
partial projection and sum-of-squares statistics. The reduction/Sinkhorn
kernel combines the partials, performs RMS scaling, affine/bias transforms,
pre/post activation and iterative normalization of the combination matrix.
That is a fused coefficient epilogue, not one kernel containing the attention
or expert GEMMs. Larger eligible batches can replace the projection with
DeepGEMM TF32 high/low components, or BF16x3 components for large prefill;
inspect `_hc_mix_stats` and the selected trace instead of assuming two fixed
kernel names for every batch.

For `M <= 8`, the side stream waits until the fused input combine/norm has
finished, reducing competition at the short input dependency. Larger decode
and verify batches can overlap statistics with the sublayer. Eligible large
prefill starts statistics after sublayer compute, immediately before its
all-reduce, to overlap communication. None of these scheduling choices removes
the coefficient dependency at post mixing.

## Which operations share a launch

`M` means rank-local kernel rows, including speculative rows and graph padding;
it is not necessarily request concurrency. These paths are shape-specific.

| Boundary | Eligible rows / phase | Fused operations | Work remaining outside |
|---|---|---|---|
| Standalone sublayer input | `1 <= M <= 96`; also V4.1 prefill-shaped `4096 <= M <= 65536` | Four-stream pre combine + RMSNorm | Coefficient statistics and sublayer compute |
| Input to an MXFP8-capable attention projection | `1 <= M <= 8` | Pre combine + RMSNorm + MXFP8 quantization; also returns BF16 input | Projection GEMM and coefficient statistics |
| Attention output, tiny TP4 | `1 <= M <= 8` | TP all-reduce + mHC post + next pre combine + RMSNorm | Statistics producer and following MoE compute |
| MoE output, tiny TP4 with deferred finalize | `1 <= M <= 8` | Routed-expert weighting/sum + shared output addition + all-reduce + mHC post; eligible next-layer handoff adds pre combine + RMSNorm + MXFP8 quantization | Expert GEMMs, shared-expert GEMMs and statistics producer |
| Attention / eligible MoE output, medium decode or target verify | `128 <= M <= 384` | All-reduce + post + next pre combine; MoE variant also includes deferred finalize | RMSNorm and quantization remain separate |
| Output after reduction, large ordinary prefill | `4096 <= M <= 65536` | Post + next pre combine + RMSNorm when a compatible next norm exists; otherwise post + combine | All-reduce and projection quantization remain separate |

The standalone pre-combine/norm kernel preserves the intermediate BF16 cast
before RMSNorm. Its small-batch implementation uses four output partitions
per row for `M <= 8`, two through M=48, then one; large prefill uses one CTA
per row to avoid repeating the combine and RMS reduction. This is independent
of the MoE router's one-warp thread model.

## Guards to inspect before transferring a result

- The specialized model paths require BF16 residuals, four HC streams and
  hidden size 5120 on supported CUDA Blackwell devices. The code also checks
  contiguity, coefficient dtype, norm semantics and batch-invariant mode.
- Collective fusions additionally need TP4, attention DP1, a registered
  compatible communicator and a sublayer that actually reduces its output.
  Medium batches need the matching fused-finalize communicator configuration.
- MoE handoff requires the supported deferred-finalize backend, no MoE
  all-to-all path and no shared-expert TP1 special case. An EP4 run cannot
  inherit the TP-only MoE fusion claim from a BS1 TP4/EP1 trace.
- Sequence parallelism and prefill context parallelism exclude these model
  shortcuts. Large post/combine prefill also checks TileLang-post selection,
  FlashInfer-mHC selection, norm compatibility and alignment.
- Quantization is included only when the next consumer accepts the swizzled
  MXFP8 tuple. BF16 normalized output alone does not prove quantization fused.
- Cross-layer reuse needs a valid next consumer. Engram, row selection and
  final-layer handling can invalidate the precomputed combined/normalized
  input; follow the actual model loop and its buffer lifetime.

Missing one launch in a trace is not sufficient evidence of which fusion ran.
Correlate the caller's state (`combined`, `normalized`, `quantized`,
`combine_only`, `overlap_only`) with the collective template/epilogue and the
next consumer. Measure the all-rank join and unprofiled iteration time;
summing overlapping kernels overstates the removable wall time.

## Source map

| Source at the inspected SHA | Read for |
|---|---|
| [deepseek_v4.py](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/srt/models/deepseek_v4.py) | `_hc_combine`, `_hc_mix_stats`, `_get_hc_stats_stream`, `_hc_post_with_combine`, `forward_hc_pre_from_prev` and cross-layer handoff eligibility |
| [mhc.py](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/ops/layernorm/mhc.py) | Statistics projection, reduction/Sinkhorn, DeepGEMM and BF16x3 dispatch |
| [hc_combine_norm.py](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/ops/layernorm/hc_combine_norm.py) | BF16 rounding, output partitions and optional MXFP8 epilogue |
| [mhc_post_combine.py](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/ops/layernorm/mhc_post_combine.py) and [prefill CUDA](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh) | Post/combine materialization and prefill norm reduction contract |
| [mhc_post_fusion.py](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/srt/layers/moe/mhc_post_fusion.py) | Scoped deferred-finalize state and statistics stream handoff |
| [all_reduce_fusion.cuh](https://github.com/sgl-project/sglang/blob/771c9d782d9ecf0324e70b7f5a08c32644d652c5/python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh) | Collective implementation, output epilogues and buffer/counter constraints |
