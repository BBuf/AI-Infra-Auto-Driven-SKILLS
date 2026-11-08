# TokenSpeed DeepSeek V4/V4.1 routing and expert execution

Inspected on 2026-11-08. The complete final diff, body and all discussion were
read, then reconciled with caller, plan, provider and kernel code at TokenSpeed
`7b21034e72bb45d1edef3d3b5483b50b58c4c492`. A fresh API check matched the
head/base below. No GPU experiment was run for this refresh.

## PR #1770 — Carry route precision and output width through the MoE pipeline

- Link: https://github.com/lightseekorg/tokenspeed/pull/1770
- State: merged 2026-10-17; head `b380dde9b9724bfb1646240e2fa9c322860c7ea1`,
  base `f774d2b9f4a0824de4686fbecf4efefcba944bd2`,
  merge `a24104a75258686de74750fc511cb390543053ab`.
- Coverage: all 2,374 diff lines, 31 files, +1,252/-120; body, 11 issue
  comments, 36 reviews and 39 inline comments. Earlier review findings were
  checked against the final diff rather than copied as unresolved defects.
- Reviewed surfaces: six runtime layer/model files and their configuration
  tests; GEMM dispatch and CuTe driver; MoE plan, CUDA/Triton/Gluon/native
  routing and FlashInfer experts; CUDA router/finalizer wrappers and kernels;
  ten kernel selection, numerical, padding and graph-replay test files.

The change reduces several small operations around expert GEMMs: router
projection, top-k selection, route scaling/conversion, activation quantization
and final routed/shared addition. These have different eligibility conditions;
a trace containing one optimization does not establish that all are active.

### Router projection and top-k

`dsv4_linear_fp32` selects the shared low-latency CuTe BF16 router above the
CUDA provider when its eligibility trait is true. Both operands must be
contiguous BF16 matrices on the same CUDA device, with matching positive K,
positive N, K divisible by eight and 16-byte-aligned pointers; M is 1–32 and
the DSV4 registration requires SM90 or later. FP32 router weights do not
qualify. The existing CUDA fallback gains K=5120 alongside 3072/6144/7168,
also requires aligned operands for its custom kernel, and otherwise uses
cuBLAS. Its FP32 output reduction now has only thread zero sum shared warp
partials, fixing competing writers and incorrect repeated contributions.

The driver takes an explicit PDL policy and includes it in compilation cache
keys. Serving-time dispatch declines the exact-M dot-product path to avoid
new compilation; kernels already captured during startup remain captured.
Current main additionally selects dot-product for K too short to give every
split-K cluster rank a tile, as well as M≤4. Thus M alone no longer identifies
the low-latency kernel. Inspect the actual selected provider and graph.

The CUDA sqrt-softplus selector retains packed score/expert-ID ordering but
implements the warp maximum as two native 32-bit reductions. It removes one
winner at a time instead of sorting each lane's eight or twelve candidates.
This specialized path admits 256/384 experts, top-6, renormalization and bias
or hash routing. Correction bias affects selection, not the returned weight;
hash routing uses the checkpoint expert table. Ties retain the smaller expert
ID. Empty inputs avoid a zero-grid launch. Both wrappers pass the supplied
PDL setting through the launch attribute; the final CUDA code waits before
reading logits and signals dependents after its output-store block.

The selected expert provider declares exactly one `topk_weights_dtype` in the
plan. CUDA and Triton score, normalize and scale in FP32, then store BF16
directly when requested. Do not round before multiplying the routed scale:

```diff
     if RENORMALIZE:
         chosen_weights /= tl.maximum(tl.sum(chosen_weights, 0), 1.1754943508222875e-38)
+    chosen_weights *= scale
     tl.store(weights_ptr + token * TOPK + lanes, chosen_weights, mask=lanes < TOPK)
```

In the final Triton diff, scale is a runtime scalar rather than a compile-time
constant. Gluon/native still perform the FP32 scale and final conversion as
separate operations. Requested output types other than FP32/BF16 use FP32
backend storage followed by conversion; deferred finalization rejects such
weight types. The current facade also offers a distinct fused router/top-k
API when logits are omitted. The inspected V4.1 caller still produces logits
with `DeepseekV4MoEGate` before calling `moe_topk`; do not attribute that later
fused API to this model path merely because it exists.

### Quantization and deferred finalization

FlashInfer TRT-LLM MXFP4 SwiGLU registrations on SM100–103 request BF16 route
weights. BF16 inputs can defer finalization; FP16 inputs have a separate
registration that cannot use the BF16-only common finalizer. Precomputed
routes retain the unpacked contiguous `(int32 ids, BF16 weights)` ABI. The
in-kernel-routing deferred result needs a BF16 view of the live weight prefix
when FlashInfer allocated the backing tensor using FP32 logits dtype; a
numeric FP32-to-BF16 cast would misinterpret those packed values.

For default internal precision, the existing CuTe MXFP8 quantizer is chosen
only when available and the input is contiguous and already has the padded
hidden width. Inputs needing padding keep CUDA quantization with explicit
alignment. The separate `bf16` precision mode passes BF16 activations, padding
them if needed; it does not perform this MXFP8 conversion.

The normal V4.1 path lets shared and routed experts finish their stream-fork
work before calling the common finalizer. That kernel applies the already
scaled route weights, reduces routed rows in FP32, adds the shared result and
stores BF16. It needs the original hidden width even with no shared expert:
the permuted GEMM rows may be padded. Other model callers and the FP8-channel
internal finalizer were updated to pass the width explicitly. The final PR
also recovers missing weights locally in V4.1:

```diff
+            if expert_weights is None:
+                # Precomputed-route kernels return no weights; routing holds them.
+                expert_weights = topk_output.topk_weights
```

Current main centralizes that recovery in `MoELayer.forward` for precomputed
deferred results, covering its other model consumers too. The V4.1 admission
now uses `supports_deferred_cuda_finalize`, which checks NVIDIA as well as
the provider's capability. It still requires BF16 activations. A supplied
`out` receives a copy of the finalizer result, so this path does not yet fuse
the destination copy. MegaMoE remains a separate `do_finalize=True` path;
its later fused-shared-expert option already includes the shared result.

The common helper checks original width against padded width and shared
output, accepts FP32/BF16 weights and INT32 permutation indices, and returns
empty BF16 output without launching for zero tokens. Its vectorized kernel
requires both original and padded widths divisible by eight; the scalar
kernel handles other widths. Extra weight columns mean weighted shared
experts and require their matching 3-D output; they cannot be silently used
without shared output. Preserve these contracts when borrowing the fusion.

### Validation and performance limits

The diff adds router accuracy against FP64 reference, shape/alignment and
missing-CuTe fallbacks, serving-time compilation guards, explicit PDL-policy
forwarding, and graph replay after changing inputs. Routing tests cover
FP32/BF16 storage, scales including zero, empty inputs, bias/hash routing,
int32/int64 token IDs, expert ties, signed zero and non-finite values. They
preserve existing non-finite semantics, not a promise to sanitize infinities.
Triton tests warm one batch then check irregular-batch reuse and scale changes.

Deferred expert tests include 0/4/1025 rows, both routing modes, quantizer
selection/padding and replay with updated input. Direct BF16 weight storage
is checked against FP32 computation followed by BF16 conversion. Shared
finalization is compared with an explicit reference; routed-only output
versus the built-in finalizer uses tolerance (`atol=2e-3`, `rtol=8e-3`), not
bitwise equivalence. Width-trimming cases exercise scalar and vectorized
paths, including original widths 510/512 with padded width 544. Runtime
mocks cover unsupported deferral and missing-weight recovery. These are
reviewed test definitions, not GPU runs performed by this repository refresh.

The body reports four B300s, DeepSeek-V4.1-Flash, attention TP4/MoE TP4,
BF16 FlashInfer TRT-LLM and vllm-bench with 8,192 input/1,024 output tokens.
Values are medians of three waves after one warmup, restricted to cycles
while the full batch remains active. Without speculation, B1 changes from
5.822 to 5.375 ms and B32 from 8.793 to 8.517 ms. DSpark uses five speculative
steps and six draft tokens; B1 changes from 7.537 to 7.044 ms and B32 from
18.075 to 17.837 ms. These speculative cycles include draft and verify work;
they are not latency per accepted output token or a complete request metric.

The baseline is described as main without a SHA, and the body does not pin
these measurements to the final reviewed head or report model accuracy.
Pre-commit is reported passing. The final-head AMD benchmark comment reports
one improvement and 564 cases within budget; that is not NVIDIA execution
evidence. CI planning comments likewise do not prove that the listed GPU
regressions ran. Preserve these limits when citing the measured gains.

Current source:
[V4.1 caller](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/models/deepseek_v41_moe.py),
[expert plan consumer and missing-weight recovery](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/moe/expert.py),
[projection facade](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/__init__.py),
[CuTe registration](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/cute_dsl.py),
[CuTe driver](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/__init__.py),
[CUDA router dispatch](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/dsv3_router_gemm.cu),
[CUDA router reduction](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/dsv3_router_gemm_float_out.cu),
[top-k facade and plan](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py),
[CUDA routing](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/routing_flash.cu),
[Triton routing](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_triton/sqrt_softplus_topk.py),
[FlashInfer experts](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer.py),
[finalizer contract](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/moe.py),
[finalizer kernels](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/moe_finalize_fuse_shared.cu).
