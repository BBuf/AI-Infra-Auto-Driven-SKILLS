# SGLang deferred MoE reduction and collective fusion

Reviewed on 2026-11-09 (Asia/Shanghai). The complete final diff, body and all
discussion were read. Current-source comparisons use SGLang
`c24285bde97c3e6424cd5d8cb35788889034d8ea` and FlashInfer
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU experiment was run for
this review. Library API availability, a framework integration proposal and
the current model's selected boundary provider are separate evidence.

## PR #39214 — Carry deferred MoE output into the next layer's collective

- Link: https://github.com/sgl-project/sglang/pull/39214
- State: closed without merge on 2026-10-18; head
  `5f53aee2a236b870ef6e9dc706628f0bf88fcaf9`, base
  `6dc7b3421b48ab962e9567cb3a595fcb7bfe1e00`.
- Coverage: all 1,053 diff lines, seven files, +876/-14; body, the sole
  issue comment and nine commits; no reviews or inline comments. The closing
  comment applies the idle-PR soft cap, not a technical rejection of fusion.
- Reviewed files: runtime
  `python/sglang/srt/arg_groups/{model_hook,overrides}.py`,
  `python/sglang/srt/layers/{communicator,flashinfer_comm_fusion}.py` and
  `python/sglang/srt/models/deepseek_v2.py`; tests
  `test/registered/unit/layers/test_flashinfer_comm_fusion.py` and
  `test/registered/unit/test_model_overrides.py`.
- Motivation: the existing DeepSeek path finalized permuted expert outputs
  and added the shared expert before a separate all-reduce/residual/RMSNorm
  boundary. The proposal carries those inputs to the next decoder layer and
  invokes FlashInfer's pure MoE reduction collective there.

The actual producer is `DeepseekV2MoE.forward_normal_dual_stream`. It needs
a nonempty, separately computed shared expert, no shared-expert TP1 path,
bypassed top-k output and a runner supporting deferred finalization. The
shared stream is joined before the payload is constructed. Forward policy
must permit MLP all-reduce fusion, and attention inputs must not be scattered.
If payload admission fails, the existing finalizer consumes the routed and
shared outputs. No cross-layer payload is returned from that fallback.

The helper admits SM100/SM103, MoE TP4/8, equal model/MoE TP, EP1, attention
DP1, PP1, attention CP1 and DCP1, with the FlashInfer TRT-LLM MoE runner and
explicit SGLang `flashinfer_allreduce_fusion_backend="trtllm"`. It excludes
two-batch overlap, torch.compile piecewise graphs and breakable graphs.
Ordinary eager execution and full decode graphs are not excluded by that
helper. Positive token count, hidden width 7168 and top-k eight are required.

GEMM2, route weights and shared output must be contiguous CUDA tensors on
one device with matching FP16/BF16 dtype. The Int32 row map can be flat
token-major or `[tokens, top_k]`. These are helper-level allowances: the
existing runner's `_make_deferred_finalize_output` requires **BF16** GEMM2
and weights. It reinterprets a legacy FP32-sized weights buffer as its live
BF16 prefix. The helper accepting FP16 does not establish an FP16 producer
in this model route.

An already initialized TRT-LLM workspace must match both process groups,
rank, world size and dtype, with adequate token/hidden bounds. Its contiguous
CUDA Int64 pointer table needs at least `3 * world_size + 1` entries, and the
manager must accept the oneshot allocation. The extra Lamport bound is
`tokens * 7168 * 2 * world_size <= 2147483647 & ~((1 << 21) - 1)`.
This is a communication-size check, not a measurement of all workspace or
temporary memory.

The API's expert-major dense input is not the runner's permuted layout:

```diff
+    active_experts_token_input = gemm2_out.index_select(
+        0, expert_major_indices
+    ).reshape(top_k, tokens, hidden_dim)
+    scale_input = expert_weights.transpose(0, 1).to(torch.float32).contiguous()
+    return active_experts_token_input, scale_input
```

Before this excerpt, the row map is transposed, made contiguous and converted
to Int64. The gather allocates `tokens * top_k * hidden_dim * 2` bytes for
the admitted input dtypes; scales become contiguous FP32 `[top_k, tokens]`.
These conversions, copies and gather launches belong in the timing window.
One collective API call does not make the whole producer-to-consumer path
one kernel. Shape checks also do not validate row-map contents; valid gather
indices remain a producer obligation.

At import, the integration requires FlashInfer's unified all-reduce/workspace
APIs and inspects `trtllm_moe_allreduce_fusion` for exactly 22 named parameters
in order, with only the final `backend` keyword-only. Missing or incompatible
APIs disable payload production before layout materialization. The consumer
calls that pure API with **`backend="cake"`**, despite the SGLang option and
workspace being named `trtllm`. It supplies scale factor one, weight bias zero,
no quantization or separate all-reduce output, and newly allocated norm and
residual outputs. PDL intent comes from the runner's architecture/token-limit
predicate; it is not unconditional.

The next `DeepseekV2DecoderLayer` consumes the dataclass with its input norm
weight and epsilon. `pre_normalized=True` bypasses the old all-reduce/norm
preparation while retaining the simple communication tail, QKV latent-input
setup and auxiliary residual capture. Payloads do not receive the Tensor-only
`_sglang_needs_allreduce_fusion` marker. These branches implement consume-once
ownership; a missing finalize or a second all-reduce is not interchangeable
with the intended fusion.

The two argument-resolution edits also exempt explicit TRT-LLM fusion from
deterministic-mode disabling while preserving explicit enforced disabling.
These files configure shared serving behavior, so their scope extends beyond
this one model caller. They do not prove deterministic equivalence of every
TRT-LLM fusion route. Frozen main has **no such exemption** in either phase:
its deterministic configuration disables the fusion selection.

### Validation and performance limits

The eight focused integration tests cover API signature, topology/workspace
boundaries, flat and two-dimensional row-map conversion, malformed shapes,
dispatch, missing-API fallback, single consumption and the producer/consumer
handoff. The happy-path test does execute CUDA allocation and layout gathering,
but mocks platform/TP/workspace and replaces the collective with a function
writing `residual + 1` and `residual + 2`. It checks plumbing, not distributed
reduction or RMSNorm numerics. Configuration tests exercise the deterministic
exception and enforced-disable behavior. The body's 8/8 focused pass belongs
to earlier SGLang head `5c272a857503971b56c99b60046f244e44154d13`.

For final SGLang head `5f53aee2a236` and FlashInfer candidate
`cc8a5d2cc35c7ad3e078e34e4a494b9ffb09ac72`, the author reports DeepSeek-V3-0324
NVFP4, TP8 on B200 and B300, deterministic Triton attention, FP8 KV,
4096-token prefill chunks and full decode graphs. Each architecture compares
16 real BF16 `[128, 7168]` norm/residual pairs, with no mismatches at
`atol=rtol=0.01`. Maximum absolute errors of 0.0625 and 0.015625 respectively
pass that combined criterion; this is not exact equality or a 0.01 maximum
absolute-error bound. TP4 and FP16 are not established by this report.

The short diagnostics report 35 decode replay observations per architecture.
The subsequent 16-question GSM8K runs report 5,221 reference and 5,184 fused
replay observations, finite normally stopped responses and consume-once checks
on all ranks. Their parser scores are 13/16 versus 15/16 on both machines.
The two gains are trailing-number parsing differences after both routes computed
the correct values. Both routes miss the same other answer. All 16 texts and
token sequences differ between routes, although they match across B200/B300
within each route. The original per-question score-equality analyses remain
**FAIL**. Preserve both the passing payload checks and this failed criterion;
neither token/logit equivalence nor a general quality improvement is established.
The body attributes the difference to low-precision reduction boundaries versus
the native FP32 reference; that explanation was not independently GPU-verified.

The recorded server cleanup return code is -9, with no remaining process group;
it is not a graceful-exit result. Diagnostic and GSM8K elapsed times include
different cleanup/recovery conditions and are explicitly not performance
comparisons. The diff has no new benchmark harness or serving-speed result.
Dependency [FlashInfer #4730](https://github.com/flashinfer-ai/flashinfer/pull/4730)
is merged and has a separate
[complete source review](flashinfer-moe-collective-source-review.md).
Its kernel timing claims cannot substitute for SGLang path measurements.

Final-head [base CI](https://github.com/sgl-project/sglang/actions/runs/34761390303)
failed at [PR gating](https://github.com/sgl-project/sglang/actions/runs/34761390303/job/103734822999):
the saved log reports the PR was draft and blocks CI. CPU and GPU test jobs
were skipped. This is neither a numerical failure of this kernel nor an
independent final-head GPU pass.

### Current source and profiler implications

Current FlashInfer still exports the pure API with the same 22-parameter
signature. Its explicit Cake branch validates FP16/BF16, SM100/103, hidden
7168, world sizes 2/4/8, positive tokens and the dense expert-major/FP32-scale
contract, then calls `run_cake_moe_allreduce_union`. The default backend remains
`trtllm`. Library TP2 support does not widen the unmerged SGLang helper's TP4/8
gate, and the export does not mean current SGLang selects this proposal.

Frozen SGLang instead has a `layer_boundary` CuTe-DSL provider. A
`MoeDeferredFinalize` handoff retains native routed rows, weights and permuted
indices through views, plus the shared contribution and an ordinary `finish`
fallback. The model installs finalize eligibility only for a supporting MoE
runner with unfused shared experts and no shared-expert TP1. Fused-shared
DeepSeek-V3/R1 configurations do not gain deferred-finalize admission from the
all-reduce/norm provider alone.

Boundary admission separately checks workspace readiness/capacity, supported
decode/extend/target-verify mode, full unscattered rows, TP greater than one,
no attention DP, CP1 and no MoE all-to-all. Finalize ordinarily requires EP1;
the EP deferred-finalize environment option is a separate gate. The model's
producer token limit and the consumer's admission both matter. Ineligible
readers call `complete()`, which invokes the saved native finalizer/reduction.

`FlashInferMNNVLCuteDSLARFusion` sends native indices/weights directly to
`allreduce_fusion` with `kMoEFinalizeARResidualRMSNorm`; it does not perform
the proposal's dense expert-major gather. Its one process-local BF16 workspace
is prepared and synchronized before graph capture. Shared-expert early-load
presets are withheld under speculative decoding because alternating draft/verify
graphs can observe an unfinished shared buffer. `launch_with_pdl=True` is caller
intent; the workspace routing profile owns the compiled choice.

For a trace, first identify the selected provider and producer eligibility.
Attribute a separate native finalizer to its fallback or unsupported producer
before calling it a missing fusion. For an experimental port of #39214, include
layout conversion, temporary bytes, shared-stream readiness and the following
collective in the measured range, and retain numerical/model-output checks.
Neither the closed proposal nor the current provider establishes a speedup
without measurements of that exact route.

Immutable source anchors:

- Proposal [admission, layout and call](https://github.com/sgl-project/sglang/blob/5f53aee2a236b870ef6e9dc706628f0bf88fcaf9/python/sglang/srt/layers/flashinfer_comm_fusion.py),
  [producer/consumer](https://github.com/sgl-project/sglang/blob/5f53aee2a236b870ef6e9dc706628f0bf88fcaf9/python/sglang/srt/models/deepseek_v2.py),
  [runner dtype contract](https://github.com/sgl-project/sglang/blob/5f53aee2a236b870ef6e9dc706628f0bf88fcaf9/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)
  and [integration tests](https://github.com/sgl-project/sglang/blob/5f53aee2a236b870ef6e9dc706628f0bf88fcaf9/test/registered/unit/layers/test_flashinfer_comm_fusion.py).
- Current [pure FlashInfer API/validator](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/comm/trtllm_ar.py),
  [native deferred handoff](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/layer_boundary/fusions/moe_finalize.py),
  [boundary admission](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/layer_boundary/fusions/cutedsl.py),
  [model dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v2.py)
  and [MNNVL service](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/flashinfer_mnnvl_cutedsl.py).
- Current deterministic [model hook](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/arg_groups/model_hook.py)
  and [override resolution](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/arg_groups/overrides.py).
