# SGLang MiniMax M2 / M2.5 / M2.7 Support and Optimization Timeline

This document is based on the latest SGLang `origin/main` snapshot `47c4b3825`, plus patch-level reading of MiniMax-related merged and open PRs. It covers the main line originally represented by the `sglang-minimax-m2-m25-optimization` skill and adds the latest MiniMax M2.7, TP QK RMSNorm allreduce fusion, DP attention, FP4/NVFP4, NPU, DeepEP, EPLB, and tool-call streaming state.

The short conclusion is: as of `47c4b3825`, the mainline MiniMax M2-series model file is `python/sglang/srt/models/minimax_m2.py`. It already supports base loading for M2/M2.1/M2.5, tool calling, reasoning parsing, Eagle3 aux hidden states, PP, DP-attention-related attention-TP grouping, M2.5 reduce-scatter/FP4 all-gather/AR fusion, and the TP QK RMSNorm allreduce fusion you mentioned. M2.7 currently appears mostly as documentation and reuse of the same model class.

## 1. Chronological Overview

| Created    |     PR | State  | Track            | Code Area                                                     | Effect                                                                                                       |
| ---------- | -----: | ------ | ---------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 2025-10-25 | #12129 | merged | M2 bring-up      | `models/minimax_m2.py`, `function_call/minimax_m2.py`, parser | Added the MiniMax M2 model, tool-call parser, reasoning parser, and docs.                                    |
| 2025-10-27 | #12186 | merged | Precision        | `MiniMaxM2RMSNormTP`                                          | Multiplied RMSNorm weight before casting back to the original dtype for better precision.                    |
| 2025-11-07 | #12798 | merged | Eagle3           | `minimax_m2.py`, memory cache                                 | Added `aux_hidden_states` capture.                                                                           |
| 2025-11-14 | #13297 | merged | Eagle3           | `minimax_m2.py`                                               | Added the missing `get_embed_and_head`.                                                                      |
| 2025-11-25 | #13892 | merged | DeepEP call      | `MiniMaxM2MoE.forward_deepep`                                 | Fixed DeepEP MoE forward arguments by passing `TopKOutput` instead of split top-k tensors.                   |
| 2025-11-27 | #14047 | merged | Router           | `layers/moe/topk.py`, `minimax_m2.py`                         | Added `topk_sigmoid` and the `scoring_func="sigmoid"` path.                                                  |
| 2025-12-04 | #14416 | merged | QK RMSNorm       | `minimax_m2.py`                                               | Fused q/k RMSNormTP sumsq, allreduce organization, and apply.                                                |
| 2026-01-05 | #16483 | merged | QK RMSNorm       | `rms_sumsq_serial`                                            | Added 512-aligned padding for the RMSNormTP allreduce buffer.                                                |
| 2026-01-27 | #17826 | open   | PP + DP          | `minimax_m2.py`                                               | Open PR for Pipeline + Data Parallelism; some ideas were absorbed by later mainline PRs.                     |
| 2026-02-04 | #18217 | merged | PCG              | `fp8_kernel.py`, `minimax_m2.py`                              | Added MiniMax-M2 piecewise CUDA graph support.                                                               |
| 2026-02-27 | #19468 | open   | DeepEP           | server args, CI, MiniMax config                               | Open PR to support DeepEP with MiniMax models.                                                               |
| 2026-02-28 | #19577 | merged | PP               | `minimax_m2.py`                                               | Added official PP support for the MiniMax M2 series.                                                         |
| 2026-03-02 | #19652 | merged | NVFP4            | quantization, Marlin fallback                                 | Runs NVFP4 on non-Blackwell GPUs through Marlin fallback.                                                    |
| 2026-03-06 | #19995 | merged | Loader           | `minimax_m2.py`                                               | Added `packed_modules_mapping`.                                                                              |
| 2026-03-06 | #20031 | open   | Loader           | `minimax_m2.py`, weight test                                  | Supports AWQ merged expert `w13` weight loading.                                                             |
| 2026-03-07 | #20067 | merged | M2.5 distributed | `layernorm.py`, `minimax_m2.py`, test                         | Added DP attention, DP reduce-scatter, FP4 all-gather, and prepare_attn AR fusion for M2.5.                  |
| 2026-03-13 | #20489 | open   | DP attention     | `minimax_m2.py`, runner, memory pool, rotary                  | Fixes MiniMax M2 DP-attn attention-TP, empty batch, and related issues.                                      |
| 2026-03-16 | #20673 | merged | TP QKNorm        | `jit_kernel/all_reduce.py`, `tp_qknorm.cuh`, `minimax_m2.py`  | Added JIT fused TP QK RMSNorm + custom allreduce.                                                            |
| 2026-03-18 | #20870 | merged | Loader           | `minimax_m2.py`                                               | Fixed KV cache scale loading being swallowed by qkv renaming.                                                |
| 2026-03-18 | #20873 | open   | M2.7 docs        | docs                                                          | Adds MiniMax-M2.7 and M2.7-highspeed to the old docs.                                                        |
| 2026-03-19 | #20905 | merged | NPU/ModelSlim    | `modelslim.py`, `minimax_m2.py`                               | Adapts the w2 quant layer suffix for Minimax2.5.                                                             |
| 2026-03-20 | #20967 | merged | TP16 bugfix      | `MiniMaxM2RMSNormTP`                                          | Fixed repeated output under TP16 caused by KV-head replication.                                              |
| 2026-03-20 | #20975 | open   | DP attention     | DP attention follow-up fixes                                  | Successor to `#20489`, continuing DP-attn, rotary empty batch, and rank-buffer fixes.                        |
| 2026-04-08 | #22300 | open   | FP8 GEMM         | `fp8.py`, `fp8_utils.py`, loader utils                        | Fixes FP8 GEMM performance/correctness issues from incorrect DeepGEMM UE8M0 scale conversion on fp16 models. |
| 2026-04-09 | #22432 | open   | NPU              | `split_qkv_tp_rmsnorm_rope`                                   | Fuses split qkv, TP RMSNorm, and RoPE on NPU.                                                                |
| 2026-04-14 | #22744 | open   | NVIDIA TF32      | server args, model runner                                     | Adds `--enable-tf32-matmul` to improve MiniMax gate GEMM performance.                                        |
| 2026-04-16 | #22934 | open   | EPLB             | `minimax_m2.py`                                               | Adds the EPLB routed expert weight interface for MiniMax.                                                    |
| 2026-04-20 | #23190 | open   | NPU + Eagle3     | `split_qkv_tp_rmsnorm_rope`, hidden states capture            | Successor to `#22432`, adding NPU empty-batch and DP-attn Eagle3 hidden-state capture fixes.                 |
| 2026-04-21 | #23301 | open   | Tool calling     | `function_call/minimax_m2.py`                                 | Streams string parameters token by token to reduce tool-call argument latency.                               |

## 2. MiniMax M2 Bring-up: Model Structure, Parser, and Weight Loading

`#12129` is the starting point for MiniMax M2 support. It added `python/sglang/srt/models/minimax_m2.py`, matching the MiniMax checkpoint structure:

- `MiniMaxM2RMSNormTP`: TP-aware RMSNorm for Q/K normalization.
- `MiniMaxM2MLP` and `MiniMaxM2MoE`: MiniMax layers are MoE layers and do not have DeepSeek-style shared experts.
- `MiniMaxM2Attention`: QK normalization, partial RoPE, `QKVParallelLinear`, and `RadixAttention`.
- `MiniMaxM2DecoderLayer`: attention followed by MoE, with TBO operations for gate, expert selection, and expert compute.
- `MiniMaxM2Model` and `MiniMaxM2ForCausalLM`: embedding, layers, norm, lm head, logits processor, and weight loading.

The same PR also added `python/sglang/srt/function_call/minimax_m2.py`. MiniMax M2 tool calls are not OpenAI-style JSON; they use an XML-like block:

```xml
<minimax:tool_call>
<invoke name="func1">
<parameter name="param1">value1</parameter>
</invoke>
</minimax:tool_call>
```

`MinimaxM2Detector` therefore has to recognize `<minimax:tool_call>`, `<invoke name="...">`, and `<parameter name="...">`. The initial streaming parser tracks state such as `_in_tool_call`, `_current_parameters`, and `_streamed_parameters`, gradually emitting the tool name and argument JSON fragments. The reasoning parser also initially registered `minimax-m2`, although the actual behavior later became closer to Qwen3-style thinking.

The initial weight loader handled several name mappings:

- `q_proj/k_proj/v_proj` are stacked into `qkv_proj`.
- `gate_proj/up_proj` are stacked into `gate_up_proj`.
- MoE expert `w1/w2/w3` map to gate/down/up.
- `rotary_emb.inv_freq` is skipped.
- Extra GPTQ bias tensors are skipped tolerantly.

`#19995` later exposed these stacking relationships explicitly:

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

This allows quantization, loaders, and external tools to know directly from the model class which checkpoint modules are packed.

`#20870` fixed KV cache scale loading. KV scales in checkpoints are named like `self_attn.k_proj.k_scale` / `self_attn.v_proj.v_scale`, but the stacked mapping loop used to rename `k_proj` to `qkv_proj` first, causing `maybe_remap_kv_scale_name` to miss the original pattern. The PR adds `_is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")`; for KV scales it skips qkv renaming and lets the original names reach `maybe_remap_kv_scale_name`, which then maps them to `self_attn.attn.k_scale/v_scale`.

`#20031` is still open and targets AWQ merged expert weights. Some checkpoints merge gate/up into `w13` instead of storing separate `w1/w3`. The PR uses `FusedMoE.make_expert_params_mapping_fused(ckpt_gate_up_proj_name="w13", ckpt_down_proj_name="w2", ...)` and tries the fused mapping before the old `w1/w2/w3` mapping.

## 3. Router and DeepEP Calls: Sigmoid Top-k and Unified TopKOutput

MiniMax M2 routing uses sigmoid scoring rather than the default softmax. `#14047` adds a `scoring_func` field to `TopKConfig` and supports `scoring_func="sigmoid"` across `topk.py`:

- CUDA/HIP imports `topk_sigmoid`.
- `fused_topk_torch_native` abstracts `scoring_func_impl`, supporting softmax and sigmoid.
- `fused_topk` calls `topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize, correction_bias)` when `scoring_func == "sigmoid"`.
- `select_experts` threads `topk_config.scoring_func` down to the fused top-k implementation.
- `MiniMaxM2MoE` removes the early workaround of setting `use_grouped_topk=True, num_expert_group=1, topk_group=1` and directly relies on sigmoid scoring.

`#13892` fixes the DeepEP MoE forward protocol. Earlier code unpacked `self.topk(...)` into `topk_weights, topk_idx, _` and passed them separately into `self.experts(topk_idx=..., topk_weights=...)`. The mainline MoE runner later standardized on `TopKOutput`, so the PR changed it to:

- `topk_output = self.topk(...)` for non-empty token batches.
- `topk_output = self.topk.empty_topk_output(device=hidden_states.device)` for empty token batches.
- experts call unified as `self.experts(hidden_states=hidden_states, topk_output=topk_output)`.

This lets normal MoE, DeepEP MoE, empty batches, and later EP/DP extensions share the same top-k data structure.

`#19468` remains open and aims to officially support DeepEP with MiniMax models. The patch touches server args, CI DeepEP installation, and MiniMax hidden-size / BF16 requirements. Current main already routes `MiniMaxM2MoE.forward` to `forward_deepep` when `get_moe_a2a_backend().is_deepep()` or Ascend FuseEP is enabled, but complete DeepEP readiness still depends on this track being merged and tested.

## 4. QK RMSNorm: From Precision Fix to TP Allreduce Fusion

`#12186` is a one-line precision fix, but an important one. The old logic was:

```python
x = x.to(orig_dtype) * self.weight
```

The PR changes it to:

```python
x = (x * self.weight).to(orig_dtype)
```

Now the weight multiplication happens on the fp32 normalized tensor, and only the final result is cast back to the original dtype.

`#14416` is the first Q/K RMSNormTP fusion. MiniMax attention applies RMSNorm to q and k, and under TP the variance must be aggregated across ranks. The PR adds Triton kernels:

- `rmsnorm_sumsq_kernel_serial`: computes q and k sum of squares together and writes fp32 `[B, 2]`.
- `rmsnorm_apply_kernel_serial`: reads the allreduced sumsq, applies `rsqrt(sum_sq / full_dim + eps)`, and multiplies q/k by their weights.
- `rms_sumsq_serial` and `rms_apply_serial` Python wrappers.
- `MiniMaxM2RMSNormTP.forward_qk`, which normalizes q/k together and reduces launch/allreduce organization overhead.
- `MiniMaxM2Attention.forward_prepare` calls `forward_qk` when `use_qk_norm` is enabled instead of calling q_norm and k_norm separately.

`#16483` optimizes this allreduce buffer. SGLang custom allreduce `sglang::cross_device_reduce_1stage` needs alignment, and MiniMax RMSNormTP reduces a `[B, 2]` fp32 tensor. The PR pads the element count to 512 alignment, avoiding performance and boundary issues for unaligned custom allreduce sizes. The PR description reports roughly 6% throughput improvement on M2.1.

`#20967` fixes repeated output under TP16. The root cause is that when attention TP size is larger than the number of KV heads, KV heads are replicated across ranks, but the old RMSNormTP weight loader still sharded directly by rank. The PR aligns `MiniMaxM2RMSNormTP` with `QKVParallelLinear`:

- If `attn_tp_size >= num_heads`, require `attn_tp_size % num_heads == 0`, keep one logical head per rank, and set `num_head_replicas = attn_tp_size // num_heads`.
- Otherwise require `num_heads % attn_tp_size == 0`, assign `num_heads // attn_tp_size` heads per rank, and set `num_head_replicas = 1`.
- The weight loader uses `shard_id = attn_tp_rank // num_head_replicas`, so replicated ranks load the same shard.
- Forward allreduce uses the attention TP group rather than blindly using global TP.

`#20673` is the “allreduce TP norm” optimization you mentioned, and it has been merged. It adds the JIT kernel `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh` and exposes the following in `python/sglang/jit_kernel/all_reduce.py`:

- `fused_parallel_qknorm`
- `get_fused_parallel_qknorm_max_occupancy`

MiniMax adds `MiniMaxM2QKRMSNorm`:

- The default `_forward_naive` still runs `rms_sumsq_serial(q, k)`, `attn_tp_all_reduce(sum_sq)`, and `rms_apply_serial(...)`.
- When `world_size > 1`, the device is CUDA, and `SGLANG_USE_FUSED_PARALLEL_QKNORM` is true, it uses the JIT fused path.
- The fused path first queries max occupancy from dtype, world size, and full q/k dimensions.
- It creates `CustomAllReduceV2` with the attention TP group.
- Max push size is derived from `chunked_prefill_size`, `context_len`, and `max_prefill_tokens`, then aligned to 512.
- At runtime, the registered custom op `fused_tp_qknorm` calls `fused_parallel_qknorm(COMM_MAP[counter].obj, q, k, q_weight, k_weight, eps)`, combining q/k norm and cross-TP reduction in one JIT kernel/communication path.

This PR also adds `test_tp_qknorm.py` and `bench_tp_qknorm.py`. The PR description reports decode throughput improving from 150 tps to 157 tps, making it one of the most important MiniMax QKNorm optimizations.

## 5. PP, DP Attention, M2.5 Distributed Path, and PCG

`#19577` is the official MiniMax PP merge. It does several things:

- `MiniMaxM2Model` uses `make_layers`, producing `self.layers, self.start_layer, self.end_layer`.
- Non-last PP ranks use `PPMissingLayer` for `norm` and `lm_head`.
- Forward accepts `pp_proxy_tensors`; the first rank starts from embeddings, while non-first ranks read `hidden_states/residual` from the proxy.
- Non-last ranks return `PPProxyTensors({"hidden_states": hidden_states, "residual": residual})`.
- `load_weights` uses `get_layer_id(name)` to skip layer weights outside the current PP shard.

`#17826` is an open PP + DP PR. It was not merged, but ideas such as attention-TP rank/size, `is_dp_attention_enabled()`, and whether embedding/lm head should use the attention TP group were gradually absorbed into later mainline PRs.

`#20067` is the main MiniMax-M2.5 distributed optimization PR. Its title is direct: DP attention, DP reduce-scatter, FP4 all-gather, and AR fusion in prepare_attn. Current main shows these results:

- `MiniMaxM2Attention` initializes QKV and O projection with `get_attention_tp_rank()` / `get_attention_tp_size()` instead of default global TP.
- `VocabParallelEmbedding(..., use_attn_tp_group=is_dp_attention_enabled())` makes embedding operate on the attention TP group in DP attention mode.
- `MiniMaxM2DecoderLayer` creates `LayerCommunicator(..., allow_reduce_scatter=True)`.
- MoE forward receives `should_allreduce_fusion` and `use_reduce_scatter`; when the next layer can fuse allreduce or the current layer can use reduce-scatter, it does not immediately run `tensor_model_parallel_all_reduce` inside MoE.
- When `should_use_flashinfer_cutlass_moe_fp4_allgather()` is true, MoE also skips its internal allreduce so the FP4 all-gather path can own communication.
- Registered tests cover M2.5 shapes such as TP8+EP8 and TP8+DP8+EP8+DP-attention.

`#18217` adds piecewise CUDA graph support for MiniMax-M2. It handles config lookup in `fp8_kernel.py` under Dynamo compiling and replaces expert distribution recorder contexts with `nullcontext()` in MoE expert selection, TBO ops, and the model forward loop where PCG capture would otherwise see incompatible dynamic context.

`#20489` and `#20975` are still-open DP attention fix lines. Their patches include:

- MiniMax attention uses attention TP size/group for head partitioning and communication instead of global TP.
- `model_runner` initializes `global_num_tokens_gpu` with `dp_size` when `require_attn_tp_gather` is true, avoiding invalid device ordinal/access on higher ranks.
- memory pool and rotary embedding handle empty batches to avoid 0-sized tensor view errors.
- follow-up patches fix function names from `get_attention_tp_world_size` to the actually available `get_attn_tensor_model_parallel_world_size` / `get_attn_tp_group`.

Current main already has many attention-TP and empty-hidden-state protections, but these PRs show that DP-attn boundaries are still being refined.

## 6. Eagle3, M2.7, and Tool-call Streaming

`#12798` adds Eagle3 aux hidden state capture for MiniMax M2:

- `MiniMaxM2Model` adds `layers_to_capture`.
- In the forward loop, if the current layer id is in the capture list, `hidden_states + residual` is appended to `aux_hidden_states`.
- `MiniMaxM2ForCausalLM.set_eagle3_layers_to_capture` sets default capture layers `[2, num_layers // 2, num_layers - 3]`, or uses caller-provided layer ids.
- The logits processor receives `aux_hidden_states` for Eagle3.

`#13297` adds `get_embed_and_head`, returning `self.model.embed_tokens.weight, self.lm_head.weight`, so Eagle3 can access the main model embedding and lm head.

`#20873` is an open old-docs PR that adds MiniMax-M2.7 and M2.7-highspeed. Although this PR is not merged, latest main already contains `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.7.mdx`, and the cookbook navigation includes MiniMax-M2.7. At the code level, M2.7 still reuses the `MiniMaxM2ForCausalLM` model family.

`#23301` is the new open tool-call streaming PR. It rewrites `MinimaxM2Detector.parse_streaming_increment` so string parameters can stream token by token:

- Adds `_STREAM_HOLD_BACK = len("</parameter>") - 1` to avoid emitting a partial end tag as parameter content.
- Adds fine-grained state such as `_in_parameter`, `_current_param_name`, `_param_raw_sent_len`, `_current_param_is_string`, and `_first_param_started`.
- For string parameters, once `<parameter name="...">` is seen, it emits the JSON key prefix and then incrementally JSON-escapes and appends value content.
- Non-string parameters such as int, bool, object, and array are still buffered until `</parameter>` and converted once.
- At `</invoke>`, it closes the JSON object with `}` when needed.

The value of this PR is not model throughput; it improves agent/tool-use experience because long string arguments no longer have to wait for the full `</parameter>` before any argument text is returned.

## 7. Quantization, NPU, TF32, EPLB, and Other Open Optimization Tracks

`#19652` is generic but important for MiniMax M2.5: NVFP4 Marlin fallback. It adds `marlin_utils_fp4.py`, allowing non-Blackwell GPUs starting at SM75 to run NVFP4 through Marlin FP4 fallback:

- Detects Blackwell; if the GPU is not Blackwell but supports Marlin FP4, fallback is enabled.
- Processes NVFP4 scales by converting FP8-S1E4M3 scales into the FP8-S0E5M3 format better suited for Marlin dequantization.
- Repacks linear and MoE weights into Marlin tile layout.
- The MoE fallback builds `MarlinMoeQuantInfo` so fused Marlin MoE can use FP4 scalar type and global scales.
- Adds `test_nvfp4_marlin_fallback.py` for linear and MoE coverage.

`#22300` is an open FP8 GEMM scale fix. The issue is that if weight scales are converted at load time into the UE8M0/R128c4 packed format required by DeepGEMM, but runtime falls back to Triton because fp16 output dtype, K shape, or backend constraints are not satisfied, Triton still expects ordinary fp32 scales. That can cause wrong results or performance issues. The PR makes `should_deepgemm_weight_requant_ue8m0` check output dtype and weight shape, and makes FlashInfer/TRTLLM fallback detect `weight_scale.format_ue8m0`.

`#20905` is the NPU ModelSlim track. MiniMax2.5 checkpoint MoE quant descriptions may use suffixes like `.0.w2.weight` instead of ordinary `.0.gate_proj.weight`. The PR updates ModelSlim MoE scheme detection so `W4A4_DYNAMIC`, `W4A8_DYNAMIC`, and `W8A8_DYNAMIC` can be recognized from MiniMax's w2 suffix.

`#22432` and `#23190` are the NPU fused attention-prepare track. They introduce `sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope.split_qkv_tp_rmsnorm_rope`, which fuses qkv split, TP RMSNorm, and RoPE after qkv projection inside `forward_prepare_npu`. `#23190` also adds empty-hidden-state short-circuiting and fixes Eagle3 hidden state capture under DP-attn.

`#22744` is the NVIDIA TF32 gate GEMM optimization. It adds `--enable-tf32-matmul`, and model runner calls `torch.set_float32_matmul_precision("high")`. The PR description reports MiniMax gate GEMM FP32 overhead dropping from 9.1% to 3.3%, and batch64 throughput rising from 3076.99 to 3302.03 tok/s.

`#22934` is the open MiniMax EPLB bugfix. It adds `get_moe_weights` to `MiniMaxM2MoE`, using `filter_moe_weight_param_global_expert` to filter local/redundant expert weights. `MiniMaxM2ForCausalLM` gains `_routed_experts_weights_of_layer = LazyValue(...)` and a `routed_experts_weights_of_layer` property. Current main already has a similar wrapper interface for Kimi K2.5, but the MiniMax version has not landed yet.

## 8. Current Main Code Shape

As of `47c4b3825`, the MiniMax mainline looks like this:

- `MiniMaxM2ForCausalLM` is the shared model class for M2/M2.1/M2.5/M2.7.
- `MiniMaxM2MoE` uses sigmoid top-k, `TopKOutput`, normal/DeepEP branches, and communication control for reduce-scatter and FP4 all-gather.
- `MiniMaxM2Attention` uses attention TP rank/size and supports DP-attention head partitioning; Q/K RMSNorm goes through `MiniMaxM2QKRMSNorm`, with JIT fused TP QKNorm allreduce enabled by `SGLANG_USE_FUSED_PARALLEL_QKNORM`.
- `MiniMaxM2DecoderLayer` uses `LayerCommunicator` for prepare_attn AR fusion, prepare_mlp, reduce-scatter, and postprocess.
- The loader supports packed mapping, KV scale remapping, and PP shard skipping; AWQ `w13` merged expert loading is still open.
- M2.7 appears in docs while the code still reuses the same M2-series implementation.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `MiniMax M2 series` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-25 | [#12129](https://github.com/sgl-project/sglang/pull/12129) | merged | Support MiniMax M2 model | model wrapper, docs/config | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2025-10-27 | [#12186](https://github.com/sgl-project/sglang/pull/12186) | merged | improve mimax-m2 rmsnorm precision | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-07 | [#12798](https://github.com/sgl-project/sglang/pull/12798) | merged | Support capturing aux_hidden_states for minimax m2. | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-14 | [#13297](https://github.com/sgl-project/sglang/pull/13297) | merged | Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-25 | [#13892](https://github.com/sgl-project/sglang/pull/13892) | merged | fix: correct usage of minimax-m2 deepep moe forward | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-27 | [#14047](https://github.com/sgl-project/sglang/pull/14047) | merged | Optimize topk sigmoid in minimax_m2 | model wrapper, MoE/router | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-04 | [#14416](https://github.com/sgl-project/sglang/pull/14416) | merged | Fusing RMSNormTP in minimax_m2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-05 | [#16483](https://github.com/sgl-project/sglang/pull/16483) | merged | Optimizing all_reduce in RMSNormTP in minimax_m2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-27 | [#17826](https://github.com/sgl-project/sglang/pull/17826) | open | Support Pipeline and Data Parallelism for MiniMax-M2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-04 | [#18217](https://github.com/sgl-project/sglang/pull/18217) | merged | [piecewise graph]: support MiniMax-M2 | model wrapper, quantization, kernel | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2026-02-27 | [#19468](https://github.com/sgl-project/sglang/pull/19468) | open | fix[minimax]: support deepep with minimax models | kernel | `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh` |
| 2026-02-28 | [#19577](https://github.com/sgl-project/sglang/pull/19577) | merged | [Feat] add PP Support for minimax-m2 series | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-02 | [#19652](https://github.com/sgl-project/sglang/pull/19652) | merged | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-03-06 | [#19995](https://github.com/sgl-project/sglang/pull/19995) | merged | Add packed_modules_mapping for MiniMax-M2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-06 | [#20031](https://github.com/sgl-project/sglang/pull/20031) | open | fix(minimax): support loading merged expert weights (w13) for awq | model wrapper, tests/benchmarks | `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-07 | [#20067](https://github.com/sgl-project/sglang/pull/20067) | merged | MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn | model wrapper, tests/benchmarks | `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py` |
| 2026-03-13 | [#20489](https://github.com/sgl-project/sglang/pull/20489) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general… | model wrapper, scheduler/runtime | `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-16 | [#20673](https://github.com/sgl-project/sglang/pull/20673) | merged | [Feature][JIT Kernel] Fused TP QK norm For Minimax | model wrapper, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py` |
| 2026-03-18 | [#20870](https://github.com/sgl-project/sglang/pull/20870) | merged | [MiniMax M2] Fix KV cache scale loading | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-18 | [#20873](https://github.com/sgl-project/sglang/pull/20873) | open | docs: add MiniMax-M2.7 and M2.7-highspeed model support | model wrapper, docs/config | `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md` |
| 2026-03-19 | [#20905](https://github.com/sgl-project/sglang/pull/20905) | merged | [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5 | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20967](https://github.com/sgl-project/sglang/pull/20967) | merged | 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20975](https://github.com/sgl-project/sglang/pull/20975) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 | model wrapper, attention/backend, scheduler/runtime | `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#22300](https://github.com/sgl-project/sglang/pull/22300) | open | [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5) | quantization | `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2026-04-09 | [#22432](https://github.com/sgl-project/sglang/pull/22432) | open | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-14 | [#22744](https://github.com/sgl-project/sglang/pull/22744) | open | [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance | scheduler/runtime, docs/config | `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-16 | [#22934](https://github.com/sgl-project/sglang/pull/22934) | open | Minimax eplb bugfix | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-20 | [#23190](https://github.com/sgl-project/sglang/pull/23190) | open | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-21 | [#23301](https://github.com/sgl-project/sglang/pull/23301) | open | [sgl] Stream MiniMax M2 string parameters token-by-token | misc | `python/sglang/srt/function_call/minimax_m2.py` |

### File-level PR diff reading notes

### PR #12129 - Support MiniMax M2 model

- Link: https://github.com/sgl-project/sglang/pull/12129
- Status/date: `merged`, created 2025-10-25, merged 2025-10-26; author `zhaochenyang20`.
- Diff scope read: `5` files, `+1320/-1`; areas: model wrapper, docs/config; keywords: expert, moe, spec, attention, config, deepep, doc, fp8, kv, processor.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
  - `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: +import ast; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call
  - `python/sglang/srt/parser/reasoning_parser.py` modified +28/-1 (29 lines); hunks: def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:; class ReasoningParser:; symbols: parse_streaming_increment, MiniMaxAppendThinkDetector, __init__, parse_streaming_increment
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: from sglang.srt.function_call.gpt_oss_detector import GptOssDetector; class FunctionCallParser:; symbols: FunctionCallParser:, __init__
  - `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: expert, moe, spec, attention, config, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12186 - improve mimax-m2 rmsnorm precision

- Link: https://github.com/sgl-project/sglang/pull/12186
- Status/date: `merged`, created 2025-10-27, merged 2025-10-27; author `haichao592`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12798 - Support capturing aux_hidden_states for minimax m2.

- Link: https://github.com/sgl-project/sglang/pull/12798
- Status/date: `merged`, created 2025-11-07, merged 2025-11-08; author `pyc96`.
- Diff scope read: `1` files, `+34/-3`; areas: model wrapper; keywords: config, eagle, expert, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: def layer_fn(idx, prefix: str) -> nn.Module:; def forward(; symbols: layer_fn, get_input_embeddings, forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, eagle, expert, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13297 - Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/13297
- Status/date: `merged`, created 2025-11-14, merged 2025-11-15; author `pyc96`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper; keywords: eagle.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13892 - fix: correct usage of minimax-m2 deepep moe forward

- Link: https://github.com/sgl-project/sglang/pull/13892
- Status/date: `merged`, created 2025-11-25, merged 2025-11-26; author `yuukidach`.
- Diff scope read: `1` files, `+3/-7`; areas: model wrapper; keywords: deepep, expert, router, topk.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: def forward_deepep(; def forward_deepep(; symbols: forward_deepep, forward_deepep
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: deepep, expert, router, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14047 - Optimize topk sigmoid in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14047
- Status/date: `merged`, created 2025-11-27, merged 2025-12-02; author `rogeryoungh`.
- Diff scope read: `2` files, `+38/-13`; areas: model wrapper, MoE/router; keywords: config, expert, topk, cuda, moe, router.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +38/-10 (48 lines); hunks: ); pass; symbols: TopKConfig:, __init__, forward_native, fused_topk_torch_native
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, expert, topk, cuda, moe, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14416 - Fusing RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14416
- Status/date: `merged`, created 2025-12-04, merged 2025-12-30; author `rogeryoungh`.
- Diff scope read: `1` files, `+189/-2`; areas: model wrapper; keywords: config, cuda, deepep, expert, kv, moe, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: from typing import Iterable, Optional, Set, Tuple, Union; logger = logging.getLogger(__name__); symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, cuda, deepep, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16483 - Optimizing all_reduce in RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/16483
- Status/date: `merged`, created 2026-01-05, merged 2026-02-01; author `rogeryoungh`.
- Diff scope read: `1` files, `+8/-2`; areas: model wrapper; keywords: triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:; def forward(; symbols: rms_sumsq_serial, forward, forward_qk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17826 - Support Pipeline and Data Parallelism for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/17826
- Status/date: `open`, created 2026-01-27; author `rogeryoungh`.
- Diff scope read: `1` files, `+167/-70`; areas: model wrapper; keywords: attention, config, cuda, deepep, eagle, expert, kv, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; from sglang.srt.distributed import (; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, cuda, deepep, eagle, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18217 - [piecewise graph]: support MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/18217
- Status/date: `merged`, created 2026-02-04, merged 2026-02-05; author `hzh0425`.
- Diff scope read: `2` files, `+28/-7`; areas: model wrapper, quantization, kernel; keywords: config, cuda, deepep, expert, fp8, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +5/-0 (5 lines); hunks: def get_w8a8_block_fp8_configs(; symbols: get_w8a8_block_fp8_configs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; keywords observed in patches: config, cuda, deepep, expert, fp8, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19468 - fix[minimax]: support deepep with minimax models

- Link: https://github.com/sgl-project/sglang/pull/19468
- Status/date: `open`, created 2026-02-27; author `ishandhanani`.
- Diff scope read: `3` files, `+10/-2`; areas: kernel; keywords: deepep, config, cuda, doc, flash, moe, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docker/Dockerfile` modified +2/-1 (3 lines); hunks: ARG HOPPER_SBO=0
  - `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: if [ "$GRACE_BLACKWELL" = "1" ]; then
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`; keywords observed in patches: deepep, config, cuda, doc, flash, moe. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19577 - [Feat] add PP Support for minimax-m2 series

- Link: https://github.com/sgl-project/sglang/pull/19577
- Status/date: `merged`, created 2026-02-28, merged 2026-03-02; author `LuYanFCP`.
- Diff scope read: `1` files, `+35/-7`; areas: model wrapper; keywords: attention, config, eagle, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; symbols: __init__, forward, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, eagle, processor, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19652 - [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

- Link: https://github.com/sgl-project/sglang/pull/19652
- Status/date: `merged`, created 2026-03-02, merged 2026-04-03; author `Godmook`.
- Diff scope read: `16` files, `+1410/-95`; areas: MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: fp4, marlin, quant, fp8, moe, expert, config, flash, topk, triton.
- Code diff details:
  - `test/registered/quant/test_nvfp4_marlin_fallback.py` added +788/-0 (788 lines); hunks: +"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""; symbols: _check_requirements, _dequant_fp4_weights, _FakeLayer, TestNvfp4MarlinLinear
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: is_blackwell_supported,; def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: CompressedTensorsMoEScheme,; class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` modified +21/-32 (53 lines); hunks: __global__ void Marlin(; __global__ void Marlin(; symbols: void, int, int, int
- Optimization/support interpretation: The concrete diff surface is `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; keywords observed in patches: fp4, marlin, quant, fp8, moe, expert. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19995 - Add packed_modules_mapping for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/19995
- Status/date: `merged`, created 2026-03-06, merged 2026-03-18; author `trevor-m`.
- Diff scope read: `1` files, `+12/-0`; areas: model wrapper; keywords: config, kv.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20031 - fix(minimax): support loading merged expert weights (w13) for awq

- Link: https://github.com/sgl-project/sglang/pull/20031
- Status/date: `open`, created 2026-03-06; author `xueliangyang-oeuler`.
- Diff scope read: `2` files, `+203/-9`; areas: model wrapper, tests/benchmarks; keywords: config, expert, moe, spec, attention, processor, quant, test.
- Code diff details:
  - `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: +import unittest; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13
  - `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, expert, moe, spec, attention, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20067 - MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn

- Link: https://github.com/sgl-project/sglang/pull/20067
- Status/date: `merged`, created 2026-03-07, merged 2026-04-10; author `trevor-m`.
- Diff scope read: `3` files, `+39/-6`; areas: model wrapper, tests/benchmarks; keywords: attention, config, cuda, expert, flash, fp4, kv, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: RowParallelLinear,; def forward_normal(; symbols: forward_normal, forward_prepare, forward_prepare, forward_core
  - `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: def test_minimax_m25(self):; def test_minimax_m25(self):; symbols: test_minimax_m25, test_minimax_m25
  - `python/sglang/srt/layers/layernorm.py` modified +4/-0 (4 lines); hunks: def forward_cuda(; symbols: forward_cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py`; keywords observed in patches: attention, config, cuda, expert, flash, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20489 - fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…

- Link: https://github.com/sgl-project/sglang/pull/20489
- Status/date: `open`, created 2026-03-13; author `xueliangyang-oeuler`.
- Diff scope read: `5` files, `+118/-20`; areas: model wrapper, scheduler/runtime; keywords: attention, config, cuda, kv, cache, expert, moe, quant, test.
- Code diff details:
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo; def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: def forward_cuda(; symbols: forward_cuda
- Optimization/support interpretation: The concrete diff surface is `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, config, cuda, kv, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20673 - [Feature][JIT Kernel] Fused TP QK norm For Minimax

- Link: https://github.com/sgl-project/sglang/pull/20673
- Status/date: `merged`, created 2026-03-16, merged 2026-04-13; author `DarkSharpness`.
- Diff scope read: `11` files, `+923/-82`; areas: model wrapper, kernel, tests/benchmarks; keywords: cuda, config, test, cache, kv, processor, spec, triton, attention, benchmark.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh` added +325/-0 (325 lines); hunks: +// Adapted from https://github.com/NVIDIA/TensorRT-LLM/pull/12163; symbols: ParallelQKNormParams, auto, KernelTrait, parameters
  - `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py` added +170/-0 (170 lines); hunks: +from __future__ import annotations; symbols: parse_args, init_distributed, bench_one, rmsnorm_baseline
  - `python/sglang/jit_kernel/tests/test_tp_qknorm.py` added +168/-0 (168 lines); hunks: +from __future__ import annotations; symbols: test_tp_qknorm, init_distributed, _all_gather_cat, _rmsnorm_ref
  - `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: import logging; ); symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm:, __init__
  - `python/sglang/jit_kernel/all_reduce.py` modified +50/-6 (56 lines); hunks: import torch; def config_pull(; symbols: config_pull, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`; keywords observed in patches: cuda, config, test, cache, kv, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20870 - [MiniMax M2] Fix KV cache scale loading

- Link: https://github.com/sgl-project/sglang/pull/20870
- Status/date: `merged`, created 2026-03-18, merged 2026-03-18; author `chadvoegele`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper; keywords: cache, expert, kv, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: cache, expert, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20873 - docs: add MiniMax-M2.7 and M2.7-highspeed model support

- Link: https://github.com/sgl-project/sglang/pull/20873
- Status/date: `open`, created 2026-03-18; author `octo-patch`.
- Diff scope read: `2` files, `+15/-3`; areas: model wrapper, docs/config; keywords: doc, moe, expert, test.
- Code diff details:
  - `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -# MiniMax M2.5/M2.1/M2 Usage; curl http://localhost:8000/v1/chat/completions \
  - `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; keywords observed in patches: doc, moe, expert, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20905 - [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5

- Link: https://github.com/sgl-project/sglang/pull/20905
- Status/date: `merged`, created 2026-03-19, merged 2026-03-24; author `shadowxz109`.
- Diff scope read: `2` files, `+22/-30`; areas: model wrapper, quantization; keywords: config, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +21/-29 (50 lines); hunks: def get_moe_scheme(; symbols: get_moe_scheme, is_layer_skipped
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20967 - 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16

- Link: https://github.com/sgl-project/sglang/pull/20967
- Status/date: `merged`, created 2026-03-20, merged 2026-04-10; author `kingkingleeljj`.
- Diff scope read: `1` files, `+34/-10`; areas: model wrapper; keywords: attention, config, kv.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: def rms_apply_serial(; def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20975 - fix(dp-attn): fix issues with dp-attention for MiniMax M2

- Link: https://github.com/sgl-project/sglang/pull/20975
- Status/date: `open`, created 2026-03-20; author `xueliangyang-oeuler`.
- Diff scope read: `6` files, `+122/-20`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, config, cuda, kv, cache, expert, moe, quant, test.
- Code diff details:
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: from sglang.kernel_api_logging import debug_kernel_api; def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
- Optimization/support interpretation: The concrete diff surface is `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, config, cuda, kv, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22300 - [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)

- Link: https://github.com/sgl-project/sglang/pull/22300
- Status/date: `open`, created 2026-04-08; author `trevor-m`.
- Diff scope read: `3` files, `+30/-6`; areas: quantization; keywords: fp8, quant, triton, config, flash.
- Code diff details:
  - `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: def post_load_weights(model: nn.Module, model_config: ModelConfig):; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_deepgemm_weight_requant_ue8m0, should_async_load
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: def process_weights_after_loading_block_quant(self, layer: Module) -> None:; symbols: process_weights_after_loading_block_quant
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`; keywords observed in patches: fp8, quant, triton, config, flash. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22432 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2

- Link: https://github.com/sgl-project/sglang/pull/22432
- Status/date: `open`, created 2026-04-09; author `shadowxz109`.
- Diff scope read: `1` files, `+69/-11`; areas: model wrapper; keywords: attention, cache, config, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: import logging; ); symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, cache, config, expert, kv, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22744 - [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance

- Link: https://github.com/sgl-project/sglang/pull/22744
- Status/date: `open`, created 2026-04-14; author `trevor-m`.
- Diff scope read: `3` files, `+11/-0`; areas: scheduler/runtime, docs/config; keywords: moe, cache, doc, fp8, kv.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: def __init__(; symbols: __init__
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md`; keywords observed in patches: moe, cache, doc, fp8, kv. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22934 - Minimax eplb bugfix

- Link: https://github.com/sgl-project/sglang/pull/22934
- Status/date: `open`, created 2026-04-16; author `DaZhUUU`.
- Diff scope read: `1` files, `+25/-0`; areas: model wrapper; keywords: attention, config, eagle, expert, moe, quant, topk, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; # Other files (custom_all_reduce.py, hf_transformers_utils.py) also use sglang.srt.utils.; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, eagle, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23190 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode

- Link: https://github.com/sgl-project/sglang/pull/23190
- Status/date: `open`, created 2026-04-20; author `heziiop`.
- Diff scope read: `1` files, `+66/-10`; areas: model wrapper; keywords: attention, cache, config, cuda, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: import logging; get_compiler_backend,; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, cache, config, cuda, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23301 - [sgl] Stream MiniMax M2 string parameters token-by-token

- Link: https://github.com/sgl-project/sglang/pull/23301
- Status/date: `open`, created 2026-04-21; author `lujiajing1126`.
- Diff scope read: `1` files, `+332/-280`; areas: misc; keywords: config, spec.
- Code diff details:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: logger = logging.getLogger(__name__); class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, MinimaxM2Detector, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/minimax_m2.py`; keywords observed in patches: config, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 29; open PRs: 12.
- Open PRs to keep tracking: [#17826](https://github.com/sgl-project/sglang/pull/17826), [#19468](https://github.com/sgl-project/sglang/pull/19468), [#20031](https://github.com/sgl-project/sglang/pull/20031), [#20489](https://github.com/sgl-project/sglang/pull/20489), [#20873](https://github.com/sgl-project/sglang/pull/20873), [#20975](https://github.com/sgl-project/sglang/pull/20975), [#22300](https://github.com/sgl-project/sglang/pull/22300), [#22432](https://github.com/sgl-project/sglang/pull/22432), [#22744](https://github.com/sgl-project/sglang/pull/22744), [#22934](https://github.com/sgl-project/sglang/pull/22934), [#23190](https://github.com/sgl-project/sglang/pull/23190), [#23301](https://github.com/sgl-project/sglang/pull/23301)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
