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
