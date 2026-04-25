# SGLang MiniMax M2 / M2.5 / M2.7 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `47c4b3825`，以及 MiniMax 相关 merged、open PR patch 阅读结果整理。范围覆盖原有 `sglang-minimax-m2-m25-optimization` skill 涉及的主线，并补充 MiniMax M2.7、TP QK RMSNorm allreduce fusion、DP attention、FP4/NVFP4、NPU、DeepEP、EPLB 和 tool-call streaming 的最新状态。

阅读结论先放前面：截至 `47c4b3825`，MiniMax M2 系列主线模型文件是 `python/sglang/srt/models/minimax_m2.py`，它已经支持 M2/M2.1/M2.5 的基础加载、tool calling、reasoning parser、Eagle3 aux hidden states、PP、DP attention 相关 attention-TP 分组、M2.5 reduce-scatter/FP4 all-gather/AR fusion，以及你提到的 TP QK RMSNorm allreduce fusion。M2.7 当前主要体现为文档和同一模型类复用。

## 1. 时间线总览

| 创建日期   |     PR | 状态   | 主线          | 代码区域                                                      | 作用                                                                                |
| ---------- | -----: | ------ | ------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 2025-10-25 | #12129 | merged | M2 bring-up   | `models/minimax_m2.py`、`function_call/minimax_m2.py`、parser | 新增 MiniMax M2 模型、tool-call parser、reasoning parser 和文档。                   |
| 2025-10-27 | #12186 | merged | 精度          | `MiniMaxM2RMSNormTP`                                          | RMSNorm 中先乘 weight 再转回原 dtype，提高精度。                                    |
| 2025-11-07 | #12798 | merged | Eagle3        | `minimax_m2.py`、memory cache                                 | 支持捕获 `aux_hidden_states`。                                                      |
| 2025-11-14 | #13297 | merged | Eagle3        | `minimax_m2.py`                                               | 补齐 `get_embed_and_head`。                                                         |
| 2025-11-25 | #13892 | merged | DeepEP 调用   | `MiniMaxM2MoE.forward_deepep`                                 | 修正 DeepEP MoE forward 参数，从拆散 top-k 改为传 `TopKOutput`。                    |
| 2025-11-27 | #14047 | merged | Router        | `layers/moe/topk.py`、`minimax_m2.py`                         | 增加 `topk_sigmoid` 和 `scoring_func="sigmoid"` 路径。                              |
| 2025-12-04 | #14416 | merged | QK RMSNorm    | `minimax_m2.py`                                               | 融合 q/k RMSNormTP 的 sumsq、allreduce 和 apply。                                   |
| 2026-01-05 | #16483 | merged | QK RMSNorm    | `rms_sumsq_serial`                                            | 给 RMSNormTP allreduce buffer 做 512 对齐 padding。                                 |
| 2026-01-27 | #17826 | open   | PP + DP       | `minimax_m2.py`                                               | 支持 Pipeline + Data Parallelism 的开放 PR，部分思想已被后续主线吸收。              |
| 2026-02-04 | #18217 | merged | PCG           | `fp8_kernel.py`、`minimax_m2.py`                              | 支持 MiniMax-M2 piecewise CUDA graph。                                              |
| 2026-02-27 | #19468 | open   | DeepEP        | server args、CI、MiniMax config                               | 让 MiniMax 模型支持 DeepEP 的开放 PR。                                              |
| 2026-02-28 | #19577 | merged | PP            | `minimax_m2.py`                                               | 正式增加 MiniMax M2 系列 PP 支持。                                                  |
| 2026-03-02 | #19652 | merged | NVFP4         | quantization、Marlin fallback                                 | 非 Blackwell GPU 上用 Marlin fallback 跑 NVFP4。                                    |
| 2026-03-06 | #19995 | merged | Loader        | `minimax_m2.py`                                               | 增加 `packed_modules_mapping`。                                                     |
| 2026-03-06 | #20031 | open   | Loader        | `minimax_m2.py`、weight test                                  | 支持 AWQ merged expert `w13` 权重加载。                                             |
| 2026-03-07 | #20067 | merged | M2.5 分布式   | `layernorm.py`、`minimax_m2.py`、test                         | M2.5 支持 DP attention、DP reduce-scatter、FP4 all-gather、prepare_attn AR fusion。 |
| 2026-03-13 | #20489 | open   | DP attention  | `minimax_m2.py`、runner、memory pool、rotary                  | 修复 MiniMax M2 DP-attn 的 attention-TP、empty batch 等问题。                       |
| 2026-03-16 | #20673 | merged | TP QKNorm     | `jit_kernel/all_reduce.py`、`tp_qknorm.cuh`、`minimax_m2.py`  | 新增 JIT fused TP QK RMSNorm + custom allreduce。                                   |
| 2026-03-18 | #20870 | merged | Loader        | `minimax_m2.py`                                               | 修复 KV cache scale 加载时被 qkv rename 吞掉的问题。                                |
| 2026-03-18 | #20873 | open   | M2.7 docs     | docs                                                          | 旧 docs 中增加 MiniMax-M2.7 和 M2.7-highspeed。                                     |
| 2026-03-19 | #20905 | merged | NPU/ModelSlim | `modelslim.py`、`minimax_m2.py`                               | 适配 Minimax2.5 的 w2 quant layer suffix。                                          |
| 2026-03-20 | #20967 | merged | TP16 bugfix   | `MiniMaxM2RMSNormTP`                                          | 修复 TP16 下 KV head replica 造成的重复输出。                                       |
| 2026-03-20 | #20975 | open   | DP attention  | DP attention 后续修复                                         | `#20489` 的后续版，继续修 DP-attn、rotary empty batch、rank buffer。                |
| 2026-04-08 | #22300 | open   | FP8 GEMM      | `fp8.py`、`fp8_utils.py`、loader utils                        | 修复 fp16 模型上 DeepGEMM UE8M0 scale 误转换导致的 FP8 GEMM 性能/正确性问题。       |
| 2026-04-09 | #22432 | open   | NPU           | `split_qkv_tp_rmsnorm_rope`                                   | NPU 上融合 split qkv、TP RMSNorm、RoPE。                                            |
| 2026-04-14 | #22744 | open   | NVIDIA TF32   | server args、model runner                                     | 增加 `--enable-tf32-matmul` 提升 MiniMax gate GEMM 性能。                           |
| 2026-04-16 | #22934 | open   | EPLB          | `minimax_m2.py`                                               | 给 MiniMax 增加 EPLB routed expert weights 接口。                                   |
| 2026-04-20 | #23190 | open   | NPU + Eagle3  | `split_qkv_tp_rmsnorm_rope`、hidden states capture            | `#22432` 后续，补 NPU empty batch 和 DP-attn 下 Eagle3 hidden states capture。      |
| 2026-04-21 | #23301 | open   | Tool calling  | `function_call/minimax_m2.py`                                 | string 参数 token-by-token streaming，降低 tool-call 参数延迟。                     |

## 2. MiniMax M2 bring-up：模型结构、parser 与权重加载

`#12129` 是 MiniMax M2 接入的起点。它新增 `python/sglang/srt/models/minimax_m2.py`，模型结构和 MiniMax checkpoint 对齐：

- `MiniMaxM2RMSNormTP`：为 Q/K normalization 设计的 TP-aware RMSNorm。
- `MiniMaxM2MLP` 与 `MiniMaxM2MoE`：MiniMax 的每层是 MoE，不像 DeepSeek 那样带 shared experts。
- `MiniMaxM2Attention`：使用 QK normalization、partial RoPE、`QKVParallelLinear` 和 `RadixAttention`。
- `MiniMaxM2DecoderLayer`：attention 后接 MoE，支持 TBO 所需的 gate、select experts、expert compute 等 operation 拆分。
- `MiniMaxM2Model` 与 `MiniMaxM2ForCausalLM`：负责 embedding、layers、norm、lm head、logits processor 和权重加载。

同一个 PR 还新增 `python/sglang/srt/function_call/minimax_m2.py`。MiniMax M2 的 tool call 格式不是 OpenAI JSON，而是 XML-like block：

```xml
<minimax:tool_call>
<invoke name="func1">
<parameter name="param1">value1</parameter>
</invoke>
</minimax:tool_call>
```

`MinimaxM2Detector` 因此要识别 `<minimax:tool_call>`、`<invoke name="...">` 和 `<parameter name="...">`。streaming 初版会维护 `_in_tool_call`、`_current_parameters`、`_streamed_parameters` 等状态，逐步发出 tool name 和参数 JSON fragment。reasoning parser 初期也接入了 `minimax-m2`，后来实际行为与 Qwen3-style thinking 更接近。

权重加载初版处理几类名字映射：

- `q_proj/k_proj/v_proj` 堆叠进 `qkv_proj`。
- `gate_proj/up_proj` 堆叠进 `gate_up_proj`。
- MoE experts 的 `w1/w2/w3` 分别映射 gate/down/up。
- 跳过 `rotary_emb.inv_freq`。
- 对 GPTQ extra bias 做容错跳过。

`#19995` 后来把这些堆叠关系显式暴露为：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

这让量化、加载器和外部工具能从模型类上直接知道哪些 checkpoint module 是 packed module。

`#20870` 修了 KV cache scale 加载。问题是 checkpoint 里 KV scale 名字类似 `self_attn.k_proj.k_scale` / `self_attn.v_proj.v_scale`，但 stacked mapping loop 会先把 `k_proj` 改成 `qkv_proj`，导致 `maybe_remap_kv_scale_name` 无法识别原始模式。PR 增加 `_is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")`，遇到 KV scale 时跳过 qkv rename，让原始名字直接进入 `maybe_remap_kv_scale_name`，最终映射到 `self_attn.attn.k_scale/v_scale`。

`#20031` 仍 open，目标是支持 AWQ merged expert weights。某些 checkpoint 把 gate/up 合并为 `w13`，而不是 `w1/w3` 分开。PR 使用 `FusedMoE.make_expert_params_mapping_fused(ckpt_gate_up_proj_name="w13", ckpt_down_proj_name="w2", ...)`，并在旧 `w1/w2/w3` mapping 前先尝试 fused mapping。

## 3. Router 与 DeepEP 调用：sigmoid top-k 和 TopKOutput 统一

MiniMax M2 的 router 使用 sigmoid scoring，而不是默认 softmax。`#14047` 把 `TopKConfig` 增加 `scoring_func` 字段，并在 `topk.py` 的多条路径中支持 `scoring_func="sigmoid"`：

- CUDA/HIP 导入 `topk_sigmoid`。
- `fused_topk_torch_native` 中抽象 `scoring_func_impl`，支持 softmax/sigmoid。
- `fused_topk` 在 `scoring_func == "sigmoid"` 时调用 `topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize, correction_bias)`。
- `select_experts` 把 `topk_config.scoring_func` 一路传到底层 fused top-k。
- `MiniMaxM2MoE` 删除早期为了复用 grouped top-k 而设置的 `use_grouped_topk=True, num_expert_group=1, topk_group=1`，直接依赖 sigmoid scoring。

`#13892` 修复 DeepEP MoE forward 的参数协议。早期代码把 `self.topk(...)` 返回拆成 `topk_weights, topk_idx, _`，然后分别传给 `self.experts(topk_idx=..., topk_weights=...)`。主线 MoE runner 后续统一为 `TopKOutput`，所以 PR 改成：

- 正常 token 时 `topk_output = self.topk(...)`。
- 空 token 时 `topk_output = self.topk.empty_topk_output(device=hidden_states.device)`。
- experts 调用统一为 `self.experts(hidden_states=hidden_states, topk_output=topk_output)`。

这使 MiniMax 的 normal MoE、DeepEP MoE、空 batch 场景和后续 EP/DP 扩展都能共享同一 top-k 数据结构。

`#19468` 仍 open，目标是让 MiniMax 模型正式支持 DeepEP。patch 涉及 server args、CI DeepEP 安装和 MiniMax hidden size / BF16 要求。当前 main 中 `MiniMaxM2MoE.forward` 已经会在 `get_moe_a2a_backend().is_deepep()` 或 Ascend FuseEP 时走 `forward_deepep`，但完整 DeepEP 可用性仍要看这个方向后续合入和测试。

## 4. QK RMSNorm：从精度修复到 TP allreduce fusion

`#12186` 是一个只有一行的精度修复，但非常关键。原逻辑是：

```python
x = x.to(orig_dtype) * self.weight
```

PR 改成：

```python
x = (x * self.weight).to(orig_dtype)
```

这样 weight 乘法发生在 fp32 normalized tensor 上，最后才 cast 回原 dtype，避免提前降精度。

`#14416` 是第一版 Q/K RMSNormTP fusion。MiniMax 的 attention 会对 q 和 k 分别做 RMSNorm，而且 TP 下 variance 需要跨 rank 汇总。PR 新增 Triton kernel：

- `rmsnorm_sumsq_kernel_serial`：同时计算 q 和 k 的 sum of squares，输出 fp32 `[B, 2]`。
- `rmsnorm_apply_kernel_serial`：读取 allreduce 后的 sumsq，对 q/k 应用 `rsqrt(sum_sq / full_dim + eps)` 和各自 weight。
- `rms_sumsq_serial` 和 `rms_apply_serial` 作为 Python wrapper。
- `MiniMaxM2RMSNormTP.forward_qk` 把 q/k 的 norm 合在一起，减少 kernel launch 和 allreduce 组织成本。
- `MiniMaxM2Attention.forward_prepare` 在 `use_qk_norm` 时调用 `forward_qk`，而不是分别调用 q_norm 和 k_norm。

`#16483` 优化了这个 allreduce buffer。SGLang 的 custom allreduce `sglang::cross_device_reduce_1stage` 需要对齐，MiniMax RMSNormTP reduce 的是 `[B, 2]` fp32 tensor。PR 把 buffer 元素数 pad 到 512 对齐，避免 custom allreduce 处理非对齐大小时的性能/边界问题。PR 描述中提到 M2.1 上吞吐约有 6% 提升。

`#20967` 修复 TP16 重复输出。根因是当 attention TP size 大于 KV head 数时，KV heads 会在多个 rank 上 replica，而旧 RMSNormTP weight sharding 仍按 rank 直接切 shard。PR 让 `MiniMaxM2RMSNormTP` 对齐 `QKVParallelLinear` 的逻辑：

- 如果 `attn_tp_size >= num_heads`，要求 `attn_tp_size % num_heads == 0`，本 rank 只有 1 个 logical head，并设置 `num_head_replicas = attn_tp_size // num_heads`。
- 否则要求 `num_heads % attn_tp_size == 0`，每 rank 负责 `num_heads // attn_tp_size` 个 head，`num_head_replicas = 1`。
- weight loader 使用 `shard_id = attn_tp_rank // num_head_replicas`，使 replicated ranks 读取同一个 shard。
- forward 中 allreduce 使用 attention TP group，而不是盲目使用全局 TP。

`#20673` 是你提到的 “allreduce TP norm” 优化，目前已经 merged。它新增 JIT kernel `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`，并在 `python/sglang/jit_kernel/all_reduce.py` 暴露：

- `fused_parallel_qknorm`
- `get_fused_parallel_qknorm_max_occupancy`

MiniMax 侧新增 `MiniMaxM2QKRMSNorm`：

- 默认 `_forward_naive` 仍然是 `rms_sumsq_serial(q, k)`、`attn_tp_all_reduce(sum_sq)`、`rms_apply_serial(...)`。
- 当 `world_size > 1`、设备是 CUDA，并且环境变量 `SGLANG_USE_FUSED_PARALLEL_QKNORM` 为 true 时，使用 JIT fused path。
- fused path 先根据 dtype、world size、q/k full dim 查询 max occupancy。
- 创建 `CustomAllReduceV2`，group 是 attention TP group。
- max push size 根据 `chunked_prefill_size`、`context_len`、`max_prefill_tokens` 推导并做 512 对齐。
- runtime 调用 registered custom op `fused_tp_qknorm`，最终执行 `fused_parallel_qknorm(COMM_MAP[counter].obj, q, k, q_weight, k_weight, eps)`，在一个 JIT kernel/通信路径中完成 q/k norm 和跨 TP reduce。

这个 PR 还加入 `test_tp_qknorm.py` 和 `bench_tp_qknorm.py`。PR 描述中的 decode benchmark 从 150 tps 提升到 157 tps，是当前 MiniMax QKNorm 路径最重要的亮点之一。

## 5. PP、DP attention、M2.5 分布式路径和 PCG

`#19577` 是 MiniMax PP 的正式合入。它做了几件事：

- `MiniMaxM2Model` 使用 `make_layers`，得到 `self.layers, self.start_layer, self.end_layer`。
- 非最后 PP rank 的 `norm` 和 `lm_head` 用 `PPMissingLayer`。
- forward 接受 `pp_proxy_tensors`，首 rank 从 embedding 开始，非首 rank 从 proxy 读取 `hidden_states/residual`。
- 非最后 rank 返回 `PPProxyTensors({"hidden_states": hidden_states, "residual": residual})`。
- `load_weights` 用 `get_layer_id(name)` 跳过不属于当前 PP shard 的 layer 权重。

`#17826` 是 open 的 PP + DP PR，虽然未合入，但其中 attention-TP rank/size、`is_dp_attention_enabled()`、embedding/lm head 是否使用 attention TP group 等思路，已经在后续 merged PR 中逐步进入主线。

`#20067` 是 MiniMax-M2.5 分布式优化的主 PR。它的标题已经很直接：DP attention、DP reduce-scatter、FP4 all-gather、prepare_attn 中 AR fusion。当前 main 中能看到这些结果：

- `MiniMaxM2Attention` 使用 `get_attention_tp_rank()` / `get_attention_tp_size()` 初始化 QKV 和 O projection，而不是默认全局 TP。
- `VocabParallelEmbedding(..., use_attn_tp_group=is_dp_attention_enabled())` 让 DP attention 模式下 embedding 按 attention TP group 工作。
- `MiniMaxM2DecoderLayer` 创建 `LayerCommunicator(..., allow_reduce_scatter=True)`。
- MoE forward 接收 `should_allreduce_fusion` 和 `use_reduce_scatter`，如果下一层可以融合 allreduce 或当前可 reduce-scatter，就不在 MoE 内部立即做 `tensor_model_parallel_all_reduce`。
- 当 `should_use_flashinfer_cutlass_moe_fp4_allgather()` 为 true 时，也跳过 MoE 内部 allreduce，让 FP4 all-gather 路径接管通信。
- registered test 覆盖 TP8+EP8、TP8+DP8+EP8+DP-attention 等 M2.5 形态。

`#18217` 让 MiniMax-M2 支持 piecewise CUDA graph。它在 `fp8_kernel.py` 中处理 Dynamo compiling 时的 config 获取，并在 MoE select experts、TBO op 和 model forward loop 里用 `nullcontext()` 替代 expert distribution recorder context，以免 PCG 捕获时引入不兼容的动态上下文。

`#20489` 和 `#20975` 是仍 open 的 DP attention 修复线。patch 内容包括：

- MiniMax attention 使用 attention TP size/group 做 head partition 和通信，而不是全局 TP。
- `model_runner` 在 `require_attn_tp_gather` 时按 `dp_size` 初始化 `global_num_tokens_gpu`，避免高 rank invalid device ordinal/access。
- memory pool 和 rotary embedding 处理空 batch，避免 0-sized tensor view 错误。
- 后续 patch 还补了函数名从 `get_attention_tp_world_size` 到实际可用的 `get_attn_tensor_model_parallel_world_size` / `get_attn_tp_group`。

当前 main 已经有不少 attention-TP 和空 hidden state 保护，但这两个 PR 说明 DP-attn 的边界仍在继续被打磨。

## 6. Eagle3、M2.7 与 tool-call streaming

`#12798` 给 MiniMax M2 增加 Eagle3 所需的 aux hidden states 捕获：

- `MiniMaxM2Model` 增加 `layers_to_capture`。
- forward loop 中如果当前 layer id 在 capture 列表里，就把 `hidden_states + residual` 放进 `aux_hidden_states`。
- `MiniMaxM2ForCausalLM.set_eagle3_layers_to_capture` 设置默认捕获层 `[2, num_layers // 2, num_layers - 3]`，或使用调用方传入的 layer ids。
- logits processor 收到 `aux_hidden_states`，供 Eagle3 使用。

`#13297` 补齐 `get_embed_and_head`，返回 `self.model.embed_tokens.weight, self.lm_head.weight`，让 Eagle3 能拿到主模型 embedding 和 lm head。

`#20873` 是 open 的旧文档 PR，增加 MiniMax-M2.7 和 M2.7-highspeed。虽然这个 PR 还没合入，但最新 main 的 `docs_new` 里已经有 `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.7.mdx`，并且 cookbook 导航里也有 MiniMax-M2.7。代码层面 M2.7 仍复用 `MiniMaxM2ForCausalLM` 系列模型类。

`#23301` 是 tool-call streaming 的新 open PR。它重写 `MinimaxM2Detector.parse_streaming_increment`，使 string 类型参数可以 token-by-token streaming：

- 增加 `_STREAM_HOLD_BACK = len("</parameter>") - 1`，避免把 end tag 的半截误当作参数内容发出去。
- 增加 `_in_parameter`、`_current_param_name`、`_param_raw_sent_len`、`_current_param_is_string`、`_first_param_started` 等 fine-grained 状态。
- string 参数在看到 `<parameter name="...">` 后先发 JSON key prefix，再逐步 JSON escape 并追加 value content。
- int/bool/object/array 等非 string 参数仍然 buffer 到 `</parameter>` 后一次性转换。
- 结束 `</invoke>` 时补齐 JSON object 的 `}`。

这个 PR 的价值不是模型吞吐，而是 agent/tool-use 体验：长 string 参数不必等完整 `</parameter>` 出现才开始返回。

## 7. 量化、NPU、TF32、EPLB 等开放优化方向

`#19652` 是通用但对 MiniMax M2.5 很重要的 NVFP4 Marlin fallback。它新增 `marlin_utils_fp4.py`，允许非 Blackwell GPU 从 SM75 起使用 Marlin FP4 fallback：

- 检测是否 Blackwell；非 Blackwell 但支持 Marlin FP4 时启用 fallback。
- 处理 NVFP4 scale：把 FP8-S1E4M3 scale 转成 Marlin dequant 更合适的 FP8-S0E5M3 格式。
- 对 linear 和 MoE 权重做 Marlin tile layout repack。
- MoE fallback 会构造 `MarlinMoeQuantInfo`，让 fused Marlin MoE 用 FP4 scalar type 和 global scale。
- 新增 `test_nvfp4_marlin_fallback.py` 覆盖 linear 和 MoE。

`#22300` 是 FP8 GEMM scale 的开放修复。问题是加载时如果把 weight scale 转成 DeepGEMM 所需的 UE8M0/R128c4 packed format，但 runtime 因 fp16 output dtype、K shape 或 backend 条件不满足而 fallback 到 Triton，Triton 仍期待普通 fp32 scale，会导致错误结果或性能问题。PR 让 `should_deepgemm_weight_requant_ue8m0` 同时检查 output dtype 和 weight shape，并在 FlashInfer/TRTLLM fallback 中检测 `weight_scale.format_ue8m0`。

`#20905` 是 NPU ModelSlim 方向。MiniMax2.5 checkpoint 的 MoE quant 描述可能使用 `.0.w2.weight` 这样的 suffix，而不是普通 `.0.gate_proj.weight`。PR 调整 ModelSlim MoE scheme 探测，让 `W4A4_DYNAMIC`、`W4A8_DYNAMIC`、`W8A8_DYNAMIC` 能从 MiniMax 的 w2 suffix 识别出来。

`#22432` 和 `#23190` 是 NPU fused attention prepare 方向。它们引入 `sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope.split_qkv_tp_rmsnorm_rope`，在 NPU 上把 qkv projection 之后的 split、TP RMSNorm 和 RoPE 合并到 `forward_prepare_npu`。`#23190` 还补了 empty hidden states 短路，以及 DP-attn 下 Eagle3 hidden states capture 的修复。

`#22744` 是 NVIDIA TF32 gate GEMM 优化。它增加 `--enable-tf32-matmul`，model runner 里调用 `torch.set_float32_matmul_precision("high")`。PR 描述显示 MiniMax gate GEMM 的 FP32 开销占比可从 9.1% 降到 3.3%，batch64 吞吐从 3076.99 提到 3302.03 tok/s。

`#22934` 是 MiniMax EPLB bugfix，仍 open。它给 `MiniMaxM2MoE` 增加 `get_moe_weights`，用 `filter_moe_weight_param_global_expert` 过滤本地/冗余 expert 权重；`MiniMaxM2ForCausalLM` 增加 `_routed_experts_weights_of_layer = LazyValue(...)` 和 `routed_experts_weights_of_layer` property。当前 main 的 Kimi K2.5 已有类似 EPLB wrapper 接口，而 MiniMax 这条还没进主线。

## 8. 当前 main 的代码形态

截至 `47c4b3825`，MiniMax 主线形态可以概括为：

- `MiniMaxM2ForCausalLM` 是 M2/M2.1/M2.5/M2.7 系列共用的模型类。
- `MiniMaxM2MoE` 使用 sigmoid top-k、`TopKOutput`、normal/DeepEP 分支、reduce-scatter 和 FP4 all-gather 相关通信控制。
- `MiniMaxM2Attention` 使用 attention TP rank/size，支持 DP attention 的 head partition；Q/K RMSNorm 走 `MiniMaxM2QKRMSNorm`，可通过 `SGLANG_USE_FUSED_PARALLEL_QKNORM` 启用 JIT fused TP QKNorm allreduce。
- `MiniMaxM2DecoderLayer` 通过 `LayerCommunicator` 支持 prepare_attn AR fusion、prepare_mlp、reduce-scatter 和 postprocess。
- loader 已支持 packed mapping、KV scale remap、PP shard skip；AWQ `w13` merged expert loader 仍 open。
- 文档层已经出现 M2.7；代码层仍复用同一个 M2 系列实现。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `MiniMax M2 series`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
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

### 逐 PR 代码 diff 阅读记录

### PR #12129 - Support MiniMax M2 model

- 链接：https://github.com/sgl-project/sglang/pull/12129
- 状态/时间：`merged`，created 2025-10-25, merged 2025-10-26；作者 `zhaochenyang20`。
- 代码 diff 已读范围：`5` 个文件，`+1320/-1`；代码面：model wrapper, docs/config；关键词：expert, moe, spec, attention, config, deepep, doc, fp8, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
  - `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunk: +import ast; 符号: _safe_val, MinimaxM2Detector, __init__, has_tool_call
  - `python/sglang/srt/parser/reasoning_parser.py` modified +28/-1 (29 lines); hunk: def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:; class ReasoningParser:; 符号: parse_streaming_increment, MiniMaxAppendThinkDetector, __init__, parse_streaming_increment
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunk: from sglang.srt.function_call.gpt_oss_detector import GptOssDetector; class FunctionCallParser:; 符号: FunctionCallParser:, __init__
  - `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunk: in the GitHub search bar.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py`；patch 关键词为 expert, moe, spec, attention, config, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12186 - improve mimax-m2 rmsnorm precision

- 链接：https://github.com/sgl-project/sglang/pull/12186
- 状态/时间：`merged`，created 2025-10-27, merged 2025-10-27；作者 `haichao592`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12798 - Support capturing aux_hidden_states for minimax m2.

- 链接：https://github.com/sgl-project/sglang/pull/12798
- 状态/时间：`merged`，created 2025-11-07, merged 2025-11-08；作者 `pyc96`。
- 代码 diff 已读范围：`1` 个文件，`+34/-3`；代码面：model wrapper；关键词：config, eagle, expert, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunk: def layer_fn(idx, prefix: str) -> nn.Module:; def forward(; 符号: layer_fn, get_input_embeddings, forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, eagle, expert, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13297 - Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3

- 链接：https://github.com/sgl-project/sglang/pull/13297
- 状态/时间：`merged`，created 2025-11-14, merged 2025-11-15；作者 `pyc96`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：model wrapper；关键词：eagle。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunk: def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):; 符号: set_eagle3_layers_to_capture, get_embed_and_head, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13892 - fix: correct usage of minimax-m2 deepep moe forward

- 链接：https://github.com/sgl-project/sglang/pull/13892
- 状态/时间：`merged`，created 2025-11-25, merged 2025-11-26；作者 `yuukidach`。
- 代码 diff 已读范围：`1` 个文件，`+3/-7`；代码面：model wrapper；关键词：deepep, expert, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunk: def forward_deepep(; def forward_deepep(; 符号: forward_deepep, forward_deepep
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 deepep, expert, router, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14047 - Optimize topk sigmoid in minimax_m2

- 链接：https://github.com/sgl-project/sglang/pull/14047
- 状态/时间：`merged`，created 2025-11-27, merged 2025-12-02；作者 `rogeryoungh`。
- 代码 diff 已读范围：`2` 个文件，`+38/-13`；代码面：model wrapper, MoE/router；关键词：config, expert, topk, cuda, moe, router。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +38/-10 (48 lines); hunk: ); pass; 符号: TopKConfig:, __init__, forward_native, fused_topk_torch_native
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, expert, topk, cuda, moe, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14416 - Fusing RMSNormTP in minimax_m2

- 链接：https://github.com/sgl-project/sglang/pull/14416
- 状态/时间：`merged`，created 2025-12-04, merged 2025-12-30；作者 `rogeryoungh`。
- 代码 diff 已读范围：`1` 个文件，`+189/-2`；代码面：model wrapper；关键词：config, cuda, deepep, expert, kv, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunk: from typing import Iterable, Optional, Set, Tuple, Union; logger = logging.getLogger(__name__); 符号: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, cuda, deepep, expert, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16483 - Optimizing all_reduce in RMSNormTP in minimax_m2

- 链接：https://github.com/sgl-project/sglang/pull/16483
- 状态/时间：`merged`，created 2026-01-05, merged 2026-02-01；作者 `rogeryoungh`。
- 代码 diff 已读范围：`1` 个文件，`+8/-2`；代码面：model wrapper；关键词：triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunk: def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:; def forward(; 符号: rms_sumsq_serial, forward, forward_qk
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17826 - Support Pipeline and Data Parallelism for MiniMax-M2

- 链接：https://github.com/sgl-project/sglang/pull/17826
- 状态/时间：`open`，created 2026-01-27；作者 `rogeryoungh`。
- 代码 diff 已读范围：`1` 个文件，`+167/-70`；代码面：model wrapper；关键词：attention, config, cuda, deepep, eagle, expert, kv, moe, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunk: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; from sglang.srt.distributed import (; 符号: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, config, cuda, deepep, eagle, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18217 - [piecewise graph]: support MiniMax-M2

- 链接：https://github.com/sgl-project/sglang/pull/18217
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-05；作者 `hzh0425`。
- 代码 diff 已读范围：`2` 个文件，`+28/-7`；代码面：model wrapper, quantization, kernel；关键词：config, cuda, deepep, expert, fp8, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunk: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; def op_select_experts(self, state):; 符号: op_select_experts, op_dispatch_a, op_dispatch_b, forward
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +5/-0 (5 lines); hunk: def get_w8a8_block_fp8_configs(; 符号: get_w8a8_block_fp8_configs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`；patch 关键词为 config, cuda, deepep, expert, fp8, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19468 - fix[minimax]: support deepep with minimax models

- 链接：https://github.com/sgl-project/sglang/pull/19468
- 状态/时间：`open`，created 2026-02-27；作者 `ishandhanani`。
- 代码 diff 已读范围：`3` 个文件，`+10/-2`；代码面：kernel；关键词：deepep, config, cuda, doc, flash, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `docker/Dockerfile` modified +2/-1 (3 lines); hunk: ARG HOPPER_SBO=0
  - `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunk: if [ "$GRACE_BLACKWELL" = "1" ]; then
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`；patch 关键词为 deepep, config, cuda, doc, flash, moe。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19577 - [Feat] add PP Support for minimax-m2 series

- 链接：https://github.com/sgl-project/sglang/pull/19577
- 状态/时间：`merged`，created 2026-02-28, merged 2026-03-02；作者 `LuYanFCP`。
- 代码 diff 已读范围：`1` 个文件，`+35/-7`；代码面：model wrapper；关键词：attention, config, eagle, processor, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunk: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; 符号: __init__, forward, load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, config, eagle, processor, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19652 - [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

- 链接：https://github.com/sgl-project/sglang/pull/19652
- 状态/时间：`merged`，created 2026-03-02, merged 2026-04-03；作者 `Godmook`。
- 代码 diff 已读范围：`16` 个文件，`+1410/-95`；代码面：MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：fp4, marlin, quant, fp8, moe, expert, config, flash, topk, triton。
- 代码 diff 细节：
  - `test/registered/quant/test_nvfp4_marlin_fallback.py` added +788/-0 (788 lines); hunk: +"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""; 符号: _check_requirements, _dequant_fp4_weights, _FakeLayer, TestNvfp4MarlinLinear
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunk: is_blackwell_supported,; def get_supported_act_dtypes(cls) -> List[torch.dtype]:; 符号: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunk: CompressedTensorsMoEScheme,; class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):; 符号: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` modified +21/-32 (53 lines); hunk: __global__ void Marlin(; __global__ void Marlin(; 符号: void, int, int, int
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；patch 关键词为 fp4, marlin, quant, fp8, moe, expert。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19995 - Add packed_modules_mapping for MiniMax-M2

- 链接：https://github.com/sgl-project/sglang/pull/19995
- 状态/时间：`merged`，created 2026-03-06, merged 2026-03-18；作者 `trevor-m`。
- 代码 diff 已读范围：`1` 个文件，`+12/-0`；代码面：model wrapper；关键词：config, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunk: def forward(; 符号: forward, MiniMaxM2ForCausalLM, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20031 - fix(minimax): support loading merged expert weights (w13) for awq

- 链接：https://github.com/sgl-project/sglang/pull/20031
- 状态/时间：`open`，created 2026-03-06；作者 `xueliangyang-oeuler`。
- 代码 diff 已读范围：`2` 个文件，`+203/-9`；代码面：model wrapper, tests/benchmarks；关键词：config, expert, moe, spec, attention, processor, quant, test。
- 代码 diff 细节：
  - `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunk: +import unittest; 符号: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13
  - `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights, load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, expert, moe, spec, attention, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20067 - MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn

- 链接：https://github.com/sgl-project/sglang/pull/20067
- 状态/时间：`merged`，created 2026-03-07, merged 2026-04-10；作者 `trevor-m`。
- 代码 diff 已读范围：`3` 个文件，`+39/-6`；代码面：model wrapper, tests/benchmarks；关键词：attention, config, cuda, expert, flash, fp4, kv, moe, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunk: RowParallelLinear,; def forward_normal(; 符号: forward_normal, forward_prepare, forward_prepare, forward_core
  - `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunk: def test_minimax_m25(self):; def test_minimax_m25(self):; 符号: test_minimax_m25, test_minimax_m25
  - `python/sglang/srt/layers/layernorm.py` modified +4/-0 (4 lines); hunk: def forward_cuda(; 符号: forward_cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py`；patch 关键词为 attention, config, cuda, expert, flash, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20489 - fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…

- 链接：https://github.com/sgl-project/sglang/pull/20489
- 状态/时间：`open`，created 2026-03-13；作者 `xueliangyang-oeuler`。
- 代码 diff 已读范围：`5` 个文件，`+118/-20`；代码面：model wrapper, scheduler/runtime；关键词：attention, config, cuda, kv, cache, expert, moe, quant, test。
- 代码 diff 细节：
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunk: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunk: from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo; def rms_apply_serial(; 符号: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunk: def _set_kv_buffer_impl(; 符号: _set_kv_buffer_impl
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunk: def _dummy_run(self, batch_size: int, run_ctx=None):; 符号: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunk: def forward_cuda(; 符号: forward_cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 attention, config, cuda, kv, cache, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20673 - [Feature][JIT Kernel] Fused TP QK norm For Minimax

- 链接：https://github.com/sgl-project/sglang/pull/20673
- 状态/时间：`merged`，created 2026-03-16, merged 2026-04-13；作者 `DarkSharpness`。
- 代码 diff 已读范围：`11` 个文件，`+923/-82`；代码面：model wrapper, kernel, tests/benchmarks；关键词：cuda, config, test, cache, kv, processor, spec, triton, attention, benchmark。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh` added +325/-0 (325 lines); hunk: +// Adapted from https://github.com/NVIDIA/TensorRT-LLM/pull/12163; 符号: ParallelQKNormParams, auto, KernelTrait, parameters
  - `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py` added +170/-0 (170 lines); hunk: +from __future__ import annotations; 符号: parse_args, init_distributed, bench_one, rmsnorm_baseline
  - `python/sglang/jit_kernel/tests/test_tp_qknorm.py` added +168/-0 (168 lines); hunk: +from __future__ import annotations; 符号: test_tp_qknorm, init_distributed, _all_gather_cat, _rmsnorm_ref
  - `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunk: import logging; ); 符号: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm:, __init__
  - `python/sglang/jit_kernel/all_reduce.py` modified +50/-6 (56 lines); hunk: import torch; def config_pull(; 符号: config_pull, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`；patch 关键词为 cuda, config, test, cache, kv, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20870 - [MiniMax M2] Fix KV cache scale loading

- 链接：https://github.com/sgl-project/sglang/pull/20870
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-18；作者 `chadvoegele`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：model wrapper；关键词：cache, expert, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 cache, expert, kv, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20873 - docs: add MiniMax-M2.7 and M2.7-highspeed model support

- 链接：https://github.com/sgl-project/sglang/pull/20873
- 状态/时间：`open`，created 2026-03-18；作者 `octo-patch`。
- 代码 diff 已读范围：`2` 个文件，`+15/-3`；代码面：model wrapper, docs/config；关键词：doc, moe, expert, test。
- 代码 diff 细节：
  - `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunk: -# MiniMax M2.5/M2.1/M2 Usage; curl http://localhost:8000/v1/chat/completions \
  - `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunk: in the GitHub search bar.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`；patch 关键词为 doc, moe, expert, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20905 - [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5

- 链接：https://github.com/sgl-project/sglang/pull/20905
- 状态/时间：`merged`，created 2026-03-19, merged 2026-03-24；作者 `shadowxz109`。
- 代码 diff 已读范围：`2` 个文件，`+22/-30`；代码面：model wrapper, quantization；关键词：config, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +21/-29 (50 lines); hunk: def get_moe_scheme(; 符号: get_moe_scheme, is_layer_skipped
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20967 - 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16

- 链接：https://github.com/sgl-project/sglang/pull/20967
- 状态/时间：`merged`，created 2026-03-20, merged 2026-04-10；作者 `kingkingleeljj`。
- 代码 diff 已读范围：`1` 个文件，`+34/-10`；代码面：model wrapper；关键词：attention, config, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunk: def rms_apply_serial(; def __init__(; 符号: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, config, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20975 - fix(dp-attn): fix issues with dp-attention for MiniMax M2

- 链接：https://github.com/sgl-project/sglang/pull/20975
- 状态/时间：`open`，created 2026-03-20；作者 `xueliangyang-oeuler`。
- 代码 diff 已读范围：`6` 个文件，`+122/-20`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, config, cuda, kv, cache, expert, moe, quant, test。
- 代码 diff 细节：
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunk: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunk: from sglang.kernel_api_logging import debug_kernel_api; def rms_apply_serial(; 符号: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunk: def _set_kv_buffer_impl(; 符号: _set_kv_buffer_impl
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunk: def get_attention_tp_size() -> int:; 符号: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunk: def _dummy_run(self, batch_size: int, run_ctx=None):; 符号: _dummy_run
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 attention, config, cuda, kv, cache, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22300 - [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)

- 链接：https://github.com/sgl-project/sglang/pull/22300
- 状态/时间：`open`，created 2026-04-08；作者 `trevor-m`。
- 代码 diff 已读范围：`3` 个文件，`+30/-6`；代码面：quantization；关键词：fp8, quant, triton, config, flash。
- 代码 diff 细节：
  - `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunk: def post_load_weights(model: nn.Module, model_config: ModelConfig):; 符号: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_deepgemm_weight_requant_ue8m0, should_async_load
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunk: def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; 符号: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunk: def process_weights_after_loading_block_quant(self, layer: Module) -> None:; 符号: process_weights_after_loading_block_quant
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`；patch 关键词为 fp8, quant, triton, config, flash。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22432 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2

- 链接：https://github.com/sgl-project/sglang/pull/22432
- 状态/时间：`open`，created 2026-04-09；作者 `shadowxz109`。
- 代码 diff 已读范围：`1` 个文件，`+69/-11`；代码面：model wrapper；关键词：attention, cache, config, expert, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunk: import logging; ); 符号: forward_prepare, forward_prepare_npu, forward_core, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, cache, config, expert, kv, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22744 - [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance

- 链接：https://github.com/sgl-project/sglang/pull/22744
- 状态/时间：`open`，created 2026-04-14；作者 `trevor-m`。
- 代码 diff 已读范围：`3` 个文件，`+11/-0`；代码面：scheduler/runtime, docs/config；关键词：moe, cache, doc, fp8, kv。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunk: def __init__(; 符号: __init__
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md`；patch 关键词为 moe, cache, doc, fp8, kv。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22934 - Minimax eplb bugfix

- 链接：https://github.com/sgl-project/sglang/pull/22934
- 状态/时间：`open`，created 2026-04-16；作者 `DaZhUUU`。
- 代码 diff 已读范围：`1` 个文件，`+25/-0`；代码面：model wrapper；关键词：attention, config, eagle, expert, moe, quant, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; # Other files (custom_all_reduce.py, hf_transformers_utils.py) also use sglang.srt.utils.; 符号: op_output, get_moe_weights, MiniMaxM2Attention, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, config, eagle, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23190 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode

- 链接：https://github.com/sgl-project/sglang/pull/23190
- 状态/时间：`open`，created 2026-04-20；作者 `heziiop`。
- 代码 diff 已读范围：`1` 个文件，`+66/-10`；代码面：model wrapper；关键词：attention, cache, config, cuda, expert, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunk: import logging; get_compiler_backend,; 符号: forward_prepare, forward_prepare_npu, forward_core, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/minimax_m2.py`；patch 关键词为 attention, cache, config, cuda, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23301 - [sgl] Stream MiniMax M2 string parameters token-by-token

- 链接：https://github.com/sgl-project/sglang/pull/23301
- 状态/时间：`open`，created 2026-04-21；作者 `lujiajing1126`。
- 代码 diff 已读范围：`1` 个文件，`+332/-280`；代码面：misc；关键词：config, spec。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunk: logger = logging.getLogger(__name__); class MinimaxM2Detector(BaseFormatDetector):; 符号: MinimaxM2Detector, MinimaxM2Detector, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/minimax_m2.py`；patch 关键词为 config, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/minimax_m2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：29；open PR 数：12。
- 仍需跟进的 open PR：[#17826](https://github.com/sgl-project/sglang/pull/17826), [#19468](https://github.com/sgl-project/sglang/pull/19468), [#20031](https://github.com/sgl-project/sglang/pull/20031), [#20489](https://github.com/sgl-project/sglang/pull/20489), [#20873](https://github.com/sgl-project/sglang/pull/20873), [#20975](https://github.com/sgl-project/sglang/pull/20975), [#22300](https://github.com/sgl-project/sglang/pull/22300), [#22432](https://github.com/sgl-project/sglang/pull/22432), [#22744](https://github.com/sgl-project/sglang/pull/22744), [#22934](https://github.com/sgl-project/sglang/pull/22934), [#23190](https://github.com/sgl-project/sglang/pull/23190), [#23301](https://github.com/sgl-project/sglang/pull/23301)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
