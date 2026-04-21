# SGLang MiniMax M2 / M2.5 / M2.7 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `47c4b3825`，以及 MiniMax 相关 merged、open PR patch 阅读结果整理。范围覆盖原有 `sglang-minimax-m2-m25-optimization` skill 涉及的主线，并补充 MiniMax M2.7、TP QK RMSNorm allreduce fusion、DP attention、FP4/NVFP4、NPU、DeepEP、EPLB 和 tool-call streaming 的最新状态。

阅读结论先放前面：截至 `47c4b3825`，MiniMax M2 系列主线模型文件是 `python/sglang/srt/models/minimax_m2.py`，它已经支持 M2/M2.1/M2.5 的基础加载、tool calling、reasoning parser、Eagle3 aux hidden states、PP、DP attention 相关 attention-TP 分组、M2.5 reduce-scatter/FP4 all-gather/AR fusion，以及你提到的 TP QK RMSNorm allreduce fusion。M2.7 当前主要体现为文档和同一模型类复用，建议 skill 名称从 `sglang-minimax-m2-m25-optimization` 改成更准确的 `sglang-minimax-m2-series-optimization` 或 `sglang-minimax-m2-m27-optimization`。

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

`#20873` 是 open 的旧文档 PR，增加 MiniMax-M2.7 和 M2.7-highspeed。虽然这个 PR 还没合入，但最新 main 的 `docs_new` 里已经有 `docs_new/cookbook/autoregressive/MiniMax/MiniMax-M2.7.mdx`，并且 cookbook 导航里也有 MiniMax-M2.7。代码层面 M2.7 仍复用 `MiniMaxM2ForCausalLM` 系列模型类，因此 skill 名称最好不要再锁死到 M2.5。

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

## 8. 当前 main 的代码形态与 skill 命名建议

截至 `47c4b3825`，MiniMax 主线形态可以概括为：

- `MiniMaxM2ForCausalLM` 是 M2/M2.1/M2.5/M2.7 系列共用的模型类。
- `MiniMaxM2MoE` 使用 sigmoid top-k、`TopKOutput`、normal/DeepEP 分支、reduce-scatter 和 FP4 all-gather 相关通信控制。
- `MiniMaxM2Attention` 使用 attention TP rank/size，支持 DP attention 的 head partition；Q/K RMSNorm 走 `MiniMaxM2QKRMSNorm`，可通过 `SGLANG_USE_FUSED_PARALLEL_QKNORM` 启用 JIT fused TP QKNorm allreduce。
- `MiniMaxM2DecoderLayer` 通过 `LayerCommunicator` 支持 prepare_attn AR fusion、prepare_mlp、reduce-scatter 和 postprocess。
- loader 已支持 packed mapping、KV scale remap、PP shard skip；AWQ `w13` merged expert loader 仍 open。
- 文档层已经出现 M2.7；代码层仍复用同一个 M2 系列实现。

因此建议把 skill 从 `sglang-minimax-m2-m25-optimization` 改成 `sglang-minimax-m2-series-optimization`。如果希望名字里显式表达最新模型，也可以叫 `sglang-minimax-m2-m27-optimization`，但 “series” 更稳，因为 M2.7、M2.7-highspeed 和后续高性能/平台优化都会继续复用同一模型族实现。
