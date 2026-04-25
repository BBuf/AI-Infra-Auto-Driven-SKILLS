# SGLang Qwen3 Core 支持与优化 PR Diff 历史

本文基于 SGLang mainline `b3e6cf60a`（2026-04-22 附近）和 sgl-cookbook `816bad5`（2026-04-21 附近）整理，覆盖 Qwen3 dense、Qwen3 MoE、Qwen3-30B-A3B、Qwen3-235B-A22B、embedding/pooled-output、parser、量化、PP/DP/EP/CP、EAGLE3、NPU/XPU/MLX 和低延迟文档。

配套 skill 证据账本在 `skills/model-optimization/sglang/sglang-qwen3-core-optimization/references/pr-history.md`。该账本保存更完整的代码块；本 README 保留模型优化历史视角的逐 PR 卡片摘要。每条都来自源码 diff 或最终合入代码，不是标题级摘要。

## 核心结论

Qwen3 Core 是后续 Qwen3.5、Qwen3-Next、Qwen3.6、Qwen3 Omni thinker-only 和 Qwen 系列量化 loader 的基础层。优化主线可以分成五条：

- 模型结构支持：dense Qwen3、Qwen3 MoE、embedding、pooled-output、RoPE 配置兼容。
- 大模型并行：EP/DeepEP/EPLB、PP/tied embedding、DP attention、TBO、context parallel。
- Kernel/后端优化：fused QK-norm/RoPE、fused KV write、FlashInfer fused all-reduce、TRTLLM-GEN-MoE、FP8 KV write。
- 量化和平台：ModelOpt FP8/NVFP4/FP4、Ascend NPU GPTQ/ModelSlim/fuseEP、Intel XPU、Apple MLX、W4AFP8 radar。
- 推理体验：LoRA、EAGLE3/DFLASH、reasoning/tool-call parser、低延迟 cookbook/docs。

## 主要代码面

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/layers/moe/`
- `python/sglang/srt/layers/quantization/`
- `python/sglang/srt/layers/attention/`
- `python/sglang/srt/distributed/`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`
- `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
- `test/registered/npu/`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

## 已合入 PR 卡片

### 模型 bring-up 与配置兼容

- [#4693](https://github.com/sgl-project/sglang/pull/4693) 初始 Qwen3/Qwen3MoE 支持。
  动机：SGLang 需要原生支持 `Qwen3ForCausalLM` 和 `Qwen3MoeForCausalLM`，不能只复用 Qwen2 路径。实现：新增 `qwen3.py`/`qwen3_moe.py`，packed QKV 后拆分 q/k/v，Q/K RMSNorm 在 RoPE 之前执行，MoE 通过 gate + `FusedMoE` 接入。关键片段：`q, k, v = qkv.split(...)`、`q, k = self._apply_qk_norm(q, k)`、`self.experts = FusedMoE(...)`。风险：这是所有后续 Qwen3 dense/MoE 修复的根基。

- [#6990](https://github.com/sgl-project/sglang/pull/6990) Qwen3 Embedding 支持。
  动机：`Qwen/Qwen3-Embedding-8B` 的权重名前缀和早期 Qwen3 loader 不匹配。实现：embedding 模型加载时给未带 `model.` 的名字加前缀。关键片段：`if "Embedding" in self.config.name_or_path: name = add_prefix(name, "model")`。风险：后续 `#17535` 证明只看模型名不够稳。

- [#17535](https://github.com/sgl-project/sglang/pull/17535) Qwen3 embedding weight rename 修正。
  动机：微调 embedding 模型可能没有 `"Embedding"` 字样，旧逻辑会触发 `layers.0.mlp.gate_up_proj.weight` 这类 KeyError。实现：只对未带 `model.` 且以 `layers.`、`embed_tokens.`、`norm.` 开头的根权重加前缀。关键片段：`if not name.startswith("model.") and (name.startswith("layers.") or name.startswith("embed_tokens.") or name.startswith("norm.")):`。

- [#17784](https://github.com/sgl-project/sglang/pull/17784) transformers 兼容升级。
  动机：HF 新版本改变了 RoPE/config 子结构，Qwen 系列可能通过 dict 子配置进入 SGLang。实现：Qwen3 读取 `config.rope_parameters`，共享 helper 兼容 legacy `rope_scaling["type"]`，`get_hf_text_config` 处理 thinker/llm/language/text 子配置。关键片段：`rope_theta = config.rope_parameters.get("rope_theta", 1000000.0)`、`rs["type"] = rs["rope_type"]`。

- [#20931](https://github.com/sgl-project/sglang/pull/20931) Qwen3 MoE RoPE 参数兼容。
  动机：部分 Qwen3 MoE checkpoint 仍使用顶层 `rope_theta`/`rope_scaling`，没有 `rope_parameters`。实现：引入 `get_rope_config(config)`，并把 `self.rope_theta` 传给 fused qk_norm_rope。关键片段：`rope_theta, rope_scaling = get_rope_config(config)`、`self.rope_theta = rope_theta`。

- [#22739](https://github.com/sgl-project/sglang/pull/22739) 恢复 dense Qwen3 RoPE fallback。
  动机：JSON override 可能创建没有 `rope_theta` 的 `rope_parameters`，直接索引会 KeyError。实现：只有 `rope_parameters` 存在且包含 `rope_theta` 时才走新字段，否则回退顶层字段。关键片段：`"rope_theta" in config.rope_parameters`，否则 `getattr(config, "rope_theta", 1000000)`。

### MoE、DeepEP、EPLB 和 dispatch 演进

- [#5917](https://github.com/sgl-project/sglang/pull/5917) Qwen3 EP MoE。
  动机：Qwen3-235B-A22B-FP8 需要 `--enable-ep-moe` 下的 expert parallel。实现：根据 flag 在 `EPMoE` 和 `FusedMoE` 间选择，并复用相同类生成 expert weight mapping。关键片段：`MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE`。

- [#6120](https://github.com/sgl-project/sglang/pull/6120) Qwen3 DeepEP。
  动机：Qwen3 MoE 需要 DeepEP all-to-all dispatch。实现：`enable_deepep_moe` 时选择 `DeepEPMoE`，构造 `DeepEPDispatcher`，通过 `select_experts` 得到 top-k 后 dispatch/combine。关键片段：`MoEImpl = DeepEPMoE if ... else ...`、`self.deepep_dispatcher = DeepEPDispatcher(...)`。验证：Qwen3-235B-A22B-FP8 TP4 DeepEP normal GSM8K `0.970`。

- [#6121](https://github.com/sgl-project/sglang/pull/6121) Qwen2/3 MoE DP attention。
  动机：EP MoE 部署需要 DP attention，issue `#6088` 中 Qwen MoE 不支持。实现：attention 使用 `get_attention_tp_rank/size`，FFN 输入分成 `SCATTERED`/`FULL`，用 `dp_gather_partial`/`dp_scatter` 做通信。关键片段：`self.num_heads = self.total_num_heads // attn_tp_size`、`class _FFNInputMode(Enum)`。

- [#6533](https://github.com/sgl-project/sglang/pull/6533) Qwen3 EPLB。
  动机：Qwen3 MoE 需要 redundant experts 和 Expert Parallel Load Balancing。实现：`get_moe_impl_class()` 创建包含冗余专家的 MoE，收集每层 expert weights，把 `ExpertLocationDispatchInfo` 传入 top-k。关键片段：`num_experts=config.num_experts + global_server_args_dict["ep_num_redundant_experts"]`、`ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)`。

- [#6709](https://github.com/sgl-project/sglang/pull/6709) Qwen3 MoE PP 修复。
  动机：EPLB 收集专家权重时会碰到非本 PP rank 的 `PPMissingLayer`。实现：只遍历 `range(self.start_layer, self.end_layer)`。关键片段：`for layer_id in range(self.start_layer, self.end_layer)`。

- [#6818](https://github.com/sgl-project/sglang/pull/6818) dynamic EPLB 权重引用修正。
  动机：EPLB 过早引用专家权重，PP/local layer 下容易拿错对象。实现：引入 `LazyValue` 惰性收集专家权重，Qwen3 MoE 保持 local-layer collection。关键片段：`self._routed_experts_weights_of_layer = LazyValue(lambda: {...})`。

- [#6964](https://github.com/sgl-project/sglang/pull/6964) EPLB expert distribution exact/approx 统计。
  动机：EPLB 既需要精确 top-k 分布，也需要 DeepEP normal 的近似统计。实现：GPU/CPU gatherer，exact 模式用 `scatter_add_`，Qwen3 的 top-k 包在 `get_global_expert_distribution_recorder().with_current_layer(...)` 中。关键片段：`self._data[layer_idx, :].scatter_add_(...)`。

- [#7580](https://github.com/sgl-project/sglang/pull/7580) EPLB 文件迁移。
  动机：EPLB 已成为独立子系统。实现：把 expert distribution/location/dispatch/updater 移到 `python/sglang/srt/eplb/`，更新 Qwen3 import。关键片段：`from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo`。

- [#8448](https://github.com/sgl-project/sglang/pull/8448) FusedMoE 支持 EPLB。
  动机：`#8398` 暴露 FusedMoE loader 不理解 expert location metadata。实现：`FusedMoE` 接收 `layer_id`，用 `logical_to_all_physical` 把逻辑 expert 映射到所有物理 expert。关键片段：`physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(self.layer_id, expert_id)`。

- [#13715](https://github.com/sgl-project/sglang/pull/13715) EPLB + FP4 兼容。
  动机：ModelOpt FP4 中有 global expert scales、swizzled blockscale 和 scalar 参数，不能按本地 expert 权重 remap。实现：`filter_moe_weight_param_global_expert` 只保留真正以 local expert 为首维的权重。关键片段：`x.data.ndim > 0 and x.data.shape[0] == num_local_experts`。

- [#6820](https://github.com/sgl-project/sglang/pull/6820) Qwen3 MoE token padding 优化修复。
  动机：Qwen3 MoE 未把非 padding token 数传给 top-k，padding 优化失效。实现：`select_experts` 和 `fused_topk` 接收 `num_token_non_padded`。关键片段：`num_token_non_padded=forward_batch.num_token_non_padded`。

- [#7222](https://github.com/sgl-project/sglang/pull/7222) DP attention + DeepEP auto。
  动机：DeepEP `auto` 曾被 DP attention 禁用，但 Qwen3 MoE 需要自动区分 prefill/decode。实现：用 `forward_batch.is_extend_in_batch` resolve DeepEP mode，并把完整 `forward_batch` 传入 experts。关键片段：`resolved_deepep_mode = self.deepep_mode.resolve(forward_batch.is_extend_in_batch)`。

- [#7723](https://github.com/sgl-project/sglang/pull/7723) Qwen MoE FlashInfer flag 修复。
  动机：Qwen MoE 未把 `enable_flashinfer_moe` 传给 `FusedMoE`。实现：全局 flag 打开时才传 `enable_flashinfer_moe=True` 和 EP 状态。关键片段：`dict(enable_flashinfer_moe=True, enable_ep_moe=...) if global_server_args_dict["enable_flashinfer_moe"] else {}`。

- [#7966](https://github.com/sgl-project/sglang/pull/7966) `select_experts` 重构。
  动机：MoE routing 重复且难扩展，输入参数过多。实现：新增 `TopKOutput` 和 `TopK` op，FusedMoE/EPMoE 接收 `topk_output`。关键片段：`class TopKOutput(NamedTuple): topk_weights; topk_ids; router_logits`、`topk_output = self.topk(hidden_states, router_logits)`。

- [#8421](https://github.com/sgl-project/sglang/pull/8421) DeepEP output 简化。
  动机：DeepEP dispatch/combine 不应散落在模型文件中。实现：新增 `DispatchOutputFormat` 和 DeepEP output classes，`DeepEPMoE.forward` 内部完成 dispatch、expert compute、combine。关键片段：`dispatch_output = self.dispatch(...)`、`hidden_states = self.moe_impl(dispatch_output)`、`hidden_states = self.combine(...)`。

- [#8658](https://github.com/sgl-project/sglang/pull/8658) MoE parallelism 参数更新。
  动机：`--enable-ep-moe` / `--enable-deepep-moe` 无法覆盖多种 A2A 后端。实现：新增 `MoeA2ABackend`，旧 flag 转换为新字段，Qwen3 MoE 使用 `moe_a2a_backend` 和 `get_moe_expert_parallel_world_size()`。关键片段：`class MoeA2ABackend(Enum): STANDARD = ("standard", "none"); DEEPEP = "deepep"`。

- [#8751](https://github.com/sgl-project/sglang/pull/8751) Slime update weights 减少 Qwen3 MoE loader 开销。
  动机：重复遍历参数和尝试加载非本 rank expert 权重导致 update-weight 开销。实现：缓存 `params_dict`，提前跳过不在本 rank 的 expert weights，惰性初始化 expert weight map。关键片段：`self._cached_params_dict = dict(self.named_parameters())`、`if is_expert_weight: continue`。

- [#9338](https://github.com/sgl-project/sglang/pull/9338) TopK 可读性和可扩展性重构。
  动机：TopK fix 不能继续硬编码在 DeepSeek 路径，Qwen3 MoE 需要匹配 Triton/FlashInfer/FP4 格式。实现：新增 `TopKOutputFormat`，按 backend/quant 选择 `TRITON_KERNEL`、`BYPASSED` 或 `STANDARD`。关键片段：`elif should_use_flashinfer_trtllm_moe(): output_format = TopKOutputFormat.BYPASSED`。

### PP 和 tied embeddings

- [#6250](https://github.com/sgl-project/sglang/pull/6250) Qwen2/Qwen3 pipeline parallelism。
  动机：大 Qwen3 模型需要 PP 切层。实现：引入 `PPMissingLayer`、`PPProxyTensors`、`get_layer_id`，first rank 放 embedding，last rank 放 norm/logits，loader 跳过非本地层。关键片段：`self.layers, self.start_layer, self.end_layer = make_layers(..., pp_rank=..., pp_size=...)`。

- [#6546](https://github.com/sgl-project/sglang/pull/6546) Qwen PP tied weights。
  动机：`tie_word_embeddings=True` 下 last PP rank 没有 `embed_tokens`，`lm_head` 无法绑定。实现：first rank 发送 embedding weight，last rank recv 并 copy 到 lm_head。关键片段：`self.pp_group.send(self.model.embed_tokens.weight, dst=...)`、`self.lm_head.weight.copy_(emb_token_weight)`。

- [#15223](https://github.com/sgl-project/sglang/pull/15223) Qwen3 PP load 修复。
  动机：Qwen3-0.6B TP2 PP4 启动失败，send/recv rank 和 shape 不对。实现：send 目标改为 `world_size - 1`，recv shape 用 `self.lm_head.weight.shape`。关键片段：`dst=self.pp_group.world_size - 1`、`size=self.lm_head.weight.shape`。

- [#15890](https://github.com/sgl-project/sglang/pull/15890) tied embedding 权重逻辑修正。
  动机：Qwen3-4B PP=2 输出异常，因为 last PP rank 过滤掉 `model.embed_tokens.weight`。实现：loader 看到 embedding weight 且 last rank tied 时，直接加载进 `lm_head.weight`，移除运行时 send/recv 依赖。关键片段：`if name == "model.embed_tokens.weight" and self.pp_group.is_last_rank and self.config.tie_word_embeddings:`。

### DP attention、TBO、CP 和 speculative

- [#6598](https://github.com/sgl-project/sglang/pull/6598) Qwen3 MoE two-batch overlap。
  动机：Qwen3-235B 需要 TBO 叠加 DP attention/DeepEP normal。实现：把 Qwen3 MoE layer 拆成 `op_*` stages，用 `MaybeTboDeepEPDispatcher`，新增 Qwen3 TBO strategy。关键片段：`self.deepep_dispatcher = MaybeTboDeepEPDispatcher(...)`、`_compute_moe_qwen3_layer_operations_strategy_tbo(...)`。

- [#6652](https://github.com/sgl-project/sglang/pull/6652) Qwen3 TBO / DP LM-head 修复。
  动机：TBO 参数和 DP LM-head group 需要修正。实现：`ParallelLMHead(..., use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"])`。关键片段同前。

- [#7681](https://github.com/sgl-project/sglang/pull/7681) dense Qwen3 DP attention。
  动机：dense Qwen3 也需要 TP8 DP8。实现：QKV/o_proj 使用 attention TP rank/size，o_proj `reduce_results=False`，decoder layer 走 `LayerCommunicator`。关键片段：`tp_rank=attn_tp_rank, tp_size=attn_tp_size`、`self.layer_communicator.prepare_attn(...)`。

- [#8280](https://github.com/sgl-project/sglang/pull/8280) DP attention 增强。
  动机：DP attention 的 padding、buffer 和通信需要统一优化。实现：新增 `DPPaddingMode.MAX_LEN/SUM_LEN`，懒分配 gathered buffer，DP+EAGLE CUDA graph 使用 max padded length。关键片段：`if sum_len * 2 > max_len * get_attention_dp_size(): return cls.MAX_LEN`。

- [#9101](https://github.com/sgl-project/sglang/pull/9101) DP attention padding reduce-scatter。
  动机：Qwen3 MoE 在 max padding 下需要 MoE/MLP 后 reduce-scatter。实现：`LayerCommunicator.should_use_reduce_scatter` 控制路径，Qwen3 MoE MLP 接收 `use_reduce_scatter` 并跳过冗余 allreduce。关键片段：`hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)`。

- [#12002](https://github.com/sgl-project/sglang/pull/12002) Qwen3 MoE EAGLE3 DP attention。
  动机：大规模 EP 部署需要 EAGLE3 + DP attention。实现：`prepare_attn_and_capture_last_layer_outputs` gather/clone 捕获 residual，Qwen3 MoE 标记 capture layers，EAGLE worker 在 DP attention 下使用 attention TP group。关键片段：`captured_last_layer_outputs.append(gathered_last_layer_output)`。

- [#18233](https://github.com/sgl-project/sglang/pull/18233) Qwen3 MoE context parallel。
  动机：长上下文 prefill 需要 attention CP 和 MoE topology 协同。实现：FlashAttention backend allgather/rerange KV cache，q 分成 prev/next 两段，Qwen3 MoE 用 MoE TP allreduce。关键片段：`cp_all_gather_rerange_kv_cache(...)`、`q_prev, q_next = torch.chunk(q.contiguous().view(...), 2, dim=0)`。

- [#21195](https://github.com/sgl-project/sglang/pull/21195) 启用 Qwen3 test。
  动机：Qwen3-30B CP 测试可以恢复进 CI。实现：`ep_size > 1` 时恢复 `moe_expert_parallel_all_reduce`，注册 4-GPU H100 test。关键片段：`if self.ep_size > 1 and not should_allreduce_fusion: final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)`。

- [#22003](https://github.com/sgl-project/sglang/pull/22003) `moe_dp_size=1` 支持不同 attention CP size。
  动机：生产希望只给 attention 开 CP，MoE DP 保持 1。实现：`attn_cp_size > moe_dp_size` 时 `_MOE_DP = _ATTN_CP`，新增 `ScatterMode.MOE_FULL`，MoE 前 allgather、后 narrow 回本地 tokens。关键片段：`hidden_states = hidden_states.narrow(0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens).contiguous()`。

- [#22358](https://github.com/sgl-project/sglang/pull/22358) DFLASH 支持。
  动机：z-lab 需要显式 aux hidden capture。实现：Qwen3 dense/MoE 增加 `set_dflash_layers_to_capture`，dense 把 HF after-layer 映射到 SGLang before-next-layer。关键片段：`self.model.layers_to_capture = [val + 1 for val in layer_ids]`。

### 量化、FlashInfer 和 TRTLLM MoE

- [#7912](https://github.com/sgl-project/sglang/pull/7912) Qwen FP8/NVFP4 ModelOpt。
  动机：Qwen ModelOpt checkpoint 需要一行量化启动。实现：`common_group_size` 递归找 group_size 并校验一致，Qwen3 loader remap KV scale name。关键片段：`if len(sizes) > 1: raise ValueError(...)`、`name = maybe_remap_kv_scale_name(name, params_dict)`。

- [#8036](https://github.com/sgl-project/sglang/pull/8036) FlashInfer MoE blockscale FP8。
  动机：低延迟 FP8 MoE 后端，目标 e2e 最高 3x。实现：`FlashInferEPMoE` 调 `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`，loader 调整 w1/w3 布局。关键片段：`return trtllm_fp8_block_scale_moe(..., routing_method_type=2)`。

- [#8450](https://github.com/sgl-project/sglang/pull/8450) FlashInfer TP MoE blockscale FP8。
  动机：`#8036` 只覆盖 EP，TP MoE 也要走 FlashInfer TRTLLM。实现：新增 `FlashInferFusedMoE` 和 `should_use_flashinfer_trtllm_moe()`，EP/TP 分别返回 FlashInfer impl。关键片段：`return FlashInferFusedMoE if should_use_flashinfer_trtllm_moe() else FusedMoE`。

- [#9973](https://github.com/sgl-project/sglang/pull/9973) Qwen3 MoE FlashInfer fused all-reduce。
  动机：profile 中 AllReduce 与 FusedNormAdd 占比高。实现：SM90/SM100 且 tokens<=4096 时用 FlashInfer allreduce fusion，Qwen3 MoE MLP 标记 `_sglang_needs_allreduce_fusion` 并跳过重复 allreduce。关键片段：`hidden_states._sglang_needs_allreduce_fusion = True`。

- [#13489](https://github.com/sgl-project/sglang/pull/13489) FlashInfer TRTLLM-GEN-MoE + Qwen3。
  动机：Qwen3-30B-A3B-Instruct-2507-FP8 应支持 `--moe-runner-backend flashinfer_trtllm --quantization fp8`，SM100 上应能自动选择。实现：Qwen3 MoE 传 `RoutingMethodType.Renormalize`，server args 在 FP8/no A2A/auto 时选择 `flashinfer_trtllm`。关键片段：`self.moe_runner_backend = "flashinfer_trtllm"`。

- [#14093](https://github.com/sgl-project/sglang/pull/14093) TRTLLM MHA fused FP8 KV cache write。
  动机：FP8 KV 路径有四个小 kernel。实现：Triton `_fused_fp8_set_kv_buffer_kernel` 把 quant K/V 和 paged cache write 融合，写完后设置 `k = None; v = None` 跳过通用写。关键片段：`self._fused_fp8_set_kv_buffer(...); k = None; v = None`。

- [#18189](https://github.com/sgl-project/sglang/pull/18189) Qwen3-235B NVFP4 launch 修复。
  动机：235B NVFP4 的 q/k/v 保持 BF16，但 Qwen3 MoE 没有 packed mapping，导致 fused `qkv_proj` 形状错误。实现：加 `packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}`。

### QK-norm、RoPE、KV-store 和 kernel fusion

- [#7740](https://github.com/sgl-project/sglang/pull/7740) Qwen3 two-stream norm。
  动机：Q/K RMSNorm 可以在 CUDA capture 下双 stream overlap。实现：`alt_stream` 传入 Qwen2/Qwen3 dense/MoE，在 capture 模式下 q_norm 当前 stream、k_norm alt stream。关键片段：`with torch.cuda.stream(self.alt_stream): k_by_head = self.k_norm(k_by_head)`。

- [#10749](https://github.com/sgl-project/sglang/pull/10749) Qwen3 MoE RoPE 内 fused KV write。
  动机：decode 可以避免单独 KV cache write。实现：CUDA BF16 KV cache 时创建 `FusedSetKVBufferArg` 传入 RoPE，attention 设置 `save_kv_cache=False`。关键片段：`fused_set_kv_buffer_arg=create_fused_set_kv_buffer_arg(...)`、`save_kv_cache=not enable_fused_set_kv_buffer(forward_batch)`。

- [#13998](https://github.com/sgl-project/sglang/pull/13998) Qwen3-MoE fused qk_norm_rope。
  动机：48 层 decode 中 qk_norm + RoPE overhead 明显。实现：CUDA 下引入 `fused_qk_norm_rope`，非 MRoPE 且 head_dim in `{64,128,256}` 时启用，支持 YaRN 参数。关键片段：`self.use_fused_qk_norm_rope = get_global_server_args().enable_fused_qk_norm_rope and self.compatible_with_fused_qk_norm_rope`。

- [#15835](https://github.com/sgl-project/sglang/pull/15835) JIT fused QK norm。
  动机：AOT/FlashInfer 路径不够通用，小 batch 带宽利用低。实现：新增 `fused_inplace_qknorm` JIT op 和共享 `apply_qk_norm`，移除模型本地重复逻辑。关键片段：`fused_inplace_qknorm(...); return q, k`。

- [#19059](https://github.com/sgl-project/sglang/pull/19059) fused qknorm_rope JIT kernel。
  动机：把 AOT fused qknorm-rope 迁到轻量 JIT，并修 NeoX active_mask UB。实现：注册 `fused_qk_norm_rope_out` 自定义 op，Qwen3 MoE 用 `can_use_fused_qk_norm_rope` gate。关键片段：`@register_custom_op(op_name="fused_qk_norm_rope_out", mutates_args=["qkv"])`。

- [#21654](https://github.com/sgl-project/sglang/pull/21654) fused qknorm_rope 优化。
  动机：JIT kernel 仍重复 `__sincosf` 并使用 `powf`。实现：模板参数 `<head_dim, interleave, yarn>`，两元素一组只算一次 sincos，freq 递推，YaRN 按需编译。关键片段：`template <int head_dim, bool interleave, bool yarn>`、`freq *= freq_ratio`。

### LoRA、EAGLE3、prefill 和共享 plumbing

- [#7312](https://github.com/sgl-project/sglang/pull/7312) Qwen3 LoRA hidden dim。
  动机：issue `#7271` 中 packed projection 的 LoRA 维度推断失败。实现：Qwen3 暂时提供 `get_hidden_dim`，覆盖 qkv/q/kv/o/gate_up/down。关键片段：`elif module_name == "gate_up_proj": return self.config.hidden_size, self.config.intermediate_size`。

- [#8987](https://github.com/sgl-project/sglang/pull/8987) 默认 LoRA hidden dim 修正。
  动机：模型内 override 重复且默认逻辑有误。实现：集中到 `lora/utils.py`，`qkv_proj` 只用于 LoRA A，`q_proj`/`kv_proj` 只用于 LoRA B。关键片段：`if module_name == "qkv_proj": return (config.hidden_size, None)`。

- [#7634](https://github.com/sgl-project/sglang/pull/7634) layer-wise prefill。
  动机：PD multiplexing 需要分段执行 decoder layers。实现：新增 `ForwardMode.SPLIT_PREFILL`，`ForwardBatch` 保存 hidden/residual/model_specific_states，Qwen3/Qwen3 MoE 实现 `forward_split_prefill`。关键片段：`ret = self.model.forward_split_prefill(..., (forward_batch.split_index, next_split_index))`。

- [#7745](https://github.com/sgl-project/sglang/pull/7745) Qwen EAGLE3。
  动机：Qwen3 dense/MoE draft model 需要 aux hidden capture。实现：在指定层前保存 `hidden_states + residual`，传入 `LogitsProcessor`。关键片段：`aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)`。

- [#10975](https://github.com/sgl-project/sglang/pull/10975) `--mem-fraction-static` 通用启发式。
  动机：默认 chunked prefill/cuda graph/mem fraction 太分散。实现：按 GPU memory bucket 选择 `chunked_prefill_size`/`cuda_graph_max_bs`，预留 DP attention/speculative memory。关键片段：`reserved_mem += self.cuda_graph_max_bs * self.dp_size * 3`。

- [#10911](https://github.com/sgl-project/sglang/pull/10911) Qwen3-Omni thinker-only plumbing。
  动机：Qwen3 Omni 需要复用 Qwen3 MoE 语言模型主体。实现：`MRotaryEmbedding.get_rope_index` 支持 `qwen3_omni_moe`，`Qwen3MoeModel` 增加 `decoder_layer_type`。关键片段：`decoder_layer_type=Qwen3MoeDecoderLayer` 参数化。

### Ascend NPU / XPU / MLX

- [#10574](https://github.com/sgl-project/sglang/pull/10574) Ascend Qwen3 优化。
  动机：NPU 上需要内存格式和 CMO prefetch。实现：W8A8 weight cast 到 format 29，新增 CMO stream prefetch，Qwen3 MLP weight 作为 cache 传给 communicator。关键片段：`torch_npu.npu_format_cast(layer.weight.data, 29)`、`cache=[self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight] if _is_npu else None`。

- [#12078](https://github.com/sgl-project/sglang/pull/12078) Ascend Qwen 优化集合。
  动机：修 W8A8 双份内存、CMO deadlock、EPLB static-index、NPU graph、fuseEP。实现：新增 `ASCEND_FUSEEP`、`NpuFuseEPMoE`，Qwen3 调 `split_qkv_rmsnorm_rope`，top-k 用 NPU `l1_norm`。关键片段：`class MoeA2ABackend(Enum): ASCEND_FUSEEP = "ascend_fuseep"`。

- [#15203](https://github.com/sgl-project/sglang/pull/15203) NPU GPTQ quantization。
  动机：NPU roadmap 需要 Qwen3 GPTQ，兼容 GPTQv2 zero-point。实现：新增 `GPTQLinearAscendMethod`，LinearBase 使用 NPU GPTQ，FusedMoE 暂不支持，matmul 用 `torch_npu.npu_weight_quant_batchmatmul`。关键片段：`return GPTQLinearAscendMethod(self)`、`out = torch_npu.npu_weight_quant_batchmatmul(...)`。

- [#15390](https://github.com/sgl-project/sglang/pull/15390) NPU Qwen3 PP bugfix。
  动机：PP 下本地首层不一定是 layer 0，RoPE sin/cos 生成条件错误。实现：`forward_prepare_npu` 接收 `forward_batch`，判断 `self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer`。关键片段同前。

- [#16115](https://github.com/sgl-project/sglang/pull/16115) NPU DP LM-head 修复。
  动机：`--enable-dp-lm-head` 下 split qkv rmsnorm rope 参数和 rotary dtype fallback 出错。实现：BF16 query + float cos/sin 时走 native；`split_qkv_rmsnorm_rope` 使用 named args；LM-head 走 attention TP group。关键片段：`use_attn_tp_group=get_global_server_args().enable_dp_lm_head`。

- [#19532](https://github.com/sgl-project/sglang/pull/19532) NPU speculative inference bugfix。
  动机：EAGLE3 target verify 会让 decode 看起来像 extend，旧 `is_extend()` 条件不够。实现：改成 `is_extend_or_draft_extend_or_mixed()`。关键片段：`or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()`。

## Open PR Radar

- [#9147](https://github.com/sgl-project/sglang/pull/9147) Qwen3-MoE W4AFP8。
  动机：支持 w4afp8-block 静态量化模型。实现草案：`W4AFp8Config` 给 `FusedMoE` 选择 TP/EP method，interleave scales 后调 `cutlass_w4a8_moe`。关键片段：`return cutlass_w4a8_moe(..., w1_q=layer.w13_weight, w2_q=layer.w2_weight, topk_ids_=topk_ids)`。风险：diff 相对当前 TopK/MoeA2ABackend 已陈旧，需要 rebase。

- [#20127](https://github.com/sgl-project/sglang/pull/20127) Qwen MoE/Qwen3Next tied embeddings。
  动机：MoE/Next 仍可能在 tied checkpoint 下创建随机 `ParallelLMHead`。实现草案：world_size==1 时 `self.lm_head = self.model.embed_tokens`，PP last rank 则在 loader 中复制 embedding weight。关键片段：`if self.pp_group.world_size == 1 and config.tie_word_embeddings: self.lm_head = self.model.embed_tokens`。

- [#20474](https://github.com/sgl-project/sglang/pull/20474) Intel XPU Qwen3。
  动机：支持 XPU 的 layernorm gated 和 MRoPE。实现草案：XPU 用 `torch.xpu.get_device_properties(...).gpu_eu_count`，`forward_xpu` 复用 Triton RoPE。关键片段：`def forward_xpu(...): return self.forward_triton(...)`。

- [#20520](https://github.com/sgl-project/sglang/pull/20520) NPU TP 通信压缩。
  动机：Qwen3 NPU prefill 通过 INT8 TP allreduce 降低通信开销。实现草案：`tensor_model_parallel_quant_all_reduce` 动态量化、allgather int8 和 scale、反量化 reduce。关键片段：`x_q, scale = npu_dynamic_quant(x, dst_type=torch.int8)`。

- [#21412](https://github.com/sgl-project/sglang/pull/21412) dense Qwen3 old-style RoPE compat。
  动机：dense Qwen3 对旧字段 checkpoint 也可能 KeyError。实现草案：替换为 `get_rope_config(config)`。关键片段：`rope_theta, rope_scaling = get_rope_config(config)`。

- [#21770](https://github.com/sgl-project/sglang/pull/21770) Apple MLX Qwen3 tests。
  动机：Apple Silicon MLX 初始正确性和 GSM8K 覆盖。实现草案：`SGLANG_USE_MLX=1` 启动，chat template 设 `enable_thinking=False`。关键片段：`env["SGLANG_USE_MLX"] = "1"`。

- [#22529](https://github.com/sgl-project/sglang/pull/22529) Qwen3 sliding window attention。
  动机：新 Qwen3 架构可交替 sliding/full attention。实现草案：`sliding_window - 1` 转为 SGLang exclusive window，按 `layer_types` 判断每层。关键片段：`is_sliding = layer_types[layer_id] == "sliding_attention"`。

- [#22674](https://github.com/sgl-project/sglang/pull/22674) NPU Qwen3.5-MoE/Qwen3-Next quant mapping。
  动机：GDN linear attention packed names 没被 loader 覆盖。实现草案：补 `in_proj_qkvz` 和 `in_proj_ba`。关键片段：`"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]`。

- [#22837](https://github.com/sgl-project/sglang/pull/22837) Qwen3 reasoning detector tool_call 修复。
  动机：`<tool_call>` 在 `</think>` 前出现会被吞进 reasoning_content。实现草案：给 base detector 传 `tool_start_token="<tool_call>"`，补 streaming/non-streaming tests。关键片段：`tool_start_token="<tool_call>"`。

- [#23372](https://github.com/sgl-project/sglang/pull/23372) NPU speculative decoding CI。
  动机：验证 A2/A3 上 EAGLE3/NEXTN、draft attention backend、token map 和 `ascend_fuseep`。实现草案：Qwen3-32B W8A8 + EAGLE3 PD 测试，注册 nightly 8-NPU A3。关键片段：`"--speculative-attention-mode", "decode"`、`register_npu_ci(..., suite="nightly-8-npu-a3")`。

- [#23397](https://github.com/sgl-project/sglang/pull/23397) Dense deterministic math。
  动机：对齐 Megatron on-policy scoring，降低 rollout/training logprob diff。实现草案：禁用部分 fusion，强制 BF16 dense math，q/k norm 用 FP32 weight，TP-invariant tree allreduce。关键片段：`get_on_policy_rms_norm_kwargs(weight_dtype=torch.float32)`。

- [#23434](https://github.com/sgl-project/sglang/pull/23434) Qwen3 pooled output embedding accessor。
  动机：Qwen3 reranker/seq-cls 缺 `get_input_embeddings`，score API embedding override 失败。实现草案：`Qwen3ForPooledOutput` 转发到底层 model。关键片段：`return self.model.get_input_embeddings()`。

## SGLang 文档和 cookbook PR

- [#22429](https://github.com/sgl-project/sglang/pull/22429) Qwen3-32B/8B Ascend 低延迟文档。
  动机：补 A3/A2 低延迟配置。实现：文档命令包含 `--attention-backend ascend`、`--device npu`、`--quantization modelslim`、`--speculative-algorithm EAGLE3`、`--dtype bfloat16`。

- [#22446](https://github.com/sgl-project/sglang/pull/22446) Qwen3-30B-A3B 低延迟文档。
  动机：补 Qwen3-30B-A3B Ascend 低延迟示例。实现：文档命令包含 `--tp-size 2`、`--mem-fraction-static 0.6/0.7`、EAGLE3。关键片段：`--tp-size 2 --mem-fraction-static 0.6 --attention-backend ascend`。

- [#22687](https://github.com/sgl-project/sglang/pull/22687) Qwen3-8B/32B 文档修复。
  动机：清理错误命令。实现：删除 `export HCCL_BUFFSIZE=400` 和重复 `--speculative-draft-model-quantization unquant`。关键片段：`-export HCCL_BUFFSIZE=400`。

- [#22450](https://github.com/sgl-project/sglang/pull/22450) open，Qwen3-14B Ascend 低延迟文档。
  动机：补 Qwen3-14B A3 配置。实现草案：`--quantization modelslim`、`--sampling-backend ascend`、EAGLE3、`--schedule-conservativeness 0.01`。

- [sgl-cookbook #74](https://github.com/sgl-project/sgl-cookbook/pull/74) Qwen3 AMD 和 tool-calling 文档。
  动机：cookbook 需要 Qwen3 AMD 示例和 tool calling 修正。实现：更新 cookbook markdown/命令。风险：这是复现上下文，不等价于 runtime 支持。

- [sgl-cookbook #245](https://github.com/sgl-project/sgl-cookbook/pull/245) Qwen cookbook refresh。
  动机：Qwen3/Qwen3.5/Qwen3-Next 之后 cookbook 内容需要刷新。实现：更新 Qwen 相关页面、示例和链接。

## 下一步建议

1. 优先跟进 open radar：`#22837` parser、`#22529` sliding window、`#20127` MoE tied embeddings、`#20520` NPU communication compression、`#9147` W4AFP8。
2. Qwen3 Core 的任何 shared helper 改动都要回归 Qwen3.5/Qwen3-Next，因为它们复用 packed mapping、RoPE fallback、TopK/MoE 和量化 loader。
3. 新增文档时继续使用 `skills/model-optimization/model-pr-diff-dossier` 的标准：逐 PR 读 diff，写 motivation、实现思路、关键代码片段和验证风险。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Core`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-03-23 | [#4693](https://github.com/sgl-project/sglang/pull/4693) | merged | [Model] Adding Qwen3 and Qwen3MoE | model wrapper, attention/backend, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-04-30 | [#5917](https://github.com/sgl-project/sglang/pull/5917) | merged | [qwen3] support qwen3 ep moe | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-08 | [#6120](https://github.com/sgl-project/sglang/pull/6120) | merged | Support qwen3 deepep | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-05-08 | [#6121](https://github.com/sgl-project/sglang/pull/6121) | merged | feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088 | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2025-05-13 | [#6250](https://github.com/sgl-project/sglang/pull/6250) | merged | Add pipeline parallelism for Qwen2 and Qwen3 Model | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-05-22 | [#6533](https://github.com/sgl-project/sglang/pull/6533) | merged | support eplb for qwen3 | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py` |
| 2025-05-23 | [#6546](https://github.com/sgl-project/sglang/pull/6546) | merged | added support for tied weights in qwen pipeline parallelism | model wrapper, tests/benchmarks | `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-05-25 | [#6598](https://github.com/sgl-project/sglang/pull/6598) | merged | qwen3moe support two batch overlap | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py` |
| 2025-05-27 | [#6652](https://github.com/sgl-project/sglang/pull/6652) | merged | Fix qwen3 tbo/dp-lm-head | model wrapper, MoE/router | `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-28 | [#6709](https://github.com/sgl-project/sglang/pull/6709) | merged | Fix PP for Qwen3 MoE | model wrapper, MoE/router, tests/benchmarks | `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-02 | [#6818](https://github.com/sgl-project/sglang/pull/6818) | merged | Fix wrong weight reference in dynamic EPLB | model wrapper, MoE/router | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-03 | [#6820](https://github.com/sgl-project/sglang/pull/6820) | merged | Fix Qwen3MoE missing token padding optimization | model wrapper, MoE/router | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-08 | [#6964](https://github.com/sgl-project/sglang/pull/6964) | merged | Support both approximate and exact expert distribution collection | model wrapper, MoE/router | `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-09 | [#6990](https://github.com/sgl-project/sglang/pull/6990) | merged | support qwen3 emebedding | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py` |
| 2025-06-16 | [#7222](https://github.com/sgl-project/sglang/pull/7222) | merged | DP Attention with Auto DeepEP Dispatch | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-18 | [#7312](https://github.com/sgl-project/sglang/pull/7312) | merged | Add get_hidden_dim to qwen3.py for correct lora | model wrapper, scheduler/runtime, tests/benchmarks | `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py` |
| 2025-06-27 | [#7580](https://github.com/sgl-project/sglang/pull/7580) | merged | Move files related to EPLB | model wrapper, MoE/router, scheduler/runtime, tests/benchmarks | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-29 | [#7634](https://github.com/sgl-project/sglang/pull/7634) | merged | [Feature] Layer-wise Prefill | model wrapper, MoE/router, scheduler/runtime | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py` |
| 2025-07-01 | [#7681](https://github.com/sgl-project/sglang/pull/7681) | merged | support qwen3 dense model dp attention | model wrapper | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-07-02 | [#7723](https://github.com/sgl-project/sglang/pull/7723) | merged | [Bug] add flashinfer bool check for fusedmoe in Qwen moe models | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-03 | [#7740](https://github.com/sgl-project/sglang/pull/7740) | merged | [optimize] add two stream norm for qwen3 | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-07-03 | [#7745](https://github.com/sgl-project/sglang/pull/7745) | merged | [feat] Support EAGLE3 for Qwen | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-07-09 | [#7912](https://github.com/sgl-project/sglang/pull/7912) | merged | Qwen FP8/NVFP4 ModelOPT Quantization support | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-07-11 | [#7966](https://github.com/sgl-project/sglang/pull/7966) | merged | [1/N] MoE Refactor: refactor `select_experts` | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-07-15 | [#8036](https://github.com/sgl-project/sglang/pull/8036) | merged | [NVIDIA] Add Flashinfer MoE blockscale fp8 backend | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-07-23 | [#8280](https://github.com/sgl-project/sglang/pull/8280) | merged | DP Enhancement | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-07-27 | [#8421](https://github.com/sgl-project/sglang/pull/8421) | merged | [3/N] MoE Refactor: Simplify DeepEP Output | model wrapper, MoE/router | `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-28 | [#8448](https://github.com/sgl-project/sglang/pull/8448) | merged | Support EPLB in FusedMoE | model wrapper, MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-07-28 | [#8450](https://github.com/sgl-project/sglang/pull/8450) | merged | [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2025-08-01 | [#8658](https://github.com/sgl-project/sglang/pull/8658) | merged | [5/N] MoE Refactor: Update MoE parallelism arguments | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-08-04 | [#8751](https://github.com/sgl-project/sglang/pull/8751) | merged | [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-08-08 | [#8987](https://github.com/sgl-project/sglang/pull/8987) | merged | Fix incorrect default get_hidden_dim logic | model wrapper | `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py` |
| 2025-08-12 | [#9101](https://github.com/sgl-project/sglang/pull/9101) | merged | Feature: support qwen and llama4 reducescatter for dp attention padding | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py` |
| 2025-08-13 | [#9147](https://github.com/sgl-project/sglang/pull/9147) | open | support Qwen3-MoE-w4afp8 | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` |
| 2025-08-19 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | model wrapper, MoE/router, kernel | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-03 | [#9973](https://github.com/sgl-project/sglang/pull/9973) | merged | Optimize Qwen3-moe model by using flashinfer fused allreduce | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-09-17 | [#10574](https://github.com/sgl-project/sglang/pull/10574) | merged | [Ascend]optimize Qwen3 on Ascend | model wrapper, quantization, scheduler/runtime | `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-09-22 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | model wrapper, MoE/router | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-09-25 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-09-27 | [#10975](https://github.com/sgl-project/sglang/pull/10975) | merged | Use more general heuristics to set the default value of --mem-fraction-static | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml` |
| 2025-10-23 | [#12002](https://github.com/sgl-project/sglang/pull/12002) | merged | Eagle3 DP attention for Qwen3 MoE | model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks | `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-10-24 | [#12078](https://github.com/sgl-project/sglang/pull/12078) | merged | [Ascend] qwen optimization | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` |
| 2025-11-18 | [#13489](https://github.com/sgl-project/sglang/pull/13489) | merged | Flashinfer TRTLLM-GEN-MoE + Qwen3 | model wrapper, MoE/router | `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-11-21 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | model wrapper, MoE/router | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-11-26 | [#13998](https://github.com/sgl-project/sglang/pull/13998) | merged | [apply][2/2] Fused qk_norm_rope for Qwen3-MoE | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py` |
| 2025-11-28 | [#14093](https://github.com/sgl-project/sglang/pull/14093) | merged | Add fused FP8 KV cache write kernel for TRTLLM MHA backend | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py` |
| 2025-12-15 | [#15203](https://github.com/sgl-project/sglang/pull/15203) | merged | [NPU] support GPTQ quantization on npu | model wrapper, quantization, tests/benchmarks | `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-12-16 | [#15223](https://github.com/sgl-project/sglang/pull/15223) | merged | [bug fix][pp] fix qwen3 model load | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2025-12-18 | [#15390](https://github.com/sgl-project/sglang/pull/15390) | merged | [NPU]qwen3 pp bugfix | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-12-25 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py` |
| 2025-12-26 | [#15890](https://github.com/sgl-project/sglang/pull/15890) | merged | [PP] fix wrong weight logic for tie_word_embeddings model | model wrapper | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2025-12-30 | [#16115](https://github.com/sgl-project/sglang/pull/16115) | merged | [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head | model wrapper, MoE/router | `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-01-21 | [#17535](https://github.com/sgl-project/sglang/pull/17535) | merged | Update weight rename check for Qwen3 Embeddings | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-01-26 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | model wrapper, MoE/router, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py` |
| 2026-02-03 | [#18189](https://github.com/sgl-project/sglang/pull/18189) | merged | [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-02-04 | [#18233](https://github.com/sgl-project/sglang/pull/18233) | merged | Support Qwen3 MoE context parallel | model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2026-02-20 | [#19059](https://github.com/sgl-project/sglang/pull/19059) | merged | [jit_kernel] Add fused_qknorm_rope JIT kernel | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` |
| 2026-02-28 | [#19532](https://github.com/sgl-project/sglang/pull/19532) | merged | [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe | model wrapper, MoE/router | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-08 | [#20127](https://github.com/sgl-project/sglang/pull/20127) | open | [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-12 | [#20474](https://github.com/sgl-project/sglang/pull/20474) | open | Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3 | attention/backend, tests/benchmarks | `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py` |
| 2026-03-13 | [#20520](https://github.com/sgl-project/sglang/pull/20520) | open | [NPU]TP Communications compression For Qwen3 models for NPU | model wrapper, quantization, tests/benchmarks, docs/config | `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py` |
| 2026-03-19 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-23 | [#21195](https://github.com/sgl-project/sglang/pull/21195) | merged | Enable the qwen3 test | model wrapper, MoE/router, tests/benchmarks | `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-25 | [#21412](https://github.com/sgl-project/sglang/pull/21412) | open | [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-03-30 | [#21654](https://github.com/sgl-project/sglang/pull/21654) | merged | [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py` |
| 2026-03-31 | [#21770](https://github.com/sgl-project/sglang/pull/21770) | open | [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon | model wrapper, tests/benchmarks | `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py` |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2026-04-08 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-09 | [#22429](https://github.com/sgl-project/sglang/pull/22429) | merged | [NPU]add Qwen3-32b and Qwen3-8b low latency md | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22446](https://github.com/sgl-project/sglang/pull/22446) | merged | [NPU] add qwen3-30b-a3b low latency example | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22450](https://github.com/sgl-project/sglang/pull/22450) | open | [NPU] Add Qwen3-14B low latency doc | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-10 | [#22529](https://github.com/sgl-project/sglang/pull/22529) | open | [Model] Support sliding window attention for Qwen3 | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-13 | [#22674](https://github.com/sgl-project/sglang/pull/22674) | open | [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization | misc | `python/sglang/srt/model_loader/loader.py` |
| 2026-04-13 | [#22687](https://github.com/sgl-project/sglang/pull/22687) | merged | [NPU]qwen3-8b and 32b md bugfix | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-14 | [#22739](https://github.com/sgl-project/sglang/pull/22739) | merged | Restore Qwen3 rope config fallback | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-15 | [#22837](https://github.com/sgl-project/sglang/pull/22837) | open | [Bug] Qwen3 reasoning detector silently swallows tool_call when </think> is missing | tests/benchmarks | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-21 | [#23372](https://github.com/sgl-project/sglang/pull/23372) | open | [NPU] Add CI tests for Speculative Decoding | attention/backend, MoE/router, tests/benchmarks | `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` |
| 2026-04-21 | [#23397](https://github.com/sgl-project/sglang/pull/23397) | open | [alignment-sglang] PR3: Dense Deterministic Math | model wrapper, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py` |
| 2026-04-22 | [#23434](https://github.com/sgl-project/sglang/pull/23434) | open | [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model | model wrapper | `python/sglang/srt/models/qwen3_classification.py` |

### 逐 PR 代码 diff 阅读记录

### PR #4693 - [Model] Adding Qwen3 and Qwen3MoE

- 链接：https://github.com/sgl-project/sglang/pull/4693
- 状态/时间：`merged`，created 2025-03-23, merged 2025-04-18；作者 `yhyang201`。
- 代码 diff 已读范围：`5` 个文件，`+780/-14`；代码面：model wrapper, attention/backend, MoE/router；关键词：config, quant, attention, expert, kv, moe, processor, spec, cache, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunk: +# Adapted from qwen2_moe.py; 符号: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention
  - `python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunk: +# Adapted from qwen2.py; 符号: Qwen3Attention, __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +13/-11 (24 lines); hunk: from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +5/-2 (7 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/qwen2.py` modified +4/-1 (5 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 config, quant, attention, expert, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5917 - [qwen3] support qwen3 ep moe

- 链接：https://github.com/sgl-project/sglang/pull/5917
- 状态/时间：`merged`，created 2025-04-30, merged 2025-04-30；作者 `laixinn`。
- 代码 diff 已读范围：`2` 个文件，`+16/-6`；代码面：model wrapper, MoE/router；关键词：attention, config, expert, moe, processor, quant, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-3 (11 lines); hunk: RowParallelLinear,; VocabParallelEmbedding,; 符号: __init__, load_weights
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunk: RowParallelLinear,; ParallelLMHead,; 符号: __init__, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 attention, config, expert, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6120 - Support qwen3 deepep

- 链接：https://github.com/sgl-project/sglang/pull/6120
- 状态/时间：`merged`，created 2025-05-08, merged 2025-05-22；作者 `sleepcoo`。
- 代码 diff 已读范围：`2` 个文件，`+125/-8`；代码面：model wrapper, MoE/router；关键词：moe, attention, config, deepep, expert, fp8, processor, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunk: get_pp_group,; RowParallelLinear,; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +4/-1 (5 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 moe, attention, config, deepep, expert, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6121 - feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088

- 链接：https://github.com/sgl-project/sglang/pull/6121
- 状态/时间：`merged`，created 2025-05-08, merged 2025-05-16；作者 `Fr4nk1inCs`。
- 代码 diff 已读范围：`4` 个文件，`+449/-70`；代码面：model wrapper, attention/backend, MoE/router, tests/benchmarks；关键词：attention, moe, config, deepep, expert, kv, processor, quant, triton, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunk: # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py; tensor_model_parallel_all_reduce,; 符号: __init__, forward, __init__, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunk: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; tensor_model_parallel_all_reduce,; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunk: def get_local_attention_dp_size():; 符号: get_local_attention_dp_size, get_local_attention_dp_rank, get_local_attention_dp_size, disable_dp_size
  - `python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunk: def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):; 符号: _maybe_prepare_dp_attn_batch
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`；patch 关键词为 attention, moe, config, deepep, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6250 - Add pipeline parallelism for Qwen2 and Qwen3 Model

- 链接：https://github.com/sgl-project/sglang/pull/6250
- 状态/时间：`merged`，created 2025-05-13, merged 2025-05-18；作者 `libratiger`。
- 代码 diff 已读范围：`5` 个文件，`+340/-73`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：attention, config, processor, quant, cache, moe, expert, kv, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2.py` modified +95/-26 (121 lines); hunk: # Adapted from llama2.py; from sglang.srt.layers.quantization.base_config import QuantizationConfig; 符号: Qwen2MLP, __init__, __init__, get_input_embedding
  - `python/sglang/srt/models/qwen2_moe.py` modified +89/-27 (116 lines); hunk: # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py; from sglang.srt.layers.quantization.base_config import QuantizationCon; 符号: Qwen2MoeMLP, __init__, __init__, forward
  - `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunk: # Adapted from qwen2.py; from sglang.srt.layers.quantization.base_config import QuantizationConfig; 符号: Qwen3Attention, __init__, __init__, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunk: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; from transformers.configuration_utils import PretrainedConfig; 符号: Qwen3MoeSparseMoeBlock, __init__, __init__, forward
  - `test/srt/test_pp_single_node.py` modified +55/-0 (55 lines); hunk: """; def test_gsm8k(self):; 符号: test_gsm8k, TestQwenPPAccuracy, setUpClass, run_gsm8k_test
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, config, processor, quant, cache, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6533 - support eplb for qwen3

- 链接：https://github.com/sgl-project/sglang/pull/6533
- 状态/时间：`merged`，created 2025-05-22, merged 2025-05-24；作者 `yizhang2077`。
- 代码 diff 已读范围：`3` 个文件，`+46/-25`；代码面：model wrapper, MoE/router；关键词：expert, moe, router, topk, config, deepep, fp8, processor, quant, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunk: RowParallelLinear,; ParallelLMHead,; 符号: Qwen3MoeSparseMoeBlock, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/topk.py` modified +4/-2 (6 lines); hunk: def fused_topk(; def fused_topk(; 符号: fused_topk, fused_topk, select_experts
  - `python/sglang/srt/managers/expert_distribution.py` modified +3/-1 (4 lines); hunk: def _convert_global_physical_count_to_logical_count(; 符号: _convert_global_physical_count_to_logical_count
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py`；patch 关键词为 expert, moe, router, topk, config, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6546 - added support for tied weights in qwen pipeline parallelism

- 链接：https://github.com/sgl-project/sglang/pull/6546
- 状态/时间：`merged`，created 2025-05-23, merged 2025-05-25；作者 `FrankLeeeee`。
- 代码 diff 已读范围：`4` 个文件，`+134/-20`；代码面：model wrapper, tests/benchmarks；关键词：config, processor, quant, test, vision, attention。
- 代码 diff 细节：
  - `test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunk: def test_pp_consistency(self):; 符号: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test
  - `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunk: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; 符号: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunk: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: __init__, load_weights
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunk: jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`；patch 关键词为 config, processor, quant, test, vision, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6598 - qwen3moe support two batch overlap

- 链接：https://github.com/sgl-project/sglang/pull/6598
- 状态/时间：`merged`，created 2025-05-25, merged 2025-05-26；作者 `yizhang2077`。
- 代码 diff 已读范围：`5` 个文件，`+351/-28`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：moe, deepep, expert, attention, config, cuda, kv, mla, processor, router。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunk: ParallelLMHead,; from sglang.srt.model_loader.weight_utils import default_weight_loader; 符号: __init__, forward_deepep, forward_deepep, op_gate
  - `python/sglang/srt/operations_strategy.py` modified +98/-7 (105 lines); hunk: def init_new_tbo(; def _compute_moe_deepseek_blog_decode(layer):; 符号: init_new_tbo, _assert_all_same, _compute_layer_operations_strategy_tbo, _compute_moe_deepseek_layer_operations_strategy_tbo
  - `test/srt/test_two_batch_overlap.py` modified +28/-0 (28 lines); hunk: from sglang.srt.utils import kill_process_tree; def test_compute_split_seq_index(self):; 符号: test_compute_split_seq_index, TestQwen3TwoBatchOverlap, setUpClass
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-6 (23 lines); hunk: from sglang.srt.managers.schedule_batch import global_server_args_dict; def forward(; 符号: forward
  - `python/sglang/srt/two_batch_overlap.py` modified +8/-4 (12 lines); hunk: def model_forward_maybe_tbo(; def _model_forward_tbo_split_inputs(; 符号: model_forward_maybe_tbo, _model_forward_tbo_split_inputs, _model_forward_tbo_split_inputs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py`；patch 关键词为 moe, deepep, expert, attention, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/operations_strategy.py`, `test/srt/test_two_batch_overlap.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6652 - Fix qwen3 tbo/dp-lm-head

- 链接：https://github.com/sgl-project/sglang/pull/6652
- 状态/时间：`merged`，created 2025-05-27, merged 2025-05-27；作者 `yizhang2077`。
- 代码 diff 已读范围：`3` 个文件，`+3/-1`；代码面：model wrapper, MoE/router；关键词：config, moe, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/two_batch_overlap.py` modified +1/-1 (2 lines); hunk: def model_forward_maybe_tbo(; 符号: model_forward_maybe_tbo
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 config, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/two_batch_overlap.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6709 - Fix PP for Qwen3 MoE

- 链接：https://github.com/sgl-project/sglang/pull/6709
- 状态/时间：`merged`，created 2025-05-28, merged 2025-05-29；作者 `jinyouzhi`。
- 代码 diff 已读范围：`2` 个文件，`+60/-4`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：moe, expert, test。
- 代码 diff 细节：
  - `test/srt/test_pp_single_node.py` modified +57/-1 (58 lines); hunk: def test_pp_consistency(self):; def test_pp_consistency(self):; 符号: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, test_pp_consistency
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 moe, expert, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6818 - Fix wrong weight reference in dynamic EPLB

- 链接：https://github.com/sgl-project/sglang/pull/6818
- 状态/时间：`merged`，created 2025-06-02, merged 2025-06-03；作者 `fzyzcjy`。
- 代码 diff 已读范围：`3` 个文件，`+27/-13`；代码面：model wrapper, MoE/router；关键词：config, expert, moe, attention, deepep, kv, processor, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunk: from sglang.srt.utils import (; def __init__(; 符号: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunk: def support_triton(backend: str) -> bool:; 符号: support_triton, cpu_has_amx_support, LazyValue:, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunk: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 config, expert, moe, attention, deepep, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6820 - Fix Qwen3MoE missing token padding optimization

- 链接：https://github.com/sgl-project/sglang/pull/6820
- 状态/时间：`merged`，created 2025-06-03, merged 2025-06-05；作者 `fzyzcjy`。
- 代码 diff 已读范围：`2` 个文件，`+5/-3`；代码面：model wrapper, MoE/router；关键词：expert, moe, topk, deepep, router。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +3/-3 (6 lines); hunk: def fused_topk(; def fused_topk(; 符号: fused_topk, fused_topk, select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunk: def forward_deepep(; def op_select_experts(self, state):; 符号: forward_deepep, op_select_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 expert, moe, topk, deepep, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6964 - Support both approximate and exact expert distribution collection

- 链接：https://github.com/sgl-project/sglang/pull/6964
- 状态/时间：`merged`，created 2025-06-08, merged 2025-06-10；作者 `fzyzcjy`。
- 代码 diff 已读范围：`4` 个文件，`+101/-71`；代码面：model wrapper, MoE/router；关键词：expert, topk, moe, router, cuda, deepep。
- 代码 diff 细节：
  - `python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunk: def init_new(; def on_forward_pass_start(self, forward_batch: ForwardBatch):; 符号: init_new, __init__, on_forward_pass_start, on_select_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunk: def op_select_experts(self, state):; 符号: op_select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunk: def op_select_experts(self, state):; 符号: op_select_experts
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunk: class ServerArgs:; 符号: ServerArgs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 expert, topk, moe, router, cuda, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6990 - support qwen3 emebedding

- 链接：https://github.com/sgl-project/sglang/pull/6990
- 状态/时间：`merged`，created 2025-06-09, merged 2025-06-09；作者 `Titan-p`。
- 代码 diff 已读范围：`2` 个文件，`+3/-0`；代码面：model wrapper, tests/benchmarks；关键词：config, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
  - `test/srt/models/test_embedding_models.py` modified +1/-0 (1 lines); hunk: ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1, 1e-5),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py`；patch 关键词为 config, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `test/srt/models/test_embedding_models.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7222 - DP Attention with Auto DeepEP Dispatch

- 链接：https://github.com/sgl-project/sglang/pull/7222
- 状态/时间：`merged`，created 2025-06-16, merged 2025-07-05；作者 `ch-wan`。
- 代码 diff 已读范围：`13` 个文件，`+136/-90`；代码面：model wrapper, MoE/router, scheduler/runtime, tests/benchmarks；关键词：deepep, moe, topk, expert, attention, cuda, quant, spec, fp8, scheduler。
- 代码 diff 细节：
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-40 (120 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass, setUpClass, setUpClass
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunk: deepep_post_reorder_triton_kernel,; def dispatch_a(; 符号: dispatch_a, dispatch_b, combine, combine_a
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunk: def forward_deepep(; def forward_deepep(; 符号: forward_deepep, forward_deepep, op_dispatch_a, op_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunk: def forward_deepep(; def forward_deepep(; 符号: forward_deepep, forward_deepep, op_dispatch_a, op_experts
  - `python/sglang/srt/two_batch_overlap.py` modified +7/-3 (10 lines); hunk: ); def replay_prepare(; 符号: replay_prepare, TboDPAttentionPreparer:, prepare_all_gather, prepare_all_gather
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 deepep, moe, topk, expert, attention, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7312 - Add get_hidden_dim to qwen3.py for correct lora

- 链接：https://github.com/sgl-project/sglang/pull/7312
- 状态/时间：`merged`，created 2025-06-18, merged 2025-07-20；作者 `logachevpa`。
- 代码 diff 已读范围：`5` 个文件，`+240/-2`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：lora, test, config, attention, cache, cuda, kv, spec, triton。
- 代码 diff 细节：
  - `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models
  - `python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunk: def __init__(; 符号: __init__, get_input_embeddings, get_hidden_dim, forward
  - `python/sglang/test/runners.py` modified +6/-1 (7 lines); hunk: def __init__(; def start_model_process(self, in_queue, out_queue, model_path, torch_dtype):; 符号: __init__, start_model_process, forward_generation_raw, forward_generation_raw
  - `test/srt/models/lora/test_lora.py` modified +0/-1 (1 lines); hunk: def ensure_reproducibility(self):; 符号: ensure_reproducibility, _run_lora_multiple_batch_on_model_cases
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: class TestFile:; 符号: TestFile:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py`；patch 关键词为 lora, test, config, attention, cache, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/test/runners.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7580 - Move files related to EPLB

- 链接：https://github.com/sgl-project/sglang/pull/7580
- 状态/时间：`merged`，created 2025-06-27, merged 2025-06-29；作者 `fzyzcjy`。
- 代码 diff 已读范围：`22` 个文件，`+42/-54`；代码面：model wrapper, MoE/router, scheduler/runtime, tests/benchmarks；关键词：expert, moe, config, attention, cuda, quant, deepep, fp8, kv, lora。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunk: set_mscclpp_all_reduce,; from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunk: get_tensor_model_parallel_world_size,; ParallelLMHead,
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunk: parallel_state,; ParallelLMHead,
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunk: tensor_model_parallel_all_gather,; ParallelLMHead,
  - `python/sglang/srt/eplb/eplb_manager.py` renamed +2/-4 (6 lines); hunk: import torch.cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 expert, moe, config, attention, cuda, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7634 - [Feature] Layer-wise Prefill

- 链接：https://github.com/sgl-project/sglang/pull/7634
- 状态/时间：`merged`，created 2025-06-29, merged 2025-07-16；作者 `jason-fxz`。
- 代码 diff 已读范围：`13` 个文件，`+464/-2`；代码面：model wrapper, MoE/router, scheduler/runtime；关键词：config, processor, expert, moe, cuda, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunk: def forward(; 符号: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunk: def forward(; 符号: forward, forward_split_prefill, get_hidden_dim
  - `python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunk: def forward(; 符号: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, forward_split_prefill, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +43/-0 (43 lines); hunk: def forward(; 符号: forward, forward_split_prefill, start_layer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`；patch 关键词为 config, processor, expert, moe, cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7681 - support qwen3 dense model dp attention

- 链接：https://github.com/sgl-project/sglang/pull/7681
- 状态/时间：`merged`，created 2025-07-01, merged 2025-07-03；作者 `yizhang2077`。
- 代码 diff 已读范围：`2` 个文件，`+49/-17`；代码面：model wrapper；关键词：attention, config, kv, quant, cache, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunk: split_tensor_along_last_dim,; def __init__(; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/qwen2.py` modified +7/-1 (8 lines); hunk: ParallelLMHead,; def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`；patch 关键词为 attention, config, kv, quant, cache, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7723 - [Bug] add flashinfer bool check for fusedmoe in Qwen moe models

- 链接：https://github.com/sgl-project/sglang/pull/7723
- 状态/时间：`merged`，created 2025-07-02, merged 2025-07-03；作者 `yilian49`。
- 代码 diff 已读范围：`2` 个文件，`+18/-0`；代码面：model wrapper, MoE/router；关键词：flash, moe, config, deepep, expert, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 flash, moe, config, deepep, expert, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7740 - [optimize] add two stream norm for qwen3

- 链接：https://github.com/sgl-project/sglang/pull/7740
- 状态/时间：`merged`，created 2025-07-03, merged 2025-07-03；作者 `yizhang2077`。
- 代码 diff 已读范围：`4` 个文件，`+54/-10`；代码面：model wrapper, MoE/router；关键词：config, cuda, quant, attention, moe, deepep。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunk: from sglang.srt.layers.rotary_embedding import get_rope; def __init__(; 符号: Qwen3Attention, __init__, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunk: VocabParallelEmbedding,; from sglang.srt.models.qwen2_moe import Qwen2MoeMLP as Qwen3MoeMLP; 符号: Qwen3MoeSparseMoeBlock, __init__, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen2.py` modified +3/-0 (3 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +3/-0 (3 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py`；patch 关键词为 config, cuda, quant, attention, moe, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7745 - [feat] Support EAGLE3 for Qwen

- 链接：https://github.com/sgl-project/sglang/pull/7745
- 状态/时间：`merged`，created 2025-07-03, merged 2025-07-05；作者 `Ximingwang-09`。
- 代码 diff 已读范围：`4` 个文件，`+81/-6`；代码面：model wrapper, MoE/router；关键词：eagle, config, cache, kv, moe, processor, spec, expert, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunk: import logging; def __init__(; 符号: __init__, get_input_embeddings, forward, set_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunk: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; def __init__(; 符号: __init__, forward, forward, start_layer
  - `python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, forward, forward
  - `python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunk: def __init__(; def forward(; 符号: __init__, get_input_embedding, forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 eagle, config, cache, kv, moe, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7912 - Qwen FP8/NVFP4 ModelOPT Quantization support

- 链接：https://github.com/sgl-project/sglang/pull/7912
- 状态/时间：`merged`，created 2025-07-09, merged 2025-09-03；作者 `jingyu-ml`。
- 代码 diff 已读范围：`2` 个文件，`+43/-4`；代码面：model wrapper, quantization；关键词：kv, cache, config, cuda, fp4, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunk: def get_min_capability(cls) -> int:; def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:; 符号: get_min_capability, get_config_filenames, common_group_size, from_config
  - `python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunk: from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`；patch 关键词为 kv, cache, config, cuda, fp4, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- 链接：https://github.com/sgl-project/sglang/pull/7966
- 状态/时间：`merged`，created 2025-07-11, merged 2025-07-19；作者 `ch-wan`。
- 代码 diff 已读范围：`39` 个文件，`+557/-872`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：expert, moe, router, topk, triton, quant, cuda, fp8, config, attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunk: +from __future__ import annotations; use_intel_amx_backend,; 符号: __init__, create_weights, apply, forward_cuda
  - `python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunk: # limitations under the License.; except ImportError:; 符号: TopKOutput, TopK, __init__, forward_native
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors; ); 符号: GPTQMarlinState, CompressedTensorsMoEMethod:, CompressedTensorsMoEMethod, __new__
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunk: import importlib; use_intel_amx_backend,; 符号: get_quant_method, apply, apply, create_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74 (87 lines); hunk: import logging; tma_align_input_scale,; 符号: __init__, __init__, determine_expert_map, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；patch 关键词为 expert, moe, router, topk, triton, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8036 - [NVIDIA] Add Flashinfer MoE blockscale fp8 backend

- 链接：https://github.com/sgl-project/sglang/pull/8036
- 状态/时间：`merged`，created 2025-07-15, merged 2025-07-27；作者 `kaixih`。
- 代码 diff 已读范围：`8` 个文件，`+179/-47`；代码面：model wrapper, MoE/router, quantization, kernel；关键词：flash, moe, quant, deepep, expert, config, fp4, router, fp8, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunk: get_bool_env_var,; from aiter.fused_moe import fused_moe; 符号: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical
  - `python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunk: RowParallelLinear,; def __init__(; 符号: __init__, __init__, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
  - `python/sglang/srt/server_args.py` modified +13/-3 (16 lines); hunk: class ServerArgs:; def __post_init__(self):; 符号: ServerArgs:, __post_init__, add_cli_args
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunk: def __init__(self, quant_config: ModelOptFp4Config):; def process_weights_after_loading(self, layer: torch.nn.Module) -> None:; 符号: __init__, create_weights, process_weights_after_loading, process_weights_after_loading
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 flash, moe, quant, deepep, expert, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8280 - DP Enhancement

- 链接：https://github.com/sgl-project/sglang/pull/8280
- 状态/时间：`merged`，created 2025-07-23, merged 2025-07-25；作者 `ch-wan`。
- 代码 diff 已读范围：`20` 个文件，`+665/-1116`；代码面：model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：spec, attention, config, cuda, eagle, cache, processor, topk, kv, triton。
- 代码 diff 细节：
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +70/-850 (920 lines); hunk: ); def test_mmlu(self):; 符号: Test0, Test00, setUpClass, test_mmlu
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunk: import triton; if TYPE_CHECKING:; 符号: ForwardBatch:, ForwardBatch:, ForwardBatch:, init_new
  - `python/sglang/srt/speculative/eagle_worker.py` modified +59/-44 (103 lines); hunk: def draft_model_runner(self):; def forward_batch_speculative_generation(; 符号: draft_model_runner, forward_batch_speculative_generation, forward_batch_speculative_generation, forward_batch_speculative_generation
  - `python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunk: import functools; _LOCAL_ATTN_DP_RANK = None; 符号: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunk: from sglang.srt.custom_op import CustomOp; def get_batch_sizes_to_capture(model_runner: ModelRunner):; 符号: get_batch_sizes_to_capture, __init__, __init__, can_run
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py`；patch 关键词为 spec, attention, config, cuda, eagle, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_worker.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8421 - [3/N] MoE Refactor: Simplify DeepEP Output

- 链接：https://github.com/sgl-project/sglang/pull/8421
- 状态/时间：`merged`，created 2025-07-27, merged 2025-07-28；作者 `ch-wan`。
- 代码 diff 已读范围：`8` 个文件，`+319/-276`；代码面：model wrapper, MoE/router；关键词：moe, deepep, expert, topk, config, router, fp8, quant, triton, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunk: +# NOTE(ch-wan): this file will be moved to sglang/srt/layers/moe/token_dispatcher/deepep.py; use_deepep = False; 符号: DeepEPNormalOutput, format, DeepEPLLOutput, format
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunk: +from __future__ import annotations; next_power_of_2,; 符号: __init__, forward, dispatch, moe_impl
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunk: def __init__(; def forward_deepep(; 符号: __init__, forward, forward_deepep, op_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunk: def forward_deepep(; def op_select_experts(self, state):; 符号: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b
  - `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0 (48 lines); hunk: +from __future__ import annotations; 符号: DispatchOutputFormat, is_standard, is_deepep_normal, is_deepep_ll
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 moe, deepep, expert, topk, config, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8448 - Support EPLB in FusedMoE

- 链接：https://github.com/sgl-project/sglang/pull/8448
- 状态/时间：`merged`，created 2025-07-28, merged 2025-07-29；作者 `ch-wan`。
- 代码 diff 已读范围：`15` 个文件，`+107/-11`；代码面：model wrapper, MoE/router, kernel；关键词：config, expert, moe, quant, deepep, flash, router, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunk: get_tensor_model_parallel_world_size,; def __init__(; 符号: __init__, __init__, weight_loader, _weight_loader_physical
  - `python/sglang/srt/eplb/expert_location.py` modified +17/-6 (23 lines); hunk: def __post_init__(self):; def init_by_mapping(; 符号: __post_init__, init_trivial, init_by_mapping, init_by_eplb
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, weight_loader, __init__
  - `python/sglang/srt/eplb/expert_distribution.py` modified +5/-0 (5 lines); hunk: def init_new(; 符号: init_new
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunk: def __init__(; def determine_num_fused_shared_experts(; 符号: __init__, determine_num_fused_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 config, expert, moe, quant, deepep, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8450 - [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE

- 链接：https://github.com/sgl-project/sglang/pull/8450
- 状态/时间：`merged`，created 2025-07-28, merged 2025-08-01；作者 `kaixih`。
- 代码 diff 已读范围：`6` 个文件，`+131/-46`；代码面：model wrapper, MoE/router, quantization, kernel；关键词：flash, moe, expert, topk, config, quant, deepep, fp8, router, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py; logger = logging; 符号: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunk: silu_and_mul_triton_kernel,; get_bool_env_var,; 符号: _get_tile_tokens_dim, EPMoE, __init__, forward
  - `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunk: def dummy_func(*args, **kwargs):; def apply(; 符号: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import (; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import (; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`；patch 关键词为 flash, moe, expert, topk, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- 链接：https://github.com/sgl-project/sglang/pull/8658
- 状态/时间：`merged`，created 2025-08-01, merged 2025-08-01；作者 `ch-wan`。
- 代码 diff 已读范围：`38` 个文件，`+342/-299`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：deepep, moe, config, expert, flash, quant, attention, cuda, fp8, topk。
- 代码 diff 细节：
  - `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-80 (160 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass, setUpClass, setUpClass
  - `python/sglang/srt/server_args.py` modified +47/-20 (67 lines); hunk: class ServerArgs:; class ServerArgs:; 符号: ServerArgs:, ServerArgs:, __post_init__, print_deprecated_warning
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunk: from __future__ import annotations; should_use_flashinfer_trtllm_moe,; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunk: +from enum import Enum; 符号: MoeA2ABackend, _missing_, is_deepep, is_standard
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunk: from transformers import PretrainedConfig; get_moe_impl_class,; 符号: __init__, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 deepep, moe, config, expert, flash, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_hybrid_dp_ep_tp_mtp.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8751 - [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead

- 链接：https://github.com/sgl-project/sglang/pull/8751
- 状态/时间：`merged`，created 2025-08-04, merged 2025-08-06；作者 `hebiao064`。
- 代码 diff 已读范围：`1` 个文件，`+26/-6`；代码面：model wrapper, MoE/router；关键词：cache, config, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights, load_weights, load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 cache, config, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8987 - Fix incorrect default get_hidden_dim logic

- 链接：https://github.com/sgl-project/sglang/pull/8987
- 状态/时间：`merged`，created 2025-08-08, merged 2025-08-09；作者 `lifuhuang`。
- 代码 diff 已读范围：`7` 个文件，`+36/-143`；代码面：model wrapper；关键词：attention, config, kv, lora, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunk: def forward_split_prefill(; 符号: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size
  - `python/sglang/srt/lora/utils.py` modified +24/-5 (29 lines); hunk: def get_hidden_dim(; 符号: get_hidden_dim, if
  - `python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunk: def forward(; 符号: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
  - `python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunk: def end_layer(self):; 符号: end_layer, get_input_embeddings, get_hidden_dim, get_module_name
  - `python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunk: def __init__(; 符号: __init__, get_input_embeddings, get_hidden_dim, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py`；patch 关键词为 attention, config, kv, lora, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/granite.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- 链接：https://github.com/sgl-project/sglang/pull/9101
- 状态/时间：`merged`，created 2025-08-12, merged 2025-08-14；作者 `Misaka9468`。
- 代码 diff 已读范围：`5` 个文件，`+68/-16`；代码面：model wrapper, MoE/router；关键词：attention, moe, config, expert, topk, deepep, lora, router。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunk: def __init__(; def get_moe_weights(self):; 符号: __init__, forward, get_moe_weights, forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunk: def __init__(; def __init__(; 符号: __init__, forward, forward, __init__
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunk: def __init__(; def __init__(; 符号: __init__, forward, forward, __init__
  - `python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunk: def __init__(; 符号: __init__, forward, forward
  - `python/sglang/srt/lora/layers.py` modified +6/-2 (8 lines); hunk: def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor; def forward(self, input_: torch.Tensor):; 符号: apply_lora, forward, forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`；patch 关键词为 attention, moe, config, expert, topk, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9147 - support Qwen3-MoE-w4afp8

- 链接：https://github.com/sgl-project/sglang/pull/9147
- 状态/时间：`open`，created 2025-08-13；作者 `zhilingjiang`。
- 代码 diff 已读范围：`636` 个文件，`+16172/-71705`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, cache, test, router, spec, attention, doc, flash, kv, cuda。
- 代码 diff 细节：
  - `sgl-router/src/routers/pd_router.rs` removed +0/-2180 (2180 lines); hunk: -// PD (Prefill-Decode) Router Implementation; 符号: PDRouter
  - `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunk: -# Copyright 2024 SGLang Team; 符号: BlockBase, __init__, get_activation, adaptive_enc_mask
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunk: -# SPDX-License-Identifier: Apache-2.0; 符号: DualChunkFlashAttentionMetadata:, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config
  - `sgl-router/tests/api_endpoints_test.rs` removed +0/-1644 (1644 lines); hunk: -mod common;; 符号: TestContext
  - `sgl-router/src/core/worker.rs` modified +16/-1387 (1403 lines); hunk: -use super::{CircuitBreaker, CircuitBreakerConfig, WorkerError, WorkerResult};; pub trait Worker: Send + Sync + fmt::Debug {; 符号: BasicWorker, DPAwareWorker
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`；patch 关键词为 config, cache, test, router, spec, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `sgl-router/src/routers/pd_router.rs`, `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9338 - Refactor TopK to ensure readability and extensibility

- 链接：https://github.com/sgl-project/sglang/pull/9338
- 状态/时间：`merged`，created 2025-08-19, merged 2025-09-15；作者 `ch-wan`。
- 代码 diff 已读范围：`14` 个文件，`+52/-47`；代码面：model wrapper, MoE/router, kernel；关键词：moe, quant, config, expert, triton, flash, fp4, fp8, topk, deepep。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunk: from dataclasses import dataclass; is_npu,; 符号: TopKConfig:, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunk: get_deepep_mode,; def __init__(; 符号: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunk: logger = logging.getLogger(__name__); 符号: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunk: def _forward_ll(dispatch_output: DeepEPLLOutput):; def get_moe_impl_class(quant_config: Optional[QuantizationConfig] = None):; 符号: _forward_ll, get_moe_impl_class, get_moe_impl_class, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunk: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: __init__, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 moe, quant, config, expert, triton, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9973 - Optimize Qwen3-moe model by using flashinfer fused allreduce

- 链接：https://github.com/sgl-project/sglang/pull/9973
- 状态/时间：`merged`，created 2025-09-03, merged 2025-09-04；作者 `yuan-luo`。
- 代码 diff 已读范围：`3` 个文件，`+52/-12`；代码面：model wrapper, MoE/router；关键词：cuda, flash, moe, attention, config, deepep, expert, fp4, processor, router。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunk: RowParallelLinear,; from sglang.srt.model_loader.weight_utils import default_weight_loader; 符号: forward, get_moe_weights, forward_normal, forward_normal
  - `python/sglang/srt/layers/communicator.py` modified +9/-3 (12 lines); hunk: ); def _gather_hidden_states_and_residual(; 符号: _gather_hidden_states_and_residual
  - `python/sglang/srt/models/qwen2_moe.py` modified +4/-1 (5 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 cuda, flash, moe, attention, config, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10574 - [Ascend]optimize Qwen3 on Ascend

- 链接：https://github.com/sgl-project/sglang/pull/10574
- 状态/时间：`merged`，created 2025-09-17, merged 2025-09-23；作者 `ping1jing2`。
- 代码 diff 已读范围：`6` 个文件，`+81/-2`；代码面：model wrapper, quantization, scheduler/runtime；关键词：attention, cache, config, cuda, flash, mla, quant。
- 代码 diff 细节：
  - `python/sglang/srt/utils.py` modified +44/-0 (44 lines); hunk: def make_layers(; 符号: make_layers, get_cmo_stream, prepare_weight_cache, wait_cmo_stream
  - `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunk: ); def forward(; 符号: Qwen3Attention, forward
  - `python/sglang/srt/layers/communicator.py` modified +8/-0 (8 lines); hunk: is_hip,; def prepare_mlp(; 符号: prepare_mlp, CommunicateContext:, is_same_group_size, _gather_hidden_states_and_residual
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunk: def add_mla_attention_backend(backend_name):; 符号: add_mla_attention_backend, RankZeroFilter
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-0 (2 lines); hunk: def process_weights_after_loading(self, layer):; def process_weights_after_loading(self, layer):; 符号: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl:, process_weights_after_loading, NPU_W8A8DynamicLinearMethod
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py`；patch 关键词为 attention, cache, config, cuda, flash, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/utils.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/communicator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- 链接：https://github.com/sgl-project/sglang/pull/10749
- 状态/时间：`merged`，created 2025-09-22, merged 2025-09-26；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+105/-34`；代码面：model wrapper, MoE/router；关键词：cache, cuda, kv, attention, moe, config, lora, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunk: from sglang.srt.managers.schedule_batch import global_server_args_dict; def forward_normal(; 符号: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunk: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def forward(; 符号: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunk: from sglang.srt.model_loader.weight_utils import default_weight_loader; def forward_prepare(; 符号: forward_prepare, forward_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`；patch 关键词为 cache, cuda, kv, attention, moe, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10911 - model: qwen3-omni (thinker-only)

- 链接：https://github.com/sgl-project/sglang/pull/10911
- 状态/时间：`merged`，created 2025-09-25, merged 2025-10-16；作者 `mickqian`。
- 代码 diff 已读范围：`16` 个文件，`+1947/-328`；代码面：model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config；关键词：vision, attention, moe, config, cache, quant, expert, processor, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunk: +# Copyright 2025 Qwen Team; 符号: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunk: +from transformers import PretrainedConfig; 符号: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, __init__
  - `python/sglang/srt/layers/rotary_embedding.py` modified +357/-2 (359 lines); hunk: def get_rope_index(; def get_rope_index(; 符号: get_rope_index, get_rope_index, get_rope_index, get_rope_index_qwen3_omni
  - `test/srt/test_vision_openai_server_common.py` modified +132/-96 (228 lines); hunk: import base64; AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"; 符号: TestOpenAIOmniServerBase, TestOpenAIMLLMServerBase, setUpClass, get_or_download_file
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunk: # ==============================================================================; class Qwen3MoeLLMModel(Qwen3MoeModel):; 符号: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 vision, attention, moe, config, cache, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10975 - Use more general heuristics to set the default value of --mem-fraction-static

- 链接：https://github.com/sgl-project/sglang/pull/10975
- 状态/时间：`merged`，created 2025-09-27, merged 2025-09-29；作者 `merrymercy`。
- 代码 diff 已读范围：`9` 个文件，`+157/-141`；代码面：model wrapper, attention/backend, tests/benchmarks；关键词：cache, test, cuda, attention, kv, lora, mla, config, eagle, moe。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunk: def _handle_missing_default_values(self):; 符号: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes
  - `python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunk: Image = Any; class GenerateReqInput:; 符号: SessionParams:, GenerateReqInput:, GenerateReqInput:, contains_mm_input
  - `.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunk: jobs:; jobs:
  - `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10 (20 lines); hunk: def find_local_hf_snapshot_dir(; def download_weights_from_hf(; 符号: find_local_hf_snapshot_dir, download_weights_from_hf
  - `test/srt/test_multi_instance_release_memory_occupation.py` modified +5/-2 (7 lines); hunk: import multiprocessing; TEST_SUITE = dict(; 符号: _run_sglang_subprocess
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml`；patch 关键词为 cache, test, cuda, attention, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`, `.github/workflows/pr-test.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12002 - Eagle3 DP attention for Qwen3 MoE

- 链接：https://github.com/sgl-project/sglang/pull/12002
- 状态/时间：`merged`，created 2025-10-23, merged 2025-10-29；作者 `qhsc`。
- 代码 diff 已读范围：`9` 个文件，`+219/-27`；代码面：model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks；关键词：eagle, spec, moe, attention, config, cuda, processor, test, cache, topk。
- 代码 diff 细节：
  - `test/srt/test_eagle_dp_attention.py` added +129/-0 (129 lines); hunk: +import unittest; 符号: TestEAGLE3EngineDPAttention, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/qwen2_moe.py` modified +30/-15 (45 lines); hunk: def forward(; def __init__(; 符号: forward, __init__, set_eagle3_layers_to_capture, forward
  - `python/sglang/srt/layers/communicator.py` modified +23/-1 (24 lines); hunk: from dataclasses import dataclass; def __init__(; 符号: __init__, prepare_attn_and_capture_last_layer_outputs, prepare_attn
  - `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunk: def forward(; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; 符号: forward, set_eagle3_layers_to_capture, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +11/-1 (12 lines); hunk: # https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py; def forward(; 符号: forward, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py`；patch 关键词为 eagle, spec, moe, attention, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_eagle_dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/communicator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12078 - [Ascend] qwen optimization

- 链接：https://github.com/sgl-project/sglang/pull/12078
- 状态/时间：`merged`，created 2025-10-24, merged 2025-11-25；作者 `Liwansi`。
- 代码 diff 已读范围：`16` 个文件，`+561/-108`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime；关键词：cache, config, cuda, kv, moe, attention, deepep, expert, mla, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunk: logger = logging.getLogger(__name__); def npu_fused_moe_without_routing_weights_bf16(; 符号: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunk: def forward_decode_graph(; 符号: forward_decode_graph
  - `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunk: +from __future__ import annotations; 符号: FuseEPDispatchOutput, format, FuseEPCombineInput, format
  - `python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunk: is_cuda,; logger = logging.getLogger(__name__); 符号: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare
  - `python/sglang/srt/models/qwen3.py` modified +40/-4 (44 lines); hunk: _is_cuda = is_cuda(); def _apply_qk_norm(; 符号: Qwen3Attention, __init__, _apply_qk_norm, forward_prepare_native
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`；patch 关键词为 cache, config, cuda, kv, moe, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13489 - Flashinfer TRTLLM-GEN-MoE + Qwen3

- 链接：https://github.com/sgl-project/sglang/pull/13489
- 状态/时间：`merged`，created 2025-11-18, merged 2025-11-18；作者 `b8zhong`。
- 代码 diff 已读范围：`2` 个文件，`+43/-1`；代码面：model wrapper, MoE/router；关键词：attention, config, moe, quant, cache, expert, flash, fp8, spec, topk。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +41/-1 (42 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 attention, config, moe, quant, cache, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- 链接：https://github.com/sgl-project/sglang/pull/13715
- 状态/时间：`merged`，created 2025-11-21, merged 2026-01-10；作者 `shifangx`。
- 代码 diff 已读范围：`8` 个文件，`+49/-3`；代码面：model wrapper, MoE/router；关键词：expert, moe, quant, config, topk, triton, attention, fp8, deepep, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunk: def get_tbo_token_distribution_threshold() -> float:; 符号: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunk: DispatchOutput,; def get_moe_weights(self):; 符号: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def get_moe_weights(self):; 符号: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; def get_moe_weights(self):; 符号: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunk: from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE; def get_moe_weights(self):; 符号: get_moe_weights, _forward_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 expert, moe, quant, config, topk, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13998 - [apply][2/2] Fused qk_norm_rope for Qwen3-MoE

- 链接：https://github.com/sgl-project/sglang/pull/13998
- 状态/时间：`merged`，created 2025-11-26, merged 2025-12-07；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+199/-22`；代码面：model wrapper, MoE/router；关键词：attention, cache, config, cuda, expert, flash, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunk: """Inference-only Qwen3MoE model compatible with HuggingFace weights."""; is_npu,; 符号: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`；patch 关键词为 attention, cache, config, cuda, expert, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14093 - Add fused FP8 KV cache write kernel for TRTLLM MHA backend

- 链接：https://github.com/sgl-project/sglang/pull/14093
- 状态/时间：`merged`，created 2025-11-28, merged 2025-12-05；作者 `harvenstar`。
- 代码 diff 已读范围：`4` 个文件，`+854/-7`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks；关键词：cache, kv, attention, fp8, cuda, quant, triton, flash, moe, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunk: +"""; 符号: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer
  - `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunk: +"""; 符号: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache
  - `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunk: The kernel supports sm100 only, with sliding window and attention sink features.; FlashInferAttnBackend,; 符号: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunk: def forward_prepare_npu(; def forward_prepare_native(; 符号: forward_prepare_npu, forward_prepare_native, forward_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`；patch 关键词为 cache, kv, attention, fp8, cuda, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `test/manual/test_trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15203 - [NPU] support GPTQ quantization on npu

- 链接：https://github.com/sgl-project/sglang/pull/15203
- 状态/时间：`merged`，created 2025-12-15, merged 2026-01-29；作者 `22dimensions`。
- 代码 diff 已读范围：`5` 个文件，`+259/-6`；代码面：model wrapper, quantization, tests/benchmarks；关键词：cache, cuda, quant, test, attention, awq, config, fp4, fp8, marlin。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunk: replace_parameter,; if _is_cuda:; 符号: __init__, __init__, __repr__, get_scaled_act_names
  - `test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunk: +import unittest; 符号: TestAscendGPTQInt8, setUpClass, test_a_gsm8k
  - `python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunk: "TPUInt8LinearMethod",
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: # NOTE: please sort the test cases alphabetically by the test file name
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py`；patch 关键词为 cache, cuda, quant, test, attention, awq。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/gptq.py`, `test/srt/ascend/test_ascend_gptq.py`, `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15223 - [bug fix][pp] fix qwen3 model load

- 链接：https://github.com/sgl-project/sglang/pull/15223
- 状态/时间：`merged`，created 2025-12-16, merged 2025-12-17；作者 `XucSh`。
- 代码 diff 已读范围：`1` 个文件，`+3/-3`；代码面：model wrapper；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15390 - [NPU]qwen3 pp bugfix

- 链接：https://github.com/sgl-project/sglang/pull/15390
- 状态/时间：`merged`，created 2025-12-18, merged 2025-12-24；作者 `Liwansi`。
- 代码 diff 已读范围：`2` 个文件，`+4/-3`；代码面：model wrapper, MoE/router；关键词：kv, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunk: def forward_prepare_native(self, positions, hidden_states):; def forward(; 符号: forward_prepare_native, forward_prepare_npu, forward_prepare_npu, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunk: def forward_prepare_npu(; 符号: forward_prepare_npu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- 链接：https://github.com/sgl-project/sglang/pull/15835
- 状态/时间：`merged`，created 2025-12-25, merged 2025-12-28；作者 `DarkSharpness`。
- 代码 diff 已读范围：`15` 个文件，`+827/-127`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks；关键词：cuda, kv, cache, test, flash, spec, triton, attention, benchmark, config。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0 (202 lines); hunk: +#include <sgl_kernel/runtime.cuh>; 符号: QKNormParams, auto, uint32_t, uint32_t
  - `python/sglang/jit_kernel/utils.py` modified +149/-1 (150 lines); hunk: from __future__ import annotations; def load_jit(; 符号: load_jit, cache_once, wrapper, is_arch_support_pdl
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0 (130 lines); hunk: +import itertools; 符号: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/jit_kernel/tests/test_qknorm.py` added +85/-0 (85 lines); hunk: +import torch; 符号: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunk: # See the License for the specific language governing permissions and; def create_fused_set_kv_buffer_arg(; 符号: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py`；patch 关键词为 cuda, kv, cache, test, flash, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/utils.py`, `python/sglang/jit_kernel/benchmark/bench_qknorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15890 - [PP] fix wrong weight logic for tie_word_embeddings model

- 链接：https://github.com/sgl-project/sglang/pull/15890
- 状态/时间：`merged`，created 2025-12-26, merged 2026-01-27；作者 `XucSh`。
- 代码 diff 已读范围：`2` 个文件，`+19/-48`；代码面：model wrapper；关键词：config, processor, vision, cache, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunk: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: __init__, load_weights, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunk: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: __init__, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`；patch 关键词为 config, processor, vision, cache, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16115 - [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head

- 链接：https://github.com/sgl-project/sglang/pull/16115
- 状态/时间：`merged`，created 2025-12-30, merged 2026-01-08；作者 `chenxu214`。
- 代码 diff 已读范围：`8` 个文件，`+52/-16`；代码面：model wrapper, MoE/router；关键词：kv, doc, cache, config, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunk: maybe_remap_kv_scale_name,; def __init__(; 符号: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunk: def forward_prepare_npu(self, positions, hidden_states, forward_batch):; def __init__(; 符号: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunk: def forward_prepare_npu(; 符号: forward_prepare_npu
  - `python/sglang/srt/layers/rotary_embedding.py` modified +2/-2 (4 lines); hunk: def forward_npu(; 符号: forward_npu
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +3/-1 (4 lines); hunk: cpu_has_amx_support,; def __post_init__(self):; 符号: __post_init__, get_masked_input_and_mask
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 kv, doc, cache, config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17535 - Update weight rename check for Qwen3 Embeddings

- 链接：https://github.com/sgl-project/sglang/pull/17535
- 状态/时间：`merged`，created 2026-01-21, merged 2026-02-03；作者 `satyamk7054`。
- 代码 diff 已读范围：`1` 个文件，`+5/-1`；代码面：model wrapper；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17784 - Upgrade transformers==5.3.0

- 链接：https://github.com/sgl-project/sglang/pull/17784
- 状态/时间：`merged`，created 2026-01-26, merged 2026-03-18；作者 `JustinTong0323`。
- 代码 diff 已读范围：`95` 个文件，`+1136/-343`；代码面：model wrapper, MoE/router, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, attention, kv, processor, vision, cache, cuda, moe, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +546/-21 (567 lines); hunk: def download_from_hf(; def get_hf_text_config(config: PretrainedConfig):; 符号: download_from_hf, get_rope_config, _patch_text_config, get_hf_text_config
  - `test/registered/vlm/test_vlm_input_format.py` modified +122/-17 (139 lines); hunk: def forward(self, x):; def setUpClass(cls):; 符号: forward, setUpClass, TestQwenVLUnderstandsImage, _init_visual
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunk: def __init__(; class Gemma3RotaryEmbedding(nn.Module):; 符号: __init__, Gemma3RotaryEmbedding, __init__, __init__
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunk: from __future__ import annotations; from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding; 符号: _get_rope_param, get_rope, get_rope, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunk: class ModelImpl(str, Enum):; def is_deepseek_nsa(config: PretrainedConfig) -> bool:; 符号: ModelImpl, is_deepseek_nsa, is_deepseek_nsa, is_deepseek_nsa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py`；patch 关键词为 config, attention, kv, processor, vision, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/utils/hf_transformers_utils.py`, `test/registered/vlm/test_vlm_input_format.py`, `python/sglang/srt/models/gemma3_causal.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18189 - [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch

- 链接：https://github.com/sgl-project/sglang/pull/18189
- 状态/时间：`merged`，created 2026-02-03, merged 2026-02-08；作者 `vincentzed`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：model wrapper, MoE/router；关键词：config, fp4, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunk: def __init__(; 符号: __init__, Qwen3MoeForCausalLM, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 config, fp4, kv, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18233 - Support Qwen3 MoE context parallel

- 链接：https://github.com/sgl-project/sglang/pull/18233
- 状态/时间：`merged`，created 2026-02-04, merged 2026-03-22；作者 `Shunkangz`。
- 代码 diff 已读范围：`19` 个文件，`+968/-73`；代码面：model wrapper, attention/backend, MoE/router, scheduler/runtime, tests/benchmarks；关键词：attention, config, cuda, flash, moe, cache, expert, kv, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/utils/cp_utils.py` added +460/-0 (460 lines); hunk: +from dataclasses import dataclass; 符号: ContextParallelMetadata:, is_prefill_context_parallel_enabled, is_prefill_cp_in_seq_split, can_cp_split
  - `python/sglang/test/attention/test_flashattn_backend.py` modified +106/-22 (128 lines); hunk: from sglang.srt.layers.radix_attention import RadixAttention; def __init__(; 符号: __init__, __init__, _verify_output, _create_forward_batch
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +73/-20 (93 lines); hunk: from sglang.srt.configs.model_config import AttentionArch; def __init__(; 符号: __init__, forward_extend, forward_extend, forward_extend
  - `test/registered/4-gpu-models/test_qwen3_30b.py` added +77/-0 (77 lines); hunk: +import unittest; 符号: TestQwen330B, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +36/-5 (41 lines); hunk: import torch; def ensure_workspace_initialized(; 符号: ensure_workspace_initialized, ensure_workspace_initialized, fake_flashinfer_allreduce_residual_rmsnorm, flashinfer_allreduce_residual_rmsnorm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`；patch 关键词为 attention, config, cuda, flash, moe, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/test/attention/test_flashattn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19059 - [jit_kernel] Add fused_qknorm_rope JIT kernel

- 链接：https://github.com/sgl-project/sglang/pull/19059
- 状态/时间：`merged`，created 2026-02-20, merged 2026-03-27；作者 `Johnsonms`。
- 代码 diff 已读范围：`5` 个文件，`+1127/-3`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks；关键词：cuda, kv, attention, config, moe, test, benchmark, cache, spec, triton。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunk: +"""; 符号: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunk: +/*; 符号: void, int, parameters, arguments
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunk: +"""; 符号: bench_fused_qknorm_rope, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0 (181 lines); hunk: +from __future__ import annotations; 符号: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, can_use_fused_qk_norm_rope, fused_qk_norm_rope
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunk: _is_cuda = is_cuda(); def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`；patch 关键词为 cuda, kv, attention, config, moe, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19532 - [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe

- 链接：https://github.com/sgl-project/sglang/pull/19532
- 状态/时间：`merged`，created 2026-02-28, merged 2026-03-03；作者 `shengzhaotian`。
- 代码 diff 已读范围：`2` 个文件，`+8/-2`；代码面：model wrapper, MoE/router；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunk: def forward_prepare(; 符号: forward_prepare
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20127 - [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next

- 链接：https://github.com/sgl-project/sglang/pull/20127
- 状态/时间：`open`，created 2026-03-08；作者 `xingsy97`。
- 代码 diff 已读范围：`3` 个文件，`+66/-25`；代码面：model wrapper, MoE/router；关键词：config, processor, quant, eagle, moe, attention, cache, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunk: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; 符号: __init__, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunk: def __init__(; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: __init__, load_weights
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunk: def __init__(; def get_embed_and_head(self):; 符号: __init__, get_embed_and_head, set_embed_and_head, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 config, processor, quant, eagle, moe, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20474 - Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3

- 链接：https://github.com/sgl-project/sglang/pull/20474
- 状态/时间：`open`，created 2026-03-12；作者 `jmunetong`。
- 代码 diff 已读范围：`6` 个文件，`+159/-7`；代码面：attention/backend, tests/benchmarks；关键词：test, triton, attention, doc, cache, cuda, kv, processor, spec。
- 代码 diff 细节：
  - `test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunk: +"""; 符号: TestQwen3, setUpClass, tearDownClass, get_request_json
  - `docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunk: ARG SG_LANG_KERNEL_BRANCH=main; RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/re
  - `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunk: def forward_npu(; 符号: forward_npu, forward_xpu, get_rope_index
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunk: device_context,; def _layer_norm_fwd_1pass_kernel(; 符号: _layer_norm_fwd_1pass_kernel, _get_sm_count
  - `.github/workflows/pr-test-xpu.yml` modified +1/-1 (2 lines); hunk: jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py`；patch 关键词为 test, triton, attention, doc, cache, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/xpu/test_qwen3.py`, `docker/xpu.Dockerfile`, `python/sglang/srt/layers/rotary_embedding/mrope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- 链接：https://github.com/sgl-project/sglang/pull/20520
- 状态/时间：`open`，created 2026-03-13；作者 `egvenediktov`。
- 代码 diff 已读范围：`12` 个文件，`+172/-10`；代码面：model wrapper, quantization, tests/benchmarks, docs/config；关键词：quant, attention, config, cuda, moe, test, cache, spec。
- 代码 diff 细节：
  - `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0 (37 lines); hunk: +import unittest; 符号: TestLlama
  - `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py` added +37/-0 (37 lines); hunk: +import os; 符号: TestQwen38BCommQuantization
  - `python/sglang/srt/distributed/device_communicators/npu_communicator.py` modified +29/-1 (30 lines); hunk: from sglang.srt.utils import is_npu; def all_reduce(self, x: torch.Tensor) -> torch.Tensor:; 符号: NpuCommunicator:, __init__, all_reduce, quant_all_reduce
  - `python/sglang/srt/server_args.py` modified +21/-0 (21 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, __post_init__, add_cli_args, from_cli_args
  - `python/sglang/srt/distributed/parallel_state.py` modified +14/-0 (14 lines); hunk: def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:; 符号: all_reduce, quant_all_reduce, fused_allreduce_rmsnorm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py`；patch 关键词为 quant, attention, config, cuda, moe, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`, `python/sglang/srt/distributed/device_communicators/npu_communicator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- 链接：https://github.com/sgl-project/sglang/pull/20931
- 状态/时间：`merged`，created 2026-03-19, merged 2026-03-20；作者 `lviy`。
- 代码 diff 已读范围：`1` 个文件，`+4/-3`；代码面：model wrapper, MoE/router；关键词：attention, config, cuda, kv, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunk: is_non_idle_and_non_empty,; def forward_prepare_native(; 符号: forward_prepare_native, apply_qk_norm_rope, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 attention, config, cuda, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21195 - Enable the qwen3 test

- 链接：https://github.com/sgl-project/sglang/pull/21195
- 状态/时间：`merged`，created 2026-03-23, merged 2026-03-24；作者 `Shunkangz`。
- 代码 diff 已读范围：`2` 个文件，`+6/-5`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：cuda, expert, fp8, moe, router, test, topk。
- 代码 diff 细节：
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +2/-5 (7 lines); hunk: popen_launch_server,
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunk: get_moe_tensor_parallel_world_size,; def forward_normal(; 符号: forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py`；patch 关键词为 cuda, expert, fp8, moe, router, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/models/qwen3_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21412 - [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints

- 链接：https://github.com/sgl-project/sglang/pull/21412
- 状态/时间：`open`，created 2026-03-25；作者 `rbqlsquf`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper；关键词：attention, config, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunk: from sglang.srt.models.utils import apply_qk_norm; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21654 - [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE

- 链接：https://github.com/sgl-project/sglang/pull/21654
- 状态/时间：`merged`，created 2026-03-30, merged 2026-04-01；作者 `Johnsonms`。
- 代码 diff 已读范围：`5` 个文件，`+208/-77`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks；关键词：kv, attention, cuda, config, moe, benchmark, cache, spec, test。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunk: namespace {; compute_freq_yarn(float base, int rotary_dim, int half_dim, float factor, float; 符号: void, void, void, void
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunk: """; ci_range=[64, 512],; 符号: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunk: @cache_once; def fused_qk_norm_rope_out(; 符号: _jit_fused_qknorm_rope_module, _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, fused_qk_norm_rope_out
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2 (4 lines); hunk: def apply_interleave(x):; def test_fused_qknorm_rope_partial_rotary(head_dim, is_neox):; 符号: apply_interleave, apply_neox, test_fused_qknorm_rope_partial_rotary
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py`；patch 关键词为 kv, attention, cuda, config, moe, benchmark。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`, `python/sglang/jit_kernel/fused_qknorm_rope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21770 - [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon

- 链接：https://github.com/sgl-project/sglang/pull/21770
- 状态/时间：`open`，created 2026-03-31；作者 `linzhonghong`。
- 代码 diff 已读范围：`2` 个文件，`+159/-0`；代码面：model wrapper, tests/benchmarks；关键词：cache, cuda, test。
- 代码 diff 细节：
  - `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunk: +import os; 符号: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat
  - `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunk: +import os; 符号: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`；patch 关键词为 cache, cuda, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- 链接：https://github.com/sgl-project/sglang/pull/22003
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-20；作者 `Shunkangz`。
- 代码 diff 已读范围：`8` 个文件，`+276/-25`；代码面：model wrapper, attention/backend, MoE/router, tests/benchmarks；关键词：moe, attention, cuda, config, expert, flash, fp4, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunk: get_dp_global_num_tokens,; class ScatterMode(Enum):; 符号: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0 (55 lines); hunk: def test_gsm8k(self):; 符号: test_gsm8k, TestQwen330BCP, setUpClass, tearDownClass
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunk: get_attn_tensor_model_parallel_rank,; def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):; 符号: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/distributed/parallel_state.py` modified +13/-7 (20 lines); hunk: def initialize_model_parallel(; def initialize_model_parallel(; 符号: initialize_model_parallel, initialize_model_parallel, destroy_model_parallel, destroy_model_parallel
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunk: def __init__(; 符号: __init__, get_input_embeddings
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`；patch 关键词为 moe, attention, cuda, config, expert, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22358 - Enable DFLASH support for additional model backends

- 链接：https://github.com/sgl-project/sglang/pull/22358
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-09；作者 `mmangkad`。
- 代码 diff 已读范围：`8` 个文件，`+152/-5`；代码面：model wrapper, MoE/router；关键词：flash, eagle, config, expert, kv, moe, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunk: def forward(; def forward(; 符号: forward, forward, get_layer, get_input_embeddings
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunk: def set_eagle3_layers_to_capture(; 符号: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunk: def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):; def forward(; 符号: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunk: def __init__(; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; 符号: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, set_dflash_layers_to_capture, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 flash, eagle, config, expert, kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22429 - [NPU]add Qwen3-32b and Qwen3-8b low latency md

- 链接：https://github.com/sgl-project/sglang/pull/22429
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-09；作者 `Liwansi`。
- 代码 diff 已读范围：`1` 个文件，`+296/-0`；代码面：docs/config；关键词：attention, benchmark, cache, config, cuda, doc, eagle, quant, spec, test。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunk: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 attention, benchmark, cache, config, cuda, doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22446 - [NPU] add qwen3-30b-a3b low latency example

- 链接：https://github.com/sgl-project/sglang/pull/22446
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-11；作者 `heziiop`。
- 代码 diff 已读范围：`1` 个文件，`+130/-0`；代码面：docs/config；关键词：attention, benchmark, cache, config, cuda, doc, eagle, quant, spec, test。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunk: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 attention, benchmark, cache, config, cuda, doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22450 - [NPU] Add Qwen3-14B low latency doc

- 链接：https://github.com/sgl-project/sglang/pull/22450
- 状态/时间：`open`，created 2026-04-09；作者 `LinyuanLi0046`。
- 代码 diff 已读范围：`1` 个文件，`+323/-0`；代码面：docs/config；关键词：attention, benchmark, cache, config, cuda, doc, eagle, quant, scheduler, spec。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunk: you encounter issues or have any questions, please [open an issue](https://githu; you encounter issues or have any questions, please [open an issue](https://git
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 attention, benchmark, cache, config, cuda, doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22529 - [Model] Support sliding window attention for Qwen3

- 链接：https://github.com/sgl-project/sglang/pull/22529
- 状态/时间：`open`，created 2026-04-10；作者 `bzantium`。
- 代码 diff 已读范围：`1` 个文件，`+29/-0`；代码面：model wrapper；关键词：attention, config, cuda, eagle, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunk: Qwen3Config = None; def __init__(; 符号: get_attention_sliding_window_size, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, config, cuda, eagle, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22674 - [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization

- 链接：https://github.com/sgl-project/sglang/pull/22674
- 状态/时间：`open`，created 2026-04-13；作者 `Dmovic`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：misc；关键词：config, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunk: def _get_quantization_config(; 符号: _get_quantization_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_loader/loader.py`；patch 关键词为 config, kv, quant。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_loader/loader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22687 - [NPU]qwen3-8b and 32b md bugfix

- 链接：https://github.com/sgl-project/sglang/pull/22687
- 状态/时间：`merged`，created 2026-04-13, merged 2026-04-13；作者 `Liwansi`。
- 代码 diff 已读范围：`1` 个文件，`+4/-8`；代码面：docs/config；关键词：cache, cuda, doc, eagle, quant, spec, topk。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunk: LOCAL_HOST2=`hostname -I\|awk -F " " '{print$2}'`; python -m sglang.launch_server --model-path $MODEL_PATH \
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 cache, cuda, doc, eagle, quant, spec。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22739 - Restore Qwen3 rope config fallback

- 链接：https://github.com/sgl-project/sglang/pull/22739
- 状态/时间：`merged`，created 2026-04-14, merged 2026-04-14；作者 `ishandhanani`。
- 代码 diff 已读范围：`1` 个文件，`+10/-2`；代码面：model wrapper；关键词：attention, config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22837 - [Bug] Qwen3 reasoning detector silently swallows tool_call when </think> is missing

- 链接：https://github.com/sgl-project/sglang/pull/22837
- 状态/时间：`open`，created 2026-04-15；作者 `gucasbrg`。
- 代码 diff 已读范围：`2` 个文件，`+43/-0`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunk: def test_streaming_qwen3_forced_reasoning_format(self):; 符号: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23372 - [NPU] Add CI tests for Speculative Decoding

- 链接：https://github.com/sgl-project/sglang/pull/23372
- 状态/时间：`open`，created 2026-04-21；作者 `EdwardXuy`。
- 代码 diff 已读范围：`7` 个文件，`+729/-14`；代码面：attention/backend, MoE/router, tests/benchmarks；关键词：eagle, spec, test, cache, attention, quant, topk, cuda, config, deepep。
- 代码 diff 细节：
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +185/-0 (185 lines); hunk: +import os; 符号: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +159/-0 (159 lines); hunk: +import os; 符号: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +156/-0 (156 lines); hunk: +import os; 符号: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +105/-0 (105 lines); hunk: +import os; 符号: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +97/-0 (97 lines); hunk: +import os; 符号: TestAscendSpeculativeMoeA2ABackend, setUpClass, test_a_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`；patch 关键词为 eagle, spec, test, cache, attention, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23397 - [alignment-sglang] PR3: Dense Deterministic Math

- 链接：https://github.com/sgl-project/sglang/pull/23397
- 状态/时间：`open`，created 2026-04-21；作者 `maocheng23`。
- 代码 diff 已读范围：`16` 个文件，`+2285/-50`；代码面：model wrapper, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：attention, cuda, moe, spec, config, flash, processor, quant, test, cache。
- 代码 diff 细节：
  - `test/registered/core/test_tp_invariant_ops.py` added +866/-0 (866 lines); hunk: +"""Tests for TP-invariant kernels (PR1).; 符号: _simulate_tp_matmul, TestTPInvariantMode, tearDown, test_mode_context_restores_previous_state
  - `test/registered/core/test_on_policy_wiring.py` added +527/-0 (527 lines); hunk: +import json; 符号: _run_server_args_script, install_openai_stubs, _mock_model_config, TestOnPolicyServerArgs
  - `test/registered/core/test_dense_deterministic_math.py` added +293/-0 (293 lines); hunk: +import json; 符号: _run_dense_math_script, install_openai_stubs, TestDenseOnPolicyHelpers, test_default_dense_math_helpers_are_inactive
  - `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunk: +from __future__ import annotations; 符号: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target
  - `python/sglang/srt/tp_invariant_ops/tp_invariant_ops.py` added +219/-0 (219 lines); hunk: +import contextlib; 符号: is_tp_invariant_mode_enabled, enable_tp_invariant_mode, disable_tp_invariant_mode, set_tp_invariant_mode
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py`；patch 关键词为 attention, cuda, moe, spec, config, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/core/test_tp_invariant_ops.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_dense_deterministic_math.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23434 - [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model

- 链接：https://github.com/sgl-project/sglang/pull/23434
- 状态/时间：`open`，created 2026-04-22；作者 `fortunecookiee`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：model wrapper；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunk: def __init__(; 符号: __init__, get_input_embeddings, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_classification.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_classification.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：79；open PR 数：13。
- 仍需跟进的 open PR：[#9147](https://github.com/sgl-project/sglang/pull/9147), [#20127](https://github.com/sgl-project/sglang/pull/20127), [#20474](https://github.com/sgl-project/sglang/pull/20474), [#20520](https://github.com/sgl-project/sglang/pull/20520), [#21412](https://github.com/sgl-project/sglang/pull/21412), [#21770](https://github.com/sgl-project/sglang/pull/21770), [#22450](https://github.com/sgl-project/sglang/pull/22450), [#22529](https://github.com/sgl-project/sglang/pull/22529), [#22674](https://github.com/sgl-project/sglang/pull/22674), [#22837](https://github.com/sgl-project/sglang/pull/22837), [#23372](https://github.com/sgl-project/sglang/pull/23372), [#23397](https://github.com/sgl-project/sglang/pull/23397), [#23434](https://github.com/sgl-project/sglang/pull/23434)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
