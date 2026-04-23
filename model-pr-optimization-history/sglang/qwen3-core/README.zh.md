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
3. 新增文档时继续使用 `skills/model-optimization/sglang/sglang-model-pr-diff-dossier` 的标准：逐 PR 读 diff，写 motivation、实现思路、关键代码片段和验证风险。
