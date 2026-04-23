# SGLang Qwen3.5 支持与优化时间线

本文不是 PR 编号清单，而是按 PR diff/source 逐个读完后写成的模型优化历史。每个条目都保留了动机、关键实现思路、核心代码片段和验证含义。详细维护准则见 `skills/model-optimization/sglang/sglang-model-pr-diff-dossier`；Qwen3.5 的 canonical skill 侧档案见 `skills/model-optimization/sglang/sglang-qwen35-optimization/references/pr-history.md`。

结论：Qwen3.5 是一个“混合架构 + 多平台 + 多 quant + 多部署形态”的模型族。优化主线不是单一 kernel，而是 GDN 线性注意力、MoE shared expert、MTP/spec-v2、PP/EP/EPLB、VLM/EPD、NIXL PD、Mamba state、FP8/NVFP4/MXFP4/NPU/ROCm 共同演化。

## 关键代码面

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/models/qwen2_moe.py`
- `python/sglang/jit_kernel/triton/gdn_fused_proj.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/disaggregation/nixl/conn.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`

## 逐 PR 读 diff 后的历史

### #18489 初始 Qwen3.5 支持

- 动机：新增 Qwen3.5 dense/MoE/VL 模型族，PR body 明确列出 `Qwen3_5MoeForConditionalGeneration` 和 `Qwen3_5ForConditionalGeneration`，并引用 HF upstream 实现。
- 关键实现：新增 `qwen3_5.py`、`qwen3_5_mtp.py`、`qwen3_5.py` config，接入 model runner、server args、speculative worker、multimodal processor。模型侧一次性引入混合 GDN linear attention、full attention、MoE、DeepStack multimodal embedding、MTP 和权重加载映射。
- 关键片段：

```python
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...

class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...
```

- 验证含义：这是后续所有 Qwen3.5 优化的基线；任何 GDN、MTP、PP、VLM、quant PR 都要回到这些 class 名和 loader mapping 检查兼容性。

### #18538 重构 Qwen3.5 MTP

- 动机：早期 MTP predictor 复用度低，checkpoint loading 容易被命名层级拖垮。
- 关键实现：用嵌套的 `Qwen3_5ForCausalLM` 作为 MTP body，新增 `fc` 和两个 `GemmaRMSNorm`，把 input embedding 和 target hidden state 归一化后 concat 再进入模型。
- 关键片段：

```python
hidden_states = self.enorm(input_embeds)
target_hidden_states = self.hnorm(target_hidden_states)
hidden_states = torch.cat([hidden_states, target_hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

- 验证含义：后续 MTP quant prefix、spec-v2、MTP weight mapping 都依赖这次结构调整。

### #18544 NPU/ModelSlim/EPLB follow-up

- 动机：初始模型仍有 CUDA-only 假设，ModelSlim prefix 和 Qwen3.5 `.linear_attn` prefix 也不稳。
- 关键实现：NPU 跳过 CUDA JIT/Triton 相关 assert，ModelSlim 归一化 `language_model.`/visual prefix，Qwen3.5 MLP prefix 正确剥离 `.linear_attn`，并暴露 EPLB expert-location config。
- 关键片段：

```python
if not is_cpu() and not is_npu():
    ...
```

```python
return ModelConfigForExpertLocation(
    num_layers=config.num_hidden_layers,
    num_logical_experts=config.num_experts,
)
```

- 验证含义：后续 fused GDN 和 NPU quant 修复都不能只按 CUDA 路径思考。

### #18926 block-wise FP8 与 prefix 对齐

- 动机：Qwen3.5 FP8 checkpoint 的 block scale 需要按 merged column shard 加载；MTP quant prefix 也需要从 `model` 改为 `mtp`。
- 关键实现：`MergedColumnParallelLinear` 增加 `_load_merged_block_scale()`；`weight_loader_v2()` 识别 `BlockQuantScaleParameter`；Qwen3.5 MTP prefix 改为 `add_prefix("mtp", prefix)`。
- 关键片段：

```python
elif isinstance(param, BlockQuantScaleParameter):
    self._load_merged_block_scale(param, loaded_weight)
    return
```

```python
prefix=add_prefix("mtp", prefix)
```

- 验证含义：FP8/NVFP4/MTP load failure 首先查 block scale slicing 和 `mtp.` prefix。

### #18937 NVFP4 checkpoint 支持

- 动机：ModelOpt FP4/NVFP4 不能直接覆盖 Qwen3.5 所有 hybrid module，linear attention、full attention、MTP 需要 quant guard。
- 关键实现：当 `quant_config.get_name() == "modelopt_fp4"` 时，GDN/full-attention/MTP 局部禁用 quant；同时加强 expert name 的严格匹配和 RoPE scaling 报错信息。
- 关键片段：

```python
linear_attn_quant_config = (
    None if quant_config and quant_config.get_name() == "modelopt_fp4" else quant_config
)
```

```python
if quant_config and quant_config.get_name() == "modelopt_fp4":
    quant_config = None
```

- 验证含义：NVFP4 正确性依赖“有些层不量化”这个事实，不能为了统一而去掉 guard。

### #19070 dense Qwen3.5 TP>1 精度修复

- 动机：dense Qwen3.5 在 TP>1 下错误继承了 MoE 风格 all-reduce fusion，导致精度问题。
- 关键实现：dense MLP 和 MoE MLP 分开调用，只在合法路径传 `should_allreduce_fusion`，并用 `_sglang_needs_allreduce_fusion` 延后 communicator postprocess。
- 关键片段：

```python
hidden_states = self.mlp(
    hidden_states,
    should_allreduce_fusion=should_allreduce_fusion,
)
hidden_states._sglang_needs_allreduce_fusion = True
```

- 验证含义：dense 27B/4B lane 不能无条件继承 MoE 通信优化。

### #19220 PCG 修复

- 动机：PCG 路径的自定义 `gdn_with_output` wrapper 和 Qwen3.5 GDN 执行/compile fake registration 冲突。
- 关键实现：移除自定义 GDN PCG wrapper，回到常规 attention 调用；给 `sgl_kernel::fp8_blockwise_scaled_mm` 增加 fake registration；恢复 `@torch.no_grad()`。
- 关键片段：

```python
hidden_states = self.attn(
    positions=positions,
    hidden_states=hidden_states,
    forward_batch=forward_batch,
)
```

- 验证含义：compile/PCG 修复会影响 graph capture 行为，即使数学路径看起来没变。

### #19391 MTP spec-v2 与 NVFP4 测试

- 动机：Qwen3.5 MTP-v2 需要携带 multimodal input embeds；NVFP4 也缺少真实 accuracy/acceptance 测试。
- 关键实现：`eagle_worker_v2` 在 draft extend 时传入 `mm_input_embeds`；Qwen3.5 MTP 用 `is_draft_extend(include_v2=True)` 判断；新增 `nvidia/Qwen3.5-397B-A17B-NVFP4` 的 non-MTP、MTP-v1、MTP-v2 测试。
- 关键片段：

```python
if mm_input_embeds is not None:
    forward_batch.mm_input_embeds = mm_input_embeds
```

```python
and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- 验证含义：Qwen3.5 speculative 测试要带 `SGLANG_ENABLE_SPEC_V2`、chat template、`qwen3` reasoning parser 和 `avg_spec_accept_length > 3.3`。

### #19411 27B repeat bug 的 last-layer 标记

- 动机：Qwen3.5-27B 出现重复输出/层通信状态问题。
- 关键实现：decoder layer communicator 构造时传入最后一层判断。
- 关键片段：

```python
is_last_layer=(layer_id == config.num_hidden_layers - 1)
```

- 验证含义：看似很小的 communicator 元信息会直接影响输出质量。

### #19670 PP 支持

- 动机：Qwen3.5 需要支持 pipeline parallel，尤其是 stage-local layer、first/last rank embed/head。
- 关键实现：加入 `PPMissingLayer`、`start_layer`/`end_layer`、PP indices、first/last rank 的 embed/head 获取与设置，并补 PP accuracy test。
- 关键片段：

```python
def get_embed_and_head(self):
    embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
    head = self.lm_head.weight if self.pp_group.is_last_rank else None
    return embed, head
```

- 验证含义：后续所有 loader 改动都要检查 PP stage 是否跳过非本地层。

### #19767 MTP + EPLB

- 动机：MTP/NEXTN draft layer 不应像目标模型 MoE 层一样参与 EPLB expert dispatch 和 expert distribution 统计。
- 关键实现：`Qwen2MoeSparseMoeBlock` 增加 `is_nextn`，nextn 路径禁用 `ExpertLocationDispatchInfo`；MTP forward 包在 expert recorder disabled region 里。
- 关键片段：

```python
if self.is_nextn:
    self.expert_location_dispatch_info = None
```

```python
with get_global_expert_distribution_recorder().disable_this_region():
    hidden_states = self.model(...)
```

- 验证含义：MTP 与 EPLB 必须联测，否则 draft layer 会污染路由统计。

### #19889 TRTLLM/FlashInfer all-reduce fusion

- 动机：Qwen3.5 MoE 要在 TRTLLM/FlashInfer 路径减少通信开销，同时保留 Gemma RMSNorm 的 `weight + 1.0` 语义。
- 关键实现：新增 `_forward_with_allreduce_fusion`；`Qwen2MoeSparseMoeBlock.forward` 接收 `should_allreduce_fusion`；server args 将 Qwen3.5 dense/MoE 架构加入可用列表。
- 关键片段：

```python
return _forward_with_allreduce_fusion(
    hidden_states,
    residual,
    self.weight + 1.0,
    self.variance_epsilon,
)
```

- 验证含义：切换 `trtllm_mha`/FlashInfer fusion 时要同时看 TP、EP、MTP acceptance 和 dense accuracy。

### #19961 GDN `A_log` 保持 FP32

- 动机：`A_log` 是 linear attention recurrent dynamics 的状态参数，不能随 BF16/FP8 主路径降低精度。
- 关键实现：初始化时明确使用 `torch.float32`。
- 关键片段：

```python
self.A_log = nn.Parameter(
    torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
)
```

- 验证含义：GDN 精度问题要优先查状态参数 dtype。

### #20386 GDN 路径去掉 `einops.rearrange`

- 动机：`einops.rearrange` 在 decode 热路径里有额外开销。
- 关键实现：用 PyTorch 原生 flatten 代替 rearrange；PR body 报告 H100 上 720 次平均约 `12.67us -> 4.74us`。
- 关键片段：

```python
core_attn_out = core_attn_out.flatten(-2)
```

- 验证含义：Qwen3.5 优化不只有大 kernel，小 tensor layout 改动也值得记录。

### #20736 AMD shared-expert fusion

- 动机：Qwen3.5 MoE shared expert 的 intermediate size 可与 routed expert 匹配，在 AMD/AITER 上可融合进 routed expert tensor，减少单独 shared expert MLP。
- 关键实现：`Qwen2MoeSparseMoeBlock` 计算 `num_fused_shared_experts`，把 shared expert id/weight append 到 `StandardTopKOutput`；Qwen3.5 loader 将 `mlp.shared_expert.*` 映射到 `mlp.experts.{num_experts_base}.*`。
- 关键片段：

```python
shared_expert_id = self.num_experts
fused_topk_ids = torch.cat([topk_output.topk_ids, shared_ids], dim=-1)
fused_topk_weights = torch.cat([topk_output.topk_weights, shared_weights], dim=-1)
```

```python
name = name.replace(
    "mlp.shared_expert.",
    f"mlp.experts.{num_experts_base}.",
)
```

- 验证含义：这是 AMD Qwen3.5 的重要性能线，但 quant checkpoint 下很脆弱，后面 #22948 就是专门给 MXFP4 收口。

### #20864 SpecV2 去 H2D/D2H 开销

- 动机：Qwen3.5 SpecV2 verify 路径中存在 Python list/CUDA scalar 造成的 host-device 开销。
- 关键实现：Mamba track indices 用 `torch.stack(...).to(torch.int64)`；text-only spec mrope 直接在 device 上构造 zero delta。
- 关键片段：

```python
batch.mamba_track_indices = torch.stack(
    [req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx] for req in batch.reqs]
).to(torch.int64)
```

```python
if all(mm_input is None for mm_input in mm_inputs):
    mrope_delta_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
```

- 验证含义：SpecV2 性能分析不应只盯 kernel，也要看 Python/tensor 构造路径。

### #21019 GDN projection split/reshape/cat Triton fusion

- 动机：Qwen3-Next checkpoint 是 fused/interleaved `in_proj_qkvz`，而 Qwen3.5 checkpoint 是分开的 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`；Qwen3.5 需要适配 contiguous layout 的 fused kernel。
- 关键实现：新增 `gdn_fused_proj.py`；把四个 projection 合成 `in_proj_qkvz` 和 `in_proj_ba`；`_make_packed_weight_loader` 同时处理 fused checkpoint 和 split checkpoint；loader mapping 把 split 权重写入 fused 参数。
- 关键片段：

```python
self.in_proj_qkvz = self.create_qkvz_proj(...)
self.in_proj_ba = self.create_ba_proj(...)
```

```python
("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
("in_proj_qkvz.", "in_proj_z.", 3),
("in_proj_ba.", "in_proj_b.", 0),
("in_proj_ba.", "in_proj_a.", 1),
```

- 验证含义：PR body 给出 H200 输出吞吐约 +7.4%。但它也引出 #22312 的非连续 B/A correctness follow-up。

### #21070 PP layer splitting 修复

- 动机：Qwen3.5 PP 仍可能实例化/加载非本地层，造成 OOM 或 missing layer 行为。
- 关键实现：`make_layers` 接入 `pp_rank`/`pp_size`；fused expert weight loading 对 `name not in params_dict` 直接跳过。
- 关键片段：

```python
make_layers(..., pp_rank=self.pp_group.rank_in_group, pp_size=self.pp_group.world_size)
```

- 验证含义：PP 验证要同时看显存和 loader，而不是只跑一次 forward。

### #21234 AMD MXFP4 Qwen3.5-397B

- 动机：gfx950 上需要 Qwen3.5-397B MXFP4，且 fused GDN projection 需要 packed mapping。
- 关键实现：`_is_gfx95` 下定义 `qkv_proj`、`gate_up_proj`、`in_proj_qkvz`、`in_proj_ba` 映射；Qwen3.5 VL subclass 复用 mapping 并关闭 HF mapper。
- 关键片段：

```python
"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
"in_proj_ba": ["in_proj_b", "in_proj_a"],
```

- 验证含义：AMD quant mapping 要尽量放在模型本地，不要散落到全局 loader hack。

### #21347 PP tied embedding loading

- 动机：Qwen3.5 4B dense tied embedding 在 PP last rank 需要把 embedding weight 装到 `lm_head.weight`。
- 关键实现：last PP rank 遇到 `model.embed_tokens.weight` 且 `tie_word_embeddings=True` 时重定向到 `lm_head.weight`。
- 关键片段：

```python
if self.config.tie_word_embeddings and name == "model.embed_tokens.weight":
    name = "lm_head.weight"
```

- 验证含义：PP + tied embedding 是加载正确性问题，不是模型结构问题。

### #21448 MoE loading 与 Mamba cache PP sharding

- 动机：PP 下 Qwen3.5 只应该为本 stage 的 Mamba layers 建状态，也只加载本 stage 的权重。
- 关键实现：Mamba pool 只接收 `[start_layer, end_layer)` 内的 layer ids；Qwen3.5 `load_weights` 通过 `get_layer_id` 跳过非本地层。
- 关键片段：

```python
mamba_layer_ids = [
    layer_id for layer_id in cache_params.layers
    if start_layer <= layer_id < end_layer
]
```

- 验证含义：PP + Mamba 不能只看 KV cache，还要检查 Mamba state layer locality。

### #21487 GB300 nightly benchmark

- 动机：GB300/4x B200 NVL4 需要覆盖 Qwen3.5 FP8/NVFP4 以及 GLM/DeepSeek/Kimi 的 nightly performance。
- 关键实现：新增 `test_qwen35_fp8.py` 和 `test_qwen35_nvfp4.py`，TP4、MTP/spec-v2、`trtllm_mha`、FlashInfer all-reduce fusion、Qwen parser 都写入测试参数。
- 关键片段：

```python
QWEN35_FP8_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
QWEN35_NVFP4_MODEL_PATH = "nvidia/Qwen3.5-397B-A17B-NVFP4"
```

```python
env={"SGLANG_ENABLE_SPEC_V2": "1"}
```

- 验证含义：GB300 是独立部署线，cookbook 与 CI 参数要同步。

### #21669 AMD FP8 nightly performance

- 动机：AMD 需要 Qwen3.5-397B FP8 性能监控，而不只是 accuracy。
- 关键实现：MI30x/MI35x perf test 使用 `SGLANG_USE_AITER=1`、TP=8、Triton attention、multithread load。
- 关键片段：

```python
QWEN35_FP8_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
```

```python
"--attention-backend", "triton",
"--model-loader-extra-config", '{"enable_multithread_load": true}',
```

- 验证含义：AMD perf test 是优化 guardrail，不是可选 smoke。

### #21692 NPU Qwen3.5 quant fix

- 动机：#21019 后 fused `in_proj_qkvz`/`in_proj_ba` 让 NPU/ModelSlim quant 找不到原始 mapping。
- 关键实现：NPU 也使用 Qwen3.5 packed mapping；ModelSlim `get_linear_scheme()` 参考 MoE scheme lookup；skip layer 同时检查局部和全局 packed mapping。
- 关键片段：

```python
if _is_gfx95 or _is_npu:
    packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
```

- 验证含义：GDN fusion 后的 quant bug 多半是名字映射问题。

### #21849 Qwen3.5 encoder disaggregation

- 动机：Qwen3.5 multimodal runtime 已支持，但 EPD allowlist 没包含 Qwen3.5 架构，导致启动阶段失败。
- 关键实现：allowlist 增加 `Qwen3_5ForConditionalGeneration`、`Qwen3_5MoeForConditionalGeneration`；Qwen VL video timestamp 处理加入 `qwen3_5`/`qwen3_5_moe`；新增 EPD image/video regression。
- 关键片段：

```python
"Qwen3_5ForConditionalGeneration",
"Qwen3_5MoeForConditionalGeneration",
```

- 验证含义：Qwen3.5-VL 支持包括 encoder/language split serving。

### #22145 NIXL heterogeneous TP KV transfer

- 动机：NIXL + prefill TP != decode TP + PP=1 会因为 notification key 使用 `pp_rank` 而永远等不到完成；GQA head 分布也因 per-rank KV head 数丢信息。
- 关键实现：用 `total_kv_head_num` 计算 head distribution；增加 GQA replication/unique head；notification 改用 `engine_rank`。
- 关键片段：

```python
total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
```

```python
notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.engine_rank}"
```

- 验证含义：这是 Qwen3.5 NIXL hetero TP 的 Step 1，不修会 decode hang。

### #22240 NIXL Mamba state slice transfer

- 动机：Mooncake 已支持 Mamba state slice，NIXL 仍直接 raise，阻塞 Qwen3.5 PD hetero TP。
- 关键实现：注册 `dst_state_item_lens`/`dst_state_dim_per_tensor`；实现 `_send_mamba_state_slice()` 按 TP-sharded dim 切 conv/temporal state。
- 关键片段：

```python
dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)
```

```python
bytes_to_send = num_dims_to_send * src_bytes_per_dim
```

- 验证含义：PD 测试要覆盖 KV cache 和 Mamba state 两类传输。

### #22312 GDN 非连续 B/A tensor 修复

- 动机：#21019 后 Qwen3.5-27B fallback BA path 的 `mixed_ba.split()` 返回非连续 view，Triton kernel 却假设 contiguous，导致 accuracy 从 49/50 掉到 3/50。
- 关键实现：`fused_gdn_gating` 显式传 `stride_a`/`stride_b`；`fused_sigmoid_gating_delta_rule_update` 使用 token-axis `stride_a`；新增 CUDA stride 回归测试。
- 关键片段：

```python
blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
```

```python
stride_a = a.stride()[-2]
p_a += stride_a
```

- 验证含义：任何 split/view 结果进入 Triton kernel，都必须把 stride 当作 correctness 输入。

### #22358 DFLASH 支持

- 动机：DFLASH 需要 Qwen3.5 等后端捕获 aux hidden state。
- 关键实现：Qwen3.5 decoder layer 使用 `prepare_attn_and_capture_last_layer_outputs`；模型记录 `layers_to_capture`；需要时返回 `(hidden_states, aux_hidden_states)`。
- 关键片段：

```python
prepare_attn_and_capture_last_layer_outputs(
    hidden_states,
    residual,
    forward_batch,
    captured_last_layer_outputs=captured_last_layer_outputs,
)
```

```python
def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
    ...
```

- 验证含义：DFLASH 会改变 forward 返回结构，logits/serving 路径必须正确 unwrap。

### #22431 Qwen3.5 processor-output video 修复

- 动机：`processor_output` 格式下 `preprocess_video()` 返回单值，Qwen3.5 处理逻辑期望 `(videos, metadata)` 两值。
- 关键实现：非 `VideoDecoderWrapper` 输入返回 `(vr, None)`。
- 关键片段：

```python
if not is_video_obj:
    return vr, None
```

- 验证含义：Qwen3.5 VLM 文档要区分 raw video 和 processor-output，但内部接口要统一。

### #22493 MambaPool retraction CPU offload

- 动机：request retraction 只保存 attention KV，丢失 Qwen3.5 Mamba conv/temporal state，恢复后生成会坏。
- 关键实现：`MambaPool` 增加 `get_cpu_copy/load_cpu_copy`；`HybridLinearKVPool` 同时 offload KV 和 Mamba；`Req` 传 `mamba_pool_idx`；scheduler 日志增加 `#mamba_num_gained`。
- 关键片段：

```python
self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
    token_indices, mamba_indices=self.mamba_pool_idx
)
```

```python
return kv_cpu, mamba_cpu
```

- 验证含义：内存压力/retraction 测试必须验证 KV 和 Mamba state round-trip。

### #22908 AMD radix cache 与 spec decoding 冲突

- 动机：Qwen3.5 MoE spec decoding + `no_buffer` + radix cache 会 hard error；CUDA 的 `extra_buffer` 建议在 ROCm 上不可用。
- 关键实现：最终 review 版本只在 `is_hip()` 下自动禁用 radix cache；CUDA/其它平台仍抛错并提示使用 `extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1`。
- 关键片段：

```python
if is_hip():
    self.disable_radix_cache = True
else:
    raise ValueError(...)
```

- 验证含义：AMD 命令可以依赖自动禁用 radix cache；CUDA spec-v2 仍应显式用 `extra_buffer`。

### #22913 拆分 B200 Qwen3.5 FP4 测试

- 动机：一个测试文件里连续启动多个 234GB NVFP4 Qwen3.5 server，慢 B200 节点容易超过 30 分钟 timeout。
- 关键实现：拆出 `test_qwen35_fp4_triton.py` 和 `test_qwen35_fp4_mtp_v2.py`；删除 v1 MTP；`stage-c-test-4-gpu-b200` 分区从 5 增到 6。
- 关键片段：

```yaml
part: [0, 1, 2, 3, 4, 5]
```

```python
envs.SGLANG_ENABLE_SPEC_V2.set(True)
```

- 验证含义：CI 分区本身也是模型优化保障，否则大模型 regressions 会被 timeout 淹没。

### #22948 MXFP4 shared-expert fusion guard

- 动机：#20736 开启 shared expert fusion 后，MXFP4 checkpoint 中 shared expert 仍是 BF16/FP32，被错误融合进量化 MoE tensor。
- 关键实现：`can_fuse_shared_expert(config, quant_config)` 检查 `exclude_layers`，如果命中 shared expert 且不是 `shared_expert_gate`/MTP，则禁用 fusion。
- 关键片段：

```python
if any(
    "shared_expert" in layer
    and "shared_expert_gate" not in layer
    and not layer.startswith("mtp.")
    for layer in exclude_layers
):
    return False
```

- 验证含义：shared expert fusion 必须同时看形状和 quant exclusion。

### #23034 Qwen3.6 文档中的 Qwen3.5 运行时规则

- 动机：Qwen3.6 docs 更新同时承载了 Qwen3.5-derived 的 MTP/Mamba 命令规则。
- 关键实现：MTP 开启时禁用 Mamba V1，强制使用 V2/`extra_buffer`。
- 关键片段：

```jsx
const mtpEnabled = values.speculative === 'enabled';
if (mtpEnabled) {
  return [
    { id: 'v1', label: 'V1', default: false, disabled: true },
    { id: 'v2', label: 'V2', default: true },
  ];
}
```

- 验证含义：Qwen3.5/Qwen3.6 cookbook snippets 共享 MTP/Mamba 运行时假设。

### #23467 FP8 ignored-layer dot-boundary 修复

- 动机：FP8 ignored-layer 的 substring 匹配会让 Qwen3.5 `in_proj_a` 误匹配 `in_proj_ba`，也会让 Qwen3.6 `mlp.gate` 误匹配 `mlp.gate_up_proj`。
- 关键实现：新增 `_module_path_match()` 做 exact/prefix/dot-boundary 匹配，并补 fused shard fallback。
- 关键片段：

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

- 验证含义：这是 Qwen3.5 fused GDN projection quant loading 的直接保护。

### #23474 hybrid linear-attention CPU offload（open radar）

- 状态：写档时仍为 open PR，因此不计入 merged history，但已读 diff。
- 动机：hybrid linear-attention 模型有 tied/view parameter，CPU offload 独立 materialize tensor 后会破坏 alias/view 关系。
- 关键实现：`OffloaderV1` 用 `state_dict(keep_vars=True)` 记录 view alias；device 侧用 `src_to_dev` 共享 tensor；用 `as_strided` 重建 view。
- 关键片段：

```python
view_aliases[name] = (src_name, tensor.size(), tensor.stride(), tensor.storage_offset())
```

```python
dev_tensor = src_to_dev[src_name].as_strided(size, stride, storage_offset)
```

- 验证含义：Qwen3.5/Qwen3.6 这种 fused/view-heavy 模型做 CPU offload 时必须检查 alias。

## 文档与公开资料

- sgl-cookbook `#164/#168/#169/#177/#179/#180/#207/#214/#230/#237` 依次覆盖初始 Qwen3.5、FP8/NVFP4、B200、H200、AMD、更多 variants、B200 all-reduce fusion、H200 MTP、FP4/NVFP4 generator、FP8 KV caution。
- SGLang 官方 Qwen3.5 文档覆盖 hybrid GDN/full-attention、shared experts、DeepStack Vision/Conv3d、AMD `--attention-backend triton`、`SGLANG_USE_AITER=1`、`--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder`。
- AMD day-0 文章从 ROCm 侧确认 GDN/Triton、shared-expert MoE/hipBLASLt/AITER、MIOpen/PyTorch multimodal kernel 路线。

## 后续维护规则

- 不要再新增“只写一句话”的 PR 条目。
- 只有打开过 diff/source 的 PR 才能进入本文。
- 每个 PR 至少写出 motivation、关键实现、核心代码片段和验证含义。
- merged history 和 open radar 分开写。
- Qwen3.5 回归矩阵必须覆盖 dense/MoE、text/VLM、BF16/FP8/NVFP4/MXFP4、CUDA/ROCm/NPU、TP/PP/EP、MTP spec-v1/v2、PD/NIXL、retraction。
