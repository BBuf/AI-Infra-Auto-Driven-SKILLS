# SGLang Qwen3.5 支持与优化时间线

本文不是 PR 编号清单，而是按 PR diff/source 逐个读完后写成的模型优化历史。每个条目都保留了动机、关键实现思路、核心代码片段和验证含义。详细维护准则见 `skills/model-optimization/model-pr-diff-dossier`；Qwen3.5 的 canonical skill 侧档案见 `skills/model-optimization/sglang/sglang-qwen35-optimization/references/pr-history.md`。

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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-10 | [#18538](https://github.com/sgl-project/sglang/pull/18538) | merged | [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation | model wrapper | `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-10 | [#18544](https://github.com/sgl-project/sglang/pull/18544) | merged | [Ascend]Support qwen3.5 | model wrapper, attention/backend, quantization | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-02-17 | [#18926](https://github.com/sgl-project/sglang/pull/18926) | merged | feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation | model wrapper, quantization | `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-17 | [#18937](https://github.com/sgl-project/sglang/pull/18937) | merged | [Qwen3.5] Enable nvfp4 checkpoint | model wrapper | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-20 | [#19070](https://github.com/sgl-project/sglang/pull/19070) | merged | fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1 | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-24 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | model wrapper, quantization | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-26 | [#19391](https://github.com/sgl-project/sglang/pull/19391) | merged | [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4 | model wrapper, scheduler/runtime, tests/benchmarks | `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py` |
| 2026-02-26 | [#19411](https://github.com/sgl-project/sglang/pull/19411) | merged | [Qwen3.5] Qwen3.5-27B inference repeat bug fix | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-02 | [#19670](https://github.com/sgl-project/sglang/pull/19670) | merged | [Qwen3.5] Support Qwen3.5 Pipeline Parallelism | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` |
| 2026-03-03 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-03-04 | [#19889](https://github.com/sgl-project/sglang/pull/19889) | merged | Use TRTLLM allreduce fusion for Qwen 3.5 | model wrapper, MoE/router | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-05 | [#19961](https://github.com/sgl-project/sglang/pull/19961) | merged | fix: change qwen 3.5 linear attention a_log to fp32 | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-11 | [#20386](https://github.com/sgl-project/sglang/pull/20386) | merged | perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe… | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-17 | [#20736](https://github.com/sgl-project/sglang/pull/20736) | merged | [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8 | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-18 | [#20864](https://github.com/sgl-project/sglang/pull/20864) | merged | [Perf]Remove H2D for Qwen3.5 SpecV2 | scheduler/runtime | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-03-20 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | model wrapper, kernel | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-21 | [#21070](https://github.com/sgl-project/sglang/pull/21070) | merged | [Qwen3.5] Fix broken pipeline parallelism layer splitting | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` |
| 2026-03-23 | [#21234](https://github.com/sgl-project/sglang/pull/21234) | merged | [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-24 | [#21347](https://github.com/sgl-project/sglang/pull/21347) | merged | [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model | model wrapper | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-26 | [#21448](https://github.com/sgl-project/sglang/pull/21448) | merged | [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-26 | [#21487](https://github.com/sgl-project/sglang/pull/21487) | merged | feat(ci): add GB300 nightly benchmark test suites | quantization, scheduler/runtime, tests/benchmarks | `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py` |
| 2026-03-30 | [#21669](https://github.com/sgl-project/sglang/pull/21669) | merged | [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x | quantization, tests/benchmarks | `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` |
| 2026-03-30 | [#21692](https://github.com/sgl-project/sglang/pull/21692) | merged | [Bugfix] [NPU] Qwen3.5 with quantization fix | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py` |
| 2026-04-01 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | multimodal/processor, tests/benchmarks | `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-05 | [#22145](https://github.com/sgl-project/sglang/pull/22145) | merged | [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) | misc | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-07 | [#22240](https://github.com/sgl-project/sglang/pull/22240) | merged | [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5) | misc | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-08 | [#22312](https://github.com/sgl-project/sglang/pull/22312) | merged | Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B | attention/backend, tests/benchmarks | `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` |
| 2026-04-08 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-09 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | multimodal/processor | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-10 | [#22493](https://github.com/sgl-project/sglang/pull/22493) | merged | Add MambaPool kvcache offloading during retraction | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py` |
| 2026-04-15 | [#22908](https://github.com/sgl-project/sglang/pull/22908) | merged | [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict. | misc | `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#22913](https://github.com/sgl-project/sglang/pull/22913) | merged | test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6 | model wrapper, quantization, kernel, tests/benchmarks | `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` |
| 2026-04-16 | [#22948](https://github.com/sgl-project/sglang/pull/22948) | merged | [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py` |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config | `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | quantization | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |

### 逐 PR 代码 diff 阅读记录

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- 链接：https://github.com/sgl-project/sglang/pull/18489
- 状态/时间：`merged`，created 2026-02-09, merged 2026-02-09；作者 `zju-stu-lizheng`。
- 代码 diff 已读范围：`17` 个文件，`+1923/-9`；代码面：model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, config, processor, spec, attention, vision, cache, cuda, expert, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunk: +# Copyright 2025 Qwen Team; 符号: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunk: +from transformers import PretrainedConfig; 符号: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunk: Lfm2Config,; def qwen3_next_config(self):; 符号: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1 (17 lines); hunk: import numpy as np; from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem; 符号: preprocess_video, QwenVLImageProcessor, process_mm_data_async
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`；patch 关键词为 moe, config, processor, spec, attention, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18538 - [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation

- 链接：https://github.com/sgl-project/sglang/pull/18538
- 状态/时间：`merged`，created 2026-02-10, merged 2026-02-12；作者 `zju-stu-lizheng`。
- 代码 diff 已读范围：`2` 个文件，`+62/-118`；代码面：model wrapper；关键词：config, moe, quant, attention, expert, processor, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunk: from sglang.srt.layers.layernorm import GemmaRMSNorm; def __init__(; 符号: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 config, moe, quant, attention, expert, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18544 - [Ascend]Support qwen3.5

- 链接：https://github.com/sgl-project/sglang/pull/18544
- 状态/时间：`merged`，created 2026-02-10, merged 2026-02-12；作者 `chenxu214`。
- 代码 diff 已读范围：`3` 个文件，`+23/-4`；代码面：model wrapper, attention/backend, quantization；关键词：attention, quant, cache, config, cuda, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunk: # Distributed; def __init__(; 符号: __init__, load_fused_expert_weights, get_model_config_for_expert_location
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +9/-0 (9 lines); hunk: def is_layer_skipped(; 符号: is_layer_skipped
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-2 (4 lines); hunk: from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu; def __init__(self, model_runner: ModelRunner):; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 attention, quant, cache, config, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18926 - feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation

- 链接：https://github.com/sgl-project/sglang/pull/18926
- 状态/时间：`merged`，created 2026-02-17, merged 2026-02-18；作者 `zju-stu-lizheng`。
- 代码 diff 已读范围：`4` 个文件，`+57/-12`；代码面：model wrapper, quantization；关键词：config, quant, kv, attention, awq, expert, fp8, test, vision。
- 代码 diff 细节：
  - `python/sglang/srt/layers/linear.py` modified +48/-0 (48 lines); hunk: def _load_fused_module_from_checkpoint(; def weight_loader_v2(; 符号: _load_fused_module_from_checkpoint, _load_merged_block_scale, weight_loader_v2, weight_loader_v2
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-2 (7 lines); hunk: def from_config(cls, config: Dict[str, Any]) -> Fp8Config:; 符号: from_config
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunk: def __init__(; def load_fused_expert_weights(; 符号: __init__, load_fused_expert_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-4 (7 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；patch 关键词为 config, quant, kv, attention, awq, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18937 - [Qwen3.5] Enable nvfp4 checkpoint

- 链接：https://github.com/sgl-project/sglang/pull/18937
- 状态/时间：`merged`，created 2026-02-17, merged 2026-02-19；作者 `hlu1`。
- 代码 diff 已读范围：`3` 个文件，`+26/-8`；代码面：model wrapper；关键词：config, fp4, quant, expert, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +3/-1 (4 lines); hunk: def get_rope(; 符号: get_rope
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；patch 关键词为 config, fp4, quant, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19070 - fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

- 链接：https://github.com/sgl-project/sglang/pull/19070
- 状态/时间：`merged`，created 2026-02-20, merged 2026-02-25；作者 `zju-stu-lizheng`。
- 代码 diff 已读范围：`1` 个文件，`+32/-6`；代码面：model wrapper；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunk: def forward(; def forward(; 符号: forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- 链接：https://github.com/sgl-project/sglang/pull/19220
- 状态/时间：`merged`，created 2026-02-24, merged 2026-02-26；作者 `zminglei`。
- 代码 diff 已读范围：`4` 个文件，`+9/-46`；代码面：model wrapper, quantization；关键词：config, attention, cuda, eagle, expert, fp8, kv, lora, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunk: import torch; make_layers,; 符号: set_eagle3_layers_to_capture, gdn_with_output
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunk: import torch.nn as nn; from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock; 符号: forward, _forward, _forward
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunk: def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=No; 符号: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunk: def get_input_embeddings(self):; 符号: get_input_embeddings, should_apply_lora, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 config, attention, cuda, eagle, expert, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19391 - [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/19391
- 状态/时间：`merged`，created 2026-02-26, merged 2026-03-04；作者 `hlu1`。
- 代码 diff 已读范围：`8` 个文件，`+252/-16`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：scheduler, spec, cache, eagle, test, attention, config, cuda, fp4, kv。
- 代码 diff 细节：
  - `test/registered/4-gpu-models/test_qwen35_models.py` added +240/-0 (240 lines); hunk: +import unittest; 符号: TestQwen35FP4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +5/-4 (9 lines); hunk: def _handle_mamba_radix_cache(; 符号: _handle_mamba_radix_cache, _handle_sampling_backend
  - `python/sglang/srt/disaggregation/decode.py` modified +0/-5 (5 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +0/-5 (5 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +4/-0 (4 lines); hunk: def _draft_extend_for_prefill(; def _draft_extend_for_prefill(; 符号: _draft_extend_for_prefill, _draft_extend_for_prefill, forward_batch_generation
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py`；patch 关键词为 scheduler, spec, cache, eagle, test, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19411 - [Qwen3.5] Qwen3.5-27B inference repeat bug fix

- 链接：https://github.com/sgl-project/sglang/pull/19411
- 状态/时间：`merged`，created 2026-02-26, merged 2026-02-26；作者 `AlfredYyong`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：model wrapper；关键词：attention, config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunk: def __init__(; def __init__(; 符号: __init__, forward, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19670 - [Qwen3.5] Support Qwen3.5 Pipeline Parallelism

- 链接：https://github.com/sgl-project/sglang/pull/19670
- 状态/时间：`merged`，created 2026-03-02, merged 2026-03-07；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+114/-13`；代码面：model wrapper, tests/benchmarks；关键词：attention, cache, config, cuda, expert, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunk: ); from sglang.srt.layers.radix_attention import RadixAttention; 符号: __init__, get_layer, get_layer, get_input_embeddings
  - `test/registered/distributed/test_pp_single_node.py` modified +54/-0 (54 lines); hunk: def test_pp_consistency(self):; 符号: test_pp_consistency, TestQwen35PPAccuracy, setUpClass, run_gsm8k_test
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`；patch 关键词为 attention, cache, config, cuda, expert, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- 链接：https://github.com/sgl-project/sglang/pull/19767
- 状态/时间：`merged`，created 2026-03-03, merged 2026-03-09；作者 `luoyuyan`。
- 代码 diff 已读范围：`5` 个文件，`+79/-16`；代码面：model wrapper, MoE/router；关键词：config, quant, expert, moe, cuda, processor, attention, deepep, router, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunk: from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunk: from transformers import PretrainedConfig; def __init__(; 符号: __init__, __init__, get_model_config_for_expert_location, get_embed_and_head
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunk: from transformers import PretrainedConfig; def __init__(; 符号: __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, get_moe_weights, _forward_deepep
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-0 (6 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；patch 关键词为 config, quant, expert, moe, cuda, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19889 - Use TRTLLM allreduce fusion for Qwen 3.5

- 链接：https://github.com/sgl-project/sglang/pull/19889
- 状态/时间：`merged`，created 2026-03-04, merged 2026-03-18；作者 `b8zhong`。
- 代码 diff 已读范围：`4` 个文件，`+88/-52`；代码面：model wrapper, MoE/router；关键词：moe, flash, attention, fp4, processor, spec, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunk: import torch_npu; def forward_with_allreduce_fusion(; 符号: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunk: def forward(; def forward(; 符号: forward, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunk: RowParallelLinear,; def forward(; 符号: forward, forward
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 moe, flash, attention, fp4, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19961 - fix: change qwen 3.5 linear attention a_log to fp32

- 链接：https://github.com/sgl-project/sglang/pull/19961
- 状态/时间：`merged`，created 2026-03-05, merged 2026-03-18；作者 `shiyu7`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20386 - perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

- 链接：https://github.com/sgl-project/sglang/pull/20386
- 状态/时间：`merged`，created 2026-03-11, merged 2026-03-12；作者 `vedantjh2`。
- 代码 diff 已读范围：`1` 个文件，`+1/-2`；代码面：model wrapper；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunk: import torch; def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20736 - [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8

- 链接：https://github.com/sgl-project/sglang/pull/20736
- 状态/时间：`merged`，created 2026-03-17, merged 2026-04-15；作者 `zhentaocc`。
- 代码 diff 已读范围：`2` 个文件，`+218/-8`；代码面：model wrapper, MoE/router；关键词：config, cuda, expert, moe, deepep, fp8, quant, router, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +108/-5 (113 lines); hunk: ); from sglang.srt.utils import (; 符号: can_fuse_shared_expert, Qwen2MoeMLP, __init__, __init__
  - `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunk: LazyValue,; _is_npu = is_npu(); 符号: __init__, __init__, __init__, _get_num_fused_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 config, cuda, expert, moe, deepep, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20864 - [Perf]Remove H2D for Qwen3.5 SpecV2

- 链接：https://github.com/sgl-project/sglang/pull/20864
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-31；作者 `Chen-0210`。
- 代码 diff 已读范围：`2` 个文件，`+17/-13`；代码面：scheduler/runtime；关键词：spec, cache, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunk: def _compute_spec_mrope_positions(; 符号: _compute_spec_mrope_positions
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunk: def prepare_for_v2_verify(; 符号: prepare_for_v2_verify
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；patch 关键词为 spec, cache, eagle。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- 链接：https://github.com/sgl-project/sglang/pull/21019
- 状态/时间：`merged`，created 2026-03-20, merged 2026-03-23；作者 `yuan-luo`。
- 代码 diff 已读范围：`3` 个文件，`+597/-202`；代码面：model wrapper, kernel；关键词：kv, triton, attention, config, cuda, cache, fp8, moe, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunk: import torch; RowParallelLinear,; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunk: +from __future__ import annotations; 符号: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunk: from typing import Any, Iterable, Optional, Set, Tuple; logger = logging.getLogger(__name__); 符号: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 kv, triton, attention, config, cuda, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21070 - [Qwen3.5] Fix broken pipeline parallelism layer splitting

- 链接：https://github.com/sgl-project/sglang/pull/21070
- 状态/时间：`merged`，created 2026-03-21, merged 2026-03-21；作者 `alisonshao`。
- 代码 diff 已读范围：`2` 个文件，`+15/-20`；代码面：model wrapper, tests/benchmarks；关键词：config, expert, moe, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunk: ); def get_layer(idx: int, prefix: str):; 符号: get_layer, load_fused_expert_weights, load_fused_expert_weights
  - `test/registered/distributed/test_pp_single_node.py` modified +7/-5 (12 lines); hunk: def setUpClass(cls):; def run_gsm8k_test(self, pp_size):; 符号: setUpClass, run_gsm8k_test, run_gsm8k_test, run_gsm8k_test
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`；patch 关键词为 config, expert, moe, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21234 - [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model

- 链接：https://github.com/sgl-project/sglang/pull/21234
- 状态/时间：`merged`，created 2026-03-23, merged 2026-03-30；作者 `hubertlu-tw`。
- 代码 diff 已读范围：`1` 个文件，`+18/-0`；代码面：model wrapper；关键词：config, cuda, expert, kv, moe, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunk: cpu_has_amx_support,; _is_cuda = is_cuda(); 符号: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 config, cuda, expert, kv, moe, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21347 - [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model

- 链接：https://github.com/sgl-project/sglang/pull/21347
- 状态/时间：`merged`，created 2026-03-24, merged 2026-04-01；作者 `edwingao28`。
- 代码 diff 已读范围：`1` 个文件，`+22/-0`；代码面：model wrapper；关键词：config, expert。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_fused_expert_weights(; 符号: load_weights, load_fused_expert_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`；patch 关键词为 config, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21448 - [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode

- 链接：https://github.com/sgl-project/sglang/pull/21448
- 状态/时间：`merged`，created 2026-03-26, merged 2026-03-30；作者 `sufeng-buaa`。
- 代码 diff 已读范围：`6` 个文件，`+78/-8`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：cache, spec, attention, config, kv, mla, cuda, expert, lora, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunk: from sglang.srt.layers.radix_attention import RadixAttention; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights, load_fused_expert_weights, load_weights, load_fused_expert_weights
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +17/-0 (17 lines); hunk: def _init_pools(self: ModelRunner):; def _init_pools(self: ModelRunner):; 符号: _init_pools, _init_pools, _init_pools
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +11/-5 (16 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +13/-0 (13 lines); hunk: def separate_deepstack_embeds(self, embedding):; 符号: separate_deepstack_embeds, start_layer, end_layer, pad_input_ids
  - `python/sglang/srt/disaggregation/decode.py` modified +4/-2 (6 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 cache, spec, attention, config, kv, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21487 - feat(ci): add GB300 nightly benchmark test suites

- 链接：https://github.com/sgl-project/sglang/pull/21487
- 状态/时间：`merged`，created 2026-03-26, merged 2026-03-29；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`11` 个文件，`+874/-4`；代码面：quantization, scheduler/runtime, tests/benchmarks；关键词：test, attention, cuda, eagle, spec, topk, flash, cache, fp4, kv。
- 代码 diff 细节：
  - `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunk: def _run_simple_eval(; def run_accuracy_test(; 符号: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunk: +import unittest; 符号: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunk: +import unittest; 符号: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunk: +import unittest; 符号: TestQwen35Nvfp4, test_qwen35_nvfp4
  - `test/registered/gb300/test_qwen35_fp8.py` added +75/-0 (75 lines); hunk: +import unittest; 符号: TestQwen35Fp8, test_qwen35_fp8
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`；patch 关键词为 test, attention, cuda, eagle, spec, topk。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21669 - [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x

- 链接：https://github.com/sgl-project/sglang/pull/21669
- 状态/时间：`merged`，created 2026-03-30, merged 2026-04-07；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`6` 个文件，`+408/-8`；代码面：quantization, tests/benchmarks；关键词：test, attention, config, fp8, triton, cache, benchmark, moe。
- 代码 diff 细节：
  - `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunk: +"""Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; 符号: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8
  - `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunk: +"""MI35x Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; 符号: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf
  - `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunk: import os; def setUpClass(cls):; 符号: setUpClass, setUpClass, tearDownClass, test_lm_eval
  - `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunk: import os; def setUpClass(cls):; 符号: setUpClass, test_lm_eval, test_lm_eval
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +26/-2 (28 lines); hunk: jobs:; jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`；patch 关键词为 test, attention, config, fp8, triton, cache。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21692 - [Bugfix] [NPU] Qwen3.5 with quantization fix

- 链接：https://github.com/sgl-project/sglang/pull/21692
- 状态/时间：`merged`，created 2026-03-30, merged 2026-04-08；作者 `OrangeRedeng`。
- 代码 diff 已读范围：`3` 个文件，`+29/-42`；代码面：model wrapper, quantization；关键词：config, moe, quant, vision, expert, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +25/-39 (64 lines); hunk: FusedMoEMethodBase,; def get_quant_method(; 符号: get_quant_method, get_quant_method, _get_scheme_from_parts, get_linear_scheme
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunk: def forward(; def load_fused_expert_weights(; 符号: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration
  - `python/sglang/srt/model_loader/loader.py` modified +1/-0 (1 lines); hunk: def _get_quantization_config(; 符号: _get_quantization_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py`；patch 关键词为 config, moe, quant, vision, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- 链接：https://github.com/sgl-project/sglang/pull/21849
- 状态/时间：`merged`，created 2026-04-01, merged 2026-04-06；作者 `Ratish1`。
- 代码 diff 已读范围：`4` 个文件，`+190/-3`；代码面：multimodal/processor, tests/benchmarks；关键词：moe, processor, config, cuda, scheduler, test。
- 代码 diff 细节：
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunk: # Omni model for local testing; override via env var EPD_OMNI_MODEL; def test_mmmu(self):; 符号: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunk: async def _process_mm_items(self, mm_items, modality):; 符号: _process_mm_items
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunk: def get_mm_data(self, prompt, embeddings, **kwargs):; 符号: get_mm_data
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunk: def _handle_encoder_disaggregation(self):; 符号: _handle_encoder_disaggregation
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；patch 关键词为 moe, processor, config, cuda, scheduler, test。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22145 - [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)

- 链接：https://github.com/sgl-project/sglang/pull/22145
- 状态/时间：`merged`，created 2026-04-05, merged 2026-04-07；作者 `YAMY1234`。
- 代码 diff 已读范围：`1` 个文件，`+20/-8`；代码面：misc；关键词：cache, config, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunk: def send_kvcache_slice(; def add_transfer_request(; 符号: send_kvcache_slice, add_transfer_request, add_transfer_request
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/disaggregation/nixl/conn.py`；patch 关键词为 cache, config, kv, mla。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/disaggregation/nixl/conn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22240 - [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)

- 链接：https://github.com/sgl-project/sglang/pull/22240
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-07；作者 `YAMY1234`。
- 代码 diff 已读范围：`1` 个文件，`+143/-2`；代码面：misc；关键词：kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunk: class KVArgsRegisterInfo:; def from_zmq(cls, msg: List[bytes]):; 符号: KVArgsRegisterInfo:, from_zmq, from_zmq, from_zmq
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/disaggregation/nixl/conn.py`；patch 关键词为 kv, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/disaggregation/nixl/conn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22312 - Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

- 链接：https://github.com/sgl-project/sglang/pull/22312
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-10；作者 `cs-cat`。
- 代码 diff 已读范围：`3` 个文件，`+272/-8`；代码面：attention/backend, tests/benchmarks；关键词：attention, triton, cache, config, cuda, test。
- 代码 diff 细节：
  - `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunk: +"""; 符号: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunk: def fused_sigmoid_gating_delta_rule_update_kernel(; def fused_sigmoid_gating_delta_rule_update_kernel(; 符号: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunk: def fused_gdn_gating_kernel(; def fused_gdn_gating_kernel(; 符号: fused_gdn_gating_kernel, fused_gdn_gating_kernel, fused_gdn_gating, fused_gdn_gating
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`；patch 关键词为 attention, triton, cache, config, cuda, test。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- 链接：https://github.com/sgl-project/sglang/pull/22431
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-18；作者 `lkhl`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：multimodal/processor；关键词：processor。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunk: async def preprocess_video(; 符号: preprocess_video
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/qwen_vl.py`；patch 关键词为 processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/qwen_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22493 - Add MambaPool kvcache offloading during retraction

- 链接：https://github.com/sgl-project/sglang/pull/22493
- 状态/时间：`merged`，created 2026-04-10, merged 2026-04-22；作者 `hlu1`。
- 代码 diff 已读范围：`5` 个文件，`+193/-16`；代码面：scheduler/runtime, tests/benchmarks；关键词：cache, kv, test, attention, cuda, mla, scheduler, triton。
- 代码 diff 细节：
  - `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunk: def make_dummy_req():; 符号: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunk: def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; def set_kv_buffer(; 符号: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos
  - `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunk: def free(self, free_index: torch.Tensor):; def clear(self):; 符号: free, get_cpu_copy, get_cpu_copy, load_cpu_copy
  - `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunk: def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:; def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch; 符号: update_running_batch, update_running_batch
  - `python/sglang/srt/managers/schedule_batch.py` modified +8/-2 (10 lines); hunk: def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):; 符号: offload_kv_cache, load_kv_cache, log_time_stats
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`；patch 关键词为 cache, kv, test, attention, cuda, mla。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22908 - [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.

- 链接：https://github.com/sgl-project/sglang/pull/22908
- 状态/时间：`merged`，created 2026-04-15, merged 2026-04-21；作者 `ChangLiu0709`。
- 代码 diff 已读范围：`1` 个文件，`+14/-4`；代码面：misc；关键词：cache, scheduler, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunk: def _handle_mamba_radix_cache(; 符号: _handle_mamba_radix_cache, _handle_sampling_backend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 cache, scheduler, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22913 - test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6

- 链接：https://github.com/sgl-project/sglang/pull/22913
- 状态/时间：`merged`，created 2026-04-15, merged 2026-04-17；作者 `alisonshao`。
- 代码 diff 已读范围：`4` 个文件，`+184/-247`；代码面：model wrapper, quantization, kernel, tests/benchmarks；关键词：cuda, test, attention, config, fp4, quant, scheduler, eagle, flash, spec。
- 代码 diff 细节：
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245 (245 lines); hunk: -import unittest; 符号: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunk: +import unittest; 符号: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunk: +import unittest; 符号: TestQwen35FP4, test_gsm8k
  - `.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunk: jobs:; jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`；patch 关键词为 cuda, test, attention, config, fp4, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22948 - [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled

- 链接：https://github.com/sgl-project/sglang/pull/22948
- 状态/时间：`merged`，created 2026-04-16, merged 2026-04-16；作者 `mqhc2020`。
- 代码 diff 已读范围：`1` 个文件，`+17/-1`；代码面：model wrapper, MoE/router；关键词：config, deepep, expert, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunk: def can_fuse_shared_expert(; def can_fuse_shared_expert(; 符号: can_fuse_shared_expert, can_fuse_shared_expert, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 config, deepep, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- 链接：https://github.com/sgl-project/sglang/pull/23034
- 状态/时间：`merged`，created 2026-04-17, merged 2026-04-17；作者 `zijiexia`。
- 代码 diff 已读范围：`73` 个文件，`+2214/-215`；代码面：model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config；关键词：doc, spec, attention, config, cuda, cache, moe, quant, eagle, expert。
- 代码 diff 细节：
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunk: +---
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunk: +---
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunk: +---; 符号: per_token_group_quant_8bit, add
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45 (286 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s; Please consult the documentation below and [server_args.py](https://github.com
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` added +219/-0 (219 lines); hunk: +export const Qwen36Deployment = () => {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`；patch 关键词为 doc, spec, attention, config, cuda, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- 链接：https://github.com/sgl-project/sglang/pull/23467
- 状态/时间：`merged`，created 2026-04-22, merged 2026-04-22；作者 `mickqian`。
- 代码 diff 已读范围：`1` 个文件，`+31/-4`；代码面：quantization；关键词：config, fp8, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunk: def __getattr__(self, name):; def is_layer_skipped(; 符号: __getattr__, _module_path_match, names, is_layer_skipped
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/utils.py`；patch 关键词为 config, fp8, kv, moe, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- 链接：https://github.com/sgl-project/sglang/pull/23474
- 状态/时间：`open`，created 2026-04-22；作者 `kawaruko`。
- 代码 diff 已读范围：`2` 个文件，`+284/-8`；代码面：tests/benchmarks；关键词：attention, cache, cuda, spec, test。
- 代码 diff 细节：
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunk: +"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).; 符号: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunk: import logging; def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:; 符号: maybe_offload_to_cpu, maybe_offload_to_cpu, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`；patch 关键词为 attention, cache, cuda, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：37；open PR 数：1。
- 仍需跟进的 open PR：[#23474](https://github.com/sgl-project/sglang/pull/23474)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
