# SGLang Qwen3-Next / Qwen3-Coder-Next 优化历史

本文件基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）、sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）、官方 Qwen3-Next 部署文档、公开优化资料，以及下列每个 PR 的代码 diff 整理。更完整的逐 PR dossier 见 `skills/model-optimization/sglang/sglang-qwen3-next-optimization/references/pr-history.md`。

Qwen3-Next 不能被当作普通 Qwen3 MoE。它的优化面包含 hybrid Gated Delta Network、Mamba/SSM state pool、RadixLinearAttention、MTP/NEXTN/EAGLE、FP8/NVFP4/ModelOpt 加载、CPU offload、FlashInfer/CuTe/Gluon GDN kernel、AMD/NPU/Blackwell 后端，以及 mixed chunk 与 `extra_buffer` 的状态一致性。

## 主要代码面

- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_next_mtp.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/linear/`
- `python/sglang/srt/layers/radix_linear_attention.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/speculative/`
- `python/sglang/srt/utils/offloader.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`

## 已合入 / current-main PR

### #10233：初始 Qwen3-Next 支持

- Motivation：SGLang 需要支持 `Qwen3NextForCausalLM` 和 MTP draft 架构，不只是普通 attention/MoE，还要管理 GDN/Mamba 状态。
- 实现：新增 `Qwen3NextConfig`、`Qwen3NextForCausalLM`、`Qwen3NextForCausalLMMTP`，引入 `HybridLayerType.linear_attention/mamba2`，新增 `MambaPool`、`HybridReqToTokenPool`、`HybridLinearKVPool` 和 hybrid linear-attention backend。
- 关键代码：

```python
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"
```

```python
if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
    self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
```

- 验证：PR 记录 GSM8K 约 `0.945`，MTP throughput 从约 `180` 提升到约 `304` tok/s。

### #10322：Norm 类型修复

- Motivation：Transformers 侧 norm 配置变化后，旧的条件分支会让 Qwen3-Next 使用错误 norm。
- 实现：统一改为 `GemmaRMSNorm`，覆盖 input/post/final norm 和 MTP pre-fc norm。
- 关键代码：

```python
self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### #10379：Ascend NPU 初始支持

- Motivation：Qwen3-Next hybrid attention 在 NPU 上需要不同的 causal conv、GDN、KV pool、page-size 和 attention backend。
- 实现：`is_npu()` 下导入 `sgl_kernel_npu` chunk/fused gating/causal conv；`HybridLinearKVPool` 使用 Ascend token pool；hybrid backend 选择 `AscendAttnBackend`；Ascend hybrid page size 强制为 `128`。
- 关键代码：

```python
full_attn_backend = AscendAttnBackend(self) if _is_npu else FlashAttentionBackend(self)
```

### #10392：MTP + DP 修复

- Motivation：Qwen3-Next speculative decoding 与 DP/cuda graph/idle batch 组合时会出错。
- 实现：draft config 设置 `num_nextn_predict_layers=1`，修复 `set_dp_buffer_len`，idle `bs=0` 特判，Mamba state size 统计覆盖所有 state tensor。
- 关键代码：

```python
self.hf_config.num_nextn_predict_layers = 1
```

```python
def get_mamba_size(self):
    return sum(get_tensor_size_bytes(t) for t in self.mamba_cache)
```

### #10466：FP8 与 L2Norm 修复

- Motivation：GDN L2Norm 精度问题会影响 Qwen3-Next；同时 FP8 path 需要把量化配置传入 GDN。
- 实现：`quant_config` 进入 `Qwen3GatedDeltaNet` 和 hybrid layer；修复 FLA recurrent/fused sigmoid gating 的 L2Norm 行为。
- 关键代码：

```python
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream)
```

### #10622：FP8 DeepEP 路径

- Motivation：支持 `Qwen/Qwen-Next-80B-A3B-Instruct-FP8` 的 TP/DP/DeepEP。
- 实现：MoE block 暴露 `get_moe_weights`；空 token 时 TopK 返回 empty output；Qwen3-Next 通过 `LazyValue` 构造 routed expert weights。
- 关键代码：

```python
def get_moe_weights(self):
    return [x.data for name, x in self.experts.named_parameters() if name not in ["correction_bias"]]
```

- 验证：TP4DP2 accuracy 约 `0.942`，TP8DP8 约 `0.940`。

### #10912：PD disaggregation 支持 hybrid state

- Motivation：Qwen3-Next 的 prefill/decode 分离不能只传 KV cache，还要传 Mamba/GDN extra state。
- 实现：KV transfer 接口增加 `extra_pool_indices`；memory pool 暴露 extra pool buffer；prefill/decode 传递 Mamba rid/req mapping；Mooncake/NIXL/fake connector 支持 extra state。
- 关键代码：

```python
def get_extra_pool_buf_infos(self):
    return self.mamba_pool.get_contiguous_buf_infos()
```

- 验证：PR 记录 Qwen-Next GSM8K 约 `0.952`。

### #11487：KTransformers CPU/GPU hybrid MoE

- Motivation：通过 KTransformers/AMX 支持 MoE CPU/GPU 混合推理和 Qwen3-Next GPTQ4/INT4 示例。
- 实现：加入 compressed-tensors WNA16 AMX MoE、AMX wrapper、`--cpuinfer`/`--num-gpu-experts` 等参数，MoE output 走 AMX/Marlin combine。
- 关键代码：

```python
output = self.amx_wrapper.forward(x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream)
```

### #11969、#16164：NPU bugfix / W8A8

- Motivation：Ascend NPU 上 decode kernel、fused TopK、DP-attention padding、W8A8 loader name/prefix 都会影响 Qwen3-Next。
- 实现：NPU/CUDA causal conv 分支；DP-attn 才补 padding；`prefix` 穿透到 `Qwen3GatedDeltaNet`。
- 关键代码：

```python
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream, prefix)
```

- 验证：A3 NPU BF16/W8A8 TP4EP4，W8A8 throughput 约 `1405` tok/s。

### #12508：fused GDN gating

- Motivation：GDN decode/verify 中 sigmoid/gating/unsqueeze 拆开执行开销大。
- 实现：新增 Triton `fused_gdn_gating` kernel，backend 直接调用 fused 版本。
- 关键代码：

```python
g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
```

- 验证：Verify 从约 `3.5us` 到 `1.4us`，H100x4 send_one throughput 约 `317 -> 319.7` tok/s。

### #12525：CPU kernel 与 AMX 路径

- Motivation：CPU 上 Qwen3-Next 缺少 fused RMSNorm/GDN/conv1d/qkvzba 等关键 kernel，TP odd-size padding 也需要处理。
- 实现：新增 `Qwen3NextRMSNormGated` CPU op、CPU causal conv、AMX conv state layout、CPU fused GDN 分支，并禁用 CPU dual-stream。
- 关键代码：

```python
class Qwen3NextRMSNormGated(CustomOp):
    def forward_cpu(self, hidden_states, gate=None):
        return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(...)
```

### #13081、#17613、#16863、#19220：PCG 演进

- Motivation：Qwen3-Next 的 GDN 参数多，最初 PCG 只能把大块 GDN 放到 fake op 外；后续需要让 projection/norm/out projection 进入图，只把 attention core 留在 eager。
- 实现：
  - `#13081` 加 `gdn_with_output` split op。
  - `#16863` 抽象 `register_split_op`。
  - `#17613` 让 `RadixLinearAttention` 接入 `unified_linear_attention_with_output`，`model_runner` 识别 `layer.linear_attn.attn`。
  - `#19220` 移除遗留 `gdn_with_output`，同时补 FP8 fake impl。
- 关键代码：

```python
if hasattr(layer.linear_attn, "attn"):
    self.attention_layers.append(layer.linear_attn.attn)
```

```python
@torch.library.register_fake("sgl_kernel::fp8_blockwise_scaled_mm")
def _fake_fp8_blockwise_scaled_mm(...):
    return mat_a.new_empty((M, N), dtype=out_dtype)
```

- 验证：`#17613` 记录 throughput 约 `2592 -> 2963` tok/s。

### #13708、#14855：小型正确性/清理修复

- `#13708` motivation：避免强制 `lm_head.float()`。实现是删除该转换，保持 BF16。
- `#14855` motivation：清理 GDN init 中混乱的 `torch.log` 逻辑和无用代码。关键保留 `self.conv_dim = self.key_dim * 2 + self.value_dim`。

### #14607：EAGLE3

- Motivation：支持 `lukeysong/qwen3-next-draft` EAGLE3。
- 实现：`set_eagle3_layers_to_capture`、捕获 aux hidden states、model 返回 `(hidden_states, aux_hidden_states)`，logits processor 接收 aux。
- 关键代码：

```python
def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
    self.capture_aux_hidden_states = True
```

- 验证：SpecForge GSM8K accept length 约 `3.13`，GSM8K 约 `0.955`。

### #15631、#17981、#17983、#23273：Blackwell/Hopper GDN kernel 方向

- `#15631` motivation：加 CuTe DSL GDN decode；通过 `SGLANG_USE_CUTEDSL_GDN_DECODE=1` 控制。验证：H200 `4.6-5.2%`、B200 `2.6-3.4%` E2E speedup。
- `#17981` motivation：Blackwell decode/MTP underutilization；改为 transposed SSM state `[B,H,V,K]`，新增 CuTe DSL transposed decode/MTP kernel。验证：decode BF16 `1.62-1.69x`，MTP BF16 `1.29-1.57x`。
- `#17983` motivation：GDN prefill/cumsum 在 Blackwell 上优化；实现 Gluon chunk/cumsum/wy_fast kernel。验证：cumsum `7us -> 3us`，chunk output `133us -> 69us`。
- `#23273` motivation：SM100+ FlashInfer GDN target_verify 之前禁用 MTP；FlashInfer 已有 pool API kernel。实现：导入 BF16 state `gated_delta_rule_mtp`，去掉 `use_state_pool` NotImplemented guard，SM100+ speculative 也默认 FlashInfer decode。
- 关键代码：

```python
USE_CUTEDSL_GDN_DECODE = os.environ.get("SGLANG_USE_CUTEDSL_GDN_DECODE", "0") == "1"
```

```python
from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
)
```

### #17373、#17660：RadixLinearAttention 抽象

- Motivation：把 Qwen3-Next linear attention 后端调用收敛成类似 `RadixAttention` 的 layer 对象，而不是从模型层传一堆散参数。
- 实现：新增 `radix_linear_attention.py`；layer 保存 `A_log`、`dt_bias`、conv weights、head dims；backend 从 `layer` 取 `q_dim/k_dim/v_dim`。
- 关键代码：

```python
class RadixLinearAttention(nn.Module):
    def forward(self, forward_batch, mixed_qkv, a, b, **kwargs):
        return forward_batch.attn_backend.forward(layer=self, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)
```

- 验证：`#17373` 记录 GSM8K 约 `0.960`。

### #17570：embedding 使用 attention TP group

- Motivation：DP-attention 模型 embedding 需要使用 attention TP group。
- 实现：

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    use_attn_tp_group=is_dp_attention_enabled(),
)
```

### #17627、#18224、#21313、#21496、#21662、#21698：FP8/NVFP4/W8A8 loader 历史

- `#17627` motivation：支持 `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`。实现：ModelOpt FP4 下禁用未量化的 `qkv_proj` quant_config，并跳过值为 `1.0` 的缺失 `_scale` tensor。
- `#18224` motivation：Qwen3-Coder-Next NVFP4 共享 Qwen3-Next 架构，需要 packed module mapping 和 KV scale name remap。
- `#21313` motivation：W8A8 fused projection loader 出错，尝试写 `_weight_loader`，但后来被 `#21496` 回滚。
- `#21662` motivation：正式修复 `weight_loader` property no setter；新增 `_override_weight_loader`。
- `#21698` motivation：NPU W8A8 精度问题；loader 需要覆盖 `weight_scale_inv/weight_scale/input_scale/weight_offset`，并使用 NPU fused qkvzba split kernel。
- 关键代码：

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
```

```python
def _override_weight_loader(module, new_loader):
    param = module.weight
    if hasattr(param, "_weight_loader"):
        param._weight_loader = new_loader
```

```python
for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale", "weight_offset"):
    param = getattr(module, attr_name, None)
```

### #18355、#17016：AMD 路径

- Motivation：Qwen3-Coder-Next 在 AMD MI300X/MI355 上需要正确 `v_head_dim`、MTP mask、dual-stream guard。
- 实现：AITER hybrid linear attention 从 `token_to_kv_pool.get_v_head_dim()` 取 `v_head_dim`；只有 CUDA 创建 `alt_stream`，AMD 上 guarded branch 不再调用 `wait_stream`。
- 关键代码：

```python
alt_stream = torch.cuda.Stream() if _is_cuda else None
```

### #18489、#21019：Qwen3.5 共享 hybrid 路径

- Motivation：Qwen3.5 引入时复用/扩展 Qwen3-Next GDN 设计；后续把 Qwen3-Next interleaved fused projection kernel 抽到共享模块，并为 Qwen3.5 增加 contiguous 版本。
- 关键代码：

```python
if isinstance(config, Qwen3NextConfig | Qwen3_5Config | Qwen3_5MoeConfig):
    return config
```

```python
def fused_qkvzba_split_reshape_cat_contiguous(...):
    ...
```

### #18917、#19321、#19434：GDN projection / norm-gate fusion

- Motivation：prefill 中 split/reshape/cat 和 GDN projection/norm/gate 都是热点。
- 实现：
  - `#18917` 把 `fused_qkvzba_split_reshape_cat` 从 CUDA graph decode 扩到 prefill。
  - `#19321` 用 `MergedColumnParallelLinear` 融合 `qkvz_proj` 和 `ba_proj`，并增加 fused/split checkpoint loader mapping。
  - `#19434` 加 `FusedRMSNormGated`，PCG 开启时回退旧 op。
- 关键代码：

```python
("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
("in_proj_qkvz.", "in_proj_z.", 3),
("in_proj_ba.", "in_proj_b.", 0),
("in_proj_ba.", "in_proj_a.", 1),
```

```python
self.norm = FusedRMSNormGated(...) if not enable_piecewise_cuda_graph else RMSNormGated(...)
```

- 验证：`#19321` throughput 约 `15314.80 -> 15733.74` tok/s；`#19434` 约 `15314.80 -> 15959.30` tok/s。

### #19767、#19812：MTP + EPLB

- Motivation：MTP draft forward 不应该污染 EPLB expert-distribution recorder，也不能用错误 layer id 生成 expert location dispatch info。
- 实现：`Qwen2MoeSparseMoeBlock` 增加 `is_nextn`；NextN 时不创建 `ExpertLocationDispatchInfo`；MTP forward 包在 `disable_this_region()`。
- 关键代码：

```python
expert_location_dispatch_info=(
    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
    if not self.is_nextn else None
)
```

```python
with get_global_expert_distribution_recorder().disable_this_region():
    hidden_states = self.model(...)
```

- 备注：`#19812` 当前 open diff 主要是 Qwen3.5 MoE hooks；Qwen3-Next current-main 行为来自 `#19767`。

### #22073、#22358、#22458、#22664：周边能力与推理稳定性

- `#22073` 是 Qwen3-ASR，主要是 shared Qwen-family import/runtime surface，不能当作 Qwen3-Next GDN 优化。
- `#22358` motivation：DFLASH 要捕获 Qwen3-Next aux hidden states；实现 `set_dflash_layers_to_capture` 并要求显式 layer ids。
- `#22458` motivation：Qwen3-Next MTP TP>1 非贪心采样 rank 间 accepted token 不一致导致 NCCL AllGather hang；实现 rank0 broadcast `predict/accept_index/accept_length`。
- `#22664` motivation：Qwen3-Coder-Next H100 未自动开启 FlashInfer all-reduce fusion；实现把 `"Qwen3NextForCausalLM"` 加入 whitelist。验证：req/s `5.49 -> 9.41`，mean TTFT `456 -> 167ms`。
- 关键代码：

```python
tp_group.broadcast(predict, src=0)
tp_group.broadcast(accept_index, src=0)
tp_group.broadcast(accept_length, src=0)
```

```python
"Qwen3NextForCausalLM",
```

## Open PR 雷达

### #10657：早期 EAGLE3，已被 #14607 覆盖

- Motivation：为 Qwen3-Next 捕获 EAGLE3 aux hidden states，并让 draft worker 保留 full attention backend。
- 实现：`layers_to_capture`，默认 `[2, num_layers // 2, num_layers - 3]`，模型返回 `(hidden_states, aux_hidden_states)`。
- 状态：open 但已被 merged `#14607` 实质覆盖。

### #12892：spec decode 避免 SSM/conv state copy

- Motivation：target verify 后的 Mamba state update 有 CPU/GPU sync 和 scatter 开销。
- 实现：`MambaPool.SpeculativeState` 增加 `last_steps`，kernel 内根据 accepted step 读取中间 state。
- 关键代码：

```python
mamba_caches.last_steps[state_indices_tensor] = accepted_indices
```

- 验证：update path 约 `339us -> 50us`，端到端约 `5-9%` speedup。

### #13964：GDN decode kernel autotune

- Motivation：提升 `fused_sigmoid_gating_delta_rule_update_kernel`。
- 实现：Triton autotune、预计算 `neg_exp_A`、`BV` 放宽到 64。
- 验证：H200 kernel avg `143747ns -> 109069ns`。

### #14502：PCG 优化

- Motivation：把 projection/out/gating 放进 PCG，只把 conv+GDN core 留在 eager。
- 实现：新增 `causal_conv1d_gdn_with_output` split op。
- 验证：H200x2 1024 TTFT `99.17ms -> 67.83ms -> 48.21ms`。

### #16488：TBO 支持

- Motivation：Qwen3-Next PCG 关闭时做 two-batch overlap。
- 实现：新增 Qwen3 hybrid layer operation strategy，decode `tbo_delta_stages=2`。
- 验证：H800 FP8 GSM8K 约 `0.936`，profile 显示 compute/comm overlap。

### #20397：NPU Qwen3-Next MTP

- Motivation：Ascend NPU 上支持 Qwen3-Next MTP。
- 实现：FIA 支持 `qk_head_dim == 256`，NPU conv state layout 加 draft step，graph metadata 增加 MTP state indices，target verify 后用 Triton helper rollback SSM/conv state。
- 关键代码：

```python
if is_npu():
    move_intermediate_cache_dynamic_h_block_v1(...)
    conv_state_rollback(...)
    return
```

### #21684：allocator clone 修复 memory leak/alias

- Motivation：allocator 返回 `free_pages` 的 view，后续修改 `free_pages` 可能污染已返回 index。
- 实现：

```python
return select_index.clone()
```

### #22876、#23075：mixed chunk + `extra_buffer` 准确率问题

- Motivation：`--enable-mixed-chunk` 与 `--mamba-scheduler-strategy extra_buffer` 并发时 GSM8K 从 `0.938` 降到 `0.876`。
- `#22876`：先加 `ValueError` guard。
- `#23075`：定位根因是 mixed mode 下 `query_start_loc` 和 `mamba_cache_indices` 混入 decode request；修复为 prefill-only prefix。
- 关键代码：

```python
if forward_batch.forward_mode.is_mixed():
    query_start_loc_for_track = query_start_loc[: num_prefills + 1]
    mamba_cache_indices_for_track = mamba_cache_indices[:num_prefills]
```

### #23474：hybrid linear-attn CPU offload 修复

- Motivation：`--cpu-offload-gb > 0` 在 Qwen3-Next/Qwen3.5/Kimi-Linear 上先因为 tied parameter 触发 `functional_call got multiple values`，绕过后又因为 cached `conv1d.weight.view` 指向旧 GPU storage 导致 garbage output。
- 实现：`state_dict(keep_vars=True)` 按 parameter id 缓存 device tensor；扫描 plain tensor attribute 的 storage alias；forward 时用 `as_strided` 临时重绑 alias，finally 恢复。
- 关键代码：

```python
for k, v in module.state_dict(keep_vars=True).items():
    dev = src_to_dev.get(id(v))
```

```python
sub.__dict__[attr_name] = dev_tensor.as_strided(size, stride, offset)
```

- 验证：新增 tied-param/view-alias 单测；Qwen3.5-2B `--cpu-offload-gb 2` 800 prompts 无 garbage。

## 文档 / cookbook 证据

- 官方 Qwen3-Next 文档保留了 `--max-mamba-cache-size`、`--mamba-ssm-dtype`、`--mamba-full-memory-ratio`、`--mamba-scheduler-strategy extra_buffer`、`--page-size 64`、NEXTN/EAGLE、`--tool-call-parser qwen`、`--reasoning-parser qwen3` 等关键参数。
- sgl-cookbook `#100`、`#123` 记录 AMD MI300X/MI325X/MI355X 相关部署环境。
- sgl-cookbook `#143` 是 Qwen3-Coder-Next cookbook，与 shared Qwen3-Next 架构强相关。

## 后续优化建议

1. MTP state copy 与 PCG/TBO 应继续分成三个 lane，不要混在一个 benchmark 中归因。
2. Blackwell GDN kernel 要分别验证 prefill、decode、MTP verify 和 fallback。
3. CPU offload 后续必须带 tied param 和 cached tensor view 的单测。
4. NPU W8A8 后续优先补 fused projection loader 的 scale/offset 覆盖测试。
5. mixed chunk + `extra_buffer` 应作为 Qwen3-Next/Qwen3.5 hybrid 的固定准确率回归项。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen3 Next`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-09 | [#10233](https://github.com/sgl-project/sglang/pull/10233) | merged | Qwen3-Next support | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/configs/qwen3_next.py` |
| 2025-09-11 | [#10322](https://github.com/sgl-project/sglang/pull/10322) | merged | [bugfix] fix norm type error in qwen3_next model | model wrapper | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-12 | [#10379](https://github.com/sgl-project/sglang/pull/10379) | merged | Support Qwen3-Next on Ascend NPU | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-09-12 | [#10392](https://github.com/sgl-project/sglang/pull/10392) | merged | [Fix] Support qwen3-next MTP+DP | model wrapper, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/models/qwen3_next_mtp.py`, `python/sglang/srt/layers/logits_processor.py` |
| 2025-09-15 | [#10466](https://github.com/sgl-project/sglang/pull/10466) | merged | feat: update support for qwen3next model | model wrapper, attention/backend | `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-09-18 | [#10622](https://github.com/sgl-project/sglang/pull/10622) | merged | support qwen3-next-fp8 deepep | model wrapper, MoE/router | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2025-09-19 | [#10657](https://github.com/sgl-project/sglang/pull/10657) | open | feat: add eagle3 support for qwen3-next model | model wrapper, scheduler/runtime | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py` |
| 2025-10-12 | [#11487](https://github.com/sgl-project/sglang/pull/11487) | merged | init support for KTransformers Heterogeneous Computing | model wrapper, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-22 | [#11969](https://github.com/sgl-project/sglang/pull/11969) | merged | [NPU] bugfix for Qwen3-Next and performance update | model wrapper, attention/backend, MoE/router | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py` |
| 2025-11-02 | [#12508](https://github.com/sgl-project/sglang/pull/12508) | merged | [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup | model wrapper, attention/backend | `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-11-03 | [#12525](https://github.com/sgl-project/sglang/pull/12525) | merged | [CPU] Optimize Qwen3-next model on CPU | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config | `sgl-kernel/python/sgl_kernel/mamba.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-11-08 | [#12892](https://github.com/sgl-project/sglang/pull/12892) | open | [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup | attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-11-11 | [#13081](https://github.com/sgl-project/sglang/pull/13081) | merged | Support piecewise cuda graph for Qwen3-next | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/models/qwen3_next.py`, `test/srt/models/test_qwen3_next_models.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-21 | [#13708](https://github.com/sgl-project/sglang/pull/13708) | merged | [Fix] Qwen3Next lmhead dtype | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-26 | [#13964](https://github.com/sgl-project/sglang/pull/13964) | open | [Performance]Qwen3 Next kernel performance optimize | attention/backend | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-12-05 | [#14502](https://github.com/sgl-project/sglang/pull/14502) | open | [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next | model wrapper, attention/backend, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-12-08 | [#14607](https://github.com/sgl-project/sglang/pull/14607) | merged | support qwen3-next eagle3 | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2025-12-11 | [#14855](https://github.com/sgl-project/sglang/pull/14855) | merged | Clean up GDN Init | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2025-12-22 | [#15631](https://github.com/sgl-project/sglang/pull/15631) | merged | [jit-kernel] Add CuTe DSL GDN Decode Kernel | attention/backend, kernel, tests/benchmarks | `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-12-30 | [#16164](https://github.com/sgl-project/sglang/pull/16164) | merged | [NPU] Adapt qwen3-next W8A8 on NPU | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-05 | [#16488](https://github.com/sgl-project/sglang/pull/16488) | open | Two-Batch Overlap (TBO) support to Qwen3-Next Models | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` |
| 2026-01-10 | [#16863](https://github.com/sgl-project/sglang/pull/16863) | merged | tiny refactor pcg split op registration | model wrapper, attention/backend, docs/config | `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py` |
| 2026-01-13 | [#17016](https://github.com/sgl-project/sglang/pull/17016) | merged | [bugfix] fix qwen3-next alt_stream none issue | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-20 | [#17373](https://github.com/sgl-project/sglang/pull/17373) | merged | refactor Qwen3-Next with a new RadixLinearAttention | model wrapper, attention/backend | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-22 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | model wrapper, MoE/router | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-23 | [#17613](https://github.com/sgl-project/sglang/pull/17613) | merged | [Perf] refactor piecewise cuda graph support of Qwen3-Next | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py` |
| 2026-01-23 | [#17627](https://github.com/sgl-project/sglang/pull/17627) | merged | [feat] Support nvfp4 quantized model of Qwen3-Next | model wrapper, quantization, tests/benchmarks | `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-23 | [#17660](https://github.com/sgl-project/sglang/pull/17660) | merged | [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention | model wrapper, attention/backend | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-30 | [#17981](https://github.com/sgl-project/sglang/pull/17981) | open | [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell. | attention/backend, kernel, scheduler/runtime, tests/benchmarks | `python/sglang/jit_kernel/cutedsl_gdn_transpose.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` |
| 2026-01-30 | [#17983](https://github.com/sgl-project/sglang/pull/17983) | open | [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel | attention/backend | `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` |
| 2026-02-04 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-06 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | model wrapper, attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-17 | [#18917](https://github.com/sgl-project/sglang/pull/18917) | merged | [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-24 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | model wrapper, quantization | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-25 | [#19321](https://github.com/sgl-project/sglang/pull/19321) | merged | [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj | model wrapper | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py` |
| 2026-02-26 | [#19434](https://github.com/sgl-project/sglang/pull/19434) | merged | [Qwen3-Next] Support gdn fused_rms_norm_gated | model wrapper, attention/backend | `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-03 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-03-04 | [#19812](https://github.com/sgl-project/sglang/pull/19812) | open | Fix Qwen3.5/Qwen3Next MTP EPLB compatibility | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-12 | [#20397](https://github.com/sgl-project/sglang/pull/20397) | open | [NPU] Qwen3 next Ascend Support MTP | model wrapper, attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-03-20 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | model wrapper, kernel | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-24 | [#21313](https://github.com/sgl-project/sglang/pull/21313) | merged | bugfix for weight loading for qwen3-next | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-26 | [#21496](https://github.com/sgl-project/sglang/pull/21496) | merged | Revert "bugfix for weight loading for qwen3-next" | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21662](https://github.com/sgl-project/sglang/pull/21662) | merged | [Fix] Fix weight_loader property assignment for qwen3-next FP8 models | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21684](https://github.com/sgl-project/sglang/pull/21684) | open | [bugfix] fix Qwen3-next memory leak | scheduler/runtime | `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-30 | [#21698](https://github.com/sgl-project/sglang/pull/21698) | open | [npu]fix: qwen3-next w8a8 precision bugs | model wrapper | `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-03 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-08 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-09 | [#22458](https://github.com/sgl-project/sglang/pull/22458) | merged | Fix NCCL AllGather hanging issue for Qwen3 Next MTP | misc | `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-04-13 | [#22664](https://github.com/sgl-project/sglang/pull/22664) | merged | Qwen3next flashinfer allreduce auto enable | misc | `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#22876](https://github.com/sgl-project/sglang/pull/22876) | open | Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac | tests/benchmarks | `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py` |
| 2026-04-17 | [#23075](https://github.com/sgl-project/sglang/pull/23075) | open | [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view. | attention/backend | `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-04-20 | [#23273](https://github.com/sgl-project/sglang/pull/23273) | open | [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell) | attention/backend, kernel | `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | tests/benchmarks | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |

### 逐 PR 代码 diff 阅读记录

### PR #10233 - Qwen3-Next support

- 链接：https://github.com/sgl-project/sglang/pull/10233
- 状态/时间：`merged`，created 2025-09-09, merged 2025-09-11；作者 `yizhang2077`。
- 代码 diff 已读范围：`19` 个文件，`+3224/-8`；代码面：model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config；关键词：attention, cache, spec, config, cuda, kv, moe, triton, eagle, expert。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunk: +import enum; 符号: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` added +581/-0 (581 lines); hunk: +from dataclasses import astuple, dataclass; 符号: ForwardMetadata:, MambaAttnBackend, __init__, _get_cached_arange
  - `python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunk: +# coding=utf-8; 符号: HybridLayerType, Qwen3NextConfig, to, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +280/-0 (280 lines); hunk: def clear(self):; def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):; 符号: clear, MambaPool:, __init__, get_mamba_params_all_layers
  - `python/sglang/srt/speculative/eagle_target_verify_cuda_graph_runner.py` added +195/-0 (195 lines); hunk: +import bisect; 符号: MambaStateUpdateCudaGraphRunner:, __init__, init_cuda_graph_state, capture
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/configs/qwen3_next.py`；patch 关键词为 attention, cache, spec, config, cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/configs/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10322 - [bugfix] fix norm type error in qwen3_next model

- 链接：https://github.com/sgl-project/sglang/pull/10322
- 状态/时间：`merged`，created 2025-09-11, merged 2025-09-11；作者 `cao1zhg`。
- 代码 diff 已读范围：`2` 个文件，`+10/-51`；代码面：model wrapper；关键词：config, attention, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, get_layer, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；patch 关键词为 config, attention, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10379 - Support Qwen3-Next on Ascend NPU

- 链接：https://github.com/sgl-project/sglang/pull/10379
- 状态/时间：`merged`，created 2025-09-12, merged 2025-09-12；作者 `iforgetmyname`。
- 代码 diff 已读范围：`10` 个文件，`+79/-26`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, cache, cuda, kv, triton, config, doc, flash, test, deepep。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +22/-4 (26 lines); hunk: from sglang.srt.model_executor.model_runner import ModelRunner; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-5 (21 lines); hunk: def init_memory_pool(; def init_memory_pool(; 符号: init_memory_pool, init_memory_pool, _get_attention_backend_from_str
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +12/-3 (15 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, set_kv_buffer
  - `scripts/ci/npu_ci_install_dependency.sh` modified +10/-4 (14 lines); hunk: wget -O "${PTA_NAME}" "${PTA_URL}" && ${PIP_INSTALL} "./${PTA_NAME}"
  - `docker/Dockerfile.npu` modified +8/-3 (11 lines); hunk: ARG PYTORCH_VERSION=2.6.0; RUN git clone https://github.com/sgl-project/sglang --branch $SGLANG_TAG && \
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 attention, cache, cuda, kv, triton, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10392 - [Fix] Support qwen3-next MTP+DP

- 链接：https://github.com/sgl-project/sglang/pull/10392
- 状态/时间：`merged`，created 2025-09-12, merged 2025-09-13；作者 `byjiang1996`。
- 代码 diff 已读范围：`4` 个文件，`+29/-18`；代码面：model wrapper, multimodal/processor, scheduler/runtime, docs/config；关键词：attention, cache, config, kv, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +22/-14 (36 lines); hunk: import torch_npu; def __init__(; 符号: get_tensor_size_bytes, ReqToTokenPool:, __init__, get_mamba_params_all_layers
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/layers/logits_processor.py` modified +1/-2 (3 lines); hunk: def compute_dp_attention_metadata(self):; 符号: compute_dp_attention_metadata
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/models/qwen3_next_mtp.py`, `python/sglang/srt/layers/logits_processor.py`；patch 关键词为 attention, cache, config, kv, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/models/qwen3_next_mtp.py`, `python/sglang/srt/layers/logits_processor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10466 - feat: update support for qwen3next model

- 链接：https://github.com/sgl-project/sglang/pull/10466
- 状态/时间：`merged`，created 2025-09-15, merged 2025-09-16；作者 `cao1zhg`。
- 代码 diff 已读范围：`3` 个文件，`+11/-7`；代码面：model wrapper, attention/backend；关键词：attention, config, cuda, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunk: def fused_recurrent_gated_delta_rule_fwd_kernel(; def fused_recurrent_gated_delta_rule_update_fwd_kernel(; 符号: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunk: def fused_sigmoid_gating_delta_rule_update_kernel(; 符号: fused_sigmoid_gating_delta_rule_update_kernel
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；patch 关键词为 attention, config, cuda, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10622 - support qwen3-next-fp8 deepep

- 链接：https://github.com/sgl-project/sglang/pull/10622
- 状态/时间：`merged`，created 2025-09-18, merged 2025-09-18；作者 `yizhang2077`。
- 代码 diff 已读范围：`2` 个文件，`+93/-9`；代码面：model wrapper, MoE/router；关键词：config, expert, moe, processor, attention, cuda, deepep, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_moe.py` modified +64/-1 (65 lines); hunk: from transformers import PretrainedConfig; RowParallelLinear,; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunk: get_tensor_model_parallel_rank,; sharded_weight_loader,; 符号: forward, __init__, routed_experts_weights_of_layer, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 config, expert, moe, processor, attention, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10657 - feat: add eagle3 support for qwen3-next model

- 链接：https://github.com/sgl-project/sglang/pull/10657
- 状态/时间：`open`，created 2025-09-19；作者 `AnnaYue`。
- 代码 diff 已读范围：`2` 个文件，`+45/-3`；代码面：model wrapper, scheduler/runtime；关键词：spec, attention, cache, config, eagle, expert, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunk: import enum; def get_layer(idx: int, prefix: str):; 符号: get_layer, forward, forward, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunk: def initialize(self, min_per_gpu_memory: float):; def _get_attention_backend(self):; 符号: initialize, _get_attention_backend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 spec, attention, cache, config, eagle, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- 链接：https://github.com/sgl-project/sglang/pull/10912
- 状态/时间：`merged`，created 2025-09-25, merged 2025-10-16；作者 `ShangmingCai`。
- 代码 diff 已读范围：`13` 个文件，`+727/-186`；代码面：attention/backend, scheduler/runtime, tests/benchmarks；关键词：kv, cache, spec, attention, config, scheduler, test, cuda, fp8, lora。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunk: def __init__(; def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; 符号: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunk: class TransferKVChunk:; class TransferInfo:; 符号: TransferKVChunk:, TransferInfo:, from_zmq, from_zmq
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunk: from collections import deque; ); 符号: clear, HybridMambaDecodeReqToTokenPool, __init__, clear
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunk: +import os; 符号: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
  - `python/sglang/srt/disaggregation/prefill.py` modified +71/-1 (72 lines); hunk: RequestStage,; def _init_kv_manager(self) -> BaseKVManager:; 符号: _init_kv_manager, send_kv_chunk
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`；patch 关键词为 kv, cache, spec, attention, config, scheduler。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11487 - init support for KTransformers Heterogeneous Computing

- 链接：https://github.com/sgl-project/sglang/pull/11487
- 状态/时间：`merged`，created 2025-10-12, merged 2025-10-21；作者 `Atream`。
- 代码 diff 已读范围：`9` 个文件，`+547/-17`；代码面：model wrapper, MoE/router, quantization, kernel, scheduler/runtime；关键词：moe, quant, config, expert, fp8, triton, cuda, topk, fp4, marlin。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunk: import enum; logger = logging.getLogger(__name__); 符号: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, GPTQMarlinState
  - `python/sglang/srt/server_args.py` modified +57/-0 (57 lines); hunk: "qoq",; class ServerArgs:; 符号: ServerArgs:, __post_init__, _handle_deprecated_args, _handle_ktransformers_configs
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunk: FusedMoEMethodBase,; def __init__(; 符号: __init__, __init__, _weight_loader_physical, _weight_loader_impl
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunk: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class; 符号: forward_normal_dual_stream, __init__, post_load_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunk: ); is_activation_quantization_format,; 符号: to_int, CompressedTensorsConfig, __init__, get_quant_method
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 moe, quant, config, expert, fp8, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11969 - [NPU] bugfix for Qwen3-Next and performance update

- 链接：https://github.com/sgl-project/sglang/pull/11969
- 状态/时间：`merged`，created 2025-10-22, merged 2025-10-30；作者 `iforgetmyname`。
- 代码 diff 已读范围：`7` 个文件，`+68/-21`；代码面：model wrapper, attention/backend, MoE/router；关键词：attention, doc, triton, config, cuda, expert, moe, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +31/-6 (37 lines); hunk: def forward_npu(; def forward_npu(; 符号: forward_npu, forward_npu
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +20/-11 (31 lines); hunk: get_tensor_model_parallel_world_size,; composed_weight_loader,
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +7/-1 (8 lines); hunk: import triton.language as tl; def _layer_norm_fwd(; 符号: rms_norm_ref, _layer_norm_fwd, rms_norm_gated
  - `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunk: def forward(; 符号: forward
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1 (2 lines); hunk: jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`；patch 关键词为 attention, doc, triton, config, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12508 - [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup

- 链接：https://github.com/sgl-project/sglang/pull/12508
- 状态/时间：`merged`，created 2025-11-02, merged 2025-11-06；作者 `byjiang1996`。
- 代码 diff 已读范围：`3` 个文件，`+71/-51`；代码面：model wrapper, attention/backend；关键词：attention, triton, cache, cuda, eagle, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunk: +from typing import Tuple; 符号: fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunk: def fused_qkvzba_split_reshape_cat(; 符号: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunk: from sglang.srt.layers.attention.base_attn_backend import AttentionBackend; from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool; 符号: forward_extend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 attention, triton, cache, cuda, eagle, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12525 - [CPU] Optimize Qwen3-next model on CPU

- 链接：https://github.com/sgl-project/sglang/pull/12525
- 状态/时间：`merged`，created 2025-11-03, merged 2026-01-30；作者 `jianan-gu`。
- 代码 diff 已读范围：`13` 个文件，`+366/-41`；代码面：model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config；关键词：attention, config, cuda, triton, cache, kv, spec, eagle, expert, fp8。
- 代码 diff 细节：
  - `sgl-kernel/python/sgl_kernel/mamba.py` modified +70/-0 (70 lines); hunk: def causal_conv1d_update(; 符号: causal_conv1d_update, causal_conv1d_fn_cpu, causal_conv1d_update_cpu, chunk_gated_delta_rule_cpu
  - `python/sglang/srt/layers/amx_utils.py` modified +49/-7 (56 lines); hunk: logger = logging.getLogger(__name__); def dim_is_supported(weight):; 符号: amx_process_weight_after_loading, amx_process_weight_after_loading, dim_is_supported, dtype_is_supported
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +35/-10 (45 lines); hunk: import triton.language as tl; from sglang.srt.server_args import get_global_server_args; 符号: __init__, forward_extend, forward_extend
  - `python/sglang/srt/configs/update_config.py` modified +43/-0 (43 lines); hunk: def get_num_heads_padding_size(tp_size, weight_block_size, head_dim):; def adjust_config_with_unaligned_cpu_tp(; 符号: get_num_heads_padding_size, adjust_tp_num_heads_if_necessary, update_intermediate_size, adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +41/-1 (42 lines); hunk: composed_weight_loader,; def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:; 符号: loader, loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/python/sgl_kernel/mamba.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 attention, config, cuda, triton, cache, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/python/sgl_kernel/mamba.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12892 - [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup

- 链接：https://github.com/sgl-project/sglang/pull/12892
- 状态/时间：`open`，created 2025-11-08；作者 `byjiang1996`。
- 代码 diff 已读范围：`6` 个文件，`+172/-241`；代码面：attention/backend, kernel, scheduler/runtime；关键词：spec, cache, attention, cuda, eagle, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunk: def causal_conv1d_fn(; def _causal_conv1d_update_kernel(; 符号: causal_conv1d_fn, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunk: def mem_usage_bytes(self):; def __init__(; 符号: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunk: from sglang.srt.utils import is_cuda, is_npu; def forward_extend(; 符号: forward_extend, forward_extend, forward_extend, forward
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunk: def fused_recurrent_gated_delta_rule(; def fused_recurrent_gated_delta_rule_update_fwd_kernel(; 符号: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2 (4 lines); hunk: void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 spec, cache, attention, cuda, eagle, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13081 - Support piecewise cuda graph for Qwen3-next

- 链接：https://github.com/sgl-project/sglang/pull/13081
- 状态/时间：`merged`，created 2025-11-11, merged 2025-11-25；作者 `Chen-0210`。
- 代码 diff 已读范围：`6` 个文件，`+112/-3`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：attention, config, cuda, triton, cache, expert, kv, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunk: import torch; _is_cuda = is_cuda(); 符号: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward
  - `test/srt/models/test_qwen3_next_models.py` modified +38/-0 (38 lines); hunk: def test_gsm8k(self):; 符号: test_gsm8k, TestQwen3NextPiecewiseCudaGraph, setUpClass, tearDownClass
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +5/-1 (6 lines); hunk: from __future__ import annotations; def at_layer_idx(self, layer: int):; 符号: at_layer_idx, mem_usage_bytes, SpeculativeState
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-0 (5 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/layers/attention/fla/chunk_o.py` modified +1/-1 (2 lines); hunk: def chunk_fwd_o(; 符号: chunk_fwd_o, grid
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `test/srt/models/test_qwen3_next_models.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 attention, config, cuda, triton, cache, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `test/srt/models/test_qwen3_next_models.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13708 - [Fix] Qwen3Next lmhead dtype

- 链接：https://github.com/sgl-project/sglang/pull/13708
- 状态/时间：`merged`，created 2025-11-21, merged 2025-11-21；作者 `ZeldaHuang`。
- 代码 diff 已读范围：`1` 个文件，`+0/-1`；代码面：model wrapper；关键词：config, expert, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 config, expert, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13964 - [Performance]Qwen3 Next kernel performance optimize

- 链接：https://github.com/sgl-project/sglang/pull/13964
- 状态/时间：`open`，created 2025-11-26；作者 `Jacki1223`。
- 代码 diff 已读范围：`1` 个文件，`+34/-24`；代码面：attention/backend；关键词：attention, config, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunk: from sglang.srt.layers.attention.fla.utils import input_guard; def fused_sigmoid_gating_delta_rule_update_kernel(; 符号: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；patch 关键词为 attention, config, spec, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14502 - [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next

- 链接：https://github.com/sgl-project/sglang/pull/14502
- 状态/时间：`open`，created 2025-12-05；作者 `Chen-0210`。
- 代码 diff 已读范围：`5` 个文件，`+248/-123`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime, docs/config；关键词：attention, cuda, cache, config, kv, spec, triton, eagle, expert。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunk: import triton.language as tl; from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput; 符号: forward_extend, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunk: import torch; make_layers,; 符号: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunk: class State:; class SpeculativeState(State):; 符号: State:, at_layer_idx, SpeculativeState, at_layer_idx
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunk: def warmup_torch_compile(self, num_tokens: int):; def capture_one_batch_size(self, num_tokens: int):; 符号: warmup_torch_compile, capture_one_batch_size
  - `python/sglang/srt/compilation/compilation_config.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 attention, cuda, cache, config, kv, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14607 - support qwen3-next eagle3

- 链接：https://github.com/sgl-project/sglang/pull/14607
- 状态/时间：`merged`，created 2025-12-08, merged 2026-02-01；作者 `sleepcoo`。
- 代码 diff 已读范围：`1` 个文件，`+73/-6`；代码面：model wrapper；关键词：cache, config, cuda, eagle, expert, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunk: def forward(; def forward(; 符号: forward, forward, get_layer, set_eagle3_layers_to_capture
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cache, config, cuda, eagle, expert, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14855 - Clean up GDN Init

- 链接：https://github.com/sgl-project/sglang/pull/14855
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-13；作者 `hebiao064`。
- 代码 diff 已读范围：`1` 个文件，`+5/-13`；代码面：model wrapper；关键词：attention, config, expert, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunk: from sglang.srt.compilation.piecewise_context_manager import get_forward_context; def __init__(; 符号: __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 attention, config, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15631 - [jit-kernel] Add CuTe DSL GDN Decode Kernel

- 链接：https://github.com/sgl-project/sglang/pull/15631
- 状态/时间：`merged`，created 2025-12-22, merged 2026-01-18；作者 `liz-badada`。
- 代码 diff 已读范围：`4` 个文件，`+1804/-1`；代码面：attention/backend, kernel, tests/benchmarks；关键词：cuda, attention, test, triton, benchmark, cache, eagle, spec。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunk: +"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""; 符号: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunk: +"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""; 符号: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunk: import triton.language as tl; from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput; 符号: __init__, forward_decode, forward_decode
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunk: class Envs:; 符号: Envs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 cuda, attention, test, triton, benchmark, cache。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16164 - [NPU] Adapt qwen3-next W8A8 on NPU

- 链接：https://github.com/sgl-project/sglang/pull/16164
- 状态/时间：`merged`，created 2025-12-30, merged 2026-01-03；作者 `shengzhaotian`。
- 代码 diff 已读范围：`1` 个文件，`+18/-5`；代码面：model wrapper；关键词：attention, config, cuda, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 attention, config, cuda, kv, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16488 - Two-Batch Overlap (TBO) support to Qwen3-Next Models

- 链接：https://github.com/sgl-project/sglang/pull/16488
- 状态/时间：`open`，created 2026-01-05；作者 `longshiW`。
- 代码 diff 已读范围：`6` 个文件，`+484/-13`；代码面：model wrapper, attention/backend, MoE/router, tests/benchmarks；关键词：attention, cuda, expert, config, moe, deepep, kv, processor, router, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunk: from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation; set_weight_attrs,; 符号: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer
  - `python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunk: is_cuda,; def forward(; 符号: forward, op_gate, op_shared_experts, op_select_experts
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunk: def init_new_tbo(; def _compute_moe_qwen3_decode(layer):; 符号: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0 (9 lines); hunk: def compute_split_seq_index(; 符号: compute_split_seq_index
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunk: def _forward_metadata(self, forward_batch: ForwardBatch):; def forward_extend(; 符号: _forward_metadata, forward_extend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`；patch 关键词为 attention, cuda, expert, config, moe, deepep。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16863 - tiny refactor pcg split op registration

- 链接：https://github.com/sgl-project/sglang/pull/16863
- 状态/时间：`merged`，created 2026-01-10, merged 2026-01-10；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`4` 个文件，`+20/-6`；代码面：model wrapper, attention/backend, docs/config；关键词：config, attention, expert。
- 代码 diff 细节：
  - `python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/compilation_config.py; def __init__(; 符号: register_split_op, decorator, __init__, add_split_op
  - `python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunk: import torch.distributed; def _register_group(group: "GroupCoordinator") -> None:; 符号: _register_group, inplace_all_reduce
  - `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunk: import torch; def forward(; 符号: forward, unified_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunk: import torch; def get_model_config_for_expert_location(cls, config):; 符号: get_model_config_for_expert_location, gdn_with_output
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py`；patch 关键词为 config, attention, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17016 - [bugfix] fix qwen3-next alt_stream none issue

- 链接：https://github.com/sgl-project/sglang/pull/17016
- 状态/时间：`merged`，created 2026-01-13, merged 2026-01-16；作者 `billishyahao`。
- 代码 diff 已读范围：`1` 个文件，`+5/-1`；代码面：model wrapper；关键词：cuda, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunk: def _forward_input_proj(self, hidden_states: torch.Tensor):; 符号: _forward_input_proj
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17373 - refactor Qwen3-Next with a new RadixLinearAttention

- 链接：https://github.com/sgl-project/sglang/pull/17373
- 状态/时间：`merged`，created 2026-01-20, merged 2026-01-22；作者 `zminglei`。
- 代码 diff 已读范围：`3` 个文件，`+200/-106`；代码面：model wrapper, attention/backend；关键词：attention, kv, cache, config, cuda, moe, quant, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +96/-69 (165 lines); hunk: Mamba2Metadata,; def __init__(self, model_runner: ModelRunner):; 符号: __init__, forward_decode, forward_decode, forward_decode
  - `python/sglang/srt/layers/radix_linear_attention.py` added +83/-0 (83 lines); hunk: +# Copyright 2025-2026 SGLang Team; 符号: RadixLinearAttention, __init__, forward
  - `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunk: from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE; import triton; 符号: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 attention, kv, cache, config, cuda, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17570 - Use attn tp group in embedding for more models

- 链接：https://github.com/sgl-project/sglang/pull/17570
- 状态/时间：`merged`，created 2026-01-22, merged 2026-01-24；作者 `ispobock`。
- 代码 diff 已读范围：`19` 个文件，`+19/-19`；代码面：model wrapper, MoE/router；关键词：attention, config, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`；patch 关键词为 attention, config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17613 - [Perf] refactor piecewise cuda graph support of Qwen3-Next

- 链接：https://github.com/sgl-project/sglang/pull/17613
- 状态/时间：`merged`，created 2026-01-23, merged 2026-02-14；作者 `zminglei`。
- 代码 diff 已读范围：`5` 个文件，`+80/-34`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：attention, cuda, kv, test, config, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +61/-7 (68 lines); hunk: import torch; def forward(; 符号: forward, unified_linear_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, _forward, _forward
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +13/-1 (14 lines); hunk: import triton.language as tl; _is_npu = is_npu(); 符号: rms_norm_ref, _get_sm_count, calc_rows_per_block
  - `test/registered/models/test_qwen3_next_models_pcg.py` modified +0/-6 (6 lines); hunk: """; register_cuda_ci(; 符号: TestQwen3NextPiecewiseCudaGraph
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-1 (5 lines); hunk: def init_piecewise_cuda_graphs(self):; 符号: init_piecewise_cuda_graphs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`；patch 关键词为 attention, cuda, kv, test, config, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17627 - [feat] Support nvfp4 quantized model of Qwen3-Next

- 链接：https://github.com/sgl-project/sglang/pull/17627
- 状态/时间：`merged`，created 2026-01-23, merged 2026-02-28；作者 `zhengd-nv`。
- 代码 diff 已读范围：`2` 个文件，`+83/-1`；代码面：model wrapper, quantization, tests/benchmarks；关键词：fp4, quant, config, cuda, kv, scheduler, test。
- 代码 diff 细节：
  - `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunk: +import unittest; 符号: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunk: def __init__(; def load_weights(; 符号: __init__, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 fp4, quant, config, cuda, kv, scheduler。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17660 - [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention

- 链接：https://github.com/sgl-project/sglang/pull/17660
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-27；作者 `zminglei`。
- 代码 diff 已读范围：`4` 个文件，`+54/-105`；代码面：model wrapper, attention/backend；关键词：attention, cache, kv。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunk: def forward_decode(; def forward_decode(; 符号: forward_decode, forward_decode, forward_decode, forward_extend
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunk: class RadixLinearAttention(nn.Module):; 符号: RadixLinearAttention, __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`；patch 关键词为 attention, cache, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17981 - [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.

- 链接：https://github.com/sgl-project/sglang/pull/17981
- 状态/时间：`open`，created 2026-01-30；作者 `Jon-WZQ`。
- 代码 diff 已读范围：`9` 个文件，`+2128/-88`；代码面：attention/backend, kernel, scheduler/runtime, tests/benchmarks；关键词：attention, cuda, cache, triton, kv, benchmark, config, test。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/cutedsl_gdn_transpose.py` added +1038/-0 (1038 lines); hunk: +import logging; 符号: reduce_dim0, L2Norm, fused_recurrent_sigmoid_update_kernel_128x32_col, fused_recurrent_sigmoid_update_128x32_col
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` modified +858/-57 (915 lines); hunk: """Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""; TRITON_AVAILABLE = False; 符号: print_summary_tables, run_triton_kernel, run_triton_kernel, run_triton_kernel
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunk: +import logging; 符号: CuteDSLGDNTransposeKernel, decode, extend, target_verify
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunk: def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; 符号: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunk: from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating; def __init__(; 符号: __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/cutedsl_gdn_transpose.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`；patch 关键词为 attention, cuda, cache, triton, kv, benchmark。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/cutedsl_gdn_transpose.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17983 - [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel

- 链接：https://github.com/sgl-project/sglang/pull/17983
- 状态/时间：`open`，created 2026-01-30；作者 `slowlyC`。
- 代码 diff 已读范围：`9` 个文件，`+1248/-97`；代码面：attention/backend；关键词：attention, triton, kv, spec, config, cuda, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunk: +from sglang.srt.layers.attention.fla.gluon import (; 符号: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunk: +from sglang.srt.layers.attention.fla.gluon import (; 符号: recompute_w_u_fwd_kernel_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunk: +from sglang.srt.layers.attention.fla.gluon import (; 符号: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunk: prepare_chunk_offsets,; def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; 符号: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18 (124 lines); hunk: import triton.language as tl; def chunk_local_cumsum_scalar(; 符号: chunk_local_cumsum_scalar_vectorization_kernel, chunk_local_cumsum_scalar
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`；patch 关键词为 attention, triton, kv, spec, config, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/18224
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-08；作者 `vincentzed`。
- 代码 diff 已读范围：`1` 个文件，`+35/-6`；代码面：model wrapper；关键词：cache, config, expert, fp8, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunk: def __init__(; class HybridLayerType(enum.Enum):; 符号: __init__, HybridLayerType, Qwen3NextForCausalLM, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cache, config, expert, fp8, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- 链接：https://github.com/sgl-project/sglang/pull/18355
- 状态/时间：`merged`，created 2026-02-06, merged 2026-02-25；作者 `yichiche`。
- 代码 diff 已读范围：`2` 个文件，`+213/-74`；代码面：model wrapper, attention/backend；关键词：cuda, attention, cache, config, flash, kv, mla, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunk: class ForwardMetadata:; def __init__(; 符号: ForwardMetadata:, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunk: def _forward_input_proj(self, hidden_states: torch.Tensor):; 符号: _forward_input_proj
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cuda, attention, cache, config, flash, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #18917 - [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill

- 链接：https://github.com/sgl-project/sglang/pull/18917
- 状态/时间：`merged`，created 2026-02-17, merged 2026-02-22；作者 `YAMY1234`。
- 代码 diff 已读范围：`1` 个文件，`+1/-7`；代码面：model wrapper；关键词：cuda, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #19321 - [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj

- 链接：https://github.com/sgl-project/sglang/pull/19321
- 状态/时间：`merged`，created 2026-02-25, merged 2026-03-20；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+107/-17`；代码面：model wrapper；关键词：quant, attention, config, fp8, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunk: from sglang.srt.layers.layernorm import GemmaRMSNorm; def __init__(; 符号: __init__, __init__, fix_query_key_value_ordering, _make_packed_weight_loader
  - `python/sglang/srt/layers/linear.py` modified +24/-6 (30 lines); hunk: def weight_loader(; def weight_loader(; 符号: weight_loader, weight_loader, _load_fused_module_from_checkpoint, _load_fused_module_from_checkpoint
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py`；patch 关键词为 quant, attention, config, fp8, kv, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19434 - [Qwen3-Next] Support gdn fused_rms_norm_gated

- 链接：https://github.com/sgl-project/sglang/pull/19434
- 状态/时间：`merged`，created 2026-02-26, merged 2026-02-27；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+411/-299`；代码面：model wrapper, attention/backend；关键词：attention, config, triton, vision, cuda, expert, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` added +388/-0 (388 lines); hunk: +# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py; 符号: layer_norm_gated_fwd_kernel, layer_norm_gated_fwd_kernel1, layer_norm_gated_fwd, LayerNormGatedFunction
  - `python/sglang/srt/layers/attention/fla/kda.py` modified +1/-290 (291 lines); hunk: # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang; def fused_recurrent_kda(; 符号: fused_recurrent_kda, layer_norm_gated_fwd_kernel, layer_norm_gated_fwd_kernel1, layer_norm_gated_fwd
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunk: ); def __init__(; 符号: __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +2/-1 (3 lines); hunk: tensor_model_parallel_all_reduce,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 attention, config, triton, vision, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #19812 - Fix Qwen3.5/Qwen3Next MTP EPLB compatibility

- 链接：https://github.com/sgl-project/sglang/pull/19812
- 状态/时间：`open`，created 2026-03-04；作者 `AjAnubolu`。
- 代码 diff 已读范围：`2` 个文件，`+26/-0`；代码面：model wrapper, MoE/router；关键词：config, expert, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunk: def __init__(; def __init__(; 符号: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；patch 关键词为 config, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20397 - [NPU] Qwen3 next Ascend Support MTP

- 链接：https://github.com/sgl-project/sglang/pull/20397
- 状态/时间：`open`，created 2026-03-12；作者 `ranjiewen`。
- 代码 diff 已读范围：`11` 个文件，`+985/-94`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime；关键词：attention, kv, spec, triton, cache, config, cuda, deepep, eagle, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunk: def fused_mamba_state_scatter_with_mask(; 符号: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunk: -from typing import Tuple, Union; causal_conv1d_fn_npu,; 符号: vllm_causal_conv1d_update, GDNKernelDispatcher:, forward_decode, forward_extend
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunk: from sglang.srt.server_args import get_global_server_args; def __init__(self, model_runner: ModelRunner):; 符号: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +29/-22 (51 lines); hunk: def forward_extend(; 符号: forward_extend
  - `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` modified +17/-0 (17 lines); hunk: from sglang.srt.layers.radix_attention import RadixAttention; 符号: _init_npu_conv_state, NPUMHATokenToKVPool, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；patch 关键词为 attention, kv, spec, triton, cache, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #21313 - bugfix for weight loading for qwen3-next

- 链接：https://github.com/sgl-project/sglang/pull/21313
- 状态/时间：`merged`，created 2026-03-24, merged 2026-03-26；作者 `McZyWu`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21496 - Revert "bugfix for weight loading for qwen3-next"

- 链接：https://github.com/sgl-project/sglang/pull/21496
- 状态/时间：`merged`，created 2026-03-26, merged 2026-03-26；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21662 - [Fix] Fix weight_loader property assignment for qwen3-next FP8 models

- 链接：https://github.com/sgl-project/sglang/pull/21662
- 状态/时间：`merged`，created 2026-03-30, merged 2026-03-30；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+17/-4`；代码面：model wrapper；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, _override_weight_loader, _make_packed_weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21684 - [bugfix] fix Qwen3-next memory leak

- 链接：https://github.com/sgl-project/sglang/pull/21684
- 状态/时间：`open`，created 2026-03-30；作者 `Chen-0210`。
- 代码 diff 已读范围：`2` 个文件，`+2/-2`；代码面：scheduler/runtime；关键词：cache。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunk: def alloc(self, need_size: int):; 符号: alloc, free
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunk: def alloc(self, need_size: int) -> Optional[torch.Tensor]:; 符号: alloc, free
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 cache。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21698 - [npu]fix: qwen3-next w8a8 precision bugs

- 链接：https://github.com/sgl-project/sglang/pull/21698
- 状态/时间：`open`，created 2026-03-30；作者 `ranjiewen`。
- 代码 diff 已读范围：`1` 个文件，`+22/-5`；代码面：model wrapper；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunk: _is_amx_available = cpu_has_amx_support(); def _override_weight_loader(module, new_loader):; 符号: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_next.py`；patch 关键词为 kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_next.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- 链接：https://github.com/sgl-project/sglang/pull/22073
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-07；作者 `adityavaid`。
- 代码 diff 已读范围：`10` 个文件，`+571/-11`；代码面：model wrapper, multimodal/processor, tests/benchmarks, docs/config；关键词：config, moe, attention, processor, vision, spec, benchmark, cache, doc, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunk: +"""Qwen3-ASR model compatible with HuggingFace weights"""; 符号: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunk: +import torch; 符号: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, __init__
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunk: +import re; 符号: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunk: TIMESTAMP_BASE_TOKEN_ID = 50365 # <\|0.00\|>; def _convert_to_internal_request(; 符号: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix
  - `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0 (29 lines); hunk: in the GitHub search bar.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；patch 关键词为 config, moe, attention, processor, vision, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #22458 - Fix NCCL AllGather hanging issue for Qwen3 Next MTP

- 链接：https://github.com/sgl-project/sglang/pull/22458
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-10；作者 `ispobock`。
- 代码 diff 已读范围：`2` 个文件，`+38/-0`；代码面：misc；关键词：attention, eagle, processor, spec, triton, flash, kv。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunk: import torch.nn.functional as F; def verify(; 符号: verify
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunk: import triton; def sample(; 符号: sample
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；patch 关键词为 attention, eagle, processor, spec, triton, flash。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22664 - Qwen3next flashinfer allreduce auto enable

- 链接：https://github.com/sgl-project/sglang/pull/22664
- 状态/时间：`merged`，created 2026-04-13, merged 2026-04-18；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+3/-1`；代码面：misc；关键词：flash, kv, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunk: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments, _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 flash, kv, moe, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22876 - Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac

- 链接：https://github.com/sgl-project/sglang/pull/22876
- 状态/时间：`open`，created 2026-04-15；作者 `flyerming`。
- 代码 diff 已读范围：`2` 个文件，`+42/-0`；代码面：tests/benchmarks；关键词：cache, cuda, scheduler, spec, test。
- 代码 diff 细节：
  - `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunk: def test_external_corpus_max_tokens_must_be_positive(self):; 符号: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunk: def _handle_mamba_radix_cache(; 符号: _handle_mamba_radix_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, cuda, scheduler, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23075 - [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.

- 链接：https://github.com/sgl-project/sglang/pull/23075
- 状态/时间：`open`，created 2026-04-17；作者 `flyerming`。
- 代码 diff 已读范围：`3` 个文件，`+51/-13`；代码面：attention/backend；关键词：attention, cache, scheduler。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunk: def prepare_mixed(; def prepare_mixed(; 符号: prepare_mixed, prepare_mixed, prepare_mixed
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunk: import logging; def _forward_metadata(self, forward_batch: ForwardBatch):; 符号: _forward_metadata
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunk: def mix_with_running(self, running_batch: "ScheduleBatch"):; def filter_batch(; 符号: mix_with_running, filter_batch, merge_batch, merge_batch
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`；patch 关键词为 attention, cache, scheduler。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23273 - [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)

- 链接：https://github.com/sgl-project/sglang/pull/23273
- 状态/时间：`open`，created 2026-04-20；作者 `wenscarl`。
- 代码 diff 已读范围：`2` 个文件，`+54/-22`；代码面：attention/backend, kernel；关键词：flash, attention, cuda, spec, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunk: Both SM90 and SM100+ use the same pool layout: [pool, HV, V, K] (K-last).; _flashinfer_chunk_gated_delta_rule = None; 符号: _get_flashinfer_gdn_kernels, _get_flashinfer_gdn_kernels, _get_flashinfer_gdn_kernels, FlashInferGDNKernel
  - `python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunk: def _handle_mamba_backend(self):; 符号: _handle_mamba_backend, _handle_linear_attn_backend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`；patch 关键词为 flash, attention, cuda, spec, topk, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

- 已覆盖 PR 数：55；open PR 数：15。
- 仍需跟进的 open PR：[#10657](https://github.com/sgl-project/sglang/pull/10657), [#12892](https://github.com/sgl-project/sglang/pull/12892), [#13964](https://github.com/sgl-project/sglang/pull/13964), [#14502](https://github.com/sgl-project/sglang/pull/14502), [#16488](https://github.com/sgl-project/sglang/pull/16488), [#17981](https://github.com/sgl-project/sglang/pull/17981), [#17983](https://github.com/sgl-project/sglang/pull/17983), [#19812](https://github.com/sgl-project/sglang/pull/19812), [#20397](https://github.com/sgl-project/sglang/pull/20397), [#21684](https://github.com/sgl-project/sglang/pull/21684), [#21698](https://github.com/sgl-project/sglang/pull/21698), [#22876](https://github.com/sgl-project/sglang/pull/22876), [#23075](https://github.com/sgl-project/sglang/pull/23075), [#23273](https://github.com/sgl-project/sglang/pull/23273), [#23474](https://github.com/sgl-project/sglang/pull/23474)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
