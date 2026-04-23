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
