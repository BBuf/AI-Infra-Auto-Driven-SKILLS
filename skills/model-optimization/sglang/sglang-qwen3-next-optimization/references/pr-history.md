# Qwen3-Next PR Diff Dossier

Evidence sweep:

- SGLang current-main snapshot inspected: `b3e6cf60a` on `2026-04-22`.
- sgl-cookbook snapshot inspected: `816bad5` on `2026-04-21`.
- Diff sources: every PR listed below was inspected through GitHub PR metadata and code diff, not only title search.
- Runtime surfaces: `qwen3_next.py`, `qwen3_next_mtp.py`, `qwen3_next.py` config, hybrid linear-attention backend, Mamba memory pools, FlashInfer/CuTe/Gluon GDN kernels, server args, CPU offloader, AMD/NPU backends, and registered Qwen3-Next tests.
- Public-doc/blog sweep: official SGLang Qwen3/Next deployment docs, SGLang cookbook Qwen3-Next/Qwen3-Coder-Next pages, FlashInfer GDN notes in PRs, and public Qwen3-Next optimization blog material referenced by kernel PRs.

## Runtime Surfaces

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
- `test/registered/4-gpu-models/test_qwen3_next_models.py`
- `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`
- `test/registered/models/test_qwen3_next_models_fp4.py`

## Source Links

- SGLang PRs: `https://github.com/sgl-project/sglang/pull/<id>`
- Official docs/cookbook surfaces:
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3-Next.mdx`
  - `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`
  - `sgl-cookbook/src/components/autoregressive/Qwen3NextConfigGenerator/`
- Public optimization context:
  - Qwen3-Next Blackwell GDN kernel blog referenced by open PR `#17981`: `https://zhuanlan.zhihu.com/p/2003887397411258684`
  - FlashInfer GDN dependency trail referenced by open PR `#23273`: `flashinfer-ai/flashinfer#2810`, `#2679`, `#3145`

## Merged / Current-Main PR Cards

### #10233 - Initial Qwen3-Next Support

- Motivation: introduce the Qwen3-Next hybrid architecture and its MTP variant into SGLang. This is the root PR for `Qwen3NextForCausalLM`, `Qwen3NextForCausalLMMTP`, GDN/Mamba state pools, and hybrid attention backend routing.
- Implementation:
  - Added `Qwen3NextConfig` and architecture registration for base and draft models.
  - Introduced `HybridLayerType` values for full attention, linear attention, and Mamba-like layers.
  - Added `MambaPool`, `HybridReqToTokenPool`, `HybridLinearKVPool`, and `hybrid_linear_attn_backend.py` so KV cache and Mamba state cache can coexist.
  - Added `qwen3_next.py` and `qwen3_next_mtp.py`, including Gated DeltaNet projection, convolution state, fused GDN paths, and MTP verify support.
  - Added server-arg plumbing such as `--mamba-ssm-dtype` and state-aware cuda graph handling.
- Key code:

```python
if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
    self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
```

```python
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"
```

```python
mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
    forward_batch.req_pool_indices
)
```

- Validation evidence in PR: GSM8K around `0.945`; MTP throughput improved from roughly `180` to `304` tok/s with accept length around `3.32`.

### #10322 - Qwen3-Next Norm Type Fix

- Motivation: Hugging Face/Transformers-side norm configuration changed, and conditional norm selection in SGLang produced the wrong normalization behavior for Qwen3-Next.
- Implementation:
  - Removed the `use_gemma_rms_norm` / `apply_layernorm_1p` conditional path.
  - Standardized input, post-attention, final, and MTP pre-fc normalization on `GemmaRMSNorm`.
- Key code:

```python
self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### #10379 - Ascend NPU Bring-up for Qwen3-Next

- Motivation: make the initial Qwen3-Next hybrid path run on Ascend NPU, where attention backend, page size, token-to-KV pool, and causal-conv/GDN kernels differ from CUDA.
- Implementation:
  - Imported `sgl_kernel_npu` chunk GDN, fused gating, and causal conv kernels under `is_npu()`.
  - Routed `HybridLinearKVPool` to `AscendTokenToKVPool` on NPU.
  - Chose `AscendAttnBackend` inside the hybrid backend and forced page size `128` for Ascend hybrid attention.
- Key code:

```python
if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu
```

```python
full_attn_backend = AscendAttnBackend(self) if _is_npu else FlashAttentionBackend(self)
```

### #10392 - Qwen3-Next MTP + DP Fixes

- Motivation: speculative decoding with DP exposed cuda graph and idle-batch bugs in the draft worker path.
- Implementation:
  - Fixed `set_dp_buffer_len` interactions.
  - Made draft Qwen3-Next set `num_nextn_predict_layers = 1`.
  - Handled idle batch size `0` in `qwen3_next_mtp.forward`.
  - Counted every Mamba state tensor in memory sizing.
- Key code:

```python
self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
self.hf_config.num_nextn_predict_layers = 1
```

```python
def get_mamba_size(self):
    return sum(get_tensor_size_bytes(t) for t in self.mamba_cache)
```

- Validation evidence in PR: TP4 DP2 MTP GSM8K around `0.945`.

### #10466 - FP8 and L2Norm Fixes for Qwen3-Next

- Motivation: fix GDN L2-normalization accuracy and prepare the FP8 Qwen3-Next path; PR notes DeepGEMM was not yet compatible and recommended `SGL_ENABLE_JIT_DEEPGEMM=0`.
- Implementation:
  - Threaded `quant_config` into `Qwen3GatedDeltaNet` and hybrid decoder layers.
  - Fixed FLA recurrent/fused sigmoid-gating L2Norm behavior.
- Key code:

```python
def __init__(..., quant_config: Optional[QuantizationConfig] = None, alt_stream=None):
    ...
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream)
```

### #10622 - FP8 DeepEP Expert Routing

- Motivation: support `Qwen/Qwen-Next-80B-A3B-Instruct-FP8` with TP/DP/DeepEP.
- Implementation:
  - Extended Qwen2 MoE to expose routed expert weights through `get_moe_weights`.
  - Made empty-token TopK return an empty output instead of crashing.
  - Built `routed_experts_weights_of_layer` with `LazyValue` so EPLB/DeepEP can inspect expert placement.
- Key code:

```python
def get_moe_weights(self):
    return [
        x.data for name, x in self.experts.named_parameters()
        if name not in ["correction_bias"]
    ]
```

```python
topk_weights, topk_idx, _ = self.topk(
    hidden_states,
    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
        layer_id=self.layer_id
    ),
)
```

- Validation evidence in PR: TP4DP2 accuracy around `0.942`; TP8DP8 around `0.940`.

### #10912 - PD Disaggregation for Hybrid Attention State

- Motivation: PD disaggregation needed to transfer more than regular KV cache for hybrid models. Qwen3-Next carries Mamba/GDN state, so prefill and decode must exchange extra pool indices.
- Implementation:
  - Added `extra_pool_indices` to KV-transfer send/recv interfaces.
  - Exposed Mamba contiguous buffer info through memory pools.
  - Passed Mamba request mappings through prefill/decode transfer paths.
  - Added Mooncake/NIXL/fake connector support and a hybrid-attention disaggregation test.
- Key code:

```python
def send(self, kv_indices, extra_pool_indices: Optional[List[int]] = None):
    ...
```

```python
def get_extra_pool_buf_infos(self):
    return self.mamba_pool.get_contiguous_buf_infos()
```

```python
extra_pool_indices = [
    self.req_to_token_pool.rid_to_mamba_index_mapping[decode_req.req.rid]
]
```

- Validation evidence in PR: Qwen-Next GSM8K around `0.952`, throughput around `3332` tok/s in the reported setup.

### #11487 - KTransformers CPU/GPU Hybrid Inference

- Motivation: enable CPU/GPU hybrid MoE inference for Qwen3-Next-style MoE checkpoints, including GPTQ4/INT4/compressed-tensors examples.
- Implementation:
  - Added compressed-tensors WNA16 AMX MoE support and AMX wrapper dispatch.
  - Added CPU infer/offload flags such as `--amx-weight-path`, `--amx-method`, `--cpuinfer`, `--subpool-count`, and `--num-gpu-experts`.
  - Routed Qwen3-Next MoE through AMX/Marlin combine paths and clipped output dimension as needed.
- Key code:

```python
class CompressedTensorsWNA16AMXMoEMethod(CompressedTensorsMoEMethod):
    ...
```

```python
output = self.amx_wrapper.forward(
    x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
)
```

### #11969 - Ascend NPU Bugfix and Performance Follow-up

- Motivation: fix Ascend decode, fused TopK, and DP-attention padding issues for Qwen3-Next-class hybrid models.
- Implementation:
  - Chose CUDA or NPU causal-conv imports based on backend.
  - Added padding to Qwen3-Next `core_attn_out` / `z` only when DP attention is enabled.
  - Used NPU fused TopK kernels for MoE routing.
- Key code:

```python
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu as causal_conv1d_fn,
    )
```

- Validation evidence in PR: BF16/W8A8 A3 NPU with `--attention-backend ascend`, DP attention, and DeepEP.

### #12508 - Fused GDN Gating

- Motivation: reduce GDN decode/verify overhead. PR notes Extend improved roughly `89us -> 79us`, Verify `3.5us -> 1.4us`, and end-to-end throughput by about `0.85%`.
- Implementation:
  - Added `fused_gdn_gating.py` Triton kernel.
  - Replaced Python-level `sigmoid`, gating, and unsqueeze sequence in the backend with a single fused call.
- Key code:

```python
g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
```

```python
fused_gdn_gating_kernel[grid](...)
return g, beta_output
```

- Validation evidence in PR: GSM8K around `0.950`; H100x4 send-one throughput around `317 -> 319.7` tok/s.

### #12525 - CPU Path Optimization

- Motivation: make Qwen3-Next CPU execution viable by adding CPU kernels for chunk GDN, fused sigmoid gating, fused qkvzba split, conv1d, and RMSNorm, while fixing TP padding and AMX state layout.
- Implementation:
  - Added `Qwen3NextRMSNormGated` CPU op.
  - Added CPU/AMX causal conv and fused GDN backend routing.
  - Added CPU TP odd-size padding logic in config and weight loading.
  - Disabled dual stream on CPU.
- Key code:

```python
class Qwen3NextRMSNormGated(CustomOp):
    def forward_cpu(self, hidden_states, gate=None):
        return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(...)
```

```python
if seq_len < DUAL_STREAM_TOKEN_THRESHOLD and not _is_cpu:
    ...
```

### #13081 - Piecewise CUDA Graph for Qwen3-Next

- Motivation: Qwen3-Next PCG needed special handling because GDN had many arguments and padded rows could carry NaNs.
- Implementation:
  - Added custom split op `gdn_with_output`.
  - Split the GDN attention region for PCG and disabled dual stream under PCG.
  - Used `torch.zeros_like` in chunk output to avoid padded-row NaN propagation.
  - Added Qwen3-Next PCG tests.
- Key code:

```python
@register_custom_op(mutates_args=["output"])
@register_split_op()
def gdn_with_output(hidden_states: torch.Tensor, output: torch.Tensor, layer_id: int):
    ret = attention_layer._forward(hidden_states, forward_batch)
    output.view(ret.shape).copy_(ret)
```

```python
DUAL_STREAM_TOKEN_THRESHOLD = (
    0 if _is_npu or get_global_server_args().enable_piecewise_cuda_graph else 1024
)
```

- Validation evidence in PR: GSM8K around `0.942`; TTFT improved in PCG benchmarks, e.g. 1024 length `99.17ms -> 67.83ms`.

### #13708 - Keep LM Head in BF16

- Motivation: `lm_head` was forced to `float`, which was unnecessary and could hurt memory/performance for BF16 Qwen3-Next runs.
- Implementation: removed the explicit `.float()` conversion on `lm_head`.
- Key code:

```python
# Removed:
# self.lm_head = self.lm_head.float()
```

### #14607 - EAGLE3 for Qwen3-Next

- Motivation: support `lukeysong/qwen3-next-draft` EAGLE3 speculative decoding.
- Implementation:
  - Added `set_eagle3_layers_to_capture`.
  - Captured auxiliary hidden states at selected layers.
  - Returned `(hidden_states, aux_hidden_states)` from the model and passed aux states into the logits processor.
- Key code:

```python
def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
    self.capture_aux_hidden_states = True
```

```python
if self.capture_aux_hidden_states:
    hidden_states, aux_hidden_states = hidden_states
return self.logits_processor(..., aux_hidden_states)
```

- Validation evidence in PR: SpecForge GSM8K accept length around `3.13`, GSM8K accuracy around `0.955`, math500 around `0.615`.

### #14855 - Clean GDN Initialization

- Motivation: remove confusing/unneeded `torch.log` initialization logic for `A_log` and clean stale code.
- Implementation: simplified `Qwen3GatedDeltaNet` initialization, removed commented code and unused imports.
- Key code:

```python
self.conv_dim = self.key_dim * 2 + self.value_dim
```

- Validation evidence in PR: GSM8K around `0.955`, throughput around `1095` tok/s.

### #15631 - CuTe DSL GDN Decode Kernel

- Motivation: add a faster CuTe DSL decode kernel for GDN, controlled by `SGLANG_USE_CUTEDSL_GDN_DECODE=1`; requires `nvidia-cutlass-dsl>=4.3.0`.
- Implementation:
  - Added `python/sglang/jit_kernel/cutedsl_gdn.py`.
  - Added lazy capability checks and a backend branch between CuTe DSL and Triton.
  - Added small/big-batch specializations, compiled-kernel cache, and precision/perf tests.
- Key code:

```python
USE_CUTEDSL_GDN_DECODE = (
    os.environ.get("SGLANG_USE_CUTEDSL_GDN_DECODE", "0") == "1"
)
```

```python
if use_cutedsl:
    core_attn_out = _cutedsl_fused_sigmoid_gating_delta_rule_update(...)
else:
    core_attn_out = fused_sigmoid_gating_delta_rule_update(...)
```

- Validation evidence in PR: E2E speedups about `4.6-5.2%` on H200, `2.6-3.4%` on B200, and `1.7-2.5%` on H20.

### #16164 - Ascend NPU W8A8 Fixes

- Motivation: fix NPU TP/EP bugs and quantized W8A8 module-name/loading issues.
- Implementation:
  - Threaded `prefix` through Qwen3-Next GDN and layers so quantized module names line up with checkpoint keys.
  - Tightened padding conditions.
- Key code:

```python
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream, prefix)
```

- Validation evidence in PR: BF16/W8A8 TP4EP4 on A3 NPU; W8A8 throughput reported around `1405` tok/s versus BF16 around `1065` tok/s.

### #16863 - Split-Op Registry Refactor

- Motivation: centralize PCG split-op registration instead of scattering special cases.
- Implementation:
  - Added `register_split_op` and `SPLIT_OPS`.
  - Registered all-reduce, unified attention, and Qwen3-Next `gdn_with_output` through the same mechanism.
- Key code:

```python
@register_custom_op(mutates_args=["output"])
@register_split_op()
def gdn_with_output(...):
    ...
```

### #17016 - AMD Alt-Stream Guard

- Motivation: AMD MI300X failed because `alt_stream` could be `None` while the GDN path still called `wait_stream`.
- Implementation: guarded the dual-stream branch with `self.alt_stream is not None`.
- Key code:

```python
if (
    seq_len < DUAL_STREAM_TOKEN_THRESHOLD
    and self.alt_stream is not None
    and get_is_capture_mode()
):
    self.alt_stream.wait_stream(current_stream)
```

- Validation evidence in PR: AMD MI300X GSM8K around `0.940`, throughput around `763` tok/s.

### #17373 - RadixLinearAttention Refactor

- Motivation: Qwen3-Next was passing many GDN-specific kwargs directly through the backend. The PR abstracted linear attention into `RadixLinearAttention`, parallel to `RadixAttention`.
- Implementation:
  - Added `python/sglang/srt/layers/radix_linear_attention.py`.
  - Moved layer metadata such as `A_log`, `dt_bias`, conv weights, and dimensions into the layer object.
  - Made backend calls take `layer`, `mixed_qkv`, `a`, and `b`.
- Key code:

```python
class RadixLinearAttention(nn.Module):
    def forward(self, forward_batch, mixed_qkv, a, b, **kwargs):
        return forward_batch.attn_backend.forward(
            layer=self, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b, **kwargs
        )
```

```python
layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
```

- Validation evidence in PR: GSM8K around `0.960` normal, `0.955` with PCG; throughput examples around `1522` and `2261` tok/s across settings.

### #17570 - Use Attention TP Group in Embedding

- Motivation: DP-attention models need embedding parallelism to use the attention TP group rather than disabling TP.
- Implementation: replaced `enable_tp=not is_dp_attention_enabled()` with `use_attn_tp_group=is_dp_attention_enabled()` in Qwen3-Next embeddings.
- Key code:

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    org_num_embeddings=config.vocab_size,
    use_attn_tp_group=is_dp_attention_enabled(),
)
```

### #17613 - PCG Refactor Around RadixLinearAttention

- Motivation: the earlier PCG path hid too much of Qwen3-Next GDN inside one fake op, so projections/out projection stayed eager. This PR moved only the attention kernel outside the graph, letting surrounding tensor work be captured.
- Implementation:
  - Added `unified_linear_attention_with_output`.
  - Let `Qwen3GatedDeltaNet` keep projections, split, norm, and output projection inside PCG.
  - Made `model_runner.init_piecewise_cuda_graphs` recognize `layer.linear_attn.attn`.
  - Returned a constant `MAX_ROWS_PER_BLOCK` in PCG to avoid torch.compile guards.
- Key code:

```python
if get_global_server_args().enable_piecewise_cuda_graph:
    return MAX_ROWS_PER_BLOCK
```

```python
if hasattr(layer.linear_attn, "attn"):
    self.attention_layers.append(layer.linear_attn.attn)
else:
    self.attention_layers.append(layer.linear_attn)
```

- Validation evidence in PR: GSM8K around `0.950 -> 0.960`; throughput around `2592 -> 2963` tok/s, about `14.3%`.

### #17627 - Qwen3-Next NVFP4

- Motivation: support `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`.
- Implementation:
  - Disabled quantization for `qkv_proj` when `quant_config.get_name() == "modelopt_fp4"` because that projection is not quantized in the checkpoint.
  - Skipped missing scale tensors that are exactly `1.0`.
  - Added a registered Qwen3-Next FP4 test.
- Key code:

```python
quant_config=(
    quant_config
    if quant_config is not None and quant_config.get_name() != "modelopt_fp4"
    else None
)
```

```python
if name.endswith("_scale") and name not in params_dict:
    assert abs(loaded_weight.item() - 1.0) < 1e-6
    continue
```

- Validation evidence in PR: accuracy around `0.955`, throughput around `1104.584` tok/s. MTP was noted as not supported by this PR.

### #17660 - RadixLinearAttention Cleanup

- Motivation: reduce redundant metadata and make the backend rely on a smaller explicit layer contract.
- Implementation:
  - Removed redundant `attention_tp_size` and local fields.
  - Stored explicit `q_dim`, `k_dim`, `v_dim`, head counts, and head dimensions on `RadixLinearAttention`.
  - Updated backend splits to use `layer.q_dim`, `layer.k_dim`, `layer.v_dim`.
- Key code:

```python
self.q_dim = num_q_heads * head_q_dim
self.k_dim = num_k_heads * head_k_dim
self.v_dim = num_v_heads * head_v_dim
```

```python
query, key, value = torch.split(
    mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
)
```

- Validation evidence in PR: Qwen3-Next GSM8K around `0.960`, throughput around `1695` tok/s; Kimi Linear was also checked.

### #18224 - ModelOpt Qwen3-Coder-Next NVFP4 Shared Loading Fix

- Motivation: Qwen3-Coder-Next NVFP4 uses the Qwen3-Next architecture and hit ModelOpt FP4/FP8 KV-scale loading gaps.
- Implementation:
  - Passed `quant_config` into `RadixAttention`.
  - Set `packed_modules_mapping` for `qkv_proj` and `gate_up_proj`.
  - Remapped ModelOpt KV scale names from `k_proj.k_scale` / `v_proj.v_scale` to `attn.k_scale` / `attn.v_scale`.
  - Used `replaced_name` carefully so stacked weight mapping does not mutate the original name too early.
- Key code:

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
elif name.endswith(".v_proj.v_scale"):
    name = name.replace(".v_proj.v_scale", ".attn.v_scale")
```

- Validation evidence in PR: GSM8K Platinum accuracy around `0.969`, throughput around `4610.959` tok/s on B300.

### #18355 - AMD Qwen3-Coder-Next

- Motivation: enable Qwen3-Coder-Next non-MTP FP8 KV and MTP on AMD MI355.
- Implementation:
  - Fixed AITER backend `v_head_dim` inference for hybrid linear attention.
  - Kept Qwen3-Next dual-stream use CUDA-only because AMD lacked that path.
  - Routed MTP draft/verify masks through Triton backend for non-MLA paths.
- Key code:

```python
elif model_runner.hybrid_gdn_config is not None:
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
```

```python
alt_stream = torch.cuda.Stream() if _is_cuda else None
```

- Validation evidence in PR: MI355x8 Qwen3-Coder-Next accuracy around `0.944`; output throughput around `3066.797` tok/s.

### #18489 - Qwen3.5 Support With Qwen3-Next Shared Patterns

- Motivation: add Qwen3.5 dense/MoE support while sharing Qwen3-Next hybrid GDN and config patterns.
- Implementation:
  - Made hybrid GDN config discovery use `hf_config.get_text_config()`.
  - Treated `Qwen3_5Config`, `Qwen3_5MoeConfig`, `Qwen3NextConfig`, JetNemotron, and JetVLM as related hybrid configs.
  - Updated Qwen3-Next attention to read `rope_parameters` when present and fall back to `rope_scaling`.
- Key code:

```python
if "rope_parameters" in config:
    self.rope_scaling = getattr(config, "rope_parameters", None)
else:
    self.rope_scaling = getattr(config, "rope_scaling", None)
```

```python
config = self.model_config.hf_config.get_text_config()
if isinstance(config, Qwen3NextConfig | Qwen3_5Config | Qwen3_5MoeConfig):
    return config
```

### #18917 - Use Fused QKVZBA Split in Prefill

- Motivation: prefill spent time in `fix_query_key_value_ordering` view/split/reshape/cat chains. The fused Triton kernel existed for CUDA graph decode and could be extended to prefill.
- Implementation: removed the cuda-graph-only guard and used `fused_qkvzba_split_reshape_cat` whenever `num_v_heads / num_k_heads` is `1`, `2`, or `4` and the backend is not CPU.
- Key code:

```python
if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(...)
```

- Validation evidence in PR: GSM8K around `0.946 -> 0.950`; TTFT improved roughly `2-4%`; per-request prefill saved about `5-6ms` for 44 GDN layers.

### #19220 - PCG Fake Impl Fix and Legacy GDN Split Cleanup

- Motivation: Qwen3.5 FP8 PCG failed because FP8 custom ops lacked fake impl; the old Qwen3-Next `gdn_with_output` became obsolete after the RadixLinearAttention PCG refactor.
- Implementation:
  - Added fake implementation for `fp8_blockwise_scaled_mm`.
  - Removed legacy `gdn_with_output` and imports from Qwen3-Next.
- Key code:

```python
@torch.library.register_fake("sgl_kernel::fp8_blockwise_scaled_mm")
def _fake_fp8_blockwise_scaled_mm(...):
    return mat_a.new_empty((M, N), dtype=out_dtype)
```

```python
# Removed from qwen3_next.py:
# @register_custom_op(mutates_args=["output"])
# @register_split_op()
# def gdn_with_output(...):
```

- Validation evidence in PR: Qwen3.5 FP8 PCG GSM8K around `0.948`.

### #19321 - Fuse Qwen3-Next GDN `qkvz_proj` and `ba_proj`

- Motivation: Qwen3-Next GDN ran separate projection modules for `qkvz` and `ba`; fusing them into `MergedColumnParallelLinear` reduced projection overhead and improved throughput.
- Implementation:
  - Extended `linear.py` weight loading to support tuple shard ids and output-size groups.
  - Replaced separate GDN projection modules with `MergedColumnParallelLinear`.
  - Added `_make_packed_weight_loader` for packed and split checkpoints.
  - Added mapping from `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a` into fused names.
- Key code:

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id: tuple[int, ...] | int | None = None):
    if isinstance(loaded_shard_id, tuple):
        return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)
```

```python
("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
("in_proj_qkvz.", "in_proj_z.", 3),
("in_proj_ba.", "in_proj_b.", 0),
("in_proj_ba.", "in_proj_a.", 1),
```

- Validation evidence in PR: no GSM8K drop; E2E throughput improved around `15314.80 -> 15733.74` tok/s.

### #19434 - Fused RMSNorm + Gate for GDN

- Motivation: GDN repeatedly performs norm + gate per token/layer; fusing reduces memory traffic. PR notes about `4%` TTFT/throughput improvement.
- Implementation:
  - Added `fla/fused_norm_gate.py` and `FusedRMSNormGated`.
  - Reused the shared kernel in KDA.
  - Used fused path for Qwen3-Next unless PCG is enabled; PCG keeps the previous `RMSNormGated`.
- Key code:

```python
self.norm = (
    RMSNormGated(..., norm_before_gate=True, ...)
    if get_global_server_args().enable_piecewise_cuda_graph
    else FusedRMSNormGated(...)
)
```

- Validation evidence in PR: E2E throughput around `15314.80 -> 15959.30` tok/s; mean TTFT around `23169 -> 22252ms`.

### #19767 - MTP + EPLB Compatibility

- Motivation: support EPLB with MTP for Qwen3.5 and Qwen3-Next without polluting expert-distribution statistics or using incorrect draft-layer ids.
- Implementation:
  - Added `is_nextn` to `Qwen2MoeSparseMoeBlock`.
  - Suppressed `ExpertLocationDispatchInfo` in NextN/draft regions.
  - Constructed Qwen3-Next MTP model with `is_nextn=True`.
  - Wrapped MTP forward with `get_global_expert_distribution_recorder().disable_this_region()`.
- Key code:

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

### #21019 - Extract GDN Fused Projection Kernels and Add Contiguous Variant

- Motivation: Qwen3-Next uses interleaved packed GDN projection layout, while Qwen3.5 uses split/contiguous layout. Shared code needed both kernels without copying logic across models.
- Implementation:
  - Moved Qwen3-Next fused kernel into `python/sglang/jit_kernel/triton/gdn_fused_proj.py`.
  - Added `fused_qkvzba_split_reshape_cat_contiguous` for Qwen3.5.
  - Kept Qwen3-Next importing the interleaved variant from the shared module.
- Key code:

```python
from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat,
)
```

```python
def fused_qkvzba_split_reshape_cat_contiguous(...):
    ...
```

- Validation evidence in PR: Qwen3.5 output throughput improved about `7.4%` and TTFT about `10.8%`; Qwen3-Next behavior remained layout-equivalent.

### #21313 - Attempted Qwen3-Next Weight Loader Fix

- Motivation: W8A8 loading failed after fused GDN projections because `weight_loader` assignment interacted with property-backed parameters.
- Implementation: attempted to write `_weight_loader` directly on fused projection weights.
- Key code:

```python
self.in_proj_qkvz.weight._weight_loader = self._make_packed_weight_loader(self.in_proj_qkvz)
self.in_proj_ba.weight._weight_loader = self._make_packed_weight_loader(self.in_proj_ba)
```

- Important status: this PR was later reverted by `#21496` and superseded by the safer helper in `#21662`.

### #21496 - Revert #21313

- Motivation: revert the incorrect #21313 loader change.
- Implementation: restored the previous direct `weight_loader` assignment shape.
- Key code:

```python
self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(self.in_proj_qkvz)
self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(self.in_proj_ba)
```

- Important status: this was an intermediate revert; current-main behavior should be read together with `#21662`.

### #21662 - Correct FP8 Weight Loader Property Assignment

- Motivation: fix `AttributeError: property 'weight_loader' of 'ModelWeightParameter' object has no setter` for Qwen3-Coder-Next-FP8 and related Qwen3-Next fused projection loading.
- Implementation:
  - Added `_override_weight_loader`.
  - If the parameter exposes `_weight_loader`, set that backing field; otherwise set `weight_loader`.
- Key code:

```python
self._override_weight_loader(
    self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)
)
```

```python
@staticmethod
def _override_weight_loader(module, new_loader):
    param = module.weight
    if hasattr(param, "_weight_loader"):
        param._weight_loader = new_loader
    else:
        param.weight_loader = new_loader
```

- Validation evidence in PR: reproduced failure on `Qwen/Qwen3-Coder-Next-FP8` TP2; fixed model loading; registered Qwen3-Next tests passed on H200.

### #22073 - Qwen3-ASR Adjacent Import/Runtime Surface

- Motivation: add Qwen3-ASR serving through `/v1/audio/transcriptions`.
- Qwen3-Next relevance: mostly adjacent; it touches Qwen-family config/model import surfaces and should not be treated as a Qwen3-Next optimization PR unless later changes share hybrid GDN plumbing.
- Key note: keep ASR model routing separate from Qwen3-Next GDN/MTP testing.

### #22358 - DFLASH Hidden-State Capture

- Motivation: enable DFLASH hidden-state collection across model backends, including Qwen3-Next.
- Implementation:
  - Added `set_dflash_layers_to_capture` on the Qwen3-Next model and wrapper.
  - Marked layers with `_is_layer_to_capture`.
  - Required explicit layer ids and offset by `+1` so hidden state after target layer `k` is captured before layer `k+1`.
- Key code:

```python
def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
    self.layers_to_capture = layers_to_capture
    for layer_id in self.layers_to_capture:
        setattr(self.layers[layer_id], "_is_layer_to_capture", True)
```

```python
if layer_ids is None:
    raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")
self.model.set_dflash_layers_to_capture([val + 1 for val in layer_ids])
```

### #22458 - Fix NCCL AllGather Hang for Qwen3-Next MTP

- Motivation: Qwen3-Next MTP with TP>1 and non-greedy sampling could hang because accepted token decisions diverged across ranks, then radix cache sequence lengths diverged, then logits all-gather sizes mismatched.
- Implementation:
  - Broadcast `predict`, `accept_index`, and `accept_length` from rank 0 after sampling in EAGLE speculative paths.
  - Use attention TP group under DP attention and regular TP group otherwise.
- Key code:

```python
tp_group = get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
if tp_group.world_size > 1:
    tp_group.broadcast(predict, src=0)
    tp_group.broadcast(accept_index, src=0)
    tp_group.broadcast(accept_length, src=0)
```

### #22664 - Auto-enable FlashInfer AllReduce Fusion

- Motivation: Qwen3-Coder-Next on H100 did not auto-enable FlashInfer all-reduce fusion; profiling showed prefill dominated by unfused cross-device reductions.
- Implementation: added `"Qwen3NextForCausalLM"` to the auto-enable whitelist for single-node SM90/SM100 TP runs.
- Key code:

```python
if model_arch in [
    "DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]:
    self.enable_flashinfer_allreduce_fusion = True
```

- Validation evidence in PR: H100 Qwen/Qwen3-Coder-Next requests/s `5.49 -> 9.41`, mean TTFT `456 -> 167ms`, TPOT `50 -> 25ms`, accuracy within variance.

## Open PR Radar Cards

### #10657 - EAGLE3 for Qwen3-Next, Superseded by #14607

- Motivation: early attempt to add EAGLE3 capture for Qwen3-Next and preserve `full_attention_backend` for draft workers.
- Implementation:
  - Draft worker uses `server_args.full_attention_backend` when present.
  - Qwen3-Next appends aux hidden states before selected layers and returns `(hidden_states, aux_hidden_states)`.
  - Defaults capture layers to `[2, num_layers // 2, num_layers - 3]`.
- Key code:

```python
elif self.is_draft_worker and hasattr(self.server_args, "full_attention_backend"):
    attn_backend = self._get_attention_backend_from_str(self.server_args.full_attention_backend)
```

```python
if i in self.layers_to_capture:
    aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
```

- Status: open but functionally superseded by merged `#14607`; keep only as historical radar.

### #12892 - Avoid SSM/Conv State Copy During Speculative Decoding

- Motivation: target verify copied intermediate SSM/conv states through extra buffers and CPU/GPU sync/scatter work. PR aims to update Mamba state lazily using accepted steps.
- Implementation:
  - Added `last_steps` to Mamba speculative state.
  - Stored accepted indices in `update_mamba_state_after_mtp_verify`.
  - Made fused recurrent and conv update kernels read `last_steps`.
  - Added EAGLE tree metadata for `topk > 1`.
- Key code:

```python
last_steps = torch.zeros((size + 1), dtype=torch.int32, device="cuda")
```

```python
mamba_caches.last_steps[state_indices_tensor] = accepted_indices
```

```python
last_step_idx = tl.load(last_steps_ptr + conv_state_batch_coord).to(tl.int64)
```

- Validation evidence in PR: GSM8K unchanged around `0.95`; update path about `339us -> 50us`; FP8 speedups around `5-9%` depending hardware/model.

### #13964 - GDN Decode Kernel Autotune

- Motivation: improve `fused_sigmoid_gating_delta_rule_update_kernel` performance.
- Implementation:
  - Added Triton autotune configs for `num_warps` and `num_stages`.
  - Precomputed `neg_exp_A`.
  - Increased `BV` up to `64`.
- Key code:

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["BK", "BV", "K", "V"],
)
```

```python
neg_exp_A = -tl.exp(b_A_log)
b_g = neg_exp_A * softplus_x
```

- Validation evidence in PR: H200 kernel average `143747ns -> 109069ns`; offline throughput `14596 -> 15179` tok/s; accuracy around `0.945`.

### #14502 - Qwen3-Next PCG Optimization

- Motivation: earlier PCG still left too much of linear attention eager. This PR moves projections/out and fused gating into graph capture, leaving only conv+GDN core outside.
- Implementation:
  - Replaced `gdn_with_output` with `causal_conv1d_gdn_with_output`.
  - Added `_causal_conv1d_gdn_core`.
  - Fixed memory-pool state access for torch.compile.
  - Fixed PCG runner dtype and `extend_prefix_lens` handling.
- Key code:

```python
self.split_ops = [
    "sglang.unified_attention_with_output",
    "sglang.causal_conv1d_gdn_with_output",
    "sglang.inplace_all_reduce",
]
```

```python
@register_custom_op(mutates_args=["core_attn_out"])
def causal_conv1d_gdn_with_output(...):
    core_attn_out_ret = forward_batch.attn_backend.linear_attn_backend._causal_conv1d_gdn_core(...)
```

- Validation evidence in PR: GSM8K around `0.948`; H200x2 TTFT at 1024 length `99.17ms` no-PCG, `67.83ms` PCG, `48.21ms` optimized PCG.

### #16488 - Two-Batch Overlap for Qwen3-Next

- Motivation: enable TBO overlap for Qwen3-Next when PCG is disabled.
- Implementation:
  - `operations_strategy.py` recognizes Qwen3 hybrid attention/linear decoder layers.
  - Added Qwen3-Next TBO prefill/decode operation strategies with `tbo_delta_stages=2`.
  - Added operation-level methods to Qwen2 MoE and Qwen3-Next layers.
  - Used `model_forward_maybe_tbo` when `forward_batch.can_run_tbo`.
- Key code:

```python
elif layer_name in ["Qwen3HybridLinearDecoderLayer", "Qwen3HybridAttentionDecoderLayer"]:
    return OperationsStrategy.concat([...])
```

```python
if forward_batch.can_run_tbo:
    hidden_states, residual = model_forward_maybe_tbo(...)
```

- Validation evidence in PR: H800 Qwen3-Next-80B-A3B-Instruct-FP8 GSM8K around `0.936`, output throughput around `245` tok/s, with profiler screenshots showing overlap.

### #17981 - CuTe DSL Decode/MTP With Transposed State

- Motivation: Blackwell underutilized Qwen3-Next GDN decode/MTP kernels. The PR transposes SSM state from `[B,H,K,V]` to `[B,H,V,K]` for contiguous K and adds CuTe DSL kernels. It references a public optimization blog.
- Implementation:
  - Added CuTe DSL transposed GDN kernels.
  - Added precision/perf tests for decode and MTP.
  - Added `SGLANG_USE_CUTEDSL_GDN_DECODE_TRANSPOSE`.
  - Auto-disables the env on non-SM100.
- Key code:

```python
if not is_sm100_supported():
    Envs.SGLANG_USE_CUTEDSL_GDN_DECODE_TRANSPOSE.set(False)
self.use_cutedsl_transpose = Envs.SGLANG_USE_CUTEDSL_GDN_DECODE_TRANSPOSE.get()
```

```python
compile_key = (dtype, B, T, H, HV, BV, use_qk_l2norm_in_kernel, cache_steps)
```

- Validation evidence in PR: decode BF16 speedups about `1.62-1.69x`; MTP BF16 about `1.29-1.57x`; Qwen3-Next FP8 E2E GSM8K around `0.961`, throughput around `1834` tok/s.

### #17983 - Gluon Prefill/Cumsum Kernels

- Motivation: optimize Qwen3-Next GDN prefill on Blackwell with Gluon kernels, transposed initial state support, and vectorized cumsum.
- Implementation:
  - Added `IS_GLUON_SUPPORTED` and `FLA_CUMSUM_SCALAR_VECTORIZATION`.
  - Added Gluon variants for `chunk_delta_h`, `chunk_o`, and `wy_fast`.
  - Added vectorized local cumsum processing multiple heads.
- Key code:

```python
IS_GLUON_SUPPORTED = (
    is_nvidia and torch.cuda.get_device_capability(0)[0] >= 10
    and os.environ.get("FLA_USE_GLUON", "1") == "1"
)
```

```python
if IS_GLUON_SUPPORTED:
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon[grid](...)
```

- Validation evidence in PR: GSM8K around `0.953`; Blackwell input:output `32K:1` examples include cumsum `7us -> 3us`, chunk output `133us -> 69us`, and wy_fast `69us -> 50us`.

### #19812 - Qwen3.5/Qwen3-Next MTP EPLB Compatibility

- Motivation: follow-up open PR for EPLB + MTP compatibility. The title includes Qwen3-Next, but the current diff mainly adds missing Qwen3.5 MoE EPLB hooks; merged `#19767` already carries Qwen3-Next-specific MTP/EPLB changes.
- Implementation:
  - Adds `self.is_nextn = is_nextn` to `Qwen2MoeSparseMoeBlock`.
  - Adds `routed_experts_weights_of_layer` and `get_model_config_for_expert_location` to Qwen3.5 MoE wrapper.
- Key code:

```python
self.is_nextn = is_nextn
```

```python
return ModelConfigForExpertLocation(
    num_layers=text_config.num_hidden_layers,
    num_logical_experts=text_config.num_experts,
    num_groups=None,
)
```

- Status: open; for Qwen3-Next documentation, reference it as EPLB radar but treat `#19767` as the merged source of Qwen3-Next behavior.

### #20397 - Ascend NPU Qwen3-Next MTP

- Motivation: bring Qwen3-Next MTP/speculative decoding to Ascend NPU.
- Implementation:
  - Uses NPU fused infer attention for Qwen3-Next `qk_head_dim == 256`.
  - Initializes NPU conv state as `[layers, pool, conv_window + draft_step, dim]`.
  - Adds MTP state-index tensors and actual sequence length tensors to hybrid backend cuda-graph metadata.
  - Uses Ascend custom conv1d update in decode and target-verify paths.
  - Adds NPU-specific fused `qkvzba` split/reshape/cat.
  - Adds Triton state rollback helpers for intermediate SSM/conv caches after MTP verify.
  - Forces MTP DeepEP dispatch envs around `qwen3_next_mtp.forward` for Ascend unquantized draft handling.
- Key code:

```python
if self.use_fia or layer.qk_head_dim == 256:
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k, value=v, input_layout="TND", ...
    )
```

```python
def _init_npu_conv_state(conv_state_in, conv_state_shape, speculative_num_draft_tokens=None):
    extra_conv_len = speculative_num_draft_tokens - 1 if speculative_num_draft_tokens else 0
    ...
```

```python
if is_npu():
    move_intermediate_cache_dynamic_h_block_v1(
        intermediate_state_cache, valid_state_indices, last_steps
    )
    conv_state_rollback(conv_states, valid_state_indices, last_steps, draft_token_num)
    return
```

- Status: open and still rough in style, but it identifies the core NPU MTP surfaces: FIA, conv state layout, graph metadata, and post-verify rollback.

### #21684 - Qwen3-Next Memory Leak / Allocator View Fix

- Motivation: Qwen3-Next memory leaked because allocator returned a view into `free_pages`; later mutation of `free_pages` could alias the returned selection.
- Implementation: clone selected page indices before returning them from both generic allocator and Mamba memory pool allocator.
- Key code:

```python
select_index = self.free_pages[:need_size]
self.free_pages = self.free_pages[need_size:]
return select_index.clone()
```

- Status: open small bugfix; relevant to hybrid cache stability.

### #21698 - NPU W8A8 Precision Fix

- Motivation: after `#19321`, Qwen3-Next W8A8 on NPU failed because fused `in_proj_qkvz` quantization parameters loaded with incomplete `_weight_loader` overrides. The NPU fused split kernel also needed to handle prefill to avoid Triton grid-size overflow.
- Implementation:
  - Imports NPU `fused_qkvzba_split_reshape_cat` from `sgl_kernel_npu.fla.utils`.
  - Extends `_override_weight_loader` to apply the loader to `weight`, `weight_scale_inv`, `weight_scale`, `input_scale`, and `weight_offset`.
- Key code:

```python
if _is_npu:
    from sgl_kernel_npu.fla.utils import (
        fused_qkvzba_split_reshape_cat as fused_qkvzba_split_reshape_cat_npu,
    )
    fused_qkvzba_split_reshape_cat = fused_qkvzba_split_reshape_cat_npu
```

```python
for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale", "weight_offset"):
    param = getattr(module, attr_name, None)
    if param is not None and hasattr(param, "_weight_loader"):
        param._weight_loader = new_loader
```

- Status: open; important for NPU quantized fused-projection correctness.

### #22876 - Guard Mixed Chunk + Mamba `extra_buffer`

- Motivation: concurrent GSM8K accuracy dropped from `93.8%` to `87.6%` when `--enable-mixed-chunk` and `--mamba-scheduler-strategy extra_buffer` were both enabled. This PR initially adds a user-facing guard instead of fixing the root cause.
- Implementation:
  - Adds a `ValueError` in `_handle_mamba_radix_cache` if `enable_mixed_chunk` and `extra_buffer` are combined.
  - Adds unit tests that verify the guard runs before CUDA checks and that extra_buffer remains allowed without mixed chunk.
- Key code:

```python
if self.enable_mixed_chunk:
    raise ValueError(
        "mamba extra_buffer is not compatible with --enable-mixed-chunk "
        "because this combination may reduce model accuracy. "
    )
```

- Status: open and largely superseded by root-cause fix `#23075`, but useful as a documentation breadcrumb for the accuracy failure mode.

### #23075 - Root-Cause Fix for Mixed Chunk + Mamba Tracking

- Motivation: after `#22876`, further debugging found the real bug: in mixed-chunk mode, `query_start_loc` and `mamba_cache_indices` were polluted by decode requests, so tracking helpers wrote Mamba conv/SSM state for prefill requests using inconsistent metadata.
- Implementation:
  - Slices tracking metadata to the prefill-only prefix in `hybrid_linear_attn_backend.py`.
  - Computes real prefill count in `mamba2_metadata.prepare_mixed`.
  - Preserves `mamba_track_indices`, mask, and sequence lengths when `ScheduleBatch.mix_with_running` merges prefill and decode batches.
- Key code:

```python
if forward_batch.forward_mode.is_mixed():
    num_prefills = forward_batch.mamba_track_mask.shape[0]
    query_start_loc_for_track = query_start_loc[: num_prefills + 1]
    mamba_cache_indices_for_track = mamba_cache_indices[:num_prefills]
```

```python
num_extend_reqs = len(forward_batch.extend_seq_lens)
num_decodes = len(forward_batch.seq_lens) - num_extend_reqs
num_prefills = num_extend_reqs - num_decodes
num_prefill_tokens = int(forward_metadata.query_start_loc[num_prefills].item())
```

- Validation evidence in PR: with both flags enabled and concurrency 16, GSM8K recovered to `0.938`.

### #23273 - FlashInfer GDN MTP Verify on SM100+

- Motivation: Blackwell SM100+ previously defaulted FlashInfer GDN decode only when speculative decoding was disabled because target verify raised `NotImplementedError`. FlashInfer already had a pool-API MTP kernel, so Qwen3-Next/Qwen3.5 MTP could use it.
- Implementation:
  - Imports `gated_delta_rule_mtp` from FlashInfer BF16-state kernel package.
  - Adapts the BF16 MTP function to the existing FP32-style target-verify interface.
  - Removes the `use_state_pool` guard in `target_verify`.
  - Allows SM100+ FlashInfer decode default even when `speculative_algorithm` is set.
- Key code:

```python
from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
)
```

```python
def _mtp_bf16_adapted(...):
    out = mtp_bf16_fn(
        A_log=A_log.float(),
        initial_state_source=initial_state,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    return out, None
```

```python
if (
    self.linear_attn_decode_backend is None
    and is_sm100_supported()
    and self.mamba_ssm_dtype == "bfloat16"
):
    self.linear_attn_decode_backend = "flashinfer"
```

- Validation evidence in PR: Qwen3.5-397B-A17B-NVFP4 B200 GSM8K Triton `0.985` vs FlashInfer `0.980`; GPQA Diamond FlashInfer `0.859`; long-output decode with MTP reached up to `1.66x` over no-MTP and around `1.03x` over Triton at OSL 4096.

### #23474 - CPU Offload for Hybrid Linear-Attention Models

- Motivation: `--cpu-offload-gb > 0` crashed on Qwen3-Next/Qwen3.5/Kimi-Linear because `functional_call` received multiple values for tied parameters such as `linear_attn.A_log` and `linear_attn.attn.A_log`. After bypassing the crash, outputs were garbage because cached plain-tensor views of `conv1d.weight` still pointed at pre-offload GPU storage.
- Implementation:
  - In `OffloaderV1`, builds an id-based cache from `state_dict(keep_vars=True)` so tied state-dict paths reuse one materialized device tensor.
  - Scans module attributes before pinning to find plain-tensor aliases of parameter storage.
  - During forward, temporarily repoints those aliases with `as_strided` to the freshly materialized device tensor, then restores them in `finally`.
  - Adds fast unit tests for tied params and view aliases.
- Key code:

```python
for k, v in module.state_dict(keep_vars=True).items():
    dev = src_to_dev.get(id(v))
    if dev is None:
        dev = v.to(device, non_blocking=True)
        src_to_dev[id(v)] = dev
    device_state[k] = dev
```

```python
sub.__dict__[attr_name] = dev_tensor.as_strided(size, stride, offset)
...
finally:
    for sub, attr_name, old in alias_restore:
        sub.__dict__[attr_name] = old
```

- Validation evidence in PR: minimal CUDA unit tests fail before patch and pass after patch; Qwen3.5-2B with `--cpu-offload-gb 2` served 800 prompts / 234k output tokens without garbage; greedy per-prompt equivalence checked for Qwen3.5-0.8B/2B and non-hybrid Qwen3-0.6B.

## Cookbook Evidence

- sgl-cookbook `#100`: AMD MI300X/MI355X support; relevant to `#17016` and `#18355`.
- sgl-cookbook `#123`: AMD MI325X support; use as cross-check for AMD command flags.
- sgl-cookbook `#143`: Qwen3-Coder-Next cookbook; adjacent because Qwen3-Coder-Next uses shared Qwen3-Next architecture/runtime.

## Validation Notes

- Keep base Qwen3-Next, Qwen3-Coder-Next, and Qwen3.5 hybrid lanes separate. They share kernels, but not all PRs affect every checkpoint equally.
- Always test no-MTP first, then MTP, then EAGLE/NEXTN.
- For GDN projection changes, run both logits/accuracy and kernel profile checks. Shape-correct fused projection can still silently corrupt GDN layout.
- For CPU offload, test tied-parameter state dicts and cached tensor views; correctness cannot be inferred from non-hybrid dense models.
- For NPU/AMD paths, do not rely on CUDA-only validation because conv state layout, dual streams, and kernel imports differ materially.
