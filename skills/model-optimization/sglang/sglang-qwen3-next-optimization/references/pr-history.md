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

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen3 Next PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
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

## Diff Cards

### PR #10233 - Qwen3-Next support

- Link: https://github.com/sgl-project/sglang/pull/10233
- Status/date: `merged`, created 2025-09-09, merged 2025-09-11; author `yizhang2077`.
- Diff scope read: `19` files, `+3224/-8`; areas: model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config; keywords: attention, cache, spec, config, cuda, kv, moe, triton, eagle, expert.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunks: +import enum; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` added +581/-0 (581 lines); hunks: +from dataclasses import astuple, dataclass; symbols: ForwardMetadata:, MambaAttnBackend, __init__, _get_cached_arange
  - `python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunks: +# coding=utf-8; symbols: HybridLayerType, Qwen3NextConfig, to, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +280/-0 (280 lines); hunks: def clear(self):; def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):; symbols: clear, MambaPool:, __init__, get_mamba_params_all_layers
  - `python/sglang/srt/speculative/eagle_target_verify_cuda_graph_runner.py` added +195/-0 (195 lines); hunks: +import bisect; symbols: MambaStateUpdateCudaGraphRunner:, __init__, init_cuda_graph_state, capture
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/configs/qwen3_next.py`; keywords observed in patches: attention, cache, spec, config, cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/configs/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10322 - [bugfix] fix norm type error in qwen3_next model

- Link: https://github.com/sgl-project/sglang/pull/10322
- Status/date: `merged`, created 2025-09-11, merged 2025-09-11; author `cao1zhg`.
- Diff scope read: `2` files, `+10/-51`; areas: model wrapper; keywords: config, attention, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, get_layer, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; keywords observed in patches: config, attention, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10379 - Support Qwen3-Next on Ascend NPU

- Link: https://github.com/sgl-project/sglang/pull/10379
- Status/date: `merged`, created 2025-09-12, merged 2025-09-12; author `iforgetmyname`.
- Diff scope read: `10` files, `+79/-26`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, cache, cuda, kv, triton, config, doc, flash, test, deepep.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +22/-4 (26 lines); hunks: from sglang.srt.model_executor.model_runner import ModelRunner; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-5 (21 lines); hunks: def init_memory_pool(; def init_memory_pool(; symbols: init_memory_pool, init_memory_pool, _get_attention_backend_from_str
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +12/-3 (15 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, set_kv_buffer
  - `scripts/ci/npu_ci_install_dependency.sh` modified +10/-4 (14 lines); hunks: wget -O "${PTA_NAME}" "${PTA_URL}" && ${PIP_INSTALL} "./${PTA_NAME}"
  - `docker/Dockerfile.npu` modified +8/-3 (11 lines); hunks: ARG PYTORCH_VERSION=2.6.0; RUN git clone https://github.com/sgl-project/sglang --branch $SGLANG_TAG && \
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, cache, cuda, kv, triton, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10392 - [Fix] Support qwen3-next MTP+DP

- Link: https://github.com/sgl-project/sglang/pull/10392
- Status/date: `merged`, created 2025-09-12, merged 2025-09-13; author `byjiang1996`.
- Diff scope read: `4` files, `+29/-18`; areas: model wrapper, multimodal/processor, scheduler/runtime, docs/config; keywords: attention, cache, config, kv, processor, spec.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +22/-14 (36 lines); hunks: import torch_npu; def __init__(; symbols: get_tensor_size_bytes, ReqToTokenPool:, __init__, get_mamba_params_all_layers
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/layers/logits_processor.py` modified +1/-2 (3 lines); hunks: def compute_dp_attention_metadata(self):; symbols: compute_dp_attention_metadata
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/models/qwen3_next_mtp.py`, `python/sglang/srt/layers/logits_processor.py`; keywords observed in patches: attention, cache, config, kv, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/models/qwen3_next_mtp.py`, `python/sglang/srt/layers/logits_processor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10466 - feat: update support for qwen3next model

- Link: https://github.com/sgl-project/sglang/pull/10466
- Status/date: `merged`, created 2025-09-15, merged 2025-09-16; author `cao1zhg`.
- Diff scope read: `3` files, `+11/-7`; areas: model wrapper, attention/backend; keywords: attention, config, cuda, kv, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunks: def fused_recurrent_gated_delta_rule_fwd_kernel(; def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunks: def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`; keywords observed in patches: attention, config, cuda, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10622 - support qwen3-next-fp8 deepep

- Link: https://github.com/sgl-project/sglang/pull/10622
- Status/date: `merged`, created 2025-09-18, merged 2025-09-18; author `yizhang2077`.
- Diff scope read: `2` files, `+93/-9`; areas: model wrapper, MoE/router; keywords: config, expert, moe, processor, attention, cuda, deepep, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +64/-1 (65 lines); hunks: from transformers import PretrainedConfig; RowParallelLinear,; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunks: get_tensor_model_parallel_rank,; sharded_weight_loader,; symbols: forward, __init__, routed_experts_weights_of_layer, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: config, expert, moe, processor, attention, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10657 - feat: add eagle3 support for qwen3-next model

- Link: https://github.com/sgl-project/sglang/pull/10657
- Status/date: `open`, created 2025-09-19; author `AnnaYue`.
- Diff scope read: `2` files, `+45/-3`; areas: model wrapper, scheduler/runtime; keywords: spec, attention, cache, config, eagle, expert, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunks: import enum; def get_layer(idx: int, prefix: str):; symbols: get_layer, forward, forward, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: def initialize(self, min_per_gpu_memory: float):; def _get_attention_backend(self):; symbols: initialize, _get_attention_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: spec, attention, cache, config, eagle, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- Link: https://github.com/sgl-project/sglang/pull/10912
- Status/date: `merged`, created 2025-09-25, merged 2025-10-16; author `ShangmingCai`.
- Diff scope read: `13` files, `+727/-186`; areas: attention/backend, scheduler/runtime, tests/benchmarks; keywords: kv, cache, spec, attention, config, scheduler, test, cuda, fp8, lora.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: def __init__(; def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: class TransferKVChunk:; class TransferInfo:; symbols: TransferKVChunk:, TransferInfo:, from_zmq, from_zmq
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: from collections import deque; ); symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, clear
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunks: +import os; symbols: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
  - `python/sglang/srt/disaggregation/prefill.py` modified +71/-1 (72 lines); hunks: RequestStage,; def _init_kv_manager(self) -> BaseKVManager:; symbols: _init_kv_manager, send_kv_chunk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`; keywords observed in patches: kv, cache, spec, attention, config, scheduler. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11487 - init support for KTransformers Heterogeneous Computing

- Link: https://github.com/sgl-project/sglang/pull/11487
- Status/date: `merged`, created 2025-10-12, merged 2025-10-21; author `Atream`.
- Diff scope read: `9` files, `+547/-17`; areas: model wrapper, MoE/router, quantization, kernel, scheduler/runtime; keywords: moe, quant, config, expert, fp8, triton, cuda, topk, fp4, marlin.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunks: import enum; logger = logging.getLogger(__name__); symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, GPTQMarlinState
  - `python/sglang/srt/server_args.py` modified +57/-0 (57 lines); hunks: "qoq",; class ServerArgs:; symbols: ServerArgs:, __post_init__, _handle_deprecated_args, _handle_ktransformers_configs
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunks: FusedMoEMethodBase,; def __init__(; symbols: __init__, __init__, _weight_loader_physical, _weight_loader_impl
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunks: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class; symbols: forward_normal_dual_stream, __init__, post_load_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunks: ); is_activation_quantization_format,; symbols: to_int, CompressedTensorsConfig, __init__, get_quant_method
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: moe, quant, config, expert, fp8, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11969 - [NPU] bugfix for Qwen3-Next and performance update

- Link: https://github.com/sgl-project/sglang/pull/11969
- Status/date: `merged`, created 2025-10-22, merged 2025-10-30; author `iforgetmyname`.
- Diff scope read: `7` files, `+68/-21`; areas: model wrapper, attention/backend, MoE/router; keywords: attention, doc, triton, config, cuda, expert, moe, router, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +31/-6 (37 lines); hunks: def forward_npu(; def forward_npu(; symbols: forward_npu, forward_npu
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +20/-11 (31 lines); hunks: get_tensor_model_parallel_world_size,; composed_weight_loader,
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +7/-1 (8 lines); hunks: import triton.language as tl; def _layer_norm_fwd(; symbols: rms_norm_ref, _layer_norm_fwd, rms_norm_gated
  - `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: def forward(; symbols: forward
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1 (2 lines); hunks: jobs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`; keywords observed in patches: attention, doc, triton, config, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12508 - [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup

- Link: https://github.com/sgl-project/sglang/pull/12508
- Status/date: `merged`, created 2025-11-02, merged 2025-11-06; author `byjiang1996`.
- Diff scope read: `3` files, `+71/-51`; areas: model wrapper, attention/backend; keywords: attention, triton, cache, cuda, eagle, kv, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunks: +from typing import Tuple; symbols: fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunks: def fused_qkvzba_split_reshape_cat(; symbols: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunks: from sglang.srt.layers.attention.base_attn_backend import AttentionBackend; from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool; symbols: forward_extend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, triton, cache, cuda, eagle, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12525 - [CPU] Optimize Qwen3-next model on CPU

- Link: https://github.com/sgl-project/sglang/pull/12525
- Status/date: `merged`, created 2025-11-03, merged 2026-01-30; author `jianan-gu`.
- Diff scope read: `13` files, `+366/-41`; areas: model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config; keywords: attention, config, cuda, triton, cache, kv, spec, eagle, expert, fp8.
- Code diff details:
  - `sgl-kernel/python/sgl_kernel/mamba.py` modified +70/-0 (70 lines); hunks: def causal_conv1d_update(; symbols: causal_conv1d_update, causal_conv1d_fn_cpu, causal_conv1d_update_cpu, chunk_gated_delta_rule_cpu
  - `python/sglang/srt/layers/amx_utils.py` modified +49/-7 (56 lines); hunks: logger = logging.getLogger(__name__); def dim_is_supported(weight):; symbols: amx_process_weight_after_loading, amx_process_weight_after_loading, dim_is_supported, dtype_is_supported
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +35/-10 (45 lines); hunks: import triton.language as tl; from sglang.srt.server_args import get_global_server_args; symbols: __init__, forward_extend, forward_extend
  - `python/sglang/srt/configs/update_config.py` modified +43/-0 (43 lines); hunks: def get_num_heads_padding_size(tp_size, weight_block_size, head_dim):; def adjust_config_with_unaligned_cpu_tp(; symbols: get_num_heads_padding_size, adjust_tp_num_heads_if_necessary, update_intermediate_size, adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +41/-1 (42 lines); hunks: composed_weight_loader,; def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:; symbols: loader, loader
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/python/sgl_kernel/mamba.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, config, cuda, triton, cache, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/python/sgl_kernel/mamba.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12892 - [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup

- Link: https://github.com/sgl-project/sglang/pull/12892
- Status/date: `open`, created 2025-11-08; author `byjiang1996`.
- Diff scope read: `6` files, `+172/-241`; areas: attention/backend, kernel, scheduler/runtime; keywords: spec, cache, attention, cuda, eagle, kv, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunks: def causal_conv1d_fn(; def _causal_conv1d_update_kernel(; symbols: causal_conv1d_fn, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel, _causal_conv1d_update_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunks: def mem_usage_bytes(self):; def __init__(; symbols: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunks: from sglang.srt.utils import is_cuda, is_npu; def forward_extend(; symbols: forward_extend, forward_extend, forward_extend, forward
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunks: def fused_recurrent_gated_delta_rule(; def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2 (4 lines); hunks: void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: spec, cache, attention, cuda, eagle, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13081 - Support piecewise cuda graph for Qwen3-next

- Link: https://github.com/sgl-project/sglang/pull/13081
- Status/date: `merged`, created 2025-11-11, merged 2025-11-25; author `Chen-0210`.
- Diff scope read: `6` files, `+112/-3`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: attention, config, cuda, triton, cache, expert, kv, spec, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunks: import torch; _is_cuda = is_cuda(); symbols: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward
  - `test/srt/models/test_qwen3_next_models.py` modified +38/-0 (38 lines); hunks: def test_gsm8k(self):; symbols: test_gsm8k, TestQwen3NextPiecewiseCudaGraph, setUpClass, tearDownClass
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +5/-1 (6 lines); hunks: from __future__ import annotations; def at_layer_idx(self, layer: int):; symbols: at_layer_idx, mem_usage_bytes, SpeculativeState
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-0 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/fla/chunk_o.py` modified +1/-1 (2 lines); hunks: def chunk_fwd_o(; symbols: chunk_fwd_o, grid
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `test/srt/models/test_qwen3_next_models.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, config, cuda, triton, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `test/srt/models/test_qwen3_next_models.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13708 - [Fix] Qwen3Next lmhead dtype

- Link: https://github.com/sgl-project/sglang/pull/13708
- Status/date: `merged`, created 2025-11-21, merged 2025-11-21; author `ZeldaHuang`.
- Diff scope read: `1` files, `+0/-1`; areas: model wrapper; keywords: config, expert, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: config, expert, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13964 - [Performance]Qwen3 Next kernel performance optimize

- Link: https://github.com/sgl-project/sglang/pull/13964
- Status/date: `open`, created 2025-11-26; author `Jacki1223`.
- Diff scope read: `1` files, `+34/-24`; areas: attention/backend; keywords: attention, config, spec, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunks: from sglang.srt.layers.attention.fla.utils import input_guard; def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`; keywords observed in patches: attention, config, spec, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14502 - [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/14502
- Status/date: `open`, created 2025-12-05; author `Chen-0210`.
- Diff scope read: `5` files, `+248/-123`; areas: model wrapper, attention/backend, kernel, scheduler/runtime, docs/config; keywords: attention, cuda, cache, config, kv, spec, triton, eagle, expert.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunks: import triton.language as tl; from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput; symbols: forward_extend, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunks: import torch; make_layers,; symbols: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunks: class State:; class SpeculativeState(State):; symbols: State:, at_layer_idx, SpeculativeState, at_layer_idx
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunks: def warmup_torch_compile(self, num_tokens: int):; def capture_one_batch_size(self, num_tokens: int):; symbols: warmup_torch_compile, capture_one_batch_size
  - `python/sglang/srt/compilation/compilation_config.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, cuda, cache, config, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14607 - support qwen3-next eagle3

- Link: https://github.com/sgl-project/sglang/pull/14607
- Status/date: `merged`, created 2025-12-08, merged 2026-02-01; author `sleepcoo`.
- Diff scope read: `1` files, `+73/-6`; areas: model wrapper; keywords: cache, config, cuda, eagle, expert, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunks: def forward(; def forward(; symbols: forward, forward, get_layer, set_eagle3_layers_to_capture
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cache, config, cuda, eagle, expert, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14855 - Clean up GDN Init

- Link: https://github.com/sgl-project/sglang/pull/14855
- Status/date: `merged`, created 2025-12-11, merged 2025-12-13; author `hebiao064`.
- Diff scope read: `1` files, `+5/-13`; areas: model wrapper; keywords: attention, config, expert, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunks: from sglang.srt.compilation.piecewise_context_manager import get_forward_context; def __init__(; symbols: __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: attention, config, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15631 - [jit-kernel] Add CuTe DSL GDN Decode Kernel

- Link: https://github.com/sgl-project/sglang/pull/15631
- Status/date: `merged`, created 2025-12-22, merged 2026-01-18; author `liz-badada`.
- Diff scope read: `4` files, `+1804/-1`; areas: attention/backend, kernel, tests/benchmarks; keywords: cuda, attention, test, triton, benchmark, cache, eagle, spec.
- Code diff details:
  - `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunks: +"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""; symbols: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunks: +"""Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""; symbols: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunks: import triton.language as tl; from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput; symbols: __init__, forward_decode, forward_decode
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: cuda, attention, test, triton, benchmark, cache. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16164 - [NPU] Adapt qwen3-next W8A8 on NPU

- Link: https://github.com/sgl-project/sglang/pull/16164
- Status/date: `merged`, created 2025-12-30, merged 2026-01-03; author `shengzhaotian`.
- Diff scope read: `1` files, `+18/-5`; areas: model wrapper; keywords: attention, config, cuda, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: attention, config, cuda, kv, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16488 - Two-Batch Overlap (TBO) support to Qwen3-Next Models

- Link: https://github.com/sgl-project/sglang/pull/16488
- Status/date: `open`, created 2026-01-05; author `longshiW`.
- Diff scope read: `6` files, `+484/-13`; areas: model wrapper, attention/backend, MoE/router, tests/benchmarks; keywords: attention, cuda, expert, config, moe, deepep, kv, processor, router, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunks: from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation; set_weight_attrs,; symbols: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer
  - `python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunks: is_cuda,; def forward(; symbols: forward, op_gate, op_shared_experts, op_select_experts
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunks: def init_new_tbo(; def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0 (9 lines); hunks: def compute_split_seq_index(; symbols: compute_split_seq_index
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunks: def _forward_metadata(self, forward_batch: ForwardBatch):; def forward_extend(; symbols: _forward_metadata, forward_extend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`; keywords observed in patches: attention, cuda, expert, config, moe, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16863 - tiny refactor pcg split op registration

- Link: https://github.com/sgl-project/sglang/pull/16863
- Status/date: `merged`, created 2026-01-10, merged 2026-01-10; author `Qiaolin-Yu`.
- Diff scope read: `4` files, `+20/-6`; areas: model wrapper, attention/backend, docs/config; keywords: config, attention, expert.
- Code diff details:
  - `python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/compilation_config.py; def __init__(; symbols: register_split_op, decorator, __init__, add_split_op
  - `python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunks: import torch.distributed; def _register_group(group: "GroupCoordinator") -> None:; symbols: _register_group, inplace_all_reduce
  - `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunks: import torch; def forward(; symbols: forward, unified_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: import torch; def get_model_config_for_expert_location(cls, config):; symbols: get_model_config_for_expert_location, gdn_with_output
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py`; keywords observed in patches: config, attention, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17016 - [bugfix] fix qwen3-next alt_stream none issue

- Link: https://github.com/sgl-project/sglang/pull/17016
- Status/date: `merged`, created 2026-01-13, merged 2026-01-16; author `billishyahao`.
- Diff scope read: `1` files, `+5/-1`; areas: model wrapper; keywords: cuda, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17373 - refactor Qwen3-Next with a new RadixLinearAttention

- Link: https://github.com/sgl-project/sglang/pull/17373
- Status/date: `merged`, created 2026-01-20, merged 2026-01-22; author `zminglei`.
- Diff scope read: `3` files, `+200/-106`; areas: model wrapper, attention/backend; keywords: attention, kv, cache, config, cuda, moe, quant, spec, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +96/-69 (165 lines); hunks: Mamba2Metadata,; def __init__(self, model_runner: ModelRunner):; symbols: __init__, forward_decode, forward_decode, forward_decode
  - `python/sglang/srt/layers/radix_linear_attention.py` added +83/-0 (83 lines); hunks: +# Copyright 2025-2026 SGLang Team; symbols: RadixLinearAttention, __init__, forward
  - `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunks: from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE; import triton; symbols: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: attention, kv, cache, config, cuda, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17570 - Use attn tp group in embedding for more models

- Link: https://github.com/sgl-project/sglang/pull/17570
- Status/date: `merged`, created 2026-01-22, merged 2026-01-24; author `ispobock`.
- Diff scope read: `19` files, `+19/-19`; areas: model wrapper, MoE/router; keywords: attention, config, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; keywords observed in patches: attention, config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17613 - [Perf] refactor piecewise cuda graph support of Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/17613
- Status/date: `merged`, created 2026-01-23, merged 2026-02-14; author `zminglei`.
- Diff scope read: `5` files, `+80/-34`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: attention, cuda, kv, test, config, triton.
- Code diff details:
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +61/-7 (68 lines); hunks: import torch; def forward(; symbols: forward, unified_linear_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, _forward, _forward
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +13/-1 (14 lines); hunks: import triton.language as tl; _is_npu = is_npu(); symbols: rms_norm_ref, _get_sm_count, calc_rows_per_block
  - `test/registered/models/test_qwen3_next_models_pcg.py` modified +0/-6 (6 lines); hunks: """; register_cuda_ci(; symbols: TestQwen3NextPiecewiseCudaGraph
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-1 (5 lines); hunks: def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`; keywords observed in patches: attention, cuda, kv, test, config, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17627 - [feat] Support nvfp4 quantized model of Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/17627
- Status/date: `merged`, created 2026-01-23, merged 2026-02-28; author `zhengd-nv`.
- Diff scope read: `2` files, `+83/-1`; areas: model wrapper, quantization, tests/benchmarks; keywords: fp4, quant, config, cuda, kv, scheduler, test.
- Code diff details:
  - `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunks: +import unittest; symbols: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunks: def __init__(; def load_weights(; symbols: __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: fp4, quant, config, cuda, kv, scheduler. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17660 - [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention

- Link: https://github.com/sgl-project/sglang/pull/17660
- Status/date: `merged`, created 2026-01-23, merged 2026-01-27; author `zminglei`.
- Diff scope read: `4` files, `+54/-105`; areas: model wrapper, attention/backend; keywords: attention, cache, kv.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunks: def forward_decode(; def forward_decode(; symbols: forward_decode, forward_decode, forward_decode, forward_extend
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunks: class RadixLinearAttention(nn.Module):; symbols: RadixLinearAttention, __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`; keywords observed in patches: attention, cache, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17981 - [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.

- Link: https://github.com/sgl-project/sglang/pull/17981
- Status/date: `open`, created 2026-01-30; author `Jon-WZQ`.
- Diff scope read: `9` files, `+2128/-88`; areas: attention/backend, kernel, scheduler/runtime, tests/benchmarks; keywords: attention, cuda, cache, triton, kv, benchmark, config, test.
- Code diff details:
  - `python/sglang/jit_kernel/cutedsl_gdn_transpose.py` added +1038/-0 (1038 lines); hunks: +import logging; symbols: reduce_dim0, L2Norm, fused_recurrent_sigmoid_update_kernel_128x32_col, fused_recurrent_sigmoid_update_128x32_col
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` modified +858/-57 (915 lines); hunks: """Tests for CuTe DSL fused sigmoid gating delta rule kernel (GDN)."""; TRITON_AVAILABLE = False; symbols: print_summary_tables, run_triton_kernel, run_triton_kernel, run_triton_kernel
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunks: +import logging; symbols: CuteDSLGDNTransposeKernel, decode, extend, target_verify
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunks: def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunks: from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating; def __init__(; symbols: __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/cutedsl_gdn_transpose.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`; keywords observed in patches: attention, cuda, cache, triton, kv, benchmark. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/cutedsl_gdn_transpose.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17983 - [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel

- Link: https://github.com/sgl-project/sglang/pull/17983
- Status/date: `open`, created 2026-01-30; author `slowlyC`.
- Diff scope read: `9` files, `+1248/-97`; areas: attention/backend; keywords: attention, triton, kv, spec, config, cuda, flash.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunks: +from sglang.srt.layers.attention.fla.gluon import (; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunks: +from sglang.srt.layers.attention.fla.gluon import (; symbols: recompute_w_u_fwd_kernel_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunks: +from sglang.srt.layers.attention.fla.gluon import (; symbols: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunks: prepare_chunk_offsets,; def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18 (124 lines); hunks: import triton.language as tl; def chunk_local_cumsum_scalar(; symbols: chunk_local_cumsum_scalar_vectorization_kernel, chunk_local_cumsum_scalar
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`; keywords observed in patches: attention, triton, kv, spec, config, cuda. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18224
- Status/date: `merged`, created 2026-02-04, merged 2026-02-08; author `vincentzed`.
- Diff scope read: `1` files, `+35/-6`; areas: model wrapper; keywords: cache, config, expert, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: def __init__(; class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cache, config, expert, fp8, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- Link: https://github.com/sgl-project/sglang/pull/18355
- Status/date: `merged`, created 2026-02-06, merged 2026-02-25; author `yichiche`.
- Diff scope read: `2` files, `+213/-74`; areas: model wrapper, attention/backend; keywords: cuda, attention, cache, config, flash, kv, mla, spec, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: class ForwardMetadata:; def __init__(; symbols: ForwardMetadata:, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cuda, attention, cache, config, flash, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/sgl-project/sglang/pull/18489
- Status/date: `merged`, created 2026-02-09, merged 2026-02-09; author `zju-stu-lizheng`.
- Diff scope read: `17` files, `+1923/-9`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, processor, spec, attention, vision, cache, cuda, expert, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: +# Copyright 2025 Qwen Team; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: +from transformers import PretrainedConfig; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: Lfm2Config,; def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1 (17 lines); hunks: import numpy as np; from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem; symbols: preprocess_video, QwenVLImageProcessor, process_mm_data_async
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`; keywords observed in patches: moe, config, processor, spec, attention, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18917 - [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill

- Link: https://github.com/sgl-project/sglang/pull/18917
- Status/date: `merged`, created 2026-02-17, merged 2026-02-22; author `YAMY1234`.
- Diff scope read: `1` files, `+1/-7`; areas: model wrapper; keywords: cuda, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/19220
- Status/date: `merged`, created 2026-02-24, merged 2026-02-26; author `zminglei`.
- Diff scope read: `4` files, `+9/-46`; areas: model wrapper, quantization; keywords: config, attention, cuda, eagle, expert, fp8, kv, lora, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: import torch; make_layers,; symbols: set_eagle3_layers_to_capture, gdn_with_output
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: import torch.nn as nn; from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock; symbols: forward, _forward, _forward
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=No; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: config, attention, cuda, eagle, expert, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19321 - [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj

- Link: https://github.com/sgl-project/sglang/pull/19321
- Status/date: `merged`, created 2026-02-25, merged 2026-03-20; author `yuan-luo`.
- Diff scope read: `2` files, `+107/-17`; areas: model wrapper; keywords: quant, attention, config, fp8, kv, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: from sglang.srt.layers.layernorm import GemmaRMSNorm; def __init__(; symbols: __init__, __init__, fix_query_key_value_ordering, _make_packed_weight_loader
  - `python/sglang/srt/layers/linear.py` modified +24/-6 (30 lines); hunks: def weight_loader(; def weight_loader(; symbols: weight_loader, weight_loader, _load_fused_module_from_checkpoint, _load_fused_module_from_checkpoint
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py`; keywords observed in patches: quant, attention, config, fp8, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19434 - [Qwen3-Next] Support gdn fused_rms_norm_gated

- Link: https://github.com/sgl-project/sglang/pull/19434
- Status/date: `merged`, created 2026-02-26, merged 2026-02-27; author `yuan-luo`.
- Diff scope read: `4` files, `+411/-299`; areas: model wrapper, attention/backend; keywords: attention, config, triton, vision, cuda, expert, flash.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` added +388/-0 (388 lines); hunks: +# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py; symbols: layer_norm_gated_fwd_kernel, layer_norm_gated_fwd_kernel1, layer_norm_gated_fwd, LayerNormGatedFunction
  - `python/sglang/srt/layers/attention/fla/kda.py` modified +1/-290 (291 lines); hunks: # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang; def fused_recurrent_kda(; symbols: fused_recurrent_kda, layer_norm_gated_fwd_kernel, layer_norm_gated_fwd_kernel1, layer_norm_gated_fwd
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunks: ); def __init__(; symbols: __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: tensor_model_parallel_all_reduce,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: attention, config, triton, vision, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- Link: https://github.com/sgl-project/sglang/pull/19767
- Status/date: `merged`, created 2026-03-03, merged 2026-03-09; author `luoyuyan`.
- Diff scope read: `5` files, `+79/-16`; areas: model wrapper, MoE/router; keywords: config, quant, expert, moe, cuda, processor, attention, deepep, router, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: from transformers import PretrainedConfig; def __init__(; symbols: __init__, __init__, get_model_config_for_expert_location, get_embed_and_head
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: from transformers import PretrainedConfig; def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, get_moe_weights, _forward_deepep
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-0 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; keywords observed in patches: config, quant, expert, moe, cuda, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19812 - Fix Qwen3.5/Qwen3Next MTP EPLB compatibility

- Link: https://github.com/sgl-project/sglang/pull/19812
- Status/date: `open`, created 2026-03-04; author `AjAnubolu`.
- Diff scope read: `2` files, `+26/-0`; areas: model wrapper, MoE/router; keywords: config, expert, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunks: def __init__(; def __init__(; symbols: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: config, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20397 - [NPU] Qwen3 next Ascend Support MTP

- Link: https://github.com/sgl-project/sglang/pull/20397
- Status/date: `open`, created 2026-03-12; author `ranjiewen`.
- Diff scope read: `11` files, `+985/-94`; areas: model wrapper, attention/backend, kernel, scheduler/runtime; keywords: attention, kv, spec, triton, cache, config, cuda, deepep, eagle, processor.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunks: def fused_mamba_state_scatter_with_mask(; symbols: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunks: -from typing import Tuple, Union; causal_conv1d_fn_npu,; symbols: vllm_causal_conv1d_update, GDNKernelDispatcher:, forward_decode, forward_extend
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunks: from sglang.srt.server_args import get_global_server_args; def __init__(self, model_runner: ModelRunner):; symbols: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +29/-22 (51 lines); hunks: def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` modified +17/-0 (17 lines); hunks: from sglang.srt.layers.radix_attention import RadixAttention; symbols: _init_npu_conv_state, NPUMHATokenToKVPool, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, kv, spec, triton, cache, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- Link: https://github.com/sgl-project/sglang/pull/21019
- Status/date: `merged`, created 2026-03-20, merged 2026-03-23; author `yuan-luo`.
- Diff scope read: `3` files, `+597/-202`; areas: model wrapper, kernel; keywords: kv, triton, attention, config, cuda, cache, fp8, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: import torch; RowParallelLinear,; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: +from __future__ import annotations; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: from typing import Any, Iterable, Optional, Set, Tuple; logger = logging.getLogger(__name__); symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv, triton, attention, config, cuda, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21313 - bugfix for weight loading for qwen3-next

- Link: https://github.com/sgl-project/sglang/pull/21313
- Status/date: `merged`, created 2026-03-24, merged 2026-03-26; author `McZyWu`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper; keywords: kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21496 - Revert "bugfix for weight loading for qwen3-next"

- Link: https://github.com/sgl-project/sglang/pull/21496
- Status/date: `merged`, created 2026-03-26, merged 2026-03-26; author `Fridge003`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper; keywords: kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21662 - [Fix] Fix weight_loader property assignment for qwen3-next FP8 models

- Link: https://github.com/sgl-project/sglang/pull/21662
- Status/date: `merged`, created 2026-03-30, merged 2026-03-30; author `Fridge003`.
- Diff scope read: `1` files, `+17/-4`; areas: model wrapper; keywords: kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, _override_weight_loader, _make_packed_weight_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21684 - [bugfix] fix Qwen3-next memory leak

- Link: https://github.com/sgl-project/sglang/pull/21684
- Status/date: `open`, created 2026-03-30; author `Chen-0210`.
- Diff scope read: `2` files, `+2/-2`; areas: scheduler/runtime; keywords: cache.
- Code diff details:
  - `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunks: def alloc(self, need_size: int):; symbols: alloc, free
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunks: def alloc(self, need_size: int) -> Optional[torch.Tensor]:; symbols: alloc, free
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: cache. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21698 - [npu]fix: qwen3-next w8a8 precision bugs

- Link: https://github.com/sgl-project/sglang/pull/21698
- Status/date: `open`, created 2026-03-30; author `ranjiewen`.
- Diff scope read: `1` files, `+22/-5`; areas: model wrapper; keywords: kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunks: _is_amx_available = cpu_has_amx_support(); def _override_weight_loader(module, new_loader):; symbols: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- Link: https://github.com/sgl-project/sglang/pull/22073
- Status/date: `merged`, created 2026-04-03, merged 2026-04-07; author `adityavaid`.
- Diff scope read: `10` files, `+571/-11`; areas: model wrapper, multimodal/processor, tests/benchmarks, docs/config; keywords: config, moe, attention, processor, vision, spec, benchmark, cache, doc, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: +"""Qwen3-ASR model compatible with HuggingFace weights"""; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: +import torch; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, __init__
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: +import re; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunks: TIMESTAMP_BASE_TOKEN_ID = 50365 # <\|0.00\|>; def _convert_to_internal_request(; symbols: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix
  - `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0 (29 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; keywords observed in patches: config, moe, attention, processor, vision, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `mmangkad`.
- Diff scope read: `8` files, `+152/-5`; areas: model wrapper, MoE/router; keywords: flash, eagle, config, expert, kv, moe, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: def forward(; def forward(; symbols: forward, forward, get_layer, get_input_embeddings
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):; def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: def __init__(; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; keywords observed in patches: flash, eagle, config, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22458 - Fix NCCL AllGather hanging issue for Qwen3 Next MTP

- Link: https://github.com/sgl-project/sglang/pull/22458
- Status/date: `merged`, created 2026-04-09, merged 2026-04-10; author `ispobock`.
- Diff scope read: `2` files, `+38/-0`; areas: misc; keywords: attention, eagle, processor, spec, triton, flash, kv.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunks: import torch.nn.functional as F; def verify(; symbols: verify
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunks: import triton; def sample(; symbols: sample
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; keywords observed in patches: attention, eagle, processor, spec, triton, flash. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22664 - Qwen3next flashinfer allreduce auto enable

- Link: https://github.com/sgl-project/sglang/pull/22664
- Status/date: `merged`, created 2026-04-13, merged 2026-04-18; author `BBuf`.
- Diff scope read: `1` files, `+3/-1`; areas: misc; keywords: flash, kv, moe, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: flash, kv, moe, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22876 - Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac

- Link: https://github.com/sgl-project/sglang/pull/22876
- Status/date: `open`, created 2026-04-15; author `flyerming`.
- Diff scope read: `2` files, `+42/-0`; areas: tests/benchmarks; keywords: cache, cuda, scheduler, spec, test.
- Code diff details:
  - `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunks: def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, cuda, scheduler, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23075 - [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.

- Link: https://github.com/sgl-project/sglang/pull/23075
- Status/date: `open`, created 2026-04-17; author `flyerming`.
- Diff scope read: `3` files, `+51/-13`; areas: attention/backend; keywords: attention, cache, scheduler.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunks: def prepare_mixed(; def prepare_mixed(; symbols: prepare_mixed, prepare_mixed, prepare_mixed
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunks: import logging; def _forward_metadata(self, forward_batch: ForwardBatch):; symbols: _forward_metadata
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunks: def mix_with_running(self, running_batch: "ScheduleBatch"):; def filter_batch(; symbols: mix_with_running, filter_batch, merge_batch, merge_batch
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`; keywords observed in patches: attention, cache, scheduler. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23273 - [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)

- Link: https://github.com/sgl-project/sglang/pull/23273
- Status/date: `open`, created 2026-04-20; author `wenscarl`.
- Diff scope read: `2` files, `+54/-22`; areas: attention/backend, kernel; keywords: flash, attention, cuda, spec, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunks: Both SM90 and SM100+ use the same pool layout: [pool, HV, V, K] (K-last).; _flashinfer_chunk_gated_delta_rule = None; symbols: _get_flashinfer_gdn_kernels, _get_flashinfer_gdn_kernels, _get_flashinfer_gdn_kernels, FlashInferGDNKernel
  - `python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunks: def _handle_mamba_backend(self):; symbols: _handle_mamba_backend, _handle_linear_attn_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: flash, attention, cuda, spec, topk, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: `open`, created 2026-04-22; author `kawaruko`.
- Diff scope read: `2` files, `+284/-8`; areas: tests/benchmarks; keywords: attention, cache, cuda, spec, test.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: +"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: import logging; def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:; symbols: maybe_offload_to_cpu, maybe_offload_to_cpu, forward
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; keywords observed in patches: attention, cache, cuda, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
