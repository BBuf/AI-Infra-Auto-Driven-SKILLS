# SGLang Qwen3-Next / Qwen3-Coder-Next Optimization History

This history is based on SGLang `origin/main` snapshot `b3e6cf60a` on `2026-04-22`, sgl-cookbook snapshot `816bad5` on `2026-04-21`, official Qwen3-Next deployment docs, public optimization material, and direct source-diff inspection of every PR below. The fuller PR-by-PR dossier lives in `skills/model-optimization/sglang/sglang-qwen3-next-optimization/references/pr-history.md`.

Qwen3-Next is not a generic Qwen3 MoE lane. It combines hybrid Gated DeltaNet, Mamba/SSM state pools, RadixLinearAttention, MTP/NEXTN/EAGLE, FP8/NVFP4/ModelOpt loading, CPU offload, FlashInfer/CuTe/Gluon GDN kernels, AMD/NPU/Blackwell backend behavior, and mixed-chunk `extra_buffer` state correctness.

## Main Code Surfaces

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

## Merged / Current-Main PRs

### #10233: Initial Qwen3-Next Support

- Motivation: add the Qwen3-Next hybrid architecture and MTP draft architecture to SGLang.
- Implementation: introduced `Qwen3NextConfig`, `Qwen3NextForCausalLM`, `Qwen3NextForCausalLMMTP`, `HybridLayerType.linear_attention/mamba2`, Mamba pools, hybrid req/token pools, hybrid KV pools, and the hybrid linear-attention backend.
- Key code:

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

- Validation: PR reports GSM8K around `0.945` and MTP throughput roughly `180 -> 304` tok/s.

### #10322: Norm Type Fix

- Motivation: transformer-side norm config changes made the old conditional path wrong for Qwen3-Next.
- Implementation: standardized input/post/final/MTP pre-fc norms on `GemmaRMSNorm`.
- Key code:

```python
self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### #10379: Initial Ascend NPU Support

- Motivation: Qwen3-Next needs NPU-specific causal conv, GDN, token-to-KV pools, page size, and attention backend routing.
- Implementation: imports `sgl_kernel_npu` under `is_npu()`, routes `HybridLinearKVPool` through Ascend token pools, chooses `AscendAttnBackend`, and forces Ascend hybrid page size.
- Key code:

```python
full_attn_backend = AscendAttnBackend(self) if _is_npu else FlashAttentionBackend(self)
```

### #10392: MTP + DP Fixes

- Motivation: speculative decoding with DP exposed cuda graph and idle-batch failures.
- Implementation: draft config sets `num_nextn_predict_layers=1`, DP buffer length is fixed, idle `bs=0` is handled, and Mamba state sizing covers all state tensors.
- Key code:

```python
self.hf_config.num_nextn_predict_layers = 1
```

### #10466 and #10622: FP8, L2Norm, and DeepEP

- Motivation: fix GDN L2Norm accuracy and enable FP8 DeepEP Qwen3-Next.
- Implementation: passes `quant_config` into `Qwen3GatedDeltaNet`, fixes FLA L2Norm behavior, exposes MoE weights, handles empty TopK output, and builds `routed_experts_weights_of_layer`.
- Key code:

```python
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream)
```

```python
def get_moe_weights(self):
    return [x.data for name, x in self.experts.named_parameters() if name not in ["correction_bias"]]
```

- Validation: FP8 DeepEP PR reports TP4DP2 accuracy around `0.942` and TP8DP8 around `0.940`.

### #10912: PD Disaggregation for Hybrid State

- Motivation: Qwen3-Next PD cannot transfer KV cache only; it must also transfer Mamba/GDN state.
- Implementation: adds `extra_pool_indices` to transfer interfaces, exposes Mamba contiguous buffers, and passes Mamba rid/req mappings through prefill/decode connectors.
- Key code:

```python
def get_extra_pool_buf_infos(self):
    return self.mamba_pool.get_contiguous_buf_infos()
```

### #11487: KTransformers CPU/GPU Hybrid MoE

- Motivation: support CPU/GPU hybrid MoE inference with AMX/compressed-tensors examples.
- Implementation: adds WNA16 AMX MoE, AMX wrapper dispatch, CPU infer/offload flags, and AMX/Marlin expert combine paths.
- Key code:

```python
output = self.amx_wrapper.forward(x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream)
```

### #11969 and #16164: NPU Bugfixes and W8A8

- Motivation: fix Ascend decode kernels, fused TopK, DP-attention padding, and W8A8 module-name loading.
- Implementation: backend-selects causal conv, pads only under DP-attention, and threads `prefix` into Qwen3-Next GDN layers.
- Key code:

```python
self.linear_attn = Qwen3GatedDeltaNet(config, layer_id, quant_config, alt_stream, prefix)
```

### #12508: Fused GDN Gating

- Motivation: reduce split sigmoid/gating/unsqueeze overhead in GDN.
- Implementation: adds a Triton `fused_gdn_gating` kernel and calls it directly from the backend.
- Key code:

```python
g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
```

- Validation: verify path improves around `3.5us -> 1.4us`.

### #12525: CPU and AMX Path

- Motivation: Qwen3-Next CPU needed fused RMSNorm/GDN/conv1d/qkvzba kernels and TP odd-size padding.
- Implementation: adds `Qwen3NextRMSNormGated`, CPU causal conv, AMX conv-state layout, CPU fused-GDN routing, and disables CPU dual stream.
- Key code:

```python
class Qwen3NextRMSNormGated(CustomOp):
    def forward_cpu(self, hidden_states, gate=None):
        return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(...)
```

### #13081, #16863, #17613, #19220: PCG Evolution

- Motivation: GDN has many inputs and was initially too coarse for piecewise cuda graphs.
- Implementation: starts with `gdn_with_output`, then centralizes split-op registration, then moves projections/norm/out projection into PCG through `RadixLinearAttention`, and finally removes the legacy split op after fake FP8 support lands.
- Key code:

```python
if hasattr(layer.linear_attn, "attn"):
    self.attention_layers.append(layer.linear_attn.attn)
```

```python
@torch.library.register_fake("sgl_kernel::fp8_blockwise_scaled_mm")
def _fake_fp8_blockwise_scaled_mm(...):
    return mat_a.new_empty((M, N), dtype=out_dtype)
```

- Validation: `#17613` reports throughput around `2592 -> 2963` tok/s.

### #13708 and #14855: Small Correctness/Cleanup Fixes

- `#13708` removes forced `lm_head.float()` so BF16 remains BF16.
- `#14855` removes confusing GDN initialization leftovers and keeps the simpler `conv_dim` computation.

```python
self.conv_dim = self.key_dim * 2 + self.value_dim
```

### #14607: EAGLE3

- Motivation: support `lukeysong/qwen3-next-draft` EAGLE3.
- Implementation: adds `set_eagle3_layers_to_capture`, captures auxiliary hidden states, returns `(hidden_states, aux_hidden_states)`, and passes aux states to the logits processor.
- Key code:

```python
def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
    self.capture_aux_hidden_states = True
```

### #15631, #17981, #17983, #23273: GDN Kernel Work

- Motivation: improve Qwen3-Next GDN decode, prefill, and MTP verify on Hopper/Blackwell.
- Implementation:
  - `#15631`: CuTe DSL GDN decode controlled by `SGLANG_USE_CUTEDSL_GDN_DECODE`.
  - `#17981`: transposed SSM state `[B,H,V,K]` and CuTe DSL decode/MTP kernels.
  - `#17983`: Gluon prefill/cumsum/wy_fast kernels.
  - `#23273`: FlashInfer BF16-state MTP verify on SM100+ and speculative-safe FlashInfer default.
- Key code:

```python
USE_CUTEDSL_GDN_DECODE = os.environ.get("SGLANG_USE_CUTEDSL_GDN_DECODE", "0") == "1"
```

```python
from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
)
```

- Validation: reports include H200/B200/H20 CuTe speedups, Blackwell decode/MTP kernel speedups, and FlashInfer MTP accuracy close to Triton.

### #17373 and #17660: RadixLinearAttention

- Motivation: avoid passing a long list of GDN kwargs through model code.
- Implementation: introduces `RadixLinearAttention`, stores dimensions/weights/state metadata on the layer, and lets backend split by `layer.q_dim/k_dim/v_dim`.
- Key code:

```python
class RadixLinearAttention(nn.Module):
    def forward(self, forward_batch, mixed_qkv, a, b, **kwargs):
        return forward_batch.attn_backend.forward(layer=self, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)
```

### #17570: Embedding Uses Attention TP Group

- Motivation: DP-attention models need embedding to use the attention TP group.
- Implementation:

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    use_attn_tp_group=is_dp_attention_enabled(),
)
```

### #17627, #18224, #21313, #21496, #21662, #21698: Quantized Loader History

- Motivation: NVFP4/FP8/W8A8 fused projection loading repeatedly exposed packed-module mapping, missing scale tensor, and property-backed loader assignment issues.
- Implementation:
  - `#17627`: ModelOpt FP4 skips unquantized qkv projection quant config and missing `1.0` scales.
  - `#18224`: Qwen3-Coder-Next NVFP4 adds packed module mapping and KV scale remapping.
  - `#21313`: attempted `_weight_loader` fix, later reverted.
  - `#21496`: reverts `#21313`.
  - `#21662`: introduces safe `_override_weight_loader`.
  - `#21698`: extends NPU W8A8 loader override to scale/offset tensors and imports NPU fused qkvzba split.
- Key code:

```python
if name.endswith(".k_proj.k_scale"):
    name = name.replace(".k_proj.k_scale", ".attn.k_scale")
```

```python
for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale", "weight_offset"):
    param = getattr(module, attr_name, None)
```

### #17016 and #18355: AMD Path

- Motivation: AMD MI300/MI355 needs correct `v_head_dim`, MTP mask routing, and no unsupported dual stream.
- Implementation: AITER reads hybrid `v_head_dim` from the KV pool; `alt_stream` is CUDA-only.
- Key code:

```python
alt_stream = torch.cuda.Stream() if _is_cuda else None
```

### #18489 and #21019: Qwen3.5 Shared Hybrid Work

- Motivation: Qwen3.5 reused and extended Qwen3-Next GDN design; later PR split interleaved Qwen3-Next and contiguous Qwen3.5 fused projection kernels.
- Key code:

```python
if isinstance(config, Qwen3NextConfig | Qwen3_5Config | Qwen3_5MoeConfig):
    return config
```

### #18917, #19321, #19434: Projection and Norm/Gate Fusion

- Motivation: prefill split/reshape/cat, GDN projection, and norm/gate were hot paths.
- Implementation: `#18917` uses fused split in prefill, `#19321` fuses `qkvz_proj` and `ba_proj` through `MergedColumnParallelLinear`, and `#19434` adds `FusedRMSNormGated`.
- Key code:

```python
("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
("in_proj_qkvz.", "in_proj_z.", 3),
("in_proj_ba.", "in_proj_b.", 0),
("in_proj_ba.", "in_proj_a.", 1),
```

- Validation: `#19321` reports throughput around `15314.80 -> 15733.74` tok/s; `#19434` around `15314.80 -> 15959.30` tok/s.

### #19767 and #19812: MTP + EPLB

- Motivation: MTP draft passes should not pollute EPLB statistics or create expert location dispatch info with draft-local layer ids.
- Implementation: `Qwen2MoeSparseMoeBlock` gains `is_nextn`, draft blocks skip expert location dispatch, and MTP forward runs under `disable_this_region()`.
- Key code:

```python
expert_location_dispatch_info=(
    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
    if not self.is_nextn else None
)
```

### #22073, #22358, #22458, #22664: Adjacent Features and Stability

- `#22073`: Qwen3-ASR adjacent import/runtime surface, not a Qwen3-Next GDN optimization.
- `#22358`: DFLASH aux hidden-state capture for Qwen3-Next.
- `#22458`: broadcasts `predict`, `accept_index`, and `accept_length` across TP ranks to prevent Qwen3-Next MTP NCCL all-gather hangs.
- `#22664`: adds `"Qwen3NextForCausalLM"` to FlashInfer all-reduce auto-enable whitelist; reports requests/s `5.49 -> 9.41` and TTFT `456 -> 167ms`.

## Open PR Radar

### #10657: Early EAGLE3

Open but superseded by merged `#14607`. It captured aux hidden states with `layers_to_capture` and returned `(hidden_states, aux_hidden_states)`.

### #12892: Avoid SSM/Conv State Copy

- Motivation: reduce target-verify state update overhead.
- Implementation: adds `last_steps` to Mamba speculative state and lets kernels read accepted steps.
- Key code:

```python
mamba_caches.last_steps[state_indices_tensor] = accepted_indices
```

### #13964: GDN Decode Autotune

Adds Triton autotune, precomputes `neg_exp_A`, and increases `BV` to improve decode kernel time from about `143747ns` to `109069ns` on H200.

### #14502: PCG Optimization

Adds `causal_conv1d_gdn_with_output` so projection/out/gating can be captured while conv+GDN core remains eager. Reported 1024-token TTFT path: `99.17ms -> 67.83ms -> 48.21ms`.

### #16488: TBO

Adds Qwen3 hybrid layer operation strategies and `tbo_delta_stages=2`; reported H800 FP8 GSM8K around `0.936` with compute/comm overlap.

### #20397: NPU MTP

Adds Ascend FIA handling for `qk_head_dim == 256`, NPU conv state with draft-step space, MTP graph metadata, and NPU SSM/conv rollback after target verify.

```python
if is_npu():
    move_intermediate_cache_dynamic_h_block_v1(...)
    conv_state_rollback(...)
    return
```

### #21684: Allocator Clone

Fixes allocator aliasing by returning `select_index.clone()` from generic and Mamba memory allocators.

### #22876 and #23075: Mixed Chunk + `extra_buffer`

- Motivation: `--enable-mixed-chunk` plus `--mamba-scheduler-strategy extra_buffer` dropped GSM8K from `0.938` to `0.876`.
- `#22876`: adds a guard.
- `#23075`: fixes the root cause by slicing `query_start_loc` and `mamba_cache_indices` to prefill-only metadata in mixed mode.

```python
if forward_batch.forward_mode.is_mixed():
    query_start_loc_for_track = query_start_loc[: num_prefills + 1]
    mamba_cache_indices_for_track = mamba_cache_indices[:num_prefills]
```

### #23474: CPU Offload for Hybrid Linear Attention

- Motivation: CPU offload crashed on tied parameters and then produced garbage because cached `conv1d.weight.view` attributes still pointed at old GPU storage.
- Implementation: `state_dict(keep_vars=True)` id-based device tensor cache, storage-alias scan before pinning, and temporary `as_strided` rebinding during forward.
- Key code:

```python
for k, v in module.state_dict(keep_vars=True).items():
    dev = src_to_dev.get(id(v))
```

```python
sub.__dict__[attr_name] = dev_tensor.as_strided(size, stride, offset)
```

## Docs / Cookbook Evidence

- Official Qwen3-Next docs preserve `--max-mamba-cache-size`, `--mamba-ssm-dtype`, `--mamba-full-memory-ratio`, `--mamba-scheduler-strategy extra_buffer`, `--page-size 64`, NEXTN/EAGLE, `--tool-call-parser qwen`, and `--reasoning-parser qwen3`.
- sgl-cookbook `#100` and `#123` cover AMD MI300X/MI325X/MI355X deployment environments.
- sgl-cookbook `#143` covers Qwen3-Coder-Next and is relevant because it shares the Qwen3-Next architecture/runtime.

## Next Optimization Work

1. Keep MTP state-copy, PCG, and TBO as separate benchmark lanes.
2. Validate Blackwell GDN prefill, decode, MTP verify, and fallback paths separately.
3. Require tied-parameter and cached-view tests for CPU offload changes.
4. Add NPU W8A8 tests for fused projection scale/offset loading.
5. Treat mixed chunk + `extra_buffer` as a fixed accuracy regression case for hybrid Qwen3-Next/Qwen3.5 models.
