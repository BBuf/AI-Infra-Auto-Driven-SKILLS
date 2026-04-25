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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3 Next` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

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

### File-level PR diff reading notes

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


### Gap and optimization follow-up

- Covered PRs: 55; open PRs: 15.
- Open PRs to keep tracking: [#10657](https://github.com/sgl-project/sglang/pull/10657), [#12892](https://github.com/sgl-project/sglang/pull/12892), [#13964](https://github.com/sgl-project/sglang/pull/13964), [#14502](https://github.com/sgl-project/sglang/pull/14502), [#16488](https://github.com/sgl-project/sglang/pull/16488), [#17981](https://github.com/sgl-project/sglang/pull/17981), [#17983](https://github.com/sgl-project/sglang/pull/17983), [#19812](https://github.com/sgl-project/sglang/pull/19812), [#20397](https://github.com/sgl-project/sglang/pull/20397), [#21684](https://github.com/sgl-project/sglang/pull/21684), [#21698](https://github.com/sgl-project/sglang/pull/21698), [#22876](https://github.com/sgl-project/sglang/pull/22876), [#23075](https://github.com/sgl-project/sglang/pull/23075), [#23273](https://github.com/sgl-project/sglang/pull/23273), [#23474](https://github.com/sgl-project/sglang/pull/23474)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
