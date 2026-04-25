# SGLang Qwen3.5 Support And Optimization History

This document is a diff-reviewed model history, not a PR-number checklist. Each PR below was filled after opening the PR metadata and source diff. The canonical skill-side dossier is `skills/model-optimization/sglang/sglang-qwen35-optimization/references/pr-history.md`; this file keeps the model-history view in sync.

Qwen3.5 is not a single-kernel optimization lane. Its SGLang history spans hybrid GDN linear attention, full attention, MoE shared experts, MTP/spec-v2, PP/EP/EPLB, VLM/encoder disaggregation, NIXL PD, Mamba state management, and FP8/NVFP4/MXFP4/NPU/ROCm deployment.

## Code Surfaces

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

## Diff-Reviewed PR Timeline

### #18489 Initial Qwen3.5 support

- Motivation: add the new Qwen3.5 dense/MoE/VL family, including `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`.
- Implementation: adds the model/config/MTP files, model registration, multimodal processor hooks, server/speculative decode wiring, and the first hybrid GDN/full-attention/MoE/DeepStack loader path.
- Key snippet:

```python
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...

class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    ...
```

- Validation meaning: all later GDN, MTP, VLM, PP and quant changes are compatibility layers on top of this baseline.

### #18538 MTP refactor

- Motivation: replace the early custom predictor body with a structure that shares the normal Qwen3.5 causal LM path.
- Implementation: nests `Qwen3_5ForCausalLM`, adds `fc`, `enorm`, and `hnorm`, then concatenates input embeddings and target hidden states before the draft model.
- Key snippet:

```python
hidden_states = self.enorm(input_embeds)
target_hidden_states = self.hnorm(target_hidden_states)
hidden_states = torch.cat([hidden_states, target_hidden_states], dim=-1)
hidden_states, _ = self.fc(hidden_states)
```

### #18544 NPU/ModelSlim/EPLB follow-up

- Motivation: the first model path still had CUDA-only assumptions and prefix mismatches.
- Implementation: avoids CUDA JIT/Triton imports on NPU, normalizes ModelSlim prefixes, fixes `.linear_attn` MLP prefixes, and exposes expert-location config for EPLB.
- Key snippet:

```python
if not is_cpu() and not is_npu():
    ...
```

### #18926 Block-wise FP8 and prefix alignment

- Motivation: Qwen3.5 FP8 checkpoints need block scale loading for merged column layers, and MTP quantization needed an `mtp` prefix.
- Implementation: adds `_load_merged_block_scale()`, routes `BlockQuantScaleParameter`, restricts an old Mistral prefix hack, and changes the Qwen3.5 MTP prefix.
- Key snippet:

```python
elif isinstance(param, BlockQuantScaleParameter):
    self._load_merged_block_scale(param, loaded_weight)
    return
```

```python
prefix=add_prefix("mtp", prefix)
```

### #18937 NVFP4 checkpoint support

- Motivation: ModelOpt FP4/NVFP4 cannot quantize every hybrid Qwen3.5 module safely.
- Implementation: disables `modelopt_fp4` quantization in linear-attention/full-attention/MTP areas and tightens expert-name checks.
- Key snippet:

```python
linear_attn_quant_config = (
    None if quant_config and quant_config.get_name() == "modelopt_fp4" else quant_config
)
```

### #19070 Dense TP>1 precision fix

- Motivation: dense Qwen3.5 incorrectly inherited a MoE-style all-reduce fusion path.
- Implementation: separates dense and MoE MLP calls, passes `should_allreduce_fusion` only where valid, and marks dense hidden states for delayed communicator postprocessing.
- Key snippet:

```python
hidden_states = self.mlp(hidden_states, should_allreduce_fusion=should_allreduce_fusion)
hidden_states._sglang_needs_allreduce_fusion = True
```

### #19220 PCG fix

- Motivation: a custom GDN PCG wrapper conflicted with the regular Qwen3.5 GDN path and compile fake registrations.
- Implementation: removes the wrapper, calls attention directly, adds fake registration for FP8 blockwise scaled MM, and restores no-grad forward.
- Key snippet:

```python
hidden_states = self.attn(
    positions=positions,
    hidden_states=hidden_states,
    forward_batch=forward_batch,
)
```

### #19391 MTP spec-v2 and NVFP4 tests

- Motivation: MTP-v2 needed multimodal input embeddings, and NVFP4 needed real Qwen3.5 accuracy/acceptance tests.
- Implementation: passes `mm_input_embeds` into draft extend, handles draft extend v2 in MTP, changes radix-cache/spec error behavior, and adds `nvidia/Qwen3.5-397B-A17B-NVFP4` tests.
- Key snippet:

```python
if mm_input_embeds is not None:
    forward_batch.mm_input_embeds = mm_input_embeds
```

```python
and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

### #19411 Last-layer communicator fix

- Motivation: Qwen3.5-27B had a repeat/output issue tied to layer communicator state.
- Implementation: marks the last decoder layer when constructing communicator state.
- Key snippet:

```python
is_last_layer=(layer_id == config.num_hidden_layers - 1)
```

### #19670 Pipeline parallel support

- Motivation: Qwen3.5 needed proper PP layer placement and first/last-rank embed/head handling.
- Implementation: adds missing-layer placeholders, `start_layer`/`end_layer`, PP indices, embed/head helpers, and PP tests.
- Key snippet:

```python
embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
head = self.lm_head.weight if self.pp_group.is_last_rank else None
```

### #19767 MTP + EPLB fixes

- Motivation: nextn/MTP draft layers should not participate in normal target-model EPLB dispatch or expert-distribution stats.
- Implementation: adds `is_nextn`, disables expert-location dispatch for draft layers, and wraps MTP forward in a disabled expert recorder region.
- Key snippet:

```python
if self.is_nextn:
    self.expert_location_dispatch_info = None
```

### #19889 TRTLLM/FlashInfer all-reduce fusion

- Motivation: reduce communication overhead in Qwen3.5 MoE while preserving Gemma-style norm semantics.
- Implementation: adds a shared all-reduce fusion helper, teaches Qwen2 MoE to accept `should_allreduce_fusion`, and marks Qwen3.5 architectures eligible in server args.
- Key snippet:

```python
return _forward_with_allreduce_fusion(
    hidden_states,
    residual,
    self.weight + 1.0,
    self.variance_epsilon,
)
```

### #19961 Keep GDN `A_log` as FP32

- Motivation: GDN recurrent dynamics are sensitive to the state parameter dtype.
- Implementation: initializes `A_log` explicitly as `torch.float32`.
- Key snippet:

```python
self.A_log = nn.Parameter(
    torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
)
```

### #20386 Replace `einops.rearrange`

- Motivation: `einops.rearrange` was expensive inside a hot GDN output path.
- Implementation: uses native flatten; the PR body reports roughly `12.67us -> 4.74us` over 720 H100 calls.
- Key snippet:

```python
core_attn_out = core_attn_out.flatten(-2)
```

### #20736 AMD shared-expert fusion

- Motivation: fuse Qwen3.5 MoE shared expert into the routed expert tensor on AMD/AITER when shape-compatible.
- Implementation: adds `num_fused_shared_experts`, appends shared expert id/weight to `StandardTopKOutput`, and remaps `mlp.shared_expert.*` weights into `mlp.experts.{num_experts_base}.*`.
- Key snippet:

```python
fused_topk_ids = torch.cat([topk_output.topk_ids, shared_ids], dim=-1)
fused_topk_weights = torch.cat([topk_output.topk_weights, shared_weights], dim=-1)
```

```python
name = name.replace("mlp.shared_expert.", f"mlp.experts.{num_experts_base}.")
```

### #20864 Remove SpecV2 H2D/D2H overhead

- Motivation: Qwen3.5 SpecV2 verify had avoidable Python-list/CUDA-scalar host-device overhead.
- Implementation: builds Mamba track indices via `torch.stack`, and creates text-only mrope deltas directly on device.
- Key snippet:

```python
batch.mamba_track_indices = torch.stack([...]).to(torch.int64)
```

```python
mrope_delta_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
```

### #21019 Triton fusion for GDN projection

- Motivation: Qwen3.5 checkpoint layout stores `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a` separately, unlike Qwen3-Next fused/interleaved layout.
- Implementation: adds `gdn_fused_proj.py`, replaces four projection modules with `in_proj_qkvz` and `in_proj_ba`, implements packed/split checkpoint loading, and maps old names into fused params.
- Key snippet:

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

### #21070 PP layer splitting fix

- Motivation: PP could instantiate/load layers outside the local pipeline stage.
- Implementation: passes PP rank/size into `make_layers` and skips absent parameters in fused expert weight loading.
- Key snippet:

```python
make_layers(..., pp_rank=self.pp_group.rank_in_group, pp_size=self.pp_group.world_size)
```

### #21234 AMD MXFP4 mapping

- Motivation: gfx950 MXFP4 Qwen3.5-397B needs packed mappings for fused GDN and MoE modules.
- Implementation: declares model-local packed mappings for `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, and `in_proj_ba`; VL subclasses reuse them.
- Key snippet:

```python
"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
"in_proj_ba": ["in_proj_b", "in_proj_a"],
```

### #21347 PP tied embedding loading

- Motivation: tied embeddings need `lm_head.weight` loaded on the last PP rank even if the checkpoint name is `model.embed_tokens.weight`.
- Implementation: redirects tied embedding load to the LM head.
- Key snippet:

```python
if self.config.tie_word_embeddings and name == "model.embed_tokens.weight":
    name = "lm_head.weight"
```

### #21448 MoE loading and Mamba cache PP sharding

- Motivation: PP Qwen3.5 should only allocate Mamba state and load weights for local layers.
- Implementation: filters Mamba layer ids by `[start_layer, end_layer)` and skips out-of-stage layer weights.
- Key snippet:

```python
mamba_layer_ids = [
    layer_id for layer_id in cache_params.layers
    if start_layer <= layer_id < end_layer
]
```

### #21487 GB300 nightly tests

- Motivation: add GB300/4x B200 NVL4 nightly coverage for Qwen3.5 FP8/NVFP4 and related frontier models.
- Implementation: adds Qwen3.5 FP8/NVFP4 tests with TP4, MTP/spec-v2, `trtllm_mha`, FlashInfer all-reduce fusion and Qwen parsers.
- Key snippet:

```python
QWEN35_FP8_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
QWEN35_NVFP4_MODEL_PATH = "nvidia/Qwen3.5-397B-A17B-NVFP4"
```

```python
env={"SGLANG_ENABLE_SPEC_V2": "1"}
```

### #21669 AMD FP8 nightly performance

- Motivation: track Qwen3.5-397B FP8 performance on AMD, not only accuracy.
- Implementation: adds MI30x/MI35x perf tests with `SGLANG_USE_AITER=1`, TP=8, Triton attention and multithread load.
- Key snippet:

```python
QWEN35_FP8_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
```

### #21692 NPU quantization fix

- Motivation: after GDN projection fusion, NPU/ModelSlim quantization needed fused-name mappings.
- Implementation: enables Qwen3.5 packed mappings on NPU, refactors ModelSlim linear scheme lookup, and checks both local and global packed maps when skipping layers.
- Key snippet:

```python
if _is_gfx95 or _is_npu:
    packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
```

### #21849 Encoder disaggregation for Qwen3.5

- Motivation: Qwen3.5 multimodal runtime worked, but EPD startup rejected Qwen3.5 architectures.
- Implementation: adds Qwen3.5 dense/MoE conditional generation to the allowlist, extends video timestamp handling, and adds EPD image/video tests.
- Key snippet:

```python
"Qwen3_5ForConditionalGeneration",
"Qwen3_5MoeForConditionalGeneration",
```

### #22145 NIXL heterogeneous TP KV transfer

- Motivation: NIXL + heterogeneous TP could hang due to `pp_rank` notification collisions and wrong GQA head distribution.
- Implementation: computes head distribution from `total_kv_head_num`, handles GQA replication, and uses `engine_rank` for notifications.
- Key snippet:

```python
total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
```

```python
notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.engine_rank}"
```

### #22240 NIXL Mamba state slice transfer

- Motivation: NIXL lacked hetero-TP Mamba state slicing for hybrid Mamba models such as Qwen3.5.
- Implementation: registers destination state metadata and adds `_send_mamba_state_slice()` for conv/temporal state transfer.
- Key snippet:

```python
dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)
```

### #22312 Non-contiguous B/A GDN fix

- Motivation: `mixed_ba.split()` can return non-contiguous B/A views, breaking kernels that assumed contiguous layout after #21019.
- Implementation: passes explicit `stride_a`/`stride_b`, advances `p_a` by token-axis stride, and adds CUDA regression tests.
- Key snippet:

```python
blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
```

### #22358 DFLASH support

- Motivation: DFLASH needed aux hidden-state capture in Qwen3.5 and sibling backends.
- Implementation: uses `prepare_attn_and_capture_last_layer_outputs`, tracks captured layers, and returns aux hidden states when requested.
- Key snippet:

```python
captured_last_layer_outputs=captured_last_layer_outputs
```

```python
def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
    ...
```

### #22431 Processor-output video fix

- Motivation: Qwen3.5 `processor_output` video path returned one value where the model processor expected `(video, metadata)`.
- Implementation: returns `(vr, None)` for preprocessed video data.
- Key snippet:

```python
if not is_video_obj:
    return vr, None
```

### #22493 MambaPool CPU offload during retraction

- Motivation: request retraction saved attention KV but dropped Qwen3.5 Mamba conv/temporal state.
- Implementation: adds CPU copy/load for MambaPool and HybridLinearKVPool, passes `mamba_pool_idx`, and logs gained Mamba slots.
- Key snippet:

```python
self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
    token_indices, mamba_indices=self.mamba_pool_idx
)
```

### #22908 AMD radix-cache/spec conflict

- Motivation: ROCm cannot use the CUDA `extra_buffer` workaround for Qwen3.5 speculative decoding with radix cache.
- Implementation: on HIP, automatically disables radix cache; on CUDA/other platforms, keeps the explicit error.
- Key snippet:

```python
if is_hip():
    self.disable_radix_cache = True
else:
    raise ValueError(...)
```

### #22913 Split B200 FP4 tests

- Motivation: one Qwen3.5 NVFP4 B200 test file launched multiple 234GB servers and timed out on slower nodes.
- Implementation: splits Triton and MTP-v2 tests, removes v1 MTP, and increases the B200 suite partition count.
- Key snippet:

```yaml
part: [0, 1, 2, 3, 4, 5]
```

### #22948 MXFP4 shared-expert fusion guard

- Motivation: shared-expert fusion broke MXFP4 when the shared expert was excluded from quantization and remained BF16/FP32.
- Implementation: disables fusion if `quant_config.exclude_layers` contains shared-expert modules, excluding `shared_expert_gate` and MTP paths.
- Key snippet:

```python
if any(
    "shared_expert" in layer
    and "shared_expert_gate" not in layer
    and not layer.startswith("mtp.")
    for layer in exclude_layers
):
    return False
```

### #23034 Qwen3.6 docs with Qwen3.5 runtime rules

- Motivation: Qwen3.6 docs also encode shared Qwen3.5 MTP/Mamba deployment behavior.
- Implementation: when speculative/MTP is enabled, the UI disables Mamba V1 and defaults to V2/`extra_buffer`.
- Key snippet:

```jsx
if (mtpEnabled) {
  return [
    { id: 'v1', label: 'V1', default: false, disabled: true },
    { id: 'v2', label: 'V2', default: true },
  ];
}
```

### #23467 FP8 ignored-layer dot-boundary matching

- Motivation: substring matching made Qwen3.5 `in_proj_a` collide with `in_proj_ba` and Qwen3.6 `mlp.gate` collide with `gate_up_proj`.
- Implementation: adds exact/prefix/dot-boundary matching and fused-shard fallback mappings.
- Key snippet:

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

### #23474 Hybrid linear-attention CPU offload open radar

- Status: open when reviewed; kept separate from merged history.
- Motivation: CPU offload can break tied/view aliases in fused/view-heavy hybrid linear-attention models.
- Implementation: records view aliases with `state_dict(keep_vars=True)`, shares device tensors for tied params, and rebuilds views with `as_strided`.
- Key snippet:

```python
view_aliases[name] = (src_name, tensor.size(), tensor.stride(), tensor.storage_offset())
```

## Docs And Public Evidence

- sgl-cookbook `#164/#168/#169/#177/#179/#180/#207/#214/#230/#237` cover initial Qwen3.5, FP8/NVFP4, B200, H200, AMD, more variants, B200 all-reduce fusion, H200 MTP, FP4/NVFP4 generator updates, and FP8 KV cautions.
- Official SGLang Qwen3.5 docs cover hybrid GDN/full attention, shared experts, DeepStack Vision/Conv3d, AMD `--attention-backend triton`, `SGLANG_USE_AITER=1`, `--reasoning-parser qwen3`, and `--tool-call-parser qwen3_coder`.
- AMD's Qwen3.5 day-0 article confirms the ROCm path: GDN via Triton, shared-expert MoE via hipBLASLt/AITER, and multimodal kernels through MIOpen/PyTorch.

## Maintenance Rules

- Do not add a PR here unless its source diff was opened.
- Every PR must include motivation, implementation idea, code snippet, and validation meaning.
- Keep merged history separate from open radar.
- Regression matrix: dense/MoE, text/VLM, BF16/FP8/NVFP4/MXFP4, CUDA/ROCm/NPU, TP/PP/EP, MTP spec-v1/v2, PD/NIXL, and retraction.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen3.5` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
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

### File-level PR diff reading notes

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

### PR #18538 - [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation

- Link: https://github.com/sgl-project/sglang/pull/18538
- Status/date: `merged`, created 2026-02-10, merged 2026-02-12; author `zju-stu-lizheng`.
- Diff scope read: `2` files, `+62/-118`; areas: model wrapper; keywords: config, moe, quant, attention, expert, processor, spec, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: from sglang.srt.layers.layernorm import GemmaRMSNorm; def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, moe, quant, attention, expert, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18544 - [Ascend]Support qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/18544
- Status/date: `merged`, created 2026-02-10, merged 2026-02-12; author `chenxu214`.
- Diff scope read: `3` files, `+23/-4`; areas: model wrapper, attention/backend, quantization; keywords: attention, quant, cache, config, cuda, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: # Distributed; def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +9/-0 (9 lines); hunks: def is_layer_skipped(; symbols: is_layer_skipped
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-2 (4 lines); hunks: from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu; def __init__(self, model_runner: ModelRunner):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; keywords observed in patches: attention, quant, cache, config, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18926 - feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation

- Link: https://github.com/sgl-project/sglang/pull/18926
- Status/date: `merged`, created 2026-02-17, merged 2026-02-18; author `zju-stu-lizheng`.
- Diff scope read: `4` files, `+57/-12`; areas: model wrapper, quantization; keywords: config, quant, kv, attention, awq, expert, fp8, test, vision.
- Code diff details:
  - `python/sglang/srt/layers/linear.py` modified +48/-0 (48 lines); hunks: def _load_fused_module_from_checkpoint(; def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_merged_block_scale, weight_loader_v2, weight_loader_v2
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-2 (7 lines); hunks: def from_config(cls, config: Dict[str, Any]) -> Fp8Config:; symbols: from_config
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: def __init__(; def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-4 (7 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; keywords observed in patches: config, quant, kv, attention, awq, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18937 - [Qwen3.5] Enable nvfp4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/18937
- Status/date: `merged`, created 2026-02-17, merged 2026-02-19; author `hlu1`.
- Diff scope read: `3` files, `+26/-8`; areas: model wrapper; keywords: config, fp4, quant, expert, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +3/-1 (4 lines); hunks: def get_rope(; symbols: get_rope
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; keywords observed in patches: config, fp4, quant, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19070 - fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

- Link: https://github.com/sgl-project/sglang/pull/19070
- Status/date: `merged`, created 2026-02-20, merged 2026-02-25; author `zju-stu-lizheng`.
- Diff scope read: `1` files, `+32/-6`; areas: model wrapper; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: def forward(; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #19391 - [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4

- Link: https://github.com/sgl-project/sglang/pull/19391
- Status/date: `merged`, created 2026-02-26, merged 2026-03-04; author `hlu1`.
- Diff scope read: `8` files, `+252/-16`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: scheduler, spec, cache, eagle, test, attention, config, cuda, fp4, kv.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_models.py` added +240/-0 (240 lines); hunks: +import unittest; symbols: TestQwen35FP4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +5/-4 (9 lines); hunks: def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
  - `python/sglang/srt/disaggregation/decode.py` modified +0/-5 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +0/-5 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +4/-0 (4 lines); hunks: def _draft_extend_for_prefill(; def _draft_extend_for_prefill(; symbols: _draft_extend_for_prefill, _draft_extend_for_prefill, forward_batch_generation
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py`; keywords observed in patches: scheduler, spec, cache, eagle, test, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_qwen35_models.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/disaggregation/decode.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19411 - [Qwen3.5] Qwen3.5-27B inference repeat bug fix

- Link: https://github.com/sgl-project/sglang/pull/19411
- Status/date: `merged`, created 2026-02-26, merged 2026-02-26; author `AlfredYyong`.
- Diff scope read: `1` files, `+2/-0`; areas: model wrapper; keywords: attention, config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: def __init__(; def __init__(; symbols: __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19670 - [Qwen3.5] Support Qwen3.5 Pipeline Parallelism

- Link: https://github.com/sgl-project/sglang/pull/19670
- Status/date: `merged`, created 2026-03-02, merged 2026-03-07; author `yuan-luo`.
- Diff scope read: `2` files, `+114/-13`; areas: model wrapper, tests/benchmarks; keywords: attention, cache, config, cuda, expert, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: ); from sglang.srt.layers.radix_attention import RadixAttention; symbols: __init__, get_layer, get_layer, get_input_embeddings
  - `test/registered/distributed/test_pp_single_node.py` modified +54/-0 (54 lines); hunks: def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwen35PPAccuracy, setUpClass, run_gsm8k_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; keywords observed in patches: attention, cache, config, cuda, expert, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #19889 - Use TRTLLM allreduce fusion for Qwen 3.5

- Link: https://github.com/sgl-project/sglang/pull/19889
- Status/date: `merged`, created 2026-03-04, merged 2026-03-18; author `b8zhong`.
- Diff scope read: `4` files, `+88/-52`; areas: model wrapper, MoE/router; keywords: moe, flash, attention, fp4, processor, spec, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: import torch_npu; def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: RowParallelLinear,; def forward(; symbols: forward, forward
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: moe, flash, attention, fp4, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19961 - fix: change qwen 3.5 linear attention a_log to fp32

- Link: https://github.com/sgl-project/sglang/pull/19961
- Status/date: `merged`, created 2026-03-05, merged 2026-03-18; author `shiyu7`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20386 - perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

- Link: https://github.com/sgl-project/sglang/pull/20386
- Status/date: `merged`, created 2026-03-11, merged 2026-03-12; author `vedantjh2`.
- Diff scope read: `1` files, `+1/-2`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: import torch; def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20736 - [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8

- Link: https://github.com/sgl-project/sglang/pull/20736
- Status/date: `merged`, created 2026-03-17, merged 2026-04-15; author `zhentaocc`.
- Diff scope read: `2` files, `+218/-8`; areas: model wrapper, MoE/router; keywords: config, cuda, expert, moe, deepep, fp8, quant, router, topk, triton.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +108/-5 (113 lines); hunks: ); from sglang.srt.utils import (; symbols: can_fuse_shared_expert, Qwen2MoeMLP, __init__, __init__
  - `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: LazyValue,; _is_npu = is_npu(); symbols: __init__, __init__, __init__, _get_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, cuda, expert, moe, deepep, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20864 - [Perf]Remove H2D for Qwen3.5 SpecV2

- Link: https://github.com/sgl-project/sglang/pull/20864
- Status/date: `merged`, created 2026-03-18, merged 2026-03-31; author `Chen-0210`.
- Diff scope read: `2` files, `+17/-13`; areas: scheduler/runtime; keywords: spec, cache, eagle.
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: def prepare_for_v2_verify(; symbols: prepare_for_v2_verify
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; keywords observed in patches: spec, cache, eagle. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #21070 - [Qwen3.5] Fix broken pipeline parallelism layer splitting

- Link: https://github.com/sgl-project/sglang/pull/21070
- Status/date: `merged`, created 2026-03-21, merged 2026-03-21; author `alisonshao`.
- Diff scope read: `2` files, `+15/-20`; areas: model wrapper, tests/benchmarks; keywords: config, expert, moe, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: ); def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights, load_fused_expert_weights
  - `test/registered/distributed/test_pp_single_node.py` modified +7/-5 (12 lines); hunks: def setUpClass(cls):; def run_gsm8k_test(self, pp_size):; symbols: setUpClass, run_gsm8k_test, run_gsm8k_test, run_gsm8k_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; keywords observed in patches: config, expert, moe, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `test/registered/distributed/test_pp_single_node.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21234 - [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model

- Link: https://github.com/sgl-project/sglang/pull/21234
- Status/date: `merged`, created 2026-03-23, merged 2026-03-30; author `hubertlu-tw`.
- Diff scope read: `1` files, `+18/-0`; areas: model wrapper; keywords: config, cuda, expert, kv, moe, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: cpu_has_amx_support,; _is_cuda = is_cuda(); symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, cuda, expert, kv, moe, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21347 - [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model

- Link: https://github.com/sgl-project/sglang/pull/21347
- Status/date: `merged`, created 2026-03-24, merged 2026-04-01; author `edwingao28`.
- Diff scope read: `1` files, `+22/-0`; areas: model wrapper; keywords: config, expert.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`; keywords observed in patches: config, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21448 - [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode

- Link: https://github.com/sgl-project/sglang/pull/21448
- Status/date: `merged`, created 2026-03-26, merged 2026-03-30; author `sufeng-buaa`.
- Diff scope read: `6` files, `+78/-8`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: cache, spec, attention, config, kv, mla, cuda, expert, lora, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: from sglang.srt.layers.radix_attention import RadixAttention; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_fused_expert_weights, load_weights, load_fused_expert_weights
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +17/-0 (17 lines); hunks: def _init_pools(self: ModelRunner):; def _init_pools(self: ModelRunner):; symbols: _init_pools, _init_pools, _init_pools
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +11/-5 (16 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +13/-0 (13 lines); hunks: def separate_deepstack_embeds(self, embedding):; symbols: separate_deepstack_embeds, start_layer, end_layer, pad_input_ids
  - `python/sglang/srt/disaggregation/decode.py` modified +4/-2 (6 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: cache, spec, attention, config, kv, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21487 - feat(ci): add GB300 nightly benchmark test suites

- Link: https://github.com/sgl-project/sglang/pull/21487
- Status/date: `merged`, created 2026-03-26, merged 2026-03-29; author `Kangyan-Zhou`.
- Diff scope read: `11` files, `+874/-4`; areas: quantization, scheduler/runtime, tests/benchmarks; keywords: test, attention, cuda, eagle, spec, topk, flash, cache, fp4, kv.
- Code diff details:
  - `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: def _run_simple_eval(; def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: +import unittest; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: +import unittest; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: +import unittest; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4
  - `test/registered/gb300/test_qwen35_fp8.py` added +75/-0 (75 lines); hunks: +import unittest; symbols: TestQwen35Fp8, test_qwen35_fp8
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`; keywords observed in patches: test, attention, cuda, eagle, spec, topk. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21669 - [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21669
- Status/date: `merged`, created 2026-03-30, merged 2026-04-07; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+408/-8`; areas: quantization, tests/benchmarks; keywords: test, attention, config, fp8, triton, cache, benchmark, moe.
- Code diff details:
  - `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: +"""Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8
  - `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: +"""MI35x Nightly performance benchmark for Qwen3.5-397B-A17B FP8.; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf
  - `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: import os; def setUpClass(cls):; symbols: setUpClass, setUpClass, tearDownClass, test_lm_eval
  - `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: import os; def setUpClass(cls):; symbols: setUpClass, test_lm_eval, test_lm_eval
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +26/-2 (28 lines); hunks: jobs:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`; keywords observed in patches: test, attention, config, fp8, triton, cache. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21692 - [Bugfix] [NPU] Qwen3.5 with quantization fix

- Link: https://github.com/sgl-project/sglang/pull/21692
- Status/date: `merged`, created 2026-03-30, merged 2026-04-08; author `OrangeRedeng`.
- Diff scope read: `3` files, `+29/-42`; areas: model wrapper, quantization; keywords: config, moe, quant, vision, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +25/-39 (64 lines); hunks: FusedMoEMethodBase,; def get_quant_method(; symbols: get_quant_method, get_quant_method, _get_scheme_from_parts, get_linear_scheme
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: def forward(; def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration
  - `python/sglang/srt/model_loader/loader.py` modified +1/-0 (1 lines); hunks: def _get_quantization_config(; symbols: _get_quantization_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py`; keywords observed in patches: config, moe, quant, vision, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/model_loader/loader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- Link: https://github.com/sgl-project/sglang/pull/21849
- Status/date: `merged`, created 2026-04-01, merged 2026-04-06; author `Ratish1`.
- Diff scope read: `4` files, `+190/-3`; areas: multimodal/processor, tests/benchmarks; keywords: moe, processor, config, cuda, scheduler, test.
- Code diff details:
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: # Omni model for local testing; override via env var EPD_OMNI_MODEL; def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation
- Optimization/support interpretation: The concrete diff surface is `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: moe, processor, config, cuda, scheduler, test. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22145 - [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)

- Link: https://github.com/sgl-project/sglang/pull/22145
- Status/date: `merged`, created 2026-04-05, merged 2026-04-07; author `YAMY1234`.
- Diff scope read: `1` files, `+20/-8`; areas: misc; keywords: cache, config, kv, mla.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: def send_kvcache_slice(; def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request, add_transfer_request
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/disaggregation/nixl/conn.py`; keywords observed in patches: cache, config, kv, mla. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/disaggregation/nixl/conn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22240 - [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)

- Link: https://github.com/sgl-project/sglang/pull/22240
- Status/date: `merged`, created 2026-04-07, merged 2026-04-07; author `YAMY1234`.
- Diff scope read: `1` files, `+143/-2`; areas: misc; keywords: kv, spec.
- Code diff details:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: class KVArgsRegisterInfo:; def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo:, from_zmq, from_zmq, from_zmq
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/disaggregation/nixl/conn.py`; keywords observed in patches: kv, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/disaggregation/nixl/conn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22312 - Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

- Link: https://github.com/sgl-project/sglang/pull/22312
- Status/date: `merged`, created 2026-04-08, merged 2026-04-10; author `cs-cat`.
- Diff scope read: `3` files, `+272/-8`; areas: attention/backend, tests/benchmarks; keywords: attention, triton, cache, config, cuda, test.
- Code diff details:
  - `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: +"""; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: def fused_sigmoid_gating_delta_rule_update_kernel(; def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: def fused_gdn_gating_kernel(; def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating_kernel, fused_gdn_gating, fused_gdn_gating
- Optimization/support interpretation: The concrete diff surface is `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`; keywords observed in patches: attention, triton, cache, config, cuda, test. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/attention/test_gdn_noncontiguous_stride.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- Link: https://github.com/sgl-project/sglang/pull/22431
- Status/date: `merged`, created 2026-04-09, merged 2026-04-18; author `lkhl`.
- Diff scope read: `1` files, `+1/-1`; areas: multimodal/processor; keywords: processor.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: async def preprocess_video(; symbols: preprocess_video
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22493 - Add MambaPool kvcache offloading during retraction

- Link: https://github.com/sgl-project/sglang/pull/22493
- Status/date: `merged`, created 2026-04-10, merged 2026-04-22; author `hlu1`.
- Diff scope read: `5` files, `+193/-16`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, test, attention, cuda, mla, scheduler, triton.
- Code diff details:
  - `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos
  - `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: def free(self, free_index: torch.Tensor):; def clear(self):; symbols: free, get_cpu_copy, get_cpu_copy, load_cpu_copy
  - `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:; def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch; symbols: update_running_batch, update_running_batch
  - `python/sglang/srt/managers/schedule_batch.py` modified +8/-2 (10 lines); hunks: def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):; symbols: offload_kv_cache, load_kv_cache, log_time_stats
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`; keywords observed in patches: cache, kv, test, attention, cuda, mla. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22908 - [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.

- Link: https://github.com/sgl-project/sglang/pull/22908
- Status/date: `merged`, created 2026-04-15, merged 2026-04-21; author `ChangLiu0709`.
- Diff scope read: `1` files, `+14/-4`; areas: misc; keywords: cache, scheduler, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: cache, scheduler, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22913 - test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6

- Link: https://github.com/sgl-project/sglang/pull/22913
- Status/date: `merged`, created 2026-04-15, merged 2026-04-17; author `alisonshao`.
- Diff scope read: `4` files, `+184/-247`; areas: model wrapper, quantization, kernel, tests/benchmarks; keywords: cuda, test, attention, config, fp4, quant, scheduler, eagle, flash, spec.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245 (245 lines); hunks: -import unittest; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: +import unittest; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: +import unittest; symbols: TestQwen35FP4, test_gsm8k
  - `.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: jobs:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`; keywords observed in patches: cuda, test, attention, config, fp4, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22948 - [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled

- Link: https://github.com/sgl-project/sglang/pull/22948
- Status/date: `merged`, created 2026-04-16, merged 2026-04-16; author `mqhc2020`.
- Diff scope read: `1` files, `+17/-1`; areas: model wrapper, MoE/router; keywords: config, deepep, expert, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: def can_fuse_shared_expert(; def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, can_fuse_shared_expert, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_moe.py`; keywords observed in patches: config, deepep, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- Link: https://github.com/sgl-project/sglang/pull/23034
- Status/date: `merged`, created 2026-04-17, merged 2026-04-17; author `zijiexia`.
- Diff scope read: `73` files, `+2214/-215`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, docs/config; keywords: doc, spec, attention, config, cuda, cache, moe, quant, eagle, expert.
- Code diff details:
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: +---
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: +---
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: +---; symbols: per_token_group_quant_8bit, add
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45 (286 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s; Please consult the documentation below and [server_args.py](https://github.com
  - `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx` added +219/-0 (219 lines); hunks: +export const Qwen36Deployment = () => {
- Optimization/support interpretation: The concrete diff surface is `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; keywords observed in patches: doc, spec, attention, config, cuda, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- Link: https://github.com/sgl-project/sglang/pull/23467
- Status/date: `merged`, created 2026-04-22, merged 2026-04-22; author `mickqian`.
- Diff scope read: `1` files, `+31/-4`; areas: quantization; keywords: config, fp8, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: def __getattr__(self, name):; def is_layer_skipped(; symbols: __getattr__, _module_path_match, names, is_layer_skipped
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/utils.py`; keywords observed in patches: config, fp8, kv, moe, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

- Covered PRs: 37; open PRs: 1.
- Open PRs to keep tracking: [#23474](https://github.com/sgl-project/sglang/pull/23474)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
