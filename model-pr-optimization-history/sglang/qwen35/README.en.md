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
