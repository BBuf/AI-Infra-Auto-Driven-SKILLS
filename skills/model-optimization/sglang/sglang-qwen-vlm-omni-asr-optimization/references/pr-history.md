# Qwen VLM / Omni / ASR PR History

This file is a manually written PR dossier for SGLang Qwen multimodal models:
Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, Qwen3-ASR, and the Qwen3.5
multimodal cross-family path that shares `qwen_vl.py` and encoder-disaggregation
infrastructure.

Evidence rules followed for this pass:

- Every SGLang PR card below was inspected with `gh pr view` and `gh pr diff --patch`
  or with the merged commit diff when the final merge commit was clearer.
- The card records motivation, the implementation shape, the most important code
  snippet, reviewed files, and validation/risk notes.
- Open or closed-unmerged PRs are marked as radar items. They are not treated as
  current-main behavior.
- Public docs/blog evidence was read separately: SGLang Qwen3-VL docs, LMSYS AMD
  Qwen3/Qwen3-VL latency blog, and SGLang issue `#18466`.

Snapshot:

- SGLang main evidence around `b3e6cf60a` on 2026-04-22.
- sgl-cookbook evidence around `816bad5` on 2026-04-21.
- Public docs/blog checked on 2026-04-23:
  - https://docs.sglang.io/basic_usage/qwen3_vl.html
  - https://www.lmsys.org/blog/2026-02-11-Qwen-latency/
  - https://github.com/sgl-project/sglang/issues/18466

## Runtime Surfaces

- `python/sglang/srt/models/qwen2_5_vl.py`
- `python/sglang/srt/models/qwen3_vl.py`
- `python/sglang/srt/models/qwen3_vl_moe.py`
- `python/sglang/srt/models/qwen3_omni_moe.py`
- `python/sglang/srt/models/qwen3_asr.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/multimodal/processors/qwen_audio.py`
- `python/sglang/srt/multimodal/processors/qwen3_asr.py`
- `python/sglang/srt/managers/mm_utils.py`
- `python/sglang/srt/disaggregation/encode_server.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`

## Current-Main PR Cards

### #10911 Qwen3-Omni Thinker-Only Support

- Link/state: https://github.com/sgl-project/sglang/pull/10911, merged.
- Diff coverage: 16 files, including `configs/qwen3_omni.py`, `model_config.py`,
  `base_processor.py`, `qwen_vl.py`, `qwen3_omni_moe.py`, and server tests.
- Motivation: bring up Qwen3-Omni in thinker-only mode and solve the missing model
  architecture/config path. The runtime needed to understand nested thinker/talker
  config, audio inputs, and Qwen Omni mRoPE metadata before it could serve the model.
- Key implementation: register `Qwen3OmniMoeForConditionalGeneration`, add a nested
  `Qwen3OmniMoeConfig`, route audio through the multimodal processor, and extend
  Qwen VL mRoPE handling with audio sequence lengths and audio token IDs.
- Key code:

```python
"Qwen3OmniMoeForConditionalGeneration",
```

```python
class Qwen3OmniMoeConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe"
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }
```

```python
if hf_config.model_type == "qwen3_omni_moe":
    hf_config = hf_config.thinker_config
audio_feature_lengths = torch.sum(audio_item.feature_attention_mask, dim=1)
MRotaryEmbedding.get_rope_index(..., audio_seqlens=audio_feature_lengths)
```

- Validation/risk: adds a `Qwen/Qwen3-Omni-30B-A3B-Instruct` server test. Later PRs
  refine Omni audio performance and audio-in-video behavior, so this PR should be
  treated as the bring-up baseline rather than the final optimized path.

### #10985 Qwen3-VL MRotaryEmbedding Launch Fix

- Link/state: https://github.com/sgl-project/sglang/pull/10985, merged.
- Diff coverage: `layers/rotary_embedding.py`, `models/qwen3_moe.py`.
- Motivation: after fused KV buffer support landed, the attention path passed
  `fused_set_kv_buffer_arg` into rotary embedding. Qwen3-VL uses `MRotaryEmbedding`,
  which cannot save KV through the same fused hook, causing launch/runtime failure.
- Key implementation: make `MRotaryEmbedding.forward` accept the optional argument
  but assert it is not used, and gate fused KV support when the rotary embedding is
  an mRoPE instance.
- Key code:

```python
def forward(..., fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None):
    assert fused_set_kv_buffer_arg is None, "save kv cache is not supported for MRotaryEmbedding."
```

```python
self.compatible_with_fused_kv_buffer = (
    False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
)
```

- Validation/risk: this is a small compatibility fix, but it documents an important
  invariant: Qwen VLM mRoPE needs separate fused-cache handling.

### #12333 Qwen3-VL Pipeline Parallelism

- Link/state: https://github.com/sgl-project/sglang/pull/12333, merged.
- Diff coverage: `schedule_policy.py`, `qwen3_omni_moe.py`, `qwen3_vl.py`,
  `test_utils.py`, `test_pp_single_node.py`.
- Motivation: `Qwen/Qwen3-VL-8B-Thinking --tp 2 --pp-size 2` could not run because
  Qwen-VL had no PP-aware language head, media embedding handoff, or weight-loading
  rank filtering.
- Key implementation: add `get_pp_group()` to the Qwen3-VL model, instantiate
  `lm_head` only on the last PP rank, pass proxy tensors through
  `general_mm_embed_routine`, return hidden states on non-last ranks, and load tied
  `lm_head` only where it exists.
- Key code:

```python
self.pp_group = get_pp_group()
if self.pp_group.is_last_rank:
    self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

```python
if self.pp_group.is_last_rank:
    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
else:
    return hidden_states
```

```python
if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
    if "lm_head.weight" in params_dict:
        weight_loader(lm_head_param, loaded_weight)
```

- Validation/risk: adds `TestQwenVLPPAccuracy`. A scheduler leak in chunked prefill
  was fixed in the same PR, so PP regressions should test both normal and chunked
  prefill requests.

### #13724 Qwen3-VL Vision Encoder Data Parallelism

- Link/state: https://github.com/sgl-project/sglang/pull/13724, merged.
- Diff coverage: `qwen3_vl.py`, `test/nightly/test_encoder_dp.py`.
- Motivation: Qwen3-VL ViT was a bottleneck at high image/video concurrency.
  Based on the earlier encoder-DP work, this PR added DP sharding for the vision
  encoder and reported TTFT reductions without MMMU accuracy loss.
- Key implementation: thread `use_data_parallel` through the vision MLP, block, and
  patch-merger modules; force vision TP size/rank to 1 under DP; and call
  `run_dp_sharded_mrope_vision_model(..., rope_type="rope_3d")` for image/video
  features.
- Key code:

```python
self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
```

```python
if self.use_data_parallel:
    return run_dp_sharded_mrope_vision_model(
        self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
    )
```

- Validation/risk: nightly encoder-DP coverage was extended. Any later change to
  position embedding, all-gather, or vision linear parallelism must retest DP and
  non-DP modes side by side.

### #13736 Qwen-VL cu_seqlens CPU-Side NumPy Optimization

- Link/state: https://github.com/sgl-project/sglang/pull/13736, merged.
- Diff coverage: `qwen2_vl.py`, `qwen3_vl.py`, `models/utils.py`,
  `test_repeat_interleave.py`, `run_suite.py`.
- Motivation: the CPU-side `torch.repeat_interleave` used to build ViT
  `cu_seqlens` was visible in TTFT profiles. The PR reports about 1.5% TTFT
  improvement from a pure CPU NumPy replacement.
- Key implementation: add `compute_cu_seqlens_from_grid_numpy`, require CPU input,
  use `np.repeat` + `cumsum(np.int32)`, and call the helper from Qwen2/Qwen3 VLM
  vision forward paths.
- Key code:

```python
def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:
    assert grid_thw.device.type == "cpu"
    arr = grid_thw.numpy()
    cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
        axis=0, dtype=np.int32
    )
    return torch.from_numpy(np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens]))
```

- Validation/risk: includes correctness and benchmark tests. Because this helper
  assumes CPU tensors, callers must not silently pass GPU `grid_thw`.

### #14292 Qwen-VL Rotary Position-ID Cache

- Link/state: https://github.com/sgl-project/sglang/pull/14292, merged.
- Diff coverage: `qwen2_5_vl.py`, `qwen3_vl.py`, `models/utils.py`.
- Motivation: repeated construction of 2D rotary position IDs in the ViT path was
  pure overhead. The PR reports around 2% TTFT improvement in its benchmark and
  calls out larger end-to-end savings when the cache is hit repeatedly.
- Key implementation: add a `RotaryPosMixin` with an LRU-cached `rot_pos_ids(h,w,
  spatial_merge_size)` helper and reuse it from Qwen2.5-VL and Qwen3-VL.
- Key code:

```python
class RotaryPosMixin:
    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        ...
        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))
```

- Validation/risk: cache key is shape-based. Bugs here usually show up as image/video
  spatial misalignment rather than text-only failures.

### #14907 Chunked ViT Attention

- Link/state: https://github.com/sgl-project/sglang/pull/14907, merged.
- Diff coverage: `qwen3_vl.py`, `mm_utils.py`.
- Motivation: Qwen3-VL-235B-A22B-Instruct-FP8 could OOM when a single request carried
  hundreds of images/frames because the ViT processed all patches in one call.
- Key implementation: introduce image/patch chunk limits via
  `SGLANG_VLM_MAX_PATCHES_PER_VIT` and `SGLANG_VLM_MAX_IMAGES_PER_VIT`, split
  `pixel_values`/`grid_thw` by image boundaries, run the visual encoder per chunk,
  and concatenate embeddings.
- Key code:

```python
max_patches_per_call = get_int_env_var("SGLANG_VLM_MAX_PATCHES_PER_VIT", 0)
max_images_per_call = get_int_env_var("SGLANG_VLM_MAX_IMAGES_PER_VIT", 0)
...
chunk_embeds = self.visual(pixel_chunk, grid_thw=grid_chunk)
all_chunk_embeds.append(chunk_embeds)
return torch.cat(all_chunk_embeds, dim=0)
```

- Validation/risk: this was later superseded structurally by `#22038`, which moved
  chunk-awareness into multimodal cache/embedding utilities. Keep this PR in the
  history because it explains the original OOM motivation.

### #15205 Qwen3-VL / GLM-4.1V Cos-Sin Cache for Vision RoPE

- Link/state: https://github.com/sgl-project/sglang/pull/15205, merged.
- Diff coverage: `layers/attention/vision.py`, `layers/rotary_embedding.py`,
  `models/glm4v.py`, `models/qwen3_vl.py`.
- Motivation: Qwen3-VL and GLM-4.1V repeatedly recomputed vision RoPE cos/sin from
  frequencies. The PR refactors the shared RoPE path so 2D vision RoPE can index a
  precomputed cos/sin cache; the PR body reports a micro path reduction from about
  490 us to 186 us and about 2% TTFT improvement on an image benchmark.
- Key implementation: expose `RotaryEmbedding.get_cos_sin`, let `VisionAttention`
  accept explicit `rotary_pos_emb_cos/sin`, replace Qwen3-VL's HF
  `Qwen2_5_VisionRotaryEmbedding` use with SGLang `get_rope`, and move Qwen3-VL
  `rot_pos_emb` to return flattened cached cos/sin tensors.
- Key code:

```python
def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

```python
elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
    cos = rotary_pos_emb_cos
    sin = rotary_pos_emb_sin
```

```python
cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
cos_combined = cos[pos_ids].flatten(1)
sin_combined = sin[pos_ids].flatten(1)
return cos_combined, sin_combined
```

- Validation/risk: MMMU validation in the PR body remained comparable. This PR is a
  cross-model VLM primitive and should be checked whenever GLM VLM or Qwen3-VL
  changes vision RoPE layout.

### #15320 Qwen3-VL ViT Piecewise CUDA Graph

- Link/state: https://github.com/sgl-project/sglang/pull/15320, merged.
- Diff coverage: `parallel_state.py`, `attention/vision.py`, `qwen2_5_vl.py`,
  `qwen3_vl.py`, `vit_cuda_graph_runner.py`, `test_vlms_vit_cuda_graph.py`.
- Motivation: capture Qwen3-VL ViT compute in piecewise CUDA graph, including TP>1
  support and deepstack output. The PR reports TTFT improvement on 8xH20
  Qwen3-VL-8B TP4 from 1384.53 ms to 1120.68 ms.
- Key implementation: allow out-of-place all-reduce with torch symmetric memory,
  remove the TP==1 restriction from vision attention graph capture, add
  `forward_with_cuda_graph`, and extend `ViTCudaGraphRunner` to Qwen3 deepstack
  blocks with full/window attention metadata.
- Key code:

```python
if get_bool_env_var("SGLANG_VIT_ENABLE_CUDA_GRAPH"):
    return self.forward_with_cuda_graph(x, grid_thw)
```

```python
return self.cuda_graph_runner.run(
    x=x,
    rotary_pos_emb_cos=rotary_pos_emb_cos,
    rotary_pos_emb_sin=rotary_pos_emb_sin,
    cu_seqlens=cu_seqlens,
)
```

```python
if self._deepstack_visual_indexes and layer_num in self._deepstack_visual_indexes:
    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](y)
    deepstack_outs.append(deepstack_out)
```

- Validation/risk: graph capture must validate nonblank image/video outputs, TP>1,
  deepstack shape, and attention backend compatibility.

### #16366 Qwen3-VL Video Memory Optimization

- Link/state: https://github.com/sgl-project/sglang/pull/16366, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: high-concurrency video requests on Qwen3-Omni/Qwen3-VL could OOM
  because per-item video features stayed resident on device after concatenation.
- Key implementation: move each item feature to the visual device just before
  concatenation, build the `pixel_values` tensor, then offload the per-item feature
  back to CPU.
- Key code:

```python
for item in items:
    item.feature = item.feature.to(self.visual.device)
pixel_values = torch.cat([item.feature for item in items], dim=0).type(self.visual.dtype)
for item in items:
    item.feature = item.feature.to("cpu")
```

- Validation/risk: helps memory pressure but interacts with `--keep-mm-feature-on-device`
  and later lazy-transfer work in `#22038`.

### #17624 Qwen3-VL DP Size Greater Than 1

- Link/state: https://github.com/sgl-project/sglang/pull/17624, merged.
- Diff coverage: `forward_batch_info.py`, `mm_utils.py`, `qwen3_vl.py`,
  `linear.py`, related tests.
- Motivation: `--mm-enable-dp-encoder` with `--enable-dp-attention` failed or
  produced precision issues when TP and DP sizes differed. Padding mRoPE positions
  also used the wrong dimension.
- Key implementation: pad `mrope_positions` by token dimension, use attention TP
  rank/group in `run_dp_sharded_mrope_vision_model`, wire `enable_dp_lm_head` into
  Qwen3-VL `lm_head`, and add attention-TP all-reduce support to row-parallel
  linear layers used by the vision path.
- Key code:

```python
self.mrope_positions = torch.cat(
    [
        self.mrope_positions,
        self.mrope_positions.new_zeros(3, num_tokens - self.mrope_positions.shape[1]),
    ],
    dim=1,
)
```

```python
tp_size = get_attention_tp_size()
if tp_size == 1:
    return vision_model(pixel_values, grid_thw=torch.tensor(grid_thw_list))
gathered_embeds = get_attention_tp_group().all_gather(image_embeds_local_padded, dim=0)
```

- Validation/risk: this PR is the basis for DP encoder correctness. Any change that
  touches DP attention groups, `mrope_positions`, or vision linear TP must repeat
  DP-size>1 launch and image accuracy checks.

### #18024 Qwen3-VL Weight Loading for Untied LM Head

- Link/state: https://github.com/sgl-project/sglang/pull/18024, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: Qwen3-VL-8B generated bad output because weight loading copied
  `embed_tokens.weight` into `lm_head.weight` unconditionally, even for models with
  `tie_word_embeddings=False`.
- Key implementation: gate the tied-weight copy on both PP last-rank ownership and
  `self.config.tie_word_embeddings`.
- Key code:

```python
if (
    self.pp_group.is_last_rank
    and "model.embed_tokens.weight" in name
    and self.config.tie_word_embeddings
):
```

- Validation/risk: primarily accuracy-facing. When adding new Qwen3-VL checkpoints,
  always inspect `tie_word_embeddings`.

### #18185 Qwen3-Omni Audio Encoder Optimization

- Link/state: https://github.com/sgl-project/sglang/pull/18185, merged.
- Diff coverage: `qwen3_omni_moe.py`.
- Motivation: Qwen3-Omni thinker ASR/audio path was slow. The PR body reports ASR
  end-to-end throughput improvement from about 0.28 req/s to 3.12 req/s.
- Key implementation: replace audio encoder FFN `nn.Linear` with
  `ColumnParallelLinear`/`RowParallelLinear`, vectorize mask construction with
  `torch.arange`, add a non-chunked convolution fast path, and move
  `feature_attention_mask` to the audio tower device.
- Key code:

```python
self.fc1 = ColumnParallelLinear(self.embed_dim, config.encoder_ffn_dim, bias=True, prefix=f"{prefix}.fc1")
self.fc2 = RowParallelLinear(config.encoder_ffn_dim, self.embed_dim, bias=True, prefix=f"{prefix}.fc2")
```

```python
idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)
```

```python
if padded_feature.size(0) <= self.conv_chunksize:
    padded_embed = F.gelu(self.conv2d1(padded_feature))
else:
    for chunk in padded_feature.split(self.conv_chunksize, dim=0):
        ...
```

- Validation/risk: audio encoder kernels and masks must be validated with ASR and
  audio+text prompts, not only with vision requests.

### #19003 FlashInfer CUDNN Prefill as Qwen3-VL ViT Backend

- Link/state: https://github.com/sgl-project/sglang/pull/19003, merged.
- Diff coverage: `attention/vision.py`, `qwen3_vl.py`, `server_args.py`,
  `test_vlms_vit_flashinfer_cudnn.py`.
- Motivation: add a `flashinfer_cudnn` VLM ViT attention backend for Qwen3-VL. The
  PR reports TTFT improvement from 1054 ms to 931 ms compared with FA3 on its test.
- Key implementation: introduce `VisionFlashInferAttention` using
  `flashinfer.prefill.cudnn_batch_prefill_with_kv_cache`, add the backend choice,
  allocate workspace, bucket batch/max-seqlen sizes, and compute packed q/k/v/o
  element indptrs for Qwen3-VL.
- Key code:

```python
output, _ = cudnn_batch_prefill_with_kv_cache(
    q, k, v, scale, self.workspace_buffer,
    max_token_per_sequence=max_seqlen,
    actual_seq_lens_q=seq_lens_4d,
    batch_offsets_q=indptr_qk,
    batch_offsets_v=indptr_v,
    batch_offsets_o=indptr_o,
    is_cuda_graph_compatible=True,
)
```

```python
def compute_flashinfer_batch_offsets_packed(self, token_cu_seqlens, *, elem_per_token):
    elem_indptr = (token_indptr * int(elem_per_token)).astype(np.int32)
    return np.concatenate([elem_indptr, elem_indptr, elem_indptr], axis=0)
```

- Validation/risk: backend is sensitive to shape buckets and graph compatibility.
  Retest long-video and high-resolution image workloads when changing indptr logic.

### #19291 Missing quant_config in Qwen3-VL

- Link/state: https://github.com/sgl-project/sglang/pull/19291, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: Qwen3.5 NVFP4 variants using the Qwen3-VL path fell back to bf16 KV
  cache because `quant_config` was not stored on the model.
- Key implementation: store `self.quant_config = quant_config` during model init.
- Key code:

```python
self.pp_group = get_pp_group()
self.quant_config = quant_config
```

- Validation/risk: tiny code change, large deployment impact for quantized VLM.
  Verify KV-cache dtype after loading NVFP4/FP8 checkpoints.

### #19333 Qwen3-VL Visual Module Weight Loading

- Link/state: https://github.com/sgl-project/sglang/pull/19333, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: Qwen3-VL visual model loading regressed because merger/visual prefix
  mapping was removed; visual weights did not load correctly and responses degraded.
- Key implementation: remap `model.visual.` back to `visual.` in the visual loader
  branch, in addition to qkv naming fixes.
- Key code:

```python
if "visual" in name:
    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
    name = name.replace(r"model.visual.", r"visual.")
```

- Validation/risk: this is an accuracy fix. Always run an image prompt after visual
  load-path changes; text-only checks will not catch it.

### #20759 Qwen3-VL DP Encoder Hang Fix

- Link/state: https://github.com/sgl-project/sglang/pull/20759, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: Qwen3-VL hung under `--mm-enable-dp-encoder` because the vision
  `pos_embed` remained a tensor-parallel `VocabParallelEmbedding`; ranks without
  image work could wait forever.
- Key implementation: disable TP for `pos_embed` when data-parallel encoder mode is
  active and only use the attention TP group when DP encoder is not forcing TP off.
- Key code:

```python
self.pos_embed = VocabParallelEmbedding(
    self.num_position_embeddings,
    self.hidden_size,
    enable_tp=not use_data_parallel,
    use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
)
```

- Validation/risk: supersedes the narrower `#20788` behavior. It should be treated
  as the current-main rule for DP encoder position embeddings.

### #20788 DP Encoder Position-Embedding TP Issue

- Link/state: https://github.com/sgl-project/sglang/pull/20788, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: with `--mm-enable-dp-encoder --tp 2`, one rank could receive an image
  while another did not, causing TP position-embedding communication to hang.
- Key implementation: add `enable_tp=False if use_data_parallel else True` to the
  Qwen3-VL position embedding.
- Key code:

```python
use_attn_tp_group=is_dp_attention_enabled(),
enable_tp=False if use_data_parallel else True,
```

- Validation/risk: this is a predecessor/narrow variant of `#20759`. Keep it in
  history because it documents the first observed hang mechanism.

### #21458 AMD Qwen3-VL Decode Fusion

- Link/state: https://github.com/sgl-project/sglang/pull/21458, merged.
- Diff coverage: `qwen3.py`.
- Motivation: on ROCm decode, Qwen3-VL paid separate costs for QKV split, QK RMSNorm,
  3D mRoPE, and KV-cache write. This PR uses an AITER fused kernel for the AMD path.
- Key implementation: detect HIP + AITER + `MRotaryEmbedding` with `mrope_section`,
  allocate graph-safe scale tensors, add `forward_prepare_fused_mrope`, write KV
  cache in the fused kernel, and call attention with `save_kv_cache=False`.
- Key code:

```python
self.use_fused_qk_norm_mrope = (
    _has_fused_qk_norm_mrope
    and isinstance(self.rotary_emb, MRotaryEmbedding)
    and getattr(self.rotary_emb, "mrope_section", None) is not None
)
```

```python
fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
    qkv_3d, self.q_norm.weight, self.k_norm.weight, cos_sin, positions,
    num_tokens, self.num_heads, self.num_kv_heads, self.num_kv_heads,
    self.head_dim, self.rotary_emb.is_neox_style,
    self.rotary_emb.mrope_section, self.rotary_emb.mrope_interleaved,
    ...
)
attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
```

- Validation/risk: AMD-only decode fusion. Regressions may appear only with
  Qwen3-VL mRoPE sections on HIP, not on NVIDIA.

### #21469 Qwen3-VL-30B-A3B-Instruct LoRA Support

- Link/state: https://github.com/sgl-project/sglang/pull/21469, merged.
- Diff coverage: `qwen3_vl_moe.py`, Qwen3-VL LoRA manual/registered tests.
- Motivation: support LoRA for `Qwen/Qwen3-VL-30B-A3B-Instruct`, especially MoE
  expert adapter targets and registered logprob-diff validation.
- Key implementation: expand the Qwen3-VL-MoE LoRA allow pattern beyond attention
  qkv/o projections to include `mlp.experts`, `lm_head`, and `model.embed_tokens`;
  add a registered H200 logprob-diff test using the Qwen3-VL-30B-A3B adapter dataset.
- Key code:

```python
_lora_pattern_moe = re.compile(
    r"^(?:model\.layers\.(\d+)\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
)
```

```python
engine = sgl.Engine(
    model_path=BASE_MODEL,
    tp_size=8,
    enable_lora=True,
    moe_runner_backend="triton",
    experts_shared_outer_loras=True,
)
```

- Validation/risk: not a ViT optimization, but it is Qwen3-VL-MoE model support.
  If adapter routing changes, run the registered logprob-diff test.

### #21849 Qwen3.5 VLM Encoder Disaggregation Allowlist

- Link/state: https://github.com/sgl-project/sglang/pull/21849, merged.
- Diff coverage: `server_args.py`, `encode_server.py`, `qwen_vl.py`,
  `test_epd_disaggregation.py`.
- Motivation: Qwen3.5 multimodal runtime support existed, but encoder-disaggregation
  startup rejected `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration` because of a stale architecture allowlist.
- Key implementation: add the Qwen3.5 dense/MoE architectures to encoder-only /
  language-only validation and extend Qwen3.5 video timestamp handling in
  `encode_server.py` and `qwen_vl.py`.
- Key code:

```python
"Qwen3_5ForConditionalGeneration",
"Qwen3_5MoeForConditionalGeneration",
```

```python
self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
```

- Validation/risk: adds EPD disaggregation regression coverage. This belongs in
  both Qwen3.5 and Qwen VLM histories because the failing path is multimodal
  encoder disaggregation.

### #22038 Chunk-Aware ViT Encoding Cache and Lazy Transfer

- Link/state: https://github.com/sgl-project/sglang/pull/22038, merged.
- Diff coverage: `mm_utils.py`, `schedule_batch.py`, `chunk_cache.py`,
  `qwen3_vl.py`, and related multimodal cache files.
- Motivation: the earlier request-level chunked ViT path encoded too much media and
  moved features to device too early. Long video/multi-image chunked prefill needed
  per-image cache and lazy CPU-to-device transfer.
- Key implementation: remove Qwen3-VL-internal env-driven chunking, add
  `_get_chunked_embedding_by_item`, check item overlap with the active token chunk,
  fetch/set per-image `EmbeddingResult` entries, and move only cache misses to the
  device immediately before visual encoding.
- Key code:

```python
for idx, (item, offset) in enumerate(zip(embedding_items_per_req, items_offset)):
    start, end = offset
    if end >= chunk_start and start < chunk_end:
        overlapping.append((idx, item, start, end))
```

```python
cached = embedding_cache.get_single(item.hash)
...
_move_items_to_device(miss_item_list, device)
all_miss_embedding = data_embedding_func(miss_item_list)
```

```python
def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
    embedding = self.mm_cache.get(mm_hash)
    if embedding is not None:
        self.mm_cache.move_to_end(mm_hash)
    return embedding
```

- Validation/risk: this is now the main chunk-aware VLM cache design. It affects all
  multimodal models, but Qwen3-VL is a primary beneficiary because of large ViT
  feature tensors and video chunks.

### #22073 Qwen3-ASR Model Support

- Link/state: https://github.com/sgl-project/sglang/pull/22073, merged.
- Diff coverage: ASR benchmark/docs, `configs/qwen3_asr.py`, `model_config.py`,
  `encode_server.py`, `serving_transcription.py`, `models/qwen3_asr.py`,
  `base_processor.py`, `processors/qwen3_asr.py`.
- Motivation: implement issue `#22025` and serve Qwen3-ASR 0.6B/1.7B through
  `/v1/audio/transcriptions`, reusing the Qwen3-Omni audio encoder with a Qwen3
  language model.
- Key implementation: add `Qwen3ASRProcessor` that expands a single audio pad token
  into the correct number of placeholders, add `Qwen3ASRForConditionalGeneration`
  with `Qwen3OmniMoeAudioEncoder` + `Qwen3ForCausalLM`, remap thinker weights, and
  add transcription adapter/postprocessing.
- Key code:

```python
audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
feat_lengths = inputs["feature_attention_mask"].sum(dim=-1)
audio_token_counts = self._get_feat_extract_output_lengths(feat_lengths)
new_ids.extend([audio_pad_id] * n)
```

```python
self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
self.language_model = Qwen3ForCausalLM(...)
```

```python
if name.startswith("thinker.audio_tower."):
    name = name.replace("thinker.audio_tower.", "audio_tower.", 1)
elif name.startswith("thinker.model."):
    name = name.replace("thinker.model.", "language_model.model.", 1)
```

- Validation/risk: the OpenAI transcription route skips Whisper segment parsing for
  Qwen3-ASR. Validate both raw transcription and model-loading remaps.

### #22089 Chunk-Based Streaming ASR for Qwen3-ASR

- Link/state: https://github.com/sgl-project/sglang/pull/22089, merged.
- Diff coverage: `streaming_asr.py`, transcription adapter structs, and
  `serving_transcription.py`.
- Motivation: `#22073` accepted uploaded audio and returned a final transcript, but
  production ASR needs partial output. This PR streams output by repeatedly running
  accumulated 2-second audio chunks with rollback for unfixed tokens.
- Key implementation: add `StreamingASRState`, `split_audio_chunks`, Qwen3-ASR
  chunk config, `_generate_chunked_asr_stream`, per-chunk model requests, SSE word
  emission, disconnection handling, and whitespace handling across chunk boundaries.
- Key code:

```python
@dataclass
class StreamingASRState:
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    def update(self, new_transcript: str) -> str:
        words = new_transcript.split()
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = " ".join(words[: -self.unfixed_token_num])
```

```python
if self._adapter.supports_chunked_streaming:
    return StreamingResponse(
        self._generate_chunked_asr_stream(adapted_request, request, raw_request),
        media_type="text/event-stream",
    )
```

```python
content = word if first_word else " " + word
first_word = False
```

- Validation/risk: test chunk boundaries, whitespace, final transcript accumulation,
  and cancellation. Full-audio correctness alone does not cover this path.

### #22230 Qwen3-VL EAGLE3 Support

- Link/state: https://github.com/sgl-project/sglang/pull/22230, merged.
- Diff coverage: `qwen3_vl.py`.
- Motivation: enable EAGLE3 speculative decoding for Qwen3-VL, including auxiliary
  hidden-state capture compatible with VLM forward.
- Key implementation: add `capture_aux_hidden_states`, unpack aux hidden states when
  capture is enabled, pass aux states into `logits_processor`, expose
  `get_embed_and_head`, and set default capture layers based on decoder depth.
- Key code:

```python
self.capture_aux_hidden_states = False
if self.capture_aux_hidden_states:
    hidden_states, aux_hidden_states = hidden_states
return self.logits_processor(
    input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
)
```

```python
self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
```

- Validation/risk: VLM EAGLE3 must validate image/video requests, not only text
  speculative decoding, because media embedding and deepstack change hidden states.

### #22266 NPU Qwen3.5 Video Processor Fix

- Link/state: https://github.com/sgl-project/sglang/pull/22266, merged.
- Diff coverage: `hardware_backend/npu/modules/qwen_vl_processor.py`.
- Motivation: Qwen3.5 video preprocessing used a high-dimensional `permute` path
  unsupported on Ascend NPU. The PR patches the Transformers Qwen3-VL video processor
  to avoid tensors with more than 8 dimensions.
- Key implementation: add `npu_wrapper_video_preprocess`, group/resize videos, fuse
  rescale+normalize, reshape/permute through a lower-dimensional layout, and patch
  `Qwen3VLVideoProcessor._preprocess`.
- Key code:

```python
patches = patches.view(
    batch_size * grid_t,
    temporal_patch_size * channel,
    grid_h // merge_size,
    merge_size,
    patch_size,
    grid_w // merge_size,
    merge_size,
    patch_size,
)
patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)
```

```python
apply_module_patch(
    "transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor",
    "_preprocess",
    [npu_wrapper_video_preprocess],
)
```

- Validation/risk: NPU-only processor patch; compare GPU and NPU video output on the
  same prompt when touching this path.

### #22431 Qwen3.5 `processor_output` Video Processing Fix

- Link/state: https://github.com/sgl-project/sglang/pull/22431, merged.
- Diff coverage: `multimodal/processors/qwen_vl.py`.
- Motivation: when video data arrived in `processor_output` format, `preprocess_video`
  returned a single value while later code expected `(video, metadata)`, causing
  `ValueError: too many values to unpack`.
- Key implementation: return `(vr, None)` for already-processed video inputs.
- Key code:

```python
is_video_obj = isinstance(vr, VideoDecoderWrapper)
if not is_video_obj:
    return vr, None
```

- Validation/risk: small but important API compatibility fix for users who preprocess
  Qwen3.5 video with Transformers and pass processor outputs directly.

## Docs / Usage PR Cards

### #12554 Qwen3-VL Usage Docs

- Link/state: https://github.com/sgl-project/sglang/pull/12554, merged.
- Diff coverage: `docs/basic_usage/qwen3_vl.md` and docs index.
- Motivation: provide first-party SGLang usage instructions for Qwen3-VL image/video
  serving rather than relying on scattered launch examples.
- Key implementation: document FP8 and BF16 launches, image request, video request,
  `--mm-attention-backend`, `--mm-max-concurrent-calls`,
  `--keep-mm-feature-on-device`, and CUDA IPC transport.
- Key snippet:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tp 8 \
  --ep 8 \
  --keep-mm-feature-on-device
```

- Validation/risk: this doc is a deployment checklist source. Keep model-history
  notes aligned with it when flags change.

### #12703 Qwen3-Omni Usage Docs

- Link/state: https://github.com/sgl-project/sglang/pull/12703, open.
- Diff coverage: `docs/basic_usage/qwen3_omni.md` and docs index.
- Motivation: Qwen3-Omni needed SGLang-specific launch and image/audio/video request
  examples; the PR body notes official examples were not enough for SGLang users.
- Key implementation: document `Qwen/Qwen3-Omni-30B-A3B-Instruct --tp 4` launch and
  OpenAI-style requests with `image_url`, `audio_url`, and `video_url`.
- Key snippet:

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Omni-30B-A3B-Instruct --tp 4
```

- Validation/risk: open docs radar. Do not claim merged behavior from this PR.

## Open / Closed Radar PR Cards

### #12662 CPU Qwen3-VL and Qwen3-Omni Support

- Link/state: https://github.com/sgl-project/sglang/pull/12662, open.
- Diff coverage: CPU config update, AMX utils, vision attention, convolution,
  `qwen3_omni_moe.py`, `qwen3_vl.py`, `qwen_vl.py`, and CPU sgl-kernel files.
- Motivation: enable frontend CPU support for Qwen3-VL and Qwen3-Omni, including
  unaligned TP padding, CPU image preprocessing, SDPA attention fallback, conv3d,
  layernorm, and CPU kernel fusion.
- Key implementation: choose `sdpa` on CPU instead of FA3/FlashAttention-3, force
  fast image processor device to CPU, pad Qwen3-VL/Omni vision/audio heads for
  unaligned CPU TP, and preserve original head size for attention.
- Key code:

```python
qkv_backend="fa3" if not _is_cpu else "sdpa"
attn_implementation="flash_attention_3" if not _is_cpu else "sdpa"
```

```python
if _is_cpu:
    kwargs["device"] = "cpu"
elif not _is_npu:
    kwargs["device"] = "cuda"
```

```python
model_config.hf_config.vision_config.original_num_heads = (
    model_config.hf_config.vision_config.num_heads
)
model_config.hf_config.vision_config.num_heads = pad_vocab_size(...)
```

- Validation/risk: open CPU path; treat as design/radar until merged.

### #12261 Qwen2.5-VL cu_seqlens Correctness Fix

- Link/state: https://github.com/sgl-project/sglang/pull/12261, open.
- Diff coverage: `qwen2_5_vl.py`.
- Motivation: Qwen2.5-VL `cu_seqlens` was wrong for multi-frame/multi-patch inputs;
  Qwen3-VL had already received a similar correction.
- Key implementation: compute per-frame patch count as `H*W`, repeat by `T`, then
  cumulative-sum to build seqlens.
- Key code:

```python
cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
).cumsum(dim=0)
```

- Validation/risk: open correctness fix for Qwen2.5-VL video/multi-frame inputs.

### #13918 Qwen3-VL EAGLE3 Early Support

- Link/state: https://github.com/sgl-project/sglang/pull/13918, open.
- Diff coverage: `llama_eagle3.py`, `qwen3_vl.py`.
- Motivation: early dense Qwen3-VL EAGLE3 support with reported 1.41x end-to-end
  speedup in the PR body.
- Key implementation: adapt EAGLE3 to mRoPE interleaving and add the same broad
  Qwen3-VL aux hidden-state capture shape later merged in `#22230`.
- Key code:

```python
self.mrope_interleaved = rope_scaling.setdefault("mrope_interleaved", False)
if not self.mrope_interleaved:
    rope_scaling["rope_type"] = "default"
```

- Validation/risk: open and mostly superseded by merged `#22230`; still useful for
  understanding EAGLE3/mRoPE compatibility.

### #14886 Qwen3-Omni DP Encoder

- Link/state: https://github.com/sgl-project/sglang/pull/14886, open.
- Diff coverage: `qwen3_omni_moe.py`, encoder-DP nightly test.
- Motivation: extend Qwen3-VL encoder DP ideas to Qwen3-Omni vision/audio towers.
  The PR body reports close MMMU accuracy and TTFT improvement for multi-image.
- Key implementation: pass `use_data_parallel` into Qwen3-Omni audio encoder,
  vision encoder, and patch merger; force merger TP size/rank to local values under
  DP; add Omni to encoder-DP tests.
- Key code:

```python
self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
self.audio_tower = Qwen3OmniMoeAudioEncoder(
    config.audio_config, self.use_data_parallel
)
self.visual = Qwen3OmniMoeVisionEncoder(..., use_data_parallel=self.use_data_parallel)
```

- Validation/risk: open. Audio and vision DP need separate checks because audio
  sequence lengths feed mRoPE differently from image/video tokens.

### #16491 Qwen3-VL-MoE PP Expert Weight Skip

- Link/state: https://github.com/sgl-project/sglang/pull/16491, open.
- Diff coverage: `qwen3_vl_moe.py`.
- Motivation: Qwen3-VL-235B-A22B-FP8 with `pp=2,tp=4` attempted to load expert
  weights that do not exist on the current PP rank.
- Key implementation: compute the mapped expert name and skip it if absent from the
  local `params_dict`.
- Key code:

```python
name_mapped = name.replace(weight_name, param_name)
if name_mapped not in params_dict:
    continue
```

- Validation/risk: open. This is PP + MoE weight loading, not dense Qwen3-VL.

### #16571 ROCm Add+LayerNorm Fusion for Qwen3-VL ViT

- Link/state: https://github.com/sgl-project/sglang/pull/16571, open.
- Diff coverage: `layernorm.py`, `qwen3_vl.py`.
- Motivation: use AITER fused add+LayerNorm in the Qwen3-VL ViT on ROCm to reduce
  kernel launches and memory traffic.
- Key implementation: add `LayerNorm.forward_aiter(x, residual)`, return fused
  `(output, residual_out)`, and carry residual through Qwen3 vision blocks and
  mergers when HIP + `SGLANG_USE_AITER` is active.
- Key code:

```python
layernorm2d_fwd_with_add(
    output, x, residual, residual_out,
    self.weight.data, self.bias.data, self.variance_epsilon,
)
return output, residual_out
```

```python
if _use_fused_layernorm:
    hidden_states, residual = self.norm1(x, residual=residual)
```

- Validation/risk: open AMD/ROCm path. Compare image accuracy and ViT latency with
  and without `SGLANG_USE_AITER`.

### #16785 Qwen3-VL Deepstack Recompile Fix

- Link/state: https://github.com/sgl-project/sglang/pull/16785, open.
- Diff coverage: `mm_utils.py`, `piecewise_cuda_graph_runner.py`, `qwen3_vl.py`,
  `qwen3_vl_moe.py`, PCG tests.
- Motivation: mixed multimodal/text traffic caused TorchDynamo recompilation churn
  because `input_deepstack_embeds` existed only for multimodal requests.
- Key implementation: let `embed_mm_inputs` accept a preallocated deepstack tensor,
  allocate graph-runner deepstack buffers for multimodal models, and make Qwen3-VL
  return zero deepstack slices when no multimodal input exists so the forward shape
  remains stable.
- Key code:

```python
if prealloc_deepstack is not None:
    assert prealloc_deepstack.shape == deepstack_embedding_shape
    input_deepstack_embeds = prealloc_deepstack
    input_deepstack_embeds.zero_()
```

```python
if self.is_multimodal and hasattr(self, "input_deepstack_embeds"):
    kwargs["input_deepstack_embeds"] = self.input_deepstack_embeds[:num_tokens]
```

```python
if input_deepstack_embeds is None:
    ...
    self.deepstack_embeds_buffer = torch.zeros(new_len, total, dtype=dtype, device=device)
```

- Validation/risk: open. It touches PCG replay and Qwen3-VL-MoE; validate both
  multimodal and text-only batches in the same server.

### #16996 Qwen3-Omni `use_audio_in_video`

- Link/state: https://github.com/sgl-project/sglang/pull/16996, open.
- Diff coverage: engine/OpenAI request schema, base multimodal processor,
  qwen_vl processor.
- Motivation: support Qwen3-Omni video files that also carry audio; without this,
  users could not ask audio-visual questions over one video source.
- Key implementation: add `use_audio_in_video` request plumbing, return
  `(video_reader, audio_waveform)` from video loading, append audio multimodal items
  when present, and compute effective sampled FPS for Qwen VL video metadata.
- Key code:

```python
use_audio_in_video: Optional[bool] = False
vr, audio_waveform = load_video(..., use_audio_in_video=use_audio_in_video)
return vr, audio_waveform
```

```python
effective_fps = round(nframes / duration, 1) if duration > 0 else video_fps
```

- Validation/risk: open. Requires video+audio tests and attention to metadata used
  by mRoPE/timestamps.

### #17202 Qwen3-VL CPU/GPU Op Removal in ViT/Embedding

- Link/state: https://github.com/sgl-project/sglang/pull/17202, open.
- Diff coverage: `attention/vision.py`, `mm_utils.py`.
- Motivation: remove avoidable device operations in Qwen3-VL forward: unnecessary
  `.contiguous()` after q/k/v reshape and `torch.where`/index-based scatter during
  multimodal embedding insertion.
- Key implementation: keep q/k/v reshape non-contiguous where supported and use
  `masked_scatter_` directly for multimodal and deepstack embeddings.
- Key code:

```python
q = q.reshape(bsz * s, head, -1)
k = k.reshape(bsz * s, kv_head, -1)
v = v.reshape(bsz * s, kv_head, -1)
```

```python
mask_1d = mask.view(-1)
input_embeds.masked_scatter_(mask_1d.unsqueeze(-1), embedding)
```

- Validation/risk: open. Scatter shape errors can corrupt multimodal token placement,
  so validate image/video outputs and deepstack embeddings.

### #17276 Qwen3-VL EAGLE3 Deepstack-Aware Capture

- Link/state: https://github.com/sgl-project/sglang/pull/17276, open.
- Diff coverage: `qwen3_vl.py`.
- Motivation: add Qwen3-VL EAGLE3 support while avoiding hidden-state capture too
  early in the decoder, because Qwen3-VL injects deepstack features in early layers.
- Key implementation: default capture layers start after deepstack injection, using
  `[4, num_layers // 2, num_layers - 3]`, and handle `config.text_config` models.
- Key code:

```python
# Qwen3VL uses deepstack at decoder layers 0, 1, 2
self.model.layers_to_capture = [4, num_layers // 2, num_layers - 3]
```

- Validation/risk: open and partially superseded by `#22230`; keep as a caution that
  EAGLE3 layer capture and deepstack timing are coupled.

### #18721 Qwen3-VL DP Encoder Hang Follow-Up

- Link/state: https://github.com/sgl-project/sglang/pull/18721, open.
- Diff coverage: `vocab_parallel_embedding.py`, `qwen3_vl.py`.
- Motivation: avoid `VocabParallelEmbedding` all-reduce hang when DP encoder mode is
  on and only one rank has a multimodal item.
- Key implementation: warn rather than assert when `use_attn_tp_group` is set but TP
  is disabled, and set Qwen3-VL `pos_embed.enable_tp` based on
  `mm_enable_dp_encoder`.
- Key code:

```python
if use_attn_tp_group:
    logger.warning("not in tp_mode, use_attn_tp_group will not work")
```

```python
enable_tp=not get_global_server_args().mm_enable_dp_encoder,
use_attn_tp_group=is_dp_attention_enabled(),
```

- Validation/risk: open; overlaps with merged `#20759`.

### #18771 Qwen3-Omni MoE Fused-MoE Tuner Handling

- Link/state: https://github.com/sgl-project/sglang/pull/18771, open.
- Diff coverage: `benchmark/kernels/fused_moe_triton/common_utils.py`.
- Motivation: the fused MoE benchmark/tuner treated Qwen3-Omni as an unknown MoE
  architecture and fell into a branch expecting `num_local_experts`.
- Key implementation: add `Qwen3OmniMoeForConditionalGeneration` to the Qwen MoE
  architecture list.
- Key code:

```python
"Qwen3OmniMoeForConditionalGeneration",
```

- Validation/risk: open benchmarking/tuning support, not runtime serving behavior.

### #19242 Early Qwen3-ASR Support Attempt

- Link/state: https://github.com/sgl-project/sglang/pull/19242, open.
- Diff coverage: Qwen3-ASR config and processor files, `model_config.py`.
- Motivation: add Qwen3-ASR support in a Whisper-like shape before the later merged
  full implementation.
- Key implementation: add ASR config classes and HF processor wrapper, register
  `Qwen3ASRForConditionalGeneration`, but does not include the final model
  implementation that later appeared in `#22073`.
- Key code:

```python
class Qwen3ASRHFProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
```

```python
"Qwen3ASRForConditionalGeneration",
```

- Validation/risk: open/incomplete compared with merged `#22073`; useful for
  provenance but not current-main behavior.

### #19693 NPU Qwen3-VL-8B Accuracy Fix

- Link/state: https://github.com/sgl-project/sglang/pull/19693, open.
- Diff coverage: NPU kernels/versioning, rotary embedding, vocab embedding,
  Qwen3/Qwen3-MoE attention paths.
- Motivation: fix Qwen3-VL-8B accuracy on NPU. The visible changes target NPU RoPE,
  compile behavior, and QKV RMSNorm/RoPE split paths.
- Key implementation: use native RoPE path when bf16 query meets float cos/sin cache,
  disable `torch.compile` for the masked embedding helper on NPU, and wire NPU
  `split_qkv_rmsnorm_rope` naming/LM-head DP attention group updates.
- Key code:

```python
if query.dtype == torch.bfloat16 and self.cos_sin_cache.dtype == torch.float:
    return self.forward_native(positions, query, key, offsets)
```

```python
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def get_masked_input_and_mask(...):
```

- Validation/risk: open NPU accuracy path; must compare NPU/GPU outputs on the same
  image/video prompts.

### #20857 Qwen3-VL EVS Support

- Link/state: https://github.com/sgl-project/sglang/pull/20857, open.
- Diff coverage: `configs/qwen3_vl.py`, `mrope_rope_index.py`, `qwen3_vl.py`,
  EVS processor integration, `qwen_vl.py`.
- Motivation: support Efficient Video Sampling for Qwen3-VL so long videos can prune
  tokens while preserving accuracy. The PR body reports Video-MME tradeoffs.
- Key implementation: add `video_pruning_rate`, make mRoPE count the actual token
  count after pruning, let `Qwen3VLForConditionalGeneration` inherit `EVS`, and add
  `_maybe_apply_qwen3_evs` in the processor.
- Key code:

```python
video_pruning_rate=0.0
self.video_pruning_rate = video_pruning_rate
```

```python
vision_pos_ids = torch.stack([t_index, h_index, w_index])
llm_pos_ids_list.append(vision_pos_ids[:, :mm_token_count] + text_len + st_idx)
st = ed + mm_token_count
```

```python
class Qwen3VLForConditionalGeneration(EVS):
    @staticmethod
    def create_evs_config(config):
        return EVSConfig(video_pruning_rate=getattr(config, "video_pruning_rate", 0.0))
```

- Validation/risk: open. mRoPE/token-count correctness is the main risk after video
  pruning.

### #22052 Precise Qwen3-VL Embedding Interpolation Default

- Link/state: https://github.com/sgl-project/sglang/pull/22052, open.
- Diff coverage: docs, server args, `qwen3_vl.py`.
- Motivation: the previous interpolation default diverged from HF reference because
  `align_corners=False` differed from the reference `torch.linspace` behavior; the
  PR body notes bf16 position embedding diffs up to 6.6.
- Key implementation: rename the flag to `disable_precise_embedding_interpolation`
  so precise mode is default, set `align_corners` from the inverse flag, and use
  `_torch_interp_indices`.
- Key code:

```python
self.align_corners = (
    not get_global_server_args().disable_precise_embedding_interpolation
)
```

```python
h_idxs = self._torch_interp_indices(h, self.device)
w_idxs = self._torch_interp_indices(w, self.device)
```

- Validation/risk: open. It can alter visual position embeddings, so accuracy should
  be checked on image and video benchmarks.

### #22839 Qwen3-VL Config `from_dict` Compatibility

- Link/state: https://github.com/sgl-project/sglang/pull/22839, open.
- Diff coverage: `configs/__init__.py`, `configs/qwen3_5.py`,
  `configs/qwen3_vl.py`, `hf_transformers_utils.py`, unit tests.
- Motivation: Transformers 5.5.0+ natively supports Qwen3-VL, which can bypass
  SGLang config conversion and leave nested `vision_config`/`text_config` as dicts.
- Key implementation: export/register Qwen3-VL config classes and add `from_dict`
  methods that convert nested dicts to config objects for Qwen3-VL and Qwen3.5
  dense/MoE configs.
- Key code:

```python
@classmethod
def from_dict(cls, config_dict, **kwargs):
    config = super().from_dict(config_dict, **kwargs)
    if isinstance(getattr(config, "vision_config", None), dict):
        config.vision_config = cls.sub_configs["vision_config"](**config.vision_config)
    if isinstance(getattr(config, "text_config", None), dict):
        config.text_config = cls.sub_configs["text_config"](**config.text_config)
    return config
```

- Validation/risk: open config compatibility. Run the added unit tests and at least
  one real Qwen3-VL model load with the target Transformers version.

### #22848 WebSocket Streaming Audio Input for ASR

- Link/state: https://github.com/sgl-project/sglang/pull/22848, open.
- Diff coverage: OpenAI server app, websocket transcription service,
  `streaming_asr.py`, server args.
- Motivation: `#22089` streamed output but still required full uploaded audio. Real
  realtime ASR needs the server to accept audio frames as they arrive and emit
  transcript deltas.
- Key implementation: register `WS /v1/audio/transcriptions/stream`, define a
  `session.start` / binary PCM16 / `session.end` protocol, add
  `--asr-max-buffer-seconds`, convert PCM to WAV chunks, and share chunk processing
  with HTTP SSE through `process_asr_chunk`.
- Key code:

```python
@app.websocket("/v1/audio/transcriptions/stream")
async def openai_v1_audio_transcriptions_ws(ws: WebSocket):
    await ws.app.state.openai_serving_transcription.handle_websocket(ws)
```

```python
def _pcm_to_wav(pcm_buffer: bytes) -> bytes:
    samples = np.frombuffer(pcm_buffer, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()
```

```python
delta = await process_asr_chunk(
    tokenizer_manager=self.tokenizer_manager,
    adapter=self._adapter,
    state=state,
    audio_data=chunk_audio,
    is_last=is_last,
)
```

- Validation/risk: open realtime ASR path; must validate protocol errors,
  disconnects, max buffer, partial deltas, and final transcript.

### #23115 / #23220 Qwen3-VL-MoE Encoder-Only Guard

- Link/state:
  - https://github.com/sgl-project/sglang/pull/23115, open.
  - https://github.com/sgl-project/sglang/pull/23220, open.
- Diff coverage: both touch `qwen3_vl_moe.py` and add the same one-line guard.
- Motivation: after earlier encoder-only changes, `Qwen3VLMoeForConditionalGeneration`
  could enter `load_weights` without `self.model` initialized, then crash on
  `hasattr(self.model, "start_layer")`.
- Key implementation: check `hasattr(self, "model")` before dereferencing
  `self.model`.
- Key code:

```python
if (
    "visual" not in name
    and layer_id is not None
    and hasattr(self, "model")
    and hasattr(self.model, "start_layer")
):
```

- Validation/risk: open duplicate fixes. Track which one lands; do not count both as
  distinct current-main behavior.

### #23304 Qwen3-VL RoPE Config Compatibility

- Link/state: https://github.com/sgl-project/sglang/pull/23304, closed unmerged.
- Diff coverage: `qwen3.py`.
- Motivation: Qwen3-VL config can lack top-level `rope_parameters`, so direct
  `config.rope_parameters["rope_theta"]` access can fail.
- Key implementation: replace direct rope-parameter probing with shared
  `get_rope_config(config)`.
- Key code:

```python
rope_theta, rope_scaling = get_rope_config(config)
```

- Validation/risk: closed unmerged. Keep as a known compatibility issue only if it
  appears through another PR or local patch.

### #23469 NPU Qwen3-ASR Audio Loading

- Link/state: https://github.com/sgl-project/sglang/pull/23469, open.
- Diff coverage: audio loading utility.
- Motivation: deploy Qwen3-ASR on NPU without relying on torchaudio CUDA
  dependencies.
- Key implementation: if `is_npu()`, read audio with `soundfile`, average stereo to
  mono, resample with `scipy.signal.resample_poly` when sample rate differs, and
  return numpy audio.
- Key code:

```python
if is_npu():
    import soundfile as sf
    if isinstance(source, bytes):
        audio, original_sr = sf.read(BytesIO(source))
    else:
        audio, original_sr = sf.read(source)
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if original_sr != sr:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, sr, original_sr)
    return audio
```

- Validation/risk: open NPU ASR input path. Validate bytes and file-path inputs, mono
  conversion, resampling, and transcript equality against the regular loader.

## sgl-cookbook Evidence

These cookbook PRs are deployment recipes rather than SGLang runtime patches, but
they are important for model-specific optimization planning:

- https://github.com/sgl-project/sgl-cookbook/pull/76: Qwen3-VL AMD MI300X config
  generator.
- https://github.com/sgl-project/sgl-cookbook/pull/84: Qwen2.5-VL AMD MI300X guide.
- https://github.com/sgl-project/sgl-cookbook/pull/102: Qwen3-VL MI355X support.
- https://github.com/sgl-project/sgl-cookbook/pull/110: Qwen2.5-VL MI355X/MI325X
  AMD support.
- https://github.com/sgl-project/sgl-cookbook/pull/124: Qwen3-VL MI325X support.

Use the cookbook as deployment evidence, not as a replacement for SGLang source-diff
inspection. Runtime motivation and implementation details must still come from the
SGLang PR diffs above.

## Public Blog / Tracking Evidence

- SGLang Qwen3-VL docs describe FP8 and BF16 launches, image/video requests,
  `--mm-attention-backend`, `--mm-max-concurrent-calls`,
  `--keep-mm-feature-on-device`, and CUDA IPC transport.
- LMSYS AMD latency blog reports Qwen3-VL-235B MI300X optimization based on SGLang,
  with TTFT improvement of 1.62x and TPOT improvement of 1.90x versus the baseline.
- SGLang issue `#18466` tracks AMD Qwen3/Qwen3-VL latency work and separates
  preprocessing acceleration, multimodal transfer, ViT DP, and ViT kernel fusion.

## Validation Checklist

- Image: single image, multi-image, processor-output image, cached image.
- Video: raw URL/path, processor-output video, long video, chunked prefill, EVS if
  active.
- Qwen3-VL encoder: no-DP, DP encoder, PP, encoder-only/language-only EPD.
- Qwen3-VL-MoE: PP/TP expert loading, LoRA adapter logprob diff, encoder-only.
- Omni: image/audio/video, audio-in-video if active, audio tower performance.
- ASR: final transcription, HTTP chunked streaming, websocket streaming if active,
  NPU audio loader.
- Hardware lanes: NVIDIA, AMD/ROCm, NPU, CPU are separate; never infer one from the
  other.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen VLM / Omni / ASR PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-25 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-09-27 | [#10985](https://github.com/sgl-project/sglang/pull/10985) | merged | Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-10-28 | [#12261](https://github.com/sgl-project/sglang/pull/12261) | open | [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl | model wrapper | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-10-29 | [#12333](https://github.com/sgl-project/sglang/pull/12333) | merged | [PP] Add pp support for Qwen3-VL | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2025-11-03 | [#12554](https://github.com/sgl-project/sglang/pull/12554) | merged | [Docs] Add docs for Qwen3-VL image and video support | docs/config | `docs/basic_usage/qwen3_vl.md`, `docs/index.rst` |
| 2025-11-05 | [#12662](https://github.com/sgl-project/sglang/pull/12662) | open | [CPU] Add support for Qwen3-vl and Qwen3-omni | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, docs/config | `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py` |
| 2025-11-05 | [#12703](https://github.com/sgl-project/sglang/pull/12703) | open | add qwen3-omni docs | docs/config | `docs/basic_usage/qwen3_omni.md`, `docs/index.rst` |
| 2025-11-21 | [#13724](https://github.com/sgl-project/sglang/pull/13724) | merged | support qwen3_vl vision model dp | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py` |
| 2025-11-21 | [#13736](https://github.com/sgl-project/sglang/pull/13736) | merged | [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series | model wrapper, tests/benchmarks | `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-11-25 | [#13918](https://github.com/sgl-project/sglang/pull/13918) | open | [VLM] support qwen3-vl eagle infer | model wrapper | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py` |
| 2025-12-02 | [#14292](https://github.com/sgl-project/sglang/pull/14292) | merged | [VLM] Introduce Cache for positional embedding ids for Qwen-VL family | model wrapper | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-11 | [#14886](https://github.com/sgl-project/sglang/pull/14886) | open | Support qwen3-omni with DP Encoder | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-11 | [#14907](https://github.com/sgl-project/sglang/pull/14907) | merged | [VLM] Support chunked vit attention | model wrapper | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-15 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | model wrapper, attention/backend, multimodal/processor | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2025-12-17 | [#15320](https://github.com/sgl-project/sglang/pull/15320) | merged | [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL | model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2026-01-04 | [#16366](https://github.com/sgl-project/sglang/pull/16366) | merged | Optimize Qwen3-VL video memory usage | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-05 | [#16491](https://github.com/sgl-project/sglang/pull/16491) | open | [Qwen3-VL][PP] Skip loading expert weights not on this rank | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-01-06 | [#16571](https://github.com/sgl-project/sglang/pull/16571) | open | [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT | model wrapper | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py` |
| 2026-01-09 | [#16785](https://github.com/sgl-project/sglang/pull/16785) | open | [Bugfix] fix recompile in qwen3 vl | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-01-13 | [#16996](https://github.com/sgl-project/sglang/pull/16996) | open | feat: Support 'use_audio_in_video' option for qwen3omnimoe model | multimodal/processor | `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-01-16 | [#17202](https://github.com/sgl-project/sglang/pull/17202) | open | [Feat] Accelerate qwen3vl by remove cpu op | attention/backend, multimodal/processor | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-01-18 | [#17276](https://github.com/sgl-project/sglang/pull/17276) | open | Add Qwen3VL Eagle3 Inference Support | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-23 | [#17624](https://github.com/sgl-project/sglang/pull/17624) | merged | [BUGFIX] Fix dp size > 1 for qwen3 vl model | model wrapper, attention/backend, multimodal/processor, scheduler/runtime | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py` |
| 2026-01-31 | [#18024](https://github.com/sgl-project/sglang/pull/18024) | merged | fix: correct weight loading prefix mapping for Qwen3-VL | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-03 | [#18185](https://github.com/sgl-project/sglang/pull/18185) | merged | [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2026-02-12 | [#18721](https://github.com/sgl-project/sglang/pull/18721) | open | [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models | model wrapper | `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-13 | [#18771](https://github.com/sgl-project/sglang/pull/18771) | open | Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-02-19 | [#19003](https://github.com/sgl-project/sglang/pull/19003) | merged | [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend | model wrapper, attention/backend, multimodal/processor, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-02-24 | [#19242](https://github.com/sgl-project/sglang/pull/19242) | open | [feat] feat: add Qwen3-ASR support like whisper | multimodal/processor, docs/config | `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py` |
| 2026-02-25 | [#19291](https://github.com/sgl-project/sglang/pull/19291) | merged | [Qwen3.5] Fix missing `quant_config` in `Qwen3VL` | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-25 | [#19333](https://github.com/sgl-project/sglang/pull/19333) | merged | fix qwen3_vl visual module loading | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-02 | [#19693](https://github.com/sgl-project/sglang/pull/19693) | open | [NPU] Fix Qwen3-VL-8B Accuracy for NPU | model wrapper, attention/backend, MoE/router, scheduler/runtime | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py` |
| 2026-03-17 | [#20759](https://github.com/sgl-project/sglang/pull/20759) | merged | [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-17 | [#20788](https://github.com/sgl-project/sglang/pull/20788) | merged | [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-18 | [#20857](https://github.com/sgl-project/sglang/pull/20857) | open | add EVS support for Qwen3-VL | model wrapper, multimodal/processor, docs/config | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-26 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-03-26 | [#21469](https://github.com/sgl-project/sglang/pull/21469) | merged | [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct | model wrapper, MoE/router, tests/benchmarks | `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-01 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | multimodal/processor, tests/benchmarks | `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-03 | [#22038](https://github.com/sgl-project/sglang/pull/22038) | merged | [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer | model wrapper, multimodal/processor, scheduler/runtime | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-04-03 | [#22052](https://github.com/sgl-project/sglang/pull/22052) | open | [Fix] Enable precise embedding interpolation by default for Qwen3-VL | model wrapper, docs/config | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-03 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-04 | [#22089](https://github.com/sgl-project/sglang/pull/22089) | merged | [Feature] Add chunk-based streaming ASR for Qwen3-ASR | multimodal/processor | `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py` |
| 2026-04-07 | [#22230](https://github.com/sgl-project/sglang/pull/22230) | merged | [Feature] Support eagle3 for qwen3-vl | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-04-07 | [#22266](https://github.com/sgl-project/sglang/pull/22266) | merged | [NPU] fix qwen3.5 video processor | multimodal/processor | `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` |
| 2026-04-09 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | multimodal/processor | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-15 | [#22839](https://github.com/sgl-project/sglang/pull/22839) | open | fix(config): Add from_dict() for Qwen3VL config classes | tests/benchmarks, docs/config | `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py` |
| 2026-04-15 | [#22848](https://github.com/sgl-project/sglang/pull/22848) | open | [Feature] WebSocket streaming audio input for ASR | model wrapper, tests/benchmarks | `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py` |
| 2026-04-18 | [#23115](https://github.com/sgl-project/sglang/pull/23115) | open | fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-20 | [#23220](https://github.com/sgl-project/sglang/pull/23220) | open | Bugfix: Qwen3-VL-MoE adapt encoder_only | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-21 | [#23304](https://github.com/sgl-project/sglang/pull/23304) | closed | [Bugfix] Fix Qwen3-VL rope config compatibility | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-22 | [#23469](https://github.com/sgl-project/sglang/pull/23469) | open | [NPU] adapt the Qwen3-ASR model for deployment on NPU | misc | `python/sglang/srt/utils/common.py` |

## Diff Cards

### PR #10911 - model: qwen3-omni (thinker-only)

- Link: https://github.com/sgl-project/sglang/pull/10911
- Status/date: `merged`, created 2025-09-25, merged 2025-10-16; author `mickqian`.
- Diff scope read: `16` files, `+1947/-328`; areas: model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config; keywords: vision, attention, moe, config, cache, quant, expert, processor, spec, test.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: +# Copyright 2025 Qwen Team; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: +from transformers import PretrainedConfig; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, __init__
  - `python/sglang/srt/layers/rotary_embedding.py` modified +357/-2 (359 lines); hunks: def get_rope_index(; def get_rope_index(; symbols: get_rope_index, get_rope_index, get_rope_index, get_rope_index_qwen3_omni
  - `test/srt/test_vision_openai_server_common.py` modified +132/-96 (228 lines); hunks: import base64; AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"; symbols: TestOpenAIOmniServerBase, TestOpenAIMLLMServerBase, setUpClass, get_or_download_file
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: # ==============================================================================; class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: vision, attention, moe, config, cache, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- Link: https://github.com/sgl-project/sglang/pull/10985
- Status/date: `merged`, created 2025-09-27, merged 2025-10-01; author `yhyang201`.
- Diff scope read: `2` files, `+14/-2`; areas: model wrapper, MoE/router; keywords: cache, kv, attention, config, moe, quant, topk.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: from sglang.srt.layers.moe.topk import TopK; def __init__(; symbols: __init__, forward_prepare, forward_core
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunks: def forward(; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: cache, kv, attention, config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12261 - [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl

- Link: https://github.com/sgl-project/sglang/pull/12261
- Status/date: `open`, created 2025-10-28; author `gjghfd`.
- Diff scope read: `1` files, `+5/-5`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_5_vl.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_5_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12333 - [PP] Add pp support for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/12333
- Status/date: `merged`, created 2025-10-29, merged 2025-12-17; author `XucSh`.
- Diff scope read: `5` files, `+119/-20`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: mla, test, attention, config, expert, fp4, moe, processor, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunks: get_tensor_model_parallel_rank,; def __init__(; symbols: __init__, __init__, forward, forward
  - `test/srt/test_pp_single_node.py` modified +57/-0 (57 lines); hunks: python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k; DEFAULT_MLA_MODEL_NAME_FOR_TEST,; symbols: test_mgsm_en, TestQwenVLPPAccuracy, setUpClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
  - `python/sglang/srt/managers/schedule_policy.py` modified +4/-0 (4 lines); hunks: def _update_prefill_budget(; symbols: _update_prefill_budget, add_chunked_req
  - `python/sglang/test/test_utils.py` modified +1/-0 (1 lines); hunks: DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN = "lmsys/sglang-ci-dsv3-test-NextN"
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py`; keywords observed in patches: mla, test, attention, config, expert, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12554 - [Docs] Add docs for Qwen3-VL image and video support

- Link: https://github.com/sgl-project/sglang/pull/12554
- Status/date: `merged`, created 2025-11-03, merged 2025-11-10; author `adarshxs`.
- Diff scope read: `2` files, `+131/-0`; areas: docs/config; keywords: doc, attention, cache, cuda, flash, fp8, quant, spec, test, vision.
- Code diff details:
  - `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunks: +# Qwen3-VL Usage
  - `docs/index.rst` modified +1/-0 (1 lines); hunks: Its core features include:
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/qwen3_vl.md`, `docs/index.rst`; keywords observed in patches: doc, attention, cache, cuda, flash, fp8. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/qwen3_vl.md`, `docs/index.rst`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12662 - [CPU] Add support for Qwen3-vl and Qwen3-omni

- Link: https://github.com/sgl-project/sglang/pull/12662
- Status/date: `open`, created 2025-11-05; author `blzheng`.
- Diff scope read: `12` files, `+496/-55`; areas: model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, docs/config; keywords: attention, config, vision, kv, quant, cuda, expert, flash, moe, processor.
- Code diff details:
  - `sgl-kernel/csrc/cpu/gemm.cpp` modified +142/-0 (142 lines); hunks: void weight_packed_linear_kernel_impl(; at::Tensor fused_linear_sigmoid_mul(; symbols: int64_t, int64_t, int64_t
  - `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunks: from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size; with_multi_stream,; symbols: forward, VisionAMXAttention, __init__, forward
  - `python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunks: def adjust_config_with_unaligned_cpu_tp(; symbols: adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunks: from sglang.srt.multimodal.processors.base_processor import (; FPS_MAX_FRAMES = 768; symbols: hacked_preprocess, smart_resize
  - `python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunks: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; logger = logging.getLogger(__name__); symbols: Qwen3_VisionMLP, __init__, __init__, forward
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`; keywords observed in patches: attention, config, vision, kv, quant, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12703 - add qwen3-omni docs

- Link: https://github.com/sgl-project/sglang/pull/12703
- Status/date: `open`, created 2025-11-05; author `jiapingW`.
- Diff scope read: `2` files, `+150/-0`; areas: docs/config; keywords: doc, spec, test.
- Code diff details:
  - `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunks: +# Qwen3-Omni Usage
  - `docs/index.rst` modified +1/-0 (1 lines); hunks: Its core features include:
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`; keywords observed in patches: doc, spec, test. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13724 - support qwen3_vl vision model dp

- Link: https://github.com/sgl-project/sglang/pull/13724
- Status/date: `merged`, created 2025-11-21, merged 2025-11-28; author `Lzhang-hub`.
- Diff scope read: `2` files, `+50/-2`; areas: model wrapper, tests/benchmarks; keywords: test, attention, awq, config, moe, processor, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: ); from sglang.srt.model_loader.weight_utils import default_weight_loader; symbols: __init__, __init__, __init__, __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: MODELS = [
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py`; keywords observed in patches: test, attention, awq, config, moe, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13736 - [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series

- Link: https://github.com/sgl-project/sglang/pull/13736
- Status/date: `merged`, created 2025-11-21, merged 2025-11-22; author `yuan-luo`.
- Diff scope read: `5` files, `+169/-13`; areas: model wrapper, tests/benchmarks; keywords: processor, test, attention, benchmark, config, fp8, quant.
- Code diff details:
  - `test/srt/ops/test_repeat_interleave.py` added +141/-0 (141 lines); hunks: +import time; symbols: torch_ref_impl, benchmark_once, _generate_random_grid, TestRepeatInterleave:
  - `python/sglang/srt/models/utils.py` modified +23/-0 (23 lines); hunks: # limitations under the License.; def permute_inv(perm: torch.Tensor) -> torch.Tensor:; symbols: permute_inv, compute_cu_seqlens_from_grid_numpy
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; symbols: forward
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: TestFile("openai_server/validation/test_matched_stop.py", 60),
- Optimization/support interpretation: The concrete diff surface is `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: processor, test, attention, benchmark, config, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13918 - [VLM] support qwen3-vl eagle infer

- Link: https://github.com/sgl-project/sglang/pull/13918
- Status/date: `open`, created 2025-11-25; author `Lzhang-hub`.
- Diff scope read: `2` files, `+30/-3`; areas: model wrapper; keywords: config, eagle, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunks: def __init__(; def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`; keywords observed in patches: config, eagle, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14292 - [VLM] Introduce Cache for positional embedding ids for Qwen-VL family

- Link: https://github.com/sgl-project/sglang/pull/14292
- Status/date: `merged`, created 2025-12-02, merged 2025-12-04; author `yuan-luo`.
- Diff scope read: `3` files, `+48/-47`; areas: model wrapper; keywords: vision, cache, moe.
- Code diff details:
  - `python/sglang/srt/models/utils.py` modified +38/-0 (38 lines); hunks: # limitations under the License.; def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens_from_grid_numpy, RotaryPosMixin:, rot_pos_ids
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen2_5_VisionTransformer, Qwen2_5_VisionTransformer, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen3VLMoeVisionModel, Qwen3VLMoeVisionModel, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: vision, cache, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14886 - Support qwen3-omni with DP Encoder

- Link: https://github.com/sgl-project/sglang/pull/14886
- Status/date: `open`, created 2025-12-11; author `apinge`.
- Diff scope read: `2` files, `+32/-3`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: attention, config, cuda, expert, moe, quant, test, triton, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunks: Qwen3OmniMoeVisionEncoderConfig,; Qwen3VLMoeForConditionalGeneration,; symbols: __init__, __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: register_cuda_ci(est_time=500, suite="nightly-4-gpu", nightly=True)
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`; keywords observed in patches: attention, config, cuda, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14907 - [VLM] Support chunked vit attention

- Link: https://github.com/sgl-project/sglang/pull/14907
- Status/date: `merged`, created 2025-12-11, merged 2025-12-15; author `yuan-luo`.
- Diff scope read: `2` files, `+363/-8`; areas: model wrapper; keywords: cache, eagle, processor, vision.
- Code diff details:
  - `python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunks: _GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None; def _get_precomputed_embedding(; symbols: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunks: # ==============================================================================; from sglang.srt.models.utils import RotaryPosMixin, compute_cu_seqlens_from_gr; symbols: get_image_feature, get_video_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: cache, eagle, processor, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/15205
- Status/date: `merged`, created 2025-12-15, merged 2025-12-18; author `yuan-luo`.
- Diff scope read: `4` files, `+100/-80`; areas: model wrapper, attention/backend, multimodal/processor; keywords: cache, vision, config, processor, quant, attention.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; symbols: forward, forward, forward, Glm4vVisionRotaryEmbedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: import torch.nn as nn; from sglang.srt.layers.logits_processor import LogitsProcessor; symbols: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`; keywords observed in patches: cache, vision, config, processor, quant, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15320 - [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/15320
- Status/date: `merged`, created 2025-12-17, merged 2025-12-20; author `yuan-luo`.
- Diff scope read: `6` files, `+233/-64`; areas: model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks; keywords: cuda, vision, attention, cache, flash, processor, spec, test, triton.
- Code diff details:
  - `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` modified +175/-57 (232 lines); hunks: from __future__ import annotations; class ViTCudaGraphRunner:; symbols: ViTCudaGraphRunner:, __init__, __init__, _get_graph_key
  - `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunks: compute_cu_seqlens_from_grid_numpy,; def forward(; symbols: forward, forward, __init__, dtype
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: def forward(; def forward(; symbols: forward, forward, forward
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-2 (4 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/distributed/parallel_state.py` modified +1/-1 (2 lines); hunks: def _all_reduce_out_place(; symbols: _all_reduce_out_place
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`; keywords observed in patches: cuda, vision, attention, cache, flash, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16366 - Optimize Qwen3-VL video memory usage

- Link: https://github.com/sgl-project/sglang/pull/16366
- Status/date: `merged`, created 2026-01-04, merged 2026-01-22; author `cen121212`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunks: def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:; symbols: get_image_feature, get_video_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16491 - [Qwen3-VL][PP] Skip loading expert weights not on this rank

- Link: https://github.com/sgl-project/sglang/pull/16491
- Status/date: `open`, created 2026-01-05; author `MtFitzRoy`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper, MoE/router; keywords: expert, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl_moe.py`; keywords observed in patches: expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16571 - [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT

- Link: https://github.com/sgl-project/sglang/pull/16571
- Status/date: `open`, created 2026-01-06; author `qichu-yun`.
- Diff scope read: `2` files, `+87/-15`; areas: model wrapper; keywords: cuda, attention, moe, processor, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunks: get_attention_tp_size,; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; symbols: Qwen3_VisionMLP, __init__, forward, forward
  - `python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunks: ); def __init__(; symbols: __init__, forward_cuda, forward_cpu, forward_aiter
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`; keywords observed in patches: cuda, attention, moe, processor, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16785 - [Bugfix] fix recompile in qwen3 vl

- Link: https://github.com/sgl-project/sglang/pull/16785
- Status/date: `open`, created 2026-01-09; author `narutolhy`.
- Diff scope read: `5` files, `+113/-36`; areas: model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks; keywords: config, cache, cuda, attention, deepep, mla, moe, test, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunks: def __init__(; def forward(; symbols: __init__, get_deepstack_embeds, forward, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunks: def __init__(; def forward(; symbols: __init__, get_input_embeddings, get_deepstack_embeds, forward
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunks: def __init__(self, model_runner: ModelRunner):; def warmup_torch_compile(self, num_tokens: int):; symbols: __init__, warmup_torch_compile, warmup_torch_compile, _cache_loc_dtype
  - `python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunks: def embed_mm_inputs(; def embed_mm_inputs(; symbols: embed_mm_inputs, embed_mm_inputs, general_mm_embed_routine, general_mm_embed_routine
  - `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3 (8 lines); hunks: ); def run_mmmu_eval(; symbols: run_mmmu_eval, _run_vlm_mmmu_test
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; keywords observed in patches: config, cache, cuda, attention, deepep, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16996 - feat: Support 'use_audio_in_video' option for qwen3omnimoe model

- Link: https://github.com/sgl-project/sglang/pull/16996
- Status/date: `open`, created 2026-01-13; author `srLi24`.
- Diff scope read: `6` files, `+129/-12`; areas: multimodal/processor; keywords: moe, processor, config, spec, test, vision.
- Code diff details:
  - `python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunks: from unittest import SkipTest; def load_audio(; symbols: load_audio, extract_audio_via_av, ImageData:, get_image_bytes
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunks: def process_mm_data(; def _load_single_item(; symbols: process_mm_data, _load_single_item, _load_single_item, submit_data_loading_tasks
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunks: async def preprocess_video(; async def preprocess_video(; symbols: preprocess_video, preprocess_video, __init__, process_mm_data_async
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunks: class ChatCompletionRequest(BaseModel):; def get_param(param_name: str):; symbols: ChatCompletionRequest, get_param
  - `python/sglang/srt/managers/io_struct.py` modified +3/-0 (3 lines); hunks: class GenerateReqInput(BaseReq, APIServingTimingMixin):; symbols: GenerateReqInput, contains_mm_input
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: moe, processor, config, spec, test, vision. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17202 - [Feat] Accelerate qwen3vl by remove cpu op

- Link: https://github.com/sgl-project/sglang/pull/17202
- Status/date: `open`, created 2026-01-16; author `ZLkanyo009`.
- Diff scope read: `2` files, `+27/-9`; areas: attention/backend, multimodal/processor; keywords: attention, kv, vision.
- Code diff details:
  - `python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunks: def embed_mm_inputs(; symbols: embed_mm_inputs
  - `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py`; keywords observed in patches: attention, kv, vision. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17276 - Add Qwen3VL Eagle3 Inference Support

- Link: https://github.com/sgl-project/sglang/pull/17276
- Status/date: `open`, created 2026-01-18; author `ardenma`.
- Diff scope read: `1` files, `+35/-0`; areas: model wrapper; keywords: config, eagle, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunks: def __init__(; def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config, eagle, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- Link: https://github.com/sgl-project/sglang/pull/17624
- Status/date: `merged`, created 2026-01-23, merged 2026-01-30; author `zju-stu-lizheng`.
- Diff scope read: `5` files, `+48/-19`; areas: model wrapper, attention/backend, multimodal/processor, scheduler/runtime; keywords: attention, vision, config, quant, cuda, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunks: from transformers.activations import ACT2FN; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunks: def run_dp_sharded_mrope_vision_model(; def run_dp_sharded_mrope_vision_model(; symbols: run_dp_sharded_mrope_vision_model, run_dp_sharded_mrope_vision_model
  - `python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunks: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; def __init__(; symbols: __init__, __init__, forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunks: def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):; symbols: _pad_inputs_to_size
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-0 (2 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`; keywords observed in patches: attention, vision, config, quant, cuda, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18024 - fix: correct weight loading prefix mapping for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/18024
- Status/date: `merged`, created 2026-01-31, merged 2026-02-02; author `Lollipop`.
- Diff scope read: `1` files, `+7/-1`; areas: model wrapper; keywords: config.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18185 - [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker

- Link: https://github.com/sgl-project/sglang/pull/18185
- Status/date: `merged`, created 2026-02-03, merged 2026-03-14; author `yuan-luo`.
- Diff scope read: `1` files, `+52/-28`; areas: model wrapper, MoE/router; keywords: attention, config, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunks: def __init__(; def forward(; symbols: __init__, forward, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_omni_moe.py`; keywords observed in patches: attention, config, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_omni_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18721 - [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models

- Link: https://github.com/sgl-project/sglang/pull/18721
- Status/date: `open`, created 2026-02-12; author `kousakawang`.
- Diff scope read: `2` files, `+3/-1`; areas: model wrapper; keywords: attention, config, quant.
- Code diff details:
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: attention, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18771 - Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton

- Link: https://github.com/sgl-project/sglang/pull/18771
- Status/date: `open`, created 2026-02-13; author `AwesomeKeyboard`.
- Diff scope read: `1` files, `+1/-0`; areas: MoE/router, kernel, tests/benchmarks; keywords: benchmark, config, expert, moe, topk, triton.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: def get_model_config(; symbols: get_model_config
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/common_utils.py`; keywords observed in patches: benchmark, config, expert, moe, topk, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/common_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19003 - [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend

- Link: https://github.com/sgl-project/sglang/pull/19003
- Status/date: `merged`, created 2026-02-19, merged 2026-02-24; author `yuan-luo`.
- Diff scope read: `4` files, `+678/-14`; areas: model wrapper, attention/backend, multimodal/processor, tests/benchmarks; keywords: attention, flash, cuda, cache, test, vision, benchmark, config, kv, processor.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunks: from sglang.srt.distributed import get_tensor_model_parallel_world_size; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; symbols: Qwen3_VisionMLP, __init__, __init__, __init__
  - `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunks: +import argparse; symbols: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test
  - `python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunks: _is_hip = is_hip(); "normal": apply_rotary_pos_emb,; symbols: SingletonCache:, forward, VisionFlashInferAttention, __init__
  - `python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunks: def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py`; keywords observed in patches: attention, flash, cuda, cache, test, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19242 - [feat] feat: add Qwen3-ASR support like whisper

- Link: https://github.com/sgl-project/sglang/pull/19242
- Status/date: `open`, created 2026-02-24; author `LuYanFCP`.
- Diff scope read: `5` files, `+475/-0`; areas: multimodal/processor, docs/config; keywords: config, attention, moe, processor, cache, spec, vision.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunks: +import logging; symbols: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length
  - `python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunks: +"""; symbols: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: from sglang.srt.configs.nemotron_h import NemotronHConfig; "Olmo3Config",
  - `python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunks: def is_generation_model(model_architectures: List[str], is_embedding: bool = Fal; def is_image_gen_model(model_architectures: List[str]):; symbols: is_generation_model, is_image_gen_model, is_audio_model
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: NemotronH_Nano_VL_V2_Config,; JetNemotronConfig,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`; keywords observed in patches: config, attention, moe, processor, cache, spec. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19291 - [Qwen3.5] Fix missing `quant_config` in `Qwen3VL`

- Link: https://github.com/sgl-project/sglang/pull/19291
- Status/date: `merged`, created 2026-02-25, merged 2026-03-02; author `mmangkad`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: config, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19333 - fix qwen3_vl visual module loading

- Link: https://github.com/sgl-project/sglang/pull/19333
- Status/date: `merged`, created 2026-02-25, merged 2026-02-27; author `WingEdge777`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: attention, kv, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: attention, kv, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19693 - [NPU] Fix Qwen3-VL-8B Accuracy for NPU

- Link: https://github.com/sgl-project/sglang/pull/19693
- Status/date: `open`, created 2026-03-02; author `Todobe`.
- Diff scope read: `14` files, `+199/-108`; areas: model wrapper, attention/backend, MoE/router, scheduler/runtime; keywords: kv, attention, cache, config, cuda, mla, spec, eagle, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunks: from functools import lru_cache, partial; def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, forward, forward
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +37/-38 (75 lines); hunks: from sglang.srt.layers.radix_attention import AttentionType; def forward_decode_graph(; symbols: forward_decode_graph, forward_decode_graph
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: maybe_remap_kv_scale_name,; def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` modified +12/-23 (35 lines); hunks: from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa; def _capture_graph(self, graph, pool, stream, run_once_fn):; symbols: _capture_graph, _get_update_attr_name, _get_update_attr_type, _get_update_attr_name
  - `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py` modified +11/-10 (21 lines); hunks: from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa; def _capture_graph(self, graph, pool, stream, run_once_fn):; symbols: _capture_graph, _get_update_attr_name, _get_update_attr_name, _get_update_attr_type
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py`; keywords observed in patches: kv, attention, cache, config, cuda, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20759 - [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable

- Link: https://github.com/sgl-project/sglang/pull/20759
- Status/date: `merged`, created 2026-03-17, merged 2026-03-19; author `ZLkanyo009`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper; keywords: attention, config, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: attention, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20788 - [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL

- Link: https://github.com/sgl-project/sglang/pull/20788
- Status/date: `merged`, created 2026-03-17, merged 2026-03-18; author `jianan-gu`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: attention, config, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: attention, config, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20857 - add EVS support for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/20857
- Status/date: `open`, created 2026-03-18; author `artetaout`.
- Diff scope read: `5` files, `+151/-4`; areas: model wrapper, multimodal/processor, docs/config; keywords: config, vision, processor, cuda, moe, quant, spec.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunks: from PIL import Image; from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration; symbols: __init__, __init__, _maybe_apply_qwen3_evs, get_mm_data
  - `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunks: def get_rope_index(; def get_rope_index(; symbols: get_rope_index, get_rope_index, get_rope_index
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunks: WeightsMapper,; def forward(; symbols: forward, Qwen3VLForConditionalGeneration, Qwen3VLForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0 (2 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config, vision, processor, cuda, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- Link: https://github.com/sgl-project/sglang/pull/21458
- Status/date: `merged`, created 2026-03-26, merged 2026-04-01; author `yctseng0211`.
- Diff scope read: `1` files, `+101/-3`; areas: model wrapper; keywords: attention, cache, config, cuda, kv, quant.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: from sglang.srt.layers.quantization.base_config import QuantizationConfig; from sglang.srt.models.qwen2 import Qwen2Model; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, cache, config, cuda, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21469 - [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct

- Link: https://github.com/sgl-project/sglang/pull/21469
- Status/date: `merged`, created 2026-03-26, merged 2026-04-01; author `yushengsu-thu`.
- Diff scope read: `3` files, `+152/-235`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: attention, lora, moe, cache, expert, kv, spec, test, config, cuda.
- Code diff details:
  - `test/manual/lora/test_lora_qwen3_vl.py` removed +0/-233 (233 lines); hunks: -import random; symbols: TestLoRAQwen3VLGating, _assert_pattern, test_qwen3_vl_should_apply_lora_regex, test_qwen3_vl_moe_should_apply_lora_regex
  - `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunks: def __init__(; symbols: __init__, should_apply_lora
- Optimization/support interpretation: The concrete diff surface is `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; keywords observed in patches: attention, lora, moe, cache, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #22038 - [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer

- Link: https://github.com/sgl-project/sglang/pull/22038
- Status/date: `merged`, created 2026-04-03, merged 2026-04-04; author `yhyang201`.
- Diff scope read: `7` files, `+167/-410`; areas: model wrapper, multimodal/processor, scheduler/runtime; keywords: cache, vision, processor, attention, cuda, eagle, spec.
- Code diff details:
  - `python/sglang/srt/managers/mm_utils.py` modified +147/-286 (433 lines); hunks: _GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None; def _get_precomputed_embedding(; symbols: _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _move_items_to_device, _get_chunked_prefill_embedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunks: """Inference-only Qwen3-VL model compatible with HuggingFace weights."""; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; symbols: get_image_feature, get_video_feature
  - `python/sglang/srt/managers/schedule_batch.py` modified +0/-15 (15 lines); hunks: def prepare_for_extend(self):; symbols: prepare_for_extend
  - `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunks: def set(; symbols: set, get_single, has
  - `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunks: def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py`; keywords observed in patches: cache, vision, processor, attention, cuda, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22052 - [Fix] Enable precise embedding interpolation by default for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/22052
- Status/date: `open`, created 2026-04-03; author `chengmengli06`.
- Diff scope read: `3` files, `+10/-11`; areas: model wrapper, docs/config; keywords: vision, moe, cache, config, doc, fp8, kv.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunks: def __init__(; def fast_pos_embed_interpolate_from_list(self, grid_thw):; symbols: __init__, fast_pos_embed_interpolate_from_list
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; keywords observed in patches: vision, moe, cache, config, doc, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #22089 - [Feature] Add chunk-based streaming ASR for Qwen3-ASR

- Link: https://github.com/sgl-project/sglang/pull/22089
- Status/date: `merged`, created 2026-04-04, merged 2026-04-09; author `SammLSH`.
- Diff scope read: `5` files, `+263/-2`; areas: multimodal/processor; keywords: config, processor, spec, cache, kv.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +125/-0 (125 lines); hunks: from __future__ import annotations; TranscriptionVerboseResponse,; symbols: _handle_streaming_request, _generate_transcription_stream, _generate_chunked_asr_stream
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` added +93/-0 (93 lines); hunks: +import io; symbols: StreamingASRState:, get_prefix_text, update, finalize
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py` modified +23/-0 (23 lines); hunks: class TranscriptionAdapter(ABC):; symbols: TranscriptionAdapter, build_sampling_params, supports_chunked_streaming, prompt_template
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunks: TranscriptionAdapter,; symbols: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunks: AUDIO_PLACEHOLDER = "<\|audio_start\|><\|audio_pad\|><\|audio_end\|>"; def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:; symbols: _build_transcription_prompt, compute_mrope_positions
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`; keywords observed in patches: config, processor, spec, cache, kv. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22230 - [Feature] Support eagle3 for qwen3-vl

- Link: https://github.com/sgl-project/sglang/pull/22230
- Status/date: `merged`, created 2026-04-07, merged 2026-04-09; author `litmei`.
- Diff scope read: `1` files, `+24/-0`; areas: model wrapper; keywords: config, eagle, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunks: def __init__(; def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config, eagle, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22266 - [NPU] fix qwen3.5 video processor

- Link: https://github.com/sgl-project/sglang/pull/22266
- Status/date: `merged`, created 2026-04-07, merged 2026-04-08; author `zhaozx-cn`.
- Diff scope read: `1` files, `+177/-21`; areas: multimodal/processor; keywords: processor, test.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunks: group_images_by_shape,; def _preprocess(; symbols: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, _preprocess
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`; keywords observed in patches: processor, test. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- Link: https://github.com/sgl-project/sglang/pull/22431
- Status/date: `merged`, created 2026-04-09, merged 2026-04-18; author `lkhl`.
- Diff scope read: `1` files, `+1/-1`; areas: multimodal/processor; keywords: processor.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: async def preprocess_video(; symbols: preprocess_video
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`; keywords observed in patches: processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/qwen_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22839 - fix(config): Add from_dict() for Qwen3VL config classes

- Link: https://github.com/sgl-project/sglang/pull/22839
- Status/date: `open`, created 2026-04-15; author `libermeng`.
- Diff scope read: `5` files, `+306/-0`; areas: tests/benchmarks, docs/config; keywords: config, moe, vision, attention, expert, topk, kv, router, test.
- Code diff details:
  - `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunks: +"""Unit tests for qwen3_vl and qwen3_5 config from_dict() handling.; symbols: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion
  - `python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunks: class Qwen3_5Config(PretrainedConfig):; class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; symbols: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunks: class Qwen3VLConfig(PretrainedConfig):; def __init__(; symbols: Qwen3VLConfig, from_dict, __init__, __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0 (4 lines); hunks: Qwen3_5Config,; DeepseekVLV2Config,
  - `python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunks: from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig; "JetVLMConfig",
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`; keywords observed in patches: config, moe, vision, attention, expert, topk. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22848 - [Feature] WebSocket streaming audio input for ASR

- Link: https://github.com/sgl-project/sglang/pull/22848
- Status/date: `open`, created 2026-04-15; author `SammLSH`.
- Diff scope read: `6` files, `+937/-43`; areas: model wrapper, tests/benchmarks; keywords: config, cache, kv, processor, test.
- Code diff details:
  - `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunks: """; TEST_AUDIO_ZH_URL = (; symbols: _normalize_for_wer, _wer, download_audio, download_audio
  - `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +376/-0 (376 lines); hunks: +"""WebSocket transport for OpenAI Realtime API-style transcription.; symbols: names, _safe_close_websocket, _pcm_to_wav, RealtimeMessageType
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunks: +import asyncio; class StreamingASRState:; symbols: StreamingASRState:, StreamingASRState:, get_prefix_text, _record_emit
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunks: import uuid; TranscriptionVerboseResponse,; symbols: _generate_chunked_asr_stream, _generate_chunked_asr_stream, _generate_chunked_asr_stream, handle_websocket
  - `python/sglang/srt/server_args.py` modified +10/-0 (10 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
- Optimization/support interpretation: The concrete diff surface is `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`; keywords observed in patches: config, cache, kv, processor, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23115 - fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights

- Link: https://github.com/sgl-project/sglang/pull/23115
- Status/date: `open`, created 2026-04-18; author `octo-patch`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper, MoE/router; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl_moe.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23220 - Bugfix: Qwen3-VL-MoE adapt encoder_only

- Link: https://github.com/sgl-project/sglang/pull/23220
- Status/date: `open`, created 2026-04-20; author `Hide-on-bushsh`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper, MoE/router; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl_moe.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23304 - [Bugfix] Fix Qwen3-VL rope config compatibility

- Link: https://github.com/sgl-project/sglang/pull/23304
- Status/date: `closed`, created 2026-04-21, closed 2026-04-21; author `Chokoyo`.
- Diff scope read: `1` files, `+1/-10`; areas: model wrapper; keywords: attention, config.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3.py`; keywords observed in patches: attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23469 - [NPU] adapt the Qwen3-ASR model for deployment on NPU

- Link: https://github.com/sgl-project/sglang/pull/23469
- Status/date: `open`, created 2026-04-22; author `xdtbynd`.
- Diff scope read: `1` files, `+18/-0`; areas: misc; keywords: cuda.
- Code diff details:
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: def load_audio(; symbols: load_audio
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/utils/common.py`; keywords observed in patches: cuda. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/utils/common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
