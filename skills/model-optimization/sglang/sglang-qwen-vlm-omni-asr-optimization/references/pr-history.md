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
