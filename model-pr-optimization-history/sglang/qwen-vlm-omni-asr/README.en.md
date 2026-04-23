# SGLang Qwen VLM / Omni / ASR Support and Optimization History

Scope: Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, Qwen3-ASR, and Qwen3.5 multimodal paths that share Qwen VLM processors or encoder-disaggregation code.

Evidence policy:

- Every SGLang PR listed here was inspected through `gh pr view` and `gh pr diff --patch`, or through the merged commit diff when that was more precise.
- The full PR-by-PR source dossier lives in `skills/model-optimization/sglang/sglang-qwen-vlm-omni-asr-optimization/references/pr-history.md`.
- This README keeps the timeline readable while preserving motivation, implementation, and key code anchors.
- Open or closed-unmerged PRs are radar items, not current-main behavior.

## Summary

Qwen multimodal optimization is not mainly a text-only forward problem. The risk is in processors, ViT attention/cache, mRoPE/3D mRoPE, DeepStack, encoder DP/PP/EPD, multimodal cache and transfer, hardware backends, Qwen3-Omni audio, and Qwen3-ASR streaming.

## Mainline Timeline

### #10911 Qwen3-Omni thinker-only

- Link: https://github.com/sgl-project/sglang/pull/10911, merged.
- Motivation: register Qwen3-Omni nested thinker/talker/code2wav config and route audio inputs through the Qwen multimodal path.
- Implementation: add `Qwen3OmniMoeConfig`, register `Qwen3OmniMoeForConditionalGeneration`, process audio in `base_processor.py`, and pass audio sequence lengths into mRoPE.
- Key code:

```python
audio_feature_lengths = torch.sum(audio_item.feature_attention_mask, dim=1)
MRotaryEmbedding.get_rope_index(..., audio_seqlens=audio_feature_lengths)
```

### #10985 Qwen3-VL MRotaryEmbedding arg fix

- Link: https://github.com/sgl-project/sglang/pull/10985, merged.
- Motivation: fused KV-buffer arguments reached `MRotaryEmbedding`, which does not support that cache-save path.
- Implementation: accept the optional argument but assert it is unused; disable fused KV compatibility for mRoPE.
- Key code:

```python
self.compatible_with_fused_kv_buffer = (
    False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
)
```

### #12333 Qwen3-VL PP support

- Link: https://github.com/sgl-project/sglang/pull/12333, merged.
- Motivation: Qwen3-VL needed PP-aware lm_head ownership, media embedding routing, logits, and rank-local weight loading.
- Implementation: add `pp_group`, create `lm_head` only on the last PP rank, return hidden states on middle ranks, and load tied head weights only where the head exists.
- Key code:

```python
if self.pp_group.is_last_rank:
    self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

### #13724 Qwen3-VL vision encoder DP

- Link: https://github.com/sgl-project/sglang/pull/13724, merged.
- Motivation: ViT was a TTFT bottleneck for image/video concurrency.
- Implementation: pass `use_data_parallel` through vision layers, force local TP settings in DP mode, and use `run_dp_sharded_mrope_vision_model(..., rope_type="rope_3d")`.
- Key code:

```python
return run_dp_sharded_mrope_vision_model(
    self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
)
```

### #13736 NumPy cu_seqlens for Qwen-VL

- Link: https://github.com/sgl-project/sglang/pull/13736, merged.
- Motivation: CPU `torch.repeat_interleave` in ViT cu_seqlens appeared in TTFT profiles.
- Implementation: add `compute_cu_seqlens_from_grid_numpy` and reuse it from Qwen2/Qwen3 VLM.
- Key code:

```python
cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
    axis=0, dtype=np.int32
)
```

### #14292 rotary position-id cache

- Link: https://github.com/sgl-project/sglang/pull/14292, merged.
- Motivation: avoid rebuilding 2D rotary position IDs for repeated image shapes.
- Implementation: add cached `RotaryPosMixin.rot_pos_ids(...)` and use it in Qwen2.5-VL and Qwen3-VL.
- Key code:

```python
@lru_cache(maxsize=1024)
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
```

### #14907 chunked ViT attention

- Link: https://github.com/sgl-project/sglang/pull/14907, merged.
- Motivation: very large image/frame counts could OOM when ViT ran all patches at once.
- Implementation: chunk by image and patch limits, run visual encoder per chunk, then concatenate.
- Key code:

```python
chunk_embeds = self.visual(pixel_chunk, grid_thw=grid_chunk)
return torch.cat(all_chunk_embeds, dim=0)
```

### #15205 vision RoPE cos/sin cache

- Link: https://github.com/sgl-project/sglang/pull/15205, merged.
- Motivation: Qwen3-VL and GLM-4.1V recomputed vision RoPE cos/sin; the PR reports a micro-path drop from about 490 us to 186 us and roughly 2% TTFT improvement.
- Implementation: expose `RotaryEmbedding.get_cos_sin`, let `VisionAttention` accept explicit cos/sin, and make Qwen3-VL index cached cos/sin tables.
- Key code:

```python
cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
cos_combined = cos[pos_ids].flatten(1)
```

### #15320 ViT piecewise CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/15320, merged.
- Motivation: capture Qwen3-VL ViT compute, including TP>1 and DeepStack, in PCG. The PR reports TTFT 1384.53 ms to 1120.68 ms on 8xH20 Qwen3-VL-8B TP4.
- Implementation: add `forward_with_cuda_graph`, relax TP restrictions, and extend `ViTCudaGraphRunner` for DeepStack visual indexes.
- Key code:

```python
if get_bool_env_var("SGLANG_VIT_ENABLE_CUDA_GRAPH"):
    return self.forward_with_cuda_graph(x, grid_thw)
```

### #16366 video memory offload

- Link: https://github.com/sgl-project/sglang/pull/16366, merged.
- Motivation: per-item video features stayed on device after concatenation and caused high-concurrency OOM.
- Implementation: move features to device only for concat, then move per-item features back to CPU.
- Key code:

```python
for item in items:
    item.feature = item.feature.to("cpu")
```

### #17624 DP size > 1

- Link: https://github.com/sgl-project/sglang/pull/17624, merged.
- Motivation: DP encoder plus DP attention failed when TP and DP sizes differed; mRoPE padding used the wrong dimension.
- Implementation: pad `mrope_positions` on token dimension, use attention TP groups for sharded vision, and wire DP LM head / row-parallel reductions.
- Key code:

```python
gathered_embeds = get_attention_tp_group().all_gather(image_embeds_local_padded, dim=0)
```

### #18024 untied lm_head weight loading

- Link: https://github.com/sgl-project/sglang/pull/18024, merged.
- Motivation: untied Qwen3-VL heads were overwritten from embeddings and produced bad output.
- Implementation: copy embedding weights into lm_head only when `tie_word_embeddings` is true.
- Key code:

```python
and self.config.tie_word_embeddings
```

### #18185 Qwen3-Omni audio encoder optimization

- Link: https://github.com/sgl-project/sglang/pull/18185, merged.
- Motivation: Qwen3-Omni audio/ASR path was slow; the PR reports ASR throughput around 0.28 to 3.12 req/s.
- Implementation: parallel linear FFN, vectorized masks, convolution fast path, and device placement for audio masks.
- Key code:

```python
self.fc1 = ColumnParallelLinear(...)
self.fc2 = RowParallelLinear(...)
```

### #19003 FlashInfer CUDNN ViT backend

- Link: https://github.com/sgl-project/sglang/pull/19003, merged.
- Motivation: add a faster Qwen3-VL ViT backend; the PR reports TTFT 1054 ms to 931 ms.
- Implementation: add `VisionFlashInferAttention`, CUDNN prefill call, backend flag, workspace, and packed offsets.
- Key code:

```python
output, _ = cudnn_batch_prefill_with_kv_cache(
    q, k, v, scale, self.workspace_buffer, batch_offsets_q=indptr_qk
)
```

### #19291 quant_config storage

- Link: https://github.com/sgl-project/sglang/pull/19291, merged.
- Motivation: quantized Qwen3.5/VL variants could use bf16 KV cache because the model did not store `quant_config`.
- Implementation: set `self.quant_config = quant_config`.

### #19333 visual weight loading

- Link: https://github.com/sgl-project/sglang/pull/19333, merged.
- Motivation: visual prefix mapping was missing, so visual weights could fail to load.
- Implementation: remap `model.visual.` to `visual.`.
- Key code:

```python
name = name.replace(r"model.visual.", r"visual.")
```

### #20759 / #20788 DP encoder position-embedding hang

- Links: https://github.com/sgl-project/sglang/pull/20759 and https://github.com/sgl-project/sglang/pull/20788, merged.
- Motivation: DP encoder could hang when TP position embedding communicated on ranks without image work.
- Implementation: disable TP for Qwen3-VL `pos_embed` in DP encoder mode; `#20759` is the fuller current rule.
- Key code:

```python
enable_tp=not use_data_parallel,
use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
```

### #21458 AMD decode fusion

- Link: https://github.com/sgl-project/sglang/pull/21458, merged.
- Motivation: ROCm decode spent separate kernels on QKV split, QK RMSNorm, 3D mRoPE, and KV-cache write.
- Implementation: use AITER fused `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` for Qwen3-VL mRoPE decode.
- Key code:

```python
self.use_fused_qk_norm_mrope = (
    _has_fused_qk_norm_mrope and isinstance(self.rotary_emb, MRotaryEmbedding)
)
```

### #21469 Qwen3-VL-MoE LoRA

- Link: https://github.com/sgl-project/sglang/pull/21469, merged.
- Motivation: support Qwen3-VL-30B-A3B-Instruct LoRA, including MoE expert targets.
- Implementation: expand the Qwen3-VL-MoE LoRA regex and add registered logprob-diff validation.
- Key code:

```python
r"^(?:model\.layers\.(\d+)\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
```

### #21849 Qwen3.5 VLM encoder disaggregation

- Link: https://github.com/sgl-project/sglang/pull/21849, merged.
- Motivation: Qwen3.5 multimodal models were supported by runtime but rejected by EPD allowlist.
- Implementation: allow Qwen3.5 dense/MoE architectures and extend video timestamp metadata handling.
- Key code:

```python
self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
```

### #22038 chunk-aware ViT cache and lazy transfer

- Link: https://github.com/sgl-project/sglang/pull/22038, merged.
- Motivation: earlier chunked ViT encoded too much media and moved features too early.
- Implementation: item-level overlap detection, per-image cache lookups, and lazy device transfer for cache misses.
- Key code:

```python
cached = embedding_cache.get_single(item.hash)
_move_items_to_device(miss_item_list, device)
```

### #22073 Qwen3-ASR support

- Link: https://github.com/sgl-project/sglang/pull/22073, merged.
- Motivation: serve Qwen3-ASR via `/v1/audio/transcriptions`.
- Implementation: add ASR config/processor/model, audio placeholder expansion, thinker weight remapping, and transcription adapter.
- Key code:

```python
audio_token_counts = self._get_feat_extract_output_lengths(feat_lengths)
new_ids.extend([audio_pad_id] * n)
```

### #22089 chunk-based Qwen3-ASR streaming

- Link: https://github.com/sgl-project/sglang/pull/22089, merged.
- Motivation: stream partial ASR output instead of waiting for full audio completion.
- Implementation: `StreamingASRState`, 2-second chunking, SSE output, rollback for unfixed words/tokens, and whitespace fixes.
- Key code:

```python
return StreamingResponse(..., media_type="text/event-stream")
```

### #22230 Qwen3-VL EAGLE3

- Link: https://github.com/sgl-project/sglang/pull/22230, merged.
- Motivation: enable EAGLE3 speculative decoding for Qwen3-VL.
- Implementation: capture auxiliary hidden states and pass them into `logits_processor`.
- Key code:

```python
return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states)
```

### #22266 NPU Qwen3.5 video processor

- Link: https://github.com/sgl-project/sglang/pull/22266, merged.
- Motivation: Qwen3.5 video preprocessing used a high-dimensional permute unsupported on NPU.
- Implementation: patch the Transformers Qwen3VL video processor with an NPU-compatible reshape/permute path.
- Key code:

```python
apply_module_patch(
    "transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor",
    "_preprocess",
    [npu_wrapper_video_preprocess],
)
```

### #22431 processor_output video fix

- Link: https://github.com/sgl-project/sglang/pull/22431, merged.
- Motivation: preprocessed video data returned one value while downstream expected `(video, metadata)`.
- Implementation: return `(vr, None)` for non-decoder video objects.
- Key code:

```python
if not is_video_obj:
    return vr, None
```

## Docs and Deployment Evidence

- `#12554` merged Qwen3-VL docs: FP8/BF16 launch, image/video request examples, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--keep-mm-feature-on-device`, and CUDA IPC.
- `#12703` open Qwen3-Omni docs: launch plus image/audio/video request examples.
- sgl-cookbook `#76/#102/#124`: Qwen3-VL AMD MI300X/MI355X/MI325X recipes.
- sgl-cookbook `#84/#110`: Qwen2.5-VL AMD recipes.
- LMSYS AMD latency blog reports Qwen3-VL-235B MI300X TTFT 1.62x and TPOT 1.90x over baseline.
- SGLang issue `#18466` organizes AMD Qwen3-VL work into preprocessing, multimodal transfer, ViT DP, and ViT kernel fusion.

## Open / Closed Radar

- `#12662`: CPU Qwen3-VL/Qwen3-Omni using SDPA, CPU processor device, and unaligned CPU TP padding.
- `#12261`: Qwen2.5-VL multi-frame `cu_seqlens` fix.
- `#13918` / `#17276`: early Qwen3-VL EAGLE3, mRoPE, and DeepStack capture layers.
- `#14886`: Qwen3-Omni DP encoder for audio/vision towers.
- `#16491`: Qwen3-VL-MoE PP expert weight skip.
- `#16571`: ROCm AITER add+LayerNorm fusion for Qwen3-VL ViT.
- `#16785`: Qwen3-VL DeepStack preallocation to avoid TorchDynamo recompiles.
- `#16996`: Qwen3-Omni `use_audio_in_video`.
- `#17202`: remove avoidable contiguous/where/scatter overhead in Qwen3-VL paths.
- `#18721`: DP encoder hang follow-up overlapping with `#20759`.
- `#18771`: Qwen3-Omni fused-MoE tuner architecture handling.
- `#19242`: early, incomplete Qwen3-ASR attempt superseded by `#22073`.
- `#19693`: NPU Qwen3-VL-8B accuracy path.
- `#20857`: Qwen3-VL EVS and mRoPE token-count handling.
- `#22052`: precise Qwen3-VL embedding interpolation default.
- `#22839`: Qwen3-VL config `from_dict` compatibility with Transformers 5.5.0+.
- `#22848`: Qwen3-ASR WebSocket realtime audio input.
- `#23115` / `#23220`: duplicate Qwen3-VL-MoE encoder-only `hasattr(self, "model")` guard.
- `#23304`: closed-unmerged Qwen3-VL RoPE config compatibility risk.
- `#23469`: NPU Qwen3-ASR audio loading with `soundfile` and `resample_poly`.

## Validation Guidance

1. Test single image, multi-image, raw video, and processor-output video separately.
2. Validate Qwen3-VL encoder DP, PP, and EPD as separate launch lanes.
3. For long video, check chunked prefill, cache hit/miss, and feature transfer.
4. For Qwen3-Omni, test audio-only, video+audio, and `feature_attention_mask`.
5. For Qwen3-ASR, test final transcription, streaming boundaries, and realtime input if enabled.
6. Do not infer AMD/NPU/CPU behavior from NVIDIA runs.
