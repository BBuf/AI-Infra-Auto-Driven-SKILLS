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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen VLM / Omni / ASR` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

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

### File-level PR diff reading notes

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


### Gap and optimization follow-up

- Covered PRs: 51; open PRs: 22.
- Open PRs to keep tracking: [#12261](https://github.com/sgl-project/sglang/pull/12261), [#12662](https://github.com/sgl-project/sglang/pull/12662), [#12703](https://github.com/sgl-project/sglang/pull/12703), [#13918](https://github.com/sgl-project/sglang/pull/13918), [#14886](https://github.com/sgl-project/sglang/pull/14886), [#16491](https://github.com/sgl-project/sglang/pull/16491), [#16571](https://github.com/sgl-project/sglang/pull/16571), [#16785](https://github.com/sgl-project/sglang/pull/16785), [#16996](https://github.com/sgl-project/sglang/pull/16996), [#17202](https://github.com/sgl-project/sglang/pull/17202), [#17276](https://github.com/sgl-project/sglang/pull/17276), [#18721](https://github.com/sgl-project/sglang/pull/18721), [#18771](https://github.com/sgl-project/sglang/pull/18771), [#19242](https://github.com/sgl-project/sglang/pull/19242), [#19693](https://github.com/sgl-project/sglang/pull/19693), [#20857](https://github.com/sgl-project/sglang/pull/20857), [#22052](https://github.com/sgl-project/sglang/pull/22052), [#22839](https://github.com/sgl-project/sglang/pull/22839), [#22848](https://github.com/sgl-project/sglang/pull/22848), [#23115](https://github.com/sgl-project/sglang/pull/23115) ...
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
