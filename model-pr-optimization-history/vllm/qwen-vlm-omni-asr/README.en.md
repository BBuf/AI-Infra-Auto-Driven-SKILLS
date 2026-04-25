# vllm Qwen VLM/Omni/ASR Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/offline_inference/qwen2_5_omni/README.md` | [#15130](https://github.com/vllm-project/vllm/pull/15130) |
| `examples/offline_inference/qwen2_5_omni/only_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#32772](https://github.com/vllm-project/vllm/pull/32772) |
| `examples/offline_inference/qwen3_omni/only_thinker.py` | [#27721](https://github.com/vllm-project/vllm/pull/27721), [#33010](https://github.com/vllm-project/vllm/pull/33010) |
| `examples/pooling/embed/template/dse_qwen2_vl.jinja` | no direct PR-number commit |
| `examples/pooling/score/template/qwen3_vl_reranker.jinja` | [#31890](https://github.com/vllm-project/vllm/pull/31890) |
| `tests/model_executor/test_qwen3_omni.py` | [#27721](https://github.com/vllm-project/vllm/pull/27721) |
| `tests/model_executor/test_qwen3_vl_mrope.py` | no direct PR-number commit |
| `tests/models/multimodal/generation/test_qwen2_5_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/generation/test_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/pooling/test_dse_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/pooling/test_qwen3_asr_forced_aligner.py` | no direct PR-number commit |
| `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` | [#35368](https://github.com/vllm-project/vllm/pull/35368) |
| `tests/models/multimodal/processing/test_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/processing/test_qwen3_omni.py` | [#29255](https://github.com/vllm-project/vllm/pull/29255) |
| `tests/models/multimodal/processing/test_qwen3_vl.py` | [#36136](https://github.com/vllm-project/vllm/pull/36136) |
| `vllm/model_executor/models/glmasr.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779), [#32540](https://github.com/vllm-project/vllm/pull/32540), [#40160](https://github.com/vllm-project/vllm/pull/40160) |
| `vllm/model_executor/models/glmasr_utils.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779) |
| `vllm/model_executor/models/qwen2_5_omni_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#16872](https://github.com/vllm-project/vllm/pull/16872), [#17301](https://github.com/vllm-project/vllm/pull/17301), [#17838](https://github.com/vllm-project/vllm/pull/17838), [#23058](https://github.com/vllm-project/vllm/pull/23058), [#24231](https://github.com/vllm-project/vllm/pull/24231), [#24420](https://github.com/vllm-project/vllm/pull/24420), [#26004](https://github.com/vllm-project/vllm/pull/26004), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#30883](https://github.com/vllm-project/vllm/pull/30883), [#32772](https://github.com/vllm-project/vllm/pull/32772), ... (17 total) |
| `vllm/model_executor/models/qwen2_5_vl.py` | [#12944](https://github.com/vllm-project/vllm/pull/12944), [#13155](https://github.com/vllm-project/vllm/pull/13155), [#13286](https://github.com/vllm-project/vllm/pull/13286), [#13533](https://github.com/vllm-project/vllm/pull/13533), [#13968](https://github.com/vllm-project/vllm/pull/13968), [#14377](https://github.com/vllm-project/vllm/pull/14377), [#15130](https://github.com/vllm-project/vllm/pull/15130), [#15200](https://github.com/vllm-project/vllm/pull/15200), [#15273](https://github.com/vllm-project/vllm/pull/15273), [#16907](https://github.com/vllm-project/vllm/pull/16907), [#16974](https://github.com/vllm-project/vllm/pull/16974), [#17726](https://github.com/vllm-project/vllm/pull/17726), ... (25 total) |
| `vllm/model_executor/models/qwen2_audio.py` | [#11258](https://github.com/vllm-project/vllm/pull/11258), [#35994](https://github.com/vllm-project/vllm/pull/35994) |
| `vllm/model_executor/models/qwen2_vl.py` | [#7905](https://github.com/vllm-project/vllm/pull/7905), [#8442](https://github.com/vllm-project/vllm/pull/8442), [#8696](https://github.com/vllm-project/vllm/pull/8696), [#8770](https://github.com/vllm-project/vllm/pull/8770), [#8837](https://github.com/vllm-project/vllm/pull/8837), [#9250](https://github.com/vllm-project/vllm/pull/9250), [#10112](https://github.com/vllm-project/vllm/pull/10112), [#10169](https://github.com/vllm-project/vllm/pull/10169), [#10221](https://github.com/vllm-project/vllm/pull/10221), [#11258](https://github.com/vllm-project/vllm/pull/11258), [#11430](https://github.com/vllm-project/vllm/pull/11430), [#11663](https://github.com/vllm-project/vllm/pull/11663), ... (31 total) |
| `vllm/model_executor/models/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312), [#33410](https://github.com/vllm-project/vllm/pull/33410), [#33644](https://github.com/vllm-project/vllm/pull/33644), [#37247](https://github.com/vllm-project/vllm/pull/37247) |
| `vllm/model_executor/models/qwen3_asr_forced_aligner.py` | no direct PR-number commit |
| `vllm/model_executor/models/qwen3_asr_realtime.py` | [#34613](https://github.com/vllm-project/vllm/pull/34613), [#35869](https://github.com/vllm-project/vllm/pull/35869) |
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | [#25550](https://github.com/vllm-project/vllm/pull/25550), [#26608](https://github.com/vllm-project/vllm/pull/26608), [#26815](https://github.com/vllm-project/vllm/pull/26815), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#29255](https://github.com/vllm-project/vllm/pull/29255), [#29828](https://github.com/vllm-project/vllm/pull/29828), [#29896](https://github.com/vllm-project/vllm/pull/29896), [#29974](https://github.com/vllm-project/vllm/pull/29974), [#31007](https://github.com/vllm-project/vllm/pull/31007), [#31790](https://github.com/vllm-project/vllm/pull/31790), ... (23 total) |
| `vllm/model_executor/models/qwen3_vl.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25337](https://github.com/vllm-project/vllm/pull/25337), [#25347](https://github.com/vllm-project/vllm/pull/25347), [#25557](https://github.com/vllm-project/vllm/pull/25557), [#25646](https://github.com/vllm-project/vllm/pull/25646), [#25648](https://github.com/vllm-project/vllm/pull/25648), [#25788](https://github.com/vllm-project/vllm/pull/25788), [#26000](https://github.com/vllm-project/vllm/pull/26000), [#27104](https://github.com/vllm-project/vllm/pull/27104), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#28663](https://github.com/vllm-project/vllm/pull/28663), ... (18 total) |
| `vllm/model_executor/models/qwen3_vl_moe.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25300](https://github.com/vllm-project/vllm/pull/25300), [#26000](https://github.com/vllm-project/vllm/pull/26000) |
| `vllm/model_executor/models/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/tokenizers/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/transformers_utils/configs/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/qwen_vl.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 101
- Extra PRs preserved from existing docs: 5
- Total PRs in this document: 106
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-09-11 | [#7905](https://github.com/vllm-project/vllm/pull/7905) | merged | [Model][VLM] Add Qwen2-VL model support | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-13 | [#8442](https://github.com/vllm-project/vllm/pull/8442) | merged | [Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8 | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-23 | [#8696](https://github.com/vllm-project/vllm/pull/8696) | merged | [Model] Support pp for qwen2-vl | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-25 | [#8770](https://github.com/vllm-project/vllm/pull/8770) | merged | [Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-26 | [#8837](https://github.com/vllm-project/vllm/pull/8837) | merged | [Misc] Update config loading for Qwen2-VL and remove Granite | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-10-16 | [#9250](https://github.com/vllm-project/vllm/pull/9250) | merged | [Misc] Standardize RoPE handling for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-11-07 | [#10112](https://github.com/vllm-project/vllm/pull/10112) | merged | [Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-11-09 | [#10169](https://github.com/vllm-project/vllm/pull/10169) | merged | [Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-11-13 | [#10221](https://github.com/vllm-project/vllm/pull/10221) | merged | [Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions | `tests/models/decoder_only/vision_language/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2024-12-19 | [#11258](https://github.com/vllm-project/vllm/pull/11258) | merged | [Model] Refactor Qwen2-VL to use merged multimodal processor | `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_audio.py` |
| 2024-12-24 | [#11430](https://github.com/vllm-project/vllm/pull/11430) | merged | [Bugfix] Fix Qwen2-VL LoRA weight loading | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-01-01 | [#11663](https://github.com/vllm-project/vllm/pull/11663) | merged | [Misc] Optimize Qwen2-VL LoRA test | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-01-19 | [#12128](https://github.com/vllm-project/vllm/pull/12128) | merged | [V1] Add V1 support of Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/test_qwen2_vl.py` |
| 2025-02-05 | [#12604](https://github.com/vllm-project/vllm/pull/12604) | merged | [VLM] Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/layers/rotary_embedding.py`, `tests/models/decoder_only/vision_language/test_models.py` |
| 2025-02-08 | [#12944](https://github.com/vllm-project/vllm/pull/12944) | merged | [Misc] Add qwen2.5-vl BNB support | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-02-12 | [#13148](https://github.com/vllm-project/vllm/pull/13148) | merged | [Bugfix] Fix num video tokens calculation for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-02-13 | [#13155](https://github.com/vllm-project/vllm/pull/13155) | merged | [Misc] Qwen2.5-VL Optimization | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-02-15 | [#13286](https://github.com/vllm-project/vllm/pull/13286) | merged | [Bugfix] Fix qwen2.5-vl image processor | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-02-20 | [#13533](https://github.com/vllm-project/vllm/pull/13533) | merged | [Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-02-27 | [#13968](https://github.com/vllm-project/vllm/pull/13968) | merged | [Bugfix] Fix qwen2.5-vl overflow issue | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-03-11 | [#14377](https://github.com/vllm-project/vllm/pull/14377) | merged | [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync | `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-03-21 | [#15200](https://github.com/vllm-project/vllm/pull/15200) | merged | [Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-03-21 | [#15273](https://github.com/vllm-project/vllm/pull/15273) | merged | [Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-04-19 | [#15130](https://github.com/vllm-project/vllm/pull/15130) | merged | [Model][VLM] Add Qwen2.5-Omni model support (thinker only) | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py` |
| 2025-04-19 | [#16872](https://github.com/vllm-project/vllm/pull/16872) | merged | [Model] Qwen2.5-Omni Cleanup | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-04-21 | [#16907](https://github.com/vllm-project/vllm/pull/16907) | merged | [Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-04-22 | [#16974](https://github.com/vllm-project/vllm/pull/16974) | merged | [Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-04-28 | [#17301](https://github.com/vllm-project/vllm/pull/17301) | merged | [Misc] Clean up Qwen2.5-Omni code | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-05-07 | [#17726](https://github.com/vllm-project/vllm/pull/17726) | merged | [Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-05-08 | [#17838](https://github.com/vllm-project/vllm/pull/17838) | merged | [Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-05-16 | [#17973](https://github.com/vllm-project/vllm/pull/17973) | merged | [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const… | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-06-03 | [#19054](https://github.com/vllm-project/vllm/pull/19054) | merged | [Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-08-02 | [#22069](https://github.com/vllm-project/vllm/pull/22069) | merged | [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform. | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-08-07 | [#22184](https://github.com/vllm-project/vllm/pull/22184) | merged | [Model] Switch to Fused RMS norm in Qwen2.5_VL model. | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-08-18 | [#23058](https://github.com/vllm-project/vllm/pull/23058) | merged | [Bugfix] fix Qwen2.5-Omni processor output mapping | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-08-25 | [#23512](https://github.com/vllm-project/vllm/pull/23512) | merged | [Bugfix] Fix Qwen2.5-VL quantized model weights loading | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-09-04 | [#24231](https://github.com/vllm-project/vllm/pull/24231) | merged | [LoRA]: Add lora support to qwen-2.5-omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-09-08 | [#24420](https://github.com/vllm-project/vllm/pull/24420) | merged | [Model] Enable BNB support for qwen2_5_omni_thinker | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-09-12 | [#24741](https://github.com/vllm-project/vllm/pull/24741) | merged | [Models] Prevent CUDA sync in Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-09-17 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-18 | [#24955](https://github.com/vllm-project/vllm/pull/24955) | merged | [MM Encoder] Apply DP ViT for Qwen3-VL model series | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py` |
| 2025-09-19 | [#25055](https://github.com/vllm-project/vllm/pull/25055) | merged | [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE | `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/kernels/core/test_mrope.py` |
| 2025-09-20 | [#25300](https://github.com/vllm-project/vllm/pull/25300) | merged | [Bugfix] Fix Qwen3-VL-MoE weight loading for EP | `vllm/model_executor/models/qwen3_vl_moe.py` |
| 2025-09-21 | [#25337](https://github.com/vllm-project/vllm/pull/25337) | merged | [MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate` | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-21 | [#25347](https://github.com/vllm-project/vllm/pull/25347) | merged | [Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate` | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-23 | [#25445](https://github.com/vllm-project/vllm/pull/25445) | merged | [Model] Enable DP for ViT in Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-25 | [#25646](https://github.com/vllm-project/vllm/pull/25646) | merged | [Misc] Fix Qwen3-VL `video_grid_thw` typing | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-25 | [#25648](https://github.com/vllm-project/vllm/pull/25648) | merged | [Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-27 | [#25788](https://github.com/vllm-project/vllm/pull/25788) | merged | [Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-28 | [#25557](https://github.com/vllm-project/vllm/pull/25557) | merged | [VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-01 | [#26000](https://github.com/vllm-project/vllm/pull/26000) | merged | [MM] Add text-only mode for Qwen3-VL | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py` |
| 2025-10-01 | [#26004](https://github.com/vllm-project/vllm/pull/26004) | merged | [BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-10-02 | [#24642](https://github.com/vllm-project/vllm/pull/24642) | merged | [Qwen][ROCm] Flash Attention Rotary Embeddings | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-03 | [#26123](https://github.com/vllm-project/vllm/pull/26123) | merged | [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642 | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-10 | [#25550](https://github.com/vllm-project/vllm/pull/25550) | merged | Add Qwen3-Omni moe thinker | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-10-11 | [#26608](https://github.com/vllm-project/vllm/pull/26608) | merged | [MM] Move Qwen3Omni MRoPE impl to model file | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-10-15 | [#26815](https://github.com/vllm-project/vllm/pull/26815) | merged | [Bugfix] Fix qwen3-omni audio truncation issue | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-10-17 | [#27104](https://github.com/vllm-project/vllm/pull/27104) | merged | [bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-10-26 | [#27190](https://github.com/vllm-project/vllm/pull/27190) | merged | [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py` |
| 2025-10-29 | [#27705](https://github.com/vllm-project/vllm/pull/27705) | merged | [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-10-30 | [#27790](https://github.com/vllm-project/vllm/pull/27790) | merged | [BugFix][VL] Fix FA selection on Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-11-02 | [#27920](https://github.com/vllm-project/vllm/pull/27920) | merged | [Bugfix] Fix Qwen Omni audio inference | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-11-12 | [#28271](https://github.com/vllm-project/vllm/pull/28271) | merged | [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-11-14 | [#28663](https://github.com/vllm-project/vllm/pull/28663) | merged | [Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-11-22 | [#29232](https://github.com/vllm-project/vllm/pull/29232) | merged | Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-11-24 | [#27721](https://github.com/vllm-project/vllm/pull/27721) | merged | [Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine. | `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-12-02 | [#29896](https://github.com/vllm-project/vllm/pull/29896) | merged | feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-12-04 | [#29974](https://github.com/vllm-project/vllm/pull/29974) | merged | [ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-12-04 | [#30037](https://github.com/vllm-project/vllm/pull/30037) | merged | support qwen3-vl handle requests with embeddings | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-12-14 | [#30542](https://github.com/vllm-project/vllm/pull/30542) | merged | [Bugfix] Revert Qwen2-VL part of change in #28271 | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-12-14 | [#29752](https://github.com/vllm-project/vllm/pull/29752) | merged | [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-12-18 | [#30883](https://github.com/vllm-project/vllm/pull/30883) | merged | [Chore] Remove v0 dead code for Qwen2.5-omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-12-24 | [#31007](https://github.com/vllm-project/vllm/pull/31007) | merged | [Qwen3-Omni] fixed _get_feat_extract_output_lengths function | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-12-31 | [#31436](https://github.com/vllm-project/vllm/pull/31436) | merged | Add GLM-ASR multimodal support | `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py` |
| 2026-01-03 | [#29255](https://github.com/vllm-project/vllm/pull/29255) | merged | Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring | `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-06 | [#31790](https://github.com/vllm-project/vllm/pull/31790) | merged | [Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni) | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-07 | [#31779](https://github.com/vllm-project/vllm/pull/31779) | merged | [Refactor] GLM-ASR Modeling | `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py` |
| 2026-01-08 | [#31890](https://github.com/vllm-project/vllm/pull/31890) | merged | [Models] Allow converting Qwen3-VL into Reranker model | `examples/pooling/score/template/qwen3_vl_reranker.jinja` |
| 2026-01-13 | [#32126](https://github.com/vllm-project/vllm/pull/32126) | merged | [Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2026-01-14 | [#32167](https://github.com/vllm-project/vllm/pull/32167) | merged | [Model] Re-implement Qwen3Omni Audio Encoder | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-18 | [#32540](https://github.com/vllm-project/vllm/pull/32540) | merged | [Bugfix] Fix GLM-ASR audio encoder RoPE dim | `vllm/model_executor/models/glmasr.py` |
| 2026-01-25 | [#32772](https://github.com/vllm-project/vllm/pull/32772) | merged | [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py` |
| 2026-01-26 | [#33010](https://github.com/vllm-project/vllm/pull/33010) | merged | [Model] Use mm_position to compute mrope positions for Qwen3-Omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `examples/offline_inference/qwen3_omni/only_thinker.py` |
| 2026-01-29 | [#33312](https://github.com/vllm-project/vllm/pull/33312) | merged | [Models] Qwen3-ASR | `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py` |
| 2026-01-31 | [#33410](https://github.com/vllm-project/vllm/pull/33410) | merged | [Bugfix] Fix `Qwen3ASR` language asr tag in output | `vllm/model_executor/models/qwen3_asr.py` |
| 2026-02-01 | [#33077](https://github.com/vllm-project/vllm/pull/33077) | merged | [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-02-03 | [#33644](https://github.com/vllm-project/vllm/pull/33644) | merged | [Bugfix] fix qwen3-asr response error | `vllm/model_executor/models/qwen3_asr.py` |
| 2026-02-04 | [#33605](https://github.com/vllm-project/vllm/pull/33605) | merged | [Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-02-04 | [#29828](https://github.com/vllm-project/vllm/pull/29828) | merged | [Model] Add transcription support for Qwen3-Omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-02-21 | [#34613](https://github.com/vllm-project/vllm/pull/34613) | merged | [Realtime] Add Qwen3-ASR realtime streaming support | `vllm/model_executor/models/qwen3_asr_realtime.py` |
| 2026-02-26 | [#35368](https://github.com/vllm-project/vllm/pull/35368) | merged | [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression | `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-02 | [#35741](https://github.com/vllm-project/vllm/pull/35741) | merged | [Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-04 | [#35869](https://github.com/vllm-project/vllm/pull/35869) | merged | [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile | `vllm/model_executor/models/qwen3_asr_realtime.py` |
| 2026-03-05 | [#36140](https://github.com/vllm-project/vllm/pull/36140) | merged | [Bugfix] Fix Qwen-VL tokenizer implementation | `vllm/tokenizers/qwen_vl.py`, `vllm/model_executor/models/qwen_vl.py`, `vllm/renderers/qwen_vl.py` |
| 2026-03-05 | [#36108](https://github.com/vllm-project/vllm/pull/36108) | merged | refactor funasr model. | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-05 | [#35994](https://github.com/vllm-project/vllm/pull/35994) | merged | [BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0 | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py` |
| 2026-03-09 | [#36319](https://github.com/vllm-project/vllm/pull/36319) | merged | Support online use_audio_in_video | `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/media/audio.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-03-11 | [#36136](https://github.com/vllm-project/vllm/pull/36136) | merged | [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps | `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-13 | [#36800](https://github.com/vllm-project/vllm/pull/36800) | merged | [Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-16 | [#37147](https://github.com/vllm-project/vllm/pull/37147) | merged | [Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-16 | [#37183](https://github.com/vllm-project/vllm/pull/37183) | merged | Remove unused EVS functions in qwen3_vl.py | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-18 | [#37439](https://github.com/vllm-project/vllm/pull/37439) | merged | [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-23 | [#35963](https://github.com/vllm-project/vllm/pull/35963) | merged | [Feature] ViT Full CUDA Graph | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` |
| 2026-04-10 | [#37247](https://github.com/vllm-project/vllm/pull/37247) | merged | [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration | `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-04-14 | [#38061](https://github.com/vllm-project/vllm/pull/38061) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-04-18 | [#40160](https://github.com/vllm-project/vllm/pull/40160) | merged | [Bugfix] Fix k_proj's bias for GLM-ASR | `vllm/model_executor/models/glmasr.py` |

## Per-PR Diff Audit Cards

### PR #7905 - [Model][VLM] Add Qwen2-VL model support

- Link: https://github.com/vllm-project/vllm/pull/7905
- Status/date: merged / 2024-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `3b7fea770f44`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1531/-31, 1844 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Qwen2-VL model support"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: This PR adding support for Qwen2-VL model. FIX #8139 FIX #8281 Requirements - ~This PR requires `transformers` with this PR merged and this bugfix PR merged (You can install it....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` added +1088/-0 (1088 lines); hunks: -0,0 +1,1088; symbols: Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP, __init__, touching `Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` added +1088/-0 (1088 lines); hunks: -0,0 +1,1088; symbols: Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -0,0 +1,1088 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` added +1088/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8442 - [Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8

- Link: https://github.com/vllm-project/vllm/pull/8442
- Status/date: merged / 2024-09-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `06311e295666`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-0, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FILL IN THE PR DESCRIPTION HERE FIX #8406 ping @DarkLight1337 **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** PR Checklist (Click to Exp....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -1055,6 +1055,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1078,6 +1081,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -1055,6 +1055,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1078,6 +1081,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -1055,6 +1055,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                # Skip loading extra bias for GPTQ models.
+                if name.endswith(".bias") and name not in params_dict:
+                    continue
@@ -1078,6 +1081,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    # Skip loading extra bias for GPTQ models.
+                    if name.endswith(".bias") and name not in params_dict:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8696 - [Model] Support pp for qwen2-vl

- Link: https://github.com/vllm-project/vllm/pull/8696
- Status/date: merged / 2024-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `a79e5229843e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +46/-14, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support pp for qwen2-vl"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FILL IN THE PR DESCRIPTION HERE Follow #7860 , this pr introduce pp support for the latest qwen2-vl. Already tested on Qwen2-VL-72B-Instruct-GPTQ-Int4 **BEFORE SUBMITTING, PLEAS....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +22/-7 (29 lines); hunks: -45,7 +45,7; -68,6 +68,9; symbols: __init__, _validate_and_reshape_mm_tensor, forward, load_weights, touching `__init__, _validate_and_reshape_mm_tensor, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-7 (29 lines); hunks: -45,7 +45,7; -68,6 +68,9; symbols: __init__, _validate_and_reshape_mm_tensor, forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -45,7 +45,7 @@
-from vllm.distributed import parallel_state
+from vllm.distributed import get_pp_group, parallel_state
@@ -68,6 +68,9 @@
+from .utils import (PPMissingLayer, is_pp_missing_parameter,
+                    make_empty_intermediate_tensors_factory)
@@ -856,15 +859,21 @@ def __init__(self,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +22/-7
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8770 - [Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend

- Link: https://github.com/vllm-project/vllm/pull/8770
- Status/date: merged / 2024-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `c23953675f78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +99/-9, 202 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) - This PR enables mrope and qwen2-vl on CPU backend. **BEFORE SUBMITTING, PLEASE READ THE....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +16/-0 (16 lines); hunks: -67,6 +67,7; -281,6 +282,21 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +16/-0 (16 lines); hunks: -67,6 +67,7; -281,6 +282,21 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -67,6 +67,7 @@
+from vllm.utils import is_cpu
@@ -281,6 +282,21 @@ def forward(
+        elif is_cpu():
+            seq_length = q.size(1)
+            q, k, v = [rearrange(x, "b s h d -> b h s d") for x in [q, k, v]]
+            attention_mask = torch.zeros([1, seq_length, seq_length],
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +16/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`, `vllm/worker/cpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8837 - [Misc] Update config loading for Qwen2-VL and remove Granite

- Link: https://github.com/vllm-project/vllm/pull/8837
- Status/date: merged / 2024-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `4bb98f2190aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +144/-224, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Update config loading for Qwen2-VL and remove Granite"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: This PR ports Qwen2-VL config from transformers v4.45 but remove rope type override to address config loading issue, and remove ported Granite config that's no longer required.....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +2/-3 (5 lines); hunks: -31,12 +31,9; -66,6 +63,8.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +2/-3 (5 lines); hunks: -31,12 +31,9; -66,6 +63,8
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -31,12 +31,9 @@
-from transformers import Qwen2VLConfig
-from transformers.models.qwen2_vl.configuration_qwen2_vl import (
-    Qwen2VLVisionConfig)
@@ -66,6 +63,8 @@
+from vllm.transformers_utils.configs.qwen2vl import (Qwen2VLConfig,
+                                                     Qwen2VLVisionConfig)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/granite.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/transformers_utils/config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9250 - [Misc] Standardize RoPE handling for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/9250
- Status/date: merged / 2024-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `7e7eae338d27`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +102/-200, 533 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Standardize RoPE handling for Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: https://github.com/huggingface/transformers/issues/33401 has been fixed in Transformers v4.45.2, but the devs have clarified that M-ROPE is intended to be configured as `rope_ty....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +4/-4 (8 lines); hunks: -34,6 +34,8; -62,8 +64,7; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +4/-4 (8 lines); hunks: -34,6 +34,8; -62,8 +64,7; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -34,6 +34,8 @@
+from transformers.models.qwen2_vl.configuration_qwen2_vl import (
+    Qwen2VLConfig, Qwen2VLVisionConfig)
@@ -62,8 +64,7 @@
-from vllm.transformers_utils.configs.qwen2vl import (Qwen2VLConfig,
-                                                     Qwen2VLVisionConfig)
+from vllm.transformers_utils.config import uses_mrope
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/test_pos_encoding.py`, `tests/lora/test_layers.py`, `tests/test_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10112 - [Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/10112
- Status/date: merged / 2024-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `999df95b4eef`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-10, 68 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Without this PR, if one wants to pass more images with lower resolution to Qwen2-VL models (e.g. 127 images with 256 tokens each): the LLM instance will be failed to create due....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +23/-10 (33 lines); hunks: -22,8 +22,8; -558,6 +558,17 @@ def forward(; symbols: forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl, get_max_qwen2_vl_mm_tokens, touching `forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +23/-10 (33 lines); hunks: -22,8 +22,8; -558,6 +558,17 @@ def forward(; symbols: forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl, get_max_qwen2_vl_mm_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -22,8 +22,8 @@
-from typing import (Any, Callable, Iterable, List, Literal, Mapping, Optional,
-                    Tuple, Type, TypedDict, Union)
+from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
+                    Optional, Tuple, Type, TypedDict, Union)
@@ -558,6 +558,17 @@ def forward(
+def get_mm_processor_kwargs(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +23/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10169 - [Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module

- Link: https://github.com/vllm-project/vllm/pull/10169
- Status/date: merged / 2024-11-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `f83feccd7f66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-2, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FIX https://github.com/vllm-project/vllm/issues/9832 This is a workaround for the fact that GPTQ configs generated by AutoGPTQ do not have a list of ignored modules to check if....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +12/-2 (14 lines); hunks: -51,7 +51,9; -982,7 +984,7 @@ def __init__(self,; symbols: __init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor, touching `__init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-2 (14 lines); hunks: -51,7 +51,9; -982,7 +984,7 @@ def __init__(self,; symbols: __init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -51,7 +51,9 @@
-from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.model_executor.layers.quantization import (GPTQConfig,
+                                                     GPTQMarlinConfig,
+                                                     QuantizationConfig)
@@ -982,7 +984,7 @@ def __init__(self,
-            quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10221 - [Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions

- Link: https://github.com/vllm-project/vllm/pull/10221
- Status/date: merged / 2024-11-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `3945c82346da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +578/-32, 709 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `tests/models/decoder_only/vision_language/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Goal 1. Add support for Qwen2-VL multiple image embeddings input with **varied resolutions** currently, vLLM implementation of Qwen2-VL's image embedding input **requires** all....
- Key implementation: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0 (428 lines); hunks: -0,0 +1,428; symbols: qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput, batch_make_image_embeddings, touching `qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput`; `vllm/model_executor/models/qwen2_vl.py` modified +149/-31 (180 lines); hunks: -79,7 +79,7; -92,17 +92,31 @@ class Qwen2VLImagePixelInputs(TypedDict):; symbols: Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs, Qwen2VLVideoPixelInputs, touching `Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs`.
- Code diff details:
  - `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0 (428 lines); hunks: -0,0 +1,428; symbols: qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput, batch_make_image_embeddings
  - `vllm/model_executor/models/qwen2_vl.py` modified +149/-31 (180 lines); hunks: -79,7 +79,7; -92,17 +92,31 @@ class Qwen2VLImagePixelInputs(TypedDict):; symbols: Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs, Qwen2VLVideoPixelInputs
- Key code excerpts:

```diff
diff -- tests/models/decoder_only/vision_language/test_qwen2_vl.py
@@ -0,0 +1,428 @@
+from typing import Any, List, Optional, Tuple, Type, TypedDict, Union
+import numpy.typing as npt
+import pytest
+import torch
+from PIL import Image
+from vllm.entrypoints.llm import LLM
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -79,7 +79,7 @@
-    data: torch.Tensor
+    pixel_values: torch.Tensor
@@ -92,17 +92,31 @@ class Qwen2VLImagePixelInputs(TypedDict):
-    data: torch.Tensor
-    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
-    `hidden_size` must match the hidden size of language model backbone.
```

- Reviewed files:
  - tests: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +149/-31
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_qwen2_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11258 - [Model] Refactor Qwen2-VL to use merged multimodal processor

- Link: https://github.com/vllm-project/vllm/pull/11258
- Status/date: merged / 2024-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `e24113a8fe5d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +277/-527, 1006 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Refactor Qwen2-VL to use merged multimodal processor"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_audio.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +187/-393 (580 lines); hunks: -22,28 +22,26; -56,14 +54,14; symbols: Qwen2VisionMLP, __init__, load_weights, get_mm_processor_kwargs, touching `Qwen2VisionMLP, __init__, load_weights`; `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127 (192 lines); hunks: -1,12 +1,9; -20,22 +17,9; symbols: image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context, processor_for_qwen2_vl, touching `image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context`; `vllm/model_executor/models/qwen2_audio.py` modified +3/-1 (4 lines); hunks: -164,7 +164,9 @@ def _get_dummy_mm_inputs(; symbols: _get_dummy_mm_inputs, touching `_get_dummy_mm_inputs`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +187/-393 (580 lines); hunks: -22,28 +22,26; -56,14 +54,14; symbols: Qwen2VisionMLP, __init__, load_weights, get_mm_processor_kwargs
  - `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127 (192 lines); hunks: -1,12 +1,9; -20,22 +17,9; symbols: image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context, processor_for_qwen2_vl
  - `vllm/model_executor/models/qwen2_audio.py` modified +3/-1 (4 lines); hunks: -164,7 +164,9 @@ def _get_dummy_mm_inputs(; symbols: _get_dummy_mm_inputs
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -22,28 +22,26 @@
-from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
-                    Optional, Set, Tuple, Type, TypedDict, Union)
+from typing import (Any, Iterable, List, Literal, Mapping, Optional, Set,
+                    Tuple, Type, TypedDict, Union)
-from transformers.image_utils import (get_image_size,
-                                      infer_channel_dimension_format,
diff -- tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py
@@ -1,12 +1,9 @@
-import torch
-from PIL.Image import Image
-from vllm.inputs import InputContext, token_inputs
-from vllm.multimodal import MultiModalRegistry
+from vllm.inputs import InputContext, InputProcessingContext
@@ -20,22 +17,9 @@
diff -- vllm/model_executor/models/qwen2_audio.py
@@ -164,7 +164,9 @@ def _get_dummy_mm_inputs(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +187/-393; `vllm/model_executor/models/qwen2_audio.py` modified +3/-1
  - tests: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11430 - [Bugfix] Fix Qwen2-VL LoRA weight loading

- Link: https://github.com/vllm-project/vllm/pull/11430
- Status/date: merged / 2024-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `b1b1038fbdc1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +168/-14, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2-VL LoRA weight loading"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FIX #11406 Upcoming PRs include: - Convert hf_to_vllm_mapper to static variable for all models - Support `substr` and `subfix` mapping.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +6/-6 (12 lines); hunks: -901,6 +901,11 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMu...; -1190,11 +1195,6 @@ def sample(; symbols: Qwen2VLForConditionalGeneration, __init__, sample, load_weights, touching `Qwen2VLForConditionalGeneration, __init__, sample`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-6 (12 lines); hunks: -901,6 +901,11 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMu...; -1190,11 +1195,6 @@ def sample(; symbols: Qwen2VLForConditionalGeneration, __init__, sample, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -901,6 +901,11 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
+    # To ensure correct weight loading and mapping.
+    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
+        "lm_head.": "language_model.lm_head.",
+        "model.": "language_model.model.",
+    })
@@ -1190,11 +1195,6 @@ def sample(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +6/-6
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_lora_checkpoints.py`, `tests/lora/test_qwen2vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11663 - [Misc] Optimize Qwen2-VL LoRA test

- Link: https://github.com/vllm-project/vllm/pull/11663
- Status/date: merged / 2025-01-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `11d8a091c6c7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +21/-4, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Optimize Qwen2-VL LoRA test"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: I retrained a QWen-VL LoRA. In my local environment using A800, I tested the model on 100 image samples, the generation results were completely aligned with the `transformers` o....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +19/-1 (20 lines); hunks: -53,6 +53,7; -925,15 +926,23 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsM...; symbols: Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping, touching `Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +19/-1 (20 lines); hunks: -53,6 +53,7; -925,15 +926,23 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsM...; symbols: Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -53,6 +53,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -925,15 +926,23 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
-    # TODO Support LoRA for the visual encoder in the future.
+        # vision tower
+        "qkv",
+        "attn.proj",  # Distinguish patch_embed.proj
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +19/-1
- Risk and verification: The diff ships test coverage in `tests/lora/test_qwen2vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12128 - [V1] Add V1 support of Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/12128
- Status/date: merged / 2025-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `81763c58a01e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +292/-85, 616 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] Add V1 support of Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/test_qwen2_vl.py`; PR body summary: Continued from #11668 for supporting Qwen2-VL on V1. Co-authored-by: @imkero who has done the great work on the V1 MRoPE implementation..
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +78/-64 (142 lines); hunks: -67,11 +67,15; -135,7 +139,7 @@ class Qwen2VLVideoEmbeddingInputs(TypedDict):; symbols: Qwen2VLVideoEmbeddingInputs, forward, load_weights, get_num_frames_with_most_features, touching `Qwen2VLVideoEmbeddingInputs, forward, load_weights`; `tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10 (18 lines); hunks: -105,7 +105,7 @@ def batch_make_image_embeddings(; -124,11 +124,10 @@ def batch_make_image_embeddings(; symbols: batch_make_image_embeddings, batch_make_video_embeddings, touching `batch_make_image_embeddings, batch_make_video_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +78/-64 (142 lines); hunks: -67,11 +67,15; -135,7 +139,7 @@ class Qwen2VLVideoEmbeddingInputs(TypedDict):; symbols: Qwen2VLVideoEmbeddingInputs, forward, load_weights, get_num_frames_with_most_features
  - `tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10 (18 lines); hunks: -105,7 +105,7 @@ def batch_make_image_embeddings(; -124,11 +124,10 @@ def batch_make_image_embeddings(; symbols: batch_make_image_embeddings, batch_make_video_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -67,11 +67,15 @@
-                    init_vllm_registered_model, maybe_prefix)
+                    init_vllm_registered_model, maybe_prefix,
+                    merge_multimodal_embeddings)
+# For profile run
+_MAX_FRAMES_PER_VIDEO = 16
@@ -135,7 +139,7 @@ class Qwen2VLVideoEmbeddingInputs(TypedDict):
diff -- tests/models/decoder_only/vision_language/test_qwen2_vl.py
@@ -105,7 +105,7 @@ def batch_make_image_embeddings(
-    # pixel values to embeddinds & grid_thws
+    # pixel values to embeddings & grid_thws
@@ -124,11 +124,10 @@ def batch_make_image_embeddings(
-        cur_batch_embed_len = sum([
-            grid_thw.prod() // merge_size // merge_size
+        cur_batch_embed_len = sum(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +78/-64
  - tests: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_qwen2_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12604 - [VLM] Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/12604
- Status/date: merged / 2025-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1315/-52, 1625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/layers/rotary_embedding.py`, `tests/models/decoder_only/vision_language/test_models.py`; PR body summary: Add support for Qwen2.5-VL, one of the most popular and anticipated vision language models, to vLLM! FIXES: #12486, #12532.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0 (1133 lines); hunks: -0,0 +1,1133; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs, touching `Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs`; `vllm/model_executor/layers/rotary_embedding.py` modified +34/-24 (58 lines); hunks: -27,6 +27,7; -772,8 +773,12 @@ def __init__(; symbols: __init__, forward, get_input_positions, get_input_positions_tensor, touching `__init__, forward, get_input_positions`; `tests/models/decoder_only/vision_language/test_models.py` modified +22/-0 (22 lines); hunks: -121,6 +121,8; -138,6 +140,26; `vllm/model_executor/models/qwen2_vl.py` modified +8/-8 (16 lines); hunks: -650,8 +650,8 @@ def load_weights(self, weights: Iterable[Tuple[str,; -683,26 +683,26 @@ def get_passthrough_data(self) -> Mapping[str, object]:; symbols: load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems, __init__, touching `load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0 (1133 lines); hunks: -0,0 +1,1133; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +34/-24 (58 lines); hunks: -27,6 +27,7; -772,8 +773,12 @@ def __init__(; symbols: __init__, forward, get_input_positions, get_input_positions_tensor
  - `tests/models/decoder_only/vision_language/test_models.py` modified +22/-0 (22 lines); hunks: -121,6 +121,8; -138,6 +140,26
  - `vllm/model_executor/models/qwen2_vl.py` modified +8/-8 (16 lines); hunks: -650,8 +650,8 @@ def load_weights(self, weights: Iterable[Tuple[str,; -683,26 +683,26 @@ def get_passthrough_data(self) -> Mapping[str, object]:; symbols: load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems, __init__
  - `docs/source/models/supported_models.md` modified +11/-0 (11 lines); hunks: -846,6 +846,13 @@ See this page for more information on how to use generativ; -880,6 +887,10 @@ The chat template for Pixtral-HF is incorrect (see [discuss...
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -0,0 +1,1133 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
diff -- vllm/model_executor/layers/rotary_embedding.py
@@ -27,6 +27,7 @@
+from transformers import PretrainedConfig
@@ -772,8 +773,12 @@ def __init__(
-        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
-                         is_neox_style, dtype)
+        # In Qwen2.5-VL, the maximum index value is related to the duration of
+        # the input video. We enlarge max_position_embeddings to 4 times to get
diff -- tests/models/decoder_only/vision_language/test_models.py
@@ -121,6 +121,8 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +34/-24; `vllm/model_executor/models/qwen2_vl.py` modified +8/-8; `vllm/entrypoints/chat_utils.py` modified +2/-2
  - tests: `tests/models/decoder_only/vision_language/test_models.py` modified +22/-0; `tests/models/registry.py` modified +2/-0; `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs: `docs/source/models/supported_models.md` modified +11/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12944 - [Misc] Add qwen2.5-vl BNB support

- Link: https://github.com/vllm-project/vllm/pull/12944
- Status/date: merged / 2025-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `4c8dd12ef347`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-30, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Add qwen2.5-vl BNB support"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: FIX https://github.com/vllm-project/vllm/pull/12604#discussion_r1940626980 - Add tp compatible BNB support to Qwen2.5-VL models..
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30 (59 lines); hunks: -40,7 +40,7; -207,11 +207,12 @@ def __init__(; symbols: __init__, split_qkv, forward, touching `__init__, split_qkv, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30 (59 lines); hunks: -40,7 +40,7; -207,11 +207,12 @@ def __init__(; symbols: __init__, split_qkv, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -40,7 +40,7 @@
-from vllm.distributed import parallel_state
+from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
@@ -207,11 +207,12 @@ def __init__(
-        world_size = parallel_state.get_tensor_model_parallel_world_size()
+        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
+        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13148 - [Bugfix] Fix num video tokens calculation for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/13148
- Status/date: merged / 2025-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `985b4a2b1989`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix num video tokens calculation for Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FIX #13099.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +5/-1 (6 lines); hunks: -800,7 +800,11 @@ def _get_vision_info(; symbols: _get_vision_info, touching `_get_vision_info`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-1 (6 lines); hunks: -800,7 +800,11 @@ def _get_vision_info(; symbols: _get_vision_info
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -800,7 +800,11 @@ def _get_vision_info(
-        grid_t = max(num_frames // temporal_patch_size, 1)
+        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
+        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
+        padded_num_frames = num_frames + num_frames % temporal_patch_size
+        grid_t = max(padded_num_frames // temporal_patch_size, 1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13155 - [Misc] Qwen2.5-VL Optimization

- Link: https://github.com/vllm-project/vllm/pull/13155
- Status/date: merged / 2025-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `02ed8a1fbe41`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-51, 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Qwen2.5-VL Optimization"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Thanks to the VLLM team for their great work! This PR includes the following: ꔷ Replaced `Qwen2RMSNorm` with vllm's official `RMSNorm`. Extensive testing has confirmed that this....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunks: -45,6 +45,7; -271,8 +272,13 @@ def forward(; symbols: forward, Qwen2RMSNorm, __init__, touching `forward, Qwen2RMSNorm, __init__`; `vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunks: -226,11 +226,15 @@ def apply_rotary_emb_torch(x: torch.Tensor,; -336,20 +340,23 @@ def forward(; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward, touching `apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunks: -45,6 +45,7; -271,8 +272,13 @@ def forward(; symbols: forward, Qwen2RMSNorm, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunks: -226,11 +226,15 @@ def apply_rotary_emb_torch(x: torch.Tensor,; -336,20 +340,23 @@ def forward(; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -45,6 +45,7 @@
+from vllm.model_executor.layers.layernorm import RMSNorm
@@ -271,8 +272,13 @@ def forward(
-            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
-            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
+            use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
+            q = apply_rotary_pos_emb_vision(q,
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -226,11 +226,15 @@ def apply_rotary_emb_torch(x: torch.Tensor,
-                                freqs: torch.Tensor) -> torch.Tensor:
+                                freqs: torch.Tensor,
+                                use_flash_attn=False) -> torch.Tensor:
-    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
+    apply_rotary_emb = apply_rotary_emb_torch
+    if use_flash_attn:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36; `vllm/model_executor/models/qwen2_vl.py` modified +22/-15
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13286 - [Bugfix] Fix qwen2.5-vl image processor

- Link: https://github.com/vllm-project/vllm/pull/13286
- Status/date: merged / 2025-02-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `7fdaaf48ef2a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-6, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix qwen2.5-vl image processor"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FIX #13285 **Need to wait Qwen team update the model repo with correct preprocessor_config.json (see https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/14)** - `tran....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5 (13 lines); hunks: -33,10 +33,11; -693,7 +694,8 @@ def get_hf_processor(; symbols: get_hf_processor, get_image_processor, touching `get_hf_processor, get_image_processor`; `vllm/model_executor/models/qwen2_vl.py` modified +9/-1 (10 lines); hunks: -31,7 +31,9; -759,7 +761,13 @@ def get_image_processor(; symbols: get_image_processor, get_supported_mm_limits, touching `get_image_processor, get_supported_mm_limits`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5 (13 lines); hunks: -33,10 +33,11; -693,7 +694,8 @@ def get_hf_processor(; symbols: get_hf_processor, get_image_processor
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-1 (10 lines); hunks: -31,7 +31,9; -759,7 +761,13 @@ def get_image_processor(; symbols: get_image_processor, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -33,10 +33,11 @@
-from transformers.models.qwen2_5_vl import (Qwen2_5_VLImageProcessor,
-                                            Qwen2_5_VLProcessor)
+from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
+from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
+                                          Qwen2VLImageProcessorFast)
@@ -693,7 +694,8 @@ def get_hf_processor(
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -31,7 +31,9 @@
+from packaging.version import Version
+from transformers import __version__ as TRANSFORMERS_VERSION
@@ -759,7 +761,13 @@ def get_image_processor(
-        assert isinstance(image_processor, Qwen2VLImageProcessor)
+        if Version(TRANSFORMERS_VERSION) >= Version("4.49"):
+            from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5; `vllm/model_executor/models/qwen2_vl.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13533 - [Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/13533
- Status/date: merged / 2025-02-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `041e29471671`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +18/-2, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: In Qwen2.5-VL online inference, the `fps` parameter in `mm_processor_kwargs` is essential for accurately calculating the `second_pre_grid_t` value. However, the OpenAI interface....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -689,7 +689,7 @@ def get_hf_processor(; symbols: get_hf_processor, touching `get_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -689,7 +689,7 @@ def get_hf_processor(; symbols: get_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -689,7 +689,7 @@ def get_hf_processor(
-        fps: Optional[float] = None,
+        fps: Optional[Union[float, List[float]]] = None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/openai/serving_engine.py`, `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13968 - [Bugfix] Fix qwen2.5-vl overflow issue

- Link: https://github.com/vllm-project/vllm/pull/13968
- Status/date: merged / 2025-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `78648758794e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-15, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix qwen2.5-vl overflow issue"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: FIX #13817 (*link existing issues this PR will resolve*) - Qwen2.5-VL-3B will overflow with float16 in its ViT encoder, we need to cast the overflow value manually..
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1 (7 lines); hunks: -63,7 +63,7; -641,6 +641,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1 (7 lines); hunks: -63,7 +63,7; -641,6 +641,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -63,7 +63,7 @@
-from .utils import (AutoWeightsLoader, WeightsMapper,
+from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
@@ -641,6 +641,11 @@ def forward(
+        # For Qwen2.5-VL-3B, float16 will overflow at last block
+        # for long visual tokens sequences.
+        if hidden_states.dtype == torch.float16:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minicpmo.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14377 - [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync

- Link: https://github.com/vllm-project/vllm/pull/14377
- Status/date: merged / 2025-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `70b808fe1a63`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +70/-24, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: qwen2-vl logic optimization: During each forward propagation, the xformer branch of Qwen2VisionTransformer will execute multiple tensor tolist methods (flash attn branch will ex....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +37/-12 (49 lines); hunks: -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -329,7 +331,6 @@ def forward(; symbols: split_qkv, forward, __init__, touching `split_qkv, forward, __init__`; `vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12 (45 lines); hunks: -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -285,7 +287,6 @@ def forward(; symbols: split_qkv, forward, __init__, touching `split_qkv, forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +37/-12 (49 lines); hunks: -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -329,7 +331,6 @@ def forward(; symbols: split_qkv, forward, __init__
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12 (45 lines); hunks: -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -285,7 +287,6 @@ def forward(; symbols: split_qkv, forward, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
-        self,
-        x: torch.Tensor,
-        cu_seqlens: torch.Tensor,
-        rotary_pos_emb: torch.Tensor,
+            self,
+            x: torch.Tensor,
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
-        self,
-        x: torch.Tensor,
-        cu_seqlens: torch.Tensor,
-        rotary_pos_emb: torch.Tensor,
+            self,
+            x: torch.Tensor,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +37/-12; `vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15200 - [Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation

- Link: https://github.com/vllm-project/vllm/pull/15200
- Status/date: merged / 2025-03-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `1e508343e1ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +37/-4, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: FIX #15122 FIX #15197 - Fix incorrect attention mask creation for qwen2.5-vl windows attention from #14377.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4 (10 lines); hunks: -647,15 +647,17 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4 (10 lines); hunks: -647,15 +647,17 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -647,15 +647,17 @@ def forward(
-        if self.attn_backend == _Backend.FLASH_ATTN:
-            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
-        elif self.attn_backend == _Backend.XFORMERS:
-            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
+            # pre-compute cu_seqlens for window attn
+            if self.attn_backend == _Backend.FLASH_ATTN:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/custom_inputs.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15273 - [Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/15273
- Status/date: merged / 2025-03-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `47c712621316`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +35/-16, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: - #15200 actually disabled the seqlens pre-computation for Qwen2.5-VL by mistake, because seqlens are computed for each layer again. - This PR adds the optimization introduced i....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10 (33 lines); hunks: -608,6 +608,17 @@ def get_window_index(self, grid_thw):; -645,25 +656,27 @@ def forward(; symbols: get_window_index, compute_attn_mask_seqlen, forward, touching `get_window_index, compute_attn_mask_seqlen, forward`; `vllm/model_executor/models/qwen2_vl.py` modified +12/-6 (18 lines); hunks: -617,6 +617,16 @@ def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:; -638,12 +648,8 @@ def forward(; symbols: rot_pos_emb, compute_attn_mask_seqlen, forward, touching `rot_pos_emb, compute_attn_mask_seqlen, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10 (33 lines); hunks: -608,6 +608,17 @@ def get_window_index(self, grid_thw):; -645,25 +656,27 @@ def forward(; symbols: get_window_index, compute_attn_mask_seqlen, forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-6 (18 lines); hunks: -617,6 +617,16 @@ def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:; -638,12 +648,8 @@ def forward(; symbols: rot_pos_emb, compute_attn_mask_seqlen, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -608,6 +608,17 @@ def get_window_index(self, grid_thw):
+    def compute_attn_mask_seqlen(
+        self,
+        cu_seqlens: torch.Tensor,
+    ) -> tuple[Optional[int], Optional[list[int]]]:
+        max_seqlen, seqlens = None, None
+        if self.attn_backend == _Backend.FLASH_ATTN:
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -617,6 +617,16 @@ def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
+    def compute_attn_mask_seqlen(
+            self, cu_seqlens: torch.Tensor
+    ) -> tuple[Optional[int], Optional[list[int]]]:
+        max_seqlen, seqlens = None, None
+        if self.attn_backend == _Backend.FLASH_ATTN:
+            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10; `vllm/model_executor/models/qwen2_vl.py` modified +12/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15130 - [Model][VLM] Add Qwen2.5-Omni model support (thinker only)

- Link: https://github.com/vllm-project/vllm/pull/15130
- Status/date: merged / 2025-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `examples/offline_inference/qwen2_5_omni/README.md`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `2c1bd848a668`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +1852/-82, 2363 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Qwen2.5-Omni model support (thinker only)"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`; PR body summary: This PR adding support for Qwen2.5-Omni model (thinker only). Requirements This PR requires this corresponding transformers PR. **Note: You need to install transformers from sou....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo, touching `_qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data`; `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__, touching `forward, all_gather_interleave, Qwen2_5_VisionAttention`; `examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query, touching `QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query`; `examples/offline_inference/qwen2_5_omni/README.md` added +32/-0 (32 lines); hunks: -0,0 +1,32.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__
  - `examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query
  - `examples/offline_inference/qwen2_5_omni/README.md` added +32/-0 (32 lines); hunks: -0,0 +1,32
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -0,0 +1,977 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -38,13 +38,14 @@
-from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
+from vllm.distributed import parallel_state
+                                               QKVParallelLinear,
@@ -195,6 +196,23 @@ def forward(self, x: torch.Tensor):
+def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
+    """All-gather the input tensor interleavely across model parallel group."""
diff -- examples/offline_inference/qwen2_5_omni/only_thinker.py
@@ -0,0 +1,160 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0; `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28
  - docs: `examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0; `examples/offline_inference/qwen2_5_omni/README.md` added +32/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16872 - [Model] Qwen2.5-Omni Cleanup

- Link: https://github.com/vllm-project/vllm/pull/16872
- Status/date: merged / 2025-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `5124f5bf51b8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +2/-5, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Qwen2.5-Omni Cleanup"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: Clean up some oversighted changes from #15130.
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3 (3 lines); hunks: -518,9 +518,6 @@ def _apply_hf_processor_main(; symbols: _apply_hf_processor_main, touching `_apply_hf_processor_main`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3 (3 lines); hunks: -518,9 +518,6 @@ def _apply_hf_processor_main(; symbols: _apply_hf_processor_main
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -518,9 +518,6 @@ def _apply_hf_processor_main(
-        print(prompt)
-        print(hf_processor_mm_kwargs)
-        print(mm_items)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16907 - [Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni

- Link: https://github.com/vllm-project/vllm/pull/16907
- Status/date: merged / 2025-04-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `26c0406555a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: This PR fixes the distribution bug in Qwen2.5-VL & Qwen2.5-Omni (introduced in #15130). When `tp_group != world_size`, current implementation will cause an allgather error..
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2 (3 lines); hunks: -198,9 +198,8 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, touching `forward, all_gather_interleave`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2 (3 lines); hunks: -198,9 +198,8 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -198,9 +198,8 @@ def forward(self, x: torch.Tensor):
-    import torch.distributed as dist
-    dist.all_gather(gathered_tensors, local_tensor)
+    parallel_state.get_tp_group().all_gather(gathered_tensors, local_tensor)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16974 - [Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni

- Link: https://github.com/vllm-project/vllm/pull/16974
- Status/date: merged / 2025-04-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `571e8dd65e2a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: I am very sorry that previous fix #16907 still have distributed bug, need to fix it agin. Related PR: #15130 and #16907.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -198,8 +198,11 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, touching `forward, all_gather_interleave`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -198,8 +198,11 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -198,8 +198,11 @@ def forward(self, x: torch.Tensor):
+    import torch.distributed as dist
-    parallel_state.get_tp_group().all_gather(gathered_tensors, local_tensor)
+    dist.all_gather(gathered_tensors,
+                    local_tensor,
+                    group=parallel_state.get_tp_group().device_group)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17301 - [Misc] Clean up Qwen2.5-Omni code

- Link: https://github.com/vllm-project/vllm/pull/17301
- Status/date: merged / 2025-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `8b464d9660a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +75/-94, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Qwen2.5-Omni code"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: Split up `apply` even more so that the override in Qwen2.5-Omni doesn't have to repeat as much code.
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51 (59 lines); hunks: -51,11 +51,9; -279,46 +277,17 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, apply, _maybe_apply_prompt_updates, _get_prompt_updates, touching `_get_mm_fields_config, apply, _maybe_apply_prompt_updates`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51 (59 lines); hunks: -51,11 +51,9; -279,46 +277,17 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, apply, _maybe_apply_prompt_updates, _get_prompt_updates
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -51,11 +51,9 @@
-from vllm.multimodal.hasher import MultiModalHasher
-                                    MultiModalInputs, MultiModalKwargs,
-                                    NestedTensors)
+                                    MultiModalKwargs, NestedTensors)
@@ -279,46 +277,17 @@ def _get_mm_fields_config(
-    def apply(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/multimodal/processing.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17726 - [Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE

- Link: https://github.com/vllm-project/vllm/pull/17726
- Status/date: merged / 2025-05-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `c3e9d5060e89`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-12, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: - Since we have ported FA's RoPE kernel in `vllm_flash_attn`, there is no need to use original FA's `apply_rotary_emb` anymore - Replace original FA's `apply_rotary_emb` with `v....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7 (9 lines); hunks: -297,13 +297,8 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/qwen2_vl.py` modified +4/-5 (9 lines); hunks: -64,7 +64,7; -230,14 +230,13 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, touching `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7 (9 lines); hunks: -297,13 +297,8 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +4/-5 (9 lines); hunks: -64,7 +64,7; -230,14 +230,13 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -297,13 +297,8 @@ def forward(
-            use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
-            q = apply_rotary_pos_emb_vision(q,
-                                            rotary_pos_emb,
-                                            use_flash_attn=use_flash_attn)
-            k = apply_rotary_pos_emb_vision(k,
-                                            rotary_pos_emb,
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -64,7 +64,7 @@
-from vllm.platforms import _Backend
+from vllm.platforms import _Backend, current_platform
@@ -230,14 +230,13 @@ def apply_rotary_emb_torch(x: torch.Tensor,
-                                freqs: torch.Tensor,
-                                use_flash_attn=False) -> torch.Tensor:
+                                freqs: torch.Tensor) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7; `vllm/model_executor/models/qwen2_vl.py` modified +4/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17838 - [Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor

- Link: https://github.com/vllm-project/vllm/pull/17838
- Status/date: merged / 2025-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `015815fe0141`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +15/-9, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: As discussed in https://vllm-dev.slack.com/archives/C07QCGVDNUF/p1746687303802249?thread_ts=1745944078.067259&cid=C07QCGVDNUF I decided not to add regression tests because the d....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3 (8 lines); hunks: -145,9 +145,11 @@ def get_hf_processor(; symbols: get_hf_processor, touching `get_hf_processor`; `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3 (8 lines); hunks: -758,9 +758,11 @@ def get_hf_processor(; symbols: get_hf_processor, touching `get_hf_processor`; `vllm/model_executor/models/qwen2_vl.py` modified +5/-3 (8 lines); hunks: -759,9 +759,11 @@ def get_hf_processor(; symbols: get_hf_processor, touching `get_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3 (8 lines); hunks: -145,9 +145,11 @@ def get_hf_processor(; symbols: get_hf_processor
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3 (8 lines); hunks: -758,9 +758,11 @@ def get_hf_processor(; symbols: get_hf_processor
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-3 (8 lines); hunks: -759,9 +759,11 @@ def get_hf_processor(; symbols: get_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -145,9 +145,11 @@ def get_hf_processor(
-            image_processor=self.get_image_processor(min_pixels=min_pixels,
-                                                     max_pixels=max_pixels,
-                                                     size=size),
+            image_processor=self.get_image_processor(
+                min_pixels=min_pixels,
+                max_pixels=max_pixels,
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -758,9 +758,11 @@ def get_hf_processor(
-            image_processor=self.get_image_processor(min_pixels=min_pixels,
-                                                     max_pixels=max_pixels,
-                                                     size=size),
+            image_processor=self.get_image_processor(
+                min_pixels=min_pixels,
+                max_pixels=max_pixels,
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -759,9 +759,11 @@ def get_hf_processor(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3; `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3; `vllm/model_executor/models/qwen2_vl.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17973 - [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…

- Link: https://github.com/vllm-project/vllm/pull/17973
- Status/date: merged / 2025-05-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `67da5720d4ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +121/-83, 285 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: Description of Problem In Qwen2.5-VL rotary position embedding constant tensors creates in the beginning of model's `forward`. Before this PR there were a mix of CPU and GPU ten....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83 (204 lines); hunks: -25,7 +25,7; -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:; symbols: __init__, dtype, device, rot_pos_emb, touching `__init__, dtype, device`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83 (204 lines); hunks: -25,7 +25,7; -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:; symbols: __init__, dtype, device, rot_pos_emb
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -25,7 +25,7 @@
-from functools import partial
+from functools import lru_cache, partial
@@ -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:
-        inv_freq = 1.0 / (theta
-                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
+        inv_freq = 1.0 / (theta**(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19054 - [Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl

- Link: https://github.com/vllm-project/vllm/pull/19054
- Status/date: merged / 2025-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `ec2dcd80bc17`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-8, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: FIX #18976 - Transformers v4.52 will map weights name for Qwen2-VL/Qwen2.5-VL: https://github.com/huggingface/transformers/blob/de4cf5a38e9678b9e465867a8a6b88ea727bea52/src/tran....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4 (13 lines); hunks: -823,10 +823,15 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Suppor...; symbols: Qwen2_5_VLForConditionalGeneration, __init__, touching `Qwen2_5_VLForConditionalGeneration, __init__`; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -1071,10 +1071,15 @@ class Qwen2VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2VLForConditionalGeneration, __init__, touching `Qwen2VLForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4 (13 lines); hunks: -823,10 +823,15 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Suppor...; symbols: Qwen2_5_VLForConditionalGeneration, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -1071,10 +1071,15 @@ class Qwen2VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2VLForConditionalGeneration, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -823,10 +823,15 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, SupportsMultiModal,
-    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
-        "lm_head.": "language_model.lm_head.",
-        "model.": "language_model.model.",
-    })
+    hf_to_vllm_mapper = WeightsMapper(
+        orig_to_new_prefix={
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -1071,10 +1071,15 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
-    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
-        "lm_head.": "language_model.lm_head.",
-        "model.": "language_model.model.",
-    })
+    hf_to_vllm_mapper = WeightsMapper(
+        orig_to_new_prefix={
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22069 - [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform.

- Link: https://github.com/vllm-project/vllm/pull/22069
- Status/date: merged / 2025-08-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `d3a6f2120bb6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +64/-39, 212 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform."; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5 (18 lines); hunks: -250,11 +250,15 @@ def __init__(; -301,10 +305,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen, touching `__init__, split_qkv, forward`; `vllm/model_executor/models/qwen2_vl.py` modified +13/-5 (18 lines); hunks: -274,10 +274,14 @@ def __init__(; -324,10 +328,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen, touching `__init__, split_qkv, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5 (18 lines); hunks: -250,11 +250,15 @@ def __init__(; -301,10 +305,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_vl.py` modified +13/-5 (18 lines); hunks: -274,10 +274,14 @@ def __init__(; -324,10 +328,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -250,11 +250,15 @@ def __init__(
-                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
+                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
+                _Backend.ROCM_AITER_FA
+        self.is_flash_attn_backend = self.attn_backend in {
+            _Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA
+        }
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -274,10 +274,14 @@ def __init__(
-                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
+                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
+                _Backend.ROCM_AITER_FA
+        self.is_flash_attn_backend = self.attn_backend in {
+            _Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA
+        }
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5; `vllm/model_executor/models/qwen2_vl.py` modified +13/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22184 - [Model] Switch to Fused RMS norm in Qwen2.5_VL model.

- Link: https://github.com/vllm-project/vllm/pull/22184
- Status/date: merged / 2025-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `cbc8457b2663`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-7, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Switch to Fused RMS norm in Qwen2.5_VL model."; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7 (14 lines); hunks: -396,13 +396,13 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7 (14 lines); hunks: -396,13 +396,13 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -396,13 +396,13 @@ def forward(
-        x = x + self.attn(self.norm1(x),
-                          cu_seqlens=cu_seqlens,
-                          rotary_pos_emb=rotary_pos_emb,
-                          max_seqlen=max_seqlen,
-                          seqlens=seqlens)
-        x = x + self.mlp(self.norm2(x))
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23058 - [Bugfix] fix Qwen2.5-Omni processor output mapping

- Link: https://github.com/vllm-project/vllm/pull/23058
- Status/date: merged / 2025-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `9f1c6422549d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix Qwen2.5-Omni processor output mapping"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: Fix #23056 Add a processor output mapping for Qwen2.5-Omni thinker, to compute video rotary embedding correctly. Use the script I provided in #23056 and check the rotary positio....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0 (5 lines); hunks: -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str...; symbols: _qwen2_5_omni_thinker_field_config, touching `_qwen2_5_omni_thinker_field_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0 (5 lines); hunks: -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str...; symbols: _qwen2_5_omni_thinker_field_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str, torch.Tensor]):
+    # vllm use `second_per_grid_ts` to compute multimodal rotary embedding
+    video_second_per_grid = hf_inputs.get("video_second_per_grid", None)
+    if video_second_per_grid is not None:
+        hf_inputs["second_per_grid_ts"] = video_second_per_grid
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23512 - [Bugfix] Fix Qwen2.5-VL quantized model weights loading

- Link: https://github.com/vllm-project/vllm/pull/23512
- Status/date: merged / 2025-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `a71e4765cc0c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2.5-VL quantized model weights loading"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: This PR makes sure quantized model weights are loaded correctly. Currently, loading `RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic` will crash on A100s: The issue is introduced in....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -135,7 +135,7 @@ class Qwen2_5_VLVideoPixelInputs(TypedDict):; -852,6 +852,10 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration, touching `Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -135,7 +135,7 @@ class Qwen2_5_VLVideoPixelInputs(TypedDict):; -852,6 +852,10 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -135,7 +135,7 @@ class Qwen2_5_VLVideoPixelInputs(TypedDict):
-    The video time interval (in seconds) for each grid along the temporal
+    The video time interval (in seconds) for each grid along the temporal
@@ -852,6 +852,10 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, SupportsMultiModal,
+    packed_modules_mapping = {
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24231 - [LoRA]: Add lora support to qwen-2.5-omni

- Link: https://github.com/vllm-project/vllm/pull/24231
- Status/date: merged / 2025-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `c9f7081f9c84`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-3, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA]: Add lora support to qwen-2.5-omni"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: Add LoRA support to `qwen-2.5-omni`, scoped strictly to the language model components. - Implements `SupportsLoRA` on `Qwen2_5OmniThinkerForConditionalGeneration`. - Adds `packe....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2 (15 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs, get_language_model, touching `_process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2 (15 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs, get_language_model
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -41,6 +41,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -66,7 +67,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -705,7 +707,7 @@ def _process_video_input(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24420 - [Model] Enable BNB support for qwen2_5_omni_thinker

- Link: https://github.com/vllm-project/vllm/pull/24420
- Status/date: merged / 2025-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `6f4a82f8b5a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-2, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Enable BNB support for qwen2_5_omni_thinker"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: - Enable bitsandbytes quantization for qwen2_5_omni_thinker, fix https://github.com/vllm-project/vllm/issues/23240 - Add `get_mm_mapping` and `SupportLoRA`, deleted by https://g....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2 (31 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str, load_weights, touching `_process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2 (31 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -41,6 +41,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -66,7 +67,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -726,14 +728,30 @@ def _process_video_input(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24741 - [Models] Prevent CUDA sync in Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/24741
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `b0d1213ac395`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Prevent CUDA sync in Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: This is a followup to #24443 from @david6666666 When I profiled Qwen2.5-VL it seems like an implicit CUDA sync is still happening during the indexing: https://github.com/vllm-pr....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -64,6 +64,7; -737,7 +738,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation, forward, touching `compute_attn_mask_seqlen, invert_permutation, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -64,6 +64,7; -737,7 +738,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -64,6 +64,7 @@
+from vllm.utils import is_pin_memory_available
@@ -737,7 +738,7 @@ def compute_attn_mask_seqlen(
-        inv = torch.empty_like(perm)
+        inv = torch.empty_like(perm, pin_memory=is_pin_memory_available())
@@ -808,6 +809,8 @@ def forward(
+        reverse_indices = reverse_indices.to(device=hidden_states.device,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24727 - [Model] Support Qwen3-VL Model Series

- Link: https://github.com/vllm-project/vllm/pull/24727
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `0f7acdd73ca6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +2084/-17, 2262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Qwen3-VL Model Series"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: This PR adds model support for the upcoming Qwen3-VL models, including both dense and MoE variants. Reference HF implementation - https://github.com/huggingface/transformers/pul....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunks: -0,0 +1,1478; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP, touching `Qwen3_VisionPatchEmbed, __init__, forward`; `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__, touching `Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel`; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -83,7 +83,7.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunks: -0,0 +1,1478; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -83,7 +83,7
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -0,0 +1,1478 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
diff -- vllm/model_executor/models/qwen3_vl_moe.py
@@ -0,0 +1,344 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -83,7 +83,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` added +1478/-0; `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24955 - [MM Encoder] Apply DP ViT for Qwen3-VL model series

- Link: https://github.com/vllm-project/vllm/pull/24955
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `3127274d022b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +77/-19, 256 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM Encoder] Apply DP ViT for Qwen3-VL model series"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; PR body summary: Add DP ViT for Qwen3-VL models - this PR should be merged only after #24727 is merged. Part of #22743 Running on `Qwen3-VL-30B-A3B-Instruct` with 4xL40s (PCI-E) with the followi....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +75/-19 (94 lines); hunks: -126,20 +126,23 @@ def __init__(self,; -158,23 +161,27 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -315,12 +315,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +75/-19 (94 lines); hunks: -126,20 +126,23 @@ def __init__(self,; -158,23 +161,27 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -315,12 +315,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -126,20 +126,23 @@ def __init__(self,
-                 prefix: str = ""):
+                 prefix: str = "",
+                 use_data_parallel: bool = False):
-                                               prefix=f"{prefix}.linear_fc1")
+                                               prefix=f"{prefix}.linear_fc1",
+                                               disable_tp=use_data_parallel)
diff -- vllm/model_executor/models/qwen3_vl_moe.py
@@ -315,12 +315,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
+            use_data_parallel=self.use_data_parallel,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +75/-19; `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25055 - [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE

- Link: https://github.com/vllm-project/vllm/pull/25055
- Status/date: merged / 2025-09-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +88/-46, 263 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/kernels/core/test_mrope.py`; PR body summary: - Following PR for #24727, implement corresponding Triton kernel for interleaved MRoPE. Test should pass.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunks: -15,7 +15,7; -30,12 +30,14 @@ def _triton_qwen2vl_mrope_forward(; symbols: _triton_qwen2vl_mrope_forward, _triton_mrope_forward, touching `_triton_qwen2vl_mrope_forward, _triton_mrope_forward`; `tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunks: -1,9 +1,12; -15,6 +18,7 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_...; symbols: generate_test_data, unroll_model_tp_dict, MRoPETestInfo, test_mrope, touching `generate_test_data, unroll_model_tp_dict, MRoPETestInfo`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunks: -15,7 +15,7; -30,12 +30,14 @@ def _triton_qwen2vl_mrope_forward(; symbols: _triton_qwen2vl_mrope_forward, _triton_mrope_forward
  - `tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunks: -1,9 +1,12; -15,6 +18,7 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_...; symbols: generate_test_data, unroll_model_tp_dict, MRoPETestInfo, test_mrope
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/mrope.py
@@ -15,7 +15,7 @@
-def _triton_qwen2vl_mrope_forward(
+def _triton_mrope_forward(
@@ -30,12 +30,14 @@ def _triton_qwen2vl_mrope_forward(
+    mrope_section_w: tl.constexpr,
+    is_interleaved: tl.constexpr,
-    # instead of (3, bsz, seq_len, head_dim)
diff -- tests/kernels/core/test_mrope.py
@@ -1,9 +1,12 @@
+from typing import NamedTuple
+from packaging.version import Version
+from transformers import __version__ as TRANSFORMERS_VERSION
@@ -15,6 +18,7 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_kv_heads: int,
+    current_platform.seed_everything(42)
@@ -33,43 +37,67 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_kv_heads: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14
  - tests: `tests/kernels/core/test_mrope.py` modified +66/-32
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_mrope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25300 - [Bugfix] Fix Qwen3-VL-MoE weight loading for EP

- Link: https://github.com/vllm-project/vllm/pull/25300
- Status/date: merged / 2025-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `be874c020196`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-5, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL-MoE weight loading for EP"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl_moe.py`; PR body summary: Previously `self.load_fused_expert_weights` check is too strict and will prevent server from launching with `--enable-expert-parallel`. This PR fixes it. The MMMU from server la....
- Key implementation: `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5 (12 lines); hunks: -122,9 +122,10 @@ def forward(; -133,9 +134,10 @@ def load_fused_expert_weights(self, name: str, params_dict:...; symbols: forward, load_fused_expert_weights, load_weights, __init__, touching `forward, load_fused_expert_weights, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5 (12 lines); hunks: -122,9 +122,10 @@ def forward(; -133,9 +134,10 @@ def load_fused_expert_weights(self, name: str, params_dict:...; symbols: forward, load_fused_expert_weights, load_weights, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl_moe.py
@@ -122,9 +122,10 @@ def forward(
-                                  num_experts: int):
+                                  num_experts: int) -> bool:
+        loaded_local_expert = False
@@ -133,9 +134,10 @@ def load_fused_expert_weights(self, name: str, params_dict: dict,
-            if not success:
-                return False
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25337 - [MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`

- Link: https://github.com/vllm-project/vllm/pull/25337
- Status/date: merged / 2025-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `30d08911f7cf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +60/-75, 177 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: 10 QPS of VisionArena on Qwen3-VL 4B on A100 Main This branch MMMU matched.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +60/-75 (135 lines); hunks: -270,6 +270,7 @@ def __init__(; -377,82 +378,68 @@ def rot_pos_emb(self, grid_thw):; symbols: __init__, rot_pos_emb, fast_pos_embed_interpolate, compute_attn_mask_seqlen, touching `__init__, rot_pos_emb, fast_pos_embed_interpolate`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +60/-75 (135 lines); hunks: -270,6 +270,7 @@ def __init__(; -377,82 +378,68 @@ def rot_pos_emb(self, grid_thw):; symbols: __init__, rot_pos_emb, fast_pos_embed_interpolate, compute_attn_mask_seqlen
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -270,6 +270,7 @@ def __init__(
+        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
@@ -377,82 +378,68 @@ def rot_pos_emb(self, grid_thw):
-    def fast_pos_embed_interpolate(self, grid_thw):
-        num_grid_per_side = int(self.num_position_embeddings**0.5)
+    def fast_pos_embed_interpolate(self,
+                                   grid_thw: list[list[int]]) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +60/-75
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25347 - [Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`

- Link: https://github.com/vllm-project/vllm/pull/25347
- Status/date: merged / 2025-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `af7dfb0d1a95`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-18, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: - Following PR for #25337 - Just found that we can further optimize weights computation for `fast_pos_embed_interpolate` by reducing duplicated multiply. - Also improve the read....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +32/-18 (50 lines); hunks: -406,25 +406,39 @@ def fast_pos_embed_interpolate(self,; symbols: fast_pos_embed_interpolate, touching `fast_pos_embed_interpolate`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +32/-18 (50 lines); hunks: -406,25 +406,39 @@ def fast_pos_embed_interpolate(self,; symbols: fast_pos_embed_interpolate
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -406,25 +406,39 @@ def fast_pos_embed_interpolate(self,
-            w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
-            w01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
-            w10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
-            w11 = (dh[:, None] * dw[None, :]).reshape(-1)
-            idx00 = (h_floor[:, None] * num_grid_per_side +
-                     w_floor[None, :]).reshape(-1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +32/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25445 - [Model] Enable DP for ViT in Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/25445
- Status/date: merged / 2025-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `c98be0a23276`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +59/-19, 252 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Enable DP for ViT in Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Part of https://github.com/vllm-project/vllm/issues/22743.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +59/-19 (78 lines); hunks: -66,6 +66,7; -217,17 +218,20 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +59/-19 (78 lines); hunks: -66,6 +66,7; -217,17 +218,20 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -66,6 +66,7 @@
+from vllm.multimodal.utils import run_dp_sharded_mrope_vision_model
@@ -217,17 +218,20 @@ def __init__(
+        use_data_parallel: bool = False,
-                                        prefix=f"{prefix}.fc1")
+                                        prefix=f"{prefix}.fc1",
+                                        disable_tp=use_data_parallel)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +59/-19
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25646 - [Misc] Fix Qwen3-VL `video_grid_thw` typing

- Link: https://github.com/vllm-project/vllm/pull/25646
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `7be9ffcd9f5c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Fix Qwen3-VL `video_grid_thw` typing"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: Without this change, torch will throw a warning `vllm/vllm/model_executor/models/qwen3_vl.py:480: UserWarning: To copy construct from a tensor, it is recommended to use sourceTe....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -1249,7 +1249,7 @@ def _process_video_input(; symbols: _process_video_input, touching `_process_video_input`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -1249,7 +1249,7 @@ def _process_video_input(; symbols: _process_video_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1249,7 +1249,7 @@ def _process_video_input(
-                                           grid_thw=grid_thw)
+                                           grid_thw=grid_thw_list)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25648 - [Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling

- Link: https://github.com/vllm-project/vllm/pull/25648
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `17b4c6685ce6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +13/-1, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: - Suspend of #25557 before we support configurable mm profiling.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +12/-0 (12 lines); hunks: -715,6 +715,18 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor, touching `_get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor`; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +12/-0 (12 lines); hunks: -715,6 +715,18 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -715,6 +715,18 @@ def _get_dummy_videos(
+    def get_dummy_processor_inputs(self, seq_len, mm_counts):
+        processor_inputs = super().get_dummy_processor_inputs(
+            seq_len, mm_counts)
+        # HACK(Isotr0py): We set do_resize to False here to reuse Qwen2-VL's
+        # profiling logic, which will be problematic for configurable mm
+        # profiling.
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -82,7 +82,7 @@
-_MAX_FRAMES_PER_VIDEO = 600
+_MAX_FRAMES_PER_VIDEO = 32
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +12/-0; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25788 - [Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL

- Link: https://github.com/vllm-project/vllm/pull/25788
- Status/date: merged / 2025-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `c242c98031b8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +75/-51, 208 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; PR body summary: Temple fix for https://github.com/vllm-project/vllm/issues/25582 Special thanks to @ywang96 for the context The default XFORMERS backend has problem and we will meet repeated to....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36 (73 lines); hunks: -274,6 +274,8 @@ def __init__(; -300,25 +302,8 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/qwen3_vl.py` modified +38/-15 (53 lines); hunks: -63,7 +63,7; -158,6 +158,8 @@ def __init__(; symbols: __init__, dtype, touching `__init__, dtype`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36 (73 lines); hunks: -274,6 +274,8 @@ def __init__(; -300,25 +302,8 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/qwen3_vl.py` modified +38/-15 (53 lines); hunks: -63,7 +63,7; -158,6 +158,8 @@ def __init__(; symbols: __init__, dtype
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -274,6 +274,8 @@ def __init__(
+        attn_backend: _Backend = _Backend.TORCH_SDPA,
+        use_upstream_fa: bool = False,
@@ -300,25 +302,8 @@ def __init__(
-        # Detect attention implementation.
-        self.attn_backend = get_vit_attn_backend(
-            head_size=self.hidden_size_per_attention_head,
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -63,7 +63,7 @@
-from vllm.platforms import _Backend
+from vllm.platforms import _Backend, current_platform
@@ -158,6 +158,8 @@ def __init__(
+        attn_backend: _Backend = _Backend.TORCH_SDPA,
+        use_upstream_fa: bool = False,
@@ -170,7 +172,9 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36; `vllm/model_executor/models/qwen3_vl.py` modified +38/-15
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25557 - [VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling

- Link: https://github.com/vllm-project/vllm/pull/25557
- Status/date: merged / 2025-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `0efd540dbc54`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +74/-9, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: - Qwen3-VL's max_num_video_tokens calculation is implemented wrong, this PR corrects it.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +65/-5 (70 lines); hunks: -33,11 +33,14; -85,6 +88,9; symbols: Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames, get_num_frames_with_most_features, touching `Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames`; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -79,7 +79,7; -932,6 +932,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens, _get_max_video_frames, touching `get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +65/-5 (70 lines); hunks: -33,11 +33,14; -85,6 +88,9; symbols: Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames, get_num_frames_with_most_features
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -79,7 +79,7; -932,6 +932,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens, _get_max_video_frames
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -33,11 +33,14 @@
-from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
+from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
+    smart_resize as image_smart_resize)
+from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
+    smart_resize as video_smart_resize)
@@ -85,6 +88,9 @@
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -79,7 +79,7 @@
-_MAX_FRAMES_PER_VIDEO = 32
+_MAX_FRAMES_PER_VIDEO = 14
@@ -932,6 +932,7 @@ def get_num_image_tokens(
+            num_frames=1,
@@ -956,6 +957,7 @@ def get_image_size_with_most_features(self) -> ImageSize:
+            num_frames=1,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +65/-5; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26000 - [MM] Add text-only mode for Qwen3-VL

- Link: https://github.com/vllm-project/vllm/pull/26000
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `66bca9b8bd0a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +45/-26, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM] Add text-only mode for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; PR body summary: Since this model is performing pretty well on text only tasks we might want to allow people to serve it as a text-only model. Running `vllm serve Qwen/Qwen3-VL-235B-A22B-Instruc....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +25/-14 (39 lines); hunks: -1125,14 +1125,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1148,11 +1151,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__, compute_logits, load_weights, get_mm_mapping, touching `__init__, compute_logits, load_weights`; `vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12 (32 lines); hunks: -319,13 +319,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -341,10 +345,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +25/-14 (39 lines); hunks: -1125,14 +1125,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1148,11 +1151,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__, compute_logits, load_weights, get_mm_mapping
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12 (32 lines); hunks: -319,13 +319,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -341,10 +345,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1125,14 +1125,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
-        self.visual = Qwen3_VisionTransformer(
-            config.vision_config,
-            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
-            quant_config=quant_config,
-            prefix=maybe_prefix(prefix, "visual"),
-            use_data_parallel=self.use_data_parallel,
diff -- vllm/model_executor/models/qwen3_vl_moe.py
@@ -319,13 +319,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.visual = Qwen3_VisionTransformer(
-            config.vision_config,
-            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
-            quant_config=quant_config,
-            prefix=maybe_prefix(prefix, "visual"),
-            use_data_parallel=self.use_data_parallel,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +25/-14; `vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26004 - [BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker

- Link: https://github.com/vllm-project/vllm/pull/26004
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `84d57342b66e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-3, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: Fix https://github.com/vllm-project/vllm/issues/25970 When omni is handling the cached video, it will produced `mm_kwargs: {'video': [None]}`. Then the error will occur when acc....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -324,9 +324,15 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, touching `_maybe_apply_prompt_updates`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -324,9 +324,15 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -324,9 +324,15 @@ def _maybe_apply_prompt_updates(
-        use_audio_in_video = (all(
-            item["use_audio_in_video"].data
-            for item in mm_kwargs["video"]) if "video" in mm_kwargs else False)
+        use_audio_in_video = False
+        if "video" in mm_kwargs:
+            video_items = [
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24642 - [Qwen][ROCm] Flash Attention Rotary Embeddings

- Link: https://github.com/vllm-project/vllm/pull/24642
- Status/date: merged / 2025-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `5e4a8223c644`, `dd96465fd744`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +28/-5, 80 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen][ROCm] Flash Attention Rotary Embeddings"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Qwen VL models previously relied on a basic PyTorch implementation for applying rotary positional embeddings on ROCm architectures. This PR adds a ROCm specialisation to use fla....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +5/-5 (10 lines); hunks: -50,6 +50,8; -63,7 +65,7; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, touching `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-5 (10 lines); hunks: -50,6 +50,8; -63,7 +65,7; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -50,6 +50,8 @@
+from vllm.model_executor.layers.rotary_embedding.common import (
+    dispatch_rotary_emb_function)
@@ -63,7 +65,7 @@
-from vllm.platforms import _Backend, current_platform
+from vllm.platforms import _Backend
@@ -272,13 +274,11 @@ def apply_rotary_emb_torch(x: torch.Tensor,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26123 - [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642

- Link: https://github.com/vllm-project/vllm/pull/26123
- Status/date: merged / 2025-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `dd96465fd744`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-4, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: 24642 introduced dispatch_rotary_emb_function method in common.py while for non_cuda or non_rocm, apply_rotary_emb_torch is differently defined in some modeling codes, so provid....
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +2/-1 (3 lines); hunks: -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, touching `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +2/-1 (3 lines); hunks: -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,
-    rotary_emb_function = dispatch_rotary_emb_function()
+    rotary_emb_function = dispatch_rotary_emb_function(
+        default=apply_rotary_emb_torch)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25550 - Add Qwen3-Omni moe thinker

- Link: https://github.com/vllm-project/vllm/pull/25550
- Status/date: merged / 2025-10-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `19a9b169bf1b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +1795/-36, 1940 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Qwen3-Omni moe thinker"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This PR from the Qwen team for: qwen3-omni-moe thinker part. Testing has been conducted internally across four configurations (v0/v1, eager/CUDA) on several representative bench....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunks: -0,0 +1,1409; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP, touching `Qwen3_VisionPatchEmbed, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunks: -0,0 +1,1409; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -0,0 +1,1409 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26608 - [MM] Move Qwen3Omni MRoPE impl to model file

- Link: https://github.com/vllm-project/vllm/pull/26608
- Status/date: merged / 2025-10-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `ddaff2938e0b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +368/-387, 859 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM] Move Qwen3Omni MRoPE impl to model file"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Move the mrope impl to the model file, and clean up some code there as well..
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26 (355 lines); hunks: -72,7 +72,12; -96,7 +101,7; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__, get_supported_mm_limits, touching `_get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26 (355 lines); hunks: -72,7 +72,12; -96,7 +101,7; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__, get_supported_mm_limits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -72,7 +72,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsMRoPE,
+    SupportsMultiModal,
+    SupportsPP,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26815 - [Bugfix] Fix qwen3-omni audio truncation issue

- Link: https://github.com/vllm-project/vllm/pull/26815
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `8c851f6d044b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-2, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix qwen3-omni audio truncation issue"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: - Fix #26630 - Before https://github.com/huggingface/transformers/pull/41473, Qwen3-omni will still truncate audio to 30s. - This PR enforces no truncation in Qwen3-omni process....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2 (18 lines); hunks: -30,7 +30,9; -711,11 +713,12 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length, touching `pad_to_hop_length`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2 (18 lines); hunks: -30,7 +30,9; -711,11 +713,12 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -30,7 +30,9 @@
+from packaging.version import Version
+from transformers import __version__ as TRANSFORMERS_VERSION
@@ -711,11 +713,12 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
+        feature_extractor = self.info.get_feature_extractor()
+        hop_length = feature_extractor.hop_length
-            hop_length = self.info.get_feature_extractor().hop_length
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27104 - [bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True

- Link: https://github.com/vllm-project/vllm/pull/27104
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `4c91a28e301d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: The current code overwrites `video_fps` with `sampled_fps`, which causes incorrect timestamp calculations. @ywang96 Hey bro, could you please help review this? Thanks!.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -735,9 +735,9 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, touching `_get_video_second_idx`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -735,9 +735,9 @@ def _get_video_second_idx(; symbols: _get_video_second_idx
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -735,9 +735,9 @@ def _get_video_second_idx(
-            video_fps = sampled_fps if sampled_fps else video_processor.fps
+            sampled_fps = sampled_fps if sampled_fps else video_processor.fps
-            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
+            num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27190 - [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA

- Link: https://github.com/vllm-project/vllm/pull/27190
- Status/date: merged / 2025-10-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +46/-12, 106 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py`; PR body summary: The refactor introduced in the following PR: https://github.com/vllm-project/vllm/pull/26104 improved the flash-attn selection, but broke the loading of models like Qwen/Qwen3-V....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -429,6 +429,12 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -462,6 +462,12 @@ def forward(; symbols: forward, touching `forward`; `vllm/attention/layer.py` modified +29/-11 (40 lines); hunks: -47,6 +47,12; -96,18 +102,29 @@ def maybe_get_vit_flash_attn_backend(; symbols: maybe_get_vit_flash_attn_backend, forward, touching `maybe_get_vit_flash_attn_backend, forward`; `vllm/platforms/rocm.py` modified +5/-1 (6 lines); hunks: -205,12 +205,16 @@ class RocmPlatform(Platform):; symbols: RocmPlatform, get_vit_attn_backend, touching `RocmPlatform, get_vit_attn_backend`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -429,6 +429,12 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -462,6 +462,12 @@ def forward(; symbols: forward
  - `vllm/attention/layer.py` modified +29/-11 (40 lines); hunks: -47,6 +47,12; -96,18 +102,29 @@ def maybe_get_vit_flash_attn_backend(; symbols: maybe_get_vit_flash_attn_backend, forward
  - `vllm/platforms/rocm.py` modified +5/-1 (6 lines); hunks: -205,12 +205,16 @@ class RocmPlatform(Platform):; symbols: RocmPlatform, get_vit_attn_backend
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -429,6 +429,12 @@ def forward(
+            from vllm.platforms import current_platform
+            if current_platform.is_rocm():
+                q = q.contiguous()
+                k = k.contiguous()
+                v = v.contiguous()
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -462,6 +462,12 @@ def forward(
+            from vllm.platforms import current_platform
+            if current_platform.is_rocm():
+                q = q.contiguous()
+                k = k.contiguous()
+                v = v.contiguous()
diff -- vllm/attention/layer.py
@@ -47,6 +47,12 @@
+if current_platform.is_rocm():
+    from vllm.platforms.rocm import on_gfx9
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0; `vllm/model_executor/models/qwen2_vl.py` modified +6/-0; `vllm/attention/layer.py` modified +29/-11; `vllm/platforms/rocm.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/attention/layer.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27705 - [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes

- Link: https://github.com/vllm-project/vllm/pull/27705
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `0d8161b07504`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +17/-16, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: 23207 Broke Qwen3VL since it relies on `Qwen2_5_VisionAttention` as well but still used the old signature: This PR updates Qwen3VL and Qwen3Omni to fix this. Note: I didn't test....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -223,8 +223,8 @@ def forward(; -488,12 +488,13 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen, get_placeholder_str, touching `forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen`; `vllm/model_executor/models/qwen3_vl.py` modified +7/-6 (13 lines); hunks: -231,8 +231,8 @@ def forward(; -512,15 +512,16 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen, touching `forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen`; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4 (6 lines); hunks: -836,10 +836,8 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, touching `compute_attn_mask_seqlen`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -223,8 +223,8 @@ def forward(; -488,12 +488,13 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen, get_placeholder_str
  - `vllm/model_executor/models/qwen3_vl.py` modified +7/-6 (13 lines); hunks: -231,8 +231,8 @@ def forward(; -512,15 +512,16 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4 (6 lines); hunks: -836,10 +836,8 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -223,8 +223,8 @@ def forward(
-        max_seqlen: int | None = None,  # Only used for Flash Attention
-        seqlens: list[int] | None = None,  # Only used for xFormers
+        max_seqlen: torch.Tensor,  # Only used for Flash Attention
+        seqlens: torch.Tensor,  # Only used for xFormers
@@ -488,12 +488,13 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
-    ) -> tuple[int | None, list[int] | None]:
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -231,8 +231,8 @@ def forward(
-        max_seqlen: int | None = None,  # Only used for Flash Attention
-        seqlens: list[int] | None = None,  # Only used for xFormers
+        max_seqlen: torch.Tensor,  # Only used for Flash Attention
+        seqlens: torch.Tensor,  # Only used for xFormers
@@ -512,15 +512,16 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
-    ) -> tuple[int | None, list[int] | None]:
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -836,10 +836,8 @@ def compute_attn_mask_seqlen(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6; `vllm/model_executor/models/qwen3_vl.py` modified +7/-6; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27790 - [BugFix][VL] Fix FA selection on Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/27790
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `e806178d2a9b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +20/-12, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][VL] Fix FA selection on Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: https://github.com/vllm-project/vllm/pull/27190 breaks AMD CI (and also qwen2.5 vl): tests/v1/entrypoints/openai/responses/test_image.py : with _Backend.FLASH_ATTN it did NOT se....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11 (30 lines); hunks: -43,10 +43,7; -318,6 +315,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11 (30 lines); hunks: -43,10 +43,7; -318,6 +315,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -43,10 +43,7 @@
-from vllm.attention.layer import (
-    check_upstream_fa_availability,
-    maybe_get_vit_flash_attn_backend,
-)
+from vllm.attention.layer import maybe_get_vit_flash_attn_backend
@@ -318,6 +315,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27920 - [Bugfix] Fix Qwen Omni audio inference

- Link: https://github.com/vllm-project/vllm/pull/27920
- Status/date: merged / 2025-11-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `853a8eb53b89`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +2/-10, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen Omni audio inference"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: FIX https://github.com/vllm-project/vllm/issues/27906 FIX #27907.
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7 (9 lines); hunks: -130,6 +130,8 @@ class Qwen2_5OmniAudioFeatureInputs(TensorSchema):; -732,13 +734,6 @@ def _process_audio_input(; symbols: Qwen2_5OmniAudioFeatureInputs, _process_audio_input, touching `Qwen2_5OmniAudioFeatureInputs, _process_audio_input`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3 (3 lines); hunks: -99,7 +99,6; -1065,8 +1064,6 @@ def _process_audio_input(; symbols: _process_audio_input, touching `_process_audio_input`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7 (9 lines); hunks: -130,6 +130,8 @@ class Qwen2_5OmniAudioFeatureInputs(TensorSchema):; -732,13 +734,6 @@ def _process_audio_input(; symbols: Qwen2_5OmniAudioFeatureInputs, _process_audio_input
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3 (3 lines); hunks: -99,7 +99,6; -1065,8 +1064,6 @@ def _process_audio_input(; symbols: _process_audio_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -130,6 +130,8 @@ class Qwen2_5OmniAudioFeatureInputs(TensorSchema):
+    audio_feature_lengths: Annotated[torch.Tensor, TensorShape("na")]
@@ -732,13 +734,6 @@ def _process_audio_input(
-        if audio_feature_lengths.shape[0] == 1:
-            audio_feature_lengths = audio_feature_lengths.squeeze(0)
-        elif audio_feature_lengths.shape[1] == 1:
-            audio_feature_lengths = audio_feature_lengths.squeeze(1)
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -99,7 +99,6 @@
-    flatten_bn,
@@ -1065,8 +1064,6 @@ def _process_audio_input(
-        audio_feature_lengths = flatten_bn(audio_feature_lengths, concat=True)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28271 - [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL

- Link: https://github.com/vllm-project/vllm/pull/28271
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `48b8456ff992`, `bc5bd45c7d1a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +1/-42, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: This code path uses head-parallel attention, where each rank holds full Q/K/V vectors for its own subset of heads. All attention backends operate per-rank on local heads. Theref....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30 (30 lines); hunks: -291,25 +291,6 @@ def forward(self, x: torch.Tensor):; -383,21 +364,10 @@ def __init__(; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__, touching `forward, all_gather_interleave, Qwen2_5_VisionAttention`; `vllm/model_executor/models/qwen2_vl.py` modified +1/-12 (13 lines); hunks: -50,7 +50,7; -396,21 +396,10 @@ def __init__(; symbols: __init__, split_qkv, touching `__init__, split_qkv`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30 (30 lines); hunks: -291,25 +291,6 @@ def forward(self, x: torch.Tensor):; -383,21 +364,10 @@ def __init__(; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-12 (13 lines); hunks: -50,7 +50,7; -396,21 +396,10 @@ def __init__(; symbols: __init__, split_qkv
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -291,25 +291,6 @@ def forward(self, x: torch.Tensor):
-def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
-    """All-gather the input tensor interleavely across model parallel group."""
-    import torch.distributed as dist
-    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
-    dist.all_gather(
-        gathered_tensors, local_tensor, group=parallel_state.get_tp_group().device_group
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -50,7 +50,7 @@
-from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
+from vllm.distributed import parallel_state
@@ -396,21 +396,10 @@ def __init__(
-        if self.tp_size > 1:
-            qkv = tensor_model_parallel_all_gather(qkv)
-        # 3 * [s, b, head * head_dim]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30; `vllm/model_executor/models/qwen2_vl.py` modified +1/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28663 - [Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure

- Link: https://github.com/vllm-project/vllm/pull/28663
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `cec275efcef6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-3, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: A quantized Qwen3-VL-32B model using GPTQModel==5.4.0 fails to load with `vllm serve Guan1794/qwen3VL-32B-4bit-gptq` The error message is: The GPTQModel quantization config: Run....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +3/-1 (4 lines); hunks: -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +3/-1 (4 lines); hunks: -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.model = Qwen3LLMModel(vllm_config=vllm_config, prefix=prefix)
+        self.model = Qwen3LLMModel(
+            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29232 - Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/29232
- Status/date: merged / 2025-11-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `d84d8f4429a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-1, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; PR body summary: **Summary** Fix a crash in `Qwen2.5-VL` when EVS (Efficient Video Sampling) is enabled and video inputs are provided through the `video_embeds` pathway. The regression occurred....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1 (17 lines); hunks: -230,6 +230,9 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; -244,6 +247,11 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; symbols: Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input, _postprocess_video_embeds_evs, touching `Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1 (17 lines); hunks: -230,6 +230,9 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; -244,6 +247,11 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; symbols: Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input, _postprocess_video_embeds_evs
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -230,6 +230,9 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):
+        - second_per_grid_ts: The video time interval (in seconds) for each
+          grid along the temporal dimension in the 3D position IDs. Returned
+          when `videos` is not `None`.
@@ -244,6 +247,11 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):
+    second_per_grid_ts: Annotated[
+        torch.Tensor | None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27721 - [Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.

- Link: https://github.com/vllm-project/vllm/pull/27721
- Status/date: merged / 2025-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `examples/offline_inference/qwen3_omni/only_thinker.py`, `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `839c6b7b72bc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +467/-59, 631 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine."; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: FIX #22268 FIX #22364 CLOSE #23888 CLOSE #25473 CLOSE https://github.com/vllm-project/vllm/issues/28046 Sanity checked with https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3....
- Key implementation: `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer, touching `print_input_ids, mock_qwen3_omni_config, mock_processor`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video, touching `_maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders, touching `_maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders`; `examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query, touching `QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query`.
- Code diff details:
  - `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders
  - `examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query
- Key code excerpts:

```diff
diff -- tests/model_executor/test_qwen3_omni.py
@@ -0,0 +1,221 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import Mock
+import pytest
+from transformers import PretrainedConfig
+from vllm.multimodal.processing import InputProcessingContext
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -68,11 +68,11 @@
-    BaseMultiModalProcessor,
+    PromptUpdateDetails,
@@ -87,7 +87,6 @@
-    Qwen2_5OmniThinkerProcessingInfo,
@@ -807,24 +806,8 @@ def _maybe_apply_prompt_updates(
-        if use_audio_in_video and "video" in mm_item_counts:
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -23,7 +23,6 @@
```

- Reviewed files:
  - tests: `tests/model_executor/test_qwen3_omni.py` added +221/-0
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25
  - docs: `examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_qwen3_omni.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29896 - feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE

- Link: https://github.com/vllm-project/vllm/pull/29896
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `a2b053dc858d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-0, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Add the necessary attributes for BitsAndBytes quantization support: - Add packed_modules_mapping with qkv_proj and gate_up_proj mappings - Add get_mm_mapping() method for multim....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0 (23 lines); hunks: -62,6 +62,7; -1137,6 +1138,18 @@ class Qwen3OmniMoeThinkerForConditionalGeneration(; symbols: Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions, get_mm_mapping, touching `Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0 (23 lines); hunks: -62,6 +62,7; -1137,6 +1138,18 @@ class Qwen3OmniMoeThinkerForConditionalGeneration(; symbols: Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions, get_mm_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -62,6 +62,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -1137,6 +1138,18 @@ class Qwen3OmniMoeThinkerForConditionalGeneration(
+    packed_modules_mapping = {
+        "qkv_proj": [
+            "q_proj",
+            "k_proj",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29974 - [ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni

- Link: https://github.com/vllm-project/vllm/pull/29974
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `3f1b03739ae1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This is a bugfix for qwen3-omni model when using AITER Flash Attention Evaluate the qwen3-omni chartqa.
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1 (5 lines); hunks: -494,7 +494,10 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, touching `compute_attn_mask_seqlen`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1 (5 lines); hunks: -494,7 +494,10 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -494,7 +494,10 @@ def compute_attn_mask_seqlen(
-        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
+        if self.attn_backend in {
+            AttentionBackendEnum.FLASH_ATTN,
+            AttentionBackendEnum.ROCM_AITER_FA,
+        }:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30037 - support qwen3-vl handle requests with embeddings

- Link: https://github.com/vllm-project/vllm/pull/30037
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `6dcb07f676ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support qwen3-vl handle requests with embeddings"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: 为了支持qwen3-vl 处理带image embedding的请求，需要使用自定义的 函数，用 类替代 即可 Test Script.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +5/-2 (7 lines); hunks: -103,7 +103,7; -884,7 +884,10 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser, _call_hf_processor, touching `_get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +5/-2 (7 lines); hunks: -103,7 +103,7; -884,7 +884,10 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser, _call_hf_processor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -103,7 +103,7 @@
-from .qwen2_vl import Qwen2VLProcessingInfo
+from .qwen2_vl import Qwen2VLMultiModalDataParser, Qwen2VLProcessingInfo
@@ -884,7 +884,10 @@ def _get_dummy_videos(
-        return MultiModalDataParser(video_needs_metadata=True)
+        return Qwen2VLMultiModalDataParser(
+            self.info.get_hf_config().vision_config.spatial_merge_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30542 - [Bugfix] Revert Qwen2-VL part of change in #28271

- Link: https://github.com/vllm-project/vllm/pull/30542
- Status/date: merged / 2025-12-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `48b8456ff992`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-1, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Revert Qwen2-VL part of change in #28271"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; PR body summary: This PR reverts qwen2_vl.py part of commit bc5bd45c7d1abbac4a63d97d383212c108e55308. Fixes #30250, the model generates incorrect output when TP is used..
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +12/-1 (13 lines); hunks: -49,7 +49,7; -359,10 +359,21 @@ def __init__(; symbols: __init__, split_qkv, touching `__init__, split_qkv`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-1 (13 lines); hunks: -49,7 +49,7; -359,10 +359,21 @@ def __init__(; symbols: __init__, split_qkv
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -49,7 +49,7 @@
-from vllm.distributed import parallel_state
+from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
@@ -359,10 +359,21 @@ def __init__(
+        if self.tp_size > 1:
+            qkv = tensor_model_parallel_all_gather(qkv)
+        # 3 * [s, b, head * head_dim]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +12/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29752 - [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL

- Link: https://github.com/vllm-project/vllm/pull/29752
- Status/date: merged / 2025-12-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `ae88aada38ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +424/-12, 539 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: This PR implements **EVS (Efficient Video Sampling)** support for Qwen3-VL models, enabling **dynamic video token pruning** to improve inference efficiency while maintaining acc....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +424/-12 (436 lines); hunks: -67,12 +67,19; -92,6 +99,7; symbols: get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__, _process_video_input, touching `get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +424/-12 (436 lines); hunks: -67,12 +67,19; -92,6 +99,7; symbols: get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__, _process_video_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -67,12 +67,19 @@
+from vllm.multimodal.evs import (
+    compute_mrope_for_media,
+    compute_retained_tokens_count,
+    compute_retention_mask,
+    recompute_mrope_positions,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +424/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30883 - [Chore] Remove v0 dead code for Qwen2.5-omni

- Link: https://github.com/vllm-project/vllm/pull/30883
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `6fe588765287`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-22, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Remove v0 dead code for Qwen2.5-omni"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: - Just found we missed Qwen2.5-omni's `embed_multimodal_v0` when removing `embed_multimodal_v0` from models..
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22 (22 lines); hunks: -70,7 +70,6; -1150,27 +1149,6 @@ def embed_input_ids(; symbols: embed_input_ids, embed_multimodal_v0, forward, touching `embed_input_ids, embed_multimodal_v0, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22 (22 lines); hunks: -70,7 +70,6; -1150,27 +1149,6 @@ def embed_input_ids(; symbols: embed_input_ids, embed_multimodal_v0, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -70,7 +70,6 @@
-    NestedTensors,
@@ -1150,27 +1149,6 @@ def embed_input_ids(
-    def embed_multimodal_v0(self, **kwargs: object) -> NestedTensors | None:
-        audio_input = self._parse_and_validate_audio_input(**kwargs)
-        image_input = self._parse_and_validate_image_input(**kwargs)
-        video_input = self._parse_and_validate_video_input(**kwargs)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31007 - [Qwen3-Omni] fixed _get_feat_extract_output_lengths function

- Link: https://github.com/vllm-project/vllm/pull/31007
- Status/date: merged / 2025-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `bb24592d139b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-12, 65 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3-Omni] fixed _get_feat_extract_output_lengths function"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; no usable PR-body summary.
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12 (20 lines); hunks: -118,7 +118,7 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Te...; -921,13 +921,11 @@ def _get_prompt_updates(; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates, _process_audio_input, touching `_get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12 (20 lines); hunks: -118,7 +118,7 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Te...; -921,13 +921,11 @@ def _get_prompt_updates(; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates, _process_audio_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -118,7 +118,7 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
-    return feat_lengths, output_lengths
+    return output_lengths
@@ -921,13 +921,11 @@ def _get_prompt_updates(
-            _, audio_output_lens = _get_feat_extract_output_lengths(
-                audio_feature_lengths
-            )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31436 - Add GLM-ASR multimodal support

- Link: https://github.com/vllm-project/vllm/pull/31436
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; associated commits `d722e9e614f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +764/-2, 833 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add GLM-ASR multimodal support"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; PR body summary: Support that model :https://huggingface.co/zai-org/GLM-ASR-Nano-2512 for download and convert test audio: https://paste.ubuntu.com/p/K7S9Thmvfg/ Start server: bench_glm_asr_e2e.....
- Key implementation: `vllm/model_executor/models/glmasr.py` added +545/-0 (545 lines); hunks: -0,0 +1,545; symbols: GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector, __init__, touching `GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector`; `vllm/model_executor/models/glmasr_utils.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: _calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts, _get_audio_output_lengths_from_lengths, touching `_calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts`.
- Code diff details:
  - `vllm/model_executor/models/glmasr.py` added +545/-0 (545 lines); hunks: -0,0 +1,545; symbols: GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector, __init__
  - `vllm/model_executor/models/glmasr_utils.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: _calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts, _get_audio_output_lengths_from_lengths
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glmasr.py
@@ -0,0 +1,545 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable, Mapping, Sequence
+from typing import Annotated, Any, Literal, TypeAlias, cast
+import numpy as np
+import torch
diff -- vllm/model_executor/models/glmasr_utils.py
@@ -0,0 +1,165 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from typing import cast
+import torch
+import torch.nn as nn
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glmasr.py` added +545/-0; `vllm/model_executor/models/glmasr_utils.py` added +165/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29255 - Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring

- Link: https://github.com/vllm-project/vllm/pull/29255
- Status/date: merged / 2026-01-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `97a01308e9ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +312/-3, 337 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: The Qwen3OmniMoeProcessor was losing the audio_sample_rate parameter during kwargs restructuring for transformers.
- Key implementation: `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count, TestQwen3OmniAudioSampleRatePreservation, touching `test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0 (25 lines); hunks: -751,6 +751,9 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np....; -760,6 +763,28 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np...; symbols: pad_to_hop_length, touching `pad_to_hop_length`.
- Code diff details:
  - `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count, TestQwen3OmniAudioSampleRatePreservation
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0 (25 lines); hunks: -751,6 +751,9 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np....; -760,6 +763,28 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np...; symbols: pad_to_hop_length
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_qwen3_omni.py
@@ -0,0 +1,285 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for Qwen3 Omni audio processing and sample rate handling."""
+from typing import Any
+import numpy as np
+import pytest
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -751,6 +751,9 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
+                # Extract audio_sample_rate before restructuring
+                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)
@@ -760,6 +763,28 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
+                # Validate and conditionally pass audio_sample_rate
+                # WhisperFeatureExtractor has a fixed sampling rate, and vLLM's
+                # audio loader already resamples audio to the target rate.
```

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_qwen3_omni.py`, `tests/multimodal/test_processing.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31790 - [Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)

- Link: https://github.com/vllm-project/vllm/pull/31790
- Status/date: merged / 2026-01-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `2c1a4f2488da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-6, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This PR fixes an issue in Qwen3OmniMoeThinker where `audio_kwargs/text_kwargs` could be overridden when restructuring kwargs for `Transformers < 4.58.0`. We now preserve existin....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -750,18 +750,20 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length, touching `pad_to_hop_length`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -750,18 +750,20 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -750,18 +750,20 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
+            mm_kwargs["audio_kwargs"] = dict(mm_kwargs.get("audio_kwargs") or {})
+            mm_kwargs["text_kwargs"] = dict(mm_kwargs.get("text_kwargs") or {})
-                mm_kwargs["audio_kwargs"] = {
-                    "truncation": mm_kwargs.pop("truncation", False)
-                }
-                mm_kwargs["text_kwargs"] = {
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31779 - [Refactor] GLM-ASR Modeling

- Link: https://github.com/vllm-project/vllm/pull/31779
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; associated commits `974138751bdb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +672/-41, 868 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] GLM-ASR Modeling"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; PR body summary: Key Improvements **1. Native vLLM Audio Encoder Implementation** (`glmasr.py`) - Completely rewrote `GlmAsrEncoder` as a vLLM-native implementation with full optimization supp....
- Key implementation: `vllm/model_executor/models/glmasr.py` modified +644/-36 (680 lines); hunks: -8,18 +8,22; -35,6 +39,8; symbols: GlmAsrEncoderRotaryEmbedding, __init__, forward, GlmAsrEncoderAttention, touching `GlmAsrEncoderRotaryEmbedding, __init__, forward`; `vllm/model_executor/models/glmasr_utils.py` modified +28/-5 (33 lines); hunks: -71,14 +71,37 @@ def _get_audio_output_lengths_for_tower(; symbols: _get_audio_output_lengths_for_tower, _flatten_audio_features_by_length, touching `_get_audio_output_lengths_for_tower, _flatten_audio_features_by_length`.
- Code diff details:
  - `vllm/model_executor/models/glmasr.py` modified +644/-36 (680 lines); hunks: -8,18 +8,22; -35,6 +39,8; symbols: GlmAsrEncoderRotaryEmbedding, __init__, forward, GlmAsrEncoderAttention
  - `vllm/model_executor/models/glmasr_utils.py` modified +28/-5 (33 lines); hunks: -71,14 +71,37 @@ def _get_audio_output_lengths_for_tower(; symbols: _get_audio_output_lengths_for_tower, _flatten_audio_features_by_length
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glmasr.py
@@ -8,18 +8,22 @@
-from transformers.models.glmasr import GlmAsrConfig, GlmAsrEncoder, GlmAsrProcessor
+from transformers.models.glmasr import GlmAsrConfig, GlmAsrProcessor
+from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
+from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
+    QKVParallelLinear,
+from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
diff -- vllm/model_executor/models/glmasr_utils.py
@@ -71,14 +71,37 @@ def _get_audio_output_lengths_for_tower(
+    """
+    Calculate the output lengths after audio processing.
+    The output length accounts for:
+    1. Convolution layers (downsampling)
+    2. Merge factor (further downsampling during projection)
+    Args:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +644/-36; `vllm/model_executor/models/glmasr_utils.py` modified +28/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31890 - [Models] Allow converting Qwen3-VL into Reranker model

- Link: https://github.com/vllm-project/vllm/pull/31890
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` found it through `examples/pooling/score/template/qwen3_vl_reranker.jinja`; associated commits `eac3b96ec04d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +287/-13, 415 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Allow converting Qwen3-VL into Reranker model"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `examples/pooling/score/template/qwen3_vl_reranker.jinja`; PR body summary: - Enable reranker support for Qwen3-VL.
- Key implementation: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23.
- Code diff details:
  - `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23
- Key code excerpts:

```diff
diff -- examples/pooling/score/template/qwen3_vl_reranker.jinja
@@ -0,0 +1,23 @@
+<|im_start|>system
+Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
+<|im_start|>user
+<Instruct>: {{
+    messages
+    | selectattr("role", "eq", "system")
```

- Reviewed files:
  - docs: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32126 - [Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/32126
- Status/date: merged / 2026-01-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `542a4059b2bb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +113/-190, 377 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; PR body summary: Followup to #28399 and #28730. This PR optimizes the M-RoPE position computation for both **Qwen2-VL** and **Qwen2.5-VL**. - Leveraging the pre-calculated mm_features to compute....
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95 (152 lines); hunks: -26,11 +26,12; -1044,121 +1045,82 @@ class Qwen2_5_VLForConditionalGeneration(; symbols: Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str, touching `Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions`; `vllm/model_executor/models/qwen2_vl.py` modified +56/-95 (151 lines); hunks: -26,7 +26,7; -1137,121 +1137,82 @@ class Qwen2VLForConditionalGeneration(; symbols: Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str, touching `Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95 (152 lines); hunks: -26,11 +26,12; -1044,121 +1045,82 @@ class Qwen2_5_VLForConditionalGeneration(; symbols: Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str
  - `vllm/model_executor/models/qwen2_vl.py` modified +56/-95 (151 lines); hunks: -26,7 +26,7; -1137,121 +1137,82 @@ class Qwen2VLForConditionalGeneration(; symbols: Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -26,11 +26,12 @@
-from collections.abc import Callable, Iterable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
+import numpy as np
@@ -1044,121 +1045,82 @@ class Qwen2_5_VLForConditionalGeneration(
+    def iter_mm_grid_thw(
+        self, mm_features: list[MultiModalFeatureSpec]
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -26,7 +26,7 @@
-from collections.abc import Callable, Iterable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
@@ -1137,121 +1137,82 @@ class Qwen2VLForConditionalGeneration(
+    def iter_mm_grid_thw(
+        self, mm_features: list[MultiModalFeatureSpec]
+    ) -> Iterator[tuple[int, int, int, int, float]]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95; `vllm/model_executor/models/qwen2_vl.py` modified +56/-95
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32167 - [Model] Re-implement Qwen3Omni Audio Encoder

- Link: https://github.com/vllm-project/vllm/pull/32167
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `b8199f604931`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +428/-29, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Re-implement Qwen3Omni Audio Encoder"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Re-implement Qwen3-Omni Audio Encoder with vLLM primitives with some vectorization improvements. - roughly 10% speedup at high batch sizes according to profiling run with TP=1.....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29 (457 lines); hunks: -31,29 +31,34; -104,11 +109,6; symbols: _get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__, forward, touching `_get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29 (457 lines); hunks: -31,29 +31,34; -104,11 +109,6; symbols: _get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -31,29 +31,34 @@
-from transformers import PretrainedConfig
-from transformers import __version__ as TRANSFORMERS_VERSION
+    Qwen3OmniMoeAudioEncoderConfig,
-from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
-    Qwen3OmniMoeAudioEncoder,
-)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32540 - [Bugfix] Fix GLM-ASR audio encoder RoPE dim

- Link: https://github.com/vllm-project/vllm/pull/32540
- Status/date: merged / 2026-01-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glmasr.py`; associated commits `38bf2ffb21d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-30, 98 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix GLM-ASR audio encoder RoPE dim"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/glmasr.py`; PR body summary: - Fix https://github.com/vllm-project/vllm/issues/32445 - GLM-ASR audio encoder's RoPE should be half rotary. - This issue only occured in native code path, because vllm_flash_a....
- Key implementation: `vllm/model_executor/models/glmasr.py` modified +12/-2 (14 lines); hunks: -181,6 +181,12 @@ def __init__(; -226,8 +232,12 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/glmasr.py` modified +12/-2 (14 lines); hunks: -181,6 +181,12 @@ def __init__(; -226,8 +232,12 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glmasr.py
@@ -181,6 +181,12 @@ def __init__(
+        rope_params = getattr(config, "rope_parameters", None)
+        if rope_params:
+            partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.5)
+        else:
+            partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
+        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glmasr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32772 - [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni

- Link: https://github.com/vllm-project/vllm/pull/32772
- Status/date: merged / 2026-01-25
- Trace source: `git log --name-only -- <model-files>` found it through `examples/offline_inference/qwen2_5_omni/only_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `a698e8e7ad4b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +386/-201, 689 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`; PR body summary: - Refactor get_mrope_input_positions() to use mm_feature.mm_position.offset directly - Follows pattern from PR #32126 for Qwen2-VL/2.5-VL Changes - Add iter_mm_features() iterat....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates, touching `_get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates`; `examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0 (26 lines); hunks: -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query, touching `get_multi_audios_query, get_multi_images_query`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates
  - `examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0 (26 lines); hunks: -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -22,10 +22,11 @@
-from collections.abc import Callable, Iterable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
+import numpy as np
@@ -85,6 +86,7 @@
+    PromptUpdateDetails,
@@ -103,7 +105,6 @@
diff -- examples/offline_inference/qwen2_5_omni/only_thinker.py
@@ -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:
+def get_multi_images_query() -> QueryResult:
+    question = "What are the differences between these two images?"
+    prompt = (
+        f"<|im_start|>system\n{default_system}<|im_end|>\n"
+        "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>"
+        "<|vision_bos|><|IMAGE|><|vision_eos|>"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198
  - docs: `examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33010 - [Model] Use mm_position to compute mrope positions for Qwen3-Omni

- Link: https://github.com/vllm-project/vllm/pull/33010
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `examples/offline_inference/qwen3_omni/only_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `6ca2c91b9663`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +293/-298, 675 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_position to compute mrope positions for Qwen3-Omni"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `examples/offline_inference/qwen3_omni/only_thinker.py`; PR body summary: Optimizes M-RoPE position calculation for Qwen3-Omni by using `mm_position.offset` directly from `MultiModalFeatureSpec` instead of token-by-token searching through `input_token....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping, touching `load_weights, get_mrope_input_positions, _compute_audio_token_count`; `examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3 (59 lines); hunks: -2,7 +2,7; -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query, main, parse_args, touching `get_multi_audios_query, get_multi_images_query, main`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping
  - `examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3 (59 lines); hunks: -2,7 +2,7; -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query, main, parse_args
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -22,7 +22,7 @@
-from collections.abc import Callable, Iterable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
@@ -104,10 +104,7 @@
-from .vision import (
-    get_llm_pos_ids_for_vision,
-    get_vit_attn_backend,
diff -- examples/offline_inference/qwen3_omni/only_thinker.py
@@ -2,7 +2,7 @@
-with the correct prompt format on Qwen2.5-Omni (thinker only).
+with the correct prompt format on Qwen3-Omni (thinker only).
@@ -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:
+def get_multi_images_query() -> QueryResult:
+    question = "What are the differences between these two images?"
+    prompt = (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295
  - docs: `examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33312 - [Models] Qwen3-ASR

- Link: https://github.com/vllm-project/vllm/pull/33312
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; associated commits `8b3f0a99dd50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1269/-0, 1335 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Qwen3-ASR"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; PR body summary: Add support for Qwen3-ASR model series - see recipe at https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-ASR.md.
- Key implementation: `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunks: -0,0 +1,567; symbols: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor, touching `_get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config`; `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig, touching `Qwen3ASRAudioEncoderConfig, to, __init__`; `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, __init__, touching `Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunks: -0,0 +1,567; symbols: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor
  - `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig
  - `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr.py
@@ -0,0 +1,567 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2026 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/transformers_utils/configs/qwen3_asr.py
@@ -0,0 +1,436 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa
+# mypy: ignore-errors
+# coding=utf-8
+# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/transformers_utils/processors/qwen3_asr.py
@@ -0,0 +1,231 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` added +567/-0; `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0; `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33410 - [Bugfix] Fix `Qwen3ASR` language asr tag in output

- Link: https://github.com/vllm-project/vllm/pull/33410
- Status/date: merged / 2026-01-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr.py`; associated commits `e77f162cf59d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +42/-2, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix `Qwen3ASR` language asr tag in output"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_asr.py`; PR body summary: It appears Qwen3-ASR necessitates of post-processing to remove spurious language tags `language English ` inserted in the prompt (https://github.com/QwenLM/Qwen3-ASR/blob/c17a13....
- Key implementation: `vllm/model_executor/models/qwen3_asr.py` modified +20/-1 (21 lines); hunks: -90,6 +90,7; -556,7 +557,7 @@ def get_generation_prompt(; symbols: _get_feat_extract_output_lengths, get_generation_prompt, post_process_output, touching `_get_feat_extract_output_lengths, get_generation_prompt, post_process_output`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` modified +20/-1 (21 lines); hunks: -90,6 +90,7; -556,7 +557,7 @@ def get_generation_prompt(; symbols: _get_feat_extract_output_lengths, get_generation_prompt, post_process_output
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr.py
@@ -90,6 +90,7 @@
+_ASR_TEXT_TAG = "<asr_text>"
@@ -556,7 +557,7 @@ def get_generation_prompt(
-                f"<|im_start|>assistant\nlanguage {full_lang_name_to}<asr_text>"
+                f"<|im_start|>assistant\nlanguage {full_lang_name_to}{_ASR_TEXT_TAG}"
@@ -565,3 +566,21 @@ def get_generation_prompt(
+    @classmethod
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +20/-1
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/translations/speech_to_text.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/qwen3_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33077 - [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM

- Link: https://github.com/vllm-project/vllm/pull/33077
- Status/date: merged / 2026-02-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `cd86fff38fee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-7, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This PR fixes a critical crash occurring on AMD (ROCm) hardware when initializing the Qwen3-Omni model (specifically the Qwen3Omni_VisionTransformer). During the memory profilin....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7 (38 lines); hunks: -907,13 +907,37 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7 (38 lines); hunks: -907,13 +907,37 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -907,13 +907,37 @@ def forward(
-        cu_seqlens = torch.repeat_interleave(
-            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
-        ).cumsum(
-            dim=0,
-            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
-        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33644 - [Bugfix] fix qwen3-asr response error

- Link: https://github.com/vllm-project/vllm/pull/33644
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr.py`; associated commits `ceab70c89d2b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-6, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix qwen3-asr response error"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_asr.py`; PR body summary: fix https://github.com/vllm-project/vllm/issues/33643.
- Key implementation: `vllm/model_executor/models/qwen3_asr.py` modified +7/-6 (13 lines); hunks: -125,6 +125,13 @@ def get_feature_extractor(self, **kwargs: object) -> Whispe...; -194,12 +201,6 @@ def _parse_audio_data(; symbols: get_feature_extractor, get_supported_mm_limits, get_data_parser, Qwen3ASRDummyInputsBuilder, touching `get_feature_extractor, get_supported_mm_limits, get_data_parser`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` modified +7/-6 (13 lines); hunks: -125,6 +125,13 @@ def get_feature_extractor(self, **kwargs: object) -> Whispe...; -194,12 +201,6 @@ def _parse_audio_data(; symbols: get_feature_extractor, get_supported_mm_limits, get_data_parser, Qwen3ASRDummyInputsBuilder
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr.py
@@ -125,6 +125,13 @@ def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
+    def get_data_parser(self) -> MultiModalDataParser:
+        feature_extractor = self.get_feature_extractor()
+        return Qwen3ASRMultiModalDataParser(
+            target_sr=feature_extractor.sampling_rate,
+            expected_hidden_size=self._get_expected_hidden_size(),
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +7/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33605 - [Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni

- Link: https://github.com/vllm-project/vllm/pull/33605
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `f8516a1ab95f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +172/-12, 247 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Fix bugs preventing `use_audio_in_video=True` from working correctly with Qwen2.5-Omni and Qwen3-Omni. **~~Bug 1: `KeyError: 'audio'` in `MultiModalBudget`~~** ~~Both `Qwen2_5Om....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3 (126 lines); hunks: -113,6 +113,95; -1286,17 +1375,48 @@ def embed_input_ids(; symbols: check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs, embed_input_ids, touching `check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9 (58 lines); hunks: -92,6 +92,8; -1780,6 +1782,19 @@ def embed_input_ids(; symbols: embed_input_ids, touching `embed_input_ids`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3 (126 lines); hunks: -113,6 +113,95; -1286,17 +1375,48 @@ def embed_input_ids(; symbols: check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs, embed_input_ids
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9 (58 lines); hunks: -92,6 +92,8; -1780,6 +1782,19 @@ def embed_input_ids(; symbols: embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -113,6 +113,95 @@
+def check_interleaved_audio_video(
+    is_video: torch.Tensor,
+    is_audio: torch.Tensor,
+    num_video: int,
+    num_audio: int,
+) -> bool:
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -92,6 +92,8 @@
+    check_interleaved_audio_video,
+    merge_interleaved_embeddings,
@@ -1780,6 +1782,19 @@ def embed_input_ids(
+        # Detect interleaved audio-in-video early, since it affects
+        # both the deepstack path and the final embedding merge.
+        video_token_id = self.config.video_token_id
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29828 - [Model] Add transcription support for Qwen3-Omni

- Link: https://github.com/vllm-project/vllm/pull/29828
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `535de06cb1d9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +104/-2, 177 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add transcription support for Qwen3-Omni"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Only 4 models are supported as a Transcription model according to the docs. This PR adds Qwen3-Omni per the feature request: https://github.com/vllm-project/vllm/issues/29405 Te....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2 (104 lines); hunks: -24,7 +24,7; -48,8 +48,9; symbols: _get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, _compute_interleaved_positions, touching `_get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2 (104 lines); hunks: -24,7 +24,7; -48,8 +48,9; symbols: _get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, _compute_interleaved_positions
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -24,7 +24,7 @@
-from typing import Any
+from typing import Any, Literal, cast
@@ -48,8 +48,9 @@
-from vllm.config import VllmConfig
+from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
+from vllm.inputs.data import PromptType
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34613 - [Realtime] Add Qwen3-ASR realtime streaming support

- Link: https://github.com/vllm-project/vllm/pull/34613
- Status/date: merged / 2026-02-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr_realtime.py`; associated commits `11be2c74dc1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +256/-1, 286 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Realtime] Add Qwen3-ASR realtime streaming support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_asr_realtime.py`; PR body summary: This PR adds real time Websocket streaming transcription support for Qwen3-ASR. Referenced issue: https://github.com/vllm-project/vllm/issues/34232 Machine configuration: Nvidia....
- Key implementation: `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: Qwen3ASRRealtimeBuffer, __init__, write_audio, read_audio, touching `Qwen3ASRRealtimeBuffer, __init__, write_audio`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: Qwen3ASRRealtimeBuffer, __init__, write_audio, read_audio
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr_realtime.py
@@ -0,0 +1,239 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2026 The Qwen team.
+# Copyright 2023 The vLLM team.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35368 - [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression

- Link: https://github.com/vllm-project/vllm/pull/35368
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `c0615a296d44`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +379/-21, 437 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: Fixes #34506 — regression introduced in #33605 where mixed modalities (audio + image + video) no longer worked correctly for Qwen2.5-Omni: the model failed to recognize audio co....
- Key implementation: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo, test_non_interleaved_audio_then_video, touching `make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16 (30 lines); hunks: -1376,23 +1376,12 @@ def embed_input_ids(; -1403,6 +1392,12 @@ def embed_input_ids(; symbols: embed_input_ids, forward, touching `embed_input_ids, forward`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5 (12 lines); hunks: -1904,15 +1904,17 @@ def embed_input_ids(; symbols: embed_input_ids, forward, touching `embed_input_ids, forward`.
- Code diff details:
  - `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo, test_non_interleaved_audio_then_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16 (30 lines); hunks: -1376,23 +1376,12 @@ def embed_input_ids(; -1403,6 +1392,12 @@ def embed_input_ids(; symbols: embed_input_ids, forward
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5 (12 lines); hunks: -1904,15 +1904,17 @@ def embed_input_ids(; symbols: embed_input_ids, forward
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_qwen2_5_omni_embed.py
@@ -0,0 +1,358 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Unit tests for Qwen2.5-Omni embed_input_ids to verify embeddings are
+correctly assigned to audio/image/video token positions.
+Regression test for: https://github.com/vllm-project/vllm/issues/34506
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -1376,23 +1376,12 @@ def embed_input_ids(
-        from .utils import _merge_multimodal_embeddings
-        inputs_embeds = self._embed_text_input_ids(
-            input_ids,
-            self.get_language_model().embed_input_ids,
-            is_multimodal=is_multimodal,
-            handle_oov_mm_token=handle_oov_mm_token,
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1904,15 +1904,17 @@ def embed_input_ids(
```

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35741 - [Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker

- Link: https://github.com/vllm-project/vllm/pull/35741
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `fa6a6be51978`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-0, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: PR #34580 added a `sequence_lengths` parameter to `Qwen2_5_VisionAttention.forward()` for the FlashInfer cuDNN backend and updated callers in `qwen3_vl.py` and `qwen2_5_vl.py`,....
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0 (17 lines); hunks: -648,13 +648,15 @@ def forward(; -975,6 +977,20 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0 (17 lines); hunks: -648,13 +648,15 @@ def forward(; -975,6 +977,20 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -648,13 +648,15 @@ def forward(
+        sequence_lengths: torch.Tensor | None,  # Only used for FlashInfer CuDNN backend
+            sequence_lengths=sequence_lengths,
@@ -975,6 +977,20 @@ def forward(
+        # Recompute cu_seqlens in numpy from grid_thw to avoid GPU->CPU sync
+        grid_thw_np = grid_thw.cpu().numpy()
+        cu_seqlens_np = np.repeat(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35869 - [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile

- Link: https://github.com/vllm-project/vllm/pull/35869
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr_realtime.py`; associated commits `36bf2131816e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-2, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_asr_realtime.py`; PR body summary: `Qwen3ASRForConditionalGeneration` (and its realtime subclass `Qwen3ASRRealtimeGeneration`) crashes during `torch.compile` warmup when `--enforce-eager` is not set. The root cau....
- Key implementation: `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2 (2 lines); hunks: -22,7 +22,6; -177,7 +176,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration, touching `_maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2 (2 lines); hunks: -22,7 +22,6; -177,7 +176,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr_realtime.py
@@ -22,7 +22,6 @@
-from vllm.compilation.decorators import support_torch_compile
@@ -177,7 +176,6 @@ def _maybe_apply_prompt_updates(
-@support_torch_compile
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_asr_realtime.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36140 - [Bugfix] Fix Qwen-VL tokenizer implementation

- Link: https://github.com/vllm-project/vllm/pull/36140
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen_vl.py`, `vllm/tokenizers/qwen_vl.py`; associated commits `719634815791`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +118/-66, 271 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen-VL tokenizer implementation"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/tokenizers/qwen_vl.py`, `vllm/model_executor/models/qwen_vl.py`, `vllm/renderers/qwen_vl.py`; PR body summary: The recent Renderer refactorings broke `Qwen/Qwen-VL` because `_get_tokenizer_without_image_pad` wasn't being used by the Renderer. To solve this, I implemented a custom tokeniz....
- Key implementation: `vllm/tokenizers/qwen_vl.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode, touching `get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize`; `vllm/model_executor/models/qwen_vl.py` modified +2/-64 (66 lines); hunks: -6,11 +6,9; -436,60 +434,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad, tokenize, touching `__init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad`; `vllm/renderers/qwen_vl.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: QwenVLRenderer, from_config, touching `QwenVLRenderer, from_config`.
- Code diff details:
  - `vllm/tokenizers/qwen_vl.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode
  - `vllm/model_executor/models/qwen_vl.py` modified +2/-64 (66 lines); hunks: -6,11 +6,9; -436,60 +434,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad, tokenize
  - `vllm/renderers/qwen_vl.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: QwenVLRenderer, from_config
- Key code excerpts:

```diff
diff -- vllm/tokenizers/qwen_vl.py
@@ -0,0 +1,67 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+import unicodedata
+from collections.abc import Collection, Set
+from transformers import AutoTokenizer
diff -- vllm/model_executor/models/qwen_vl.py
@@ -6,11 +6,9 @@
-import copy
-import unicodedata
-from collections.abc import Callable, Collection, Mapping, Sequence, Set
-from functools import lru_cache, partial
+from collections.abc import Callable, Mapping, Sequence
+from functools import partial
diff -- vllm/renderers/qwen_vl.py
@@ -0,0 +1,29 @@
```

- Reviewed files:
  - runtime: `vllm/tokenizers/qwen_vl.py` added +67/-0; `vllm/model_executor/models/qwen_vl.py` modified +2/-64; `vllm/renderers/qwen_vl.py` added +29/-0
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36108 - refactor funasr model.

- Link: https://github.com/vllm-project/vllm/pull/36108
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `3ee68590c7fa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-57, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor funasr model."; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This PR is a small refactor for the FunASR model. Please take a look @DarkLight1337 @Isotr0py HF's allendou/Fun-ASR-Nano-2512-vllm has also been updated..
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1 (2 lines); hunks: -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModal...; symbols: embed_multimodal, touching `embed_multimodal`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1 (2 lines); hunks: -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModal...; symbols: embed_multimodal
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
-        # tensor correspoending to a multimodal data item (image or video).
+        # tensor corresponding to a multimodal data item (image or video).
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/funasr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/transformers_utils/processors/funasr_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35994 - [BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0

- Link: https://github.com/vllm-project/vllm/pull/35994
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `e998fa76b99a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +86/-0, 107 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`; PR body summary: **Bug Description** When running offline inference with the Qwen2.5-Omni model using audio inputs `python examples/offline_inference/audio_language.py --model-type qwen2_5_omni`....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0 (33 lines); hunks: -353,6 +353,39 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2_5OmniThinkerDummyInputsBuilder, touching `get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0 (33 lines); hunks: -1163,6 +1163,39 @@ def get_feature_extractor(self, **kwargs: object):; symbols: get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item, touching `get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item`; `vllm/model_executor/models/qwen2_audio.py` modified +20/-0 (20 lines); hunks: -179,6 +179,26 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2AudioDummyInputsBuilder, touching `get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0 (33 lines); hunks: -353,6 +353,39 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2_5OmniThinkerDummyInputsBuilder
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0 (33 lines); hunks: -1163,6 +1163,39 @@ def get_feature_extractor(self, **kwargs: object):; symbols: get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item
  - `vllm/model_executor/models/qwen2_audio.py` modified +20/-0 (20 lines); hunks: -179,6 +179,26 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2AudioDummyInputsBuilder
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -353,6 +353,39 @@ def get_target_channels(self) -> int:
+    def get_mm_max_tokens_per_item(
+        self,
+        seq_len: int,
+        mm_counts: Mapping[str, int] | None = None,
+    ) -> Mapping[str, int] | None:
+        mm_counts = mm_counts or {}
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1163,6 +1163,39 @@ def get_feature_extractor(self, **kwargs: object):
+    def get_mm_max_tokens_per_item(
+        self,
+        seq_len: int,
+        mm_counts: Mapping[str, int] | None = None,
+    ) -> Mapping[str, int] | None:
+        mm_counts = mm_counts or {}
diff -- vllm/model_executor/models/qwen2_audio.py
@@ -179,6 +179,26 @@ def get_target_channels(self) -> int:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0; `vllm/model_executor/models/qwen2_audio.py` modified +20/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36319 - Support online use_audio_in_video

- Link: https://github.com/vllm-project/vllm/pull/36319
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +152/-10, 403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support online use_audio_in_video"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/media/audio.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; PR body summary: - Main branch - This PR.
- Key implementation: `vllm/entrypoints/chat_utils.py` modified +54/-8 (62 lines); hunks: -564,7 +564,9 @@ def add(self, modality: ModalityStr, item: _T) -> str | None:; -690,8 +692,10 @@ def resolve_items(; symbols: add, create_parser, resolve_items, touching `add, create_parser, resolve_items`; `vllm/multimodal/media/audio.py` modified +31/-0 (31 lines); hunks: -82,6 +82,35 @@ def extract_audio_from_video_bytes(; -100,6 +129,8 @@ def __init__(self, **kwargs) -> None:; symbols: extract_audio_from_video_bytes, is_video, AudioMediaIO, __init__, touching `extract_audio_from_video_bytes, is_video, AudioMediaIO`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1 (16 lines); hunks: -78,7 +78,11; -811,6 +815,16 @@ def get_replacement_qwen2_use_audio_in_video(item_idx: int):; symbols: get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main, touching `get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main`; `vllm/entrypoints/openai/engine/serving.py` modified +1/-0 (1 lines); hunks: -908,6 +908,7 @@ async def _preprocess_chat(; symbols: _preprocess_chat, touching `_preprocess_chat`.
- Code diff details:
  - `vllm/entrypoints/chat_utils.py` modified +54/-8 (62 lines); hunks: -564,7 +564,9 @@ def add(self, modality: ModalityStr, item: _T) -> str | None:; -690,8 +692,10 @@ def resolve_items(; symbols: add, create_parser, resolve_items
  - `vllm/multimodal/media/audio.py` modified +31/-0 (31 lines); hunks: -82,6 +82,35 @@ def extract_audio_from_video_bytes(; -100,6 +129,8 @@ def __init__(self, **kwargs) -> None:; symbols: extract_audio_from_video_bytes, is_video, AudioMediaIO, __init__
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1 (16 lines); hunks: -78,7 +78,11; -811,6 +815,16 @@ def get_replacement_qwen2_use_audio_in_video(item_idx: int):; symbols: get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main
  - `vllm/entrypoints/openai/engine/serving.py` modified +1/-0 (1 lines); hunks: -908,6 +908,7 @@ async def _preprocess_chat(; symbols: _preprocess_chat
  - `vllm/renderers/params.py` modified +41/-1 (42 lines); hunks: -40,6 +40,34 @@ def merge_kwargs(; -56,12 +84,20 @@ class ChatParams:; symbols: merge_kwargs, recursively_merge_kwargs, ChatParams, with_defaults
- Key code excerpts:

```diff
diff -- vllm/entrypoints/chat_utils.py
@@ -564,7 +564,9 @@ def add(self, modality: ModalityStr, item: _T) -> str | None:
-    def create_parser(self) -> "BaseMultiModalContentParser":
+    def create_parser(
+        self, mm_processor_kwargs: dict[str, Any] | None = None
+    ) -> "BaseMultiModalContentParser":
@@ -690,8 +692,10 @@ def resolve_items(
-    def create_parser(self) -> "BaseMultiModalContentParser":
diff -- vllm/multimodal/media/audio.py
@@ -82,6 +82,35 @@ def extract_audio_from_video_bytes(
+def is_video(data: bytes) -> bool:
+    """Check if the fetched bytes are video"""
+    if len(data) < 12:
+        return False
+    box_type = data[4:8]
+    major_brand = data[8:12]
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -78,7 +78,11 @@
```

- Reviewed files:
  - runtime: `vllm/entrypoints/chat_utils.py` modified +54/-8; `vllm/multimodal/media/audio.py` modified +31/-0; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1; `vllm/entrypoints/openai/engine/serving.py` modified +1/-0; `vllm/renderers/params.py` modified +41/-1; `vllm/renderers/deepseek_v32.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36136 - [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps

- Link: https://github.com/vllm-project/vllm/pull/36136
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `724759684cd9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +116/-4, 150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; PR body summary: Fixes #35909. When `num_frames` is provided via `mm_processor_kwargs` without `fps`, `_get_video_second_idx` was falling back to the default fps (2) to compute the number of fra....
- Key implementation: `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0 (94 lines); hunks: -0,0 +1,94; symbols: _build_video_mm_data, test_processor_num_frames_timestamp, touching `_build_video_mm_data, test_processor_num_frames_timestamp`; `vllm/model_executor/models/qwen3_vl.py` modified +22/-4 (26 lines); hunks: -768,6 +768,7 @@ def _get_video_second_idx(; -782,11 +783,20 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, _call_hf_processor, default, touching `_get_video_second_idx, _call_hf_processor, default`.
- Code diff details:
  - `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0 (94 lines); hunks: -0,0 +1,94; symbols: _build_video_mm_data, test_processor_num_frames_timestamp
  - `vllm/model_executor/models/qwen3_vl.py` modified +22/-4 (26 lines); hunks: -768,6 +768,7 @@ def _get_video_second_idx(; -782,11 +783,20 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, _call_hf_processor, default
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_qwen3_vl.py
@@ -0,0 +1,94 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Regression tests for Qwen3-VL processor.
+Covers the fix for num_frames-based timestamp calculation
+(issue vllm-project/vllm#35909).
+"""
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -768,6 +768,7 @@ def _get_video_second_idx(
+        sampled_num_frames: int | None = None,
@@ -782,11 +783,20 @@ def _get_video_second_idx(
-            # here video_fps is the fps of the sampled video, and
-            # metadata["fps"] refers to the fps of the original video.
-            sampled_fps = sampled_fps if sampled_fps else video_processor.fps
-            num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
```

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +22/-4
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_qwen3_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36800 - [Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request

- Link: https://github.com/vllm-project/vllm/pull/36800
- Status/date: merged / 2026-03-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `abf61aaa8ef2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +128/-12, 169 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: - Following PR for https://github.com/vllm-project/vllm/pull/36319 - Actually, Qwen2.5-Omni and Qwen3-Omni processor's `apply_prompt_updates` is incorrect for `audio_in_video=Tr....
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12 (23 lines); hunks: -80,8 +80,6; -609,6 +607,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main, touching `_maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0 (11 lines); hunks: -1326,6 +1326,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, touching `_maybe_apply_prompt_updates`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12 (23 lines); hunks: -80,8 +80,6; -609,6 +607,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0 (11 lines); hunks: -1326,6 +1326,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -80,8 +80,6 @@
-    ProcessorInputs,
-    TimingContext,
@@ -609,6 +607,17 @@ def _maybe_apply_prompt_updates(
+            # for mutilmodality cache
+            if any(item is None for item in mm_kwargs["video"]):
+                video_token_id = self.info.get_hf_config().video_token_id
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1326,6 +1326,17 @@ def _maybe_apply_prompt_updates(
+            # for mutilmodality cache
+            if any(item is None for item in mm_kwargs["video"]):
+                video_token_id = self.info.get_hf_config().video_token_id
+                audio_token_id = self.info.get_hf_config().audio_token_id
+                video_audio_item_num = sum(
+                    id in (video_token_id, audio_token_id) for id in prompt_ids
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_audio_in_video.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37147 - [Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs

- Link: https://github.com/vllm-project/vllm/pull/37147
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `912fbe9555f9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +117/-17, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: - Fix index out of range when Qwen2.5-Omni/Qwen3-omni received multiple video inputs, because ` audio_in_video_item_idx + item_idx` is incorrect: All newly added tests passed..
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3 (4 lines); hunks: -774,9 +774,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video, touching `get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3 (4 lines); hunks: -1489,9 +1489,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality:...; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video, touching `get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3 (4 lines); hunks: -774,9 +774,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3 (4 lines); hunks: -1489,9 +1489,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality:...; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -774,9 +774,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):
-            audio_num_features = audio_output_lengths[
-                audio_in_video_item_idx + item_idx
-            ]
+            audio_num_features = audio_output_lengths[audio_in_video_item_idx]
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1489,9 +1489,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):
-            audio_num_features = audio_output_lengths[
-                audio_in_video_item_idx + item_idx
-            ]
+            audio_num_features = audio_output_lengths[audio_in_video_item_idx]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/test_audio_in_video.py`, `tests/models/multimodal/processing/test_audio_in_video.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37183 - Remove unused EVS functions in qwen3_vl.py

- Link: https://github.com/vllm-project/vllm/pull/37183
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `43a73f853bac`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-101, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove unused EVS functions in qwen3_vl.py"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: In qwen3_vl.py, functions including _get_evs_mask_segments, _extract_frame_offsets_from_mask, and _get_actual_frame_token_counts were introduced in #29752. However, their calls....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +0/-101 (101 lines); hunks: -1960,107 +1960,6 @@ def _iter_mm_grid_hw(; symbols: _iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask, _get_actual_frame_token_counts, touching `_iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +0/-101 (101 lines); hunks: -1960,107 +1960,6 @@ def _iter_mm_grid_hw(; symbols: _iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask, _get_actual_frame_token_counts
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1960,107 +1960,6 @@ def _iter_mm_grid_hw(
-    def _get_evs_mask_segments(
-        self, mm_position: PlaceholderRange, expected_frames: int
-    ) -> list[torch.Tensor] | None:
-        """Extract contiguous segments from EVS is_embed mask.
-        The EVS (Efficient Video Sampling) mask marks which placeholder
-        positions should be filled with video embeddings. This method splits
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +0/-101
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37439 - [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation

- Link: https://github.com/vllm-project/vllm/pull/37439
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `738d0a281fab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: This PR fixes an incorrect use of `merge_size` in Qwen3-VL video timestamp processing. (This bug also affects Qwen3.5) Previously, `_get_video_second_idx` passed `video_processo....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -767,7 +767,7 @@ def _get_video_second_idx(; -806,7 +806,7 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, touching `_get_video_second_idx`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -767,7 +767,7 @@ def _get_video_second_idx(; -806,7 +806,7 @@ def _get_video_second_idx(; symbols: _get_video_second_idx
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -767,7 +767,7 @@ def _get_video_second_idx(
-        merge_size = video_processor.merge_size
+        temporal_patch_size = video_processor.temporal_patch_size
@@ -806,7 +806,7 @@ def _get_video_second_idx(
-        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
+        timestamps = self._calculate_timestamps(indices, video_fps, temporal_patch_size)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35963 - [Feature] ViT Full CUDA Graph

- Link: https://github.com/vllm-project/vllm/pull/35963
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1584/-31, 1731 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] ViT Full CUDA Graph"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py`; PR body summary: Add full CUDA graph for the ViT to reduce kernel launch overheads. **Features:** - **Budget-based graphs with a maximum batch size**: - Capture CUDA graphs at configurable token....
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +270/-30 (300 lines); hunks: -103,6 +103,7; -528,54 +529,120 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[...; symbols: fast_pos_embed_interpolate, forward, prepare_encoder_metadata, __init__, touching `fast_pos_embed_interpolate, forward, prepare_encoder_metadata`; `vllm/model_executor/models/interfaces.py` modified +141/-0 (141 lines); hunks: -13,6 +13,7; -46,6 +47,11; symbols: supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config, get_encoder_cudagraph_budget_range, touching `supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config`; `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0 (576 lines); hunks: -0,0 +1,576; symbols: BudgetGraphMetadata, EncoderCudaGraphManager, __init__, _generate_budgets, touching `BudgetGraphMetadata, EncoderCudaGraphManager, __init__`; `tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: _make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2, test_max_not_power_of_2, touching `_make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +270/-30 (300 lines); hunks: -103,6 +103,7; -528,54 +529,120 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[...; symbols: fast_pos_embed_interpolate, forward, prepare_encoder_metadata, __init__
  - `vllm/model_executor/models/interfaces.py` modified +141/-0 (141 lines); hunks: -13,6 +13,7; -46,6 +47,11; symbols: supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config, get_encoder_cudagraph_budget_range
  - `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0 (576 lines); hunks: -0,0 +1,576; symbols: BudgetGraphMetadata, EncoderCudaGraphManager, __init__, _generate_budgets
  - `tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: _make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2, test_max_not_power_of_2
  - `vllm/v1/worker/gpu/mm/encoder_cudagraph_defs.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: EncoderCudaGraphConfig, EncoderCudaGraphCaptureInputs, EncoderCudaGraphReplayBuffers
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -103,6 +103,7 @@
+    SupportsEncoderCudaGraph,
@@ -528,54 +529,120 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
-    def forward(
+    def prepare_encoder_metadata(
-        x: torch.Tensor,
-        grid_thw: torch.Tensor | list[list[int]],
diff -- vllm/model_executor/models/interfaces.py
@@ -13,6 +13,7 @@
+    Any,
@@ -46,6 +47,11 @@
+    from vllm.v1.worker.gpu.mm.encoder_cudagraph_defs import (
+        EncoderCudaGraphCaptureInputs,
+        EncoderCudaGraphConfig,
+        EncoderCudaGraphReplayBuffers,
diff -- vllm/v1/worker/gpu/mm/encoder_cudagraph.py
@@ -0,0 +1,576 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +270/-30; `vllm/model_executor/models/interfaces.py` modified +141/-0; `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0; `vllm/v1/worker/gpu/mm/encoder_cudagraph_defs.py` added +66/-0; `vllm/v1/worker/gpu_model_runner.py` modified +48/-1; `vllm/config/compilation.py` modified +32/-0
  - tests: `tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0
- Risk and verification: The diff ships test coverage in `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37247 - [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration

- Link: https://github.com/vllm-project/vllm/pull/37247
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `8d0f908b98cd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +63/-5, 126 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Implement LoRA support for Qwen3ASRForConditionalGeneration"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; PR body summary: This PR adds LoRA support for `Qwen3ASRForConditionalGeneration` model. For this to work for the audio tower, I had to make a few additional changes: - Implement `get_num_mm_enc....
- Key implementation: `vllm/model_executor/models/qwen3_asr.py` modified +26/-0 (26 lines); hunks: -37,6 +37,7; -266,7 +267,21 @@ class Qwen3ASRForConditionalGeneration(; symbols: Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens, get_speech_to_text_config, touching `Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3 (25 lines); hunks: -57,6 +57,7; -357,7 +358,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` modified +26/-0 (26 lines); hunks: -37,6 +37,7; -266,7 +267,21 @@ class Qwen3ASRForConditionalGeneration(; symbols: Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens, get_speech_to_text_config
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3 (25 lines); hunks: -57,6 +57,7; -357,7 +358,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr.py
@@ -37,6 +37,7 @@
+    SupportsLoRA,
@@ -266,7 +267,21 @@ class Qwen3ASRForConditionalGeneration(
+    SupportsLoRA,
+    # LoRA support
+    packed_modules_mapping = {
+        "qkv_proj": [
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -57,6 +57,7 @@
+    ReplicatedLinear,
@@ -357,7 +358,13 @@ def __init__(
-        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)
+        self.conv_out = ReplicatedLinear(
+            conv_out_dim,
+            config.d_model,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +26/-0; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38061 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference

- Link: https://github.com/vllm-project/vllm/pull/38061
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `80118853f42a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +583/-68, 1042 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; PR body summary: Following https://github.com/vllm-project/vllm/pull/35963 (only supports image inference), this PR continues to work on it to support video inference for Qwen3-VL..
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +138/-42 (180 lines); hunks: -99,6 +99,7; -689,6 +690,7 @@ def prepare_encoder_metadata(; symbols: prepare_encoder_metadata, get_encoder_cudagraph_config, touching `prepare_encoder_metadata, get_encoder_cudagraph_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +138/-42 (180 lines); hunks: -99,6 +99,7; -689,6 +690,7 @@ def prepare_encoder_metadata(; symbols: prepare_encoder_metadata, get_encoder_cudagraph_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -99,6 +99,7 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
@@ -689,6 +690,7 @@ def prepare_encoder_metadata(
+        max_frames_per_batch: int | None = None,
@@ -701,6 +703,10 @@ def prepare_encoder_metadata(
+            max_frames_per_batch: If set, overrides max_batch_size for
+                cu_seqlens padding. For video inputs each item contributes
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +138/-42
- Risk and verification: The diff ships test coverage in `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40160 - [Bugfix] Fix k_proj's bias for GLM-ASR

- Link: https://github.com/vllm-project/vllm/pull/40160
- Status/date: merged / 2026-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glmasr.py`; associated commits `aeee7ef93910`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix k_proj's bias for GLM-ASR"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/glmasr.py`; PR body summary: GLM-ASR has `bias = true` only for `q_proj` and `v_proj` not for `k_proj` So, if we use `QKVParallelLinear` to fuse its `qkv_proj`, the bias part for `k_proj` will remain uninit....
- Key implementation: `vllm/model_executor/models/glmasr.py` modified +3/-1 (4 lines); hunks: -66,7 +66,7; -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: GlmAsrEncoderRotaryEmbedding, load_weights, touching `GlmAsrEncoderRotaryEmbedding, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/glmasr.py` modified +3/-1 (4 lines); hunks: -66,7 +66,7; -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: GlmAsrEncoderRotaryEmbedding, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/glmasr.py
@@ -66,7 +66,7 @@
-from .whisper import ISO639_1_SUPPORTED_LANGS
+from .whisper import ISO639_1_SUPPORTED_LANGS, _create_fake_bias_for_k_proj
@@ -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/glmasr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
