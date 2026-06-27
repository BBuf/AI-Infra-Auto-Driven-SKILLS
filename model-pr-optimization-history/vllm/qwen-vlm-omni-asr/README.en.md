# vllm Qwen VLM/Omni/ASR Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/generate/multimodal/qwen2_5_omni/README.md` | no direct PR-number commit |
| `examples/generate/multimodal/qwen2_5_omni/only_thinker.py` | no direct PR-number commit |
| `examples/generate/multimodal/qwen3_omni/only_thinker.py` | no direct PR-number commit |
| `examples/pooling/embed/template/dse_qwen2_vl.jinja` | no direct PR-number commit |
| `examples/pooling/score/template/qwen3_vl_reranker.jinja` | [#31890](https://github.com/vllm-project/vllm/pull/31890) |
| `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py` | [#35415](https://github.com/vllm-project/vllm/pull/35415) |
| `tests/model_executor/test_qwen3_omni.py` | [#27721](https://github.com/vllm-project/vllm/pull/27721) |
| `tests/model_executor/test_qwen3_vl_mrope.py` | no direct PR-number commit |
| `tests/models/multimodal/generation/test_qwen2_5_vl.py` | [#40830](https://github.com/vllm-project/vllm/pull/40830) |
| `tests/models/multimodal/generation/test_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/pooling/test_dse_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/pooling/test_qwen3_asr_forced_aligner.py` | no direct PR-number commit |
| `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` | [#35368](https://github.com/vllm-project/vllm/pull/35368) |
| `tests/models/multimodal/processing/test_qwen2_vl.py` | no direct PR-number commit |
| `tests/models/multimodal/processing/test_qwen3_omni.py` | [#29255](https://github.com/vllm-project/vllm/pull/29255) |
| `tests/models/multimodal/processing/test_qwen3_vl.py` | [#36136](https://github.com/vllm-project/vllm/pull/36136), [#46026](https://github.com/vllm-project/vllm/pull/46026), [#46305](https://github.com/vllm-project/vllm/pull/46305) |
| `vllm/model_executor/models/glmasr.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779), [#32540](https://github.com/vllm-project/vllm/pull/32540), [#40160](https://github.com/vllm-project/vllm/pull/40160) |
| `vllm/model_executor/models/glmasr_utils.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779) |
| `vllm/model_executor/models/mimo_v2_omni.py` | no direct PR-number commit |
| `vllm/model_executor/models/qwen2_5_omni_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#16872](https://github.com/vllm-project/vllm/pull/16872), [#17301](https://github.com/vllm-project/vllm/pull/17301), [#17838](https://github.com/vllm-project/vllm/pull/17838), [#23058](https://github.com/vllm-project/vllm/pull/23058), [#24231](https://github.com/vllm-project/vllm/pull/24231), [#24420](https://github.com/vllm-project/vllm/pull/24420), [#26004](https://github.com/vllm-project/vllm/pull/26004), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#30883](https://github.com/vllm-project/vllm/pull/30883), [#32772](https://github.com/vllm-project/vllm/pull/32772), ... (17 total) |
| `vllm/model_executor/models/qwen2_5_vl.py` | [#12944](https://github.com/vllm-project/vllm/pull/12944), [#13155](https://github.com/vllm-project/vllm/pull/13155), [#13286](https://github.com/vllm-project/vllm/pull/13286), [#13533](https://github.com/vllm-project/vllm/pull/13533), [#13968](https://github.com/vllm-project/vllm/pull/13968), [#14377](https://github.com/vllm-project/vllm/pull/14377), [#15130](https://github.com/vllm-project/vllm/pull/15130), [#15200](https://github.com/vllm-project/vllm/pull/15200), [#15273](https://github.com/vllm-project/vllm/pull/15273), [#16907](https://github.com/vllm-project/vllm/pull/16907), [#16974](https://github.com/vllm-project/vllm/pull/16974), [#17726](https://github.com/vllm-project/vllm/pull/17726), ... (29 total) |
| `vllm/model_executor/models/qwen2_audio.py` | [#11258](https://github.com/vllm-project/vllm/pull/11258), [#35994](https://github.com/vllm-project/vllm/pull/35994) |
| `vllm/model_executor/models/qwen2_vl.py` | [#7905](https://github.com/vllm-project/vllm/pull/7905), [#8442](https://github.com/vllm-project/vllm/pull/8442), [#8696](https://github.com/vllm-project/vllm/pull/8696), [#8770](https://github.com/vllm-project/vllm/pull/8770), [#8837](https://github.com/vllm-project/vllm/pull/8837), [#9250](https://github.com/vllm-project/vllm/pull/9250), [#10112](https://github.com/vllm-project/vllm/pull/10112), [#10169](https://github.com/vllm-project/vllm/pull/10169), [#10221](https://github.com/vllm-project/vllm/pull/10221), [#11258](https://github.com/vllm-project/vllm/pull/11258), [#11430](https://github.com/vllm-project/vllm/pull/11430), [#11663](https://github.com/vllm-project/vllm/pull/11663), ... (32 total) |
| `vllm/model_executor/models/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312), [#33410](https://github.com/vllm-project/vllm/pull/33410), [#33644](https://github.com/vllm-project/vllm/pull/33644), [#35415](https://github.com/vllm-project/vllm/pull/35415), [#37247](https://github.com/vllm-project/vllm/pull/37247) |
| `vllm/model_executor/models/qwen3_asr_forced_aligner.py` | no direct PR-number commit |
| `vllm/model_executor/models/qwen3_asr_realtime.py` | [#34613](https://github.com/vllm-project/vllm/pull/34613), [#35869](https://github.com/vllm-project/vllm/pull/35869) |
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | [#25550](https://github.com/vllm-project/vllm/pull/25550), [#26608](https://github.com/vllm-project/vllm/pull/26608), [#26815](https://github.com/vllm-project/vllm/pull/26815), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#29255](https://github.com/vllm-project/vllm/pull/29255), [#29828](https://github.com/vllm-project/vllm/pull/29828), [#29896](https://github.com/vllm-project/vllm/pull/29896), [#29974](https://github.com/vllm-project/vllm/pull/29974), [#31007](https://github.com/vllm-project/vllm/pull/31007), [#31790](https://github.com/vllm-project/vllm/pull/31790), ... (26 total) |
| `vllm/model_executor/models/qwen3_vl.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25337](https://github.com/vllm-project/vllm/pull/25337), [#25347](https://github.com/vllm-project/vllm/pull/25347), [#25557](https://github.com/vllm-project/vllm/pull/25557), [#25646](https://github.com/vllm-project/vllm/pull/25646), [#25648](https://github.com/vllm-project/vllm/pull/25648), [#25788](https://github.com/vllm-project/vllm/pull/25788), [#26000](https://github.com/vllm-project/vllm/pull/26000), [#27104](https://github.com/vllm-project/vllm/pull/27104), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#28663](https://github.com/vllm-project/vllm/pull/28663), ... (24 total) |
| `vllm/model_executor/models/qwen3_vl_moe.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25300](https://github.com/vllm-project/vllm/pull/25300), [#26000](https://github.com/vllm-project/vllm/pull/26000), [#42394](https://github.com/vllm-project/vllm/pull/42394), [#42716](https://github.com/vllm-project/vllm/pull/42716) |
| `vllm/transformers_utils/configs/mimo_v2_omni.py` | no direct PR-number commit |
| `vllm/transformers_utils/configs/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/mimo_v2_omni.py` | no direct PR-number commit |
| `vllm/transformers_utils/processors/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |

## PR Coverage Summary

- Git-traced PRs: 63
- Extra PRs preserved from existing docs: 23
- Total PRs in this document: 77
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-09-11 | [#7905](https://github.com/vllm-project/vllm/pull/7905) | merged | [Model][VLM] Add Qwen2-VL model support | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-13 | [#8442](https://github.com/vllm-project/vllm/pull/8442) | merged | [Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8 | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-25 | [#8770](https://github.com/vllm-project/vllm/pull/8770) | merged | [Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-09-26 | [#8837](https://github.com/vllm-project/vllm/pull/8837) | merged | [Misc] Update config loading for Qwen2-VL and remove Granite | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-10-16 | [#9250](https://github.com/vllm-project/vllm/pull/9250) | merged | [Misc] Standardize RoPE handling for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2024-12-24 | [#11430](https://github.com/vllm-project/vllm/pull/11430) | merged | [Bugfix] Fix Qwen2-VL LoRA weight loading | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-02-08 | [#12944](https://github.com/vllm-project/vllm/pull/12944) | merged | [Misc] Add qwen2.5-vl BNB support | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-02-20 | [#13533](https://github.com/vllm-project/vllm/pull/13533) | merged | [Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-02-27 | [#13968](https://github.com/vllm-project/vllm/pull/13968) | merged | [Bugfix] Fix qwen2.5-vl overflow issue | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-03-11 | [#14377](https://github.com/vllm-project/vllm/pull/14377) | merged | [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync | `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-04-21 | [#16907](https://github.com/vllm-project/vllm/pull/16907) | merged | [Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-04-22 | [#16974](https://github.com/vllm-project/vllm/pull/16974) | merged | [Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-05-16 | [#17973](https://github.com/vllm-project/vllm/pull/17973) | merged | [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const… | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-06-03 | [#19054](https://github.com/vllm-project/vllm/pull/19054) | merged | [Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-08-02 | [#22069](https://github.com/vllm-project/vllm/pull/22069) | merged | [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform. | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-08-07 | [#22184](https://github.com/vllm-project/vllm/pull/22184) | merged | [Model] Switch to Fused RMS norm in Qwen2.5_VL model. | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-08-18 | [#23058](https://github.com/vllm-project/vllm/pull/23058) | merged | [Bugfix] fix Qwen2.5-Omni processor output mapping | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-09-04 | [#24231](https://github.com/vllm-project/vllm/pull/24231) | merged | [LoRA]: Add lora support to qwen-2.5-omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-09-12 | [#24741](https://github.com/vllm-project/vllm/pull/24741) | merged | [Models] Prevent CUDA sync in Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-09-17 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-20 | [#25300](https://github.com/vllm-project/vllm/pull/25300) | merged | [Bugfix] Fix Qwen3-VL-MoE weight loading for EP | `vllm/model_executor/models/qwen3_vl_moe.py` |
| 2025-09-21 | [#25337](https://github.com/vllm-project/vllm/pull/25337) | merged | [MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate` | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-21 | [#25347](https://github.com/vllm-project/vllm/pull/25347) | merged | [Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate` | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-25 | [#25646](https://github.com/vllm-project/vllm/pull/25646) | merged | [Misc] Fix Qwen3-VL `video_grid_thw` typing | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-09-25 | [#25648](https://github.com/vllm-project/vllm/pull/25648) | merged | [Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-09-27 | [#25788](https://github.com/vllm-project/vllm/pull/25788) | merged | [Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2025-10-02 | [#24642](https://github.com/vllm-project/vllm/pull/24642) | merged | [Qwen][ROCm] Flash Attention Rotary Embeddings | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-03 | [#26104](https://github.com/vllm-project/vllm/pull/26104) | merged | [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-03 | [#26123](https://github.com/vllm-project/vllm/pull/26123) | merged | [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642 | `vllm/model_executor/models/qwen2_vl.py` |
| 2025-10-15 | [#26815](https://github.com/vllm-project/vllm/pull/26815) | merged | [Bugfix] Fix qwen3-omni audio truncation issue | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-10-26 | [#27190](https://github.com/vllm-project/vllm/pull/27190) | merged | [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py` |
| 2025-10-29 | [#27705](https://github.com/vllm-project/vllm/pull/27705) | merged | [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-10-30 | [#27790](https://github.com/vllm-project/vllm/pull/27790) | merged | [BugFix][VL] Fix FA selection on Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-11-12 | [#28271](https://github.com/vllm-project/vllm/pull/28271) | merged | [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py` |
| 2025-11-22 | [#29232](https://github.com/vllm-project/vllm/pull/29232) | merged | Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2025-11-24 | [#27721](https://github.com/vllm-project/vllm/pull/27721) | merged | [Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine. | `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-12-02 | [#29896](https://github.com/vllm-project/vllm/pull/29896) | merged | feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2025-12-14 | [#29752](https://github.com/vllm-project/vllm/pull/29752) | merged | [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL | `vllm/model_executor/models/qwen3_vl.py` |
| 2025-12-18 | [#30883](https://github.com/vllm-project/vllm/pull/30883) | merged | [Chore] Remove v0 dead code for Qwen2.5-omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2025-12-31 | [#31436](https://github.com/vllm-project/vllm/pull/31436) | merged | Add GLM-ASR multimodal support | `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py` |
| 2026-01-03 | [#29255](https://github.com/vllm-project/vllm/pull/29255) | merged | Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring | `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-06 | [#31790](https://github.com/vllm-project/vllm/pull/31790) | merged | [Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni) | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-07 | [#31779](https://github.com/vllm-project/vllm/pull/31779) | merged | [Refactor] GLM-ASR Modeling | `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py` |
| 2026-01-08 | [#31890](https://github.com/vllm-project/vllm/pull/31890) | merged | [Models] Allow converting Qwen3-VL into Reranker model | `examples/pooling/score/template/qwen3_vl_reranker.jinja` |
| 2026-01-25 | [#32772](https://github.com/vllm-project/vllm/pull/32772) | merged | [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-01-26 | [#33010](https://github.com/vllm-project/vllm/pull/33010) | merged | [Model] Use mm_position to compute mrope positions for Qwen3-Omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-01-29 | [#33312](https://github.com/vllm-project/vllm/pull/33312) | merged | [Models] Qwen3-ASR | `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py` |
| 2026-01-31 | [#33410](https://github.com/vllm-project/vllm/pull/33410) | merged | [Bugfix] Fix `Qwen3ASR` language asr tag in output | `vllm/model_executor/models/qwen3_asr.py` |
| 2026-02-04 | [#29828](https://github.com/vllm-project/vllm/pull/29828) | merged | [Model] Add transcription support for Qwen3-Omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-02-26 | [#35368](https://github.com/vllm-project/vllm/pull/35368) | merged | [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression | `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-04 | [#35869](https://github.com/vllm-project/vllm/pull/35869) | merged | [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile | `vllm/model_executor/models/qwen3_asr_realtime.py` |
| 2026-03-05 | [#36108](https://github.com/vllm-project/vllm/pull/36108) | merged | refactor funasr model. | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-05 | [#35994](https://github.com/vllm-project/vllm/pull/35994) | merged | [BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0 | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py` |
| 2026-03-18 | [#37439](https://github.com/vllm-project/vllm/pull/37439) | merged | [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-23 | [#35963](https://github.com/vllm-project/vllm/pull/35963) | merged | [Feature] ViT Full CUDA Graph | `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` |
| 2026-04-14 | [#38061](https://github.com/vllm-project/vllm/pull/38061) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-04-18 | [#40160](https://github.com/vllm-project/vllm/pull/40160) | merged | [Bugfix] Fix k_proj's bias for GLM-ASR | `vllm/model_executor/models/glmasr.py` |
| 2026-04-27 | [#38065](https://github.com/vllm-project/vllm/pull/38065) | merged | [Perf] FP8 FlashInfer Attn for ViT | `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/vision.py`, `vllm/config/multimodal.py` |
| 2026-04-27 | [#36464](https://github.com/vllm-project/vllm/pull/36464) | merged | [Examples] Resettle generate examples. | `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md` |
| 2026-04-27 | [#40967](https://github.com/vllm-project/vllm/pull/40967) | merged | [Model] Add MiMo-V2.5 support | `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py` |
| 2026-05-02 | [#40830](https://github.com/vllm-project/vllm/pull/40830) | merged | [MM][CG] Support ViT CG for Qwen2.5-VL | `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py` |
| 2026-05-13 | [#41736](https://github.com/vllm-project/vllm/pull/41736) | merged | [MM][CG] Support ViT CG for Qwen2-VL | `vllm/model_executor/models/qwen2_vl.py` |
| 2026-05-14 | [#42412](https://github.com/vllm-project/vllm/pull/42412) | merged | [Feature] Add instruction support for score/rerank chat templates | `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py`, `vllm/entrypoints/pooling/scoring/protocol.py`, `vllm/entrypoints/pooling/scoring/io_processor.py` |
| 2026-05-14 | [#38040](https://github.com/vllm-project/vllm/pull/38040) | merged | [Fix] Misc Fixes in ViT CUDA Graph | `vllm/model_executor/models/qwen3_vl.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`, `vllm/v1/worker/encoder_cudagraph.py` |
| 2026-05-17 | [#42716](https://github.com/vllm-project/vllm/pull/42716) | merged | Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer | `vllm/model_executor/models/qwen3_vl_moe.py` |
| 2026-05-19 | [#42347](https://github.com/vllm-project/vllm/pull/42347) | merged | [Perf][4/n] Eliminate various GPU CPU syncs | `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py` |
| 2026-05-27 | [#43647](https://github.com/vllm-project/vllm/pull/43647) | merged | [ROCm][CI] Fix ROCm multimodal Qwen2.5-VL activation compile and Phi4MM ragged image mask handling | `vllm/model_executor/models/qwen2_5_vl.py` |
| 2026-05-27 | [#43617](https://github.com/vllm-project/vllm/pull/43617) | merged | Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2026-06-04 | [#44205](https://github.com/vllm-project/vllm/pull/44205) | merged | [Bugfix] fix EVS for qwen3-vl | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-06-09 | [#44264](https://github.com/vllm-project/vllm/pull/44264) | merged | [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-06-10 | [#45131](https://github.com/vllm-project/vllm/pull/45131) | merged | Deprecated 1st generation Qwen and QwenVL models | `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py` |
| 2026-06-10 | [#35415](https://github.com/vllm-project/vllm/pull/35415) | merged | feat(qwen3-asr): support prompt parameter in v1/audio/transcriptions | `vllm/model_executor/models/qwen3_asr.py`, `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py` |
| 2026-06-11 | [#45161](https://github.com/vllm-project/vllm/pull/45161) | merged | Deprecate Transformers v4 support | `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py` |
| 2026-06-13 | [#42700](https://github.com/vllm-project/vllm/pull/42700) | merged | [Bugfix] Replace deprecated Qwen2VLImageProcessorFast with Qwen2VLImageProcessor | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |
| 2026-06-21 | [#45424](https://github.com/vllm-project/vllm/pull/45424) | merged | [Core] Ensure memory is pinned prior to async h2d copy | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py` |
| 2026-06-21 | [#46305](https://github.com/vllm-project/vllm/pull/46305) | merged | [Bugfix][Qwen3-VL] Fix multi-video crash with list-valued fps/num_frames | `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py` |

## Per-PR Diff Audit Cards

### PR #7905 - [Model][VLM] Add Qwen2-VL model support

- Link: https://github.com/vllm-project/vllm/pull/7905
- Status/date: merged / 2024-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `3b7fea770f44`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1531/-31, 1844 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][VLM] Add Qwen2-VL model support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Model][VLM] Add Qwen2-VL model support"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #8770 - [Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend

- Link: https://github.com/vllm-project/vllm/pull/8770
- Status/date: merged / 2024-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `c23953675f78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +99/-9, 202 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Misc] Update config loading for Qwen2-VL and remove Granite"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Misc] Update config loading for Qwen2-VL and remove Granite"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Misc] Standardize RoPE handling for Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Misc] Standardize RoPE handling for Qwen2-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #11430 - [Bugfix] Fix Qwen2-VL LoRA weight loading

- Link: https://github.com/vllm-project/vllm/pull/11430
- Status/date: merged / 2024-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `b1b1038fbdc1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +168/-14, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2-VL LoRA weight loading"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Bugfix] Fix Qwen2-VL LoRA weight loading"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #12944 - [Misc] Add qwen2.5-vl BNB support

- Link: https://github.com/vllm-project/vllm/pull/12944
- Status/date: merged / 2025-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `4c8dd12ef347`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-30, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Add qwen2.5-vl BNB support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Misc] Add qwen2.5-vl BNB support"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #13533 - [Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/13533
- Status/date: merged / 2025-02-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `041e29471671`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +18/-2, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Fix qwen2.5-vl overflow issue"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Bugfix] Fix qwen2.5-vl overflow issue"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #16907 - [Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni

- Link: https://github.com/vllm-project/vllm/pull/16907
- Status/date: merged / 2025-04-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `26c0406555a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #17973 - [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…

- Link: https://github.com/vllm-project/vllm/pull/17973
- Status/date: merged / 2025-05-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `67da5720d4ed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +121/-83, 285 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform."; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform."; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Switch to Fused RMS norm in Qwen2.5_VL model."; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Model] Switch to Fused RMS norm in Qwen2.5_VL model."; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] fix Qwen2.5-Omni processor output mapping"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Bugfix] fix Qwen2.5-Omni processor output mapping"; the main implementation surface is `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #24231 - [LoRA]: Add lora support to qwen-2.5-omni

- Link: https://github.com/vllm-project/vllm/pull/24231
- Status/date: merged / 2025-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `c9f7081f9c84`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-3, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA]: Add lora support to qwen-2.5-omni"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[LoRA]: Add lora support to qwen-2.5-omni"; the main implementation surface is `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #24741 - [Models] Prevent CUDA sync in Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/24741
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `b0d1213ac395`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Prevent CUDA sync in Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Models] Prevent CUDA sync in Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Support Qwen3-VL Model Series"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Model] Support Qwen3-VL Model Series"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #25300 - [Bugfix] Fix Qwen3-VL-MoE weight loading for EP

- Link: https://github.com/vllm-project/vllm/pull/25300
- Status/date: merged / 2025-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `be874c020196`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-5, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL-MoE weight loading for EP"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl_moe.py`; technical summary: Covers "[Bugfix] Fix Qwen3-VL-MoE weight loading for EP"; the main implementation surface is `vllm/model_executor/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #25646 - [Misc] Fix Qwen3-VL `video_grid_thw` typing

- Link: https://github.com/vllm-project/vllm/pull/25646
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `7be9ffcd9f5c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Fix Qwen3-VL `video_grid_thw` typing"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Misc] Fix Qwen3-VL `video_grid_thw` typing"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #24642 - [Qwen][ROCm] Flash Attention Rotary Embeddings

- Link: https://github.com/vllm-project/vllm/pull/24642
- Status/date: merged / 2025-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `5e4a8223c644`, `dd96465fd744`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +28/-5, 80 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen][ROCm] Flash Attention Rotary Embeddings"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Qwen][ROCm] Flash Attention Rotary Embeddings"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #26104 - [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm

- Link: https://github.com/vllm-project/vllm/pull/26104
- Status/date: merged / 2025-10-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +154/-141, 553 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl.py` modified +23/-26 (49 lines); hunks: -35,7 +35,8; -176,14 +177,18 @@ def __init__(; symbols: __init__, forward, compute_attn_mask_seqlen, touching `__init__, forward, compute_attn_mask_seqlen`; `vllm/model_executor/models/dots_ocr.py` modified +19/-22 (41 lines); hunks: -10,7 +10,8; -267,10 +268,12 @@ def __init__(self,; symbols: __init__, forward, compute_attn_mask_seqlen, touching `__init__, forward, compute_attn_mask_seqlen`; `vllm/model_executor/models/qwen2_vl.py` modified +18/-22 (40 lines); hunks: -42,7 +42,8; -319,18 +320,20 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17 (34 lines); hunks: -39,7 +39,8; -302,6 +303,11 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +23/-26 (49 lines); hunks: -35,7 +35,8; -176,14 +177,18 @@ def __init__(; symbols: __init__, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/dots_ocr.py` modified +19/-22 (41 lines); hunks: -10,7 +10,8; -267,10 +268,12 @@ def __init__(self,; symbols: __init__, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_vl.py` modified +18/-22 (40 lines); hunks: -42,7 +42,8; -319,18 +320,20 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17 (34 lines); hunks: -39,7 +39,8; -302,6 +303,11 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/glm4_1v.py` modified +17/-14 (31 lines); hunks: -47,7 +47,8; -263,19 +264,26 @@ def __init__(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/ernie45_vl.py
@@ -35,7 +35,8 @@
-from vllm.attention.layer import check_upstream_fa_availability
+from vllm.attention.layer import (check_upstream_fa_availability,
+                                  maybe_get_vit_flash_attn_backend)
@@ -176,14 +177,18 @@ def __init__(
-        if self.attn_backend != _Backend.FLASH_ATTN and \
-            check_upstream_fa_availability(torch.get_default_dtype()):
diff -- vllm/model_executor/models/dots_ocr.py
@@ -10,7 +10,8 @@
-from vllm.attention.layer import check_upstream_fa_availability
+from vllm.attention.layer import (check_upstream_fa_availability,
+                                  maybe_get_vit_flash_attn_backend)
@@ -267,10 +268,12 @@ def __init__(self,
-        if self.attn_backend != _Backend.FLASH_ATTN and \
-                check_upstream_fa_availability(torch.get_default_dtype()):
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -42,7 +42,8 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +23/-26; `vllm/model_executor/models/dots_ocr.py` modified +19/-22; `vllm/model_executor/models/qwen2_vl.py` modified +18/-22; `vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17; `vllm/model_executor/models/glm4_1v.py` modified +17/-14; `vllm/model_executor/models/siglip2navit.py` modified +8/-14
- Risk and verification: Runtime changes concentrate in `vllm/attention/layer.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/ernie45_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26123 - [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642

- Link: https://github.com/vllm-project/vllm/pull/26123
- Status/date: merged / 2025-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `dd96465fd744`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-4, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #26815 - [Bugfix] Fix qwen3-omni audio truncation issue

- Link: https://github.com/vllm-project/vllm/pull/26815
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `8c851f6d044b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-2, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix qwen3-omni audio truncation issue"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Bugfix] Fix qwen3-omni audio truncation issue"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #27190 - [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA

- Link: https://github.com/vllm-project/vllm/pull/27190
- Status/date: merged / 2025-10-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +46/-12, 106 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py`; technical summary: Covers "[BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[BugFix][VL] Fix FA selection on Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[BugFix][VL] Fix FA selection on Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #28271 - [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL

- Link: https://github.com/vllm-project/vllm/pull/28271
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; associated commits `48b8456ff992`, `bc5bd45c7d1a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +1/-42, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #29232 - Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/29232
- Status/date: merged / 2025-11-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `d84d8f4429a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-1, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `839c6b7b72bc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +467/-59, 631 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine."; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine."; the main implementation surface is `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer, touching `print_input_ids, mock_qwen3_omni_config, mock_processor`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video, touching `_maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders, touching `_maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders`.
- Code diff details:
  - `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders
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
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_qwen3_omni.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29896 - feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE

- Link: https://github.com/vllm-project/vllm/pull/29896
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `a2b053dc858d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-0, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #29752 - [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL

- Link: https://github.com/vllm-project/vllm/pull/29752
- Status/date: merged / 2025-12-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `ae88aada38ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +424/-12, 539 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Chore] Remove v0 dead code for Qwen2.5-omni"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Chore] Remove v0 dead code for Qwen2.5-omni"; the main implementation surface is `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #31436 - Add GLM-ASR multimodal support

- Link: https://github.com/vllm-project/vllm/pull/31436
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; associated commits `d722e9e614f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +764/-2, 833 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add GLM-ASR multimodal support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; technical summary: Covers "Add GLM-ASR multimodal support"; the main implementation surface is `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring"; the main implementation surface is `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Refactor] GLM-ASR Modeling"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`; technical summary: Covers "[Refactor] GLM-ASR Modeling"; the main implementation surface is `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Models] Allow converting Qwen3-VL into Reranker model"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `examples/pooling/score/template/qwen3_vl_reranker.jinja`; technical summary: Covers "[Models] Allow converting Qwen3-VL into Reranker model"; the main implementation surface is `examples/pooling/score/template/qwen3_vl_reranker.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #32772 - [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni

- Link: https://github.com/vllm-project/vllm/pull/32772
- Status/date: merged / 2026-01-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_omni_thinker.py`; associated commits `a698e8e7ad4b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +386/-201, 689 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`; technical summary: Covers "[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni"; the main implementation surface is `vllm/model_executor/models/qwen2_5_omni_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates, touching `_get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates
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
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33010 - [Model] Use mm_position to compute mrope positions for Qwen3-Omni

- Link: https://github.com/vllm-project/vllm/pull/33010
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `6ca2c91b9663`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +293/-298, 675 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_position to compute mrope positions for Qwen3-Omni"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Model] Use mm_position to compute mrope positions for Qwen3-Omni"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping, touching `load_weights, get_mrope_input_positions, _compute_audio_token_count`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping
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
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33312 - [Models] Qwen3-ASR

- Link: https://github.com/vllm-project/vllm/pull/33312
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; associated commits `8b3f0a99dd50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1269/-0, 1335 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models] Qwen3-ASR"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`; technical summary: Covers "[Models] Qwen3-ASR"; the main implementation surface is `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Fix `Qwen3ASR` language asr tag in output"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_asr.py`; technical summary: Covers "[Bugfix] Fix `Qwen3ASR` language asr tag in output"; the main implementation surface is `vllm/model_executor/models/qwen3_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #29828 - [Model] Add transcription support for Qwen3-Omni

- Link: https://github.com/vllm-project/vllm/pull/29828
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `535de06cb1d9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +104/-2, 177 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add transcription support for Qwen3-Omni"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Model] Add transcription support for Qwen3-Omni"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #35368 - [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression

- Link: https://github.com/vllm-project/vllm/pull/35368
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `c0615a296d44`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +379/-21, 437 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression"; the main implementation surface is `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #35869 - [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile

- Link: https://github.com/vllm-project/vllm/pull/35869
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_asr_realtime.py`; associated commits `36bf2131816e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-2, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_asr_realtime.py`; technical summary: Covers "[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile"; the main implementation surface is `vllm/model_executor/models/qwen3_asr_realtime.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #36108 - refactor funasr model.

- Link: https://github.com/vllm-project/vllm/pull/36108
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `3ee68590c7fa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-57, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor funasr model."; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "refactor funasr model."; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`; technical summary: Covers "[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0"; the main implementation surface is `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #37439 - [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation

- Link: https://github.com/vllm-project/vllm/pull/37439
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `738d0a281fab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Feature] ViT Full CUDA Graph"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py`; technical summary: Covers "[Feature] ViT Full CUDA Graph"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #38061 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference

- Link: https://github.com/vllm-project/vllm/pull/38061
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `80118853f42a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +583/-68, 1042 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Bugfix] Fix k_proj's bias for GLM-ASR"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/glmasr.py`; technical summary: Covers "[Bugfix] Fix k_proj's bias for GLM-ASR"; the main implementation surface is `vllm/model_executor/models/glmasr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #38065 - [Perf] FP8 FlashInfer Attn for ViT

- Link: https://github.com/vllm-project/vllm/pull/38065
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1830/-50, 2151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] FP8 FlashInfer Attn for ViT"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/vision.py`, `vllm/config/multimodal.py`; technical summary: Covers "[Perf] FP8 FlashInfer Attn for ViT"; the main implementation surface is `vllm/model_executor/layers/attention/mm_encoder_attention.py`, `vllm/model_executor/models/vision.py`, `vllm/config/multimodal.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +336/-15 (351 lines); hunks: -1,13 +1,32; -20,6 +39,108; symbols: _load_fp8_scales_file, _maybe_save_fp8_scales, maybe_recompute_cu_seqlens, __init__, touching `_load_fp8_scales_file, _maybe_save_fp8_scales, maybe_recompute_cu_seqlens`; `vllm/model_executor/models/vision.py` modified +32/-28 (60 lines); hunks: -10,14 +10,15; -102,45 +103,48 @@ def get_vit_attn_backend(; symbols: get_vit_attn_backend, get_multimodal_config, get_fp8_padded_hidden_size, is_vit_use_data_parallel, touching `get_vit_attn_backend, get_multimodal_config, get_fp8_padded_hidden_size`; `vllm/config/multimodal.py` modified +51/-0 (51 lines); hunks: -2,6 +2,7; -158,6 +159,24 @@ class MultiModalConfig:; symbols: MultiModalConfig, _validate_multimodal_config, compute_hash, touching `MultiModalConfig, _validate_multimodal_config, compute_hash`; `tests/config/test_multimodal_config.py` modified +18/-0 (18 lines); hunks: -41,3 +41,21 @@ def test_language_model_only_affects_model_hash():; symbols: test_language_model_only_affects_model_hash, test_mm_encoder_fp8_scale_path_requires_fp8, test_mm_encoder_attn_dtype_hash_updates, touching `test_language_model_only_affects_model_hash, test_mm_encoder_fp8_scale_path_requires_fp8, test_mm_encoder_attn_dtype_hash_updates`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +336/-15 (351 lines); hunks: -1,13 +1,32; -20,6 +39,108; symbols: _load_fp8_scales_file, _maybe_save_fp8_scales, maybe_recompute_cu_seqlens, __init__
  - `vllm/model_executor/models/vision.py` modified +32/-28 (60 lines); hunks: -10,14 +10,15; -102,45 +103,48 @@ def get_vit_attn_backend(; symbols: get_vit_attn_backend, get_multimodal_config, get_fp8_padded_hidden_size, is_vit_use_data_parallel
  - `vllm/config/multimodal.py` modified +51/-0 (51 lines); hunks: -2,6 +2,7; -158,6 +159,24 @@ class MultiModalConfig:; symbols: MultiModalConfig, _validate_multimodal_config, compute_hash
  - `tests/config/test_multimodal_config.py` modified +18/-0 (18 lines); hunks: -41,3 +41,21 @@ def test_language_model_only_affects_model_hash():; symbols: test_language_model_only_affects_model_hash, test_mm_encoder_fp8_scale_path_requires_fp8, test_mm_encoder_attn_dtype_hash_updates
  - `vllm/model_executor/model_loader/utils.py` modified +9/-5 (14 lines); hunks: -15,7 +15,11; -106,12 +110,12 @@ def process_weights_after_loading(; symbols: process_weights_after_loading
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mm_encoder_attention.py
@@ -1,13 +1,32 @@
+import functools
+import json
+from vllm.config import MultiModalConfig
+from vllm.kernels.triton.qkv_padded_fp8_quant import (
+    quantize_fp8_maybe_pad_head_dim,
+)
diff -- vllm/model_executor/models/vision.py
@@ -10,14 +10,15 @@
-from vllm.config import MultiModalConfig, VllmConfig, get_current_vllm_config
+from vllm.config import MultiModalConfig, get_current_vllm_config_or_none
+from vllm.utils.math_utils import round_up
@@ -102,45 +103,48 @@ def get_vit_attn_backend(
-    try:
-        vllm_config: VllmConfig = get_current_vllm_config()
diff -- vllm/config/multimodal.py
@@ -2,6 +2,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +336/-15; `vllm/model_executor/models/vision.py` modified +32/-28; `vllm/config/multimodal.py` modified +51/-0; `vllm/model_executor/model_loader/utils.py` modified +9/-5; `vllm/model_executor/models/qwen3_vl.py` modified +9/-0
  - tests: `tests/config/test_multimodal_config.py` modified +18/-0; `tests/kernels/core/test_vit_fp8_attn.py` added +279/-0
  - other: `benchmarks/kernels/benchmark_vit_fp8_attn.py` added +324/-0
- Risk and verification: The diff ships test coverage in `tests/config/test_multimodal_config.py`, `tests/kernels/core/test_vit_fp8_attn.py`, `tests/kernels/core/test_vit_fp8_quant.py`, `tests/kernels/core/test_vit_fp8_scaling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36464 - [Examples] Resettle generate examples.

- Link: https://github.com/vllm-project/vllm/pull/36464
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 36 files, +46/-50, 267 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Examples] Resettle generate examples."; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md`; technical summary: Covers "[Examples] Resettle generate examples."; the main implementation surface is `docs/features/multimodal_inputs.md`, `examples/generate/multimodal/qwen2_5_omni/README.md`, `docs/features/reasoning_outputs.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/features/multimodal_inputs.md` modified +7/-7 (14 lines); hunks: -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the mult...; -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, y...; `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6 (12 lines); hunks: -6,15 +6,15 @@ This folder provides several example scripts on how to inferen...; -24,16 +24,16 @@ You can also test Qwen2.5-Omni on a single modality:; `docs/features/reasoning_outputs.md` modified +1/-1 (2 lines); hunks: -202,7 +202,7 @@ The reasoning content is also available when both tool calli...; `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1 (2 lines); hunks: -1402,7 +1402,7 @@ def run_mantis(questions: list[str], modality: str) -> Mod...; symbols: run_mantis, run_minicpmv_base, touching `run_mantis, run_minicpmv_base`.
- Code diff details:
  - `docs/features/multimodal_inputs.md` modified +7/-7 (14 lines); hunks: -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the mult...; -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, y...
  - `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6 (12 lines); hunks: -6,15 +6,15 @@ This folder provides several example scripts on how to inferen...; -24,16 +24,16 @@ You can also test Qwen2.5-Omni on a single modality:
  - `docs/features/reasoning_outputs.md` modified +1/-1 (2 lines); hunks: -202,7 +202,7 @@ The reasoning content is also available when both tool calli...
  - `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1 (2 lines); hunks: -1402,7 +1402,7 @@ def run_mantis(questions: list[str], modality: str) -> Mod...; symbols: run_mantis, run_minicpmv_base
  - `examples/generate/multimodal/audio_language_offline.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- docs/features/multimodal_inputs.md
@@ -68,7 +68,7 @@ You can pass a single image to the `'image'` field of the multi-modal dictionary
-Full example: [examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)
+Full example: [examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)
@@ -101,7 +101,7 @@ To substitute multiple images inside the same text prompt, you can pass in a lis
-Full example: [examples/offline_inference/vision_language_multi_image.py](../../examples/offline_inference/vision_language_multi_image.py)
+Full example: [examples/generate/multimodal/vision_language_multi_image_offline.py](../../examples/generate/multimodal/vision_language_multi_image_offline.py)
@@ -287,13 +287,13 @@ Instead of NumPy arrays, you can also pass `'torch.Tensor'` instances, as shown
diff -- examples/generate/multimodal/qwen2_5_omni/README.md
@@ -6,15 +6,15 @@ This folder provides several example scripts on how to inference Qwen2.5-Omni of
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
-python examples/offline_inference/qwen2_5_omni/only_thinker.py \
+python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
diff -- docs/features/reasoning_outputs.md
@@ -202,7 +202,7 @@ The reasoning content is also available when both tool calling and the reasoning
```

- Reviewed files:
  - docs: `docs/features/multimodal_inputs.md` modified +7/-7; `examples/generate/multimodal/qwen2_5_omni/README.md` renamed +6/-6; `docs/features/reasoning_outputs.md` modified +1/-1; `examples/generate/multimodal/vision_language_offline.py` renamed +1/-1; `examples/generate/multimodal/audio_language_offline.py` renamed +0/-0; `examples/generate/multimodal/encoder_decoder_multimodal_offline.py` renamed +0/-0
- Risk and verification: This is mostly docs/examples in `docs/features/multimodal_inputs.md`, `docs/features/reasoning_outputs.md`, `docs/serving/openai_compatible_server.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #40967 - [Model] Add MiMo-V2.5 support

- Link: https://github.com/vllm-project/vllm/pull/40967
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +4737/-5, 4920 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MiMo-V2.5 support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`; technical summary: Covers "[Model] Add MiMo-V2.5 support"; the main implementation surface is `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__, touching `MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger`; `vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init, touching `_vq_default, _ema_inplace, _laplace_smoothing`; `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput, touching `ImageInput, VideoInput, AudioInput`; `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers, touching `MiMoV2MTPLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__
  - `vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init
  - `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput
  - `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers
  - `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: Mimo_VLVisionConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -0,0 +1,1488 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import math
+from collections.abc import Callable, Iterable, Mapping, Sequence
+from functools import partial
+from typing import Any
diff -- vllm/model_executor/models/mimo_audio.py
@@ -0,0 +1,1389 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MiMo audio: tokenizer, encoding utilities, and audio encoder.
+Ported from SGLang's mimo_audio.py.
+Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
+"""
diff -- vllm/transformers_utils/processors/mimo_v2_omni.py
@@ -0,0 +1,1285 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0; `vllm/model_executor/models/mimo_audio.py` added +1389/-0; `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0; `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0; `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0; `vllm/model_executor/models/mimo_v2.py` renamed +22/-2
  - tests: `tests/models/registry.py` modified +18/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40830 - [MM][CG] Support ViT CG for Qwen2.5-VL

- Link: https://github.com/vllm-project/vllm/pull/40830
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `964a4bc2a57a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +539/-22, 669 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`; technical summary: Covers "[MM][CG] Support ViT CG for Qwen2.5-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21 (450 lines); hunks: -85,11 +85,13; -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:; symbols: invert_permutation, forward, prepare_encoder_metadata, touching `invert_permutation, forward, prepare_encoder_metadata`; `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0 (95 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_evs_batched_videos, touching `qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21 (450 lines); hunks: -85,11 +85,13; -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:; symbols: invert_permutation, forward, prepare_encoder_metadata
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0 (95 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: qwen2_5_vl_chat_template, _window_attention_regression_image, _encoder_cudagraph_config, test_qwen2_5_vl_evs_batched_videos
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -85,11 +85,13 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
+    SupportsEncoderCudaGraph,
@@ -771,22 +773,54 @@ def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
-    def forward(
+    def prepare_encoder_metadata(
-        x: torch.Tensor,
diff -- tests/models/multimodal/generation/test_qwen2_5_vl.py
@@ -3,6 +3,7 @@
+from vllm.assets.image import ImageAsset
@@ -11,6 +12,7 @@
+IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
@@ -28,6 +30,25 @@ def qwen2_5_vl_chat_template(*query):
+WINDOW_ATTN_IMAGE_PROMPT = qwen2_5_vl_chat_template(
+    IMAGE_PLACEHOLDER,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +429/-21
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` modified +95/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_qwen2_5_vl.py`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41736 - [MM][CG] Support ViT CG for Qwen2-VL

- Link: https://github.com/vllm-project/vllm/pull/41736
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_vl.py`; associated commits `b3c69595a63f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +315/-21, 415 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][CG] Support ViT CG for Qwen2-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/qwen2_vl.py`; technical summary: Covers "[MM][CG] Support ViT CG for Qwen2-VL"; the main implementation surface is `vllm/model_executor/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20 (320 lines); hunks: -89,9 +89,11; -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tenso...; symbols: compute_attn_mask_seqlen, prepare_encoder_metadata, forward, _get_mm_fields_config, touching `compute_attn_mask_seqlen, prepare_encoder_metadata, forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_vl.py` modified +300/-20 (320 lines); hunks: -89,9 +89,11; -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tenso...; symbols: compute_attn_mask_seqlen, prepare_encoder_metadata, forward, _get_mm_fields_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -89,9 +89,11 @@
+from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers
+    SupportsEncoderCudaGraph,
@@ -646,38 +648,84 @@ def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None:
+    def prepare_encoder_metadata(
+        self,
+        grid_thw: list[list[int]],
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +300/-20
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42412 - [Feature] Add instruction support for score/rerank chat templates

- Link: https://github.com/vllm-project/vllm/pull/42412
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +182/-12, 285 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add instruction support for score/rerank chat templates"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py`, `vllm/entrypoints/pooling/scoring/protocol.py`, `vllm/entrypoints/pooling/scoring/io_processor.py`; technical summary: Covers "[Feature] Add instruction support for score/rerank chat templates"; the main implementation surface is `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py`, `vllm/entrypoints/pooling/scoring/protocol.py`, `vllm/entrypoints/pooling/scoring/io_processor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py` modified +132/-0 (132 lines); hunks: -377,3 +377,135 @@ async def test_score_api_queries_list_documents_list(; symbols: test_score_api_queries_list_documents_list, test_score_api_instruction_field, test_rerank_api_instruction_field, test_rerank_api_instruction_field_matches_chat_template_kwargs, touching `test_score_api_queries_list_documents_list, test_score_api_instruction_field, test_rerank_api_instruction_field`; `vllm/entrypoints/pooling/scoring/protocol.py` modified +31/-2 (33 lines); hunks: -1,9 +1,9; -35,8 +35,37 @@ class ScoringRequestMixin(PoolingBasicRequestMixin, ClassifyR...; symbols: ScoringRequestMixin, _merge_instruction_into_kwargs, build_tok_params, touching `ScoringRequestMixin, _merge_instruction_into_kwargs, build_tok_params`; `vllm/entrypoints/pooling/scoring/io_processor.py` modified +17/-2 (19 lines); hunks: -157,7 +157,7 @@ def pre_process_online(self, ctx: ScoringServeContext):; -384,7 +384,7 @@ def pre_process_online(self, ctx: ScoringServeContext):; symbols: pre_process_online, pre_process_offline, _pre_process, get_score_prompt, touching `pre_process_online, pre_process_offline, _pre_process`; `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7 (8 lines); hunks: -1,13 +1,7.
- Code diff details:
  - `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py` modified +132/-0 (132 lines); hunks: -377,3 +377,135 @@ async def test_score_api_queries_list_documents_list(; symbols: test_score_api_queries_list_documents_list, test_score_api_instruction_field, test_rerank_api_instruction_field, test_rerank_api_instruction_field_matches_chat_template_kwargs
  - `vllm/entrypoints/pooling/scoring/protocol.py` modified +31/-2 (33 lines); hunks: -1,9 +1,9; -35,8 +35,37 @@ class ScoringRequestMixin(PoolingBasicRequestMixin, ClassifyR...; symbols: ScoringRequestMixin, _merge_instruction_into_kwargs, build_tok_params
  - `vllm/entrypoints/pooling/scoring/io_processor.py` modified +17/-2 (19 lines); hunks: -157,7 +157,7 @@ def pre_process_online(self, ctx: ScoringServeContext):; -384,7 +384,7 @@ def pre_process_online(self, ctx: ScoringServeContext):; symbols: pre_process_online, pre_process_offline, _pre_process, get_score_prompt
  - `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7 (8 lines); hunks: -1,13 +1,7
  - `examples/pooling/score/template/qwen3_reranker.jinja` modified +1/-1 (2 lines); hunks: -1,7 +1,7
- Key code excerpts:

```diff
diff -- tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py
@@ -377,3 +377,135 @@ async def test_score_api_queries_list_documents_list(
+INSTRUCTION = (
+    "Given a multimodal retrieval query, retrieve candidates that "
+    "visually or textually match the requested scene, object, or action."
+)
+@pytest.mark.asyncio
+async def test_score_api_instruction_field(
diff -- vllm/entrypoints/pooling/scoring/protocol.py
@@ -1,9 +1,9 @@
-from typing import TypeAlias
+from typing import Any, TypeAlias
-from pydantic import BaseModel, Field
+from pydantic import BaseModel, Field, model_validator
@@ -35,8 +35,37 @@ class ScoringRequestMixin(PoolingBasicRequestMixin, ClassifyRequestMixin):
+    instruction: str | None = Field(
diff -- vllm/entrypoints/pooling/scoring/io_processor.py
@@ -157,7 +157,7 @@ def pre_process_online(self, ctx: ScoringServeContext):
```

- Reviewed files:
  - tests: `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py` modified +132/-0
  - runtime: `vllm/entrypoints/pooling/scoring/protocol.py` modified +31/-2; `vllm/entrypoints/pooling/scoring/io_processor.py` modified +17/-2
  - docs: `examples/pooling/score/template/qwen3_vl_reranker.jinja` modified +1/-7; `examples/pooling/score/template/qwen3_reranker.jinja` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38040 - [Fix] Misc Fixes in ViT CUDA Graph

- Link: https://github.com/vllm-project/vllm/pull/38040
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +242/-21, 309 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Misc Fixes in ViT CUDA Graph"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`, `vllm/v1/worker/encoder_cudagraph.py`; technical summary: Covers "[Fix] Misc Fixes in ViT CUDA Graph"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`, `vllm/v1/worker/encoder_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +10/-9 (19 lines); hunks: -1768,14 +1768,12 @@ def get_encoder_cudagraph_config(self):; -1923,7 +1921,10 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: get_encoder_cudagraph_config, prepare_encoder_cudagraph_capture_inputs, touching `get_encoder_cudagraph_config, prepare_encoder_cudagraph_capture_inputs`; `tests/v1/cudagraph/test_encoder_cudagraph.py` modified +172/-0 (172 lines); hunks: -32,6 +32,68; -760,3 +822,113 @@ def test_image_and_video_share_manager(self):; symbols: _MockCompilationConfig, __init__, _MockMultimodalConfig, get_limit_per_prompt, touching `_MockCompilationConfig, __init__, _MockMultimodalConfig`; `vllm/v1/worker/encoder_cudagraph.py` modified +52/-12 (64 lines); hunks: -72,25 +72,65 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/config/compilation.py` modified +8/-0 (8 lines); hunks: -1005,6 +1005,14 @@ def __post_init__(self) -> None:; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +10/-9 (19 lines); hunks: -1768,14 +1768,12 @@ def get_encoder_cudagraph_config(self):; -1923,7 +1921,10 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: get_encoder_cudagraph_config, prepare_encoder_cudagraph_capture_inputs
  - `tests/v1/cudagraph/test_encoder_cudagraph.py` modified +172/-0 (172 lines); hunks: -32,6 +32,68; -760,3 +822,113 @@ def test_image_and_video_share_manager(self):; symbols: _MockCompilationConfig, __init__, _MockMultimodalConfig, get_limit_per_prompt
  - `vllm/v1/worker/encoder_cudagraph.py` modified +52/-12 (64 lines); hunks: -72,25 +72,65 @@ def __init__(; symbols: __init__
  - `vllm/config/compilation.py` modified +8/-0 (8 lines); hunks: -1005,6 +1005,14 @@ def __post_init__(self) -> None:; symbols: __post_init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1768,14 +1768,12 @@ def get_encoder_cudagraph_config(self):
-        modalities = ["image"]
-        # NOTE: When EVS (Efficient Video Sampling) pruning is enabled, the number
-        # of tokens becomes data-dependent (i.e., the retained tokens are
-        # dynamically selected based on inter-frame differences) and therefore
-        # cannot be captured by CUDA Graphs. As a result, video CUDA Graphs are
-        # only enabled when EVS is disabled.
diff -- tests/v1/cudagraph/test_encoder_cudagraph.py
@@ -32,6 +32,68 @@
+class _MockCompilationConfig:
+    """Minimal mock for VllmConfig.compilation_config."""
+    def __init__(
+        self,
+        token_budgets: list[int] | None = None,
+        max_mm_items: int = 0,
diff -- vllm/v1/worker/encoder_cudagraph.py
@@ -72,25 +72,65 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +10/-9; `vllm/v1/worker/encoder_cudagraph.py` modified +52/-12; `vllm/config/compilation.py` modified +8/-0
  - tests: `tests/v1/cudagraph/test_encoder_cudagraph.py` modified +172/-0
- Risk and verification: The diff ships test coverage in `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42716 - Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer

- Link: https://github.com/vllm-project/vllm/pull/42716
- Status/date: merged / 2026-05-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl_moe.py`; associated commits `a94189295b8b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl_moe.py`; technical summary: Covers "Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer"; the main implementation surface is `vllm/model_executor/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-2 (4 lines); hunks: -152,8 +152,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights, touching `load_fused_expert_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-2 (4 lines); hunks: -152,8 +152,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl_moe.py
@@ -152,8 +152,8 @@ def load_fused_expert_weights(
-                shard_id,
-                expert_id,
+                shard_id=shard_id,
+                expert_id=expert_id,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42347 - [Perf][4/n] Eliminate various GPU CPU syncs

- Link: https://github.com/vllm-project/vllm/pull/42347
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +129/-108, 606 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][4/n] Eliminate various GPU CPU syncs"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`; technical summary: Covers "[Perf][4/n] Eliminate various GPU CPU syncs"; the main implementation surface is `vllm/model_executor/models/utils.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/granite_speech.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/utils.py` modified +7/-15 (22 lines); hunks: -30,10 +30,8; -498,10 +496,9 @@ def isin_list(; symbols: isin_list, extract_layer_index, cast_overflow_tensors, fast_topk, touching `isin_list, extract_layer_index, cast_overflow_tensors`; `vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7 (19 lines); hunks: -84,6 +84,7; -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):; symbols: rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config, _call_hf_processor, touching `rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config`; `vllm/model_executor/models/granite_speech.py` modified +7/-7 (14 lines); hunks: -143,7 +143,7 @@ def _get_mm_fields_config(; -717,13 +717,13 @@ def _build_input_features_mask(; symbols: _get_mm_fields_config, _get_prompt_updates, _build_input_features_mask, _pad_and_stack_input_features, touching `_get_mm_fields_config, _get_prompt_updates, _build_input_features_mask`; `vllm/model_executor/models/phi4mm_audio.py` modified +9/-3 (12 lines); hunks: -586,7 +586,9 @@ def forward_embeddings(; -605,7 +607,9 @@ def forward_embeddings(; symbols: forward_embeddings, calculate_hs_mask, touching `forward_embeddings, calculate_hs_mask`.
- Code diff details:
  - `vllm/model_executor/models/utils.py` modified +7/-15 (22 lines); hunks: -30,10 +30,8; -498,10 +496,9 @@ def isin_list(; symbols: isin_list, extract_layer_index, cast_overflow_tensors, fast_topk
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7 (19 lines); hunks: -84,6 +84,7; -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):; symbols: rotary_pos_emb_thw, get_rope_by_thw, _get_mm_fields_config, _call_hf_processor
  - `vllm/model_executor/models/granite_speech.py` modified +7/-7 (14 lines); hunks: -143,7 +143,7 @@ def _get_mm_fields_config(; -717,13 +717,13 @@ def _build_input_features_mask(; symbols: _get_mm_fields_config, _get_prompt_updates, _build_input_features_mask, _pad_and_stack_input_features
  - `vllm/model_executor/models/phi4mm_audio.py` modified +9/-3 (12 lines); hunks: -586,7 +586,9 @@ def forward_embeddings(; -605,7 +607,9 @@ def forward_embeddings(; symbols: forward_embeddings, calculate_hs_mask
  - `vllm/model_executor/models/bert.py` modified +3/-6 (9 lines); hunks: -559,13 +559,10 @@ def _encode_token_type_ids(; symbols: _encode_token_type_ids, _decode_token_type_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/utils.py
@@ -30,10 +30,8 @@
-from vllm.utils.platform_utils import (
-    is_pin_memory_available,
-)
+    async_tensor_h2d,
@@ -498,10 +496,9 @@ def isin_list(
-    test_elements = torch.tensor(
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -84,6 +84,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -679,6 +680,7 @@ def rotary_pos_emb_thw(self, t, h, w):
+        pos_ids = pos_ids.to(cos.device, non_blocking=True)
@@ -737,9 +739,10 @@ def get_rope_by_thw(self, t, h, w):
-        cos_thw = cos_thw[window_index_thw, :, :]
+        window_index_thw_dev = window_index_thw.to(cos_thw.device, non_blocking=True)
diff -- vllm/model_executor/models/granite_speech.py
@@ -143,7 +143,7 @@ def _get_mm_fields_config(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/utils.py` modified +7/-15; `vllm/model_executor/models/qwen2_5_vl.py` modified +12/-7; `vllm/model_executor/models/granite_speech.py` modified +7/-7; `vllm/model_executor/models/phi4mm_audio.py` modified +9/-3; `vllm/model_executor/models/bert.py` modified +3/-6; `vllm/model_executor/models/qwen3_vl.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `tests/v1/logits_processors/test_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43647 - [ROCm][CI] Fix ROCm multimodal Qwen2.5-VL activation compile and Phi4MM ragged image mask handling

- Link: https://github.com/vllm-project/vllm/pull/43647
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen2_5_vl.py`; associated commits `5bdb181df5bd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +52/-9, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix ROCm multimodal Qwen2.5-VL activation compile and Phi4MM ragged image mask handling"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen2_5_vl.py`; technical summary: Covers "[ROCm][CI] Fix ROCm multimodal Qwen2.5-VL activation compile and Phi4MM ragged image mask handling"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -81,6 +81,7; -641,7 +642,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -81,6 +81,7; -641,7 +642,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -81,6 +81,7 @@
+from vllm.platforms import current_platform
@@ -641,7 +642,10 @@ def __init__(
-                    act_fn=get_act_and_mul_fn(vision_config.hidden_act),
+                    act_fn=get_act_and_mul_fn(
+                        vision_config.hidden_act,
+                        compile_native=not current_platform.is_rocm(),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/activation.py`, `vllm/model_executor/models/phi4mm.py`, `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43617 - Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile

- Link: https://github.com/vllm-project/vllm/pull/43617
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `5963c194787d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +28/-22, 92 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +14/-11 (25 lines); hunks: -1778,8 +1778,8 @@ def _get_deepstack_input_embeds(; -1791,22 +1791,25 @@ def _get_deepstack_input_embeds(; symbols: _get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds, touching `_get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds`; `vllm/model_executor/models/qwen3_vl.py` modified +14/-11 (25 lines); hunks: -1715,8 +1715,8 @@ def _get_deepstack_input_embeds(; -1728,22 +1728,25 @@ def _get_deepstack_input_embeds(; symbols: _get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds, touching `_get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +14/-11 (25 lines); hunks: -1778,8 +1778,8 @@ def _get_deepstack_input_embeds(; -1791,22 +1791,25 @@ def _get_deepstack_input_embeds(; symbols: _get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds
  - `vllm/model_executor/models/qwen3_vl.py` modified +14/-11 (25 lines); hunks: -1715,8 +1715,8 @@ def _get_deepstack_input_embeds(; -1728,22 +1728,25 @@ def _get_deepstack_input_embeds(; symbols: _get_deepstack_input_embeds, _resize_deepstack_input_embeds, _set_deepstack_input_embeds
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1778,8 +1778,8 @@ def _get_deepstack_input_embeds(
-        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
-            return None
+        if num_tokens > self.deepstack_input_embeds[0].size(0):
+            self._resize_deepstack_input_embeds(num_tokens)
@@ -1791,22 +1791,25 @@ def _get_deepstack_input_embeds(
+    def _resize_deepstack_input_embeds(self, num_tokens: int) -> None:
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1715,8 +1715,8 @@ def _get_deepstack_input_embeds(
-        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
-            return None
+        if num_tokens > self.deepstack_input_embeds[0].size(0):
+            self._resize_deepstack_input_embeds(num_tokens)
@@ -1728,22 +1728,25 @@ def _get_deepstack_input_embeds(
+    def _resize_deepstack_input_embeds(self, num_tokens: int) -> None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +14/-11; `vllm/model_executor/models/qwen3_vl.py` modified +14/-11
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44205 - [Bugfix] fix EVS for qwen3-vl

- Link: https://github.com/vllm-project/vllm/pull/44205
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_vl.py`; associated commits `4b87b3e845fc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix EVS for qwen3-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Bugfix] fix EVS for qwen3-vl"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +4/-4 (8 lines); hunks: -2269,6 +2269,8 @@ def _create_final_video_embeddings(; -2283,10 +2285,8 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings, touching `_create_final_video_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +4/-4 (8 lines); hunks: -2269,6 +2269,8 @@ def _create_final_video_embeddings(; -2283,10 +2285,8 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -2269,6 +2269,8 @@ def _create_final_video_embeddings(
+        device = video_embeddings.device
@@ -2283,10 +2285,8 @@ def _create_final_video_embeddings(
-        repl_token_ids = torch.tensor(video_repl.full)
-        embed_token_id = _cached_tensor(
-            self.config.video_token_id, repl_token_ids.device
-        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44264 - [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention

- Link: https://github.com/vllm-project/vllm/pull/44264
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; associated commits `540aaf21406b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; technical summary: Covers "[Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention"; the main implementation surface is `vllm/model_executor/models/qwen3_omni_moe_thinker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +3/-0 (3 lines); hunks: -991,6 +991,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +3/-0 (3 lines); hunks: -991,6 +991,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -991,6 +991,9 @@ def forward(
+        # Move cu_seqlens to GPU; grid_thw may be on CPU during profile_run
+        # and FA3 vit attention requires cu_seqlens on CUDA.
+        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45131 - Deprecated 1st generation Qwen and QwenVL models

- Link: https://github.com/vllm-project/vllm/pull/45131
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +6/-1349, 1585 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecated 1st generation Qwen and QwenVL models"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py`; technical summary: Covers "Deprecated 1st generation Qwen and QwenVL models"; the main implementation surface is `vllm/model_executor/models/qwen_vl.py`, `vllm/model_executor/models/qwen.py`, `vllm/tokenizers/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen_vl.py` removed +0/-688 (688 lines); hunks: -1,688 +0,0; symbols: QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention, __init__, touching `QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention`; `vllm/model_executor/models/qwen.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: QWenMLP, __init__, forward, QWenAttention, touching `QWenMLP, __init__, forward`; `vllm/tokenizers/qwen_vl.py` removed +0/-71 (71 lines); hunks: -1,71 +0,0; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode, touching `get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize`; `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44 (44 lines); hunks: -1042,49 +1042,6 @@ def load_phi4siglip(question: str, image_urls: list[str])...; -1544,7 +1501,6 @@ def load_molmo2(question: str, image_urls: list[str]) -> M...; symbols: load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl, load_molmo2, touching `load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl`.
- Code diff details:
  - `vllm/model_executor/models/qwen_vl.py` removed +0/-688 (688 lines); hunks: -1,688 +0,0; symbols: QwenImagePixelInputs, QwenImageEmbeddingInputs, VisualAttention, __init__
  - `vllm/model_executor/models/qwen.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: QWenMLP, __init__, forward, QWenAttention
  - `vllm/tokenizers/qwen_vl.py` removed +0/-71 (71 lines); hunks: -1,71 +0,0; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode
  - `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44 (44 lines); hunks: -1042,49 +1042,6 @@ def load_phi4siglip(question: str, image_urls: list[str])...; -1544,7 +1501,6 @@ def load_molmo2(question: str, image_urls: list[str]) -> M...; symbols: load_phi4siglip, load_qwen_vl_chat, load_qwen2_vl, load_molmo2
  - `vllm/transformers_utils/processors/qwen_vl.py` removed +0/-42 (42 lines); hunks: -1,42 +0,0; symbols: QwenVLImageProcessorFast, QwenVLProcessor, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen_vl.py
@@ -1,688 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Adapted from
-# https://huggingface.co/Qwen/Qwen-VL/blob/main/modeling_qwen.py
-# Copyright (c) Alibaba Cloud.
-"""Inference-only Qwen-VL model compatible with HuggingFace weights."""
diff -- vllm/model_executor/models/qwen.py
@@ -1,377 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Adapted from
-# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
-# Copyright (c) Alibaba Cloud.
-# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
diff -- vllm/tokenizers/qwen_vl.py
@@ -1,71 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen_vl.py` removed +0/-688; `vllm/model_executor/models/qwen.py` removed +0/-377; `vllm/tokenizers/qwen_vl.py` removed +0/-71; `vllm/transformers_utils/processors/qwen_vl.py` removed +0/-42
  - docs: `examples/generate/multimodal/vision_language_multi_image_offline.py` modified +0/-44; `examples/generate/multimodal/vision_language_offline.py` modified +0/-22
  - tests: `tests/models/registry.py` modified +0/-18; `tests/tokenizers_/conftest.py` removed +0/-14
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`, `tests/models/multimodal/conftest.py`, `tests/models/multimodal/generation/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35415 - feat(qwen3-asr): support prompt parameter in v1/audio/transcriptions

- Link: https://github.com/vllm-project/vllm/pull/35415
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py`, `vllm/model_executor/models/qwen3_asr.py`; associated commits `12f3f19c1959`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +145/-13, 241 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(qwen3-asr): support prompt parameter in v1/audio/transcriptions"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen3_asr.py`, `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py`; technical summary: Covers "feat(qwen3-asr): support prompt parameter in v1/audio/transcriptions"; the main implementation surface is `vllm/model_executor/models/qwen3_asr.py`, `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_asr.py` modified +54/-12 (66 lines); hunks: -25,6 +25,7; -90,6 +91,31; symbols: _sanitize_transcription_user_text, _get_feat_extract_output_lengths, get_speech_to_text_config, get_generation_prompt, touching `_sanitize_transcription_user_text, _get_feat_extract_output_lengths, get_speech_to_text_config`; `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: test_sanitize_strips_control_tokens, test_sanitize_handles_falsy_inputs, test_sanitize_is_idempotent, touching `test_sanitize_strips_control_tokens, test_sanitize_handles_falsy_inputs, test_sanitize_is_idempotent`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_asr.py` modified +54/-12 (66 lines); hunks: -25,6 +25,7; -90,6 +91,31; symbols: _sanitize_transcription_user_text, _get_feat_extract_output_lengths, get_speech_to_text_config, get_generation_prompt
  - `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: test_sanitize_strips_control_tokens, test_sanitize_handles_falsy_inputs, test_sanitize_is_idempotent
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_asr.py
@@ -25,6 +25,7 @@
+import regex as re
@@ -90,6 +91,31 @@
+# User-supplied `prompt` / `response_prefix` must not inject extra ChatML turns.
+_CHATML_LIKE_TOKEN = re.compile(r"<\|[^|]+\|>")
+def _sanitize_transcription_user_text(text: str) -> str:
+    """Strip ChatML-style special tokens from user-controlled transcription fields.
diff -- tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py
@@ -0,0 +1,64 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for ``Qwen3ASR``'s user-text sanitizer.
+The sanitizer is the security boundary between user-supplied transcription
+fields (``prompt`` / ``response_prefix``) and the structured ChatML prompt
+template. It must strip both ``<|...|>`` control tokens and the
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +54/-12
  - tests: `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py` added +64/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45161 - Deprecate Transformers v4 support

- Link: https://github.com/vllm-project/vllm/pull/45161
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +62/-268, 612 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecate Transformers v4 support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`; technical summary: Covers "Deprecate Transformers v4 support"; the main implementation surface is `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings, touching `_patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length, touching `pad_to_hop_length`; `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm, touching `enable_hf_transfer, enable_xet_high_performance, DisabledTqdm`; `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length
  - `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12 (17 lines); hunks: -100,18 +100,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/transformers/base.py
@@ -27,6 +27,10 @@
+from transformers.conversion_mapping import (
+    WeightRenaming,
+    get_model_conversion_mapping,
+)
@@ -212,16 +216,9 @@ def _patch_config(self):
-        - Propagates this dtype to any sub-configs because Transformers model
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -30,9 +30,7 @@
-from packaging.version import Version
-from transformers import __version__ as TRANSFORMERS_VERSION
@@ -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
-            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
-                # Extract audio_sample_rate before restructuring
-                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -77,30 +77,13 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/transformers/base.py` modified +16/-42; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36; `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18; `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12; `vllm/model_executor/models/ultravox.py` modified +0/-15
- Risk and verification: Runtime changes concentrate in `vllm/config/vllm.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/gemma3n_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42700 - [Bugfix] Replace deprecated Qwen2VLImageProcessorFast with Qwen2VLImageProcessor

- Link: https://github.com/vllm-project/vllm/pull/42700
- Status/date: merged / 2026-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Replace deprecated Qwen2VLImageProcessorFast with Qwen2VLImageProcessor"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Bugfix] Replace deprecated Qwen2VLImageProcessorFast with Qwen2VLImageProcessor"; the main implementation surface is `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_vl.py` modified +3/-3 (6 lines); hunks: -34,7 +34,7; -871,7 +871,7 @@ def get_hf_processor(self, **kwargs: object) -> Qwen3VLProce...; symbols: get_hf_processor, get_image_processor, get_video_processor, _get_vision_info, touching `get_hf_processor, get_image_processor, get_video_processor`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_vl.py` modified +3/-3 (6 lines); hunks: -34,7 +34,7; -871,7 +871,7 @@ def get_hf_processor(self, **kwargs: object) -> Qwen3VLProce...; symbols: get_hf_processor, get_image_processor, get_video_processor, _get_vision_info
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -34,7 +34,7 @@
-from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
+from transformers.models.qwen2_vl import Qwen2VLImageProcessor
@@ -871,7 +871,7 @@ def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
-    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessorFast:
+    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
@@ -891,7 +891,7 @@ def _get_vision_info(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- Link: https://github.com/vllm-project/vllm/pull/43586
- Status/date: merged / 2026-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +809/-69, 1559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features, touching `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__`; `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata, touching `BudgetGraphMetadata`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template, touching `VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template`; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features
  - `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template
  - `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2
  - `vllm/model_executor/models/interfaces.py` modified +5/-0 (5 lines); hunks: -1623,6 +1623,7 @@ def postprocess_encoder_output(; -1643,6 +1644,7 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, prepare_encoder_cudagraph_replay_buffers, encoder_cudagraph_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -4,7 +4,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -15,6 +15,7 @@
+    SupportsEncoderCudaGraph,
@@ -52,6 +53,7 @@
+    IMAGE_SIZE,
diff -- docs/design/cuda_graphs_multimodal.md
@@ -2,6 +2,8 @@
+For two-tower vision encoders (e.g., DeepSeek-OCR's SAM + CLIP with dynamic tiling), a **dual-path graph** mode captures two independent sets of CUDA graphs — one for the global i
@@ -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on the host side. Th
+For two-tower vision encoders such as DeepSeek-OCR (SAM + CLIP with dynamic tiling), the global image path and local patch path have independent token profiles (272 tokens per glo
@@ -37,17 +41,57 @@ class BudgetGraphMetadata:
+When `EncoderCudaGraphConfig.enable_dual_path_graph` is `True`, the manager generates two independent budget lists — `global_token_budgets` (multiples of `global_token_per_image`)
+For dual-path models, the manager routes to `_execute_local_dual_path()`, which constrains both global and local token budgets simultaneously during packing (see [Dual-Path graph
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -29,6 +29,7 @@ class VitCudagraphTestConfig:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5; `vllm/model_executor/models/interfaces.py` modified +5/-0; `vllm/model_executor/models/step3_vl.py` modified +5/-0; `vllm/model_executor/models/glm4_1v.py` modified +4/-0; `vllm/model_executor/models/internvl.py` modified +4/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +63/-16; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45424 - [Core] Ensure memory is pinned prior to async h2d copy

- Link: https://github.com/vllm-project/vllm/pull/45424
- Status/date: merged / 2026-06-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 49 files, +254/-264, 1718 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Core] Ensure memory is pinned prior to async h2d copy"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`; technical summary: Covers "[Core] Ensure memory is pinned prior to async h2d copy"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/pooler/seqwise/methods.py`, `vllm/multimodal/inputs.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build, touching `build`; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward, touching `forward`; `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data, touching `_reduce_data`; `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata, touching `_apply_rope_input_validation, prepare_encoder_metadata`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8 (18 lines); hunks: -1684,12 +1684,13 @@ def build(; -1746,12 +1747,13 @@ def build(; symbols: build
  - `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8 (16 lines); hunks: -10,6 +10,7; -74,15 +75,14 @@ def forward(; symbols: forward
  - `vllm/multimodal/inputs.py` modified +14/-2 (16 lines); hunks: -488,7 +488,13 @@ def _reduce_data(; -538,7 +544,13 @@ def _reduce_data(; symbols: _reduce_data
  - `vllm/model_executor/models/moonvit.py` modified +3/-2 (5 lines); hunks: -66,6 +66,7; -758,7 +759,7 @@ def prepare_encoder_metadata(; symbols: _apply_rope_input_validation, prepare_encoder_metadata
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -83,9 +83,8; -825,7 +824,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -1684,12 +1684,13 @@ def build(
-                chunk_starts = (
+                chunk_starts = torch.empty(
+                    num_chunks, num_prefills, dtype=torch.int32, pin_memory=True
+                ).copy_(
+                    .multiply_(max_context_chunk)
-                    .expand(-1, num_prefills)
diff -- vllm/model_executor/layers/pooler/seqwise/methods.py
@@ -10,6 +10,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -74,15 +75,14 @@ def forward(
-        # Build segment_ids on CPU so repeat_interleave doesn't need to sync
-        # GPU->CPU to learn its data-dependent output length, then upload
-        # non-blocking. eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
+        prompt_lens = async_tensor_h2d(
diff -- vllm/multimodal/inputs.py
@@ -488,7 +488,13 @@ def _reduce_data(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +10/-8; `vllm/model_executor/layers/pooler/seqwise/methods.py` modified +8/-8; `vllm/multimodal/inputs.py` modified +14/-2; `vllm/model_executor/models/moonvit.py` modified +3/-2; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-3; `vllm/model_executor/layers/attention/mm_encoder_attention.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/v1/logits_processors/test_correctness.py`, `tests/v1/streaming_input/test_gpu_model_runner_streaming.py`, `tests/v1/worker/test_gpu_input_batch.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46305 - [Bugfix][Qwen3-VL] Fix multi-video crash with list-valued fps/num_frames

- Link: https://github.com/vllm-project/vllm/pull/46305
- Status/date: merged / 2026-06-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; associated commits `12fe2a9aac8e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +59/-2, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Qwen3-VL] Fix multi-video crash with list-valued fps/num_frames"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "[Bugfix][Qwen3-VL] Fix multi-video crash with list-valued fps/num_frames"; the main implementation surface is `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_qwen3_vl.py` modified +46/-0 (46 lines); hunks: -138,3 +138,49 @@ def test_processor_multi_video(; symbols: test_processor_multi_video, test_processor_multi_video_list_kwargs, touching `test_processor_multi_video, test_processor_multi_video_list_kwargs`; `vllm/model_executor/models/qwen3_vl.py` modified +13/-2 (15 lines); hunks: -1271,7 +1271,7 @@ def _call_hf_processor(; -1282,6 +1282,12 @@ def _call_hf_processor(; symbols: _call_hf_processor, touching `_call_hf_processor`.
- Code diff details:
  - `tests/models/multimodal/processing/test_qwen3_vl.py` modified +46/-0 (46 lines); hunks: -138,3 +138,49 @@ def test_processor_multi_video(; symbols: test_processor_multi_video, test_processor_multi_video_list_kwargs
  - `vllm/model_executor/models/qwen3_vl.py` modified +13/-2 (15 lines); hunks: -1271,7 +1271,7 @@ def _call_hf_processor(; -1282,6 +1282,12 @@ def _call_hf_processor(; symbols: _call_hf_processor
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_qwen3_vl.py
@@ -138,3 +138,49 @@ def test_processor_multi_video(
+@pytest.mark.parametrize("model_id", [MODEL_ID])
+@pytest.mark.parametrize(
+    "hf_mm_kwargs",
+    [{"num_frames": [8, 16]}, {"fps": [2.0, 4.0]}],
+)
+def test_processor_multi_video_list_kwargs(
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1271,7 +1271,7 @@ def _call_hf_processor(
-            for item in videos:
+            for item_idx, item in enumerate(videos):
@@ -1282,6 +1282,12 @@ def _call_hf_processor(
+                sampled_fps = video_mm_kwargs.get("fps")
+                if is_list_of(sampled_fps, float):
+                    video_mm_kwargs["fps"] = sampled_fps[item_idx]
```

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_qwen3_vl.py` modified +46/-0
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +13/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_qwen3_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
