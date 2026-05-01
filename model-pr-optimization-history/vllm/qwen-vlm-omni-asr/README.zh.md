# vllm Qwen VLM/Omni/ASR 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/generate/multimodal/qwen2_5_omni/README.md` | 无直接 PR 号提交 |
| `examples/generate/multimodal/qwen2_5_omni/only_thinker.py` | 无直接 PR 号提交 |
| `examples/generate/multimodal/qwen3_omni/only_thinker.py` | 无直接 PR 号提交 |
| `examples/pooling/embed/template/dse_qwen2_vl.jinja` | 无直接 PR 号提交 |
| `examples/pooling/score/template/qwen3_vl_reranker.jinja` | [#31890](https://github.com/vllm-project/vllm/pull/31890) |
| `tests/model_executor/test_qwen3_omni.py` | [#27721](https://github.com/vllm-project/vllm/pull/27721) |
| `tests/model_executor/test_qwen3_vl_mrope.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/generation/test_qwen2_5_vl.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/generation/test_qwen2_vl.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/pooling/test_dse_qwen2_vl.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/pooling/test_qwen3_asr_forced_aligner.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` | [#35368](https://github.com/vllm-project/vllm/pull/35368) |
| `tests/models/multimodal/processing/test_qwen2_vl.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/processing/test_qwen3_omni.py` | [#29255](https://github.com/vllm-project/vllm/pull/29255) |
| `tests/models/multimodal/processing/test_qwen3_vl.py` | [#36136](https://github.com/vllm-project/vllm/pull/36136) |
| `vllm/model_executor/models/glmasr.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779), [#32540](https://github.com/vllm-project/vllm/pull/32540), [#40160](https://github.com/vllm-project/vllm/pull/40160) |
| `vllm/model_executor/models/glmasr_utils.py` | [#31436](https://github.com/vllm-project/vllm/pull/31436), [#31779](https://github.com/vllm-project/vllm/pull/31779) |
| `vllm/model_executor/models/mimo_v2_omni.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen2_5_omni_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#16872](https://github.com/vllm-project/vllm/pull/16872), [#17301](https://github.com/vllm-project/vllm/pull/17301), [#17838](https://github.com/vllm-project/vllm/pull/17838), [#23058](https://github.com/vllm-project/vllm/pull/23058), [#24231](https://github.com/vllm-project/vllm/pull/24231), [#24420](https://github.com/vllm-project/vllm/pull/24420), [#26004](https://github.com/vllm-project/vllm/pull/26004), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#30883](https://github.com/vllm-project/vllm/pull/30883), [#32772](https://github.com/vllm-project/vllm/pull/32772), ... (17 total) |
| `vllm/model_executor/models/qwen2_5_vl.py` | [#12944](https://github.com/vllm-project/vllm/pull/12944), [#13155](https://github.com/vllm-project/vllm/pull/13155), [#13286](https://github.com/vllm-project/vllm/pull/13286), [#13533](https://github.com/vllm-project/vllm/pull/13533), [#13968](https://github.com/vllm-project/vllm/pull/13968), [#14377](https://github.com/vllm-project/vllm/pull/14377), [#15130](https://github.com/vllm-project/vllm/pull/15130), [#15200](https://github.com/vllm-project/vllm/pull/15200), [#15273](https://github.com/vllm-project/vllm/pull/15273), [#16907](https://github.com/vllm-project/vllm/pull/16907), [#16974](https://github.com/vllm-project/vllm/pull/16974), [#17726](https://github.com/vllm-project/vllm/pull/17726), ... (25 total) |
| `vllm/model_executor/models/qwen2_audio.py` | [#11258](https://github.com/vllm-project/vllm/pull/11258), [#35994](https://github.com/vllm-project/vllm/pull/35994) |
| `vllm/model_executor/models/qwen2_vl.py` | [#7905](https://github.com/vllm-project/vllm/pull/7905), [#8442](https://github.com/vllm-project/vllm/pull/8442), [#8696](https://github.com/vllm-project/vllm/pull/8696), [#8770](https://github.com/vllm-project/vllm/pull/8770), [#8837](https://github.com/vllm-project/vllm/pull/8837), [#9250](https://github.com/vllm-project/vllm/pull/9250), [#10112](https://github.com/vllm-project/vllm/pull/10112), [#10169](https://github.com/vllm-project/vllm/pull/10169), [#10221](https://github.com/vllm-project/vllm/pull/10221), [#11258](https://github.com/vllm-project/vllm/pull/11258), [#11430](https://github.com/vllm-project/vllm/pull/11430), [#11663](https://github.com/vllm-project/vllm/pull/11663), ... (31 total) |
| `vllm/model_executor/models/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312), [#33410](https://github.com/vllm-project/vllm/pull/33410), [#33644](https://github.com/vllm-project/vllm/pull/33644), [#37247](https://github.com/vllm-project/vllm/pull/37247) |
| `vllm/model_executor/models/qwen3_asr_forced_aligner.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3_asr_realtime.py` | [#34613](https://github.com/vllm-project/vllm/pull/34613), [#35869](https://github.com/vllm-project/vllm/pull/35869) |
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | [#25550](https://github.com/vllm-project/vllm/pull/25550), [#26608](https://github.com/vllm-project/vllm/pull/26608), [#26815](https://github.com/vllm-project/vllm/pull/26815), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#29255](https://github.com/vllm-project/vllm/pull/29255), [#29828](https://github.com/vllm-project/vllm/pull/29828), [#29896](https://github.com/vllm-project/vllm/pull/29896), [#29974](https://github.com/vllm-project/vllm/pull/29974), [#31007](https://github.com/vllm-project/vllm/pull/31007), [#31790](https://github.com/vllm-project/vllm/pull/31790), ... (24 total) |
| `vllm/model_executor/models/qwen3_vl.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25337](https://github.com/vllm-project/vllm/pull/25337), [#25347](https://github.com/vllm-project/vllm/pull/25347), [#25557](https://github.com/vllm-project/vllm/pull/25557), [#25646](https://github.com/vllm-project/vllm/pull/25646), [#25648](https://github.com/vllm-project/vllm/pull/25648), [#25788](https://github.com/vllm-project/vllm/pull/25788), [#26000](https://github.com/vllm-project/vllm/pull/26000), [#27104](https://github.com/vllm-project/vllm/pull/27104), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#28663](https://github.com/vllm-project/vllm/pull/28663), ... (19 total) |
| `vllm/model_executor/models/qwen3_vl_moe.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25300](https://github.com/vllm-project/vllm/pull/25300), [#26000](https://github.com/vllm-project/vllm/pull/26000) |
| `vllm/model_executor/models/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/tokenizers/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/transformers_utils/configs/mimo_v2_omni.py` | 无直接 PR 号提交 |
| `vllm/transformers_utils/configs/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/mimo_v2_omni.py` | 无直接 PR 号提交 |
| `vllm/transformers_utils/processors/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/qwen_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 102
- 原文档显式引用补充 PR 数: 6
- 当前文档总 PR 数: 108
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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
| 2025-04-19 | [#15130](https://github.com/vllm-project/vllm/pull/15130) | merged | [Model][VLM] Add Qwen2.5-Omni model support (thinker only) | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py` |
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
| 2025-10-03 | [#26104](https://github.com/vllm-project/vllm/pull/26104) | merged | [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py` |
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
| 2026-01-25 | [#32772](https://github.com/vllm-project/vllm/pull/32772) | merged | [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| 2026-01-26 | [#33010](https://github.com/vllm-project/vllm/pull/33010) | merged | [Model] Use mm_position to compute mrope positions for Qwen3-Omni | `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
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
| 2026-04-27 | [#40932](https://github.com/vllm-project/vllm/pull/40932) | merged | [Bugfix] Remove invalid deepstack boundary check for Qwen3-VL | `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py` |

## 逐 PR diff 审计卡

### PR #7905 - [Model][VLM] Add Qwen2-VL model support

- 链接: https://github.com/vllm-project/vllm/pull/7905
- 状态/时间: merged / 2024-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `3b7fea770f44`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1531/-31，可读 patch 1844 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Add Qwen2-VL model support」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model][VLM] Add Qwen2-VL model support」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` added +1088/-0 (1088 lines); hunks: -0,0 +1,1088; symbols: Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP, __init__，涉及 `Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` added +1088/-0 (1088 lines); hunks: -0,0 +1,1088; symbols: Qwen2VLImageInputs, Qwen2VLVideoInputs, Qwen2VisionMLP, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` added +1088/-0
- 验证与风险: diff 自带测试面 `tests/models/test_registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8442 - [Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8

- 链接: https://github.com/vllm-project/vllm/pull/8442
- 状态/时间: merged / 2024-09-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `06311e295666`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -1055,6 +1055,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1078,6 +1081,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -1055,6 +1055,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -1078,6 +1081,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8696 - [Model] Support pp for qwen2-vl

- 链接: https://github.com/vllm-project/vllm/pull/8696
- 状态/时间: merged / 2024-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `a79e5229843e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+46/-14，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support pp for qwen2-vl」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model] Support pp for qwen2-vl」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +22/-7 (29 lines); hunks: -45,7 +45,7; -68,6 +68,9; symbols: __init__, _validate_and_reshape_mm_tensor, forward, load_weights，涉及 `__init__, _validate_and_reshape_mm_tensor, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-7 (29 lines); hunks: -45,7 +45,7; -68,6 +68,9; symbols: __init__, _validate_and_reshape_mm_tensor, forward, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +22/-7
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8770 - [Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend

- 链接: https://github.com/vllm-project/vllm/pull/8770
- 状态/时间: merged / 2024-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c23953675f78`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+99/-9，可读 patch 202 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +16/-0 (16 lines); hunks: -67,6 +67,7; -281,6 +282,21 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +16/-0 (16 lines); hunks: -67,6 +67,7; -281,6 +282,21 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +16/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`, `vllm/worker/cpu_model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8837 - [Misc] Update config loading for Qwen2-VL and remove Granite

- 链接: https://github.com/vllm-project/vllm/pull/8837
- 状态/时间: merged / 2024-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `4bb98f2190aa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+144/-224，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Update config loading for Qwen2-VL and remove Granite」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Update config loading for Qwen2-VL and remove Granite」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +2/-3 (5 lines); hunks: -31,12 +31,9; -66,6 +63,8。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +2/-3 (5 lines); hunks: -31,12 +31,9; -66,6 +63,8
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +2/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/granite.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/transformers_utils/config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9250 - [Misc] Standardize RoPE handling for Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/9250
- 状态/时间: merged / 2024-10-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `7e7eae338d27`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+102/-200，可读 patch 533 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Standardize RoPE handling for Qwen2-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Standardize RoPE handling for Qwen2-VL」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +4/-4 (8 lines); hunks: -34,6 +34,8; -62,8 +64,7; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +4/-4 (8 lines); hunks: -34,6 +34,8; -62,8 +64,7; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/kernels/test_pos_encoding.py`, `tests/lora/test_layers.py`, `tests/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10112 - [Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/10112
- 状态/时间: merged / 2024-11-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `999df95b4eef`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-10，可读 patch 68 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +23/-10 (33 lines); hunks: -22,8 +22,8; -558,6 +558,17 @@ def forward(; symbols: forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl, get_max_qwen2_vl_mm_tokens，涉及 `forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +23/-10 (33 lines); hunks: -22,8 +22,8; -558,6 +558,17 @@ def forward(; symbols: forward, get_mm_processor_kwargs, mm_input_mapper_for_qwen2_vl, get_max_qwen2_vl_mm_tokens
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +23/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10169 - [Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module

- 链接: https://github.com/vllm-project/vllm/pull/10169
- 状态/时间: merged / 2024-11-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `f83feccd7f66`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +12/-2 (14 lines); hunks: -51,7 +51,9; -982,7 +984,7 @@ def __init__(self,; symbols: __init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor，涉及 `__init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-2 (14 lines); hunks: -51,7 +51,9; -982,7 +984,7 @@ def __init__(self,; symbols: __init__, _maybe_ignore_quant_config, _validate_and_reshape_mm_tensor
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10221 - [Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions

- 链接: https://github.com/vllm-project/vllm/pull/10221
- 状态/时间: merged / 2024-11-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `3945c82346da`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+578/-32，可读 patch 709 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `tests/models/decoder_only/vision_language/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions」；主要实现面是 `tests/models/decoder_only/vision_language/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0 (428 lines); hunks: -0,0 +1,428; symbols: qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput, batch_make_image_embeddings，涉及 `qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput`；`vllm/model_executor/models/qwen2_vl.py` modified +149/-31 (180 lines); hunks: -79,7 +79,7; -92,17 +92,31 @@ class Qwen2VLImagePixelInputs(TypedDict):; symbols: Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs, Qwen2VLVideoPixelInputs，涉及 `Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs`。
- 代码 diff 细节:
  - `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0 (428 lines); hunks: -0,0 +1,428; symbols: qwen2_vl_chat_template, Qwen2VLPromptImageEmbeddingInput, Qwen2VLPromptVideoEmbeddingInput, batch_make_image_embeddings
  - `vllm/model_executor/models/qwen2_vl.py` modified +149/-31 (180 lines); hunks: -79,7 +79,7; -92,17 +92,31 @@ class Qwen2VLImagePixelInputs(TypedDict):; symbols: Qwen2VLImagePixelInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLVideoInputs, Qwen2VLVideoPixelInputs
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` added +428/-0
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +149/-31
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_qwen2_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11258 - [Model] Refactor Qwen2-VL to use merged multimodal processor

- 链接: https://github.com/vllm-project/vllm/pull/11258
- 状态/时间: merged / 2024-12-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `e24113a8fe5d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+277/-527，可读 patch 1006 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Refactor Qwen2-VL to use merged multimodal processor」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_audio.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +187/-393 (580 lines); hunks: -22,28 +22,26; -56,14 +54,14; symbols: Qwen2VisionMLP, __init__, load_weights, get_mm_processor_kwargs，涉及 `Qwen2VisionMLP, __init__, load_weights`；`tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127 (192 lines); hunks: -1,12 +1,9; -20,22 +17,9; symbols: image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context, processor_for_qwen2_vl，涉及 `image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context`；`vllm/model_executor/models/qwen2_audio.py` modified +3/-1 (4 lines); hunks: -164,7 +164,9 @@ def _get_dummy_mm_inputs(; symbols: _get_dummy_mm_inputs，涉及 `_get_dummy_mm_inputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +187/-393 (580 lines); hunks: -22,28 +22,26; -56,14 +54,14; symbols: Qwen2VisionMLP, __init__, load_weights, get_mm_processor_kwargs
  - `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127 (192 lines); hunks: -1,12 +1,9; -20,22 +17,9; symbols: image_input_mapper_for_qwen2_vl, input_processor_for_qwen2_vl, qwen2_vl_context, processor_for_qwen2_vl
  - `vllm/model_executor/models/qwen2_audio.py` modified +3/-1 (4 lines); hunks: -164,7 +164,9 @@ def _get_dummy_mm_inputs(; symbols: _get_dummy_mm_inputs
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +187/-393; `vllm/model_executor/models/qwen2_audio.py` modified +3/-1
  - tests: `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py` modified +65/-127
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11430 - [Bugfix] Fix Qwen2-VL LoRA weight loading

- 链接: https://github.com/vllm-project/vllm/pull/11430
- 状态/时间: merged / 2024-12-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `b1b1038fbdc1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+168/-14，可读 patch 298 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen2-VL LoRA weight loading」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen2-VL LoRA weight loading」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +6/-6 (12 lines); hunks: -901,6 +901,11 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMu...; -1190,11 +1195,6 @@ def sample(; symbols: Qwen2VLForConditionalGeneration, __init__, sample, load_weights，涉及 `Qwen2VLForConditionalGeneration, __init__, sample`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-6 (12 lines); hunks: -901,6 +901,11 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsMu...; -1190,11 +1195,6 @@ def sample(; symbols: Qwen2VLForConditionalGeneration, __init__, sample, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +6/-6
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_lora_checkpoints.py`, `tests/lora/test_qwen2vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11663 - [Misc] Optimize Qwen2-VL LoRA test

- 链接: https://github.com/vllm-project/vllm/pull/11663
- 状态/时间: merged / 2025-01-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `11d8a091c6c7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+21/-4，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Optimize Qwen2-VL LoRA test」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Optimize Qwen2-VL LoRA test」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +19/-1 (20 lines); hunks: -53,6 +53,7; -925,15 +926,23 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsM...; symbols: Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping，涉及 `Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +19/-1 (20 lines); hunks: -53,6 +53,7; -925,15 +926,23 @@ class Qwen2VLForConditionalGeneration(nn.Module, SupportsM...; symbols: Qwen2VLForConditionalGeneration, load_weights, get_mm_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +19/-1
- 验证与风险: diff 自带测试面 `tests/lora/test_qwen2vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12128 - [V1] Add V1 support of Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/12128
- 状态/时间: merged / 2025-01-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `81763c58a01e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+292/-85，可读 patch 616 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] Add V1 support of Qwen2-VL」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/test_qwen2_vl.py`；技术摘要: 覆盖「[V1] Add V1 support of Qwen2-VL」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/test_qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +78/-64 (142 lines); hunks: -67,11 +67,15; -135,7 +139,7 @@ class Qwen2VLVideoEmbeddingInputs(TypedDict):; symbols: Qwen2VLVideoEmbeddingInputs, forward, load_weights, get_num_frames_with_most_features，涉及 `Qwen2VLVideoEmbeddingInputs, forward, load_weights`；`tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10 (18 lines); hunks: -105,7 +105,7 @@ def batch_make_image_embeddings(; -124,11 +124,10 @@ def batch_make_image_embeddings(; symbols: batch_make_image_embeddings, batch_make_video_embeddings，涉及 `batch_make_image_embeddings, batch_make_video_embeddings`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +78/-64 (142 lines); hunks: -67,11 +67,15; -135,7 +139,7 @@ class Qwen2VLVideoEmbeddingInputs(TypedDict):; symbols: Qwen2VLVideoEmbeddingInputs, forward, load_weights, get_num_frames_with_most_features
  - `tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10 (18 lines); hunks: -105,7 +105,7 @@ def batch_make_image_embeddings(; -124,11 +124,10 @@ def batch_make_image_embeddings(; symbols: batch_make_image_embeddings, batch_make_video_embeddings
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +78/-64
  - tests: `tests/models/decoder_only/vision_language/test_qwen2_vl.py` modified +8/-10
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_qwen2_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12604 - [VLM] Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/12604
- 状态/时间: merged / 2025-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1315/-52，可读 patch 1625 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/layers/rotary_embedding.py`, `tests/models/decoder_only/vision_language/test_models.py`；技术摘要: 覆盖「[VLM] Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/layers/rotary_embedding.py`, `tests/models/decoder_only/vision_language/test_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0 (1133 lines); hunks: -0,0 +1,1133; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs，涉及 `Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs`；`vllm/model_executor/layers/rotary_embedding.py` modified +34/-24 (58 lines); hunks: -27,6 +27,7; -772,8 +773,12 @@ def __init__(; symbols: __init__, forward, get_input_positions, get_input_positions_tensor，涉及 `__init__, forward, get_input_positions`；`tests/models/decoder_only/vision_language/test_models.py` modified +22/-0 (22 lines); hunks: -121,6 +121,8; -138,6 +140,26；`vllm/model_executor/models/qwen2_vl.py` modified +8/-8 (16 lines); hunks: -650,8 +650,8 @@ def load_weights(self, weights: Iterable[Tuple[str,; -683,26 +683,26 @@ def get_passthrough_data(self) -> Mapping[str, object]:; symbols: load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems, __init__，涉及 `load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0 (1133 lines); hunks: -0,0 +1,1133; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs
  - `vllm/model_executor/layers/rotary_embedding.py` modified +34/-24 (58 lines); hunks: -27,6 +27,7; -772,8 +773,12 @@ def __init__(; symbols: __init__, forward, get_input_positions, get_input_positions_tensor
  - `tests/models/decoder_only/vision_language/test_models.py` modified +22/-0 (22 lines); hunks: -121,6 +121,8; -138,6 +140,26
  - `vllm/model_executor/models/qwen2_vl.py` modified +8/-8 (16 lines); hunks: -650,8 +650,8 @@ def load_weights(self, weights: Iterable[Tuple[str,; -683,26 +683,26 @@ def get_passthrough_data(self) -> Mapping[str, object]:; symbols: load_weights, Qwen2EmbeddingItems, Qwen2VLEmbeddingItems, __init__
  - `docs/source/models/supported_models.md` modified +11/-0 (11 lines); hunks: -846,6 +846,13 @@ See this page for more information on how to use generativ; -880,6 +887,10 @@ The chat template for Pixtral-HF is incorrect (see [discuss...
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` added +1133/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +34/-24; `vllm/model_executor/models/qwen2_vl.py` modified +8/-8; `vllm/entrypoints/chat_utils.py` modified +2/-2
  - tests: `tests/models/decoder_only/vision_language/test_models.py` modified +22/-0; `tests/models/registry.py` modified +2/-0; `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs: `docs/source/models/supported_models.md` modified +11/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12944 - [Misc] Add qwen2.5-vl BNB support

- 链接: https://github.com/vllm-project/vllm/pull/12944
- 状态/时间: merged / 2025-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `4c8dd12ef347`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-30，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Add qwen2.5-vl BNB support」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Misc] Add qwen2.5-vl BNB support」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30 (59 lines); hunks: -40,7 +40,7; -207,11 +207,12 @@ def __init__(; symbols: __init__, split_qkv, forward，涉及 `__init__, split_qkv, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30 (59 lines); hunks: -40,7 +40,7; -207,11 +207,12 @@ def __init__(; symbols: __init__, split_qkv, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +29/-30
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13148 - [Bugfix] Fix num video tokens calculation for Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/13148
- 状态/时间: merged / 2025-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `985b4a2b1989`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix num video tokens calculation for Qwen2-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Fix num video tokens calculation for Qwen2-VL」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +5/-1 (6 lines); hunks: -800,7 +800,11 @@ def _get_vision_info(; symbols: _get_vision_info，涉及 `_get_vision_info`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-1 (6 lines); hunks: -800,7 +800,11 @@ def _get_vision_info(; symbols: _get_vision_info
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -800,7 +800,11 @@ def _get_vision_info(
-        grid_t = max(num_frames // temporal_patch_size, 1)
+        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
+        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
+        padded_num_frames = num_frames + num_frames % temporal_patch_size
+        grid_t = max(padded_num_frames // temporal_patch_size, 1)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13155 - [Misc] Qwen2.5-VL Optimization

- 链接: https://github.com/vllm-project/vllm/pull/13155
- 状态/时间: merged / 2025-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `02ed8a1fbe41`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+47/-51，可读 patch 152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Qwen2.5-VL Optimization」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Qwen2.5-VL Optimization」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunks: -45,6 +45,7; -271,8 +272,13 @@ def forward(; symbols: forward, Qwen2RMSNorm, __init__，涉及 `forward, Qwen2RMSNorm, __init__`；`vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunks: -226,11 +226,15 @@ def apply_rotary_emb_torch(x: torch.Tensor,; -336,20 +340,23 @@ def forward(; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward，涉及 `apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36 (61 lines); hunks: -45,6 +45,7; -271,8 +272,13 @@ def forward(; symbols: forward, Qwen2RMSNorm, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +22/-15 (37 lines); hunks: -226,11 +226,15 @@ def apply_rotary_emb_torch(x: torch.Tensor,; -336,20 +340,23 @@ def forward(; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +25/-36; `vllm/model_executor/models/qwen2_vl.py` modified +22/-15
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13286 - [Bugfix] Fix qwen2.5-vl image processor

- 链接: https://github.com/vllm-project/vllm/pull/13286
- 状态/时间: merged / 2025-02-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `7fdaaf48ef2a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-6，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix qwen2.5-vl image processor」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Fix qwen2.5-vl image processor」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5 (13 lines); hunks: -33,10 +33,11; -693,7 +694,8 @@ def get_hf_processor(; symbols: get_hf_processor, get_image_processor，涉及 `get_hf_processor, get_image_processor`；`vllm/model_executor/models/qwen2_vl.py` modified +9/-1 (10 lines); hunks: -31,7 +31,9; -759,7 +761,13 @@ def get_image_processor(; symbols: get_image_processor, get_supported_mm_limits，涉及 `get_image_processor, get_supported_mm_limits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5 (13 lines); hunks: -33,10 +33,11; -693,7 +694,8 @@ def get_hf_processor(; symbols: get_hf_processor, get_image_processor
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-1 (10 lines); hunks: -31,7 +31,9; -759,7 +761,13 @@ def get_image_processor(; symbols: get_image_processor, get_supported_mm_limits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +8/-5; `vllm/model_executor/models/qwen2_vl.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13533 - [Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/13533
- 状态/时间: merged / 2025-02-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `041e29471671`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+18/-2，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -689,7 +689,7 @@ def get_hf_processor(; symbols: get_hf_processor，涉及 `get_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: -689,7 +689,7 @@ def get_hf_processor(; symbols: get_hf_processor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -689,7 +689,7 @@ def get_hf_processor(
-        fps: Optional[float] = None,
+        fps: Optional[Union[float, List[float]]] = None,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/protocol.py`, `vllm/entrypoints/openai/serving_engine.py`, `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13968 - [Bugfix] Fix qwen2.5-vl overflow issue

- 链接: https://github.com/vllm-project/vllm/pull/13968
- 状态/时间: merged / 2025-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `78648758794e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+22/-15，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix qwen2.5-vl overflow issue」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Bugfix] Fix qwen2.5-vl overflow issue」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1 (7 lines); hunks: -63,7 +63,7; -641,6 +641,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1 (7 lines); hunks: -63,7 +63,7; -641,6 +641,11 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minicpmo.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14377 - [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync

- 链接: https://github.com/vllm-project/vllm/pull/14377
- 状态/时间: merged / 2025-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `70b808fe1a63`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+70/-24，可读 patch 186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +37/-12 (49 lines); hunks: -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -329,7 +331,6 @@ def forward(; symbols: split_qkv, forward, __init__，涉及 `split_qkv, forward, __init__`；`vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12 (45 lines); hunks: -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -285,7 +287,6 @@ def forward(; symbols: split_qkv, forward, __init__，涉及 `split_qkv, forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +37/-12 (49 lines); hunks: -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -329,7 +331,6 @@ def forward(; symbols: split_qkv, forward, __init__
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12 (45 lines); hunks: -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tens...; -285,7 +287,6 @@ def forward(; symbols: split_qkv, forward, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +37/-12; `vllm/model_executor/models/qwen2_5_vl.py` modified +33/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15200 - [Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation

- 链接: https://github.com/vllm-project/vllm/pull/15200
- 状态/时间: merged / 2025-03-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `1e508343e1ec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+37/-4，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4 (10 lines); hunks: -647,15 +647,17 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4 (10 lines); hunks: -647,15 +647,17 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-4
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/custom_inputs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15273 - [Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/15273
- 状态/时间: merged / 2025-03-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `47c712621316`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+35/-16，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10 (33 lines); hunks: -608,6 +608,17 @@ def get_window_index(self, grid_thw):; -645,25 +656,27 @@ def forward(; symbols: get_window_index, compute_attn_mask_seqlen, forward，涉及 `get_window_index, compute_attn_mask_seqlen, forward`；`vllm/model_executor/models/qwen2_vl.py` modified +12/-6 (18 lines); hunks: -617,6 +617,16 @@ def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:; -638,12 +648,8 @@ def forward(; symbols: rot_pos_emb, compute_attn_mask_seqlen, forward，涉及 `rot_pos_emb, compute_attn_mask_seqlen, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10 (33 lines); hunks: -608,6 +608,17 @@ def get_window_index(self, grid_thw):; -645,25 +656,27 @@ def forward(; symbols: get_window_index, compute_attn_mask_seqlen, forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-6 (18 lines); hunks: -617,6 +617,16 @@ def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:; -638,12 +648,8 @@ def forward(; symbols: rot_pos_emb, compute_attn_mask_seqlen, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +23/-10; `vllm/model_executor/models/qwen2_vl.py` modified +12/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15130 - [Model][VLM] Add Qwen2.5-Omni model support (thinker only)

- 链接: https://github.com/vllm-project/vllm/pull/15130
- 状态/时间: merged / 2025-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `2c1bd848a668`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+1852/-82，可读 patch 2363 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][VLM] Add Qwen2.5-Omni model support (thinker only)」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Model][VLM] Add Qwen2.5-Omni model support (thinker only)」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo，涉及 `_qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data`；`vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__，涉及 `forward, all_gather_interleave, Qwen2_5_VisionAttention`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__
- 关键代码摘录:

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
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0; `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16872 - [Model] Qwen2.5-Omni Cleanup

- 链接: https://github.com/vllm-project/vllm/pull/16872
- 状态/时间: merged / 2025-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `5124f5bf51b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-5，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Qwen2.5-Omni Cleanup」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Model] Qwen2.5-Omni Cleanup」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3 (3 lines); hunks: -518,9 +518,6 @@ def _apply_hf_processor_main(; symbols: _apply_hf_processor_main，涉及 `_apply_hf_processor_main`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3 (3 lines); hunks: -518,9 +518,6 @@ def _apply_hf_processor_main(; symbols: _apply_hf_processor_main
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -518,9 +518,6 @@ def _apply_hf_processor_main(
-        print(prompt)
-        print(hf_processor_mm_kwargs)
-        print(mm_items)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16907 - [Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni

- 链接: https://github.com/vllm-project/vllm/pull/16907
- 状态/时间: merged / 2025-04-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `26c0406555a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2 (3 lines); hunks: -198,9 +198,8 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave，涉及 `forward, all_gather_interleave`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2 (3 lines); hunks: -198,9 +198,8 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -198,9 +198,8 @@ def forward(self, x: torch.Tensor):
-    import torch.distributed as dist
-    dist.all_gather(gathered_tensors, local_tensor)
+    parallel_state.get_tp_group().all_gather(gathered_tensors, local_tensor)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16974 - [Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni

- 链接: https://github.com/vllm-project/vllm/pull/16974
- 状态/时间: merged / 2025-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `571e8dd65e2a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -198,8 +198,11 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave，涉及 `forward, all_gather_interleave`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -198,8 +198,11 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -198,8 +198,11 @@ def forward(self, x: torch.Tensor):
+    import torch.distributed as dist
-    parallel_state.get_tp_group().all_gather(gathered_tensors, local_tensor)
+    dist.all_gather(gathered_tensors,
+                    local_tensor,
+                    group=parallel_state.get_tp_group().device_group)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17301 - [Misc] Clean up Qwen2.5-Omni code

- 链接: https://github.com/vllm-project/vllm/pull/17301
- 状态/时间: merged / 2025-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `8b464d9660a8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+75/-94，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Clean up Qwen2.5-Omni code」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Misc] Clean up Qwen2.5-Omni code」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51 (59 lines); hunks: -51,11 +51,9; -279,46 +277,17 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, apply, _maybe_apply_prompt_updates, _get_prompt_updates，涉及 `_get_mm_fields_config, apply, _maybe_apply_prompt_updates`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51 (59 lines); hunks: -51,11 +51,9; -279,46 +277,17 @@ def _get_mm_fields_config(; symbols: _get_mm_fields_config, apply, _maybe_apply_prompt_updates, _get_prompt_updates
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +8/-51
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/multimodal/processing.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17726 - [Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE

- 链接: https://github.com/vllm-project/vllm/pull/17726
- 状态/时间: merged / 2025-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c3e9d5060e89`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-12，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7 (9 lines); hunks: -297,13 +297,8 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/qwen2_vl.py` modified +4/-5 (9 lines); hunks: -64,7 +64,7; -230,14 +230,13 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision，涉及 `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7 (9 lines); hunks: -297,13 +297,8 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +4/-5 (9 lines); hunks: -64,7 +64,7; -230,14 +230,13 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-7; `vllm/model_executor/models/qwen2_vl.py` modified +4/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17838 - [Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor

- 链接: https://github.com/vllm-project/vllm/pull/17838
- 状态/时间: merged / 2025-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `015815fe0141`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+15/-9，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3 (8 lines); hunks: -145,9 +145,11 @@ def get_hf_processor(; symbols: get_hf_processor，涉及 `get_hf_processor`；`vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3 (8 lines); hunks: -758,9 +758,11 @@ def get_hf_processor(; symbols: get_hf_processor，涉及 `get_hf_processor`；`vllm/model_executor/models/qwen2_vl.py` modified +5/-3 (8 lines); hunks: -759,9 +759,11 @@ def get_hf_processor(; symbols: get_hf_processor，涉及 `get_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3 (8 lines); hunks: -145,9 +145,11 @@ def get_hf_processor(; symbols: get_hf_processor
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3 (8 lines); hunks: -758,9 +758,11 @@ def get_hf_processor(; symbols: get_hf_processor
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-3 (8 lines); hunks: -759,9 +759,11 @@ def get_hf_processor(; symbols: get_hf_processor
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-3; `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-3; `vllm/model_executor/models/qwen2_vl.py` modified +5/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17973 - [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…

- 链接: https://github.com/vllm-project/vllm/pull/17973
- 状态/时间: merged / 2025-05-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `67da5720d4ed`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+121/-83，可读 patch 285 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83 (204 lines); hunks: -25,7 +25,7; -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:; symbols: __init__, dtype, device, rot_pos_emb，涉及 `__init__, dtype, device`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83 (204 lines); hunks: -25,7 +25,7; -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:; symbols: __init__, dtype, device, rot_pos_emb
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +121/-83
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19054 - [Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl

- 链接: https://github.com/vllm-project/vllm/pull/19054
- 状态/时间: merged / 2025-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `ec2dcd80bc17`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-8，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4 (13 lines); hunks: -823,10 +823,15 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Suppor...; symbols: Qwen2_5_VLForConditionalGeneration, __init__，涉及 `Qwen2_5_VLForConditionalGeneration, __init__`；`vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -1071,10 +1071,15 @@ class Qwen2VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2VLForConditionalGeneration, __init__，涉及 `Qwen2VLForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4 (13 lines); hunks: -823,10 +823,15 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Suppor...; symbols: Qwen2_5_VLForConditionalGeneration, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -1071,10 +1071,15 @@ class Qwen2VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2VLForConditionalGeneration, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +9/-4; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22069 - [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform.

- 链接: https://github.com/vllm-project/vllm/pull/22069
- 状态/时间: merged / 2025-08-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `d3a6f2120bb6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+64/-39，可读 patch 212 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform.」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5 (18 lines); hunks: -250,11 +250,15 @@ def __init__(; -301,10 +305,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen，涉及 `__init__, split_qkv, forward`；`vllm/model_executor/models/qwen2_vl.py` modified +13/-5 (18 lines); hunks: -274,10 +274,14 @@ def __init__(; -324,10 +328,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen，涉及 `__init__, split_qkv, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5 (18 lines); hunks: -250,11 +250,15 @@ def __init__(; -301,10 +305,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_vl.py` modified +13/-5 (18 lines); hunks: -274,10 +274,14 @@ def __init__(; -324,10 +328,13 @@ def forward(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +13/-5; `vllm/model_executor/models/qwen2_vl.py` modified +13/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/vision.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22184 - [Model] Switch to Fused RMS norm in Qwen2.5_VL model.

- 链接: https://github.com/vllm-project/vllm/pull/22184
- 状态/时间: merged / 2025-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `cbc8457b2663`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-7，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Switch to Fused RMS norm in Qwen2.5_VL model.」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7 (14 lines); hunks: -396,13 +396,13 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7 (14 lines); hunks: -396,13 +396,13 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-7
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23058 - [Bugfix] fix Qwen2.5-Omni processor output mapping

- 链接: https://github.com/vllm-project/vllm/pull/23058
- 状态/时间: merged / 2025-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `9f1c6422549d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-0，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix Qwen2.5-Omni processor output mapping」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Bugfix] fix Qwen2.5-Omni processor output mapping」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0 (5 lines); hunks: -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str...; symbols: _qwen2_5_omni_thinker_field_config，涉及 `_qwen2_5_omni_thinker_field_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0 (5 lines); hunks: -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str...; symbols: _qwen2_5_omni_thinker_field_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_5_omni_thinker.py
@@ -88,6 +88,11 @@ def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str, torch.Tensor]):
+    # vllm use `second_per_grid_ts` to compute multimodal rotary embedding
+    video_second_per_grid = hf_inputs.get("video_second_per_grid", None)
+    if video_second_per_grid is not None:
+        hf_inputs["second_per_grid_ts"] = video_second_per_grid
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23512 - [Bugfix] Fix Qwen2.5-VL quantized model weights loading

- 链接: https://github.com/vllm-project/vllm/pull/23512
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `a71e4765cc0c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen2.5-VL quantized model weights loading」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen2.5-VL quantized model weights loading」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -135,7 +135,7 @@ class Qwen2_5_VLVideoPixelInputs(TypedDict):; -852,6 +852,10 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration，涉及 `Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1 (6 lines); hunks: -135,7 +135,7 @@ class Qwen2_5_VLVideoPixelInputs(TypedDict):; -852,6 +852,10 @@ class Qwen2_5_VLForConditionalGeneration(nn.Module, Support...; symbols: Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLForConditionalGeneration
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24231 - [LoRA]: Add lora support to qwen-2.5-omni

- 链接: https://github.com/vllm-project/vllm/pull/24231
- 状态/时间: merged / 2025-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `c9f7081f9c84`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-3，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA]: Add lora support to qwen-2.5-omni」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[LoRA]: Add lora support to qwen-2.5-omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2 (15 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs, get_language_model，涉及 `_process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2 (15 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, _parse_and_validate_multimodal_inputs, get_language_model
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +13/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24420 - [Model] Enable BNB support for qwen2_5_omni_thinker

- 链接: https://github.com/vllm-project/vllm/pull/24420
- 状态/时间: merged / 2025-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `6f4a82f8b5a1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-2，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable BNB support for qwen2_5_omni_thinker」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Model] Enable BNB support for qwen2_5_omni_thinker」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2 (31 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str, load_weights，涉及 `_process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2 (31 lines); hunks: -41,6 +41,7; -66,7 +67,8; symbols: _process_video_input, Qwen2_5OmniThinkerForConditionalGeneration, get_placeholder_str, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +29/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24741 - [Models] Prevent CUDA sync in Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/24741
- 状态/时间: merged / 2025-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `b0d1213ac395`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Prevent CUDA sync in Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Models] Prevent CUDA sync in Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -64,6 +64,7; -737,7 +738,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation, forward，涉及 `compute_attn_mask_seqlen, invert_permutation, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1 (5 lines); hunks: -64,6 +64,7; -737,7 +738,7 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen, invert_permutation, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24727 - [Model] Support Qwen3-VL Model Series

- 链接: https://github.com/vllm-project/vllm/pull/24727
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `0f7acdd73ca6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+2084/-17，可读 patch 2262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Qwen3-VL Model Series」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model] Support Qwen3-VL Model Series」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunks: -0,0 +1,1478; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP，涉及 `Qwen3_VisionPatchEmbed, __init__, forward`；`vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__，涉及 `Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel`；`vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -83,7 +83,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` added +1478/-0 (1478 lines); hunks: -0,0 +1,1478; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
  - `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: Qwen3VLMoeProcessingInfo, get_hf_config, Qwen3MoeLLMModel, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -83,7 +83,7
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` added +1478/-0; `vllm/model_executor/models/qwen3_vl_moe.py` added +344/-0; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24955 - [MM Encoder] Apply DP ViT for Qwen3-VL model series

- 链接: https://github.com/vllm-project/vllm/pull/24955
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `3127274d022b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+77/-19，可读 patch 256 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM Encoder] Apply DP ViT for Qwen3-VL model series」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；技术摘要: 覆盖「[MM Encoder] Apply DP ViT for Qwen3-VL model series」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +75/-19 (94 lines); hunks: -126,20 +126,23 @@ def __init__(self,; -158,23 +161,27 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -315,12 +315,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +75/-19 (94 lines); hunks: -126,20 +126,23 @@ def __init__(self,; -158,23 +161,27 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0 (2 lines); hunks: -315,12 +315,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +75/-19; `vllm/model_executor/models/qwen3_vl_moe.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25055 - [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE

- 链接: https://github.com/vllm-project/vllm/pull/25055
- 状态/时间: merged / 2025-09-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+88/-46，可读 patch 263 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/kernels/core/test_mrope.py`；技术摘要: 覆盖「[Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/kernels/core/test_mrope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunks: -15,7 +15,7; -30,12 +30,14 @@ def _triton_qwen2vl_mrope_forward(; symbols: _triton_qwen2vl_mrope_forward, _triton_mrope_forward，涉及 `_triton_qwen2vl_mrope_forward, _triton_mrope_forward`；`tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunks: -1,9 +1,12; -15,6 +18,7 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_...; symbols: generate_test_data, unroll_model_tp_dict, MRoPETestInfo, test_mrope，涉及 `generate_test_data, unroll_model_tp_dict, MRoPETestInfo`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14 (36 lines); hunks: -15,7 +15,7; -30,12 +30,14 @@ def _triton_qwen2vl_mrope_forward(; symbols: _triton_qwen2vl_mrope_forward, _triton_mrope_forward
  - `tests/kernels/core/test_mrope.py` modified +66/-32 (98 lines); hunks: -1,9 +1,12; -15,6 +18,7 @@ def generate_test_data(num_tokens: int, num_q_heads: int, num_...; symbols: generate_test_data, unroll_model_tp_dict, MRoPETestInfo, test_mrope
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +22/-14
  - tests: `tests/kernels/core/test_mrope.py` modified +66/-32
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_mrope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25300 - [Bugfix] Fix Qwen3-VL-MoE weight loading for EP

- 链接: https://github.com/vllm-project/vllm/pull/25300
- 状态/时间: merged / 2025-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `be874c020196`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-5，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL-MoE weight loading for EP」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl_moe.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3-VL-MoE weight loading for EP」；主要实现面是 `vllm/model_executor/models/qwen3_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5 (12 lines); hunks: -122,9 +122,10 @@ def forward(; -133,9 +134,10 @@ def load_fused_expert_weights(self, name: str, params_dict:...; symbols: forward, load_fused_expert_weights, load_weights, __init__，涉及 `forward, load_fused_expert_weights, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5 (12 lines); hunks: -122,9 +122,10 @@ def forward(; -133,9 +134,10 @@ def load_fused_expert_weights(self, name: str, params_dict:...; symbols: forward, load_fused_expert_weights, load_weights, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl_moe.py` modified +7/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25337 - [MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`

- 链接: https://github.com/vllm-project/vllm/pull/25337
- 状态/时间: merged / 2025-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `30d08911f7cf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+60/-75，可读 patch 177 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +60/-75 (135 lines); hunks: -270,6 +270,7 @@ def __init__(; -377,82 +378,68 @@ def rot_pos_emb(self, grid_thw):; symbols: __init__, rot_pos_emb, fast_pos_embed_interpolate, compute_attn_mask_seqlen，涉及 `__init__, rot_pos_emb, fast_pos_embed_interpolate`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +60/-75 (135 lines); hunks: -270,6 +270,7 @@ def __init__(; -377,82 +378,68 @@ def rot_pos_emb(self, grid_thw):; symbols: __init__, rot_pos_emb, fast_pos_embed_interpolate, compute_attn_mask_seqlen
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +60/-75
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25347 - [Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`

- 链接: https://github.com/vllm-project/vllm/pull/25347
- 状态/时间: merged / 2025-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `af7dfb0d1a95`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-18，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +32/-18 (50 lines); hunks: -406,25 +406,39 @@ def fast_pos_embed_interpolate(self,; symbols: fast_pos_embed_interpolate，涉及 `fast_pos_embed_interpolate`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +32/-18 (50 lines); hunks: -406,25 +406,39 @@ def fast_pos_embed_interpolate(self,; symbols: fast_pos_embed_interpolate
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +32/-18
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25445 - [Model] Enable DP for ViT in Qwen2-VL

- 链接: https://github.com/vllm-project/vllm/pull/25445
- 状态/时间: merged / 2025-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c98be0a23276`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+59/-19，可读 patch 252 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable DP for ViT in Qwen2-VL」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model] Enable DP for ViT in Qwen2-VL」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +59/-19 (78 lines); hunks: -66,6 +66,7; -217,17 +218,20 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +59/-19 (78 lines); hunks: -66,6 +66,7; -217,17 +218,20 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +59/-19
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25646 - [Misc] Fix Qwen3-VL `video_grid_thw` typing

- 链接: https://github.com/vllm-project/vllm/pull/25646
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `7be9ffcd9f5c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Fix Qwen3-VL `video_grid_thw` typing」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Misc] Fix Qwen3-VL `video_grid_thw` typing」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -1249,7 +1249,7 @@ def _process_video_input(; symbols: _process_video_input，涉及 `_process_video_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: -1249,7 +1249,7 @@ def _process_video_input(; symbols: _process_video_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1249,7 +1249,7 @@ def _process_video_input(
-                                           grid_thw=grid_thw)
+                                           grid_thw=grid_thw_list)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25648 - [Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling

- 链接: https://github.com/vllm-project/vllm/pull/25648
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `17b4c6685ce6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+13/-1，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +12/-0 (12 lines); hunks: -715,6 +715,18 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor，涉及 `_get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor`；`vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +12/-0 (12 lines); hunks: -715,6 +715,18 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, get_dummy_processor_inputs, Qwen3VLMultiModalProcessor
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +12/-0; `vllm/model_executor/models/qwen2_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25788 - [Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL

- 链接: https://github.com/vllm-project/vllm/pull/25788
- 状态/时间: merged / 2025-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `c242c98031b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+75/-51，可读 patch 208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36 (73 lines); hunks: -274,6 +274,8 @@ def __init__(; -300,25 +302,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/qwen3_vl.py` modified +38/-15 (53 lines); hunks: -63,7 +63,7; -158,6 +158,8 @@ def __init__(; symbols: __init__, dtype，涉及 `__init__, dtype`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36 (73 lines); hunks: -274,6 +274,8 @@ def __init__(; -300,25 +302,8 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/qwen3_vl.py` modified +38/-15 (53 lines); hunks: -63,7 +63,7; -158,6 +158,8 @@ def __init__(; symbols: __init__, dtype
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +37/-36; `vllm/model_executor/models/qwen3_vl.py` modified +38/-15
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25557 - [VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling

- 链接: https://github.com/vllm-project/vllm/pull/25557
- 状态/时间: merged / 2025-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `0efd540dbc54`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+74/-9，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +65/-5 (70 lines); hunks: -33,11 +33,14; -85,6 +88,9; symbols: Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames, get_num_frames_with_most_features，涉及 `Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames`；`vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -79,7 +79,7; -932,6 +932,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens, _get_max_video_frames，涉及 `get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +65/-5 (70 lines); hunks: -33,11 +33,14; -85,6 +88,9; symbols: Qwen3_VisionPatchEmbed, _get_vision_info, _get_max_video_frames, get_num_frames_with_most_features
  - `vllm/model_executor/models/qwen2_vl.py` modified +9/-4 (13 lines); hunks: -79,7 +79,7; -932,6 +932,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens, get_image_size_with_most_features, get_max_image_tokens, _get_max_video_frames
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +65/-5; `vllm/model_executor/models/qwen2_vl.py` modified +9/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26000 - [MM] Add text-only mode for Qwen3-VL

- 链接: https://github.com/vllm-project/vllm/pull/26000
- 状态/时间: merged / 2025-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `66bca9b8bd0a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+45/-26，可读 patch 105 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM] Add text-only mode for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；技术摘要: 覆盖「[MM] Add text-only mode for Qwen3-VL」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +25/-14 (39 lines); hunks: -1125,14 +1125,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1148,11 +1151,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__, compute_logits, load_weights, get_mm_mapping，涉及 `__init__, compute_logits, load_weights`；`vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12 (32 lines); hunks: -319,13 +319,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -341,10 +345,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +25/-14 (39 lines); hunks: -1125,14 +1125,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1148,11 +1151,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__, compute_logits, load_weights, get_mm_mapping
  - `vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12 (32 lines); hunks: -319,13 +319,17 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -341,10 +345,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +25/-14; `vllm/model_executor/models/qwen3_vl_moe.py` modified +20/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26004 - [BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker

- 链接: https://github.com/vllm-project/vllm/pull/26004
- 状态/时间: merged / 2025-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `84d57342b66e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-3，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -324,9 +324,15 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates，涉及 `_maybe_apply_prompt_updates`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3 (12 lines); hunks: -324,9 +324,15 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +9/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24642 - [Qwen][ROCm] Flash Attention Rotary Embeddings

- 链接: https://github.com/vllm-project/vllm/pull/24642
- 状态/时间: merged / 2025-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `5e4a8223c644`, `dd96465fd744`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+28/-5，可读 patch 80 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen][ROCm] Flash Attention Rotary Embeddings」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Qwen][ROCm] Flash Attention Rotary Embeddings」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +5/-5 (10 lines); hunks: -50,6 +50,8; -63,7 +65,7; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision，涉及 `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +5/-5 (10 lines); hunks: -50,6 +50,8; -63,7 +65,7; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +5/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26104 - [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm

- 链接: https://github.com/vllm-project/vllm/pull/26104
- 状态/时间: merged / 2025-10-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+154/-141，可读 patch 553 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm」；主要实现面是 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl.py` modified +23/-26 (49 lines); hunks: -35,7 +35,8; -176,14 +177,18 @@ def __init__(; symbols: __init__, forward, compute_attn_mask_seqlen，涉及 `__init__, forward, compute_attn_mask_seqlen`；`vllm/model_executor/models/dots_ocr.py` modified +19/-22 (41 lines); hunks: -10,7 +10,8; -267,10 +268,12 @@ def __init__(self,; symbols: __init__, forward, compute_attn_mask_seqlen，涉及 `__init__, forward, compute_attn_mask_seqlen`；`vllm/model_executor/models/qwen2_vl.py` modified +18/-22 (40 lines); hunks: -42,7 +42,8; -319,18 +320,20 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17 (34 lines); hunks: -39,7 +39,8; -302,6 +303,11 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl.py` modified +23/-26 (49 lines); hunks: -35,7 +35,8; -176,14 +177,18 @@ def __init__(; symbols: __init__, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/dots_ocr.py` modified +19/-22 (41 lines); hunks: -10,7 +10,8; -267,10 +268,12 @@ def __init__(self,; symbols: __init__, forward, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_vl.py` modified +18/-22 (40 lines); hunks: -42,7 +42,8; -319,18 +320,20 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17 (34 lines); hunks: -39,7 +39,8; -302,6 +303,11 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/glm4_1v.py` modified +17/-14 (31 lines); hunks: -47,7 +47,8; -263,19 +264,26 @@ def __init__(; symbols: __init__, split_qkv, forward, compute_attn_mask_seqlen
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +23/-26; `vllm/model_executor/models/dots_ocr.py` modified +19/-22; `vllm/model_executor/models/qwen2_vl.py` modified +18/-22; `vllm/model_executor/models/qwen2_5_vl.py` modified +17/-17; `vllm/model_executor/models/glm4_1v.py` modified +17/-14; `vllm/model_executor/models/siglip2navit.py` modified +8/-14
- 验证与风险: runtime 路径改动集中在 `vllm/attention/layer.py`, `vllm/model_executor/models/dots_ocr.py`, `vllm/model_executor/models/ernie45_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26123 - [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642

- 链接: https://github.com/vllm-project/vllm/pull/26123
- 状态/时间: merged / 2025-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `dd96465fd744`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-4，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +2/-1 (3 lines); hunks: -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision，涉及 `apply_rotary_emb_torch, apply_rotary_pos_emb_vision`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +2/-1 (3 lines); hunks: -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,; symbols: apply_rotary_emb_torch, apply_rotary_pos_emb_vision
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen2_vl.py
@@ -276,7 +276,8 @@ def apply_rotary_emb_torch(x: torch.Tensor,
-    rotary_emb_function = dispatch_rotary_emb_function()
+    rotary_emb_function = dispatch_rotary_emb_function(
+        default=apply_rotary_emb_torch)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25550 - Add Qwen3-Omni moe thinker

- 链接: https://github.com/vllm-project/vllm/pull/25550
- 状态/时间: merged / 2025-10-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `19a9b169bf1b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+1795/-36，可读 patch 1940 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3-Omni moe thinker」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「Add Qwen3-Omni moe thinker」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunks: -0,0 +1,1409; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP，涉及 `Qwen3_VisionPatchEmbed, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0 (1409 lines); hunks: -0,0 +1,1409; symbols: Qwen3_VisionPatchEmbed, __init__, forward, Qwen3_VisionMLP
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` added +1409/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26608 - [MM] Move Qwen3Omni MRoPE impl to model file

- 链接: https://github.com/vllm-project/vllm/pull/26608
- 状态/时间: merged / 2025-10-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `ddaff2938e0b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+368/-387，可读 patch 859 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM] Move Qwen3Omni MRoPE impl to model file」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[MM] Move Qwen3Omni MRoPE impl to model file」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26 (355 lines); hunks: -72,7 +72,12; -96,7 +101,7; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__, get_supported_mm_limits，涉及 `_get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26 (355 lines); hunks: -72,7 +72,12; -96,7 +101,7; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, __init__, get_supported_mm_limits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +329/-26
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/vision.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26815 - [Bugfix] Fix qwen3-omni audio truncation issue

- 链接: https://github.com/vllm-project/vllm/pull/26815
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `8c851f6d044b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-2，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix qwen3-omni audio truncation issue」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix qwen3-omni audio truncation issue」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2 (18 lines); hunks: -30,7 +30,9; -711,11 +713,12 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length，涉及 `pad_to_hop_length`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2 (18 lines); hunks: -30,7 +30,9; -711,11 +713,12 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +16/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27104 - [bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True

- 链接: https://github.com/vllm-project/vllm/pull/27104
- 状态/时间: merged / 2025-10-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `4c91a28e301d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -735,9 +735,9 @@ def _get_video_second_idx(; symbols: _get_video_second_idx，涉及 `_get_video_second_idx`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -735,9 +735,9 @@ def _get_video_second_idx(; symbols: _get_video_second_idx
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -735,9 +735,9 @@ def _get_video_second_idx(
-            video_fps = sampled_fps if sampled_fps else video_processor.fps
+            sampled_fps = sampled_fps if sampled_fps else video_processor.fps
-            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
+            num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27190 - [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA

- 链接: https://github.com/vllm-project/vllm/pull/27190
- 状态/时间: merged / 2025-10-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+46/-12，可读 patch 106 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py`；技术摘要: 覆盖「[BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`, `vllm/attention/layer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -429,6 +429,12 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -462,6 +462,12 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/attention/layer.py` modified +29/-11 (40 lines); hunks: -47,6 +47,12; -96,18 +102,29 @@ def maybe_get_vit_flash_attn_backend(; symbols: maybe_get_vit_flash_attn_backend, forward，涉及 `maybe_get_vit_flash_attn_backend, forward`；`vllm/platforms/rocm.py` modified +5/-1 (6 lines); hunks: -205,12 +205,16 @@ class RocmPlatform(Platform):; symbols: RocmPlatform, get_vit_attn_backend，涉及 `RocmPlatform, get_vit_attn_backend`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -429,6 +429,12 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -462,6 +462,12 @@ def forward(; symbols: forward
  - `vllm/attention/layer.py` modified +29/-11 (40 lines); hunks: -47,6 +47,12; -96,18 +102,29 @@ def maybe_get_vit_flash_attn_backend(; symbols: maybe_get_vit_flash_attn_backend, forward
  - `vllm/platforms/rocm.py` modified +5/-1 (6 lines); hunks: -205,12 +205,16 @@ class RocmPlatform(Platform):; symbols: RocmPlatform, get_vit_attn_backend
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +6/-0; `vllm/model_executor/models/qwen2_vl.py` modified +6/-0; `vllm/attention/layer.py` modified +29/-11; `vllm/platforms/rocm.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/attention/layer.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27705 - [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes

- 链接: https://github.com/vllm-project/vllm/pull/27705
- 状态/时间: merged / 2025-10-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `0d8161b07504`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-16，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -223,8 +223,8 @@ def forward(; -488,12 +488,13 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen, get_placeholder_str，涉及 `forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen`；`vllm/model_executor/models/qwen3_vl.py` modified +7/-6 (13 lines); hunks: -231,8 +231,8 @@ def forward(; -512,15 +512,16 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen，涉及 `forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen`；`vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4 (6 lines); hunks: -836,10 +836,8 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen，涉及 `compute_attn_mask_seqlen`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -223,8 +223,8 @@ def forward(; -488,12 +488,13 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen, get_placeholder_str
  - `vllm/model_executor/models/qwen3_vl.py` modified +7/-6 (13 lines); hunks: -231,8 +231,8 @@ def forward(; -512,15 +512,16 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[i...; symbols: forward, fast_pos_embed_interpolate, compute_attn_mask_seqlen
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4 (6 lines); hunks: -836,10 +836,8 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6; `vllm/model_executor/models/qwen3_vl.py` modified +7/-6; `vllm/model_executor/models/qwen2_5_vl.py` modified +2/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27790 - [BugFix][VL] Fix FA selection on Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/27790
- 状态/时间: merged / 2025-10-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `e806178d2a9b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+20/-12，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][VL] Fix FA selection on Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「[BugFix][VL] Fix FA selection on Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11 (30 lines); hunks: -43,10 +43,7; -318,6 +315,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11 (30 lines); hunks: -43,10 +43,7; -318,6 +315,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +19/-11
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27920 - [Bugfix] Fix Qwen Omni audio inference

- 链接: https://github.com/vllm-project/vllm/pull/27920
- 状态/时间: merged / 2025-11-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `853a8eb53b89`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-10，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen Omni audio inference」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen Omni audio inference」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7 (9 lines); hunks: -130,6 +130,8 @@ class Qwen2_5OmniAudioFeatureInputs(TensorSchema):; -732,13 +734,6 @@ def _process_audio_input(; symbols: Qwen2_5OmniAudioFeatureInputs, _process_audio_input，涉及 `Qwen2_5OmniAudioFeatureInputs, _process_audio_input`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3 (3 lines); hunks: -99,7 +99,6; -1065,8 +1064,6 @@ def _process_audio_input(; symbols: _process_audio_input，涉及 `_process_audio_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7 (9 lines); hunks: -130,6 +130,8 @@ class Qwen2_5OmniAudioFeatureInputs(TensorSchema):; -732,13 +734,6 @@ def _process_audio_input(; symbols: Qwen2_5OmniAudioFeatureInputs, _process_audio_input
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3 (3 lines); hunks: -99,7 +99,6; -1065,8 +1064,6 @@ def _process_audio_input(; symbols: _process_audio_input
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +2/-7; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28271 - [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL

- 链接: https://github.com/vllm-project/vllm/pull/28271
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `48b8456ff992`, `bc5bd45c7d1a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+1/-42，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30 (30 lines); hunks: -291,25 +291,6 @@ def forward(self, x: torch.Tensor):; -383,21 +364,10 @@ def __init__(; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__，涉及 `forward, all_gather_interleave, Qwen2_5_VisionAttention`；`vllm/model_executor/models/qwen2_vl.py` modified +1/-12 (13 lines); hunks: -50,7 +50,7; -396,21 +396,10 @@ def __init__(; symbols: __init__, split_qkv，涉及 `__init__, split_qkv`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30 (30 lines); hunks: -291,25 +291,6 @@ def forward(self, x: torch.Tensor):; -383,21 +364,10 @@ def __init__(; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__
  - `vllm/model_executor/models/qwen2_vl.py` modified +1/-12 (13 lines); hunks: -50,7 +50,7; -396,21 +396,10 @@ def __init__(; symbols: __init__, split_qkv
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +0/-30; `vllm/model_executor/models/qwen2_vl.py` modified +1/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28663 - [Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure

- 链接: https://github.com/vllm-project/vllm/pull/28663
- 状态/时间: merged / 2025-11-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `cec275efcef6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-3，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +3/-1 (4 lines); hunks: -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +3/-1 (4 lines); hunks: -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1138,7 +1138,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.model = Qwen3LLMModel(vllm_config=vllm_config, prefix=prefix)
+        self.model = Qwen3LLMModel(
+            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
+        )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29232 - Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/29232
- 状态/时间: merged / 2025-11-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `d84d8f4429a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-1，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`；技术摘要: 覆盖「Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1 (17 lines); hunks: -230,6 +230,9 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; -244,6 +247,11 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; symbols: Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input, _postprocess_video_embeds_evs，涉及 `Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1 (17 lines); hunks: -230,6 +230,9 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; -244,6 +247,11 @@ class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):; symbols: Qwen2_5_VLVideoEmbeddingInputs, _parse_and_validate_video_input, _process_image_input, _postprocess_video_embeds_evs
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +16/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27721 - [Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.

- 链接: https://github.com/vllm-project/vllm/pull/27721
- 状态/时间: merged / 2025-11-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `839c6b7b72bc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+467/-59，可读 patch 631 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.」；主要实现面是 `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer，涉及 `print_input_ids, mock_qwen3_omni_config, mock_processor`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video，涉及 `_maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`；`vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders，涉及 `_maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders`。
- 代码 diff 细节:
  - `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/model_executor/test_qwen3_omni.py` added +221/-0
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25
- 验证与风险: diff 自带测试面 `tests/model_executor/test_qwen3_omni.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29896 - feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE

- 链接: https://github.com/vllm-project/vllm/pull/29896
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `a2b053dc858d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-0，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0 (23 lines); hunks: -62,6 +62,7; -1137,6 +1138,18 @@ class Qwen3OmniMoeThinkerForConditionalGeneration(; symbols: Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions, get_mm_mapping，涉及 `Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0 (23 lines); hunks: -62,6 +62,7; -1137,6 +1138,18 @@ class Qwen3OmniMoeThinkerForConditionalGeneration(; symbols: Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, get_mrope_input_positions, get_mm_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +23/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29974 - [ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni

- 链接: https://github.com/vllm-project/vllm/pull/29974
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `3f1b03739ae1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1 (5 lines); hunks: -494,7 +494,10 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen，涉及 `compute_attn_mask_seqlen`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1 (5 lines); hunks: -494,7 +494,10 @@ def compute_attn_mask_seqlen(; symbols: compute_attn_mask_seqlen
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -494,7 +494,10 @@ def compute_attn_mask_seqlen(
-        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
+        if self.attn_backend in {
+            AttentionBackendEnum.FLASH_ATTN,
+            AttentionBackendEnum.ROCM_AITER_FA,
+        }:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30037 - support qwen3-vl handle requests with embeddings

- 链接: https://github.com/vllm-project/vllm/pull/30037
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `6dcb07f676ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3-vl handle requests with embeddings」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「support qwen3-vl handle requests with embeddings」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +5/-2 (7 lines); hunks: -103,7 +103,7; -884,7 +884,10 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser, _call_hf_processor，涉及 `_get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +5/-2 (7 lines); hunks: -103,7 +103,7; -884,7 +884,10 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Qwen3VLMultiModalProcessor, _get_data_parser, _call_hf_processor
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30542 - [Bugfix] Revert Qwen2-VL part of change in #28271

- 链接: https://github.com/vllm-project/vllm/pull/30542
- 状态/时间: merged / 2025-12-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `48b8456ff992`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-1，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Revert Qwen2-VL part of change in #28271」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Bugfix] Revert Qwen2-VL part of change in #28271」；主要实现面是 `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_vl.py` modified +12/-1 (13 lines); hunks: -49,7 +49,7; -359,10 +359,21 @@ def __init__(; symbols: __init__, split_qkv，涉及 `__init__, split_qkv`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_vl.py` modified +12/-1 (13 lines); hunks: -49,7 +49,7; -359,10 +359,21 @@ def __init__(; symbols: __init__, split_qkv
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_vl.py` modified +12/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29752 - [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL

- 链接: https://github.com/vllm-project/vllm/pull/29752
- 状态/时间: merged / 2025-12-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `ae88aada38ec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+424/-12，可读 patch 539 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +424/-12 (436 lines); hunks: -67,12 +67,19; -92,6 +99,7; symbols: get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__, _process_video_input，涉及 `get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +424/-12 (436 lines); hunks: -67,12 +67,19; -92,6 +99,7; symbols: get_video_replacement_qwen3vl, Qwen3VLForConditionalGeneration, __init__, _process_video_input
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +424/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30883 - [Chore] Remove v0 dead code for Qwen2.5-omni

- 链接: https://github.com/vllm-project/vllm/pull/30883
- 状态/时间: merged / 2025-12-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `6fe588765287`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-22，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Remove v0 dead code for Qwen2.5-omni」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Chore] Remove v0 dead code for Qwen2.5-omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22 (22 lines); hunks: -70,7 +70,6; -1150,27 +1149,6 @@ def embed_input_ids(; symbols: embed_input_ids, embed_multimodal_v0, forward，涉及 `embed_input_ids, embed_multimodal_v0, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22 (22 lines); hunks: -70,7 +70,6; -1150,27 +1149,6 @@ def embed_input_ids(; symbols: embed_input_ids, embed_multimodal_v0, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-22
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31007 - [Qwen3-Omni] fixed _get_feat_extract_output_lengths function

- 链接: https://github.com/vllm-project/vllm/pull/31007
- 状态/时间: merged / 2025-12-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `bb24592d139b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-12，可读 patch 65 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Omni] fixed _get_feat_extract_output_lengths function」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12 (20 lines); hunks: -118,7 +118,7 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Te...; -921,13 +921,11 @@ def _get_prompt_updates(; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates, _process_audio_input，涉及 `_get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12 (20 lines); hunks: -118,7 +118,7 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Te...; -921,13 +921,11 @@ def _get_prompt_updates(; symbols: _get_feat_extract_output_lengths, Qwen3_VisionPatchEmbed, _get_prompt_updates, _process_audio_input
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31436 - Add GLM-ASR multimodal support

- 链接: https://github.com/vllm-project/vllm/pull/31436
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；关联提交 `d722e9e614f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+764/-2，可读 patch 833 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add GLM-ASR multimodal support」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；技术摘要: 覆盖「Add GLM-ASR multimodal support」；主要实现面是 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glmasr.py` added +545/-0 (545 lines); hunks: -0,0 +1,545; symbols: GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector, __init__，涉及 `GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector`；`vllm/model_executor/models/glmasr_utils.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: _calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts, _get_audio_output_lengths_from_lengths，涉及 `_calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glmasr.py` added +545/-0 (545 lines); hunks: -0,0 +1,545; symbols: GlmAsrFeatureInputs, GlmAsrEmbeddingInputs, GlmAsrMultiModalProjector, __init__
  - `vllm/model_executor/models/glmasr_utils.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: _calculate_conv_output_length, _as_list_chunk_counts, _normalize_chunk_counts, _get_audio_output_lengths_from_lengths
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glmasr.py` added +545/-0; `vllm/model_executor/models/glmasr_utils.py` added +165/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29255 - Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring

- 链接: https://github.com/vllm-project/vllm/pull/29255
- 状态/时间: merged / 2026-01-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `97a01308e9ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+312/-3，可读 patch 337 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring」；主要实现面是 `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count, TestQwen3OmniAudioSampleRatePreservation，涉及 `test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0 (25 lines); hunks: -751,6 +751,9 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np....; -760,6 +763,28 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np...; symbols: pad_to_hop_length，涉及 `pad_to_hop_length`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0 (285 lines); hunks: -0,0 +1,285; symbols: test_processor_with_audio_sample_rate, test_longer_audio_generates_more_tokens, get_token_count, TestQwen3OmniAudioSampleRatePreservation
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0 (25 lines); hunks: -751,6 +751,9 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np....; -760,6 +763,28 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np...; symbols: pad_to_hop_length
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_qwen3_omni.py` added +285/-0
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +25/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_qwen3_omni.py`, `tests/multimodal/test_processing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31790 - [Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)

- 链接: https://github.com/vllm-project/vllm/pull/31790
- 状态/时间: merged / 2026-01-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `2c1a4f2488da`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-6，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -750,18 +750,20 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length，涉及 `pad_to_hop_length`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6 (14 lines); hunks: -750,18 +750,20 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> n...; symbols: pad_to_hop_length
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +8/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31779 - [Refactor] GLM-ASR Modeling

- 链接: https://github.com/vllm-project/vllm/pull/31779
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；关联提交 `974138751bdb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+672/-41，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] GLM-ASR Modeling」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；技术摘要: 覆盖「[Refactor] GLM-ASR Modeling」；主要实现面是 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glmasr.py` modified +644/-36 (680 lines); hunks: -8,18 +8,22; -35,6 +39,8; symbols: GlmAsrEncoderRotaryEmbedding, __init__, forward, GlmAsrEncoderAttention，涉及 `GlmAsrEncoderRotaryEmbedding, __init__, forward`；`vllm/model_executor/models/glmasr_utils.py` modified +28/-5 (33 lines); hunks: -71,14 +71,37 @@ def _get_audio_output_lengths_for_tower(; symbols: _get_audio_output_lengths_for_tower, _flatten_audio_features_by_length，涉及 `_get_audio_output_lengths_for_tower, _flatten_audio_features_by_length`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glmasr.py` modified +644/-36 (680 lines); hunks: -8,18 +8,22; -35,6 +39,8; symbols: GlmAsrEncoderRotaryEmbedding, __init__, forward, GlmAsrEncoderAttention
  - `vllm/model_executor/models/glmasr_utils.py` modified +28/-5 (33 lines); hunks: -71,14 +71,37 @@ def _get_audio_output_lengths_for_tower(; symbols: _get_audio_output_lengths_for_tower, _flatten_audio_features_by_length
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +644/-36; `vllm/model_executor/models/glmasr_utils.py` modified +28/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31890 - [Models] Allow converting Qwen3-VL into Reranker model

- 链接: https://github.com/vllm-project/vllm/pull/31890
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/template/qwen3_vl_reranker.jinja`；关联提交 `eac3b96ec04d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+287/-13，可读 patch 415 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Allow converting Qwen3-VL into Reranker model」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `examples/pooling/score/template/qwen3_vl_reranker.jinja`；技术摘要: 覆盖「[Models] Allow converting Qwen3-VL into Reranker model」；主要实现面是 `examples/pooling/score/template/qwen3_vl_reranker.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23。
- 代码 diff 细节:
  - `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0 (23 lines); hunks: -0,0 +1,23
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/pooling/score/template/qwen3_vl_reranker.jinja` added +23/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32126 - [Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL

- 链接: https://github.com/vllm-project/vllm/pull/32126
- 状态/时间: merged / 2026-01-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `542a4059b2bb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+113/-190，可读 patch 377 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；技术摘要: 覆盖「[Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL」；主要实现面是 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95 (152 lines); hunks: -26,11 +26,12; -1044,121 +1045,82 @@ class Qwen2_5_VLForConditionalGeneration(; symbols: Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str，涉及 `Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions`；`vllm/model_executor/models/qwen2_vl.py` modified +56/-95 (151 lines); hunks: -26,7 +26,7; -1137,121 +1137,82 @@ class Qwen2VLForConditionalGeneration(; symbols: Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str，涉及 `Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95 (152 lines); hunks: -26,11 +26,12; -1044,121 +1045,82 @@ class Qwen2_5_VLForConditionalGeneration(; symbols: Qwen2_5_VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str
  - `vllm/model_executor/models/qwen2_vl.py` modified +56/-95 (151 lines); hunks: -26,7 +26,7; -1137,121 +1137,82 @@ class Qwen2VLForConditionalGeneration(; symbols: Qwen2VLForConditionalGeneration, iter_mm_grid_thw, get_mrope_input_positions, get_placeholder_str
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +57/-95; `vllm/model_executor/models/qwen2_vl.py` modified +56/-95
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32167 - [Model] Re-implement Qwen3Omni Audio Encoder

- 链接: https://github.com/vllm-project/vllm/pull/32167
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `b8199f604931`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+428/-29，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Re-implement Qwen3Omni Audio Encoder」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Model] Re-implement Qwen3Omni Audio Encoder」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29 (457 lines); hunks: -31,29 +31,34; -104,11 +109,6; symbols: _get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__, forward，涉及 `_get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29 (457 lines); hunks: -31,29 +31,34; -104,11 +109,6; symbols: _get_feat_extract_output_lengths, SinusoidsPositionEmbedding, __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +428/-29
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32540 - [Bugfix] Fix GLM-ASR audio encoder RoPE dim

- 链接: https://github.com/vllm-project/vllm/pull/32540
- 状态/时间: merged / 2026-01-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`；关联提交 `38bf2ffb21d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+40/-30，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM-ASR audio encoder RoPE dim」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glmasr.py`；技术摘要: 覆盖「[Bugfix] Fix GLM-ASR audio encoder RoPE dim」；主要实现面是 `vllm/model_executor/models/glmasr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glmasr.py` modified +12/-2 (14 lines); hunks: -181,6 +181,12 @@ def __init__(; -226,8 +232,12 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glmasr.py` modified +12/-2 (14 lines); hunks: -181,6 +181,12 @@ def __init__(; -226,8 +232,12 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glmasr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32772 - [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni

- 链接: https://github.com/vllm-project/vllm/pull/32772
- 状态/时间: merged / 2026-01-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `a698e8e7ad4b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+386/-201，可读 patch 689 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates，涉及 `_get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33010 - [Model] Use mm_position to compute mrope positions for Qwen3-Omni

- 链接: https://github.com/vllm-project/vllm/pull/33010
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `6ca2c91b9663`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+293/-298，可读 patch 675 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use mm_position to compute mrope positions for Qwen3-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Model] Use mm_position to compute mrope positions for Qwen3-Omni」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping，涉及 `load_weights, get_mrope_input_positions, _compute_audio_token_count`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33312 - [Models] Qwen3-ASR

- 链接: https://github.com/vllm-project/vllm/pull/33312
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`；关联提交 `8b3f0a99dd50`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1269/-0，可读 patch 1335 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Qwen3-ASR」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`；技术摘要: 覆盖「[Models] Qwen3-ASR」；主要实现面是 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunks: -0,0 +1,567; symbols: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor，涉及 `_get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config`；`vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig，涉及 `Qwen3ASRAudioEncoderConfig, to, __init__`；`vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, __init__，涉及 `Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr.py` added +567/-0 (567 lines); hunks: -0,0 +1,567; symbols: _get_feat_extract_output_lengths, Qwen3ASRProcessingInfo, get_hf_config, get_hf_processor
  - `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: Qwen3ASRAudioEncoderConfig, to, __init__, Qwen3ASRTextConfig
  - `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0 (231 lines); hunks: -0,0 +1,231; symbols: Qwen3ASRProcessorKwargs, _get_feat_extract_output_lengths, Qwen3ASRProcessor, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` added +567/-0; `vllm/transformers_utils/configs/qwen3_asr.py` added +436/-0; `vllm/transformers_utils/processors/qwen3_asr.py` added +231/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33410 - [Bugfix] Fix `Qwen3ASR` language asr tag in output

- 链接: https://github.com/vllm-project/vllm/pull/33410
- 状态/时间: merged / 2026-01-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`；关联提交 `e77f162cf59d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+42/-2，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix `Qwen3ASR` language asr tag in output」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_asr.py`；技术摘要: 覆盖「[Bugfix] Fix `Qwen3ASR` language asr tag in output」；主要实现面是 `vllm/model_executor/models/qwen3_asr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr.py` modified +20/-1 (21 lines); hunks: -90,6 +90,7; -556,7 +557,7 @@ def get_generation_prompt(; symbols: _get_feat_extract_output_lengths, get_generation_prompt, post_process_output，涉及 `_get_feat_extract_output_lengths, get_generation_prompt, post_process_output`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr.py` modified +20/-1 (21 lines); hunks: -90,6 +90,7; -556,7 +557,7 @@ def get_generation_prompt(; symbols: _get_feat_extract_output_lengths, get_generation_prompt, post_process_output
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +20/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/translations/speech_to_text.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/qwen3_asr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33077 - [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM

- 链接: https://github.com/vllm-project/vllm/pull/33077
- 状态/时间: merged / 2026-02-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `cd86fff38fee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+31/-7，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7 (38 lines); hunks: -907,13 +907,37 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7 (38 lines); hunks: -907,13 +907,37 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +31/-7
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33644 - [Bugfix] fix qwen3-asr response error

- 链接: https://github.com/vllm-project/vllm/pull/33644
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`；关联提交 `ceab70c89d2b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-6，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix qwen3-asr response error」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_asr.py`；技术摘要: 覆盖「[Bugfix] fix qwen3-asr response error」；主要实现面是 `vllm/model_executor/models/qwen3_asr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr.py` modified +7/-6 (13 lines); hunks: -125,6 +125,13 @@ def get_feature_extractor(self, **kwargs: object) -> Whispe...; -194,12 +201,6 @@ def _parse_audio_data(; symbols: get_feature_extractor, get_supported_mm_limits, get_data_parser, Qwen3ASRDummyInputsBuilder，涉及 `get_feature_extractor, get_supported_mm_limits, get_data_parser`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr.py` modified +7/-6 (13 lines); hunks: -125,6 +125,13 @@ def get_feature_extractor(self, **kwargs: object) -> Whispe...; -194,12 +201,6 @@ def _parse_audio_data(; symbols: get_feature_extractor, get_supported_mm_limits, get_data_parser, Qwen3ASRDummyInputsBuilder
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +7/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_asr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33605 - [Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni

- 链接: https://github.com/vllm-project/vllm/pull/33605
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `f8516a1ab95f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+172/-12，可读 patch 247 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3 (126 lines); hunks: -113,6 +113,95; -1286,17 +1375,48 @@ def embed_input_ids(; symbols: check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs, embed_input_ids，涉及 `check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9 (58 lines); hunks: -92,6 +92,8; -1780,6 +1782,19 @@ def embed_input_ids(; symbols: embed_input_ids，涉及 `embed_input_ids`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3 (126 lines); hunks: -113,6 +113,95; -1286,17 +1375,48 @@ def embed_input_ids(; symbols: check_interleaved_audio_video, merge_interleaved_embeddings, Qwen2_5OmniAudioFeatureInputs, embed_input_ids
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9 (58 lines); hunks: -92,6 +92,8; -1780,6 +1782,19 @@ def embed_input_ids(; symbols: embed_input_ids
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +123/-3; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +49/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29828 - [Model] Add transcription support for Qwen3-Omni

- 链接: https://github.com/vllm-project/vllm/pull/29828
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `535de06cb1d9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+104/-2，可读 patch 177 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add transcription support for Qwen3-Omni」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Model] Add transcription support for Qwen3-Omni」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2 (104 lines); hunks: -24,7 +24,7; -48,8 +48,9; symbols: _get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, _compute_interleaved_positions，涉及 `_get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2 (104 lines); hunks: -24,7 +24,7; -48,8 +48,9; symbols: _get_feat_extract_output_lengths, Qwen3OmniMoeThinkerForConditionalGeneration, get_placeholder_str, _compute_interleaved_positions
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +102/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34613 - [Realtime] Add Qwen3-ASR realtime streaming support

- 链接: https://github.com/vllm-project/vllm/pull/34613
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr_realtime.py`；关联提交 `11be2c74dc1e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+256/-1，可读 patch 286 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Realtime] Add Qwen3-ASR realtime streaming support」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_asr_realtime.py`；技术摘要: 覆盖「[Realtime] Add Qwen3-ASR realtime streaming support」；主要实现面是 `vllm/model_executor/models/qwen3_asr_realtime.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: Qwen3ASRRealtimeBuffer, __init__, write_audio, read_audio，涉及 `Qwen3ASRRealtimeBuffer, __init__, write_audio`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: Qwen3ASRRealtimeBuffer, __init__, write_audio, read_audio
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr_realtime.py` added +239/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35368 - [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression

- 链接: https://github.com/vllm-project/vllm/pull/35368
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `c0615a296d44`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+379/-21，可读 patch 437 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression」；主要实现面是 `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo, test_non_interleaved_audio_then_video，涉及 `make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo`；`vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16 (30 lines); hunks: -1376,23 +1376,12 @@ def embed_input_ids(; -1403,6 +1392,12 @@ def embed_input_ids(; symbols: embed_input_ids, forward，涉及 `embed_input_ids, forward`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5 (12 lines); hunks: -1904,15 +1904,17 @@ def embed_input_ids(; symbols: embed_input_ids, forward，涉及 `embed_input_ids, forward`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0 (358 lines); hunks: -0,0 +1,358; symbols: make_token_seq, make_interleaved_seq, TestCheckInterleavedAudioVideo, test_non_interleaved_audio_then_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16 (30 lines); hunks: -1376,23 +1376,12 @@ def embed_input_ids(; -1403,6 +1392,12 @@ def embed_input_ids(; symbols: embed_input_ids, forward
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5 (12 lines); hunks: -1904,15 +1904,17 @@ def embed_input_ids(; symbols: embed_input_ids, forward
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py` added +358/-0
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +14/-16; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +7/-5
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35741 - [Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker

- 链接: https://github.com/vllm-project/vllm/pull/35741
- 状态/时间: merged / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `fa6a6be51978`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-0，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0 (17 lines); hunks: -648,13 +648,15 @@ def forward(; -975,6 +977,20 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0 (17 lines); hunks: -648,13 +648,15 @@ def forward(; -975,6 +977,20 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +17/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35869 - [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile

- 链接: https://github.com/vllm-project/vllm/pull/35869
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr_realtime.py`；关联提交 `36bf2131816e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_asr_realtime.py`；技术摘要: 覆盖「[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile」；主要实现面是 `vllm/model_executor/models/qwen3_asr_realtime.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2 (2 lines); hunks: -22,7 +22,6; -177,7 +176,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration，涉及 `_maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2 (2 lines); hunks: -22,7 +22,6; -177,7 +176,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, Qwen3ASRRealtimeGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_asr_realtime.py
@@ -22,7 +22,6 @@
-from vllm.compilation.decorators import support_torch_compile
@@ -177,7 +176,6 @@ def _maybe_apply_prompt_updates(
-@support_torch_compile
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr_realtime.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_asr_realtime.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36140 - [Bugfix] Fix Qwen-VL tokenizer implementation

- 链接: https://github.com/vllm-project/vllm/pull/36140
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen_vl.py`, `vllm/tokenizers/qwen_vl.py`；关联提交 `719634815791`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+118/-66，可读 patch 271 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen-VL tokenizer implementation」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/tokenizers/qwen_vl.py`, `vllm/model_executor/models/qwen_vl.py`, `vllm/renderers/qwen_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen-VL tokenizer implementation」；主要实现面是 `vllm/tokenizers/qwen_vl.py`, `vllm/model_executor/models/qwen_vl.py`, `vllm/renderers/qwen_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/qwen_vl.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode，涉及 `get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize`；`vllm/model_executor/models/qwen_vl.py` modified +2/-64 (66 lines); hunks: -6,11 +6,9; -436,60 +434,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad, tokenize，涉及 `__init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad`；`vllm/renderers/qwen_vl.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: QwenVLRenderer, from_config，涉及 `QwenVLRenderer, from_config`。
- 代码 diff 细节:
  - `vllm/tokenizers/qwen_vl.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: get_qwen_vl_tokenizer, TokenizerWithoutImagePad, tokenize, _decode
  - `vllm/model_executor/models/qwen_vl.py` modified +2/-64 (66 lines); hunks: -6,11 +6,9; -436,60 +434,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, _get_tokenizer_without_image_pad, TokenizerWithoutImagePad, tokenize
  - `vllm/renderers/qwen_vl.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: QwenVLRenderer, from_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tokenizers/qwen_vl.py` added +67/-0; `vllm/model_executor/models/qwen_vl.py` modified +2/-64; `vllm/renderers/qwen_vl.py` added +29/-0
- 验证与风险: diff 自带测试面 `tests/tokenizers_/test_basic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36108 - refactor funasr model.

- 链接: https://github.com/vllm-project/vllm/pull/36108
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `3ee68590c7fa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-57，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor funasr model.」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「refactor funasr model.」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1 (2 lines); hunks: -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModal...; symbols: embed_multimodal，涉及 `embed_multimodal`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1 (2 lines); hunks: -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModal...; symbols: embed_multimodal
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1794,7 +1794,7 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
-        # tensor correspoending to a multimodal data item (image or video).
+        # tensor corresponding to a multimodal data item (image or video).
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/funasr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/transformers_utils/processors/funasr_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35994 - [BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0

- 链接: https://github.com/vllm-project/vllm/pull/35994
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `e998fa76b99a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+86/-0，可读 patch 107 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`；技术摘要: 覆盖「[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0 (33 lines); hunks: -353,6 +353,39 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2_5OmniThinkerDummyInputsBuilder，涉及 `get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0 (33 lines); hunks: -1163,6 +1163,39 @@ def get_feature_extractor(self, **kwargs: object):; symbols: get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item，涉及 `get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item`；`vllm/model_executor/models/qwen2_audio.py` modified +20/-0 (20 lines); hunks: -179,6 +179,26 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2AudioDummyInputsBuilder，涉及 `get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0 (33 lines); hunks: -353,6 +353,39 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2_5OmniThinkerDummyInputsBuilder
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0 (33 lines); hunks: -1163,6 +1163,39 @@ def get_feature_extractor(self, **kwargs: object):; symbols: get_feature_extractor, get_supported_mm_limits, get_mm_max_tokens_per_item
  - `vllm/model_executor/models/qwen2_audio.py` modified +20/-0 (20 lines); hunks: -179,6 +179,26 @@ def get_target_channels(self) -> int:; symbols: get_target_channels, get_supported_mm_limits, get_mm_max_tokens_per_item, Qwen2AudioDummyInputsBuilder
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +33/-0; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +33/-0; `vllm/model_executor/models/qwen2_audio.py` modified +20/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36319 - Support online use_audio_in_video

- 链接: https://github.com/vllm-project/vllm/pull/36319
- 状态/时间: merged / 2026-03-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+152/-10，可读 patch 403 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support online use_audio_in_video」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/media/audio.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`；技术摘要: 覆盖「Support online use_audio_in_video」；主要实现面是 `vllm/entrypoints/chat_utils.py`, `vllm/multimodal/media/audio.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/chat_utils.py` modified +54/-8 (62 lines); hunks: -564,7 +564,9 @@ def add(self, modality: ModalityStr, item: _T) -> str | None:; -690,8 +692,10 @@ def resolve_items(; symbols: add, create_parser, resolve_items，涉及 `add, create_parser, resolve_items`；`vllm/multimodal/media/audio.py` modified +31/-0 (31 lines); hunks: -82,6 +82,35 @@ def extract_audio_from_video_bytes(; -100,6 +129,8 @@ def __init__(self, **kwargs) -> None:; symbols: extract_audio_from_video_bytes, is_video, AudioMediaIO, __init__，涉及 `extract_audio_from_video_bytes, is_video, AudioMediaIO`；`vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1 (16 lines); hunks: -78,7 +78,11; -811,6 +815,16 @@ def get_replacement_qwen2_use_audio_in_video(item_idx: int):; symbols: get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main，涉及 `get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main`；`vllm/entrypoints/openai/engine/serving.py` modified +1/-0 (1 lines); hunks: -908,6 +908,7 @@ async def _preprocess_chat(; symbols: _preprocess_chat，涉及 `_preprocess_chat`。
- 代码 diff 细节:
  - `vllm/entrypoints/chat_utils.py` modified +54/-8 (62 lines); hunks: -564,7 +564,9 @@ def add(self, modality: ModalityStr, item: _T) -> str | None:; -690,8 +692,10 @@ def resolve_items(; symbols: add, create_parser, resolve_items
  - `vllm/multimodal/media/audio.py` modified +31/-0 (31 lines); hunks: -82,6 +82,35 @@ def extract_audio_from_video_bytes(; -100,6 +129,8 @@ def __init__(self, **kwargs) -> None:; symbols: extract_audio_from_video_bytes, is_video, AudioMediaIO, __init__
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1 (16 lines); hunks: -78,7 +78,11; -811,6 +815,16 @@ def get_replacement_qwen2_use_audio_in_video(item_idx: int):; symbols: get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main
  - `vllm/entrypoints/openai/engine/serving.py` modified +1/-0 (1 lines); hunks: -908,6 +908,7 @@ async def _preprocess_chat(; symbols: _preprocess_chat
  - `vllm/renderers/params.py` modified +41/-1 (42 lines); hunks: -40,6 +40,34 @@ def merge_kwargs(; -56,12 +84,20 @@ class ChatParams:; symbols: merge_kwargs, recursively_merge_kwargs, ChatParams, with_defaults
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/entrypoints/chat_utils.py` modified +54/-8; `vllm/multimodal/media/audio.py` modified +31/-0; `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +15/-1; `vllm/entrypoints/openai/engine/serving.py` modified +1/-0; `vllm/renderers/params.py` modified +41/-1; `vllm/renderers/deepseek_v32.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/chat_utils.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36136 - [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps

- 链接: https://github.com/vllm-project/vllm/pull/36136
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `724759684cd9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+116/-4，可读 patch 150 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps」；主要实现面是 `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0 (94 lines); hunks: -0,0 +1,94; symbols: _build_video_mm_data, test_processor_num_frames_timestamp，涉及 `_build_video_mm_data, test_processor_num_frames_timestamp`；`vllm/model_executor/models/qwen3_vl.py` modified +22/-4 (26 lines); hunks: -768,6 +768,7 @@ def _get_video_second_idx(; -782,11 +783,20 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, _call_hf_processor, default，涉及 `_get_video_second_idx, _call_hf_processor, default`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0 (94 lines); hunks: -0,0 +1,94; symbols: _build_video_mm_data, test_processor_num_frames_timestamp
  - `vllm/model_executor/models/qwen3_vl.py` modified +22/-4 (26 lines); hunks: -768,6 +768,7 @@ def _get_video_second_idx(; -782,11 +783,20 @@ def _get_video_second_idx(; symbols: _get_video_second_idx, _call_hf_processor, default
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_qwen3_vl.py` added +94/-0
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +22/-4
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_qwen3_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36800 - [Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request

- 链接: https://github.com/vllm-project/vllm/pull/36800
- 状态/时间: merged / 2026-03-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `abf61aaa8ef2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+128/-12，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12 (23 lines); hunks: -80,8 +80,6; -609,6 +607,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main，涉及 `_maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0 (11 lines); hunks: -1326,6 +1326,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates，涉及 `_maybe_apply_prompt_updates`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12 (23 lines); hunks: -80,8 +80,6; -609,6 +607,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_use_audio_in_video, _cached_apply_hf_processor, _apply_hf_processor_main
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0 (11 lines); hunks: -1326,6 +1326,17 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +11/-12; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +11/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_audio_in_video.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37147 - [Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs

- 链接: https://github.com/vllm-project/vllm/pull/37147
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `912fbe9555f9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+117/-17，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs」；主要实现面是 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3 (4 lines); hunks: -774,9 +774,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video，涉及 `get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3 (4 lines); hunks: -1489,9 +1489,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality:...; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video，涉及 `get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3 (4 lines); hunks: -774,9 +774,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality: str):; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3 (4 lines); hunks: -1489,9 +1489,7 @@ def get_replacement_qwen2_vision(item_idx: int, modality:...; symbols: get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +1/-3; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +1/-3
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/test_audio_in_video.py`, `tests/models/multimodal/processing/test_audio_in_video.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37183 - Remove unused EVS functions in qwen3_vl.py

- 链接: https://github.com/vllm-project/vllm/pull/37183
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `43a73f853bac`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-101，可读 patch 108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove unused EVS functions in qwen3_vl.py」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「Remove unused EVS functions in qwen3_vl.py」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +0/-101 (101 lines); hunks: -1960,107 +1960,6 @@ def _iter_mm_grid_hw(; symbols: _iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask, _get_actual_frame_token_counts，涉及 `_iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +0/-101 (101 lines); hunks: -1960,107 +1960,6 @@ def _iter_mm_grid_hw(; symbols: _iter_mm_grid_hw, _get_evs_mask_segments, _extract_frame_offsets_from_mask, _get_actual_frame_token_counts
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +0/-101
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37439 - [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation

- 链接: https://github.com/vllm-project/vllm/pull/37439
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `738d0a281fab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -767,7 +767,7 @@ def _get_video_second_idx(; -806,7 +806,7 @@ def _get_video_second_idx(; symbols: _get_video_second_idx，涉及 `_get_video_second_idx`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -767,7 +767,7 @@ def _get_video_second_idx(; -806,7 +806,7 @@ def _get_video_second_idx(; symbols: _get_video_second_idx
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -767,7 +767,7 @@ def _get_video_second_idx(
-        merge_size = video_processor.merge_size
+        temporal_patch_size = video_processor.temporal_patch_size
@@ -806,7 +806,7 @@ def _get_video_second_idx(
-        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
+        timestamps = self._calculate_timestamps(indices, video_fps, temporal_patch_size)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35963 - [Feature] ViT Full CUDA Graph

- 链接: https://github.com/vllm-project/vllm/pull/35963
- 状态/时间: merged / 2026-03-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1584/-31，可读 patch 1731 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] ViT Full CUDA Graph」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py`；技术摘要: 覆盖「[Feature] ViT Full CUDA Graph」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/v1/worker/gpu/mm/encoder_cudagraph.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +270/-30 (300 lines); hunks: -103,6 +103,7; -528,54 +529,120 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[...; symbols: fast_pos_embed_interpolate, forward, prepare_encoder_metadata, __init__，涉及 `fast_pos_embed_interpolate, forward, prepare_encoder_metadata`；`vllm/model_executor/models/interfaces.py` modified +141/-0 (141 lines); hunks: -13,6 +13,7; -46,6 +47,11; symbols: supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config, get_encoder_cudagraph_budget_range，涉及 `supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config`；`vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0 (576 lines); hunks: -0,0 +1,576; symbols: BudgetGraphMetadata, EncoderCudaGraphManager, __init__, _generate_budgets，涉及 `BudgetGraphMetadata, EncoderCudaGraphManager, __init__`；`tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: _make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2, test_max_not_power_of_2，涉及 `_make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +270/-30 (300 lines); hunks: -103,6 +103,7; -528,54 +529,120 @@ def fast_pos_embed_interpolate(self, grid_thw: list[list[...; symbols: fast_pos_embed_interpolate, forward, prepare_encoder_metadata, __init__
  - `vllm/model_executor/models/interfaces.py` modified +141/-0 (141 lines); hunks: -13,6 +13,7; -46,6 +47,11; symbols: supports_xdrope, SupportsEncoderCudaGraph, get_encoder_cudagraph_config, get_encoder_cudagraph_budget_range
  - `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0 (576 lines); hunks: -0,0 +1,576; symbols: BudgetGraphMetadata, EncoderCudaGraphManager, __init__, _generate_budgets
  - `tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: _make_manager_with_budgets, TestGenerateBudgets, test_exact_powers_of_2, test_max_not_power_of_2
  - `vllm/v1/worker/gpu/mm/encoder_cudagraph_defs.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: EncoderCudaGraphConfig, EncoderCudaGraphCaptureInputs, EncoderCudaGraphReplayBuffers
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +270/-30; `vllm/model_executor/models/interfaces.py` modified +141/-0; `vllm/v1/worker/gpu/mm/encoder_cudagraph.py` added +576/-0; `vllm/v1/worker/gpu/mm/encoder_cudagraph_defs.py` added +66/-0; `vllm/v1/worker/gpu_model_runner.py` modified +48/-1; `vllm/config/compilation.py` modified +32/-0
  - tests: `tests/v1/cudagraph/test_encoder_cudagraph.py` added +451/-0
- 验证与风险: diff 自带测试面 `tests/v1/cudagraph/test_encoder_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37247 - [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration

- 链接: https://github.com/vllm-project/vllm/pull/37247
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `8d0f908b98cd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+63/-5，可读 patch 126 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Implement LoRA support for Qwen3ASRForConditionalGeneration」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；技术摘要: 覆盖「[Model] Implement LoRA support for Qwen3ASRForConditionalGeneration」；主要实现面是 `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_asr.py` modified +26/-0 (26 lines); hunks: -37,6 +37,7; -266,7 +267,21 @@ class Qwen3ASRForConditionalGeneration(; symbols: Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens, get_speech_to_text_config，涉及 `Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3 (25 lines); hunks: -57,6 +57,7; -357,7 +358,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_asr.py` modified +26/-0 (26 lines); hunks: -37,6 +37,7; -266,7 +267,21 @@ class Qwen3ASRForConditionalGeneration(; symbols: Qwen3ASRForConditionalGeneration, get_mm_mapping, get_num_mm_encoder_tokens, get_speech_to_text_config
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3 (25 lines); hunks: -57,6 +57,7; -357,7 +358,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_asr.py` modified +26/-0; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +22/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38061 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference

- 链接: https://github.com/vllm-project/vllm/pull/38061
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `80118853f42a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+583/-68，可读 patch 1042 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference」；主要实现面是 `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_vl.py` modified +138/-42 (180 lines); hunks: -99,6 +99,7; -689,6 +690,7 @@ def prepare_encoder_metadata(; symbols: prepare_encoder_metadata, get_encoder_cudagraph_config，涉及 `prepare_encoder_metadata, get_encoder_cudagraph_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_vl.py` modified +138/-42 (180 lines); hunks: -99,6 +99,7; -689,6 +690,7 @@ def prepare_encoder_metadata(; symbols: prepare_encoder_metadata, get_encoder_cudagraph_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_vl.py` modified +138/-42
- 验证与风险: diff 自带测试面 `tests/v1/cudagraph/test_encoder_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40160 - [Bugfix] Fix k_proj's bias for GLM-ASR

- 链接: https://github.com/vllm-project/vllm/pull/40160
- 状态/时间: merged / 2026-04-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`；关联提交 `aeee7ef93910`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix k_proj's bias for GLM-ASR」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glmasr.py`；技术摘要: 覆盖「[Bugfix] Fix k_proj's bias for GLM-ASR」；主要实现面是 `vllm/model_executor/models/glmasr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glmasr.py` modified +3/-1 (4 lines); hunks: -66,7 +66,7; -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: GlmAsrEncoderRotaryEmbedding, load_weights，涉及 `GlmAsrEncoderRotaryEmbedding, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glmasr.py` modified +3/-1 (4 lines); hunks: -66,7 +66,7; -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: GlmAsrEncoderRotaryEmbedding, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glmasr.py
@@ -66,7 +66,7 @@
-from .whisper import ISO639_1_SUPPORTED_LANGS
+from .whisper import ISO639_1_SUPPORTED_LANGS, _create_fake_bias_for_k_proj
@@ -499,6 +499,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glmasr.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glmasr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40932 - [Bugfix] Remove invalid deepstack boundary check for Qwen3-VL

- 链接: https://github.com/vllm-project/vllm/pull/40932
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `22631f80a01a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+0/-22，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Remove invalid deepstack boundary check for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「[Bugfix] Remove invalid deepstack boundary check for Qwen3-VL」；主要实现面是 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-11 (11 lines); hunks: -1778,11 +1778,6 @@ def _get_deepstack_input_embeds(; -1824,12 +1819,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int)...; symbols: _get_deepstack_input_embeds, _clear_deepstack_input_embeds，涉及 `_get_deepstack_input_embeds, _clear_deepstack_input_embeds`；`vllm/model_executor/models/qwen3_vl.py` modified +0/-11 (11 lines); hunks: -1707,11 +1707,6 @@ def _get_deepstack_input_embeds(; -1753,12 +1748,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int)...; symbols: _get_deepstack_input_embeds, _clear_deepstack_input_embeds，涉及 `_get_deepstack_input_embeds, _clear_deepstack_input_embeds`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-11 (11 lines); hunks: -1778,11 +1778,6 @@ def _get_deepstack_input_embeds(; -1824,12 +1819,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int)...; symbols: _get_deepstack_input_embeds, _clear_deepstack_input_embeds
  - `vllm/model_executor/models/qwen3_vl.py` modified +0/-11 (11 lines); hunks: -1707,11 +1707,6 @@ def _get_deepstack_input_embeds(; -1753,12 +1748,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int)...; symbols: _get_deepstack_input_embeds, _clear_deepstack_input_embeds
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -1778,11 +1778,6 @@ def _get_deepstack_input_embeds(
-        if num_tokens > self.deepstack_input_embeds_num_tokens:
-            raise ValueError(
-                "Requested more deepstack tokens than available in buffer: "
-                f"{num_tokens=} > {self.deepstack_input_embeds_num_tokens=}"
-            )
@@ -1824,12 +1819,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -1707,11 +1707,6 @@ def _get_deepstack_input_embeds(
-        if num_tokens > self.deepstack_input_embeds_num_tokens:
-            raise ValueError(
-                "Requested more deepstack tokens than available in buffer: "
-                f"{num_tokens=} > {self.deepstack_input_embeds_num_tokens=}"
-            )
@@ -1753,12 +1748,6 @@ def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-11; `vllm/model_executor/models/qwen3_vl.py` modified +0/-11
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
