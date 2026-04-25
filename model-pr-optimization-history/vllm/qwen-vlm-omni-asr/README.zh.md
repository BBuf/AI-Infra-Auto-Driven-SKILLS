# vllm Qwen VLM/Omni/ASR 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/offline_inference/qwen2_5_omni/README.md` | [#15130](https://github.com/vllm-project/vllm/pull/15130) |
| `examples/offline_inference/qwen2_5_omni/only_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#32772](https://github.com/vllm-project/vllm/pull/32772) |
| `examples/offline_inference/qwen3_omni/only_thinker.py` | [#27721](https://github.com/vllm-project/vllm/pull/27721), [#33010](https://github.com/vllm-project/vllm/pull/33010) |
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
| `vllm/model_executor/models/qwen2_5_omni_thinker.py` | [#15130](https://github.com/vllm-project/vllm/pull/15130), [#16872](https://github.com/vllm-project/vllm/pull/16872), [#17301](https://github.com/vllm-project/vllm/pull/17301), [#17838](https://github.com/vllm-project/vllm/pull/17838), [#23058](https://github.com/vllm-project/vllm/pull/23058), [#24231](https://github.com/vllm-project/vllm/pull/24231), [#24420](https://github.com/vllm-project/vllm/pull/24420), [#26004](https://github.com/vllm-project/vllm/pull/26004), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#30883](https://github.com/vllm-project/vllm/pull/30883), [#32772](https://github.com/vllm-project/vllm/pull/32772), ... (17 total) |
| `vllm/model_executor/models/qwen2_5_vl.py` | [#12944](https://github.com/vllm-project/vllm/pull/12944), [#13155](https://github.com/vllm-project/vllm/pull/13155), [#13286](https://github.com/vllm-project/vllm/pull/13286), [#13533](https://github.com/vllm-project/vllm/pull/13533), [#13968](https://github.com/vllm-project/vllm/pull/13968), [#14377](https://github.com/vllm-project/vllm/pull/14377), [#15130](https://github.com/vllm-project/vllm/pull/15130), [#15200](https://github.com/vllm-project/vllm/pull/15200), [#15273](https://github.com/vllm-project/vllm/pull/15273), [#16907](https://github.com/vllm-project/vllm/pull/16907), [#16974](https://github.com/vllm-project/vllm/pull/16974), [#17726](https://github.com/vllm-project/vllm/pull/17726), ... (25 total) |
| `vllm/model_executor/models/qwen2_audio.py` | [#11258](https://github.com/vllm-project/vllm/pull/11258), [#35994](https://github.com/vllm-project/vllm/pull/35994) |
| `vllm/model_executor/models/qwen2_vl.py` | [#7905](https://github.com/vllm-project/vllm/pull/7905), [#8442](https://github.com/vllm-project/vllm/pull/8442), [#8696](https://github.com/vllm-project/vllm/pull/8696), [#8770](https://github.com/vllm-project/vllm/pull/8770), [#8837](https://github.com/vllm-project/vllm/pull/8837), [#9250](https://github.com/vllm-project/vllm/pull/9250), [#10112](https://github.com/vllm-project/vllm/pull/10112), [#10169](https://github.com/vllm-project/vllm/pull/10169), [#10221](https://github.com/vllm-project/vllm/pull/10221), [#11258](https://github.com/vllm-project/vllm/pull/11258), [#11430](https://github.com/vllm-project/vllm/pull/11430), [#11663](https://github.com/vllm-project/vllm/pull/11663), ... (31 total) |
| `vllm/model_executor/models/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312), [#33410](https://github.com/vllm-project/vllm/pull/33410), [#33644](https://github.com/vllm-project/vllm/pull/33644), [#37247](https://github.com/vllm-project/vllm/pull/37247) |
| `vllm/model_executor/models/qwen3_asr_forced_aligner.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3_asr_realtime.py` | [#34613](https://github.com/vllm-project/vllm/pull/34613), [#35869](https://github.com/vllm-project/vllm/pull/35869) |
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | [#25550](https://github.com/vllm-project/vllm/pull/25550), [#26608](https://github.com/vllm-project/vllm/pull/26608), [#26815](https://github.com/vllm-project/vllm/pull/26815), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#27721](https://github.com/vllm-project/vllm/pull/27721), [#27920](https://github.com/vllm-project/vllm/pull/27920), [#29255](https://github.com/vllm-project/vllm/pull/29255), [#29828](https://github.com/vllm-project/vllm/pull/29828), [#29896](https://github.com/vllm-project/vllm/pull/29896), [#29974](https://github.com/vllm-project/vllm/pull/29974), [#31007](https://github.com/vllm-project/vllm/pull/31007), [#31790](https://github.com/vllm-project/vllm/pull/31790), ... (23 total) |
| `vllm/model_executor/models/qwen3_vl.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25337](https://github.com/vllm-project/vllm/pull/25337), [#25347](https://github.com/vllm-project/vllm/pull/25347), [#25557](https://github.com/vllm-project/vllm/pull/25557), [#25646](https://github.com/vllm-project/vllm/pull/25646), [#25648](https://github.com/vllm-project/vllm/pull/25648), [#25788](https://github.com/vllm-project/vllm/pull/25788), [#26000](https://github.com/vllm-project/vllm/pull/26000), [#27104](https://github.com/vllm-project/vllm/pull/27104), [#27705](https://github.com/vllm-project/vllm/pull/27705), [#28663](https://github.com/vllm-project/vllm/pull/28663), ... (18 total) |
| `vllm/model_executor/models/qwen3_vl_moe.py` | [#24727](https://github.com/vllm-project/vllm/pull/24727), [#24955](https://github.com/vllm-project/vllm/pull/24955), [#25300](https://github.com/vllm-project/vllm/pull/25300), [#26000](https://github.com/vllm-project/vllm/pull/26000) |
| `vllm/model_executor/models/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/tokenizers/qwen_vl.py` | [#36140](https://github.com/vllm-project/vllm/pull/36140) |
| `vllm/transformers_utils/configs/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/qwen3_asr.py` | [#33312](https://github.com/vllm-project/vllm/pull/33312) |
| `vllm/transformers_utils/processors/qwen_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 101
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 102
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
| 2026-03-11 | [#36136](https://github.com/vllm-project/vllm/pull/36136) | merged | [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps | `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-13 | [#36800](https://github.com/vllm-project/vllm/pull/36800) | merged | [Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-16 | [#37147](https://github.com/vllm-project/vllm/pull/37147) | merged | [Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs | `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-03-16 | [#37183](https://github.com/vllm-project/vllm/pull/37183) | merged | Remove unused EVS functions in qwen3_vl.py | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-03-18 | [#37439](https://github.com/vllm-project/vllm/pull/37439) | merged | [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-04-10 | [#37247](https://github.com/vllm-project/vllm/pull/37247) | merged | [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration | `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py` |
| 2026-04-14 | [#38061](https://github.com/vllm-project/vllm/pull/38061) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference | `vllm/model_executor/models/qwen3_vl.py` |
| 2026-04-18 | [#40160](https://github.com/vllm-project/vllm/pull/40160) | merged | [Bugfix] Fix k_proj's bias for GLM-ASR | `vllm/model_executor/models/glmasr.py` |

## 逐 PR diff 审计卡

### PR #7905 - [Model][VLM] Add Qwen2-VL model support

- 链接: https://github.com/vllm-project/vllm/pull/7905
- 状态/时间: merged / 2024-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `3b7fea770f44`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1531/-31，可读 patch 1844 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model][VLM] Add Qwen2-VL model support」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：This PR adding support for Qwen2-VL model. FIX #8139 FIX #8281 ## Requirements - ~This PR requires `transformers` with this PR merged and this bugfix PR merged (You can install...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `06311e295666`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Misc] Skip loading extra bias for Qwen2-VL GPTQ-Int8」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #8406 ping @DarkLight1337 **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** --- section, markdown rend...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `a79e5229843e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+46/-14，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Support pp for qwen2-vl」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE Follow #7860 , this pr introduce pp support for the latest qwen2-vl. Already tested on Qwen2-VL-72B-Instruct-GPTQ-Int4 **BEFORE SUBMITTING, PLEAS...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c23953675f78`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+99/-9，可读 patch 202 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Hardware][CPU] Enable mrope and support Qwen2-VL on CPU backend」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FILL IN THE PR DESCRIPTION HERE FIX #xxxx (*link existing issues this PR will resolve*) - This PR enables mrope and qwen2-vl on CPU backend. **BEFORE SUBMITTING, PLEASE READ THE...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `4bb98f2190aa`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+144/-224，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Misc] Update config loading for Qwen2-VL and remove Granite」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：This PR ports Qwen2-VL config from transformers v4.45 but remove rope type override to address config loading issue, and remove ported Granite config that's no longer required....
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `7e7eae338d27`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+102/-200，可读 patch 533 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Misc] Standardize RoPE handling for Qwen2-VL」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：https://github.com/huggingface/transformers/issues/33401 has been fixed in Transformers v4.45.2, but the devs have clarified that M-ROPE is intended to be configured as `rope_ty...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `999df95b4eef`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-10，可读 patch 68 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Make image processor respect `mm_processor_kwargs` for Qwen2-VL」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：Without this PR, if one wants to pass more images with lower resolution to Qwen2-VL models (e.g. 127 images with 256 tokens each): the LLM instance will be failed to create due...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `f83feccd7f66`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Ignore GPTQ quantization of Qwen2-VL visual module」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FIX https://github.com/vllm-project/vllm/issues/9832 This is a workaround for the fact that GPTQ configs generated by AutoGPTQ do not have a list of ignored modules to check if...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `3945c82346da`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+578/-32，可读 patch 709 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Add support for Qwen2-VL video embeddings input & multiple image embeddings input with varied resolutions」，变更集中在 `tests/models/decoder_only/vision_language/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Goal 1. Add support for Qwen2-VL multiple image embeddings input with **varied resolutions** currently, vLLM implementation of Qwen2-VL's image embedding input **requires** a...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `e24113a8fe5d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+277/-527，可读 patch 1006 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Model] Refactor Qwen2-VL to use merged multimodal processor」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/mm_processor_kwargs/test_qwen2_vl.py`, `vllm/model_executor/models/qwen2_audio.py`。PR 正文没有提供额外背景，判断主要来自标题、文件列表和 patch。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `b1b1038fbdc1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+168/-14，可读 patch 298 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix Qwen2-VL LoRA weight loading」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FIX #11406 ## Upcoming PRs include: - Convert hf_to_vllm_mapper to static variable for all models - Support `substr` and `subfix` mapping
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `11d8a091c6c7`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+21/-4，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Misc] Optimize Qwen2-VL LoRA test」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：I retrained a QWen-VL LoRA. In my local environment using A800, I tested the model on 100 image samples, the generation results were completely aligned with the `transformers` o...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `81763c58a01e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+292/-85，可读 patch 616 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[V1] Add V1 support of Qwen2-VL」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`, `tests/models/decoder_only/vision_language/test_qwen2_vl.py`。PR 描述补充为：Continued from #11668 for supporting Qwen2-VL on V1. Co-authored-by: @imkero who has done the great work on the V1 MRoPE implementation.
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

### PR #12944 - [Misc] Add qwen2.5-vl BNB support

- 链接: https://github.com/vllm-project/vllm/pull/12944
- 状态/时间: merged / 2025-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `4c8dd12ef347`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-30，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Misc] Add qwen2.5-vl BNB support」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：FIX https://github.com/vllm-project/vllm/pull/12604#discussion_r1940626980 - Add tp compatible BNB support to Qwen2.5-VL models.
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `985b4a2b1989`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix num video tokens calculation for Qwen2-VL」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FIX #13099
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
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Misc] Qwen2.5-VL Optimization」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：Thanks to the VLLM team for their great work! This PR includes the following: ꔷ Replaced `Qwen2RMSNorm` with vllm's official `RMSNorm`. Extensive testing has confirmed that this...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `7fdaaf48ef2a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-6，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix qwen2.5-vl image processor」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FIX #13285 **Need to wait Qwen team update the model repo with correct preprocessor_config.json (see https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/14)** - `tran...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `041e29471671`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+18/-2，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Misc] add mm_processor_kwargs to extra_body for Qwen2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：In Qwen2.5-VL online inference, the `fps` parameter in `mm_processor_kwargs` is essential for accurately calculating the `second_pre_grid_t` value. However, the OpenAI interface...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `78648758794e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+22/-15，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix qwen2.5-vl overflow issue」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：FIX #13817 (*link existing issues this PR will resolve*) - Qwen2.5-VL-3B will overflow with float16 in its ViT encoder, we need to cast the overflow value manually.
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `70b808fe1a63`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+70/-24，可读 patch 186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：qwen2-vl logic optimization: During each forward propagation, the xformer branch of Qwen2VisionTransformer will execute multiple tensor tolist methods (flash attn branch will ex...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `1e508343e1ec`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+37/-4，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：FIX #15122 FIX #15197 - Fix incorrect attention mask creation for qwen2.5-vl windows attention from #14377
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `47c712621316`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+35/-16，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：- #15200 actually disabled the seqlens pre-computation for Qwen2.5-VL by mistake, because seqlens are computed for each layer again. - This PR adds the optimization introduced i...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/offline_inference/qwen2_5_omni/README.md`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `2c1bd848a668`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+1852/-82，可读 patch 2363 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model][VLM] Add Qwen2.5-Omni model support (thinker only)」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`。PR 描述补充为：This PR adding support for Qwen2.5-Omni model (thinker only). ## Requirements This PR requires this corresponding transformers PR. **Note: You need to install transformers from...
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo，涉及 `_qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data`；`vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__，涉及 `forward, all_gather_interleave, Qwen2_5_VisionAttention`；`examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query，涉及 `QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query`；`examples/offline_inference/qwen2_5_omni/README.md` added +32/-0 (32 lines); hunks: -0,0 +1,32。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0 (977 lines); hunks: -0,0 +1,977; symbols: _qwen2_5_omni_thinker_field_config, Qwen2_5OmniThinkerMultiModalDataParser, _parse_audio_data, Qwen2_5OmniThinkerProcessingInfo
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28 (79 lines); hunks: -38,13 +38,14; -195,6 +196,23 @@ def forward(self, x: torch.Tensor):; symbols: forward, all_gather_interleave, Qwen2_5_VisionAttention, __init__
  - `examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query
  - `examples/offline_inference/qwen2_5_omni/README.md` added +32/-0 (32 lines); hunks: -0,0 +1,32
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
diff -- examples/offline_inference/qwen2_5_omni/only_thinker.py
@@ -0,0 +1,160 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` added +977/-0; `vllm/model_executor/models/qwen2_5_vl.py` modified +51/-28
  - docs: `examples/offline_inference/qwen2_5_omni/only_thinker.py` added +160/-0; `examples/offline_inference/qwen2_5_omni/README.md` added +32/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16872 - [Model] Qwen2.5-Omni Cleanup

- 链接: https://github.com/vllm-project/vllm/pull/16872
- 状态/时间: merged / 2025-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `5124f5bf51b8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-5，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Model] Qwen2.5-Omni Cleanup」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：Clean up some oversighted changes from #15130
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `26c0406555a5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix distributed bug in Qwen2.5-VL & Qwen2.5-Omni」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：This PR fixes the distribution bug in Qwen2.5-VL & Qwen2.5-Omni (introduced in #15130). When `tp_group != world_size`, current implementation will cause an allgather error.
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `571e8dd65e2a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix distributed bug again in Qwen2.5-VL & Qwen2.5-Omni」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：I am very sorry that previous fix #16907 still have distributed bug, need to fix it agin. Related PR: #15130 and #16907
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `8b464d9660a8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+75/-94，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Misc] Clean up Qwen2.5-Omni code」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：Split up `apply` even more so that the override in Qwen2.5-Omni doesn't have to repeat as much code
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c3e9d5060e89`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-12，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：- Since we have ported FA's RoPE kernel in `vllm_flash_attn`, there is no need to use original FA's `apply_rotary_emb` anymore - Replace original FA's `apply_rotary_emb` with `v...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `015815fe0141`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+15/-9，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] `use_fast` failing to be propagated to Qwen2-VL image processor」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：As discussed in https://vllm-dev.slack.com/archives/C07QCGVDNUF/p1746687303802249?thread_ts=1745944078.067259&cid=C07QCGVDNUF I decided not to add regression tests because the d...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `67da5720d4ed`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+121/-83，可读 patch 285 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：### Description of Problem In Qwen2.5-VL rotary position embedding constant tensors creates in the beginning of model's `forward`. Before this PR there were a mix of CPU and GPU...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `ec2dcd80bc17`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-8，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Misc] Update `WeightsMapper` for qwen2-vl/qwen2.5-vl」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：FIX #18976 - Transformers v4.52 will map weights name for Qwen2-VL/Qwen2.5-VL: https://github.com/huggingface/transformers/blob/de4cf5a38e9678b9e465867a8a6b88ea727bea52/src/tran...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `d3a6f2120bb6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+64/-39，可读 patch 212 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform.」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：# Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test p...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `cbc8457b2663`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-7，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Model] Switch to Fused RMS norm in Qwen2.5_VL model.」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：# Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test p...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `9f1c6422549d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-0，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] fix Qwen2.5-Omni processor output mapping」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：## Purpose Fix #23056 Add a processor output mapping for Qwen2.5-Omni thinker, to compute video rotary embedding correctly. ## Test Plan Use the script I provided in #23056 and...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `a71e4765cc0c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix Qwen2.5-VL quantized model weights loading」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：This PR makes sure quantized model weights are loaded correctly. Currently, loading `RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic` will crash on A100s: The issue is introduced in...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `c9f7081f9c84`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-3，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[LoRA]: Add lora support to qwen-2.5-omni」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：## Purpose Add LoRA support to `qwen-2.5-omni`, scoped strictly to the language model components. - Implements `SupportsLoRA` on `Qwen2_5OmniThinkerForConditionalGeneration`. -...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `6f4a82f8b5a1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-2，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Enable BNB support for qwen2_5_omni_thinker」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：## Purpose - Enable bitsandbytes quantization for qwen2_5_omni_thinker, fix https://github.com/vllm-project/vllm/issues/23240 - Add `get_mm_mapping` and `SupportLoRA`, deleted b...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `b0d1213ac395`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Models] Prevent CUDA sync in Qwen2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：This is a followup to #24443 from @david6666666 When I profiled Qwen2.5-VL it seems like an implicit CUDA sync is still happening during the indexing: https://github.com/vllm-pr...
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
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Support Qwen3-VL Model Series」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose This PR adds model support for the upcoming Qwen3-VL models, including both dense and MoE variants. Reference HF implementation - https://github.com/huggingface/trans...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `3127274d022b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+77/-19，可读 patch 256 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[MM Encoder] Apply DP ViT for Qwen3-VL model series」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`。PR 描述补充为：## Purpose Add DP ViT for Qwen3-VL models - this PR should be merged only after #24727 is merged. Part of #22743 ## Test Plan Running on `Qwen3-VL-30B-A3B-Instruct` with 4xL40s...
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
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE」，变更集中在 `vllm/model_executor/layers/rotary_embedding/mrope.py`, `tests/kernels/core/test_mrope.py`。PR 描述补充为：## Purpose - Following PR for #24727, implement corresponding Triton kernel for interleaved MRoPE. ## Test Plan ## Test Result Test should pass --- Essential Elements of an Effe...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `be874c020196`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-5，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix Qwen3-VL-MoE weight loading for EP」，变更集中在 `vllm/model_executor/models/qwen3_vl_moe.py`。PR 描述补充为：## Purpose Previously `self.load_fused_expert_weights` check is too strict and will prevent server from launching with `--enable-expert-parallel`. This PR fixes it. ## Test Plan...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `30d08911f7cf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+60/-75，可读 patch 177 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate`」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose ## Test Plan 10 QPS of VisionArena on Qwen3-VL 4B on A100 ## Test Result Main This branch MMMU matched --- Essential Elements of an Effective PR Description Checklist...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `af7dfb0d1a95`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-18，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate`」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose - Following PR for #25337 - Just found that we can further optimize weights computation for `fast_pos_embed_interpolate` by reducing duplicated multiply. - Also impro...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `c98be0a23276`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+59/-19，可读 patch 252 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Enable DP for ViT in Qwen2-VL」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose Part of https://github.com/vllm-project/vllm/issues/22743 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpos...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `7be9ffcd9f5c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Misc] Fix Qwen3-VL `video_grid_thw` typing」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose Without this change, torch will throw a warning `vllm/vllm/model_executor/models/qwen3_vl.py:480: UserWarning: To copy construct from a tensor, it is recommended to u...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `17b4c6685ce6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+13/-1，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose - Suspend of #25557 before we support configurable mm profiling ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [x] The...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `c242c98031b8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+75/-51，可读 patch 208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose Temple fix for https://github.com/vllm-project/vllm/issues/25582 Special thanks to @ywang96 for the context The default XFORMERS backend has problem and we will meet...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `0efd540dbc54`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+74/-9，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose - Qwen3-VL's max_num_video_tokens calculation is implemented wrong, this PR corrects it ## Test Plan ## Test Result --- Essential Elements of an Effective PR Descript...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`；关联提交 `66bca9b8bd0a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+45/-26，可读 patch 105 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[MM] Add text-only mode for Qwen3-VL」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl_moe.py`。PR 描述补充为：## Purpose Since this model is performing pretty well on text only tasks we might want to allow people to serve it as a text-only model. ## Test Plan ## Test Result Running `vll...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `84d57342b66e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-3，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：Fix https://github.com/vllm-project/vllm/issues/25970 When omni is handling the cached video, it will produced `mm_kwargs: {'video': [None]}`. Then the error will occur when acc...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `5e4a8223c644`, `dd96465fd744`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+28/-5，可读 patch 80 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Qwen][ROCm] Flash Attention Rotary Embeddings」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose Qwen VL models previously relied on a basic PyTorch implementation for applying rotary positional embeddings on ROCm architectures. This PR adds a ROCm specialisation...
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

### PR #26123 - [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642

- 链接: https://github.com/vllm-project/vllm/pull/26123
- 状态/时间: merged / 2025-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `dd96465fd744`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-4，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose #24642 introduced dispatch_rotary_emb_function method in common.py while for non_cuda or non_rocm, apply_rotary_emb_torch is differently defined in some modeling code...
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
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「Add Qwen3-Omni moe thinker」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：This PR from the Qwen team for: qwen3-omni-moe thinker part. Testing has been conducted internally across four configurations (v0/v1, eager/CUDA) on several representative bench...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `ddaff2938e0b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+368/-387，可读 patch 859 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补强部署文档、测试或 CI 验证面，标题为「[MM] Move Qwen3Omni MRoPE impl to model file」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose Move the mrope impl to the model file, and clean up some code there as well. ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checkli...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `8c851f6d044b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-2，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix qwen3-omni audio truncation issue」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose - Fix #26630 - Before https://github.com/huggingface/transformers/pull/41473, Qwen3-omni will still truncate audio to 30s. - This PR enforces no truncation in Qwen3-o...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `4c91a28e301d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：The current code overwrites `video_fps` with `sampled_fps`, which causes incorrect timestamp calculations. @ywang96 Hey bro, could you please help review this? Thanks!
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

### PR #27705 - [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes

- 链接: https://github.com/vllm-project/vllm/pull/27705
- 状态/时间: merged / 2025-10-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `0d8161b07504`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-16，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen3_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：## Purpose #23207 Broke Qwen3VL since it relies on `Qwen2_5_VisionAttention` as well but still used the old signature: This PR updates Qwen3VL and Qwen3Omni to fix this. ## Test...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `e806178d2a9b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+20/-12，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[BugFix][VL] Fix FA selection on Qwen2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：## Purpose https://github.com/vllm-project/vllm/pull/27190 breaks AMD CI (and also qwen2.5 vl): tests/v1/entrypoints/openai/responses/test_image.py : with _Backend.FLASH_ATTN it...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `853a8eb53b89`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-10，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen Omni audio inference」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose FIX https://github.com/vllm-project/vllm/issues/27906 FIX #27907 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `48b8456ff992`, `bc5bd45c7d1a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+1/-42，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Refactor] Remove redundant TP gather/split in split_qkv in QwenVL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose This code path uses head-parallel attention, where each rank holds full Q/K/V vectors for its own subset of heads. All attention backends operate per-rank on local he...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `cec275efcef6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-3，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose A quantized Qwen3-VL-32B model using GPTQModel==5.4.0 fails to load with `vllm serve Guan1794/qwen3VL-32B-4bit-gptq` The error message is: The GPTQModel quantization...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`；关联提交 `d84d8f4429a5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-1，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`。PR 描述补充为：### **Summary** Fix a crash in `Qwen2.5-VL` when EVS (Efficient Video Sampling) is enabled and video inputs are provided through the `video_embeds` pathway. The regression occur...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/offline_inference/qwen3_omni/only_thinker.py`, `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `839c6b7b72bc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+467/-59，可读 patch 631 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.」，变更集中在 `tests/model_executor/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：## Purpose FIX #22268 FIX #22364 CLOSE #23888 CLOSE #25473 CLOSE https://github.com/vllm-project/vllm/issues/28046 ## Test Plan Sanity checked with https://qianwen-res.oss-cn-be...
- 实现要点: `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer，涉及 `print_input_ids, mock_qwen3_omni_config, mock_processor`；`vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video，涉及 `_maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video`；`vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders，涉及 `_maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders`；`examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query，涉及 `QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query`。
- 代码 diff 细节:
  - `tests/model_executor/test_qwen3_omni.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: print_input_ids, mock_qwen3_omni_config, mock_processor, mock_tokenizer
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +76/-34 (110 lines); hunks: -68,11 +68,11; -87,7 +87,6; symbols: _maybe_apply_prompt_updates, get_replacement_qwen2_vision, get_replacement_qwen2_use_audio_in_video
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +0/-25 (25 lines); hunks: -23,7 +23,6; -387,15 +386,6 @@ def _maybe_apply_prompt_updates(; symbols: _maybe_apply_prompt_updates, _apply_hf_processor_mm_only, _validate_mm_placeholders
  - `examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: QueryResult, get_mixed_modalities_query, get_use_audio_in_video_query, get_multi_audios_query
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
  - docs: `examples/offline_inference/qwen3_omni/only_thinker.py` added +170/-0
- 验证与风险: diff 自带测试面 `tests/model_executor/test_qwen3_omni.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29896 - feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE

- 链接: https://github.com/vllm-project/vllm/pull/29896
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `a2b053dc858d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-0，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：Add the necessary attributes for BitsAndBytes quantization support: - Add packed_modules_mapping with qkv_proj and gate_up_proj mappings - Add get_mm_mapping() method for multim...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `3f1b03739ae1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose This is a bugfix for qwen3-omni model when using AITER Flash Attention ## Test Plan Evaluate the qwen3-omni chartqa ## Test Result --- Essential Elements of an Effect...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `6dcb07f676ae`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「support qwen3-vl handle requests with embeddings」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：为了支持qwen3-vl 处理带image embedding的请求，需要使用自定义的 函数，用 类替代 即可 ## Test Script ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_vl.py`；关联提交 `48b8456ff992`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-1，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Revert Qwen2-VL part of change in #28271」，变更集中在 `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：This PR reverts qwen2_vl.py part of commit bc5bd45c7d1abbac4a63d97d383212c108e55308. Fixes #30250, the model generates incorrect output when TP is used.
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `ae88aada38ec`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+424/-12，可读 patch 539 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Summary This PR implements **EVS (Efficient Video Sampling)** support for Qwen3-VL models, enabling **dynamic video token pruning** to improve inference efficiency while main...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `6fe588765287`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-22，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补强部署文档、测试或 CI 验证面，标题为「[Chore] Remove v0 dead code for Qwen2.5-omni」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`。PR 描述补充为：## Purpose - Just found we missed Qwen2.5-omni's `embed_multimodal_v0` when removing `embed_multimodal_v0` from models. ## Test Plan ## Test Result --- Essential Elements of an...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `bb24592d139b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-12，可读 patch 65 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Qwen3-Omni] fixed _get_feat_extract_output_lengths function」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；关联提交 `d722e9e614f6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+764/-2，可读 patch 833 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「Add GLM-ASR multimodal support」，变更集中在 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`。PR 描述补充为：## Purpose Support that model :https://huggingface.co/zai-org/GLM-ASR-Nano-2512 ## Test Plan ## Test Result for download and convert test audio: https://paste.ubuntu.com/p/K7S9T...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `97a01308e9ce`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+312/-3，可读 patch 337 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring」，变更集中在 `tests/models/multimodal/processing/test_qwen3_omni.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：The Qwen3OmniMoeProcessor was losing the audio_sample_rate parameter during kwargs restructuring for transformers
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `2c1a4f2488da`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-6，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni)」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：This PR fixes an issue in Qwen3OmniMoeThinker where `audio_kwargs/text_kwargs` could be overridden when restructuring kwargs for `Transformers < 4.58.0`. We now preserve existin...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`；关联提交 `974138751bdb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+672/-41，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Refactor] GLM-ASR Modeling」，变更集中在 `vllm/model_executor/models/glmasr.py`, `vllm/model_executor/models/glmasr_utils.py`。PR 描述补充为：### 🚀 Key Improvements **1. Native vLLM Audio Encoder Implementation** (`glmasr.py`) - Completely rewrote `GlmAsrEncoder` as a vLLM-native implementation with full optimization...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/template/qwen3_vl_reranker.jinja`；关联提交 `eac3b96ec04d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+287/-13，可读 patch 415 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Models] Allow converting Qwen3-VL into Reranker model」，变更集中在 `examples/pooling/score/template/qwen3_vl_reranker.jinja`。PR 描述补充为：## Purpose - Enable reranker support for Qwen3-VL ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such a...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`；关联提交 `542a4059b2bb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+113/-190，可读 patch 377 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL」，变更集中在 `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/qwen2_vl.py`。PR 描述补充为：## Purpose Followup to #28399 and #28730. This PR optimizes the M-RoPE position computation for both **Qwen2-VL** and **Qwen2.5-VL**. - Leveraging the pre-calculated mm_features...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `b8199f604931`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+428/-29，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「[Model] Re-implement Qwen3Omni Audio Encoder」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose Re-implement Qwen3-Omni Audio Encoder with vLLM primitives with some vectorization improvements. - roughly 10% speedup at high batch sizes according to profiling run...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`；关联提交 `38bf2ffb21d5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+40/-30，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix GLM-ASR audio encoder RoPE dim」，变更集中在 `vllm/model_executor/models/glmasr.py`。PR 描述补充为：## Purpose - Fix https://github.com/vllm-project/vllm/issues/32445 - GLM-ASR audio encoder's RoPE should be half rotary. - This issue only occured in native code path, because v...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/offline_inference/qwen2_5_omni/only_thinker.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`；关联提交 `a698e8e7ad4b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+386/-201，可读 patch 689 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Use mm_position to compute mrope positions for Qwen2.5-Omni」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `examples/offline_inference/qwen2_5_omni/only_thinker.py`。PR 描述补充为：## Summary - Refactor get_mrope_input_positions() to use mm_feature.mm_position.offset directly - Follows pattern from PR #32126 for Qwen2-VL/2.5-VL ## Changes - Add iter_mm_fea...
- 实现要点: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates，涉及 `_get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates`；`examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0 (26 lines); hunks: -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query，涉及 `get_multi_audios_query, get_multi_images_query`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198 (549 lines); hunks: -22,10 +22,11; -85,6 +86,7; symbols: _get_mm_fields_config, _derive_audio_from_video_placeholders, _maybe_apply_prompt_updates
  - `examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0 (26 lines); hunks: -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query
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
diff -- examples/offline_inference/qwen2_5_omni/only_thinker.py
@@ -112,10 +112,36 @@ def get_multi_audios_query() -> QueryResult:
+def get_multi_images_query() -> QueryResult:
+    question = "What are the differences between these two images?"
+    prompt = (
+        f"<|im_start|>system\n{default_system}<|im_end|>\n"
+        "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>"
+        "<|vision_bos|><|IMAGE|><|vision_eos|>"
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen2_5_omni_thinker.py` modified +351/-198
  - docs: `examples/offline_inference/qwen2_5_omni/only_thinker.py` modified +26/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/v1/worker/gpu_model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33010 - [Model] Use mm_position to compute mrope positions for Qwen3-Omni

- 链接: https://github.com/vllm-project/vllm/pull/33010
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/offline_inference/qwen3_omni/only_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `6ca2c91b9663`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+293/-298，可读 patch 675 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 优化关键推理路径或后端选择，标题为「[Model] Use mm_position to compute mrope positions for Qwen3-Omni」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `examples/offline_inference/qwen3_omni/only_thinker.py`。PR 描述补充为：Optimizes M-RoPE position calculation for Qwen3-Omni by using `mm_position.offset` directly from `MultiModalFeatureSpec` instead of token-by-token searching through `input_token...
- 实现要点: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping，涉及 `load_weights, get_mrope_input_positions, _compute_audio_token_count`；`examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3 (59 lines); hunks: -2,7 +2,7; -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query, main, parse_args，涉及 `get_multi_audios_query, get_multi_images_query, main`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295 (532 lines); hunks: -22,7 +22,7; -104,10 +104,7; symbols: load_weights, get_mrope_input_positions, _compute_audio_token_count, _get_audio_for_video_mapping
  - `examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3 (59 lines); hunks: -2,7 +2,7; -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:; symbols: get_multi_audios_query, get_multi_images_query, main, parse_args
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
diff -- examples/offline_inference/qwen3_omni/only_thinker.py
@@ -2,7 +2,7 @@
-with the correct prompt format on Qwen2.5-Omni (thinker only).
+with the correct prompt format on Qwen3-Omni (thinker only).
@@ -112,23 +112,51 @@ def get_multi_audios_query() -> QueryResult:
+def get_multi_images_query() -> QueryResult:
+    question = "What are the differences between these two images?"
+    prompt = (
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +237/-295
  - docs: `examples/offline_inference/qwen3_omni/only_thinker.py` modified +56/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33312 - [Models] Qwen3-ASR

- 链接: https://github.com/vllm-project/vllm/pull/33312
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`；关联提交 `8b3f0a99dd50`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1269/-0，可读 patch 1335 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Models] Qwen3-ASR」，变更集中在 `vllm/model_executor/models/qwen3_asr.py`, `vllm/transformers_utils/configs/qwen3_asr.py`, `vllm/transformers_utils/processors/qwen3_asr.py`。PR 描述补充为：## Purpose Add support for Qwen3-ASR model series - see recipe at https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-ASR.md ## Test Plan ## Test Result --- Essential E...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`；关联提交 `e77f162cf59d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+42/-2，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix `Qwen3ASR` language asr tag in output」，变更集中在 `vllm/model_executor/models/qwen3_asr.py`。PR 描述补充为：It appears Qwen3-ASR necessitates of post-processing to remove spurious language tags `language English ` inserted in the prompt (https://github.com/QwenLM/Qwen3-ASR/blob/c17a13...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `cd86fff38fee`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+31/-7，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：This PR fixes a critical crash occurring on AMD (ROCm) hardware when initializing the Qwen3-Omni model (specifically the Qwen3Omni_VisionTransformer). During the memory profilin...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`；关联提交 `ceab70c89d2b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-6，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] fix qwen3-asr response error」，变更集中在 `vllm/model_executor/models/qwen3_asr.py`。PR 描述补充为：## Purpose fix https://github.com/vllm-project/vllm/issues/33643 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `f8516a1ab95f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+172/-12，可读 patch 247 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose Fix bugs preventing `use_audio_in_video=True` from working correctly with Qwen2.5-Omni and Qwen3-Omni. **~~Bug 1: `KeyError: 'audio'` in `MultiModalBudget`~~** ~~Both...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `535de06cb1d9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+104/-2，可读 patch 177 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Add transcription support for Qwen3-Omni」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose Only 4 models are supported as a Transcription model according to the docs. This PR adds Qwen3-Omni per the feature request: https://github.com/vllm-project/vllm/issu...
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
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Realtime] Add Qwen3-ASR realtime streaming support」，变更集中在 `vllm/model_executor/models/qwen3_asr_realtime.py`。PR 描述补充为：## Purpose This PR adds real time Websocket streaming transcription support for Qwen3-ASR. Referenced issue: https://github.com/vllm-project/vllm/issues/34232 ## Test Plan Machi...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `c0615a296d44`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+379/-21，可读 patch 437 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression」，变更集中在 `tests/models/multimodal/processing/test_qwen2_5_omni_embed.py`, `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Summary Fixes #34506 — regression introduced in #33605 where mixed modalities (audio + image + video) no longer worked correctly for Qwen2.5-Omni: the model failed to recogni...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `fa6a6be51978`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-0，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose PR #34580 added a `sequence_lengths` parameter to `Qwen2_5_VisionAttention.forward()` for the FlashInfer cuDNN backend and updated callers in `qwen3_vl.py` and `qwen2...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr_realtime.py`；关联提交 `36bf2131816e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile」，变更集中在 `vllm/model_executor/models/qwen3_asr_realtime.py`。PR 描述补充为：## Summary `Qwen3ASRForConditionalGeneration` (and its realtime subclass `Qwen3ASRRealtimeGeneration`) crashes during `torch.compile` warmup when `--enforce-eager` is not set. T...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen_vl.py`, `vllm/tokenizers/qwen_vl.py`；关联提交 `719634815791`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+118/-66，可读 patch 271 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen-VL tokenizer implementation」，变更集中在 `vllm/tokenizers/qwen_vl.py`, `vllm/model_executor/models/qwen_vl.py`, `vllm/renderers/qwen_vl.py`。PR 描述补充为：## Purpose The recent Renderer refactorings broke `Qwen/Qwen-VL` because `_get_tokenizer_without_image_pad` wasn't being used by the Renderer. To solve this, I implemented a cus...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `3ee68590c7fa`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-57，可读 patch 184 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 调整模型相关实现，标题为「refactor funasr model.」，变更集中在 `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：This PR is a small refactor for the FunASR model. Please take a look @DarkLight1337 @Isotr0py HF's allendou/Fun-ASR-Nano-2512-vllm has also been updated.
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `e998fa76b99a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+86/-0，可读 patch 107 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/models/qwen2_audio.py`。PR 描述补充为：**Bug Description** When running offline inference with the Qwen2.5-Omni model using audio inputs `python examples/offline_inference/audio_language.py --model-type qwen2_5_omni`...
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

### PR #36136 - [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps

- 链接: https://github.com/vllm-project/vllm/pull/36136
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`；关联提交 `724759684cd9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+116/-4，可读 patch 150 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps」，变更集中在 `tests/models/multimodal/processing/test_qwen3_vl.py`, `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Summary Fixes #35909. When `num_frames` is provided via `mm_processor_kwargs` without `fps`, `_get_video_second_idx` was falling back to the default fps (2) to compute the nu...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `abf61aaa8ef2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+128/-12，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose - Following PR for https://github.com/vllm-project/vllm/pull/36319 - Actually, Qwen2.5-Omni and Qwen3-Omni processor's `apply_prompt_updates` is incorrect for `audio_...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `912fbe9555f9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+117/-17，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs」，变更集中在 `vllm/model_executor/models/qwen2_5_omni_thinker.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose - Fix index out of range when Qwen2.5-Omni/Qwen3-omni received multiple video inputs, because ` audio_in_video_item_idx + item_idx` is incorrect: ## Test Plan ## Test...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `43a73f853bac`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-101，可读 patch 108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「Remove unused EVS functions in qwen3_vl.py」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose In qwen3_vl.py, functions including _get_evs_mask_segments, _extract_frame_offsets_from_mask, and _get_actual_frame_token_counts were introduced in #29752. However, t...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `738d0a281fab`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose This PR fixes an incorrect use of `merge_size` in Qwen3-VL video timestamp processing. (This bug also affects Qwen3.5) Previously, `_get_video_second_idx` passed `vid...
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

### PR #37247 - [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration

- 链接: https://github.com/vllm-project/vllm/pull/37247
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`；关联提交 `8d0f908b98cd`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+63/-5，可读 patch 126 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[Model] Implement LoRA support for Qwen3ASRForConditionalGeneration」，变更集中在 `vllm/model_executor/models/qwen3_asr.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`。PR 描述补充为：## Purpose This PR adds LoRA support for `Qwen3ASRForConditionalGeneration` model. For this to work for the audio tower, I had to make a few additional changes: - Implement `get...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_vl.py`；关联提交 `80118853f42a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+583/-68，可读 patch 1042 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 补齐模型支持入口或运行时能力，标题为「[MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference」，变更集中在 `vllm/model_executor/models/qwen3_vl.py`。PR 描述补充为：## Purpose Following https://github.com/vllm-project/vllm/pull/35963 (only supports image inference), this PR continues to work on it to support video inference for Qwen3-VL. **...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glmasr.py`；关联提交 `aeee7ef93910`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Qwen VLM/Omni/ASR 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix k_proj's bias for GLM-ASR」，变更集中在 `vllm/model_executor/models/glmasr.py`。PR 描述补充为：## Purpose GLM-ASR has `bias = true` only for `q_proj` and `v_proj` not for `k_proj` So, if we use `QKVParallelLinear` to fuse its `qkv_proj`, the bias part for `k_proj` will re...
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

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
