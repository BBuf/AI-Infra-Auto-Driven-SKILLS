# sglang Qwen VLM/Omni/ASR 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/qwen3_vl.md` | [#12554](https://github.com/sgl-project/sglang/pull/12554) |
| `docs_new/cookbook/omni/FishAudio/S2-Pro.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/omni/intro.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/qwen3_vl.mdx` | 无直接 PR 号提交 |
| `examples/chat_template/qwen3_vl_reranker.jinja` | 无直接 PR 号提交 |
| `examples/runtime/qwen3_vl_reranker.py` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/configs/qwen3_omni.py` | [#10911](https://github.com/sgl-project/sglang/pull/10911) |
| `python/sglang/srt/configs/qwen3_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911) |
| `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` | [#22089](https://github.com/sgl-project/sglang/pull/22089), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/glmasr.py` | [#15570](https://github.com/sgl-project/sglang/pull/15570), [#15772](https://github.com/sgl-project/sglang/pull/15772) |
| `python/sglang/srt/models/qwen2_5_vl.py` | [#5003](https://github.com/sgl-project/sglang/pull/5003), [#5349](https://github.com/sgl-project/sglang/pull/5349), [#6136](https://github.com/sgl-project/sglang/pull/6136), [#8801](https://github.com/sgl-project/sglang/pull/8801), [#13055](https://github.com/sgl-project/sglang/pull/13055), [#13075](https://github.com/sgl-project/sglang/pull/13075), [#13126](https://github.com/sgl-project/sglang/pull/13126), [#13904](https://github.com/sgl-project/sglang/pull/13904), [#14292](https://github.com/sgl-project/sglang/pull/14292), [#15138](https://github.com/sgl-project/sglang/pull/15138), [#15320](https://github.com/sgl-project/sglang/pull/15320) |
| `python/sglang/srt/models/qwen2_audio.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/qwen2_vl.py` | [#2055](https://github.com/sgl-project/sglang/pull/2055), [#5003](https://github.com/sgl-project/sglang/pull/5003), [#5349](https://github.com/sgl-project/sglang/pull/5349), [#5783](https://github.com/sgl-project/sglang/pull/5783), [#6136](https://github.com/sgl-project/sglang/pull/6136), [#13055](https://github.com/sgl-project/sglang/pull/13055), [#13736](https://github.com/sgl-project/sglang/pull/13736) |
| `python/sglang/srt/models/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073) |
| `python/sglang/srt/models/qwen3_omni_moe.py` | [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11791](https://github.com/sgl-project/sglang/pull/11791), [#12333](https://github.com/sgl-project/sglang/pull/12333), [#18185](https://github.com/sgl-project/sglang/pull/18185) |
| `python/sglang/srt/models/qwen3_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11458](https://github.com/sgl-project/sglang/pull/11458), [#11481](https://github.com/sgl-project/sglang/pull/11481), [#12333](https://github.com/sgl-project/sglang/pull/12333), [#13724](https://github.com/sgl-project/sglang/pull/13724), [#13736](https://github.com/sgl-project/sglang/pull/13736), [#14292](https://github.com/sgl-project/sglang/pull/14292), [#15205](https://github.com/sgl-project/sglang/pull/15205), [#15320](https://github.com/sgl-project/sglang/pull/15320), [#16366](https://github.com/sgl-project/sglang/pull/16366), [#16781](https://github.com/sgl-project/sglang/pull/16781), ... (15 total) |
| `python/sglang/srt/models/qwen3_vl_moe.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11481](https://github.com/sgl-project/sglang/pull/11481), [#13983](https://github.com/sgl-project/sglang/pull/13983), [#21469](https://github.com/sgl-project/sglang/pull/21469) |
| `python/sglang/srt/multimodal/processors/glmasr.py` | [#15570](https://github.com/sgl-project/sglang/pull/15570), [#15772](https://github.com/sgl-project/sglang/pull/15772) |
| `python/sglang/srt/multimodal/processors/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073), [#22089](https://github.com/sgl-project/sglang/pull/22089), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/multimodal/processors/qwen_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11377](https://github.com/sgl-project/sglang/pull/11377), [#12240](https://github.com/sgl-project/sglang/pull/12240), [#12458](https://github.com/sgl-project/sglang/pull/12458) |
| `python/sglang/test/external_models/custom_qwen2_vl.py` | 无直接 PR 号提交 |
| `test/manual/models/test_qwen3_asr.py` | [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `test/registered/ascend/vlm_models/test_npu_qwen2_5_vl_3b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_qwen2_5_vl_72b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_235b_a22b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_30b_a3b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_4b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_8b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` | [#21469](https://github.com/sgl-project/sglang/pull/21469) |

## PR 覆盖总览

- git 追溯 PR 数: 39
- 原文档显式引用补充 PR 数: 37
- 当前文档总 PR 数: 76
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-11-21 | [#2055](https://github.com/sgl-project/sglang/pull/2055) | merged | Add support for Qwen2-VL-based embedding models | `python/sglang/srt/models/qwen2_vl.py` |
| 2025-04-14 | [#5003](https://github.com/sgl-project/sglang/pull/5003) | merged | Support for Qwen2.5-VL Model in bitsandbytes Format | `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-04-24 | [#5349](https://github.com/sgl-project/sglang/pull/5349) | merged | vlm: enable radix cache for qwen-vl models | `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-05-01 | [#5783](https://github.com/sgl-project/sglang/pull/5783) | merged | Remove unused method `calculate_num_image_tokens` from qwen2_vl.py | `python/sglang/srt/models/qwen2_vl.py` |
| 2025-05-16 | [#6136](https://github.com/sgl-project/sglang/pull/6136) | merged | Support precomputed multimodal features for Qwen-VL and Gemma3 models. | `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py` |
| 2025-09-08 | [#8801](https://github.com/sgl-project/sglang/pull/8801) | merged | Qwen2.5-VL eagle3 infer | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-09-23 | [#10323](https://github.com/sgl-project/sglang/pull/10323) | merged | model: support qwen3-vl series | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2025-09-26 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-01 | [#10985](https://github.com/sgl-project/sglang/pull/10985) | merged | Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-10-12 | [#11481](https://github.com/sgl-project/sglang/pull/11481) | merged | [smol] [perf] Qwen3-VL in place op. | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2025-10-15 | [#11458](https://github.com/sgl-project/sglang/pull/11458) | merged | [BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl | `python/sglang/srt/models/qwen3_vl.py` |
| 2025-10-16 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2025-10-21 | [#11377](https://github.com/sgl-project/sglang/pull/11377) | merged | [BugFix][Qwen3-VL]: add metadata for video in qwen3-vl | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2025-10-28 | [#12261](https://github.com/sgl-project/sglang/pull/12261) | open | [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-11-05 | [#12662](https://github.com/sgl-project/sglang/pull/12662) | open | [CPU] Add support for Qwen3-vl and Qwen3-omni | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2025-11-05 | [#12703](https://github.com/sgl-project/sglang/pull/12703) | open | add qwen3-omni docs | `docs/basic_usage/qwen3_omni.md`, `docs/index.rst` |
| 2025-11-06 | [#12240](https://github.com/sgl-project/sglang/pull/12240) | merged | [VLM] Optimize qwen_vl preprocess_video | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2025-11-10 | [#12554](https://github.com/sgl-project/sglang/pull/12554) | merged | [Docs] Add docs for Qwen3-VL image and video support | `docs/basic_usage/qwen3_vl.md` |
| 2025-11-12 | [#12458](https://github.com/sgl-project/sglang/pull/12458) | merged | fix: duplicate resize images logic of qwen-vl series models | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2025-11-12 | [#13075](https://github.com/sgl-project/sglang/pull/13075) | merged | [VLM] Support PP for Qwen2.5-VL | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-11-18 | [#13126](https://github.com/sgl-project/sglang/pull/13126) | merged | [VLM][feat] Support encoder DP for Qwen2.5-VL | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-11-20 | [#13055](https://github.com/sgl-project/sglang/pull/13055) | merged | [VLM] Support Piecewise CUDA Graph for Qwen2.5-VL | `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py` |
| 2025-11-22 | [#13736](https://github.com/sgl-project/sglang/pull/13736) | merged | [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_vl.py` |
| 2025-11-25 | [#13918](https://github.com/sgl-project/sglang/pull/13918) | open | [VLM] support qwen3-vl eagle infer | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py` |
| 2025-11-26 | [#13983](https://github.com/sgl-project/sglang/pull/13983) | merged | Support KTransformers for Qwen3-VL moe | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2025-11-28 | [#13904](https://github.com/sgl-project/sglang/pull/13904) | merged | [Bugfix] qwen2.5-vl spec decode accept_len low | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-11-28 | [#13724](https://github.com/sgl-project/sglang/pull/13724) | merged | support qwen3_vl vision model dp | `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-04 | [#14292](https://github.com/sgl-project/sglang/pull/14292) | merged | [VLM] Introduce Cache for positional embedding ids for Qwen-VL family | `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-06 | [#11791](https://github.com/sgl-project/sglang/pull/11791) | merged | fix rmsnorm -> layernorm in qwen3 omni | `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2025-12-11 | [#14886](https://github.com/sgl-project/sglang/pull/14886) | open | Support qwen3-omni with DP Encoder | `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-15 | [#14907](https://github.com/sgl-project/sglang/pull/14907) | merged | [VLM] Support chunked vit attention | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/mm_utils.py` |
| 2025-12-17 | [#12333](https://github.com/sgl-project/sglang/pull/12333) | merged | [PP] Add pp support for Qwen3-VL | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2025-12-17 | [#15138](https://github.com/sgl-project/sglang/pull/15138) | merged | [bug fix][pp] fix weight load for qwen2.5-vl | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-12-18 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-20 | [#15320](https://github.com/sgl-project/sglang/pull/15320) | merged | [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-12-23 | [#15570](https://github.com/sgl-project/sglang/pull/15570) | merged | [GLM-ASR] GLM-ASR Support | `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py` |
| 2025-12-25 | [#15772](https://github.com/sgl-project/sglang/pull/15772) | merged | Change GLM-ASR class name | `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py` |
| 2026-01-05 | [#16491](https://github.com/sgl-project/sglang/pull/16491) | open | [Qwen3-VL][PP] Skip loading expert weights not on this rank | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-01-06 | [#16571](https://github.com/sgl-project/sglang/pull/16571) | open | [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py` |
| 2026-01-09 | [#16785](https://github.com/sgl-project/sglang/pull/16785) | open | [Bugfix] fix recompile in qwen3 vl | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-01-13 | [#16996](https://github.com/sgl-project/sglang/pull/16996) | open | feat: Support 'use_audio_in_video' option for qwen3omnimoe model | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/entrypoints/openai/protocol.py` |
| 2026-01-16 | [#17202](https://github.com/sgl-project/sglang/pull/17202) | open | [Feat] Accelerate qwen3vl by remove cpu op | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py` |
| 2026-01-18 | [#17276](https://github.com/sgl-project/sglang/pull/17276) | open | Add Qwen3VL Eagle3 Inference Support | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-22 | [#16366](https://github.com/sgl-project/sglang/pull/16366) | merged | Optimize Qwen3-VL video memory usage | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-30 | [#17624](https://github.com/sgl-project/sglang/pull/17624) | merged | [BUGFIX] Fix dp size > 1 for qwen3 vl model | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py` |
| 2026-02-02 | [#18024](https://github.com/sgl-project/sglang/pull/18024) | merged | fix: correct weight loading prefix mapping for Qwen3-VL | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-05 | [#16781](https://github.com/sgl-project/sglang/pull/16781) | merged | Refactor(qwen3-vl) optimize position encoding interpolation | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-12 | [#18721](https://github.com/sgl-project/sglang/pull/18721) | open | [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models | `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-13 | [#18771](https://github.com/sgl-project/sglang/pull/18771) | open | Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-02-24 | [#19242](https://github.com/sgl-project/sglang/pull/19242) | open | [feat] feat: add Qwen3-ASR support like whisper | `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py` |
| 2026-02-24 | [#19003](https://github.com/sgl-project/sglang/pull/19003) | merged | [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` |
| 2026-02-27 | [#19333](https://github.com/sgl-project/sglang/pull/19333) | merged | fix qwen3_vl visual module loading | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-02 | [#19693](https://github.com/sgl-project/sglang/pull/19693) | open | [NPU] Fix Qwen3-VL-8B Accuracy for NPU | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-02 | [#19291](https://github.com/sgl-project/sglang/pull/19291) | merged | [Qwen3.5] Fix missing `quant_config` in `Qwen3VL` | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-14 | [#18185](https://github.com/sgl-project/sglang/pull/18185) | merged | [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker | `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2026-03-18 | [#20788](https://github.com/sgl-project/sglang/pull/20788) | merged | [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-18 | [#20857](https://github.com/sgl-project/sglang/pull/20857) | open | add EVS support for Qwen3-VL | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-19 | [#20759](https://github.com/sgl-project/sglang/pull/20759) | merged | [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-04-01 | [#21469](https://github.com/sgl-project/sglang/pull/21469) | merged | [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct | `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` |
| 2026-04-01 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | `python/sglang/srt/models/qwen3.py` |
| 2026-04-03 | [#19135](https://github.com/sgl-project/sglang/pull/19135) | merged | qwen3 vl skip layer id for pp | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-03 | [#22052](https://github.com/sgl-project/sglang/pull/22052) | open | [Fix] Enable precise embedding interpolation by default for Qwen3-VL | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-04 | [#22038](https://github.com/sgl-project/sglang/pull/22038) | merged | [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`, `python/sglang/srt/models/deepseek_vl2.py` |
| 2026-04-06 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py` |
| 2026-04-07 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-08 | [#22266](https://github.com/sgl-project/sglang/pull/22266) | merged | [NPU] fix qwen3.5 video processor | `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` |
| 2026-04-08 | [#22181](https://github.com/sgl-project/sglang/pull/22181) | merged | [refactor] [asr] Add transcription adapter for extensible ASR models support | `python/sglang/srt/configs/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` |
| 2026-04-09 | [#22230](https://github.com/sgl-project/sglang/pull/22230) | merged | [Feature] Support eagle3 for qwen3-vl | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-04-09 | [#22089](https://github.com/sgl-project/sglang/pull/22089) | merged | [Feature] Add chunk-based streaming ASR for Qwen3-ASR | `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-15 | [#22839](https://github.com/sgl-project/sglang/pull/22839) | open | fix(config): Add from_dict() for Qwen3VL config classes | `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py` |
| 2026-04-15 | [#22848](https://github.com/sgl-project/sglang/pull/22848) | open | [Feature] WebSocket streaming audio input for ASR | `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py` |
| 2026-04-18 | [#23115](https://github.com/sgl-project/sglang/pull/23115) | open | fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-18 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-20 | [#23220](https://github.com/sgl-project/sglang/pull/23220) | open | Bugfix: Qwen3-VL-MoE adapt encoder_only | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-21 | [#23304](https://github.com/sgl-project/sglang/pull/23304) | closed | [Bugfix] Fix Qwen3-VL rope config compatibility | `python/sglang/srt/models/qwen3.py` |
| 2026-04-22 | [#23469](https://github.com/sgl-project/sglang/pull/23469) | open | [NPU] adapt the Qwen3-ASR model for deployment on NPU | `python/sglang/srt/utils/common.py` |

## 逐 PR diff 审计卡

### PR #2055 - Add support for Qwen2-VL-based embedding models

- 链接: https://github.com/sgl-project/sglang/pull/2055
- 状态/时间: merged / 2024-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_vl.py`；关联提交 `f6f713797bcb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+39/-12，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support for Qwen2-VL-based embedding models」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_vl.py`；PR 正文摘要: Add support for Qwen2-VL based embedding models (e.g. mcdse-2b-v1), as requested in https://github.com/sgl-project/sglang/issues/2032.。
- 实现要点: `python/sglang/srt/models/qwen2_vl.py` modified +12/-5 (17 lines); hunks: -44,6 +44,7; -559,6 +560,7 @@ def __init__(; symbols: __init__, _process_image_input, forward，涉及 `__init__, _process_image_input, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_vl.py` modified +12/-5 (17 lines); hunks: -44,6 +44,7; -559,6 +560,7 @@ def __init__(; symbols: __init__, _process_image_input, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -44,6 +44,7 @@
+from sglang.srt.layers.pooler import Pooler, PoolingType
@@ -559,6 +560,7 @@ def __init__(
+        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
@@ -577,6 +579,7 @@ def forward(
+        get_embedding: bool = False,
@@ -599,8 +602,8 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +12/-5
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/models/test_embedding_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5003 - Support for Qwen2.5-VL Model in bitsandbytes Format

- 链接: https://github.com/sgl-project/sglang/pull/5003
- 状态/时间: merged / 2025-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；关联提交 `072df753546b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+375/-45，可读 patch 531 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support for Qwen2.5-VL Model in bitsandbytes Format」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: Support for Qwen2.5-VL Model in bitsandbytes Format 1. Added code in `scheduler.py` to automatically load the bnb model format, in line with the definition in the documentation...。
- 实现要点: `python/sglang/srt/models/qwen2_vl.py` modified +24/-23 (47 lines); hunks: -152,7 +152,7 @@ def __init__(; -351,7 +351,7 @@ def __init__(; symbols: __init__, dtype, device, forward，涉及 `__init__, dtype, device`；`python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22 (46 lines); hunks: -141,7 +141,7 @@ def __init__(; -325,7 +325,7 @@ def get_window_index(self, grid_thw):; symbols: __init__, get_window_index, dtype, device，涉及 `__init__, get_window_index, dtype`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_vl.py` modified +24/-23 (47 lines); hunks: -152,7 +152,7 @@ def __init__(; -351,7 +351,7 @@ def __init__(; symbols: __init__, dtype, device, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22 (46 lines); hunks: -141,7 +141,7 @@ def __init__(; -325,7 +325,7 @@ def get_window_index(self, grid_thw):; symbols: __init__, get_window_index, dtype, device
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -152,7 +152,7 @@ def __init__(
-            use_qkv_parallel=False,
+            use_qkv_parallel=True,
@@ -351,7 +351,7 @@ def __init__(
-        return next(self.parameters()).dtype
+        return self.patch_embed.proj.weight.dtype
@@ -423,6 +423,25 @@ def forward(
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -141,7 +141,7 @@ def __init__(
-            use_qkv_parallel=False,
+            use_qkv_parallel=True,
@@ -325,7 +325,7 @@ def get_window_index(self, grid_thw):
-        return self.blocks[0].mlp.gate_proj.weight.dtype
+        return self.patch_embed.proj.weight.dtype
@@ -429,6 +429,25 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +24/-23; `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_bnb.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5349 - vlm: enable radix cache for qwen-vl models

- 链接: https://github.com/sgl-project/sglang/pull/5349
- 状态/时间: merged / 2025-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；关联提交 `c998d04b4692`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+435/-337，可读 patch 1363 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「vlm: enable radix cache for qwen-vl models」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: 1. support qwen2vl/qwen2.5vl radix cache 2. fix bugs about mrope calculation in qwen2vl/qwen2.5vl Background **mrope** is a special 3D position_id for qwen-vl models. In order t...。
- 实现要点: `python/sglang/srt/models/qwen2_vl.py` modified +3/-7 (10 lines); hunks: -42,7 +42,7; -490,15 +490,11 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature，涉及 `__init__, pad_input_ids, get_image_feature`；`python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6 (9 lines); hunks: -49,7 +49,7; -488,11 +488,8 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature，涉及 `__init__, pad_input_ids, get_image_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_vl.py` modified +3/-7 (10 lines); hunks: -42,7 +42,7; -490,15 +490,11 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6 (9 lines); hunks: -49,7 +49,7; -488,11 +488,8 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -42,7 +42,7 @@
-    MultiModalityDataPaddingPatternTokenPairs,
+    MultiModalityDataPaddingPatternMultimodalTokens,
@@ -490,15 +490,11 @@ def __init__(
-    # Use grid_t * grid_w * grid_h to pad tokens for each image
-    # add replaced padding by unique image hash
-        im_start_id: int = mm_inputs.im_start_id
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -49,7 +49,7 @@
-    MultiModalityDataPaddingPatternTokenPairs,
+    MultiModalityDataPaddingPatternMultimodalTokens,
@@ -488,11 +488,8 @@ def __init__(
-        im_start_id: int = mm_inputs.im_start_id
-        im_end_id: int = mm_inputs.im_end_id
-        media_token_pairs = [(im_start_id, im_end_id)]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +3/-7; `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/run_suite.py`, `test/srt/test_vision_openai_server.py`, `test/srt/test_vlm_accuracy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5783 - Remove unused method `calculate_num_image_tokens` from qwen2_vl.py

- 链接: https://github.com/sgl-project/sglang/pull/5783
- 状态/时间: merged / 2025-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_vl.py`；关联提交 `e97e57e699e5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-12，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove unused method `calculate_num_image_tokens` from qwen2_vl.py」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen2_vl.py`；PR 正文摘要: `calculate_num_image_tokens` is not used anywhere, just remove it.。
- 实现要点: `python/sglang/srt/models/qwen2_vl.py` modified +0/-12 (12 lines); hunks: -442,18 +442,6 @@ class Qwen2VLForConditionalGeneration(nn.Module):; symbols: Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__，涉及 `Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-12 (12 lines); hunks: -442,18 +442,6 @@ class Qwen2VLForConditionalGeneration(nn.Module):; symbols: Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -442,18 +442,6 @@ class Qwen2VLForConditionalGeneration(nn.Module):
-    def calculate_num_image_tokens(self, image_grid_thw: Tuple[int, int, int]):
-        processor = cached_get_processor(self.config._name_or_path)
-        grid_t, grid_h, grid_w = image_grid_thw
-        num_image_tokens = (
-            grid_t
-            * grid_h
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +0/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6136 - Support precomputed multimodal features for Qwen-VL and Gemma3 models.

- 链接: https://github.com/sgl-project/sglang/pull/6136
- 状态/时间: merged / 2025-05-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；关联提交 `f19a9204cdbd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+592/-125，可读 patch 1118 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support precomputed multimodal features for Qwen-VL and Gemma3 models.」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；PR 正文摘要: When running a VLM via SGLang, it's useful to be able to supply image embeddings directly. For example maybe you've frozen your vision encoder and want to precompute features, o...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -497,6 +497,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature，涉及 `pad_input_ids, get_image_feature`；`python/sglang/srt/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -486,6 +486,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature，涉及 `pad_input_ids, get_image_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -497,6 +497,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature
  - `python/sglang/srt/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -486,6 +486,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -497,6 +497,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
+        if any(item.precomputed_features is not None for item in items):
+            if not all(item.precomputed_features is not None for item in items):
+                raise NotImplementedError(
+                    "MM inputs where only some items are precomputed."
+                )
+            return torch.concat([item.precomputed_features for item in items])
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -486,6 +486,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
+        if any(item.precomputed_features is not None for item in items):
+            if not all(item.precomputed_features is not None for item in items):
+                raise NotImplementedError(
+                    "MM inputs where only some items are precomputed."
+                )
+            return torch.concat([item.precomputed_features for item in items])
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0; `python/sglang/srt/models/qwen2_vl.py` modified +6/-0
- 验证与风险: diff 自带测试面 `test/srt/test_skip_tokenizer_init.py`, `test/srt/test_vlm_accuracy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8801 - Qwen2.5-VL eagle3 infer

- 链接: https://github.com/sgl-project/sglang/pull/8801
- 状态/时间: merged / 2025-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`；关联提交 `37d83c6e6d8a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+114/-5，可读 patch 260 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Qwen2.5-VL eagle3 infer」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: support qwen2.5-vl eagle3 infer 1. add set_eagle3_layers_to_capture in qwen2 and qwen2.5_vl 2. change raw_bs to raw_num_token for mrope when target_verify 3. llama_eagle3 suppor...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1 (25 lines); hunks: -517,6 +517,9 @@ def __init__(; -587,9 +590,13 @@ def forward(; symbols: __init__, pad_input_ids, forward, load_weights，涉及 `__init__, pad_input_ids, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1 (25 lines); hunks: -517,6 +517,9 @@ def __init__(; -587,9 +590,13 @@ def forward(; symbols: __init__, pad_input_ids, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -517,6 +517,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
@@ -587,9 +590,13 @@ def forward(
+        aux_hidden_states = None
+        if self.capture_aux_hidden_states:
+            hidden_states, aux_hidden_states = hidden_states
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/forward_batch_info.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10323 - model: support qwen3-vl series

- 链接: https://github.com/sgl-project/sglang/pull/10323
- 状态/时间: merged / 2025-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；关联提交 `4f564b9e8378`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1898/-8，可读 patch 2006 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support qwen3-vl series」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: This PR introduces support for the upcoming **Qwen3-VL** models — including both dense and MoE variants, as well as Instruct and Thinking editions. As the next generation of the...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` added +787/-0 (787 lines); hunks: -0,0 +1,787; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLVisionPatchEmbed，涉及 `Qwen3_VisionMLP, __init__, forward`；`python/sglang/srt/configs/qwen3_vl.py` added +586/-0 (586 lines); hunks: -0,0 +1,586; symbols: Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig, to，涉及 `Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig`；`python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature，涉及 `Qwen3MoeLLMModel, __init__, get_input_embeddings`；`python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1 (9 lines); hunks: -12,6 +12,8; -209,7 +211,12 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, __init__，涉及 `preprocess_video, Qwen2_5VLImageProcessor, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` added +787/-0 (787 lines); hunks: -0,0 +1,787; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLVisionPatchEmbed
  - `python/sglang/srt/configs/qwen3_vl.py` added +586/-0 (586 lines); hunks: -0,0 +1,586; symbols: Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig, to
  - `python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1 (9 lines); hunks: -12,6 +12,8; -209,7 +211,12 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -0,0 +1,787 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/configs/qwen3_vl.py
@@ -0,0 +1,586 @@
+from typing import Optional, Union
+from transformers import PretrainedConfig
+from transformers.modeling_rope_utils import rope_config_validation
+class Qwen3VLVisionConfig(PretrainedConfig):
+    model_type = "qwen3_vl"
+    base_config_key = "vision_config"
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -0,0 +1,471 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` added +787/-0; `python/sglang/srt/configs/qwen3_vl.py` added +586/-0; `python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- 链接: https://github.com/sgl-project/sglang/pull/10749
- 状态/时间: merged / 2025-09-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+105/-34，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fuse write kv buffer into rope for qwen3 moe & bailing moe」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`；PR 正文摘要: Fused write kv buffer into rope for qwen3 moe and bailing moe models. Got minor e2e speedup. Inspired by https://github.com/sgl-project/sglang/pull/9014 gsm8k result:。
- 实现要点: `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg，涉及 `enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg`；`python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention，涉及 `forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg`；`python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core，涉及 `forward_prepare, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -0,0 +1,51 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,6 +66,10 @@
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
@@ -193,33 +197,6 @@ def forward_normal(
-def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
diff -- python/sglang/srt/models/bailing_moe.py
@@ -72,6 +72,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/utils.py` added +51/-0; `python/sglang/srt/models/gpt_oss.py` modified +7/-30; `python/sglang/srt/models/bailing_moe.py` modified +25/-2; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- 链接: https://github.com/sgl-project/sglang/pull/10985
- 状态/时间: merged / 2025-10-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-2，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`；PR 正文摘要: https://github.com/sgl-project/sglang/pull/10749 This PR fixes an issue in Qwen3-MOE where `fused_set_kv_buffer_arg` was passed to `q, k = self.rotary_emb`, but only `RotaryEmbe...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core，涉及 `__init__, forward_prepare, forward_core`；`python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunks: -1065,6 +1065,7 @@ def forward(; -1075,6 +1076,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunks: -1065,6 +1065,7 @@ def forward(; -1075,6 +1076,9 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -51,7 +51,7 @@
-from sglang.srt.layers.rotary_embedding import get_rope
+from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
@@ -358,6 +358,10 @@ def __init__(
+        self.compatible_with_fused_kv_buffer = (
+            False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
+        )
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -1065,6 +1065,7 @@ def forward(
+        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
@@ -1075,6 +1076,9 @@ def forward(
+        assert (
+            fused_set_kv_buffer_arg is None
+        ), "save kv cache is not supported for MRotaryEmbedding."
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11481 - [smol] [perf] Qwen3-VL in place op.

- 链接: https://github.com/sgl-project/sglang/pull/11481
- 状态/时间: merged / 2025-10-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；关联提交 `be740acdb0ad`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-11，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[smol] [perf] Qwen3-VL in place op.」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: Because it is not touched by `compile`。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +4/-7 (11 lines); hunks: -189,10 +189,10 @@ def forward(; -441,7 +441,7 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4 (7 lines); hunks: -114,7 +114,7 @@ def forward(; -130,9 +130,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +4/-7 (11 lines); hunks: -189,10 +189,10 @@ def forward(; -441,7 +441,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4 (7 lines); hunks: -114,7 +114,7 @@ def forward(; -130,9 +130,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -189,10 +189,10 @@ def forward(
-        x = x + attn
+        x += attn
-        x = x + mlp
+        x += mlp
@@ -441,7 +441,7 @@ def forward(
-        x = x + pos_embeds
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -114,7 +114,7 @@ def forward(
-            layer_idx = layer_idx + self.start_layer
+            layer_idx += self.start_layer
@@ -130,9 +130,8 @@ def forward(
-                hidden_states = (
-                    hidden_states
-                    + input_deepstack_embeds[:, sep : sep + self.hidden_size]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +4/-7; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11458 - [BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl

- 链接: https://github.com/sgl-project/sglang/pull/11458
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `b2c856692092`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-3，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: The implementation of cur_seqlens is inconsistent with the HF implementation, resulting in accuracy issues with multi-frame/multi-patch inputs. Fixed the implementation of cur_s...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +5/-3 (8 lines); hunks: -452,13 +452,15 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-3 (8 lines); hunks: -452,13 +452,15 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -452,13 +452,15 @@ def forward(
+        cu_seqlens = torch.repeat_interleave(
+            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
+        ).cumsum(dim=0)
-                torch.tensor([0], device=grid_thw.device),
-                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).cumsum(dim=0),
+                torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +5/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10911 - model: qwen3-omni (thinker-only)

- 链接: https://github.com/sgl-project/sglang/pull/10911
- 状态/时间: merged / 2025-10-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py` 等 6 个文件；关联提交 `86b04d25b3f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1947/-328，可读 patch 2837 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: qwen3-omni (thinker-only)」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: solve #11343。
- 实现要点: `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding，涉及 `Qwen3OmniMoeAudioEncoderLayer, __init__, forward`；`python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: -0,0 +1,613; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeTextConfig，涉及 `Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig`；`python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: -14,41 +14,27; -60,28 +46,16 @@ class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature，涉及 `Qwen3MoeLLMModel, __init__, get_input_embeddings`；`python/sglang/srt/models/qwen3_vl.py` modified +38/-24 (62 lines); hunks: -15,7 +15,7; -27,7 +27,11; symbols: forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger, __init__，涉及 `forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: -0,0 +1,613; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeTextConfig
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: -14,41 +14,27; -60,28 +46,16 @@ class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
  - `python/sglang/srt/models/qwen3_vl.py` modified +38/-24 (62 lines); hunks: -15,7 +15,7; -27,7 +27,11; symbols: forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger, __init__
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +40/-6 (46 lines); hunks: -12,6 +12,7; -209,22 +210,31 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, QwenVLImageProcessor, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -0,0 +1,661 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/configs/qwen3_omni.py
@@ -0,0 +1,613 @@
+from transformers import PretrainedConfig
+from transformers.configuration_utils import layer_type_validation
+from transformers.modeling_rope_utils import rope_config_validation
+from sglang.utils import logger
+class Qwen3OmniMoeAudioEncoderConfig(PretrainedConfig):
+    model_type = "qwen3_omni_moe_audio_encoder"
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -14,41 +14,27 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0; `python/sglang/srt/configs/qwen3_omni.py` added +613/-0; `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168; `python/sglang/srt/models/qwen3_vl.py` modified +38/-24; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +40/-6; `python/sglang/srt/configs/qwen3_vl.py` modified +0/-10
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_b.py`, `test/srt/test_vision_openai_server_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11377 - [BugFix][Qwen3-VL]: add metadata for video in qwen3-vl

- 链接: https://github.com/sgl-project/sglang/pull/11377
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/qwen_vl.py`；关联提交 `fde2decf8b59`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-8，可读 patch 49 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][Qwen3-VL]: add metadata for video in qwen3-vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`；PR 正文摘要: In Qwen3's pre-processing pipeline within the transformers repository, `do_sample_frames=True` is enabled by default, which has already been handled by sglang itself, resulting...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8 (34 lines); hunks: -214,7 +214,14 @@ async def preprocess_video(; -279,14 +286,25 @@ async def process_mm_data_async(; symbols: preprocess_video, process_mm_data_async，涉及 `preprocess_video, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8 (34 lines); hunks: -214,7 +214,14 @@ async def preprocess_video(; -279,14 +286,25 @@ async def process_mm_data_async(; symbols: preprocess_video, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -214,7 +214,14 @@ async def preprocess_video(
-    return video
+    video_metadata = {
+        "fps": video_fps,
+        "duration": total_frames / video_fps,
+        "total_num_frames": total_frames,
+        "frames_indices": idx,
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/qwen_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12261 - [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl

- 链接: https://github.com/sgl-project/sglang/pull/12261
- 状态/时间: open / 2025-10-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-5，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: The implementation of cu_seqlens is incorrect and leads to accuracy issues with multi-frame/multi-patch inputs. This exact issue was already solved for Qwen3-VL in #11458 . This...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunks: -430,15 +430,15 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunks: -430,15 +430,15 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -430,15 +430,15 @@ def forward(
+        cu_seqlens = torch.repeat_interleave(
+            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
+        ).cumsum(dim=0)
-                torch.tensor([0], device=x.device, dtype=torch.int32),
-                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
-                .cumsum(dim=0)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12662 - [CPU] Add support for Qwen3-vl and Qwen3-omni

- 链接: https://github.com/sgl-project/sglang/pull/12662
- 状态/时间: open / 2025-11-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+496/-55，可读 patch 884 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Add support for Qwen3-vl and Qwen3-omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；PR 正文摘要: This PR added frontend support for Qwen3-VL and Qwen3-Omni on CPU. - Fixed the tp=3/6 padding issue in the vision encoder. - Replaced nn.Linear with ReplicatedLinear - Applied o...。
- 实现要点: `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunks: -16,9 +16,11; -30,9 +32,11; symbols: forward, VisionAMXAttention, __init__，涉及 `forward, VisionAMXAttention, __init__`；`python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunks: -189,28 +189,62 @@ def adjust_config_with_unaligned_cpu_tp(; symbols: adjust_config_with_unaligned_cpu_tp，涉及 `adjust_config_with_unaligned_cpu_tp`；`python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunks: -32,6 +32,7; -57,6 +58,65; symbols: hacked_preprocess, smart_resize，涉及 `hacked_preprocess, smart_resize`；`python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunks: -72,7 +72,14; -88,6 +95,10; symbols: Qwen3_VisionMLP, __init__, forward，涉及 `Qwen3_VisionMLP, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunks: -16,9 +16,11; -30,9 +32,11; symbols: forward, VisionAMXAttention, __init__
  - `python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunks: -189,28 +189,62 @@ def adjust_config_with_unaligned_cpu_tp(; symbols: adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunks: -32,6 +32,7; -57,6 +58,65; symbols: hacked_preprocess, smart_resize
  - `python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunks: -72,7 +72,14; -88,6 +95,10; symbols: Qwen3_VisionMLP, __init__, forward
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +34/-11 (45 lines); hunks: -32,7 +32,11; -43,7 +47,9; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/vision.py
@@ -16,9 +16,11 @@
+    cpu_has_amx_support,
+    is_cpu,
@@ -30,9 +32,11 @@
+_is_cpu = is_cpu()
+_is_cpu_amx_available = cpu_has_amx_support()
@@ -42,6 +46,9 @@
diff -- python/sglang/srt/configs/update_config.py
@@ -189,28 +189,62 @@ def adjust_config_with_unaligned_cpu_tp(
-    if (
-        hasattr(model_config.hf_config, "vision_config")
-        and model_config.hf_config.vision_config.model_type == "siglip_vision_model"
-    ):
-        model_config.hf_config.vision_config.original_num_attention_heads = (
-            model_config.num_attention_heads
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -32,6 +32,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +80/-12; `python/sglang/srt/configs/update_config.py` modified +54/-20; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0; `python/sglang/srt/models/qwen3_vl.py` modified +44/-6; `python/sglang/srt/models/qwen3_omni_moe.py` modified +34/-11; `python/sglang/srt/layers/conv.py` modified +19/-0
  - other: `sgl-kernel/csrc/cpu/gemm.cpp` modified +142/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/vision.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12703 - add qwen3-omni docs

- 链接: https://github.com/sgl-project/sglang/pull/12703
- 状态/时间: open / 2025-11-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+150/-0，可读 patch 158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add qwen3-omni docs」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`；PR 正文摘要: When I was fixing bugs related to qwen3-omni, I couldn't find any user manuals for qwen3-omni. The official qwen3-omni website only provides a vllm version. I also noticed that...。
- 实现要点: `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunks: -0,0 +1,149；`docs/index.rst` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ Its core features include:。
- 代码 diff 细节:
  - `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunks: -0,0 +1,149
  - `docs/index.rst` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ Its core features include:
- 关键代码摘录:

```diff
diff -- docs/basic_usage/qwen3_omni.md
@@ -0,0 +1,149 @@
+# Qwen3-Omni Usage
+[Qwen3-Omni](https://huggingface.co/collections/Qwen/qwen3-omni) is a natively end-to-end multilingual omni-modal foundation model. It processes text, images, audio, and video, an
+## Launch Qwen3-Omni with SGLang
+To serve Qwen3-Omni models on 4xH20 GPUs:
+'''bash
+python3 -m sglang.launch_server --model Qwen/Qwen3-Omni-30B-A3B-Instruct --tp 4
diff -- docs/index.rst
@@ -31,6 +31,7 @@ Its core features include:
+   basic_usage/qwen3_omni.md
```

- 已读文件:
  - docs: `docs/basic_usage/qwen3_omni.md` added +149/-0; `docs/index.rst` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #12240 - [VLM] Optimize qwen_vl preprocess_video

- 链接: https://github.com/sgl-project/sglang/pull/12240
- 状态/时间: merged / 2025-11-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/qwen_vl.py`；关联提交 `fd3034da7515`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+43/-10，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Optimize qwen_vl preprocess_video」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`；PR 正文摘要: The preprocess_video function can be divided into several stages. Through a combination of optimizations, each stage achieved varying degrees of performance improvement: get_bat...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10 (53 lines); hunks: -2,8 +2,10; -175,19 +177,25 @@ async def preprocess_video(; symbols: preprocess_video, process_mm_data_async，涉及 `preprocess_video, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10 (53 lines); hunks: -2,8 +2,10; -175,19 +177,25 @@ async def preprocess_video(; symbols: preprocess_video, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -2,8 +2,10 @@
+import time
+import numpy as np
@@ -175,19 +177,25 @@ async def preprocess_video(
+    entry_time = time.perf_counter()
-    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
-    video = vr.get_batch(idx).asnumpy()
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/qwen_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12554 - [Docs] Add docs for Qwen3-VL image and video support

- 链接: https://github.com/sgl-project/sglang/pull/12554
- 状态/时间: merged / 2025-11-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/qwen3_vl.md`；关联提交 `583bb1804e4c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+131/-0，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add docs for Qwen3-VL image and video support」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `docs/basic_usage/qwen3_vl.md`；PR 正文摘要: Added standalone page for Qwen3 VL model family. cc @mickqian。
- 实现要点: `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunks: -0,0 +1,130。
- 代码 diff 细节:
  - `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunks: -0,0 +1,130
- 关键代码摘录:

```diff
diff -- docs/basic_usage/qwen3_vl.md
@@ -0,0 +1,130 @@
+# Qwen3-VL Usage
+[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)
+is Alibaba’s latest multimodal large language model with strong text, vision, and reasoning capabilities.
+SGLang supports Qwen3-VL Family of models with Image and Video input support.
+## Launch commands for SGLang
+Below are suggested launch commands tailored for different hardware / precision modes
```

- 已读文件:
  - docs: `docs/basic_usage/qwen3_vl.md` added +130/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/qwen3_vl.md`, `docs/index.rst`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #12458 - fix: duplicate resize images logic of qwen-vl series models

- 链接: https://github.com/sgl-project/sglang/pull/12458
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/qwen_vl.py`；关联提交 `ffeb28ba6fb7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+1/-67，可读 patch 136 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: duplicate resize images logic of qwen-vl series models」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`；PR 正文摘要: fix https://github.com/sgl-project/sglang/issues/12390 and https://github.com/sgl-project/sglang/issues/11896 1. set `MIN_PIXELS`, `MAX_PIXELS`, `IMAGE_FACTOR` according to mode...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40 (40 lines); hunks: -1,4 +1,3; -79,26 +78,6 @@ def smart_resize(; symbols: smart_resize, resize_image, round_by_factor, floor_by_factor，涉及 `smart_resize, resize_image, round_by_factor`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40 (40 lines); hunks: -1,4 +1,3; -79,26 +78,6 @@ def smart_resize(; symbols: smart_resize, resize_image, round_by_factor, floor_by_factor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -1,4 +1,3 @@
-import asyncio
@@ -79,26 +78,6 @@ def smart_resize(
-def resize_image(
-    image,
-    min_pixels: int = MIN_PIXELS,
-    max_pixels: int = MAX_PIXELS,
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/dots_vlm.py`, `python/sglang/srt/multimodal/processors/points_v15_chat.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13075 - [VLM] Support PP for Qwen2.5-VL

- 链接: https://github.com/sgl-project/sglang/pull/13075
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`；关联提交 `706502ff6cff`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+88/-54，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support PP for Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: This PR is to support PP for Qwen2.5-VL model.。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15 (59 lines); hunks: -40,6 +40,7; -50,13 +51,14; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15 (59 lines); hunks: -40,6 +40,7; -50,13 +51,14; symbols: __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -40,6 +40,7 @@
+from sglang.srt.distributed.parallel_state import get_pp_group
@@ -50,13 +51,14 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
@@ -482,6 +484,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13126 - [VLM][feat] Support encoder DP for Qwen2.5-VL

- 链接: https://github.com/sgl-project/sglang/pull/13126
- 状态/时间: merged / 2025-11-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`；关联提交 `ac81db66c2d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+595/-6，可读 patch 790 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM][feat] Support encoder DP for Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: Based on this PR https://github.com/vllm-project/vllm/pull/22742, vLLM has introduced support for data parallelism (DP) in the vision transformer (ViT) component while maintaini...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -40,6 +40,10 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -62,6 +66,8 @@
+from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2
- 验证与风险: diff 自带测试面 `test/srt/nightly/test_encoder_dp.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13055 - [VLM] Support Piecewise CUDA Graph for Qwen2.5-VL

- 链接: https://github.com/sgl-project/sglang/pull/13055
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；关联提交 `af6bcadcf723`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+710/-29，可读 patch 970 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support Piecewise CUDA Graph for Qwen2.5-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；PR 正文摘要: Address https://github.com/sgl-project/sglang/issues/12838 This PR cooperates closely with @yhyang201 . Per comparing the SGLang and vLLM performance on VLM, one of the key infl...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7 (45 lines); hunks: -57,11 +57,12; -566,6 +567,25 @@ def get_video_feature(self, items: List[MultimodalDataItem]...; symbols: get_video_feature, post_process, get_input_embeddings, forward，涉及 `get_video_feature, post_process, get_input_embeddings`；`python/sglang/srt/models/qwen2_vl.py` modified +16/-7 (23 lines); hunks: -39,10 +39,7; -509,6 +506,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7 (45 lines); hunks: -57,11 +57,12; -566,6 +567,25 @@ def get_video_feature(self, items: List[MultimodalDataItem]...; symbols: get_video_feature, post_process, get_input_embeddings, forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +16/-7 (23 lines); hunks: -39,10 +39,7; -509,6 +506,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -57,11 +57,12 @@
-from sglang.srt.managers.mm_utils import (
-    MultiModalityDataPaddingPatternMultimodalTokens,
-    general_mm_embed_routine,
+from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
+from sglang.srt.managers.schedule_batch import (
+    Modality,
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -39,10 +39,7 @@
-from sglang.srt.managers.mm_utils import (
-    MultiModalityDataPaddingPatternMultimodalTokens,
-    general_mm_embed_routine,
-)
+from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
@@ -509,6 +506,7 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7; `python/sglang/srt/models/qwen2_vl.py` modified +16/-7
- 验证与风险: diff 自带测试面 `test/srt/nightly/test_vlms_piecewise_cuda_graph.py`, `test/srt/run_suite.py`, `test/srt/test_piecewise_cuda_graph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13736 - [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series

- 链接: https://github.com/sgl-project/sglang/pull/13736
- 状态/时间: merged / 2025-11-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；关联提交 `5625e32cae12`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+169/-13，可读 patch 229 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_vl.py`；PR 正文摘要: Replace `torch.repeat_interleave` with `faster np.repeat` for Qwen2-VL, Qwen3-VL, Qwen3-VL-MoE. E2E TTFT reduces 1.5%. gsm8k matched. More benchmark and profile will be updated.。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -434,15 +435,7 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunks: -44,6 +44,7; -387,10 +388,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -434,15 +435,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunks: -44,6 +44,7; -387,10 +388,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -46,6 +46,7 @@
+from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
@@ -434,15 +435,7 @@ def forward(
-        cu_seqlens = torch.repeat_interleave(
-            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
-        ).cumsum(dim=0)
-        cu_seqlens = torch.cat(
diff -- python/sglang/srt/models/qwen2_vl.py
@@ -44,6 +44,7 @@
+from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
@@ -387,10 +388,7 @@ def forward(
-        cu_seqlens = torch.repeat_interleave(
-            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
-        ).cumsum(dim=0, dtype=torch.int32)
-        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +2/-9; `python/sglang/srt/models/qwen2_vl.py` modified +2/-4
- 验证与风险: diff 自带测试面 `test/srt/ops/test_repeat_interleave.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13918 - [VLM] support qwen3-vl eagle infer

- 链接: https://github.com/sgl-project/sglang/pull/13918
- 状态/时间: open / 2025-11-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+30/-3，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] support qwen3-vl eagle infer」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`；PR 正文摘要: Support qwen3-vl dense model eagle infer. use qwen3-vl-8b eagle model for test. server: acc_bench: result: - with eagle: | Tasks |Version|Filter|n-shot| Metric | |Value | |Stder...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunks: -624,6 +624,9 @@ def __init__(; -711,9 +714,13 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights，涉及 `__init__, separate_deepstack_embeds, forward`；`python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunks: -116,9 +116,14 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunks: -624,6 +624,9 @@ def __init__(; -711,9 +714,13 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunks: -116,9 +116,14 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -624,6 +624,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
@@ -711,9 +714,13 @@ def forward(
+        aux_hidden_states = None
+        if self.capture_aux_hidden_states:
+            hidden_states, aux_hidden_states = hidden_states
diff -- python/sglang/srt/models/llama_eagle3.py
@@ -116,9 +116,14 @@ def __init__(
-        # fix rope_scaling for qwen2.5-vl
+        # fix rope_scaling for qwen2.5-vl/qwen3-vl
-            config.rope_scaling["rope_type"] = "default"
+            rope_scaling = config.rope_scaling
+            self.mrope_interleaved = rope_scaling.setdefault("mrope_interleaved", False)
+            if not self.mrope_interleaved:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +23/-1; `python/sglang/srt/models/llama_eagle3.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama_eagle3.py`, `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13983 - Support KTransformers for Qwen3-VL moe

- 链接: https://github.com/sgl-project/sglang/pull/13983
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl_moe.py`；关联提交 `15ff6982b94f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-0，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support KTransformers for Qwen3-VL moe」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: This pr adds KTransformers Support for qwen3-vl moe models add `get_model_config_for_expert_location` in `Qwen3VLMoeForConditionalGeneration`。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0 (9 lines); hunks: -22,6 +22,7; -326,5 +327,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location，涉及 `load_weights, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0 (9 lines); hunks: -22,6 +22,7; -326,5 +327,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -22,6 +22,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -326,5 +327,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+    @classmethod
+    def get_model_config_for_expert_location(cls, config):
+        return ModelConfigForExpertLocation(
+            num_layers=config.text_config.num_hidden_layers,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13904 - [Bugfix] qwen2.5-vl spec decode accept_len low

- 链接: https://github.com/sgl-project/sglang/pull/13904
- 状态/时间: merged / 2025-11-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`；关联提交 `f6e37d3edb94`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] qwen2.5-vl spec decode accept_len low」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: After this pr, qwen2.5-vl eagle3 infer draft accept_len is low. Because `aux_hidden_states` is miss in logits_processor. Releated issue: https://github.com/sgl-project/SpecForge...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -654,6 +654,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -654,6 +654,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -654,6 +654,7 @@ def forward(
+                    aux_hidden_states,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13724 - support qwen3_vl vision model dp

- 链接: https://github.com/sgl-project/sglang/pull/13724
- 状态/时间: merged / 2025-11-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `ea1e9f6b3c3d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+50/-2，可读 patch 208 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3_vl vision model dp」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Based on PR 13126, add support for the Qwen3_VL vision model DP. Qwen/Qwen3-VL-32B-Instruct server cmd: bench cmd：。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -28,6 +28,10 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -47,6 +51,8 @@
+from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2
- 验证与风险: diff 自带测试面 `test/nightly/test_encoder_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14292 - [VLM] Introduce Cache for positional embedding ids for Qwen-VL family

- 链接: https://github.com/sgl-project/sglang/pull/14292
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；关联提交 `b2b09f5f24b9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+48/-47，可读 patch 149 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Introduce Cache for positional embedding ids for Qwen-VL family」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Introduce a cache for rot_pos_emb index computation to boost the calculation. Introduce a mixin class for broader reuse for this mechanism. For cached rotary position embedding,...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunks: -69,7 +69,7; -246,7 +246,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen2_5_VisionTransformer, __init__, device，涉及 `forward, Qwen2_5_VisionTransformer, __init__`；`python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunks: -50,7 +50,7; -257,7 +257,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen3VLMoeVisionModel, __init__, device，涉及 `forward, Qwen3VLMoeVisionModel, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunks: -69,7 +69,7; -246,7 +246,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen2_5_VisionTransformer, __init__, device
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunks: -50,7 +50,7; -257,7 +257,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen3VLMoeVisionModel, __init__, device
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -69,7 +69,7 @@
-from sglang.srt.models.utils import permute_inv
+from sglang.srt.models.utils import RotaryPosMixin, permute_inv
@@ -246,7 +246,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-class Qwen2_5_VisionTransformer(nn.Module):
+class Qwen2_5_VisionTransformer(nn.Module, RotaryPosMixin):
@@ -362,30 +362,10 @@ def device(self) -> torch.device:
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -50,7 +50,7 @@
-from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
+from sglang.srt.models.utils import RotaryPosMixin, compute_cu_seqlens_from_grid_numpy
@@ -257,7 +257,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-class Qwen3VLMoeVisionModel(nn.Module):
+class Qwen3VLMoeVisionModel(nn.Module, RotaryPosMixin):
@@ -339,26 +339,9 @@ def device(self) -> torch.device:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25; `python/sglang/srt/models/qwen3_vl.py` modified +5/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11791 - fix rmsnorm -> layernorm in qwen3 omni

- 链接: https://github.com/sgl-project/sglang/pull/11791
- 状态/时间: merged / 2025-12-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_omni_moe.py`；关联提交 `2ac5b9839508`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix rmsnorm -> layernorm in qwen3 omni」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_omni_moe.py`；PR 正文摘要: If you look at the transformer's modeling code, then you will notice that it's actually supposed to be layer norm instead of RMS norm. You can transform RMS norm to be the same...。
- 实现要点: `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2 (3 lines); hunks: -31,7 +31,6; -318,7 +317,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2 (3 lines); hunks: -31,7 +31,6; -318,7 +317,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -31,7 +31,6 @@
-from sglang.srt.layers.layernorm import RMSNorm
@@ -318,7 +317,7 @@ def __init__(
-        self.ln_q = RMSNorm(
+        self.ln_q = nn.LayerNorm(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_omni_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14886 - Support qwen3-omni with DP Encoder

- 链接: https://github.com/sgl-project/sglang/pull/14886
- 状态/时间: open / 2025-12-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+32/-3，可读 patch 151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support qwen3-omni with DP Encoder」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`；PR 正文摘要: Add support for Qwen3-Omni with vision model DP based on #13724 Tested Qwen/Qwen3-Omni-30B-A3B-Instruct with dp | Tasks |Version|Filter|n-shot| Metric | |Value | |Stderr| |-----...。
- 实现要点: `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunks: -30,6 +30,10; -42,6 +46,7; symbols: __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder，涉及 `__init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder`；`test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -21,6 +21,7。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunks: -30,6 +30,10; -42,6 +46,7; symbols: __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -21,6 +21,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -30,6 +30,10 @@
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -42,6 +46,7 @@
+from sglang.srt.server_args import get_global_server_args
diff -- test/nightly/test_encoder_dp.py
@@ -21,6 +21,7 @@
+    SimpleNamespace(model="Qwen/Qwen3-Omni-30B-A3B-Instruct", mmmu_accuracy=0.55),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3
  - tests: `test/nightly/test_encoder_dp.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/nightly/test_encoder_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14907 - [VLM] Support chunked vit attention

- 链接: https://github.com/sgl-project/sglang/pull/14907
- 状态/时间: merged / 2025-12-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+363/-8，可读 patch 413 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support chunked vit attention」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/mm_utils.py`；PR 正文摘要: Inspired by @merrymercy @mickqian . One request with 500 images (or a video with 500 frames) on Qwen3-VL-235B-A22B-Instruct-FP8 will have OOM. The PR have two parts: - Adds supp...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunks: -14,6 +14,7; -53,7 +54,7; symbols: get_image_feature, get_video_feature，涉及 `get_image_feature, get_video_feature`；`python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunks: -41,6 +41,9; -414,6 +417,67 @@ def _get_precomputed_embedding(; symbols: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding，涉及 `init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunks: -14,6 +14,7; -53,7 +54,7; symbols: get_image_feature, get_video_feature
  - `python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunks: -41,6 +41,9; -414,6 +417,67 @@ def _get_precomputed_embedding(; symbols: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -14,6 +14,7 @@
+import math
@@ -53,7 +54,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, get_int_env_var
@@ -666,13 +667,101 @@ def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
-        if self.use_data_parallel:
diff -- python/sglang/srt/managers/mm_utils.py
@@ -41,6 +41,9 @@
+_EXTRA_PRE_TOKENS = 0  # pre chunk extra token (0 for the moment)
+_EXTRA_POST_TOKENS = 0  # post chunk extra token (0 for the moment)
@@ -414,6 +417,67 @@ def _get_precomputed_embedding(
+def get_embedding_items_per_chunk_with_extra_padding(
+    embedding_items_per_req: List["MultimodalDataItem"],
+    extend_prefix_len: int,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +97/-8; `python/sglang/srt/managers/mm_utils.py` modified +266/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12333 - [PP] Add pp support for Qwen3-VL

- 链接: https://github.com/sgl-project/sglang/pull/12333
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/models/qwen3_vl.py`；关联提交 `45a959d3e971`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+119/-20，可读 patch 243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PP] Add pp support for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py`；PR 正文摘要: see issue 11947 test with command : `python3 -m sglang.launch_server --model Qwen/Qwen3-VL-8B-Thinking --tp 2 --pp-size 2`。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunks: -32,11 +32,13; -599,6 +601,7 @@ def __init__(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunks: -614,7 +614,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunks: -32,11 +32,13; -599,6 +601,7 @@ def __init__(; symbols: __init__, forward, load_weights
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunks: -614,7 +614,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -32,11 +32,13 @@
+from sglang.srt.distributed.parallel_state import get_pp_group
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -599,6 +601,7 @@ def __init__(
+        self.pp_group = get_pp_group()
@@ -626,19 +629,22 @@ def __init__(
-            if self.config.tie_word_embeddings:
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -614,7 +614,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                        param = params_dict[name_mapped]
+                        if name_mapped in params_dict.keys():
+                            param = params_dict[name_mapped]
+                        else:
+                            continue
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +53/-19; `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15138 - [bug fix][pp] fix weight load for qwen2.5-vl

- 链接: https://github.com/sgl-project/sglang/pull/15138
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`；关联提交 `0071fe9c407a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-4，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bug fix][pp] fix weight load for qwen2.5-vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: Now， There are two bug for Qwen2.5-vl 1. when loading weights for Qwen2.5-vl, errors ouucrs(brought by EPD feature): File "/root/leipi/sglang/python/sglang/srt/managers/schedule...。
- 实现要点: `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4 (14 lines); hunks: -743,6 +743,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -789,10 +797,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4 (14 lines); hunks: -743,6 +743,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -789,10 +797,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -743,6 +743,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
+                if "lm_head.weight" in params_dict:
+                    lm_head_param = params_dict["lm_head.weight"]
+                    weight_loader = getattr(
+                        lm_head_param, "weight_loader", default_weight_loader
+                    )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_5_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- 链接: https://github.com/sgl-project/sglang/pull/15205
- 状态/时间: merged / 2025-12-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `8fa3dc36c565`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+100/-80，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Support cos sin cache for Qwen3-VL & GLM-4.1V. This PR refactors the rotary positional embedding (RoPE) implementation to expose an explicit cosine/sine cache interface and reus...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device，涉及 `forward, __init__, dtype`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -24,9 +24,6 @@
-from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
-    Qwen2_5_VisionRotaryEmbedding,
-)
@@ -39,6 +36,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -188,14 +186,16 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +41/-20
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4v.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15320 - [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL

- 链接: https://github.com/sgl-project/sglang/pull/15320
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；关联提交 `019517a35610`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+233/-64，可读 patch 497 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`；PR 正文摘要: This PR is to enable ViT Piecewise CUDA Graph for Qwen3-VL. Building logic upon ViTCudaGraphRunner to support both Qwen2.5-VL and Qwen3-VL. TP>1 is supported. Benchmark show 8xH...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunks: -57,8 +57,9; -188,6 +189,7 @@ def forward(; symbols: forward, __init__, dtype，涉及 `forward, __init__, dtype`；`python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -170,7 +170,6 @@ def forward(; -182,7 +181,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunks: -57,8 +57,9; -188,6 +189,7 @@ def forward(; symbols: forward, __init__, dtype
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -170,7 +170,6 @@ def forward(; -182,7 +181,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -57,8 +57,9 @@
+from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
-from sglang.srt.utils import add_prefix, get_int_env_var
+from sglang.srt.utils import add_prefix, get_bool_env_var, get_int_env_var
@@ -188,6 +189,7 @@ def forward(
+        output_ws: Optional[torch.Tensor] = None,
@@ -196,6 +198,7 @@ def forward(
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -170,7 +170,6 @@ def forward(
-        ws = output_ws
@@ -182,7 +181,7 @@ def forward(
-            output_ws=ws,
+            output_ws=output_ws,
@@ -390,7 +389,7 @@ def forward(
-        if get_bool_env_var("SGLANG_VIT_ENABLE_CUDA_GRAPH") and self.tp_size == 1:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +52/-1; `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3
- 验证与风险: diff 自带测试面 `test/manual/nightly/test_vlms_vit_cuda_graph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15570 - [GLM-ASR] GLM-ASR Support

- 链接: https://github.com/sgl-project/sglang/pull/15570
- 状态/时间: merged / 2025-12-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`；关联提交 `82f1d6157f8e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+226/-1，可读 patch 250 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-ASR] GLM-ASR Support」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`；PR 正文摘要: Note that this PR needs to wait for support from the transformers PR PR。
- 实现要点: `python/sglang/srt/models/glmasr.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: GlmasrForConditionalGeneration, __init__, pad_input_ids, get_audio_feature，涉及 `GlmasrForConditionalGeneration, __init__, pad_input_ids`；`python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: GlmasrProcessor, __init__, process_mm_data_async，涉及 `GlmasrProcessor, __init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glmasr.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: GlmasrForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: GlmasrProcessor, __init__, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glmasr.py
@@ -0,0 +1,171 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/glmasr.py
@@ -0,0 +1,53 @@
+import re
+from sglang.srt.models.glmasr import GlmasrForConditionalGeneration
+from sglang.srt.multimodal.processors.base_processor import (
+    BaseMultimodalProcessor,
+    MultimodalSpecialTokens,
+)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glmasr.py` added +171/-0; `python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/base_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15772 - Change GLM-ASR class name

- 链接: https://github.com/sgl-project/sglang/pull/15772
- 状态/时间: merged / 2025-12-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`；关联提交 `f3ba71166262`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+19/-15，可读 patch 103 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Change GLM-ASR class name」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`；PR 正文摘要: The class name for GLM-ASR was changed in the transformers PR, so the original code will be unable to import and find the processor.。
- 实现要点: `python/sglang/srt/models/glmasr.py` modified +9/-9 (18 lines); hunks: -22,10 +22,10; -46,7 +46,7; symbols: GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__，涉及 `GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__`；`python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3 (6 lines); hunks: -1,14 +1,14; symbols: GlmasrProcessor, GlmAsrProcessor, __init__，涉及 `GlmasrProcessor, GlmAsrProcessor, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glmasr.py` modified +9/-9 (18 lines); hunks: -22,10 +22,10; -46,7 +46,7; symbols: GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3 (6 lines); hunks: -1,14 +1,14; symbols: GlmasrProcessor, GlmAsrProcessor, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glmasr.py
@@ -22,10 +22,10 @@
-from transformers import GlmasrConfig, GlmasrEncoderConfig
+from transformers import GlmAsrConfig, GlmAsrEncoderConfig
-    GlmasrEncoder,
-    GlmasrMultiModalProjector,
+    GlmAsrEncoder,
+    GlmAsrMultiModalProjector,
diff -- python/sglang/srt/multimodal/processors/glmasr.py
@@ -1,14 +1,14 @@
-from sglang.srt.models.glmasr import GlmasrForConditionalGeneration
+from sglang.srt.models.glmasr import GlmAsrForConditionalGeneration
-class GlmasrProcessor(BaseMultimodalProcessor):
-    models = [GlmasrForConditionalGeneration]
+class GlmAsrProcessor(BaseMultimodalProcessor):
+    models = [GlmAsrForConditionalGeneration]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glmasr.py` modified +9/-9; `python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/models/glmasr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16491 - [Qwen3-VL][PP] Skip loading expert weights not on this rank

- 链接: https://github.com/sgl-project/sglang/pull/16491
- 状态/时间: open / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-VL][PP] Skip loading expert weights not on this rank」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: Failed to run Qwen3-vl-235B-A22B-FP8 with pp=2, tp=4 Fix expert weights loading logic, skip loading expert weights not on this PP rank.。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunks: -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunks: -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    if name_mapped not in params_dict:
+                        # Expert weight not on this rank, will be skipped below
+                        continue
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16571 - [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT

- 链接: https://github.com/sgl-project/sglang/pull/16571
- 状态/时间: open / 2026-01-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+87/-15，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`；PR 正文摘要: Support Add & LayerNorm fused for Qwen3.5-Next VIT before： after： **layernorm.py :** - In LayerNorm.__init__: when _use_aiter is True, set _forward_method = self.forward_aiter -...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunks: -42,6 +42,7; -72,7 +73,7; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLMoeVisionPatchMerger，涉及 `Qwen3_VisionMLP, __init__, forward`；`python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunks: -67,6 +67,7; -367,6 +368,9 @@ def __init__(; symbols: __init__, forward_cuda, forward_cpu, forward_aiter，涉及 `__init__, forward_cuda, forward_cpu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunks: -42,6 +42,7; -72,7 +73,7; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLMoeVisionPatchMerger
  - `python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunks: -67,6 +67,7; -367,6 +368,9 @@ def __init__(; symbols: __init__, forward_cuda, forward_cpu, forward_aiter
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -42,6 +42,7 @@
+from sglang.srt.layers.layernorm import LayerNorm
@@ -72,7 +73,7 @@
-from sglang.srt.utils import add_prefix, get_int_env_var, is_npu, round_up
+from sglang.srt.utils import add_prefix, get_int_env_var, is_hip, is_npu, round_up
@@ -86,6 +87,8 @@
+_is_hip = is_hip()
diff -- python/sglang/srt/layers/layernorm.py
@@ -67,6 +67,7 @@
+    from aiter import layer_norm, layernorm2d_fwd_with_add
@@ -367,6 +368,9 @@ def __init__(
+        if _use_aiter:
+            self._forward_method = self.forward_aiter
@@ -419,6 +423,29 @@ def forward_cpu(
+    def forward_aiter(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +60/-15; `python/sglang/srt/layers/layernorm.py` modified +27/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16785 - [Bugfix] fix recompile in qwen3 vl

- 链接: https://github.com/sgl-project/sglang/pull/16785
- 状态/时间: open / 2026-01-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+113/-36，可读 patch 307 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix recompile in qwen3 vl」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；PR 正文摘要: Qwen3-VL automatically injects input_deepstack_embeds into the language model inputs when requests contain mm_inputs (multimodal inputs). The current implementation does not han...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunks: -913,16 +913,37 @@ def __init__(; -963,7 +984,7 @@ def forward(; symbols: __init__, get_deepstack_embeds, forward，涉及 `__init__, get_deepstack_embeds, forward`；`python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunks: -57,19 +57,40 @@ def __init__(; -109,7 +130,7 @@ def forward(; symbols: __init__, get_input_embeddings, get_deepstack_embeds, forward，涉及 `__init__, get_input_embeddings, get_deepstack_embeds`；`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunks: -245,6 +245,18 @@ def __init__(self, model_runner: ModelRunner):; -392,6 +404,9 @@ def warmup_torch_compile(self, num_tokens: int):; symbols: __init__, warmup_torch_compile, _cache_loc_dtype, capture_one_batch_size，涉及 `__init__, warmup_torch_compile, _cache_loc_dtype`；`python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunks: -919,6 +919,7 @@ def embed_mm_inputs(; -1019,12 +1020,16 @@ def embed_mm_inputs(; symbols: embed_mm_inputs, general_mm_embed_routine，涉及 `embed_mm_inputs, general_mm_embed_routine`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunks: -913,16 +913,37 @@ def __init__(; -963,7 +984,7 @@ def forward(; symbols: __init__, get_deepstack_embeds, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunks: -57,19 +57,40 @@ def __init__(; -109,7 +130,7 @@ def forward(; symbols: __init__, get_input_embeddings, get_deepstack_embeds, forward
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunks: -245,6 +245,18 @@ def __init__(self, model_runner: ModelRunner):; -392,6 +404,9 @@ def warmup_torch_compile(self, num_tokens: int):; symbols: __init__, warmup_torch_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunks: -919,6 +919,7 @@ def embed_mm_inputs(; -1019,12 +1020,16 @@ def embed_mm_inputs(; symbols: embed_mm_inputs, general_mm_embed_routine
  - `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3 (8 lines); hunks: -18,7 +18,9; -59,7 +61,7 @@ def run_mmmu_eval(; symbols: run_mmmu_eval, _run_vlm_mmmu_test
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -913,16 +913,37 @@ def __init__(
+        self.deepstack_embeds_buffer = None
-        self, layer_idx: int, input_deepstack_embeds: Optional[torch.Tensor]
+        self,
+        layer_idx: int,
+        input_deepstack_embeds: Optional[torch.Tensor],
+        seq_len: int,
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -57,19 +57,40 @@ def __init__(
+        self.deepstack_embeds_buffer = None
-        self, layer_idx: int, input_deepstack_embeds: Optional[torch.Tensor]
+        self,
+        layer_idx: int,
+        input_deepstack_embeds: Optional[torch.Tensor],
+        seq_len: int,
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -245,6 +245,18 @@ def __init__(self, model_runner: ModelRunner):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +41/-18; `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1; `python/sglang/srt/managers/mm_utils.py` modified +13/-6
  - tests: `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3
- 验证与风险: diff 自带测试面 `test/manual/nightly/test_vlms_piecewise_cuda_graph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16996 - feat: Support 'use_audio_in_video' option for qwen3omnimoe model

- 链接: https://github.com/sgl-project/sglang/pull/16996
- 状态/时间: open / 2026-01-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+129/-12，可读 patch 330 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Support 'use_audio_in_video' option for qwen3omnimoe model」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/entrypoints/openai/protocol.py`；PR 正文摘要: This PR adds support for the `use_audio_in_video` configuration parameter in the `Qwen3-Omni-Moe` model. - Passed this parameter to the input of 'Engine' class and openai reques...。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunks: -324,7 +324,22 @@ def process_mm_data(; -405,6 +420,7 @@ def _load_single_item(; symbols: process_mm_data, _load_single_item, submit_data_loading_tasks，涉及 `process_mm_data, _load_single_item, submit_data_loading_tasks`；`python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunks: -151,6 +151,7 @@ async def preprocess_video(; -201,8 +202,16 @@ async def preprocess_video(; symbols: preprocess_video, __init__, process_mm_data_async，涉及 `preprocess_video, __init__, process_mm_data_async`；`python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunks: -552,6 +552,8 @@ class ChatCompletionRequest(BaseModel):; -698,6 +700,7 @@ def get_param(param_name: str):; symbols: ChatCompletionRequest, get_param，涉及 `ChatCompletionRequest, get_param`；`python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunks: -72,6 +72,7; -878,6 +879,43 @@ def load_audio(; symbols: load_audio, extract_audio_via_av, ImageData, get_image_bytes，涉及 `load_audio, extract_audio_via_av, ImageData`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunks: -324,7 +324,22 @@ def process_mm_data(; -405,6 +420,7 @@ def _load_single_item(; symbols: process_mm_data, _load_single_item, submit_data_loading_tasks
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunks: -151,6 +151,7 @@ async def preprocess_video(; -201,8 +202,16 @@ async def preprocess_video(; symbols: preprocess_video, __init__, process_mm_data_async
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunks: -552,6 +552,8 @@ class ChatCompletionRequest(BaseModel):; -698,6 +700,7 @@ def get_param(param_name: str):; symbols: ChatCompletionRequest, get_param
  - `python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunks: -72,6 +72,7; -878,6 +879,43 @@ def load_audio(; symbols: load_audio, extract_audio_via_av, ImageData, get_image_bytes
  - `python/sglang/srt/managers/io_struct.py` modified +3/-0 (3 lines); hunks: -273,6 +273,9 @@ class GenerateReqInput(BaseReq, APIServingTimingMixin):; symbols: GenerateReqInput, contains_mm_input
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -324,7 +324,22 @@ def process_mm_data(
+        # For Qwen3_Omni: construct correct FPS and use_audio_in_video kwargs
+        if self._processor.__class__.__name__ == "Qwen3OmniMoeProcessor" and videos:
+            videos_kwargs = kwargs.get("videos_kwargs", {})
+            video_metadata = kwargs.get("video_metadata")
+            if video_metadata and "fps" not in videos_kwargs:
+                if isinstance(video_metadata, list):
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -151,6 +151,7 @@ async def preprocess_video(
+    duration = total_frames / video_fps if video_fps > 0 else 0
@@ -201,8 +202,16 @@ async def preprocess_video(
+    # Fix: correct fps from sampled video
+    if duration > 0:
+        effective_fps = round(nframes / duration, 1)
+    else:
diff -- python/sglang/srt/entrypoints/openai/protocol.py
@@ -552,6 +552,8 @@ class ChatCompletionRequest(BaseModel):
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2; `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0; `python/sglang/srt/utils/common.py` modified +63/-7; `python/sglang/srt/managers/io_struct.py` modified +3/-0; `python/pyproject_npu.toml` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/pyproject_npu.toml`, `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/managers/io_struct.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17202 - [Feat] Accelerate qwen3vl by remove cpu op

- 链接: https://github.com/sgl-project/sglang/pull/17202
- 状态/时间: open / 2026-01-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+27/-9，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feat] Accelerate qwen3vl by remove cpu op」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py`；PR 正文摘要: Accelerate qwen3vl by delete contiguous in vision_attention && delete D2D in torch.where and index_put During the forward pass of the ViT in the Qwen3VL multimodal model, some D...。
- 实现要点: `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunks: -1005,9 +1005,9 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunks: -1029,18 +1029,36 @@ def embed_mm_inputs(; symbols: embed_mm_inputs，涉及 `embed_mm_inputs`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunks: -1005,9 +1005,9 @@ def forward(; symbols: forward
  - `python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunks: -1029,18 +1029,36 @@ def embed_mm_inputs(; symbols: embed_mm_inputs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/vision.py
@@ -1005,9 +1005,9 @@ def forward(
-            q = q.reshape(bsz * s, head, -1).contiguous()
-            k = k.reshape(bsz * s, kv_head, -1).contiguous()
-            v = v.reshape(bsz * s, kv_head, -1).contiguous()
+            q = q.reshape(bsz * s, head, -1)
+            k = k.reshape(bsz * s, kv_head, -1)
+            v = v.reshape(bsz * s, kv_head, -1)
diff -- python/sglang/srt/managers/mm_utils.py
@@ -1029,18 +1029,36 @@ def embed_mm_inputs(
+    # Use masked_scatter_ to completely avoid D2D synchronization
-        # in-place update
-        indices = torch.where(mask.squeeze(dim=-1))[0]
-        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)
+        mask_1d = mask.view(-1)
+        # Convert embedding to target device/dtype if needed
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +3/-3; `python/sglang/srt/managers/mm_utils.py` modified +24/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17276 - Add Qwen3VL Eagle3 Inference Support

- 链接: https://github.com/sgl-project/sglang/pull/17276
- 状态/时间: open / 2026-01-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+35/-0，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3VL Eagle3 Inference Support」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Add support for EAGLE3 for Qwen3VL. I have not extensively tried to optimize benchmark performance, but the accuracy/performance numbers at the bottom are there to show that inf...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunks: -733,6 +733,9 @@ def __init__(; -920,13 +923,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights，涉及 `__init__, separate_deepstack_embeds, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunks: -733,6 +733,9 @@ def __init__(; -920,13 +923,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -733,6 +733,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
@@ -920,13 +923,18 @@ def forward(
+        aux_hidden_states = None
+        if self.capture_aux_hidden_states:
+            hidden_states, aux_hidden_states = hidden_states
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +35/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16366 - Optimize Qwen3-VL video memory usage

- 链接: https://github.com/sgl-project/sglang/pull/16366
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `0c2993eed03a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize Qwen3-VL video memory usage」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: We observed that when launching the Qwen3-Omni-30B-A3B-Instruct model with sglang, an OOM (Out-of-Memory) error occurs under high-concurrency inference. After analyzing the prof...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunks: -852,10 +852,18 @@ def get_image_feature(self, items: List[MultimodalDataItem...; symbols: get_image_feature, get_video_feature，涉及 `get_image_feature, get_video_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunks: -852,10 +852,18 @@ def get_image_feature(self, items: List[MultimodalDataItem...; symbols: get_image_feature, get_video_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -852,10 +852,18 @@ def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
+        for item in items:
+            item.feature = item.feature.to(self.visual.device)
+        # Memory optimization for item.feature:
+        # 1. item.feature is released when request finished
+        # 2. High concurrency may cause device OOM due to delayed release
+        # 3. Fix: Offload item.feature to CPU, move to device only when needed
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- 链接: https://github.com/sgl-project/sglang/pull/17624
- 状态/时间: merged / 2026-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+48/-19，可读 patch 185 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] Fix dp size > 1 for qwen3 vl model」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`；PR 正文摘要: Question In my local tests, the Qwen3-VL service fails to start when both `--mm-enable-dp-encoder` and `--enable-dp-attention` are enabled. I came across the related PR: pr17157...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunks: -25,14 +25,15; -85,10 +86,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunks: -495,11 +495,19 @@ def run_dp_sharded_mrope_vision_model(; -611,7 +619,9 @@ def run_dp_sharded_mrope_vision_model(; symbols: run_dp_sharded_mrope_vision_model，涉及 `run_dp_sharded_mrope_vision_model`；`python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -1262,6 +1265,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunks: -860,7 +860,15 @@ def _pad_inputs_to_size(self, model_runner: ModelRunner, nu...; symbols: _pad_inputs_to_size，涉及 `_pad_inputs_to_size`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunks: -25,14 +25,15; -85,10 +86,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunks: -495,11 +495,19 @@ def run_dp_sharded_mrope_vision_model(; -611,7 +619,9 @@ def run_dp_sharded_mrope_vision_model(; symbols: run_dp_sharded_mrope_vision_model
  - `python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -1262,6 +1265,7 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunks: -860,7 +860,15 @@ def _pad_inputs_to_size(self, model_runner: ModelRunner, nu...; symbols: _pad_inputs_to_size
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-0 (2 lines); hunks: -538,6 +538,7 @@ def __init__(; -640,6 +641,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -25,14 +25,15 @@
-from sglang.srt.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
-)
+from sglang.srt.distributed import get_tensor_model_parallel_world_size
-from sglang.srt.layers.dp_attention import is_dp_attention_enabled
diff -- python/sglang/srt/multimodal/mm_utils.py
@@ -495,11 +495,19 @@ def run_dp_sharded_mrope_vision_model(
-    tp_size = get_tensor_model_parallel_world_size()
+    from sglang.srt.layers.dp_attention import (
+        get_attention_tp_group,
+        get_attention_tp_rank,
+        get_attention_tp_size,
+    )
diff -- python/sglang/srt/layers/linear.py
@@ -21,7 +21,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +14/-13; `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3; `python/sglang/srt/layers/linear.py` modified +10/-2; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1; `python/sglang/srt/layers/attention/vision.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/model_executor/forward_batch_info.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18024 - fix: correct weight loading prefix mapping for Qwen3-VL

- 链接: https://github.com/sgl-project/sglang/pull/18024
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `522e13b4d2c1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-1，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: correct weight loading prefix mapping for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Fix Qwen3-VL-8B model producing garbage output due to incorrect weight loading. Fixes #17887 Problem The weight loading code unconditionally copies `embed_tokens.weight` to `lm_...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -959,7 +959,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -959,7 +959,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -959,7 +959,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-            if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
+            # Only copy embed_tokens to lm_head when tie_word_embeddings=True
+            # For models with tie_word_embeddings=False (e.g. 8B), lm_head has independent weights
+            if (
+                self.pp_group.is_last_rank
+                and "model.embed_tokens.weight" in name
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16781 - Refactor(qwen3-vl) optimize position encoding interpolation

- 链接: https://github.com/sgl-project/sglang/pull/16781
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `6a4b81e2d9fc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+126/-25，可读 patch 175 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor(qwen3-vl) optimize position encoding interpolation」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Sync Slime patches. Optimize fast_pos_embed_interpolate for Qwen3-VL by vectorizing core embedding lookups and interpolation math. This reduces GPU kernel launch overhead and im...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +123/-23 (146 lines); hunks: -19,6 +19,7; -396,31 +397,130 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices, _calculate_indices_and_weights，涉及 `rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +123/-23 (146 lines); hunks: -19,6 +19,7; -396,31 +397,130 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices, _calculate_indices_and_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -19,6 +19,7 @@
+import numpy as np
@@ -396,31 +397,130 @@ def rot_pos_emb(
-    def fast_pos_embed_interpolate(self, grid_thw):
-        patch_pos_embeds_permute = []
-        m_size = self.spatial_merge_size
-        embeds = torch.arange(self.num_grid, device=self.pos_embed.weight.device)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +123/-23
- 验证与风险: diff 自带测试面 `test/srt/test_embed_interpolate_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18721 - [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models

- 链接: https://github.com/sgl-project/sglang/pull/18721
- 状态/时间: open / 2026-02-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+3/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Launch server with --mm-enable-dp-encoder option for Qwen3-VL model( for example LLM parts use tp2 and vision encoder part use DP2) will hang currently. Root cause is the VocabP...。
- 实现要点: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunks: -224,7 +224,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -308,6 +308,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunks: -224,7 +224,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -308,6 +308,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/vocab_parallel_embedding.py
@@ -224,7 +224,8 @@ def __init__(
-            assert use_attn_tp_group is False
+            if use_attn_tp_group:
+                logger.warning("not in tp_mode, use_attn_tp_group will not work")
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -308,6 +308,7 @@ def __init__(
+                enable_tp=not get_global_server_args().mm_enable_dp_encoder,
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18771 - Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton

- 链接: https://github.com/sgl-project/sglang/pull/18771
- 状态/时间: open / 2026-02-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `benchmark/kernels/fused_moe_triton/common_utils.py`；PR 正文摘要: What This PR Does This PR simply adds: "Qwen3OmniMoeForConditionalGeneration" to the existing Qwen MoE architecture list. Since Qwen3-Omni follows the same configuration schema...。
- 实现要点: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ def get_model_config(; symbols: get_model_config，涉及 `get_model_config`。
- 代码 diff 细节:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ def get_model_config(; symbols: get_model_config
- 关键代码摘录:

```diff
diff -- benchmark/kernels/fused_moe_triton/common_utils.py
@@ -67,6 +67,7 @@ def get_model_config(
+        "Qwen3OmniMoeForConditionalGeneration",
```

- 已读文件:
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #19242 - [feat] feat: add Qwen3-ASR support like whisper

- 链接: https://github.com/sgl-project/sglang/pull/19242
- 状态/时间: open / 2026-02-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+475/-0，可读 patch 519 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] feat: add Qwen3-ASR support like whisper」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length，涉及 `Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor`；`python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig，涉及 `Qwen3ASRHFProcessor, __init__, from_pretrained`；`python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -47,6 +48,7；`python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunks: -1259,6 +1259,7 @@ def is_generation_model(model_architectures: List[str], is...; -1299,6 +1300,7 @@ def is_image_gen_model(model_architectures: List[str]):; symbols: is_generation_model, is_image_gen_model, is_audio_model，涉及 `is_generation_model, is_image_gen_model, is_audio_model`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length
  - `python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -47,6 +48,7
  - `python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunks: -1259,6 +1259,7 @@ def is_generation_model(model_architectures: List[str], is...; -1299,6 +1300,7 @@ def is_image_gen_model(model_architectures: List[str]):; symbols: is_generation_model, is_image_gen_model, is_audio_model
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -64,6 +64,7; -104,6 +105,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen3_asr.py
@@ -0,0 +1,252 @@
+import logging
+import re
+from typing import Any, Dict, List, Optional
+import torch
+from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.models.qwen3_asr import (
diff -- python/sglang/srt/configs/qwen3_asr.py
@@ -0,0 +1,217 @@
+"""
+Copy from [configuration_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)
+and add some typing.
++ Qwen3ASRAudioEncoderConfig
++ Qwen3ASRConfig
++ Qwen3ASRTextConfig
diff -- python/sglang/srt/configs/__init__.py
@@ -22,6 +22,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0; `python/sglang/srt/configs/qwen3_asr.py` added +217/-0; `python/sglang/srt/configs/__init__.py` modified +2/-0; `python/sglang/srt/configs/model_config.py` modified +2/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19003 - [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend

- 链接: https://github.com/sgl-project/sglang/pull/19003
- 状态/时间: merged / 2026-02-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+678/-14，可读 patch 862 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`；PR 正文摘要: FlashInfer CUDNN Prefill demonstrates strong performance. This PR is to introduce it to SGLang as one of VLM ViT attention backends. A new "flashinfer" mm attention backend is a...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunks: -30,7 +30,12; -66,15 +71,12; symbols: Qwen3_VisionMLP, __init__, forward，涉及 `Qwen3_VisionMLP, __init__, forward`；`python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunks: -34,6 +34,7; -64,6 +65,24; symbols: SingletonCache, forward, VisionFlashInferAttention, __init__，涉及 `SingletonCache, forward, VisionFlashInferAttention`；`test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test，涉及 `TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval`；`python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunks: -3859,7 +3859,15 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args，涉及 `add_cli_args`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunks: -30,7 +30,12; -66,15 +71,12; symbols: Qwen3_VisionMLP, __init__, forward
  - `python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunks: -34,6 +34,7; -64,6 +65,24; symbols: SingletonCache, forward, VisionFlashInferAttention, __init__
  - `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test
  - `python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunks: -3859,7 +3859,15 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -30,7 +30,12 @@
-from sglang.srt.layers.attention.vision import VisionAttention
+from sglang.srt.layers.attention.vision import (
+    BATCH_BUCKETS,
+    FLASHINFER_MAX_SEQLEN_BUCKETS,
+    FLASHINFER_WORKSPACE_SIZE_BYTES,
+    VisionAttention,
diff -- python/sglang/srt/layers/attention/vision.py
@@ -34,6 +34,7 @@
+    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
@@ -64,6 +65,24 @@
+# === Vision Encoder === #
+FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
+# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
+# This avoids creating a new graph for each unique batch size at runtime
diff -- test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py
@@ -0,0 +1,258 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +259/-13; `python/sglang/srt/layers/attention/vision.py` modified +152/-0; `python/sglang/srt/server_args.py` modified +9/-1
  - tests: `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0
- 验证与风险: diff 自带测试面 `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19333 - fix qwen3_vl visual module loading

- 链接: https://github.com/sgl-project/sglang/pull/19333
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `d566816d838c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix qwen3_vl visual module loading」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Fix https://github.com/sgl-project/sglang/issues/19335 Qwen3_vl's visual model weight loading is broken. Somehow the name mapping logic was deleted. add name replace logic for v...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    name = name.replace(r"model.visual.", r"visual.")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19693 - [NPU] Fix Qwen3-VL-8B Accuracy for NPU

- 链接: https://github.com/sgl-project/sglang/pull/19693
- 状态/时间: open / 2026-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+199/-108，可读 patch 518 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Fix Qwen3-VL-8B Accuracy for NPU」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/layers/rotary_embedding.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunks: -19,6 +19,7; -397,30 +398,89 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, forward, Qwen3VLForConditionalGeneration，涉及 `rot_pos_emb, fast_pos_embed_interpolate, forward`；`python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: -52,10 +52,14; -185,15 +189,44 @@ def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu，涉及 `LlamaMLP, __init__, forward_prepare_native`；`python/sglang/srt/layers/rotary_embedding.py` modified +4/-3 (7 lines); hunks: -115,9 +115,10 @@ def __init__(; -294,8 +295,8 @@ def forward_npu(; symbols: __init__, forward_npu，涉及 `__init__, forward_npu`；`python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__，涉及 `forward_prepare_npu, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunks: -19,6 +19,7; -397,30 +398,89 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, forward, Qwen3VLForConditionalGeneration
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: -52,10 +52,14; -185,15 +189,44 @@ def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-3 (7 lines); hunks: -115,9 +115,10 @@ def __init__(; -294,8 +295,8 @@ def forward_npu(; symbols: __init__, forward_npu
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -19,6 +19,7 @@
+import numpy as np
@@ -397,30 +398,89 @@ def rot_pos_emb(
-        patch_pos_embeds_permute = []
-        m_size = self.spatial_merge_size
+        num_grid_per_side = int(self.num_position_embeddings**0.5)
-        embeds = torch.arange(self.num_grid, device=self.pos_embed.weight.device)
diff -- python/sglang/srt/models/llama.py
@@ -52,10 +52,14 @@
-from sglang.srt.utils import add_prefix, make_layers
+from sglang.srt.utils import add_prefix, is_npu, make_layers
+_is_npu = is_npu()
+if _is_npu:
+    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
@@ -185,15 +189,44 @@ def __init__(
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -115,9 +115,10 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +79/-20; `python/sglang/srt/models/llama.py` modified +37/-4; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-3; `python/sglang/srt/models/qwen3.py` modified +4/-3; `python/sglang/srt/models/qwen3_moe.py` modified +3/-3; `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19291 - [Qwen3.5] Fix missing `quant_config` in `Qwen3VL`

- 链接: https://github.com/sgl-project/sglang/pull/19291
- 状态/时间: merged / 2026-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Fix missing `quant_config` in `Qwen3VL`」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Fix missing `quant_config` in `Qwen3VL` causing Qwen3.5 NVFP4 versions to use bf16 KV cache instead of fp8.。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1025,6 +1025,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1025,6 +1025,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -1025,6 +1025,7 @@ def __init__(
+        self.quant_config = quant_config
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18185 - [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker

- 链接: https://github.com/sgl-project/sglang/pull/18185
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_omni_moe.py`；关联提交 `22e67876d6aa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+52/-28，可读 patch 130 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_omni_moe.py`；PR 正文摘要: Optimize Qwen3_Omni_Thinker with: 1. Support audio encoder layer's FFN TP>1 2. AudioEncoder vectorized mask and conv fast path in order to reduce CPU small tensor allocator's pr...。
- 实现要点: `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunks: -70,8 +70,18 @@ def __init__(; -98,9 +108,9 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunks: -70,8 +70,18 @@ def __init__(; -98,9 +108,9 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -70,8 +70,18 @@ def __init__(
-        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
-        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
+        self.fc1 = ColumnParallelLinear(
+            self.embed_dim,
+            config.encoder_ffn_dim,
+            bias=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_omni_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20788 - [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL

- 链接: https://github.com/sgl-project/sglang/pull/20788
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Issue description: Enable DP encoder for qwen3-vl model and TP size = 2: Where DP size shall also be equal to TP size, depending image distribution. However, in current init cod...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -330,6 +330,7 @@ def __init__(
+                enable_tp=False if use_data_parallel else True,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20857 - add EVS support for Qwen3-VL

- 链接: https://github.com/sgl-project/sglang/pull/20857
- 状态/时间: open / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+151/-4，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add EVS support for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: We support EVS(Efficient Video Sampling) for Qwen3-VL. Long video token can be pruned while keeping accuracy * add pruning functions in Qwen-VL's processor * make Qwen3's MROPE...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunks: -10,6 +10,7; -22,6 +23,9; symbols: __init__, _maybe_apply_qwen3_evs, get_mm_data, process_mm_data_async，涉及 `__init__, _maybe_apply_qwen3_evs, get_mm_data`；`python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunks: -121,6 +121,7 @@ def get_rope_index(; -134,13 +135,22 @@ def get_rope_index(; symbols: get_rope_index，涉及 `get_rope_index`；`python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunks: -69,6 +69,7; -1047,7 +1048,7 @@ def forward(; symbols: forward, Qwen3VLForConditionalGeneration, __init__，涉及 `forward, Qwen3VLForConditionalGeneration, __init__`；`python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunks: -65,6 +65,16 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunks: -10,6 +10,7; -22,6 +23,9; symbols: __init__, _maybe_apply_qwen3_evs, get_mm_data, process_mm_data_async
  - `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunks: -121,6 +121,7 @@ def get_rope_index(; -134,13 +135,22 @@ def get_rope_index(; symbols: get_rope_index
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunks: -69,6 +69,7; -1047,7 +1048,7 @@ def forward(; symbols: forward, Qwen3VLForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunks: -65,6 +65,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0 (2 lines); hunks: -244,6 +244,7 @@ def __init__(; -261,6 +262,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -10,6 +10,7 @@
+from sglang.srt.configs.qwen3_vl import Qwen3VLConfig
@@ -22,6 +23,9 @@
+from sglang.srt.multimodal.evs import EVSProcessor
+from sglang.srt.multimodal.evs.evs_core import tokens_per_frame
+from sglang.srt.multimodal.evs.evs_module import VideoEVSDataItem
@@ -250,6 +254,10 @@ def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
diff -- python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py
@@ -121,6 +121,7 @@ def get_rope_index(
+                    mm_token_id = image_token_id
@@ -134,13 +135,22 @@ def get_rope_index(
+                    mm_token_id = video_token_id
+                mm_token_count = 0
+                mm_cursor = ed
+                while (
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -69,6 +69,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0; `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2; `python/sglang/srt/models/qwen3_vl.py` modified +10/-2; `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0; `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20759 - [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable

- 链接: https://github.com/sgl-project/sglang/pull/20759
- 状态/时间: merged / 2026-03-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: The Qwen3-VL type model hangs during forward when --mm-enable-dp-encoder is enabled. This is because the VocabParallelEmbedding part remains as TP8 after enabling --mm-enable-dp...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -329,8 +329,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -329,8 +329,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -329,8 +329,8 @@ def __init__(
-                use_attn_tp_group=is_dp_attention_enabled(),
-                enable_tp=False if use_data_parallel else True,
+                enable_tp=not use_data_parallel,
+                use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21469 - [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct

- 链接: https://github.com/sgl-project/sglang/pull/21469
- 状态/时间: merged / 2026-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`；关联提交 `cffc95edf455`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+152/-235，可读 patch 397 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`；PR 正文摘要: - Support Qwen3-VL-30B-A3B-Instruct。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunks: -179,9 +179,8 @@ def __init__(; symbols: __init__, should_apply_lora，涉及 `__init__, should_apply_lora`；`test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunks: -179,9 +179,8 @@ def __init__(; symbols: __init__, should_apply_lora
  - `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -179,9 +179,8 @@ def __init__(
-    # Only allow LoRA on attention projections within text layers for MoE.
-        r"^model\.layers\.(\d+)\.self_attn\.(?:qkv_proj|o_proj)$"
+        r"^(?:model\.layers\.(\d+)\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
diff -- test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py
@@ -0,0 +1,151 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2
  - tests: `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0
- 验证与风险: diff 自带测试面 `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- 链接: https://github.com/sgl-project/sglang/pull/21458
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+101/-3，可读 patch 152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Use aiter's fused_qk_norm_mrope_3d_cache_pts_quant_shuffle kernel to replace 4 separate kernel launches (QKV split, QK RMSNorm, 3D mRoPE, KV cache write) with a single HIP kerne...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope，涉及 `__init__, forward_prepare_native, forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -19,6 +19,7 @@
+from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
@@ -30,13 +31,25 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu
+from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19135 - qwen3 vl skip layer id for pp

- 链接: https://github.com/sgl-project/sglang/pull/19135
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「qwen3 vl skip layer id for pp」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: This PR fixes Qwen3-VL MoE failing to start when pipeline parallelism (PP) is enabled. Previously, launching with --pipeline-parallel-size > 1 could crash during weight loading...。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0 (12 lines); hunks: -26,6 +26,7; -232,6 +233,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0 (12 lines); hunks: -26,6 +26,7; -232,6 +233,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -26,6 +26,7 @@
+from sglang.srt.layers.utils import get_layer_id
@@ -232,6 +233,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            layer_id = get_layer_id(name)
+            if (
+                "visual" not in name
+                and layer_id is not None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22052 - [Fix] Enable precise embedding interpolation by default for Qwen3-VL

- 链接: https://github.com/sgl-project/sglang/pull/22052
- 状态/时间: open / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+10/-11，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Enable precise embedding interpolation by default for Qwen3-VL」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`；PR 正文摘要: The previous default (enable_precise_embedding_interpolation=False) uses an align_corners=False interpolation scheme that diverges from the HuggingFace transformers reference im...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunks: -310,7 +310,7 @@ def __init__(; -524,12 +524,8 @@ def fast_pos_embed_interpolate_from_list(self, grid_thw):; symbols: __init__, fast_pos_embed_interpolate_from_list，涉及 `__init__, fast_pos_embed_interpolate_from_list`；`python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: -675,7 +675,7 @@ class ServerArgs:; -5732,9 +5732,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args，涉及 `ServerArgs, add_cli_args`；`docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: -469,7 +469,7 @@ Please consult the documentation below and [server_args.py](...。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunks: -310,7 +310,7 @@ def __init__(; -524,12 +524,8 @@ def fast_pos_embed_interpolate_from_list(self, grid_thw):; symbols: __init__, fast_pos_embed_interpolate_from_list
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: -675,7 +675,7 @@ class ServerArgs:; -5732,9 +5732,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: -469,7 +469,7 @@ Please consult the documentation below and [server_args.py](...
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -310,7 +310,7 @@ def __init__(
-            get_global_server_args().enable_precise_embedding_interpolation
+            not get_global_server_args().disable_precise_embedding_interpolation
@@ -524,12 +524,8 @@ def fast_pos_embed_interpolate_from_list(self, grid_thw):
-            h_idxs = torch.linspace(
-                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
-            )
diff -- python/sglang/srt/server_args.py
@@ -675,7 +675,7 @@ class ServerArgs:
-    enable_precise_embedding_interpolation: bool = False
+    disable_precise_embedding_interpolation: bool = False
@@ -5732,9 +5732,12 @@ def add_cli_args(parser: argparse.ArgumentParser):
-            "--enable-precise-embedding-interpolation",
+            "--disable-precise-embedding-interpolation",
-            help="Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
diff -- docs/advanced_features/server_arguments.md
@@ -469,7 +469,7 @@ Please consult the documentation below and [server_args.py](https://github.com/s
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +3/-7; `python/sglang/srt/server_args.py` modified +6/-3
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22038 - [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer

- 链接: https://github.com/sgl-project/sglang/pull/22038
- 状态/时间: merged / 2026-04-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+167/-410，可读 patch 696 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`, `python/sglang/srt/models/deepseek_vl2.py`；PR 正文摘要: - Per-image embedding cache: Switch multimodal embedding cache granularity from per-request (combine_hashes(all_items)) to per-image (item.hash), improving cache reuse under LRU...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunks: -15,7 +15,6; -73,7 +72,7; symbols: get_image_feature, get_video_feature，涉及 `get_image_feature, get_video_feature`；`python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunks: -120,6 +120,13 @@ def set(; symbols: set, get_single, has，涉及 `set, get_single, has`；`python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunks: -270,9 +270,7 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature，涉及 `get_image_feature`；`python/sglang/srt/models/phi4mm.py` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ def get_audio_feature(self, items: List[MultimodalDataItem])...; symbols: get_audio_feature，涉及 `get_audio_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunks: -15,7 +15,6; -73,7 +72,7; symbols: get_image_feature, get_video_feature
  - `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunks: -120,6 +120,13 @@ def set(; symbols: set, get_single, has
  - `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunks: -270,9 +270,7 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature
  - `python/sglang/srt/models/phi4mm.py` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ def get_audio_feature(self, items: List[MultimodalDataItem])...; symbols: get_audio_feature
  - `python/sglang/srt/models/step3_vl_10b.py` modified +1/-1 (2 lines); hunks: -484,7 +484,7 @@ def get_image_feature(self, items: List[MultimodalDataItem])...; symbols: get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -15,7 +15,6 @@
-import math
@@ -73,7 +72,7 @@
-from sglang.srt.utils import add_prefix, get_int_env_var, is_npu, round_up
+from sglang.srt.utils import add_prefix, is_npu, round_up
@@ -1167,114 +1166,21 @@ def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
-        max_patches_per_call = get_int_env_var("SGLANG_VLM_MAX_PATCHES_PER_VIT", 0)
diff -- python/sglang/srt/mem_cache/multimodal_cache.py
@@ -120,6 +120,13 @@ def set(
+    def get_single(self, mm_hash: int) -> Optional[EmbeddingResult]:
+        """Get a single cached embedding by its hash (no combine_hashes)."""
+        embedding = self.mm_cache.get(mm_hash)
+        if embedding is not None:
+            self.mm_cache.move_to_end(mm_hash)
+        return embedding
diff -- python/sglang/srt/models/deepseek_vl2.py
@@ -270,9 +270,7 @@ def get_image_feature(self, items: List[MultimodalDataItem]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +10/-104; `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0; `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3; `python/sglang/srt/models/phi4mm.py` modified +1/-1; `python/sglang/srt/models/step3_vl_10b.py` modified +1/-1; `python/sglang/srt/managers/mm_utils.py` modified +147/-286
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- 链接: https://github.com/sgl-project/sglang/pull/21849
- 状态/时间: merged / 2026-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+190/-3，可读 patch 230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM]: allow Qwen3.5 models for encoder disaggregation」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`；PR 正文摘要: Fixes #21805. SGLang already supports Qwen3.5 multimodal models in the runtime, but encoder disaggregation rejected them during servervstartup. This blocked valid EPD deployment...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data，涉及 `get_mm_data`；`test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode，涉及 `test_mmmu, TestEPDDisaggregationQwen35, setUpClass`；`python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items，涉及 `_process_mm_items`；`python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation，涉及 `_handle_encoder_disaggregation`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):
-            self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
+            self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
diff -- test/registered/distributed/test_epd_disaggregation.py
@@ -33,6 +33,7 @@
+QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
@@ -813,6 +814,189 @@ def test_mmmu(self):
+@unittest.skipIf(
+    is_in_ci(),
+    "Qwen3.5 EPD image/video test runs locally only",
+)
diff -- python/sglang/srt/disaggregation/encode_server.py
@@ -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):
-                self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
+                self.model_type
+                in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
-                # For qwen3-vl models, we need to store the video timestamps
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1; `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2; `python/sglang/srt/server_args.py` modified +2/-0
  - tests: `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0
- 验证与风险: diff 自带测试面 `test/registered/distributed/test_epd_disaggregation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- 链接: https://github.com/sgl-project/sglang/pull/22073
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；关联提交 `f6e85676b578`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+571/-11，可读 patch 689 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Adding Qwen3-asr Model Support」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；PR 正文摘要: Issue : https://github.com/sgl-project/sglang/issues/22025 This PR adds support so users can serve Qwen3-ASR via the existing `/v1/audio/transcriptions` endpoint. References - v...。
- 实现要点: `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature，涉及 `Qwen3ASRForConditionalGeneration, __init__, pad_input_ids`；`python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config，涉及 `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`；`python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions，涉及 `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_asr.py
@@ -0,0 +1,199 @@
+"""Qwen3-ASR model compatible with HuggingFace weights"""
+import logging
+from typing import Any, Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig
diff -- python/sglang/srt/configs/qwen3_asr.py
@@ -0,0 +1,172 @@
+import torch
+from transformers import (
+    AutoConfig,
+    AutoFeatureExtractor,
+    AutoTokenizer,
+    PretrainedConfig,
diff -- python/sglang/srt/multimodal/processors/qwen3_asr.py
@@ -0,0 +1,95 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_asr.py` added +199/-0; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22266 - [NPU] fix qwen3.5 video processor

- 链接: https://github.com/sgl-project/sglang/pull/22266
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+177/-21，可读 patch 235 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] fix qwen3.5 video processor」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`；PR 正文摘要: In the Qwen3VLVideoProcessor, there is a permute operation with more than 8 dimensions which is not supported on the NPU. Following PR #20189 , this PR applies a patch to the Qw...。
- 实现要点: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunks: -7,13 +7,62; -90,31 +139,16 @@ def _preprocess(; symbols: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, npu_wrapper_video_preprocess，涉及 `transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess`。
- 代码 diff 细节:
  - `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunks: -7,13 +7,62; -90,31 +139,16 @@ def _preprocess(; symbols: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, npu_wrapper_video_preprocess
- 关键代码摘录:

```diff
diff -- python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py
@@ -7,13 +7,62 @@
-from transformers.image_utils import SizeDict
+from transformers.image_utils import (
+    ChannelDimension,
+    PILImageResampling,
+    SizeDict,
+    get_image_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22181 - [refactor] [asr] Add transcription adapter for extensible ASR models support

- 链接: https://github.com/sgl-project/sglang/pull/22181
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`；关联提交 `a5ed507a1639`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+473/-223，可读 patch 809 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[refactor] [asr] Add transcription adapter for extensible ASR models support」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`；PR 正文摘要: Follow up of #22073. To remove hardcoded model family detection and branching. See reviews in #22073 and #22089. Mainly scoped under python/sglang/srt/entrypoints/openai/ Will c...。
- 实现要点: `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67 (130 lines); hunks: -14,72 +14,6; -167,6 +101,68 @@ def __call__(self, text=None, audio=None, audio_kwargs=None...; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config，涉及 `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`；`test/manual/models/test_qwen3_asr.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: download_audio, TestQwen3ASRTranscription, setUpClass, tearDownClass，涉及 `download_audio, TestQwen3ASRTranscription, setUpClass`；`python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: Qwen3ASRAdapter, build_sampling_params, postprocess_text, build_verbose_response，涉及 `Qwen3ASRAdapter, build_sampling_params, postprocess_text`；`python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5 (13 lines); hunks: -10,11 +10,13; -23,7 +25,7 @@ class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt，涉及 `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67 (130 lines); hunks: -14,72 +14,6; -167,6 +101,68 @@ def __call__(self, text=None, audio=None, audio_kwargs=None...; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `test/manual/models/test_qwen3_asr.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: download_audio, TestQwen3ASRTranscription, setUpClass, tearDownClass
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: Qwen3ASRAdapter, build_sampling_params, postprocess_text, build_verbose_response
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5 (13 lines); hunks: -10,11 +10,13; -23,7 +25,7 @@ class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/qwen3_asr.py
@@ -14,72 +14,6 @@
-class Qwen3ASRThinkerConfig(PretrainedConfig):
-    model_type = "qwen3_asr_thinker"
-    sub_configs = {
-        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
-    }
-    def __init__(
diff -- test/manual/models/test_qwen3_asr.py
@@ -0,0 +1,118 @@
+"""
+Test Qwen3-ASR model support in SGLang.
+Tests /v1/audio/transcriptions endpoint (OpenAI-compatible).
+Usage:
+    python test/manual/models/test_qwen3_asr.py
+"""
diff -- python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py
@@ -0,0 +1,49 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67; `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5
  - tests: `test/manual/models/test_qwen3_asr.py` added +118/-0
- 验证与风险: diff 自带测试面 `test/manual/models/test_qwen3_asr.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22230 - [Feature] Support eagle3 for qwen3-vl

- 链接: https://github.com/sgl-project/sglang/pull/22230
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_vl.py`；关联提交 `a69be2e866fb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+24/-0，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support eagle3 for qwen3-vl」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_vl.py`；PR 正文摘要: Adapt Eagle3 capture for the Qwen3-VL model Add `set_eagle3_layers_to_capture` function on `qwen3_vl.py`. Like: qwen2_5_vl.py Launch Script gsm8k: without eagle3: with eagle3: m...。
- 实现要点: `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunks: -1130,6 +1130,9 @@ def __init__(; -1246,13 +1249,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights，涉及 `__init__, separate_deepstack_embeds, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunks: -1130,6 +1130,9 @@ def __init__(; -1246,13 +1249,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -1130,6 +1130,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
@@ -1246,13 +1249,18 @@ def forward(
+        aux_hidden_states = None
+        if self.capture_aux_hidden_states:
+            hidden_states, aux_hidden_states = hidden_states
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +24/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22089 - [Feature] Add chunk-based streaming ASR for Qwen3-ASR

- 链接: https://github.com/sgl-project/sglang/pull/22089
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；关联提交 `8b991d98a12c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+263/-2，可读 patch 325 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add chunk-based streaming ASR for Qwen3-ASR」；模型线: Qwen VLM/Omni/ASR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；PR 正文摘要: Motivation Issue: #22025 (streaming input design and implementation) This PR adds chunk-based streaming transcription support for Qwen3-ASR via the existing /v1/audio/transcript...。
- 实现要点: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunks: -9,12 +9,32; symbols: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template，涉及 `Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config`；`python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -47,7 +47,7 @@ def _build_transcription_prompt(self, input_text: Union[str, l...; symbols: _build_transcription_prompt, compute_mrope_positions，涉及 `_build_transcription_prompt, compute_mrope_positions`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunks: -9,12 +9,32; symbols: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -47,7 +47,7 @@ def _build_transcription_prompt(self, input_text: Union[str, l...; symbols: _build_transcription_prompt, compute_mrope_positions
- 关键代码摘录:

```diff
diff -- python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py
@@ -9,12 +9,32 @@
+from sglang.srt.multimodal.processors.qwen3_asr import DEFAULT_ASR_PROMPT
+    @property
+    def supports_chunked_streaming(self) -> bool:
+        return True
+    @property
+    def chunked_streaming_config(self) -> dict:
diff -- python/sglang/srt/multimodal/processors/qwen3_asr.py
@@ -12,7 +12,7 @@
-_DEFAULT_ASR_PROMPT = (
+DEFAULT_ASR_PROMPT = (
@@ -47,7 +47,7 @@ def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:
-            return _DEFAULT_ASR_PROMPT
+            return DEFAULT_ASR_PROMPT
```

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22839 - fix(config): Add from_dict() for Qwen3VL config classes

- 链接: https://github.com/sgl-project/sglang/pull/22839
- 状态/时间: open / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+306/-0，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(config): Add from_dict() for Qwen3VL config classes」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`；PR 正文摘要: Fix Qwen3-VL and Qwen3.5 model loading with transformers 5.5.0+. Transformers 5.5.0+ natively supports Qwen3-VL, causing `AutoConfig.from_pretrained()` to skip sglang's config c...。
- 实现要点: `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion，涉及 `TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object`；`python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunks: -78,6 +78,17 @@ class Qwen3_5Config(PretrainedConfig):; -112,6 +123,55 @@ class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; symbols: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig，涉及 `Qwen3_5Config, from_dict, __init__`；`python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunks: -236,6 +236,17 @@ class Qwen3VLConfig(PretrainedConfig):; -251,11 +262,15 @@ def __init__(; symbols: Qwen3VLConfig, from_dict, __init__, Qwen3VLMoeConfig，涉及 `Qwen3VLConfig, from_dict, __init__`；`python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunks: -25,6 +25,7; -65,4 +66,6。
- 代码 diff 细节:
  - `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion
  - `python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunks: -78,6 +78,17 @@ class Qwen3_5Config(PretrainedConfig):; -112,6 +123,55 @@ class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; symbols: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunks: -236,6 +236,17 @@ class Qwen3VLConfig(PretrainedConfig):; -251,11 +262,15 @@ def __init__(; symbols: Qwen3VLConfig, from_dict, __init__, Qwen3VLMoeConfig
  - `python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunks: -25,6 +25,7; -65,4 +66,6
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0 (4 lines); hunks: -87,6 +87,8; -121,6 +123,8
- 关键代码摘录:

```diff
diff -- test/registered/unit/configs/test_qwen3_vl_config.py
@@ -0,0 +1,198 @@
+"""Unit tests for qwen3_vl and qwen3_5 config from_dict() handling.
+This tests the fix for transformers 5.5.0 compatibility where nested
+vision_config and text_config dicts need to be converted to config objects.
+"""
+import json
+import unittest
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -78,6 +78,17 @@ class Qwen3_5Config(PretrainedConfig):
+    @classmethod
+    def from_dict(cls, config_dict, **kwargs):
+        config = super().from_dict(config_dict, **kwargs)
+        if isinstance(getattr(config, "vision_config", None), dict):
+            config.vision_config = cls.sub_configs["vision_config"](
+                **config.vision_config
diff -- python/sglang/srt/configs/qwen3_vl.py
@@ -236,6 +236,17 @@ class Qwen3VLConfig(PretrainedConfig):
```

- 已读文件:
  - tests: `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0
  - runtime: `python/sglang/srt/configs/qwen3_5.py` modified +71/-0; `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0; `python/sglang/srt/configs/__init__.py` modified +3/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/configs/test_qwen3_vl_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22848 - [Feature] WebSocket streaming audio input for ASR

- 链接: https://github.com/sgl-project/sglang/pull/22848
- 状态/时间: open / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+937/-43，可读 patch 1151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] WebSocket streaming audio input for ASR」；模型线: Qwen VLM/Omni/ASR；类别: 模型实现调整；主要 diff: `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`；PR 正文摘要: Implements M1 of the RFC in sgl-project/sglang#22474. PR #22089 shipped chunked streaming **output** for Qwen3-ASR via `POST /v1/audio/transcriptions?stream=true` (SSE over an H...。
- 实现要点: `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunks: -1,17 +1,30; -29,8 +42,96; symbols: _normalize_for_wer, _wer, download_audio, _pcm16_from_audio_bytes，涉及 `_normalize_for_wer, _wer, download_audio`；`python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +376/-0 (376 lines); hunks: -0,0 +1,376; symbols: names, _safe_close_websocket, _pcm_to_wav, RealtimeMessageType，涉及 `names, _safe_close_websocket, _pcm_to_wav`；`python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunks: -1,9 +1,21; -22,13 +34,23 @@ class StreamingASRState:; symbols: StreamingASRState, get_prefix_text, _record_emit, update，涉及 `StreamingASRState, get_prefix_text, _record_emit`；`python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunks: -29,7 +29,7; -43,8 +43,12; symbols: _generate_chunked_asr_stream, handle_websocket，涉及 `_generate_chunked_asr_stream, handle_websocket`。
- 代码 diff 细节:
  - `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunks: -1,17 +1,30; -29,8 +42,96; symbols: _normalize_for_wer, _wer, download_audio, _pcm16_from_audio_bytes
  - `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +376/-0 (376 lines); hunks: -0,0 +1,376; symbols: names, _safe_close_websocket, _pcm_to_wav, RealtimeMessageType
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunks: -1,9 +1,21; -22,13 +34,23 @@ class StreamingASRState:; symbols: StreamingASRState, get_prefix_text, _record_emit, update
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunks: -29,7 +29,7; -43,8 +43,12; symbols: _generate_chunked_asr_stream, handle_websocket
  - `python/sglang/srt/entrypoints/http_server.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -1580,6 +1581,12 @@ async def openai_v1_audio_transcriptions(; symbols: openai_v1_audio_transcriptions, openai_v1_audio_transcriptions_ws, available_models
- 关键代码摘录:

```diff
diff -- test/manual/models/test_qwen3_asr.py
@@ -1,17 +1,30 @@
-Tests /v1/audio/transcriptions endpoint (OpenAI-compatible).
+Tests /v1/audio/transcriptions (HTTP) and
+/v1/audio/transcriptions/stream (WebSocket live audio input).
+import asyncio
+import json
+import re
diff -- python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py
@@ -0,0 +1,376 @@
+"""WebSocket transport for OpenAI Realtime API-style transcription.
+The wire protocol mirrors OpenAI's Realtime API conventions
+(``session.start`` / ``transcript.delta`` / ``transcript.final``) so the
+``Realtime*`` symbol prefix refers to the protocol identity, not the
+transport. A future gRPC streaming variant for the same OpenAI-style
+protocol would live in ``serving_transcription_grpc.py`` and could reuse
diff -- python/sglang/srt/entrypoints/openai/streaming_asr.py
@@ -1,9 +1,21 @@
```

- 已读文件:
  - tests: `test/manual/models/test_qwen3_asr.py` modified +451/-3
  - runtime: `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +376/-0; `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33; `python/sglang/srt/entrypoints/http_server.py` modified +7/-0; `python/sglang/srt/server_args.py` modified +10/-0
- 验证与风险: diff 自带测试面 `test/manual/models/test_qwen3_asr.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23115 - fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights

- 链接: https://github.com/sgl-project/sglang/pull/23115
- 状态/时间: open / 2026-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: Fixes #23063 Problem When launching a Qwen3-VL-MoE model in `--encoder-only` mode, the server crashes during weight loading with `AttributeError: 'Qwen3VLMoeForConditionalGenera...。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                and hasattr(self, "model")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- 链接: https://github.com/sgl-project/sglang/pull/22431
- 状态/时间: merged / 2026-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Qwen3.5 video processing when passing video_data in "processor_output" format」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`；PR 正文摘要: Currently, the video preprocessing function of Qwen3.5 requires two return values: https://github.com/sgl-project/sglang/blob/ef6bfc1197ab45290e33941881f23c39fbf30ad9/python/sgl...。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video，涉及 `preprocess_video`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -162,7 +162,7 @@ async def preprocess_video(
-        return vr
+        return vr, None
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/qwen_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23220 - Bugfix: Qwen3-VL-MoE adapt encoder_only

- 链接: https://github.com/sgl-project/sglang/pull/23220
- 状态/时间: open / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Bugfix: Qwen3-VL-MoE adapt encoder_only」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl_moe.py`；PR 正文摘要: This bug is caused by PR https://github.com/sgl-project/sglang/pull/19135, `self.model` is created in the parent class and is only created when `--encoder-only` is not enabled....。
- 实现要点: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                and hasattr(self, "model")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23304 - [Bugfix] Fix Qwen3-VL rope config compatibility

- 链接: https://github.com/sgl-project/sglang/pull/23304
- 状态/时间: closed / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-10，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL rope config compatibility」；模型线: Qwen VLM/Omni/ASR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: `Qwen3-VL` can fail during model initialization because `qwen3.py` assumes `config.rope_parameters["rope_theta"]` is always available. In practice, Qwen3-VL series model exposes...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunks: -316,16 +316,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunks: -316,16 +316,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -316,16 +316,7 @@ def __init__(
-        if (
-            hasattr(config, "rope_parameters")
-            and config.rope_parameters
-            and "rope_theta" in config.rope_parameters
-        ):
-            rope_theta = config.rope_parameters["rope_theta"]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +1/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23469 - [NPU] adapt the Qwen3-ASR model for deployment on NPU

- 链接: https://github.com/sgl-project/sglang/pull/23469
- 状态/时间: open / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-0，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] adapt the Qwen3-ASR model for deployment on NPU」；模型线: Qwen VLM/Omni/ASR；类别: 性能/后端优化；主要 diff: `python/sglang/srt/utils/common.py`；PR 正文摘要: Enable Qwen3-ASR model deployment on NPU devices. The current audio loading implementation relies on torchaudio with CUDA dependencies, which is not compatible with NPU environm...。
- 实现要点: `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: -740,6 +740,24 @@ def load_audio(; symbols: load_audio，涉及 `load_audio`。
- 代码 diff 细节:
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: -740,6 +740,24 @@ def load_audio(; symbols: load_audio
- 关键代码摘录:

```diff
diff -- python/sglang/srt/utils/common.py
@@ -740,6 +740,24 @@ def load_audio(
+    if is_npu():
+        import soundfile as sf
+        if isinstance(source, bytes):
+            audio, original_sr = sf.read(BytesIO(source))
+        else:
+            audio, original_sr = sf.read(source)
```

- 已读文件:
  - runtime: `python/sglang/srt/utils/common.py` modified +18/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/utils/common.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
