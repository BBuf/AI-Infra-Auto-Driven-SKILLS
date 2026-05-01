# sglang Qwen VLM/Omni/ASR Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/qwen3_vl.md` | [#12554](https://github.com/sgl-project/sglang/pull/12554) |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` | [#23907](https://github.com/sgl-project/sglang/pull/23907), [#23968](https://github.com/sgl-project/sglang/pull/23968) |
| `docs_new/cookbook/omni/FishAudio/S2-Pro.mdx` | no direct PR-number commit |
| `docs_new/cookbook/omni/intro.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/qwen3_vl.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` | [#23907](https://github.com/sgl-project/sglang/pull/23907) |
| `examples/chat_template/qwen3_vl_reranker.jinja` | no direct PR-number commit |
| `examples/runtime/qwen3_vl_reranker.py` | no direct PR-number commit |
| `python/sglang/srt/configs/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/configs/qwen3_omni.py` | [#10911](https://github.com/sgl-project/sglang/pull/10911) |
| `python/sglang/srt/configs/qwen3_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911) |
| `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` | [#22089](https://github.com/sgl-project/sglang/pull/22089), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` | no direct PR-number commit |
| `python/sglang/srt/models/glmasr.py` | [#15570](https://github.com/sgl-project/sglang/pull/15570), [#15772](https://github.com/sgl-project/sglang/pull/15772) |
| `python/sglang/srt/models/qwen2_5_vl.py` | [#5003](https://github.com/sgl-project/sglang/pull/5003), [#5349](https://github.com/sgl-project/sglang/pull/5349), [#6136](https://github.com/sgl-project/sglang/pull/6136), [#8801](https://github.com/sgl-project/sglang/pull/8801), [#13055](https://github.com/sgl-project/sglang/pull/13055), [#13075](https://github.com/sgl-project/sglang/pull/13075), [#13126](https://github.com/sgl-project/sglang/pull/13126), [#13904](https://github.com/sgl-project/sglang/pull/13904), [#14292](https://github.com/sgl-project/sglang/pull/14292), [#15138](https://github.com/sgl-project/sglang/pull/15138), [#15320](https://github.com/sgl-project/sglang/pull/15320) |
| `python/sglang/srt/models/qwen2_audio.py` | no direct PR-number commit |
| `python/sglang/srt/models/qwen2_vl.py` | [#2055](https://github.com/sgl-project/sglang/pull/2055), [#5003](https://github.com/sgl-project/sglang/pull/5003), [#5349](https://github.com/sgl-project/sglang/pull/5349), [#5783](https://github.com/sgl-project/sglang/pull/5783), [#6136](https://github.com/sgl-project/sglang/pull/6136), [#13055](https://github.com/sgl-project/sglang/pull/13055), [#13736](https://github.com/sgl-project/sglang/pull/13736) |
| `python/sglang/srt/models/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073) |
| `python/sglang/srt/models/qwen3_omni_moe.py` | [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11791](https://github.com/sgl-project/sglang/pull/11791), [#12333](https://github.com/sgl-project/sglang/pull/12333), [#18185](https://github.com/sgl-project/sglang/pull/18185) |
| `python/sglang/srt/models/qwen3_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11458](https://github.com/sgl-project/sglang/pull/11458), [#11481](https://github.com/sgl-project/sglang/pull/11481), [#12333](https://github.com/sgl-project/sglang/pull/12333), [#13724](https://github.com/sgl-project/sglang/pull/13724), [#13736](https://github.com/sgl-project/sglang/pull/13736), [#14292](https://github.com/sgl-project/sglang/pull/14292), [#15205](https://github.com/sgl-project/sglang/pull/15205), [#15320](https://github.com/sgl-project/sglang/pull/15320), [#16366](https://github.com/sgl-project/sglang/pull/16366), [#16781](https://github.com/sgl-project/sglang/pull/16781), ... (15 total) |
| `python/sglang/srt/models/qwen3_vl_moe.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11481](https://github.com/sgl-project/sglang/pull/11481), [#13983](https://github.com/sgl-project/sglang/pull/13983), [#21469](https://github.com/sgl-project/sglang/pull/21469) |
| `python/sglang/srt/multimodal/processors/glmasr.py` | [#15570](https://github.com/sgl-project/sglang/pull/15570), [#15772](https://github.com/sgl-project/sglang/pull/15772) |
| `python/sglang/srt/multimodal/processors/qwen3_asr.py` | [#22073](https://github.com/sgl-project/sglang/pull/22073), [#22089](https://github.com/sgl-project/sglang/pull/22089), [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `python/sglang/srt/multimodal/processors/qwen_vl.py` | [#10323](https://github.com/sgl-project/sglang/pull/10323), [#10911](https://github.com/sgl-project/sglang/pull/10911), [#11377](https://github.com/sgl-project/sglang/pull/11377), [#12240](https://github.com/sgl-project/sglang/pull/12240), [#12458](https://github.com/sgl-project/sglang/pull/12458) |
| `python/sglang/test/external_models/custom_qwen2_vl.py` | no direct PR-number commit |
| `test/manual/models/test_qwen3_asr.py` | [#22181](https://github.com/sgl-project/sglang/pull/22181) |
| `test/registered/ascend/vlm_models/test_npu_qwen2_5_vl_3b_instruct.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_qwen2_5_vl_72b_instruct.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_235b_a22b_instruct.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_30b_a3b_instruct.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_4b_instruct.py` | no direct PR-number commit |
| `test/registered/ascend/vlm_models/test_npu_qwen3_vl_8b_instruct.py` | no direct PR-number commit |
| `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` | [#21469](https://github.com/sgl-project/sglang/pull/21469) |

## PR Coverage Summary

- Git-traced PRs: 41
- Extra PRs preserved from existing docs: 38
- Total PRs in this document: 79
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-11-21 | [#2055](https://github.com/sgl-project/sglang/pull/2055) | merged | Add support for Qwen2-VL-based embedding models | `python/sglang/srt/models/qwen2_vl.py` |
| 2025-04-14 | [#5003](https://github.com/sgl-project/sglang/pull/5003) | merged | Support for Qwen2.5-VL Model in bitsandbytes Format | `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-04-24 | [#5349](https://github.com/sgl-project/sglang/pull/5349) | merged | vlm: enable radix cache for qwen-vl models | `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-05-01 | [#5783](https://github.com/sgl-project/sglang/pull/5783) | merged | Remove unused method `calculate_num_image_tokens` from qwen2_vl.py | `python/sglang/srt/models/qwen2_vl.py` |
| 2025-05-16 | [#6136](https://github.com/sgl-project/sglang/pull/6136) | merged | Support precomputed multimodal features for Qwen-VL and Gemma3 models. | `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py` |
| 2025-08-12 | [#9014](https://github.com/sgl-project/sglang/pull/9014) | merged | Fuse writing KV buffer into rope kernel (part 2: srt) | `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py` |
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
| 2026-04-28 | [#23907](https://github.com/sgl-project/sglang/pull/23907) | merged | [Docs] add Nemotron 3 Nano Omni cookbook | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` |
| 2026-04-28 | [#23968](https://github.com/sgl-project/sglang/pull/23968) | merged | [Docs] update Docker image for Nemotron 3 Nano Omni | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |

## Per-PR Diff Audit Cards

### PR #2055 - Add support for Qwen2-VL-based embedding models

- Link: https://github.com/sgl-project/sglang/pull/2055
- Status/date: merged / 2024-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_vl.py`; associated commits `f6f713797bcb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +39/-12, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support for Qwen2-VL-based embedding models"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_vl.py`; technical summary: Covers "Add support for Qwen2-VL-based embedding models"; the main implementation surface is `python/sglang/srt/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_vl.py` modified +12/-5 (17 lines); hunks: -44,6 +44,7; -559,6 +560,7 @@ def __init__(; symbols: __init__, _process_image_input, forward, touching `__init__, _process_image_input, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_vl.py` modified +12/-5 (17 lines); hunks: -44,6 +44,7; -559,6 +560,7 @@ def __init__(; symbols: __init__, _process_image_input, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +12/-5
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/srt/models/test_embedding_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5003 - Support for Qwen2.5-VL Model in bitsandbytes Format

- Link: https://github.com/sgl-project/sglang/pull/5003
- Status/date: merged / 2025-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; associated commits `072df753546b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +375/-45, 531 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support for Qwen2.5-VL Model in bitsandbytes Format"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "Support for Qwen2.5-VL Model in bitsandbytes Format"; the main implementation surface is `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_vl.py` modified +24/-23 (47 lines); hunks: -152,7 +152,7 @@ def __init__(; -351,7 +351,7 @@ def __init__(; symbols: __init__, dtype, device, forward, touching `__init__, dtype, device`; `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22 (46 lines); hunks: -141,7 +141,7 @@ def __init__(; -325,7 +325,7 @@ def get_window_index(self, grid_thw):; symbols: __init__, get_window_index, dtype, device, touching `__init__, get_window_index, dtype`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_vl.py` modified +24/-23 (47 lines); hunks: -152,7 +152,7 @@ def __init__(; -351,7 +351,7 @@ def __init__(; symbols: __init__, dtype, device, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22 (46 lines); hunks: -141,7 +141,7 @@ def __init__(; -325,7 +325,7 @@ def get_window_index(self, grid_thw):; symbols: __init__, get_window_index, dtype, device
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +24/-23; `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-22
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_bnb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5349 - vlm: enable radix cache for qwen-vl models

- Link: https://github.com/sgl-project/sglang/pull/5349
- Status/date: merged / 2025-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; associated commits `c998d04b4692`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +435/-337, 1363 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "vlm: enable radix cache for qwen-vl models"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "vlm: enable radix cache for qwen-vl models"; the main implementation surface is `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_vl.py` modified +3/-7 (10 lines); hunks: -42,7 +42,7; -490,15 +490,11 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, touching `__init__, pad_input_ids, get_image_feature`; `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6 (9 lines); hunks: -49,7 +49,7; -488,11 +488,8 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, touching `__init__, pad_input_ids, get_image_feature`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_vl.py` modified +3/-7 (10 lines); hunks: -42,7 +42,7; -490,15 +490,11 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6 (9 lines); hunks: -49,7 +49,7; -488,11 +488,8 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +3/-7; `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-6
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/srt/run_suite.py`, `test/srt/test_vision_openai_server.py`, `test/srt/test_vlm_accuracy.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5783 - Remove unused method `calculate_num_image_tokens` from qwen2_vl.py

- Link: https://github.com/sgl-project/sglang/pull/5783
- Status/date: merged / 2025-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_vl.py`; associated commits `e97e57e699e5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-12, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove unused method `calculate_num_image_tokens` from qwen2_vl.py"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `python/sglang/srt/models/qwen2_vl.py`; technical summary: Covers "Remove unused method `calculate_num_image_tokens` from qwen2_vl.py"; the main implementation surface is `python/sglang/srt/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_vl.py` modified +0/-12 (12 lines); hunks: -442,18 +442,6 @@ class Qwen2VLForConditionalGeneration(nn.Module):; symbols: Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__, touching `Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-12 (12 lines); hunks: -442,18 +442,6 @@ class Qwen2VLForConditionalGeneration(nn.Module):; symbols: Qwen2VLForConditionalGeneration, calculate_num_image_tokens, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_vl.py` modified +0/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6136 - Support precomputed multimodal features for Qwen-VL and Gemma3 models.

- Link: https://github.com/sgl-project/sglang/pull/6136
- Status/date: merged / 2025-05-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; associated commits `f19a9204cdbd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +592/-125, 1118 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support precomputed multimodal features for Qwen-VL and Gemma3 models."; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; technical summary: Covers "Support precomputed multimodal features for Qwen-VL and Gemma3 models."; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -497,6 +497,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature, touching `pad_input_ids, get_image_feature`; `python/sglang/srt/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -486,6 +486,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature, touching `pad_input_ids, get_image_feature`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0 (6 lines); hunks: -497,6 +497,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature
  - `python/sglang/srt/models/qwen2_vl.py` modified +6/-0 (6 lines); hunks: -486,6 +486,12 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, get_image_feature
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +6/-0; `python/sglang/srt/models/qwen2_vl.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_skip_tokenizer_init.py`, `test/srt/test_vlm_accuracy.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9014 - Fuse writing KV buffer into rope kernel (part 2: srt)

- Link: https://github.com/sgl-project/sglang/pull/9014
- Status/date: merged / 2025-08-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +65/-6, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse writing KV buffer into rope kernel (part 2: srt)"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py`; technical summary: Covers "Fuse writing KV buffer into rope kernel (part 2: srt)"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, touching `GptOssConfig, forward_normal, _enable_fused_set_kv_buffer`; `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`; `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config, touching `_set_envs_and_config`; `.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg
  - `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
  - `.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:
  - `docker/Dockerfile.gb200` modified +1/-1 (2 lines); hunks: -64,7 +64,7 @@ RUN python3 -m pip install --no-cache-dir --upgrade pip setupt...
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,10 +66,15 @@
+_is_cuda = is_cuda()
+if _is_cuda:
+    from sgl_kernel import FusedSetKVBufferArg
@@ -196,6 +201,32 @@ def forward_normal(
+def _enable_fused_set_kv_buffer():
+    return _is_cuda
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -222,6 +222,7 @@ def forward_cuda(
+        fused_set_kv_buffer_arg=None,  # Optional[FusedSetKVBufferArg]
@@ -231,8 +232,17 @@ def forward_cuda(
+                # Compatible with old sgl-kernel
+                **(
+                    dict(fused_set_kv_buffer_arg=fused_set_kv_buffer_arg)
+                    if fused_set_kv_buffer_arg is not None
diff -- python/sglang/srt/entrypoints/engine.py
@@ -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +51/-2; `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/pyproject.toml` modified +1/-1
  - ci: `.github/workflows/pr-test-pd-router.yml` modified +1/-1
  - other: `docker/Dockerfile.gb200` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8801 - Qwen2.5-VL eagle3 infer

- Link: https://github.com/sgl-project/sglang/pull/8801
- Status/date: merged / 2025-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`; associated commits `37d83c6e6d8a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +114/-5, 260 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Qwen2.5-VL eagle3 infer"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "Qwen2.5-VL eagle3 infer"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1 (25 lines); hunks: -517,6 +517,9 @@ def __init__(; -587,9 +590,13 @@ def forward(; symbols: __init__, pad_input_ids, forward, load_weights, touching `__init__, pad_input_ids, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1 (25 lines); hunks: -517,6 +517,9 @@ def __init__(; -587,9 +590,13 @@ def forward(; symbols: __init__, pad_input_ids, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +24/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10323 - model: support qwen3-vl series

- Link: https://github.com/sgl-project/sglang/pull/10323
- Status/date: merged / 2025-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; associated commits `4f564b9e8378`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1898/-8, 2006 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: support qwen3-vl series"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "model: support qwen3-vl series"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` added +787/-0 (787 lines); hunks: -0,0 +1,787; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLVisionPatchEmbed, touching `Qwen3_VisionMLP, __init__, forward`; `python/sglang/srt/configs/qwen3_vl.py` added +586/-0 (586 lines); hunks: -0,0 +1,586; symbols: Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig, to, touching `Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig`; `python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature, touching `Qwen3MoeLLMModel, __init__, get_input_embeddings`; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1 (9 lines); hunks: -12,6 +12,8; -209,7 +211,12 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, __init__, touching `preprocess_video, Qwen2_5VLImageProcessor, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` added +787/-0 (787 lines); hunks: -0,0 +1,787; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLVisionPatchEmbed
  - `python/sglang/srt/configs/qwen3_vl.py` added +586/-0 (586 lines); hunks: -0,0 +1,586; symbols: Qwen3VLVisionConfig, __init__, Qwen3VLTextConfig, to
  - `python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0 (471 lines); hunks: -0,0 +1,471; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1 (9 lines); hunks: -12,6 +12,8; -209,7 +211,12 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` added +787/-0; `python/sglang/srt/configs/qwen3_vl.py` added +586/-0; `python/sglang/srt/models/qwen3_vl_moe.py` added +471/-0; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- Link: https://github.com/sgl-project/sglang/pull/10749
- Status/date: merged / 2025-09-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +105/-34, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; the main implementation surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg, touching `enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg`; `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention, touching `forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg`; `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core, touching `forward_prepare, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/utils.py` added +51/-0; `python/sglang/srt/models/gpt_oss.py` modified +7/-30; `python/sglang/srt/models/bailing_moe.py` modified +25/-2; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- Link: https://github.com/sgl-project/sglang/pull/10985
- Status/date: merged / 2025-10-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-2, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; technical summary: Covers "Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core, touching `__init__, forward_prepare, forward_core`; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunks: -1065,6 +1065,7 @@ def forward(; -1075,6 +1076,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunks: -1065,6 +1065,7 @@ def forward(; -1075,6 +1076,9 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11481 - [smol] [perf] Qwen3-VL in place op.

- Link: https://github.com/sgl-project/sglang/pull/11481
- Status/date: merged / 2025-10-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; associated commits `be740acdb0ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-11, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[smol] [perf] Qwen3-VL in place op."; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "[smol] [perf] Qwen3-VL in place op."; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +4/-7 (11 lines); hunks: -189,10 +189,10 @@ def forward(; -441,7 +441,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4 (7 lines); hunks: -114,7 +114,7 @@ def forward(; -130,9 +130,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +4/-7 (11 lines); hunks: -189,10 +189,10 @@ def forward(; -441,7 +441,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4 (7 lines); hunks: -114,7 +114,7 @@ def forward(; -130,9 +130,8 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +4/-7; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11458 - [BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl

- Link: https://github.com/sgl-project/sglang/pull/11458
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `b2c856692092`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-3, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[BugFix][Qwen3-VL]: fix cu_seqlens in qwen3-vl"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +5/-3 (8 lines); hunks: -452,13 +452,15 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-3 (8 lines); hunks: -452,13 +452,15 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10911 - model: qwen3-omni (thinker-only)

- Link: https://github.com/sgl-project/sglang/pull/10911
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py` and 6 files; associated commits `86b04d25b3f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1947/-328, 2837 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: qwen3-omni (thinker-only)"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "model: qwen3-omni (thinker-only)"; the main implementation surface is `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding, touching `Qwen3OmniMoeAudioEncoderLayer, __init__, forward`; `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: -0,0 +1,613; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeTextConfig, touching `Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig`; `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: -14,41 +14,27; -60,28 +46,16 @@ class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature, touching `Qwen3MoeLLMModel, __init__, get_input_embeddings`; `python/sglang/srt/models/qwen3_vl.py` modified +38/-24 (62 lines); hunks: -15,7 +15,7; -27,7 +27,11; symbols: forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger, __init__, touching `forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunks: -0,0 +1,661; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunks: -0,0 +1,613; symbols: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeTextConfig
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunks: -14,41 +14,27; -60,28 +46,16 @@ class Qwen3MoeLLMModel(Qwen3MoeModel):; symbols: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
  - `python/sglang/srt/models/qwen3_vl.py` modified +38/-24 (62 lines); hunks: -15,7 +15,7; -27,7 +27,11; symbols: forward, Qwen3_VisionPatchMerger, Qwen3VLMoeVisionPatchMerger, __init__
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +40/-6 (46 lines); hunks: -12,6 +12,7; -209,22 +210,31 @@ async def preprocess_video(; symbols: preprocess_video, Qwen2_5VLImageProcessor, QwenVLImageProcessor, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0; `python/sglang/srt/configs/qwen3_omni.py` added +613/-0; `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168; `python/sglang/srt/models/qwen3_vl.py` modified +38/-24; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +40/-6; `python/sglang/srt/configs/qwen3_vl.py` modified +0/-10
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_b.py`, `test/srt/test_vision_openai_server_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11377 - [BugFix][Qwen3-VL]: add metadata for video in qwen3-vl

- Link: https://github.com/sgl-project/sglang/pull/11377
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/qwen_vl.py`; associated commits `fde2decf8b59`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-8, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][Qwen3-VL]: add metadata for video in qwen3-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`; technical summary: Covers "[BugFix][Qwen3-VL]: add metadata for video in qwen3-vl"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8 (34 lines); hunks: -214,7 +214,14 @@ async def preprocess_video(; -279,14 +286,25 @@ async def process_mm_data_async(; symbols: preprocess_video, process_mm_data_async, touching `preprocess_video, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8 (34 lines); hunks: -214,7 +214,14 @@ async def preprocess_video(; -279,14 +286,25 @@ async def process_mm_data_async(; symbols: preprocess_video, process_mm_data_async
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +26/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/qwen_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12261 - [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl

- Link: https://github.com/sgl-project/sglang/pull/12261
- Status/date: open / 2025-10-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunks: -430,15 +430,15 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunks: -430,15 +430,15 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12662 - [CPU] Add support for Qwen3-vl and Qwen3-omni

- Link: https://github.com/sgl-project/sglang/pull/12662
- Status/date: open / 2025-11-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +496/-55, 884 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add support for Qwen3-vl and Qwen3-omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; technical summary: Covers "[CPU] Add support for Qwen3-vl and Qwen3-omni"; the main implementation surface is `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunks: -16,9 +16,11; -30,9 +32,11; symbols: forward, VisionAMXAttention, __init__, touching `forward, VisionAMXAttention, __init__`; `python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunks: -189,28 +189,62 @@ def adjust_config_with_unaligned_cpu_tp(; symbols: adjust_config_with_unaligned_cpu_tp, touching `adjust_config_with_unaligned_cpu_tp`; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunks: -32,6 +32,7; -57,6 +58,65; symbols: hacked_preprocess, smart_resize, touching `hacked_preprocess, smart_resize`; `python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunks: -72,7 +72,14; -88,6 +95,10; symbols: Qwen3_VisionMLP, __init__, forward, touching `Qwen3_VisionMLP, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunks: -16,9 +16,11; -30,9 +32,11; symbols: forward, VisionAMXAttention, __init__
  - `python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunks: -189,28 +189,62 @@ def adjust_config_with_unaligned_cpu_tp(; symbols: adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunks: -32,6 +32,7; -57,6 +58,65; symbols: hacked_preprocess, smart_resize
  - `python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunks: -72,7 +72,14; -88,6 +95,10; symbols: Qwen3_VisionMLP, __init__, forward
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +34/-11 (45 lines); hunks: -32,7 +32,11; -43,7 +47,9; symbols: Qwen3OmniMoeAudioEncoderLayer, __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +80/-12; `python/sglang/srt/configs/update_config.py` modified +54/-20; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0; `python/sglang/srt/models/qwen3_vl.py` modified +44/-6; `python/sglang/srt/models/qwen3_omni_moe.py` modified +34/-11; `python/sglang/srt/layers/conv.py` modified +19/-0
  - other: `sgl-kernel/csrc/cpu/gemm.cpp` modified +142/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/layers/amx_utils.py`, `python/sglang/srt/layers/attention/vision.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12703 - add qwen3-omni docs

- Link: https://github.com/sgl-project/sglang/pull/12703
- Status/date: open / 2025-11-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +150/-0, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add qwen3-omni docs"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`; technical summary: Covers "add qwen3-omni docs"; the main implementation surface is `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunks: -0,0 +1,149; `docs/index.rst` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ Its core features include:.
- Code diff details:
  - `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunks: -0,0 +1,149
  - `docs/index.rst` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ Its core features include:
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/basic_usage/qwen3_omni.md` added +149/-0; `docs/index.rst` modified +1/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12240 - [VLM] Optimize qwen_vl preprocess_video

- Link: https://github.com/sgl-project/sglang/pull/12240
- Status/date: merged / 2025-11-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/qwen_vl.py`; associated commits `fd3034da7515`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +43/-10, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Optimize qwen_vl preprocess_video"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`; technical summary: Covers "[VLM] Optimize qwen_vl preprocess_video"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10 (53 lines); hunks: -2,8 +2,10; -175,19 +177,25 @@ async def preprocess_video(; symbols: preprocess_video, process_mm_data_async, touching `preprocess_video, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10 (53 lines); hunks: -2,8 +2,10; -175,19 +177,25 @@ async def preprocess_video(; symbols: preprocess_video, process_mm_data_async
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +43/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/qwen_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12554 - [Docs] Add docs for Qwen3-VL image and video support

- Link: https://github.com/sgl-project/sglang/pull/12554
- Status/date: merged / 2025-11-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/qwen3_vl.md`; associated commits `583bb1804e4c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +131/-0, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add docs for Qwen3-VL image and video support"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `docs/basic_usage/qwen3_vl.md`; technical summary: Covers "[Docs] Add docs for Qwen3-VL image and video support"; the main implementation surface is `docs/basic_usage/qwen3_vl.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunks: -0,0 +1,130.
- Code diff details:
  - `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunks: -0,0 +1,130
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/basic_usage/qwen3_vl.md` added +130/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/qwen3_vl.md`, `docs/index.rst`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12458 - fix: duplicate resize images logic of qwen-vl series models

- Link: https://github.com/sgl-project/sglang/pull/12458
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/qwen_vl.py`; associated commits `ffeb28ba6fb7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +1/-67, 136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: duplicate resize images logic of qwen-vl series models"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`; technical summary: Covers "fix: duplicate resize images logic of qwen-vl series models"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40 (40 lines); hunks: -1,4 +1,3; -79,26 +78,6 @@ def smart_resize(; symbols: smart_resize, resize_image, round_by_factor, floor_by_factor, touching `smart_resize, resize_image, round_by_factor`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40 (40 lines); hunks: -1,4 +1,3; -79,26 +78,6 @@ def smart_resize(; symbols: smart_resize, resize_image, round_by_factor, floor_by_factor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +0/-40
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/dots_vlm.py`, `python/sglang/srt/multimodal/processors/points_v15_chat.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13075 - [VLM] Support PP for Qwen2.5-VL

- Link: https://github.com/sgl-project/sglang/pull/13075
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`; associated commits `706502ff6cff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +88/-54, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support PP for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[VLM] Support PP for Qwen2.5-VL"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15 (59 lines); hunks: -40,6 +40,7; -50,13 +51,14; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15 (59 lines); hunks: -40,6 +40,7; -50,13 +51,14; symbols: __init__, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13126 - [VLM][feat] Support encoder DP for Qwen2.5-VL

- Link: https://github.com/sgl-project/sglang/pull/13126
- Status/date: merged / 2025-11-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`; associated commits `ac81db66c2d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +595/-6, 790 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM][feat] Support encoder DP for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[VLM][feat] Support encoder DP for Qwen2.5-VL"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2 (46 lines); hunks: -40,6 +40,10; -62,6 +66,8; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +44/-2
- Risk and verification: The diff ships test coverage in `test/srt/nightly/test_encoder_dp.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13055 - [VLM] Support Piecewise CUDA Graph for Qwen2.5-VL

- Link: https://github.com/sgl-project/sglang/pull/13055
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; associated commits `af6bcadcf723`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +710/-29, 970 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support Piecewise CUDA Graph for Qwen2.5-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; technical summary: Covers "[VLM] Support Piecewise CUDA Graph for Qwen2.5-VL"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7 (45 lines); hunks: -57,11 +57,12; -566,6 +567,25 @@ def get_video_feature(self, items: List[MultimodalDataItem]...; symbols: get_video_feature, post_process, get_input_embeddings, forward, touching `get_video_feature, post_process, get_input_embeddings`; `python/sglang/srt/models/qwen2_vl.py` modified +16/-7 (23 lines); hunks: -39,10 +39,7; -509,6 +506,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7 (45 lines); hunks: -57,11 +57,12; -566,6 +567,25 @@ def get_video_feature(self, items: List[MultimodalDataItem]...; symbols: get_video_feature, post_process, get_input_embeddings, forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +16/-7 (23 lines); hunks: -39,10 +39,7; -509,6 +506,7 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +38/-7; `python/sglang/srt/models/qwen2_vl.py` modified +16/-7
- Risk and verification: The diff ships test coverage in `test/srt/nightly/test_vlms_piecewise_cuda_graph.py`, `test/srt/run_suite.py`, `test/srt/test_piecewise_cuda_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13736 - [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series

- Link: https://github.com/sgl-project/sglang/pull/13736
- Status/date: merged / 2025-11-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; associated commits `5625e32cae12`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +169/-13, 229 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_vl.py`; technical summary: Covers "[VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -434,15 +435,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunks: -44,6 +44,7; -387,10 +388,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunks: -46,6 +46,7; -434,15 +435,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunks: -44,6 +44,7; -387,10 +388,7 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +2/-9; `python/sglang/srt/models/qwen2_vl.py` modified +2/-4
- Risk and verification: The diff ships test coverage in `test/srt/ops/test_repeat_interleave.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13918 - [VLM] support qwen3-vl eagle infer

- Link: https://github.com/sgl-project/sglang/pull/13918
- Status/date: open / 2025-11-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +30/-3, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] support qwen3-vl eagle infer"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`; technical summary: Covers "[VLM] support qwen3-vl eagle infer"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunks: -624,6 +624,9 @@ def __init__(; -711,9 +714,13 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights, touching `__init__, separate_deepstack_embeds, forward`; `python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunks: -116,9 +116,14 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunks: -624,6 +624,9 @@ def __init__(; -711,9 +714,13 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunks: -116,9 +116,14 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +23/-1; `python/sglang/srt/models/llama_eagle3.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/llama_eagle3.py`, `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13983 - Support KTransformers for Qwen3-VL moe

- Link: https://github.com/sgl-project/sglang/pull/13983
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl_moe.py`; associated commits `15ff6982b94f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-0, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support KTransformers for Qwen3-VL moe"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "Support KTransformers for Qwen3-VL moe"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0 (9 lines); hunks: -22,6 +22,7; -326,5 +327,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location, touching `load_weights, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0 (9 lines); hunks: -22,6 +22,7; -326,5 +327,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, get_model_config_for_expert_location
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13904 - [Bugfix] qwen2.5-vl spec decode accept_len low

- Link: https://github.com/sgl-project/sglang/pull/13904
- Status/date: merged / 2025-11-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`; associated commits `f6e37d3edb94`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] qwen2.5-vl spec decode accept_len low"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[Bugfix] qwen2.5-vl spec decode accept_len low"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -654,6 +654,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: -654,6 +654,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen2_5_vl.py
@@ -654,6 +654,7 @@ def forward(
+                    aux_hidden_states,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13724 - support qwen3_vl vision model dp

- Link: https://github.com/sgl-project/sglang/pull/13724
- Status/date: merged / 2025-11-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `ea1e9f6b3c3d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +50/-2, 208 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support qwen3_vl vision model dp"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "support qwen3_vl vision model dp"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunks: -28,6 +28,10; -47,6 +51,8; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +49/-2
- Risk and verification: The diff ships test coverage in `test/nightly/test_encoder_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14292 - [VLM] Introduce Cache for positional embedding ids for Qwen-VL family

- Link: https://github.com/sgl-project/sglang/pull/14292
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; associated commits `b2b09f5f24b9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +48/-47, 149 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Introduce Cache for positional embedding ids for Qwen-VL family"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[VLM] Introduce Cache for positional embedding ids for Qwen-VL family"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunks: -69,7 +69,7; -246,7 +246,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen2_5_VisionTransformer, __init__, device, touching `forward, Qwen2_5_VisionTransformer, __init__`; `python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunks: -50,7 +50,7; -257,7 +257,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen3VLMoeVisionModel, __init__, device, touching `forward, Qwen3VLMoeVisionModel, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunks: -69,7 +69,7; -246,7 +246,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen2_5_VisionTransformer, __init__, device
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunks: -50,7 +50,7; -257,7 +257,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, Qwen3VLMoeVisionModel, __init__, device
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25; `python/sglang/srt/models/qwen3_vl.py` modified +5/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11791 - fix rmsnorm -> layernorm in qwen3 omni

- Link: https://github.com/sgl-project/sglang/pull/11791
- Status/date: merged / 2025-12-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_omni_moe.py`; associated commits `2ac5b9839508`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix rmsnorm -> layernorm in qwen3 omni"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_omni_moe.py`; technical summary: Covers "fix rmsnorm -> layernorm in qwen3 omni"; the main implementation surface is `python/sglang/srt/models/qwen3_omni_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2 (3 lines); hunks: -31,7 +31,6; -318,7 +317,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2 (3 lines); hunks: -31,7 +31,6; -318,7 +317,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_omni_moe.py
@@ -31,7 +31,6 @@
-from sglang.srt.layers.layernorm import RMSNorm
@@ -318,7 +317,7 @@ def __init__(
-        self.ln_q = RMSNorm(
+        self.ln_q = nn.LayerNorm(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_omni_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14886 - Support qwen3-omni with DP Encoder

- Link: https://github.com/sgl-project/sglang/pull/14886
- Status/date: open / 2025-12-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +32/-3, 151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support qwen3-omni with DP Encoder"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`; technical summary: Covers "Support qwen3-omni with DP Encoder"; the main implementation surface is `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunks: -30,6 +30,10; -42,6 +46,7; symbols: __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder, touching `__init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder`; `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -21,6 +21,7.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunks: -30,6 +30,10; -42,6 +46,7; symbols: __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: -21,6 +21,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3
  - tests: `test/nightly/test_encoder_dp.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/nightly/test_encoder_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14907 - [VLM] Support chunked vit attention

- Link: https://github.com/sgl-project/sglang/pull/14907
- Status/date: merged / 2025-12-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +363/-8, 413 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support chunked vit attention"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/mm_utils.py`; technical summary: Covers "[VLM] Support chunked vit attention"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/mm_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunks: -14,6 +14,7; -53,7 +54,7; symbols: get_image_feature, get_video_feature, touching `get_image_feature, get_video_feature`; `python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunks: -41,6 +41,9; -414,6 +417,67 @@ def _get_precomputed_embedding(; symbols: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding, touching `init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunks: -14,6 +14,7; -53,7 +54,7; symbols: get_image_feature, get_video_feature
  - `python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunks: -41,6 +41,9; -414,6 +417,67 @@ def _get_precomputed_embedding(; symbols: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +97/-8; `python/sglang/srt/managers/mm_utils.py` modified +266/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12333 - [PP] Add pp support for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/12333
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/models/qwen3_vl.py`; associated commits `45a959d3e971`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +119/-20, 243 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PP] Add pp support for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py`; technical summary: Covers "[PP] Add pp support for Qwen3-VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_omni_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunks: -32,11 +32,13; -599,6 +601,7 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunks: -614,7 +614,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunks: -32,11 +32,13; -599,6 +601,7 @@ def __init__(; symbols: __init__, forward, load_weights
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunks: -614,7 +614,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +53/-19; `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15138 - [bug fix][pp] fix weight load for qwen2.5-vl

- Link: https://github.com/sgl-project/sglang/pull/15138
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`; associated commits `0071fe9c407a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-4, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bug fix][pp] fix weight load for qwen2.5-vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[bug fix][pp] fix weight load for qwen2.5-vl"; the main implementation surface is `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4 (14 lines); hunks: -743,6 +743,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -789,10 +797,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4 (14 lines); hunks: -743,6 +743,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -789,10 +797,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_5_vl.py` modified +10/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/15205
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `8fa3dc36c565`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +100/-80, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device, touching `forward, __init__, dtype`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: -24,9 +24,6; -39,6 +36,7; symbols: forward, __init__, dtype, device
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +41/-20
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/glm4v.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15320 - [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/15320
- Status/date: merged / 2025-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; associated commits `019517a35610`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +233/-64, 497 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`; technical summary: Covers "[VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunks: -57,8 +57,9; -188,6 +189,7 @@ def forward(; symbols: forward, __init__, dtype, touching `forward, __init__, dtype`; `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -170,7 +170,6 @@ def forward(; -182,7 +181,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunks: -57,8 +57,9; -188,6 +189,7 @@ def forward(; symbols: forward, __init__, dtype
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunks: -170,7 +170,6 @@ def forward(; -182,7 +181,7 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +52/-1; `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_vlms_vit_cuda_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15570 - [GLM-ASR] GLM-ASR Support

- Link: https://github.com/sgl-project/sglang/pull/15570
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`; associated commits `82f1d6157f8e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +226/-1, 250 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM-ASR] GLM-ASR Support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`; technical summary: Covers "[GLM-ASR] GLM-ASR Support"; the main implementation surface is `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glmasr.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: GlmasrForConditionalGeneration, __init__, pad_input_ids, get_audio_feature, touching `GlmasrForConditionalGeneration, __init__, pad_input_ids`; `python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: GlmasrProcessor, __init__, process_mm_data_async, touching `GlmasrProcessor, __init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/glmasr.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: GlmasrForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: GlmasrProcessor, __init__, process_mm_data_async
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/glmasr.py` added +171/-0; `python/sglang/srt/multimodal/processors/glmasr.py` added +53/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15772 - Change GLM-ASR class name

- Link: https://github.com/sgl-project/sglang/pull/15772
- Status/date: merged / 2025-12-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`; associated commits `f3ba71166262`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +19/-15, 103 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Change GLM-ASR class name"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`; technical summary: Covers "Change GLM-ASR class name"; the main implementation surface is `python/sglang/srt/models/glmasr.py`, `python/sglang/srt/multimodal/processors/glmasr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glmasr.py` modified +9/-9 (18 lines); hunks: -22,10 +22,10; -46,7 +46,7; symbols: GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__, touching `GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__`; `python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3 (6 lines); hunks: -1,14 +1,14; symbols: GlmasrProcessor, GlmAsrProcessor, __init__, touching `GlmasrProcessor, GlmAsrProcessor, __init__`.
- Code diff details:
  - `python/sglang/srt/models/glmasr.py` modified +9/-9 (18 lines); hunks: -22,10 +22,10; -46,7 +46,7; symbols: GlmasrForConditionalGeneration, GlmAsrForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3 (6 lines); hunks: -1,14 +1,14; symbols: GlmasrProcessor, GlmAsrProcessor, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/glmasr.py` modified +9/-9; `python/sglang/srt/multimodal/processors/glmasr.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/models/glmasr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16491 - [Qwen3-VL][PP] Skip loading expert weights not on this rank

- Link: https://github.com/sgl-project/sglang/pull/16491
- Status/date: open / 2026-01-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3-VL][PP] Skip loading expert weights not on this rank"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "[Qwen3-VL][PP] Skip loading expert weights not on this rank"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunks: -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunks: -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -274,6 +274,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    if name_mapped not in params_dict:
+                        # Expert weight not on this rank, will be skipped below
+                        continue
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16571 - [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT

- Link: https://github.com/sgl-project/sglang/pull/16571
- Status/date: open / 2026-01-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +87/-15, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`; technical summary: Covers "[Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunks: -42,6 +42,7; -72,7 +73,7; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLMoeVisionPatchMerger, touching `Qwen3_VisionMLP, __init__, forward`; `python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunks: -67,6 +67,7; -367,6 +368,9 @@ def __init__(; symbols: __init__, forward_cuda, forward_cpu, forward_aiter, touching `__init__, forward_cuda, forward_cpu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunks: -42,6 +42,7; -72,7 +73,7; symbols: Qwen3_VisionMLP, __init__, forward, Qwen3VLMoeVisionPatchMerger
  - `python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunks: -67,6 +67,7; -367,6 +368,9 @@ def __init__(; symbols: __init__, forward_cuda, forward_cpu, forward_aiter
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +60/-15; `python/sglang/srt/layers/layernorm.py` modified +27/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16785 - [Bugfix] fix recompile in qwen3 vl

- Link: https://github.com/sgl-project/sglang/pull/16785
- Status/date: open / 2026-01-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +113/-36, 307 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix recompile in qwen3 vl"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; technical summary: Covers "[Bugfix] fix recompile in qwen3 vl"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunks: -913,16 +913,37 @@ def __init__(; -963,7 +984,7 @@ def forward(; symbols: __init__, get_deepstack_embeds, forward, touching `__init__, get_deepstack_embeds, forward`; `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunks: -57,19 +57,40 @@ def __init__(; -109,7 +130,7 @@ def forward(; symbols: __init__, get_input_embeddings, get_deepstack_embeds, forward, touching `__init__, get_input_embeddings, get_deepstack_embeds`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunks: -245,6 +245,18 @@ def __init__(self, model_runner: ModelRunner):; -392,6 +404,9 @@ def warmup_torch_compile(self, num_tokens: int):; symbols: __init__, warmup_torch_compile, _cache_loc_dtype, capture_one_batch_size, touching `__init__, warmup_torch_compile, _cache_loc_dtype`; `python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunks: -919,6 +919,7 @@ def embed_mm_inputs(; -1019,12 +1020,16 @@ def embed_mm_inputs(; symbols: embed_mm_inputs, general_mm_embed_routine, touching `embed_mm_inputs, general_mm_embed_routine`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunks: -913,16 +913,37 @@ def __init__(; -963,7 +984,7 @@ def forward(; symbols: __init__, get_deepstack_embeds, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunks: -57,19 +57,40 @@ def __init__(; -109,7 +130,7 @@ def forward(; symbols: __init__, get_input_embeddings, get_deepstack_embeds, forward
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunks: -245,6 +245,18 @@ def __init__(self, model_runner: ModelRunner):; -392,6 +404,9 @@ def warmup_torch_compile(self, num_tokens: int):; symbols: __init__, warmup_torch_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunks: -919,6 +919,7 @@ def embed_mm_inputs(; -1019,12 +1020,16 @@ def embed_mm_inputs(; symbols: embed_mm_inputs, general_mm_embed_routine
  - `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3 (8 lines); hunks: -18,7 +18,9; -59,7 +61,7 @@ def run_mmmu_eval(; symbols: run_mmmu_eval, _run_vlm_mmmu_test
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +41/-18; `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1; `python/sglang/srt/managers/mm_utils.py` modified +13/-6
  - tests: `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_vlms_piecewise_cuda_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16996 - feat: Support 'use_audio_in_video' option for qwen3omnimoe model

- Link: https://github.com/sgl-project/sglang/pull/16996
- Status/date: open / 2026-01-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +129/-12, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Support 'use_audio_in_video' option for qwen3omnimoe model"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/entrypoints/openai/protocol.py`; technical summary: Covers "feat: Support 'use_audio_in_video' option for qwen3omnimoe model"; the main implementation surface is `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/entrypoints/openai/protocol.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunks: -324,7 +324,22 @@ def process_mm_data(; -405,6 +420,7 @@ def _load_single_item(; symbols: process_mm_data, _load_single_item, submit_data_loading_tasks, touching `process_mm_data, _load_single_item, submit_data_loading_tasks`; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunks: -151,6 +151,7 @@ async def preprocess_video(; -201,8 +202,16 @@ async def preprocess_video(; symbols: preprocess_video, __init__, process_mm_data_async, touching `preprocess_video, __init__, process_mm_data_async`; `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunks: -552,6 +552,8 @@ class ChatCompletionRequest(BaseModel):; -698,6 +700,7 @@ def get_param(param_name: str):; symbols: ChatCompletionRequest, get_param, touching `ChatCompletionRequest, get_param`; `python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunks: -72,6 +72,7; -878,6 +879,43 @@ def load_audio(; symbols: load_audio, extract_audio_via_av, ImageData, get_image_bytes, touching `load_audio, extract_audio_via_av, ImageData`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunks: -324,7 +324,22 @@ def process_mm_data(; -405,6 +420,7 @@ def _load_single_item(; symbols: process_mm_data, _load_single_item, submit_data_loading_tasks
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunks: -151,6 +151,7 @@ async def preprocess_video(; -201,8 +202,16 @@ async def preprocess_video(; symbols: preprocess_video, __init__, process_mm_data_async
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunks: -552,6 +552,8 @@ class ChatCompletionRequest(BaseModel):; -698,6 +700,7 @@ def get_param(param_name: str):; symbols: ChatCompletionRequest, get_param
  - `python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunks: -72,6 +72,7; -878,6 +879,43 @@ def load_audio(; symbols: load_audio, extract_audio_via_av, ImageData, get_image_bytes
  - `python/sglang/srt/managers/io_struct.py` modified +3/-0 (3 lines); hunks: -273,6 +273,9 @@ class GenerateReqInput(BaseReq, APIServingTimingMixin):; symbols: GenerateReqInput, contains_mm_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2; `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0; `python/sglang/srt/utils/common.py` modified +63/-7; `python/sglang/srt/managers/io_struct.py` modified +3/-0; `python/pyproject_npu.toml` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/pyproject_npu.toml`, `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/managers/io_struct.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17202 - [Feat] Accelerate qwen3vl by remove cpu op

- Link: https://github.com/sgl-project/sglang/pull/17202
- Status/date: open / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +27/-9, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] Accelerate qwen3vl by remove cpu op"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py`; technical summary: Covers "[Feat] Accelerate qwen3vl by remove cpu op"; the main implementation surface is `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunks: -1005,9 +1005,9 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunks: -1029,18 +1029,36 @@ def embed_mm_inputs(; symbols: embed_mm_inputs, touching `embed_mm_inputs`.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunks: -1005,9 +1005,9 @@ def forward(; symbols: forward
  - `python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunks: -1029,18 +1029,36 @@ def embed_mm_inputs(; symbols: embed_mm_inputs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +3/-3; `python/sglang/srt/managers/mm_utils.py` modified +24/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/mm_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17276 - Add Qwen3VL Eagle3 Inference Support

- Link: https://github.com/sgl-project/sglang/pull/17276
- Status/date: open / 2026-01-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +35/-0, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Qwen3VL Eagle3 Inference Support"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "Add Qwen3VL Eagle3 Inference Support"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunks: -733,6 +733,9 @@ def __init__(; -920,13 +923,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights, touching `__init__, separate_deepstack_embeds, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunks: -733,6 +733,9 @@ def __init__(; -920,13 +923,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +35/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16366 - Optimize Qwen3-VL video memory usage

- Link: https://github.com/sgl-project/sglang/pull/16366
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `0c2993eed03a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize Qwen3-VL video memory usage"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "Optimize Qwen3-VL video memory usage"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunks: -852,10 +852,18 @@ def get_image_feature(self, items: List[MultimodalDataItem...; symbols: get_image_feature, get_video_feature, touching `get_image_feature, get_video_feature`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunks: -852,10 +852,18 @@ def get_image_feature(self, items: List[MultimodalDataItem...; symbols: get_image_feature, get_video_feature
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- Link: https://github.com/sgl-project/sglang/pull/17624
- Status/date: merged / 2026-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +48/-19, 185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] Fix dp size > 1 for qwen3 vl model"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`; technical summary: Covers "[BUGFIX] Fix dp size > 1 for qwen3 vl model"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunks: -25,14 +25,15; -85,10 +86,8 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunks: -495,11 +495,19 @@ def run_dp_sharded_mrope_vision_model(; -611,7 +619,9 @@ def run_dp_sharded_mrope_vision_model(; symbols: run_dp_sharded_mrope_vision_model, touching `run_dp_sharded_mrope_vision_model`; `python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -1262,6 +1265,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunks: -860,7 +860,15 @@ def _pad_inputs_to_size(self, model_runner: ModelRunner, nu...; symbols: _pad_inputs_to_size, touching `_pad_inputs_to_size`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunks: -25,14 +25,15; -85,10 +86,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunks: -495,11 +495,19 @@ def run_dp_sharded_mrope_vision_model(; -611,7 +619,9 @@ def run_dp_sharded_mrope_vision_model(; symbols: run_dp_sharded_mrope_vision_model
  - `python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -1262,6 +1265,7 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunks: -860,7 +860,15 @@ def _pad_inputs_to_size(self, model_runner: ModelRunner, nu...; symbols: _pad_inputs_to_size
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-0 (2 lines); hunks: -538,6 +538,7 @@ def __init__(; -640,6 +641,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +14/-13; `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3; `python/sglang/srt/layers/linear.py` modified +10/-2; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1; `python/sglang/srt/layers/attention/vision.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18024 - fix: correct weight loading prefix mapping for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/18024
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `522e13b4d2c1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-1, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: correct weight loading prefix mapping for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "fix: correct weight loading prefix mapping for Qwen3-VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -959,7 +959,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunks: -959,7 +959,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16781 - Refactor(qwen3-vl) optimize position encoding interpolation

- Link: https://github.com/sgl-project/sglang/pull/16781
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `6a4b81e2d9fc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +126/-25, 175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor(qwen3-vl) optimize position encoding interpolation"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "Refactor(qwen3-vl) optimize position encoding interpolation"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +123/-23 (146 lines); hunks: -19,6 +19,7; -396,31 +397,130 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices, _calculate_indices_and_weights, touching `rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +123/-23 (146 lines); hunks: -19,6 +19,7; -396,31 +397,130 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, _get_interpolation_indices, _calculate_indices_and_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +123/-23
- Risk and verification: The diff ships test coverage in `test/srt/test_embed_interpolate_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18721 - [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models

- Link: https://github.com/sgl-project/sglang/pull/18721
- Status/date: open / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models"; the main implementation surface is `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunks: -224,7 +224,8 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -308,6 +308,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunks: -224,7 +224,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -308,6 +308,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18771 - Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton

- Link: https://github.com/sgl-project/sglang/pull/18771
- Status/date: open / 2026-02-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `benchmark/kernels/fused_moe_triton/common_utils.py`; technical summary: Covers "Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton"; the main implementation surface is `benchmark/kernels/fused_moe_triton/common_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ def get_model_config(; symbols: get_model_config, touching `get_model_config`.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ def get_model_config(; symbols: get_model_config
- Key code excerpts:

```diff
diff -- benchmark/kernels/fused_moe_triton/common_utils.py
@@ -67,6 +67,7 @@ def get_model_config(
+        "Qwen3OmniMoeForConditionalGeneration",
```

- Reviewed files:
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #19242 - [feat] feat: add Qwen3-ASR support like whisper

- Link: https://github.com/sgl-project/sglang/pull/19242
- Status/date: open / 2026-02-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +475/-0, 519 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feat] feat: add Qwen3-ASR support like whisper"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`; technical summary: Covers "[feat] feat: add Qwen3-ASR support like whisper"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length, touching `Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor`; `python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig, touching `Qwen3ASRHFProcessor, __init__, from_pretrained`; `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -47,6 +48,7; `python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunks: -1259,6 +1259,7 @@ def is_generation_model(model_architectures: List[str], is...; -1299,6 +1300,7 @@ def is_image_gen_model(model_architectures: List[str]):; symbols: is_generation_model, is_image_gen_model, is_audio_model, touching `is_generation_model, is_image_gen_model, is_audio_model`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length
  - `python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunks: -0,0 +1,217; symbols: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -47,6 +48,7
  - `python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunks: -1259,6 +1259,7 @@ def is_generation_model(model_architectures: List[str], is...; -1299,6 +1300,7 @@ def is_image_gen_model(model_architectures: List[str]):; symbols: is_generation_model, is_image_gen_model, is_audio_model
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -64,6 +64,7; -104,6 +105,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0; `python/sglang/srt/configs/qwen3_asr.py` added +217/-0; `python/sglang/srt/configs/__init__.py` modified +2/-0; `python/sglang/srt/configs/model_config.py` modified +2/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19003 - [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend

- Link: https://github.com/sgl-project/sglang/pull/19003
- Status/date: merged / 2026-02-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +678/-14, 862 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`; technical summary: Covers "[VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunks: -30,7 +30,12; -66,15 +71,12; symbols: Qwen3_VisionMLP, __init__, forward, touching `Qwen3_VisionMLP, __init__, forward`; `python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunks: -34,6 +34,7; -64,6 +65,24; symbols: SingletonCache, forward, VisionFlashInferAttention, __init__, touching `SingletonCache, forward, VisionFlashInferAttention`; `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test, touching `TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval`; `python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunks: -3859,7 +3859,15 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args, touching `add_cli_args`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunks: -30,7 +30,12; -66,15 +71,12; symbols: Qwen3_VisionMLP, __init__, forward
  - `python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunks: -34,6 +34,7; -64,6 +65,24; symbols: SingletonCache, forward, VisionFlashInferAttention, __init__
  - `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test
  - `python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunks: -3859,7 +3859,15 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: add_cli_args
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +259/-13; `python/sglang/srt/layers/attention/vision.py` modified +152/-0; `python/sglang/srt/server_args.py` modified +9/-1
  - tests: `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19333 - fix qwen3_vl visual module loading

- Link: https://github.com/sgl-project/sglang/pull/19333
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `d566816d838c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix qwen3_vl visual module loading"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "fix qwen3_vl visual module loading"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -1357,6 +1357,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    name = name.replace(r"model.visual.", r"visual.")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19693 - [NPU] Fix Qwen3-VL-8B Accuracy for NPU

- Link: https://github.com/sgl-project/sglang/pull/19693
- Status/date: open / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +199/-108, 518 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Fix Qwen3-VL-8B Accuracy for NPU"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/layers/rotary_embedding.py`; technical summary: Covers "[NPU] Fix Qwen3-VL-8B Accuracy for NPU"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/layers/rotary_embedding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunks: -19,6 +19,7; -397,30 +398,89 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, forward, Qwen3VLForConditionalGeneration, touching `rot_pos_emb, fast_pos_embed_interpolate, forward`; `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: -52,10 +52,14; -185,15 +189,44 @@ def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu, touching `LlamaMLP, __init__, forward_prepare_native`; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-3 (7 lines); hunks: -115,9 +115,10 @@ def __init__(; -294,8 +295,8 @@ def forward_npu(; symbols: __init__, forward_npu, touching `__init__, forward_npu`; `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__, touching `forward_prepare_npu, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunks: -19,6 +19,7; -397,30 +398,89 @@ def rot_pos_emb(; symbols: rot_pos_emb, fast_pos_embed_interpolate, forward, Qwen3VLForConditionalGeneration
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunks: -52,10 +52,14; -185,15 +189,44 @@ def __init__(; symbols: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-3 (7 lines); hunks: -115,9 +115,10 @@ def __init__(; -294,8 +295,8 @@ def forward_npu(; symbols: __init__, forward_npu
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +79/-20; `python/sglang/srt/models/llama.py` modified +37/-4; `python/sglang/srt/layers/rotary_embedding.py` modified +4/-3; `python/sglang/srt/models/qwen3.py` modified +4/-3; `python/sglang/srt/models/qwen3_moe.py` modified +3/-3; `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19291 - [Qwen3.5] Fix missing `quant_config` in `Qwen3VL`

- Link: https://github.com/sgl-project/sglang/pull/19291
- Status/date: merged / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen3.5] Fix missing `quant_config` in `Qwen3VL`"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[Qwen3.5] Fix missing `quant_config` in `Qwen3VL`"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1025,6 +1025,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -1025,6 +1025,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -1025,6 +1025,7 @@ def __init__(
+        self.quant_config = quant_config
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18185 - [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker

- Link: https://github.com/sgl-project/sglang/pull/18185
- Status/date: merged / 2026-03-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_omni_moe.py`; associated commits `22e67876d6aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +52/-28, 130 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_omni_moe.py`; technical summary: Covers "[Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker"; the main implementation surface is `python/sglang/srt/models/qwen3_omni_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunks: -70,8 +70,18 @@ def __init__(; -98,9 +108,9 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunks: -70,8 +70,18 @@ def __init__(; -98,9 +108,9 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_omni_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20788 - [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL

- Link: https://github.com/sgl-project/sglang/pull/20788
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -330,6 +330,7 @@ def __init__(
+                enable_tp=False if use_data_parallel else True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20857 - add EVS support for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/20857
- Status/date: open / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +151/-4, 269 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add EVS support for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "add EVS support for Qwen3-VL"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunks: -10,6 +10,7; -22,6 +23,9; symbols: __init__, _maybe_apply_qwen3_evs, get_mm_data, process_mm_data_async, touching `__init__, _maybe_apply_qwen3_evs, get_mm_data`; `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunks: -121,6 +121,7 @@ def get_rope_index(; -134,13 +135,22 @@ def get_rope_index(; symbols: get_rope_index, touching `get_rope_index`; `python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunks: -69,6 +69,7; -1047,7 +1048,7 @@ def forward(; symbols: forward, Qwen3VLForConditionalGeneration, __init__, touching `forward, Qwen3VLForConditionalGeneration, __init__`; `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunks: -65,6 +65,16 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunks: -10,6 +10,7; -22,6 +23,9; symbols: __init__, _maybe_apply_qwen3_evs, get_mm_data, process_mm_data_async
  - `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunks: -121,6 +121,7 @@ def get_rope_index(; -134,13 +135,22 @@ def get_rope_index(; symbols: get_rope_index
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunks: -69,6 +69,7; -1047,7 +1048,7 @@ def forward(; symbols: forward, Qwen3VLForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunks: -65,6 +65,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0 (2 lines); hunks: -244,6 +244,7 @@ def __init__(; -261,6 +262,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0; `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2; `python/sglang/srt/models/qwen3_vl.py` modified +10/-2; `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0; `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/qwen3_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20759 - [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable

- Link: https://github.com/sgl-project/sglang/pull/20759
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -329,8 +329,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunks: -329,8 +329,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -329,8 +329,8 @@ def __init__(
-                use_attn_tp_group=is_dp_attention_enabled(),
-                enable_tp=False if use_data_parallel else True,
+                enable_tp=not use_data_parallel,
+                use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21469 - [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct

- Link: https://github.com/sgl-project/sglang/pull/21469
- Status/date: merged / 2026-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`; associated commits `cffc95edf455`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +152/-235, 397 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`; technical summary: Covers "[3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunks: -179,9 +179,8 @@ def __init__(; symbols: __init__, should_apply_lora, touching `__init__, should_apply_lora`; `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy, touching `kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunks: -179,9 +179,8 @@ def __init__(; symbols: __init__, should_apply_lora
  - `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2
  - tests: `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0
- Risk and verification: The diff ships test coverage in `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- Link: https://github.com/sgl-project/sglang/pull/21458
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +101/-3, 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope, touching `__init__, forward_prepare_native, forward_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19135 - qwen3 vl skip layer id for pp

- Link: https://github.com/sgl-project/sglang/pull/19135
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "qwen3 vl skip layer id for pp"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "qwen3 vl skip layer id for pp"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0 (12 lines); hunks: -26,6 +26,7; -232,6 +233,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0 (12 lines); hunks: -26,6 +26,7; -232,6 +233,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22052 - [Fix] Enable precise embedding interpolation by default for Qwen3-VL

- Link: https://github.com/sgl-project/sglang/pull/22052
- Status/date: open / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +10/-11, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Enable precise embedding interpolation by default for Qwen3-VL"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`; technical summary: Covers "[Fix] Enable precise embedding interpolation by default for Qwen3-VL"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunks: -310,7 +310,7 @@ def __init__(; -524,12 +524,8 @@ def fast_pos_embed_interpolate_from_list(self, grid_thw):; symbols: __init__, fast_pos_embed_interpolate_from_list, touching `__init__, fast_pos_embed_interpolate_from_list`; `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: -675,7 +675,7 @@ class ServerArgs:; -5732,9 +5732,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`; `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: -469,7 +469,7 @@ Please consult the documentation below and [server_args.py](....
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunks: -310,7 +310,7 @@ def __init__(; -524,12 +524,8 @@ def fast_pos_embed_interpolate_from_list(self, grid_thw):; symbols: __init__, fast_pos_embed_interpolate_from_list
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: -675,7 +675,7 @@ class ServerArgs:; -5732,9 +5732,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunks: -469,7 +469,7 @@ Please consult the documentation below and [server_args.py](...
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +3/-7; `python/sglang/srt/server_args.py` modified +6/-3
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22038 - [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer

- Link: https://github.com/sgl-project/sglang/pull/22038
- Status/date: merged / 2026-04-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +167/-410, 696 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`, `python/sglang/srt/models/deepseek_vl2.py`; technical summary: Covers "[VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`, `python/sglang/srt/models/deepseek_vl2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunks: -15,7 +15,6; -73,7 +72,7; symbols: get_image_feature, get_video_feature, touching `get_image_feature, get_video_feature`; `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunks: -120,6 +120,13 @@ def set(; symbols: set, get_single, has, touching `set, get_single, has`; `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunks: -270,9 +270,7 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, touching `get_image_feature`; `python/sglang/srt/models/phi4mm.py` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ def get_audio_feature(self, items: List[MultimodalDataItem])...; symbols: get_audio_feature, touching `get_audio_feature`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunks: -15,7 +15,6; -73,7 +72,7; symbols: get_image_feature, get_video_feature
  - `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunks: -120,6 +120,13 @@ def set(; symbols: set, get_single, has
  - `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunks: -270,9 +270,7 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature
  - `python/sglang/srt/models/phi4mm.py` modified +1/-1 (2 lines); hunks: -440,7 +440,7 @@ def get_audio_feature(self, items: List[MultimodalDataItem])...; symbols: get_audio_feature
  - `python/sglang/srt/models/step3_vl_10b.py` modified +1/-1 (2 lines); hunks: -484,7 +484,7 @@ def get_image_feature(self, items: List[MultimodalDataItem])...; symbols: get_image_feature
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +10/-104; `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0; `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3; `python/sglang/srt/models/phi4mm.py` modified +1/-1; `python/sglang/srt/models/step3_vl_10b.py` modified +1/-1; `python/sglang/srt/managers/mm_utils.py` modified +147/-286
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/mem_cache/multimodal_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- Link: https://github.com/sgl-project/sglang/pull/21849
- Status/date: merged / 2026-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +190/-3, 230 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM]: allow Qwen3.5 models for encoder disaggregation"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`; technical summary: Covers "[VLM]: allow Qwen3.5 models for encoder disaggregation"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, touching `get_mm_data`; `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode, touching `test_mmmu, TestEPDDisaggregationQwen35, setUpClass`; `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items, touching `_process_mm_items`; `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation, touching `_handle_encoder_disaggregation`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunks: -33,6 +33,7; -813,6 +814,189 @@ def test_mmmu(self):; symbols: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunks: -867,10 +867,11 @@ async def _process_mm_items(self, mm_items, modality):; symbols: _process_mm_items
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -3326,6 +3326,8 @@ def _handle_encoder_disaggregation(self):; symbols: _handle_encoder_disaggregation
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1; `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2; `python/sglang/srt/server_args.py` modified +2/-0
  - tests: `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_epd_disaggregation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- Link: https://github.com/sgl-project/sglang/pull/22073
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; associated commits `f6e85676b578`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +571/-11, 689 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Adding Qwen3-asr Model Support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; technical summary: Covers "[Feature] Adding Qwen3-asr Model Support"; the main implementation surface is `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature, touching `Qwen3ASRForConditionalGeneration, __init__, pad_input_ids`; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config, touching `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions, touching `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_asr.py` added +199/-0; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22266 - [NPU] fix qwen3.5 video processor

- Link: https://github.com/sgl-project/sglang/pull/22266
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +177/-21, 235 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] fix qwen3.5 video processor"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`; technical summary: Covers "[NPU] fix qwen3.5 video processor"; the main implementation surface is `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunks: -7,13 +7,62; -90,31 +139,16 @@ def _preprocess(; symbols: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, npu_wrapper_video_preprocess, touching `transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess`.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunks: -7,13 +7,62; -90,31 +139,16 @@ def _preprocess(; symbols: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, npu_wrapper_video_preprocess
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22181 - [refactor] [asr] Add transcription adapter for extensible ASR models support

- Link: https://github.com/sgl-project/sglang/pull/22181
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`; associated commits `a5ed507a1639`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +473/-223, 809 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[refactor] [asr] Add transcription adapter for extensible ASR models support"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/configs/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`; technical summary: Covers "[refactor] [asr] Add transcription adapter for extensible ASR models support"; the main implementation surface is `python/sglang/srt/configs/qwen3_asr.py`, `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67 (130 lines); hunks: -14,72 +14,6; -167,6 +101,68 @@ def __call__(self, text=None, audio=None, audio_kwargs=None...; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config, touching `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`; `test/manual/models/test_qwen3_asr.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: download_audio, TestQwen3ASRTranscription, setUpClass, tearDownClass, touching `download_audio, TestQwen3ASRTranscription, setUpClass`; `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: Qwen3ASRAdapter, build_sampling_params, postprocess_text, build_verbose_response, touching `Qwen3ASRAdapter, build_sampling_params, postprocess_text`; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5 (13 lines); hunks: -10,11 +10,13; -23,7 +25,7 @@ class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, touching `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`.
- Code diff details:
  - `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67 (130 lines); hunks: -14,72 +14,6; -167,6 +101,68 @@ def __call__(self, text=None, audio=None, audio_kwargs=None...; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `test/manual/models/test_qwen3_asr.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: download_audio, TestQwen3ASRTranscription, setUpClass, tearDownClass
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: Qwen3ASRAdapter, build_sampling_params, postprocess_text, build_verbose_response
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5 (13 lines); hunks: -10,11 +10,13; -23,7 +25,7 @@ class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/configs/qwen3_asr.py` modified +63/-67; `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` added +49/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +8/-5
  - tests: `test/manual/models/test_qwen3_asr.py` added +118/-0
- Risk and verification: The diff ships test coverage in `test/manual/models/test_qwen3_asr.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22230 - [Feature] Support eagle3 for qwen3-vl

- Link: https://github.com/sgl-project/sglang/pull/22230
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_vl.py`; associated commits `a69be2e866fb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-0, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support eagle3 for qwen3-vl"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[Feature] Support eagle3 for qwen3-vl"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunks: -1130,6 +1130,9 @@ def __init__(; -1246,13 +1249,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights, touching `__init__, separate_deepstack_embeds, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunks: -1130,6 +1130,9 @@ def __init__(; -1246,13 +1249,18 @@ def forward(; symbols: __init__, separate_deepstack_embeds, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl.py` modified +24/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22089 - [Feature] Add chunk-based streaming ASR for Qwen3-ASR

- Link: https://github.com/sgl-project/sglang/pull/22089
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; associated commits `8b991d98a12c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +263/-2, 325 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add chunk-based streaming ASR for Qwen3-ASR"; model line: Qwen VLM/Omni/ASR; category: model support/runtime entry; main diff: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`; technical summary: Covers "[Feature] Add chunk-based streaming ASR for Qwen3-ASR"; the main implementation surface is `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunks: -9,12 +9,32; symbols: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template, touching `Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config`; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -47,7 +47,7 @@ def _build_transcription_prompt(self, input_text: Union[str, l...; symbols: _build_transcription_prompt, compute_mrope_positions, touching `_build_transcription_prompt, compute_mrope_positions`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunks: -9,12 +9,32; symbols: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -47,7 +47,7 @@ def _build_transcription_prompt(self, input_text: Union[str, l...; symbols: _build_transcription_prompt, compute_mrope_positions
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22839 - fix(config): Add from_dict() for Qwen3VL config classes

- Link: https://github.com/sgl-project/sglang/pull/22839
- Status/date: open / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +306/-0, 389 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(config): Add from_dict() for Qwen3VL config classes"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`; technical summary: Covers "fix(config): Add from_dict() for Qwen3VL config classes"; the main implementation surface is `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion, touching `TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object`; `python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunks: -78,6 +78,17 @@ class Qwen3_5Config(PretrainedConfig):; -112,6 +123,55 @@ class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; symbols: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig, touching `Qwen3_5Config, from_dict, __init__`; `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunks: -236,6 +236,17 @@ class Qwen3VLConfig(PretrainedConfig):; -251,11 +262,15 @@ def __init__(; symbols: Qwen3VLConfig, from_dict, __init__, Qwen3VLMoeConfig, touching `Qwen3VLConfig, from_dict, __init__`; `python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunks: -25,6 +25,7; -65,4 +66,6.
- Code diff details:
  - `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion
  - `python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunks: -78,6 +78,17 @@ class Qwen3_5Config(PretrainedConfig):; -112,6 +123,55 @@ class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; symbols: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunks: -236,6 +236,17 @@ class Qwen3VLConfig(PretrainedConfig):; -251,11 +262,15 @@ def __init__(; symbols: Qwen3VLConfig, from_dict, __init__, Qwen3VLMoeConfig
  - `python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunks: -25,6 +25,7; -65,4 +66,6
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0 (4 lines); hunks: -87,6 +87,8; -121,6 +123,8
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0
  - runtime: `python/sglang/srt/configs/qwen3_5.py` modified +71/-0; `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0; `python/sglang/srt/configs/__init__.py` modified +3/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/configs/test_qwen3_vl_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22848 - [Feature] WebSocket streaming audio input for ASR

- Link: https://github.com/sgl-project/sglang/pull/22848
- Status/date: open / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +948/-43, 1176 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] WebSocket streaming audio input for ASR"; model line: Qwen VLM/Omni/ASR; category: model implementation change; main diff: `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`; technical summary: Covers "[Feature] WebSocket streaming audio input for ASR"; the main implementation surface is `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunks: -1,17 +1,30; -29,8 +42,96; symbols: _normalize_for_wer, _wer, download_audio, _pcm16_from_audio_bytes, touching `_normalize_for_wer, _wer, download_audio`; `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _pcm_to_wav, SessionStartEvent, SessionEndEvent, SessionStartedEvent, touching `_pcm_to_wav, SessionStartEvent, SessionEndEvent`; `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunks: -1,9 +1,21; -22,13 +34,23 @@ class StreamingASRState:; symbols: StreamingASRState, get_prefix_text, _record_emit, update, touching `StreamingASRState, get_prefix_text, _record_emit`; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunks: -29,7 +29,7; -43,8 +43,12; symbols: _generate_chunked_asr_stream, handle_websocket, touching `_generate_chunked_asr_stream, handle_websocket`.
- Code diff details:
  - `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunks: -1,17 +1,30; -29,8 +42,96; symbols: _normalize_for_wer, _wer, download_audio, _pcm16_from_audio_bytes
  - `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +377/-0 (377 lines); hunks: -0,0 +1,377; symbols: _pcm_to_wav, SessionStartEvent, SessionEndEvent, SessionStartedEvent
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunks: -1,9 +1,21; -22,13 +34,23 @@ class StreamingASRState:; symbols: StreamingASRState, get_prefix_text, _record_emit, update
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunks: -29,7 +29,7; -43,8 +43,12; symbols: _generate_chunked_asr_stream, handle_websocket
  - `python/sglang/srt/entrypoints/http_server.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -1597,6 +1598,12 @@ async def openai_v1_audio_transcriptions(; symbols: openai_v1_audio_transcriptions, openai_v1_audio_transcriptions_ws, available_models
- Key code excerpts:

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
@@ -0,0 +1,377 @@
+"""WebSocket transport for OpenAI Realtime API-style transcription.
+Protocol:
+    Client -> Server:
+        {"type": "session.start"}                                              # minimum
+        {"type": "session.start", "model": "<model-name>", "language": "en"}   # with hints
+        <binary PCM16 frame>     # 16 kHz mono LE, length must be % 2 == 0
diff -- python/sglang/srt/entrypoints/openai/streaming_asr.py
@@ -1,9 +1,21 @@
```

- Reviewed files:
  - tests: `test/manual/models/test_qwen3_asr.py` modified +451/-3
  - runtime: `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +377/-0; `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33; `python/sglang/srt/entrypoints/http_server.py` modified +7/-0; `python/sglang/srt/server_args.py` modified +20/-0
- Risk and verification: The diff ships test coverage in `test/manual/models/test_qwen3_asr.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23115 - fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights

- Link: https://github.com/sgl-project/sglang/pull/23115
- Status/date: open / 2026-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                and hasattr(self, "model")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- Link: https://github.com/sgl-project/sglang/pull/22431
- Status/date: merged / 2026-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Qwen3.5 video processing when passing video_data in "processor_output" format"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`; technical summary: Covers "Fix Qwen3.5 video processing when passing video_data in "processor_output" format"; the main implementation surface is `python/sglang/srt/multimodal/processors/qwen_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video, touching `preprocess_video`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunks: -162,7 +162,7 @@ async def preprocess_video(; symbols: preprocess_video
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -162,7 +162,7 @@ async def preprocess_video(
-        return vr
+        return vr, None
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/qwen_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23220 - Bugfix: Qwen3-VL-MoE adapt encoder_only

- Link: https://github.com/sgl-project/sglang/pull/23220
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Bugfix: Qwen3-VL-MoE adapt encoder_only"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl_moe.py`; technical summary: Covers "Bugfix: Qwen3-VL-MoE adapt encoder_only"; the main implementation surface is `python/sglang/srt/models/qwen3_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunks: -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_vl_moe.py
@@ -235,6 +235,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                and hasattr(self, "model")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23304 - [Bugfix] Fix Qwen3-VL rope config compatibility

- Link: https://github.com/sgl-project/sglang/pull/23304
- Status/date: closed / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-10, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3-VL rope config compatibility"; model line: Qwen VLM/Omni/ASR; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Bugfix] Fix Qwen3-VL rope config compatibility"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunks: -316,16 +316,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunks: -316,16 +316,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +1/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23469 - [NPU] adapt the Qwen3-ASR model for deployment on NPU

- Link: https://github.com/sgl-project/sglang/pull/23469
- Status/date: open / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-0, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] adapt the Qwen3-ASR model for deployment on NPU"; model line: Qwen VLM/Omni/ASR; category: performance/backend optimization; main diff: `python/sglang/srt/utils/common.py`; technical summary: Covers "[NPU] adapt the Qwen3-ASR model for deployment on NPU"; the main implementation surface is `python/sglang/srt/utils/common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: -740,6 +740,24 @@ def load_audio(; symbols: load_audio, touching `load_audio`.
- Code diff details:
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunks: -740,6 +740,24 @@ def load_audio(; symbols: load_audio
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/utils/common.py` modified +18/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/utils/common.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23907 - [Docs] add Nemotron 3 Nano Omni cookbook

- Link: https://github.com/sgl-project/sglang/pull/23907
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`; associated commits `ad785a229911`
- Diff scope read: GitHub Pull Request files API returned 5 files, +746/-1, 771 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] add Nemotron 3 Nano Omni cookbook"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`; technical summary: Covers "[Docs] add Nemotron 3 Nano Omni cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542; `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200.
- Code diff details:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542
  - `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -0,0 +1,542 @@
+---
+title: Nemotron 3 Nano Omni
+metatags:
+    description: "Deploy NVIDIA Nemotron 3 Nano Omni multimodal MoE model with SGLang - text, image, video, and audio inputs with reasoning and tool calling."
+tag:
+    NEW
diff -- docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx
@@ -0,0 +1,200 @@
+export const Nemotron3NanoOmniDeployment = () => {
+  const MODEL_PATHS = {
+    reasoning: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
+    bf16: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16',
+    fp8: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8',
+    nvfp4: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4',
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0; `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/cookbook/intro copy.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23968 - [Docs] update Docker image for Nemotron 3 Nano Omni

- Link: https://github.com/sgl-project/sglang/pull/23968
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`; associated commits `387c932dfc88`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] update Docker image for Nemotron 3 Nano Omni"; model line: Qwen VLM/Omni/ASR; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`; technical summary: Covers "[Docs] update Docker image for Nemotron 3 Nano Omni"; the main implementation surface is `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang.
- Code diff details:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ pip install sglang
-docker pull lmsysorg/sglang:nightly
+docker pull lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
