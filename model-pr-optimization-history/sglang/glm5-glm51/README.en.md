# sglang GLM-5/5.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | [#22712](https://github.com/sgl-project/sglang/pull/22712), [#23708](https://github.com/sgl-project/sglang/pull/23708) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` | [#27708](https://github.com/sgl-project/sglang/pull/27708), [#28437](https://github.com/sgl-project/sglang/pull/28437), [#29194](https://github.com/sgl-project/sglang/pull/29194), [#29313](https://github.com/sgl-project/sglang/pull/29313) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28454](https://github.com/sgl-project/sglang/pull/28454), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` | [#25814](https://github.com/sgl-project/sglang/pull/25814) |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | [#28433](https://github.com/sgl-project/sglang/pull/28433) |
| `docs_new/docs/hardware-platforms/ascend-npus/best_practice/glm5_1.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` | [#25814](https://github.com/sgl-project/sglang/pull/25814) |
| `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` | [#23540](https://github.com/sgl-project/sglang/pull/23540), [#27708](https://github.com/sgl-project/sglang/pull/27708), [#29194](https://github.com/sgl-project/sglang/pull/29194), [#29313](https://github.com/sgl-project/sglang/pull/29313) |
| `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380), [#29486](https://github.com/sgl-project/sglang/pull/29486) |
| `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380), [#29466](https://github.com/sgl-project/sglang/pull/29466), [#29486](https://github.com/sgl-project/sglang/pull/29486) |
| `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi30x/test_glm51_hisparse_eval_mi30x.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi35x/test_glm51_hisparse_eval_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` | [#26396](https://github.com/sgl-project/sglang/pull/26396) |
| `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/ascend/performance/glm5_1/test_npu_glm5_1_w4a8_1p1d_32p_in64k_out1k_50ms_aime26.py` | no direct PR-number commit |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp4.py` | no direct PR-number commit |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` | [#27053](https://github.com/sgl-project/sglang/pull/27053) |
| `test/registered/gb300/test_glm5_fp8.py` | [#22399](https://github.com/sgl-project/sglang/pull/22399) |
| `test/registered/gb300/test_glm5_nvfp4.py` | no direct PR-number commit |
| `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py` | no direct PR-number commit |
| `test/registered/models_e2e/test_dsa_glm5_hisparse.py` | [#28607](https://github.com/sgl-project/sglang/pull/28607) |
| `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 22
- Extra PRs preserved from existing docs: 22
- Total PRs in this document: 44
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-10 | [#18521](https://github.com/sgl-project/sglang/pull/18521) | merged | Support GlmMoeDsaForCausalLM | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py` |
| 2026-02-16 | [#18804](https://github.com/sgl-project/sglang/pull/18804) | merged | Fix GLM-5 fused shared expert | `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-25 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` |
| 2026-03-09 | [#20062](https://github.com/sgl-project/sglang/pull/20062) | merged | [V32/GLM5] Control the threshold of applying dense attention with an environ | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-08 | [#22314](https://github.com/sgl-project/sglang/pull/22314) | merged | [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300 | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#21710](https://github.com/sgl-project/sglang/pull/21710) | merged | [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` |
| 2026-04-08 | [#22285](https://github.com/sgl-project/sglang/pull/22285) | merged | Add CI tests for GLM-5 | `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-04-09 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | `test/registered/gb300/test_glm5_fp8.py` |
| 2026-04-09 | [#22336](https://github.com/sgl-project/sglang/pull/22336) | merged | [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x | `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` |
| 2026-04-13 | [#22712](https://github.com/sgl-project/sglang/pull/22712) | merged | [NPU] update GLM5 running guide | `docs/platforms/ascend/ascend_npu_glm5_examples.md` |
| 2026-04-14 | [#22543](https://github.com/sgl-project/sglang/pull/22543) | merged | GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#21773](https://github.com/sgl-project/sglang/pull/21773) | merged | [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x | `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` |
| 2026-04-16 | [#22595](https://github.com/sgl-project/sglang/pull/22595) | merged | fix: normalize tool message content for GLM5.1 chat template | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py` |
| 2026-04-19 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-23 | [#23060](https://github.com/sgl-project/sglang/pull/23060) | merged | [fix] Fix dynamic chunking profiling crash on GLM-5 models | `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-04-23 | [#23540](https://github.com/sgl-project/sglang/pull/23540) | merged | docs: split MI300X and MI325X options in GLM-5.1 generator | `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` |
| 2026-05-08 | [#23708](https://github.com/sgl-project/sglang/pull/23708) | merged | [NPU][Doc] Update GLM-5 docs, enabling deepep by default | `docs/platforms/ascend/ascend_npu_glm5_examples.md` |
| 2026-05-16 | [#25453](https://github.com/sgl-project/sglang/pull/25453) | merged | [CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85 | `test/registered/8-gpu-models/test_glm_51_fp8.py` |
| 2026-05-19 | [#25735](https://github.com/sgl-project/sglang/pull/25735) | merged | [NPU] [DOCS] Improved the usability of Ascend NPU documents | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` |
| 2026-05-20 | [#25814](https://github.com/sgl-project/sglang/pull/25814) | merged | Update GLM-5 H200 FP8 | `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` |
| 2026-05-20 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | merged | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-05-21 | [#25266](https://github.com/sgl-project/sglang/pull/25266) | merged | [AMD][CI] Clean up AMD nightly + pr-test workflows | `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `.github/workflows/pr-test-amd-rocm720.yml` |
| 2026-05-27 | [#26396](https://github.com/sgl-project/sglang/pull/26396) | merged | [AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate | `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-10 | [#27708](https://github.com/sgl-project/sglang/pull/27708) | merged | [Docs] Add GLM-5.1 NVFP4 to cookbook | `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-16 | [#28437](https://github.com/sgl-project/sglang/pull/28437) | merged | docs(cookbook): add GLM-5.2 deployment cookbook | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` |
| 2026-06-16 | [#28448](https://github.com/sgl-project/sglang/pull/28448) | merged | docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` |
| 2026-06-16 | [#28454](https://github.com/sgl-project/sglang/pull/28454) | merged | docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort | `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` |
| 2026-06-17 | [#28433](https://github.com/sgl-project/sglang/pull/28433) | merged | [Ascend]GLM 5.2 deployment | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` |
| 2026-06-17 | [#28460](https://github.com/sgl-project/sglang/pull/28460) | merged | docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16) | `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` |
| 2026-06-18 | [#28607](https://github.com/sgl-project/sglang/pull/28607) | merged | [misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e | `test/registered/models_e2e/test_dsa_glm5_hisparse.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-19 | [#28536](https://github.com/sgl-project/sglang/pull/28536) | merged | ci: run GB300 nightly suite in the standard Nvidia nightly workflow | `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py` |
| 2026-06-22 | [#27893](https://github.com/sgl-project/sglang/pull/27893) | merged | [NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` |
| 2026-06-24 | [#27053](https://github.com/sgl-project/sglang/pull/27053) | merged | [BCG][GLM5] perf: BCG support and prefill enhancements | `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` |
| 2026-06-25 | [#29194](https://github.com/sgl-project/sglang/pull/29194) | merged | [AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook | `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` |
| 2026-06-25 | [#28103](https://github.com/sgl-project/sglang/pull/28103) | merged | Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test | `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml` |
| 2026-06-26 | [#29313](https://github.com/sgl-project/sglang/pull/29313) | merged | [AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook | `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` |
| 2026-06-26 | [#29380](https://github.com/sgl-project/sglang/pull/29380) | merged | [Docs] Add NVFP4 quantization to GLM-5.2 cookbook | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` |
| 2026-06-26 | [#29466](https://github.com/sgl-project/sglang/pull/29466) | merged | Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` |

## Per-PR Diff Audit Cards

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- Status/date: merged / 2026-02-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +22/-7, 98 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support GlmMoeDsaForCausalLM"; model line: GLM-5/5.1; category: model support/runtime entry; main diff: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "Support GlmMoeDsaForCausalLM"; the main implementation surface is `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes, touching `is_deepseek_nsa, from_server_args, _config_draft_model`; `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, touching `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM`; `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params, touching `_handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference`.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:
+            "GlmMoeDsaForCausalLM",
@@ -271,10 +272,10 @@ def from_server_args(
-        if (
-            is_draft_model
-            and self.hf_config.architectures[0] == "DeepseekV3ForCausalLM"
-        ):
diff -- python/sglang/srt/models/glm4_moe.py
@@ -79,6 +79,7 @@
+from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
@@ -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-EntryClass = [Glm4MoeForCausalLM]
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
+EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
diff -- python/sglang/srt/server_args.py
@@ -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +6/-5; `python/sglang/srt/models/glm4_moe.py` modified +6/-1; `python/sglang/srt/server_args.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GLM-5 fused shared expert"; model line: GLM-5/5.1; category: bug fix; main diff: `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "Fix GLM-5 fused shared expert"; the main implementation surface is `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts, touching `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-    pass
+    def determine_num_fused_shared_experts(self):
+        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`; associated commits `23adb50751d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +635/-1, 725 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`; technical summary: Covers "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm5 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py
@@ -0,0 +1,244 @@
+"""AMD GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-glm5 suite
+"""
+import ast
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- Link: https://github.com/sgl-project/sglang/pull/20062
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +32/-59, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V32/GLM5] Control the threshold of applying dense attention with an environ"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`; technical summary: Covers "[V32/GLM5] Control the threshold of applying dense attention with an environ"; the main implementation surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl, touching `NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed`; `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: -377,8 +377,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -16,10 +16,6 @@
-from sglang.srt.layers.attention.nsa.nsa_mtp_verification import (
-    verify_multi_backend_fused_metadata_copy,
-    verify_single_backend_fused_metadata_copy,
-)
@@ -71,15 +67,10 @@
-# Control whether to use fused metadata copy kernel (default: enabled)
diff -- python/sglang/srt/server_args.py
@@ -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):
-            if is_deepseek_nsa(hf_config):  # DeepSeek 3.2, GlmMoeDsaForCausalLM
+            if is_deepseek_nsa(hf_config):  # DeepSeek 3.2/GLM 5
-                    envs.SGLANG_NSA_FORCE_MLA.set(True)
+                    envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
-                        "Force NSA prefill to use MLA (i.e. disable MHA_ONE_SHOT) for GlmMoeDsaForCausalLM on Blackwell."
+                        "Force NSA prefill to use sparse MLA (i.e. disable MHA_ONE_SHOT) for GlmMoeDsaForCausalLM on Blackwell."
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -34,8 +34,6 @@ def setUpClass(cls):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46; `python/sglang/srt/server_args.py` modified +26/-3; `python/sglang/srt/environ.py` modified +1/-2
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4
  - docs: `docs/references/environment_variables.md` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: merged / 2026-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-12, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation"; model line: GLM-5/5.1; category: bug fix; main diff: `docs/basic_usage/deepseek_v32.md`; technical summary: Covers "[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation"; the main implementation surface is `docs/basic_usage/deepseek_v32.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -3,7 +3,7 @@
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
+Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
@@ -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
-- **MHA prefill threshold relax** To apply MHA attention to requests longer than 2048 tokens, please set flag `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` to a value larger th
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +11/-12
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- Link: https://github.com/sgl-project/sglang/pull/22314
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +27/-31, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300"; model line: GLM-5/5.1; category: bug fix; main diff: `python/sglang/srt/mem_cache/memory_pool.py`; technical summary: Covers "[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300"; the main implementation surface is `python/sglang/srt/mem_cache/memory_pool.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer, touching `set_mla_kv_buffer`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -45,7 +45,7 @@
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
+from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
@@ -1575,37 +1575,33 @@ def set_mla_kv_buffer(
-        if self.nsa_kv_cache_store_fp8:
-            if _is_hip:
-                # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21710
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`; associated commits `db60a620dbf1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +345/-5, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`; technical summary: Covers "[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x"; the main implementation surface is `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf, touching `generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5, touching `generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass`; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name, touching `get_display_name`; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name, touching `get_display_name`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py
@@ -0,0 +1,143 @@
+"""MI35x Nightly performance benchmark for GLM-5.
+Tests GLM-5 with NSA attention backend using bench_one_batch on 8 GPUs.
+Registry: nightly-perf-8-gpu-mi35x-glm5 suite
+"""
+import os
+os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
diff -- test/registered/amd/perf/mi30x/test_glm5_perf_amd.py
@@ -0,0 +1,140 @@
+"""Nightly performance benchmark for GLM-5 on MI30x.
+Tests GLM-5 with NSA attention backend using bench_one_batch on 8 GPUs.
+Model paths can be configured via environment variables:
+- GLM5_MODEL_PATH: Path to GLM-5 model (default: zai-org/GLM-5-FP8)
+Example usage:
+    python -m pytest test_glm5_perf_amd.py -v
diff -- test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py
@@ -59,13 +59,17 @@ def get_display_name(self) -> str:
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22285 - Add CI tests for GLM-5

- Link: https://github.com/sgl-project/sglang/pull/22285
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +153/-30, 301 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add CI tests for GLM-5"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; technical summary: Covers "Add CI tests for GLM-5"; the main implementation surface is `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass, touching `TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP`; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed, touching `TestDeepseekV32DPMTP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_dsa_models_basic.py
@@ -14,9 +14,10 @@
-register_cuda_ci(est_time=360, suite="stage-c-test-8-gpu-h200")
+register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200")
+GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
@@ -138,5 +139,124 @@ def test_bs_1_speed(self):
+class TestGLM5DP(CustomTestCase):
+    @classmethod
diff -- test/registered/8-gpu-models/test_dsa_models_mtp.py
@@ -20,6 +20,7 @@
+GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
@@ -47,12 +48,13 @@ def setUpClass(cls):
-        cls.process = popen_launch_server(
-            cls.model,
-            cls.base_url,
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- Link: https://github.com/sgl-project/sglang/pull/22399
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/gb300/test_glm5_fp8.py`; associated commits `46c2b7762765`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +82/-6, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/gb300/test_glm5_fp8.py`; technical summary: Covers "[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model"; the main implementation surface is `test/registered/gb300/test_glm5_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8, touching `TestGlm5Fp8, test_glm5_fp8`.
- Code diff details:
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_glm5_fp8.py
@@ -8,7 +8,7 @@
-MODEL_PATH = "zai-org/GLM-5-FP8"
+MODEL_PATH = "zai-org/GLM-5.1-FP8"
@@ -27,7 +27,7 @@
-    """GLM-5 FP8 on GB300 (4x B200 NVL4, tp=4)."""
+    """GLM-5.1 FP8 on GB300 (4x B200 NVL4, tp=4)."""
@@ -56,7 +56,7 @@ def test_glm5_fp8(self):
```

- Reviewed files:
  - tests: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/22336
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; associated commits `ef6bfc1197ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +918/-25, 1064 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; technical summary: Covers "[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf, touching `generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51, touching `generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py
@@ -0,0 +1,242 @@
+"""MI35x GLM-5.1 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5.1-FP8 with NSA attention backend using few-shot
+completion benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm51 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py
@@ -0,0 +1,238 @@
+"""AMD GLM-5.1 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5.1-FP8 with NSA attention backend using few-shot
+completion benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-glm51 suite
+"""
+import ast
diff -- test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22712 - [NPU] update GLM5 running guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- Status/date: merged / 2026-04-13
- Trace source: `git log --name-only -- <model-files>` found it through `docs/platforms/ascend/ascend_npu_glm5_examples.md`; associated commits `13a4aafdbe69`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] update GLM5 running guide"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs/platforms/ascend/ascend_npu_glm5_examples.md`; technical summary: Covers "[NPU] update GLM5 running guide"; the main implementation surface is `docs/platforms/ascend/ascend_npu_glm5_examples.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA....
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...
- Key code excerpts:

```diff
diff -- docs/platforms/ascend/ascend_npu_glm5_examples.md
@@ -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
-Note: Using this image, you need to update transformers to main branch
+### Best Practices
+Note: Using this image for **best practices**, you need to update transformers to version 5.3.0
-pip install git+https://github.com/huggingface/transformers.git
+# Install transformers version 5.3.0 from PyPI
+pip install transformers==5.3.0
```

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_glm5_examples.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- Link: https://github.com/sgl-project/sglang/pull/22543
- Status/date: merged / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix"; model line: GLM-5/5.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix"; the main implementation surface is `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config, touching `_get_quantization_config`; `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, touching `_handle_missing_default_values`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -560,6 +560,9 @@ def post_load_weights(
+                and self.config.architectures
+                and self.config.architectures[0]
+                == "DeepseekV3ForCausalLM"  # Avoid processing other models like GlmMoeDsaForCausalLM
diff -- python/sglang/srt/model_loader/loader.py
@@ -198,6 +198,9 @@ def _get_quantization_config(
+    if model_config.quantization == "quark":
+        packed_modules_mapping.update({"gate_up_proj": ["gate_proj", "up_proj"]})
diff -- python/sglang/srt/server_args.py
@@ -1016,6 +1016,8 @@ def _handle_missing_default_values(self):
+        # strip device index from user if any (e.g. "cuda:0" -> "cuda")
+        self.device = self.device.split(":")[0]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0; `python/sglang/srt/model_loader/loader.py` modified +3/-0; `python/sglang/srt/server_args.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- Link: https://github.com/sgl-project/sglang/pull/21773
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`; associated commits `39c6bf730c41`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +528/-130, 821 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`; technical summary: Covers "[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name, touching `get_model_path, ModelConfig, __post_init__`; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py
@@ -0,0 +1,281 @@
+"""MI35x GLM-5-MXFP4 GSM8K Completion Evaluation Test (8-GPU)
+Tests the AMD Quark MXFP4-quantized GLM-5 model using few-shot
+completion benchmark on MI35x.
+Model: amd/GLM-5-MXFP4 (MOE-only MXFP4 quantization of zai-org/GLM-5)
+Reference: https://huggingface.co/amd/GLM-5-MXFP4
+Registry: nightly-amd-8-gpu-mi35x-glm5-mxfp4 suite
diff -- test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py
@@ -0,0 +1,187 @@
+"""MI35x Nightly performance benchmark for GLM-5-MXFP4 model.
+Benchmarks the AMD Quark MXFP4-quantized GLM-5 model on MI35x with 8 GPUs.
+Model: amd/GLM-5-MXFP4 (MOE-only MXFP4 quantization of zai-org/GLM-5)
+Reference: https://huggingface.co/amd/GLM-5-MXFP4
+Registry: nightly-perf-8-gpu-mi35x-glm5-mxfp4 suite
+"""
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/22595
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +67/-1, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: normalize tool message content for GLM5.1 chat template"; model line: GLM-5/5.1; category: bug fix; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`; technical summary: Covers "fix: normalize tool message content for GLM5.1 chat template"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template, touching `normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template`; `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined, touching `test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -60,6 +60,28 @@
+def normalize_tool_content(role: str, content):
+    """Normalize tool message content from OpenAI array format to plain string.
+    OpenAI clients may send tool content as a list of content parts
+    (e.g. [{"type":"text","text":"..."}]) but most chat templates expect
+    a plain string for tool messages. Only flatten when ALL items are
+    pure OpenAI text parts; preserve lists containing non-text-type items
diff -- test/registered/openai_server/basic/test_serving_chat.py
@@ -19,7 +19,10 @@
-from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
+from sglang.srt.entrypoints.openai.serving_chat import (
+    OpenAIServingChat,
+    normalize_tool_content,
+)
@@ -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0
  - tests: `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1
- Risk and verification: The diff ships test coverage in `test/registered/openai_server/basic/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: merged / 2026-04-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-5, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)"; model line: GLM-5/5.1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; technical summary: Covers "[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache, touching `__init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -14,7 +14,7 @@
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
+from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
@@ -32,14 +32,16 @@
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+if _use_aiter:
+    from aiter.ops.cache import indexer_k_quant_and_cache
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-15, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable MTP for GLM-5-mxfp4 model"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_nextn.py`; technical summary: Covers "[AMD] Enable MTP for GLM-5-mxfp4 model"; the main implementation surface is `python/sglang/srt/models/deepseek_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -42,6 +42,7 @@
+from sglang.srt.layers.linear import ReplicatedLinear
@@ -99,7 +100,18 @@ def __init__(
-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        if quant_config is not None and quant_config.get_name() == "quark":
+            self.eh_proj = ReplicatedLinear(
+                2 * config.hidden_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- Link: https://github.com/sgl-project/sglang/pull/23060
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] Fix dynamic chunking profiling crash on GLM-5 models"; model line: GLM-5/5.1; category: bug fix; main diff: `python/sglang/srt/managers/scheduler_pp_mixin.py`; technical summary: Covers "[fix] Fix dynamic chunking profiling crash on GLM-5 models"; the main implementation surface is `python/sglang/srt/managers/scheduler_pp_mixin.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor, touching `profile_and_init_predictor`.
- Code diff details:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- Key code excerpts:

```diff
diff -- python/sglang/srt/managers/scheduler_pp_mixin.py
@@ -20,6 +20,7 @@
+    set_is_extend_in_batch,
@@ -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):
+                set_is_extend_in_batch(batch.forward_mode.is_extend())
```

- Reviewed files:
  - runtime: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/scheduler_pp_mixin.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- Link: https://github.com/sgl-project/sglang/pull/23540
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; associated commits `9b2f7f8a91d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +15/-13, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs: split MI300X and MI325X options in GLM-5.1 generator"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; technical summary: Covers "docs: split MI300X and MI325X options in GLM-5.1 generator"; the main implementation surface is `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/glm-51-deployment.jsx
@@ -14,7 +14,8 @@ export const GLM51Deployment = () => {
-        { id: 'mi300x', label: 'MI300X/MI325X', default: false },
+        { id: 'mi300x', label: 'MI300X',        default: false },
+        { id: 'mi325x', label: 'MI325X',        default: false },
@@ -23,7 +24,7 @@ export const GLM51Deployment = () => {
-        const isAMD = hw === 'mi300x' || hw === 'mi355x';
+        const isAMD = ['mi300x', 'mi325x', 'mi355x'].includes(hw);
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23708 - [NPU][Doc] Update GLM-5 docs, enabling deepep by default

- Link: https://github.com/sgl-project/sglang/pull/23708
- Status/date: merged / 2026-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/platforms/ascend/ascend_npu_glm5_examples.md`; associated commits `461bc8af494c`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU][Doc] Update GLM-5 docs, enabling deepep by default"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs/platforms/ascend/ascend_npu_glm5_examples.md`; technical summary: Covers "[NPU][Doc] Update GLM-5 docs, enabling deepep by default"; the main implementation surface is `docs/platforms/ascend/ascend_npu_glm5_examples.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3 (6 lines); hunks: -161,8 +161,6 @@ P_IP=('your ip1' 'your ip2'); -182,7 +180,9 @@ do.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3 (6 lines); hunks: -161,8 +161,6 @@ P_IP=('your ip1' 'your ip2'); -182,7 +180,9 @@ do
- Key code excerpts:

```diff
diff -- docs/platforms/ascend/ascend_npu_glm5_examples.md
@@ -161,8 +161,6 @@ P_IP=('your ip1' 'your ip2')
-export SGLANG_ENABLE_SPEC_V2=1
-export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
@@ -182,7 +180,9 @@ do
-        --cuda-graph-max-bs 16 \
+        --cuda-graph-max-bs 32 \
+        --moe-a2a-backend deepep \
```

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_glm5_examples.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25453 - [CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85

- Link: https://github.com/sgl-project/sglang/pull/25453
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/8-gpu-models/test_glm_51_fp8.py`; technical summary: Covers "[CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85"; the main implementation surface is `test/registered/8-gpu-models/test_glm_51_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7.
- Code diff details:
  - `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_glm_51_fp8.py
@@ -15,7 +15,7 @@
-    "--mem-fraction-static=0.9",
+    "--mem-fraction-static=0.85",
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_glm_51_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25735 - [NPU] [DOCS] Improved the usability of Ascend NPU documents

- Link: https://github.com/sgl-project/sglang/pull/25735
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +468/-49, 743 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] [DOCS] Improved the usability of Ascend NPU documents"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`; technical summary: Covers "[NPU] [DOCS] Improved the usability of Ascend NPU documents"; the main implementation surface is `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22 (244 lines); hunks: -53,19 +53,29 @@ You can install SGLang using any of the methods below. Pleas...; -142,14 +152,42 @@ pip install -e python[all_npu]; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0 (60 lines); hunks: -221,3 +221,63 @@ python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__pa...; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11 (59 lines); hunks: -7,15 +7,18 @@ metatags:; -42,9 +45,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7 (55 lines); hunks: -11,22 +11,29 @@ The GLM (General Language Model) series is an open-source bi...; -53,9 +60,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM....
- Code diff details:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22 (244 lines); hunks: -53,19 +53,29 @@ You can install SGLang using any of the methods below. Pleas...; -142,14 +152,42 @@ pip install -e python[all_npu]
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0 (60 lines); hunks: -221,3 +221,63 @@ python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__pa...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11 (59 lines); hunks: -7,15 +7,18 @@ metatags:; -42,9 +45,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7 (55 lines); hunks: -11,22 +11,29 @@ The GLM (General Language Model) series is an open-source bi...; -53,9 +60,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quick_start.mdx` modified +42/-5 (47 lines); hunks: -13,14 +13,23 @@ metatags:; -39,6 +48,34 @@ docker run -it --rm --privileged --network=host --ipc=host --...
- Key code excerpts:

```diff
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx
@@ -53,19 +53,29 @@ You can install SGLang using any of the methods below. Please go through `System
+<Warning>
+Ensure sufficient disk space before pulling images. Each Docker image requires at least **30 GB** of free space.
+</Warning>
-<Note>
-CANN images and SGLang images are hosted at different registry addresses. Make sure to pull them from the correct location.
-</Note>
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx
@@ -221,3 +221,63 @@ python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
+## 6. `[Errno 101] Network is unreachable` when downloading HuggingFace datasets
+### Error message
+'''text highlight=1-2
+'[Errno 101] Network is unreachable' thrown while requesting HEAD https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cle
+Retrying in 1s [Retry 1/5].
+Traceback (most recent call last):
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx
@@ -7,15 +7,18 @@ metatags:
```

- Reviewed files:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quick_start.mdx` modified +42/-5; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_accuracy_evaluation.mdx` modified +31/-3
- Risk and verification: This is mostly docs/examples in `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_accuracy_evaluation.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25814 - Update GLM-5 H200 FP8

- Link: https://github.com/sgl-project/sglang/pull/25814
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; associated commits `da6d549ab28a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-0, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update GLM-5 H200 FP8"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`; technical summary: Covers "Update GLM-5 H200 FP8"; the main implementation surface is `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0 (5 lines); hunks: -206,6 +206,11 @@ export const GLM5Deployment = () => {; `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0 (1 lines); hunks: -109,6 +109,7 @@ sglang serve \.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0 (5 lines); hunks: -206,6 +206,11 @@ export const GLM5Deployment = () => {
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0 (1 lines); hunks: -109,6 +109,7 @@ sglang serve \
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -206,6 +206,11 @@ export const GLM5Deployment = () => {
+    // H200 FP8: flashinfer allreduce fusion.
+    if (hardware === 'h200' && effectiveQuant === 'fp8') {
+      cmd += ' \\\n  --enable-flashinfer-allreduce-fusion';
+    }
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.mdx
@@ -109,6 +109,7 @@ sglang serve \
+  --enable-flashinfer-allreduce-fusion \
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- Link: https://github.com/sgl-project/sglang/pull/25821
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 162 files, +11303/-10745, 15980 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; technical summary: Covers "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines); `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines); `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines); `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines).
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,1746 +1,10 @@
-from __future__ import annotations
+# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
+import warnings
-import contextlib
-import logging
-from abc import ABC, abstractmethod
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -0,0 +1,1746 @@
+from __future__ import annotations
+import contextlib
+import logging
+from abc import ABC, abstractmethod
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -1,814 +1,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25266 - [AMD][CI] Clean up AMD nightly + pr-test workflows

- Link: https://github.com/sgl-project/sglang/pull/25266
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1034/-927, 3120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][CI] Clean up AMD nightly + pr-test workflows"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `.github/workflows/pr-test-amd-rocm720.yml`; technical summary: Covers "[AMD][CI] Clean up AMD nightly + pr-test workflows"; the main implementation surface is `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `.github/workflows/pr-test-amd-rocm720.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490 (1014 lines); hunks: -27,43 +27,59 @@ on:; -108,6 +124,11 @@ concurrency:; `.github/workflows/nightly-test-amd.yml` modified +454/-409 (863 lines); hunks: -27,41 +27,56 @@ on:; -106,6 +121,11 @@ concurrency:; `.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15 (53 lines); hunks: -46,7 +46,7 @@ on:; -68,6 +68,11 @@ on:; `.github/workflows/pr-test-amd.yml` modified +10/-5 (15 lines); hunks: -38,7 +38,7 @@ on:; -67,6 +67,11 @@ on:.
- Code diff details:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490 (1014 lines); hunks: -27,43 +27,59 @@ on:; -108,6 +124,11 @@ concurrency:
  - `.github/workflows/nightly-test-amd.yml` modified +454/-409 (863 lines); hunks: -27,41 +27,56 @@ on:; -106,6 +121,11 @@ concurrency:
  - `.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15 (53 lines); hunks: -46,7 +46,7 @@ on:; -68,6 +68,11 @@ on:
  - `.github/workflows/pr-test-amd.yml` modified +10/-5 (15 lines); hunks: -38,7 +38,7 @@ on:; -67,6 +67,11 @@ on:
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ class TestGLM51EvalAMD(unittest.TestCase):; symbols: TestGLM51EvalAMD, setUpClass, test_glm51_accuracy
- Key code excerpts:

```diff
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -27,43 +27,59 @@ on:
+          # 1-GPU Unit Tests (MI30x + MI35x)
+          - nightly-test-1-gpu-mi35x-rocm720
+          # 2-GPU and 4-GPU Tests (MI30x)
+          # 8-GPU GPT-OSS (MI30x mixes Grok1-FP8; MI35x mixes Qwen3-Coder-Next)
+          - nightly-accuracy-8-gpu-mi35x-rocm720
+          # 8-GPU Grok1-INT4 (MI30x + MI35x)
diff -- .github/workflows/nightly-test-amd.yml
@@ -27,41 +27,56 @@ on:
+          # 1-GPU Unit Tests (MI30x + MI35x)
+          - nightly-test-1-gpu-mi35x
+          # 2-GPU and 4-GPU Tests (MI30x)
+          # 8-GPU GPT-OSS (MI30x mixes Grok1-FP8; MI35x mixes Qwen3-Coder-Next)
+          - nightly-accuracy-8-gpu-mi35x
+          # 8-GPU Grok1-INT4 (MI30x + MI35x)
diff -- .github/workflows/pr-test-amd-rocm720.yml
@@ -46,7 +46,7 @@ on:
```

- Reviewed files:
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490; `.github/workflows/nightly-test-amd.yml` modified +454/-409; `.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15; `.github/workflows/pr-test-amd.yml` modified +10/-5
  - tests: `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` modified +1/-1; `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` modified +1/-1; `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` modified +1/-1; `test/registered/amd/disaggregation/test_disaggregation_basic.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/disaggregation/test_disaggregation_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26396 - [AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate

- Link: https://github.com/sgl-project/sglang/pull/26396
- Status/date: merged / 2026-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`; associated commits `d44584e8d8b2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +144/-0, 145 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`; technical summary: Covers "[AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: _raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x, setUpClass, touching `_raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: _raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py
@@ -0,0 +1,144 @@
+"""MI355X GLM-5.1-MXFP4 TP=2 GSM8K accuracy gate.
+This is a PR Test (AMD) regression test for the GLM-5.1-MXFP4 TP=2
+accuracy drop seen on MI355X/gfx950 when aiter selected a bad BF16 GEMM path.
+Registry: stage-c-test-large-8-gpu-amd-mi35x suite
+"""
+import os
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- Link: https://github.com/sgl-project/sglang/pull/25813
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 47 files, +1262/-2154, 4187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): port popular model usage guides into cookbook pages"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`; technical summary: Covers "docs(cookbook): port popular model usage guides into cookbook pages"; the main implementation surface is `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64, touching `image_to_base64`.
- Code diff details:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- Key code excerpts:

```diff
diff -- docs_new/docs/basic_usage/deepseek_v32.mdx
@@ -1,601 +0,0 @@
-title: "DeepSeek V3.2/GLM-5 Usage"
-metatags:
-    description: "Deploy DeepSeek V3.2/GLM-5 with SGLang: DeepSeek Sparse Attention (DSA), long-context optimization, MTP speculative decoding, function calling. Supports H200, B2
-DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism power
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
-## Installation
diff -- docs_new/docs/basic_usage/deepseek_v3.mdx
@@ -1,375 +0,0 @@
-title: "DeepSeek V3/V3.1/R1 Usage"
-metatags:
-    description: "Deploy DeepSeek V3/R1 with SGLang: MLA optimization, FP8 quantization, multi-node TP, DP attention, MTP speculative decoding. Supports H200, B200, MI300X, A100."
-SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/dee
-This document outlines current optimizations for DeepSeek.
-For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx
@@ -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose the most suitable in
```

- Reviewed files:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- Link: https://github.com/sgl-project/sglang/pull/27001
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +11/-471, 936 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`; technical summary: Covers "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; the main implementation surface is `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models, touching `get_model_path, ModelConfig, get_display_name`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py
@@ -2,19 +2,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py
@@ -3,19 +3,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py
@@ -3,19 +3,10 @@
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27708 - [Docs] Add GLM-5.1 NVFP4 to cookbook

- Link: https://github.com/sgl-project/sglang/pull/27708
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; associated commits `91ff7baa2860`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +92/-38, 275 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add GLM-5.1 NVFP4 to cookbook"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; technical summary: Covers "[Docs] Add GLM-5.1 NVFP4 to cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21 (90 lines); hunks: -1,7 +1,7; -11,6 +11,7 @@ tag: NEW; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17 (40 lines); hunks: -1,19 +1,19; -25,10 +25,13 @@ export const GLM51Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21 (90 lines); hunks: -1,7 +1,7; -11,6 +11,7 @@ tag: NEW
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17 (40 lines); hunks: -1,19 +1,19; -25,10 +25,13 @@ export const GLM51Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx
@@ -1,7 +1,7 @@
-    description: "Deploy GLM-5.1 with SGLang on NVIDIA H100/H200/B200/GB300 and AMD MI300X/MI325X/MI355X."
+    description: "Deploy GLM-5.1 with SGLang on NVIDIA H100/H200/B300/GB300 and AMD MI300X/MI325X/MI355X."
@@ -11,6 +11,7 @@ tag: NEW
+- **NVFP4 (4-bit quantized)**: [nvidia/GLM-5.1-NVFP4](https://huggingface.co/nvidia/GLM-5.1-NVFP4)
@@ -24,7 +25,7 @@ This section provides deployment configurations optimized for different hardware
-**Interactive Command Generator**: Use the configuration selector below to automatically generate the appropriate deployment command for your hardware platform, quantization metho
diff -- docs_new/src/snippets/autoregressive/glm-51-deployment.jsx
@@ -1,19 +1,19 @@
-  // Supported quantization per hardware:
-  //   H100 / H200 / B200 → BF16 + FP8
-  //   GB300 → FP8 only
+  // Recommended quantization per hardware:
+  //   H100 / H200 → FP8
+  //   B300 / GB300 → NVFP4
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27964 - [Spec] Retire Spec V1

- Link: https://github.com/sgl-project/sglang/pull/27964
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 46 files, +111/-252, 1422 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Retire Spec V1"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`; technical summary: Covers "[Spec] Retire Spec V1"; the main implementation surface is `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass, touching `TestDeepseekMTP, setUpClass, tearDownClass`; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do; `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family, touching `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu....
- Code diff details:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- Key code excerpts:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- Reviewed files:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28437 - docs(cookbook): add GLM-5.2 deployment cookbook

- Link: https://github.com/sgl-project/sglang/pull/28437
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; associated commits `0cb6183432dc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +970/-2, 996 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): add GLM-5.2 deployment cookbook"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; technical summary: Covers "docs(cookbook): add GLM-5.2 deployment cookbook"; the main implementation surface is `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0 (657 lines); hunks: -0,0 +1,657; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0 (116 lines); hunks: -0,0 +1,116; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0 (195 lines); hunks: -0,0 +1,195; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6.
- Code diff details:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0 (657 lines); hunks: -0,0 +1,657
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0 (116 lines); hunks: -0,0 +1,116
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0 (195 lines); hunks: -0,0 +1,195
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2.jsx
@@ -0,0 +1,657 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+export const config = {
+  modelName: "GLM-5.2",
+  supportedHardware: [
+    "h200", "b200", "gb300", "b300",
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx
@@ -0,0 +1,116 @@
+// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
+// See _deployment.jsx for the speed/accuracy schema.
+// Numbers pending: each entry is a bare `match` stub (renders "pending") until measured
+// end-to-end on the corresponding hardware, then filled with sglang_version + speed/accuracy.
+export const benchmarks = [
+  // ---- H200 + FP8 ----  (measured on the v0.5.13.post1 release image, flush-cache on every run)
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
@@ -0,0 +1,195 @@
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28448 - docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags

- Link: https://github.com/sgl-project/sglang/pull/28448
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; associated commits `00081a00d5b3`
- Diff scope read: GitHub Pull Request files API returned 4 files, +36/-54, 354 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; technical summary: Covers "docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags"; the main implementation surface is `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45 (65 lines); hunks: -36,7 +36,6 @@ export const config = {; -141,10 +140,10 @@ sgl-eval run aime25 \\; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6 (17 lines); hunks: -5,13 +5,16; -36,13 +39,14 @@ export const benchmarks = [; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2 (4 lines); hunks: -92,8 +92,8 @@ import { Playground } from "/src/snippets/_playground.jsx";.
- Code diff details:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45 (65 lines); hunks: -36,7 +36,6 @@ export const config = {; -141,10 +140,10 @@ sgl-eval run aime25 \\
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6 (17 lines); hunks: -5,13 +5,16; -36,13 +39,14 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2 (4 lines); hunks: -92,8 +92,8 @@ import { Playground } from "/src/snippets/_playground.jsx";
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2.jsx
@@ -36,7 +36,6 @@ export const config = {
-    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
@@ -141,10 +140,10 @@ sgl-eval run aime25 \\
-        { id: "mtp-314", label: "EAGLE / MTP 3-1-4",
-          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
-                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
-        { id: "mtp-112", label: "EAGLE / MTP 1-1-2",
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx
@@ -5,13 +5,16 @@
+    // EAGLE MTP 5-1-6 (was 3-1-4): accept ~5.96/6 → +31%/+15% throughput, -25%/-11% TPOT vs 3-1-4.
+    // KV stays bf16 (Hopper auto-default). fp8 KV measured worse on H200 (slower flashmla_kv prefill
+    // + lower decode throughput): conc=1 31 gpu / TTFT 838, conc=16 96 gpu / TTFT 6650.
-        ttft_ms: 740, tpot_ms: 4.06, tokens_per_sec_per_gpu: 26 },
+        ttft_ms: 662, tpot_ms: 3.03, tokens_per_sec_per_gpu: 34 },
-        ttft_ms: 5980, tpot_ms: 13.97, tokens_per_sec_per_gpu: 98 },
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
@@ -92,8 +92,8 @@ import { Playground } from "/src/snippets/_playground.jsx";
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28454 - docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort

- Link: https://github.com/sgl-project/sglang/pull/28454
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; associated commits `33f205d8c5b2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-2, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort"; model line: GLM-5/5.1; category: bug fix; main diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; technical summary: Covers "docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort"; the main implementation surface is `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2 (12 lines); hunks: -104,7 +104,15 @@ import { Playground } from "/src/snippets/_playground.jsx";; -115,7 +123,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke....
- Code diff details:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2 (12 lines); hunks: -104,7 +104,15 @@ import { Playground } from "/src/snippets/_playground.jsx";; -115,7 +123,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke...
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
@@ -104,7 +104,15 @@ import { Playground } from "/src/snippets/_playground.jsx";
-GLM-5.2 is a hybrid-reasoning model. Enable the `glm45` reasoning parser (toggle **Reasoning Parser** in the **Parsers** card of the [Playground above](#playground)) to separate t
+GLM-5.2 is a hybrid-reasoning model. Enable the `glm45` reasoning parser (toggle **Reasoning Parser** in the **Parsers** card of the [Playground above](#playground)) to separate t
+**Reasoning effort.** Pass `chat_template_kwargs: {"reasoning_effort": ...}` to inject a `Reasoning Effort: <level>` system line (only while thinking is on). **The template wires
+| `reasoning_effort` | Injected system line | Effect |
+|---|---|---|
+| *(not passed / unset)* | `Reasoning Effort: Max` | **default — highest reasoning** |
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28433 - [Ascend]GLM 5.2 deployment

- Link: https://github.com/sgl-project/sglang/pull/28433
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`; associated commits `71b090a8e785`
- Diff scope read: GitHub Pull Request files API returned 1 files, +376/-0, 377 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Ascend]GLM 5.2 deployment"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`; technical summary: Covers "[Ascend]GLM 5.2 deployment"; the main implementation surface is `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0 (376 lines); hunks: -0,0 +1,376.
- Code diff details:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0 (376 lines); hunks: -0,0 +1,376
- Key code excerpts:

```diff
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx
@@ -0,0 +1,376 @@
+# GLM-5.2 examples
+## Introduction
+The GLM (General Language Model) series is an open-source bilingual large language model family jointly developed by the KEG Laboratory of Tsinghua University and Zhipu AI. This s
+## Environment Preparation
+### Model Weight
+- `GLM-5.2`(BF16 version): [Download model weight](https://huggingface.co/collections/zai-org/glm-52).
```

- Reviewed files:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0
- Risk and verification: This is mostly docs/examples in `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28460 - docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)

- Link: https://github.com/sgl-project/sglang/pull/28460
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; associated commits `72ccfec5949d`
- Diff scope read: GitHub Pull Request files API returned 3 files, +76/-23, 170 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; technical summary: Covers "docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)"; the main implementation surface is `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8 (69 lines); hunks: -100,14 +100,67 @@ export const benchmarks = [; symbols: kernels, touching `kernels`; `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12 (24 lines); hunks: -354,13 +354,13 @@ sgl-eval run aime25 \\; -377,7 +377,7 @@ sgl-eval run aime25 \\; symbols: kernels, touching `kernels`; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3 (6 lines); hunks: -1,6 +1,6; -94,10 +94,10 @@ import { Playground } from "/src/snippets/_playground.jsx";.
- Code diff details:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8 (69 lines); hunks: -100,14 +100,67 @@ export const benchmarks = [; symbols: kernels
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12 (24 lines); hunks: -354,13 +354,13 @@ sgl-eval run aime25 \\; -377,7 +377,7 @@ sgl-eval run aime25 \\; symbols: kernels
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3 (6 lines); hunks: -1,6 +1,6; -94,10 +94,10 @@ import { Playground } from "/src/snippets/_playground.jsx";
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx
@@ -100,14 +100,67 @@ export const benchmarks = [
-  // ---- B300 + FP8 ----  (inferred from B200; benchmarks pending → render "pending")
-  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
-  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
-  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
-  // ---- B300 + BF16 ----  (unquantized zai-org/GLM-5.2; benchmarks pending → render "pending")
-  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2.jsx
@@ -354,13 +354,13 @@ sgl-eval run aime25 \\
-    // B300 + FP8 (Blackwell Ultra, 8-GPU single node) — TP8.
-    // Inferred from the verified B200 (sm100) FP8 recipe; B300 is the same Blackwell
-    // family (sm103). Benchmarks pending → verified:false.
+    // B300 + FP8 (Blackwell Ultra, 8-GPU single node) — TP8. Verified on 8xB300 (v0.5.13.post1).
+    // Recipe mirrors the verified B200 (sm100) FP8 path. B300 (sm103) currently trails B200 per-GPU
+    // because deep_gemm/DSA are tuned for sm100; expected to improve as sm103 gets first-class kernels.
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
@@ -1,6 +1,6 @@
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8; `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28607 - [misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e

- Link: https://github.com/sgl-project/sglang/pull/28607
- Status/date: merged / 2026-06-18
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/models_e2e/test_dsa_glm5_hisparse.py`; associated commits `8f6d9ef9a55e`
- Diff scope read: GitHub Pull Request files API returned 4 files, +124/-203, 368 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `test/registered/models_e2e/test_dsa_glm5_hisparse.py`; technical summary: Covers "[misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e"; the main implementation surface is `test/registered/models_e2e/test_dsa_glm5_hisparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM5HiSparse, touching `TestGLM5HiSparse`.
- Code diff details:
  - `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM5HiSparse
- Key code excerpts:

```diff
diff -- test/registered/models_e2e/test_dsa_glm5_hisparse.py
@@ -0,0 +1,54 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
+from sglang.test.server_fixtures.default_fixture import DefaultServerBase
+register_cuda_ci(est_time=720, stage="extra-b", runner_config="8-gpu-h200")
+GLM5_FP8_MODEL_PATH = "zai-org/GLM-5-FP8"
```

- Reviewed files:
  - tests: `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_dsa_models_hisparse.py`, `test/registered/models_e2e/test_dsa_glm5_hisparse.py`, `test/registered/unit/managers/test_schedule_batch_req_pool_indices.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28536 - ci: run GB300 nightly suite in the standard Nvidia nightly workflow

- Link: https://github.com/sgl-project/sglang/pull/28536
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +72/-197, 438 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: run GB300 nightly suite in the standard Nvidia nightly workflow"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`; technical summary: Covers "ci: run GB300 nightly suite in the standard Nvidia nightly workflow"; the main implementation surface is `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81 (81 lines); hunks: -1,81 +0,0; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4, touching `TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4`; `test/registered/gb300/test_deepseek_v32.py` removed +0/-78 (78 lines); hunks: -1,78 +0,0; symbols: TestDeepseekV32, test_deepseek_v32, touching `TestDeepseekV32, test_deepseek_v32`; `test/registered/gb300/test_qwen35_fp8.py` modified +14/-14 (28 lines); hunks: -17,43 +17,43; -62,7 +62,7 @@ def test_qwen35_fp8(self):; symbols: TestQwen35Fp8, test_qwen35_fp8, touching `TestQwen35Fp8, test_qwen35_fp8`; `.github/workflows/nightly-test-nvidia.yml` modified +27/-0 (27 lines); hunks: -24,6 +24,7 @@ on:; -512,6 +513,31 @@ jobs:.
- Code diff details:
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81 (81 lines); hunks: -1,81 +0,0; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` removed +0/-78 (78 lines); hunks: -1,78 +0,0; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_fp8.py` modified +14/-14 (28 lines); hunks: -17,43 +17,43; -62,7 +62,7 @@ def test_qwen35_fp8(self):; symbols: TestQwen35Fp8, test_qwen35_fp8
  - `.github/workflows/nightly-test-nvidia.yml` modified +27/-0 (27 lines); hunks: -24,6 +24,7 @@ on:; -512,6 +513,31 @@ jobs:
  - `test/registered/gb300/test_glm5_nvfp4.py` modified +12/-12 (24 lines); hunks: -16,42 +16,42; symbols: TestGlm5Nvfp4, test_glm5_nvfp4
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_deepseek_v32_nvfp4.py
@@ -1,81 +0,0 @@
-import unittest
-from sglang.test.accuracy_test_runner import AccuracyTestParams
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.performance_test_runner import PerformanceTestParams
-from sglang.test.run_combined_tests import run_combined_tests
-from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_deepseek_v32.py
@@ -1,78 +0,0 @@
-import unittest
-from sglang.test.accuracy_test_runner import AccuracyTestParams
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.performance_test_runner import PerformanceTestParams
-from sglang.test.run_combined_tests import run_combined_tests
-from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_qwen35_fp8.py
@@ -17,43 +17,43 @@
```

- Reviewed files:
  - tests: `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81; `test/registered/gb300/test_deepseek_v32.py` removed +0/-78; `test/registered/gb300/test_qwen35_fp8.py` modified +14/-14; `test/registered/gb300/test_glm5_nvfp4.py` modified +12/-12; `test/registered/gb300/test_qwen35_nvfp4.py` modified +5/-3; `test/registered/gb300/test_glm5_fp8.py` modified +4/-2
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +27/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/performance_test_runner.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_glm5_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27893 - [NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU

- Link: https://github.com/sgl-project/sglang/pull/27893
- Status/date: merged / 2026-06-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 34 files, +2984/-8438, 4706 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`; technical summary: Covers "[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU"; the main implementation surface is `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927 (6927 lines); `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0 (424 lines); hunks: -0,0 +1,424; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354 (354 lines); hunks: -1,354 +0,0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311 (311 lines); hunks: -1,311 +0,0.
- Code diff details:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927 (6927 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0 (424 lines); hunks: -0,0 +1,424
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354 (354 lines); hunks: -1,354 +0,0
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311 (311 lines); hunks: -1,311 +0,0
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx` removed +0/-300 (300 lines); hunks: -1,300 +0,0; symbols: gsm8k
- Key code excerpts:

```diff
diff -- docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx
@@ -0,0 +1,424 @@
+---
+title: "Qwen3-235B-A22B"
+metatags:
+  description: "Deploy Qwen3-235B-A22B model with SGLang on Ascend NPUs, including single-node, multi-node, and PD disaggregation modes."
+---
+## Introduction
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx
@@ -1,354 +0,0 @@
-title: "Qwen3.5 examples"
-metatags:
-  description: "Documentation for Qwen3.5 examples"
-## Environment Preparation
-### Installation
-<Warning>
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx
@@ -1,311 +0,0 @@
```

- Reviewed files:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927; `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx` removed +0/-300; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_examples.mdx` removed +0/-293
- Risk and verification: This is mostly docs/examples in `docs_new/docs.json`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27053 - [BCG][GLM5] perf: BCG support and prefill enhancements

- Link: https://github.com/sgl-project/sglang/pull/27053
- Status/date: merged / 2026-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`; associated commits `d5e9176f6581`
- Diff scope read: GitHub Pull Request files API returned 7 files, +694/-224, 1292 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BCG][GLM5] perf: BCG support and prefill enhancements"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`; technical summary: Covers "[BCG][GLM5] perf: BCG support and prefill enhancements"; the main implementation surface is `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestBCGGlm5Fp8TP8, setUpClass, tearDownClass, test_gsm8k, touching `TestBCGGlm5Fp8TP8, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestBCGGlm5Fp8TP8, setUpClass, tearDownClass, test_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py
@@ -0,0 +1,75 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
+from sglang.test.test_utils import (
```

- Reviewed files:
  - tests: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0
- Risk and verification: The diff ships test coverage in `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29194 - [AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook

- Link: https://github.com/sgl-project/sglang/pull/29194
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; associated commits `0075c8f02b1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-17, 188 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; technical summary: Covers "[AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1 (35 lines); hunks: -43,6 +43,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -51,43 +52,49 @@ import { GLM51Deployment } from '/src/snippets/autoregressiv...; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16 (35 lines); hunks: -4,7 +4,9 @@ export const GLM51Deployment = () => {; -25,11 +27,13 @@ export const GLM51Deployment = () => {.
- Code diff details:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1 (35 lines); hunks: -43,6 +43,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -51,43 +52,49 @@ import { GLM51Deployment } from '/src/snippets/autoregressiv...
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16 (35 lines); hunks: -4,7 +4,9 @@ export const GLM51Deployment = () => {; -25,11 +27,13 @@ export const GLM51Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx
@@ -43,6 +43,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/glm-51-deployment.
+      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>MXFP4</th>
@@ -51,43 +52,49 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/glm-51-deployment.
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>—</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>—</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>—</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>—</td>
diff -- docs_new/src/snippets/autoregressive/glm-51-deployment.jsx
@@ -4,7 +4,9 @@ export const GLM51Deployment = () => {
-  //   MI300X/MI325X/MI355X → BF16 (FP8 not verified on AMD)
+  //   MI300X / MI325X → BF16 (FP8 not verified on AMD)
+  //   MI355X (gfx950) → MXFP4 (amd/GLM-5.1-MXFP4); BF16 also supported.
+  //     MI350X is identical to MI355X (cooling only) and is omitted here.
@@ -25,11 +27,13 @@ export const GLM51Deployment = () => {
+        const isGfx950 = hw === 'mi355x'; // MI350X identical (cooling only)
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- Link: https://github.com/sgl-project/sglang/pull/28103
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +218/-19, 334 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`; technical summary: Covers "Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test"; the main implementation surface is `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4, touching `TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4`; `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4, touching `TestKimiK25Nvfp4, test_kimi_k25_nvfp4`; `.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:; `test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15.
- Code diff details:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4
  - `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4
  - `.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:
  - `test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15
  - `test/registered/gb300/test_glm5_fp8.py` modified +4/-1 (5 lines); hunks: -7,7 +7,10
- Key code excerpts:

```diff
diff -- test/registered/gb300/test_deepseek_v4_pro_fp4.py
@@ -0,0 +1,152 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_kimi_k25_nvfp4.py
@@ -6,9 +6,12 @@
-register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True)
+register_cuda_ci(
+    est_time=7200, suite="nightly-4-gpu-gb300-kimi-k25-nvfp4", nightly=True
+)
+DRAFT_MODEL_PATH = "lightseekorg/kimi-k2.5-eagle3-mla"
@@ -19,30 +22,43 @@
diff -- .github/workflows/nightly-test-nvidia.yml
@@ -539,7 +539,20 @@ jobs:
```

- Reviewed files:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0; `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10; `test/run_suite.py` modified +8/-1; `test/registered/gb300/test_glm5_fp8.py` modified +4/-1; `test/registered/gb300/test_kimi_k25.py` modified +4/-1; `test/registered/gb300/test_qwen35_nvfp4.py` modified +4/-1
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +18/-3
- Risk and verification: The diff ships test coverage in `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`, `test/registered/gb300/test_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29313 - [AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook

- Link: https://github.com/sgl-project/sglang/pull/29313
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; associated commits `7f376644e0cc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +22/-9, 66 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook"; model line: GLM-5/5.1; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`; technical summary: Covers "[AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8 (18 lines); hunks: -170,14 +170,16 @@ export const GLM51Deployment = () => {; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1 (13 lines); hunks: -94,7 +94,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -183,6 +183,7 @@ SGLANG_DSA_TRITON_PREFILL=1 sglang serve \.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8 (18 lines); hunks: -170,14 +170,16 @@ export const GLM51Deployment = () => {
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1 (13 lines); hunks: -94,7 +94,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -183,6 +183,7 @@ SGLANG_DSA_TRITON_PREFILL=1 sglang serve \
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/glm-51-deployment.jsx
@@ -170,14 +170,16 @@ export const GLM51Deployment = () => {
-    // EAGLE MTP speculative decoding: emitted by default (recommended). Excluded
-    // only on MI300X/MI325X (gfx942), where it is not yet verified.
-    if (!['mi300x', 'mi325x'].includes(hardware)) {
-      cmd += ' \\\n  --speculative-algorithm EAGLE';
-      cmd += ' \\\n  --speculative-num-steps 3';
-      cmd += ' \\\n  --speculative-eagle-topk 1';
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx
@@ -94,7 +94,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/glm-51-deployment.
-- **AMD GPUs**: BF16 and FP8 checkpoints run on MI300X/MI325X/MI355X at tp=8. On MI355X (gfx950), the MXFP4 checkpoint `amd/GLM-5.1-MXFP4` is also supported at tp=4 with `--kv-cac
+- **AMD GPUs**: BF16 and FP8 checkpoints run on MI300X/MI325X/MI355X at tp=8. On MI355X (gfx950), the MXFP4 checkpoint `amd/GLM-5.1-MXFP4` is also supported at tp=4 with `--kv-cac
@@ -183,6 +183,7 @@ SGLANG_DSA_TRITON_PREFILL=1 sglang serve \
+  --disable-custom-all-reduce \
@@ -201,6 +202,11 @@ sglang serve \
+  --speculative-algorithm EAGLE \
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29380 - [Docs] Add NVFP4 quantization to GLM-5.2 cookbook

- Link: https://github.com/sgl-project/sglang/pull/29380
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; associated commits `dd56a9f06948`
- Diff scope read: GitHub Pull Request files API returned 8 files, +112/-12, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add NVFP4 quantization to GLM-5.2 cookbook"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`; technical summary: Covers "[Docs] Add NVFP4 quantization to GLM-5.2 cookbook"; the main implementation surface is `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0 (79 lines); hunks: -15,6 +15,7 @@ export const config = {; -29,6 +30,7 @@ export const config = {; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0 (5 lines); hunks: -171,4 +171,9 @@ export const benchmarks = [; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2 (13 lines); hunks: -52,6 +52,10 @@ import { benchmarks } from "/src/snippets/configs/zai-org/glm...; -62,7 +66,7 @@ import { Playground } from "/src/snippets/_playground.jsx";.
- Code diff details:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0 (79 lines); hunks: -15,6 +15,7 @@ export const config = {; -29,6 +30,7 @@ export const config = {
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0 (5 lines); hunks: -171,4 +171,9 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2 (13 lines); hunks: -52,6 +52,10 @@ import { benchmarks } from "/src/snippets/configs/zai-org/glm...; -62,7 +66,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2.jsx
@@ -15,6 +15,7 @@ export const config = {
+    { id: "nvfp4", label: "NVFP4" },
@@ -29,6 +30,7 @@ export const config = {
+    "default|nvfp4": "nvidia/GLM-5.2-NVFP4",
@@ -90,6 +92,9 @@ sgl-eval run aime25 \\
+    // NVFP4 needs the dev image with modelopt_fp4 support (per-quant override).
+    "b300|nvfp4":  "lmsysorg/sglang:dev-glm52-nvfp4",
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx
@@ -171,4 +171,9 @@ export const benchmarks = [
+  // ---- NVFP4 (Blackwell Ultra) ----  benchmarks pending
+  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
+  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "balanced",    nodes: "single" } },
+  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
+  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced",    nodes: "single" } },
diff -- docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
@@ -52,6 +52,10 @@ import { benchmarks } from "/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx
+<Note>
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29466 - Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings

- Link: https://github.com/sgl-project/sglang/pull/29466
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; associated commits `12f76d115cd8`
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-8, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings"; model line: GLM-5/5.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; technical summary: Covers "Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings"; the main implementation surface is `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8 (16 lines); hunks: -652,8 +652,8 @@ sgl-eval run aime25 \\; -667,8 +667,8 @@ sgl-eval run aime25 \\.
- Code diff details:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8 (16 lines); hunks: -652,8 +652,8 @@ sgl-eval run aime25 \\; -667,8 +667,8 @@ sgl-eval run aime25 \\
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/configs/zai-org/glm-5.2.jsx
@@ -652,8 +652,8 @@ sgl-eval run aime25 \\
-        "--chunked-prefill-size 131072",
-        "--mem-fraction-static 0.70",
+        "--chunked-prefill-size 8192",
+        "--mem-fraction-static 0.8",
@@ -667,8 +667,8 @@ sgl-eval run aime25 \\
-        "--chunked-prefill-size 131072",
```

- Reviewed files:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8
- Risk and verification: This is mostly docs/examples in `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
