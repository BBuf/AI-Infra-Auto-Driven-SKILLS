# sglang GLM-5/5.1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | [#22712](https://github.com/sgl-project/sglang/pull/22712), [#23708](https://github.com/sgl-project/sglang/pull/23708) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` | [#27708](https://github.com/sgl-project/sglang/pull/27708), [#28437](https://github.com/sgl-project/sglang/pull/28437), [#29194](https://github.com/sgl-project/sglang/pull/29194), [#29313](https://github.com/sgl-project/sglang/pull/29313) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28454](https://github.com/sgl-project/sglang/pull/28454), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380) |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` | [#25814](https://github.com/sgl-project/sglang/pull/25814) |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | [#28433](https://github.com/sgl-project/sglang/pull/28433) |
| `docs_new/docs/hardware-platforms/ascend-npus/best_practice/glm5_1.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` | [#25814](https://github.com/sgl-project/sglang/pull/25814) |
| `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` | [#23540](https://github.com/sgl-project/sglang/pull/23540), [#27708](https://github.com/sgl-project/sglang/pull/27708), [#29194](https://github.com/sgl-project/sglang/pull/29194), [#29313](https://github.com/sgl-project/sglang/pull/29313) |
| `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380), [#29486](https://github.com/sgl-project/sglang/pull/29486) |
| `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | [#28437](https://github.com/sgl-project/sglang/pull/28437), [#28448](https://github.com/sgl-project/sglang/pull/28448), [#28460](https://github.com/sgl-project/sglang/pull/28460), [#29380](https://github.com/sgl-project/sglang/pull/29380), [#29466](https://github.com/sgl-project/sglang/pull/29466), [#29486](https://github.com/sgl-project/sglang/pull/29486) |
| `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi30x/test_glm51_hisparse_eval_mi30x.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/accuracy/mi35x/test_glm51_hisparse_eval_mi35x.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` | [#26396](https://github.com/sgl-project/sglang/pull/26396) |
| `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` | [#18911](https://github.com/sgl-project/sglang/pull/18911), [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` | [#22336](https://github.com/sgl-project/sglang/pull/22336) |
| `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` | [#21773](https://github.com/sgl-project/sglang/pull/21773) |
| `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` | [#21710](https://github.com/sgl-project/sglang/pull/21710) |
| `test/registered/ascend/performance/glm5_1/test_npu_glm5_1_w4a8_1p1d_32p_in64k_out1k_50ms_aime26.py` | 无直接 PR 号提交 |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp4.py` | 无直接 PR 号提交 |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` | [#27053](https://github.com/sgl-project/sglang/pull/27053) |
| `test/registered/gb300/test_glm5_fp8.py` | [#22399](https://github.com/sgl-project/sglang/pull/22399) |
| `test/registered/gb300/test_glm5_nvfp4.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_dsa_glm5_hisparse.py` | [#28607](https://github.com/sgl-project/sglang/pull/28607) |
| `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 22
- 原文档显式引用补充 PR 数: 22
- 当前文档总 PR 数: 44
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #18521 - Support GlmMoeDsaForCausalLM

- 链接: https://github.com/sgl-project/sglang/pull/18521
- 状态/时间: merged / 2026-02-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+22/-7，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support GlmMoeDsaForCausalLM」；模型线: GLM-5/5.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Support GlmMoeDsaForCausalLM」；主要实现面是 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes，涉及 `is_deepseek_nsa, from_server_args, _config_draft_model`；`python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM，涉及 `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM`；`python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params，涉及 `_handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: -61,6 +61,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; -271,10 +272,10 @@ def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -79,6 +79,7; -1279,4 +1280,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: -1194,9 +1194,15 @@ def _handle_model_specific_adjustments(self):; -2323,6 +2329,7 @@ def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +6/-5; `python/sglang/srt/models/glm4_moe.py` modified +6/-1; `python/sglang/srt/server_args.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18804 - Fix GLM-5 fused shared expert

- 链接: https://github.com/sgl-project/sglang/pull/18804
- 状态/时间: merged / 2026-02-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-5 fused shared expert」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Fix GLM-5 fused shared expert」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts，涉及 `set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1281,7 +1281,8 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-    pass
+    def determine_num_fused_shared_experts(self):
+        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18911
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`；关联提交 `23adb50751d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+635/-1，可读 patch 725 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [GLM-5 Day 0] Add GLM-5 nightly test」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`；技术摘要: 覆盖「[AMD] [GLM-5 Day 0] Add GLM-5 nightly test」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- 链接: https://github.com/sgl-project/sglang/pull/20062
- 状态/时间: merged / 2026-03-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+32/-59，可读 patch 200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V32/GLM5] Control the threshold of applying dense attention with an environ」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`；技术摘要: 覆盖「[V32/GLM5] Control the threshold of applying dense attention with an environ」；主要实现面是 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl，涉及 `NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed`；`python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`；`test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: -1353,12 +1353,35 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: -377,8 +377,7 @@ class Envs:; symbols: Envs
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46; `python/sglang/srt/server_args.py` modified +26/-3; `python/sglang/srt/environ.py` modified +1/-2
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4
  - docs: `docs/references/environment_variables.md` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- 链接: https://github.com/sgl-project/sglang/pull/22179
- 状态/时间: merged / 2026-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-12，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `docs/basic_usage/deepseek_v32.md`；技术摘要: 覆盖「[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation」；主要实现面是 `docs/basic_usage/deepseek_v32.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...。
- 代码 diff 细节:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +11/-12
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/deepseek_v32.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- 链接: https://github.com/sgl-project/sglang/pull/22314
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+27/-31，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/memory_pool.py`；技术摘要: 覆盖「[AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300」；主要实现面是 `python/sglang/srt/mem_cache/memory_pool.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer，涉及 `set_mla_kv_buffer`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: -45,7 +45,7; -1575,37 +1575,33 @@ def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21710
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`；关联提交 `db60a620dbf1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+345/-5，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`；技术摘要: 覆盖「[AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x」；主要实现面是 `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf，涉及 `generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5，涉及 `generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass`；`test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name，涉及 `get_display_name`；`test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name，涉及 `get_display_name`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: -59,13 +59,17 @@ def get_display_name(self) -> str:; -77,7 +81,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -64,13 +64,17 @@ def get_display_name(self) -> str:; -82,7 +86,7 @@ def get_display_name(self) -> str:; symbols: get_display_name
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0; `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` modified +6/-2
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22285 - Add CI tests for GLM-5

- 链接: https://github.com/sgl-project/sglang/pull/22285
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+153/-30，可读 patch 301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add CI tests for GLM-5」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；技术摘要: 覆盖「Add CI tests for GLM-5」；主要实现面是 `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass，涉及 `TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP`；`test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed，涉及 `TestDeepseekV32DPMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: -14,9 +14,10; -138,5 +139,124 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: -20,6 +20,7; -47,12 +48,13 @@ def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- 链接: https://github.com/sgl-project/sglang/pull/22399
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/gb300/test_glm5_fp8.py`；关联提交 `46c2b7762765`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+82/-6，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_glm5_fp8.py`；技术摘要: 覆盖「[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model」；主要实现面是 `test/registered/gb300/test_glm5_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8，涉及 `TestGlm5Fp8, test_glm5_fp8`。
- 代码 diff 细节:
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: -8,7 +8,7; -27,7 +27,7; symbols: TestGlm5Fp8, test_glm5_fp8
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/gb300/test_glm5_fp8.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/22336
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；关联提交 `ef6bfc1197ab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+918/-25，可读 patch 1064 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；技术摘要: 覆盖「[AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples，涉及 `ModelConfig, get_display_name, get_one_example`；`test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf，涉及 `generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51，涉及 `generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0; `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0; `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22712 - [NPU] update GLM5 running guide

- 链接: https://github.com/sgl-project/sglang/pull/22712
- 状态/时间: merged / 2026-04-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；关联提交 `13a4aafdbe69`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] update GLM5 running guide」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs/platforms/ascend/ascend_npu_glm5_examples.md`；技术摘要: 覆盖「[NPU] update GLM5 running guide」；主要实现面是 `docs/platforms/ascend/ascend_npu_glm5_examples.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: -53,10 +53,16 @@ docker run -itd --shm-size=16g --privileged=true --name ${NA...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- 链接: https://github.com/sgl-project/sglang/pull/22543
- 状态/时间: merged / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+8/-0，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix」；主要实现面是 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights，涉及 `post_load_weights`；`python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config，涉及 `_get_quantization_config`；`python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values，涉及 `_handle_missing_default_values`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: -560,6 +560,9 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: -198,6 +198,9 @@ def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1016,6 +1016,8 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0; `python/sglang/srt/model_loader/loader.py` modified +3/-0; `python/sglang/srt/server_args.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21773
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；关联提交 `39c6bf730c41`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+528/-130，可读 patch 821 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；技术摘要: 覆盖「[AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name，涉及 `get_model_path, ModelConfig, __post_init__`；`test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: -0,0 +1,187; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0; `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- 链接: https://github.com/sgl-project/sglang/pull/22595
- 状态/时间: merged / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+67/-1，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: normalize tool message content for GLM5.1 chat template」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`；技术摘要: 覆盖「fix: normalize tool message content for GLM5.1 chat template」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template，涉及 `normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template`；`test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined，涉及 `test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: -60,6 +60,28; -457,6 +479,10 @@ def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: -19,7 +19,10; -894,5 +897,42 @@ def test_required_without_parser_invalid_json_returns_none(...; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0
  - tests: `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1
- 验证与风险: diff 自带测试面 `test/registered/openai_server/basic/test_serving_chat.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- 链接: https://github.com/sgl-project/sglang/pull/22850
- 状态/时间: merged / 2026-04-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+24/-5，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)」；模型线: GLM-5/5.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；技术摘要: 覆盖「[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)」；主要实现面是 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache，涉及 `__init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- 链接: https://github.com/sgl-project/sglang/pull/23219
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+41/-15，可读 patch 87 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable MTP for GLM-5-mxfp4 model」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_nextn.py`；技术摘要: 覆盖「[AMD] Enable MTP for GLM-5-mxfp4 model」；主要实现面是 `python/sglang/srt/models/deepseek_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- 链接: https://github.com/sgl-project/sglang/pull/23060
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Fix dynamic chunking profiling crash on GLM-5 models」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/managers/scheduler_pp_mixin.py`；技术摘要: 覆盖「[fix] Fix dynamic chunking profiling crash on GLM-5 models」；主要实现面是 `python/sglang/srt/managers/scheduler_pp_mixin.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor，涉及 `profile_and_init_predictor`。
- 代码 diff 细节:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/managers/scheduler_pp_mixin.py
@@ -20,6 +20,7 @@
+    set_is_extend_in_batch,
@@ -631,6 +632,8 @@ def profile_and_init_predictor(self: Scheduler):
+                set_is_extend_in_batch(batch.forward_mode.is_extend())
```

- 已读文件:
  - runtime: `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/scheduler_pp_mixin.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- 链接: https://github.com/sgl-project/sglang/pull/23540
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；关联提交 `9b2f7f8a91d4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+15/-13，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: split MI300X and MI325X options in GLM-5.1 generator」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；技术摘要: 覆盖「docs: split MI300X and MI325X options in GLM-5.1 generator」；主要实现面是 `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: -14,7 +14,8 @@ export const GLM51Deployment = () => {; -23,7 +24,7 @@ export const GLM51Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23708 - [NPU][Doc] Update GLM-5 docs, enabling deepep by default

- 链接: https://github.com/sgl-project/sglang/pull/23708
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；关联提交 `461bc8af494c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU][Doc] Update GLM-5 docs, enabling deepep by default」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs/platforms/ascend/ascend_npu_glm5_examples.md`；技术摘要: 覆盖「[NPU][Doc] Update GLM-5 docs, enabling deepep by default」；主要实现面是 `docs/platforms/ascend/ascend_npu_glm5_examples.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3 (6 lines); hunks: -161,8 +161,6 @@ P_IP=('your ip1' 'your ip2'); -182,7 +180,9 @@ do。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3 (6 lines); hunks: -161,8 +161,6 @@ P_IP=('your ip1' 'your ip2'); -182,7 +180,9 @@ do
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_glm5_examples.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25453 - [CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85

- 链接: https://github.com/sgl-project/sglang/pull/25453
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_glm_51_fp8.py`；技术摘要: 覆盖「[CI] Lower mem-fraction-static for GLM-5.1 FP8 8-GPU test to 0.85」；主要实现面是 `test/registered/8-gpu-models/test_glm_51_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_glm_51_fp8.py
@@ -15,7 +15,7 @@
-    "--mem-fraction-static=0.9",
+    "--mem-fraction-static=0.85",
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_glm_51_fp8.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_glm_51_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25735 - [NPU] [DOCS] Improved the usability of Ascend NPU documents

- 链接: https://github.com/sgl-project/sglang/pull/25735
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+468/-49，可读 patch 743 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [DOCS] Improved the usability of Ascend NPU documents」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`；技术摘要: 覆盖「[NPU] [DOCS] Improved the usability of Ascend NPU documents」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22 (244 lines); hunks: -53,19 +53,29 @@ You can install SGLang using any of the methods below. Pleas...; -142,14 +152,42 @@ pip install -e python[all_npu]；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0 (60 lines); hunks: -221,3 +221,63 @@ python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__pa...；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11 (59 lines); hunks: -7,15 +7,18 @@ metatags:; -42,9 +45,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7 (55 lines); hunks: -11,22 +11,29 @@ The GLM (General Language Model) series is an open-source bi...; -53,9 +60,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...。
- 代码 diff 细节:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22 (244 lines); hunks: -53,19 +53,29 @@ You can install SGLang using any of the methods below. Pleas...; -142,14 +152,42 @@ pip install -e python[all_npu]
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0 (60 lines); hunks: -221,3 +221,63 @@ python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__pa...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11 (59 lines); hunks: -7,15 +7,18 @@ metatags:; -42,9 +45,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7 (55 lines); hunks: -11,22 +11,29 @@ The GLM (General Language Model) series is an open-source bi...; -53,9 +60,39 @@ docker run -itd --shm-size=16g --privileged=true --name ${NAM...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quick_start.mdx` modified +42/-5 (47 lines); hunks: -13,14 +13,23 @@ metatags:; -39,6 +48,34 @@ docker run -it --rm --privileged --network=host --ipc=host --...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +222/-22; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` modified +60/-0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +48/-11; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5_examples.mdx` modified +48/-7; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quick_start.mdx` modified +42/-5; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_accuracy_evaluation.mdx` modified +31/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_accuracy_evaluation.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25814 - Update GLM-5 H200 FP8

- 链接: https://github.com/sgl-project/sglang/pull/25814
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；关联提交 `da6d549ab28a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update GLM-5 H200 FP8」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`；技术摘要: 覆盖「Update GLM-5 H200 FP8」；主要实现面是 `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0 (5 lines); hunks: -206,6 +206,11 @@ export const GLM5Deployment = () => {；`docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0 (1 lines); hunks: -109,6 +109,7 @@ sglang serve \。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0 (5 lines); hunks: -206,6 +206,11 @@ export const GLM5Deployment = () => {
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0 (1 lines); hunks: -109,6 +109,7 @@ sglang serve \
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +5/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- 链接: https://github.com/sgl-project/sglang/pull/25821
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 162 个文件，+11303/-10745，可读 patch 15980 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；技术摘要: 覆盖「[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename」；主要实现面是 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)；`python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)；`python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)；`python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25266 - [AMD][CI] Clean up AMD nightly + pr-test workflows

- 链接: https://github.com/sgl-project/sglang/pull/25266
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1034/-927，可读 patch 3120 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][CI] Clean up AMD nightly + pr-test workflows」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `.github/workflows/pr-test-amd-rocm720.yml`；技术摘要: 覆盖「[AMD][CI] Clean up AMD nightly + pr-test workflows」；主要实现面是 `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `.github/workflows/pr-test-amd-rocm720.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490 (1014 lines); hunks: -27,43 +27,59 @@ on:; -108,6 +124,11 @@ concurrency:；`.github/workflows/nightly-test-amd.yml` modified +454/-409 (863 lines); hunks: -27,41 +27,56 @@ on:; -106,6 +121,11 @@ concurrency:；`.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15 (53 lines); hunks: -46,7 +46,7 @@ on:; -68,6 +68,11 @@ on:；`.github/workflows/pr-test-amd.yml` modified +10/-5 (15 lines); hunks: -38,7 +38,7 @@ on:; -67,6 +67,11 @@ on:。
- 代码 diff 细节:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490 (1014 lines); hunks: -27,43 +27,59 @@ on:; -108,6 +124,11 @@ concurrency:
  - `.github/workflows/nightly-test-amd.yml` modified +454/-409 (863 lines); hunks: -27,41 +27,56 @@ on:; -106,6 +121,11 @@ concurrency:
  - `.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15 (53 lines); hunks: -46,7 +46,7 @@ on:; -68,6 +68,11 @@ on:
  - `.github/workflows/pr-test-amd.yml` modified +10/-5 (15 lines); hunks: -38,7 +38,7 @@ on:; -67,6 +67,11 @@ on:
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ class TestGLM51EvalAMD(unittest.TestCase):; symbols: TestGLM51EvalAMD, setUpClass, test_glm51_accuracy
- 关键代码摘录:

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

- 已读文件:
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +524/-490; `.github/workflows/nightly-test-amd.yml` modified +454/-409; `.github/workflows/pr-test-amd-rocm720.yml` modified +38/-15; `.github/workflows/pr-test-amd.yml` modified +10/-5
  - tests: `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` modified +1/-1; `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` modified +1/-1; `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` modified +1/-1; `test/registered/amd/disaggregation/test_disaggregation_basic.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/disaggregation/test_disaggregation_basic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26396 - [AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate

- 链接: https://github.com/sgl-project/sglang/pull/26396
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`；关联提交 `d44584e8d8b2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+144/-0，可读 patch 145 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Add GLM-5.1 MXFP4 TP2 accuracy gate」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: _raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x, setUpClass，涉及 `_raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: _raise_nofile_limit, _get_model_path, TestGLM51MXFP4TP2GSM8KMI35x, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py` added +144/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_glm51_mxfp4_tp2_gsm8k_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0；`docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0；`docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...；`docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64，涉及 `image_to_base64`。
- 代码 diff 细节:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27001
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+11/-471，可读 patch 936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`；`test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass，涉及 `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`；`test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models，涉及 `get_model_path, ModelConfig, get_display_name`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27708 - [Docs] Add GLM-5.1 NVFP4 to cookbook

- 链接: https://github.com/sgl-project/sglang/pull/27708
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；关联提交 `91ff7baa2860`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+92/-38，可读 patch 275 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add GLM-5.1 NVFP4 to cookbook」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；技术摘要: 覆盖「[Docs] Add GLM-5.1 NVFP4 to cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21 (90 lines); hunks: -1,7 +1,7; -11,6 +11,7 @@ tag: NEW；`docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17 (40 lines); hunks: -1,19 +1,19; -25,10 +25,13 @@ export const GLM51Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21 (90 lines); hunks: -1,7 +1,7; -11,6 +11,7 @@ tag: NEW
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17 (40 lines); hunks: -1,19 +1,19; -25,10 +25,13 @@ export const GLM51Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +69/-21; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +23/-17
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27964 - [Spec] Retire Spec V1

- 链接: https://github.com/sgl-project/sglang/pull/27964
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 46 个文件，+111/-252，可读 patch 1422 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Retire Spec V1」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`；技术摘要: 覆盖「[Spec] Retire Spec V1」；主要实现面是 `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass，涉及 `TestDeepseekMTP, setUpClass, tearDownClass`；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do；`python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family，涉及 `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`；`docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...。
- 代码 diff 细节:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28437 - docs(cookbook): add GLM-5.2 deployment cookbook

- 链接: https://github.com/sgl-project/sglang/pull/28437
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；关联提交 `0cb6183432dc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+970/-2，可读 patch 996 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add GLM-5.2 deployment cookbook」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；技术摘要: 覆盖「docs(cookbook): add GLM-5.2 deployment cookbook」；主要实现面是 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0 (657 lines); hunks: -0,0 +1,657；`docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0 (116 lines); hunks: -0,0 +1,116；`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0 (195 lines); hunks: -0,0 +1,195；`docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0 (657 lines); hunks: -0,0 +1,657
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0 (116 lines); hunks: -0,0 +1,116
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0 (195 lines); hunks: -0,0 +1,195
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` added +657/-0; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` added +116/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` added +195/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28448 - docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags

- 链接: https://github.com/sgl-project/sglang/pull/28448
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；关联提交 `00081a00d5b3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+36/-54，可读 patch 354 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；技术摘要: 覆盖「docs(cookbook): tune GLM-5.2 MTP to 5-1-6 and simplify launch flags」；主要实现面是 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45 (65 lines); hunks: -36,7 +36,6 @@ export const config = {; -141,10 +140,10 @@ sgl-eval run aime25 \\；`docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6 (17 lines); hunks: -5,13 +5,16; -36,13 +39,14 @@ export const benchmarks = [；`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2 (4 lines); hunks: -92,8 +92,8 @@ import { Playground } from "/src/snippets/_playground.jsx";。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45 (65 lines); hunks: -36,7 +36,6 @@ export const config = {; -141,10 +140,10 @@ sgl-eval run aime25 \\
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6 (17 lines); hunks: -5,13 +5,16; -36,13 +39,14 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2 (4 lines); hunks: -92,8 +92,8 @@ import { Playground } from "/src/snippets/_playground.jsx";
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +20/-45; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +11/-6; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28454 - docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort

- 链接: https://github.com/sgl-project/sglang/pull/28454
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；关联提交 `33f205d8c5b2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort」；模型线: GLM-5/5.1；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；技术摘要: 覆盖「docs(cookbook): fix GLM-5.2 thinking toggle kwarg + document reasoning effort」；主要实现面是 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2 (12 lines); hunks: -104,7 +104,15 @@ import { Playground } from "/src/snippets/_playground.jsx";; -115,7 +123,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2 (12 lines); hunks: -104,7 +104,15 @@ import { Playground } from "/src/snippets/_playground.jsx";; -115,7 +123,7 @@ client = OpenAI(base_url="http://localhost:30000/v1", api_ke...
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +10/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28433 - [Ascend]GLM 5.2 deployment

- 链接: https://github.com/sgl-project/sglang/pull/28433
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`；关联提交 `71b090a8e785`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+376/-0，可读 patch 377 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend]GLM 5.2 deployment」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`；技术摘要: 覆盖「[Ascend]GLM 5.2 deployment」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0 (376 lines); hunks: -0,0 +1,376。
- 代码 diff 细节:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0 (376 lines); hunks: -0,0 +1,376
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` added +376/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28460 - docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)

- 链接: https://github.com/sgl-project/sglang/pull/28460
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；关联提交 `72ccfec5949d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+76/-23，可读 patch 170 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；技术摘要: 覆盖「docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16)」；主要实现面是 `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8 (69 lines); hunks: -100,14 +100,67 @@ export const benchmarks = [; symbols: kernels，涉及 `kernels`；`docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12 (24 lines); hunks: -354,13 +354,13 @@ sgl-eval run aime25 \\; -377,7 +377,7 @@ sgl-eval run aime25 \\; symbols: kernels，涉及 `kernels`；`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3 (6 lines); hunks: -1,6 +1,6; -94,10 +94,10 @@ import { Playground } from "/src/snippets/_playground.jsx";。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8 (69 lines); hunks: -100,14 +100,67 @@ export const benchmarks = [; symbols: kernels
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12 (24 lines); hunks: -354,13 +354,13 @@ sgl-eval run aime25 \\; -377,7 +377,7 @@ sgl-eval run aime25 \\; symbols: kernels
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3 (6 lines); hunks: -1,6 +1,6; -94,10 +94,10 @@ import { Playground } from "/src/snippets/_playground.jsx";
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +61/-8; `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +12/-12; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28607 - [misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e

- 链接: https://github.com/sgl-project/sglang/pull/28607
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_dsa_glm5_hisparse.py`；关联提交 `8f6d9ef9a55e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+124/-203，可读 patch 368 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `test/registered/models_e2e/test_dsa_glm5_hisparse.py`；技术摘要: 覆盖「[misc] Drop redundant req_pool_indices_cpu guards; fold hisparse into GLM-5.1 e2e」；主要实现面是 `test/registered/models_e2e/test_dsa_glm5_hisparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM5HiSparse，涉及 `TestGLM5HiSparse`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM5HiSparse
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/models_e2e/test_dsa_glm5_hisparse.py` added +54/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_dsa_models_hisparse.py`, `test/registered/models_e2e/test_dsa_glm5_hisparse.py`, `test/registered/unit/managers/test_schedule_batch_req_pool_indices.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167；`docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {；`docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28536 - ci: run GB300 nightly suite in the standard Nvidia nightly workflow

- 链接: https://github.com/sgl-project/sglang/pull/28536
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+72/-197，可读 patch 438 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`；技术摘要: 覆盖「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；主要实现面是 `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81 (81 lines); hunks: -1,81 +0,0; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4，涉及 `TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4`；`test/registered/gb300/test_deepseek_v32.py` removed +0/-78 (78 lines); hunks: -1,78 +0,0; symbols: TestDeepseekV32, test_deepseek_v32，涉及 `TestDeepseekV32, test_deepseek_v32`；`test/registered/gb300/test_qwen35_fp8.py` modified +14/-14 (28 lines); hunks: -17,43 +17,43; -62,7 +62,7 @@ def test_qwen35_fp8(self):; symbols: TestQwen35Fp8, test_qwen35_fp8，涉及 `TestQwen35Fp8, test_qwen35_fp8`；`.github/workflows/nightly-test-nvidia.yml` modified +27/-0 (27 lines); hunks: -24,6 +24,7 @@ on:; -512,6 +513,31 @@ jobs:。
- 代码 diff 细节:
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81 (81 lines); hunks: -1,81 +0,0; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` removed +0/-78 (78 lines); hunks: -1,78 +0,0; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_fp8.py` modified +14/-14 (28 lines); hunks: -17,43 +17,43; -62,7 +62,7 @@ def test_qwen35_fp8(self):; symbols: TestQwen35Fp8, test_qwen35_fp8
  - `.github/workflows/nightly-test-nvidia.yml` modified +27/-0 (27 lines); hunks: -24,6 +24,7 @@ on:; -512,6 +513,31 @@ jobs:
  - `test/registered/gb300/test_glm5_nvfp4.py` modified +12/-12 (24 lines); hunks: -16,42 +16,42; symbols: TestGlm5Nvfp4, test_glm5_nvfp4
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/gb300/test_deepseek_v32_nvfp4.py` removed +0/-81; `test/registered/gb300/test_deepseek_v32.py` removed +0/-78; `test/registered/gb300/test_qwen35_fp8.py` modified +14/-14; `test/registered/gb300/test_glm5_nvfp4.py` modified +12/-12; `test/registered/gb300/test_qwen35_nvfp4.py` modified +5/-3; `test/registered/gb300/test_glm5_fp8.py` modified +4/-2
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +27/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/performance_test_runner.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_glm5_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27893 - [NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU

- 链接: https://github.com/sgl-project/sglang/pull/27893
- 状态/时间: merged / 2026-06-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+2984/-8438，可读 patch 4706 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`；技术摘要: 覆盖「[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927 (6927 lines)；`docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0 (424 lines); hunks: -0,0 +1,424；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354 (354 lines); hunks: -1,354 +0,0；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311 (311 lines); hunks: -1,311 +0,0。
- 代码 diff 细节:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927 (6927 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0 (424 lines); hunks: -0,0 +1,424
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354 (354 lines); hunks: -1,354 +0,0
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311 (311 lines); hunks: -1,311 +0,0
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx` removed +0/-300 (300 lines); hunks: -1,300 +0,0; symbols: gsm8k
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` removed +0/-6927; `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx` added +424/-0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` removed +0/-354; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` removed +0/-311; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx` removed +0/-300; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_examples.mdx` removed +0/-293
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/docs.json`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_deepseek_example.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27053 - [BCG][GLM5] perf: BCG support and prefill enhancements

- 链接: https://github.com/sgl-project/sglang/pull/27053
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`；关联提交 `d5e9176f6581`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+694/-224，可读 patch 1292 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BCG][GLM5] perf: BCG support and prefill enhancements」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`；技术摘要: 覆盖「[BCG][GLM5] perf: BCG support and prefill enhancements」；主要实现面是 `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestBCGGlm5Fp8TP8, setUpClass, tearDownClass, test_gsm8k，涉及 `TestBCGGlm5Fp8TP8, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestBCGGlm5Fp8TP8, setUpClass, tearDownClass, test_gsm8k
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0
- 验证与风险: diff 自带测试面 `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29194 - [AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29194
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；关联提交 `0075c8f02b1e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+53/-17，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；技术摘要: 覆盖「[AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1 (35 lines); hunks: -43,6 +43,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -51,43 +52,49 @@ import { GLM51Deployment } from '/src/snippets/autoregressiv...；`docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16 (35 lines); hunks: -4,7 +4,9 @@ export const GLM51Deployment = () => {; -25,11 +27,13 @@ export const GLM51Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1 (35 lines); hunks: -43,6 +43,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -51,43 +52,49 @@ import { GLM51Deployment } from '/src/snippets/autoregressiv...
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16 (35 lines); hunks: -4,7 +4,9 @@ export const GLM51Deployment = () => {; -25,11 +27,13 @@ export const GLM51Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +34/-1; `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +19/-16
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/28103
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+218/-19，可读 patch 334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`；技术摘要: 覆盖「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；主要实现面是 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4，涉及 `TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4`；`test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4，涉及 `TestKimiK25Nvfp4, test_kimi_k25_nvfp4`；`.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:；`test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15。
- 代码 diff 细节:
  - `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: TestDeepSeekV4ProFp4, test_deepseek_v4_pro_fp4
  - `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10 (36 lines); hunks: -6,9 +6,12; -19,30 +22,43; symbols: TestKimiK25Nvfp4, test_kimi_k25_nvfp4
  - `.github/workflows/nightly-test-nvidia.yml` modified +18/-3 (21 lines); hunks: -539,7 +539,20 @@ jobs:; -549,8 +562,10 @@ jobs:
  - `test/run_suite.py` modified +8/-1 (9 lines); hunks: -121,8 +121,15
  - `test/registered/gb300/test_glm5_fp8.py` modified +4/-1 (5 lines); hunks: -7,7 +7,10
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/gb300/test_deepseek_v4_pro_fp4.py` added +152/-0; `test/registered/gb300/test_kimi_k25_nvfp4.py` modified +26/-10; `test/run_suite.py` modified +8/-1; `test/registered/gb300/test_glm5_fp8.py` modified +4/-1; `test/registered/gb300/test_kimi_k25.py` modified +4/-1; `test/registered/gb300/test_qwen35_nvfp4.py` modified +4/-1
  - ci: `.github/workflows/nightly-test-nvidia.yml` modified +18/-3
- 验证与风险: diff 自带测试面 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`, `test/registered/gb300/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29313 - [AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29313
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；关联提交 `7f376644e0cc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+22/-9，可读 patch 66 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook」；模型线: GLM-5/5.1；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`；技术摘要: 覆盖「[AMD] [GLM5] Mark EAGLE verified on MI300X/MI325X (gfx942) in GLM-5.1 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8 (18 lines); hunks: -170,14 +170,16 @@ export const GLM51Deployment = () => {；`docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1 (13 lines); hunks: -94,7 +94,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -183,6 +183,7 @@ SGLANG_DSA_TRITON_PREFILL=1 sglang serve \。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8 (18 lines); hunks: -170,14 +170,16 @@ export const GLM51Deployment = () => {
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1 (13 lines); hunks: -94,7 +94,7 @@ import { GLM51Deployment } from '/src/snippets/autoregressive/...; -183,6 +183,7 @@ SGLANG_DSA_TRITON_PREFILL=1 sglang serve \
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +10/-8; `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` modified +12/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29380 - [Docs] Add NVFP4 quantization to GLM-5.2 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29380
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；关联提交 `dd56a9f06948`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+112/-12，可读 patch 224 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add NVFP4 quantization to GLM-5.2 cookbook」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`；技术摘要: 覆盖「[Docs] Add NVFP4 quantization to GLM-5.2 cookbook」；主要实现面是 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`, `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0 (79 lines); hunks: -15,6 +15,7 @@ export const config = {; -29,6 +30,7 @@ export const config = {；`docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0 (5 lines); hunks: -171,4 +171,9 @@ export const benchmarks = [；`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2 (13 lines); hunks: -52,6 +52,10 @@ import { benchmarks } from "/src/snippets/configs/zai-org/glm...; -62,7 +66,7 @@ import { Playground } from "/src/snippets/_playground.jsx";。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0 (79 lines); hunks: -15,6 +15,7 @@ export const config = {; -29,6 +30,7 @@ export const config = {
  - `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0 (5 lines); hunks: -171,4 +171,9 @@ export const benchmarks = [
  - `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2 (13 lines); hunks: -52,6 +52,10 @@ import { benchmarks } from "/src/snippets/configs/zai-org/glm...; -62,7 +66,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +79/-0; `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` modified +5/-0; `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` modified +11/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `docs_new/src/snippets/_deployment.jsx`, `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29466 - Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings

- 链接: https://github.com/sgl-project/sglang/pull/29466
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；关联提交 `12f76d115cd8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-8，可读 patch 44 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings」；模型线: GLM-5/5.1；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；技术摘要: 覆盖「Update GLM-5.2 B300 and GB300 NVFP4 cookbook settings」；主要实现面是 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8 (16 lines); hunks: -652,8 +652,8 @@ sgl-eval run aime25 \\; -667,8 +667,8 @@ sgl-eval run aime25 \\。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8 (16 lines); hunks: -652,8 +652,8 @@ sgl-eval run aime25 \\; -667,8 +667,8 @@ sgl-eval run aime25 \\
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` modified +8/-8
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
