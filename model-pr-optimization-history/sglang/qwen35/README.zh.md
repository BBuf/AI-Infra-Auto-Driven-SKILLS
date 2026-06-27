# sglang Qwen3.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/qwen3_5.md` | 无直接 PR 号提交 |
| `docs/platforms/ascend/ascend_npu_qwen3_5_examples.md` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` | [#26695](https://github.com/sgl-project/sglang/pull/26695), [#27296](https://github.com/sgl-project/sglang/pull/27296), [#27660](https://github.com/sgl-project/sglang/pull/27660) |
| `docs_new/docs/hardware-platforms/ascend-npus/best_practice/qwen3_5_397b.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_5_397b.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` | [#26695](https://github.com/sgl-project/sglang/pull/26695), [#27296](https://github.com/sgl-project/sglang/pull/27296), [#27660](https://github.com/sgl-project/sglang/pull/27660) |
| `python/sglang/srt/configs/qwen3_5.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489) |
| `python/sglang/srt/models/qwen3_5.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489), [#18538](https://github.com/sgl-project/sglang/pull/18538), [#18544](https://github.com/sgl-project/sglang/pull/18544), [#18937](https://github.com/sgl-project/sglang/pull/18937), [#19070](https://github.com/sgl-project/sglang/pull/19070), [#19220](https://github.com/sgl-project/sglang/pull/19220), [#19411](https://github.com/sgl-project/sglang/pull/19411), [#19484](https://github.com/sgl-project/sglang/pull/19484), [#19670](https://github.com/sgl-project/sglang/pull/19670), [#19767](https://github.com/sgl-project/sglang/pull/19767), [#20386](https://github.com/sgl-project/sglang/pull/20386), [#20736](https://github.com/sgl-project/sglang/pull/20736), ... (29 total) |
| `python/sglang/srt/models/qwen3_5_mtp.py` | [#18489](https://github.com/sgl-project/sglang/pull/18489), [#18538](https://github.com/sgl-project/sglang/pull/18538), [#18926](https://github.com/sgl-project/sglang/pull/18926), [#18937](https://github.com/sgl-project/sglang/pull/18937), [#19391](https://github.com/sgl-project/sglang/pull/19391), [#19767](https://github.com/sgl-project/sglang/pull/19767), [#20918](https://github.com/sgl-project/sglang/pull/20918), [#23146](https://github.com/sgl-project/sglang/pull/23146), [#23331](https://github.com/sgl-project/sglang/pull/23331) |
| `test/lm_eval_configs/Qwen3.5-397B-A17B.yaml` | 无直接 PR 号提交 |
| `test/manual/4-gpu-models/test_qwen35_fp4_triton.py` | 无直接 PR 号提交 |
| `test/manual/4-gpu-models/test_qwen35_models_archived.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_qwen35.py` | [#19906](https://github.com/sgl-project/sglang/pull/19906), [#22399](https://github.com/sgl-project/sglang/pull/22399) |
| `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` | [#21669](https://github.com/sgl-project/sglang/pull/21669) |
| `test/registered/attention/test_qwen35_deterministic.py` | [#27869](https://github.com/sgl-project/sglang/pull/27869) |
| `test/registered/gb300/test_qwen35_fp8.py` | 无直接 PR 号提交 |
| `test/registered/gb300/test_qwen35_nvfp4.py` | 无直接 PR 号提交 |
| `test/registered/hicache/test_qwen35_hicache.py` | 无直接 PR 号提交 |
| `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py` | [#23594](https://github.com/sgl-project/sglang/pull/23594) |
| `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py` | [#23594](https://github.com/sgl-project/sglang/pull/23594) |
| `test/registered/models_e2e/test_qwen35_fp4_flashinfer.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_qwen35_fp4_mtp.py` | 无直接 PR 号提交 |
| `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py` | [#23062](https://github.com/sgl-project/sglang/pull/23062) |

## PR 覆盖总览

- git 追溯 PR 数: 41
- 原文档显式引用补充 PR 数: 46
- 当前文档总 PR 数: 87
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-12 | [#18538](https://github.com/sgl-project/sglang/pull/18538) | merged | [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation | `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-12 | [#18544](https://github.com/sgl-project/sglang/pull/18544) | merged | [Ascend]Support qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-18 | [#18926](https://github.com/sgl-project/sglang/pull/18926) | merged | feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-19 | [#18937](https://github.com/sgl-project/sglang/pull/18937) | merged | [Qwen3.5] Enable nvfp4 checkpoint | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-02-25 | [#19070](https://github.com/sgl-project/sglang/pull/19070) | merged | fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-26 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-02-26 | [#19411](https://github.com/sgl-project/sglang/pull/19411) | merged | [Qwen3.5] Qwen3.5-27B inference repeat bug fix | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-04 | [#19391](https://github.com/sgl-project/sglang/pull/19391) | merged | [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4 | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-03-06 | [#19906](https://github.com/sgl-project/sglang/pull/19906) | merged | Add Qwen3.5-397B-A17B nightly test (8-GPU) | `test/registered/8-gpu-models/test_qwen35.py` |
| 2026-03-07 | [#19670](https://github.com/sgl-project/sglang/pull/19670) | merged | [Qwen3.5] Support Qwen3.5 Pipeline Parallelism | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-09 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-03-12 | [#20386](https://github.com/sgl-project/sglang/pull/20386) | merged | perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe… | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-15 | [#20540](https://github.com/sgl-project/sglang/pull/20540) | merged | [CI]: Add CI For HiMambaRadixTree and qwen3.5 | `test/registered/4-gpu-models/test_qwen35_models.py`, `.github/workflows/pr-test.yml` |
| 2026-03-18 | [#19150](https://github.com/sgl-project/sglang/pull/19150) | merged | [NVIDIA] Integrate FlashInfer decode kernel (Blackwell) for Qwen3.5 | `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `test/registered/4-gpu-models/test_qwen35_models.py` |
| 2026-03-18 | [#19889](https://github.com/sgl-project/sglang/pull/19889) | merged | Use TRTLLM allreduce fusion for Qwen 3.5 | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-18 | [#19961](https://github.com/sgl-project/sglang/pull/19961) | merged | fix: change qwen 3.5 linear attention a_log to fp32 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-20 | [#19321](https://github.com/sgl-project/sglang/pull/19321) | merged | [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py` |
| 2026-03-21 | [#21081](https://github.com/sgl-project/sglang/pull/21081) | merged | Fix test_qwen35_models | `test/registered/4-gpu-models/test_qwen35_models.py` |
| 2026-03-21 | [#21070](https://github.com/sgl-project/sglang/pull/21070) | merged | [Qwen3.5] Fix broken pipeline parallelism layer splitting | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-23 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-25 | [#21371](https://github.com/sgl-project/sglang/pull/21371) | merged | [CI] Fix TestQwen35WithHiCache | `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen35_models.py` |
| 2026-03-29 | [#21487](https://github.com/sgl-project/sglang/pull/21487) | merged | feat(ci): add GB300 nightly benchmark test suites | `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py` |
| 2026-03-30 | [#21448](https://github.com/sgl-project/sglang/pull/21448) | merged | [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-30 | [#21234](https://github.com/sgl-project/sglang/pull/21234) | merged | [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model | `python/sglang/srt/models/qwen3_5.py` |
| 2026-03-31 | [#20864](https://github.com/sgl-project/sglang/pull/20864) | merged | [Perf]Remove H2D for Qwen3.5 SpecV2 | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-04-01 | [#21347](https://github.com/sgl-project/sglang/pull/21347) | merged | [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-06 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py` |
| 2026-04-07 | [#21669](https://github.com/sgl-project/sglang/pull/21669) | merged | [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x | `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` |
| 2026-04-07 | [#22145](https://github.com/sgl-project/sglang/pull/22145) | merged | [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-07 | [#22240](https://github.com/sgl-project/sglang/pull/22240) | merged | [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5) | `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-04-08 | [#21692](https://github.com/sgl-project/sglang/pull/21692) | merged | [Bugfix] [NPU] Qwen3.5 with quantization fix | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-09 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | `test/registered/8-gpu-models/test_qwen35.py` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22312](https://github.com/sgl-project/sglang/pull/22312) | merged | Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `test/registered/attention/test_gdn_noncontiguous_stride.py` |
| 2026-04-15 | [#20736](https://github.com/sgl-project/sglang/pull/20736) | merged | [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-16 | [#22948](https://github.com/sgl-project/sglang/pull/22948) | merged | [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled | `python/sglang/srt/models/qwen2_moe.py` |
| 2026-04-17 | [#22913](https://github.com/sgl-project/sglang/pull/22913) | merged | test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6 | `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` |
| 2026-04-17 | [#23034](https://github.com/sgl-project/sglang/pull/23034) | merged | docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs | `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` |
| 2026-04-18 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-21 | [#22908](https://github.com/sgl-project/sglang/pull/22908) | merged | [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict. | `python/sglang/srt/server_args.py` |
| 2026-04-22 | [#22493](https://github.com/sgl-project/sglang/pull/22493) | merged | Add MambaPool kvcache offloading during retraction | `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-22 | [#23467](https://github.com/sgl-project/sglang/pull/23467) | merged | fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert | `python/sglang/srt/layers/quantization/utils.py` |
| 2026-04-26 | [#19484](https://github.com/sgl-project/sglang/pull/19484) | merged | [CPU] Add Qwen3.5 model optimization for CPU | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-27 | [#20918](https://github.com/sgl-project/sglang/pull/20918) | merged | [NPU] Support MTP for Qwen3.5 | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-04-28 | [#23471](https://github.com/sgl-project/sglang/pull/23471) | merged | [Fix] NVFP4 qwen3.5 quant error fix by add packed_modules_mapping | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-29 | [#23815](https://github.com/sgl-project/sglang/pull/23815) | merged | [NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-04-30 | [#23594](https://github.com/sgl-project/sglang/pull/23594) | merged | LoRA support for qwen3.5 and nemotron3 | `python/sglang/srt/models/qwen3_5.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py` |
| 2026-04-30 | [#23062](https://github.com/sgl-project/sglang/pull/23062) | merged | [bugfix]fix(qwen3_5): broadcast per-tensor scale in _make_packed_weight_loader for FP8 models | `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py`, `python/sglang/srt/models/qwen3_5.py` |
| 2026-05-05 | [#23146](https://github.com/sgl-project/sglang/pull/23146) | merged | [AMD] Enable EAGLE speculative decoding for Qwen3.5 FP8 and MXFP4 models with aiter's unified attention | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-05-15 | [#24906](https://github.com/sgl-project/sglang/pull/24906) | merged | Support Qwen3.5 NVFP4 MTP DeepEP | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| 2026-05-18 | [#21668](https://github.com/sgl-project/sglang/pull/21668) | merged | [XPU] Enable qwen3.5 on XPU | `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/chunk.py`, `python/sglang/srt/layers/attention/fla/kda.py` |
| 2026-05-18 | [#25401](https://github.com/sgl-project/sglang/pull/25401) | merged | Add output_gate_type to Qwen3NextConfig and update models to utilize it | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/configs/qwen3_next.py` |
| 2026-05-19 | [#25735](https://github.com/sgl-project/sglang/pull/25735) | merged | [NPU] [DOCS] Improved the usability of Ascend NPU documents | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` |
| 2026-05-19 | [#23331](https://github.com/sgl-project/sglang/pull/23331) | merged | [BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN) | `python/sglang/srt/models/qwen3_5_mtp.py` |
| 2026-05-20 | [#23925](https://github.com/sgl-project/sglang/pull/23925) | merged | [NPU]use triton split_qkvgate_gemma_rmsnorm_rope for Qwen3.5 and Qwen3_next | `python/sglang/srt/models/qwen3_5.py` |
| 2026-05-23 | [#26069](https://github.com/sgl-project/sglang/pull/26069) | merged | [NPU]Ascend NPU Performance Profiling Guide and Ascend NPU Operator Development Guide | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-05-29 | [#26695](https://github.com/sgl-project/sglang/pull/26695) | merged | [docs] Qwen3.5 cookbook: multi-node, MTP TP overrides, dense mamba flag | `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` |
| 2026-05-30 | [#26389](https://github.com/sgl-project/sglang/pull/26389) | merged | 【NPU】【bugfix】fix server error when mtp unquant | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-04 | [#27296](https://github.com/sgl-project/sglang/pull/27296) | merged | Add --enable-symm-mem for Qwen3.5 | `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` |
| 2026-06-05 | [#25885](https://github.com/sgl-project/sglang/pull/25885) | merged | [AMD] Support alt stream for Qwen3.5 on AMD platform | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-06 | [#27248](https://github.com/sgl-project/sglang/pull/27248) | merged | [Doc][CPU]Update Cookbook with Xeon support info | `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` |
| 2026-06-09 | [#27660](https://github.com/sgl-project/sglang/pull/27660) | merged | [AMD] Update amd qwen3.5 cookbook | `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-10 | [#27656](https://github.com/sgl-project/sglang/pull/27656) | merged | [AMD][Perf] Fuse QK RMSNorm + gate extraction Triton kernel for Qwen3.5 on HIP | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-11 | [#27630](https://github.com/sgl-project/sglang/pull/27630) | merged | [AMD] Fuse sigmoid + mul attention output gate into single Triton kernel | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py` |
| 2026-06-11 | [#27846](https://github.com/sgl-project/sglang/pull/27846) | merged | fix: per-sequence last-token embedding in EAGLE3/MTP draft for batched multimodal spec decoding | `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/llama_eagle3.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-12 | [#23862](https://github.com/sgl-project/sglang/pull/23862) | merged | Fix --mem-fraction-static not accounting for EAGLE draft model KV cache | `python/sglang/srt/model_executor/model_runner.py`, `test/registered/unit/configs/test_model_config_shapes.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-06-13 | [#27057](https://github.com/sgl-project/sglang/pull/27057) | merged | [AMD] move shared expert check function to quark | `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-06-13 | [#26924](https://github.com/sgl-project/sglang/pull/26924) | merged | [4/N] Qwen3.5Opt: Overlap mamba verify update with draft extend | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-13 | [#28129](https://github.com/sgl-project/sglang/pull/28129) | merged | [Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py` |
| 2026-06-14 | [#27869](https://github.com/sgl-project/sglang/pull/27869) | merged | Fix Qwen3.5 deterministic batch-invariant logprobs | `test/registered/attention/test_qwen35_deterministic.py` |
| 2026-06-15 | [#27868](https://github.com/sgl-project/sglang/pull/27868) | merged | fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885) | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-16 | [#28293](https://github.com/sgl-project/sglang/pull/28293) | merged | [NPU] Add NPU fallback for fused Triton gating kernels | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-19 | [#28536](https://github.com/sgl-project/sglang/pull/28536) | merged | ci: run GB300 nightly suite in the standard Nvidia nightly workflow | `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py` |
| 2026-06-22 | [#27893](https://github.com/sgl-project/sglang/pull/27893) | merged | [NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` |
| 2026-06-24 | [#27870](https://github.com/sgl-project/sglang/pull/27870) | merged | [qwen3.5][XPU]Add XPU support for set_embed_and_head and fused QK RMSNorm kernel | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-25 | [#28320](https://github.com/sgl-project/sglang/pull/28320) | merged | Fused QK GemmaRMSNorm + RoPE + gate kernel for Qwen3.5 | `python/sglang/srt/models/qwen3_5.py` |
| 2026-06-25 | [#28103](https://github.com/sgl-project/sglang/pull/28103) | merged | Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test | `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml` |
| 2026-06-25 | [#29267](https://github.com/sgl-project/sglang/pull/29267) | merged | [CPU] add indices in chunk_gated_delta_rule | `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`, `python/sglang/srt/model_executor/cpu_graph_runner.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py` |

## 逐 PR diff 审计卡

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- 链接: https://github.com/sgl-project/sglang/pull/18489
- 状态/时间: merged / 2026-02-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `27c447653d9c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1923/-9，可读 patch 2159 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MODEL] Adding Support for Qwen3.5 Models」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`；技术摘要: 覆盖「[MODEL] Adding Support for Qwen3.5 Models」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward，涉及 `Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering`；`python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward，涉及 `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`；`python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config，涉及 `Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -0,0 +1,1310 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -0,0 +1,415 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -0,0 +1,113 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` added +1310/-0; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0; `python/sglang/srt/configs/qwen3_5.py` added +113/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18538 - [Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation

- 链接: https://github.com/sgl-project/sglang/pull/18538
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `4ed2548427a0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+62/-118，可读 patch 275 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Qwen3_5] Refactor `Qwen3_5ForCausalLMMTP` class implementation」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: -24,114 +24,15; -140,7 +41,7 @@ def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward，涉及 `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`；`python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: -330,6 +330,9 @@ def __init__(; -338,15 +341,18 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112 (156 lines); hunks: -24,114 +24,15; -140,7 +41,7 @@ def __init__(; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-6 (24 lines); hunks: -330,6 +330,9 @@ def __init__(; -338,15 +341,18 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -24,114 +24,15 @@
-from sglang.srt.layers.vocab_parallel_embedding import (
-    ParallelLMHead,
-    VocabParallelEmbedding,
-)
+from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
-from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
diff -- python/sglang/srt/models/qwen3_5.py
@@ -330,6 +330,9 @@ def __init__(
+            is_layer_sparse = True
+            is_previous_layer_sparse = True
+            is_next_layer_sparse = True
@@ -338,15 +341,18 @@ def __init__(
+            is_layer_sparse = False
+            is_previous_layer_sparse = False
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +44/-112; `python/sglang/srt/models/qwen3_5.py` modified +18/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18544 - [Ascend]Support qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/18544
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `1edc69be0854`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+23/-4，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend]Support qwen3.5」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Ascend]Support qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -34,6 +34,7; -328,15 +329,15 @@ def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location，涉及 `__init__, load_fused_expert_weights, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -34,6 +34,7; -328,15 +329,15 @@ def __init__(; symbols: __init__, load_fused_expert_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -34,6 +34,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -328,15 +329,15 @@ def __init__(
-                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
+                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
-                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
+                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18926 - feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation

- 链接: https://github.com/sgl-project/sglang/pull/18926
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `fa5698d79164`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+57/-12，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「feat: [Qwen3.5] Support block-wise FP8 quantization and model adaptation」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: -64,7 +64,7 @@ def __init__(; -214,16 +214,11 @@ def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights，涉及 `__init__, load_fused_expert_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6 (7 lines); hunks: -64,7 +64,7 @@ def __init__(; -214,16 +214,11 @@ def load_fused_expert_weights(; symbols: __init__, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -64,7 +64,7 @@ def __init__(
-            prefix=add_prefix("model", prefix),
+            prefix=add_prefix("mtp", prefix),
@@ -214,16 +214,11 @@ def load_fused_expert_weights(
-            # Some checkpoints use model.language_model.mtp.* prefix
-            if "language_model" in name:
-                name = name.replace(r"model.language_model.", r"model.")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18937 - [Qwen3.5] Enable nvfp4 checkpoint

- 链接: https://github.com/sgl-project/sglang/pull/18937
- 状态/时间: merged / 2026-02-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `bba2fc49a170`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+26/-8，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Enable nvfp4 checkpoint」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[Qwen3.5] Enable nvfp4 checkpoint」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: -318,8 +318,14 @@ def __init__(; -458,13 +464,19 @@ def __init__(; symbols: __init__, load_weights, load_fused_expert_weights，涉及 `__init__, load_weights, load_fused_expert_weights`；`python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -48,6 +48,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +19/-7 (26 lines); hunks: -318,8 +318,14 @@ def __init__(; -458,13 +464,19 @@ def __init__(; symbols: __init__, load_weights, load_fused_expert_weights
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -48,6 +48,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -318,8 +318,14 @@ def __init__(
+        linear_attn_quant_config = (
+            None
+            if quant_config and quant_config.get_name() == "modelopt_fp4"
+            else quant_config
+        )
-            config, layer_id, quant_config, alt_stream, prefix
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -48,6 +48,10 @@ def __init__(
+        # The MTP model is unquantized in the nvfp4 checkpoint.
+        if quant_config and quant_config.get_name() == "modelopt_fp4":
+            quant_config = None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +19/-7; `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19070 - fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

- 链接: https://github.com/sgl-project/sglang/pull/19070
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `d38c0e537d95`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-6，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: -400,11 +400,24 @@ def forward(; -633,11 +646,24 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +32/-6 (38 lines); hunks: -400,11 +400,24 @@ def forward(; -633,11 +646,24 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -400,11 +400,24 @@ def forward(
-        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
-        hidden_states, residual = self.layer_communicator.postprocess_layer(
-            hidden_states, residual, forward_batch
+        should_allreduce_fusion = (
+            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
+                forward_batch
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +32/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/19220
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `b3202fe6d072`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+9/-46，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] fix piecewise cuda graph for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[PCG] fix piecewise cuda graph for Qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward，涉及 `forward, _forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -22,9 +22,6 @@
-# Model Executor
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -72,7 +69,6 @@
-from sglang.srt.models.qwen3_next import gdn_with_output
@@ -253,22 +249,6 @@ def forward(
-    ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-21
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19411 - [Qwen3.5] Qwen3.5-27B inference repeat bug fix

- 链接: https://github.com/sgl-project/sglang/pull/19411
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `bdc1e46e5ac9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Qwen3.5-27B inference repeat bug fix」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Qwen3.5] Qwen3.5-27B inference repeat bug fix」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -352,6 +352,7 @@ def __init__(; -542,6 +543,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -352,6 +352,7 @@ def __init__(; -542,6 +543,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -352,6 +352,7 @@ def __init__(
+            is_last_layer=(layer_id == config.num_hidden_layers - 1),
@@ -542,6 +543,7 @@ def __init__(
+            is_last_layer=(layer_id == config.num_hidden_layers - 1),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19391 - [Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/19391
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `9457c049e19e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+252/-16，可读 patch 332 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -111,7 +111,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1 (2 lines); hunks: -111,7 +111,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -111,7 +111,7 @@ def forward(
-            and not forward_batch.forward_mode.is_draft_extend()
+            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19906 - Add Qwen3.5-397B-A17B nightly test (8-GPU)

- 链接: https://github.com/sgl-project/sglang/pull/19906
- 状态/时间: merged / 2026-03-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_qwen35.py`；关联提交 `ac453b253f58`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+74/-0，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Qwen3.5-397B-A17B nightly test (8-GPU)」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_qwen35.py`；技术摘要: 覆盖「Add Qwen3.5-397B-A17B nightly test (8-GPU)」；主要实现面是 `test/registered/8-gpu-models/test_qwen35.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_qwen35.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestQwen35, for, test_qwen35，涉及 `TestQwen35, for, test_qwen35`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_qwen35.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestQwen35, for, test_qwen35
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_qwen35.py
@@ -0,0 +1,74 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_qwen35.py` added +74/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_qwen35.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19670 - [Qwen3.5] Support Qwen3.5 Pipeline Parallelism

- 链接: https://github.com/sgl-project/sglang/pull/19670
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `7da590d4d069`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+114/-13，可读 patch 194 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Support Qwen3.5 Pipeline Parallelism」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Qwen3.5] Support Qwen3.5 Pipeline Parallelism」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: -30,7 +30,7; -59,6 +59,7; symbols: __init__, get_layer, get_input_embeddings，涉及 `__init__, get_layer, get_input_embeddings`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +60/-13 (73 lines); hunks: -30,7 +30,7; -59,6 +59,7; symbols: __init__, get_layer, get_input_embeddings
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -30,7 +30,7 @@
-from sglang.srt.distributed import get_pp_group
+from sglang.srt.distributed import get_pp_group, get_pp_indices
@@ -59,6 +59,7 @@
+from sglang.srt.layers.utils import PPMissingLayer
@@ -680,6 +681,8 @@ def __init__(
+        else:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +60/-13
- 验证与风险: diff 自带测试面 `test/registered/distributed/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- 链接: https://github.com/sgl-project/sglang/pull/19767
- 状态/时间: merged / 2026-03-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `cabe171b6ce3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+79/-16，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix qwen3.5 mtp eplb related issues」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「Fix qwen3.5 mtp eplb related issues」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward，涉及 `__init__, get_model_config_for_expert_location, get_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -72,7 +72,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_npu,
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -22,6 +22,8 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -69,6 +71,7 @@ def __init__(
+            is_nextn=True,
@@ -84,6 +87,15 @@ def __init__(
+    @classmethod
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-1; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20386 - perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

- 链接: https://github.com/sgl-project/sglang/pull/20386
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `9b55a98a6705`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: -20,7 +20,6; -287,7 +286,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-2 (3 lines); hunks: -20,7 +20,6; -287,7 +286,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,7 +20,6 @@
-from einops import rearrange
@@ -287,7 +286,7 @@ def forward(
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20540 - [CI]: Add CI For HiMambaRadixTree and qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/20540
- 状态/时间: merged / 2026-03-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+80/-7，可读 patch 120 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI]: Add CI For HiMambaRadixTree and qwen3.5」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `test/registered/4-gpu-models/test_qwen35_models.py`, `.github/workflows/pr-test.yml`；技术摘要: 覆盖「[CI]: Add CI For HiMambaRadixTree and qwen3.5」；主要实现面是 `test/registered/4-gpu-models/test_qwen35_models.py`, `.github/workflows/pr-test.yml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_qwen35_models.py` modified +78/-5 (83 lines); hunks: -1,3 +1,5; -17,13 +19,11; symbols: TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache, setUpClass，涉及 `TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache`；`.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: -1664,7 +1664,7 @@ jobs:; -1693,7 +1693,7 @@ jobs:。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +78/-5 (83 lines); hunks: -1,3 +1,5; -17,13 +19,11; symbols: TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache, setUpClass
  - `.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: -1664,7 +1664,7 @@ jobs:; -1693,7 +1693,7 @@ jobs:
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -1,3 +1,5 @@
+import shutil
+import tempfile
@@ -17,13 +19,11 @@
-register_cuda_ci(est_time=1000, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=1400, suite="stage-c-test-4-gpu-b200")
-ACC_THRESHOLDS = {
diff -- .github/workflows/pr-test.yml
@@ -1664,7 +1664,7 @@ jobs:
-        part: [0, 1, 2]
+        part: [0, 1, 2, 3]
@@ -1693,7 +1693,7 @@ jobs:
-          IS_BLACKWELL=1 python3 run_suite.py --hw cuda --suite stage-c-test-4-gpu-b200 --auto-partition-id ${{ matrix.part }} --auto-partition-size 3 --timeout-per-file 1800 $CON
+          IS_BLACKWELL=1 python3 run_suite.py --hw cuda --suite stage-c-test-4-gpu-b200 --auto-partition-id ${{ matrix.part }} --auto-partition-size 4 --timeout-per-file 1800 $CON
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_qwen35_models.py` modified +78/-5
  - ci: `.github/workflows/pr-test.yml` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19150 - [NVIDIA] Integrate FlashInfer decode kernel (Blackwell) for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/19150
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+160/-127，可读 patch 492 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Integrate FlashInfer decode kernel (Blackwell) for Qwen3.5」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；技术摘要: 覆盖「[NVIDIA] Integrate FlashInfer decode kernel (Blackwell) for Qwen3.5」；主要实现面是 `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `test/registered/4-gpu-models/test_qwen35_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +66/-71 (137 lines); hunks: -1,16 +1,11; -52,17 +47,12 @@ def _get_flashinfer_gdn_kernels():; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__，涉及 `_get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__`；`python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +3/-2 (5 lines); hunks: -72,7 +72,7 @@ def __init__(; -91,7 +91,7 @@ def __init__(; symbols: __init__, forward_extend，涉及 `__init__, forward_extend`；`test/registered/4-gpu-models/test_qwen35_models.py` modified +53/-53 (106 lines); hunks: -7,15 +7,18; -26,62 +29,59; symbols: TestQwen35FP4, setUpClass, test_gsm8k, tearDownClass，涉及 `TestQwen35FP4, setUpClass, test_gsm8k`；`python/sglang/srt/server_args.py` modified +31/-0 (31 lines); hunks: -767,6 +767,7 @@ def __post_init__(self):; -2031,6 +2032,19 @@ def _handle_mamba_radix_cache(; symbols: __post_init__, _handle_mamba_radix_cache, _handle_mamba_backend, _handle_linear_attn_backend，涉及 `__post_init__, _handle_mamba_radix_cache, _handle_mamba_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +66/-71 (137 lines); hunks: -1,16 +1,11; -52,17 +47,12 @@ def _get_flashinfer_gdn_kernels():; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +3/-2 (5 lines); hunks: -72,7 +72,7 @@ def __init__(; -91,7 +91,7 @@ def __init__(; symbols: __init__, forward_extend
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +53/-53 (106 lines); hunks: -7,15 +7,18; -26,62 +29,59; symbols: TestQwen35FP4, setUpClass, test_gsm8k, tearDownClass
  - `python/sglang/srt/server_args.py` modified +31/-0 (31 lines); hunks: -767,6 +767,7 @@ def __post_init__(self):; -2031,6 +2032,19 @@ def _handle_mamba_radix_cache(; symbols: __post_init__, _handle_mamba_radix_cache, _handle_mamba_backend, _handle_linear_attn_backend
  - `python/sglang/test/accuracy_test_runner.py` modified +7/-1 (8 lines); hunks: -27,6 +27,7 @@ class AccuracyTestParams:; -83,6 +84,7 @@ def _run_simple_eval(; symbols: AccuracyTestParams, _run_simple_eval, run_accuracy_test
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
@@ -1,16 +1,11 @@
-Provides K-last SSM layout support using FlashInfer CUTLASS kernels (SM90+).
-The K-last layout stores SSM states as [pool, HV, V, K] instead of V-last
-[pool, HV, K, V], enabling more efficient memory access patterns for decode.
+Both SM90 and SM100+ use the same pool layout: [pool, HV, V, K] (K-last).
-Requires ``flashinfer`` with GDN kernel support to be installed, e.g.
-``pip install -e ".[cutlass]"`` from the FlashInfer repo.
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -72,7 +72,7 @@ def __init__(
-                raise ValueError("FlashInfer backend requires CUDA")
+                raise ValueError("FlashInfer GDN backend requires CUDA")
@@ -91,7 +91,7 @@ def __init__(
-                raise ValueError("FlashInfer backend requires CUDA")
+                raise ValueError("FlashInfer GDN backend requires CUDA")
@@ -399,6 +399,7 @@ def forward_extend(
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -7,15 +7,18 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +66/-71; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +3/-2; `python/sglang/srt/server_args.py` modified +31/-0
  - tests: `test/registered/4-gpu-models/test_qwen35_models.py` modified +53/-53; `python/sglang/test/accuracy_test_runner.py` modified +7/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/accuracy_test_runner.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19889 - Use TRTLLM allreduce fusion for Qwen 3.5

- 链接: https://github.com/sgl-project/sglang/pull/19889
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+88/-52，可读 patch 210 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use TRTLLM allreduce fusion for Qwen 3.5」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「Use TRTLLM allreduce fusion for Qwen 3.5」；主要实现面是 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: -86,6 +86,53; -303,53 +350,10 @@ def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion，涉及 `_forward_with_allreduce_fusion, RMSNorm, __init__`；`python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -397,7 +397,12 @@ def forward(; -646,7 +651,12 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: -54,7 +54,10; -310,6 +313,7 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1978,6 +1978,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/layernorm.py` modified +63/-48 (111 lines); hunks: -86,6 +86,53; -303,53 +350,10 @@ def forward_with_allreduce_fusion(; symbols: _forward_with_allreduce_fusion, RMSNorm, __init__, forward_with_allreduce_fusion
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-2 (14 lines); hunks: -397,7 +397,12 @@ def forward(; -646,7 +651,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +11/-2 (13 lines); hunks: -54,7 +54,10; -310,6 +313,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -1978,6 +1978,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/layernorm.py
@@ -86,6 +86,53 @@
+def _forward_with_allreduce_fusion(
+    norm_module,
+    x: torch.Tensor,
+    residual: Optional[torch.Tensor],
+    post_residual_addition: Optional[torch.Tensor],
+    weight: torch.Tensor,
diff -- python/sglang/srt/models/qwen3_5.py
@@ -397,7 +397,12 @@ def forward(
-            hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
+            hidden_states = self.mlp(
+                hidden_states,
+                forward_batch,
+                use_reduce_scatter,
+                should_allreduce_fusion,
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -54,7 +54,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/layernorm.py` modified +63/-48; `python/sglang/srt/models/qwen3_5.py` modified +12/-2; `python/sglang/srt/models/qwen2_moe.py` modified +11/-2; `python/sglang/srt/server_args.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19961 - fix: change qwen 3.5 linear attention a_log to fp32

- 链接: https://github.com/sgl-project/sglang/pull/19961
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: change qwen 3.5 linear attention a_log to fp32」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「fix: change qwen 3.5 linear attention a_log to fp32」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -186,7 +186,7 @@ def __init__(
-            torch.empty(self.num_v_heads // self.attn_tp_size),
+            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19321 - [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj

- 链接: https://github.com/sgl-project/sglang/pull/19321
- 状态/时间: merged / 2026-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+107/-17，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py`；技术摘要: 覆盖「[Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader，涉及 `__init__, fix_query_key_value_ordering, _make_packed_weight_loader`；`python/sglang/srt/layers/linear.py` modified +24/-6 (30 lines); hunks: -531,8 +531,15 @@ def weight_loader(; -699,7 +706,10 @@ def weight_loader(; symbols: weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2，涉及 `weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader
  - `python/sglang/srt/layers/linear.py` modified +24/-6 (30 lines); hunks: -531,8 +531,15 @@ def weight_loader(; -699,7 +706,10 @@ def weight_loader(; symbols: weight_loader, _load_fused_module_from_checkpoint, weight_loader_v2
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -20,6 +20,7 @@
+    MergedColumnParallelLinear,
@@ -245,28 +246,38 @@ def __init__(
-        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
-        projection_size_ba = self.num_v_heads * 2
-        self.in_proj_qkvz = ColumnParallelLinear(
-            input_size=self.hidden_size,
diff -- python/sglang/srt/layers/linear.py
@@ -531,8 +531,15 @@ def weight_loader(
-        loaded_shard_id: Optional[int] = None,
+        loaded_shard_id: tuple[int, ...] | int | None = None,
+        if isinstance(loaded_shard_id, tuple):
+            if hasattr(param, "load_merged_column_weight"):
+                return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)
+            raise NotImplementedError(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +83/-11; `python/sglang/srt/layers/linear.py` modified +24/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21081 - Fix test_qwen35_models

- 链接: https://github.com/sgl-project/sglang/pull/21081
- 状态/时间: merged / 2026-03-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-5，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix test_qwen35_models」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `test/registered/4-gpu-models/test_qwen35_models.py`；技术摘要: 覆盖「Fix test_qwen35_models」；主要实现面是 `test/registered/4-gpu-models/test_qwen35_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_qwen35_models.py` modified +6/-5 (11 lines); hunks: -60,11 +60,12 @@ def test_gsm8k(self):; symbols: test_gsm8k，涉及 `test_gsm8k`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +6/-5 (11 lines); hunks: -60,11 +60,12 @@ def test_gsm8k(self):; symbols: test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -60,11 +60,12 @@ def test_gsm8k(self):
-            ModelLaunchSettings(
-                QWEN35_FP4_MODEL,
-                extra_args=base_args + ["--linear-attn-decode-backend", "flashinfer"],
-                variant="FlashInfer",
-            ),
+            # TODO: Fix this and re-enable it
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_qwen35_models.py` modified +6/-5
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21070 - [Qwen3.5] Fix broken pipeline parallelism layer splitting

- 链接: https://github.com/sgl-project/sglang/pull/21070
- 状态/时间: merged / 2026-03-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `852e112ebf00`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-20，可读 patch 94 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Fix broken pipeline parallelism layer splitting」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Qwen3.5] Fix broken pipeline parallelism layer splitting」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: -29,7 +29,7; -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights，涉及 `get_layer, load_fused_expert_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +8/-15 (23 lines); hunks: -29,7 +29,7; -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -29,7 +29,7 @@
-from sglang.srt.distributed import get_pp_group, get_pp_indices
+from sglang.srt.distributed import get_pp_group
@@ -721,25 +721,14 @@ def get_layer(idx: int, prefix: str):
-        self.layers = make_layers(
+        self.layers, self._start_layer, self._end_layer = make_layers(
+            pp_rank=self.pp_group.rank_in_group,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +8/-15
- 验证与风险: diff 自带测试面 `test/registered/distributed/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- 链接: https://github.com/sgl-project/sglang/pull/21019
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `5bdc07d974f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+597/-202，可读 patch 953 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,6 +20,11 @@
+import triton
+from sglang.jit_kernel.triton.gdn_fused_proj import (
+    fused_qkvzba_split_reshape_cat_contiguous,
+)
@@ -54,6 +59,10 @@
+from sglang.srt.layers.parameter import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +285/-65
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21371 - [CI] Fix TestQwen35WithHiCache

- 链接: https://github.com/sgl-project/sglang/pull/21371
- 状态/时间: merged / 2026-03-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+128/-103，可读 patch 249 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix TestQwen35WithHiCache」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；技术摘要: 覆盖「[CI] Fix TestQwen35WithHiCache」；主要实现面是 `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_qwen35_hicache.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: TestQwen35WithHiCache, setUpClass, tearDownClass, _run_gsm8k，涉及 `TestQwen35WithHiCache, setUpClass, tearDownClass`；`test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-103 (104 lines); hunks: -1,6 +1,3; -26,8 +23,7; symbols: TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache, setUpClass，涉及 `TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_qwen35_hicache.py` added +127/-0 (127 lines); hunks: -0,0 +1,127; symbols: TestQwen35WithHiCache, setUpClass, tearDownClass, _run_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-103 (104 lines); hunks: -1,6 +1,3; -26,8 +23,7; symbols: TestQwen35FP4, test_gsm8k, TestQwen35WithHiCache, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_hicache.py
@@ -0,0 +1,127 @@
+import shutil
+import tempfile
+import time
+import unittest
+from types import SimpleNamespace
+import requests
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -1,6 +1,3 @@
-import shutil
-import tempfile
-import time
@@ -26,8 +23,7 @@
-QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
-ACC_THRESHOLDS = {QWEN35_FP4_MODEL: {"gsm8k": 0.95}, QWEN35_27B_MODEL: {"gsm8k": 0.8}}
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_qwen35_hicache.py` added +127/-0; `test/registered/4-gpu-models/test_qwen35_models.py` modified +1/-103
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21487 - feat(ci): add GB300 nightly benchmark test suites

- 链接: https://github.com/sgl-project/sglang/pull/21487
- 状态/时间: merged / 2026-03-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+874/-4，可读 patch 926 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(ci): add GB300 nightly benchmark test suites」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`；技术摘要: 覆盖「feat(ci): add GB300 nightly benchmark test suites」；主要实现面是 `python/sglang/test/accuracy_test_runner.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: -150,6 +150,288 @@ def _run_simple_eval(; -224,13 +506,24 @@ def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval，涉及 `_run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared`；`test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4，涉及 `TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4`；`test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV32, test_deepseek_v32，涉及 `TestDeepseekV32, test_deepseek_v32`；`test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4，涉及 `TestQwen35Nvfp4, test_qwen35_nvfp4`。
- 代码 diff 细节:
  - `python/sglang/test/accuracy_test_runner.py` modified +296/-3 (299 lines); hunks: -150,6 +150,288 @@ def _run_simple_eval(; -224,13 +506,24 @@ def run_accuracy_test(; symbols: _run_simple_eval, _get_nemo_venv, _ensure_nemo_data_prepared, _run_nemo_skills_eval
  - `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: TestDeepseekV32Nvfp4, test_deepseek_v32_nvfp4
  - `test/registered/gb300/test_deepseek_v32.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestDeepseekV32, test_deepseek_v32
  - `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: TestQwen35Nvfp4, test_qwen35_nvfp4
  - `test/registered/gb300/test_qwen35_fp8.py` added +75/-0 (75 lines); hunks: -0,0 +1,75; symbols: TestQwen35Fp8, test_qwen35_fp8
- 关键代码摘录:

```diff
diff -- python/sglang/test/accuracy_test_runner.py
@@ -150,6 +150,288 @@ def _run_simple_eval(
+# Cached uv venv for NeMo Skills (persists across variants within a process).
+_nemo_venv_dir: Optional[str] = None
+_nemo_data_prepared: set = set()
+def _get_nemo_venv() -> Tuple[str, dict]:
+    """Get or create a uv venv with nemo_skills installed.
+    Returns (venv_python_path, env_dict) reusable across calls.
diff -- test/registered/gb300/test_deepseek_v32_nvfp4.py
@@ -0,0 +1,82 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
diff -- test/registered/gb300/test_deepseek_v32.py
@@ -0,0 +1,79 @@
```

- 已读文件:
  - tests: `python/sglang/test/accuracy_test_runner.py` modified +296/-3; `test/registered/gb300/test_deepseek_v32_nvfp4.py` added +82/-0; `test/registered/gb300/test_deepseek_v32.py` added +79/-0; `test/registered/gb300/test_qwen35_nvfp4.py` added +79/-0; `test/registered/gb300/test_qwen35_fp8.py` added +75/-0; `test/registered/gb300/test_glm5_nvfp4.py` added +71/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_deepseek_v32_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21448 - [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode

- 链接: https://github.com/sgl-project/sglang/pull/21448
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `9b4dd274787c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+78/-8，可读 patch 262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding in PP mode」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: -67,7 +67,7; -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, load_fused_expert_weights，涉及 `load_weights, load_fused_expert_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +31/-1 (32 lines); hunks: -67,7 +67,7; -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -67,7 +67,7 @@
-from sglang.srt.layers.utils import PPMissingLayer
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -1038,6 +1038,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            layer_id = get_layer_id(name)
+            if (
+                layer_id is not None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +31/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/mem_cache/test_mamba_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21234 - [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model

- 链接: https://github.com/sgl-project/sglang/pull/21234
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `e6071e60c097`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-0，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: -88,6 +88,7; -98,6 +99,7; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights，涉及 `forward, Qwen3_5ForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-0 (18 lines); hunks: -88,6 +88,7; -98,6 +99,7; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -88,6 +88,7 @@
+    is_gfx95_supported,
@@ -98,6 +99,7 @@
+_is_gfx95 = is_gfx95_supported()
@@ -879,6 +881,14 @@ def forward(
+    if _is_gfx95:
+        packed_modules_mapping = {
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +18/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20864 - [Perf]Remove H2D for Qwen3.5 SpecV2

- 链接: https://github.com/sgl-project/sglang/pull/20864
- 状态/时间: merged / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-13，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf]Remove H2D for Qwen3.5 SpecV2」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；技术摘要: 覆盖「[Perf]Remove H2D for Qwen3.5 SpecV2」；主要实现面是 `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: -715,15 +715,21 @@ def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions，涉及 `_compute_spec_mrope_positions`；`python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: -234,14 +234,12 @@ def prepare_for_v2_verify(; symbols: prepare_for_v2_verify，涉及 `prepare_for_v2_verify`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8 (22 lines); hunks: -715,15 +715,21 @@ def _compute_spec_mrope_positions(; symbols: _compute_spec_mrope_positions
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5 (8 lines); hunks: -234,14 +234,12 @@ def prepare_for_v2_verify(; symbols: prepare_for_v2_verify
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -715,15 +715,21 @@ def _compute_spec_mrope_positions(
-            mrope_deltas = [
-                (
-                    torch.tensor([0], dtype=torch.int64)
-                    if mm_inputs[i] is None
-                    else mm_inputs[i].mrope_position_delta.squeeze(0)
+            # Split text-only and mixed batches here because SpecV2 text-only batches can avoid an extra D2H.
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -234,14 +234,12 @@ def prepare_for_v2_verify(
-                batch.mamba_track_indices = torch.tensor(
+                batch.mamba_track_indices = torch.stack(
-                    ],
-                    dtype=torch.int64,
-                    device=device,
-                )
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/forward_batch_info.py` modified +14/-8; `python/sglang/srt/speculative/eagle_info_v2.py` modified +3/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21347 - [Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model

- 链接: https://github.com/sgl-project/sglang/pull/21347
- 状态/时间: merged / 2026-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `2861596fc683`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-0，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1549,6 +1560,17 @@ def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights，涉及 `load_weights, load_fused_expert_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +22/-0 (22 lines); hunks: -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; -1549,6 +1560,17 @@ def load_fused_expert_weights(; symbols: load_weights, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -1384,6 +1384,17 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if (
+                self.config.tie_word_embeddings
+                and self.pp_group.is_last_rank
+                and "model.embed_tokens.weight" in name
+            ):
+                if "lm_head.weight" in params_dict:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +22/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- 链接: https://github.com/sgl-project/sglang/pull/21849
- 状态/时间: merged / 2026-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+190/-3，可读 patch 230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM]: allow Qwen3.5 models for encoder disaggregation」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`；技术摘要: 覆盖「[VLM]: allow Qwen3.5 models for encoder disaggregation」；主要实现面是 `python/sglang/srt/multimodal/processors/qwen_vl.py`, `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #21669 - [AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21669
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`；关联提交 `ba78f6e0efb9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+408/-8，可读 patch 538 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`；技术摘要: 覆盖「[AMD] Add Qwen3.5-397B FP8 nightly perf benchmarks for MI30x and MI35x」；主要实现面是 `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8，涉及 `generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass`；`test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf，涉及 `generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass`；`test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: -8,14 +8,20; -38,7 +44,7 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_lm_eval，涉及 `setUpClass, tearDownClass, test_lm_eval`；`test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: -8,16 +8,21; -40,12 +45,12 @@ def setUpClass(cls):; symbols: setUpClass, test_lm_eval，涉及 `setUpClass, test_lm_eval`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestNightlyQwen35Fp8Performance, setUpClass, test_bench_qwen35_fp8
  - `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0 (139 lines); hunks: -0,0 +1,139; symbols: generate_simple_markdown_report, TestQwen35Fp8PerfMI35x, setUpClass, test_qwen35_fp8_perf
  - `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1 (43 lines); hunks: -8,14 +8,20; -38,7 +44,7 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_lm_eval
  - `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3 (39 lines); hunks: -8,16 +8,21; -40,12 +45,12 @@ def setUpClass(cls):; symbols: setUpClass, test_lm_eval
- 关键代码摘录:

```diff
diff -- test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py
@@ -0,0 +1,139 @@
+"""Nightly performance benchmark for Qwen3.5-397B-A17B FP8.
+Tests Qwen3.5-397B-A17B-FP8 (MoE, Hybrid Attention with Gated Delta Networks)
+on 8 GPUs with triton attention backend.
+Model path can be configured via environment variable:
+- QWEN35_FP8_MODEL_PATH: Path to Qwen3.5-FP8 model
+  (default: Qwen/Qwen3.5-397B-A17B-FP8)
diff -- test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py
@@ -0,0 +1,139 @@
+"""MI35x Nightly performance benchmark for Qwen3.5-397B-A17B FP8.
+Tests Qwen3.5-397B-A17B-FP8 (MoE, Hybrid Attention with Gated Delta Networks)
+on 8 GPUs with triton attention backend.
+Registry: nightly-perf-8-gpu-mi35x-qwen35-fp8 suite
+"""
+import os
diff -- test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py
@@ -8,14 +8,20 @@
```

- 已读文件:
  - tests: `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py` added +139/-0; `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` added +139/-0; `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py` modified +42/-1; `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py` modified +36/-3
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_qwen35_fp8_perf_amd.py`, `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22145 - [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)

- 链接: https://github.com/sgl-project/sglang/pull/22145
- 状态/时间: merged / 2026-04-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-8，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/disaggregation/nixl/conn.py`；技术摘要: 覆盖「[Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)」；主要实现面是 `python/sglang/srt/disaggregation/nixl/conn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: -477,25 +477,35 @@ def send_kvcache_slice(; -748,7 +758,9 @@ def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request，涉及 `send_kvcache_slice, add_transfer_request`。
- 代码 diff 细节:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8 (28 lines); hunks: -477,25 +477,35 @@ def send_kvcache_slice(; -748,7 +758,9 @@ def add_transfer_request(; symbols: send_kvcache_slice, add_transfer_request
- 关键代码摘录:

```diff
diff -- python/sglang/srt/disaggregation/nixl/conn.py
@@ -477,25 +477,35 @@ def send_kvcache_slice(
-        num_kv_heads = self.kv_args.kv_head_num
-        # Calculate head distribution
-        src_heads_per_rank = num_kv_heads
-        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size
+        # Use total KV head count (not per-rank) for correct head distribution.
+        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
```

- 已读文件:
  - runtime: `python/sglang/srt/disaggregation/nixl/conn.py` modified +20/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/nixl/conn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22240 - [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)

- 链接: https://github.com/sgl-project/sglang/pull/22240
- 状态/时间: merged / 2026-04-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+143/-2，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/disaggregation/nixl/conn.py`；技术摘要: 覆盖「[Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)」；主要实现面是 `python/sglang/srt/disaggregation/nixl/conn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: -84,6 +84,8 @@ class KVArgsRegisterInfo:; -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo, from_zmq, _send_mamba_state，涉及 `KVArgsRegisterInfo, from_zmq, _send_mamba_state`。
- 代码 diff 细节:
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2 (145 lines); hunks: -84,6 +84,8 @@ class KVArgsRegisterInfo:; -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):; symbols: KVArgsRegisterInfo, from_zmq, _send_mamba_state
- 关键代码摘录:

```diff
diff -- python/sglang/srt/disaggregation/nixl/conn.py
@@ -84,6 +84,8 @@ class KVArgsRegisterInfo:
+    dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
+    dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)
@@ -93,6 +95,15 @@ def from_zmq(cls, msg: List[bytes]):
+        dst_state_item_lens = []
+        dst_state_dim_per_tensor = []
+        if len(msg) > 12 and len(msg[12]) > 0:
```

- 已读文件:
  - runtime: `python/sglang/srt/disaggregation/nixl/conn.py` modified +143/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/nixl/conn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21692 - [Bugfix] [NPU] Qwen3.5 with quantization fix

- 链接: https://github.com/sgl-project/sglang/pull/21692
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `cd373667cdfa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+29/-42，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [NPU] Qwen3.5 with quantization fix」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Bugfix] [NPU] Qwen3.5 with quantization fix」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: -881,7 +881,7 @@ def forward(; -1310,7 +1310,7 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration，涉及 `forward, Qwen3_5ForCausalLM, load_fused_expert_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-3 (6 lines); hunks: -881,7 +881,7 @@ def forward(; -1310,7 +1310,7 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, load_fused_expert_weights, Qwen3_5ForConditionalGeneration
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -881,7 +881,7 @@ def forward(
-    if _is_gfx95:
+    if _is_gfx95 or _is_npu:
@@ -1310,7 +1310,7 @@ def load_fused_expert_weights(
-    if _is_gfx95:
+    if _is_gfx95 or _is_npu:
@@ -1447,7 +1447,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- 链接: https://github.com/sgl-project/sglang/pull/22399
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_qwen35.py`；关联提交 `46c2b7762765`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+82/-6，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_qwen35.py`；技术摘要: 覆盖「[CI] Add GLM-5.1 nightly tests and update Qwen3.5 model」；主要实现面是 `test/registered/8-gpu-models/test_qwen35.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: -9,7 +9,7; -30,6 +30,7 @@ def test_qwen35(self):; symbols: TestQwen35, test_qwen35，涉及 `TestQwen35, test_qwen35`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: -9,7 +9,7; -30,6 +30,7 @@ def test_qwen35(self):; symbols: TestQwen35, test_qwen35
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_qwen35.py
@@ -9,7 +9,7 @@
-QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
+QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
@@ -30,6 +30,7 @@ def test_qwen35(self):
+        dp_args = ["--dp=8", "--enable-dp-attention"]
@@ -48,8 +49,14 @@ def test_qwen35(self):
-                extra_args=base_args + mtp_args,
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22358 - Enable DFLASH support for additional model backends

- 链接: https://github.com/sgl-project/sglang/pull/22358
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+152/-5，可读 patch 299 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DFLASH support for additional model backends」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Enable DFLASH support for additional model backends」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture，涉及 `forward, get_layer, get_input_embeddings`；`python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`；`python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture，涉及 `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22312 - Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

- 链接: https://github.com/sgl-project/sglang/pull/22312
- 状态/时间: merged / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+272/-8，可读 patch 346 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `test/registered/attention/test_gdn_noncontiguous_stride.py`；技术摘要: 覆盖「Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B」；主要实现面是 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `test/registered/attention/test_gdn_noncontiguous_stride.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update，涉及 `fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update`；`python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: -16,6 +16,8 @@ def fused_gdn_gating_kernel(; -26,8 +28,8 @@ def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating，涉及 `fused_gdn_gating_kernel, fused_gdn_gating`；`test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small，涉及 `_make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6 (15 lines); hunks: -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2 (10 lines); hunks: -16,6 +16,8 @@ def fused_gdn_gating_kernel(; -26,8 +28,8 @@ def fused_gdn_gating_kernel(; symbols: fused_gdn_gating_kernel, fused_gdn_gating
  - `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: _make_noncontiguous_ab, TestFusedGdnGatingNonContiguous, _run_test, test_small
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -30,6 +30,7 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
+    stride_a,
@@ -81,10 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
-        p_a = a + (bos * HV + i_hv) * K + o_k
+        p_a = a + bos * stride_a + i_hv * K + o_k
-        p_a = a + bos * HV + i_hv
+        p_a = a + bos * stride_a + i_hv
diff -- python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
@@ -16,6 +16,8 @@ def fused_gdn_gating_kernel(
+    stride_a,
+    stride_b,
@@ -26,8 +28,8 @@ def fused_gdn_gating_kernel(
-    blk_a = tl.load(a + off, mask=mask)
-    blk_b = tl.load(b + off, mask=mask)
+    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
diff -- test/registered/attention/test_gdn_noncontiguous_stride.py
@@ -0,0 +1,255 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +9/-6; `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` modified +8/-2
  - tests: `test/registered/attention/test_gdn_noncontiguous_stride.py` added +255/-0
- 验证与风险: diff 自带测试面 `test/registered/attention/test_gdn_noncontiguous_stride.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20736 - [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8

- 链接: https://github.com/sgl-project/sglang/pull/20736
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `ea05ea5abed1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+218/-8，可读 patch 383 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: -86,9 +86,11; -100,6 +102,8; symbols: __init__, _get_num_fused_shared_experts, get_embed_and_head，涉及 `__init__, _get_num_fused_shared_experts, get_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +110/-3 (113 lines); hunks: -86,9 +86,11; -100,6 +102,8; symbols: __init__, _get_num_fused_shared_experts, get_embed_and_head
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -86,9 +86,11 @@
+    get_bool_env_var,
+    is_hip,
@@ -100,6 +102,8 @@
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
@@ -528,6 +532,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +110/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22948 - [AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled

- 链接: https://github.com/sgl-project/sglang/pull/22948
- 状态/时间: merged / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-1，可读 patch 39 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「[AMD] Qwen3.5 MXFP4 breaks after shared expert fusion is enabled」；主要实现面是 `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: -108,6 +108,7; -120,6 +121,20 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, __init__，涉及 `can_fuse_shared_expert, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_moe.py` modified +17/-1 (18 lines); hunks: -108,6 +108,7; -120,6 +121,20 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -108,6 +108,7 @@
+    quant_config: Optional[QuantizationConfig],
@@ -120,6 +121,20 @@ def can_fuse_shared_expert(
+    # If the shared expert is excluded from quantization (stored as FP32 in the
+    # checkpoint), fusing it into the quantized MoE weight tensor requires online
+    # quantization which is not supported. Disable fusion in this case.
+    if quant_config is not None:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +17/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22913 - test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6

- 链接: https://github.com/sgl-project/sglang/pull/22913
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+184/-247，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`；技术摘要: 覆盖「test(4-gpu-b200): split test_qwen35_models.py + bump partitions 5→6」；主要实现面是 `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass，涉及 `TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP`；`test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k，涉及 `TestQwen35FP4MTPV2, setUpClass, tearDownClass`；`test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestQwen35FP4, test_gsm8k，涉及 `TestQwen35FP4, test_gsm8k`；`.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: -1232,7 +1232,7 @@ jobs:; -1263,7 +1263,7 @@ jobs:。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestQwen35FP4, test_gsm8k
  - `.github/workflows/pr-test.yml` modified +2/-2 (4 lines); hunks: -1232,7 +1232,7 @@ jobs:; -1263,7 +1263,7 @@ jobs:
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -1,245 +0,0 @@
-import unittest
-from types import SimpleNamespace
-import requests
-from sglang.srt.environ import envs
-from sglang.srt.utils import kill_process_tree
-from sglang.test.accuracy_test_runner import AccuracyTestParams
diff -- test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py
@@ -0,0 +1,105 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.environ import envs
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
diff -- test/registered/4-gpu-models/test_qwen35_fp4_triton.py
@@ -0,0 +1,77 @@
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-245; `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` added +105/-0; `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` added +77/-0
  - ci: `.github/workflows/pr-test.yml` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23034 - docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs

- 链接: https://github.com/sgl-project/sglang/pull/23034
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 73 个文件，+2214/-215，可读 patch 3198 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`；技术摘要: 覆盖「docs: fix links, add Qwen3.6, update Qwen3.5/GLM-5 docs」；主要实现面是 `docs_new/docs/advanced_features/separate_reasoning.mdx`, `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")；`docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that，涉及 `NewModelDetector, that`；`docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509；`docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471。
- 代码 diff 细节:
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3 (5 lines); hunks: -207,7 +207,7 @@ print_highlight("==== Text ===="); -226,7 +226,7 @@ print_highlight("==== Original Output ====")
  - `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2 (3 lines); hunks: -718,7 +718,7 @@ for tool_call in tool_calls:; -738,4 +738,3 @@ terminate_process(server_process); symbols: NewModelDetector, that
  - `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0 (509 lines); hunks: -0,0 +1,509
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0 (471 lines); hunks: -0,0 +1,471
  - `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0 (299 lines); hunks: -0,0 +1,299; symbols: per_token_group_quant_8bit, add
- 关键代码摘录:

```diff
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -207,7 +207,7 @@ print_highlight("==== Text ====")
-The reasoning separation is enable by default when specify .
+The reasoning separation is enable by default when specify .
@@ -226,7 +226,7 @@ print_highlight("==== Original Output ====")
-### SGLang Native API
+### SGLang Native API
@@ -315,4 +315,3 @@ llm.shutdown()
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -718,7 +718,7 @@ for tool_call in tool_calls:
-> **Note:**
+> **Note:**
@@ -738,4 +738,3 @@ terminate_process(server_process)
diff -- docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx
@@ -0,0 +1,509 @@
+---
+title: "DP, DPA and SGLang DP Router"
+metatags:
```

- 已读文件:
  - docs: `docs_new/docs/advanced_features/separate_reasoning.mdx` modified +2/-3; `docs_new/docs/advanced_features/tool_parser.mdx` modified +1/-2; `docs_new/docs/advanced_features/dp_dpa_smg_guide.mdx` added +509/-0; `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx` added +471/-0; `docs_new/docs/advanced_features/piecewise_cuda_graph.mdx` added +299/-0; `docs_new/docs/advanced_features/server_arguments.mdx` modified +241/-45
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/.gitignore`, `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- 链接: https://github.com/sgl-project/sglang/pull/22431
- 状态/时间: merged / 2026-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Qwen3.5 video processing when passing video_data in "processor_output" format」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`；技术摘要: 覆盖「Fix Qwen3.5 video processing when passing video_data in "processor_output" format」；主要实现面是 `python/sglang/srt/multimodal/processors/qwen_vl.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #22908 - [AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.

- 链接: https://github.com/sgl-project/sglang/pull/22908
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-4，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[AMD] Resolve Qwen3.5 MTP (speculative decoding) radix cache conflict.」；主要实现面是 `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend，涉及 `_handle_mamba_radix_cache, _handle_sampling_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/server_args.py` modified +14/-4 (18 lines); hunks: -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, _handle_sampling_backend
- 关键代码摘录:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2326,10 +2326,20 @@ def _handle_mamba_radix_cache(
-                    raise ValueError(
-                        f"Speculative decoding for {model_arch} is not compatible with radix cache when using --mamba-scheduler-strategy no_buffer."
-                        "To use radix cache with speculative decoding, please use --mamba-scheduler-strategy extra_buffer and set SGLANG_ENABLE_SPEC_V2=1."
-                    )
+                    if is_hip():
+                        # On ROCm, extra_buffer is unsupported.
```

- 已读文件:
  - runtime: `python/sglang/srt/server_args.py` modified +14/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22493 - Add MambaPool kvcache offloading during retraction

- 链接: https://github.com/sgl-project/sglang/pull/22493
- 状态/时间: merged / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+193/-16，可读 patch 311 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add MambaPool kvcache offloading during retraction」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`；技术摘要: 覆盖「Add MambaPool kvcache offloading during retraction」；主要实现面是 `test/registered/unit/mem_cache/test_mamba_unittest.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: -388,6 +388,129 @@ def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len，涉及 `make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload`；`python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; -728,10 +750,10 @@ def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos，涉及 `fork_from, get_cpu_copy, load_cpu_copy`；`python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):; -512,8 +512,8 @@ def clear(self):; symbols: free, get_cpu_copy, load_cpu_copy，涉及 `free, get_cpu_copy, load_cpu_copy`；`python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: -2681,11 +2681,20 @@ def update_running_batch(self, batch: ScheduleBatch) ->...; -2715,6 +2724,8 @@ def update_running_batch(self, batch: ScheduleBatch) -> Op...; symbols: update_running_batch，涉及 `update_running_batch`。
- 代码 diff 细节:
  - `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0 (123 lines); hunks: -388,6 +388,129 @@ def make_dummy_req():; symbols: make_dummy_req, test_mamba_pool_cpu_offload, test_hybrid_kv_pool_cpu_offload, test_insert_prev_prefix_len
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6 (49 lines); hunks: -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; -728,10 +750,10 @@ def set_kv_buffer(; symbols: fork_from, get_cpu_copy, load_cpu_copy, get_contiguous_buf_infos
  - `python/sglang/srt/mem_cache/allocator.py` modified +8/-8 (16 lines); hunks: -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):; -512,8 +512,8 @@ def clear(self):; symbols: free, get_cpu_copy, load_cpu_copy
  - `python/sglang/srt/managers/scheduler.py` modified +11/-0 (11 lines); hunks: -2681,11 +2681,20 @@ def update_running_batch(self, batch: ScheduleBatch) ->...; -2715,6 +2724,8 @@ def update_running_batch(self, batch: ScheduleBatch) -> Op...; symbols: update_running_batch
  - `python/sglang/srt/managers/schedule_batch.py` modified +8/-2 (10 lines); hunks: -1241,13 +1241,19 @@ def offload_kv_cache(self, req_to_token_pool, token_to_k...; symbols: offload_kv_cache, load_kv_cache, log_time_stats
- 关键代码摘录:

```diff
diff -- test/registered/unit/mem_cache/test_mamba_unittest.py
@@ -388,6 +388,129 @@ def make_dummy_req():
+    def test_mamba_pool_cpu_offload(self):
+        """MambaPool.get_cpu_copy / load_cpu_copy round-trips conv and temporal state."""
+        _, _, req_to_token_pool, _ = self._setup_tree_and_allocator()
+        mamba_pool = req_to_token_pool.mamba_pool
+        n = 3
+        indices = mamba_pool.alloc(n)
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -388,6 +388,28 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
+    def get_cpu_copy(self, indices, **kwargs):
+        torch.cuda.synchronize()
+        conv_cpu = [
+            conv[:, indices].to("cpu", non_blocking=True)
+            for conv in self.mamba_cache.conv
+        ]
diff -- python/sglang/srt/mem_cache/allocator.py
@@ -164,11 +164,11 @@ def free(self, free_index: torch.Tensor):
```

- 已读文件:
  - tests: `test/registered/unit/mem_cache/test_mamba_unittest.py` modified +123/-0
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +43/-6; `python/sglang/srt/mem_cache/allocator.py` modified +8/-8; `python/sglang/srt/managers/scheduler.py` modified +11/-0; `python/sglang/srt/managers/schedule_batch.py` modified +8/-2
- 验证与风险: diff 自带测试面 `test/registered/unit/mem_cache/test_mamba_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- 链接: https://github.com/sgl-project/sglang/pull/23474
- 状态/时间: open / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+284/-8，可读 patch 330 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`；技术摘要: 覆盖「[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models」；主要实现面是 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent，涉及 `_TiedChild, __init__, forward`；`python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward，涉及 `maybe_offload_to_cpu, forward`。
- 代码 diff 细节:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- 关键代码摘录:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- 已读文件:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- 验证与风险: diff 自带测试面 `test/registered/unit/utils/test_offloader_tied_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23467 - fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert

- 链接: https://github.com/sgl-project/sglang/pull/23467
- 状态/时间: merged / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+31/-4，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/utils.py`；技术摘要: 覆盖「fix: dot-boundary match in is_layer_skipped for FP8 modules_to_not_convert」；主要实现面是 `python/sglang/srt/layers/quantization/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped，涉及 `__getattr__, _module_path_match, is_layer_skipped`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/utils.py` modified +31/-4 (35 lines); hunks: -43,6 +43,28 @@ def __getattr__(self, name):; -56,16 +78,19 @@ def is_layer_skipped(; symbols: __getattr__, _module_path_match, is_layer_skipped
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -43,6 +43,28 @@ def __getattr__(self, name):
+def _module_path_match(ignored: str, prefix: str) -> bool:
+    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
+    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
+    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
+    # that collide with fused dense MLP names by plain substring.
+    if ignored == prefix:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/utils.py` modified +31/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19484 - [CPU] Add Qwen3.5 model optimization for CPU

- 链接: https://github.com/sgl-project/sglang/pull/19484
- 状态/时间: merged / 2026-04-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `10fd0faccd85`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+768/-209，可读 patch 1454 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Add Qwen3.5 model optimization for CPU」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[CPU] Add Qwen3.5 model optimization for CPU」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights，涉及 `__init__, weight_loader, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -124,8 +124,16 @@ def __init__(
-        self.num_v_heads = config.linear_num_value_heads
-        self.num_k_heads = config.linear_num_key_heads
+        self.num_v_heads = (
+            config.linear_num_value_heads
+            if not _is_cpu
+            else config.linear_num_value_heads_cpu
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +37/-4
- 验证与风险: diff 自带测试面 `test/srt/cpu/test_mamba.py`, `test/srt/cpu/test_qwen3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20918 - [NPU] Support MTP for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/20918
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `32c3513816b0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+809/-10，可读 patch 963 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support MTP for Qwen3.5」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[NPU] Support MTP for Qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +23/-1 (24 lines); hunks: -15,13 +15,15; -31,7 +33,8; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +23/-1 (24 lines); hunks: -15,13 +15,15; -31,7 +33,8; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -15,13 +15,15 @@
+from contextlib import ExitStack
+from sglang.srt.environ import envs
@@ -31,7 +33,8 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.server_args import get_global_server_args
+from sglang.srt.utils import add_prefix, is_npu
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +23/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23471 - [Fix] NVFP4 qwen3.5 quant error fix by add packed_modules_mapping

- 链接: https://github.com/sgl-project/sglang/pull/23471
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `9814cc89ce03`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-13，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] NVFP4 qwen3.5 quant error fix by add packed_modules_mapping」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[Fix] NVFP4 qwen3.5 quant error fix by add packed_modules_mapping」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +11/-13 (24 lines); hunks: -900,13 +900,12 @@ def forward(; -1345,9 +1344,9 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights，涉及 `forward, Qwen3_5ForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +11/-13 (24 lines); hunks: -900,13 +900,12 @@ def forward(; -1345,9 +1344,9 @@ def load_fused_expert_weights(; symbols: forward, Qwen3_5ForCausalLM, __init__, load_fused_expert_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -900,13 +900,12 @@ def forward(
-    if _is_gfx95 or _is_npu:
-        packed_modules_mapping = {
-            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
-            "gate_up_proj": ["gate_proj", "up_proj"],
-            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
-            "in_proj_ba": ["in_proj_b", "in_proj_a"],
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +11/-13
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23815 - [NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/23815
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `08699bb1b2d3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-2，可读 patch 37 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +7/-1 (8 lines); hunks: -464,7 +464,11 @@ def forward(; -488,6 +492,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +7/-1 (8 lines); hunks: -464,7 +464,11 @@ def forward(; -488,6 +492,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -464,7 +464,11 @@ def forward(
-        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
+        if (
+            self.num_v_heads // self.num_k_heads in [1, 2, 4]
+            and not _is_cpu
+            and not _is_npu
+        ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23594 - LoRA support for qwen3.5 and nemotron3

- 链接: https://github.com/sgl-project/sglang/pull/23594
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`；关联提交 `c8c1c9261d72`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+1131/-127，可读 patch 1734 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「LoRA support for qwen3.5 and nemotron3」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`；技术摘要: 覆盖「LoRA support for qwen3.5 and nemotron3」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +76/-0 (76 lines); hunks: -945,6 +945,65 @@ class Qwen3_5ForCausalLM(nn.Module):; -1386,6 +1445,8 @@ class Qwen3_5ForConditionalGeneration(Qwen3VLForConditiona...; symbols: Qwen3_5ForCausalLM, get_hidden_dim, __init__, Qwen3_5ForConditionalGeneration，涉及 `Qwen3_5ForCausalLM, get_hidden_dim, __init__`；`test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_35B_A3B_LogprobDiff, test_lora_qwen3_5_35b_a3b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_35B_A3B_LogprobDiff`；`test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_4BLogprobDiff, test_lora_qwen3_5_4b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_4BLogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +76/-0 (76 lines); hunks: -945,6 +945,65 @@ class Qwen3_5ForCausalLM(nn.Module):; -1386,6 +1445,8 @@ class Qwen3_5ForConditionalGeneration(Qwen3VLForConditiona...; symbols: Qwen3_5ForCausalLM, get_hidden_dim, __init__, Qwen3_5ForConditionalGeneration
  - `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_35B_A3B_LogprobDiff, test_lora_qwen3_5_35b_a3b_logprob_accuracy
  - `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: kl_v2, get_prompt_logprobs, TestLoRAQwen3_5_4BLogprobDiff, test_lora_qwen3_5_4b_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -945,6 +945,65 @@ class Qwen3_5ForCausalLM(nn.Module):
+    supported_lora_modules = [
+        "qkv_proj",
+        "o_proj",
+        "out_proj",
+        "in_proj_qkvz",
+        "gate_up_proj",
diff -- test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py
@@ -0,0 +1,157 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py
@@ -0,0 +1,146 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +76/-0
  - tests: `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py` added +157/-0; `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py` added +146/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_chunked_sgmv_backend.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23062 - [bugfix]fix(qwen3_5): broadcast per-tensor scale in _make_packed_weight_loader for FP8 models

- 链接: https://github.com/sgl-project/sglang/pull/23062
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`, `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py`；关联提交 `936c9c235596`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+220/-7，可读 patch 235 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix]fix(qwen3_5): broadcast per-tensor scale in _make_packed_weight_loader for FP8 models」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py`, `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[bugfix]fix(qwen3_5): broadcast per-tensor scale in _make_packed_weight_loader for FP8 models」；主要实现面是 `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py`, `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: _make_mock_module, _make_per_tensor_scale_param, TestMakePackedWeightLoader, test_scalar_weight_broadcast，涉及 `_make_mock_module, _make_per_tensor_scale_param, TestMakePackedWeightLoader`；`python/sglang/srt/models/qwen3_5.py` modified +4/-7 (11 lines); hunks: -320,13 +320,10 @@ def weight_loader(param, loaded_weight, loaded_shard_id=No...; symbols: weight_loader，涉及 `weight_loader`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py` added +216/-0 (216 lines); hunks: -0,0 +1,216; symbols: _make_mock_module, _make_per_tensor_scale_param, TestMakePackedWeightLoader, test_scalar_weight_broadcast
  - `python/sglang/srt/models/qwen3_5.py` modified +4/-7 (11 lines); hunks: -320,13 +320,10 @@ def weight_loader(param, loaded_weight, loaded_shard_id=No...; symbols: weight_loader
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_qwen3_5_packed_weight_loader.py
@@ -0,0 +1,216 @@
+"""
+Unit tests for Qwen3_5GatedDeltaNet._make_packed_weight_loader.
+Validates that per-tensor FP8 scales (scalar or single-element tensors)
+are broadcast to every logical shard, while normal multi-element weights
+are split correctly.
+Regression test for https://github.com/sgl-project/sglang/issues/23051
diff -- python/sglang/srt/models/qwen3_5.py
@@ -320,13 +320,10 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):
-                if len(loaded_weight.shape) == 0:
-                    # Scalar only makes sense for a single logical shard.
-                    assert len(split_sizes) == 1 and split_sizes[0] == 1, (
-                        f"Unexpected scalar for tuple shard load: "
-                        f"{loaded_shard_id=}, {split_sizes=}"
-                    )
```

- 已读文件:
  - tests: `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py` added +216/-0
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +4/-7
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_qwen3_5_packed_weight_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23146 - [AMD] Enable EAGLE speculative decoding for Qwen3.5 FP8 and MXFP4 models with aiter's unified attention

- 链接: https://github.com/sgl-project/sglang/pull/23146
- 状态/时间: merged / 2026-05-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `c2db19ffa40e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+588/-148，可读 patch 964 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable EAGLE speculative decoding for Qwen3.5 FP8 and MXFP4 models with aiter's unified attention」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[AMD] Enable EAGLE speculative decoding for Qwen3.5 FP8 and MXFP4 models with aiter's unified attention」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +12/-0 (12 lines); hunks: -62,6 +62,18 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +12/-0 (12 lines); hunks: -62,6 +62,18 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -62,6 +62,18 @@ def __init__(
+        # Quark-quantized Qwen3.5 MXFP4 checkpoints ship the MTP module in
+        # bf16; every `mtp.*` layer appears under the quantization exclude
+        # list. Detect that and skip quantization here so linear/MoE weight
+        # loaders allocate bf16 shapes (see sgl-project/sglang#23113).
+        if quant_config and quant_config.get_name() == "quark":
+            exclude_layers = getattr(quant_config, "exclude_layers", [])
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/triton_ops/aiter_unified_attention.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24906 - Support Qwen3.5 NVFP4 MTP DeepEP

- 链接: https://github.com/sgl-project/sglang/pull/24906
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+58/-5，可读 patch 137 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Qwen3.5 NVFP4 MTP DeepEP」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`；技术摘要: 覆盖「Support Qwen3.5 NVFP4 MTP DeepEP」；主要实现面是 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +47/-2 (49 lines); hunks: -4,6 +4,7; -137,17 +138,23 @@ def __init__(; symbols: __init__, run_moe_core, forward_aiter, forward_unquantized_deepep_ll，涉及 `__init__, run_moe_core, forward_aiter`；`python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +6/-2 (8 lines); hunks: -105,8 +105,12 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +4/-1 (5 lines); hunks: -625,16 +625,19 @@ def _dispatch_core(; symbols: _dispatch_core，涉及 `_dispatch_core`；`python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +1/-0 (1 lines); hunks: -98,6 +98,7 @@ def __init__(self):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +47/-2 (49 lines); hunks: -4,6 +4,7; -137,17 +138,23 @@ def __init__(; symbols: __init__, run_moe_core, forward_aiter, forward_unquantized_deepep_ll
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +6/-2 (8 lines); hunks: -105,8 +105,12 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +4/-1 (5 lines); hunks: -625,16 +625,19 @@ def _dispatch_core(; symbols: _dispatch_core
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +1/-0 (1 lines); hunks: -98,6 +98,7 @@ def __init__(self):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -4,6 +4,7 @@
+import torch.nn.functional as F
@@ -137,17 +138,23 @@ def __init__(
+        if quant_config is None and hasattr(self.dispatcher, "set_quant_config"):
+            self.dispatcher.set_quant_config({"bf16_dispatch": True})
+                and self.quant_config is not None
+            and quant_config is not None
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -105,8 +105,12 @@ def __init__(
-        # Verify kernel: use FlashInfer if either decode or prefill selected it
-        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
+        # Verify kernel: use FlashInfer only when the selected FlashInfer kernel
+        # supports MTP verify. On SM100+ FlashInfer GDN decode is supported, but
+        # its MTP verify path is not, so keep Triton as the verify fallback.
+        if (
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -625,16 +625,19 @@ def _dispatch_core(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +47/-2; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +6/-2; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +4/-1; `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21668 - [XPU] Enable qwen3.5 on XPU

- 链接: https://github.com/sgl-project/sglang/pull/21668
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+757/-13，可读 patch 895 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Enable qwen3.5 on XPU」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/chunk.py`, `python/sglang/srt/layers/attention/fla/kda.py`；技术摘要: 覆盖「[XPU] Enable qwen3.5 on XPU」；主要实现面是 `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/chunk.py`, `python/sglang/srt/layers/attention/fla/kda.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +12/-0 (12 lines); hunks: -278,6 +278,18 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index，涉及 `forward_npu, forward_xpu, get_rope_index`；`python/sglang/srt/layers/attention/fla/chunk.py` modified +9/-0 (9 lines); hunks: -19,8 +19,17；`python/sglang/srt/layers/attention/fla/kda.py` modified +7/-0 (7 lines); hunks: -26,8 +26,15；`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +5/-1 (6 lines); hunks: -3,7 +3,7; -29,6 +29,10; symbols: TritonGDNKernel，涉及 `TritonGDNKernel`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +12/-0 (12 lines); hunks: -278,6 +278,18 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index
  - `python/sglang/srt/layers/attention/fla/chunk.py` modified +9/-0 (9 lines); hunks: -19,8 +19,17
  - `python/sglang/srt/layers/attention/fla/kda.py` modified +7/-0 (7 lines); hunks: -26,8 +26,15
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +5/-1 (6 lines); hunks: -3,7 +3,7; -29,6 +29,10; symbols: TritonGDNKernel
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +3/-0 (3 lines); hunks: -172,6 +172,9 @@ def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/rotary_embedding/mrope.py
@@ -278,6 +278,18 @@ def forward_npu(
+    def forward_xpu(
+        self,
+        positions: torch.Tensor,
+        query: torch.Tensor,
+        key: torch.Tensor,
+        fused_set_kv_buffer_arg=None,
diff -- python/sglang/srt/layers/attention/fla/chunk.py
@@ -19,8 +19,17 @@
+    is_intel,
+if is_intel:
+    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
+        chunk_gated_delta_rule_fwd_h,
+    )
+    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_fwd import (
diff -- python/sglang/srt/layers/attention/fla/kda.py
@@ -26,8 +26,15 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +12/-0; `python/sglang/srt/layers/attention/fla/chunk.py` modified +9/-0; `python/sglang/srt/layers/attention/fla/kda.py` modified +7/-0; `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +5/-1; `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +3/-0; `python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_fwd.py` added +315/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/ci/ci_register.py`, `test/registered/attention/test_chunk_gated_delta_rule.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25401 - Add output_gate_type to Qwen3NextConfig and update models to utilize it

- 链接: https://github.com/sgl-project/sglang/pull/25401
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+21/-1，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add output_gate_type to Qwen3NextConfig and update models to utilize it」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/configs/qwen3_next.py`；技术摘要: 覆盖「Add output_gate_type to Qwen3NextConfig and update models to utilize it」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/configs/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +11/-1 (12 lines); hunks: -106,6 +106,7 @@ def __init__(; -186,12 +187,21 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_5.py` modified +6/-0 (6 lines); hunks: -143,6 +143,7 @@ def __init__(; -237,6 +238,11 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/configs/qwen3_next.py` modified +4/-0 (4 lines); hunks: -68,6 +68,8 @@ class Qwen3NextConfig(PretrainedConfig):; -186,6 +188,7 @@ def __init__(; symbols: Qwen3NextConfig, __init__，涉及 `Qwen3NextConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +11/-1 (12 lines); hunks: -106,6 +106,7 @@ def __init__(; -186,12 +187,21 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_5.py` modified +6/-0 (6 lines); hunks: -143,6 +143,7 @@ def __init__(; -237,6 +238,11 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/configs/qwen3_next.py` modified +4/-0 (4 lines); hunks: -68,6 +68,8 @@ class Qwen3NextConfig(PretrainedConfig):; -186,6 +188,7 @@ def __init__(; symbols: Qwen3NextConfig, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -106,6 +106,7 @@ def __init__(
+        self.output_gate_type = config.output_gate_type
@@ -186,12 +187,21 @@ def __init__(
+                **(
+                    {"activation": self.output_gate_type}
+                    if self.output_gate_type is not None
+                    else {}
diff -- python/sglang/srt/models/qwen3_5.py
@@ -143,6 +143,7 @@ def __init__(
+        self.output_gate_type = config.output_gate_type
@@ -237,6 +238,11 @@ def __init__(
+            **(
+                {"activation": self.output_gate_type}
+                if self.output_gate_type is not None
+                else {}
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -68,6 +68,8 @@ class Qwen3NextConfig(PretrainedConfig):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +11/-1; `python/sglang/srt/models/qwen3_5.py` modified +6/-0; `python/sglang/srt/configs/qwen3_next.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25735 - [NPU] [DOCS] Improved the usability of Ascend NPU documents

- 链接: https://github.com/sgl-project/sglang/pull/25735
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+468/-49，可读 patch 743 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] [DOCS] Improved the usability of Ascend NPU documents」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`；技术摘要: 覆盖「[NPU] [DOCS] Improved the usability of Ascend NPU documents」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23331 - [BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN)

- 链接: https://github.com/sgl-project/sglang/pull/23331
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5_mtp.py`；关联提交 `b9c2bf717ba4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+156/-68，可读 patch 444 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN)」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`；技术摘要: 覆盖「[BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN)」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -51,6 +52,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0 (4 lines); hunks: -14,6 +14,7; -51,6 +52,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -14,6 +14,7 @@
+import copy
@@ -51,6 +52,9 @@ def __init__(
+        # Deep-copy so MTP mutations below don't leak into the target's config.
+        config = copy.deepcopy(config)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/spec/test_adaptive_spec_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23925 - [NPU]use triton split_qkvgate_gemma_rmsnorm_rope for Qwen3.5 and Qwen3_next

- 链接: https://github.com/sgl-project/sglang/pull/23925
- 状态/时间: merged / 2026-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `55ba03db6a46`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+194/-19，可读 patch 284 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]use triton split_qkvgate_gemma_rmsnorm_rope for Qwen3.5 and Qwen3_next」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[NPU]use triton split_qkvgate_gemma_rmsnorm_rope for Qwen3.5 and Qwen3_next」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +52/-8 (60 lines); hunks: -109,6 +109,11; -841,15 +846,8 @@ def _apply_qk_norm(; symbols: Qwen3_5GatedDeltaNet, __init__, _apply_qk_norm, self_attention，涉及 `Qwen3_5GatedDeltaNet, __init__, _apply_qk_norm`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +52/-8 (60 lines); hunks: -109,6 +109,11; -841,15 +846,8 @@ def _apply_qk_norm(; symbols: Qwen3_5GatedDeltaNet, __init__, _apply_qk_norm, self_attention
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -109,6 +109,11 @@
+if _is_npu:
+    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
+        split_qkvgate_gemma_rmsnorm_rope,
+    )
@@ -841,15 +846,8 @@ def _apply_qk_norm(
-    def self_attention(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +52/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26069 - [NPU]Ascend NPU Performance Profiling Guide and Ascend NPU Operator Development Guide

- 链接: https://github.com/sgl-project/sglang/pull/26069
- 状态/时间: merged / 2026-05-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+92/-1，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Ascend NPU Performance Profiling Guide and Ascend NPU Operator Development Guide」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`；技术摘要: 覆盖「[NPU]Ascend NPU Performance Profiling Guide and Ascend NPU Operator Development Guide」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +71/-0 (71 lines); hunks: -271,6 +271,77 @@ python3 -m sglang.launch_server \；`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +21/-1 (22 lines); hunks: -82,13 +82,33 @@ docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11。
- 代码 diff 细节:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +71/-0 (71 lines); hunks: -271,6 +271,77 @@ python3 -m sglang.launch_server \
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +21/-1 (22 lines); hunks: -82,13 +82,33 @@ docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
- 关键代码摘录:

```diff
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx
@@ -271,6 +271,77 @@ python3 -m sglang.launch_server \
+### Multi-node Deployment
+<Tip>
+Recommended model: [`Qwen/Qwen3.5-35B-A3B`](https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)
+Other Qwen3.5 series models can also be deployed in multi-node configurations following this workflow. Simply change `--model-path` to the corresponding model, and adjust paramete
+</Tip>
+**A2 series**
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx
@@ -82,13 +82,33 @@ docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
-Only `python==3.11` is supported currently. If you don't want to break system pre-installed python, try installing with [conda](https://github.com/conda/conda).
+**Only `python==3.11` is supported currently**. If you don't want to break system pre-installed python, try installing with [conda](https://github.com/conda/conda).
+Note on Anaconda repository restrictions
+If you encounter an error like “Terms of Service have not been accepted” during the conda create step, the default Anaconda repository is blocking package downloads. To resolve th
+'''bash Command
+# Add Tsinghua mirrors
```

- 已读文件:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx` modified +71/-0; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` modified +21/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- 链接: https://github.com/sgl-project/sglang/pull/26610
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+611/-816，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；主要实现面是 `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`；`python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache，涉及 `_random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`；`test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass，涉及 `TestStep3p5FlashChainMTP, setUpClass, tearDownClass`；`test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV3MTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache
  - `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105 (105 lines); hunks: -1,105 +0,0; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,212 +0,0 @@
-import unittest
-from types import SimpleNamespace
-import requests
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.run_eval import run_eval
diff -- python/sglang/test/kits/unified_radix_cache_kit.py
@@ -1,25 +1,12 @@
-import unittest
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-    get_input_ids,
-    make_mamba_decode_assert,
-    make_mamba_prefill_assert,
diff -- test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
@@ -1,28 +1,20 @@
```

- 已读文件:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110; `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105; `test/registered/quant/test_deepseek_v3_fp4_4gpu.py` removed +0/-80
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26695 - [docs] Qwen3.5 cookbook: multi-node, MTP TP overrides, dense mamba flag

- 链接: https://github.com/sgl-project/sglang/pull/26695
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；关联提交 `7fb7b41a3e06`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+77/-18，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Qwen3.5 cookbook: multi-node, MTP TP overrides, dense mamba flag」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`；技术摘要: 覆盖「[docs] Qwen3.5 cookbook: multi-node, MTP TP overrides, dense mamba flag」；主要实现面是 `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +76/-17 (93 lines); hunks: -8,14 +8,15 @@ export const Qwen35Deployment = () => {; -142,7 +143,7 @@ export const Qwen35Deployment = () => {；`docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ SGLang from the main branch is required for Qwen3.5. You can i...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +76/-17 (93 lines); hunks: -8,14 +8,15 @@ export const Qwen35Deployment = () => {; -142,7 +143,7 @@ export const Qwen35Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ SGLang from the main branch is required for Qwen3.5. You can i...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen35-deployment.jsx
@@ -8,14 +8,15 @@ export const Qwen35Deployment = () => {
-  //   397B-A17B: H100 tp=16, H200 tp=8, B200 tp=8, B300 tp=4, MI300X tp=8, MI325X tp=4, MI355X tp=4
-  //   122B-A10B: H100 tp=4,  H200 tp=2, B200 tp=2, B300 tp=1, MI300X tp=2, MI325X tp=1, MI355X tp=1
-  //   35B-A3B:   H100 tp=1,  H200 tp=1, B200 tp=1, B300 tp=1, MI300X tp=1, MI325X tp=1, MI355X tp=1
-  //   27B/9B/4B/2B/0.8B: tp=1 on all hardware (including MI300X, MI325X, MI355X)
+  //   397B-A17B: H100 tp=16 (2 nodes), H200 tp=8, B200 tp=8, B300 tp=4, MI300X tp=8, MI325X tp=4, MI355X tp=4
+  //   122B-A10B: H100 tp=4,  H200 tp=4, B200 tp=2, B300 tp=2, MI300X tp=2, MI325X tp=1, MI355X tp=1
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx
@@ -95,7 +95,7 @@ SGLang from the main branch is required for Qwen3.5. You can install from source
-docker pull lmsysorg/sglang:nightly-dev-20260216-d3bae71e
+docker pull lmsysorg/sglang:latest
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +76/-17; `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26389 - 【NPU】【bugfix】fix server error when mtp unquant

- 链接: https://github.com/sgl-project/sglang/pull/26389
- 状态/时间: merged / 2026-05-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+148/-98，可读 patch 346 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「【NPU】【bugfix】fix server error when mtp unquant」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；技术摘要: 覆盖「【NPU】【bugfix】fix server error when mtp unquant」；主要实现面是 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_nextn.py` modified +69/-51 (120 lines); hunks: -16,6 +16,7; -169,70 +170,87 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_5_mtp.py` modified +43/-24 (67 lines); hunks: -16,13 +16,15; -140,38 +142,55 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_next_mtp.py` modified +33/-17 (50 lines); hunks: -16,13 +16,15; -94,25 +96,39 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-6 (8 lines); hunks: -1,7 +1,6; -437,11 +436,8 @@ def _validate_and_adjust_dtype(self) -> None:; symbols: _validate_and_adjust_dtype, _update_int8_quant_env, set_overlap_args，涉及 `_validate_and_adjust_dtype, _update_int8_quant_env, set_overlap_args`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +69/-51 (120 lines); hunks: -16,6 +16,7; -169,70 +170,87 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +43/-24 (67 lines); hunks: -16,13 +16,15; -140,38 +142,55 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +33/-17 (50 lines); hunks: -16,13 +16,15; -94,25 +96,39 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-6 (8 lines); hunks: -1,7 +1,6; -437,11 +436,8 @@ def _validate_and_adjust_dtype(self) -> None:; symbols: _validate_and_adjust_dtype, _update_int8_quant_env, set_overlap_args
  - `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py` modified +1/-0 (1 lines); hunks: -60,6 +60,7 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -16,6 +16,7 @@
+from contextlib import ExitStack
@@ -169,70 +170,87 @@ def forward(
-        zero_allocator = BumpAllocator(
-            buffer_size=2,
-            dtype=torch.float32,
-            device=(
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -16,13 +16,15 @@
+from contextlib import ExitStack
+from sglang.srt.environ import envs
@@ -140,38 +142,55 @@ def forward(
-        assert input_embeds is None
-        input_embeds = forward_batch.mm_input_embeds
+        exit_stack = ExitStack()
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -16,13 +16,15 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +69/-51; `python/sglang/srt/models/qwen3_5_mtp.py` modified +43/-24; `python/sglang/srt/models/qwen3_next_mtp.py` modified +33/-17; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-6
  - tests: `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27296 - Add --enable-symm-mem for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/27296
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；关联提交 `0e4aa081bad9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-0，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add --enable-symm-mem for Qwen3.5」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`；技术摘要: 覆盖「Add --enable-symm-mem for Qwen3.5」；主要实现面是 `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +5/-0 (5 lines); hunks: -376,6 +376,11 @@ export const Qwen35Deployment = () => {；`docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-0 (1 lines); hunks: -124,6 +124,7 @@ This section provides deployment configurations optimized fo...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +5/-0 (5 lines); hunks: -376,6 +376,11 @@ export const Qwen35Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-0 (1 lines); hunks: -124,6 +124,7 @@ This section provides deployment configurations optimized fo...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen35-deployment.jsx
@@ -376,6 +376,11 @@ export const Qwen35Deployment = () => {
+    // Enable NCCL symmetric memory for H100 FP8 deployments.
+    if (hardware === 'h100' && quantization === 'fp8' && hwConfig.tp > 1) {
+      cmd += ` \\\n  --enable-symm-mem`;
+    }
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx
@@ -124,6 +124,7 @@ This section provides deployment configurations optimized for different hardware
+- **H100 FP8:** Add `--enable-symm-mem` to enable NCCL symmetric memory for faster collectives and better performance under multi-GPU settings.
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +5/-0; `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25885 - [AMD] Support alt stream for Qwen3.5 on AMD platform

- 链接: https://github.com/sgl-project/sglang/pull/25885
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `7f919edf006b`, `d5899b95c4d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-7，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support alt stream for Qwen3.5 on AMD platform」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[AMD] Support alt stream for Qwen3.5 on AMD platform」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +26/-7 (33 lines); hunks: -105,10 +105,24; -445,6 +459,7 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _disable_shared_experts_fusion, _forward_input_proj, __init__, _apply_qk_norm，涉及 `_disable_shared_experts_fusion, _forward_input_proj, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +26/-7 (33 lines); hunks: -105,10 +105,24; -445,6 +459,7 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _disable_shared_experts_fusion, _forward_input_proj, __init__, _apply_qk_norm
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -105,10 +105,24 @@
+_hip_use_alt_stream = get_bool_env_var("SGLANG_ALT_STREAM") and _is_hip
+_gdn_use_alt_stream = (
+    get_bool_env_var("SGLANG_GDN_QKVZ_BA_ALT_STREAM", "False") and _hip_use_alt_stream
+)
+_qknorm_use_alt_stream = (
+    get_bool_env_var("SGLANG_QK_NORM_ALT_STREAM", "False") and _hip_use_alt_stream
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +26/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27248 - [Doc][CPU]Update Cookbook with Xeon support info

- 链接: https://github.com/sgl-project/sglang/pull/27248
- 状态/时间: merged / 2026-06-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+443/-121，可读 patch 1524 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc][CPU]Update Cookbook with Xeon support info」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`；技术摘要: 覆盖「[Doc][CPU]Update Cookbook with Xeon support info」；主要实现面是 `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {；`docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {；`docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20 (85 lines); hunks: -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {; -57,8 +67,11 @@ export const DeepSeekV3Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15 (65 lines); hunks: -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {; -26,9 +28,9 @@ export const DeepSeekV31Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7 (56 lines); hunks: -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {; -35,9 +46,9 @@ export const DeepSeekR1BasicDeployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18 (51 lines); hunks: -64,7 +64,8 @@ export const Qwen35Deployment = () => {; -74,12 +75,13 @@ export const Qwen35Deployment = () => {
  - `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10 (41 lines); hunks: -13,7 +13,8 @@ export const Hunyuan3PreviewDeployment = () => {; -35,18 +36,22 @@ export const Hunyuan3PreviewDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx
@@ -10,26 +10,36 @@ export const DeepSeekV3Deployment = () => {
-        { id: 'mi355x', label: 'MI355X', default: false }
+        { id: 'mi355x', label: 'MI355X', default: false },
+        { id: 'xeon', label: 'XEON', default: false }
-      items: [
-        { id: 'fp8', label: 'FP8', default: true },
-        { id: 'fp4', label: 'FP4', default: false }
diff -- docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx
@@ -9,15 +9,17 @@ export const DeepSeekV31Deployment = () => {
-        { id: 'mi355x', label: 'MI355X', default: false }
+        { id: 'mi355x', label: 'MI355X', default: false },
+        { id: 'xeon', label: 'XEON', default: false }
-        { id: 'v31terminus', label: 'DeepSeek-V3.1-Terminus', default: false }
+        { id: 'v31terminus', label: 'DeepSeek-V3.1-Terminus', default: false },
+        { id: 'v31terminusint8', label: 'DeepSeek-V3.1-Terminus-Channel-int8', default: false, xeonOnly: true }
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx
@@ -10,21 +10,32 @@ export const DeepSeekR1BasicDeployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` modified +65/-20; `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` modified +50/-15; `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` modified +49/-7; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +33/-18; `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx` modified +31/-10; `docs_new/src/snippets/autoregressive/deepseek-ocr-deployment.jsx` modified +29/-9
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27660 - [AMD] Update amd qwen3.5 cookbook

- 链接: https://github.com/sgl-project/sglang/pull/27660
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；关联提交 `9ab7a64ee1e3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+62/-29，可读 patch 192 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Update amd qwen3.5 cookbook」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`；技术摘要: 覆盖「[AMD] Update amd qwen3.5 cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +43/-18 (61 lines); hunks: -20,7 +20,7 @@ export const Qwen35Deployment = () => {; -64,7 +64,7 @@ export const Qwen35Deployment = () => {；`docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +19/-11 (30 lines); hunks: -42,7 +42,7 @@ Qwen3.5 features a Gated Delta Networks combined with sparse M...; -103,10 +103,10 @@ uv pip install 'git+https://github.com/sgl-project/sglang....。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +43/-18 (61 lines); hunks: -20,7 +20,7 @@ export const Qwen35Deployment = () => {; -64,7 +64,7 @@ export const Qwen35Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +19/-11 (30 lines); hunks: -42,7 +42,7 @@ Qwen3.5 features a Gated Delta Networks combined with sparse M...; -103,10 +103,10 @@ uv pip install 'git+https://github.com/sgl-project/sglang....
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen35-deployment.jsx
@@ -20,7 +20,7 @@ export const Qwen35Deployment = () => {
-  // FP4 (397B only, Blackwell required): B200 tp=4, B300 tp=2
+  // FP4 (397B only): NVFP4 on Blackwell B200 tp=4, B300 tp=2; AMD MXFP4 on MI355X tp=2
@@ -64,7 +64,7 @@ export const Qwen35Deployment = () => {
-          { id: 'mi355x', label: 'MI355X', default: false,     disabled: isNvfp4 },
+          { id: 'mi355x', label: 'MI355X', default: false,     disabled: false },
@@ -152,7 +152,7 @@ export const Qwen35Deployment = () => {
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx
@@ -42,7 +42,7 @@ Qwen3.5 features a Gated Delta Networks combined with sparse Mixture-of-Experts
-      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>[nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>NVIDIA NVFP4: [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-N
@@ -103,10 +103,10 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=pytho
-docker pull lmsysorg/sglang:v0.5.9-rocm720-mi30x
+docker pull lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi30x-20260604
-docker pull lmsysorg/sglang:v0.5.9-rocm720-mi35x
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +43/-18; `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx` modified +19/-11
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`, `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool，涉及 `freeze_gc, _to_torch, patch_model`；`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype，涉及 `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`；`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode，涉及 `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`；`python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers，涉及 `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27656 - [AMD][Perf] Fuse QK RMSNorm + gate extraction Triton kernel for Qwen3.5 on HIP

- 链接: https://github.com/sgl-project/sglang/pull/27656
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `0da18f8d916e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+315/-7，可读 patch 371 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][Perf] Fuse QK RMSNorm + gate extraction Triton kernel for Qwen3.5 on HIP」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[AMD][Perf] Fuse QK RMSNorm + gate extraction Triton kernel for Qwen3.5 on HIP」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +37/-3 (40 lines); hunks: -79,7 +79,10; -884,6 +887,33 @@ def forward_prepare_native(self, positions, hidden_states):; symbols: forward_prepare_native, forward_prepare_hip, forward_prepare_npu, self_attention，涉及 `forward_prepare_native, forward_prepare_hip, forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +37/-3 (40 lines); hunks: -79,7 +79,10; -884,6 +887,33 @@ def forward_prepare_native(self, positions, hidden_states):; symbols: forward_prepare_native, forward_prepare_hip, forward_prepare_npu, self_attention
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -79,7 +79,10 @@
-from sglang.srt.models.utils import fused_qk_gemma_rmsnorm
+from sglang.srt.models.utils import (
+    fused_qk_gemma_rmsnorm,
+    fused_qk_gemma_rmsnorm_with_gate,
+)
@@ -884,6 +887,33 @@ def forward_prepare_native(self, positions, hidden_states):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +37/-3
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_qk_gemma_rmsnorm_gate.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27630 - [AMD] Fuse sigmoid + mul attention output gate into single Triton kernel

- 链接: https://github.com/sgl-project/sglang/pull/27630
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+130/-4，可读 patch 164 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fuse sigmoid + mul attention output gate into single Triton kernel」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py`；技术摘要: 覆盖「[AMD] Fuse sigmoid + mul attention output gate into single Triton kernel」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +11/-2 (13 lines); hunks: -49,6 +49,7; -60,6 +61,7; symbols: self_attention，涉及 `self_attention`；`python/sglang/srt/models/qwen3_5.py` modified +9/-2 (11 lines); hunks: -930,8 +930,15 @@ def self_attention(; symbols: self_attention，涉及 `self_attention`；`python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: reference_sigmoid_gate_mul, test_sigmoid_gate_mul_correctness, test_sigmoid_gate_mul_does_not_modify_inputs, test_sigmoid_gate_mul_output_dtype，涉及 `reference_sigmoid_gate_mul, test_sigmoid_gate_mul_correctness, test_sigmoid_gate_mul_does_not_modify_inputs`；`python/sglang/jit_kernel/triton/sigmoid_gate_mul.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: _sigmoid_gate_mul_kernel, sigmoid_gate_mul，涉及 `_sigmoid_gate_mul_kernel, sigmoid_gate_mul`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +11/-2 (13 lines); hunks: -49,6 +49,7; -60,6 +61,7; symbols: self_attention
  - `python/sglang/srt/models/qwen3_5.py` modified +9/-2 (11 lines); hunks: -930,8 +930,15 @@ def self_attention(; symbols: self_attention
  - `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: reference_sigmoid_gate_mul, test_sigmoid_gate_mul_correctness, test_sigmoid_gate_mul_does_not_modify_inputs, test_sigmoid_gate_mul_output_dtype
  - `python/sglang/jit_kernel/triton/sigmoid_gate_mul.py` added +29/-0 (29 lines); hunks: -0,0 +1,29; symbols: _sigmoid_gate_mul_kernel, sigmoid_gate_mul
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -49,6 +49,7 @@
+    is_hip,
@@ -60,6 +61,7 @@
+_is_hip = is_hip()
@@ -819,8 +821,15 @@ def self_attention(
-            gate = torch.sigmoid(gate)
-            attn_output = attn_output * gate
diff -- python/sglang/srt/models/qwen3_5.py
@@ -930,8 +930,15 @@ def self_attention(
-            gate = torch.sigmoid(gate)
-            attn_output = attn_output * gate
+            if _is_hip:
+                from sglang.jit_kernel.triton.sigmoid_gate_mul import (
+                    sigmoid_gate_mul,
+                )
diff -- python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py
@@ -0,0 +1,81 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +11/-2; `python/sglang/srt/models/qwen3_5.py` modified +9/-2; `python/sglang/jit_kernel/triton/sigmoid_gate_mul.py` added +29/-0
  - tests: `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py` added +81/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_sigmoid_gate_mul.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27846 - fix: per-sequence last-token embedding in EAGLE3/MTP draft for batched multimodal spec decoding

- 链接: https://github.com/sgl-project/sglang/pull/27846
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-8，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: per-sequence last-token embedding in EAGLE3/MTP draft for batched multimodal spec decoding」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/llama_eagle3.py`；技术摘要: 覆盖「fix: per-sequence last-token embedding in EAGLE3/MTP draft for batched multimodal spec decoding」；主要实现面是 `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/llama_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5_mtp.py` modified +5/-5 (10 lines); hunks: -163,11 +163,11 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/llama_eagle3.py` modified +4/-3 (7 lines); hunks: -198,9 +198,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +5/-5 (10 lines); hunks: -163,11 +163,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/llama_eagle3.py` modified +4/-3 (7 lines); hunks: -198,9 +198,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -163,11 +163,11 @@ def forward(
-                input_embeds = torch.cat(
-                    [
-                        input_embeds[:-1],
-                        self.model.embed_tokens(input_ids[-1].unsqueeze(0)),
-                    ]
+                last_indices = (
diff -- python/sglang/srt/models/llama_eagle3.py
@@ -198,9 +198,10 @@ def forward(
-                embeds = torch.cat(
-                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
-                )
+                last_indices = (
+                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
+                ).long()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5_mtp.py` modified +5/-5; `python/sglang/srt/models/llama_eagle3.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama_eagle3.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27964 - [Spec] Retire Spec V1

- 链接: https://github.com/sgl-project/sglang/pull/27964
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 46 个文件，+111/-252，可读 patch 1422 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Retire Spec V1」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`；技术摘要: 覆盖「[Spec] Retire Spec V1」；主要实现面是 `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23862 - Fix --mem-fraction-static not accounting for EAGLE draft model KV cache

- 链接: https://github.com/sgl-project/sglang/pull/23862
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+688/-295，可读 patch 1511 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix --mem-fraction-static not accounting for EAGLE draft model KV cache」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `test/registered/unit/configs/test_model_config_shapes.py`, `python/sglang/srt/configs/model_config.py`；技术摘要: 覆盖「Fix --mem-fraction-static not accounting for EAGLE draft model KV cache」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `test/registered/unit/configs/test_model_config_shapes.py`, `python/sglang/srt/configs/model_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +124/-63 (187 lines); hunks: -449,41 +449,54 @@ def __init__(; -541,8 +554,9 @@ def __init__(; symbols: __init__, _build_model_config，涉及 `__init__, _build_model_config`；`test/registered/unit/configs/test_model_config_shapes.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: _make_text_config, TestModelConfigShapes, _derive_shapes, test_optional_head_dims_default_when_none，涉及 `_make_text_config, TestModelConfigShapes, _derive_shapes`；`python/sglang/srt/configs/model_config.py` modified +34/-27 (61 lines); hunks: -12,6 +12,7; -198,13 +199,18 @@ def __init__(; symbols: __init__, _derive_context_length, _derive_model_shapes，涉及 `__init__, _derive_context_length, _derive_model_shapes`；`test/registered/unit/model_executor/test_pool_configurator.py` modified +31/-0 (31 lines); hunks: -83,6 +83,8 @@ def _make_model_runner(; -303,6 +305,35 @@ def test_constraint_respected(self):; symbols: _make_model_runner, test_constraint_respected, TestEagleConfigurator, test_eagle_does_not_exceed_budget，涉及 `_make_model_runner, test_constraint_respected, TestEagleConfigurator`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +124/-63 (187 lines); hunks: -449,41 +449,54 @@ def __init__(; -541,8 +554,9 @@ def __init__(; symbols: __init__, _build_model_config
  - `test/registered/unit/configs/test_model_config_shapes.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: _make_text_config, TestModelConfigShapes, _derive_shapes, test_optional_head_dims_default_when_none
  - `python/sglang/srt/configs/model_config.py` modified +34/-27 (61 lines); hunks: -12,6 +12,7; -198,13 +199,18 @@ def __init__(; symbols: __init__, _derive_context_length, _derive_model_shapes
  - `test/registered/unit/model_executor/test_pool_configurator.py` modified +31/-0 (31 lines); hunks: -83,6 +83,8 @@ def _make_model_runner(; -303,6 +305,35 @@ def test_constraint_respected(self):; symbols: _make_model_runner, test_constraint_respected, TestEagleConfigurator, test_eagle_does_not_exceed_budget
  - `python/sglang/srt/model_executor/pool_configurator.py` modified +18/-0 (18 lines); hunks: -104,6 +104,24 @@ def __init__(self, mr: ModelRunner):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -449,41 +449,54 @@ def __init__(
+        self.eagle_draft_num_layers = None
-        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
-            # load draft config
-            draft_model_config = ModelConfig.from_server_args(
+        if (
+            (self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone())
diff -- test/registered/unit/configs/test_model_config_shapes.py
@@ -0,0 +1,71 @@
+"""Unit tests for ModelConfig shape normalization."""
+import unittest
+from types import SimpleNamespace
+from sglang.srt.configs.model_config import ModelConfig
+from sglang.test.ci.ci_register import register_cpu_ci
+from sglang.test.test_utils import CustomTestCase
diff -- python/sglang/srt/configs/model_config.py
@@ -12,6 +12,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +124/-63; `python/sglang/srt/configs/model_config.py` modified +34/-27; `python/sglang/srt/model_executor/pool_configurator.py` modified +18/-0; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +11/-6; `python/sglang/srt/models/qwen3_5_mtp.py` modified +4/-3; `python/sglang/srt/models/qwen3_next_mtp.py` modified +4/-3
  - tests: `test/registered/unit/configs/test_model_config_shapes.py` added +71/-0; `test/registered/unit/model_executor/test_pool_configurator.py` modified +31/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/configs/test_model_config_shapes.py`, `test/registered/unit/model_executor/test_pool_configurator.py`, `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27057 - [AMD] move shared expert check function to quark

- 链接: https://github.com/sgl-project/sglang/pull/27057
- 状态/时间: merged / 2026-06-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+74/-11，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] move shared expert check function to quark」；模型线: Qwen3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「[AMD] move shared expert check function to quark」；主要实现面是 `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0 (44 lines); hunks: -37,6 +37,18; -492,6 +504,38 @@ def get_moe_scheme(; symbols: QuarkConfig, get_moe_scheme, get_scaled_act_names, can_fuse_shared_expert，涉及 `QuarkConfig, get_moe_scheme, get_scaled_act_names`；`python/sglang/srt/models/qwen3_5.py` modified +23/-1 (24 lines); hunks: -80,7 +80,11; -1117,6 +1121,21 @@ def get_hidden_dim(self, module_name: str, layer_idx: int):; symbols: get_hidden_dim, _maybe_autodisable_shared_experts_fusion, __init__，涉及 `get_hidden_dim, _maybe_autodisable_shared_experts_fusion, __init__`；`python/sglang/srt/models/qwen2_moe.py` modified +7/-10 (17 lines); hunks: -156,20 +156,17 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert，涉及 `can_fuse_shared_expert`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0 (44 lines); hunks: -37,6 +37,18; -492,6 +504,38 @@ def get_moe_scheme(; symbols: QuarkConfig, get_moe_scheme, get_scaled_act_names, can_fuse_shared_expert
  - `python/sglang/srt/models/qwen3_5.py` modified +23/-1 (24 lines); hunks: -80,7 +80,11; -1117,6 +1121,21 @@ def get_hidden_dim(self, module_name: str, layer_idx: int):; symbols: get_hidden_dim, _maybe_autodisable_shared_experts_fusion, __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-10 (17 lines); hunks: -156,20 +156,17 @@ def can_fuse_shared_expert(; symbols: can_fuse_shared_expert
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -37,6 +37,18 @@
+_MOE_SHARED_EXPERT_QUANT_LAYER0_BASES: tuple[str, ...] = (
+    "model.layers.0",
+    "model.language_model.layers.0",
+)
+_SHARED_EXPERT_BODY_PROJ_SUFFIXES: tuple[str, ...] = (
+    "gate_proj",
diff -- python/sglang/srt/models/qwen3_5.py
@@ -80,7 +80,11 @@
-from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
+from sglang.srt.models.qwen2_moe import (
+    Qwen2MoeMLP,
+    Qwen2MoeSparseMoeBlock,
+    can_fuse_shared_expert,
+)
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -156,20 +156,17 @@ def can_fuse_shared_expert(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/quark/quark.py` modified +44/-0; `python/sglang/srt/models/qwen3_5.py` modified +23/-1; `python/sglang/srt/models/qwen2_moe.py` modified +7/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26924 - [4/N] Qwen3.5Opt: Overlap mamba verify update with draft extend

- 链接: https://github.com/sgl-project/sglang/pull/26924
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `aea0e308537a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+534/-16，可读 patch 647 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[4/N] Qwen3.5Opt: Overlap mamba verify update with draft extend」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[4/N] Qwen3.5Opt: Overlap mamba verify update with draft extend」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +3/-10 (13 lines); hunks: -47,6 +47,7; -883,7 +884,7 @@ def forward_prepare_native(self, positions, hidden_states):; symbols: forward_prepare_native, self_attention，涉及 `forward_prepare_native, self_attention`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +3/-10 (13 lines); hunks: -47,6 +47,7; -883,7 +884,7 @@ def forward_prepare_native(self, positions, hidden_states):; symbols: forward_prepare_native, self_attention
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -47,6 +47,7 @@
+from sglang.srt.layers.elementwise import fused_sigmoid_mul
@@ -883,7 +884,7 @@ def forward_prepare_native(self, positions, hidden_states):
-            gate = gate.reshape(*orig_shape, -1)
+            # gate stays as 3D strided view; fused_sigmoid_mul handles it directly
@@ -970,15 +971,7 @@ def self_attention(
-            if _is_hip:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +3/-10
- 验证与风险: diff 自带测试面 `test/manual/layers/test_fused_gate_sigmoid_mul_add.py`, `test/manual/layers/test_fused_sigmoid_mul.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28129 - [Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode

- 链接: https://github.com/sgl-project/sglang/pull/28129
- 状态/时间: merged / 2026-06-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+219/-2555，可读 patch 3937 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py`；技术摘要: 覆盖「[Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157 (159 lines); hunks: -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBat...; -1941,72 +1859,6 @@ def _apply_cuda_graph_metadata(; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend，涉及 `init_forward_metadata, _apply_cuda_graph_metadata, forward_extend`；`python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60 (79 lines); hunks: -94,8 +94,6 @@ class ForwardMode(IntEnum):; -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):; symbols: ForwardMode, is_extend, is_decode_or_idle, is_target_verify，涉及 `ForwardMode, is_extend, is_decode_or_idle`；`python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52 (72 lines); hunks: -454,31 +454,25 @@ def _update_draft_extend_buffers(; -693,32 +687,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata, _apply_cuda_graph_metadata，涉及 `_update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata`；`python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40 (47 lines); hunks: -352,7 +352,7 @@ def init_forward_metadata_out_graph(; -645,9 +645,10 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn, _bind_metadata_buffers，涉及 `init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157 (159 lines); hunks: -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBat...; -1941,72 +1859,6 @@ def _apply_cuda_graph_metadata(; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60 (79 lines); hunks: -94,8 +94,6 @@ class ForwardMode(IntEnum):; -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):; symbols: ForwardMode, is_extend, is_decode_or_idle, is_target_verify
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52 (72 lines); hunks: -454,31 +454,25 @@ def _update_draft_extend_buffers(; -693,32 +687,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata, _apply_cuda_graph_metadata
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40 (47 lines); hunks: -352,7 +352,7 @@ def init_forward_metadata_out_graph(; -645,9 +645,10 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn, _bind_metadata_buffers
  - `python/sglang/srt/layers/attention/dsa_backend.py` modified +11/-33 (44 lines); hunks: -406,9 +406,7 @@ def _build_paged_mqa_schedule_2d_ctx_lens(; -488,7 +486,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _build_paged_mqa_schedule_2d_ctx_lens, init_forward_metadata
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-        elif forward_batch.forward_mode.is_draft_extend():
-            # EAGLE V1: DRAFT_EXTEND mode - uses spec_info.num_accept_tokens
-            if self.use_mla:
-                kv_indices, kv_indptr, qo_indptr, custom_mask = (
-                    spec_info.generate_attn_arg_prefill(
-                        forward_batch.req_pool_indices,
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -94,8 +94,6 @@ class ForwardMode(IntEnum):
-    DRAFT_EXTEND = auto()
@@ -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):
-            or self == ForwardMode.DRAFT_EXTEND
@@ -148,19 +145,13 @@ def is_decode_or_idle(self):
-    def is_draft_extend(self, include_v2: bool = False):
-        return self == ForwardMode.DRAFT_EXTEND or (
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -454,31 +454,25 @@ def _update_draft_extend_buffers(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157; `python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60; `python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40; `python/sglang/srt/layers/attention/dsa_backend.py` modified +11/-33; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-24
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27869 - Fix Qwen3.5 deterministic batch-invariant logprobs

- 链接: https://github.com/sgl-project/sglang/pull/27869
- 状态/时间: merged / 2026-06-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/attention/test_qwen35_deterministic.py`；关联提交 `171037c3e73d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+56/-5，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Qwen3.5 deterministic batch-invariant logprobs」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `test/registered/attention/test_qwen35_deterministic.py`；技术摘要: 覆盖「Fix Qwen3.5 deterministic batch-invariant logprobs」；主要实现面是 `test/registered/attention/test_qwen35_deterministic.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/attention/test_qwen35_deterministic.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: TestQwen35Fa3Deterministic, get_model, get_server_args，涉及 `TestQwen35Fa3Deterministic, get_model, get_server_args`。
- 代码 diff 细节:
  - `test/registered/attention/test_qwen35_deterministic.py` added +44/-0 (44 lines); hunks: -0,0 +1,44; symbols: TestQwen35Fa3Deterministic, get_model, get_server_args
- 关键代码摘录:

```diff
diff -- test/registered/attention/test_qwen35_deterministic.py
@@ -0,0 +1,44 @@
+"""
+Usage:
+cd test/srt
+python3 -m unittest test_qwen35_deterministic.TestQwen35Fa3Deterministic
+"""
+import unittest
```

- 已读文件:
  - tests: `test/registered/attention/test_qwen35_deterministic.py` added +44/-0
- 验证与风险: diff 自带测试面 `test/registered/attention/test_qwen35_deterministic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27868 - fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885)

- 链接: https://github.com/sgl-project/sglang/pull/27868
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `d5899b95c4d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 39 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885)」；模型线: Qwen3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885)」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +12/-4 (16 lines); hunks: -119,10 +119,10; -594,7 +594,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +12/-4 (16 lines); hunks: -119,10 +119,10; -594,7 +594,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -119,10 +119,10 @@
-_gdn_use_alt_stream = (
+_gdn_use_alt_stream = _is_cuda or (
-_qknorm_use_alt_stream = (
+_qknorm_use_alt_stream = _is_cuda or (
@@ -594,7 +594,11 @@ def __init__(
-                alt_stream=(alt_stream if _disable_shared_experts_fusion() else None),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +12/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28293 - [NPU] Add NPU fallback for fused Triton gating kernels

- 链接: https://github.com/sgl-project/sglang/pull/28293
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-1，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add NPU fallback for fused Triton gating kernels」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「[NPU] Add NPU fallback for fused Triton gating kernels」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +5/-1 (6 lines); hunks: -975,7 +975,11 @@ def self_attention(; symbols: self_attention，涉及 `self_attention`；`python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -540,6 +540,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +5/-1 (6 lines); hunks: -975,7 +975,11 @@ def self_attention(; symbols: self_attention
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -540,6 +540,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -975,7 +975,11 @@ def self_attention(
-            attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
+            if not _is_npu:
+                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
+            else:
+                gate_val = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
+                attn_output.mul_(torch.sigmoid(gate_val))
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -540,6 +540,7 @@ def forward(
+            and not is_npu()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +5/-1; `python/sglang/srt/models/qwen2_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Qwen3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`；技术摘要: 覆盖「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；主要实现面是 `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU」；模型线: Qwen3.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`；技术摘要: 覆盖「[NPU] [DOC] Create deployment tutorials for mainstream models on Ascend NPU」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_5_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27870 - [qwen3.5][XPU]Add XPU support for set_embed_and_head and fused QK RMSNorm kernel

- 链接: https://github.com/sgl-project/sglang/pull/27870
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `bf231f01a3b3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-8，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[qwen3.5][XPU]Add XPU support for set_embed_and_head and fused QK RMSNorm kernel」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「[qwen3.5][XPU]Add XPU support for set_embed_and_head and fused QK RMSNorm kernel」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +18/-8 (26 lines); hunks: -105,6 +105,7; -116,6 +117,7; symbols: _apply_qk_norm, forward_prepare_native, forward_prepare_hip, forward_prepare_fused_gate，涉及 `_apply_qk_norm, forward_prepare_native, forward_prepare_hip`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +18/-8 (26 lines); hunks: -105,6 +105,7; -116,6 +117,7; symbols: _apply_qk_norm, forward_prepare_native, forward_prepare_hip, forward_prepare_fused_gate
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -105,6 +105,7 @@
+    is_xpu,
@@ -116,6 +117,7 @@
+_is_xpu = is_xpu()
@@ -867,7 +869,7 @@ def _apply_qk_norm(
-        elif _is_hip:
+        elif _is_hip or _is_xpu:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +18/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28320 - Fused QK GemmaRMSNorm + RoPE + gate kernel for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/28320
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_5.py`；关联提交 `4a8200565e1c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+244/-7，可读 patch 292 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fused QK GemmaRMSNorm + RoPE + gate kernel for Qwen3.5」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`；技术摘要: 覆盖「Fused QK GemmaRMSNorm + RoPE + gate kernel for Qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +40/-1 (41 lines); hunks: -137,6 +137,11 @@ def _disable_shared_experts_fusion() -> bool:; -887,6 +892,35 @@ def _apply_qk_norm(; symbols: _disable_shared_experts_fusion, _apply_qk_norm, forward_prepare_cuda_fused, forward_prepare_native，涉及 `_disable_shared_experts_fusion, _apply_qk_norm, forward_prepare_cuda_fused`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +40/-1 (41 lines); hunks: -137,6 +137,11 @@ def _disable_shared_experts_fusion() -> bool:; -887,6 +892,35 @@ def _apply_qk_norm(; symbols: _disable_shared_experts_fusion, _apply_qk_norm, forward_prepare_cuda_fused, forward_prepare_native
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -137,6 +137,11 @@ def _disable_shared_experts_fusion() -> bool:
+if _is_cuda:
+    from sglang.srt.layers.fused_qk_rmsnorm_rope_gate import (
+        fused_qk_gemma_rmsnorm_rope_gate,
+    )
@@ -887,6 +892,35 @@ def _apply_qk_norm(
+    def forward_prepare_cuda_fused(self, positions, hidden_states):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +40/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/hardware_backend/mlx/test_attention_patching.py`, `test/registered/unit/hardware_backend/mlx/test_mlx_runner_pool_contract.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/28103
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+218/-19，可读 patch 334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`；技术摘要: 覆盖「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；主要实现面是 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #29267 - [CPU] add indices in chunk_gated_delta_rule

- 链接: https://github.com/sgl-project/sglang/pull/29267
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+49/-17，可读 patch 236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] add indices in chunk_gated_delta_rule」；模型线: Qwen3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`, `python/sglang/srt/model_executor/cpu_graph_runner.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`；技术摘要: 覆盖「[CPU] add indices in chunk_gated_delta_rule」；主要实现面是 `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`, `python/sglang/srt/model_executor/cpu_graph_runner.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +2/-1 (3 lines); hunks: -141,9 +141,10 @@ def extend(; symbols: extend，涉及 `extend`；`python/sglang/srt/model_executor/cpu_graph_runner.py` modified +2/-1 (3 lines); hunks: -534,7 +534,8 @@ def _(; symbols: _，涉及 `_`；`python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +1/-1 (2 lines); hunks: -503,7 +503,7 @@ def forward_extend(; symbols: forward_extend，涉及 `forward_extend`；`python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -984,7 +984,7 @@ def self_attention(; symbols: self_attention，涉及 `self_attention`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +2/-1 (3 lines); hunks: -141,9 +141,10 @@ def extend(; symbols: extend
  - `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +2/-1 (3 lines); hunks: -534,7 +534,8 @@ def _(; symbols: _
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +1/-1 (2 lines); hunks: -503,7 +503,7 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-1 (2 lines); hunks: -984,7 +984,7 @@ def self_attention(; symbols: self_attention
  - `sgl-kernel/csrc/cpu/mamba/fla.cpp` modified +21/-6 (27 lines); hunks: -864,6 +864,7 @@ template; -956,7 +957,7 @@ void chunk_gated_delta_rule_fwd_inter_kernel_impl(
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py
@@ -141,9 +141,10 @@ def extend(
-        if is_npu() or is_cpu():
+        if is_npu():
diff -- python/sglang/srt/model_executor/cpu_graph_runner.py
@@ -534,7 +534,8 @@ def _(
-        eps,
+        initial_state_indices,
+        eps=1e-6,
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -503,7 +503,7 @@ def forward_extend(
-            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
+            if is_npu() and last_recurrent_state is not None:
diff -- python/sglang/srt/models/qwen3_5.py
@@ -984,7 +984,7 @@ def self_attention(
-            if not _is_npu:
+            if not (_is_npu or _is_cpu):
diff -- sgl-kernel/csrc/cpu/mamba/fla.cpp
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` modified +2/-1; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +2/-1; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +1/-1; `python/sglang/srt/models/qwen3_5.py` modified +1/-1
  - other: `sgl-kernel/csrc/cpu/mamba/fla.cpp` modified +21/-6; `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` modified +3/-2; `sgl-kernel/python/sgl_kernel/mamba.py` modified +2/-0
  - tests: `test/registered/cpu/test_mamba.py` modified +17/-5
- 验证与风险: diff 自带测试面 `test/registered/cpu/test_mamba.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
