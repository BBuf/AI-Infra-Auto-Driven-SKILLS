# sglang Nemotron Super 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` | [#23907](https://github.com/sgl-project/sglang/pull/23907), [#23968](https://github.com/sgl-project/sglang/pull/23968), [#23998](https://github.com/sgl-project/sglang/pull/23998), [#25198](https://github.com/sgl-project/sglang/pull/25198), [#27240](https://github.com/sgl-project/sglang/pull/27240) |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Super.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` | [#26969](https://github.com/sgl-project/sglang/pull/26969), [#27240](https://github.com/sgl-project/sglang/pull/27240), [#28087](https://github.com/sgl-project/sglang/pull/28087), [#28675](https://github.com/sgl-project/sglang/pull/28675) |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` | [#23907](https://github.com/sgl-project/sglang/pull/23907), [#25198](https://github.com/sgl-project/sglang/pull/25198) |
| `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` | [#27184](https://github.com/sgl-project/sglang/pull/27184) |
| `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` | [#26969](https://github.com/sgl-project/sglang/pull/26969), [#28087](https://github.com/sgl-project/sglang/pull/28087), [#28675](https://github.com/sgl-project/sglang/pull/28675), [#29200](https://github.com/sgl-project/sglang/pull/29200) |
| `python/sglang/srt/arg_groups/nemotron_h_hook.py` | [#24328](https://github.com/sgl-project/sglang/pull/24328), [#25655](https://github.com/sgl-project/sglang/pull/25655), [#26733](https://github.com/sgl-project/sglang/pull/26733) |
| `python/sglang/srt/configs/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/configs/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857), [#25023](https://github.com/sgl-project/sglang/pull/25023) |
| `python/sglang/srt/configs/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#19950](https://github.com/sgl-project/sglang/pull/19950), [#20458](https://github.com/sgl-project/sglang/pull/20458), [#24429](https://github.com/sgl-project/sglang/pull/24429) |
| `python/sglang/srt/models/jet_nemotron.py` | [#12448](https://github.com/sgl-project/sglang/pull/12448) |
| `python/sglang/srt/models/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857), [#25023](https://github.com/sgl-project/sglang/pull/25023) |
| `python/sglang/srt/models/nemotron_h.py` | [#10909](https://github.com/sgl-project/sglang/pull/10909), [#11866](https://github.com/sgl-project/sglang/pull/11866), [#12015](https://github.com/sgl-project/sglang/pull/12015), [#12277](https://github.com/sgl-project/sglang/pull/12277), [#12690](https://github.com/sgl-project/sglang/pull/12690), [#16172](https://github.com/sgl-project/sglang/pull/16172), [#16227](https://github.com/sgl-project/sglang/pull/16227), [#16569](https://github.com/sgl-project/sglang/pull/16569), [#17013](https://github.com/sgl-project/sglang/pull/17013), [#18546](https://github.com/sgl-project/sglang/pull/18546), [#19903](https://github.com/sgl-project/sglang/pull/19903), [#20580](https://github.com/sgl-project/sglang/pull/20580), ... (21 total) |
| `python/sglang/srt/models/nemotron_h_mtp.py` | [#17013](https://github.com/sgl-project/sglang/pull/17013), [#19433](https://github.com/sgl-project/sglang/pull/19433), [#24429](https://github.com/sgl-project/sglang/pull/24429), [#24955](https://github.com/sgl-project/sglang/pull/24955), [#28346](https://github.com/sgl-project/sglang/pull/28346) |
| `python/sglang/srt/models/nemotron_h_utils.py` | [#24955](https://github.com/sgl-project/sglang/pull/24955), [#28102](https://github.com/sgl-project/sglang/pull/28102), [#28346](https://github.com/sgl-project/sglang/pull/28346) |
| `python/sglang/srt/models/nemotron_nas.py` | [#9067](https://github.com/sgl-project/sglang/pull/9067) |
| `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` | [#12277](https://github.com/sgl-project/sglang/pull/12277), [#14051](https://github.com/sgl-project/sglang/pull/14051), [#23568](https://github.com/sgl-project/sglang/pull/23568), [#23857](https://github.com/sgl-project/sglang/pull/23857) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` | [#18119](https://github.com/sgl-project/sglang/pull/18119) |
| `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml` | [#25655](https://github.com/sgl-project/sglang/pull/25655) |
| `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` | [#25655](https://github.com/sgl-project/sglang/pull/25655) |
| `test/manual/models/test_nvidia_nemotron_nano_v2.py` | 无直接 PR 号提交 |
| `test/manual/models/test_nvidia_nemotron_nano_v2_vl.py` | 无直接 PR 号提交 |
| `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` | [#20575](https://github.com/sgl-project/sglang/pull/20575), [#20616](https://github.com/sgl-project/sglang/pull/20616), [#21516](https://github.com/sgl-project/sglang/pull/21516), [#27838](https://github.com/sgl-project/sglang/pull/27838) |
| `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` | [#20616](https://github.com/sgl-project/sglang/pull/20616), [#27838](https://github.com/sgl-project/sglang/pull/27838) |
| `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` | [#23594](https://github.com/sgl-project/sglang/pull/23594) |
| `test/registered/models_e2e/test_nvidia_nemotron_3_nano.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_nvidia_nemotron_3_super_bf16.py` | 无直接 PR 号提交 |
| `test/registered/models_e2e/test_nvidia_nemotron_3_super_bf16_mtp.py` | 无直接 PR 号提交 |
| `test/registered/unit/models/test_nemotron_h_weight_loading.py` | [#24434](https://github.com/sgl-project/sglang/pull/24434), [#26522](https://github.com/sgl-project/sglang/pull/26522) |

## PR 覆盖总览

- git 追溯 PR 数: 48
- 原文档显式引用补充 PR 数: 19
- 当前文档总 PR 数: 67
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-11 | [#5073](https://github.com/sgl-project/sglang/pull/5073) | closed | [Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1 | `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py` |
| 2025-08-17 | [#9067](https://github.com/sgl-project/sglang/pull/9067) | merged | model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1 | `python/sglang/srt/models/nemotron_nas.py` |
| 2025-10-08 | [#10909](https://github.com/sgl-project/sglang/pull/10909) | merged | model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2) | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2025-10-23 | [#11866](https://github.com/sgl-project/sglang/pull/11866) | merged | Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4 | `python/sglang/srt/models/nemotron_h.py` |
| 2025-10-23 | [#12015](https://github.com/sgl-project/sglang/pull/12015) | merged | Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4" | `python/sglang/srt/models/nemotron_h.py` |
| 2025-11-09 | [#12448](https://github.com/sgl-project/sglang/pull/12448) | merged | Add Jet-Nemotron | `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py` |
| 2025-11-21 | [#12690](https://github.com/sgl-project/sglang/pull/12690) | merged | Feat/nemotron nano v3 support | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2025-11-26 | [#12277](https://github.com/sgl-project/sglang/pull/12277) | merged | Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H) | `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py` |
| 2025-12-31 | [#16172](https://github.com/sgl-project/sglang/pull/16172) | merged | [NemotronH] PP support | `python/sglang/srt/models/nemotron_h.py` |
| 2026-01-02 | [#16227](https://github.com/sgl-project/sglang/pull/16227) | merged | [NemotronH] Add latent MoE support | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py` |
| 2026-01-05 | [#14051](https://github.com/sgl-project/sglang/pull/14051) | merged | EVS Framework: Support NemotronH_Nano_VL_V2 | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py` |
| 2026-01-14 | [#17013](https://github.com/sgl-project/sglang/pull/17013) | merged | Feat/support nemotron h mtp | `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py` |
| 2026-01-14 | [#16569](https://github.com/sgl-project/sglang/pull/16569) | merged | [NemotronH] Use ReplicatedLinear for fc1_latent_proj | `python/sglang/srt/models/nemotron_h.py` |
| 2026-02-06 | [#18119](https://github.com/sgl-project/sglang/pull/18119) | merged | Add Nemotron 3 Nano tests | `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` |
| 2026-02-21 | [#18546](https://github.com/sgl-project/sglang/pull/18546) | merged | [Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-03 | [#19433](https://github.com/sgl-project/sglang/pull/19433) | merged | Fix/nemotron mtp quantaized | `python/sglang/srt/models/nemotron_h_mtp.py` |
| 2026-03-07 | [#19950](https://github.com/sgl-project/sglang/pull/19950) | merged | Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support | `python/sglang/srt/configs/nemotron_h.py` |
| 2026-03-12 | [#19903](https://github.com/sgl-project/sglang/pull/19903) | merged | Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-14 | [#20407](https://github.com/sgl-project/sglang/pull/20407) | merged | [Model] Support Nemotron 3 Super NVFP4 | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py` |
| 2026-03-14 | [#20575](https://github.com/sgl-project/sglang/pull/20575) | merged | [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4 | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` |
| 2026-03-15 | [#20458](https://github.com/sgl-project/sglang/pull/20458) | merged | fix: Nemotron chunk size alias | `python/sglang/srt/configs/nemotron_h.py` |
| 2026-03-16 | [#20616](https://github.com/sgl-project/sglang/pull/20616) | merged | [CI] Add Nemotron 3 Super 120B nightly 8-GPU tests | `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` |
| 2026-03-17 | [#20580](https://github.com/sgl-project/sglang/pull/20580) | merged | [Model] Fix NemotronH OOM on unified-mem systems: stream weights | `python/sglang/srt/models/nemotron_h.py` |
| 2026-03-27 | [#21516](https://github.com/sgl-project/sglang/pull/21516) | merged | [CI] Fix nemotron nvfp4 test estimated time | `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` |
| 2026-04-25 | [#23568](https://github.com/sgl-project/sglang/pull/23568) | merged | Parakeet nemotron encoder | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py` |
| 2026-04-28 | [#23907](https://github.com/sgl-project/sglang/pull/23907) | merged | [Docs] add Nemotron 3 Nano Omni cookbook | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` |
| 2026-04-28 | [#23874](https://github.com/sgl-project/sglang/pull/23874) | merged | Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk` | `python/sglang/srt/models/nemotron_h.py` |
| 2026-04-28 | [#23968](https://github.com/sgl-project/sglang/pull/23968) | merged | [Docs] update Docker image for Nemotron 3 Nano Omni | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-04-29 | [#23857](https://github.com/sgl-project/sglang/pull/23857) | merged | Nemotron-omni-v3-alias | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py` |
| 2026-04-29 | [#21321](https://github.com/sgl-project/sglang/pull/21321) | merged | [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron) | `python/sglang/srt/models/nemotron_h.py` |
| 2026-04-30 | [#23594](https://github.com/sgl-project/sglang/pull/23594) | merged | LoRA support for qwen3.5 and nemotron3 | `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` |
| 2026-04-30 | [#24163](https://github.com/sgl-project/sglang/pull/24163) | merged | Revert "[ci] split stage-c-test-4-gpu-b200 to enable a low-disk runner pool" | `.github/workflows/pr-test.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-05-03 | [#24328](https://github.com/sgl-project/sglang/pull/24328) | merged | introduce arg_groups/ with nemotron_h hook | `python/sglang/srt/arg_groups/nemotron_h_hook.py` |
| 2026-05-05 | [#23998](https://github.com/sgl-project/sglang/pull/23998) | merged | update Nemotron3 Nano Omni cookbook benchmarks | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-05-08 | [#24434](https://github.com/sgl-project/sglang/pull/24434) | merged | [NemotronH] Fix expert scale weight loading | `test/registered/unit/models/test_nemotron_h_weight_loading.py`, `python/sglang/srt/models/nemotron_h.py` |
| 2026-05-08 | [#24721](https://github.com/sgl-project/sglang/pull/24721) | merged | ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/ | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/manual/models/test_nvidia_nemotron_nano_v2.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-14 | [#25197](https://github.com/sgl-project/sglang/pull/25197) | merged | ci: decouple stage and runner for cuda registry | `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py` |
| 2026-05-14 | [#25203](https://github.com/sgl-project/sglang/pull/25203) | merged | ci: B200 conditional split + LPT_SLOP removal (stage-c partition 8→3) | `scripts/ci/utils/compute_partitions.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` |
| 2026-05-14 | [#25236](https://github.com/sgl-project/sglang/pull/25236) | merged | ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2) | `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-05-14 | [#24725](https://github.com/sgl-project/sglang/pull/24725) | merged | ci: tag-gated nightly migration — foundation + 40 whole-file moves | `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py` |
| 2026-05-16 | [#25420](https://github.com/sgl-project/sglang/pull/25420) | merged | [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI | `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py` |
| 2026-05-20 | [#25831](https://github.com/sgl-project/sglang/pull/25831) | merged | [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py` |
| 2026-05-21 | [#25983](https://github.com/sgl-project/sglang/pull/25983) | merged | feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-05-26 | [#15829](https://github.com/sgl-project/sglang/pull/15829) | merged | [feat] Support `extra_buffer` in Mamba2-based models | `test/manual/models/test_nvidia_nemotron_nano_v2.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `test/manual/models/test_granite_moe_hybrid.py` |
| 2026-05-26 | [#25023](https://github.com/sgl-project/sglang/pull/25023) | merged | [NemotronH] V3 Omni wrapper: WeightsMapper + config round-trip | `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py` |
| 2026-05-27 | [#24429](https://github.com/sgl-project/sglang/pull/24429) | merged | Support NemotronHPuzzleForCausalLM | `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py` |
| 2026-05-28 | [#26522](https://github.com/sgl-project/sglang/pull/26522) | merged | [NemotronH] Fix weight-loading unit test broken by Puzzle support | `test/registered/unit/models/test_nemotron_h_weight_loading.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-06-03 | [#25655](https://github.com/sgl-project/sglang/pull/25655) | merged | Feat/add w4a16 moe support to nemotron | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml` |
| 2026-06-03 | [#27184](https://github.com/sgl-project/sglang/pull/27184) | merged | docs: fix Nemotron Super MTP deployment command (spec-v2 + B200) | `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` |
| 2026-06-03 | [#25198](https://github.com/sgl-project/sglang/pull/25198) | merged | [Docs] Update Nemotron3-Nano-Omni cookbook to reflect new model paths | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` |
| 2026-06-04 | [#26969](https://github.com/sgl-project/sglang/pull/26969) | merged | docs: add Nemotron 3 Ultra cookbook entry | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` |
| 2026-06-04 | [#27240](https://github.com/sgl-project/sglang/pull/27240) | merged | [Docs] re-organize nemotron cookbook | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |
| 2026-06-06 | [#26733](https://github.com/sgl-project/sglang/pull/26733) | merged | Nemotron perf changes | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-10 | [#27838](https://github.com/sgl-project/sglang/pull/27838) | merged | Disable async assert in Nemotron nightly tests | `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` |
| 2026-06-12 | [#24955](https://github.com/sgl-project/sglang/pull/24955) | merged | Support Nemotron DP attention and MTP | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py` |
| 2026-06-12 | [#28087](https://github.com/sgl-project/sglang/pull/28087) | merged | [Doc] Fix some inconsistencies in the Nemotron Cookbook | `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` |
| 2026-06-13 | [#28102](https://github.com/sgl-project/sglang/pull/28102) | merged | Fix DP attention + EP mode of Nemotron | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28346](https://github.com/sgl-project/sglang/pull/28346) | merged | Use Flashinfer allreduce fusion for MNNVL allreduce for Nemotron | `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-22 | [#28675](https://github.com/sgl-project/sglang/pull/28675) | merged | [Cookbook] Nemotron3-Ultra: Add mamba-backend and SSM dtype flags | `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` |
| 2026-06-24 | [#29200](https://github.com/sgl-project/sglang/pull/29200) | merged | [Cookbook] Nemotron3-Ultra: align MTP draft depth with NVIDIA reference (num_steps 5) | `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` |
| 2026-06-25 | [#29261](https://github.com/sgl-project/sglang/pull/29261) | merged | [Docs] Fix broken links in cookbook | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` |

## 逐 PR diff 审计卡

### PR #5073 - [Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1

- 链接: https://github.com/sgl-project/sglang/pull/5073
- 状态/时间: closed / 2025-08-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+898/-1，可读 patch 929 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py`；技术摘要: 覆盖「[Model] Add support for nvidia/Llama-3_3-Nemotron-Super-49B-v1」；主要实现面是 `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_nas.py` added +516/-0 (516 lines); hunks: -0,0 +1,516; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__，涉及 `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`；`python/sglang/srt/configs/model_config.py` modified +8/-0 (8 lines); hunks: -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:; symbols: get_total_num_kv_heads，涉及 `get_total_num_kv_heads`；`python/sglang/srt/utils.py` modified +374/-1 (375 lines); hunks: -55,6 +55,8; -439,8 +441,10 @@ def set_cpu_offload_max_bytes(max_bytes: int) -> None:; symbols: set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn, __call__，涉及 `set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_nas.py` added +516/-0 (516 lines); hunks: -0,0 +1,516; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
  - `python/sglang/srt/configs/model_config.py` modified +8/-0 (8 lines); hunks: -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:; symbols: get_total_num_kv_heads
  - `python/sglang/srt/utils.py` modified +374/-1 (375 lines); hunks: -55,6 +55,8; -439,8 +441,10 @@ def set_cpu_offload_max_bytes(max_bytes: int) -> None:; symbols: set_cpu_offload_max_bytes, maybe_offload_to_cpu, LayerFn, __call__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_nas.py
@@ -0,0 +1,516 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/model_config.py
@@ -320,6 +320,14 @@ def get_total_num_kv_heads(self) -> int:
+        if self.hf_config.model_type in ["nemotron-nas"]:
+            for block in self.hf_config.block_configs:
+                if not block.attention.no_op:
+                    return (
+                        self.hf_config.num_attention_heads
+                        // block.attention.n_heads_in_group
diff -- python/sglang/srt/utils.py
@@ -55,6 +55,8 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_nas.py` added +516/-0; `python/sglang/srt/configs/model_config.py` modified +8/-0; `python/sglang/srt/utils.py` modified +374/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/nemotron_nas.py`, `python/sglang/srt/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9067 - model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1

- 链接: https://github.com/sgl-project/sglang/pull/9067
- 状态/时间: merged / 2025-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_nas.py`；关联提交 `845d12a979fb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+465/-5，可读 patch 505 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_nas.py`；技术摘要: 覆盖「model: support nvidia/Llama-3_3-Nemotron-Super-49B-v1」；主要实现面是 `python/sglang/srt/models/nemotron_nas.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_nas.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__，涉及 `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_nas.py` added +435/-0 (435 lines); hunks: -0,0 +1,435; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_nas.py
@@ -0,0 +1,435 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_nas.py` added +435/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/models/test_generation_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10909 - model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)

- 链接: https://github.com/sgl-project/sglang/pull/10909
- 状态/时间: merged / 2025-10-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6837aea4d2c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+3279/-853，可读 patch 4929 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2)」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer，涉及 `NemotronHMLP, __init__, forward`；`python/sglang/srt/configs/nemotron_h.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: NemotronHConfig, to, __init__, mamba_layer_ids，涉及 `NemotronHConfig, to, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` added +514/-0 (514 lines); hunks: -0,0 +1,514; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `python/sglang/srt/configs/nemotron_h.py` added +286/-0 (286 lines); hunks: -0,0 +1,286; symbols: NemotronHConfig, to, __init__, mamba_layer_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -0,0 +1,514 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -0,0 +1,286 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` added +514/-0; `python/sglang/srt/configs/nemotron_h.py` added +286/-0
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11866 - Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/11866
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `d6fee73d1f59`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+207/-127，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +19/-22 (41 lines); hunks: -48,6 +48,8; -155,6 +157,7 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model，涉及 `__init__, forward, NemotronHForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +19/-22 (41 lines); hunks: -48,6 +48,8; -155,6 +157,7 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,6 +48,8 @@
+    replace_prefix,
+    replace_substrings,
@@ -155,6 +157,7 @@ def __init__(
+            prefix=f"{prefix}.mixer",
@@ -381,16 +384,19 @@ def forward(
+    stacked_params_mapping = [
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +19/-22
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12015 - Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"

- 链接: https://github.com/sgl-project/sglang/pull/12015
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `6c18addb6f53`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+127/-207，可读 patch 628 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Revert "Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4"」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +22/-19 (41 lines); hunks: -48,8 +48,6; -157,7 +155,6 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model，涉及 `__init__, forward, NemotronHForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +22/-19 (41 lines); hunks: -48,8 +48,6; -157,7 +155,6 @@ def __init__(; symbols: __init__, forward, NemotronHForCausalLM, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,8 +48,6 @@
-    replace_prefix,
-    replace_substrings,
@@ -157,7 +155,6 @@ def __init__(
-            prefix=f"{prefix}.mixer",
@@ -384,19 +381,16 @@ def forward(
-    stacked_params_mapping = [
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +22/-19
- 验证与风险: diff 自带测试面 `test/srt/layers/attention/mamba/test_causal_conv1d.py`, `test/srt/layers/attention/mamba/test_mamba2_mixer.py`, `test/srt/layers/attention/mamba/test_mamba_ssm.py`, `test/srt/layers/attention/mamba/test_mamba_ssm_ssd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12448 - Add Jet-Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/12448
- 状态/时间: merged / 2025-11-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/jet_nemotron.py`, `python/sglang/srt/models/jet_nemotron.py`；关联提交 `3633f8b0cfef`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+678/-2，可读 patch 733 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Jet-Nemotron」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py`；技术摘要: 覆盖「Add Jet-Nemotron」；主要实现面是 `python/sglang/srt/models/jet_nemotron.py`, `python/sglang/srt/configs/jet_nemotron.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/jet_nemotron.py` added +596/-0 (596 lines); hunks: -0,0 +1,596; symbols: DynamicShortConvolutionKernelGenerator, __init__, forward, DynamicShortConvolution，涉及 `DynamicShortConvolutionKernelGenerator, __init__, forward`；`python/sglang/srt/configs/jet_nemotron.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: JetBlockConfig, JetNemotronConfig, full_attention_layer_ids, linear_layer_ids，涉及 `JetBlockConfig, JetNemotronConfig, full_attention_layer_ids`。
- 代码 diff 细节:
  - `python/sglang/srt/models/jet_nemotron.py` added +596/-0 (596 lines); hunks: -0,0 +1,596; symbols: DynamicShortConvolutionKernelGenerator, __init__, forward, DynamicShortConvolution
  - `python/sglang/srt/configs/jet_nemotron.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: JetBlockConfig, JetNemotronConfig, full_attention_layer_ids, linear_layer_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/jet_nemotron.py
@@ -0,0 +1,596 @@
+from collections.abc import Iterable
+from typing import cast
+import einops
+import torch
+import torch.nn as nn
+from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
diff -- python/sglang/srt/configs/jet_nemotron.py
@@ -0,0 +1,74 @@
+from dataclasses import dataclass
+from typing import Any
+from transformers.configuration_utils import PretrainedConfig
+from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
+@dataclass
+class JetBlockConfig:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/jet_nemotron.py` added +596/-0; `python/sglang/srt/configs/jet_nemotron.py` added +74/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12690 - Feat/nemotron nano v3 support

- 链接: https://github.com/sgl-project/sglang/pull/12690
- 状态/时间: merged / 2025-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `1b48e1b97484`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+775/-67，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat/nemotron nano v3 support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「Feat/nemotron nano v3 support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +259/-28 (287 lines); hunks: -22,8 +22,13; -34,9 +39,13; symbols: NemotronHMLP, __init__, forward, _get_or_create_alt_stream，涉及 `NemotronHMLP, __init__, forward`；`python/sglang/srt/configs/nemotron_h.py` modified +25/-6 (31 lines); hunks: -26,6 +26,7; -189,6 +190,15 @@ def __init__(; symbols: NemotronHConfig, __init__，涉及 `NemotronHConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +259/-28 (287 lines); hunks: -22,8 +22,13; -34,9 +39,13; symbols: NemotronHMLP, __init__, forward, _get_or_create_alt_stream
  - `python/sglang/srt/configs/nemotron_h.py` modified +25/-6 (31 lines); hunks: -26,6 +26,7; -189,6 +190,15 @@ def __init__(; symbols: NemotronHConfig, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -22,8 +22,13 @@
-from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP
-from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
+from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP, MOE
+from sglang.srt.distributed import (
+    get_moe_ep_group,
+    get_pp_group,
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -26,6 +26,7 @@
+MOE = "E"
@@ -189,6 +190,15 @@ def __init__(
+        n_routed_experts=8,
+        n_shared_experts=1,
+        moe_intermediate_size=7688,
+        moe_shared_expert_intermediate_size=7688,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +259/-28; `python/sglang/srt/configs/nemotron_h.py` modified +25/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_L40S.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12277 - Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)

- 链接: https://github.com/sgl-project/sglang/pull/12277
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `082b54c6890a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1334/-17，可读 patch 1528 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；技术摘要: 覆盖「Support nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (and nvidia/C-RADIOv2-H)」；主要实现面是 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0 (219 lines); hunks: -0,0 +1,219; symbols: NemotronH_Nano_VL_V2, __init__, pad_input_ids, pixel_shuffle，涉及 `NemotronH_Nano_VL_V2, __init__, pad_input_ids`；`python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image，涉及 `NanoNemotronVLImageProcessor, __init__, preprocess_image`；`python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: float_triplet, NemotronH_Nano_VL_V2_Config, __init__, create_radio_config，涉及 `float_triplet, NemotronH_Nano_VL_V2_Config, __init__`；`python/sglang/srt/models/nemotron_h.py` modified +3/-6 (9 lines); hunks: -542,9 +542,6 @@ def get_layer(idx: int, prefix: str):; -557,7 +554,7 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, _init_model，涉及 `get_layer, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0 (219 lines); hunks: -0,0 +1,219; symbols: NemotronH_Nano_VL_V2, __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image
  - `python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: float_triplet, NemotronH_Nano_VL_V2_Config, __init__, create_radio_config
  - `python/sglang/srt/models/nemotron_h.py` modified +3/-6 (9 lines); hunks: -542,9 +542,6 @@ def get_layer(idx: int, prefix: str):; -557,7 +554,7 @@ def forward(; symbols: get_layer, get_input_embeddings, forward, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -0,0 +1,219 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -0,0 +1,197 @@
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -0,0 +1,114 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nano_nemotron_vl.py` added +219/-0; `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` added +197/-0; `python/sglang/srt/configs/nano_nemotron_vl.py` added +114/-0; `python/sglang/srt/models/nemotron_h.py` modified +3/-6
- 验证与风险: diff 自带测试面 `test/srt/models/test_nvidia_nemotron_nano_v2_vl.py`, `test/srt/run_suite.py`, `test/srt/test_video_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16172 - [NemotronH] PP support

- 链接: https://github.com/sgl-project/sglang/pull/16172
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `47a660d5b925`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+94/-35，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] PP support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] PP support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunks: -48,6 +48,7; -65,7 +66,7; symbols: __init__, get_layer, forward，涉及 `__init__, get_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +88/-35 (123 lines); hunks: -48,6 +48,7; -65,7 +66,7; symbols: __init__, get_layer, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -48,6 +48,7 @@
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -65,7 +66,7 @@
-    make_layers_non_pp,
+    make_layers,
@@ -526,21 +527,32 @@ def __init__(
+        self.pp_group = get_pp_group()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +88/-35
- 验证与风险: diff 自带测试面 `test/srt/models/test_nvidia_nemotron_nano_v2.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16227 - [NemotronH] Add latent MoE support

- 链接: https://github.com/sgl-project/sglang/pull/16227
- 状态/时间: merged / 2026-01-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `b0213323397c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+2957/-2，可读 patch 3056 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Add latent MoE support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Add latent MoE support」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +32/-1 (33 lines); hunks: -138,6 +138,10 @@ def __init__(; -165,7 +169,7 @@ def __init__(; symbols: __init__, _forward_core, _forward_core_normal，涉及 `__init__, _forward_core, _forward_core_normal`；`python/sglang/srt/configs/nemotron_h.py` modified +2/-0 (2 lines); hunks: -194,6 +194,7 @@ def __init__(; -259,6 +260,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +32/-1 (33 lines); hunks: -138,6 +138,10 @@ def __init__(; -165,7 +169,7 @@ def __init__(; symbols: __init__, _forward_core, _forward_core_normal
  - `python/sglang/srt/configs/nemotron_h.py` modified +2/-0 (2 lines); hunks: -194,6 +194,7 @@ def __init__(; -259,6 +260,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -138,6 +138,10 @@ def __init__(
+        self.use_latent_moe = getattr(config, "moe_latent_size", None) is not None
+        self.moe_hidden_size = (
+            config.moe_latent_size if self.use_latent_moe else config.hidden_size
+        )
@@ -165,7 +169,7 @@ def __init__(
-            hidden_size=config.hidden_size,
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -194,6 +194,7 @@ def __init__(
+        moe_latent_size=None,
@@ -259,6 +260,7 @@ def __init__(
+        self.moe_latent_size = moe_latent_size
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +32/-1; `python/sglang/srt/configs/nemotron_h.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14051 - EVS Framework: Support NemotronH_Nano_VL_V2

- 链接: https://github.com/sgl-project/sglang/pull/14051
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `bebd625ba145`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+821/-56，可读 patch 1171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「EVS Framework: Support NemotronH_Nano_VL_V2」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；技术摘要: 覆盖「EVS Framework: Support NemotronH_Nano_VL_V2」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22 (57 lines); hunks: -11,14 +11,16; -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image，涉及 `NanoNemotronVLImageProcessor, __init__, preprocess_image`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2 (9 lines); hunks: -36,19 +36,24; symbols: NemotronH_Nano_VL_V2, create_evs_config, __init__，涉及 `NemotronH_Nano_VL_V2, create_evs_config, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22 (57 lines); hunks: -11,14 +11,16; -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):; symbols: NanoNemotronVLImageProcessor, __init__, preprocess_image, render_image
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2 (9 lines); hunks: -36,19 +36,24; symbols: NemotronH_Nano_VL_V2, create_evs_config, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,14 +11,16 @@
+from math import sqrt
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
+from sglang.srt.multimodal.evs import EVSProcessor
@@ -40,6 +42,9 @@ class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
+        self.evs = EVSProcessor(
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -36,19 +36,24 @@
+from sglang.srt.multimodal.evs import EVS, EVSConfig
-class NemotronH_Nano_VL_V2(nn.Module):
+class NemotronH_Nano_VL_V2(EVS):
+    @staticmethod
+    def create_evs_config(config: NemotronH_Nano_VL_V2_Config):
+        return EVSConfig(video_pruning_rate=config.video_pruning_rate)
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +35/-22; `python/sglang/srt/models/nano_nemotron_vl.py` modified +7/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_evs.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17013 - Feat/support nemotron h mtp

- 链接: https://github.com/sgl-project/sglang/pull/17013
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `ba625c2d908a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+378/-1，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat/support nemotron h mtp」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Feat/support nemotron h mtp」；主要实现面是 `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer，涉及 `NemotronHMTPAttentionDecoderLayer, __init__, forward`；`python/sglang/srt/models/nemotron_h.py` modified +28/-1 (29 lines); hunks: -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **k...; -749,6 +762,20 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights, get_embed_and_head，涉及 `copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `python/sglang/srt/models/nemotron_h.py` modified +28/-1 (29 lines); hunks: -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **k...; -749,6 +762,20 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: copy_inputs_before_cuda_graphs, get_seqlen_agnostic_capture_inputs, load_weights, get_embed_and_head
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -0,0 +1,340 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/nemotron_h.py
@@ -728,7 +728,20 @@ def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
-    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
+    def get_embed_and_head(self):
+        return self.model.embed_tokens.weight, self.lm_head.weight
+    def set_embed_and_head(self, embed, head):
+        del self.model.embed_tokens.weight
+        del self.lm_head.weight
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h_mtp.py` added +340/-0; `python/sglang/srt/models/nemotron_h.py` modified +28/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16569 - [NemotronH] Use ReplicatedLinear for fc1_latent_proj

- 链接: https://github.com/sgl-project/sglang/pull/16569
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `72bacc88c8a0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +1/-2 (3 lines); hunks: -191,12 +191,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +1/-2 (3 lines); hunks: -191,12 +191,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -191,12 +191,11 @@ def __init__(
-            self.fc1_latent_proj = ColumnParallelLinear(
+            self.fc1_latent_proj = ReplicatedLinear(
-                gather_output=True,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18119 - Add Nemotron 3 Nano tests

- 链接: https://github.com/sgl-project/sglang/pull/18119
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`；关联提交 `c6aa1863be84`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+177/-0，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Nemotron 3 Nano tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`；技术摘要: 覆盖「Add Nemotron 3 Nano tests」；主要实现面是 `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13；`test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13。
- 代码 diff 细节:
  - `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
  - `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml
@@ -0,0 +1,13 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
+tasks:
+- name: "gsm8k"
+  metrics:
+  - name: "exact_match,strict-match"
+    value: 0.847
diff -- test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml
@@ -0,0 +1,13 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
+tasks:
+- name: "gsm8k"
+  metrics:
+  - name: "exact_match,strict-match"
+    value: 0.847
```

- 已读文件:
  - tests: `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml` added +13/-0; `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml` added +13/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/lm_eval_kit.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18546 - [Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/18546
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `33c33a7de9bb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+100/-71，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Quantization] Support config.json quantization_config format, fix exclude_modules matching, and fix KV cache scale loading for Nemotron」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +7/-0 (7 lines); hunks: -61,6 +61,7; -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):; symbols: NemotronHForCausalLM, __init__，涉及 `NemotronHForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +7/-0 (7 lines); hunks: -61,6 +61,7; -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):; symbols: NemotronHForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -61,6 +61,7 @@
+from sglang.srt.models.utils import WeightsMapper
@@ -640,6 +641,12 @@ class NemotronHForCausalLM(nn.Module):
+    hf_to_sglang_mapper = WeightsMapper(
+        orig_to_new_prefix={
+            "backbone.": "model.",
+        }
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19433 - Fix/nemotron mtp quantaized

- 链接: https://github.com/sgl-project/sglang/pull/19433
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `4c95953b7733`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+73/-3，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix/nemotron mtp quantaized」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Fix/nemotron mtp quantaized」；主要实现面是 `python/sglang/srt/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1 (2 lines); hunks: -297,7 +297,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1 (2 lines); hunks: -297,7 +297,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -297,7 +297,7 @@ def __init__(
-            prefix=add_prefix("model", prefix),
+            prefix=add_prefix("mtp", prefix),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/model_loading/test_modelopt_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19950 - Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support

- 链接: https://github.com/sgl-project/sglang/pull/19950
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `f8bbf56de7b2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+182/-17，可读 patch 281 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「Refactor NemotronHConfig to canonical layers_block_type and add MTP block-type support」；主要实现面是 `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/nemotron_h.py` modified +182/-17 (199 lines); hunks: -15,7 +15,6; -31,6 +30,8; symbols: NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type，涉及 `NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/nemotron_h.py` modified +182/-17 (199 lines); hunks: -15,7 +15,6; -31,6 +30,8; symbols: NemotronHConfig, _validate_layers_block_type, _resolve_layers_block_type
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -15,7 +15,6 @@
-import regex as re
@@ -31,6 +30,8 @@
+DEFAULT_LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
+DEFAULT_MTP_LAYERS_BLOCK_TYPE = ["attention", "moe"]
@@ -53,13 +54,17 @@ class NemotronHConfig(PretrainedConfig):
-        num_hidden_layers (`int`, *optional*, defaults to 52):
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/nemotron_h.py` modified +182/-17
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19903 - Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models

- 链接: https://github.com/sgl-project/sglang/pull/19903
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `25bd83033d09`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+91/-24，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Enable Piecewise CUDA Graph for NemotronH Hybrid (Mamba+Attention) Models」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunks: -21,6 +21,11; -69,6 +74,7; symbols: _forward_core, __init__, _forward_mamba, forward，涉及 `_forward_core, __init__, _forward_mamba`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +70/-18 (88 lines); hunks: -21,6 +21,11; -69,6 +74,7; symbols: _forward_core, __init__, _forward_mamba, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -21,6 +21,11 @@
+from sglang.srt.compilation.compilation_config import register_split_op
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -69,6 +74,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +70/-18
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20407 - [Model] Support Nemotron 3 Super NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/20407
- 状态/时间: merged / 2026-03-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+277/-11，可读 patch 413 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Nemotron 3 Super NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`；技术摘要: 覆盖「[Model] Support Nemotron 3 Super NVFP4」；主要实现面是 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/quantization/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunks: -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):; symbols: __init__, ModelOptMixedPrecisionConfig, override_quantization_method, get_name，涉及 `__init__, ModelOptMixedPrecisionConfig, override_quantization_method`；`python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunks: -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: d...; -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:; symbols: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization，涉及 `_parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config`；`python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):; -57,6 +58,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method，涉及 `override_quantization_method`；`test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunks: -14,7 +14,11; -620,5 +624,66 @@ def test_non_modelopt_quant_method_unchanged(self):; symbols: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8，涉及 `test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0 (177 lines); hunks: -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):; symbols: __init__, ModelOptMixedPrecisionConfig, override_quantization_method, get_name
  - `python/sglang/srt/configs/model_config.py` modified +12/-0 (12 lines); hunks: -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: d...; -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:; symbols: _parse_modelopt_quant_config, _get_modelopt_quant_type, _validate_quantize_and_serve_config, _verify_quantization
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):; -57,6 +58,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0 (65 lines); hunks: -14,7 +14,11; -620,5 +624,66 @@ def test_non_modelopt_quant_method_unchanged(self):; symbols: test_non_modelopt_quant_method_unchanged, TestModelOptMixedPrecisionConfig, test_nemotron_mixed_precision_uses_modelopt_mixed, test_mixed_precision_override_does_not_hijack_w4afp8
  - `python/sglang/srt/server_args.py` modified +17/-9 (26 lines); hunks: -105,6 +105,7; -1546,7 +1547,8 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_moe_kernel_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -591,6 +591,183 @@ def __init__(self, quant_config: ModelOptFp8Config):
+class ModelOptMixedPrecisionConfig(ModelOptQuantConfig):
+    """Configuration for ModelOpt MIXED_PRECISION checkpoints."""
+    def __init__(
+        self,
+        kv_cache_quant_algo: Optional[str],
+        exclude_modules: Optional[List[str]],
diff -- python/sglang/srt/configs/model_config.py
@@ -793,6 +793,11 @@ def _parse_modelopt_quant_config(self, quant_config_dict: dict) -> Optional[dict
+            architectures = getattr(self.hf_config, "architectures", []) or []
+            if getattr(self.hf_config, "model_type", None) == "nemotron_h" or any(
+                arch.startswith("NemotronH") for arch in architectures
+            ):
+                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
@@ -842,6 +847,10 @@ def _get_modelopt_quant_type(self) -> str:
diff -- python/sglang/srt/layers/quantization/__init__.py
@@ -31,6 +31,7 @@ def override_quantization_method(self, *args, **kwargs):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +177/-0; `python/sglang/srt/configs/model_config.py` modified +12/-0; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +17/-9; `python/sglang/srt/model_loader/loader.py` modified +4/-2
  - tests: `test/registered/model_loading/test_modelopt_loader.py` modified +65/-0
- 验证与风险: diff 自带测试面 `test/registered/model_loading/test_modelopt_loader.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20575 - [CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/20575
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；关联提交 `3e643967e6d7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+212/-0，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；技术摘要: 覆盖「[CI] Add Nemotron 3 Super 120B CI tests for BF16 and NVFP4」；主要实现面是 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass，涉及 `_run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass`。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _run_gsm8k, TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -0,0 +1,106 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
+from sglang.test.test_utils import (
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` added +106/-0
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20458 - fix: Nemotron chunk size alias

- 链接: https://github.com/sgl-project/sglang/pull/20458
- 状态/时间: merged / 2026-03-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`；关联提交 `1ac6a2646437`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-1，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Nemotron chunk size alias」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/nemotron_h.py`；技术摘要: 覆盖「fix: Nemotron chunk size alias」；主要实现面是 `python/sglang/srt/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/nemotron_h.py` modified +26/-1 (27 lines); hunks: -32,6 +32,7; -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, k...; symbols: NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size, __init__，涉及 `NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/nemotron_h.py` modified +26/-1 (27 lines); hunks: -32,6 +32,7; -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, k...; symbols: NemotronHConfig, _resolve_mtp_layers_block_type, _resolve_mamba_chunk_size, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -32,6 +32,7 @@
+DEFAULT_MAMBA_CHUNK_SIZE = 256
@@ -213,6 +214,28 @@ def _resolve_mtp_layers_block_type(mtp_layers_block_type, kwargs) -> list[str]:
+    @staticmethod
+    def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:
+        """Resolve canonical mamba_chunk_size from new and legacy config fields."""
+        chunk_size = kwargs.pop("chunk_size", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/nemotron_h.py` modified +26/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20616 - [CI] Add Nemotron 3 Super 120B nightly 8-GPU tests

- 链接: https://github.com/sgl-project/sglang/pull/20616
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；关联提交 `3879c466b432`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+145/-6，可读 patch 180 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Nemotron 3 Super 120B nightly 8-GPU tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；技术摘要: 覆盖「[CI] Add Nemotron 3 Super 120B nightly 8-GPU tests」；主要实现面是 `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4，涉及 `TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16`；`test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: TestNvidiaNemotron3SuperNightly, for, test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3 (8 lines); hunks: -37,6 +37,10; -89,9 +93,7 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py
@@ -0,0 +1,135 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -37,6 +37,10 @@
+    "--max-running-requests",
+    "200",
+    "--mem-fraction-static",
+    "0.75",
@@ -89,9 +93,7 @@ def setUpClass(cls):
-            other_args=NEMOTRON_3_SUPER_NVFP4_ARGS
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` added +135/-0; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +5/-3
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20580 - [Model] Fix NemotronH OOM on unified-mem systems: stream weights

- 链接: https://github.com/sgl-project/sglang/pull/20580
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `466ff20e5148`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-7，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix NemotronH OOM on unified-mem systems: stream weights」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Model] Fix NemotronH OOM on unified-mem systems: stream weights」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +7/-7 (14 lines); hunks: -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):; -793,7 +787,13 @@ def load_weights(; symbols: set_embed_and_head, load_weights，涉及 `set_embed_and_head, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +7/-7 (14 lines); hunks: -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):; -793,7 +787,13 @@ def load_weights(; symbols: set_embed_and_head, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -774,12 +774,6 @@ def set_embed_and_head(self, embed, head):
-        updated_weights = []
-        for name, loaded_weight in weights:
-            name = replace_prefix(name, self.remap_prefix)
-            name = replace_substrings(name, self.remap_substr)
-            updated_weights.append((name, loaded_weight))
@@ -793,7 +787,13 @@ def load_weights(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +7/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21516 - [CI] Fix nemotron nvfp4 test estimated time

- 链接: https://github.com/sgl-project/sglang/pull/21516
- 状态/时间: merged / 2026-03-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；关联提交 `0138129d3cfc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix nemotron nvfp4 test estimated time」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；技术摘要: 覆盖「[CI] Fix nemotron nvfp4 test estimated time」；主要实现面是 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7。
- 代码 diff 细节:
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7
- 关键代码摘录:

```diff
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
-register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-b200")
```

- 已读文件:
  - tests: `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23568 - Parakeet nemotron encoder

- 链接: https://github.com/sgl-project/sglang/pull/23568
- 状态/时间: merged / 2026-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `4a3fe2a0913c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1289/-116，可读 patch 1817 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Parakeet nemotron encoder」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；技术摘要: 覆盖「Parakeet nemotron encoder」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic，涉及 `__init__, preprocess_image, render_image`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle，涉及 `__init__, pad_input_ids, pixel_shuffle`；`python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0 (38 lines); hunks: -38,6 +38,7 @@ def __init__(; -51,6 +52,9 @@ def __init__(; symbols: __init__, create_radio_config，涉及 `__init__, create_radio_config`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0 (38 lines); hunks: -38,6 +38,7 @@ def __init__(; -51,6 +52,9 @@ def __init__(; symbols: __init__, create_radio_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,23 +11,39 @@
+import logging
+import math
-from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -35,8 +35,10 @@
+from sglang.srt.models.parakeet import ProjectedParakeet
+from sglang.srt.multimodal.evs.evs_module import VideoEVSDataItem
@@ -66,9 +68,13 @@ def __init__(
-        self.rmsnorm_hidden_size = vit_hidden_size * int(1 / self.downsample_ratio) ** 2
+        self.rmsnorm_hidden_size = (
+            vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -38,6 +38,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36; `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20; `python/sglang/srt/configs/nano_nemotron_vl.py` modified +38/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/configs/parakeet.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23907 - [Docs] add Nemotron 3 Nano Omni cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23907
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；关联提交 `ad785a229911`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+746/-1，可读 patch 771 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] add Nemotron 3 Nano Omni cookbook」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；技术摘要: 覆盖「[Docs] add Nemotron 3 Nano Omni cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542；`docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0 (542 lines); hunks: -0,0 +1,542
  - `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0 (200 lines); hunks: -0,0 +1,200
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` added +542/-0; `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` added +200/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/cookbook/intro copy.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23874 - Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`

- 链接: https://github.com/sgl-project/sglang/pull/23874
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `ddcacaf1bd4e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+223/-19，可读 patch 282 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk`」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -923,6 +923,8 @@ def nemotron_mamba2_with_output(; symbols: nemotron_mamba2_with_output，涉及 `nemotron_mamba2_with_output`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -923,6 +923,8 @@ def nemotron_mamba2_with_output(; symbols: nemotron_mamba2_with_output
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -923,6 +923,8 @@ def nemotron_mamba2_with_output(
+    if output.shape[0] != num_actual_tokens:
+        output[num_actual_tokens:].zero_()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +2/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_grouped_topk.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23968 - [Docs] update Docker image for Nemotron 3 Nano Omni

- 链接: https://github.com/sgl-project/sglang/pull/23968
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；关联提交 `387c932dfc88`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] update Docker image for Nemotron 3 Nano Omni」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「[Docs] update Docker image for Nemotron 3 Nano Omni」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ pip install sglang
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ pip install sglang
-docker pull lmsysorg/sglang:nightly
+docker pull lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23857 - Nemotron-omni-v3-alias

- 链接: https://github.com/sgl-project/sglang/pull/23857
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；关联提交 `b437f6be48a1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+36/-6，可读 patch 111 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron-omni-v3-alias」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Nemotron-omni-v3-alias」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4 (18 lines); hunks: -19,13 +19,19; -51,15 +57,19; symbols: NanoNemotronVLImageProcessor, __init__，涉及 `NanoNemotronVLImageProcessor, __init__`；`python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0 (9 lines); hunks: -150,3 +150,12 @@ def create_radio_config(self):; symbols: create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__，涉及 `create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1 (6 lines); hunks: -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:; symbols: is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3，涉及 `is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4 (18 lines); hunks: -19,13 +19,19; -51,15 +57,19; symbols: NanoNemotronVLImageProcessor, __init__
  - `python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0 (9 lines); hunks: -150,3 +150,12 @@ def create_radio_config(self):; symbols: create_radio_config, NemotronH_Nano_Omni_Reasoning_V3_Config, __init__
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1 (6 lines); hunks: -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:; symbols: is_sound_weights, NemotronH_Nano_Omni_Reasoning_V3
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -19,13 +19,19 @@
-from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
+from sglang.srt.configs.nano_nemotron_vl import (
+    NemotronH_Nano_Omni_Reasoning_V3_Config,
+    NemotronH_Nano_VL_V2_Config,
+)
-from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -150,3 +150,12 @@ def create_radio_config(self):
+class NemotronH_Nano_Omni_Reasoning_V3_Config(NemotronH_Nano_VL_V2_Config):
+    model_type = "NemotronH_Nano_Omni_Reasoning_V3"
+    def __init__(self, *args, **kwargs):
+        # Explicit __init__ prevents PretrainedConfig.__init_subclass__ from
+        # replacing the parent's custom __init__ with a dataclass-generated one.
+        super().__init__(*args, **kwargs)
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -372,4 +372,8 @@ def is_sound_weights(name: str) -> bool:
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +14/-4; `python/sglang/srt/configs/nano_nemotron_vl.py` modified +9/-0; `python/sglang/srt/models/nano_nemotron_vl.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21321 - [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)

- 链接: https://github.com/sgl-project/sglang/pull/21321
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`；关联提交 `8327270c7263`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+341/-53，可读 patch 758 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron)」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -190,6 +191,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -190,6 +191,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -51,6 +51,7 @@
+from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -190,6 +191,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.DeepSeekV3,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23594 - LoRA support for qwen3.5 and nemotron3

- 链接: https://github.com/sgl-project/sglang/pull/23594
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`；关联提交 `c8c1c9261d72`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 21 个文件，+1131/-127，可读 patch 1734 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「LoRA support for qwen3.5 and nemotron3」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`；技术摘要: 覆盖「LoRA support for qwen3.5 and nemotron3」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +105/-0 (105 lines); hunks: -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):; -748,6 +759,100 @@ def _init_model(; symbols: NemotronHForCausalLM, _init_model, get_input_embeddings, get_stacked_multiply，涉及 `NemotronHForCausalLM, _init_model, get_input_embeddings`；`test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff, test_lora_nemotron_3_super_120b_a12b_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +105/-0 (105 lines); hunks: -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):; -748,6 +759,100 @@ def _init_model(; symbols: NemotronHForCausalLM, _init_model, get_input_embeddings, get_stacked_multiply
  - `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: kl_v2, get_prompt_logprobs, TestLoRANemotron3Super120B_A12B_LogprobDiff, test_lora_nemotron_3_super_120b_a12b_logprob_accuracy
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -665,6 +665,17 @@ class NemotronHForCausalLM(nn.Module):
+    supported_lora_modules = [
+        "qkv_proj",
+        "o_proj",
+        "out_proj",
+        "in_proj",
+        "up_proj",
diff -- test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py
@@ -0,0 +1,154 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +105/-0
  - tests: `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` added +154/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_chunked_sgmv_backend.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_5_4b_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24163 - Revert "[ci] split stage-c-test-4-gpu-b200 to enable a low-disk runner pool"

- 链接: https://github.com/sgl-project/sglang/pull/24163
- 状态/时间: merged / 2026-04-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+29/-99，可读 patch 290 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[ci] split stage-c-test-4-gpu-b200 to enable a low-disk runner pool"」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`；技术摘要: 覆盖「Revert "[ci] split stage-c-test-4-gpu-b200 to enable a low-disk runner pool"」；主要实现面是 `.github/workflows/pr-test.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/pr-test.yml` modified +15/-82 (97 lines); hunks: -101,7 +101,6 @@ jobs:; -283,10 +282,8 @@ jobs:；`scripts/ci/utils/slash_command_handler.py` modified +0/-2 (2 lines); hunks: -303,7 +303,6 @@ def handle_rerun_stage(; -486,7 +485,6 @@ def handle_rerun_stage(; symbols: handle_rerun_stage，涉及 `handle_rerun_stage`；`test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; symbols: TestGptOss4Gpu，涉及 `TestGptOss4Gpu`；`test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7。
- 代码 diff 细节:
  - `.github/workflows/pr-test.yml` modified +15/-82 (97 lines); hunks: -101,7 +101,6 @@ jobs:; -283,10 +282,8 @@ jobs:
  - `scripts/ci/utils/slash_command_handler.py` modified +0/-2 (2 lines); hunks: -303,7 +303,6 @@ def handle_rerun_stage(; -486,7 +485,6 @@ def handle_rerun_stage(; symbols: handle_rerun_stage
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; symbols: TestGptOss4Gpu
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7
  - `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test.yml
@@ -101,7 +101,6 @@ jobs:
-      b200_low_disk_runner: ${{ steps.set-runner.outputs.b200_low_disk_runner }}
@@ -283,10 +282,8 @@ jobs:
-            echo "b200_low_disk_runner=4-gpu-b200-kernel-low-disk" >> $GITHUB_OUTPUT
-            echo "b200_low_disk_runner=4-gpu-b200-low-disk" >> $GITHUB_OUTPUT
@@ -337,20 +334,19 @@ jobs:
-            echo "| Component            | Changed |"
diff -- scripts/ci/utils/slash_command_handler.py
@@ -303,7 +303,6 @@ def handle_rerun_stage(
-        "stage-c-test-4-gpu-b200-small",
@@ -486,7 +485,6 @@ def handle_rerun_stage(
-    "stage-c-test-4-gpu-b200-small": "4-gpu-b200-low-disk",
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -4,7 +4,7 @@
-register_cuda_ci(est_time=584, suite="stage-c-test-4-gpu-b200-small")
+register_cuda_ci(est_time=740, suite="stage-c-test-4-gpu-b200")
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
```

- 已读文件:
  - ci: `.github/workflows/pr-test.yml` modified +15/-82
  - other: `scripts/ci/utils/slash_command_handler.py` modified +0/-2
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +1/-1; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_fp4_triton.py` modified +1/-1; `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` modified +1/-1; `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24328 - introduce arg_groups/ with nemotron_h hook

- 链接: https://github.com/sgl-project/sglang/pull/24328
- 状态/时间: merged / 2026-05-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/nemotron_h_hook.py`；关联提交 `00d620b77d1b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+55/-39，可读 patch 103 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「introduce arg_groups/ with nemotron_h hook」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/arg_groups/nemotron_h_hook.py`；技术摘要: 覆盖「introduce arg_groups/ with nemotron_h hook」；主要实现面是 `python/sglang/srt/arg_groups/nemotron_h_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/arg_groups/nemotron_h_hook.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: apply_nemotron_h_defaults，涉及 `apply_nemotron_h_defaults`。
- 代码 diff 细节:
  - `python/sglang/srt/arg_groups/nemotron_h_hook.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: apply_nemotron_h_defaults
- 关键代码摘录:

```diff
diff -- python/sglang/srt/arg_groups/nemotron_h_hook.py
@@ -0,0 +1,51 @@
+import logging
+from typing import TYPE_CHECKING
+from sglang.srt.utils.common import is_sm100_supported
+if TYPE_CHECKING:
+    from sglang.srt.server_args import ServerArgs
+logger = logging.getLogger(__name__)
```

- 已读文件:
  - runtime: `python/sglang/srt/arg_groups/nemotron_h_hook.py` added +51/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/__init__.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23998 - update Nemotron3 Nano Omni cookbook benchmarks

- 链接: https://github.com/sgl-project/sglang/pull/23998
- 状态/时间: merged / 2026-05-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；关联提交 `83b48fd5237a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+119/-4，可读 patch 165 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update Nemotron3 Nano Omni cookbook benchmarks」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「update Nemotron3 Nano Omni cookbook benchmarks」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +119/-4 (123 lines); hunks: -478,7 +478,7 @@ Nemotron 3 Nano Omni achieves **9x higher throughput** than...; -492,6 +492,7 @@ sglang serve \。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +119/-4 (123 lines); hunks: -478,7 +478,7 @@ Nemotron 3 Nano Omni achieves **9x higher throughput** than...; -492,6 +492,7 @@ sglang serve \
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -478,7 +478,7 @@ Nemotron 3 Nano Omni achieves **9x higher throughput** than other open omni mode
-- Hardware: H100 (4×)
+- Hardware: B200 (8×)
@@ -492,6 +492,7 @@ sglang serve \
+  --attention-backend flashinfer \
@@ -510,12 +511,52 @@ python3 -m sglang.bench_serving \
-### 5.3 Accuracy Benchmark
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +119/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24434 - [NemotronH] Fix expert scale weight loading

- 链接: https://github.com/sgl-project/sglang/pull/24434
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `test/registered/unit/models/test_nemotron_h_weight_loading.py`；关联提交 `672f778512bc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+89/-0，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Fix expert scale weight loading」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `test/registered/unit/models/test_nemotron_h_weight_loading.py`, `python/sglang/srt/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Fix expert scale weight loading」；主要实现面是 `test/registered/unit/models/test_nemotron_h_weight_loading.py`, `python/sglang/srt/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_nemotron_h_weight_loading.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _FakePPGroup, _FakeParam, __init__, weight_loader，涉及 `_FakePPGroup, _FakeParam, __init__`；`python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -972,6 +972,8 @@ def load_weights(; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_nemotron_h_weight_loading.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _FakePPGroup, _FakeParam, __init__, weight_loader
  - `python/sglang/srt/models/nemotron_h.py` modified +2/-0 (2 lines); hunks: -972,6 +972,8 @@ def load_weights(; symbols: load_weights
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_nemotron_h_weight_loading.py
@@ -0,0 +1,87 @@
+"""
+Unit tests for NemotronHForCausalLM.load_weights.
+Regression test for Nemotron-H expert scale checkpoint tensors that map to
+parameters absent from the current runtime model.
+"""
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/models/nemotron_h.py
@@ -972,6 +972,8 @@ def load_weights(
+                    if name_mapped not in params_dict:
+                        continue
```

- 已读文件:
  - tests: `test/registered/unit/models/test_nemotron_h_weight_loading.py` added +87/-0
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_nemotron_h_weight_loading.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24721 - ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/

- 链接: https://github.com/sgl-project/sglang/pull/24721
- 状态/时间: merged / 2026-05-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 45 个文件，+818/-525，可读 patch 1510 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/manual/models/test_nvidia_nemotron_nano_v2.py`；技术摘要: 覆盖「ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/」；主要实现面是 `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/manual/models/test_nvidia_nemotron_nano_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer，涉及 `TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer`；`test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-25 (26 lines); hunks: -5,7 +5,7; -18,30 +18,6; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BFP8，涉及 `TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BFP8`；`test/manual/models/test_nvidia_nemotron_nano_v2.py` renamed +0/-0 (0 lines)；`test/manual/models/test_nvidia_nemotron_nano_v2_vl.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-25 (26 lines); hunks: -5,7 +5,7; -18,30 +18,6; symbols: TestNvidiaNemotron3Nano30BBF16, TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BFP8
  - `test/manual/models/test_nvidia_nemotron_nano_v2.py` renamed +0/-0 (0 lines)
  - `test/manual/models/test_nvidia_nemotron_nano_v2_vl.py` renamed +0/-0 (0 lines)
  - `test/manual/models/test_qwen_models.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- test/manual/models/test_nvidia_nemotron_3_nano_archived.py
@@ -0,0 +1,47 @@
+"""Archived test classes split out of test/registered/models/test_nvidia_nemotron_3_nano.py.
+Originally registered with `register_cuda_ci(...)`. Moved here as part of
+the per-commit pruning effort to keep the code reachable manually.
+Run with `python3 test/manual/models/test_nvidia_nemotron_3_nano_archived.py`.
+"""
+import unittest
diff -- test/registered/models/test_nvidia_nemotron_3_nano.py
@@ -5,7 +5,7 @@
-    est_time=564,
+    est_time=190,
@@ -18,30 +18,6 @@
-class TestNvidiaNemotron3Nano30BBF16(LMEvalMixin, DefaultServerBase):
-    """Test Nemotron-3-Nano-30B BF16 model with lm-eval GSM8K evaluation."""
-    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
diff -- test/manual/4-gpu-models/test_qwen35_models_archived.py
@@ -0,0 +1,168 @@
```

- 已读文件:
  - tests: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` added +47/-0; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-25; `test/manual/models/test_nvidia_nemotron_nano_v2.py` renamed +0/-0; `test/manual/models/test_nvidia_nemotron_nano_v2_vl.py` renamed +0/-0; `test/manual/models/test_qwen_models.py` renamed +0/-0; `test/manual/4-gpu-models/test_qwen35_models_archived.py` added +168/-0
- 验证与风险: diff 自带测试面 `test/manual/4-gpu-models/test_qwen35_fp4_triton.py`, `test/manual/4-gpu-models/test_qwen35_models_archived.py`, `test/manual/4-gpu-models/test_qwen3_next_models.py`, `test/manual/4-gpu-models/test_qwen3_next_models_mtp_archived.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- 链接: https://github.com/sgl-project/sglang/pull/25182
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 136 个文件，+255/-0，可读 patch 872 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: add vLLM SPDX copyright headers to ported files」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`；技术摘要: 覆盖「chore: add vLLM SPDX copyright headers to ported files」；主要实现面是 `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6；`python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6。
- 代码 diff 细节:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- 已读文件:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25197 - ci: decouple stage and runner for cuda registry

- 链接: https://github.com/sgl-project/sglang/pull/25197
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 261 个文件，+388/-293，可读 patch 2625 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: decouple stage and runner for cuda registry」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`；技术摘要: 覆盖「ci: decouple stage and runner for cuda registry」；主要实现面是 `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8；`test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8；`test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8；`test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8。
- 代码 diff 细节:
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8
  - `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8
  - `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1 (3 lines); hunks: -6,7 +6,8
- 关键代码摘录:

```diff
diff -- test/registered/layers/test_fla_layernorm_guard.py
@@ -19,7 +19,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_dummy_grok_models.py
@@ -5,7 +5,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_ministral3_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-1-gpu-small",
+    stage="stage-b",
+    runner_config="1-gpu-small",
diff -- test/registered/models/test_ministral4_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-2-gpu-large",
```

- 已读文件:
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1; `test/registered/models/test_dummy_grok_models.py` modified +2/-1; `test/registered/models/test_ministral3_models.py` modified +2/-1; `test/registered/models/test_ministral4_models.py` modified +2/-1; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ci/ci_register.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25203 - ci: B200 conditional split + LPT_SLOP removal (stage-c partition 8→3)

- 链接: https://github.com/sgl-project/sglang/pull/25203
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+33/-34，可读 patch 209 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: B200 conditional split + LPT_SLOP removal (stage-c partition 8→3)」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `scripts/ci/utils/compute_partitions.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`；技术摘要: 覆盖「ci: B200 conditional split + LPT_SLOP removal (stage-c partition 8→3)」；主要实现面是 `scripts/ci/utils/compute_partitions.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `scripts/ci/utils/compute_partitions.py` modified +9/-10 (19 lines); hunks: -39,15 +39,13; -97,10 +95,11 @@ def compute_partitions(tests, full_parallel=False):; symbols: compute_partitions，涉及 `compute_partitions`；`test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9；`test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9；`test/registered/lora/test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9。
- 代码 diff 细节:
  - `scripts/ci/utils/compute_partitions.py` modified +9/-10 (19 lines); hunks: -39,15 +39,13; -97,10 +95,11 @@ def compute_partitions(tests, full_parallel=False):; symbols: compute_partitions
  - `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9
  - `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9
  - `test/registered/lora/test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` modified +3/-3 (6 lines); hunks: -35,9 +35,9
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; symbols: TestGptOss4Gpu
- 关键代码摘录:

```diff
diff -- scripts/ci/utils/compute_partitions.py
@@ -39,15 +39,13 @@
-# Per-partition wall-clock target + ceiling. Single knob for the whole
-# pipeline. ~17 min avg under perfect LPT (TARGET / LPT_SLOP), ~22 min under
-# worst-case LPT 4/3 imbalance, fail-fast above 30 min.
+# Per-partition wall-clock target. ~20 min avg naive; worst-case LPT 4/3
+# imbalance is ~27 min, still below the 30-min job-level timeout that acts
+# as the real safety net. No LPT slop applied — we lean on the runtime
diff -- test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
@@ -35,9 +35,9 @@
-    est_time=300,
-    stage="stage-c",
-    runner_config="4-gpu-b200",
+    est_time=90,
+    suite="nightly-4-gpu-b200",
+    nightly=True,
diff -- test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py
@@ -35,9 +35,9 @@
```

- 已读文件:
  - other: `scripts/ci/utils/compute_partitions.py` modified +9/-10
  - tests: `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` modified +3/-3; `test/registered/lora/test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` modified +3/-3; `test/registered/lora/test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` modified +3/-3; `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +1/-1; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25236 - ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)

- 链接: https://github.com/sgl-project/sglang/pull/25236
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+13/-13，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`；技术摘要: 覆盖「ci: H200 conditional split + dsv4 est_time recalibration (h200 partition 6→2)」；主要实现面是 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7；`test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7；`test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7；`test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash，涉及 `TestMiMoV2Flash`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -6,7 +6,7; symbols: TestMiMoV2Flash
  - `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1 (2 lines); hunks: -14,7 +14,7
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
@@ -13,7 +13,7 @@
-register_cuda_ci(est_time=492, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=450, suite="nightly-8-gpu-h200", nightly=True)
diff -- test/registered/8-gpu-models/test_deepseek_v3_mtp.py
@@ -17,7 +17,7 @@
-register_cuda_ci(est_time=309, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=300, stage="stage-c", runner_config="8-gpu-h200")
diff -- test/registered/8-gpu-models/test_dsa_models_mtp.py
@@ -17,7 +17,7 @@
-    est_time=1048,
+    est_time=1030,
diff -- test/registered/8-gpu-models/test_mimo_models.py
@@ -6,7 +6,7 @@
-register_cuda_ci(est_time=610, stage="stage-c", runner_config="8-gpu-h200")
+register_cuda_ci(est_time=500, stage="stage-c", runner_config="8-gpu-h200")
diff -- test/registered/8-gpu-models/test_minimax_m25_basic.py
@@ -14,7 +14,7 @@
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` modified +1/-1; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +1/-1; `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1; `test/registered/8-gpu-models/test_minimax_m25_basic.py` modified +1/-1; `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_bf16.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`, `test/registered/8-gpu-models/test_mimo_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24725 - ci: tag-gated nightly migration — foundation + 40 whole-file moves

- 链接: https://github.com/sgl-project/sglang/pull/24725
- 状态/时间: merged / 2026-05-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 78 个文件，+2263/-2140，可读 patch 4964 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: tag-gated nightly migration — foundation + 40 whole-file moves」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`；技术摘要: 覆盖「ci: tag-gated nightly migration — foundation + 40 whole-file moves」；主要实现面是 `test/registered/models/test_ministral4_models.py`, `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_generation_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7；`test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7；`test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7；`test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7。
- 代码 diff 细节:
  - `test/registered/models/test_ministral4_models.py` modified +1/-5 (6 lines); hunks: -6,11 +6,7
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/registered/models/test_generation_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7
  - `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
  - `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- test/registered/models/test_ministral4_models.py
@@ -6,11 +6,7 @@
-register_cuda_ci(
-    est_time=200,
-    stage="stage-b",
-    runner_config="2-gpu-large",
-)
+register_cuda_ci(est_time=200, stage="extra-a", runner_config="2-gpu-large")
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -13,7 +13,7 @@
-register_cuda_ci(est_time=65, stage="stage-b", runner_config="1-gpu-large")
+register_cuda_ci(est_time=65, stage="extra-a", runner_config="1-gpu-large")
diff -- test/registered/models/test_generation_models.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=150, stage="stage-b", runner_config="1-gpu-large")
+register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-large")
diff -- test/registered/models/test_vlm_models.py
@@ -13,7 +13,7 @@
```

- 已读文件:
  - tests: `test/registered/models/test_ministral4_models.py` modified +1/-5; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1; `test/registered/models/test_vlm_models.py` modified +1/-1; `test/manual/openai_server/function_call/test_tool_choice.py` renamed +0/-0; `test/registered/sessions/test_streaming_session.py` modified +62/-1072
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/streaming_session_kit.py`, `python/sglang/test/server_fixtures/hybrid_attn_backend_fixture.py`, `python/sglang/test/server_fixtures/ngram_fixture.py`, `python/sglang/test/server_fixtures/pcg_spec_fixture.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25420 - [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI

- 链接: https://github.com/sgl-project/sglang/pull/25420
- 状态/时间: merged / 2026-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 473 个文件，+746/-747，可读 patch 5614 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`；技术摘要: 覆盖「[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI」；主要实现面是 `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:；`test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port，涉及 `_free_port`；`test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool，涉及 `_make_tool`；`test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7。
- 代码 diff 细节:
  - `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:
  - `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port
  - `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-multimodal-gen.yml
@@ -31,7 +31,7 @@ on:
-      skip_stage_health_check:
+      skip_pr_test_health_check:
@@ -42,7 +42,7 @@ env:
-  SKIP_STAGE_HEALTH_CHECK: ${{ inputs.skip_stage_health_check == 'true' }}
+  SKIP_PR_TEST_HEALTH_CHECK: ${{ inputs.skip_pr_test_health_check == 'true' }}
@@ -90,7 +90,7 @@ jobs:
diff -- test/registered/bench_fn/test_bench_serving_reasoning_stream.py
@@ -24,7 +24,7 @@
-register_cpu_ci(est_time=10, suite="stage-a-test-cpu")
+register_cpu_ci(est_time=10, suite="base-a-test-cpu")
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -11,7 +11,7 @@
-register_cpu_ci(5, "stage-a-test-cpu")
+register_cpu_ci(5, "base-a-test-cpu")
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -2,7 +2,7 @@
```

- 已读文件:
  - runtime: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7
  - tests: `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1; `test/registered/function_call/test_kimik2_detector.py` modified +1/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`, `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/jit_kernel/tests/test_add_constant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25831 - [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests

- 链接: https://github.com/sgl-project/sglang/pull/25831
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+572/-639，可读 patch 1504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`；技术摘要: 覆盖「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；主要实现面是 `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4；`python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate，涉及 `ServerSanityMixin, _sanity_generate, test_health`；`python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests，涉及 `BasicSchedulerStressMixin, _stress_generate, test_streaming_response`；`python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math，涉及 `BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france`。
- 代码 diff 细节:
  - `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4
  - `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate
  - `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests
  - `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math
  - `test/registered/language/test_srt_backend.py` removed +0/-94 (94 lines); hunks: -1,94 +0,0; symbols: TestSRTBackend, setUpClass, tearDownClass, test_few_shot_qa
- 关键代码摘录:

```diff
diff -- test/manual/models/test_nvidia_nemotron_3_nano_archived.py
@@ -1,4 +1,4 @@
-"""Archived test classes split out of test/registered/models/test_nvidia_nemotron_3_nano.py.
+"""Archived test classes split out of test/registered/models_e2e/test_nvidia_nemotron_3_nano.py.
diff -- python/sglang/test/kits/server_sanity_kit.py
@@ -1,228 +0,0 @@
-"""Black-box server sanity prompts: cheap checks that catch silent
-correctness regressions (gibberish / repetition collapse / encoding),
-streaming/concurrent path bugs, and endpoint health.
-Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
-and ``self.process``. Each test is independent and fast (≤ 5 s after
-warmup); the whole kit completes in < 1 min."""
diff -- python/sglang/test/kits/basic_scheduler_stress_kit.py
@@ -0,0 +1,135 @@
+"""Basic scheduler / cache / streaming stress sanity kit.
+Probes that catch bugs which only fire under multi-request or large-
+prompt conditions: scheduler hangs, radix prefix-cache cross-
+contamination, chunked-prefill multi-chunk kernel crashes, and SSE
```

- 已读文件:
  - tests: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1; `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228; `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0; `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0; `test/registered/language/test_srt_backend.py` removed +0/-94; `test/registered/core/test_engine_child_pids.py` modified +40/-51
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/basic_api_contract_kit.py`, `python/sglang/test/kits/basic_decode_correctness_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`, `python/sglang/test/kits/server_sanity_kit.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- 链接: https://github.com/sgl-project/sglang/pull/25983
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 77 个文件，+1227/-905，可读 patch 5236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；技术摘要: 覆盖「feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode，涉及 `get_spec_info, run_once, maybe_init_ngram_embedding`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once，涉及 `capture_one_batch_size, run_once`；`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size，涉及 `warmup_compile, _cache_loc_dtype, capture_one_batch_size`；`python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp，涉及 `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -146,6 +146,11 @@
+from sglang.srt.model_executor.forward_context import (
+    ForwardContext,
+    forward_context,
+    has_forward_context,
+)
@@ -2638,9 +2643,6 @@ def get_spec_info():
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -65,6 +65,7 @@
+from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
@@ -1016,9 +1017,6 @@ def capture_one_batch_size(
-            req_to_token_pool=self.model_runner.req_to_token_pool,
-            token_to_kv_pool=self.model_runner.token_to_kv_pool,
-            attn_backend=attn_backend,
@@ -1040,85 +1038,90 @@ def capture_one_batch_size(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -58,6 +58,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- 验证与风险: diff 自带测试面 `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- 链接: https://github.com/sgl-project/sglang/pull/24751
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+45/-44，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data，涉及 `_process_loaded_mm_data, load_mm_data`；`python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async，涉及 `__init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15829 - [feat] Support `extra_buffer` in Mamba2-based models

- 链接: https://github.com/sgl-project/sglang/pull/15829
- 状态/时间: merged / 2026-05-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+405/-130，可读 patch 965 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] Support `extra_buffer` in Mamba2-based models」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `test/manual/models/test_nvidia_nemotron_nano_v2.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `test/manual/models/test_granite_moe_hybrid.py`；技术摘要: 覆盖「[feat] Support `extra_buffer` in Mamba2-based models」；主要实现面是 `test/manual/models/test_nvidia_nemotron_nano_v2.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `test/manual/models/test_granite_moe_hybrid.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/models/test_nvidia_nemotron_nano_v2.py` modified +65/-52 (117 lines); hunks: -2,91 +2,104; symbols: TestNvidiaNemotronNanoV2BF16, TestNvidiaNemotronNanoV2BF16PP, TestNvidiaNemotronNanoV2FP8, TestNvidiaNemotronNanoV2BF16ExtraBuffer，涉及 `TestNvidiaNemotronNanoV2BF16, TestNvidiaNemotronNanoV2BF16PP, TestNvidiaNemotronNanoV2FP8`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +69/-29 (98 lines); hunks: -23,12 +23,6; -278,9 +272,9 @@ def _init_track_conv_indices(; symbols: _init_track_conv_indices, _init_track_ssm_indices, _track_mamba_state_extend，涉及 `_init_track_conv_indices, _init_track_ssm_indices, _track_mamba_state_extend`；`test/manual/models/test_granite_moe_hybrid.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGraniteMoeHybrid, TestGraniteMoeHybridExtraBuffer，涉及 `TestGraniteMoeHybrid, TestGraniteMoeHybridExtraBuffer`；`python/sglang/srt/layers/attention/mamba/mamba.py` modified +15/-1 (16 lines); hunks: -26,6 +26,7; -410,6 +411,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `test/manual/models/test_nvidia_nemotron_nano_v2.py` modified +65/-52 (117 lines); hunks: -2,91 +2,104; symbols: TestNvidiaNemotronNanoV2BF16, TestNvidiaNemotronNanoV2BF16PP, TestNvidiaNemotronNanoV2FP8, TestNvidiaNemotronNanoV2BF16ExtraBuffer
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +69/-29 (98 lines); hunks: -23,12 +23,6; -278,9 +272,9 @@ def _init_track_conv_indices(; symbols: _init_track_conv_indices, _init_track_ssm_indices, _track_mamba_state_extend
  - `test/manual/models/test_granite_moe_hybrid.py` added +33/-0 (33 lines); hunks: -0,0 +1,33; symbols: TestGraniteMoeHybrid, TestGraniteMoeHybridExtraBuffer
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +15/-1 (16 lines); hunks: -26,6 +26,7; -410,6 +411,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +10/-0 (10 lines); hunks: -171,6 +171,11 @@ def prepare_decode(; -239,6 +244,11 @@ def prepare_mixed(; symbols: prepare_decode, prepare_mixed
- 关键代码摘录:

```diff
diff -- test/manual/models/test_nvidia_nemotron_nano_v2.py
@@ -2,91 +2,104 @@
+from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
+from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
+NVIDIA_NEMOTRON_NANO_V2_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
-    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
+    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
-    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -23,12 +23,6 @@
-from sglang.srt.utils import is_cpu
-if not is_cpu():
-    from sglang.srt.layers.attention.fla.chunk_delta_h import (
-        CHUNK_SIZE as FLA_CHUNK_SIZE,
-    )
@@ -278,9 +272,9 @@ def _init_track_conv_indices(
diff -- test/manual/models/test_granite_moe_hybrid.py
@@ -0,0 +1,33 @@
```

- 已读文件:
  - tests: `test/manual/models/test_nvidia_nemotron_nano_v2.py` modified +65/-52; `test/manual/models/test_granite_moe_hybrid.py` added +33/-0
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +69/-29; `python/sglang/srt/layers/attention/mamba/mamba.py` modified +15/-1; `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +10/-0; `python/sglang/srt/models/falcon_h1.py` modified +1/-0; `python/sglang/srt/models/granitemoehybrid.py` modified +1/-0; `python/sglang/srt/models/nemotron_h.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/manual/models/test_granite_moe_hybrid.py`, `test/manual/models/test_nvidia_nemotron_nano_v2.py`, `test/registered/disaggregation/test_disaggregation_hybrid_attention.py`, `test/registered/unit/mem_cache/test_mamba_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25023 - [NemotronH] V3 Omni wrapper: WeightsMapper + config round-trip

- 链接: https://github.com/sgl-project/sglang/pull/25023
- 状态/时间: merged / 2026-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；关联提交 `499eecce22ea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-0，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] V3 Omni wrapper: WeightsMapper + config round-trip」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`；技术摘要: 覆盖「[NemotronH] V3 Omni wrapper: WeightsMapper + config round-trip」；主要实现面是 `python/sglang/srt/models/nano_nemotron_vl.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nano_nemotron_vl.py` modified +10/-0 (10 lines); hunks: -39,6 +39,7; -47,6 +48,15; symbols: NemotronH_Nano_VL_V2, when, create_evs_config，涉及 `NemotronH_Nano_VL_V2, when, create_evs_config`；`python/sglang/srt/configs/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -60,6 +60,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +10/-0 (10 lines); hunks: -39,6 +39,7; -47,6 +48,15; symbols: NemotronH_Nano_VL_V2, when, create_evs_config
  - `python/sglang/srt/configs/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -60,6 +60,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nano_nemotron_vl.py
@@ -39,6 +39,7 @@
+from sglang.srt.models.utils import WeightsMapper
@@ -47,6 +48,15 @@
+    # The loader reads `hf_to_sglang_mapper` off the outer model class when
+    # applying name rewrites to the quant config's `quantized_layers` keys;
+    # the inner NemotronHForCausalLM mapper is not consulted there.
+    hf_to_sglang_mapper = WeightsMapper(
diff -- python/sglang/srt/configs/nano_nemotron_vl.py
@@ -60,6 +60,13 @@ def __init__(
+        # Round-trip: `to_dict()` emits `raw_vision_config` (V2's storage
+        # name) but `from_dict()` rebuilds via this `vision_config` kwarg.
+        # Without this alias, the V3->V2 alias rebuild in `get_config` loses
+        # the vision config across the round-trip.
+        if vision_config is None:
+            vision_config = kwargs.pop("raw_vision_config", None)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nano_nemotron_vl.py` modified +10/-0; `python/sglang/srt/configs/nano_nemotron_vl.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24429 - Support NemotronHPuzzleForCausalLM

- 链接: https://github.com/sgl-project/sglang/pull/24429
- 状态/时间: merged / 2026-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；关联提交 `0abe6a85a51f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+74/-7，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support NemotronHPuzzleForCausalLM」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Support NemotronHPuzzleForCausalLM」；主要实现面是 `python/sglang/srt/configs/nemotron_h.py`, `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/nemotron_h.py` modified +53/-0 (53 lines); hunks: -17,6 +17,9; -240,6 +243,7 @@ def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:; symbols: _resolve_mamba_chunk_size, __init__, _pattern_to_list, get_nemotron_h_config_for_layer，涉及 `_resolve_mamba_chunk_size, __init__, _pattern_to_list`；`python/sglang/srt/models/nemotron_h.py` modified +11/-4 (15 lines); hunks: -358,9 +358,10 @@ def __init__(; -510,6 +511,7 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0 (1 lines); hunks: -283,6 +283,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/nemotron_h.py` modified +53/-0 (53 lines); hunks: -17,6 +17,9; -240,6 +243,7 @@ def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:; symbols: _resolve_mamba_chunk_size, __init__, _pattern_to_list, get_nemotron_h_config_for_layer
  - `python/sglang/srt/models/nemotron_h.py` modified +11/-4 (15 lines); hunks: -358,9 +358,10 @@ def __init__(; -510,6 +511,7 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0 (1 lines); hunks: -283,6 +283,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/nemotron_h.py
@@ -17,6 +17,9 @@
+import copy
+from typing import Any
@@ -240,6 +243,7 @@ def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:
+        *,
@@ -504,3 +508,52 @@ def _pattern_to_list(pattern: str) -> list[str]:
+    def get_nemotron_h_config_for_layer(self, layer_idx: int) -> "NemotronHConfig":
diff -- python/sglang/srt/models/nemotron_h.py
@@ -358,9 +358,10 @@ def __init__(
+        layer_config = config.get_nemotron_h_config_for_layer(layer_idx)
-            config,
+            layer_config,
@@ -510,6 +511,7 @@ def __init__(
+            sliding_window_size=config.sliding_window,
@@ -533,9 +535,10 @@ def __init__(
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -283,6 +283,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/nemotron_h.py` modified +53/-0; `python/sglang/srt/models/nemotron_h.py` modified +11/-4; `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26522 - [NemotronH] Fix weight-loading unit test broken by Puzzle support

- 链接: https://github.com/sgl-project/sglang/pull/26522
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/unit/models/test_nemotron_h_weight_loading.py`；关联提交 `68e5b4fdd66b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Fix weight-loading unit test broken by Puzzle support」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `test/registered/unit/models/test_nemotron_h_weight_loading.py`；技术摘要: 覆盖「[NemotronH] Fix weight-loading unit test broken by Puzzle support」；主要实现面是 `test/registered/unit/models/test_nemotron_h_weight_loading.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_nemotron_h_weight_loading.py` modified +1/-1 (2 lines); hunks: -35,7 +35,7 @@ def weight_loader(; symbols: weight_loader, TestNemotronHWeightLoading, _make_minimal_model，涉及 `weight_loader, TestNemotronHWeightLoading, _make_minimal_model`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_nemotron_h_weight_loading.py` modified +1/-1 (2 lines); hunks: -35,7 +35,7 @@ def weight_loader(; symbols: weight_loader, TestNemotronHWeightLoading, _make_minimal_model
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_nemotron_h_weight_loading.py
@@ -35,7 +35,7 @@ def weight_loader(
-        model.config = SimpleNamespace(n_routed_experts=2)
+        model.config = SimpleNamespace(n_routed_experts=2, max_n_routed_experts=2)
```

- 已读文件:
  - tests: `test/registered/unit/models/test_nemotron_h_weight_loading.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_nemotron_h_weight_loading.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- 链接: https://github.com/sgl-project/sglang/pull/26610
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+611/-816，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`；技术摘要: 覆盖「test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)」；主要实现面是 `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #25655 - Feat/add w4a16 moe support to nemotron

- 链接: https://github.com/sgl-project/sglang/pull/25655
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml`, `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`；关联提交 `b8d7351a74c3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+999/-61，可读 patch 1548 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat/add w4a16 moe support to nemotron」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml`；技术摘要: 覆盖「Feat/add w4a16 moe support to nemotron」；主要实现面是 `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +23/-0 (23 lines); hunks: -7,6 +7,7; -43,5 +44,27 @@ class TestNvidiaNemotron3Nano30BBF16FlashInfer(LMEvalMixin, D...; symbols: TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BNVFP4Marlin，涉及 `TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BNVFP4Marlin`；`python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +14/-1 (15 lines); hunks: -1,7 +1,7; -36,6 +36,19 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", mode...; symbols: apply_nemotron_h_defaults，涉及 `apply_nemotron_h_defaults`；`test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13。
- 代码 diff 细节:
  - `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +23/-0 (23 lines); hunks: -7,6 +7,7; -43,5 +44,27 @@ class TestNvidiaNemotron3Nano30BBF16FlashInfer(LMEvalMixin, D...; symbols: TestNvidiaNemotron3Nano30BBF16FlashInfer, TestNvidiaNemotron3Nano30BNVFP4Marlin
  - `python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +14/-1 (15 lines); hunks: -1,7 +1,7; -36,6 +36,19 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", mode...; symbols: apply_nemotron_h_defaults
  - `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- test/manual/models/test_nvidia_nemotron_3_nano_archived.py
@@ -7,6 +7,7 @@
+from sglang.srt.utils import is_sm80_supported, is_sm90_supported
@@ -43,5 +44,27 @@ class TestNvidiaNemotron3Nano30BBF16FlashInfer(LMEvalMixin, DefaultServerBase):
+@unittest.skip("Skip, test pass locally but compiling takes too long in CI")
+@unittest.skipIf(
+    not (is_sm80_supported() or is_sm90_supported()),
+    "NVFP4 Marlin fallback test requires CUDA SM8X/SM9X",
diff -- python/sglang/srt/arg_groups/nemotron_h_hook.py
@@ -1,7 +1,7 @@
-from sglang.srt.utils.common import is_sm100_supported
+from sglang.srt.utils.common import get_device_capability, is_cuda, is_sm100_supported
@@ -36,6 +36,19 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", model_arch: str) -> Non
+            elif (
+                (
+                    model_config.quantization in ("modelopt_fp4", "modelopt_mixed")
diff -- test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml
@@ -0,0 +1,13 @@
```

- 已读文件:
  - tests: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +23/-0; `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml` added +13/-0
  - runtime: `python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +14/-1
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_gptq_marlin.py`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`, `python/sglang/test/test_marlin_utils.py`, `test/lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27184 - docs: fix Nemotron Super MTP deployment command (spec-v2 + B200)

- 链接: https://github.com/sgl-project/sglang/pull/27184
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx`；关联提交 `90985117a5f7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-3，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: fix Nemotron Super MTP deployment command (spec-v2 + B200)」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx`；技术摘要: 覆盖「docs: fix Nemotron Super MTP deployment command (spec-v2 + B200)」；主要实现面是 `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` modified +7/-3 (10 lines); hunks: -39,7 +39,10 @@ export const Nemotron3SuperDeployment = () => {; -75,7 +78,8 @@ export const Nemotron3SuperDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` modified +7/-3 (10 lines); hunks: -39,7 +39,10 @@ export const Nemotron3SuperDeployment = () => {; -75,7 +78,8 @@ export const Nemotron3SuperDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx
@@ -39,7 +39,10 @@ export const Nemotron3SuperDeployment = () => {
-      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-t
+      // trtllm_mha is Blackwell-only; on B200 it replaces the flashinfer default,
+      // whose per-step plan() host-sync breaks the spec-v2 overlap scheduler. H200
+      // defaults to fa3 (no such sync), so no override is needed there.
+      commandRule: (value, state) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-
@@ -75,7 +78,8 @@ export const Nemotron3SuperDeployment = () => {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx` modified +7/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/nemotron3-super-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25198 - [Docs] Update Nemotron3-Nano-Omni cookbook to reflect new model paths

- 链接: https://github.com/sgl-project/sglang/pull/25198
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；关联提交 `8980eb82de90`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-28，可读 patch 194 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Update Nemotron3-Nano-Omni cookbook to reflect new model paths」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；技术摘要: 覆盖「[Docs] Update Nemotron3-Nano-Omni cookbook to reflect new model paths」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +20/-21 (41 lines); hunks: -30,10 +30,9 @@ Architecture and key features:; -76,7 +75,7 @@ This section provides a progressive guide from quick deploymen...；`docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` modified +5/-7 (12 lines); hunks: -1,18 +1,16; -77,7 +75,7 @@ export const Nemotron3NanoOmniDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +20/-21 (41 lines); hunks: -30,10 +30,9 @@ Architecture and key features:; -76,7 +75,7 @@ This section provides a progressive guide from quick deploymen...
  - `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` modified +5/-7 (12 lines); hunks: -1,18 +1,16; -77,7 +75,7 @@ export const Nemotron3NanoOmniDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -30,10 +30,9 @@ Architecture and key features:
-- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning)
-- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16)
-- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8)
-- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4)
+- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
+- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
diff -- docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx
@@ -1,18 +1,16 @@
-    reasoning: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
-    bf16: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16',
-    fp8: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8',
-    nvfp4: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4',
+    bf16: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16',
+    fp8: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8',
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +20/-21; `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx` modified +5/-7
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-nano-omni-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26969 - docs: add Nemotron 3 Ultra cookbook entry

- 链接: https://github.com/sgl-project/sglang/pull/26969
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；关联提交 `1463e5fbdd54`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1045/-1，可读 patch 1062 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add Nemotron 3 Ultra cookbook entry」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；技术摘要: 覆盖「docs: add Nemotron 3 Ultra cookbook entry」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` added +535/-0 (535 lines); hunks: -0,0 +1,535; symbols: support, defined, is, RadixCache，涉及 `support, defined, is`；`docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` added +507/-0 (507 lines); hunks: -0,0 +1,507。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` added +535/-0 (535 lines); hunks: -0,0 +1,535; symbols: support, defined, is, RadixCache
  - `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` added +507/-0 (507 lines); hunks: -0,0 +1,507
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx
@@ -0,0 +1,535 @@
+---
+title: NVIDIA Nemotron3-Ultra
+metatags:
+    description: "Deploy NVIDIA Nemotron3-Ultra with SGLang - 550B hybrid MoE model (55B active) with 1M context window, BF16/NVFP4 support, built for long-running autonomous agen
+tag:
+    NEW
diff -- docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx
@@ -0,0 +1,507 @@
+export const Nemotron3UltraDeployment = () => {
+  const MODEL_PATHS = {
+    bf16: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
+    nvfp4: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4',
+  };
+  // Verified {model, hardware, tp} combinations. Any tuple not in this list is
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` added +535/-0; `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` added +507/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/cookbook/intro copy.mdx`, `docs_new/docs.json`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #27240 - [Docs] re-organize nemotron cookbook

- 链接: https://github.com/sgl-project/sglang/pull/27240
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`；关联提交 `b89686710d2d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+4/-7，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] re-organize nemotron cookbook」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「[Docs] re-organize nemotron cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +1/-2 (3 lines); hunks: -1,7 +1,6；`docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +0/-2 (2 lines); hunks: -2,8 +2,6。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +1/-2 (3 lines); hunks: -1,7 +1,6
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +0/-2 (2 lines); hunks: -2,8 +2,6
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx
@@ -1,7 +1,6 @@
-metatags:
-    description: "Deploy NVIDIA Nemotron3-Ultra with SGLang - 550B hybrid MoE model (55B active) with 1M context window, BF16/NVFP4 support, built for long-running autonomous agen
+description: "Deploy NVIDIA Nemotron3-Ultra with SGLang - 550B hybrid MoE model (55B active) with 1M context window, BF16/NVFP4 support, built for long-running autonomous agents."
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -2,8 +2,6 @@
-tag:
-    NEW
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +1/-2; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +0/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #26733 - Nemotron perf changes

- 链接: https://github.com/sgl-project/sglang/pull/26733
- 状态/时间: merged / 2026-06-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/nemotron_h_hook.py`, `python/sglang/srt/models/nemotron_h.py`；关联提交 `38ae22e08c73`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+297/-58，可读 patch 713 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron perf changes」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`；技术摘要: 覆盖「Nemotron perf changes」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/arg_groups/nemotron_h_hook.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +25/-25 (50 lines); hunks: -164,24 +164,13 @@ def __init__(; -195,6 +184,18 @@ def __init__(; symbols: __init__, _forward_core_normal, _forward_core_shared_routed_overlap，涉及 `__init__, _forward_core_normal, _forward_core_shared_routed_overlap`；`python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +27/-24 (51 lines); hunks: -12,12 +12,13; -29,28 +30,30 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", mod...; symbols: apply_nemotron_h_defaults，涉及 `apply_nemotron_h_defaults`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +25/-25 (50 lines); hunks: -164,24 +164,13 @@ def __init__(; -195,6 +184,18 @@ def __init__(; symbols: __init__, _forward_core_normal, _forward_core_shared_routed_overlap
  - `python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +27/-24 (51 lines); hunks: -12,12 +12,13; -29,28 +30,30 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", mod...; symbols: apply_nemotron_h_defaults
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -164,24 +164,13 @@ def __init__(
-            params_dtype=torch.float32,
-        self.topk = TopK(
-            top_k=config.num_experts_per_tok,
-            use_grouped_topk=True,
-            topk_group=config.topk_group,
-            num_expert_group=config.n_group,
diff -- python/sglang/srt/arg_groups/nemotron_h_hook.py
@@ -12,12 +12,13 @@
-    if model_config.quantization in [
+    is_modelopt = model_config.quantization in [
-    ]:
+    ]
+    if is_modelopt:
@@ -29,28 +30,30 @@ def apply_nemotron_h_defaults(server_args: "ServerArgs", model_arch: str) -> Non
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +25/-25; `python/sglang/srt/arg_groups/nemotron_h_hook.py` modified +27/-24
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_activation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27838 - Disable async assert in Nemotron nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27838
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；关联提交 `125ef888921b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+52/-46，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Disable async assert in Nemotron nightly tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`；技术摘要: 覆盖「Disable async assert in Nemotron nightly tests」；主要实现面是 `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +37/-34 (71 lines); hunks: -1,5 +1,6; -76,23 +77,24 @@ def test_nemotron_3_super_bf16(self):; symbols: test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4，涉及 `test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4`；`test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +15/-12 (27 lines); hunks: -1,6 +1,7; -69,12 +70,13 @@ class TestNvidiaNemotron3SuperNVFP4(CustomTestCase):; symbols: TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass, TestNvidiaNemotron3SuperNVFP4MTP，涉及 `TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +37/-34 (71 lines); hunks: -1,5 +1,6; -76,23 +77,24 @@ def test_nemotron_3_super_bf16(self):; symbols: test_nemotron_3_super_bf16, test_nemotron_3_super_nvfp4
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +15/-12 (27 lines); hunks: -1,6 +1,7; -69,12 +70,13 @@ class TestNvidiaNemotron3SuperNVFP4(CustomTestCase):; symbols: TestNvidiaNemotron3SuperNVFP4, setUpClass, tearDownClass, TestNvidiaNemotron3SuperNVFP4MTP
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py
@@ -1,5 +1,6 @@
+from sglang.srt.environ import envs
@@ -76,23 +77,24 @@ def test_nemotron_3_super_bf16(self):
-        run_combined_tests(
-            models=variants,
-            test_name="Nemotron-3-Super-120B-BF16",
-            accuracy_params=AccuracyTestParams(
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -1,6 +1,7 @@
+from sglang.srt.environ import envs
@@ -69,12 +70,13 @@ class TestNvidiaNemotron3SuperNVFP4(CustomTestCase):
-        cls.process = popen_launch_server(
-            cls.model,
-            cls.base_url,
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py` modified +37/-34; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +15/-12
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24955 - Support Nemotron DP attention and MTP

- 链接: https://github.com/sgl-project/sglang/pull/24955
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h_utils.py`；关联提交 `6e0fa5afe1c6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+472/-95，可读 patch 1135 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Nemotron DP attention and MTP」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Support Nemotron DP attention and MTP」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +139/-35 (174 lines); hunks: -42,6 +42,12; -76,6 +82,12; symbols: forward, NemotronHMLPDecoderLayer, NemotronHMLPLikeDecoderLayer，涉及 `forward, NemotronHMLPDecoderLayer, NemotronHMLPLikeDecoderLayer`；`python/sglang/srt/models/nemotron_h_utils.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: is_attn_layer, get_real_num_tokens, pad_to_original_num_tokens, _build_layer_scatter_modes，涉及 `is_attn_layer, get_real_num_tokens, pad_to_original_num_tokens`；`python/sglang/srt/models/nemotron_h_mtp.py` modified +37/-3 (40 lines); hunks: -19,6 +19,13; -33,6 +40,7; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +139/-35 (174 lines); hunks: -42,6 +42,12; -76,6 +82,12; symbols: forward, NemotronHMLPDecoderLayer, NemotronHMLPLikeDecoderLayer
  - `python/sglang/srt/models/nemotron_h_utils.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: is_attn_layer, get_real_num_tokens, pad_to_original_num_tokens, _build_layer_scatter_modes
  - `python/sglang/srt/models/nemotron_h_mtp.py` modified +37/-3 (40 lines); hunks: -19,6 +19,13; -33,6 +40,7; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -42,6 +42,12 @@
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_reduce,
+    get_attention_tp_rank,
+    get_attention_tp_size,
+    is_dp_attention_enabled,
+)
diff -- python/sglang/srt/models/nemotron_h_utils.py
@@ -0,0 +1,67 @@
+"""DP-attention helpers for the Nemotron-H model."""
+import torch
+from torch import nn
+from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA
+from sglang.srt.layers.communicator import (
+    LayerCommunicator,
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -19,6 +19,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +139/-35; `python/sglang/srt/models/nemotron_h_utils.py` added +67/-0; `python/sglang/srt/models/nemotron_h_mtp.py` modified +37/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/layers/attention/hybrid_attn_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28087 - [Doc] Fix some inconsistencies in the Nemotron Cookbook

- 链接: https://github.com/sgl-project/sglang/pull/28087
- 状态/时间: merged / 2026-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；关联提交 `95867f0932a6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+54/-39，可读 patch 230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Fix some inconsistencies in the Nemotron Cookbook」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`；技术摘要: 覆盖「[Doc] Fix some inconsistencies in the Nemotron Cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +26/-23 (49 lines); hunks: -97,6 +97,16 @@ export const Nemotron3UltraDeployment = () => {; -126,25 +136,14 @@ export const Nemotron3UltraDeployment = () => {；`docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +28/-16 (44 lines); hunks: -60,7 +60,11 @@ The generator only emits a runnable command for combinations...; -76,6 +80,14 @@ The generator only emits a runnable command for combinations...; symbols: defined，涉及 `defined`。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +26/-23 (49 lines); hunks: -97,6 +97,16 @@ export const Nemotron3UltraDeployment = () => {; -126,25 +136,14 @@ export const Nemotron3UltraDeployment = () => {
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +28/-16 (44 lines); hunks: -60,7 +60,11 @@ The generator only emits a runnable command for combinations...; -76,6 +80,14 @@ The generator only emits a runnable command for combinations...; symbols: defined
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx
@@ -97,6 +97,16 @@ export const Nemotron3UltraDeployment = () => {
+    ep: {
+      name: 'ep',
+      title: 'Expert Parallel (EP)',
+      items: [
+        { id: 'enabled',  label: 'Enabled', subtitle: 'EP = TP' },
+        { id: 'disabled', label: 'Disabled', default: true }
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx
@@ -60,7 +60,11 @@ The generator only emits a runnable command for combinations that NVIDIA / SGLan
-    **B200/GB200/B300/GB300**: Use flashinfer backend by default.
+    **B200/GB200/B300/GB300**: Append `--attention-backend trtllm_mha`. The flashinfer default breaks the overlap scheduler on Blackwell, so `trtllm_mha` is required there.
+- **Mamba scheduler strategy**:
+    Always launch with `--mamba-scheduler-strategy extra_buffer`. This hybrid Transformer-Mamba model requires the `extra_buffer` strategy for correct scheduling of its Mamba stat
@@ -76,6 +80,14 @@ The generator only emits a runnable command for combinations that NVIDIA / SGLan
+- **Expert parallel (EP)**:
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +26/-23; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +28/-16
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28102 - Fix DP attention + EP mode of Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/28102
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`；关联提交 `1a19f66acbf2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+27/-9，可读 patch 98 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DP attention + EP mode of Nemotron」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`；技术摘要: 覆盖「Fix DP attention + EP mode of Nemotron」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +25/-8 (33 lines); hunks: -55,7 +55,10; -129,10 +132,10 @@ def __init__(; symbols: __init__, forward, _forward_core_shared_routed_overlap，涉及 `__init__, forward, _forward_core_shared_routed_overlap`；`python/sglang/srt/models/nemotron_h_utils.py` modified +2/-1 (3 lines); hunks: -57,11 +57,12 @@ def _build_layer_scatter_modes() -> LayerScatterModes:; symbols: _build_layer_scatter_modes, make_layer_communicator，涉及 `_build_layer_scatter_modes, make_layer_communicator`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +25/-8 (33 lines); hunks: -55,7 +55,10; -129,10 +132,10 @@ def __init__(; symbols: __init__, forward, _forward_core_shared_routed_overlap
  - `python/sglang/srt/models/nemotron_h_utils.py` modified +2/-1 (3 lines); hunks: -57,11 +57,12 @@ def _build_layer_scatter_modes() -> LayerScatterModes:; symbols: _build_layer_scatter_modes, make_layer_communicator
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -55,7 +55,10 @@
-from sglang.srt.layers.moe.utils import RoutingMethodType
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    should_skip_post_experts_all_reduce,
+)
@@ -129,10 +132,10 @@ def __init__(
diff -- python/sglang/srt/models/nemotron_h_utils.py
@@ -57,11 +57,12 @@ def _build_layer_scatter_modes() -> LayerScatterModes:
-    layer_norm: RMSNorm, *, for_attn: bool
+    layer_norm: RMSNorm, *, for_attn: bool, allow_reduce_scatter: bool = False
+        allow_reduce_scatter=allow_reduce_scatter,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +25/-8; `python/sglang/srt/models/nemotron_h_utils.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28346 - Use Flashinfer allreduce fusion for MNNVL allreduce for Nemotron

- 链接: https://github.com/sgl-project/sglang/pull/28346
- 状态/时间: merged / 2026-06-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h_utils.py`；关联提交 `ea407df4b023`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+111/-47，可读 patch 385 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use Flashinfer allreduce fusion for MNNVL allreduce for Nemotron」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Use Flashinfer allreduce fusion for MNNVL allreduce for Nemotron」；主要实现面是 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_utils.py`, `python/sglang/srt/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/nemotron_h.py` modified +76/-46 (122 lines); hunks: -18,7 +18,6; -85,6 +84,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/nemotron_h_utils.py` modified +32/-1 (33 lines); hunks: -4,10 +4,12; -57,12 +59,41 @@ def _build_layer_scatter_modes() -> LayerScatterModes:; symbols: _build_layer_scatter_modes, make_layer_communicator, input_norm_maybe_fuse_allreduce，涉及 `_build_layer_scatter_modes, make_layer_communicator, input_norm_maybe_fuse_allreduce`；`python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0 (1 lines); hunks: -148,6 +148,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/nemotron_h.py` modified +76/-46 (122 lines); hunks: -18,7 +18,6; -85,6 +84,7; symbols: __init__, forward
  - `python/sglang/srt/models/nemotron_h_utils.py` modified +32/-1 (33 lines); hunks: -4,10 +4,12; -57,12 +59,41 @@ def _build_layer_scatter_modes() -> LayerScatterModes:; symbols: _build_layer_scatter_modes, make_layer_communicator, input_norm_maybe_fuse_allreduce
  - `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0 (1 lines); hunks: -148,6 +148,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/nemotron_h.py
@@ -18,7 +18,6 @@
-from typing import Optional, Union
@@ -85,6 +84,7 @@
+    input_norm_maybe_fuse_allreduce,
@@ -108,7 +108,7 @@ def __init__(
-        quant_config: Optional[QuantizationConfig] = None,
+        quant_config: QuantizationConfig | None = None,
diff -- python/sglang/srt/models/nemotron_h_utils.py
@@ -4,10 +4,12 @@
+from sglang.srt.distributed import tensor_model_parallel_all_reduce
+    apply_flashinfer_allreduce_fusion,
@@ -57,12 +59,41 @@ def _build_layer_scatter_modes() -> LayerScatterModes:
-    layer_norm: RMSNorm, *, for_attn: bool, allow_reduce_scatter: bool = False
+    layer_norm: RMSNorm,
+    *,
diff -- python/sglang/srt/models/nemotron_h_mtp.py
@@ -148,6 +148,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/nemotron_h.py` modified +76/-46; `python/sglang/srt/models/nemotron_h_utils.py` modified +32/-1; `python/sglang/srt/models/nemotron_h_mtp.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/nemotron_h.py`, `python/sglang/srt/models/nemotron_h_mtp.py`, `python/sglang/srt/models/nemotron_h_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28675 - [Cookbook] Nemotron3-Ultra: Add mamba-backend and SSM dtype flags

- 链接: https://github.com/sgl-project/sglang/pull/28675
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；关联提交 `34e5e38604bb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+31/-5，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Cookbook] Nemotron3-Ultra: Add mamba-backend and SSM dtype flags」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；技术摘要: 覆盖「[Cookbook] Nemotron3-Ultra: Add mamba-backend and SSM dtype flags」；主要实现面是 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +13/-5 (18 lines); hunks: -60,12 +60,20 @@ The generator only emits a runnable command for combinations...; -82,23 +90,23 @@ The generator only emits a runnable command for combinations...；`docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +18/-0 (18 lines); hunks: -154,6 +154,24 @@ export const Nemotron3UltraDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +13/-5 (18 lines); hunks: -60,12 +60,20 @@ The generator only emits a runnable command for combinations...; -82,23 +90,23 @@ The generator only emits a runnable command for combinations...
  - `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +18/-0 (18 lines); hunks: -154,6 +154,24 @@ export const Nemotron3UltraDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx
@@ -60,12 +60,20 @@ The generator only emits a runnable command for combinations that NVIDIA / SGLan
-    **B200/GB200/B300/GB300**: Append `--attention-backend trtllm_mha`. The flashinfer default breaks the overlap scheduler on Blackwell, so `trtllm_mha` is required there.
+    **B200/GB200/B300/GB300**: Set `--attention-backend trtllm_mha`. The flashinfer default breaks the overlap scheduler on Blackwell, so `trtllm_mha` is required there.
+- **Mamba backend**:
+    The Mamba layers use the Triton SSM kernels by default. For better performance, set `--mamba-backend flashinfer` to use the FlashInfer Mamba kernels instead.
+- **Mamba SSM precision**:
+    The SSM state dtype defaults to the model config value. Set `--mamba-ssm-dtype float16` to store the Mamba states in FP16, which reduces mamba cache memory without significant
diff -- docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx
@@ -154,6 +154,24 @@ export const Nemotron3UltraDeployment = () => {
+    mambabackend: {
+      name: 'mambabackend',
+      title: 'Mamba Backend',
+      items: [
+        { id: 'triton',     label: 'Triton',     subtitle: 'Default', default: true  },
+        { id: 'flashinfer', label: 'FlashInfer', subtitle: 'Faster',  default: false }
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx` modified +13/-5; `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +18/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx`, `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29200 - [Cookbook] Nemotron3-Ultra: align MTP draft depth with NVIDIA reference (num_steps 5)

- 链接: https://github.com/sgl-project/sglang/pull/29200
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；关联提交 `2c697daf5f92`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Cookbook] Nemotron3-Ultra: align MTP draft depth with NVIDIA reference (num_steps 5)」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；技术摘要: 覆盖「[Cookbook] Nemotron3-Ultra: align MTP draft depth with NVIDIA reference (num_steps 5)」；主要实现面是 `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +1/-1 (2 lines); hunks: -143,7 +143,7 @@ export const Nemotron3UltraDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +1/-1 (2 lines); hunks: -143,7 +143,7 @@ export const Nemotron3UltraDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx
@@ -143,7 +143,7 @@ export const Nemotron3UltraDeployment = () => {
-      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-t
+      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 5 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-t
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/nemotron3-ultra-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29261 - [Docs] Fix broken links in cookbook

- 链接: https://github.com/sgl-project/sglang/pull/29261
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Fix broken links in cookbook」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；技术摘要: 覆盖「[Docs] Fix broken links in cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...；`docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...；`docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1 (2 lines); hunks: -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1 (2 lines); hunks: -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackw...
  - `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#...
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
@@ -89,7 +89,7 @@ import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchm
-  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](/demo/deepseek_v4_flash).
+  For a runnable end-to-end example, see the [DeepSeek-V4-Flash demo notebook](https://github.com/sgl-project/sglang/blob/main/docs_new/demo/deepseek_v4_flash.ipynb).
diff -- docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx
@@ -127,7 +127,7 @@ Pick a weight format by hardware: **NVFP4** on NVIDIA Blackwell (B200, GB200), *
-For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](../../../docs/basic_usage/glm45). Per-hardware bench com
+For general GLM-4.x family launch guidance (AMD ROCm notes and more), see [Launch GLM-4.5 / GLM-4.6 / GLM-4.7 with SGLang](/cookbook/autoregressive/GLM/GLM-4.5). Per-hardware benc
diff -- docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx
@@ -52,7 +52,7 @@ uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=pytho
-For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/installation).
+For the full Docker setup and other installation methods, refer to the [official SGLang installation guide](../../../docs/get-started/install).
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`, `docs_new/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
