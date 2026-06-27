# vllm GPT-OSS Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/evals/gpt_oss/README.md` | no direct PR-number commit |
| `tests/evals/gpt_oss/__init__.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-baseline.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-bf16-cutlass.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-bf16-trtllm.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-mxfp8-cutlass.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-marlin.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` | [#36179](https://github.com/vllm-project/vllm/pull/36179) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-sm100-fi-mxfp4-mxfp8-trtllm.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-sm120.yaml` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/models-b200.txt` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/models-gfx942.txt` | [#36179](https://github.com/vllm-project/vllm/pull/36179) |
| `tests/evals/gpt_oss/configs/models-gfx950.txt` | [#36179](https://github.com/vllm-project/vllm/pull/36179), [#38292](https://github.com/vllm-project/vllm/pull/38292) |
| `tests/evals/gpt_oss/configs/models-h100.txt` | no direct PR-number commit |
| `tests/evals/gpt_oss/configs/models-spark.txt` | no direct PR-number commit |
| `tests/evals/gpt_oss/conftest.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920) |
| `tests/evals/gpt_oss/test_gpqa_correctness.py` | [#24920](https://github.com/vllm-project/vllm/pull/24920), [#26030](https://github.com/vllm-project/vllm/pull/26030) |
| `tests/evals/gsm8k/configs/humming/gpt-oss-20b-humming-act-fp8.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/humming/gpt-oss-20b-humming.yaml` | no direct PR-number commit |
| `tests/kernels/moe/test_gpt_oss_triton_kernels.py` | [#22421](https://github.com/vllm-project/vllm/pull/22421), [#29008](https://github.com/vllm-project/vllm/pull/29008), [#37683](https://github.com/vllm-project/vllm/pull/37683), [#39007](https://github.com/vllm-project/vllm/pull/39007) |
| `tests/models/quantization/test_gpt_oss.py` | [#29008](https://github.com/vllm-project/vllm/pull/29008), [#35806](https://github.com/vllm-project/vllm/pull/35806), [#35887](https://github.com/vllm-project/vllm/pull/35887), [#36174](https://github.com/vllm-project/vllm/pull/36174), [#43571](https://github.com/vllm-project/vllm/pull/43571) |
| `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` | [#39007](https://github.com/vllm-project/vllm/pull/39007), [#43135](https://github.com/vllm-project/vllm/pull/43135) |
| `vllm/model_executor/models/gpt_oss.py` | [#22327](https://github.com/vllm-project/vllm/pull/22327), [#22401](https://github.com/vllm-project/vllm/pull/22401), [#22508](https://github.com/vllm-project/vllm/pull/22508), [#22538](https://github.com/vllm-project/vllm/pull/22538), [#22678](https://github.com/vllm-project/vllm/pull/22678), [#22948](https://github.com/vllm-project/vllm/pull/22948), [#22951](https://github.com/vllm-project/vllm/pull/22951), [#23613](https://github.com/vllm-project/vllm/pull/23613), [#23680](https://github.com/vllm-project/vllm/pull/23680), [#23815](https://github.com/vllm-project/vllm/pull/23815), [#24032](https://github.com/vllm-project/vllm/pull/24032), [#25246](https://github.com/vllm-project/vllm/pull/25246), ... (28 total) |

## PR Coverage Summary

- Git-traced PRs: 40
- Extra PRs preserved from existing docs: 23
- Total PRs in this document: 63
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-06 | [#22327](https://github.com/vllm-project/vllm/pull/22327) | merged | Add GPT-OSS model code and config [1/N] | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-07 | [#22401](https://github.com/vllm-project/vllm/pull/22401) | merged | [gpt-oss] fix model config with hf_config | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-08 | [#22421](https://github.com/vllm-project/vllm/pull/22421) | merged | [gpt-oss] triton kernel mxfp4 | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2025-08-10 | [#22508](https://github.com/vllm-project/vllm/pull/22508) | merged | [oss] Init gpt-oss bf16 support | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-13 | [#22678](https://github.com/vllm-project/vllm/pull/22678) | merged | Force TRTLLM attention for gpt-oss on SM100 | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-15 | [#22538](https://github.com/vllm-project/vllm/pull/22538) | merged | [Kernel] Add cuda kernel for gpt_oss activation | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-15 | [#22948](https://github.com/vllm-project/vllm/pull/22948) | merged | Revert "[Kernel] Add cuda kernel for gpt_oss activation" | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-17 | [#22951](https://github.com/vllm-project/vllm/pull/22951) | merged | [Kernel] Add cuda kernel for gpt_oss activation | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-27 | [#23613](https://github.com/vllm-project/vllm/pull/23613) | merged | [Bugfix][gpt-oss] passing the cache config in gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23680](https://github.com/vllm-project/vllm/pull/23680) | merged | [Model] Add PP support and VLM backbone compatability for GPT-OSS | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23815](https://github.com/vllm-project/vllm/pull/23815) | merged | [Model] [gpt-oss] fix gpt-oss pp support | `vllm/model_executor/models/gpt_oss.py` |
| 2025-08-28 | [#23819](https://github.com/vllm-project/vllm/pull/23819) | merged | [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE | `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py` |
| 2025-09-17 | [#24920](https://github.com/vllm-project/vllm/pull/24920) | merged | [CI] GPT-OSS GPQA eval test for Blackwell | `tests/evals/gpt_oss/test_gpqa_correctness.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/__init__.py` |
| 2025-09-22 | [#25246](https://github.com/vllm-project/vllm/pull/25246) | merged | Enable Eagle3 speculative decoding for GPT-OSS model | `vllm/model_executor/models/gpt_oss.py` |
| 2025-10-01 | [#26030](https://github.com/vllm-project/vllm/pull/26030) | merged | [CI] Tweaks to GPT-OSS Eval (Blackwell) for stability | `tests/evals/gpt_oss/test_gpqa_correctness.py` |
| 2025-10-18 | [#25515](https://github.com/vllm-project/vllm/pull/25515) | merged | [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot | `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `vllm/reasoning/gptoss_reasoning_parser.py` |
| 2025-10-21 | [#24032](https://github.com/vllm-project/vllm/pull/24032) | merged | [BugFix] GPT-OSS Attention DP + MoE TP weight loading issue | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-05 | [#27786](https://github.com/vllm-project/vllm/pull/27786) | merged | [XPU] Add gpt-oss model support for Intel GPU | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-11 | [#27334](https://github.com/vllm-project/vllm/pull/27334) | merged | [Quantization] fix attention quantization of gpt_oss model | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-12 | [#28536](https://github.com/vllm-project/vllm/pull/28536) | merged | [Bugfix] Fix gpt_oss packed_modules_mapping | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-16 | [#28715](https://github.com/vllm-project/vllm/pull/28715) | merged | Fixed gpt-oss _load_weights_other() parameter position bug | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-16 | [#28765](https://github.com/vllm-project/vllm/pull/28765) | merged | Fix gpt oss weight loading with EP + bf16 | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-20 | [#28244](https://github.com/vllm-project/vllm/pull/28244) | merged | Add truncate arg to yarn to match openai implementation of gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2025-11-28 | [#29506](https://github.com/vllm-project/vllm/pull/29506) | merged | Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights | `vllm/model_executor/models/gpt_oss.py` |
| 2026-01-28 | [#30976](https://github.com/vllm-project/vllm/pull/30976) | merged | Use aiter triton fused_add_rmsnorm_pad for gpt-oss | `vllm/model_executor/models/gpt_oss.py` |
| 2026-02-10 | [#29008](https://github.com/vllm-project/vllm/pull/29008) | merged | [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations | `vllm/model_executor/models/gpt_oss.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2026-02-11 | [#34337](https://github.com/vllm-project/vllm/pull/34337) | merged | [GPT-OSS] Remove unnecessary contiguous | `vllm/model_executor/models/gpt_oss.py` |
| 2026-02-27 | [#35404](https://github.com/vllm-project/vllm/pull/35404) | merged | [Bugfix][Model] Fix gpt-oss batch invariance | `vllm/model_executor/models/gpt_oss.py` |
| 2026-03-02 | [#35658](https://github.com/vllm-project/vllm/pull/35658) | merged | [ROCm] add amd-quark package in requirements for rocm to use quantized models | `tests/quantization/test_quark.py`, `requirements/rocm.txt` |
| 2026-03-03 | [#35806](https://github.com/vllm-project/vllm/pull/35806) | merged | [ROCm][CI] Fix Assertion Logic For `test_gpt_oss` | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-03 | [#35887](https://github.com/vllm-project/vllm/pull/35887) | merged | [ROCm][CI] Fix TP size issue for `test_gpt_oss` | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-07 | [#36174](https://github.com/vllm-project/vllm/pull/36174) | merged | [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355 | `tests/models/quantization/test_gpt_oss.py` |
| 2026-03-10 | [#36179](https://github.com/vllm-project/vllm/pull/36179) | merged | [ROCm][CI] Fix ROCm GPT-OSS Eval test group | `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt` |
| 2026-03-18 | [#30647](https://github.com/vllm-project/vllm/pull/30647) | merged | [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE | `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` |
| 2026-03-18 | [#37205](https://github.com/vllm-project/vllm/pull/37205) | merged | [Kernel] Add gpt-oss Router GEMM kernel | `vllm/model_executor/models/gpt_oss.py` |
| 2026-03-20 | [#37683](https://github.com/vllm-project/vllm/pull/37683) | merged | [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2026-04-02 | [#38778](https://github.com/vllm-project/vllm/pull/38778) | merged | Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)" | `vllm/model_executor/models/gpt_oss.py` |
| 2026-04-02 | [#38292](https://github.com/vllm-project/vllm/pull/38292) | merged | [CI][ROCm] Add gpt-oss w4a8 in CI | `tests/evals/gpt_oss/configs/models-gfx950.txt` |
| 2026-04-13 | [#39604](https://github.com/vllm-project/vllm/pull/39604) | merged | [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod" | `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py` |
| 2026-04-14 | [#39007](https://github.com/vllm-project/vllm/pull/39007) | merged | [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/ | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-04-26 | [#40338](https://github.com/vllm-project/vllm/pull/40338) | merged | [LoRA] MoE LoRA Refactor | `vllm/lora/layers/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py` |
| 2026-04-27 | [#40860](https://github.com/vllm-project/vllm/pull/40860) | merged | [Feat] DeepSeek V4 Rebased | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-05-05 | [#39136](https://github.com/vllm-project/vllm/pull/39136) | merged | [ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle | `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-12 | [#42334](https://github.com/vllm-project/vllm/pull/42334) | merged | [MoE Refactor] Move remaining experts classes to experts directory | `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/fp8.py` |
| 2026-05-13 | [#41566](https://github.com/vllm-project/vllm/pull/41566) | merged | [Quantization] Rework quantization_config to use QuantKey and allow for activation override | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/online/base.py`, `vllm/model_executor/model_loader/weight_utils.py` |
| 2026-05-15 | [#37826](https://github.com/vllm-project/vllm/pull/37826) | merged | [ROCm] Widen OAI Triton MoE capability range to include gfx12 (RDNA4) | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |
| 2026-05-20 | [#43135](https://github.com/vllm-project/vllm/pull/43135) | merged | [Perf][gpt-oss] Downgrade triton_kernels to v3.5.1 | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |
| 2026-05-24 | [#43385](https://github.com/vllm-project/vllm/pull/43385) | merged | [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP | `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py` |
| 2026-05-29 | [#43108](https://github.com/vllm-project/vllm/pull/43108) | merged | [MoE Refactor] Remove supports_expert_map | `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`, `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py` |
| 2026-05-30 | [#43571](https://github.com/vllm-project/vllm/pull/43571) | merged | [BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py | `tests/models/quantization/test_gpt_oss.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-11 | [#45067](https://github.com/vllm-project/vllm/pull/45067) | merged | [Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor | `vllm/model_executor/models/gpt_oss.py` |
| 2026-06-11 | [#44992](https://github.com/vllm-project/vllm/pull/44992) | merged | Deprecations for v0.23 and v0.24 | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py` |
| 2026-06-15 | [#45381](https://github.com/vllm-project/vllm/pull/45381) | merged | [Model] Add MiniMax M3 support | `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py` |
| 2026-06-17 | [#45896](https://github.com/vllm-project/vllm/pull/45896) | merged | [feature] MiniMax-M3-MXFP4 support added | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/minimax_m3/amd/model.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-06-23 | [#46441](https://github.com/vllm-project/vllm/pull/46441) | merged | fix gpt_oss pp>1 with ep | `vllm/model_executor/models/gpt_oss.py` |
| 2026-06-23 | [#45818](https://github.com/vllm-project/vllm/pull/45818) | merged | [Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r… | `vllm/model_executor/models/gpt_oss.py` |
| 2026-06-23 | [#46142](https://github.com/vllm-project/vllm/pull/46142) | merged | [AMD][OCP MX][CI] Fix tests to not dispatch on `UNFUSED_TRITON` backend on MI300, improve w_mxfp4_a_fp8 emulation support | `vllm/model_executor/layers/fused_moe/utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` |
| 2026-06-24 | [#46406](https://github.com/vllm-project/vllm/pull/46406) | merged | [Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |
| 2026-06-24 | [#46408](https://github.com/vllm-project/vllm/pull/46408) | merged | [Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE | `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` |

## Per-PR Diff Audit Cards

### PR #22327 - Add GPT-OSS model code and config [1/N]

- Link: https://github.com/vllm-project/vllm/pull/22327
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `de98252f497b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +503/-0, 530 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add GPT-OSS model code and config [1/N]"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Add GPT-OSS model code and config [1/N]"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: OAIAttention, __init__, forward, MLPBlock, touching `OAIAttention, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: OAIAttention, __init__, forward, MLPBlock
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -0,0 +1,472 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Optional
+import torch
+import torch.distributed as dist
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` added +472/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22401 - [gpt-oss] fix model config with hf_config

- Link: https://github.com/vllm-project/vllm/pull/22401
- Status/date: merged / 2025-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5c7cc33f4daf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[gpt-oss] fix model config with hf_config"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[gpt-oss] fix model config with hf_config"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +3/-3 (6 lines); hunks: -61,9 +61,9 @@ def __init__(; -154,7 +154,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +3/-3 (6 lines); hunks: -61,9 +61,9 @@ def __init__(; -154,7 +154,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -61,9 +61,9 @@ def __init__(
-                config.rope_ntk_beta,
+                config.rope_scaling["beta_fast"],
-                config.rope_ntk_alpha,
+                config.rope_scaling["beta_slow"],
@@ -154,7 +154,7 @@ def __init__(
-                                top_k=config.num_experts_per_token,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22421 - [gpt-oss] triton kernel mxfp4

- Link: https://github.com/vllm-project/vllm/pull/22421
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; associated commits `e789cad6b8b5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +755/-9, 859 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[gpt-oss] triton kernel mxfp4"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; technical summary: Covers "[gpt-oss] triton kernel mxfp4"; the main implementation surface is `tests/kernels/moe/test_gpt_oss_triton_kernels.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: deshuffle, init_compute_data, ModelConfig, swiglu, touching `deshuffle, init_compute_data, ModelConfig`.
- Code diff details:
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0 (375 lines); hunks: -0,0 +1,375; symbols: deshuffle, init_compute_data, ModelConfig, swiglu
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -0,0 +1,375 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from dataclasses import dataclass, fields
+import pytest
+import torch
+import torch.nn.functional as F
```

- Reviewed files:
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` added +375/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22508 - [oss] Init gpt-oss bf16 support

- Link: https://github.com/vllm-project/vllm/pull/22508
- Status/date: merged / 2025-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `0c5254b82acc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +342/-125, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[oss] Init gpt-oss bf16 support"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[oss] Init gpt-oss bf16 support"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +149/-3 (152 lines); hunks: -160,7 +160,9 @@ def __init__(; -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,; symbols: __init__, forward, compute_logits, load_weights, touching `__init__, forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +149/-3 (152 lines); hunks: -160,7 +160,9 @@ def __init__(; -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,; symbols: __init__, forward, compute_logits, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -160,7 +160,9 @@ def __init__(
-                                apply_router_weight_on_input=False)
+                                apply_router_weight_on_input=False,
+                                has_bias=True,
+                                activation="swiglu_oai")
@@ -262,8 +264,8 @@ def compute_logits(self, hidden_states: torch.Tensor,
-    def load_weights(self, weights: Iterable[tuple[str,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +149/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22678 - Force TRTLLM attention for gpt-oss on SM100

- Link: https://github.com/vllm-project/vllm/pull/22678
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `c6b928798e96`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +20/-9, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Force TRTLLM attention for gpt-oss on SM100"; model line: GPT-OSS; category: model implementation change; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Force TRTLLM attention for gpt-oss on SM100"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-4 (5 lines); hunks: -8,7 +8,6; -70,11 +69,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-4 (5 lines); hunks: -8,7 +8,6; -70,11 +69,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -8,7 +8,6 @@
-from vllm import envs
@@ -70,11 +69,9 @@ def __init__(
-        attention_sink_dtype = (torch.float32 if envs.VLLM_USE_TRTLLM_ATTENTION
-                                else torch.bfloat16)
-                        dtype=attention_sink_dtype,
+                        dtype=torch.bfloat16,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`, `vllm/utils/flashinfer.py`, `vllm/v1/attention/backends/flashinfer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22538 - [Kernel] Add cuda kernel for gpt_oss activation

- Link: https://github.com/vllm-project/vllm/pull/22538
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `81f4b9648117`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +150/-24, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add cuda kernel for gpt_oss activation"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Kernel] Add cuda kernel for gpt_oss activation"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swiglu_oai")
+                                activation="swigluoai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22948 - Revert "[Kernel] Add cuda kernel for gpt_oss activation"

- Link: https://github.com/vllm-project/vllm/pull/22948
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `f1f0d2fab8a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +24/-150, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Kernel] Add cuda kernel for gpt_oss activation""; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Revert "[Kernel] Add cuda kernel for gpt_oss activation""; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swigluoai")
+                                activation="swiglu_oai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22951 - [Kernel] Add cuda kernel for gpt_oss activation

- Link: https://github.com/vllm-project/vllm/pull/22951
- Status/date: merged / 2025-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `4d4061b6e73d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +157/-42, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add cuda kernel for gpt_oss activation"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Kernel] Add cuda kernel for gpt_oss activation"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -159,7 +159,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -159,7 +159,7 @@ def __init__(
-                                activation="swiglu_oai")
+                                activation="swigluoai")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23613 - [Bugfix][gpt-oss] passing the cache config in gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/23613
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `fecbb7c78298`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][gpt-oss] passing the cache config in gpt-oss"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Bugfix][gpt-oss] passing the cache config in gpt-oss"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +6/-1 (7 lines); hunks: -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):; -203,6 +206,7 @@ def __init__(; symbols: TransformerBlock, __init__, touching `TransformerBlock, __init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +6/-1 (7 lines); hunks: -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):; -203,6 +206,7 @@ def __init__(; symbols: TransformerBlock, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -174,12 +174,15 @@ class TransformerBlock(torch.nn.Module):
+        cache_config: CacheConfig,
-        self.attn = OAIAttention(config, prefix=f"{prefix}.attn")
+        self.attn = OAIAttention(config,
+                                 prefix=f"{prefix}.attn",
+                                 cache_config=cache_config)
@@ -203,6 +206,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23680 - [Model] Add PP support and VLM backbone compatability for GPT-OSS

- Link: https://github.com/vllm-project/vllm/pull/23680
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `c5d004aaaf3b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +87/-34, 232 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add PP support and VLM backbone compatability for GPT-OSS"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Model] Add PP support and VLM backbone compatability for GPT-OSS"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +86/-33 (119 lines); hunks: -11,7 +11,8; -27,7 +28,10; symbols: __init__, forward, MLPBlock, touching `__init__, forward, MLPBlock`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +86/-33 (119 lines); hunks: -11,7 +11,8; -27,7 +28,10; symbols: __init__, forward, MLPBlock
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -11,7 +11,8 @@
-from vllm.distributed import (get_ep_group, get_tensor_model_parallel_rank,
+from vllm.distributed import (get_ep_group, get_pp_group,
+                              get_tensor_model_parallel_rank,
@@ -27,7 +28,10 @@
+from .interfaces import SupportsPP
+                    is_pp_missing_parameter,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +86/-33
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23815 - [Model] [gpt-oss] fix gpt-oss pp support

- Link: https://github.com/vllm-project/vllm/pull/23815
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `bfab219648fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-3, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] [gpt-oss] fix gpt-oss pp support"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Model] [gpt-oss] fix gpt-oss pp support"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +2/-3 (5 lines); hunks: -668,9 +668,8 @@ def forward(self,; symbols: forward, compute_logits, touching `forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +2/-3 (5 lines); hunks: -668,9 +668,8 @@ def forward(self,; symbols: forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -668,9 +668,8 @@ def forward(self,
-        assert intermediate_tensors is None
-        assert inputs_embeds is None
-        return self.model(input_ids, positions)
+        return self.model(input_ids, positions, intermediate_tensors,
+                          inputs_embeds)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23819 - [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE

- Link: https://github.com/vllm-project/vllm/pull/23819
- Status/date: merged / 2025-08-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +14/-15, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; technical summary: Covers "[Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE"; the main implementation surface is `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunks: -190,12 +190,6 @@ def use_deepep_ll_kernels(self):; -404,7 +398,14 @@ def use_deepep_ll_kernels(self):; symbols: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make, touching `use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make`; `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunks: -920,7 +920,7 @@ def __init__(; -974,7 +974,7 @@ def use_deepep_ll_kernels(self):; symbols: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map, touching `__init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels`; `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunks: -623,8 +623,6 @@ def apply(; -650,12 +648,12 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunks: -190,12 +190,6 @@ def use_deepep_ll_kernels(self):; -404,7 +398,14 @@ def use_deepep_ll_kernels(self):; symbols: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunks: -920,7 +920,7 @@ def __init__(; -974,7 +974,7 @@ def use_deepep_ll_kernels(self):; symbols: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunks: -623,8 +623,6 @@ def apply(; -650,12 +648,12 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/config.py
@@ -190,12 +190,6 @@ def use_deepep_ll_kernels(self):
-    @property
-    def use_flashinfer_cutlass_kernels(self):
-        return (envs.VLLM_USE_FLASHINFER_MOE_FP4
-                and has_flashinfer_cutlass_fused_moe()
-                and envs.VLLM_FLASHINFER_MOE_BACKEND == "throughput")
@@ -404,7 +398,14 @@ def use_deepep_ll_kernels(self):
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -920,7 +920,7 @@ def __init__(
-                or self.moe_parallel_config.use_flashinfer_cutlass_kernels):
+                or self.moe_config.use_flashinfer_cutlass_kernels):
@@ -974,7 +974,7 @@ def use_deepep_ll_kernels(self):
-        return self.moe_parallel_config.use_flashinfer_cutlass_kernels
+        return self.moe_config.use_flashinfer_cutlass_kernels
@@ -1665,7 +1665,7 @@ def forward_impl(self, hidden_states: torch.Tensor,
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -623,8 +623,6 @@ def apply(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7; `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4; `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24920 - [CI] GPT-OSS GPQA eval test for Blackwell

- Link: https://github.com/vllm-project/vllm/pull/24920
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/__init__.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/test_gpqa_correctness.py`; associated commits `493b10f8bf38`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +136/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] GPT-OSS GPQA eval test for Blackwell"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/evals/gpt_oss/test_gpqa_correctness.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/__init__.py`; technical summary: Covers "[CI] GPT-OSS GPQA eval test for Blackwell"; the main implementation surface is `tests/evals/gpt_oss/test_gpqa_correctness.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: run_gpqa_eval, test_gpqa_correctness, touching `run_gpqa_eval, test_gpqa_correctness`; `tests/evals/gpt_oss/conftest.py` added +18/-0 (18 lines); hunks: -0,0 +1,18; symbols: pytest_addoption, touching `pytest_addoption`; `tests/evals/gpt_oss/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: run_gpqa_eval, test_gpqa_correctness
  - `tests/evals/gpt_oss/conftest.py` added +18/-0 (18 lines); hunks: -0,0 +1,18; symbols: pytest_addoption
  - `tests/evals/gpt_oss/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/test_gpqa_correctness.py
@@ -0,0 +1,102 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+GPQA evaluation using vLLM server and GPT-OSS evaluation package.
+Usage:
+pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
diff -- tests/evals/gpt_oss/conftest.py
@@ -0,0 +1,18 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Pytest configuration for GPT-OSS evaluation tests.
+"""
+def pytest_addoption(parser):
diff -- tests/evals/gpt_oss/__init__.py
@@ -0,0 +1,2 @@
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/test_gpqa_correctness.py` added +102/-0; `tests/evals/gpt_oss/conftest.py` added +18/-0; `tests/evals/gpt_oss/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/__init__.py`, `tests/evals/gpt_oss/conftest.py`, `tests/evals/gpt_oss/test_gpqa_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25246 - Enable Eagle3 speculative decoding for GPT-OSS model

- Link: https://github.com/vllm-project/vllm/pull/25246
- Status/date: merged / 2025-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `21467f9a1c62`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +41/-12, 111 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable Eagle3 speculative decoding for GPT-OSS model"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Enable Eagle3 speculative decoding for GPT-OSS model"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunks: -28,7 +28,7; -239,6 +239,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, _load_weights_mxfp4, touching `__init__, get_input_embeddings, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunks: -28,7 +28,7; -239,6 +239,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, _load_weights_mxfp4
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -28,7 +28,7 @@
-from .interfaces import SupportsPP
+from .interfaces import SupportsEagle3, SupportsPP
@@ -239,6 +239,7 @@ def __init__(
+        self.aux_hidden_state_layers = tuple[int, ...]()
@@ -262,15 +263,22 @@ def forward(
+        aux_hidden_states = []
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +17/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/v1/spec_decode/eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26030 - [CI] Tweaks to GPT-OSS Eval (Blackwell) for stability

- Link: https://github.com/vllm-project/vllm/pull/26030
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/test_gpqa_correctness.py`; associated commits `ee04c0cd04cf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-4, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Tweaks to GPT-OSS Eval (Blackwell) for stability"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/evals/gpt_oss/test_gpqa_correctness.py`; technical summary: Covers "[CI] Tweaks to GPT-OSS Eval (Blackwell) for stability"; the main implementation surface is `tests/evals/gpt_oss/test_gpqa_correctness.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3 (5 lines); hunks: -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:; -72,8 +73,6 @@ def test_gpqa_correctness(request):; symbols: run_gpqa_eval, test_gpqa_correctness, touching `run_gpqa_eval, test_gpqa_correctness`.
- Code diff details:
  - `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3 (5 lines); hunks: -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:; -72,8 +73,6 @@ def test_gpqa_correctness(request):; symbols: run_gpqa_eval, test_gpqa_correctness
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/test_gpqa_correctness.py
@@ -26,7 +26,8 @@ def run_gpqa_eval(model_name: str, base_url: str) -> float:
-        model_name, "--reasoning-effort", "low", "--base-url", base_url
+        model_name, "--reasoning-effort", "low", "--base-url", base_url,
+        "--n-threads", "200"
@@ -72,8 +73,6 @@ def test_gpqa_correctness(request):
-        "--max-model-len",
-        "32768",
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/test_gpqa_correctness.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/test_gpqa_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25515 - [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot

- Link: https://github.com/vllm-project/vllm/pull/25515
- Status/date: merged / 2025-10-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +911/-32, 1107 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `vllm/reasoning/gptoss_reasoning_parser.py`; technical summary: Covers "[GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot"; the main implementation surface is `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `vllm/reasoning/gptoss_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser, tool_server_with_python, touching `TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser`; `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config, mock_vllm_config, touching `TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config`; `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunks: -1,17 +1,61; -81,3 +125,33 @@ def extract_reasoning_content(; symbols: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content, touching `from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser`; `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0 (46 lines); hunks: -864,3 +864,49 @@ def test_structured_output_batched_with_non_structured_outp...; symbols: test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag, touching `test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag`.
- Code diff details:
  - `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: TestGptOssStructuralTagsIntegration, mock_tokenizer, gptoss_parser, tool_server_with_python
  - `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: TestReasoningStructuredOutput, mock_model_config, mock_scheduler_config, mock_vllm_config
  - `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunks: -1,17 +1,61; -81,3 +125,33 @@ def extract_reasoning_content(; symbols: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content
  - `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0 (46 lines); hunks: -864,3 +864,49 @@ def test_structured_output_batched_with_non_structured_outp...; symbols: test_structured_output_batched_with_non_structured_outputs_requests, test_structured_output_with_structural_tag
  - `vllm/entrypoints/openai/protocol.py` modified +21/-5 (26 lines); hunks: -200,27 +200,39 @@ class JsonSchemaResponseFormat(OpenAIBaseModel):; -823,7 +835,11 @@ def to_sampling_params(; symbols: JsonSchemaResponseFormat, StructuralTag, LegacyStructuralTag, StructuralTagResponseFormat
- Key code excerpts:

```diff
diff -- tests/entrypoints/openai/test_gptoss_structural_tags_integration.py
@@ -0,0 +1,280 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Integration tests for GPT-OSS structural tags functionality (PR #25515)."""
+import json
+from unittest.mock import Mock
+import pytest
diff -- tests/v1/structured_output/test_reasoning_structured_output.py
@@ -0,0 +1,207 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""
+from unittest.mock import Mock
+import pytest
+from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
diff -- vllm/reasoning/gptoss_reasoning_parser.py
@@ -1,17 +1,61 @@
```

- Reviewed files:
  - tests: `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0; `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0; `tests/v1/entrypoints/llm/test_struct_output_generate.py` modified +46/-0; `tests/v1/structured_output/test_gptoss_structural_tags.py` added +172/-0
  - runtime: `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1; `vllm/entrypoints/openai/protocol.py` modified +21/-5; `vllm/entrypoints/openai/serving_responses.py` modified +15/-1; `vllm/reasoning/abs_reasoning_parsers.py` modified +12/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/entrypoints/llm/test_struct_output_generate.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24032 - [BugFix] GPT-OSS Attention DP + MoE TP weight loading issue

- Link: https://github.com/vllm-project/vllm/pull/24032
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `aef368aa0857`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +30/-13, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] GPT-OSS Attention DP + MoE TP weight loading issue"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[BugFix] GPT-OSS Attention DP + MoE TP weight loading issue"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +16/-4 (20 lines); hunks: -11,13 +11,15; -305,8 +307,13 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other, touching `_load_weights_mxfp4, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +16/-4 (20 lines); hunks: -11,13 +11,15; -305,8 +307,13 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -11,13 +11,15 @@
+    get_dp_group,
+from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
@@ -305,8 +307,13 @@ def _load_weights_mxfp4(
-        tp_rank = get_tensor_model_parallel_rank()
-        tp_size = get_tensor_model_parallel_world_size()
+        # In MoE, we need to flatten the tensor parallel size across the data
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +16/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27786 - [XPU] Add gpt-oss model support for Intel GPU

- Link: https://github.com/vllm-project/vllm/pull/27786
- Status/date: merged / 2025-11-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `18b39828d904`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +101/-6, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU] Add gpt-oss model support for Intel GPU"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[XPU] Add gpt-oss model support for Intel GPU"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +0/-3 (3 lines); hunks: -329,9 +329,6 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, touching `_load_weights_mxfp4`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-3 (3 lines); hunks: -329,9 +329,6 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -329,9 +329,6 @@ def _load_weights_mxfp4(
-            # FIXME(woosuk): Remove this after testing.
-            weight = weight.cuda()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/attention/utils/fa_utils.py`, `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27334 - [Quantization] fix attention quantization of gpt_oss model

- Link: https://github.com/vllm-project/vllm/pull/27334
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5a1271d83a65`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +101/-4, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] fix attention quantization of gpt_oss model"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Quantization] fix attention quantization of gpt_oss model"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +8/-2 (10 lines); hunks: -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):; -207,7 +208,10 @@ def __init__(; symbols: TransformerBlock, __init__, touching `TransformerBlock, __init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +8/-2 (10 lines); hunks: -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):; -207,7 +208,10 @@ def __init__(; symbols: TransformerBlock, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -198,6 +198,7 @@ class TransformerBlock(torch.nn.Module):
+        quant_config: QuantizationConfig,
@@ -207,7 +208,10 @@ def __init__(
-            config, prefix=f"{prefix}.attn", cache_config=cache_config
+            config,
+            prefix=f"{prefix}.attn",
+            quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss_attn_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28536 - [Bugfix] Fix gpt_oss packed_modules_mapping

- Link: https://github.com/vllm-project/vllm/pull/28536
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `a9d18b51078d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix gpt_oss packed_modules_mapping"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Bugfix] Fix gpt_oss packed_modules_mapping"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +5/-5 (10 lines); hunks: -92,7 +92,7 @@ def __init__(; -129,7 +129,7 @@ def __init__(; symbols: __init__, forward, _load_weights_other, load_weights, touching `__init__, forward, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +5/-5 (10 lines); hunks: -92,7 +92,7 @@ def __init__(; -129,7 +129,7 @@ def __init__(; symbols: __init__, forward, _load_weights_other, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -92,7 +92,7 @@ def __init__(
-        self.qkv = QKVParallelLinear(
+        self.qkv_proj = QKVParallelLinear(
@@ -129,7 +129,7 @@ def __init__(
-        qkv, _ = self.qkv(hidden_states)
+        qkv, _ = self.qkv_proj(hidden_states)
@@ -606,9 +606,9 @@ def _load_weights_other(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28715 - Fixed gpt-oss _load_weights_other() parameter position bug

- Link: https://github.com/vllm-project/vllm/pull/28715
- Status/date: merged / 2025-11-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `af02c409702f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fixed gpt-oss _load_weights_other() parameter position bug"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Fixed gpt-oss _load_weights_other() parameter position bug"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -641,8 +641,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                ep_rank_end,
+                ep_rank_end,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28765 - Fix gpt oss weight loading with EP + bf16

- Link: https://github.com/vllm-project/vllm/pull/28765
- Status/date: merged / 2025-11-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `8d259fad6cd5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gpt oss weight loading with EP + bf16"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Fix gpt oss weight loading with EP + bf16"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -494,8 +494,8 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other, touching `_load_weights_mxfp4, _load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -494,8 +494,8 @@ def _load_weights_mxfp4(; symbols: _load_weights_mxfp4, _load_weights_other
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -494,8 +494,8 @@ def _load_weights_mxfp4(
-        ep_rank_start: int,
+        ep_rank_start: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28244 - Add truncate arg to yarn to match openai implementation of gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/28244
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `6eb745d9bdf5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +12/-7, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add truncate arg to yarn to match openai implementation of gpt-oss"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Add truncate arg to yarn to match openai implementation of gpt-oss"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-0 (1 lines); hunks: -78,6 +78,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -78,6 +78,7 @@ def __init__(
+                "truncate": config.rope_parameters.get("truncate", True),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/model_executor/layers/rotary_embedding/common.py`, `vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29506 - Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights

- Link: https://github.com/vllm-project/vllm/pull/29506
- Status/date: merged / 2025-11-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `5f5521bd5d7d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -647,8 +647,8 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                ep_rank_start,
+                ep_rank_start,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30976 - Use aiter triton fused_add_rmsnorm_pad for gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/30976
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `59bcc5b6f2e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +327/-11, 489 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use aiter triton fused_add_rmsnorm_pad for gpt-oss"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Use aiter triton fused_add_rmsnorm_pad for gpt-oss"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -187,7 +187,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-        x = self.experts(hidden_states=x, router_logits=g)
+        x = self.experts(hidden_states=x, router_logits=g)[:, : self.hidden_size]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/compile/test_fuse_act_padding.py`, `tests/compile/test_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29008 - [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations

- Link: https://github.com/vllm-project/vllm/pull/29008
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `vllm/model_executor/models/gpt_oss.py`; associated commits `b129136c7a73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1094/-213, 1860 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/models/gpt_oss.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; technical summary: Covers "[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +491/-18 (509 lines); hunks: -1,6 +1,7; -25,13 +26,17; symbols: __init__, forward, get_expert_mapping, _load_weights_mxfp4, touching `__init__, forward, get_expert_mapping`; `tests/models/quantization/test_gpt_oss.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: has_huggingface_access, ModelCase, EvaluationConfig, get_model_args, touching `has_huggingface_access, ModelCase, EvaluationConfig`; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7 (20 lines); hunks: -22,7 +22,7; -298,12 +298,18 @@ def test_equiv(num_token, a_dtype, w_dtype, tp, workspace_...; symbols: test_equiv, touching `test_equiv`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +491/-18 (509 lines); hunks: -1,6 +1,7; -25,13 +26,17; symbols: __init__, forward, get_expert_mapping, _load_weights_mxfp4
  - `tests/models/quantization/test_gpt_oss.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: has_huggingface_access, ModelCase, EvaluationConfig, get_model_args
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7 (20 lines); hunks: -22,7 +22,7; -298,12 +298,18 @@ def test_equiv(num_token, a_dtype, w_dtype, tp, workspace_...; symbols: test_equiv
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -1,6 +1,7 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -25,13 +26,17 @@
+from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
diff -- tests/models/quantization/test_gpt_oss.py
@@ -0,0 +1,110 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+End-to-end accuracy test for GPT-OSS model quantization.
+Config:
+    Task:   gsm8k_platinum
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -22,7 +22,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +491/-18
  - tests: `tests/models/quantization/test_gpt_oss.py` added +110/-0; `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +13/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/models/quantization/test_gpt_oss_attn_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34337 - [GPT-OSS] Remove unnecessary contiguous

- Link: https://github.com/vllm-project/vllm/pull/34337
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `83e26c834ef1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GPT-OSS] Remove unnecessary contiguous"; model line: GPT-OSS; category: model implementation change; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[GPT-OSS] Remove unnecessary contiguous"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +0/-1 (1 lines); hunks: -140,7 +140,6 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-1 (1 lines); hunks: -140,7 +140,6 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -140,7 +140,6 @@ def forward(
-        v = v.contiguous()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35404 - [Bugfix][Model] Fix gpt-oss batch invariance

- Link: https://github.com/vllm-project/vllm/pull/35404
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `1f3dbd95fd13`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-8, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Fix gpt-oss batch invariance"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Bugfix][Model] Fix gpt-oss batch invariance"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +13/-2 (15 lines); hunks: -23,7 +23,11; -165,7 +169,14 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +13/-2 (15 lines); hunks: -23,7 +23,11; -165,7 +169,14 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -23,7 +23,11 @@
-from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
+from vllm.model_executor.layers.linear import (
+    QKVParallelLinear,
+    ReplicatedLinear,
+    RowParallelLinear,
+)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +13/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35658 - [ROCm] add amd-quark package in requirements for rocm to use quantized models

- Link: https://github.com/vllm-project/vllm/pull/35658
- Status/date: merged / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +24/-6, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] add amd-quark package in requirements for rocm to use quantized models"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/quantization/test_quark.py`, `requirements/rocm.txt`; technical summary: Covers "[ROCm] add amd-quark package in requirements for rocm to use quantized models"; the main implementation surface is `tests/quantization/test_quark.py`, `requirements/rocm.txt`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark, touching `get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness`; `requirements/rocm.txt` modified +4/-1 (5 lines); hunks: -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0.
- Code diff details:
  - `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark
  - `requirements/rocm.txt` modified +4/-1 (5 lines); hunks: -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0
- Key code excerpts:

```diff
diff -- tests/quantization/test_quark.py
@@ -26,9 +26,12 @@
+# Minimum amd-quark version for MXFP4/OCP_MX tests (single source of truth).
+QUARK_MXFP4_MIN_VERSION = "0.8.99"
-) >= version.parse("0.8.99")
+) >= version.parse(QUARK_MXFP4_MIN_VERSION)
@@ -200,7 +203,10 @@ def get_model_args(
-@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
diff -- requirements/rocm.txt
@@ -19,4 +19,7 @@ setuptools>=77.0.3,<80.0.0
-timm>=1.0.17
+timm>=1.0.17
+# amd-quark: required for Quark quantization on ROCm
+# To be consistent with test_quark.py
+amd-quark>=0.8.99
```

- Reviewed files:
  - tests: `tests/quantization/test_quark.py` modified +20/-5
  - other: `requirements/rocm.txt` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35806 - [ROCm][CI] Fix Assertion Logic For `test_gpt_oss`

- Link: https://github.com/vllm-project/vllm/pull/35806
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `8b9e8b74541e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix Assertion Logic For `test_gpt_oss`"; model line: GPT-OSS; category: bug fix; main diff: `tests/models/quantization/test_gpt_oss.py`; technical summary: Covers "[ROCm][CI] Fix Assertion Logic For `test_gpt_oss`"; the main implementation surface is `tests/models/quantization/test_gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +5/-5 (10 lines); hunks: -12,8 +12,8; -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization, touching `test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +5/-5 (10 lines); hunks: -12,8 +12,8; -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -12,8 +12,8 @@
-import importlib
+import importlib.util
@@ -104,7 +104,7 @@ def test_gpt_oss_attention_quantization(
-    assert (
-        measured_accuracy - rtol < expected_accuracy
-        and measured_accuracy + rtol > expected_accuracy
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +5/-5
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35887 - [ROCm][CI] Fix TP size issue for `test_gpt_oss`

- Link: https://github.com/vllm-project/vllm/pull/35887
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `e7213003cbf6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix TP size issue for `test_gpt_oss`"; model line: GPT-OSS; category: bug fix; main diff: `tests/models/quantization/test_gpt_oss.py`; technical summary: Covers "[ROCm][CI] Fix TP size issue for `test_gpt_oss`"; the main implementation surface is `tests/models/quantization/test_gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -21,6 +21,8; -83,6 +85,9 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization, touching `get_model_args, test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -21,6 +21,8; -83,6 +85,9 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -21,6 +21,8 @@
+from vllm.utils.torch_utils import cuda_device_count_stateless
@@ -83,6 +85,9 @@ def get_model_args(self, tp_size: int):
+    if tp_size > cuda_device_count_stateless():
+        pytest.skip("Not enough GPUs to run this test case")
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36174 - [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355

- Link: https://github.com/vllm-project/vllm/pull/36174
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `fc4657756ff0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/models/quantization/test_gpt_oss.py`; technical summary: Covers "[ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355"; the main implementation surface is `tests/models/quantization/test_gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -21,6 +21,7; -83,11 +84,17 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization, touching `get_model_args, test_gpt_oss_attention_quantization`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -21,6 +21,7; -83,11 +84,17 @@ def get_model_args(self, tp_size: int):; symbols: get_model_args, test_gpt_oss_attention_quantization
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -21,6 +21,7 @@
+from vllm.platforms.rocm import on_gfx950
@@ -83,11 +84,17 @@ def get_model_args(self, tp_size: int):
-    model_name: str, tp_size: int, expected_accuracy: float
+    model_name: str,
+    tp_size: int,
+    expected_accuracy: float,
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36179 - [ROCm][CI] Fix ROCm GPT-OSS Eval test group

- Link: https://github.com/vllm-project/vllm/pull/36179
- Status/date: merged / 2026-03-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; associated commits `179547d62c73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +16/-4, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix ROCm GPT-OSS Eval test group"; model line: GPT-OSS; category: bug fix; main diff: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; technical summary: Covers "[ROCm][CI] Fix ROCm GPT-OSS Eval test group"; the main implementation surface is `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6; `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3; `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3.
- Code diff details:
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3
  - `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0 (3 lines); hunks: -0,0 +1,3
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml
@@ -0,0 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+model_name: openai/gpt-oss-20b
+metric_threshold: 0.568
+reasoning_effort: low
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN"
diff -- tests/evals/gpt_oss/configs/models-gfx942.txt
@@ -0,0 +1,3 @@
+# GFX942 model configurations for GPQA evaluation
+# Tests different environment variable combinations
+gpt-oss-20b-rocm-baseline.yaml
diff -- tests/evals/gpt_oss/configs/models-gfx950.txt
@@ -0,0 +1,3 @@
+# GFX950 model configurations for GPQA evaluation
+# Tests different environment variable combinations
+gpt-oss-20b-rocm-baseline.yaml
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml` added +6/-0; `tests/evals/gpt_oss/configs/models-gfx942.txt` added +3/-0; `tests/evals/gpt_oss/configs/models-gfx950.txt` added +3/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/models-gfx942.txt`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30647 - [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- Link: https://github.com/vllm-project/vllm/pull/30647
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +40/-3, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`; technical summary: Covers "[Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE"; the main implementation surface is `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunks: -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):; -1130,9 +1136,17 @@ def apply_monolithic(; symbols: __init__, skip_forward_padding, create_weights, apply_monolithic, touching `__init__, skip_forward_padding, create_weights`; `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunks: -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:; symbols: topk_indices_dtype, skip_forward_padding, supports_eplb, touching `topk_indices_dtype, skip_forward_padding, supports_eplb`; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunks: -415,7 +415,10 @@ def forward(; symbols: forward, touching `forward`; `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunks: -162,3 +162,12.
- Code diff details:
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunks: -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):; -1130,9 +1136,17 @@ def apply_monolithic(; symbols: __init__, skip_forward_padding, create_weights, apply_monolithic
  - `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunks: -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:; symbols: topk_indices_dtype, skip_forward_padding, supports_eplb
  - `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunks: -415,7 +415,10 @@ def forward(; symbols: forward
  - `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunks: -162,3 +162,12
  - `tests/compile/fusions_e2e/conftest.py` modified +4/-0 (4 lines); hunks: -82,6 +82,10 @@ def run(; symbols: run
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -294,6 +294,12 @@ def __init__(self, moe: FusedMoEConfig):
+    @property
+    def skip_forward_padding(self) -> bool:
+        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
+        # so can skip the padding in the forward before applying the moe method
+        return self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
@@ -1130,9 +1136,17 @@ def apply_monolithic(
diff -- vllm/model_executor/layers/fused_moe/fused_moe_method_base.py
@@ -101,6 +101,11 @@ def topk_indices_dtype(self) -> torch.dtype | None:
+    @property
+    def skip_forward_padding(self) -> bool:
+        """Whether to skip the padding in the forward before applying the moe method."""
+        return False
diff -- vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py
@@ -415,7 +415,10 @@ def forward(
-        if self.moe_config.hidden_dim != transformed_hidden_dim:
+        if (
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1; `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0; `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1
  - tests: `tests/compile/fusions_e2e/models.py` modified +9/-0; `tests/compile/fusions_e2e/conftest.py` modified +4/-0; `tests/compile/fusions_e2e/test_tp2_ar_rms.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/compile/fusions_e2e/models.py`, `tests/compile/fusions_e2e/test_tp2_ar_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37205 - [Kernel] Add gpt-oss Router GEMM kernel

- Link: https://github.com/vllm-project/vllm/pull/37205
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `9bd723110689`, `b1169d7be8ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +875/-13, 1035 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add gpt-oss Router GEMM kernel"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Kernel] Add gpt-oss Router GEMM kernel"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +3/-7 (10 lines); hunks: -20,12 +20,11; -175,13 +174,11 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +3/-7 (10 lines); hunks: -20,12 +20,11; -175,13 +174,11 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -20,12 +20,11 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
-    ReplicatedLinear,
@@ -175,13 +174,11 @@ def __init__(
-        self.router = ReplicatedLinear(
+        self.router = GateLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +3/-7
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37683 - [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels

- Link: https://github.com/vllm-project/vllm/pull/37683
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; associated commits `d0532bf38da5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-4, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels"; model line: GPT-OSS; category: performance/backend optimization; main diff: `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; technical summary: Covers "[Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels"; the main implementation surface is `tests/kernels/moe/test_gpt_oss_triton_kernels.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0 (44 lines); hunks: -21,12 +21,16; -355,3 +359,43 @@ def test_unit_shuffle():; symbols: test_unit_shuffle, test_legacy_routing, touching `test_unit_shuffle, test_legacy_routing`.
- Code diff details:
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0 (44 lines); hunks: -21,12 +21,16; -355,3 +359,43 @@ def test_unit_shuffle():; symbols: test_unit_shuffle, test_legacy_routing
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -21,12 +21,16 @@
+from triton_kernels.topk import topk as topk_fn
+    legacy_routing,
+    make_routing_data,
+from vllm.utils.torch_utils import set_random_seed
@@ -355,3 +359,43 @@ def test_unit_shuffle():
+@pytest.mark.parametrize("num_tokens", [2, 8, 64])
```

- Reviewed files:
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +44/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38778 - Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)"

- Link: https://github.com/vllm-project/vllm/pull/38778
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `9bd723110689`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +12/-875, 1027 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)""; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)""; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -20,11 +20,12; -174,11 +175,13 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -20,11 +20,12; -174,11 +175,13 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -20,11 +20,12 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
+from vllm.model_executor.layers.fused_moe import FusedMoE
+    ReplicatedLinear,
@@ -174,11 +175,13 @@ def __init__(
-        self.router = GateLinear(
+        self.router = ReplicatedLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38292 - [CI][ROCm] Add gpt-oss w4a8 in CI

- Link: https://github.com/vllm-project/vllm/pull/38292
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/models-gfx950.txt`; associated commits `82a006beebf0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI][ROCm] Add gpt-oss w4a8 in CI"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/evals/gpt_oss/configs/models-gfx950.txt`; technical summary: Covers "[CI][ROCm] Add gpt-oss w4a8 in CI"; the main implementation surface is `tests/evals/gpt_oss/configs/models-gfx950.txt`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1 (3 lines); hunks: -1,3 +1,4.
- Code diff details:
  - `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1 (3 lines); hunks: -1,3 +1,4
- Key code excerpts:

```diff
diff -- tests/evals/gpt_oss/configs/models-gfx950.txt
@@ -1,3 +1,4 @@
-gpt-oss-20b-rocm-baseline.yaml
+gpt-oss-20b-rocm-baseline.yaml
+gpt-oss-20b-rocm-mxfp4-fp8.yaml
```

- Reviewed files:
  - tests: `tests/evals/gpt_oss/configs/models-gfx950.txt` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-mxfp4-fp8.yaml`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39604 - [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod"

- Link: https://github.com/vllm-project/vllm/pull/39604
- Status/date: merged / 2026-04-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +129/-34, 431 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod""; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`; technical summary: Covers "[Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod""; the main implementation surface is `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/mxfp4.py` modified +48/-7 (55 lines); hunks: -19,11 +19,11; -38,6 +38,12; symbols: Mxfp4Config, __init__, get_supported_act_dtypes, get_config_filenames, touching `Mxfp4Config, __init__, get_supported_act_dtypes`; `vllm/model_executor/models/gpt_oss.py` modified +25/-3 (28 lines); hunks: -560,6 +560,14 @@ def _load_weights_quark(; -578,7 +586,7 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _load_weights_quark, _is_mxfp4, _get_moe_weight_dtype, kv_cache_scale_loader, touching `_load_weights_quark, _is_mxfp4, _get_moe_weight_dtype`; `vllm/model_executor/models/config.py` modified +17/-0 (17 lines); hunks: -108,6 +108,23 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; symbols: verify_and_update_config, GptOssForCausalLMConfig, verify_and_update_model_config, touching `verify_and_update_config, GptOssForCausalLMConfig, verify_and_update_model_config`; `vllm/model_executor/layers/quantization/base_config.py` modified +11/-2 (13 lines); hunks: -110,13 +110,22 @@ def from_config(cls, config: dict[str, Any]) -> "Quantizat...; symbols: from_config, override_quantization_method, touching `from_config, override_quantization_method`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +48/-7 (55 lines); hunks: -19,11 +19,11; -38,6 +38,12; symbols: Mxfp4Config, __init__, get_supported_act_dtypes, get_config_filenames
  - `vllm/model_executor/models/gpt_oss.py` modified +25/-3 (28 lines); hunks: -560,6 +560,14 @@ def _load_weights_quark(; -578,7 +586,7 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _load_weights_quark, _is_mxfp4, _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/models/config.py` modified +17/-0 (17 lines); hunks: -108,6 +108,23 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; symbols: verify_and_update_config, GptOssForCausalLMConfig, verify_and_update_model_config
  - `vllm/model_executor/layers/quantization/base_config.py` modified +11/-2 (13 lines); hunks: -110,13 +110,22 @@ def from_config(cls, config: dict[str, Any]) -> "Quantizat...; symbols: from_config, override_quantization_method
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-3 (9 lines); hunks: -194,7 +194,7 @@ def _backend_activation_key(backend: Mxfp4MoeBackend) -> Qua...; -400,7 +400,7 @@ def mxfp4_round_up_hidden_size_and_intermediate_size(; symbols: _backend_activation_key, select_mxfp4_moe_backend, select_gpt_oss_mxfp4_moe_backend, mxfp4_round_up_hidden_size_and_intermediate_size
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/mxfp4.py
@@ -19,11 +19,11 @@
-    convert_to_mxfp4_moe_kernel_format,
+    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
-    select_mxfp4_moe_backend,
+    select_gpt_oss_mxfp4_moe_backend,
@@ -38,6 +38,12 @@
+    """Canonical base config for MXFP4 quantization.
diff -- vllm/model_executor/models/gpt_oss.py
@@ -560,6 +560,14 @@ def _load_weights_quark(
+        def _is_mxfp4(weight_dtype: str | None) -> bool:
+            """Return True for any MXFP4 weight-dtype variant.
+            Covers "gpt_oss_mxfp4" (GptOssMxfp4MoEMethod) and "mxfp4"
+            (QuarkMoEMethod with fp4 weights) and any future variants.
+            """
+            return weight_dtype is not None and "mxfp4" in weight_dtype
diff -- vllm/model_executor/models/config.py
@@ -108,6 +108,23 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/mxfp4.py` modified +48/-7; `vllm/model_executor/models/gpt_oss.py` modified +25/-3; `vllm/model_executor/models/config.py` modified +17/-0; `vllm/model_executor/layers/quantization/base_config.py` modified +11/-2; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-3; `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39007 - [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/

- Link: https://github.com/vllm-project/vllm/pull/39007
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; associated commits `1a9353bb02e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +16/-12, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; technical summary: Covers "[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0 (0 lines); `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0 (0 lines)
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7
- Key code excerpts:

```diff
diff -- tests/kernels/moe/test_gpt_oss_triton_kernels.py
@@ -25,7 +25,7 @@
-from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
+from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` renamed +0/-0
  - tests: `tests/kernels/moe/test_gpt_oss_triton_kernels.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_oai_triton_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake, touching `_resolve_layer_name, _moe_forward, _moe_forward_shared`; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__, touching `FusedMoE, __init__`; `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40338 - [LoRA] MoE LoRA Refactor

- Link: https://github.com/vllm-project/vllm/pull/40338
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +736/-328, 1280 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] MoE LoRA Refactor"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/lora/layers/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py`; technical summary: Covers "[LoRA] MoE LoRA Refactor"; the main implementation surface is `vllm/lora/layers/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/lora/layers/fused_moe.py` modified +36/-297 (333 lines); hunks: -1,6 +1,5; -14,31 +13,17; symbols: FusedMoEWithLoRA, __init__, _normalize_keys, _get_lora_moe_configs, touching `FusedMoEWithLoRA, __init__, _normalize_keys`; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +106/-6 (112 lines); hunks: -17,6 +17,7; -655,7 +656,7 @@ def moe_problem_size(; symbols: moe_problem_size, MarlinExperts, supports_expert_map, apply, touching `moe_problem_size, MarlinExperts, supports_expert_map`; `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: LoRAExpertsMixin, in, set_lora_context, supports_lora, touching `LoRAExpertsMixin, in, set_lora_context`; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +52/-3 (55 lines); hunks: -1,7 +1,6; -16,6 +15,7; symbols: apply, UnfusedOAITritonExperts, that, touching `apply, UnfusedOAITritonExperts, that`.
- Code diff details:
  - `vllm/lora/layers/fused_moe.py` modified +36/-297 (333 lines); hunks: -1,6 +1,5; -14,31 +13,17; symbols: FusedMoEWithLoRA, __init__, _normalize_keys, _get_lora_moe_configs
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +106/-6 (112 lines); hunks: -17,6 +17,7; -655,7 +656,7 @@ def moe_problem_size(; symbols: moe_problem_size, MarlinExperts, supports_expert_map, apply
  - `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: LoRAExpertsMixin, in, set_lora_context, supports_lora
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +52/-3 (55 lines); hunks: -1,7 +1,6; -16,6 +15,7; symbols: apply, UnfusedOAITritonExperts, that
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +48/-1 (49 lines); hunks: -25,6 +25,7; -1886,7 +1887,7 @@ def fused_experts_impl(; symbols: fused_experts_impl, TritonExperts, __init__, apply
- Key code excerpts:

```diff
diff -- vllm/lora/layers/fused_moe.py
@@ -1,6 +1,5 @@
-import functools
@@ -14,31 +13,17 @@
-from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
-from vllm.model_executor.layers.fused_moe.config import (
-    _get_config_dtype_str,
-)
diff -- vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
@@ -17,6 +17,7 @@
+from vllm.model_executor.layers.fused_moe.lora_experts_mixin import LoRAExpertsMixin
@@ -655,7 +656,7 @@ def moe_problem_size(
-class MarlinExperts(MarlinExpertsBase):
+class MarlinExperts(LoRAExpertsMixin, MarlinExpertsBase):
@@ -720,7 +721,108 @@ def apply(
-        fused_marlin_moe(
diff -- vllm/model_executor/layers/fused_moe/lora_experts_mixin.py
@@ -0,0 +1,111 @@
```

- Reviewed files:
  - runtime: `vllm/lora/layers/fused_moe.py` modified +36/-297; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +106/-6; `vllm/model_executor/layers/fused_moe/lora_experts_mixin.py` added +111/-0; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +52/-3; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +48/-1; `vllm/model_executor/layers/fused_moe/lora_context.py` added +44/-0
- Risk and verification: Runtime changes concentrate in `vllm/lora/layers/fused_moe.py`, `vllm/lora/layers/utils.py`, `vllm/lora/ops/triton_ops/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40860 - [Feat] DeepSeek V4 Rebased

- Link: https://github.com/vllm-project/vllm/pull/40860
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 150 files, +16313/-717, 20516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] DeepSeek V4 Rebased"; model line: GPT-OSS; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; technical summary: Covers "[Feat] DeepSeek V4 Rebased"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0 (1076 lines); hunks: -0,0 +1,1076; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does, touching `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0 (1076 lines); hunks: -0,0 +1,1076; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/mhc.py` added +450/-0 (450 lines); hunks: -0,0 +1,450; symbols: compute_num_split, mhc_pre_big_fuse_tilelang, mhc_pre, _mhc_pre_fake
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1437 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -0,0 +1,1076 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+DeepseekV4 MLA Attention Layer
+"""
+from dataclasses import dataclass
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0; `vllm/model_executor/layers/mhc.py` added +450/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +438/-0
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39136 - [ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle

- Link: https://github.com/vllm-project/vllm/pull/39136
- Status/date: merged / 2026-05-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +875/-347, 1610 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: aiter_triton_kernel_w4a8_moe_forward, triton_kernel_fused_mxfp4_w4a8_experts, AiterW4A8ExpertsMonolithic, __init__, touching `aiter_triton_kernel_w4a8_moe_forward, triton_kernel_fused_mxfp4_w4a8_experts, AiterW4A8ExpertsMonolithic`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +66/-200 (266 lines); hunks: -35,19 +35,19; -62,7 +62,6; symbols: get_moe_method, __init__, touching `get_moe_method, __init__`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +126/-13 (139 lines); hunks: -19,16 +19,19; -59,8 +62,9 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss, touching `Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend`; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +0/-123 (123 lines); hunks: -5,7 +5,6; -286,35 +285,6 @@ def triton_kernel_moe_forward(; symbols: triton_kernel_moe_forward, triton_kernel_fused_experts, triton_kernel_fused_mxfp4_w4a8_experts, make_routing_data, touching `triton_kernel_moe_forward, triton_kernel_fused_experts, triton_kernel_fused_mxfp4_w4a8_experts`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: aiter_triton_kernel_w4a8_moe_forward, triton_kernel_fused_mxfp4_w4a8_experts, AiterW4A8ExpertsMonolithic, __init__
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +66/-200 (266 lines); hunks: -35,19 +35,19; -62,7 +62,6; symbols: get_moe_method, __init__
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +126/-13 (139 lines); hunks: -19,16 +19,19; -59,8 +62,9 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +0/-123 (123 lines); hunks: -5,7 +5,6; -286,35 +285,6 @@ def triton_kernel_moe_forward(; symbols: triton_kernel_moe_forward, triton_kernel_fused_experts, triton_kernel_fused_mxfp4_w4a8_experts, make_routing_data
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +3/-3 (6 lines); hunks: -24,7 +24,7; -140,7 +140,7 @@ class GptOssMxfp4MoEMethod(FusedMoEMethodBase):; symbols: GptOssMxfp4MoEMethod, __init__, Mxfp4MoEMethod
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py
@@ -0,0 +1,292 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+import vllm.model_executor.layers.fused_moe.modular_kernel as mk
+from vllm._aiter_ops import rocm_aiter_ops
+from vllm.model_executor.layers.fused_moe.activation import MoEActivation
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -35,19 +35,19 @@
-    select_gpt_oss_mxfp4_moe_backend,
+    select_mxfp4_moe_backend,
-from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
-    _swizzle_mxfp4,
-)
-from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -19,16 +19,19 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py` added +292/-0; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +66/-200; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +126/-13; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +0/-123; `vllm/model_executor/layers/quantization/mxfp4.py` modified +3/-3
  - tests: `tests/kernels/moe/test_ocp_mx_moe.py` modified +388/-8
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_ocp_mx_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42334 - [MoE Refactor] Move remaining experts classes to experts directory

- Link: https://github.com/vllm-project/vllm/pull/42334
- Status/date: merged / 2026-05-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +48/-42, 315 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move remaining experts classes to experts directory"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/fp8.py`; technical summary: Covers "[MoE Refactor] Move remaining experts classes to experts directory"; the main implementation surface is `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/__init__.py` modified +6/-6 (12 lines); hunks: -85,9 +85,15 @@ def get_config() -> dict[str, Any] | None:; -97,9 +103,6 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config, touching `get_config`; `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +3/-3 (6 lines); hunks: -12,15 +12,15; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +3/-3 (6 lines); hunks: -130,7 +130,7 @@ def backend_to_kernel_cls(; -158,7 +158,7 @@ def backend_to_kernel_cls(; symbols: backend_to_kernel_cls, touching `backend_to_kernel_cls`; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +3/-1 (4 lines); hunks: -14,7 +14,9.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +6/-6 (12 lines); hunks: -85,9 +85,15 @@ def get_config() -> dict[str, Any] | None:; -97,9 +103,6 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config
  - `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +3/-3 (6 lines); hunks: -12,15 +12,15
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +3/-3 (6 lines); hunks: -130,7 +130,7 @@ def backend_to_kernel_cls(; -158,7 +158,7 @@ def backend_to_kernel_cls(; symbols: backend_to_kernel_cls
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +3/-1 (4 lines); hunks: -14,7 +14,9
  - `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` modified +3/-1 (4 lines); hunks: -17,7 +17,9
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/__init__.py
@@ -85,9 +85,15 @@ def get_config() -> dict[str, Any] | None:
+    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
+        BatchedTritonExperts,
+    )
+    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
+        TritonOrDeepGemmExperts,
+    )
diff -- vllm/model_executor/layers/fused_moe/experts/triton_moe.py
@@ -12,15 +12,15 @@
+from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
+    LoRAExpertsMixin,
+)
-from vllm.model_executor.layers.fused_moe.lora_experts_mixin import (
-    LoRAExpertsMixin,
-)
diff -- vllm/model_executor/layers/fused_moe/oracle/fp8.py
@@ -130,7 +130,7 @@ def backend_to_kernel_cls(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/__init__.py` modified +6/-6; `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +3/-1; `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` modified +3/-1; `vllm/model_executor/layers/quantization/humming.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41566 - [Quantization] Rework quantization_config to use QuantKey and allow for activation override

- Link: https://github.com/vllm-project/vllm/pull/41566
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +626/-327, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Rework quantization_config to use QuantKey and allow for activation override"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/online/base.py`, `vllm/model_executor/model_loader/weight_utils.py`; technical summary: Covers "[Quantization] Rework quantization_config to use QuantKey and allow for activation override"; the main implementation surface is `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/online/base.py`, `vllm/model_executor/model_loader/weight_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +173/-122 (295 lines); hunks: -1,13 +1,15; -207,42 +209,55 @@ def backend_to_kernel_cls(; symbols: backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss, _backend_activation_key, touching `backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss`; `vllm/model_executor/layers/quantization/online/base.py` modified +65/-38 (103 lines); hunks: -5,10 +5,7; -41,31 +38,50; symbols: handles, OnlineQuantizationConfig, for, __init__, touching `handles, OnlineQuantizationConfig, for`; `vllm/model_executor/model_loader/weight_utils.py` modified +7/-23 (30 lines); hunks: -300,12 +300,10 @@ def get_quant_config(; -330,12 +328,6 @@ def get_quant_config(; symbols: get_quant_config, touching `get_quant_config`; `vllm/model_executor/layers/quantization/__init__.py` modified +11/-13 (24 lines); hunks: -35,10 +35,9; -111,7 +110,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, touching `get_quantization_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +173/-122 (295 lines); hunks: -1,13 +1,15; -207,42 +209,55 @@ def backend_to_kernel_cls(; symbols: backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss, _backend_activation_key
  - `vllm/model_executor/layers/quantization/online/base.py` modified +65/-38 (103 lines); hunks: -5,10 +5,7; -41,31 +38,50; symbols: handles, OnlineQuantizationConfig, for, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +7/-23 (30 lines); hunks: -300,12 +300,10 @@ def get_quant_config(; -330,12 +328,6 @@ def get_quant_config(; symbols: get_quant_config
  - `vllm/model_executor/layers/quantization/__init__.py` modified +11/-13 (24 lines); hunks: -35,10 +35,9; -111,7 +110,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config
  - `vllm/entrypoints/llm.py` modified +2/-4 (6 lines); hunks: -35,7 +35,7; -248,9 +248,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -1,13 +1,15 @@
-from typing import TYPE_CHECKING, Union
+from typing import TYPE_CHECKING, Literal, Union
+from vllm.config import get_current_vllm_config
+from vllm.config.quantization import QuantizationConfigArgs
@@ -207,42 +209,55 @@ def backend_to_kernel_cls(
-def map_mxfp4_backend(runner_backend: MoEBackend) -> Mxfp4MoeBackend:
diff -- vllm/model_executor/layers/quantization/online/base.py
@@ -5,10 +5,7 @@
-from vllm.config.quantization import (
-    OnlineQuantizationConfigArgs,
-    OnlineQuantScheme,
-)
+from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
@@ -41,31 +38,50 @@
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -300,12 +300,10 @@ def get_quant_config(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +173/-122; `vllm/model_executor/layers/quantization/online/base.py` modified +65/-38; `vllm/model_executor/model_loader/weight_utils.py` modified +7/-23; `vllm/model_executor/layers/quantization/__init__.py` modified +11/-13; `vllm/entrypoints/llm.py` modified +2/-4; `vllm/config/quantization.py` modified +143/-94
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-mxfp8-cutlass.yaml` modified +1/-3; `tests/evals/gpt_oss/configs/gpt-oss-20b-sm100-fi-mxfp4-mxfp8-trtllm.yaml` modified +1/-3
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/compile/fusions_e2e/models.py`, `tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-mxfp8-cutlass.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-sm100-fi-mxfp4-mxfp8-trtllm.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37826 - [ROCm] Widen OAI Triton MoE capability range to include gfx12 (RDNA4)

- Link: https://github.com/vllm-project/vllm/pull/37826
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-22, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Widen OAI Triton MoE capability range to include gfx12 (RDNA4)"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; technical summary: Covers "[ROCm] Widen OAI Triton MoE capability range to include gfx12 (RDNA4)"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +26/-22 (48 lines); hunks: -34,6 +34,30; -524,17 +548,7 @@ def expects_unquantized_inputs(self) -> bool:; symbols: _triton_kernel_moe_supports_current_device, _patch_make_bitmatrix_metadata, expects_unquantized_inputs, _supports_current_device, touching `_triton_kernel_moe_supports_current_device, _patch_make_bitmatrix_metadata, expects_unquantized_inputs`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +26/-22 (48 lines); hunks: -34,6 +34,30; -524,17 +548,7 @@ def expects_unquantized_inputs(self) -> bool:; symbols: _triton_kernel_moe_supports_current_device, _patch_make_bitmatrix_metadata, expects_unquantized_inputs, _supports_current_device
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
@@ -34,6 +34,30 @@
+def _triton_kernel_moe_supports_current_device() -> bool:
+    # Shared device gate for the OAI Triton MoE expert classes.
+    # Platform-aware to avoid ROCm capability aliasing — cap (9, 0)
+    # matches both gfx90a (verified) and gfx906 (unverified), so we
+    # dispatch on gfx-string helpers instead of the cap tuple on ROCm.
+    p = current_platform
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +26/-22
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43135 - [Perf][gpt-oss] Downgrade triton_kernels to v3.5.1

- Link: https://github.com/vllm-project/vllm/pull/43135
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; associated commits `5774aad9c5b6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +73/-57, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][gpt-oss] Downgrade triton_kernels to v3.5.1"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; technical summary: Covers "[Perf][gpt-oss] Downgrade triton_kernels to v3.5.1"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +57/-32 (89 lines); hunks: -209,6 +209,16 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmat...; -233,11 +243,10 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitma...; symbols: _make_bitmatrix_metadata_pow2_safe, triton_kernel_moe_forward, touching `_make_bitmatrix_metadata_pow2_safe, triton_kernel_moe_forward`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +57/-32 (89 lines); hunks: -209,6 +209,16 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmat...; -233,11 +243,10 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitma...; symbols: _make_bitmatrix_metadata_pow2_safe, triton_kernel_moe_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
@@ -209,6 +209,16 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatrix):
+# Two API generations of triton_kernels are supported:
+#   - v3.5.1 (the version bundled with vLLM): exposes `routing()` and
+#     `routing_from_bitmatrix()` in triton_kernels.routing; the `Bitmatrix`
+#     constructor takes a `scratchpad` argument.
+#   - v3.6.0+: removes the `routing` module in favor of a `SparseMatrix`
+#     based path, and adds a `dtype=BIT` kwarg to `Bitmatrix`. Used only
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +57/-32
- Risk and verification: The diff ships test coverage in `tests/kernels/quantization/test_mxfp4_triton_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43385 - [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP

- Link: https://github.com/vllm-project/vllm/pull/43385
- Status/date: merged / 2026-05-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +2340/-52, 2496 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`; technical summary: Covers "[ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP"; the main implementation surface is `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/mtp.py`, `vllm/models/deepseek_v4/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel, touching `DeepseekV4MLP, __init__, forward`; `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`; `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel, touching `_build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices`; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +22/-11 (33 lines); hunks: -81,17 +81,28 @@ def _patch_make_bitmatrix_metadata() -> None:; symbols: _patch_make_bitmatrix_metadata, touching `_patch_make_bitmatrix_metadata`.
- Code diff details:
  - `vllm/models/deepseek_v4/amd/model.py` added +1612/-0 (1612 lines); hunks: -0,0 +1,1612; symbols: DeepseekV4MLP, __init__, forward, _deepseek_v4_stage_mega_moe_inputs_kernel
  - `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0 (520 lines); hunks: -0,0 +1,520; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23 (157 lines); hunks: -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> tor...; -704,38 +825,28 @@ def _forward_prefill(; symbols: _build_indptr_from_lengths, _combine_topk_swa_indices_kernel, combine_topk_swa_indices, _compute_topk_lens_kernel
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +22/-11 (33 lines); hunks: -81,17 +81,28 @@ def _patch_make_bitmatrix_metadata() -> None:; symbols: _patch_make_bitmatrix_metadata
  - `vllm/models/deepseek_v4/amd/model.py` removed +0/-1 (1 lines); hunks: -1 +0,0
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/amd/model.py
@@ -0,0 +1,1612 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4/amd/mtp.py
@@ -0,0 +1,520 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MTP draft model for DeepSeek V4 (internal codename: DeepseekV4).
+Split from ``deepseek_mtp.py`` because the V4 architecture introduces several
+pieces that have no analogue in V3/V32:
+  * separate ``e_proj`` / ``h_proj`` with fp8 linear quantization (instead of
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -44,6 +44,127 @@ def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/amd/model.py` added +1612/-0; `vllm/models/deepseek_v4/amd/mtp.py` added +520/-0; `vllm/models/deepseek_v4/amd/rocm.py` modified +134/-23; `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +22/-11; `vllm/models/deepseek_v4/amd/model.py` removed +0/-1; `vllm/models/deepseek_v4/amd/mtp.py` removed +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/deepseek_v4/amd/model.py`, `vllm/models/deepseek_v4/amd/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43108 - [MoE Refactor] Remove supports_expert_map

- Link: https://github.com/vllm-project/vllm/pull/43108
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +26/-148, 614 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove supports_expert_map"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`, `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`; technical summary: Covers "[MoE Refactor] Remove supports_expert_map"; the main implementation surface is `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`, `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` modified +12/-8 (20 lines); hunks: -17,7 +17,7 @@ def _quantize_and_setup_dispatch(; -33,7 +33,7 @@ def _quantize_and_setup_dispatch(; symbols: _quantize_and_setup_dispatch, _unwrap_scale_and_prepare_for_moe, prepare, touching `_quantize_and_setup_dispatch, _unwrap_scale_and_prepare_for_moe, prepare`; `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` modified +2/-16 (18 lines); hunks: -378,7 +378,8 @@ def apply(; -418,9 +419,6 @@ def _supports_parallel_config(moe_parallel_config: FusedMoEP...; symbols: apply, _supports_parallel_config, supports_expert_map, finalize_weight_and_reduce_impl, touching `apply, _supports_parallel_config, supports_expert_map`; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +0/-13 (13 lines); hunks: -751,13 +751,6 @@ def supports_lora() -> bool:; -1567,12 +1560,6 @@ def _post_init_setup(self):; symbols: supports_lora, supports_expert_map, supports, supports_packed_ue8m0_act_scales, touching `supports_lora, supports_expert_map, supports`; `vllm/model_executor/layers/fused_moe/experts/fallback.py` modified +0/-10 (10 lines); hunks: -92,16 +92,6 @@ def _supports_parallel_config(; symbols: _supports_parallel_config, supports_expert_map, finalize_weight_and_reduce_impl, touching `_supports_parallel_config, supports_expert_map, finalize_weight_and_reduce_impl`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` modified +12/-8 (20 lines); hunks: -17,7 +17,7 @@ def _quantize_and_setup_dispatch(; -33,7 +33,7 @@ def _quantize_and_setup_dispatch(; symbols: _quantize_and_setup_dispatch, _unwrap_scale_and_prepare_for_moe, prepare
  - `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` modified +2/-16 (18 lines); hunks: -378,7 +378,8 @@ def apply(; -418,9 +419,6 @@ def _supports_parallel_config(moe_parallel_config: FusedMoEP...; symbols: apply, _supports_parallel_config, supports_expert_map, finalize_weight_and_reduce_impl
  - `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +0/-13 (13 lines); hunks: -751,13 +751,6 @@ def supports_lora() -> bool:; -1567,12 +1560,6 @@ def _post_init_setup(self):; symbols: supports_lora, supports_expert_map, supports, supports_packed_ue8m0_act_scales
  - `vllm/model_executor/layers/fused_moe/experts/fallback.py` modified +0/-10 (10 lines); hunks: -92,16 +92,6 @@ def _supports_parallel_config(; symbols: _supports_parallel_config, supports_expert_map, finalize_weight_and_reduce_impl
  - `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py` modified +1/-6 (7 lines); hunks: -34,11 +34,6 @@ def __init__(; -103,7 +98,7 @@ def apply(; symbols: __init__, apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
@@ -17,7 +17,7 @@ def _quantize_and_setup_dispatch(
-) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
+) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
@@ -33,7 +33,7 @@ def _quantize_and_setup_dispatch(
-        a1q, a1q_scale = a1q, a1q_scale = moe_kernel_quantize_input(
+        a1q, a1q_scale = moe_kernel_quantize_input(
@@ -49,7 +49,7 @@ def _quantize_and_setup_dispatch(
diff -- vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py
@@ -378,7 +378,8 @@ def apply(
-            expert_map,
+            # the fp8 cutlass experts use their own expert map.
+            None,
@@ -418,9 +419,6 @@ def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bo
-    def supports_expert_map(self) -> bool:
-        return False
diff -- vllm/model_executor/layers/fused_moe/modular_kernel.py
@@ -751,13 +751,6 @@ def supports_lora() -> bool:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` modified +12/-8; `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` modified +2/-16; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +0/-13; `vllm/model_executor/layers/fused_moe/experts/fallback.py` modified +0/-10; `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py` modified +1/-6; `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` modified +0/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/test_modular_kernel_combinations.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43571 - [BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py

- Link: https://github.com/vllm-project/vllm/pull/43571
- Status/date: merged / 2026-05-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/quantization/test_gpt_oss.py`; associated commits `e9499996df89`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py"; model line: GPT-OSS; category: bug fix; main diff: `tests/models/quantization/test_gpt_oss.py`; technical summary: Covers "[BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py"; the main implementation surface is `tests/models/quantization/test_gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -22,7 +22,14; symbols: on_gfx950, touching `on_gfx950`.
- Code diff details:
  - `tests/models/quantization/test_gpt_oss.py` modified +8/-1 (9 lines); hunks: -22,7 +22,14; symbols: on_gfx950
- Key code excerpts:

```diff
diff -- tests/models/quantization/test_gpt_oss.py
@@ -22,7 +22,14 @@
-from vllm.platforms.rocm import on_gfx950
+if current_platform.is_rocm():
+    from vllm.platforms.rocm import on_gfx950
+else:
+    def on_gfx950() -> bool:
+        return False
```

- Reviewed files:
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- Link: https://github.com/vllm-project/vllm/pull/43167
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 56 files, +88/-731, 1251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove KV cache scale boilerplate from model weight loading methods"; model line: GPT-OSS; category: docs/tests/CI; main diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`; technical summary: Covers "Remove KV cache scale boilerplate from model weight loading methods"; the main implementation surface is `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name, touching `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`; `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader, touching `_get_moe_weight_dtype, kv_cache_scale_loader`; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod, touching `KVCacheScaleParameter, __new__, weight_loader`; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter, touching `get_quant_method, get_cache_scale, get_cache_scale_mapper`.
- Code diff details:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- Key code excerpts:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- Reviewed files:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_eagle_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- Link: https://github.com/vllm-project/vllm/pull/41184
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 90 files, +2734/-2027, 7329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`; technical summary: Covers "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts, touching `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method, touching `FusedMoeWeightScaleSupported, RoutedExperts, __init__`; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward, touching `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`; `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__, touching `FusedMoEWithLoRA, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1,1424 +1,404 @@
-from collections.abc import Callable, Iterable
-from enum import Enum
-from typing import Literal, cast, overload
+from collections.abc import Callable
+from typing import Any
-from torch.nn.parameter import UninitializedParameter
diff -- vllm/model_executor/layers/fused_moe/routed_experts.py
@@ -0,0 +1,1144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable, Iterable
+from enum import Enum
+from typing import TYPE_CHECKING, Any, Literal, cast, overload
+import torch
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@ -1,28 +1,39 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- Risk and verification: The diff ships test coverage in `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45067 - [Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor

- Link: https://github.com/vllm-project/vllm/pull/45067
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `7920ccb97c2d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +7/-0 (7 lines); hunks: -635,6 +635,13 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, touching `_get_moe_weight_dtype`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +7/-0 (7 lines); hunks: -635,6 +635,13 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,6 +635,13 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
+                # The MoE refactor (#41184) moved expert params under
+                # `mlp.experts.routed_experts.*`; remap the legacy checkpoint
+                # name so keys like w2_bias resolve against params_dict.
+                fused_name = fused_name.replace(
+                    ".mlp.experts.", ".mlp.experts.routed_experts."
+                )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44992 - Deprecations for v0.23 and v0.24

- Link: https://github.com/vllm-project/vllm/pull/44992
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +102/-676, 1334 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecations for v0.23 and v0.24"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`; technical summary: Covers "Deprecations for v0.23 and v0.24"; the main implementation surface is `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend, touching `select_mxfp4_moe_backend`; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise, touching `NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise`; `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel, touching `_get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel`; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise, touching `_return_or_raise`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend
  - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise
  - `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise
  - `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45 (45 lines); hunks: -19,9 +19,7; -230,49 +228,6 @@ def _return_or_raise(; symbols: _return_or_raise
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -6,7 +6,6 @@
-from vllm import envs
@@ -465,74 +464,6 @@ def select_mxfp4_moe_backend(
-    # Handle explicit FlashInfer MXFP4 BF16 configuration.
-    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"):
-        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
-            for _b in (
diff -- vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
@@ -22,10 +22,6 @@
-from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
-    FlashinferMoeBackend,
-    get_flashinfer_moe_backend,
-)
@@ -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):
-fi_2_vllm_backend_map: dict[FlashinferMoeBackend, NvFp4MoeBackend] = {
diff -- vllm/model_executor/kernels/linear/__init__.py
@@ -212,6 +212,9 @@ def _get_linear_backend() -> str:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59; `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45; `vllm/entrypoints/pooling/offline.py` modified +0/-44
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_async_tp.py`, `tests/conftest.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/entrypoints/pooling/reward/test_token_reward_offline.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45381 - [Model] Add MiniMax M3 support

- Link: https://github.com/vllm-project/vllm/pull/45381
- Status/date: merged / 2026-06-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 108 files, +14746/-323, 16807 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MiniMax M3 support"; model line: GPT-OSS; category: model support/runtime entry; main diff: `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py`; technical summary: Covers "[Model] Add MiniMax M3 support"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` added +1216/-0 (1216 lines); hunks: -0,0 +1,1216; symbols: _sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb, MiniMAXGemmaRMSNorm, touching `_sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb`; `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0 (1177 lines); hunks: -0,0 +1,1177; symbols: _sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm, __init__, touching `_sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm`; `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0 (898 lines); hunks: -0,0 +1,898; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel, touching `_compare_and_swap, _bitonic_merge, _index_block_score_kernel`; `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: MiniMaxVLPatchEmbed, __init__, forward, MiniMaxVLAttention, touching `MiniMaxVLPatchEmbed, __init__, forward`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` added +1216/-0 (1216 lines); hunks: -0,0 +1,1216; symbols: _sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb, MiniMAXGemmaRMSNorm
  - `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0 (1177 lines); hunks: -0,0 +1,1177; symbols: _sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm, __init__
  - `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0 (898 lines); hunks: -0,0 +1,898; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel
  - `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: MiniMaxVLPatchEmbed, __init__, forward, MiniMaxVLAttention
  - `vllm/transformers_utils/processors/minimax_m3.py` added +736/-0 (736 lines); hunks: -0,0 +1,736; symbols: round_by_factor, ceil_by_factor, floor_by_factor, _smart_resize_by_long_side
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -0,0 +1,1216 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only MiniMax M3 (text backbone) model — AMD ROCm implementation.
+Self-contained per-platform impl (mirrors ``deepseek_v4/amd``). It is identical
+to ``../nvidia/model.py`` except for RMS normalization: FlashInfer's Gemma
+RMSNorm kernels are CUDA-only, so ``MiniMAXGemmaRMSNorm`` here uses a native
diff -- vllm/models/minimax_m3/nvidia/model.py
@@ -0,0 +1,1177 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only MiniMax M3 (text backbone) model.
+The MiniMax-M3-preview config selects a single set of branches:
+    * qk_norm_type == "per_head"
+    * hidden_act == "swigluoai"
diff -- vllm/models/minimax_m3/common/ops/index_topk.py
@@ -0,0 +1,898 @@
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` added +1216/-0; `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0; `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0; `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0; `vllm/transformers_utils/processors/minimax_m3.py` added +736/-0; `vllm/models/minimax_m3/common/ops/sparse_attn.py` added +593/-0
- Risk and verification: The diff ships test coverage in `requirements/test/cuda.txt`, `requirements/test/rocm.txt`, `requirements/test/xpu.txt`, `tests/kernels/attention/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45896 - [feature] MiniMax-M3-MXFP4 support added

- Link: https://github.com/vllm-project/vllm/pull/45896
- Status/date: merged / 2026-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +44/-3, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] MiniMax-M3-MXFP4 support added"; model line: GPT-OSS; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/minimax_m3/amd/model.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[feature] MiniMax-M3-MXFP4 support added"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/models/minimax_m3/amd/model.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +14/-0 (14 lines); hunks: -755,6 +755,7 @@ def _supports_activation(activation: MoEActivation) -> bool:; -811,6 +812,19 @@ def activation(; symbols: _supports_activation, activation, touching `_supports_activation, activation`; `vllm/models/minimax_m3/amd/model.py` modified +12/-2 (14 lines); hunks: -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(; symbols: load_weights, MiniMaxM3SparseForCausalLM, __init__, MiniMaxM3SparseForConditionalGeneration, touching `load_weights, MiniMaxM3SparseForCausalLM, __init__`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +12/-1 (13 lines); hunks: -503,7 +503,18 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend, touching `select_mxfp4_moe_backend`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-0 (6 lines); hunks: -1303,6 +1303,9 @@ def get_fused_moe_quant_config(; -1339,6 +1342,9 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, touching `get_fused_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +14/-0 (14 lines); hunks: -755,6 +755,7 @@ def _supports_activation(activation: MoEActivation) -> bool:; -811,6 +812,19 @@ def activation(; symbols: _supports_activation, activation
  - `vllm/models/minimax_m3/amd/model.py` modified +12/-2 (14 lines); hunks: -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(; symbols: load_weights, MiniMaxM3SparseForCausalLM, __init__, MiniMaxM3SparseForConditionalGeneration
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +12/-1 (13 lines); hunks: -503,7 +503,18 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-0 (6 lines); hunks: -1303,6 +1303,9 @@ def get_fused_moe_quant_config(; -1339,6 +1342,9 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
@@ -755,6 +755,7 @@ def _supports_activation(activation: MoEActivation) -> bool:
+            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
@@ -811,6 +812,19 @@ def activation(
+        elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
+            assert quant_config.gemm1_clamp_limit is not None
+            alpha = (
+                quant_config.gemm1_alpha
diff -- vllm/models/minimax_m3/amd/model.py
@@ -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    packed_modules_mapping = {
+        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
@@ -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(
+    packed_modules_mapping = {
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -503,7 +503,18 @@ def select_mxfp4_moe_backend(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +14/-0; `vllm/models/minimax_m3/amd/model.py` modified +12/-2; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +12/-1; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46441 - fix gpt_oss pp>1 with ep

- Link: https://github.com/vllm-project/vllm/pull/46441
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `901a3b091cf1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix gpt_oss pp>1 with ep"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "fix gpt_oss pp>1 with ep"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1078,7 +1078,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1078,7 +1078,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -1078,7 +1078,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        ep_rank = get_ep_group().rank
+        ep_rank = get_ep_group().rank_in_group
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45818 - [Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r…

- Link: https://github.com/vllm-project/vllm/pull/45818
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gpt_oss.py`; associated commits `f4d5f73ffa40`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r…"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/models/gpt_oss.py`; technical summary: Covers "[Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r…"; the main implementation surface is `vllm/model_executor/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -979,7 +979,11 @@ def _load_weights_other(; symbols: _load_weights_other, touching `_load_weights_other`.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -979,7 +979,11 @@ def _load_weights_other(; symbols: _load_weights_other
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gpt_oss.py
@@ -979,7 +979,11 @@ def _load_weights_other(
-        for name, weight in weights:
+        # Use centralized weight remapping for MoE expert parameters.
+        # The FusedMoE refactor moved expert params under
+        # `mlp.experts.routed_experts.*`; this remaps checkpoint names so
+        # MoE weight/bias keys resolve against params_dict.
+        for name, weight in remap_moe_expert_weights(weights, params_dict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46142 - [AMD][OCP MX][CI] Fix tests to not dispatch on `UNFUSED_TRITON` backend on MI300, improve w_mxfp4_a_fp8 emulation support

- Link: https://github.com/vllm-project/vllm/pull/46142
- Status/date: merged / 2026-06-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +74/-17, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][OCP MX][CI] Fix tests to not dispatch on `UNFUSED_TRITON` backend on MI300, improve w_mxfp4_a_fp8 emulation support"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`; technical summary: Covers "[AMD][OCP MX][CI] Fix tests to not dispatch on `UNFUSED_TRITON` backend on MI300, improve w_mxfp4_a_fp8 emulation support"; the main implementation surface is `vllm/model_executor/layers/fused_moe/utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/utils.py` modified +16/-12 (28 lines); hunks: -200,6 +200,16 @@ def _mxfp4_quantize(; -268,23 +278,17 @@ def moe_kernel_quantize_input(; symbols: _mxfp4_quantize, _fp8_quantize_dequantize, _mxfp8_e4m3_quantize, moe_kernel_quantize_input, touching `_mxfp4_quantize, _fp8_quantize_dequantize, _mxfp8_e4m3_quantize`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +25/-2 (27 lines); hunks: -674,6 +674,8 @@ def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(; -1191,8 +1193,29 @@ def swap_every_two_rows(x, axis=-1):; symbols: convert_gpt_oss_weight_to_mxfp4_moe_kernel_format, swap_every_two_rows, touching `convert_gpt_oss_weight_to_mxfp4_moe_kernel_format, swap_every_two_rows`; `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +12/-1 (13 lines); hunks: -332,12 +332,23 @@ def apply(; symbols: apply, _base_w13_fn, touching `apply, _base_w13_fn`; `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -104,6 +104,11 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization, touching `test_gpt_oss_attention_quantization`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/utils.py` modified +16/-12 (28 lines); hunks: -200,6 +200,16 @@ def _mxfp4_quantize(; -268,23 +278,17 @@ def moe_kernel_quantize_input(; symbols: _mxfp4_quantize, _fp8_quantize_dequantize, _mxfp8_e4m3_quantize, moe_kernel_quantize_input
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +25/-2 (27 lines); hunks: -674,6 +674,8 @@ def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(; -1191,8 +1193,29 @@ def swap_every_two_rows(x, axis=-1):; symbols: convert_gpt_oss_weight_to_mxfp4_moe_kernel_format, swap_every_two_rows
  - `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +12/-1 (13 lines); hunks: -332,12 +332,23 @@ def apply(; symbols: apply, _base_w13_fn
  - `tests/models/quantization/test_gpt_oss.py` modified +5/-0 (5 lines); hunks: -104,6 +104,11 @@ def test_gpt_oss_attention_quantization(; symbols: test_gpt_oss_attention_quantization
  - `vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py` modified +2/-2 (4 lines); hunks: -26,6 +26,7; -83,8 +84,7 @@ def __init__(; symbols: __init__, quant_dtype
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/utils.py
@@ -200,6 +200,16 @@ def _mxfp4_quantize(
+def _fp8_quantize_dequantize(
+    A: torch.Tensor,
+    A_scale: torch.Tensor,
+):
+    qA, qA_scale = ops.scaled_fp8_quant(A, A_scale, use_per_token_if_dynamic=False)
+    A = per_tensor_dequantize(qA, qA_scale).to(A.dtype)
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -674,6 +674,8 @@ def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
+    w13_input_scale: torch.Tensor | None = None,
+    w2_input_scale: torch.Tensor | None = None,
@@ -1191,8 +1193,29 @@ def swap_every_two_rows(x, axis=-1):
-        # No additional transformation needed for emulation backend,
-        # weights are dequantized on the fly in the experts class.
+        w13_has_per_expert_scale = (
diff -- vllm/model_executor/layers/fused_moe/experts/triton_moe.py
@@ -332,12 +332,23 @@ def apply(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/utils.py` modified +16/-12; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +25/-2; `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +12/-1; `vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py` modified +2/-2; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-0
  - tests: `tests/models/quantization/test_gpt_oss.py` modified +5/-0; `tests/quantization/test_quark.py` modified +12/-0
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_gpt_oss.py`, `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #46406 - [Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing

- Link: https://github.com/vllm-project/vllm/pull/46406
- Status/date: merged / 2026-06-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +238/-0, 252 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; technical summary: Covers "[Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +238/-0 (238 lines); hunks: -220,6 +220,241 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitma...; -260,6 +495,9 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatr...; symbols: _make_bitmatrix_metadata_pow2_safe, _patch_legacy_routing_for_nonpow2_topk, _routing_compute_indx_pow2, _combined_routing_compute_pow2, touching `_make_bitmatrix_metadata_pow2_safe, _patch_legacy_routing_for_nonpow2_topk, _routing_compute_indx_pow2`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +238/-0 (238 lines); hunks: -220,6 +220,241 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitma...; -260,6 +495,9 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatr...; symbols: _make_bitmatrix_metadata_pow2_safe, _patch_legacy_routing_for_nonpow2_topk, _routing_compute_indx_pow2, _combined_routing_compute_pow2
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
@@ -220,6 +220,241 @@ def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatrix):
+def _patch_legacy_routing_for_nonpow2_topk() -> None:
+    """Monkey-patch the legacy (v3.5.1) triton_kernels routing path to support
+    non-power-of-2 top_k (e.g. DeepSeek-V4 top_k=6).
+    The bundled ``_routing_compute_indx`` does ``tl.arange(0, N_EXPTS_ACT *
+    BLOCK_M)``, which fails to compile when ``N_EXPTS_ACT`` (top_k) is not a
+    power of 2 (6 * 32 = 192). This installs a pow2-safe variant that pads the
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +238/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46408 - [Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE

- Link: https://github.com/vllm-project/vllm/pull/46408
- Status/date: merged / 2026-06-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +88/-7, 137 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE"; model line: GPT-OSS; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; technical summary: Covers "[Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +88/-7 (95 lines); hunks: -4,7 +4,6; -577,6 +576,85 @@ def make_routing_data(; symbols: make_routing_data, _masked_topk_sum_kernel, masked_moe_sum, _remap_topk_to_local_kernel, touching `make_routing_data, _masked_topk_sum_kernel, masked_moe_sum`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +88/-7 (95 lines); hunks: -4,7 +4,6; -577,6 +576,85 @@ def make_routing_data(; symbols: make_routing_data, _masked_topk_sum_kernel, masked_moe_sum, _remap_topk_to_local_kernel
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
@@ -4,7 +4,6 @@
-from vllm import _custom_ops as ops
@@ -577,6 +576,85 @@ def make_routing_data(
+@triton.jit
+def _masked_topk_sum_kernel(
+    inp_ptr,  # (M, topk, K) contiguous
+    topk_ids_ptr,  # (M, topk) int: -1 marks an invalid / non-local slot
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py` modified +88/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
