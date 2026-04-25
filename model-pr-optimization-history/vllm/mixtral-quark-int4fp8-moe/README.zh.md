# vllm Mixtral Quark INT4/FP8 MoE 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/features/quantization/quark.md` | [#17888](https://github.com/vllm-project/vllm/pull/17888), [#24239](https://github.com/vllm-project/vllm/pull/24239) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` | [#38504](https://github.com/vllm-project/vllm/pull/38504), [#38774](https://github.com/vllm-project/vllm/pull/38774) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` | [#38504](https://github.com/vllm-project/vllm/pull/38504), [#38774](https://github.com/vllm-project/vllm/pull/38774) |
| `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` | [#38504](https://github.com/vllm-project/vllm/pull/38504), [#38774](https://github.com/vllm-project/vllm/pull/38774) |
| `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` | [#31827](https://github.com/vllm-project/vllm/pull/31827), [#33715](https://github.com/vllm-project/vllm/pull/33715), [#33807](https://github.com/vllm-project/vllm/pull/33807) |
| `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` | [#31827](https://github.com/vllm-project/vllm/pull/31827) |
| `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` | [#31759](https://github.com/vllm-project/vllm/pull/31759), [#33807](https://github.com/vllm-project/vllm/pull/33807) |
| `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` | [#31759](https://github.com/vllm-project/vllm/pull/31759) |
| `tests/lora/test_mixtral.py` | [#2831](https://github.com/vllm-project/vllm/pull/2831), [#9008](https://github.com/vllm-project/vllm/pull/9008), [#11390](https://github.com/vllm-project/vllm/pull/11390), [#28322](https://github.com/vllm-project/vllm/pull/28322) |
| `tests/quantization/test_quark.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#12612](https://github.com/vllm-project/vllm/pull/12612), [#16236](https://github.com/vllm-project/vllm/pull/16236), [#17888](https://github.com/vllm-project/vllm/pull/17888), [#24649](https://github.com/vllm-project/vllm/pull/24649), [#26364](https://github.com/vllm-project/vllm/pull/26364), [#35658](https://github.com/vllm-project/vllm/pull/35658), [#36320](https://github.com/vllm-project/vllm/pull/36320) |
| `tests/quantization/test_quark_maybe_update_config.py` | 无直接 PR 号提交 |
| `vllm/model_executor/layers/quantization/quark/__init__.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/layers/quantization/quark/quark.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#15734](https://github.com/vllm-project/vllm/pull/15734), [#16943](https://github.com/vllm-project/vllm/pull/16943), [#17215](https://github.com/vllm-project/vllm/pull/17215), [#17888](https://github.com/vllm-project/vllm/pull/17888), [#20251](https://github.com/vllm-project/vllm/pull/20251), [#24239](https://github.com/vllm-project/vllm/pull/24239), [#28275](https://github.com/vllm-project/vllm/pull/28275), [#28638](https://github.com/vllm-project/vllm/pull/28638), [#29008](https://github.com/vllm-project/vllm/pull/29008), [#30071](https://github.com/vllm-project/vllm/pull/30071), [#32779](https://github.com/vllm-project/vllm/pull/32779), ... (14 total) |
| `vllm/model_executor/layers/quantization/quark/quark_moe.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#11528](https://github.com/vllm-project/vllm/pull/11528), [#13784](https://github.com/vllm-project/vllm/pull/13784), [#14245](https://github.com/vllm-project/vllm/pull/14245), [#17888](https://github.com/vllm-project/vllm/pull/17888), [#22035](https://github.com/vllm-project/vllm/pull/22035), [#22537](https://github.com/vllm-project/vllm/pull/22537), [#23123](https://github.com/vllm-project/vllm/pull/23123), [#24649](https://github.com/vllm-project/vllm/pull/24649), [#26545](https://github.com/vllm-project/vllm/pull/26545), [#26739](https://github.com/vllm-project/vllm/pull/26739), [#27029](https://github.com/vllm-project/vllm/pull/27029), ... (44 total) |
| `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#16943](https://github.com/vllm-project/vllm/pull/16943), [#35316](https://github.com/vllm-project/vllm/pull/35316) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` | [#36100](https://github.com/vllm-project/vllm/pull/36100), [#36232](https://github.com/vllm-project/vllm/pull/36232) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_scheme.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` | [#35316](https://github.com/vllm-project/vllm/pull/35316) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#12612](https://github.com/vllm-project/vllm/pull/12612), [#14245](https://github.com/vllm-project/vllm/pull/14245), [#14390](https://github.com/vllm-project/vllm/pull/14390), [#14578](https://github.com/vllm-project/vllm/pull/14578), [#16236](https://github.com/vllm-project/vllm/pull/16236), [#19830](https://github.com/vllm-project/vllm/pull/19830), [#20251](https://github.com/vllm-project/vllm/pull/20251), [#27814](https://github.com/vllm-project/vllm/pull/27814), [#33892](https://github.com/vllm-project/vllm/pull/33892) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#16236](https://github.com/vllm-project/vllm/pull/16236), [#27814](https://github.com/vllm-project/vllm/pull/27814) |
| `vllm/model_executor/layers/quantization/quark/utils.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/models/mixtral.py` | [#2011](https://github.com/vllm-project/vllm/pull/2011), [#2015](https://github.com/vllm-project/vllm/pull/2015), [#2036](https://github.com/vllm-project/vllm/pull/2036), [#2090](https://github.com/vllm-project/vllm/pull/2090), [#2208](https://github.com/vllm-project/vllm/pull/2208), [#2542](https://github.com/vllm-project/vllm/pull/2542), [#2677](https://github.com/vllm-project/vllm/pull/2677), [#2769](https://github.com/vllm-project/vllm/pull/2769), [#2831](https://github.com/vllm-project/vllm/pull/2831), [#2875](https://github.com/vllm-project/vllm/pull/2875), [#2880](https://github.com/vllm-project/vllm/pull/2880), [#3597](https://github.com/vllm-project/vllm/pull/3597), ... (40 total) |

## PR 覆盖总览

- git 追溯 PR 数: 108
- 原文档显式引用补充 PR 数: 11
- 当前文档总 PR 数: 119
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2023-03-02 | [#4](https://github.com/vllm-project/vllm/pull/4) | merged | Use FlashAttention for `multi_query_kv_attention` | `cacheflow/models/attention.py`, `tests/kernels/attention.py`, `server.py` |
| 2023-12-11 | [#2011](https://github.com/vllm-project/vllm/pull/2011) | merged | Mixtral 8x7B support | `vllm/model_executor/models/mixtral.py` |
| 2023-12-11 | [#2015](https://github.com/vllm-project/vllm/pull/2015) | merged | Minor fixes for Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2023-12-11 | [#2036](https://github.com/vllm-project/vllm/pull/2036) | merged | [Minor] Fix type annotation in Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2023-12-14 | [#2090](https://github.com/vllm-project/vllm/pull/2090) | merged | Mixtral expert parallelism | `vllm/model_executor/models/mixtral.py` |
| 2023-12-20 | [#2208](https://github.com/vllm-project/vllm/pull/2208) | merged | [BugFix] Fix weight loading for Mixtral with TP | `vllm/model_executor/models/mixtral.py` |
| 2024-01-30 | [#2293](https://github.com/vllm-project/vllm/pull/2293) | closed | tensor parallel MOE implementation | `vllm/model_executor/layers/moe.py`, `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe_grouped_matmul.py` |
| 2024-01-30 | [#2453](https://github.com/vllm-project/vllm/pull/2453) | merged | DeepseekMoE support with Fused MoE kernel | `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/layers/fused_moe.py`, `vllm/model_executor/models/__init__.py` |
| 2024-01-30 | [#2542](https://github.com/vllm-project/vllm/pull/2542) | merged | Fused MOE for Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2024-01-31 | [#2677](https://github.com/vllm-project/vllm/pull/2677) | merged | Add unit test for Mixtral MoE layer | `vllm/model_executor/models/mixtral.py` |
| 2024-02-06 | [#2769](https://github.com/vllm-project/vllm/pull/2769) | merged | Add fused top-K softmax kernel for MoE | `vllm/model_executor/models/mixtral.py` |
| 2024-02-13 | [#2831](https://github.com/vllm-project/vllm/pull/2831) | merged | Add LoRA support for Mixtral | `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py` |
| 2024-02-15 | [#2880](https://github.com/vllm-project/vllm/pull/2880) | merged | Align LoRA code between Mistral and Mixtral (fixes #2875) | `vllm/model_executor/models/mixtral.py` |
| 2024-02-15 | [#2875](https://github.com/vllm-project/vllm/pull/2875) | closed | Fix AttributeError: MixtralModel object has no attribute org_vocab_size. | `vllm/model_executor/models/mixtral.py` |
| 2024-03-24 | [#3597](https://github.com/vllm-project/vllm/pull/3597) | merged | [BugFix] 1D query fix for MoE models | `vllm/model_executor/models/mixtral.py` |
| 2024-04-24 | [#4244](https://github.com/vllm-project/vllm/pull/4244) | merged | [Kernel] FP8 support for MoE kernel / Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2024-04-27 | [#4343](https://github.com/vllm-project/vllm/pull/4343) | merged | [Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales | `vllm/model_executor/models/mixtral.py` |
| 2024-04-30 | [#4332](https://github.com/vllm-project/vllm/pull/4332) | merged | [Kernel] Support Fp8 Checkpoints (Dynamic + Static) | `vllm/model_executor/layers/quantization/fp8.py`, `tests/models/test_fp8.py`, `vllm/model_executor/layers/linear.py` |
| 2024-05-01 | [#4436](https://github.com/vllm-project/vllm/pull/4436) | closed | [Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static) | `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/model_loader/weight_utils.py` |
| 2024-05-01 | [#4543](https://github.com/vllm-project/vllm/pull/4543) | merged | [Misc] Remove Mixtral device="cuda" declarations | `vllm/model_executor/models/mixtral.py` |
| 2024-05-04 | [#4527](https://github.com/vllm-project/vllm/pull/4527) | merged | [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations) | `vllm/model_executor/models/mixtral.py` |
| 2024-05-13 | [#4793](https://github.com/vllm-project/vllm/pull/4793) | merged | [Bugfix] Fix dynamic FP8 quantization for Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2024-05-22 | [#4893](https://github.com/vllm-project/vllm/pull/4893) | merged | [Misc] Load FP8 kv-cache scaling factors from checkpoints | `vllm/model_executor/models/mixtral.py` |
| 2024-05-31 | [#5039](https://github.com/vllm-project/vllm/pull/5039) | merged | [Model] Enable FP8 QKV in MoE and refine kernel tuning script | `vllm/model_executor/models/mixtral.py` |
| 2024-06-05 | [#5231](https://github.com/vllm-project/vllm/pull/5231) | merged | [Model] Correct Mixtral FP8 checkpoint loading | `vllm/model_executor/models/mixtral.py` |
| 2024-06-08 | [#5353](https://github.com/vllm-project/vllm/pull/5353) | merged | [Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale | `vllm/model_executor/models/mixtral.py` |
| 2024-07-02 | [#5970](https://github.com/vllm-project/vllm/pull/5970) | merged | [ Misc ] Refactor MoE to isolate Fp8 From Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2024-07-10 | [#6287](https://github.com/vllm-project/vllm/pull/6287) | merged | [Bugfix] Support 2D input shape in MoE layer | `vllm/model_executor/models/mixtral.py` |
| 2024-07-14 | [#6417](https://github.com/vllm-project/vllm/pull/6417) | merged | [ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8 | `vllm/model_executor/models/mixtral.py` |
| 2024-07-18 | [#6516](https://github.com/vllm-project/vllm/pull/6516) | merged | [Model] Pipeline parallel support for Mixtral | `vllm/model_executor/models/mixtral.py` |
| 2024-08-13 | [#7334](https://github.com/vllm-project/vllm/pull/7334) | merged | [Misc] Update Fused MoE weight loading | `vllm/model_executor/models/mixtral.py` |
| 2024-08-21 | [#7527](https://github.com/vllm-project/vllm/pull/7527) | merged | [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel | `vllm/model_executor/models/mixtral.py` |
| 2024-08-22 | [#7764](https://github.com/vllm-project/vllm/pull/7764) | merged | Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)" | `vllm/model_executor/models/mixtral.py` |
| 2024-08-27 | [#7766](https://github.com/vllm-project/vllm/pull/7766) | merged | [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel | `vllm/model_executor/models/mixtral.py` |
| 2024-09-10 | [#8217](https://github.com/vllm-project/vllm/pull/8217) | merged | [Misc] Fused MoE Marlin support for GPTQ | `vllm/model_executor/models/mixtral.py` |
| 2024-10-04 | [#9008](https://github.com/vllm-project/vllm/pull/9008) | merged | [Model] add a bunch of supported lora modules for mixtral | `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py` |
| 2024-10-28 | [#9632](https://github.com/vllm-project/vllm/pull/9632) | merged | [torch.compile] support moe models | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py` |
| 2024-10-28 | [#9758](https://github.com/vllm-project/vllm/pull/9758) | merged | [torch.compile] Adding "torch compile" annotations to some models | `vllm/model_executor/models/mixtral.py` |
| 2024-12-22 | [#11390](https://github.com/vllm-project/vllm/pull/11390) | merged | [Bugfix] Fix fully sharded LoRAs with Mixtral | `tests/lora/test_mixtral.py` |
| 2025-01-15 | [#10765](https://github.com/vllm-project/vllm/pull/10765) | merged | [Misc][Quark] Upstream Quark format to VLLM | `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` |
| 2025-01-23 | [#11528](https://github.com/vllm-project/vllm/pull/11528) | merged | [BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-02-17 | [#3208](https://github.com/vllm-project/vllm/pull/3208) | closed | [RFC/WIP] First steps towards FP8 for Mixtral | `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| 2025-02-22 | [#13591](https://github.com/vllm-project/vllm/pull/13591) | merged | [core] set up data parallel communication | `vllm/distributed/utils.py`, `examples/offline_inference/data_parallel.py`, `vllm/distributed/parallel_state.py` |
| 2025-02-25 | [#13784](https://github.com/vllm-project/vllm/pull/13784) | merged | [Bugfix][Quantization] Fix FP8 + EP | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-03-05 | [#13931](https://github.com/vllm-project/vllm/pull/13931) | merged | [V1] EP/TP MoE + DP Attention | `vllm/model_executor/models/mixtral.py` |
| 2025-03-07 | [#14390](https://github.com/vllm-project/vllm/pull/14390) | merged | [FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` |
| 2025-03-11 | [#14245](https://github.com/vllm-project/vllm/pull/14245) | merged | dynamic distpatch of fp8 kernels | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-03-17 | [#14961](https://github.com/vllm-project/vllm/pull/14961) | merged | [Bugfix][Model] Mixtral: use unused head_dim config argument | `vllm/model_executor/models/mixtral.py` |
| 2025-03-28 | [#14578](https://github.com/vllm-project/vllm/pull/14578) | merged | [Quantization][FP8] Adding support for fp8 gemm layer input in fp8 | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` |
| 2025-04-09 | [#15961](https://github.com/vllm-project/vllm/pull/15961) | merged | Add support to modelopt quantization of Mixtral model | `vllm/model_executor/models/mixtral_quant.py` |
| 2025-04-10 | [#16325](https://github.com/vllm-project/vllm/pull/16325) | merged | [Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral | `vllm/model_executor/models/mixtral.py` |
| 2025-04-11 | [#16236](https://github.com/vllm-project/vllm/pull/16236) | merged | [Bugfix] Fix bugs of running Quark quantized models | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py` |
| 2025-04-25 | [#15734](https://github.com/vllm-project/vllm/pull/15734) | merged | [Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization | `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-04-26 | [#17215](https://github.com/vllm-project/vllm/pull/17215) | merged | [AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary | `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-05-07 | [#16943](https://github.com/vllm-project/vllm/pull/16943) | merged | [Quantization] Quark MXFP4 format loading | `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` |
| 2025-05-08 | [#12612](https://github.com/vllm-project/vllm/pull/12612) | merged | [Bugfix] Fix quark fp8 format loading on AMD GPUs | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py` |
| 2025-06-30 | [#20251](https://github.com/vllm-project/vllm/pull/20251) | merged | [Bugfix] fix quark ptpc | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-07-09 | [#17888](https://github.com/vllm-project/vllm/pull/17888) | merged | [Feature][Quantization] MXFP4 support for MOE models | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py` |
| 2025-07-11 | [#19830](https://github.com/vllm-project/vllm/pull/19830) | merged | [Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` |
| 2025-07-14 | [#20893](https://github.com/vllm-project/vllm/pull/20893) | merged | [Quantization] add BNB for MixtralForCausalLM | `vllm/model_executor/models/mixtral.py` |
| 2025-08-15 | [#22035](https://github.com/vllm-project/vllm/pull/22035) | merged | [Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods. | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-08-30 | [#23123](https://github.com/vllm-project/vllm/pull/23123) | merged | Add routed_scaling_factor to MoE grouped topk | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-09-17 | [#24649](https://github.com/vllm-project/vllm/pull/24649) | merged | [Rocm] [quantization] Fix quark ptpc moe and add test case | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/quantization/test_quark.py` |
| 2025-09-17 | [#22842](https://github.com/vllm-project/vllm/pull/22842) | merged | [EPLB] Support EPLB for Mixtral Model | `vllm/model_executor/models/mixtral.py` |
| 2025-09-17 | [#22537](https://github.com/vllm-project/vllm/pull/22537) | merged | [Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-10-07 | [#21166](https://github.com/vllm-project/vllm/pull/21166) | merged | [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4 | `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| 2025-10-07 | [#26364](https://github.com/vllm-project/vllm/pull/26364) | merged | [ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py` | `tests/quantization/test_quark.py` |
| 2025-10-14 | [#26739](https://github.com/vllm-project/vllm/pull/26739) | merged | [torch.compile] Unwrap fused_marlin_moe custom op | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-10-17 | [#26545](https://github.com/vllm-project/vllm/pull/26545) | merged | [ROCM] MoE fp4 CK kernel | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-10-17 | [#27029](https://github.com/vllm-project/vllm/pull/27029) | merged | [Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-11-04 | [#27123](https://github.com/vllm-project/vllm/pull/27123) | merged | [Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses. | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-11-10 | [#28322](https://github.com/vllm-project/vllm/pull/28322) | merged | [CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness | `tests/lora/test_mixtral.py` |
| 2025-11-10 | [#27474](https://github.com/vllm-project/vllm/pull/27474) | merged | [Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-11-11 | [#24239](https://github.com/vllm-project/vllm/pull/24239) | merged | [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model | `vllm/model_executor/layers/quantization/quark/quark.py`, `docs/features/quantization/quark.md` |
| 2025-11-14 | [#28275](https://github.com/vllm-project/vllm/pull/28275) | merged | [Misc] add ignore mapper for quark quantization | `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-11-18 | [#28638](https://github.com/vllm-project/vllm/pull/28638) | merged | [ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss | `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-11-24 | [#29067](https://github.com/vllm-project/vllm/pull/29067) | merged | [MoE][Refactor] Make select_experts a non-static method | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-12-08 | [#29773](https://github.com/vllm-project/vllm/pull/29773) | merged | [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-12-09 | [#29066](https://github.com/vllm-project/vllm/pull/29066) | merged | [MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-12-18 | [#30071](https://github.com/vllm-project/vllm/pull/30071) | merged | [Quantization] Support Quark int4-fp8 w4a8 for MoE | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2025-12-20 | [#28891](https://github.com/vllm-project/vllm/pull/28891) | merged | [MoE Refactor][5/N] Isolate zero expert to LongCatFlash | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2025-12-29 | [#31499](https://github.com/vllm-project/vllm/pull/31499) | merged | [MoE Refactor][12/N] Marlin Fp8 MoE Pure Function | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-01-06 | [#31759](https://github.com/vllm-project/vllm/pull/31759) | merged | [MoE Refactor] Add Temporary Integration Tests - H100/B200 | `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` |
| 2026-01-08 | [#31415](https://github.com/vllm-project/vllm/pull/31415) | merged | [MoE Refactor][15/N] Apply Refactor to Fp8 | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-01-08 | [#30519](https://github.com/vllm-project/vllm/pull/30519) | merged | [Misc][Refactor] Add FusedMoERouter object | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-01-15 | [#31827](https://github.com/vllm-project/vllm/pull/31827) | merged | [MoE Refactor][17/N] Apply Refactor to Bf16 | `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` |
| 2026-01-18 | [#30623](https://github.com/vllm-project/vllm/pull/30623) | merged | [MoE Refactor] Separate Router into OO Classes | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-01-20 | [#27814](https://github.com/vllm-project/vllm/pull/27814) | merged | [Refactor] Make FP8 Linear Ops use kernel abstraction | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` |
| 2026-01-22 | [#31996](https://github.com/vllm-project/vllm/pull/31996) | merged | [MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE` | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-01-29 | [#32954](https://github.com/vllm-project/vllm/pull/32954) | merged | [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe | `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` |
| 2026-02-05 | [#33375](https://github.com/vllm-project/vllm/pull/33375) | merged | [Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-02-10 | [#29008](https://github.com/vllm-project/vllm/pull/29008) | merged | [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2026-02-11 | [#32344](https://github.com/vllm-project/vllm/pull/32344) | merged | [MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-02-11 | [#33715](https://github.com/vllm-project/vllm/pull/33715) | merged | [NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE | `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` |
| 2026-02-12 | [#33843](https://github.com/vllm-project/vllm/pull/33843) | merged | [Refactor] Replace `activation: str` with `MoEActivation` enum | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-02-12 | [#34192](https://github.com/vllm-project/vllm/pull/34192) | merged | [ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-02-20 | [#34386](https://github.com/vllm-project/vllm/pull/34386) | merged | [Quark] Fix MoE fp8 activation scale handling on mi300 | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-02-26 | [#33807](https://github.com/vllm-project/vllm/pull/33807) | merged | [UX] Add `--moe-backend` arg for explicit kernel selection | `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` |
| 2026-02-26 | [#30357](https://github.com/vllm-project/vllm/pull/30357) | merged | [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-02 | [#35658](https://github.com/vllm-project/vllm/pull/35658) | merged | [ROCm] add amd-quark package in requirements for rocm to use quantized models | `tests/quantization/test_quark.py` |
| 2026-03-04 | [#35893](https://github.com/vllm-project/vllm/pull/35893) | merged | [ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-13 | [#35316](https://github.com/vllm-project/vllm/pull/35316) | merged | [ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer | `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` |
| 2026-03-17 | [#32779](https://github.com/vllm-project/vllm/pull/32779) | merged | Fix infinite recursive search issue in quark.py | `vllm/model_executor/layers/quantization/quark/quark.py` |
| 2026-03-20 | [#36232](https://github.com/vllm-project/vllm/pull/36232) | merged | [ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization | `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-21 | [#37128](https://github.com/vllm-project/vllm/pull/37128) | merged | [MoE Refactor] Mxfp4 oracle rebased | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-23 | [#36100](https://github.com/vllm-project/vllm/pull/36100) | merged | [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs | `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-25 | [#37833](https://github.com/vllm-project/vllm/pull/37833) | merged | [ROCm] Fix MoE kernel test failures on gfx950 | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-03-27 | [#34285](https://github.com/vllm-project/vllm/pull/34285) | merged | [Refactor] Move FusedMoE hidden_size roundup to quant_method | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-01 | [#35153](https://github.com/vllm-project/vllm/pull/35153) | merged | [MoE Refactor] Make SharedExperts class for use with DefaultMoERunner | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-03 | [#38774](https://github.com/vllm-project/vllm/pull/38774) | merged | [ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` |
| 2026-04-07 | [#38504](https://github.com/vllm-project/vllm/pull/38504) | merged | [Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path | `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` |
| 2026-04-09 | [#33892](https://github.com/vllm-project/vllm/pull/33892) | merged | [W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections. | `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` |
| 2026-04-09 | [#36320](https://github.com/vllm-project/vllm/pull/36320) | merged | [Quantization] Support Quark W8A8 INT8 MoE inference | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py` |
| 2026-04-13 | [#39604](https://github.com/vllm-project/vllm/pull/39604) | merged | [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod" | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-14 | [#39107](https://github.com/vllm-project/vllm/pull/39107) | merged | [MoE Refactor] Remove MoE DP chunking | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-14 | [#39007](https://github.com/vllm-project/vllm/pull/39007) | merged | [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/ | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/models/mixtral.py` |
| 2026-04-22 | [#35737](https://github.com/vllm-project/vllm/pull/35737) | merged | [NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/models/mixtral.py` |

## 逐 PR diff 审计卡

### PR #4 - Use FlashAttention for `multi_query_kv_attention`

- 链接: https://github.com/vllm-project/vllm/pull/4
- 状态/时间: merged / 2023-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+108/-34，可读 patch 236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use FlashAttention for `multi_query_kv_attention`」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `cacheflow/models/attention.py`, `tests/kernels/attention.py`, `server.py`；PR 正文摘要: This PR is to use FlashAttention kernels for `multi_query_kv_attention`, which performs masked attention for the prompt inputs. Pros - FlashAttention is fast and memory-efficien...。
- 实现要点: `cacheflow/models/attention.py` modified +37/-30 (67 lines); hunks: -1,5 +1,6; -14,20 +15,7 @@ def __init__(self, scale: float) -> None:; symbols: __init__, _masked_attention, multi_query_kv_attention, single_query_cached_kv_attention，涉及 `__init__, _masked_attention, multi_query_kv_attention`；`tests/kernels/attention.py` modified +65/-2 (67 lines); hunks: -1,10 +1,13; -79,7 +82,7 @@ def test_single_query_cached_kv_attention(; symbols: ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention, test_attention，涉及 `ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention`；`server.py` modified +5/-2 (7 lines); hunks: -9,10 +9,12; -27,6 +29,7 @@ def main():; symbols: main，涉及 `main`；`README.md` modified +1/-0 (1 lines); hunks: -4,6 +4,7。
- 代码 diff 细节:
  - `cacheflow/models/attention.py` modified +37/-30 (67 lines); hunks: -1,5 +1,6; -14,20 +15,7 @@ def __init__(self, scale: float) -> None:; symbols: __init__, _masked_attention, multi_query_kv_attention, single_query_cached_kv_attention
  - `tests/kernels/attention.py` modified +65/-2 (67 lines); hunks: -1,10 +1,13; -79,7 +82,7 @@ def test_single_query_cached_kv_attention(; symbols: ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention, test_attention
  - `server.py` modified +5/-2 (7 lines); hunks: -9,10 +9,12; -27,6 +29,7 @@ def main():; symbols: main
  - `README.md` modified +1/-0 (1 lines); hunks: -4,6 +4,7
- 关键代码摘录:

```diff
diff -- cacheflow/models/attention.py
@@ -1,5 +1,6 @@
+from flash_attn.flash_attention import FlashAttention
@@ -14,20 +15,7 @@ def __init__(self, scale: float) -> None:
-    def _masked_attention(
-        self,
-        query: torch.Tensor,                        # [num_queries, num_heads, head_size]
-        key: torch.Tensor,                          # [num_keys, num_heads, head_size]
diff -- tests/kernels/attention.py
@@ -1,10 +1,13 @@
+from flash_attn.flash_attention import FlashAttention
+MAX_SEQ_LEN = 4096
@@ -79,7 +82,7 @@ def test_single_query_cached_kv_attention(
-    context_lens = [random.randint(1, 4096) for _ in range(num_tokens)]
+    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
@@ -123,11 +126,60 @@ def test_single_query_cached_kv_attention(
diff -- server.py
@@ -9,10 +9,12 @@
```

- 已读文件:
  - runtime: `cacheflow/models/attention.py` modified +37/-30
  - tests: `tests/kernels/attention.py` modified +65/-2
  - other: `server.py` modified +5/-2; `README.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/kernels/attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2011 - Mixtral 8x7B support

- 链接: https://github.com/vllm-project/vllm/pull/2011
- 状态/时间: merged / 2023-12-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `b5f882cc98e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+538/-0，可读 patch 566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Mixtral 8x7B support」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Adding support for `mistralai/Mixtral-8x7B-v0.1` and `mistralai/Mixtral-8x7B-Instruct-v0.1` models as described in our blogpost. This is joint work between @zhuohan123, @WoosukK...。
- 实现要点: `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunks: -0,0 +1,534; symbols: promote_scalar, MixtralAttention, __init__, forward，涉及 `promote_scalar, MixtralAttention, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunks: -0,0 +1,534; symbols: promote_scalar, MixtralAttention, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -0,0 +1,534 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` added +534/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader.py`, `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2015 - Minor fixes for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/2015
- 状态/时间: merged / 2023-12-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `4ff0203987ff`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+5/-6，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Minor fixes for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR 1) adds Mixtral to supported model list doc, and 2) did minor code cleaning for `mixtral.py`.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +2/-6 (8 lines); hunks: -21,7 +21,7; -453,10 +453,6 @@ def __init__(; symbols: __init__, forward, sample，涉及 `__init__, forward, sample`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +2/-6 (8 lines); hunks: -21,7 +21,7; -453,10 +453,6 @@ def __init__(; symbols: __init__, forward, sample
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -21,7 +21,7 @@
-from typing import List, Optional, Tuple, Union
+from typing import List, Optional, Tuple
@@ -453,10 +453,6 @@ def __init__(
-        self.tok_embeddings: Union[nn.Embedding, None] = None
-        self.layers: nn.ModuleList = None
-        self.output: Union[nn.Linear, None] = None
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2036 - [Minor] Fix type annotation in Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/2036
- 状态/时间: merged / 2023-12-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `81ce2a4b26c7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Minor] Fix type annotation in Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: The following error is raised for users who didn't install `megablocks`. This PR removes this misleading error so that users can see the warning messages on the import statement...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -251,7 +251,7 @@ def sparse_transpose(; symbols: sparse_transpose, topology，涉及 `sparse_transpose, topology`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -251,7 +251,7 @@ def sparse_transpose(; symbols: sparse_transpose, topology
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -251,7 +251,7 @@ def sparse_transpose(
-                 padded_bins: torch.Tensor) -> stk.Matrix:
+                 padded_bins: torch.Tensor) -> "stk.Matrix":
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2090 - Mixtral expert parallelism

- 链接: https://github.com/vllm-project/vllm/pull/2090
- 状态/时间: merged / 2023-12-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `21d93c140d0a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+221/-334，可读 patch 737 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Mixtral expert parallelism」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR implements a more efficient parallelism scheme for the Mixtral model. Instead of sharding the layers of each expert by rank, we instead shard whole experts across ranks....。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunks: -31,22 +31,11; -66,8 +55,134; symbols: promote_scalar, MixtralMLP, __init__, forward，涉及 `promote_scalar, MixtralMLP, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunks: -31,22 +31,11; -66,8 +55,134; symbols: promote_scalar, MixtralMLP, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -31,22 +31,11 @@
-try:
-    import megablocks.ops as ops
-except ImportError as e:
-    raise ImportError("MegaBlocks not found. "
-                      "Please install it by `pip install megablocks`.") from e
-try:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +207/-307
- 验证与风险: runtime 路径改动集中在 `vllm/config.py`, `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2208 - [BugFix] Fix weight loading for Mixtral with TP

- 链接: https://github.com/vllm-project/vllm/pull/2208
- 状态/时间: merged / 2023-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `ba4f82673884`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-26，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix weight loading for Mixtral with TP」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Fixes #2202 Currently, the Mixtral model does not support quantization with TP > 1 because `DummyModule` does not use quantized linear methods. This PR removes `DummyModule` and...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +5/-26 (31 lines); hunks: -49,7 +49,6; -94,30 +93,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, DummyModule, __init__, dummy_weight_loader，涉及 `forward, DummyModule, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +5/-26 (31 lines); hunks: -49,7 +49,6; -94,30 +93,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, DummyModule, __init__, dummy_weight_loader
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -49,7 +49,6 @@
-from vllm.model_executor.utils import set_weight_attrs
@@ -94,30 +93,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-class DummyModule(nn.Module):
-    def __init__(self) -> None:
-        super().__init__()
-        self.w1 = nn.Linear(0, 0, bias=False)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-26
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2293 - tensor parallel MOE implementation

- 链接: https://github.com/vllm-project/vllm/pull/2293
- 状态/时间: closed / 2024-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+515/-126，可读 patch 720 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「tensor parallel MOE implementation」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/moe.py`, `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe_grouped_matmul.py`；PR 正文摘要: This PR implements tensor parallel MOE by sharding each expert across all ranks. concretely, it does following: 1. column parallel each expert's w1 and w3 weights 2. row paralle...。
- 实现要点: `vllm/model_executor/layers/moe.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MoE, __init__, weight_loader, forward，涉及 `MoE, __init__, weight_loader`；`vllm/model_executor/models/mixtral.py` modified +34/-126 (160 lines); hunks: -23,10 +23,7; -35,17 +32,15; symbols: MixtralMLP, __init__, forward, MixtralMoE，涉及 `MixtralMLP, __init__, forward`；`tests/kernels/test_moe_grouped_matmul.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: ref_grouped_matmul, test_moe_grouped_matmul，涉及 `ref_grouped_matmul, test_moe_grouped_matmul`；`csrc/misc_kernels.cu` added +35/-0 (35 lines); hunks: -0,0 +1,35。
- 代码 diff 细节:
  - `vllm/model_executor/layers/moe.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MoE, __init__, weight_loader, forward
  - `vllm/model_executor/models/mixtral.py` modified +34/-126 (160 lines); hunks: -23,10 +23,7; -35,17 +32,15; symbols: MixtralMLP, __init__, forward, MixtralMoE
  - `tests/kernels/test_moe_grouped_matmul.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: ref_grouped_matmul, test_moe_grouped_matmul
  - `csrc/misc_kernels.cu` added +35/-0 (35 lines); hunks: -0,0 +1,35
  - `csrc/dispatch_utils.h` modified +11/-0 (11 lines); hunks: -14,3 +14,14
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/moe.py
@@ -0,0 +1,369 @@
+from typing import Tuple
+import torch
+from torch import nn
+import torch.nn.functional as F
+import triton
+import triton.language as tl
diff -- vllm/model_executor/models/mixtral.py
@@ -23,10 +23,7 @@
-import numpy as np
-import torch.nn.functional as F
@@ -35,17 +32,15 @@
-                                               ReplicatedLinear,
+from vllm.model_executor.layers.moe import MoE
-from vllm.model_executor.parallel_utils.communication_op import (
diff -- tests/kernels/test_moe_grouped_matmul.py
@@ -0,0 +1,57 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/moe.py` added +369/-0; `vllm/model_executor/models/mixtral.py` modified +34/-126
  - tests: `tests/kernels/test_moe_grouped_matmul.py` added +57/-0
  - other: `csrc/misc_kernels.cu` added +35/-0; `csrc/dispatch_utils.h` modified +11/-0; `csrc/ops.h` modified +4/-0; `csrc/pybind.cpp` modified +4/-0; `setup.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe_grouped_matmul.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2453 - DeepseekMoE support with Fused MoE kernel

- 链接: https://github.com/vllm-project/vllm/pull/2453
- 状态/时间: merged / 2024-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+924/-0，可读 patch 957 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DeepseekMoE support with Fused MoE kernel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/layers/fused_moe.py`, `vllm/model_executor/models/__init__.py`；PR 正文摘要: Adding support for DeepseekMoE as described in here. This work was partly done by @esmeetu and DeepSeek-AI We have fixed some bugs in the @esmeetu's code and added support for e...。
- 实现要点: `vllm/model_executor/models/deepseek.py` added +453/-0 (453 lines); hunks: -0,0 +1,453; symbols: DeepseekMLP, __init__, forward, DeepseekMoE，涉及 `DeepseekMLP, __init__, forward`；`vllm/model_executor/layers/fused_moe.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel, fused_moe，涉及 `fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel`；`vllm/model_executor/models/__init__.py` modified +1/-0 (1 lines); hunks: -18,6 +18,7；`csrc/moe_align_block_size_kernels.cu` added +108/-0 (108 lines); hunks: -0,0 +1,108。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek.py` added +453/-0 (453 lines); hunks: -0,0 +1,453; symbols: DeepseekMLP, __init__, forward, DeepseekMoE
  - `vllm/model_executor/layers/fused_moe.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel, fused_moe
  - `vllm/model_executor/models/__init__.py` modified +1/-0 (1 lines); hunks: -18,6 +18,7
  - `csrc/moe_align_block_size_kernels.cu` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/kernels/test_fused_moe.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: torch_moe, test_fused_moe
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek.py
@@ -0,0 +1,453 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/layers/fused_moe.py
@@ -0,0 +1,287 @@
+"""Fused MoE kernel."""
+import torch
+import triton
+import triton.language as tl
+from vllm._C import ops
+@triton.jit
diff -- vllm/model_executor/models/__init__.py
@@ -18,6 +18,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek.py` added +453/-0; `vllm/model_executor/layers/fused_moe.py` added +287/-0; `vllm/model_executor/models/__init__.py` modified +1/-0
  - other: `csrc/moe_align_block_size_kernels.cu` added +108/-0; `csrc/dispatch_utils.h` modified +11/-0; `csrc/ops.h` modified +9/-0; `csrc/pybind.cpp` modified +4/-0
  - tests: `tests/kernels/test_fused_moe.py` added +50/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2542 - Fused MOE for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/2542
- 状态/时间: merged / 2024-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `ab406446691f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+115/-109，可读 patch 327 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fused MOE for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This builds on https://github.com/vllm-project/vllm/pull/2453 and https://github.com/vllm-project/vllm/pull/2293 to fuse the MOE kernel for the Mixtral model. It seems to give a...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunks: -23,8 +23,6; -33,10 +31,11; symbols: MixtralMLP, MixtralMoE, __init__, forward，涉及 `MixtralMLP, MixtralMoE, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunks: -23,8 +23,6; -33,10 +31,11; symbols: MixtralMLP, MixtralMoE, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -23,8 +23,6 @@
-import numpy as np
@@ -33,10 +31,11 @@
+from vllm.model_executor.layers.fused_moe import fused_moe
-                                               ReplicatedLinear,
+                                               ReplicatedLinear,
@@ -47,92 +46,85 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +104/-96
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2677 - Add unit test for Mixtral MoE layer

- 链接: https://github.com/vllm-project/vllm/pull/2677
- 状态/时间: merged / 2024-01-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `d0d93b92b190`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+119/-55，可读 patch 209 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add unit test for Mixtral MoE layer」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This is PR adds a unit test for the Mixtral MoE layer to vLLM. It is based on @casper-hansen 's test in https://github.com/casper-hansen/AutoAWQ/blob/mixtral_fused/tests/test_fu...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +6/-4 (10 lines); hunks: -70,13 +70,14 @@ def __init__(; -141,8 +142,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +6/-4 (10 lines); hunks: -70,13 +70,14 @@ def __init__(; -141,8 +142,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -70,13 +70,14 @@ def __init__(
+        tp_size: Optional[int] = None,
-        tp_size = get_tensor_model_parallel_world_size()
+        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
-        self.intermediate_size = intermediate_size // tp_size
+        self.intermediate_size = intermediate_size // self.tp_size
@@ -141,8 +142,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +6/-4
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_moe.py`, `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2769 - Add fused top-K softmax kernel for MoE

- 链接: https://github.com/vllm-project/vllm/pull/2769
- 状态/时间: merged / 2024-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `f0d4e145575b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+591/-50，可读 patch 772 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add fused top-K softmax kernel for MoE」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR ports a fused topk-softmax kernel from TensorRT-LLM v0.7.1.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +3/-11 (14 lines); hunks: -24,8 +24,6; -128,18 +126,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +3/-11 (14 lines); hunks: -24,8 +24,6; -128,18 +126,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -24,8 +24,6 @@
-import torch.nn.functional as F
@@ -128,18 +126,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
-        routing_weights, selected_experts = torch.topk(routing_weights,
-                                                       self.top_k,
-                                                       dim=-1)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-11
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2831 - Add LoRA support for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/2831
- 状态/时间: merged / 2024-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_mixtral.py`, `vllm/model_executor/models/mixtral.py`；关联提交 `2a543d6efecc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+251/-121，可读 patch 751 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add LoRA support for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`；PR 正文摘要: **Problem**: We don't have LoRA support for Mixtral. **Solution**: Add LoRA configurations for Mixtral and refactor relevant parts to allow this. **Testing**: added correctness...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +37/-3 (40 lines); hunks: -27,6 +27,7; -38,7 +39,7; symbols: __init__, forward, MixtralForCausalLM，涉及 `__init__, forward, MixtralForCausalLM`；`tests/lora/test_mixtral.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: do_sample, test_mixtral_lora，涉及 `do_sample, test_mixtral_lora`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +37/-3 (40 lines); hunks: -27,6 +27,7; -38,7 +39,7; symbols: __init__, forward, MixtralForCausalLM
  - `tests/lora/test_mixtral.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: do_sample, test_mixtral_lora
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -27,6 +27,7 @@
+from vllm.config import LoRAConfig
@@ -38,7 +39,7 @@
-    VocabParallelEmbedding, ParallelLMHead)
+    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
@@ -292,6 +293,7 @@ def __init__(
+            org_num_embeddings=self.org_vocab_size,
diff -- tests/lora/test_mixtral.py
@@ -0,0 +1,53 @@
+import pytest
+import torch
+import vllm
+from vllm.lora.request import LoRARequest
+MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
+def do_sample(llm, lora_path: str, lora_id: int):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +37/-3
  - tests: `tests/lora/test_mixtral.py` added +53/-0
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_lora_manager.py`, `tests/lora/test_mixtral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #2880 - Align LoRA code between Mistral and Mixtral (fixes #2875)

- 链接: https://github.com/vllm-project/vllm/pull/2880
- 状态/时间: merged / 2024-02-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `31348dff03d6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-4，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Align LoRA code between Mistral and Mixtral (fixes #2875)」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This aligns the LoRA code between Mistral and Mixtral and fixes https://github.com/vllm-project/vllm/pull/2875 as a result.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +10/-4 (14 lines); hunks: -285,15 +285,19 @@ def __init__(; -350,7 +354,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +10/-4 (14 lines); hunks: -285,15 +285,19 @@ def __init__(; -350,7 +354,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -285,15 +285,19 @@ def __init__(
+        lora_config: Optional[LoRAConfig] = None,
-        self.vocab_size = config.vocab_size
+        lora_vocab = (lora_config.lora_extra_vocab_size *
+                      (lora_config.max_loras or 1)) if lora_config else 0
+        self.vocab_size = config.vocab_size + lora_vocab
+        self.org_vocab_size = config.vocab_size
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +10/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2875 - Fix AttributeError: MixtralModel object has no attribute org_vocab_size.

- 链接: https://github.com/vllm-project/vllm/pull/2875
- 状态/时间: closed / 2024-02-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `31348dff03d6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix AttributeError: MixtralModel object has no attribute org_vocab_size.」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Mixtral cannot start because of this error: Related PR: #2831。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -293,7 +293,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -293,7 +293,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -293,7 +293,7 @@ def __init__(
-            org_num_embeddings=self.org_vocab_size,
+            org_num_embeddings=self.vocab_size,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #3597 - [BugFix] 1D query fix for MoE models

- 链接: https://github.com/vllm-project/vllm/pull/3597
- 状态/时间: merged / 2024-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `41deac4a3d78`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+15/-15，可读 patch 93 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] 1D query fix for MoE models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: MoE models were broken by #3236.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +3/-4 (7 lines); hunks: -124,9 +124,9 @@ def weight_loader(self, param: nn.Parameter, loaded_weight:...; -140,8 +140,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: weight_loader, forward, MixtralAttention，涉及 `weight_loader, forward, MixtralAttention`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +3/-4 (7 lines); hunks: -124,9 +124,9 @@ def weight_loader(self, param: nn.Parameter, loaded_weight:...; -140,8 +140,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: weight_loader, forward, MixtralAttention
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -124,9 +124,9 @@ def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
-        batch_size, sequence_length, hidden_size = hidden_states.shape
+        num_tokens, hidden_size = hidden_states.shape
-        # router_logits: (batch * sequence_length, n_experts)
+        # router_logits: (num_tokens, n_experts)
@@ -140,8 +140,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        return final_hidden_states.view(batch_size, sequence_length,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-4
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4244 - [Kernel] FP8 support for MoE kernel / Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/4244
- 状态/时间: merged / 2024-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `eace8bf0b911`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+385/-21，可读 patch 626 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] FP8 support for MoE kernel / Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR is the first step towards fixing https://github.com/vllm-project/vllm/pull/3208 It implements dynamic per-tensor scaling (see https://github.com/vllm-project/vllm/pull/4...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +42/-2 (44 lines); hunks: -39,6 +39,8; -47,6 +49,7; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading，涉及 `MixtralMoE, __init__, weight_loader`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +42/-2 (44 lines); hunks: -39,6 +39,8; -47,6 +49,7; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -39,6 +39,8 @@
+from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod,
+                                                         per_tensor_quantize)
@@ -47,6 +49,7 @@
+from vllm.utils import print_warning_once
@@ -66,13 +69,17 @@ def __init__(
+        linear_method: Optional[LinearMethodBase] = None,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +42/-2
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4343 - [Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales

- 链接: https://github.com/vllm-project/vllm/pull/4343
- 状态/时间: merged / 2024-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `12628d3c787e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+95/-18，可读 patch 233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR contains the performance improvements mentioned in https://github.com/vllm-project/vllm/pull/4244 The performance numbers are as follows (TP2, H100) -- with mistralai/Mi...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +37/-7 (44 lines); hunks: -105,6 +105,13 @@ def __init__(; -115,12 +122,23 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading，涉及 `__init__, weight_loader, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +37/-7 (44 lines); hunks: -105,6 +105,13 @@ def __init__(; -115,12 +122,23 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -105,6 +105,13 @@ def __init__(
+        set_weight_attrs(self.ws, {
+            "weight_loader": self.weight_loader,
+        })
+        set_weight_attrs(self.w2s, {
+            "weight_loader": self.weight_loader,
+        })
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +37/-7
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4332 - [Kernel] Support Fp8 Checkpoints (Dynamic + Static)

- 链接: https://github.com/vllm-project/vllm/pull/4332
- 状态/时间: merged / 2024-04-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+307/-40，可读 patch 549 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support Fp8 Checkpoints (Dynamic + Static)」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/fp8.py`, `tests/models/test_fp8.py`, `vllm/model_executor/layers/linear.py`；PR 正文摘要: This PR does two things: - Supports loading serialized fp8 models with static weight and act scales - Supports loading serialized fp8 models with static weight scales and dynami...。
- 实现要点: `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30 (199 lines); hunks: -1,23 +1,36; -30,22 +43,22 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: Fp8Config, for, __init__, get_supported_act_dtypes，涉及 `Fp8Config, for, __init__`；`tests/models/test_fp8.py` added +90/-0 (90 lines); hunks: -0,0 +1,90; symbols: test_models，涉及 `test_models`；`vllm/model_executor/layers/linear.py` modified +48/-10 (58 lines); hunks: -246,6 +246,10 @@ def __init__(; -254,6 +258,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader，涉及 `__init__, weight_loader`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30 (199 lines); hunks: -1,23 +1,36; -30,22 +43,22 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: Fp8Config, for, __init__, get_supported_act_dtypes
  - `tests/models/test_fp8.py` added +90/-0 (90 lines); hunks: -0,0 +1,90; symbols: test_models
  - `vllm/model_executor/layers/linear.py` modified +48/-10 (58 lines); hunks: -246,6 +246,10 @@ def __init__(; -254,6 +258,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/fp8.py
@@ -1,23 +1,36 @@
-from typing import Any, Dict, List, Optional
+from typing import Any, Dict, List, Optional, Tuple, Union
+from vllm.logger import init_logger
-    QuantizationConfig, QuantizeMethodBase)
+    QuantizationConfig)
+ACTIVATION_SCHEMES = ["static", "dynamic"]
diff -- tests/models/test_fp8.py
@@ -0,0 +1,90 @@
+# flake8: noqa
+"""Tests fp8 models against ground truth generation
+Note: these tests will only pass on L4 GPU.
+"""
+import os
+import pytest
diff -- vllm/model_executor/layers/linear.py
@@ -246,6 +246,10 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30; `vllm/model_executor/layers/linear.py` modified +48/-10
  - tests: `tests/models/test_fp8.py` added +90/-0
- 验证与风险: diff 自带测试面 `tests/models/test_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4436 - [Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static)

- 链接: https://github.com/vllm-project/vllm/pull/4436
- 状态/时间: closed / 2024-05-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+105/-40，可读 patch 223 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static)」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/model_loader/weight_utils.py`；PR 正文摘要: Follow on to #4332 (should wait until 4332 is merged) This PR does a few things: - Supports loading fp8 checkpoints for Mixtral, such as this test model - Supports static activa...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +100/-40 (140 lines); hunks: -46,7 +46,8; -78,6 +79,8 @@ def __init__(; symbols: __init__, weight_loader，涉及 `__init__, weight_loader`；`vllm/model_executor/model_loader/weight_utils.py` modified +5/-0 (5 lines); hunks: -370,3 +370,8 @@ def initialize_dummy_weights(; symbols: initialize_dummy_weights, all_close_1d，涉及 `initialize_dummy_weights, all_close_1d`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +100/-40 (140 lines); hunks: -46,7 +46,8; -78,6 +79,8 @@ def __init__(; symbols: __init__, weight_loader
  - `vllm/model_executor/model_loader/weight_utils.py` modified +5/-0 (5 lines); hunks: -370,3 +370,8 @@ def initialize_dummy_weights(; symbols: initialize_dummy_weights, all_close_1d
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -46,7 +46,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    all_close_1d, default_weight_loader)
@@ -78,6 +79,8 @@ def __init__(
+        self.quant_config = quant_config
@@ -86,24 +89,28 @@ def __init__(
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -370,3 +370,8 @@ def initialize_dummy_weights(
+def all_close_1d(x: torch.Tensor) -> bool:
+    assert len(x.shape) == 1
+    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +100/-40; `vllm/model_executor/model_loader/weight_utils.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4543 - [Misc] Remove Mixtral device="cuda" declarations

- 链接: https://github.com/vllm-project/vllm/pull/4543
- 状态/时间: merged / 2024-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `c9d852d601ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-8，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Remove Mixtral device="cuda" declarations」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Remove the `device="cuda"` declarations in mixtral as promised in https://github.com/vllm-project/vllm/pull/4343 PR Checklist (Click to Expand) Thank you for your contribution t...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +4/-8 (12 lines); hunks: -96,13 +96,11 @@ def __init__(; -114,22 +112,20 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +4/-8 (12 lines); hunks: -96,13 +96,11 @@ def __init__(; -114,22 +112,20 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -96,13 +96,11 @@ def __init__(
-                        device="cuda",
-                        device="cuda",
@@ -114,22 +112,20 @@ def __init__(
-            torch.ones(
-                self.num_total_experts, device="cuda", dtype=torch.float32),
+            torch.ones(self.num_total_experts, dtype=torch.float32),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +4/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4527 - [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)

- 链接: https://github.com/vllm-project/vllm/pull/4527
- 状态/时间: merged / 2024-05-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `2a052011ca47`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+122/-53，可读 patch 262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Follow on to https://github.com/vllm-project/vllm/pull/4332 to enable FP8 checkpoint loading for Mixtral and supersedes https://github.com/vllm-project/vllm/pull/4436. This PR e...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunks: -78,6 +78,8 @@ def __init__(; -86,55 +88,79 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading，涉及 `__init__, weight_loader, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunks: -78,6 +78,8 @@ def __init__(; -86,55 +88,79 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -78,6 +78,8 @@ def __init__(
+        self.quant_config = quant_config
@@ -86,55 +88,79 @@ def __init__(
+        # Gate always runs at half / full precision for now.
-        self.ws = nn.Parameter(
+        if self.use_fp8:
+            params_dtype = torch.float8_e4m3fn
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +120/-51
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4793 - [Bugfix] Fix dynamic FP8 quantization for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/4793
- 状态/时间: merged / 2024-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `33d3914b1e6d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix dynamic FP8 quantization for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: If the checkpoint is not an FP8 checkpoint, we first load the weights as FP16 and then quantize them after they are loaded. A small bug was introduced in https://github.com/vllm...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -95,7 +95,7 @@ def __init__(
-        if self.use_fp8:
+        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4893 - [Misc] Load FP8 kv-cache scaling factors from checkpoints

- 链接: https://github.com/vllm-project/vllm/pull/4893
- 状态/时间: merged / 2024-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `a3a73ab0696b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 40 个文件，+284/-158，可读 patch 869 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Load FP8 kv-cache scaling factors from checkpoints」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: The 2nd PR for #4532. This PR supports loading FP8 kv-cache scaling factors from a FP8 checkpoint (with `.kv_scale` parameter). Specifically, 1. We now support `--kv-cache-dtype...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +21/-8 (29 lines); hunks: -308,14 +308,13 @@ def __init__(self,; -581,6 +580,20 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +21/-8 (29 lines); hunks: -308,14 +308,13 @@ def __init__(self,; -581,6 +580,20 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -308,14 +308,13 @@ def __init__(self,
-        self.attn = Attention(
-            self.num_heads,
-            self.head_dim,
-            self.scaling,
-            num_kv_heads=self.num_kv_heads,
-            sliding_window=self.sliding_window,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +21/-8
- 验证与风险: diff 自带测试面 `tests/models/test_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5039 - [Model] Enable FP8 QKV in MoE and refine kernel tuning script

- 链接: https://github.com/vllm-project/vllm/pull/5039
- 状态/时间: merged / 2024-05-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `e9899fb7a4d9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+711/-114，可读 patch 1000 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable FP8 QKV in MoE and refine kernel tuning script」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: In this PR, we enable QKV in FP8 for Mixtral models. Gate linear is still in FP16. Since most MoE models now have 8 experts, the shape of gate weights is not divisible by 16, wh...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +0/-9 (9 lines); hunks: -278,15 +278,6 @@ def __init__(self,; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +0/-9 (9 lines); hunks: -278,15 +278,6 @@ def __init__(self,; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -278,15 +278,6 @@ def __init__(self,
-        if isinstance(
-                quant_config,
-                Fp8Config) and not quant_config.is_checkpoint_fp8_serialized:
-            print_warning_once(
-                "For Mixtral FP8 quantization, we currently do not quantize "
-                "the attention layers until their FP8 performance is improved."
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/configs/E=8,N=14336,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5231 - [Model] Correct Mixtral FP8 checkpoint loading

- 链接: https://github.com/vllm-project/vllm/pull/5231
- 状态/时间: merged / 2024-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `5563a4dea86e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+80/-35，可读 patch 182 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Correct Mixtral FP8 checkpoint loading」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR fixes the issue the the weight scaling factors of w1 and w3 are not properly loaded. Since we merge w1 and w3 in vLLM for better performance, their weights and scaling f...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +76/-32 (108 lines); hunks: -41,7 +41,9; -98,16 +100,16 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading，涉及 `__init__, weight_loader, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +76/-32 (108 lines); hunks: -41,7 +41,9; -98,16 +100,16 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -41,7 +41,9 @@
-from vllm.model_executor.layers.quantization.fp8 import Fp8Config
+from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
+                                                         per_tensor_dequantize,
+                                                         per_tensor_quantize)
@@ -98,16 +100,16 @@ def __init__(
-        self.w13_weight = nn.Parameter(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +76/-32
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5353 - [Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale

- 链接: https://github.com/vllm-project/vllm/pull/5353
- 状态/时间: merged / 2024-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `c09dade2a263`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+23/-23，可读 patch 127 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: In tandem with https://github.com/neuralmagic/AutoFP8/pull/11. BREAKING CHANGE: Because there can be input and output scales (`kv_scale` is an example of an "output" scale for k...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +8/-8 (16 lines); hunks: -147,7 +147,7 @@ def __init__(; -182,11 +182,11 @@ def weight_loader(self, param: nn.Parameter, loaded_weight...; symbols: __init__, weight_loader, process_weights_after_loading, load_weights，涉及 `__init__, weight_loader, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +8/-8 (16 lines); hunks: -147,7 +147,7 @@ def __init__(; -182,11 +182,11 @@ def weight_loader(self, param: nn.Parameter, loaded_weight...; symbols: __init__, weight_loader, process_weights_after_loading, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -147,7 +147,7 @@ def __init__(
-            # ACT_SCALE (for fp8)
+            # INPUT_SCALE (for fp8)
@@ -182,11 +182,11 @@ def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
-        if "act_scale" in weight_name or "w2.weight_scale" in weight_name:
+        if "input_scale" in weight_name or "w2.weight_scale" in weight_name:
-                    "act_scales of w1 and w3 of a layer "
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +8/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5970 - [ Misc ] Refactor MoE to isolate Fp8 From Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/5970
- 状态/时间: merged / 2024-07-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `7c008c51a9aa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+537/-306，可读 patch 1033 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ Misc ] Refactor MoE to isolate Fp8 From Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: MOTIVATION: * Prior to this PR, MoE layer + weight loading logic is repeated for all models * Prior to this PR, fp8 logic for MoE is coupled with Mixtral and unsupported for oth...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +41/-235 (276 lines); hunks: -27,30 +27,23; -66,227 +59,40 @@ class MixtralMoE(nn.Module):; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading，涉及 `MixtralMoE, __init__, weight_loader`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +41/-235 (276 lines); hunks: -27,30 +27,23; -66,227 +59,40 @@ class MixtralMoE(nn.Module):; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -27,30 +27,23 @@
-from vllm import _custom_ops as ops
-from vllm.distributed import (get_tensor_model_parallel_rank,
-                              get_tensor_model_parallel_world_size,
-                              tensor_model_parallel_all_reduce)
-from vllm.model_executor.layers.fused_moe import fused_moe
+from vllm.distributed import get_tensor_model_parallel_world_size
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +41/-235
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6287 - [Bugfix] Support 2D input shape in MoE layer

- 链接: https://github.com/vllm-project/vllm/pull/6287
- 状态/时间: merged / 2024-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `e72ae80b0640`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Support 2D input shape in MoE layer」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: The MoE layer current assumes 1D input shape, i.e., `[num_tokens, hidden_size]`. This PR enhances this to support both 1D and 2D (`[batch_size, seq_len, hidden_size]`) input sha...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +3/-2 (5 lines); hunks: -88,12 +88,13 @@ def __init__(self,; symbols: __init__, forward, MixtralAttention，涉及 `__init__, forward, MixtralAttention`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +3/-2 (5 lines); hunks: -88,12 +88,13 @@ def __init__(self,; symbols: __init__, forward, MixtralAttention
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -88,12 +88,13 @@ def __init__(self,
-        num_tokens, hidden_size = hidden_states.shape
+        # NOTE: hidden_states can have either 1D or 2D shape.
+        orig_shape = hidden_states.shape
-        return final_hidden_states.view(num_tokens, hidden_size)
+        return final_hidden_states.view(orig_shape)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/qwen2_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6417 - [ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8

- 链接: https://github.com/vllm-project/vllm/pull/6417
- 状态/时间: merged / 2024-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `fb6af8bc0863`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+223/-137，可读 patch 564 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: SUMMARY: * apply MoE refactor to DeepSeekv2 to support `fp8` * * apply MoE refactor to Qwen2Moe to support `fp8` * in combination with #6415, this will support fused AWQ as well...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +7/-25 (32 lines); hunks: -372,31 +372,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +7/-25 (32 lines); hunks: -372,31 +372,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -372,31 +372,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        expert_params_mapping = [
-            # These are the weight scales for the experts
-            # (param_name, weight_name, expert_id, shard_id)
-            ("experts.w13_scale"
-             if weight_name in ["w1", "w3"] else "experts.w2_scale",
-             f"experts.{expert_id}.{weight_name}.weight_scale", expert_id,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +7/-25
- 验证与风险: runtime 路径改动集中在 `.buildkite/lm-eval-harness/configs/DeepSeek-V2-Lite-Chat.yaml`, `.buildkite/lm-eval-harness/configs/models-large.txt`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6516 - [Model] Pipeline parallel support for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/6516
- 状态/时间: merged / 2024-07-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `b5af8c223c3d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+60/-19，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Pipeline parallel support for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Take from #6403. Co-authored by @binxuan cc @youkaichao。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +48/-13 (61 lines); hunks: -29,7 +29,7; -48,6 +48,7; symbols: MixtralMoE, __init__, forward，涉及 `MixtralMoE, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +48/-13 (61 lines); hunks: -29,7 +29,7; -48,6 +48,7; symbols: MixtralMoE, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -29,7 +29,7 @@
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
@@ -48,6 +48,7 @@
+from .utils import is_pp_missing_parameter, make_layers
@@ -255,12 +256,11 @@ def __init__(
-        self.layers = nn.ModuleList([
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +48/-13
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7334 - [Misc] Update Fused MoE weight loading

- 链接: https://github.com/vllm-project/vllm/pull/7334
- 状态/时间: merged / 2024-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `d3bdfd3ab9ba`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+264/-201，可读 patch 614 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Update Fused MoE weight loading」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: - Splits up #6422 into two separate PRs. This is the first of the two. The second will leverage the weight loading changes introduced in this PR while adding the AWQ Fused MoE K...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                                  weight_name,
+                                  name,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7527 - [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel

- 链接: https://github.com/vllm-project/vllm/pull/7527
- 状态/时间: merged / 2024-08-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `8678a69ab519`, `aae74ef95c37`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+2374/-84，可读 patch 2645 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: - Expands weight loading to support grouped and per channel weight quantization. Cleans-up fp8 MoE to use the updated weight loading - Adds Marlin Fused MoE Kernel for `w4a16` b...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,6 +73,7 @@ def __init__(self,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/weight_loading/models.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7764 - Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)"

- 链接: https://github.com/vllm-project/vllm/pull/7764
- 状态/时间: merged / 2024-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `aae74ef95c37`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+84/-2374，可读 patch 2645 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)"」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This reverts commit 8678a69ab51956031e3bb70bdf1a781a8652e67d from PR https://github.com/vllm-project/vllm/pull/7527 since it causes a failure in the AMD build。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self,; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self,; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,7 +73,6 @@ def __init__(self,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-1
- 验证与风险: diff 自带测试面 `tests/weight_loading/models.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7766 - [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel

- 链接: https://github.com/vllm-project/vllm/pull/7766
- 状态/时间: merged / 2024-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `fc911880cc50`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2382/-85，可读 patch 2666 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: - Expands weight loading to support grouped and per channel weight quantization. Cleans-up fp8 MoE to use the updated weight loading - Adds Marlin Fused MoE Kernel for `w4a16` b...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,6 +73,7 @@ def __init__(self,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_compressed_tensors.py`, `tests/weight_loading/models.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8217 - [Misc] Fused MoE Marlin support for GPTQ

- 链接: https://github.com/vllm-project/vllm/pull/8217
- 状态/时间: merged / 2024-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `6cd5e5b07e44`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+912/-204，可读 patch 1442 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Fused MoE Marlin support for GPTQ」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: - Add GPTQ Marlin MoE Support; marlin MoE kernels currently support int4 - Update/add optional testing for large MoE models for GPTQ and llm-compressor Co-authored by @ElizaWszo...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +7/-2 (9 lines); hunks: -435,7 +435,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -454,6 +455,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +7/-2 (9 lines); hunks: -435,7 +435,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -454,6 +455,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -435,7 +435,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                if name.endswith(".bias") and name not in params_dict:
+                if ((name.endswith(".bias") or name.endswith("_bias"))
+                        and name not in params_dict):
@@ -454,6 +455,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                    if ((name.endswith(".bias") or name.endswith("_bias"))
+                            and name not in params_dict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +7/-2
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`, `tests/weight_loading/models-large.txt`, `tests/weight_loading/models.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9008 - [Model] add a bunch of supported lora modules for mixtral

- 链接: https://github.com/vllm-project/vllm/pull/9008
- 状态/时间: merged / 2024-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_mixtral.py`, `vllm/model_executor/models/mixtral.py`；关联提交 `9ade8bbc8dc6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+69/-20，可读 patch 123 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] add a bunch of supported lora modules for mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`；PR 正文摘要: `all-linear` option for mixtral produces these `target modules`: This PR adds the missing modules (`w1`, `w2`, `w3` and `gate`) to the supported_lora_modules in `mixtral`. Updat...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +2/-4 (6 lines); hunks: -319,10 +319,8 @@ class MixtralForCausalLM(nn.Module, SupportsLoRA):; symbols: MixtralForCausalLM，涉及 `MixtralForCausalLM`；`tests/lora/test_mixtral.py` modified +62/-16 (78 lines); hunks: -9,12 +9,9; -33,22 +30,71 @@ def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -...; symbols: do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules，涉及 `do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +2/-4 (6 lines); hunks: -319,10 +319,8 @@ class MixtralForCausalLM(nn.Module, SupportsLoRA):; symbols: MixtralForCausalLM
  - `tests/lora/test_mixtral.py` modified +62/-16 (78 lines); hunks: -9,12 +9,9; -33,22 +30,71 @@ def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -...; symbols: do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -319,10 +319,8 @@ class MixtralForCausalLM(nn.Module, SupportsLoRA):
-        "qkv_proj",
-        "o_proj",
-        "embed_tokens",
-        "lm_head",
+        "qkv_proj", "o_proj", "embed_tokens", "lm_head", "w1", "w2", "w3",
+        "gate"
diff -- tests/lora/test_mixtral.py
@@ -9,12 +9,9 @@
-def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
-    prompts = [
-        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This f
-        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This f
-        "[system] Given a target sentence construct the underlying meaning representation\nof the input sentence as a single function with attributes and attribute\nvalues. This f
-    ]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-4
  - tests: `tests/lora/test_mixtral.py` modified +62/-16
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_mixtral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9632 - [torch.compile] support moe models

- 链接: https://github.com/vllm-project/vllm/pull/9632
- 状态/时间: merged / 2024-10-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+217/-78，可读 patch 605 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile] support moe models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`；PR 正文摘要: moe models will read config file to determine the triton config to run. reading files during forward is a disaster for `torch.compile` . this pr wraps the config reading part in...。
- 实现要点: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7 (100 lines); hunks: -358,9 +358,10 @@ def try_get_optimal_moe_config(; -465,19 +466,109 @@ def get_config_dtype_str(dtype: torch.dtype,; symbols: try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts, _，涉及 `try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts`；`vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9 (51 lines); hunks: -1,6 +1,6; -18,6 +18,7 @@ def get_scalar_type(num_bits: int, has_zp: bool):; symbols: get_scalar_type, single_marlin_moe，涉及 `get_scalar_type, single_marlin_moe`；`vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11 (29 lines); hunks: -12,7 +12,16; -96,9 +105,6 @@ def forward_cuda(; symbols: forward_cuda, forward_tpu, FusedMoE，涉及 `forward_cuda, forward_tpu, FusedMoE`；`vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4 (28 lines); hunks: -1,23 +1,43; symbols: override_config, get_config，涉及 `override_config, get_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7 (100 lines); hunks: -358,9 +358,10 @@ def try_get_optimal_moe_config(; -465,19 +466,109 @@ def get_config_dtype_str(dtype: torch.dtype,; symbols: try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts, _
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9 (51 lines); hunks: -1,6 +1,6; -18,6 +18,7 @@ def get_scalar_type(num_bits: int, has_zp: bool):; symbols: get_scalar_type, single_marlin_moe
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11 (29 lines); hunks: -12,7 +12,16; -96,9 +105,6 @@ def forward_cuda(; symbols: forward_cuda, forward_tpu, FusedMoE
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4 (28 lines); hunks: -1,23 +1,43; symbols: override_config, get_config
  - `vllm/model_executor/layers/quantization/awq_marlin.py` modified +2/-5 (7 lines); hunks: -3,6 +3,7; -435,10 +436,6 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -358,9 +358,10 @@ def try_get_optimal_moe_config(
-    override_config: Optional[Dict[str, Any]] = None,
+    from vllm.model_executor.layers.fused_moe import get_config
+    override_config = get_config()
@@ -465,19 +466,109 @@ def get_config_dtype_str(dtype: torch.dtype,
+@torch.library.custom_op("vllm::inplace_fused_experts",
+                         mutates_args=["hidden_states"])
diff -- vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
@@ -1,6 +1,6 @@
-from typing import Any, Dict, Optional
+from typing import Optional
@@ -18,6 +18,7 @@ def get_scalar_type(num_bits: int, has_zp: bool):
+@torch.library.custom_op("vllm::single_marlin_moe", mutates_args=[])
@@ -28,7 +29,6 @@ def single_marlin_moe(
-    override_config: Optional[Dict[str, Any]] = None,
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -12,7 +12,16 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9; `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11; `vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4; `vllm/model_executor/layers/quantization/awq_marlin.py` modified +2/-5; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-5
- 验证与风险: diff 自带测试面 `tests/compile/test_basic_correctness.py`, `tests/kernels/test_awq_marlin.py`, `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9758 - [torch.compile] Adding "torch compile" annotations to some models

- 链接: https://github.com/vllm-project/vllm/pull/9758
- 状态/时间: merged / 2024-10-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `aa0addb39726`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+8/-0，可读 patch 64 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile] Adding "torch compile" annotations to some models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This PR is to complete #9589 and #9632, adding "torch compile" annotations to some moe models and testing whether they can pass the compilation.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -245,6 +246,7 @@ def forward(; symbols: forward, MixtralModel, __init__，涉及 `forward, MixtralModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -245,6 +246,7 @@ def forward(; symbols: forward, MixtralModel, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -28,6 +28,7 @@
+from vllm.compilation.decorators import support_torch_compile
@@ -245,6 +246,7 @@ def forward(
+@support_torch_compile
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/olmoe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11390 - [Bugfix] Fix fully sharded LoRAs with Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/11390
- 状态/时间: merged / 2024-12-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_mixtral.py`；关联提交 `f1d1bf6288ab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+5/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix fully sharded LoRAs with Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `tests/lora/test_mixtral.py`；PR 正文摘要: Fixes a regression introduced by #9008 , which leads to an assertion error when `--fully-sharded-loras` is enabled with an adaptor that includes a gate target: This occurs becau...。
- 实现要点: `tests/lora/test_mixtral.py` modified +3/-1 (4 lines); hunks: -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_al...; symbols: test_mixtral_lora, test_mixtral_lora_all_target_modules，涉及 `test_mixtral_lora, test_mixtral_lora_all_target_modules`。
- 代码 diff 细节:
  - `tests/lora/test_mixtral.py` modified +3/-1 (4 lines); hunks: -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_al...; symbols: test_mixtral_lora, test_mixtral_lora_all_target_modules
- 关键代码摘录:

```diff
diff -- tests/lora/test_mixtral.py
@@ -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):
+@pytest.mark.parametrize("fully_shard", [True, False])
-                                         tp_size):
+                                         tp_size, fully_shard):
@@ -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_all_target_modules,
+        fully_sharded_loras=fully_shard,
```

- 已读文件:
  - tests: `tests/lora/test_mixtral.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/lora/test_mixtral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10765 - [Misc][Quark] Upstream Quark format to VLLM

- 链接: https://github.com/vllm-project/vllm/pull/10765
- 状态/时间: merged / 2025-01-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/__init__.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` 等 10 个文件；关联提交 `de0526f668d6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 32 个文件，+1264/-70，可读 patch 1679 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Quark] Upstream Quark format to VLLM」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；PR 正文摘要: This PR is related to the issue discussed in this https://github.com/vllm-project/vllm/issues/10294.。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0 (387 lines); hunks: -0,0 +1,387; symbols: QuarkConfig, __init__, get_linear_method, get_supported_act_dtypes，涉及 `QuarkConfig, __init__, get_linear_method`；`vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod, __init__，涉及 `QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod`；`vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading，涉及 `QuarkW8A8Fp8, __init__, get_min_capability`；`vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: QuarkW8A8Int8, __init__, get_min_capability, create_weights，涉及 `QuarkW8A8Int8, __init__, get_min_capability`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0 (387 lines); hunks: -0,0 +1,387; symbols: QuarkConfig, __init__, get_linear_method, get_supported_act_dtypes
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: QuarkW8A8Int8, __init__, get_min_capability, create_weights
  - `vllm/model_executor/layers/quantization/quark/utils.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: deep_compare, should_ignore_layer, check_equal_or_regex_match, _is_equal_or_regex_match
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -0,0 +1,387 @@
+import fnmatch
+import re
+from typing import Any, Dict, List, Optional, cast
+import torch
+from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -0,0 +1,225 @@
+from typing import Any, Callable, Dict, Optional
+import torch
+import vllm.model_executor.layers.fused_moe  # noqa
+from vllm import _custom_ops as ops
+from vllm.logger import init_logger
+from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -0,0 +1,140 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0; `vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0; `vllm/model_executor/layers/quantization/quark/utils.py` added +99/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_scheme.py` added +52/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11528 - [BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order

- 链接: https://github.com/vllm-project/vllm/pull/11528
- 状态/时间: merged / 2025-01-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `eb5cb5e5280c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+243/-148，可读 patch 756 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Summary - Fix parameter names to ensure proper weight loading - Fix `process_after_weight_loading` if running group act order - Don't shard w2 weight_scales when running actorde...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9 (20 lines); hunks: -60,24 +60,26 @@ def __init__(self, weight_config: Dict[str, Any], input_conf...; symbols: __init__, create_weights，涉及 `__init__, create_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9 (20 lines); hunks: -60,24 +60,26 @@ def __init__(self, weight_config: Dict[str, Any], input_conf...; symbols: __init__, create_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -60,24 +60,26 @@ def __init__(self, weight_config: Dict[str, Any], input_config: Dict[str,
-                       hidden_size: int, intermediate_size: int,
+                       hidden_size: int, intermediate_size_per_partition: int,
-        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
-                                                    2 * intermediate_size,
-                                                    hidden_size,
-                                                    dtype=params_dtype),
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/awq_marlin.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #3208 - [RFC/WIP] First steps towards FP8 for Mixtral

- 链接: https://github.com/vllm-project/vllm/pull/3208
- 状态/时间: closed / 2025-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+203/-71，可读 patch 473 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[RFC/WIP] First steps towards FP8 for Mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；PR 正文摘要: Even after the optimizations https://github.com/vllm-project/vllm/pull/2979 and https://github.com/vllm-project/vllm/pull/2542, most of the time in the Mixtral forward pass is s...。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22 (44 lines); hunks: -1,24 +1,24；`vllm/model_executor/models/mixtral.py` modified +36/-2 (38 lines); hunks: -21,6 +21,7; -93,13 +94,23 @@ def __init__(; symbols: __init__, weight_loader, forward，涉及 `__init__, weight_loader, forward`；`vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14 (32 lines); hunks: -114,7 +114,7 @@ def fused_moe_kernel(; -186,7 +186,7 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, fused_moe，涉及 `fused_moe_kernel, invoke_fused_moe_kernel, fused_moe`；`vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22 (44 lines); hunks: -1,24 +1,24
  - `vllm/model_executor/models/mixtral.py` modified +36/-2 (38 lines); hunks: -21,6 +21,7; -93,13 +94,23 @@ def __init__(; symbols: __init__, weight_loader, forward
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14 (32 lines); hunks: -114,7 +114,7 @@ def fused_moe_kernel(; -186,7 +186,7 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, fused_moe
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` removed +0/-20 (20 lines); hunks: -1,20 +0,0
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json
@@ -1,24 +1,24 @@
-    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 4},
-    "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
-    "4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
-    "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 4},
-    "16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 4},
-    "24": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 4},
diff -- vllm/model_executor/models/mixtral.py
@@ -21,6 +21,7 @@
+import os
@@ -93,13 +94,23 @@ def __init__(
-                        dtype=self.params_dtype))
+                        dtype=torch.float8_e4m3fn))
-                        dtype=self.params_dtype))
+                        dtype=torch.float8_e4m3fn))
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -114,7 +114,7 @@ def fused_moe_kernel(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22; `vllm/model_executor/models/mixtral.py` modified +36/-2; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` removed +0/-20
  - other: `csrc/activation_kernels.cu` modified +40/-0; `benchmarks/kernels/benchmark_mixtral_moe.py` modified +4/-7
  - docs: `examples/quantization/mixtral/mixtral_fp8_quantization.py` added +34/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13591 - [core] set up data parallel communication

- 链接: https://github.com/vllm-project/vllm/pull/13591
- 状态/时间: merged / 2025-02-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+416/-28，可读 patch 719 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[core] set up data parallel communication」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/distributed/utils.py`, `examples/offline_inference/data_parallel.py`, `vllm/distributed/parallel_state.py`；PR 正文摘要: We need to explore data parallel in many cases, e.g. in deepseek models, and moe models. While the end-user interface is still to be designed, this PR first creates the necessar...。
- 实现要点: `vllm/distributed/utils.py` modified +90/-1 (91 lines); hunks: -11,7 +11,11; -227,3 +231,88 @@ def create(; symbols: create, stateless_init_torch_distributed_process_group，涉及 `create, stateless_init_torch_distributed_process_group`；`examples/offline_inference/data_parallel.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: main，涉及 `main`；`vllm/distributed/parallel_state.py` modified +62/-14 (76 lines); hunks: -750,6 +750,13 @@ def get_tp_group() -> GroupCoordinator:; -811,6 +818,21 @@ def init_distributed_environment(; symbols: get_tp_group, get_dp_group, get_pp_group, init_distributed_environment，涉及 `get_tp_group, get_dp_group, get_pp_group`；`vllm/config.py` modified +57/-0 (57 lines); hunks: -16,6 +16,7; -1290,6 +1291,11 @@ class ParallelConfig:; symbols: ParallelConfig, get_next_dp_init_port, stateless_init_dp_group, has_unfinished_dp，涉及 `ParallelConfig, get_next_dp_init_port, stateless_init_dp_group`。
- 代码 diff 细节:
  - `vllm/distributed/utils.py` modified +90/-1 (91 lines); hunks: -11,7 +11,11; -227,3 +231,88 @@ def create(; symbols: create, stateless_init_torch_distributed_process_group
  - `examples/offline_inference/data_parallel.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: main
  - `vllm/distributed/parallel_state.py` modified +62/-14 (76 lines); hunks: -750,6 +750,13 @@ def get_tp_group() -> GroupCoordinator:; -811,6 +818,21 @@ def init_distributed_environment(; symbols: get_tp_group, get_dp_group, get_pp_group, init_distributed_environment
  - `vllm/config.py` modified +57/-0 (57 lines); hunks: -16,6 +16,7; -1290,6 +1291,11 @@ class ParallelConfig:; symbols: ParallelConfig, get_next_dp_init_port, stateless_init_dp_group, has_unfinished_dp
  - `vllm/forward_context.py` modified +31/-3 (34 lines); hunks: -4,9 +4,10; -32,6 +33,8 @@ class ForwardContext:; symbols: ForwardContext, get_forward_context, set_forward_context
- 关键代码摘录:

```diff
diff -- vllm/distributed/utils.py
@@ -11,7 +11,11 @@
-from torch.distributed import TCPStore
+from torch.distributed import ProcessGroup, TCPStore
+from torch.distributed.distributed_c10d import (Backend, PrefixStore,
+                                                _get_default_timeout,
+                                                is_nccl_available)
+from torch.distributed.rendezvous import rendezvous
diff -- examples/offline_inference/data_parallel.py
@@ -0,0 +1,76 @@
+# SPDX-License-Identifier: Apache-2.0
+# usage: VLLM_USE_V1=1 python examples/offline_inference/data_parallel.py
+# we need to have a launcher to create multiple data parallel
+# ranks. And each rank will create a vLLM instance to process its own prompts.
+import os
+from vllm import LLM, SamplingParams
diff -- vllm/distributed/parallel_state.py
@@ -750,6 +750,13 @@ def get_tp_group() -> GroupCoordinator:
```

- 已读文件:
  - runtime: `vllm/distributed/utils.py` modified +90/-1; `vllm/distributed/parallel_state.py` modified +62/-14; `vllm/config.py` modified +57/-0; `vllm/forward_context.py` modified +31/-3; `vllm/v1/engine/llm_engine.py` modified +24/-2; `vllm/envs.py` modified +20/-0
  - docs: `examples/offline_inference/data_parallel.py` added +76/-0
- 验证与风险: runtime 路径改动集中在 `vllm/config.py`, `vllm/distributed/device_communicators/cuda_communicator.py`, `vllm/distributed/device_communicators/custom_all_reduce.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13784 - [Bugfix][Quantization] Fix FP8 + EP

- 链接: https://github.com/vllm-project/vllm/pull/13784
- 状态/时间: merged / 2025-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `1e15aaef562d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+22/-22，可读 patch 138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Quantization] Fix FP8 + EP」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: `Fp8MoEMethod` is reaching into the `layer` to get the number of experts during `process_weights_after_loading`, which isn't right when using expert parallelism. Would hit the f...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-        for expert_id in range(layer.num_experts):
+        for expert_id in range(layer.local_num_experts):
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/awq_marlin.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13931 - [V1] EP/TP MoE + DP Attention

- 链接: https://github.com/vllm-project/vllm/pull/13931
- 状态/时间: merged / 2025-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `72c62eae5f01`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+250/-75，可读 patch 778 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] EP/TP MoE + DP Attention」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: Based on https://github.com/vllm-project/vllm/pull/13591 DP+EP implemented via collective ops in the fused_moe layer's forward pass.。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -71,6 +71,7 @@ def __init__(self,; -93,6 +94,7 @@ def __init__(self,; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -71,6 +71,7 @@ def __init__(self,; -93,6 +94,7 @@ def __init__(self,; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -71,6 +71,7 @@ def __init__(self,
+                 dp_size: Optional[int] = None,
@@ -93,6 +94,7 @@ def __init__(self,
+                                dp_size=dp_size,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14390 - [FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object

- 链接: https://github.com/vllm-project/vllm/pull/14390
- 状态/时间: merged / 2025-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `e1744502c21f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+268/-242，可读 patch 760 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；PR 正文摘要: This PR replaces `apply_fp8_linear` and `apply_fp8_linear_generic` with objects so that VllmConfig can be accessed in their `__init__` method as opposed to the `forward` method.。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11 (18 lines); hunks: -7,8 +7,7; -22,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; symbols: QuarkW8A8Fp8, __init__, get_min_capability, apply_weights，涉及 `QuarkW8A8Fp8, __init__, get_min_capability`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11 (18 lines); hunks: -7,8 +7,7; -22,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; symbols: QuarkW8A8Fp8, __init__, get_min_capability, apply_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -7,8 +7,7 @@
-    apply_fp8_linear, cutlass_fp8_supported, normalize_e4m3fn_to_e4m3fnuz,
-    requantize_with_max_scale)
+    Fp8LinearOp, normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
@@ -22,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):
-        self.cutlass_fp8_supported = cutlass_fp8_supported()
+        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14245 - dynamic distpatch of fp8 kernels

- 链接: https://github.com/vllm-project/vllm/pull/14245
- 状态/时间: merged / 2025-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `a1c8f3796c89`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 25 个文件，+293/-159，可读 patch 944 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「dynamic distpatch of fp8 kernels」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This mostly affects ROCm with hardware that can support one or the other FP8 type. All fp8 kernels are now templated on fp8_type instead of assuming a single fp8_type via `using...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2 (4 lines); hunks: -39,7 +39,7 @@ def process_weights_after_loading(self, layer) -> None:; -55,7 +55,7 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`；`vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2 (3 lines); hunks: -142,8 +142,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2 (4 lines); hunks: -39,7 +39,7 @@ def process_weights_after_loading(self, layer) -> None:; -55,7 +55,7 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2 (3 lines); hunks: -142,8 +142,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -39,7 +39,7 @@ def process_weights_after_loading(self, layer) -> None:
-            if current_platform.is_rocm():
+            if current_platform.is_fp8_fnuz():
@@ -55,7 +55,7 @@ def process_weights_after_loading(self, layer) -> None:
-            if current_platform.is_rocm():
+            if current_platform.is_fp8_fnuz():
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -142,8 +142,7 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-        # If rocm, normalize the weights and scales to e4m3fnuz
-        if current_platform.is_rocm():
+        if current_platform.is_fp8_fnuz():
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2
- 验证与风险: diff 自带测试面 `tests/kernels/quant_utils.py`, `tests/kernels/test_triton_scaled_mm.py`, `tests/quantization/test_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14961 - [Bugfix][Model] Mixtral: use unused head_dim config argument

- 链接: https://github.com/vllm-project/vllm/pull/14961
- 状态/时间: merged / 2025-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `aaaec52ad907`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-2，可读 patch 54 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Mixtral: use unused head_dim config argument」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: `MixtralConfig` has a a `head_dim` argument that will never be used as `MixtralAttention` always uses `hidden_size // self.total_num_heads` as its value. This change brings `Mix...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +5/-1 (6 lines); hunks: -111,6 +111,7 @@ class MixtralAttention(nn.Module):; -136,7 +137,9 @@ def __init__(; symbols: MixtralAttention, __init__，涉及 `MixtralAttention, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +5/-1 (6 lines); hunks: -111,6 +111,7 @@ class MixtralAttention(nn.Module):; -136,7 +137,9 @@ def __init__(; symbols: MixtralAttention, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -111,6 +111,7 @@ class MixtralAttention(nn.Module):
+        config: MixtralConfig,
@@ -136,7 +137,9 @@ def __init__(
-        self.head_dim = hidden_size // self.total_num_heads
+        # MixtralConfig has an optional head_dim argument
+        self.head_dim = getattr(config, "head_dim",
+                                self.hidden_size // self.total_num_heads)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/mixtral_quant.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14578 - [Quantization][FP8] Adding support for fp8 gemm layer input in fp8

- 链接: https://github.com/vllm-project/vllm/pull/14578
- 状态/时间: merged / 2025-03-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `4d0ec37267af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+41/-9，可读 patch 155 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization][FP8] Adding support for fp8 gemm layer input in fp8」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；PR 正文摘要: Adding support for the case where both inputs to the FP8 GEMM are in FP8 datatype and not only weights (in preparation for attention with fused FP8 conversion) Functionality por...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optio...; -134,5 +135,6 @@ def apply_weights(self,; symbols: __init__, get_min_capability, apply_weights，涉及 `__init__, get_min_capability, apply_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optio...; -134,5 +135,6 @@ def apply_weights(self,; symbols: __init__, get_min_capability, apply_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optional[bool]):
+        self.out_dtype = torch.get_default_dtype()
@@ -134,5 +135,6 @@ def apply_weights(self,
+                                     out_dtype=self.out_dtype,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/fbgemm_fp8.py`, `vllm/model_executor/layers/quantization/fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15961 - Add support to modelopt quantization of Mixtral model

- 链接: https://github.com/vllm-project/vllm/pull/15961
- 状态/时间: merged / 2025-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-1，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support to modelopt quantization of Mixtral model」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral_quant.py`；PR 正文摘要: Add vllm support to modelopt fp8 quantization of Mixtral8x7b model。
- 实现要点: `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunks: -45,7 +45,8; -420,6 +421,11 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunks: -45,7 +45,8; -420,6 +421,11 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral_quant.py
@@ -45,7 +45,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader, maybe_remap_kv_scale_name)
@@ -420,6 +421,11 @@ def load_weights(self, weights: Iterable[Tuple[str,
+            if name.endswith("scale"):
+                # Remapping the name of FP8 kv-scale.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral_quant.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral_quant.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16325 - [Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral

- 链接: https://github.com/vllm-project/vllm/pull/16325
- 状态/时间: merged / 2025-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `a564797151a0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+353/-323，可读 patch 836 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: fix (partially) #15697 * GraniteForCausalLM * GraniteMoeForCausalLM * GraniteMoeShardedForCausalLM * Grok1ForCausalLM * MixtralForCausalLM。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +90/-86 (176 lines); hunks: -49,7 +49,7; -260,6 +260,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, MixtralForCausalLM, get_input_embeddings，涉及 `__init__, forward, MixtralForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +90/-86 (176 lines); hunks: -49,7 +49,7; -260,6 +260,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, MixtralForCausalLM, get_input_embeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -49,7 +49,7 @@
-from .utils import (is_pp_missing_parameter,
+from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
@@ -260,6 +260,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.config = config
+        self.quant_config = quant_config
@@ -313,88 +315,6 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +90/-86
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/granite.py`, `vllm/model_executor/models/granitemoe.py`, `vllm/model_executor/models/granitemoeshared.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16236 - [Bugfix] Fix bugs of running Quark quantized models

- 链接: https://github.com/vllm-project/vllm/pull/16236
- 状态/时间: merged / 2025-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`；关联提交 `9e90c9f73f94`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+67/-22，可读 patch 183 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix bugs of running Quark quantized models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`；PR 正文摘要: 1. Fix fp8 activation dynamic and weight per-channel 2. Fix int8 weight per-channel and weight_zero_point/input_zero_point loading issues。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10 (34 lines); hunks: -35,7 +35,7 @@ def create_weights(self, layer: torch.nn.Module,; -63,16 +63,28 @@ def create_weights(self, layer: torch.nn.Module,; symbols: create_weights, process_weights_after_loading，涉及 `create_weights, process_weights_after_loading`；`vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4 (10 lines); hunks: -21,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; -41,10 +41,11 @@ def process_weights_after_loading(self, layer) -> None:; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, create_weights，涉及 `QuarkW8A8Fp8, __init__, process_weights_after_loading`；`tests/quantization/test_quark.py` modified +37/-8 (45 lines); hunks: -4,17 +4,28; -26,11 +37,29 @@ def check_model(model):; symbols: test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor, check_model，涉及 `test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10 (34 lines); hunks: -35,7 +35,7 @@ def create_weights(self, layer: torch.nn.Module,; -63,16 +63,28 @@ def create_weights(self, layer: torch.nn.Module,; symbols: create_weights, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4 (10 lines); hunks: -21,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; -41,10 +41,11 @@ def process_weights_after_loading(self, layer) -> None:; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, create_weights
  - `tests/quantization/test_quark.py` modified +37/-8 (45 lines); hunks: -4,17 +4,28; -26,11 +37,29 @@ def check_model(model):; symbols: test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor, check_model
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py
@@ -35,7 +35,7 @@ def create_weights(self, layer: torch.nn.Module,
-        self.logical_widths = output_partition_sizes
+        layer.logical_widths = output_partition_sizes
@@ -63,16 +63,28 @@ def create_weights(self, layer: torch.nn.Module,
-                data=torch.empty((sum(output_partition_sizes), 1),
+                data=torch.empty((sum(output_partition_sizes)),
+            ChannelQuantZPParameter = ChannelQuantScaleParameter
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -21,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):
-        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)
+        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=False)
@@ -41,10 +41,11 @@ def process_weights_after_loading(self, layer) -> None:
+                input_scale = getattr(layer, 'input_scale', None)
-                    input_scale=layer.input_scale)
+                    input_scale=input_scale)
diff -- tests/quantization/test_quark.py
@@ -4,17 +4,28 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4
  - tests: `tests/quantization/test_quark.py` modified +37/-8
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15734 - [Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization

- 链接: https://github.com/vllm-project/vllm/pull/15734
- 状态/时间: merged / 2025-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `a41351f363f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+105/-20，可读 patch 212 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: This PR adds fp8 quantization support for: 1) Preserving FP8 quantization after FA output, so that output of FA into the next layer will be FP8 - this uses the model parameter s...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20 (47 lines); hunks: -1,7 +1,6; -125,6 +124,13 @@ def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":; symbols: from_config, get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod，涉及 `from_config, get_cache_scale, has_fp8_layer_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20 (47 lines); hunks: -1,7 +1,6; -125,6 +124,13 @@ def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":; symbols: from_config, get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -1,7 +1,6 @@
-import re
@@ -125,6 +124,13 @@ def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":
+            # In case q_proj output is also quantized, remove the configuration
+            # to keep qkv consistency.
+            q_proj_q_config = cast(Dict[str, Any],
+                                   layer_quant_config.get("*q_proj"))
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20
- 验证与风险: runtime 路径改动集中在 `vllm/attention/backends/abstract.py`, `vllm/attention/backends/rocm_flash_attn.py`, `vllm/attention/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17215 - [AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary

- 链接: https://github.com/vllm-project/vllm/pull/17215
- 状态/时间: merged / 2025-04-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `68af5f6c5ce4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+0/-29，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: This PR removes a check in arg_utils.py for FP8 that is not necessary. It turns out that a different backend is used for FP8 when running FP8 on V1 so it will not hit the Triton...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12 (12 lines); hunks: -307,18 +307,6 @@ def get_cache_scale(self, name: str) -> Optional[str]:; symbols: get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod，涉及 `get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12 (12 lines); hunks: -307,18 +307,6 @@ def get_cache_scale(self, name: str) -> Optional[str]:; symbols: get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -307,18 +307,6 @@ def get_cache_scale(self, name: str) -> Optional[str]:
-    def has_fp8_layer_weights(self):
-        layer_quant_config = self.quant_config.get("layer_quant_config")
-        to_dict = lambda obj: cast(Dict[str, Any], obj) or {}
-        return any([
-            'fp8' in cast(
-                str,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12
- 验证与风险: runtime 路径改动集中在 `vllm/engine/arg_utils.py`, `vllm/model_executor/layers/quantization/quark/quark.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16943 - [Quantization] Quark MXFP4 format loading

- 链接: https://github.com/vllm-project/vllm/pull/16943
- 状态/时间: merged / 2025-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`；关联提交 `db593aa67f8d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+289/-3，可读 patch 378 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Quark MXFP4 format loading」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`；PR 正文摘要: Initial PR to integrate loading MXFP4 models quantized by Quark. This PR supports running MXFP4 emulation for devices where micro-scaling datatype is not natively supported. Nex...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1 (56 lines); hunks: -5,6 +5,7; -15,13 +16,15; symbols: QuarkConfig, get_quant_method, _is_static_tensor_w8a8, _is_mx_fp4，涉及 `QuarkConfig, get_quant_method, _is_static_tensor_w8a8`；`vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1 (3 lines); hunks: -1,7 +1,8。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1 (56 lines); hunks: -5,6 +5,7; -15,13 +16,15; symbols: QuarkConfig, get_quant_method, _is_static_tensor_w8a8, _is_mx_fp4
  - `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1 (3 lines); hunks: -1,7 +1,8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -5,6 +5,7 @@
+from vllm.logger import init_logger
@@ -15,13 +16,15 @@
-    QuarkScheme, QuarkW8A8Fp8, QuarkW8A8Int8)
+    QuarkScheme, QuarkW4A4MXFP4, QuarkW8A8Fp8, QuarkW8A8Int8)
+logger = init_logger(__name__)
@@ -67,6 +70,7 @@ def get_quant_method(self, layer: torch.nn.Module,
diff -- vllm/model_executor/layers/quantization/quark/schemes/__init__.py
@@ -1,7 +1,8 @@
+from .quark_w4a4_mxfp4 import QuarkW4A4MXFP4
-__all__ = ["QuarkScheme", "QuarkW8A8Fp8", "QuarkW8A8Int8"]
+__all__ = ["QuarkScheme", "QuarkW8A8Fp8", "QuarkW8A8Int8", "QuarkW4A4MXFP4"]
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/models/quantization/test_mxfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12612 - [Bugfix] Fix quark fp8 format loading on AMD GPUs

- 链接: https://github.com/vllm-project/vllm/pull/12612
- 状态/时间: merged / 2025-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `bb239a730f2a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-9，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix quark fp8 format loading on AMD GPUs」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`；PR 正文摘要: https://github.com/vllm-project/vllm/pull/10765 was merged into vllm to support Quark quantized models. Unfortunately, tests are somewhat limited and the issue fixed in this PR...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9 (21 lines); hunks: -34,21 +34,24 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`；`tests/quantization/test_quark.py` modified +26/-0 (26 lines); hunks: -5,6 +5,7; -63,3 +64,28 @@ def check_model(model):; symbols: check_model, test_quark_fp8_parity，涉及 `check_model, test_quark_fp8_parity`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9 (21 lines); hunks: -34,21 +34,24 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading
  - `tests/quantization/test_quark.py` modified +26/-0 (26 lines); hunks: -5,6 +5,7; -63,3 +64,28 @@ def check_model(model):; symbols: check_model, test_quark_fp8_parity
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -34,21 +34,24 @@ def process_weights_after_loading(self, layer) -> None:
-            max_w_scale, weight = requantize_with_max_scale(
-                weight=layer.weight,
-                weight_scale=layer.weight_scale,
-                logical_widths=layer.logical_widths,
-            )
-            if current_platform.is_fp8_fnuz():
diff -- tests/quantization/test_quark.py
@@ -5,6 +5,7 @@
+import torch
@@ -63,3 +64,28 @@ def check_model(model):
+def test_quark_fp8_parity(vllm_runner):
+    quark_model_id = "amd-quark/llama-tiny-fp8-quark-quant-method"
+    fp8_model_id = "amd-quark/llama-tiny-fp8-quant-method"
+    llm_kwargs = {
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9
  - tests: `tests/quantization/test_quark.py` modified +26/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20251 - [Bugfix] fix quark ptpc

- 链接: https://github.com/vllm-project/vllm/pull/20251
- 状态/时间: merged / 2025-06-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `1c50e100a9c5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+23/-16，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix quark ptpc」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: This is a fix for importing the **quark** ptpc quantization format.。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11 (33 lines); hunks: -1,7 +1,7; -19,10 +19,19; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading，涉及 `QuarkW8A8Fp8, __init__, process_weights_after_loading`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5 (6 lines); hunks: -312,11 +312,7 @@ def _get_scheme_from_config(self, config: dict[str, Any]) -...; symbols: _get_scheme_from_config，涉及 `_get_scheme_from_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11 (33 lines); hunks: -1,7 +1,7; -19,10 +19,19; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5 (6 lines); hunks: -312,11 +312,7 @@ def _get_scheme_from_config(self, config: dict[str, Any]) -...; symbols: _get_scheme_from_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -1,7 +1,7 @@
-from typing import Callable, Optional
+from typing import Any, Callable, Optional, cast
@@ -19,10 +19,19 @@
-    def __init__(self, qscheme: str, is_static_input_scheme: Optional[bool]):
-        self.qscheme = qscheme
-        self.is_static_input_scheme = is_static_input_scheme
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -312,11 +312,7 @@ def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkScheme":
-                weight_qscheme = cast(str, weight_config.get("qscheme"))
-                input_static = (input_config is not None and
-                                not cast(bool, input_config.get("is_dynamic")))
-                return QuarkW8A8Fp8(qscheme=weight_qscheme,
-                                    is_static_input_scheme=input_static)
+                return QuarkW8A8Fp8(weight_config, input_config)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11; `vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17888 - [Feature][Quantization] MXFP4 support for MOE models

- 链接: https://github.com/vllm-project/vllm/pull/17888
- 状态/时间: merged / 2025-07-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/features/quantization/quark.md`, `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `332d4cb17b7c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+875/-106，可读 patch 1416 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature][Quantization] MXFP4 support for MOE models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`；PR 正文摘要: This PR follows https://github.com/vllm-project/vllm/pull/16943, and adds the possibility to load MOE models using MXFP4 weights with dynamic per-group MXFP4 quantization for ac...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3 (170 lines); hunks: -5,19 +5,22; -40,6 +43,8 @@ def get_moe_method(; symbols: QuarkMoEMethod, get_moe_method, apply, QuarkW4A4MXFp4MoEMethod，涉及 `QuarkMoEMethod, get_moe_method, apply`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6 (6 lines); hunks: -237,12 +237,6 @@ def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],; symbols: _is_mx_fp4，涉及 `_is_mx_fp4`；`tests/quantization/test_quark.py` modified +171/-0 (171 lines); hunks: -3,15 +3,44; -90,3 +119,145 @@ def test_quark_fp8_parity(vllm_runner):; symbols: use_v0_only, test_quark_fp8_parity, ModelCase, GSM8KAccuracyTestConfig，涉及 `use_v0_only, test_quark_fp8_parity, ModelCase`；`docs/features/quantization/quark.md` modified +25/-0 (25 lines); hunks: -232,3 +232,28 @@ python3 quantize_quark.py --model_dir meta-llama/Llama-2-70...。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3 (170 lines); hunks: -5,19 +5,22; -40,6 +43,8 @@ def get_moe_method(; symbols: QuarkMoEMethod, get_moe_method, apply, QuarkW4A4MXFp4MoEMethod
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6 (6 lines); hunks: -237,12 +237,6 @@ def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],; symbols: _is_mx_fp4
  - `tests/quantization/test_quark.py` modified +171/-0 (171 lines); hunks: -3,15 +3,44; -90,3 +119,145 @@ def test_quark_fp8_parity(vllm_runner):; symbols: use_v0_only, test_quark_fp8_parity, ModelCase, GSM8KAccuracyTestConfig
  - `docs/features/quantization/quark.md` modified +25/-0 (25 lines); hunks: -232,3 +232,28 @@ python3 quantize_quark.py --model_dir meta-llama/Llama-2-70...
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -5,19 +5,22 @@
-import vllm.model_executor.layers.fused_moe  # noqa
+from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
+    OCP_MX_BLOCK_SIZE)
-__all__ = ["QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod"]
+__all__ = [
+    "QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod", "QuarkW4A4MXFp4MoEMethod"
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -237,12 +237,6 @@ def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],
-        # Weights need to use static quantization.
-        if weight_quant.get("is_dynamic") is True:
-            logger.debug(
-                "Quark model is not in MX-FP4 format: not weight static")
-            return False
diff -- tests/quantization/test_quark.py
@@ -3,15 +3,44 @@
+See also `tests/kernels/moe/test_mxfp4_moe.py`.
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3; `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6
  - tests: `tests/quantization/test_quark.py` modified +171/-0
  - docs: `docs/features/quantization/quark.md` modified +25/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_moe.py`, `tests/kernels/moe/test_mxfp4_moe.py`, `tests/quantization/reference_mxfp4.py`, `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19830 - [Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf

- 链接: https://github.com/vllm-project/vllm/pull/19830
- 状态/时间: merged / 2025-07-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `31d5c1797f32`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+368/-104，可读 patch 858 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5 (16 lines); hunks: -7,6 +7,8; -28,10 +30,14 @@ def __init__(self, weight_config: dict[str, Any],; symbols: __init__, process_weights_after_loading，涉及 `__init__, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5 (16 lines); hunks: -7,6 +7,8; -28,10 +30,14 @@ def __init__(self, weight_config: dict[str, Any],; symbols: __init__, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -7,6 +7,8 @@
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    GroupShape)
@@ -28,10 +30,14 @@ def __init__(self, weight_config: dict[str, Any],
-        self.use_per_token_if_dynamic = (not self.is_static_input_scheme \
-            and self.input_qscheme == "per_channel")
+        per_token = (not self.is_static_input_scheme
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion.py`, `tests/compile/test_fusion_attn.py`, `tests/compile/test_silu_mul_quant_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20893 - [Quantization] add BNB for MixtralForCausalLM

- 链接: https://github.com/vllm-project/vllm/pull/20893
- 状态/时间: merged / 2025-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `a99b9f7dee0a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+128/-20，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] add BNB for MixtralForCausalLM」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +13/-8 (21 lines); hunks: -317,6 +317,15 @@ def forward(; -326,16 +335,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights，涉及 `forward, get_expert_mapping, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +13/-8 (21 lines); hunks: -317,6 +317,15 @@ def forward(; -326,16 +335,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -317,6 +317,15 @@ def forward(
+    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+        # Params for weights, fp8 weight scales, fp8 activation scales
+        # (param_name, weight_name, expert_id, shard_id)
+        return FusedMoE.make_expert_params_mapping(
+            ckpt_gate_proj_name="w1",
+            ckpt_down_proj_name="w2",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +13/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/granitemoe.py`, `vllm/model_executor/models/granitemoeshared.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22035 - [Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods.

- 链接: https://github.com/vllm-project/vllm/pull/22035
- 状态/时间: merged / 2025-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `8ad7285ea28a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 54 个文件，+2022/-1305，可读 patch 5535 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods.」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: - Pass `FusedMoEConfig` to all `FusedMoEMethodBase` object constructors - Make sure `self.fused_experts` is set to None in the constructor and only set once by `init_prepare_fin...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9 (39 lines); hunks: -7,7 +7,8; -25,6 +26,9; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW8A8Fp8MoEMethod，涉及 `QuarkMoEMethod, __init__, get_moe_method`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9 (39 lines); hunks: -7,7 +7,8; -25,6 +26,9; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW8A8Fp8MoEMethod
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -7,7 +7,8 @@
-from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
+from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
+                                                  FusedMoEMethodBase,
@@ -25,6 +26,9 @@
+    def __init__(self, moe: FusedMoEConfig):
+        super().__init__(moe)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py`, `tests/kernels/moe/modular_kernel_tools/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23123 - Add routed_scaling_factor to MoE grouped topk

- 链接: https://github.com/vllm-project/vllm/pull/23123
- 状态/时间: merged / 2025-08-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `8fb85b7bb674`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+77/-4，可读 patch 570 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add routed_scaling_factor to MoE grouped topk」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This PR add `routed_scaling_factor` to MoE grouped topk. * In `vllm/model_executor/layers/fused_moe/layer.py`, add `routed_scaling_factor` parameter to `grouped_topk()`. * In `v...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0 (4 lines); hunks: -218,6 +218,7 @@ def apply(; -244,6 +245,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0 (4 lines); hunks: -218,6 +218,7 @@ def apply(; -244,6 +245,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -218,6 +218,7 @@ def apply(
+        routed_scaling_factor: float = 1.0,
@@ -244,6 +245,7 @@ def apply(
+            routed_scaling_factor=routed_scaling_factor,
@@ -380,6 +382,7 @@ def apply(
+        routed_scaling_factor: float = 1.0,
@@ -406,6 +409,7 @@ def apply(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24649 - [Rocm] [quantization] Fix quark ptpc moe and add test case

- 链接: https://github.com/vllm-project/vllm/pull/24649
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `ca2d1925ef5a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+196/-52，可读 patch 325 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Rocm] [quantization] Fix quark ptpc moe and add test case」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/quantization/test_quark.py`；PR 正文摘要: 1.Add support for loading quark's ptpc-format moe models 2.add test case for quark ptpc。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52 (223 lines); hunks: -5,17 +5,25; -67,21 +75,45 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, apply，涉及 `__init__, create_weights, process_weights_after_loading`；`tests/quantization/test_quark.py` modified +25/-0 (25 lines); hunks: -77,6 +77,31 @@ def check_model(model):; symbols: check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor，涉及 `check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52 (223 lines); hunks: -5,17 +5,25; -67,21 +75,45 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, apply
  - `tests/quantization/test_quark.py` modified +25/-0 (25 lines); hunks: -77,6 +77,31 @@ def check_model(model):; symbols: check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -5,17 +5,25 @@
+import vllm.envs as envs
+from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
+    is_rocm_aiter_moe_enabled)
+from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
+    prepare_moe_fp8_layer_for_marlin)
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
diff -- tests/quantization/test_quark.py
@@ -77,6 +77,31 @@ def check_model(model):
+@pytest.mark.parametrize('tp', [1])
+def test_quark_fp8_w_per_channel_a_per_token(vllm_runner, tp):
+    model_path = "amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts"
+    with vllm_runner(model_path, tensor_parallel_size=tp) as llm:
+        def check_model(model):
+            layer = model.model.layers[0]
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52
  - tests: `tests/quantization/test_quark.py` modified +25/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22842 - [EPLB] Support EPLB for Mixtral Model

- 链接: https://github.com/vllm-project/vllm/pull/22842
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `b77bf34e531a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+137/-23，可读 patch 265 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[EPLB] Support EPLB for Mixtral Model」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunks: -23,7 +23,8; -33,8 +34,9; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunks: -23,7 +23,8; -33,8 +34,9; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -23,7 +23,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -33,8 +34,9 @@
-from vllm.config import CacheConfig, VllmConfig
-from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +137/-23
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22537 - [Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses

- 链接: https://github.com/vllm-project/vllm/pull/22537
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `5963b98b4650`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 68 个文件，+2675/-2503，可读 patch 9071 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: - Delegate construction of `FusedMoEQuantConfig` objects to the subclass of `FusedMoEMethodBase` that will use that info. - Move all/most quantization info into `FusedMoEQuantCo...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19 (46 lines); hunks: -11,6 +11,9; -287,6 +290,16 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply，涉及 `process_weights_after_loading, get_fused_moe_quant_config, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19 (46 lines); hunks: -11,6 +11,9; -287,6 +290,16 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -11,6 +11,9 @@
+from vllm.model_executor.layers.fused_moe.config import (
+    FusedMoEQuantConfig, fp8_w8a8_moe_quant_config,
+    mxfp4_w4a4_moe_quant_config)
@@ -287,6 +290,16 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+    def get_fused_moe_quant_config(
+            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/make_feature_matrix.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/test_batched_deepgemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21166 - [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4

- 链接: https://github.com/vllm-project/vllm/pull/21166
- 状态/时间: merged / 2025-10-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+658/-182，可读 patch 1462 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`；PR 正文摘要: As per title. Support for MXFP4 in vllm was added in https://github.com/vllm-project/vllm/pull/16943 and https://github.com/vllm-project/vllm/pull/17888. However, some hardware...。
- 实现要点: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: _quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6, _dequant_mxfp6_fake，涉及 `_quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6`；`vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45 (140 lines); hunks: -1,22 +1,34; -96,23 +108,57 @@ def gemm_with_dynamic_quant_fake(; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4, QuarkOCP_MX，涉及 `is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4`；`vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21 (90 lines); hunks: -42,6 +42,8; -1323,7 +1325,7 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts，涉及 `inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts`；`vllm/model_executor/layers/fused_moe/config.py` modified +56/-17 (73 lines); hunks: -9,6 +9,10; -30,7 +34,7 @@ def _get_config_dtype_str(; symbols: _get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4, ocp_mx_scheme，涉及 `_get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: _quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6, _dequant_mxfp6_fake
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45 (140 lines); hunks: -1,22 +1,34; -96,23 +108,57 @@ def gemm_with_dynamic_quant_fake(; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4, QuarkOCP_MX
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21 (90 lines); hunks: -42,6 +42,8; -1323,7 +1325,7 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `vllm/model_executor/layers/fused_moe/config.py` modified +56/-17 (73 lines); hunks: -9,6 +9,10; -30,7 +34,7 @@ def _get_config_dtype_str(; symbols: _get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4, ocp_mx_scheme
  - `vllm/model_executor/layers/fused_moe/utils.py` modified +60/-13 (73 lines); hunks: -16,11 +16,15; -106,12 +110,14 @@ def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) ->...; symbols: _resize_cache, _fp4_quantize, _nvfp4_quantize, _fp8_quantize
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/mxfp6_utils.py
@@ -0,0 +1,142 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
+from vllm.utils import direct_register_custom_op
+def _quant_dequant_mxfp6(
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py
@@ -1,22 +1,34 @@
-from functools import cache
-from typing import Any, Callable, Optional
+from fractions import Fraction
+from functools import cache, partial
+from typing import Any, Callable, Optional, Union
-from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -42,6 +42,8 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21; `vllm/model_executor/layers/fused_moe/config.py` modified +56/-17; `vllm/model_executor/layers/fused_moe/utils.py` modified +60/-13; `vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py` added +54/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_ocp_mx_moe.py`, `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26364 - [ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py`

- 链接: https://github.com/vllm-project/vllm/pull/26364
- 状态/时间: merged / 2025-10-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`；关联提交 `a38c1bfe09f4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py`」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `tests/quantization/test_quark.py`；PR 正文摘要: As per title, as reported in https://github.com/vllm-project/vllm/pull/21166#issuecomment-3377626487 `test_mxfp4_moe.py` was renamed, but not in `.buildkite/test-pipeline.yaml`....。
- 实现要点: `tests/quantization/test_quark.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7。
- 代码 diff 细节:
  - `tests/quantization/test_quark.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7
- 关键代码摘录:

```diff
diff -- tests/quantization/test_quark.py
@@ -4,7 +4,7 @@
-See also `tests/kernels/moe/test_mxfp4_moe.py`.
+See also `tests/kernels/moe/test_ocp_mx_moe.py`.
```

- 已读文件:
  - tests: `tests/quantization/test_quark.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26739 - [torch.compile] Unwrap fused_marlin_moe custom op

- 链接: https://github.com/vllm-project/vllm/pull/26739
- 状态/时间: merged / 2025-10-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `8ae169286f93`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+22/-52，可读 patch 228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile] Unwrap fused_marlin_moe custom op」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Unwrap unnecessary CustomOP'ing of `fused_marlin_moe`. `fused_marlin_moe` was first wrapped into a custom op in the PR https://github.com/vllm-project/vllm/pull/9632 as `fused_m...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -20,6 +20,7; -402,7 +403,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -20,6 +20,7; -402,7 +403,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -20,6 +20,7 @@
+from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
@@ -402,7 +403,7 @@ def apply(
-            return torch.ops.vllm.fused_marlin_moe(
+            return fused_marlin_moe(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26545 - [ROCM] MoE fp4 CK kernel

- 链接: https://github.com/vllm-project/vllm/pull/26545
- 状态/时间: merged / 2025-10-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `0925b28a8e92`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+73/-24，可读 patch 144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCM] MoE fp4 CK kernel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: server: client or returns reasonable results. Correctness:。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24 (92 lines); hunks: -23,6 +23,7; -472,22 +473,22 @@ def __init__(; symbols: __init__, get_packed_dim, create_weights, process_weights_after_loading，涉及 `__init__, get_packed_dim, create_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24 (92 lines); hunks: -23,6 +23,7; -472,22 +473,22 @@ def __init__(; symbols: __init__, get_packed_dim, create_weights, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -23,6 +23,7 @@
+    use_mxfp4_aiter_moe,
@@ -472,22 +473,22 @@ def __init__(
-        if not current_platform.supports_mx():
-            self.emulate = True
+        self.emulate = not current_platform.supports_mx() or not (
+            use_mxfp4_aiter_moe() and self.ocp_mx_scheme == "w_mxfp4_a_mxfp4"
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27029 - [Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic

- 链接: https://github.com/vllm-project/vllm/pull/27029
- 状态/时间: merged / 2025-10-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `e33ee23ee3cd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+14/-3，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Bug fix the incorrect initialization of Quant Config in Quark MoE and the missing PTPC quant type logic in Fused MoE and the w8a8_utils.py incorrect condition checks. Evaluate Q...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -341,7 +341,8 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply，涉及 `get_fused_moe_quant_config, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -341,7 +341,8 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -341,7 +341,8 @@ def get_fused_moe_quant_config(
-            per_act_token_quant=self.weight_qscheme == "per_channel",
+            per_act_token_quant=self.input_qscheme == "per_channel",
+            per_out_ch_quant=self.weight_qscheme == "per_channel",
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/utils/w8a8_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27123 - [Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses.

- 链接: https://github.com/vllm-project/vllm/pull/27123
- 状态/时间: merged / 2025-11-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `938772af03ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+271/-311，可读 patch 1049 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses.」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Make a new `FusedMoEModularMethod` subclass of `FusedMoeMethodBase` for use with modular kernels. Instead of having every subclass of `FusedMoEMethodBase` check `self.fused_expe...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28 (53 lines); hunks: -310,7 +310,6 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; -322,17 +321,11 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply，涉及 `process_weights_after_loading, get_fused_moe_quant_config, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28 (53 lines); hunks: -310,7 +310,6 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; -322,17 +321,11 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -310,7 +310,6 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-                rocm_aiter_fused_experts,
@@ -322,17 +321,11 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-            self.rocm_aiter_fused_experts_func = rocm_aiter_fused_experts
-            self.fused_experts_func = None
-        else:
-            from vllm.model_executor.layers.fused_moe import fused_experts
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28
- 验证与风险: runtime 路径改动集中在 `vllm/distributed/device_communicators/base_device_communicator.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28322 - [CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness

- 链接: https://github.com/vllm-project/vllm/pull/28322
- 状态/时间: merged / 2025-11-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_mixtral.py`；关联提交 `6b2b9fd934af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-11，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `tests/lora/test_mixtral.py`；PR 正文摘要: Distributed A100 tests fail the nightly due to `lora/test_mixtral.py` link This PR attempts to fix that test. Running the test locally, I found it to be flaky. However, the outp...。
- 实现要点: `tests/lora/test_mixtral.py` modified +18/-11 (29 lines); hunks: -56,15 +56,22 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; symbols: test_mixtral_lora, check_outputs，涉及 `test_mixtral_lora, check_outputs`。
- 代码 diff 细节:
  - `tests/lora/test_mixtral.py` modified +18/-11 (29 lines); hunks: -56,15 +56,22 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; symbols: test_mixtral_lora, check_outputs
- 关键代码摘录:

```diff
diff -- tests/lora/test_mixtral.py
@@ -56,15 +56,22 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):
-        "give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])",  # noqa: E501
-        "give_opinion(name[SpellForce 3], developer[Grimlore Games], release_year[2017], rating[poor])",  # noqa: E501
-        "inform(name[BioShock], release_year[2007], rating[good], genres[action-adventure, role-playing, shooter], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_l
+        [
+            "give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])"  # noqa: E501
+        ],
```

- 已读文件:
  - tests: `tests/lora/test_mixtral.py` modified +18/-11
- 验证与风险: diff 自带测试面 `tests/lora/test_mixtral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27474 - [Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model

- 链接: https://github.com/vllm-project/vllm/pull/27474
- 状态/时间: merged / 2025-11-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `b06b9470ca88`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: doing view for moe weight to torch.float4_e2m1fn_x2 for aiter FP4 fused moe kernel with this PR, aiter cannot find the suitable kernel for the weight, whose type is uint8 with t...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0 (12 lines); hunks: -458,6 +458,7 @@ def __init__(; -581,6 +582,17 @@ def process_weights_after_loading(self, layer):; symbols: __init__, process_weights_after_loading, get_fused_moe_quant_config，涉及 `__init__, process_weights_after_loading, get_fused_moe_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0 (12 lines); hunks: -458,6 +458,7 @@ def __init__(; -581,6 +582,17 @@ def process_weights_after_loading(self, layer):; symbols: __init__, process_weights_after_loading, get_fused_moe_quant_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -458,6 +458,7 @@ def __init__(
+        self.fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
@@ -581,6 +582,17 @@ def process_weights_after_loading(self, layer):
+        if self.fp4_dtype is not None:
+            layer.w13_weight = torch.nn.Parameter(
+                layer.w13_weight.view(self.fp4_dtype),
+                requires_grad=layer.w13_weight.requires_grad,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24239 - [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model

- 链接: https://github.com/vllm-project/vllm/pull/24239
- 状态/时间: merged / 2025-11-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/features/quantization/quark.md`, `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `05576df85c52`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+127/-8，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `docs/features/quantization/quark.md`；PR 正文摘要: This PR aims to support _**layerwise**_ mixed-precision quantization model inference, extending from quantized models in single scheme such as MXFP4, FP8 (aka PTQ models). Here,...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7 (32 lines); hunks: -114,7 +114,14 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":; -124,10 +131,15 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConf...; symbols: from_config, _find_matched_config, _matches_pattern，涉及 `from_config, _find_matched_config, _matches_pattern`；`docs/features/quantization/quark.md` modified +33/-1 (34 lines); hunks: -281,4 +281,36 @@ python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B...。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7 (32 lines); hunks: -114,7 +114,14 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":; -124,10 +131,15 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConf...; symbols: from_config, _find_matched_config, _matches_pattern
  - `docs/features/quantization/quark.md` modified +33/-1 (34 lines); hunks: -281,4 +281,36 @@ python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B...
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -114,7 +114,14 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
-            if not kv_cache_set.issubset(layer_quant_set):
+            if not (
+                kv_cache_set.issubset(layer_quant_set)
+                or any(
+                    fnmatch.fnmatchcase(layer_quant, pat)
+                    for layer_quant in list(layer_quant_set)
diff -- docs/features/quantization/quark.md
@@ -281,4 +281,36 @@ python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B-Chat \
-The current integration supports [all combination of FP4, FP6_E3M2, FP6_E2M3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/ocp_mx_u
+The current integration supports [all combination of FP4, FP6_E3M2, FP6_E2M3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/ocp_mx_u
+## Using Quark Quantized layerwise Auto Mixed Precision (AMP) Models
+vLLM also supports loading layerwise mixed precision model quantized using AMD Quark. Currently, mixed scheme of {MXFP4, FP8} is supported, where FP8 here denotes for FP8 per-tens
+- Unquantized Linear and/or MoE layer(s) as an option for each layer, i.e., mixed of {MXFP4, FP8, BF16/FP16}
+- MXFP6 quantization extension, i.e., {MXFP4, MXFP6, FP8, BF16/FP16}
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7
  - docs: `docs/features/quantization/quark.md` modified +33/-1
- 验证与风险: diff 自带测试面 `tests/quantization/test_mixed_precision.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28275 - [Misc] add ignore mapper for quark quantization

- 链接: https://github.com/vllm-project/vllm/pull/28275
- 状态/时间: merged / 2025-11-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `0b25498990f0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-3，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] add ignore mapper for quark quantization」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: Qwen3vlmoe has the following mapping, and quark format requires a function to complete this mapping. orig_to_new_prefix={ "model.visual.": "visual.", "lm_head.": "language_model...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3 (12 lines); hunks: -2,7 +2,7; -34,6 +34,9; symbols: __init__, get_linear_method, get_quant_method, apply_vllm_mapper，涉及 `__init__, get_linear_method, get_quant_method`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3 (12 lines); hunks: -2,7 +2,7; -34,6 +34,9; symbols: __init__, get_linear_method, get_quant_method, apply_vllm_mapper
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -2,7 +2,7 @@
-from typing import Any, Optional, cast
+from typing import TYPE_CHECKING, Any, Optional, cast
@@ -34,6 +34,9 @@
+if TYPE_CHECKING:
+    from vllm.model_executor.models.utils import WeightsMapper
@@ -54,6 +57,7 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28638 - [ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss

- 链接: https://github.com/vllm-project/vllm/pull/28638
- 状态/时间: merged / 2025-11-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `d0a73620cc85`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+30/-5，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: Need to call https://github.com/vllm-project/vllm/blob/c9fe6abe7c0b03d552420edd63c6c678ed683dea/vllm/model_executor/models/gpt_oss.py#L656 to build a map, such as `self_attn` to...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5 (35 lines); hunks: -32,6 +32,7; -57,7 +58,6 @@ def __init__(; symbols: __init__, get_linear_method, get_min_capability, get_name，涉及 `__init__, get_linear_method, get_min_capability`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5 (35 lines); hunks: -32,6 +32,7; -57,7 +58,6 @@ def __init__(; symbols: __init__, get_linear_method, get_min_capability, get_name
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -32,6 +32,7 @@
+from vllm.model_executor.models.utils import WeightsMapper
@@ -57,7 +58,6 @@ def __init__(
-        self.ignore: list[str] = cast(list[str], self.quant_config.get("exclude", []))
@@ -72,14 +72,42 @@ def get_min_capability(cls) -> int:
+    def apply_vllm_mapper(  # noqa: B027
+        self, hf_to_vllm_mapper: "WeightsMapper"
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29067 - [MoE][Refactor] Make select_experts a non-static method

- 链接: https://github.com/vllm-project/vllm/pull/29067
- 状态/时间: merged / 2025-11-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `8f066146c395`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+163/-472，可读 patch 1200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE][Refactor] Make select_experts a non-static method」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: There didn't appear to be a good reason why `select_experts` was a static method. Making it non-static allows access to the attributes of the `FusedMoE` layer so that we don't n...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34 (38 lines); hunks: -334,7 +334,7 @@ def get_fused_moe_quant_config(; -355,24 +355,9 @@ def apply(; symbols: get_fused_moe_quant_config, apply, allow_inplace，涉及 `get_fused_moe_quant_config, apply, allow_inplace`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34 (38 lines); hunks: -334,7 +334,7 @@ def get_fused_moe_quant_config(; -355,24 +355,9 @@ def apply(; symbols: get_fused_moe_quant_config, apply, allow_inplace
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -334,7 +334,7 @@ def get_fused_moe_quant_config(
-        layer: torch.nn.Module,
+        layer: FusedMoE,
@@ -355,24 +355,9 @@ def apply(
-        if enable_eplb:
-            raise NotImplementedError(
-                "EPLB not supported for `QuarkW8A8Fp8MoEMethod` yet."
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_flashinfer.py`, `tests/test_routing_simulator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29773 - [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel

- 链接: https://github.com/vllm-project/vllm/pull/29773
- 状态/时间: merged / 2025-12-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `c6df05ebb499`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+5/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Quark Fused Moe meets acc issue in DeepSeek-R1 Expert Parallelism. This PR fixes it through setting the right expert mask with format required by aiter kernel. The aiter kernel...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -633,6 +633,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -633,6 +633,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -633,6 +633,7 @@ def apply(
+                expert_map=expert_map,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29066 - [MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply

- 链接: https://github.com/vllm-project/vllm/pull/29066
- 状态/时间: merged / 2025-12-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `00e5cbb96789`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+318/-872，可读 patch 2081 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: After making `select_experts` a non-static method (https://github.com/vllm-project/vllm/pull/29067), we can avoid passing most of the arguments to `FusedMoEMethodBase.apply` and...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52 (72 lines); hunks: -1,7 +1,6; -337,23 +336,6 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52 (72 lines); hunks: -1,7 +1,6; -337,23 +336,6 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1,7 +1,6 @@
-from collections.abc import Callable
@@ -337,23 +336,6 @@ def apply(
-        top_k: int,
-        renormalize: bool,
-        use_grouped_topk: bool = False,
-        topk_group: int | None = None,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30071 - [Quantization] Support Quark int4-fp8 w4a8 for MoE

- 链接: https://github.com/vllm-project/vllm/pull/30071
- 状态/时间: merged / 2025-12-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `0c738b58bc0e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+201/-2，可读 patch 224 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Support Quark int4-fp8 w4a8 for MoE」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: This PR extends support of Quark quantized model for int4-fp8 w4a8 quantization spec. Specifically: - Weights are double quantized: first at INT4 per channel, then further at FP...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2 (160 lines); hunks: -63,8 +63,9 @@ def get_moe_method(; -396,6 +397,161 @@ def apply(; symbols: get_moe_method, apply, QuarkW4A8Fp8MoEMethod, __init__，涉及 `get_moe_method, apply, QuarkW4A8Fp8MoEMethod`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0 (43 lines); hunks: -218,6 +218,49 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8，涉及 `_check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2 (160 lines); hunks: -63,8 +63,9 @@ def get_moe_method(; -396,6 +397,161 @@ def apply(; symbols: get_moe_method, apply, QuarkW4A8Fp8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0 (43 lines); hunks: -218,6 +218,49 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -63,8 +63,9 @@ def get_moe_method(
-        if quant_config._is_fp8_w8a8(weight_config, input_config):
+        if quant_config._is_fp8_w4a8(weight_config, input_config):
+            return QuarkW4A8Fp8MoEMethod(weight_config, input_config, module.moe_config)
+        elif quant_config._is_fp8_w8a8(weight_config, input_config):
@@ -396,6 +397,161 @@ def apply(
+class QuarkW4A8Fp8MoEMethod(QuarkMoEMethod):
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -218,6 +218,49 @@ def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bo
+    def _is_fp8_w4a8(
+        self,
+        weight_quant: list[dict[str, Any]] | None,
+        input_quant: dict[str, Any] | None,
+    ) -> bool:
+        # Confirm weights and input quantized.
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2; `vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28891 - [MoE Refactor][5/N] Isolate zero expert to LongCatFlash

- 链接: https://github.com/vllm-project/vllm/pull/28891
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `54c892438479`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+263/-108，可读 patch 709 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor][5/N] Isolate zero expert to LongCatFlash」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: 28152 28745 Fix dimension mismatch error in LongCat Flash MoE model when using zero experts. The issue occurred when `router_logits` (including zero experts) and `e_score_correc...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -338,7 +338,7 @@ def apply(; -530,7 +530,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -338,7 +338,7 @@ def apply(; -530,7 +530,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -338,7 +338,7 @@ def apply(
-        topk_weights, topk_ids, _ = layer.select_experts(
+        topk_weights, topk_ids = layer.select_experts(
@@ -530,7 +530,7 @@ def apply(
-        topk_weights, topk_ids, _ = layer.select_experts(
+        topk_weights, topk_ids = layer.select_experts(
@@ -738,7 +738,7 @@ def apply(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/test_routing_simulator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31499 - [MoE Refactor][12/N] Marlin Fp8 MoE Pure Function

- 链接: https://github.com/vllm-project/vllm/pull/31499
- 状态/时间: merged / 2025-12-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `9152a30d8f6f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+92/-76，可读 patch 270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor][12/N] Marlin Fp8 MoE Pure Function」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: SUMMARY: * make `prepare_moe_fp8_layer_for_marlin` into a pure function for consistency with other methods and compatibility with the `replace_parameter` semantics that are need...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4 (24 lines); hunks: -315,10 +315,26 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config，涉及 `process_weights_after_loading, get_fused_moe_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4 (24 lines); hunks: -315,10 +315,26 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -315,10 +315,26 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-            prepare_moe_fp8_layer_for_marlin(layer, False)
-            # Activations not quantized for marlin.
-            del layer.w13_input_scale
-            del layer.w2_input_scale
+            (workspace, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale) = (
+                prepare_moe_fp8_layer_for_marlin(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31759 - [MoE Refactor] Add Temporary Integration Tests - H100/B200

- 链接: https://github.com/vllm-project/vllm/pull/31759
- 状态/时间: merged / 2026-01-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml`；关联提交 `d3e477c01342`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 30 个文件，+247/-0，可读 patch 280 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Add Temporary Integration Tests - H100/B200」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 文档/测试/CI；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml`；PR 正文摘要: * add ci job to validate MoE refactor * this is very compute intensive and duplicative, so this is a temporary job that I will run on my PRs。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9；`tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml
@@ -0,0 +1,9 @@
+# TODO(rob): enable
+# model_name: "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV"
+# accuracy_threshold: 0.62
+# num_questions: 1319
+# num_fewshot: 5
+# server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml
@@ -0,0 +1,5 @@
+model_name: "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV"
+accuracy_threshold: 0.62
+num_questions: 1319
+num_fewshot: 5
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-marlin.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31415 - [MoE Refactor][15/N] Apply Refactor to Fp8

- 链接: https://github.com/vllm-project/vllm/pull/31415
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `5dcd7ef1f219`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+1446/-1535，可读 patch 4055 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor][15/N] Apply Refactor to Fp8」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: SUMMARY: * continue refactoring fp8 moe * CUTLASS MoE --> make CutlassExpertsFp8 manage the strides data * CUTLASS MoE --> remove cutlass_moe_fp8 entrypoint * factor out MoE ker...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4 (7 lines); hunks: -22,7 +22,7; -315,16 +315,15 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4 (7 lines); hunks: -22,7 +22,7; -315,16 +315,15 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -22,7 +22,7 @@
-    prepare_moe_fp8_layer_for_marlin,
+    prepare_fp8_moe_layer_for_marlin,
@@ -315,16 +315,15 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-            (workspace, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale) = (
-                prepare_moe_fp8_layer_for_marlin(
+            w13_weight, w2_weight, w13_weight_scale, w2_weight_scale = (
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm-deepep-ht.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm-deepep-ll.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30519 - [Misc][Refactor] Add FusedMoERouter object

- 链接: https://github.com/vllm-project/vllm/pull/30519
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `e74698c27ad5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+165/-36，可读 patch 744 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Refactor] Add FusedMoERouter object」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Add abstract `FusedMoERouter` class that provides `select_experts`. There's a concrete subclass that wraps `FusedMoE._select_experts` that is passed to all the quantization meth...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2 (8 lines); hunks: -13,6 +13,7; -350,10 +351,11 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, allow_inplace，涉及 `get_fused_moe_quant_config, apply, allow_inplace`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2 (8 lines); hunks: -13,6 +13,7; -350,10 +351,11 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, allow_inplace
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -13,6 +13,7 @@
+    FusedMoERouter,
@@ -350,10 +351,11 @@ def get_fused_moe_quant_config(
+        router: FusedMoERouter,
-        topk_weights, topk_ids = layer.select_experts(
+        topk_weights, topk_ids = router.select_experts(
@@ -542,6 +544,7 @@ def get_fused_moe_quant_config(self, layer):
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2
- 验证与风险: diff 自带测试面 `tests/test_routing_simulator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31827 - [MoE Refactor][17/N] Apply Refactor to Bf16

- 链接: https://github.com/vllm-project/vllm/pull/31827
- 状态/时间: merged / 2026-01-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml`；关联提交 `31c29257c852`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+257/-87，可读 patch 445 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor][17/N] Apply Refactor to Bf16」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml`；PR 正文摘要: gsm8k result for Triton kernel, flashinfer_cutlass kernel and aiter rocm kernel for `Qwen/Qwen3-30B-A3B` in TP(triton), TEP (flashinfer cutlass) and rocm. Triton Flashinfer。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0 (7 lines); hunks: -0,0 +1,7；`tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0 (7 lines); hunks: -0,0 +1,7
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml
@@ -0,0 +1,7 @@
+model_name: "mistralai/Mixtral-8x7B-v0.1"
+accuracy_threshold: 0.58
+num_questions: 1319
+num_fewshot: 5
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel"
+env:
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml
@@ -0,0 +1,5 @@
+model_name: "mistralai/Mixtral-8x7B-v0.1"
+accuracy_threshold: 0.58
+num_questions: 1319
+num_fewshot: 5
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-BF16-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/config-b200.txt`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-triton.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30623 - [MoE Refactor] Separate Router into OO Classes

- 链接: https://github.com/vllm-project/vllm/pull/30623
- 状态/时间: merged / 2026-01-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `327a02d8db86`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 45 个文件，+1754/-688，可读 patch 2998 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Separate Router into OO Classes」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This PR is part of the effort to separate the expert selection code from `FusedMoEMethod`. Move all the MoE router implementations into separate subclasses of `FusedMoERouter`....。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -548,7 +548,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -548,7 +548,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -548,7 +548,7 @@ def apply(
-        topk_weights, topk_ids = layer.select_experts(
+        topk_weights, topk_ids = router.select_experts(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_cutlass_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27814 - [Refactor] Make FP8 Linear Ops use kernel abstraction

- 链接: https://github.com/vllm-project/vllm/pull/27814
- 状态/时间: merged / 2026-01-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`；关联提交 `148117ea2e68`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 30 个文件，+1458/-1029，可读 patch 3752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Make FP8 Linear Ops use kernel abstraction」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`；PR 正文摘要: This PR refactors the FP8 linear kernel integration in vLLM to improve code clarity, maintainability, and path consistency. Changes: 1- A more generic ScaledMMLinearKernel inter...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19 (47 lines); hunks: -7,10 +7,18; -23,6 +31,8; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading，涉及 `QuarkW8A8Fp8, __init__, process_weights_after_loading`；`vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22 (32 lines); hunks: -7,8 +7,7; -22,8 +21,6; symbols: QuarkW8A8Int8, __init__, create_weights，涉及 `QuarkW8A8Int8, __init__, create_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19 (47 lines); hunks: -7,10 +7,18; -23,6 +31,8; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22 (32 lines); hunks: -7,8 +7,7; -22,8 +21,6; symbols: QuarkW8A8Int8, __init__, create_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -7,10 +7,18 @@
+from vllm.logger import init_logger
+from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
+    init_fp8_linear_kernel,
+)
-from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py
@@ -7,8 +7,7 @@
-    ScaledMMLinearLayerConfig,
-    choose_scaled_mm_linear_kernel,
+    init_int8_linear_kernel,
@@ -22,8 +21,6 @@
-    _kernel_backends_being_used: set[str] = set()
@@ -50,18 +47,13 @@ def create_weights(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22
- 验证与风险: diff 自带测试面 `tests/compile/distributed/test_fusion_all_reduce.py`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`, `tests/compile/test_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31996 - [MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE`

- 链接: https://github.com/vllm-project/vllm/pull/31996
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `dc917cceb877`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+495/-530，可读 patch 1822 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE`」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Move (almost all) the `select_experts` calls to the `FusedMoE` layer instead of each `FusedMoEMethodBase.apply` method. Change the `apply` signature to take `topk_weights` and `...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23 (29 lines); hunks: -13,7 +13,6; -351,15 +350,10 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply，涉及 `get_fused_moe_quant_config, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23 (29 lines); hunks: -13,7 +13,6; -351,15 +350,10 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -13,7 +13,6 @@
-    FusedMoERouter,
@@ -351,15 +350,10 @@ def get_fused_moe_quant_config(
-        router: FusedMoERouter,
-        router_logits: torch.Tensor,
+        topk_weights: torch.Tensor,
+        topk_ids: torch.Tensor,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32954 - [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe

- 链接: https://github.com/vllm-project/vllm/pull/32954
- 状态/时间: merged / 2026-01-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+290/-17，可读 patch 452 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`；PR 正文摘要: - Integrate flashinfer trtllm-gen BF16 moe to supported models - This is a rebased version of PR 28238 by @jiahanc and includes adaptation to the latest moe refactoring changes....。
- 实现要点: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0 (103 lines); hunks: -77,6 +77,18 @@ def _supports_routing_method(; -115,6 +127,34 @@ def _make_reason(reason: str) -> str:; symbols: _supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config, _make_reason，涉及 `_supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config`；`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11 (85 lines); hunks: -6,6 +6,7; -32,6 +33,9; symbols: UnquantizedFusedMoEMethod, __init__, _select_monolithic, forward_native，涉及 `UnquantizedFusedMoEMethod, __init__, _select_monolithic`；`vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0 (75 lines); hunks: -214,6 +214,81 @@ def is_flashinfer_supporting_global_sf(backend: FlashinferM...; symbols: is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi，涉及 `is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi`；`vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6 (41 lines); hunks: -14,19 +14,23; -40,6 +44,7 @@ class UnquantizedMoeBackend(Enum):; symbols: UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend，涉及 `UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0 (103 lines); hunks: -77,6 +77,18 @@ def _supports_routing_method(; -115,6 +127,34 @@ def _make_reason(reason: str) -> str:; symbols: _supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config, _make_reason
  - `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11 (85 lines); hunks: -6,6 +6,7; -32,6 +33,9; symbols: UnquantizedFusedMoEMethod, __init__, _select_monolithic, forward_native
  - `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0 (75 lines); hunks: -214,6 +214,81 @@ def is_flashinfer_supporting_global_sf(backend: FlashinferM...; symbols: is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi
  - `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6 (41 lines); hunks: -14,19 +14,23; -40,6 +44,7 @@ class UnquantizedMoeBackend(Enum):; symbols: UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend
  - `vllm/utils/flashinfer.py` modified +3/-0 (3 lines); hunks: -105,6 +105,9 @@ def wrapper(*args, **kwargs):; symbols: wrapper
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py
@@ -77,6 +77,18 @@ def _supports_routing_method(
+def _supports_routing_method_bf16(
+    routing_method: RoutingMethodType,
+) -> bool:
+    return routing_method in [
+        RoutingMethodType.Default,
+        RoutingMethodType.Renormalize,
diff -- vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py
@@ -6,6 +6,7 @@
+from torch.nn.parameter import Parameter
@@ -32,6 +33,9 @@
+from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
+    convert_moe_weights_to_flashinfer_trtllm_block_layout,
+)
@@ -54,6 +58,7 @@ class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
diff -- vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
@@ -214,6 +214,81 @@ def is_flashinfer_supporting_global_sf(backend: FlashinferMoeBackend | None) ->
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0; `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11; `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6; `vllm/utils/flashinfer.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33375 - [Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor

- 链接: https://github.com/vllm-project/vllm/pull/33375
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `a57c8228ffb3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+132/-109，可读 patch 869 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Pass the inplace flag to the FusedMoEModularKernel constructor. Knowing the inplace (or being able to disable) behavior ahead of time will help simplify some of the runtime logi...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6 (9 lines); hunks: -388,6 +388,7 @@ def apply(; -398,7 +399,7 @@ def apply(; symbols: apply, get_fused_moe_quant_config, allow_inplace，涉及 `apply, get_fused_moe_quant_config, allow_inplace`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6 (9 lines); hunks: -388,6 +388,7 @@ def apply(; -398,7 +399,7 @@ def apply(; symbols: apply, get_fused_moe_quant_config, allow_inplace
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -388,6 +388,7 @@ def apply(
+                inplace=not self.moe.disable_inplace,
@@ -398,7 +399,7 @@ def apply(
-                inplace=True,
+                inplace=not self.moe.disable_inplace,
@@ -734,10 +735,6 @@ def get_fused_moe_quant_config(
-    @property
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_cutlass_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29008 - [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations

- 链接: https://github.com/vllm-project/vllm/pull/29008
- 状态/时间: merged / 2026-02-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `b129136c7a73`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1094/-213，可读 patch 1860 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: This PR aims for: - Models: - Quantization schemes: - TP: See results below. (Sub)-tasks。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54 (352 lines); hunks: -8,6 +8,7; -18,9 +19,15; symbols: QuarkMoEMethod, __init__, get_moe_method，涉及 `QuarkMoEMethod, __init__, get_moe_method`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23 (70 lines); hunks: -320,38 +320,45 @@ def _is_static_tensor_w8a8(; -360,14 +367,31 @@ def _is_ocp_mx(; symbols: _is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x, is_mxfp4_quant，涉及 `_is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54 (352 lines); hunks: -8,6 +8,7; -18,9 +19,15; symbols: QuarkMoEMethod, __init__, get_moe_method
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23 (70 lines); hunks: -320,38 +320,45 @@ def _is_static_tensor_w8a8(; -360,14 +367,31 @@ def _is_ocp_mx(; symbols: _is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x, is_mxfp4_quant
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -8,6 +8,7 @@
+from vllm.config import get_current_vllm_config
@@ -18,9 +19,15 @@
+    mxfp4_w4a8_moe_quant_config,
+    mxfp4_w4a16_moe_quant_config,
+from vllm.model_executor.layers.quantization.mxfp4 import (
+    Mxfp4Backend,
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -320,38 +320,45 @@ def _is_static_tensor_w8a8(
-    def _is_ocp_mx(
-        self,
-        weight_quant: dict[str, Any] | None,
-        input_quant: dict[str, Any] | None,
+    def _is_w_ocp_mx_a_x(
+        self, weight_quant: dict[str, Any] | None, input_quant: dict[str, Any] | None
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54; `vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/models/quantization/test_gpt_oss_attn_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32344 - [MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner

- 链接: https://github.com/vllm-project/vllm/pull/32344
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `d1481ba78323`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 25 个文件，+913/-753，可读 patch 2246 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: The execution logic in `FusedMoE` is rather convoluted due to the number of inter-dependent features, e.g. shared experts, gating, modular vs. non-modular kernels, logic for par...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0 (3 lines); hunks: -419,6 +419,7 @@ def apply(; -607,6 +608,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0 (3 lines); hunks: -419,6 +419,7 @@ def apply(; -607,6 +608,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -419,6 +419,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
@@ -607,6 +608,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
@@ -977,6 +979,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33715 - [NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE

- 链接: https://github.com/vllm-project/vllm/pull/33715
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`；关联提交 `275e0d2a993b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+296/-1，可读 patch 337 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`；PR 正文摘要: Adding tests for trtllm bf16 moe backend added in PR [[NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe #32954](https://github.com/vllm-project/vllm/pull/32954) - unit and...。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0 (1 lines); hunks: -5,3 +5,4 @@ num_fewshot: 5。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0 (1 lines); hunks: -5,3 +5,4 @@ num_fewshot: 5
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml
@@ -5,3 +5,4 @@ num_fewshot: 5
+  VLLM_FLASHINFER_MOE_BACKEND: "throughput"
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/kernels/moe/test_flashinfer.py`, `tests/kernels/moe/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33843 - [Refactor] Replace `activation: str` with `MoEActivation` enum

- 链接: https://github.com/vllm-project/vllm/pull/33843
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `ff1f83b056ae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 48 个文件，+474/-282，可读 patch 2289 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Replace `activation: str` with `MoEActivation` enum」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: We have had `activation` defined, validated, and passed around as a raw string forever. Now that we have popular models that don't just use the `silu` default and also need to s...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -15,6 +15,7; -438,7 +439,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -15,6 +15,7; -438,7 +439,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -15,6 +15,7 @@
+    MoEActivation,
@@ -438,7 +439,7 @@ def apply(
-            assert layer.activation == "silu", (
+            assert layer.activation == MoEActivation.SILU, (
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_cpu_fused_moe.py`, `tests/kernels/moe/test_cutlass_moe.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34192 - [ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter

- 链接: https://github.com/vllm-project/vllm/pull/34192
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `8a798be929d6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This commit introduces hardware-specific optimization for MXFP4 quantized Mixture-of-Experts (MoE) layers on AMD gfx950 GPUs and updates the aiter dependency to support this fun...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0 (8 lines); hunks: -933,7 +933,15 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, get_fused_moe_quant_config，涉及 `process_weights_after_loading, get_fused_moe_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0 (8 lines); hunks: -933,7 +933,15 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, get_fused_moe_quant_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -933,7 +933,15 @@ def process_weights_after_loading(self, layer):
+        # Pre-shuffle weight
+        shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
+            layer.w13_weight.data, layer.w2_weight.data
+        )
+        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
+        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34386 - [Quark] Fix MoE fp8 activation scale handling on mi300

- 链接: https://github.com/vllm-project/vllm/pull/34386
- 状态/时间: merged / 2026-02-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `d9e62c03eb98`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quark] Fix MoE fp8 activation scale handling on mi300」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Small fix follow-up after #29008 for running gpt-oss on mi300. Also ensure the fp8 scale conversion only runs when activation is fp8 quantized.。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -858,7 +858,7 @@ def create_weights(; -883,14 +883,14 @@ def process_weights_after_loading(self, layer):; symbols: create_weights, process_weights_after_loading，涉及 `create_weights, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -858,7 +858,7 @@ def create_weights(; -883,14 +883,14 @@ def process_weights_after_loading(self, layer):; symbols: create_weights, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -858,7 +858,7 @@ def create_weights(
-        if self.static_input_scales:
+        if self.static_input_scales and self.input_dtype == "fp8":
@@ -883,14 +883,14 @@ def process_weights_after_loading(self, layer):
-                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fnuz),
+                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fn),
-                    torch.empty_like(layer.w2_weight, dtype=torch.float8_e4m3fnuz),
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33807 - [UX] Add `--moe-backend` arg for explicit kernel selection

- 链接: https://github.com/vllm-project/vllm/pull/33807
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`；关联提交 `de527e1cec82`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 37 个文件，+260/-140，可读 patch 720 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[UX] Add `--moe-backend` arg for explicit kernel selection」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`；PR 正文摘要: Adds `--moe-backend` argument for explicit MoE kernel selection, allowing users to override the automatic backend selection logic (e.g., `--moe-backend triton`, `--moe-backend m...。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -2,7 +2,4 @@ model_name: "mistralai/Mixtral-8x7B-v0.1"；`tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -3,7 +3,4。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -2,7 +2,4 @@ model_name: "mistralai/Mixtral-8x7B-v0.1"
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -3,7 +3,4
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml
@@ -2,7 +2,4 @@ model_name: "mistralai/Mixtral-8x7B-v0.1"
-server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel"
-env:
-  VLLM_USE_FLASHINFER_MOE_FP16: "1"
-  VLLM_FLASHINFER_MOE_BACKEND: "throughput"
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel --moe-backend=flashinfer_cutlass"
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml
@@ -3,7 +3,4 @@
-# server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
-# env:
-#   VLLM_USE_FLASHINFER_MOE_FP8: "1"
-#   VLLM_FLASHINFER_MOE_BACKEND: "throughput"
+# server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2 --moe-backend=flashinfer_cutlass"
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Qwen3-Next-80B-A3B-NVFP4-EP2.yaml`, `tests/evals/gsm8k/configs/Qwen3-Next-FP8-EP2.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-NvFp4-CT-fi-cutedsl-deepep-ll.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30357 - [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales

- 链接: https://github.com/vllm-project/vllm/pull/30357
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `01923eec7092`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+315/-37，可读 patch 494 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: gpt-oss120b-w-mxfp4-a-fp8 server: > HIP_VISIBLE_DEVICES=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 vl...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29 (202 lines); hunks: -5,8 +5,8; -32,6 +32,7; symbols: QuarkMoEMethod, get_moe_method, __init__, create_weights，涉及 `QuarkMoEMethod, get_moe_method, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29 (202 lines); hunks: -5,8 +5,8; -32,6 +32,7; symbols: QuarkMoEMethod, get_moe_method, __init__, create_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -5,8 +5,8 @@
-import vllm.envs as envs
+from vllm import envs
@@ -32,6 +32,7 @@
+from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
@@ -49,7 +50,11 @@
-__all__ = ["QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod", "QuarkOCP_MX_MoEMethod"]
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35658 - [ROCm] add amd-quark package in requirements for rocm to use quantized models

- 链接: https://github.com/vllm-project/vllm/pull/35658
- 状态/时间: merged / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`；关联提交 `f26650d649aa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+24/-6，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] add amd-quark package in requirements for rocm to use quantized models」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `tests/quantization/test_quark.py`；PR 正文摘要: Fix https://github.com/vllm-project/vllm/issues/35633 - Added amd-quark to requirements/rocm.txt. This way, it can be picked up for building docker, wheel or building from sourc...。
- 实现要点: `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark，涉及 `get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness`。
- 代码 diff 细节:
  - `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark
- 关键代码摘录:

```diff
diff -- tests/quantization/test_quark.py
@@ -26,9 +26,12 @@
+# Minimum amd-quark version for MXFP4/OCP_MX tests (single source of truth).
+QUARK_MXFP4_MIN_VERSION = "0.8.99"
-) >= version.parse("0.8.99")
+) >= version.parse(QUARK_MXFP4_MIN_VERSION)
@@ -200,7 +203,10 @@ def get_model_args(
-@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
```

- 已读文件:
  - tests: `tests/quantization/test_quark.py` modified +20/-5
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35893 - [ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported

- 链接: https://github.com/vllm-project/vllm/pull/35893
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `5dc3538736e4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+63/-1，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Fix a `RuntimeError` crash when using AITER fused MoE with MXFP4 quantization on certain model/TP configurations where CK's pre-compiled GEMM kernel instances don't support the...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1 (31 lines); hunks: -32,7 +32,10; -732,6 +735,32 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1 (31 lines); hunks: -32,7 +32,10; -732,6 +735,32 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -32,7 +32,10 @@
-from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
+from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
+    CK_MXFP4_MOE_DIM_ALIGNMENT,
+    _swizzle_mxfp4,
+)
@@ -732,6 +735,32 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35316 - [ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer

- 链接: https://github.com/vllm-project/vllm/pull/35316
- 状态/时间: merged / 2026-03-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`；关联提交 `6341d4304351`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+311/-1，可读 patch 362 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`；PR 正文摘要: weights: mxfp4 with static scales activations: fp8 1. Supports Eager + Torch.compile mode 2. Supports Emulation mode 3. Updated test results below. ~~Tested on this script for q...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: QuarkW4A8_MXFP4_FP8, __init__, get_min_capability, get_packed_dim，涉及 `QuarkW4A8_MXFP4_FP8, __init__, get_min_capability`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0 (32 lines); hunks: -26,6 +26,7; -350,6 +351,31 @@ def _is_static_tensor_w8a8(; symbols: _is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x, _get_scheme_from_config，涉及 `_is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x`；`vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1 (9 lines); hunks: -3,7 +3,14。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: QuarkW4A8_MXFP4_FP8, __init__, get_min_capability, get_packed_dim
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0 (32 lines); hunks: -26,6 +26,7; -350,6 +351,31 @@ def _is_static_tensor_w8a8(; symbols: _is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x, _get_scheme_from_config
  - `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1 (9 lines); hunks: -3,7 +3,14
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py
@@ -0,0 +1,218 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable
+from fractions import Fraction
+from typing import Any
+import torch
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -26,6 +26,7 @@
+    QuarkW4A8_MXFP4_FP8,
@@ -350,6 +351,31 @@ def _is_static_tensor_w8a8(
+    def _is_w4a8_mxfp4_fp8(
+        self,
+        weight_quant: dict[str, Any] | None,
+        input_quant: dict[str, Any] | None,
diff -- vllm/model_executor/layers/quantization/quark/schemes/__init__.py
@@ -3,7 +3,14 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0; `vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32779 - Fix infinite recursive search issue in quark.py

- 链接: https://github.com/vllm-project/vllm/pull/32779
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark.py`；关联提交 `24b4272a8ca6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-4，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix infinite recursive search issue in quark.py」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark.py`；PR 正文摘要: Fix infinite recursive search issue in _find_matched_config() when test granite 4 small fp8 model on rocm/vllm-dev:nightly docker, the issue fix will be generic for other enviro...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4 (15 lines); hunks: -467,10 +467,17 @@ def _find_matched_config(; symbols: _find_matched_config，涉及 `_find_matched_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4 (15 lines); hunks: -467,10 +467,17 @@ def _find_matched_config(; symbols: _find_matched_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -467,10 +467,17 @@ def _find_matched_config(
-            shard_configs = [
-                self._find_matched_config(shard_name, module)
-                for shard_name in shard_names
-            ]
+            shard_configs = []
+            for shard_name in shard_names:
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36232 - [ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization

- 链接: https://github.com/vllm-project/vllm/pull/36232
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`；关联提交 `44eea10f6846`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+19/-5，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This PR aims to improve the robustness of Quark's weight-only OCP MX quantization. Changes here are to fix errors due to None type of `input_quant_spec`. Test Plan & Result - Qw...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4 (21 lines); hunks: -176,7 +176,7 @@ class QuarkOCP_MX(QuarkScheme):; -185,7 +185,13 @@ def __init__(; symbols: QuarkOCP_MX, __init__，涉及 `QuarkOCP_MX, __init__`；`vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def get_moe_method(; symbols: get_moe_method，涉及 `get_moe_method`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4 (21 lines); hunks: -176,7 +176,7 @@ class QuarkOCP_MX(QuarkScheme):; -185,7 +185,13 @@ def __init__(; symbols: QuarkOCP_MX, __init__
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def get_moe_method(; symbols: get_moe_method
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py
@@ -176,7 +176,7 @@ class QuarkOCP_MX(QuarkScheme):
-        input_quant_spec: dict[str, Any],
+        input_quant_spec: dict[str, Any] | None,
@@ -185,7 +185,13 @@ def __init__(
-        self.input_dtype = input_quant_spec["dtype"].replace("fp", "mxfp")
+        self.input_dtype: str | None = None
+        if input_quant_spec is not None:
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -92,7 +92,8 @@ def get_moe_method(
-                input_config.get("dtype") == "fp8_e4m3"
+                input_config is not None
+                and input_config.get("dtype") == "fp8_e4m3"
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37128 - [MoE Refactor] Mxfp4 oracle rebased

- 链接: https://github.com/vllm-project/vllm/pull/37128
- 状态/时间: merged / 2026-03-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `87bd91892f8c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+1695/-1369，可读 patch 3508 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Mxfp4 oracle rebased」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Rebased and improve version of #34983 Ongoing MXFP4 MoE refactor - Refactor MXFP4 MoE from a monolithic 1299-line `Mxfp4MoEMethod` class to the oracle pattern used by FP8 and Nv...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -25,9 +25,9; -698,9 +698,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -25,9 +25,9; -698,9 +698,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -25,9 +25,9 @@
-from vllm.model_executor.layers.quantization.mxfp4 import (
-    Mxfp4Backend,
-    get_mxfp4_backend,
+from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
+    Mxfp4MoeBackend,
+    select_mxfp4_moe_backend,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5
- 验证与风险: diff 自带测试面 `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_ocp_mx_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36100 - [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs

- 链接: https://github.com/vllm-project/vllm/pull/36100
- 状态/时间: merged / 2026-03-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`；关联提交 `e99fb98867c2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+16/-26，可读 patch 122 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Fix a real logic bug and several related issues in ROCm AITER ops: Main bugfix: `_rocm_aiter_fused_moe_fake` signature mismatch The fake/meta implementation for the `rocm_aiter_...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20 (25 lines); hunks: -3,13 +3,12; -37,22 +36,6; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__，涉及 `is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__`；`vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -765,7 +765,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20 (25 lines); hunks: -3,13 +3,12; -37,22 +36,6; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -765,7 +765,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py
@@ -3,13 +3,12 @@
-from functools import cache, partial
+from functools import partial
-from vllm import envs
@@ -37,22 +36,6 @@
-# TODO: move registration of custom op to aiter_ops.py
-# `from vllm._aiter_ops import rocm_aiter_ops`
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -765,7 +765,7 @@ def __init__(
-                f"use_mxfp4_aiter_moe={self.use_rocm_aiter_moe}, "
+                f"use_rocm_aiter_moe={self.use_rocm_aiter_moe}, "
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37833 - [ROCm] Fix MoE kernel test failures on gfx950

- 链接: https://github.com/vllm-project/vllm/pull/37833
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `7d6917bef552`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+480/-88，可读 patch 928 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Fix MoE kernel test failures on gfx950」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Fixes ROCm-specific MoE kernel test failures on MI355 (gfx950). - `gpt_oss_triton_kernels_moe.py`: Handle installed `triton_kernels.topk` returning tuple instead of `SparseMatri...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0 (15 lines); hunks: -741,11 +741,14 @@ def __init__(; -819,6 +822,18 @@ def create_weights(; symbols: __init__, create_weights，涉及 `__init__, create_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0 (15 lines); hunks: -741,11 +741,14 @@ def __init__(; -819,6 +822,18 @@ def create_weights(; symbols: __init__, create_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -741,11 +741,14 @@ def __init__(
+        # For gpt_oss models, create_weights rounds up the dimensions
+        # internally, so the alignment check is skipped.
+            and self.model_type != "gpt_oss"
@@ -819,6 +822,18 @@ def create_weights(
+        # On GFX950, the GFX950MXScaleLayout swizzle requires
+        # hidden_size to be a multiple of 256 (SCALE_K = hidden_size / 32
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_kernel_combinations.py`, `tests/kernels/moe/test_routing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34285 - [Refactor] Move FusedMoE hidden_size roundup to quant_method

- 链接: https://github.com/vllm-project/vllm/pull/34285
- 状态/时间: merged / 2026-03-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `0ae89f18fd75`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+204/-222，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Move FusedMoE hidden_size roundup to quant_method」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: - Refactor hidden_size and intermediate roundup logic to be handled by QuantMethod. - Store padded and unpadded sizes under MoeConfig. - ~~Update ROCm padding logic to improve p...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79 (130 lines); hunks: -18,6 +18,7; -27,13 +28,13; symbols: create_weights, apply，涉及 `create_weights, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79 (130 lines); hunks: -18,6 +18,7; -27,13 +28,13; symbols: create_weights, apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -18,6 +18,7 @@
+    FusedMoEParallelConfig,
@@ -27,13 +28,13 @@
+    mxfp4_round_up_hidden_size_and_intermediate_size,
-    CK_MXFP4_MOE_DIM_ALIGNMENT,
@@ -49,7 +50,6 @@
-from vllm.utils.math_utils import round_up
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35153 - [MoE Refactor] Make SharedExperts class for use with DefaultMoERunner

- 链接: https://github.com/vllm-project/vllm/pull/35153
- 状态/时间: merged / 2026-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `7cf56a59a267`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+556/-397，可读 patch 1816 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Make SharedExperts class for use with DefaultMoERunner」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Move shared experts functionality into `SharedExperts` class. The `DefaultMoERunner` and modular kernels defer to `SharedExperts` for executing shared experts (making sure that...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -444,7 +444,7 @@ def apply(; -634,7 +634,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -444,7 +444,7 @@ def apply(; -634,7 +634,7 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -444,7 +444,7 @@ def apply(
-    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
+    ) -> torch.Tensor:
@@ -634,7 +634,7 @@ def apply(
-    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
+    ) -> torch.Tensor:
@@ -1027,7 +1027,7 @@ def apply(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/kernels/moe/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38774 - [ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle

- 链接: https://github.com/vllm-project/vllm/pull/38774
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `103f0de565f2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+170/-15，可读 patch 316 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`；PR 正文摘要: - Refactor quark_moe mxfp4 w4a16 to run through oracle and kernel backend. - Rename 'CK' backend to 'AITER', matching with existing '--moe-backend aiter' cli arg. - Add quark w4...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7 (153 lines); hunks: -5,6 +5,7; -27,7 +28,11; symbols: __init__, process_weights_after_loading, _setup_kernel_via_oracle，涉及 `__init__, process_weights_after_loading, _setup_kernel_via_oracle`；`tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8；`tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6；`tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7 (153 lines); hunks: -5,6 +5,7; -27,7 +28,11; symbols: __init__, process_weights_after_loading, _setup_kernel_via_oracle
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -5,6 +5,7 @@
+import vllm.model_executor.layers.fused_moe.modular_kernel as mk
@@ -27,7 +28,11 @@
+    TRITON_BACKENDS,
+    convert_to_mxfp4_moe_kernel_format,
+    make_mxfp4_moe_kernel,
+    make_mxfp4_moe_quant_config,
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml
@@ -0,0 +1,8 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+model_name: amd/gpt-oss-20b-w-mxfp4-a-bf16
+metric_threshold: 0.568
+reasoning_effort: low
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend aiter"
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml
@@ -0,0 +1,6 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0
- 验证与风险: diff 自带测试面 `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`, `tests/evals/gpt_oss/configs/models-gfx950.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38504 - [Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path

- 链接: https://github.com/vllm-project/vllm/pull/38504
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`；关联提交 `2df2c85be494`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+84/-216，可读 patch 444 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 缺陷修复；主要 diff: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`；PR 正文摘要: Three issues prevented `test_gpt_oss_triton_kernels` and GPQA serving from working on gfx950 (CDNA4) after the legacy routing deprecation: 1. `pack_bitmatrix` sets bit 31 spurio...。
- 实现要点: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2 (4 lines); hunks: -3,6 +3,6；`tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1 (2 lines); hunks: -3,4 +3,4；`tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1 (2 lines); hunks: -3,6 +3,6。
- 代码 diff 细节:
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2 (4 lines); hunks: -3,6 +3,6
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1 (2 lines); hunks: -3,4 +3,4
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1 (2 lines); hunks: -3,6 +3,6
- 关键代码摘录:

```diff
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml
@@ -3,6 +3,6 @@
-server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend aiter"
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend aiter --tokenizer openai/gpt-oss-20b --tensor-parallel-size 2"
-  VLLM_ROCM_USE_AITER: "1"
+  VLLM_ROCM_USE_AITER: "1"
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml
@@ -3,4 +3,4 @@
-server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend triton"
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --moe-backend triton --tokenizer openai/gpt-oss-20b --tensor-parallel-size 2"
diff -- tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml
@@ -3,6 +3,6 @@
-server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN"
+server_args: "--attention-backend ROCM_AITER_UNIFIED_ATTN --tensor-parallel-size 2"
```

- 已读文件:
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33892 - [W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections.

- 链接: https://github.com/vllm-project/vllm/pull/33892
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；关联提交 `2e9034c998e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+1710/-904，可读 patch 3623 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections.」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`；PR 正文摘要: This PR refactors block scaled linear kernel into kernel abstraction. changes: - Introduces `MMLinearKernel` base interface for all linear kernels. - Introduces `Params`, `Fp8Pa...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1 (6 lines); hunks: -7,6 +7,7; -57,6 +58,7 @@ def __init__(; symbols: __init__, get_min_capability, create_weights，涉及 `__init__, get_min_capability, create_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1 (6 lines); hunks: -7,6 +7,7; -57,6 +58,7 @@ def __init__(; symbols: __init__, get_min_capability, create_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -7,6 +7,7 @@
+from vllm.config import get_current_vllm_config
@@ -57,6 +58,7 @@ def __init__(
+        self.input_dtype = get_current_vllm_config().model_config.dtype
@@ -175,7 +177,9 @@ def create_weights(
-            out_dtype=torch.get_default_dtype(),
+            weight_shape=layer.weight.shape,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1
- 验证与风险: diff 自带测试面 `tests/compile/passes/distributed/test_fusion_all_reduce.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`, `tests/compile/passes/test_functionalization.py`, `tests/compile/passes/test_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36320 - [Quantization] Support Quark W8A8 INT8 MoE inference

- 链接: https://github.com/vllm-project/vllm/pull/36320
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `827268e98d92`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+360/-2，可读 patch 411 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] Support Quark W8A8 INT8 MoE inference」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`；PR 正文摘要: MoE models quantized by AMD Quark with W8A8 INT8 (per-channel weight + per-token dynamic activation) cannot be loaded in vLLM. For example, quantizing MiniMax-M2.1 (456B MoE) wi...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0 (282 lines); hunks: -109,6 +109,12 @@ def get_moe_method(; -505,6 +511,282 @@ def apply(; symbols: get_moe_method, apply, QuarkW8A8Int8MoEMethod, __init__，涉及 `get_moe_method, apply, QuarkW8A8Int8MoEMethod`；`vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0 (38 lines); hunks: -389,6 +389,37 @@ def _is_w4a8_mxfp4_fp8(; -556,6 +587,13 @@ def _get_scheme_from_config(; symbols: _is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x, _get_scheme_from_config，涉及 `_is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x`；`tests/quantization/test_quark.py` modified +31/-0 (31 lines); hunks: -22,6 +22,9; -126,6 +129,34 @@ def check_model(model):; symbols: check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity，涉及 `check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0 (282 lines); hunks: -109,6 +109,12 @@ def get_moe_method(; -505,6 +511,282 @@ def apply(; symbols: get_moe_method, apply, QuarkW8A8Int8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0 (38 lines); hunks: -389,6 +389,37 @@ def _is_w4a8_mxfp4_fp8(; -556,6 +587,13 @@ def _get_scheme_from_config(; symbols: _is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x, _get_scheme_from_config
  - `tests/quantization/test_quark.py` modified +31/-0 (31 lines); hunks: -22,6 +22,9; -126,6 +129,34 @@ def check_model(model):; symbols: check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -109,6 +109,12 @@ def get_moe_method(
+        elif quant_config._is_static_tensor_w8a8(
+            weight_config, input_config
+        ) or quant_config._is_dynamic_per_token_w8a8(weight_config, input_config):
+            return QuarkW8A8Int8MoEMethod(
+                weight_config, input_config, module.moe_config
+            )
diff -- vllm/model_executor/layers/quantization/quark/quark.py
@@ -389,6 +389,37 @@ def _is_w4a8_mxfp4_fp8(
+    def _is_dynamic_per_token_w8a8(
+        self,
+        weight_quant: dict[str, Any] | None,
+        input_quant: dict[str, Any] | None,
+    ) -> bool:
+        """Detect W8A8 INT8 with per-tensor or per-channel
diff -- tests/quantization/test_quark.py
@@ -22,6 +22,9 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0; `vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0
  - tests: `tests/quantization/test_quark.py` modified +31/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39604 - [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod"

- 链接: https://github.com/vllm-project/vllm/pull/39604
- 状态/时间: merged / 2026-04-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `739e5945dc4b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+129/-34，可读 patch 431 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod"」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: (Generated with Claude) This PR scopes the MXFP4 quantization implementation to GPT-OSS checkpoints while keeping generic MXFP4 support for other models (e.g. Quark). Changes **...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4 (8 lines); hunks: -30,11 +30,11; -995,7 +995,7 @@ def __init__(; symbols: __init__, _setup_kernel_via_oracle，涉及 `__init__, _setup_kernel_via_oracle`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4 (8 lines); hunks: -30,11 +30,11; -995,7 +995,7 @@ def __init__(; symbols: __init__, _setup_kernel_via_oracle
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -30,11 +30,11 @@
-    convert_to_mxfp4_moe_kernel_format,
+    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
-    select_mxfp4_moe_backend,
+    select_gpt_oss_mxfp4_moe_backend,
@@ -995,7 +995,7 @@ def __init__(
-            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4
- 验证与风险: runtime 路径改动集中在 `vllm/config/model.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39107 - [MoE Refactor] Remove MoE DP chunking

- 链接: https://github.com/vllm-project/vllm/pull/39107
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `e1e318af010b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+76/-389，可读 patch 782 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Remove MoE DP chunking」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: Remove DP chunking MoE runner. Use `max_num_batched_tokens` as default for `max_num_tokens` in `FusedMoEConfig`. CI Ran DeepEP related tests/kernels/moe tests locally. cc @rober...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):
-        # only apply to  batched mode
+        # only apply to batched mode
-            num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
+            num_warps = 4 if self.moe.max_num_tokens <= 512 else 8
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`, `tests/kernels/moe/test_flashinfer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39007 - [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/

- 链接: https://github.com/vllm-project/vllm/pull/39007
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `1a9353bb02e6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+16/-12，可读 patch 100 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: - Moves `gpt_oss_triton_kernels_moe.py` from `fused_moe/` root into `fused_moe/experts/`, consistent with the ongoing migration of expert kernel files (e.g. `trtllm_nvfp4_moe.py...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -1591,7 +1591,7 @@ def apply_monolithic(; symbols: apply_monolithic，涉及 `apply_monolithic`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -1591,7 +1591,7 @@ def apply_monolithic(; symbols: apply_monolithic
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1591,7 +1591,7 @@ def apply_monolithic(
-        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
+        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_oai_triton_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- 链接: https://github.com/vllm-project/vllm/pull/35949
- 状态/时间: merged / 2026-04-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `726efe177bf2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+325/-702，可读 patch 2430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: - Move the summation of the shared and fused expert outputs into `MoERunnerBase` and cleanup the final shared/fused all reduce code. Remove corresponding code from models that u...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -132,7 +132,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -132,7 +132,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -132,7 +132,6 @@ def __init__(
-            reduce_results=True,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-1
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35737 - [NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation

- 链接: https://github.com/vllm-project/vllm/pull/35737
- 状态/时间: merged / 2026-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/quantization/quark/quark_moe.py`；关联提交 `d622e27d2be9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+601/-121，可读 patch 1072 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`；PR 正文摘要: This PR enables running NVFP4 MOE models on Hopper and AMD Instinct MI300, MI350. This is useful for researchers, anybody trying out microscaling formats, and people who would l...。
- 实现要点: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54 (97 lines); hunks: -30,6 +30,7; -986,6 +987,8 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, process_weights_after_loading，涉及 `__init__, maybe_roundup_sizes, process_weights_after_loading`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54 (97 lines); hunks: -30,6 +30,7; -986,6 +987,8 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -30,6 +30,7 @@
+    backend_to_kernel_cls,
@@ -986,6 +987,8 @@ def __init__(
+        # TODO(bowenbao): refactor and introduce backends for other OCP MX schemes,
+        # use kernel abstraction for all OCP MX MOE implementations.
@@ -994,12 +997,6 @@ def __init__(
-        if self.ocp_mx_scheme == "w_mxfp4":
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/models-mi3xx.txt`, `tests/models/quantization/test_nvfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mixtral.py`；关联提交 `1c2c1eb8b9fd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: Mixtral Quark INT4/FP8 MoE；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mixtral.py`；PR 正文摘要: This is prep work for deleting the `FusedMoE` class and replacing it with `MoERunner`. This PR is just a rename of `FusedMoE.make_expert_params_mapping` to `fused_moe_make_exper...。
- 实现要点: `vllm/model_executor/models/mixtral.py` modified +5/-2 (7 lines); hunks: -40,7 +40,10; -364,7 +367,7 @@ def forward(; symbols: forward, get_expert_mapping，涉及 `forward, get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mixtral.py` modified +5/-2 (7 lines); hunks: -40,7 +40,10; -364,7 +367,7 @@ def forward(; symbols: forward, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -40,7 +40,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -364,7 +367,7 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
