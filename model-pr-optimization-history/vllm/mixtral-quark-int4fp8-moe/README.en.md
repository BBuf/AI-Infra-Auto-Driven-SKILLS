# vllm Mixtral Quark INT4/FP8 MoE Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
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
| `vllm/model_executor/layers/quantization/quark/__init__.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/layers/quantization/quark/quark.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#15734](https://github.com/vllm-project/vllm/pull/15734), [#16943](https://github.com/vllm-project/vllm/pull/16943), [#17215](https://github.com/vllm-project/vllm/pull/17215), [#17888](https://github.com/vllm-project/vllm/pull/17888), [#20251](https://github.com/vllm-project/vllm/pull/20251), [#24239](https://github.com/vllm-project/vllm/pull/24239), [#28275](https://github.com/vllm-project/vllm/pull/28275), [#28638](https://github.com/vllm-project/vllm/pull/28638), [#29008](https://github.com/vllm-project/vllm/pull/29008), [#30071](https://github.com/vllm-project/vllm/pull/30071), [#32779](https://github.com/vllm-project/vllm/pull/32779), ... (14 total) |
| `vllm/model_executor/layers/quantization/quark/quark_moe.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#11528](https://github.com/vllm-project/vllm/pull/11528), [#13784](https://github.com/vllm-project/vllm/pull/13784), [#14245](https://github.com/vllm-project/vllm/pull/14245), [#17888](https://github.com/vllm-project/vllm/pull/17888), [#22035](https://github.com/vllm-project/vllm/pull/22035), [#22537](https://github.com/vllm-project/vllm/pull/22537), [#23123](https://github.com/vllm-project/vllm/pull/23123), [#24649](https://github.com/vllm-project/vllm/pull/24649), [#26545](https://github.com/vllm-project/vllm/pull/26545), [#26739](https://github.com/vllm-project/vllm/pull/26739), [#27029](https://github.com/vllm-project/vllm/pull/27029), ... (46 total) |
| `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#16943](https://github.com/vllm-project/vllm/pull/16943), [#35316](https://github.com/vllm-project/vllm/pull/35316) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` | [#36100](https://github.com/vllm-project/vllm/pull/36100), [#36232](https://github.com/vllm-project/vllm/pull/36232) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_scheme.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` | [#35316](https://github.com/vllm-project/vllm/pull/35316) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#12612](https://github.com/vllm-project/vllm/pull/12612), [#14245](https://github.com/vllm-project/vllm/pull/14245), [#14390](https://github.com/vllm-project/vllm/pull/14390), [#14578](https://github.com/vllm-project/vllm/pull/14578), [#16236](https://github.com/vllm-project/vllm/pull/16236), [#19830](https://github.com/vllm-project/vllm/pull/19830), [#20251](https://github.com/vllm-project/vllm/pull/20251), [#27814](https://github.com/vllm-project/vllm/pull/27814), [#33892](https://github.com/vllm-project/vllm/pull/33892) |
| `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765), [#16236](https://github.com/vllm-project/vllm/pull/16236), [#27814](https://github.com/vllm-project/vllm/pull/27814) |
| `vllm/model_executor/layers/quantization/quark/utils.py` | [#10765](https://github.com/vllm-project/vllm/pull/10765) |
| `vllm/model_executor/models/mixtral.py` | [#2011](https://github.com/vllm-project/vllm/pull/2011), [#2015](https://github.com/vllm-project/vllm/pull/2015), [#2036](https://github.com/vllm-project/vllm/pull/2036), [#2090](https://github.com/vllm-project/vllm/pull/2090), [#2208](https://github.com/vllm-project/vllm/pull/2208), [#2542](https://github.com/vllm-project/vllm/pull/2542), [#2677](https://github.com/vllm-project/vllm/pull/2677), [#2769](https://github.com/vllm-project/vllm/pull/2769), [#2831](https://github.com/vllm-project/vllm/pull/2831), [#2875](https://github.com/vllm-project/vllm/pull/2875), [#2880](https://github.com/vllm-project/vllm/pull/2880), [#3597](https://github.com/vllm-project/vllm/pull/3597), ... (40 total) |

## PR Coverage Summary

- Git-traced PRs: 110
- Extra PRs preserved from existing docs: 12
- Total PRs in this document: 122
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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
| 2024-02-26 | [#2979](https://github.com/vllm-project/vllm/pull/2979) | merged | Optimized fused MoE Kernel, take 2 | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` |
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
| 2026-04-29 | [#41175](https://github.com/vllm-project/vllm/pull/41175) | merged | [ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |
| 2026-04-29 | [#41165](https://github.com/vllm-project/vllm/pull/41165) | merged | [ROCm][Bugfix][GPTOSS]: fix input_ids and expert_map args for quark w4a8 gptoss | `vllm/model_executor/layers/quantization/quark/quark_moe.py` |

## Per-PR Diff Audit Cards

### PR #4 - Use FlashAttention for `multi_query_kv_attention`

- Link: https://github.com/vllm-project/vllm/pull/4
- Status/date: merged / 2023-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +108/-34, 236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use FlashAttention for `multi_query_kv_attention`"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `cacheflow/models/attention.py`, `tests/kernels/attention.py`, `server.py`; technical summary: Covers "Use FlashAttention for `multi_query_kv_attention`"; the main implementation surface is `cacheflow/models/attention.py`, `tests/kernels/attention.py`, `server.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `cacheflow/models/attention.py` modified +37/-30 (67 lines); hunks: -1,5 +1,6; -14,20 +15,7 @@ def __init__(self, scale: float) -> None:; symbols: __init__, _masked_attention, multi_query_kv_attention, single_query_cached_kv_attention, touching `__init__, _masked_attention, multi_query_kv_attention`; `tests/kernels/attention.py` modified +65/-2 (67 lines); hunks: -1,10 +1,13; -79,7 +82,7 @@ def test_single_query_cached_kv_attention(; symbols: ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention, test_attention, touching `ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention`; `server.py` modified +5/-2 (7 lines); hunks: -9,10 +9,12; -27,6 +29,7 @@ def main():; symbols: main, touching `main`; `README.md` modified +1/-0 (1 lines); hunks: -4,6 +4,7.
- Code diff details:
  - `cacheflow/models/attention.py` modified +37/-30 (67 lines); hunks: -1,5 +1,6; -14,20 +15,7 @@ def __init__(self, scale: float) -> None:; symbols: __init__, _masked_attention, multi_query_kv_attention, single_query_cached_kv_attention
  - `tests/kernels/attention.py` modified +65/-2 (67 lines); hunks: -1,10 +1,13; -79,7 +82,7 @@ def test_single_query_cached_kv_attention(; symbols: ref_masked_attention, test_single_query_cached_kv_attention, test_multi_query_kv_attention, test_attention
  - `server.py` modified +5/-2 (7 lines); hunks: -9,10 +9,12; -27,6 +29,7 @@ def main():; symbols: main
  - `README.md` modified +1/-0 (1 lines); hunks: -4,6 +4,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `cacheflow/models/attention.py` modified +37/-30
  - tests: `tests/kernels/attention.py` modified +65/-2
  - other: `server.py` modified +5/-2; `README.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2011 - Mixtral 8x7B support

- Link: https://github.com/vllm-project/vllm/pull/2011
- Status/date: merged / 2023-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `b5f882cc98e2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +538/-0, 566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Mixtral 8x7B support"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Mixtral 8x7B support"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunks: -0,0 +1,534; symbols: promote_scalar, MixtralAttention, __init__, forward, touching `promote_scalar, MixtralAttention, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunks: -0,0 +1,534; symbols: promote_scalar, MixtralAttention, __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` added +534/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader.py`, `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2015 - Minor fixes for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2015
- Status/date: merged / 2023-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `4ff0203987ff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-6, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Minor fixes for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Minor fixes for Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +2/-6 (8 lines); hunks: -21,7 +21,7; -453,10 +453,6 @@ def __init__(; symbols: __init__, forward, sample, touching `__init__, forward, sample`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +2/-6 (8 lines); hunks: -21,7 +21,7; -453,10 +453,6 @@ def __init__(; symbols: __init__, forward, sample
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2036 - [Minor] Fix type annotation in Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2036
- Status/date: merged / 2023-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `81ce2a4b26c7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Minor] Fix type annotation in Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Minor] Fix type annotation in Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -251,7 +251,7 @@ def sparse_transpose(; symbols: sparse_transpose, topology, touching `sparse_transpose, topology`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -251,7 +251,7 @@ def sparse_transpose(; symbols: sparse_transpose, topology
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -251,7 +251,7 @@ def sparse_transpose(
-                 padded_bins: torch.Tensor) -> stk.Matrix:
+                 padded_bins: torch.Tensor) -> "stk.Matrix":
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2090 - Mixtral expert parallelism

- Link: https://github.com/vllm-project/vllm/pull/2090
- Status/date: merged / 2023-12-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `21d93c140d0a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +221/-334, 737 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Mixtral expert parallelism"; model line: Mixtral Quark INT4/FP8 MoE; category: docs/tests/CI; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Mixtral expert parallelism"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunks: -31,22 +31,11; -66,8 +55,134; symbols: promote_scalar, MixtralMLP, __init__, forward, touching `promote_scalar, MixtralMLP, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunks: -31,22 +31,11; -66,8 +55,134; symbols: promote_scalar, MixtralMLP, __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +207/-307
- Risk and verification: Runtime changes concentrate in `vllm/config.py`, `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2208 - [BugFix] Fix weight loading for Mixtral with TP

- Link: https://github.com/vllm-project/vllm/pull/2208
- Status/date: merged / 2023-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `ba4f82673884`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-26, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix weight loading for Mixtral with TP"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[BugFix] Fix weight loading for Mixtral with TP"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +5/-26 (31 lines); hunks: -49,7 +49,6; -94,30 +93,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, DummyModule, __init__, dummy_weight_loader, touching `forward, DummyModule, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +5/-26 (31 lines); hunks: -49,7 +49,6; -94,30 +93,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, DummyModule, __init__, dummy_weight_loader
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-26
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2293 - tensor parallel MOE implementation

- Link: https://github.com/vllm-project/vllm/pull/2293
- Status/date: closed / 2024-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +515/-126, 720 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "tensor parallel MOE implementation"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/model_executor/layers/moe.py`, `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe_grouped_matmul.py`; technical summary: Covers "tensor parallel MOE implementation"; the main implementation surface is `vllm/model_executor/layers/moe.py`, `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe_grouped_matmul.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/moe.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MoE, __init__, weight_loader, forward, touching `MoE, __init__, weight_loader`; `vllm/model_executor/models/mixtral.py` modified +34/-126 (160 lines); hunks: -23,10 +23,7; -35,17 +32,15; symbols: MixtralMLP, __init__, forward, MixtralMoE, touching `MixtralMLP, __init__, forward`; `tests/kernels/test_moe_grouped_matmul.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: ref_grouped_matmul, test_moe_grouped_matmul, touching `ref_grouped_matmul, test_moe_grouped_matmul`; `csrc/misc_kernels.cu` added +35/-0 (35 lines); hunks: -0,0 +1,35.
- Code diff details:
  - `vllm/model_executor/layers/moe.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MoE, __init__, weight_loader, forward
  - `vllm/model_executor/models/mixtral.py` modified +34/-126 (160 lines); hunks: -23,10 +23,7; -35,17 +32,15; symbols: MixtralMLP, __init__, forward, MixtralMoE
  - `tests/kernels/test_moe_grouped_matmul.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: ref_grouped_matmul, test_moe_grouped_matmul
  - `csrc/misc_kernels.cu` added +35/-0 (35 lines); hunks: -0,0 +1,35
  - `csrc/dispatch_utils.h` modified +11/-0 (11 lines); hunks: -14,3 +14,14
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/moe.py` added +369/-0; `vllm/model_executor/models/mixtral.py` modified +34/-126
  - tests: `tests/kernels/test_moe_grouped_matmul.py` added +57/-0
  - other: `csrc/misc_kernels.cu` added +35/-0; `csrc/dispatch_utils.h` modified +11/-0; `csrc/ops.h` modified +4/-0; `csrc/pybind.cpp` modified +4/-0; `setup.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe_grouped_matmul.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2453 - DeepseekMoE support with Fused MoE kernel

- Link: https://github.com/vllm-project/vllm/pull/2453
- Status/date: merged / 2024-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +924/-0, 957 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepseekMoE support with Fused MoE kernel"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/layers/fused_moe.py`, `vllm/model_executor/models/__init__.py`; technical summary: Covers "DeepseekMoE support with Fused MoE kernel"; the main implementation surface is `vllm/model_executor/models/deepseek.py`, `vllm/model_executor/layers/fused_moe.py`, `vllm/model_executor/models/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek.py` added +453/-0 (453 lines); hunks: -0,0 +1,453; symbols: DeepseekMLP, __init__, forward, DeepseekMoE, touching `DeepseekMLP, __init__, forward`; `vllm/model_executor/layers/fused_moe.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel, fused_moe, touching `fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel`; `vllm/model_executor/models/__init__.py` modified +1/-0 (1 lines); hunks: -18,6 +18,7; `csrc/moe_align_block_size_kernels.cu` added +108/-0 (108 lines); hunks: -0,0 +1,108.
- Code diff details:
  - `vllm/model_executor/models/deepseek.py` added +453/-0 (453 lines); hunks: -0,0 +1,453; symbols: DeepseekMLP, __init__, forward, DeepseekMoE
  - `vllm/model_executor/layers/fused_moe.py` added +287/-0 (287 lines); hunks: -0,0 +1,287; symbols: fused_moe_kernel, moe_align_block_size, invoke_fused_moe_kernel, fused_moe
  - `vllm/model_executor/models/__init__.py` modified +1/-0 (1 lines); hunks: -18,6 +18,7
  - `csrc/moe_align_block_size_kernels.cu` added +108/-0 (108 lines); hunks: -0,0 +1,108
  - `tests/kernels/test_fused_moe.py` added +50/-0 (50 lines); hunks: -0,0 +1,50; symbols: torch_moe, test_fused_moe
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek.py` added +453/-0; `vllm/model_executor/layers/fused_moe.py` added +287/-0; `vllm/model_executor/models/__init__.py` modified +1/-0
  - other: `csrc/moe_align_block_size_kernels.cu` added +108/-0; `csrc/dispatch_utils.h` modified +11/-0; `csrc/ops.h` modified +9/-0; `csrc/pybind.cpp` modified +4/-0
  - tests: `tests/kernels/test_fused_moe.py` added +50/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2542 - Fused MOE for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2542
- Status/date: merged / 2024-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `ab406446691f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +115/-109, 327 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fused MOE for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Fused MOE for Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunks: -23,8 +23,6; -33,10 +31,11; symbols: MixtralMLP, MixtralMoE, __init__, forward, touching `MixtralMLP, MixtralMoE, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunks: -23,8 +23,6; -33,10 +31,11; symbols: MixtralMLP, MixtralMoE, __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +104/-96
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2677 - Add unit test for Mixtral MoE layer

- Link: https://github.com/vllm-project/vllm/pull/2677
- Status/date: merged / 2024-01-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `d0d93b92b190`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +119/-55, 209 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add unit test for Mixtral MoE layer"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Add unit test for Mixtral MoE layer"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +6/-4 (10 lines); hunks: -70,13 +70,14 @@ def __init__(; -141,8 +142,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +6/-4 (10 lines); hunks: -70,13 +70,14 @@ def __init__(; -141,8 +142,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +6/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_moe.py`, `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2769 - Add fused top-K softmax kernel for MoE

- Link: https://github.com/vllm-project/vllm/pull/2769
- Status/date: merged / 2024-02-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `f0d4e145575b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +591/-50, 772 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add fused top-K softmax kernel for MoE"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Add fused top-K softmax kernel for MoE"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +3/-11 (14 lines); hunks: -24,8 +24,6; -128,18 +126,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +3/-11 (14 lines); hunks: -24,8 +24,6; -128,18 +126,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-11
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2831 - Add LoRA support for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2831
- Status/date: merged / 2024-02-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_mixtral.py`, `vllm/model_executor/models/mixtral.py`; associated commits `2a543d6efecc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +251/-121, 751 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add LoRA support for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: docs/tests/CI; main diff: `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`; technical summary: Covers "Add LoRA support for Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +37/-3 (40 lines); hunks: -27,6 +27,7; -38,7 +39,7; symbols: __init__, forward, MixtralForCausalLM, touching `__init__, forward, MixtralForCausalLM`; `tests/lora/test_mixtral.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: do_sample, test_mixtral_lora, touching `do_sample, test_mixtral_lora`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +37/-3 (40 lines); hunks: -27,6 +27,7; -38,7 +39,7; symbols: __init__, forward, MixtralForCausalLM
  - `tests/lora/test_mixtral.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: do_sample, test_mixtral_lora
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +37/-3
  - tests: `tests/lora/test_mixtral.py` added +53/-0
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_lora_manager.py`, `tests/lora/test_mixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #2880 - Align LoRA code between Mistral and Mixtral (fixes #2875)

- Link: https://github.com/vllm-project/vllm/pull/2880
- Status/date: merged / 2024-02-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `31348dff03d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-4, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Align LoRA code between Mistral and Mixtral (fixes #2875)"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Align LoRA code between Mistral and Mixtral (fixes #2875)"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +10/-4 (14 lines); hunks: -285,15 +285,19 @@ def __init__(; -350,7 +354,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +10/-4 (14 lines); hunks: -285,15 +285,19 @@ def __init__(; -350,7 +354,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +10/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2875 - Fix AttributeError: MixtralModel object has no attribute org_vocab_size.

- Link: https://github.com/vllm-project/vllm/pull/2875
- Status/date: closed / 2024-02-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `31348dff03d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix AttributeError: MixtralModel object has no attribute org_vocab_size."; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Fix AttributeError: MixtralModel object has no attribute org_vocab_size."; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -293,7 +293,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -293,7 +293,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -293,7 +293,7 @@ def __init__(
-            org_num_embeddings=self.org_vocab_size,
+            org_num_embeddings=self.vocab_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2979 - Optimized fused MoE Kernel, take 2

- Link: https://github.com/vllm-project/vllm/pull/2979
- Status/date: merged / 2024-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +297/-15, 364 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimized fused MoE Kernel, take 2"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json`; technical summary: Covers "Optimized fused MoE Kernel, take 2"; the main implementation surface is `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/fused_moe.py` renamed +63/-14 (77 lines); hunks: -1,11 +1,19; -210,6 +218,34 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tenso...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, get_moe_configs, fused_moe, touching `fused_moe_kernel, invoke_fused_moe_kernel, get_moe_configs`; `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` added +20/-0 (20 lines); hunks: -0,0 +1,20; `vllm/model_executor/layers/fused_moe/configs/README` added +10/-0 (10 lines); hunks: -0,0 +1,10.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` renamed +63/-14 (77 lines); hunks: -1,11 +1,19; -210,6 +218,34 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tenso...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, get_moe_configs, fused_moe
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` added +20/-0 (20 lines); hunks: -0,0 +1,20
  - `vllm/model_executor/layers/fused_moe/configs/README` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `vllm/model_executor/layers/fused_moe/__init__.py` added +5/-0 (5 lines); hunks: -0,0 +1,5
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -1,11 +1,19 @@
+import functools
+import json
+import os
+from typing import Any, Dict, Optional
+from vllm.logger import init_logger
+logger = init_logger(__name__)
diff -- vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json
@@ -0,0 +1,24 @@
+{
+    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 4},
+    "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
+    "4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
+    "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 4},
+    "16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 4},
diff -- vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json
@@ -0,0 +1,20 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/fused_moe.py` renamed +63/-14; `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` added +20/-0; `vllm/model_executor/layers/fused_moe/configs/README` added +10/-0; `vllm/model_executor/layers/fused_moe/__init__.py` added +5/-0
  - other: `benchmarks/kernels/benchmark_mixtral_moe.py` added +172/-0; `setup.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3597 - [BugFix] 1D query fix for MoE models

- Link: https://github.com/vllm-project/vllm/pull/3597
- Status/date: merged / 2024-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `41deac4a3d78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +15/-15, 93 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] 1D query fix for MoE models"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[BugFix] 1D query fix for MoE models"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +3/-4 (7 lines); hunks: -124,9 +124,9 @@ def weight_loader(self, param: nn.Parameter, loaded_weight:...; -140,8 +140,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: weight_loader, forward, MixtralAttention, touching `weight_loader, forward, MixtralAttention`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +3/-4 (7 lines); hunks: -124,9 +124,9 @@ def weight_loader(self, param: nn.Parameter, loaded_weight:...; -140,8 +140,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: weight_loader, forward, MixtralAttention
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4244 - [Kernel] FP8 support for MoE kernel / Mixtral

- Link: https://github.com/vllm-project/vllm/pull/4244
- Status/date: merged / 2024-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `eace8bf0b911`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +385/-21, 626 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] FP8 support for MoE kernel / Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Kernel] FP8 support for MoE kernel / Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +42/-2 (44 lines); hunks: -39,6 +39,8; -47,6 +49,7; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading, touching `MixtralMoE, __init__, weight_loader`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +42/-2 (44 lines); hunks: -39,6 +39,8; -47,6 +49,7; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +42/-2
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4343 - [Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales

- Link: https://github.com/vllm-project/vllm/pull/4343
- Status/date: merged / 2024-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `12628d3c787e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +95/-18, 233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Kernel] Optimize FP8 support for MoE kernel / Mixtral via static scales"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +37/-7 (44 lines); hunks: -105,6 +105,13 @@ def __init__(; -115,12 +122,23 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, touching `__init__, weight_loader, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +37/-7 (44 lines); hunks: -105,6 +105,13 @@ def __init__(; -115,12 +122,23 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +37/-7
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4332 - [Kernel] Support Fp8 Checkpoints (Dynamic + Static)

- Link: https://github.com/vllm-project/vllm/pull/4332
- Status/date: merged / 2024-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +307/-40, 549 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Support Fp8 Checkpoints (Dynamic + Static)"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/fp8.py`, `tests/models/test_fp8.py`, `vllm/model_executor/layers/linear.py`; technical summary: Covers "[Kernel] Support Fp8 Checkpoints (Dynamic + Static)"; the main implementation surface is `vllm/model_executor/layers/quantization/fp8.py`, `tests/models/test_fp8.py`, `vllm/model_executor/layers/linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30 (199 lines); hunks: -1,23 +1,36; -30,22 +43,22 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: Fp8Config, for, __init__, get_supported_act_dtypes, touching `Fp8Config, for, __init__`; `tests/models/test_fp8.py` added +90/-0 (90 lines); hunks: -0,0 +1,90; symbols: test_models, touching `test_models`; `vllm/model_executor/layers/linear.py` modified +48/-10 (58 lines); hunks: -246,6 +246,10 @@ def __init__(; -254,6 +258,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader, touching `__init__, weight_loader`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30 (199 lines); hunks: -1,23 +1,36; -30,22 +43,22 @@ def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: Fp8Config, for, __init__, get_supported_act_dtypes
  - `tests/models/test_fp8.py` added +90/-0 (90 lines); hunks: -0,0 +1,90; symbols: test_models
  - `vllm/model_executor/layers/linear.py` modified +48/-10 (58 lines); hunks: -246,6 +246,10 @@ def __init__(; -254,6 +258,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +169/-30; `vllm/model_executor/layers/linear.py` modified +48/-10
  - tests: `tests/models/test_fp8.py` added +90/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4436 - [Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static)

- Link: https://github.com/vllm-project/vllm/pull/4436
- Status/date: closed / 2024-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +105/-40, 223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static)"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/model_loader/weight_utils.py`; technical summary: Covers "[Kernel] Support Fp8 Checkpoints for Mixtral (Dynamic + Static)"; the main implementation surface is `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/model_loader/weight_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +100/-40 (140 lines); hunks: -46,7 +46,8; -78,6 +79,8 @@ def __init__(; symbols: __init__, weight_loader, touching `__init__, weight_loader`; `vllm/model_executor/model_loader/weight_utils.py` modified +5/-0 (5 lines); hunks: -370,3 +370,8 @@ def initialize_dummy_weights(; symbols: initialize_dummy_weights, all_close_1d, touching `initialize_dummy_weights, all_close_1d`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +100/-40 (140 lines); hunks: -46,7 +46,8; -78,6 +79,8 @@ def __init__(; symbols: __init__, weight_loader
  - `vllm/model_executor/model_loader/weight_utils.py` modified +5/-0 (5 lines); hunks: -370,3 +370,8 @@ def initialize_dummy_weights(; symbols: initialize_dummy_weights, all_close_1d
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +100/-40; `vllm/model_executor/model_loader/weight_utils.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4543 - [Misc] Remove Mixtral device="cuda" declarations

- Link: https://github.com/vllm-project/vllm/pull/4543
- Status/date: merged / 2024-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `c9d852d601ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-8, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Remove Mixtral device="cuda" declarations"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Misc] Remove Mixtral device="cuda" declarations"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +4/-8 (12 lines); hunks: -96,13 +96,11 @@ def __init__(; -114,22 +112,20 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +4/-8 (12 lines); hunks: -96,13 +96,11 @@ def __init__(; -114,22 +112,20 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +4/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4527 - [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)

- Link: https://github.com/vllm-project/vllm/pull/4527
- Status/date: merged / 2024-05-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `2a052011ca47`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +122/-53, 262 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunks: -78,6 +78,8 @@ def __init__(; -86,55 +88,79 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, touching `__init__, weight_loader, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunks: -78,6 +78,8 @@ def __init__(; -86,55 +88,79 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +120/-51
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4793 - [Bugfix] Fix dynamic FP8 quantization for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/4793
- Status/date: merged / 2024-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `33d3914b1e6d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix dynamic FP8 quantization for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Bugfix] Fix dynamic FP8 quantization for Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -95,7 +95,7 @@ def __init__(
-        if self.use_fp8:
+        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4893 - [Misc] Load FP8 kv-cache scaling factors from checkpoints

- Link: https://github.com/vllm-project/vllm/pull/4893
- Status/date: merged / 2024-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `a3a73ab0696b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 40 files, +284/-158, 869 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Load FP8 kv-cache scaling factors from checkpoints"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Misc] Load FP8 kv-cache scaling factors from checkpoints"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +21/-8 (29 lines); hunks: -308,14 +308,13 @@ def __init__(self,; -581,6 +580,20 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +21/-8 (29 lines); hunks: -308,14 +308,13 @@ def __init__(self,; -581,6 +580,20 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +21/-8
- Risk and verification: The diff ships test coverage in `tests/models/test_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5039 - [Model] Enable FP8 QKV in MoE and refine kernel tuning script

- Link: https://github.com/vllm-project/vllm/pull/5039
- Status/date: merged / 2024-05-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `e9899fb7a4d9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +711/-114, 1000 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Enable FP8 QKV in MoE and refine kernel tuning script"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Model] Enable FP8 QKV in MoE and refine kernel tuning script"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +0/-9 (9 lines); hunks: -278,15 +278,6 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +0/-9 (9 lines); hunks: -278,15 +278,6 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/configs/E=8,N=14336,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5231 - [Model] Correct Mixtral FP8 checkpoint loading

- Link: https://github.com/vllm-project/vllm/pull/5231
- Status/date: merged / 2024-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `5563a4dea86e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +80/-35, 182 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Correct Mixtral FP8 checkpoint loading"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Model] Correct Mixtral FP8 checkpoint loading"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +76/-32 (108 lines); hunks: -41,7 +41,9; -98,16 +100,16 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading, touching `__init__, weight_loader, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +76/-32 (108 lines); hunks: -41,7 +41,9; -98,16 +100,16 @@ def __init__(; symbols: __init__, weight_loader, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +76/-32
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5353 - [Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale

- Link: https://github.com/vllm-project/vllm/pull/5353
- Status/date: merged / 2024-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `c09dade2a263`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +23/-23, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Misc][Breaking] Change FP8 checkpoint format from act_scale -> input_scale"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +8/-8 (16 lines); hunks: -147,7 +147,7 @@ def __init__(; -182,11 +182,11 @@ def weight_loader(self, param: nn.Parameter, loaded_weight...; symbols: __init__, weight_loader, process_weights_after_loading, load_weights, touching `__init__, weight_loader, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +8/-8 (16 lines); hunks: -147,7 +147,7 @@ def __init__(; -182,11 +182,11 @@ def weight_loader(self, param: nn.Parameter, loaded_weight...; symbols: __init__, weight_loader, process_weights_after_loading, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +8/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5970 - [ Misc ] Refactor MoE to isolate Fp8 From Mixtral

- Link: https://github.com/vllm-project/vllm/pull/5970
- Status/date: merged / 2024-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `7c008c51a9aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +537/-306, 1033 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ Misc ] Refactor MoE to isolate Fp8 From Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[ Misc ] Refactor MoE to isolate Fp8 From Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +41/-235 (276 lines); hunks: -27,30 +27,23; -66,227 +59,40 @@ class MixtralMoE(nn.Module):; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading, touching `MixtralMoE, __init__, weight_loader`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +41/-235 (276 lines); hunks: -27,30 +27,23; -66,227 +59,40 @@ class MixtralMoE(nn.Module):; symbols: MixtralMoE, __init__, weight_loader, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +41/-235
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6287 - [Bugfix] Support 2D input shape in MoE layer

- Link: https://github.com/vllm-project/vllm/pull/6287
- Status/date: merged / 2024-07-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `e72ae80b0640`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Support 2D input shape in MoE layer"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Bugfix] Support 2D input shape in MoE layer"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +3/-2 (5 lines); hunks: -88,12 +88,13 @@ def __init__(self,; symbols: __init__, forward, MixtralAttention, touching `__init__, forward, MixtralAttention`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +3/-2 (5 lines); hunks: -88,12 +88,13 @@ def __init__(self,; symbols: __init__, forward, MixtralAttention
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -88,12 +88,13 @@ def __init__(self,
-        num_tokens, hidden_size = hidden_states.shape
+        # NOTE: hidden_states can have either 1D or 2D shape.
+        orig_shape = hidden_states.shape
-        return final_hidden_states.view(num_tokens, hidden_size)
+        return final_hidden_states.view(orig_shape)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/qwen2_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6417 - [ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8

- Link: https://github.com/vllm-project/vllm/pull/6417
- Status/date: merged / 2024-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `fb6af8bc0863`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +223/-137, 564 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[ Misc ] Apply MoE Refactor to Qwen2 + Deepseekv2 To Support Fp8"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +7/-25 (32 lines); hunks: -372,31 +372,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +7/-25 (32 lines); hunks: -372,31 +372,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +7/-25
- Risk and verification: Runtime changes concentrate in `.buildkite/lm-eval-harness/configs/DeepSeek-V2-Lite-Chat.yaml`, `.buildkite/lm-eval-harness/configs/models-large.txt`, `vllm/model_executor/layers/fused_moe/fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6516 - [Model] Pipeline parallel support for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/6516
- Status/date: merged / 2024-07-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `b5af8c223c3d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +60/-19, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Pipeline parallel support for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Model] Pipeline parallel support for Mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +48/-13 (61 lines); hunks: -29,7 +29,7; -48,6 +48,7; symbols: MixtralMoE, __init__, forward, touching `MixtralMoE, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +48/-13 (61 lines); hunks: -29,7 +29,7; -48,6 +48,7; symbols: MixtralMoE, __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +48/-13
- Risk and verification: The diff ships test coverage in `tests/distributed/test_pipeline_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7334 - [Misc] Update Fused MoE weight loading

- Link: https://github.com/vllm-project/vllm/pull/7334
- Status/date: merged / 2024-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `d3bdfd3ab9ba`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +264/-201, 614 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Update Fused MoE weight loading"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Misc] Update Fused MoE weight loading"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-1 (2 lines); hunks: -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -452,7 +452,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                                  weight_name,
+                                  name,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7527 - [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel

- Link: https://github.com/vllm-project/vllm/pull/7527
- Status/date: merged / 2024-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `8678a69ab519`, `aae74ef95c37`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +2374/-84, 2645 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,6 +73,7 @@ def __init__(self,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/weight_loading/models.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7764 - Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)"

- Link: https://github.com/vllm-project/vllm/pull/7764
- Status/date: merged / 2024-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `aae74ef95c37`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +84/-2374, 2645 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)""; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "Revert "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel (#7527)""; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,7 +73,6 @@ def __init__(self,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `tests/weight_loading/models.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7766 - [Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel

- Link: https://github.com/vllm-project/vllm/pull/7766
- Status/date: merged / 2024-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `fc911880cc50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +2382/-85, 2666 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Kernel] Expand MoE weight loading + Add Fused Marlin MoE Kernel"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +1/-0 (1 lines); hunks: -73,6 +73,7 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -73,6 +73,7 @@ def __init__(self,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_compressed_tensors.py`, `tests/weight_loading/models.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8217 - [Misc] Fused MoE Marlin support for GPTQ

- Link: https://github.com/vllm-project/vllm/pull/8217
- Status/date: merged / 2024-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `6cd5e5b07e44`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +912/-204, 1442 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Fused MoE Marlin support for GPTQ"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Misc] Fused MoE Marlin support for GPTQ"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +7/-2 (9 lines); hunks: -435,7 +435,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -454,6 +455,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +7/-2 (9 lines); hunks: -435,7 +435,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; -454,6 +455,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +7/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`, `tests/weight_loading/models-large.txt`, `tests/weight_loading/models.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9008 - [Model] add a bunch of supported lora modules for mixtral

- Link: https://github.com/vllm-project/vllm/pull/9008
- Status/date: merged / 2024-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_mixtral.py`, `vllm/model_executor/models/mixtral.py`; associated commits `9ade8bbc8dc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +69/-20, 123 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] add a bunch of supported lora modules for mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`; technical summary: Covers "[Model] add a bunch of supported lora modules for mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`, `tests/lora/test_mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +2/-4 (6 lines); hunks: -319,10 +319,8 @@ class MixtralForCausalLM(nn.Module, SupportsLoRA):; symbols: MixtralForCausalLM, touching `MixtralForCausalLM`; `tests/lora/test_mixtral.py` modified +62/-16 (78 lines); hunks: -9,12 +9,9; -33,22 +30,71 @@ def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -...; symbols: do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules, touching `do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +2/-4 (6 lines); hunks: -319,10 +319,8 @@ class MixtralForCausalLM(nn.Module, SupportsLoRA):; symbols: MixtralForCausalLM
  - `tests/lora/test_mixtral.py` modified +62/-16 (78 lines); hunks: -9,12 +9,9; -33,22 +30,71 @@ def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -...; symbols: do_sample, test_mixtral_lora, test_mixtral_lora_all_target_modules
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-4
  - tests: `tests/lora/test_mixtral.py` modified +62/-16
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_mixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9632 - [torch.compile] support moe models

- Link: https://github.com/vllm-project/vllm/pull/9632
- Status/date: merged / 2024-10-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +217/-78, 605 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile] support moe models"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; technical summary: Covers "[torch.compile] support moe models"; the main implementation surface is `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7 (100 lines); hunks: -358,9 +358,10 @@ def try_get_optimal_moe_config(; -465,19 +466,109 @@ def get_config_dtype_str(dtype: torch.dtype,; symbols: try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts, _, touching `try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts`; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9 (51 lines); hunks: -1,6 +1,6; -18,6 +18,7 @@ def get_scalar_type(num_bits: int, has_zp: bool):; symbols: get_scalar_type, single_marlin_moe, touching `get_scalar_type, single_marlin_moe`; `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11 (29 lines); hunks: -12,7 +12,16; -96,9 +105,6 @@ def forward_cuda(; symbols: forward_cuda, forward_tpu, FusedMoE, touching `forward_cuda, forward_tpu, FusedMoE`; `vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4 (28 lines); hunks: -1,23 +1,43; symbols: override_config, get_config, touching `override_config, get_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7 (100 lines); hunks: -358,9 +358,10 @@ def try_get_optimal_moe_config(; -465,19 +466,109 @@ def get_config_dtype_str(dtype: torch.dtype,; symbols: try_get_optimal_moe_config, get_config_dtype_str, inplace_fused_experts, _
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9 (51 lines); hunks: -1,6 +1,6; -18,6 +18,7 @@ def get_scalar_type(num_bits: int, has_zp: bool):; symbols: get_scalar_type, single_marlin_moe
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11 (29 lines); hunks: -12,7 +12,16; -96,9 +105,6 @@ def forward_cuda(; symbols: forward_cuda, forward_tpu, FusedMoE
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4 (28 lines); hunks: -1,23 +1,43; symbols: override_config, get_config
  - `vllm/model_executor/layers/quantization/awq_marlin.py` modified +2/-5 (7 lines); hunks: -3,6 +3,7; -435,10 +436,6 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +93/-7; `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +42/-9; `vllm/model_executor/layers/fused_moe/layer.py` modified +18/-11; `vllm/model_executor/layers/fused_moe/__init__.py` modified +24/-4; `vllm/model_executor/layers/quantization/awq_marlin.py` modified +2/-5; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-5
- Risk and verification: The diff ships test coverage in `tests/compile/test_basic_correctness.py`, `tests/kernels/test_awq_marlin.py`, `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9758 - [torch.compile] Adding "torch compile" annotations to some models

- Link: https://github.com/vllm-project/vllm/pull/9758
- Status/date: merged / 2024-10-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `aa0addb39726`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +8/-0, 64 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile] Adding "torch compile" annotations to some models"; model line: Mixtral Quark INT4/FP8 MoE; category: docs/tests/CI; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[torch.compile] Adding "torch compile" annotations to some models"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -245,6 +246,7 @@ def forward(; symbols: forward, MixtralModel, __init__, touching `forward, MixtralModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -245,6 +246,7 @@ def forward(; symbols: forward, MixtralModel, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -28,6 +28,7 @@
+from vllm.compilation.decorators import support_torch_compile
@@ -245,6 +246,7 @@ def forward(
+@support_torch_compile
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/arctic.py`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/olmoe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11390 - [Bugfix] Fix fully sharded LoRAs with Mixtral

- Link: https://github.com/vllm-project/vllm/pull/11390
- Status/date: merged / 2024-12-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_mixtral.py`; associated commits `f1d1bf6288ab`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix fully sharded LoRAs with Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `tests/lora/test_mixtral.py`; technical summary: Covers "[Bugfix] Fix fully sharded LoRAs with Mixtral"; the main implementation surface is `tests/lora/test_mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/lora/test_mixtral.py` modified +3/-1 (4 lines); hunks: -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_al...; symbols: test_mixtral_lora, test_mixtral_lora_all_target_modules, touching `test_mixtral_lora, test_mixtral_lora_all_target_modules`.
- Code diff details:
  - `tests/lora/test_mixtral.py` modified +3/-1 (4 lines); hunks: -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_al...; symbols: test_mixtral_lora, test_mixtral_lora_all_target_modules
- Key code excerpts:

```diff
diff -- tests/lora/test_mixtral.py
@@ -62,8 +62,9 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):
+@pytest.mark.parametrize("fully_shard", [True, False])
-                                         tp_size):
+                                         tp_size, fully_shard):
@@ -82,6 +83,7 @@ def test_mixtral_lora_all_target_modules(mixtral_lora_files_all_target_modules,
+        fully_sharded_loras=fully_shard,
```

- Reviewed files:
  - tests: `tests/lora/test_mixtral.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/lora/test_mixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10765 - [Misc][Quark] Upstream Quark format to VLLM

- Link: https://github.com/vllm-project/vllm/pull/10765
- Status/date: merged / 2025-01-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/__init__.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` and 10 files; associated commits `de0526f668d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 32 files, +1264/-70, 1679 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][Quark] Upstream Quark format to VLLM"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; technical summary: Covers "[Misc][Quark] Upstream Quark format to VLLM"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0 (387 lines); hunks: -0,0 +1,387; symbols: QuarkConfig, __init__, get_linear_method, get_supported_act_dtypes, touching `QuarkConfig, __init__, get_linear_method`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod, __init__, touching `QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod`; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading, touching `QuarkW8A8Fp8, __init__, get_min_capability`; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: QuarkW8A8Int8, __init__, get_min_capability, create_weights, touching `QuarkW8A8Int8, __init__, get_min_capability`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0 (387 lines); hunks: -0,0 +1,387; symbols: QuarkConfig, __init__, get_linear_method, get_supported_act_dtypes
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: QuarkMoEMethod, get_moe_method, QuarkW8A8Fp8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: QuarkW8A8Fp8, __init__, get_min_capability, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: QuarkW8A8Int8, __init__, get_min_capability, create_weights
  - `vllm/model_executor/layers/quantization/quark/utils.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: deep_compare, should_ignore_layer, check_equal_or_regex_match, _is_equal_or_regex_match
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` added +387/-0; `vllm/model_executor/layers/quantization/quark/quark_moe.py` added +225/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` added +140/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` added +105/-0; `vllm/model_executor/layers/quantization/quark/utils.py` added +99/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_scheme.py` added +52/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11528 - [BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order

- Link: https://github.com/vllm-project/vllm/pull/11528
- Status/date: merged / 2025-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `eb5cb5e5280c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +243/-148, 756 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[BugFix] Fix parameter names and `process_after_weight_loading` for W4A16 MoE Group Act Order"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9 (20 lines); hunks: -60,24 +60,26 @@ def __init__(self, weight_config: Dict[str, Any], input_conf...; symbols: __init__, create_weights, touching `__init__, create_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9 (20 lines); hunks: -60,24 +60,26 @@ def __init__(self, weight_config: Dict[str, Any], input_conf...; symbols: __init__, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +11/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/awq_marlin.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3208 - [RFC/WIP] First steps towards FP8 for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/3208
- Status/date: closed / 2025-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +203/-71, 473 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[RFC/WIP] First steps towards FP8 for Mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`; technical summary: Covers "[RFC/WIP] First steps towards FP8 for Mixtral"; the main implementation surface is `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22 (44 lines); hunks: -1,24 +1,24; `vllm/model_executor/models/mixtral.py` modified +36/-2 (38 lines); hunks: -21,6 +21,7; -93,13 +94,23 @@ def __init__(; symbols: __init__, weight_loader, forward, touching `__init__, weight_loader, forward`; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14 (32 lines); hunks: -114,7 +114,7 @@ def fused_moe_kernel(; -186,7 +186,7 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, fused_moe, touching `fused_moe_kernel, invoke_fused_moe_kernel, fused_moe`; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22 (44 lines); hunks: -1,24 +1,24
  - `vllm/model_executor/models/mixtral.py` modified +36/-2 (38 lines); hunks: -21,6 +21,7; -93,13 +94,23 @@ def __init__(; symbols: __init__, weight_loader, forward
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14 (32 lines); hunks: -114,7 +114,7 @@ def fused_moe_kernel(; -186,7 +186,7 @@ def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor...; symbols: fused_moe_kernel, invoke_fused_moe_kernel, fused_moe
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0 (24 lines); hunks: -0,0 +1,24
  - `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` removed +0/-20 (20 lines); hunks: -1,20 +0,0
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json` modified +22/-22; `vllm/model_executor/models/mixtral.py` modified +36/-2; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +18/-14; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json` added +24/-0; `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json` removed +0/-20
  - other: `csrc/activation_kernels.cu` modified +40/-0; `benchmarks/kernels/benchmark_mixtral_moe.py` modified +4/-7
  - docs: `examples/quantization/mixtral/mixtral_fp8_quantization.py` added +34/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json`, `vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13591 - [core] set up data parallel communication

- Link: https://github.com/vllm-project/vllm/pull/13591
- Status/date: merged / 2025-02-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +416/-28, 719 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[core] set up data parallel communication"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/distributed/utils.py`, `examples/offline_inference/data_parallel.py`, `vllm/distributed/parallel_state.py`; technical summary: Covers "[core] set up data parallel communication"; the main implementation surface is `vllm/distributed/utils.py`, `examples/offline_inference/data_parallel.py`, `vllm/distributed/parallel_state.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/distributed/utils.py` modified +90/-1 (91 lines); hunks: -11,7 +11,11; -227,3 +231,88 @@ def create(; symbols: create, stateless_init_torch_distributed_process_group, touching `create, stateless_init_torch_distributed_process_group`; `examples/offline_inference/data_parallel.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: main, touching `main`; `vllm/distributed/parallel_state.py` modified +62/-14 (76 lines); hunks: -750,6 +750,13 @@ def get_tp_group() -> GroupCoordinator:; -811,6 +818,21 @@ def init_distributed_environment(; symbols: get_tp_group, get_dp_group, get_pp_group, init_distributed_environment, touching `get_tp_group, get_dp_group, get_pp_group`; `vllm/config.py` modified +57/-0 (57 lines); hunks: -16,6 +16,7; -1290,6 +1291,11 @@ class ParallelConfig:; symbols: ParallelConfig, get_next_dp_init_port, stateless_init_dp_group, has_unfinished_dp, touching `ParallelConfig, get_next_dp_init_port, stateless_init_dp_group`.
- Code diff details:
  - `vllm/distributed/utils.py` modified +90/-1 (91 lines); hunks: -11,7 +11,11; -227,3 +231,88 @@ def create(; symbols: create, stateless_init_torch_distributed_process_group
  - `examples/offline_inference/data_parallel.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: main
  - `vllm/distributed/parallel_state.py` modified +62/-14 (76 lines); hunks: -750,6 +750,13 @@ def get_tp_group() -> GroupCoordinator:; -811,6 +818,21 @@ def init_distributed_environment(; symbols: get_tp_group, get_dp_group, get_pp_group, init_distributed_environment
  - `vllm/config.py` modified +57/-0 (57 lines); hunks: -16,6 +16,7; -1290,6 +1291,11 @@ class ParallelConfig:; symbols: ParallelConfig, get_next_dp_init_port, stateless_init_dp_group, has_unfinished_dp
  - `vllm/forward_context.py` modified +31/-3 (34 lines); hunks: -4,9 +4,10; -32,6 +33,8 @@ class ForwardContext:; symbols: ForwardContext, get_forward_context, set_forward_context
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/distributed/utils.py` modified +90/-1; `vllm/distributed/parallel_state.py` modified +62/-14; `vllm/config.py` modified +57/-0; `vllm/forward_context.py` modified +31/-3; `vllm/v1/engine/llm_engine.py` modified +24/-2; `vllm/envs.py` modified +20/-0
  - docs: `examples/offline_inference/data_parallel.py` added +76/-0
- Risk and verification: Runtime changes concentrate in `vllm/config.py`, `vllm/distributed/device_communicators/cuda_communicator.py`, `vllm/distributed/device_communicators/custom_all_reduce.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13784 - [Bugfix][Quantization] Fix FP8 + EP

- Link: https://github.com/vllm-project/vllm/pull/13784
- Status/date: merged / 2025-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `1e15aaef562d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +22/-22, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Quantization] Fix FP8 + EP"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Bugfix][Quantization] Fix FP8 + EP"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading, touching `process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -174,7 +174,7 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
-        for expert_id in range(layer.num_experts):
+        for expert_id in range(layer.local_num_experts):
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/awq_marlin.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13931 - [V1] EP/TP MoE + DP Attention

- Link: https://github.com/vllm-project/vllm/pull/13931
- Status/date: merged / 2025-03-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `72c62eae5f01`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +250/-75, 778 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] EP/TP MoE + DP Attention"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[V1] EP/TP MoE + DP Attention"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -71,6 +71,7 @@ def __init__(self,; -93,6 +94,7 @@ def __init__(self,; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +2/-0 (2 lines); hunks: -71,6 +71,7 @@ def __init__(self,; -93,6 +94,7 @@ def __init__(self,; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -71,6 +71,7 @@ def __init__(self,
+                 dp_size: Optional[int] = None,
@@ -93,6 +94,7 @@ def __init__(self,
+                                dp_size=dp_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14390 - [FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object

- Link: https://github.com/vllm-project/vllm/pull/14390
- Status/date: merged / 2025-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `e1744502c21f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +268/-242, 760 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; technical summary: Covers "[FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11 (18 lines); hunks: -7,8 +7,7; -22,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; symbols: QuarkW8A8Fp8, __init__, get_min_capability, apply_weights, touching `QuarkW8A8Fp8, __init__, get_min_capability`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11 (18 lines); hunks: -7,8 +7,7; -22,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; symbols: QuarkW8A8Fp8, __init__, get_min_capability, apply_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +7/-11
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14245 - dynamic distpatch of fp8 kernels

- Link: https://github.com/vllm-project/vllm/pull/14245
- Status/date: merged / 2025-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `a1c8f3796c89`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +293/-159, 944 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "dynamic distpatch of fp8 kernels"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "dynamic distpatch of fp8 kernels"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2 (4 lines); hunks: -39,7 +39,7 @@ def process_weights_after_loading(self, layer) -> None:; -55,7 +55,7 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading, touching `process_weights_after_loading`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2 (3 lines); hunks: -142,8 +142,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading, touching `process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2 (4 lines); hunks: -39,7 +39,7 @@ def process_weights_after_loading(self, layer) -> None:; -55,7 +55,7 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2 (3 lines); hunks: -142,8 +142,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-2; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/quant_utils.py`, `tests/kernels/test_triton_scaled_mm.py`, `tests/quantization/test_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14961 - [Bugfix][Model] Mixtral: use unused head_dim config argument

- Link: https://github.com/vllm-project/vllm/pull/14961
- Status/date: merged / 2025-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `aaaec52ad907`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-2, 54 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Mixtral: use unused head_dim config argument"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Bugfix][Model] Mixtral: use unused head_dim config argument"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +5/-1 (6 lines); hunks: -111,6 +111,7 @@ class MixtralAttention(nn.Module):; -136,7 +137,9 @@ def __init__(; symbols: MixtralAttention, __init__, touching `MixtralAttention, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +5/-1 (6 lines); hunks: -111,6 +111,7 @@ class MixtralAttention(nn.Module):; -136,7 +137,9 @@ def __init__(; symbols: MixtralAttention, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/mixtral_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14578 - [Quantization][FP8] Adding support for fp8 gemm layer input in fp8

- Link: https://github.com/vllm-project/vllm/pull/14578
- Status/date: merged / 2025-03-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `4d0ec37267af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +41/-9, 155 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization][FP8] Adding support for fp8 gemm layer input in fp8"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; technical summary: Covers "[Quantization][FP8] Adding support for fp8 gemm layer input in fp8"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optio...; -134,5 +135,6 @@ def apply_weights(self,; symbols: __init__, get_min_capability, apply_weights, touching `__init__, get_min_capability, apply_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optio...; -134,5 +135,6 @@ def apply_weights(self,; symbols: __init__, get_min_capability, apply_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py
@@ -22,6 +22,7 @@ def __init__(self, qscheme: str, is_static_input_scheme: Optional[bool]):
+        self.out_dtype = torch.get_default_dtype()
@@ -134,5 +135,6 @@ def apply_weights(self,
+                                     out_dtype=self.out_dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/fbgemm_fp8.py`, `vllm/model_executor/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15961 - Add support to modelopt quantization of Mixtral model

- Link: https://github.com/vllm-project/vllm/pull/15961
- Status/date: merged / 2025-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-1, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support to modelopt quantization of Mixtral model"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral_quant.py`; technical summary: Covers "Add support to modelopt quantization of Mixtral model"; the main implementation surface is `vllm/model_executor/models/mixtral_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunks: -45,7 +45,8; -420,6 +421,11 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunks: -45,7 +45,8; -420,6 +421,11 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral_quant.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16325 - [Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral

- Link: https://github.com/vllm-project/vllm/pull/16325
- Status/date: merged / 2025-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `a564797151a0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +353/-323, 836 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Model] use AutoWeightsLoader for granite, granitemoe, granitemoeshared, grok1, mixtral"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +90/-86 (176 lines); hunks: -49,7 +49,7; -260,6 +260,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, MixtralForCausalLM, get_input_embeddings, touching `__init__, forward, MixtralForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +90/-86 (176 lines); hunks: -49,7 +49,7; -260,6 +260,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, MixtralForCausalLM, get_input_embeddings
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +90/-86
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/granite.py`, `vllm/model_executor/models/granitemoe.py`, `vllm/model_executor/models/granitemoeshared.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16236 - [Bugfix] Fix bugs of running Quark quantized models

- Link: https://github.com/vllm-project/vllm/pull/16236
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`; associated commits `9e90c9f73f94`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +67/-22, 183 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix bugs of running Quark quantized models"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`; technical summary: Covers "[Bugfix] Fix bugs of running Quark quantized models"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10 (34 lines); hunks: -35,7 +35,7 @@ def create_weights(self, layer: torch.nn.Module,; -63,16 +63,28 @@ def create_weights(self, layer: torch.nn.Module,; symbols: create_weights, process_weights_after_loading, touching `create_weights, process_weights_after_loading`; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4 (10 lines); hunks: -21,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; -41,10 +41,11 @@ def process_weights_after_loading(self, layer) -> None:; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, create_weights, touching `QuarkW8A8Fp8, __init__, process_weights_after_loading`; `tests/quantization/test_quark.py` modified +37/-8 (45 lines); hunks: -4,17 +4,28; -26,11 +37,29 @@ def check_model(model):; symbols: test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor, check_model, touching `test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10 (34 lines); hunks: -35,7 +35,7 @@ def create_weights(self, layer: torch.nn.Module,; -63,16 +63,28 @@ def create_weights(self, layer: torch.nn.Module,; symbols: create_weights, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4 (10 lines); hunks: -21,7 +21,7 @@ class QuarkW8A8Fp8(QuarkScheme):; -41,10 +41,11 @@ def process_weights_after_loading(self, layer) -> None:; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, create_weights
  - `tests/quantization/test_quark.py` modified +37/-8 (45 lines); hunks: -4,17 +4,28; -26,11 +37,29 @@ def check_model(model):; symbols: test_quark_fp8, use_v0_only, test_quark_fp8_w_per_tensor_a_per_tensor, check_model
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +24/-10; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +6/-4
  - tests: `tests/quantization/test_quark.py` modified +37/-8
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15734 - [Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization

- Link: https://github.com/vllm-project/vllm/pull/15734
- Status/date: merged / 2025-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `a41351f363f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +105/-20, 212 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20 (47 lines); hunks: -1,7 +1,6; -125,6 +124,13 @@ def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":; symbols: from_config, get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod, touching `from_config, get_cache_scale, has_fp8_layer_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20 (47 lines); hunks: -1,7 +1,6; -125,6 +124,13 @@ def from_config(cls, config: Dict[str, Any]) -> "QuarkConfig":; symbols: from_config, get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +27/-20
- Risk and verification: Runtime changes concentrate in `vllm/attention/backends/abstract.py`, `vllm/attention/backends/rocm_flash_attn.py`, `vllm/attention/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17215 - [AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary

- Link: https://github.com/vllm-project/vllm/pull/17215
- Status/date: merged / 2025-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `68af5f6c5ce4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +0/-29, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[AMD][FP8][BugFix] Remove V1 check in arg_utils.py for FP8 since it is not necessary"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12 (12 lines); hunks: -307,18 +307,6 @@ def get_cache_scale(self, name: str) -> Optional[str]:; symbols: get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod, touching `get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12 (12 lines); hunks: -307,18 +307,6 @@ def get_cache_scale(self, name: str) -> Optional[str]:; symbols: get_cache_scale, has_fp8_layer_weights, QuarkLinearMethod
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-12
- Risk and verification: Runtime changes concentrate in `vllm/engine/arg_utils.py`, `vllm/model_executor/layers/quantization/quark/quark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16943 - [Quantization] Quark MXFP4 format loading

- Link: https://github.com/vllm-project/vllm/pull/16943
- Status/date: merged / 2025-05-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`; associated commits `db593aa67f8d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +289/-3, 378 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Quark MXFP4 format loading"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`; technical summary: Covers "[Quantization] Quark MXFP4 format loading"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1 (56 lines); hunks: -5,6 +5,7; -15,13 +16,15; symbols: QuarkConfig, get_quant_method, _is_static_tensor_w8a8, _is_mx_fp4, touching `QuarkConfig, get_quant_method, _is_static_tensor_w8a8`; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1 (3 lines); hunks: -1,7 +1,8.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1 (56 lines); hunks: -5,6 +5,7; -15,13 +16,15; symbols: QuarkConfig, get_quant_method, _is_static_tensor_w8a8, _is_mx_fp4
  - `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1 (3 lines); hunks: -1,7 +1,8
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +55/-1; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12612 - [Bugfix] Fix quark fp8 format loading on AMD GPUs

- Link: https://github.com/vllm-project/vllm/pull/12612
- Status/date: merged / 2025-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `bb239a730f2a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +38/-9, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix quark fp8 format loading on AMD GPUs"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`; technical summary: Covers "[Bugfix] Fix quark fp8 format loading on AMD GPUs"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9 (21 lines); hunks: -34,21 +34,24 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading, touching `process_weights_after_loading`; `tests/quantization/test_quark.py` modified +26/-0 (26 lines); hunks: -5,6 +5,7; -63,3 +64,28 @@ def check_model(model):; symbols: check_model, test_quark_fp8_parity, touching `check_model, test_quark_fp8_parity`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9 (21 lines); hunks: -34,21 +34,24 @@ def process_weights_after_loading(self, layer) -> None:; symbols: process_weights_after_loading
  - `tests/quantization/test_quark.py` modified +26/-0 (26 lines); hunks: -5,6 +5,7; -63,3 +64,28 @@ def check_model(model):; symbols: check_model, test_quark_fp8_parity
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +12/-9
  - tests: `tests/quantization/test_quark.py` modified +26/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20251 - [Bugfix] fix quark ptpc

- Link: https://github.com/vllm-project/vllm/pull/20251
- Status/date: merged / 2025-06-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `1c50e100a9c5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +23/-16, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix quark ptpc"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[Bugfix] fix quark ptpc"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11 (33 lines); hunks: -1,7 +1,7; -19,10 +19,19; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, touching `QuarkW8A8Fp8, __init__, process_weights_after_loading`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5 (6 lines); hunks: -312,11 +312,7 @@ def _get_scheme_from_config(self, config: dict[str, Any]) -...; symbols: _get_scheme_from_config, touching `_get_scheme_from_config`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11 (33 lines); hunks: -1,7 +1,7; -19,10 +19,19; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5 (6 lines); hunks: -312,11 +312,7 @@ def _get_scheme_from_config(self, config: dict[str, Any]) -...; symbols: _get_scheme_from_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +22/-11; `vllm/model_executor/layers/quantization/quark/quark.py` modified +1/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17888 - [Feature][Quantization] MXFP4 support for MOE models

- Link: https://github.com/vllm-project/vllm/pull/17888
- Status/date: merged / 2025-07-09
- Trace source: `git log --name-only -- <model-files>` found it through `docs/features/quantization/quark.md`, `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `332d4cb17b7c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +875/-106, 1416 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature][Quantization] MXFP4 support for MOE models"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`; technical summary: Covers "[Feature][Quantization] MXFP4 support for MOE models"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3 (170 lines); hunks: -5,19 +5,22; -40,6 +43,8 @@ def get_moe_method(; symbols: QuarkMoEMethod, get_moe_method, apply, QuarkW4A4MXFp4MoEMethod, touching `QuarkMoEMethod, get_moe_method, apply`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6 (6 lines); hunks: -237,12 +237,6 @@ def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],; symbols: _is_mx_fp4, touching `_is_mx_fp4`; `tests/quantization/test_quark.py` modified +171/-0 (171 lines); hunks: -3,15 +3,44; -90,3 +119,145 @@ def test_quark_fp8_parity(vllm_runner):; symbols: use_v0_only, test_quark_fp8_parity, ModelCase, GSM8KAccuracyTestConfig, touching `use_v0_only, test_quark_fp8_parity, ModelCase`; `docs/features/quantization/quark.md` modified +25/-0 (25 lines); hunks: -232,3 +232,28 @@ python3 quantize_quark.py --model_dir meta-llama/Llama-2-70....
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3 (170 lines); hunks: -5,19 +5,22; -40,6 +43,8 @@ def get_moe_method(; symbols: QuarkMoEMethod, get_moe_method, apply, QuarkW4A4MXFp4MoEMethod
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6 (6 lines); hunks: -237,12 +237,6 @@ def _is_mx_fp4(self, weight_quant: Optional[dict[str, Any]],; symbols: _is_mx_fp4
  - `tests/quantization/test_quark.py` modified +171/-0 (171 lines); hunks: -3,15 +3,44; -90,3 +119,145 @@ def test_quark_fp8_parity(vllm_runner):; symbols: use_v0_only, test_quark_fp8_parity, ModelCase, GSM8KAccuracyTestConfig
  - `docs/features/quantization/quark.md` modified +25/-0 (25 lines); hunks: -232,3 +232,28 @@ python3 quantize_quark.py --model_dir meta-llama/Llama-2-70...
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +167/-3; `vllm/model_executor/layers/quantization/quark/quark.py` modified +0/-6
  - tests: `tests/quantization/test_quark.py` modified +171/-0
  - docs: `docs/features/quantization/quark.md` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe.py`, `tests/kernels/moe/test_mxfp4_moe.py`, `tests/quantization/reference_mxfp4.py`, `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19830 - [Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf

- Link: https://github.com/vllm-project/vllm/pull/19830
- Status/date: merged / 2025-07-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `31d5c1797f32`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +368/-104, 858 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; technical summary: Covers "[Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5 (16 lines); hunks: -7,6 +7,8; -28,10 +30,14 @@ def __init__(self, weight_config: dict[str, Any],; symbols: __init__, process_weights_after_loading, touching `__init__, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5 (16 lines); hunks: -7,6 +7,8; -28,10 +30,14 @@ def __init__(self, weight_config: dict[str, Any],; symbols: __init__, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +11/-5
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion.py`, `tests/compile/test_fusion_attn.py`, `tests/compile/test_silu_mul_quant_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20893 - [Quantization] add BNB for MixtralForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/20893
- Status/date: merged / 2025-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `a99b9f7dee0a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +128/-20, 251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] add BNB for MixtralForCausalLM"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[Quantization] add BNB for MixtralForCausalLM"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +13/-8 (21 lines); hunks: -317,6 +317,15 @@ def forward(; -326,16 +335,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights, touching `forward, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +13/-8 (21 lines); hunks: -317,6 +317,15 @@ def forward(; -326,16 +335,9 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: forward, get_expert_mapping, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +13/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/models/granitemoe.py`, `vllm/model_executor/models/granitemoeshared.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22035 - [Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods.

- Link: https://github.com/vllm-project/vllm/pull/22035
- Status/date: merged / 2025-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `8ad7285ea28a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 54 files, +2022/-1305, 5535 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods."; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Kernels] Clean up FusedMoeMethodBase and modular kernel setup. Remove extra arguments from modular kernel methods."; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9 (39 lines); hunks: -7,7 +7,8; -25,6 +26,9; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW8A8Fp8MoEMethod, touching `QuarkMoEMethod, __init__, get_moe_method`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9 (39 lines); hunks: -7,7 +7,8; -25,6 +26,9; symbols: QuarkMoEMethod, __init__, get_moe_method, QuarkW8A8Fp8MoEMethod
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-9
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py`, `tests/kernels/moe/modular_kernel_tools/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23123 - Add routed_scaling_factor to MoE grouped topk

- Link: https://github.com/vllm-project/vllm/pull/23123
- Status/date: merged / 2025-08-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `8fb85b7bb674`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +77/-4, 570 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add routed_scaling_factor to MoE grouped topk"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "Add routed_scaling_factor to MoE grouped topk"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0 (4 lines); hunks: -218,6 +218,7 @@ def apply(; -244,6 +245,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0 (4 lines); hunks: -218,6 +218,7 @@ def apply(; -244,6 +245,7 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24649 - [Rocm] [quantization] Fix quark ptpc moe and add test case

- Link: https://github.com/vllm-project/vllm/pull/24649
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `ca2d1925ef5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +196/-52, 325 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Rocm] [quantization] Fix quark ptpc moe and add test case"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/quantization/test_quark.py`; technical summary: Covers "[Rocm] [quantization] Fix quark ptpc moe and add test case"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52 (223 lines); hunks: -5,17 +5,25; -67,21 +75,45 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, apply, touching `__init__, create_weights, process_weights_after_loading`; `tests/quantization/test_quark.py` modified +25/-0 (25 lines); hunks: -77,6 +77,31 @@ def check_model(model):; symbols: check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor, touching `check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52 (223 lines); hunks: -5,17 +5,25; -67,21 +75,45 @@ def __init__(; symbols: __init__, create_weights, process_weights_after_loading, apply
  - `tests/quantization/test_quark.py` modified +25/-0 (25 lines); hunks: -77,6 +77,31 @@ def check_model(model):; symbols: check_model, test_quark_fp8_w_per_channel_a_per_token, test_quark_int8_w_per_tensor_a_per_tensor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +171/-52
  - tests: `tests/quantization/test_quark.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22842 - [EPLB] Support EPLB for Mixtral Model

- Link: https://github.com/vllm-project/vllm/pull/22842
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `b77bf34e531a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +137/-23, 265 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[EPLB] Support EPLB for Mixtral Model"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[EPLB] Support EPLB for Mixtral Model"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunks: -23,7 +23,8; -33,8 +34,9; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunks: -23,7 +23,8; -33,8 +34,9; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +137/-23
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22537 - [Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses

- Link: https://github.com/vllm-project/vllm/pull/22537
- Status/date: merged / 2025-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `5963b98b4650`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 68 files, +2675/-2503, 9071 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19 (46 lines); hunks: -11,6 +11,9; -287,6 +290,16 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply, touching `process_weights_after_loading, get_fused_moe_quant_config, apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19 (46 lines); hunks: -11,6 +11,9; -287,6 +290,16 @@ def process_weights_after_loading(self, layer: torch.nn.Mod...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +27/-19
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/make_feature_matrix.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/test_batched_deepgemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21166 - [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4

- Link: https://github.com/vllm-project/vllm/pull/21166
- Status/date: merged / 2025-10-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +658/-182, 1462 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`; technical summary: Covers "[Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4"; the main implementation surface is `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: _quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6, _dequant_mxfp6_fake, touching `_quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6`; `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45 (140 lines); hunks: -1,22 +1,34; -96,23 +108,57 @@ def gemm_with_dynamic_quant_fake(; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4, QuarkOCP_MX, touching `is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4`; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21 (90 lines); hunks: -42,6 +42,8; -1323,7 +1325,7 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts, touching `inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts`; `vllm/model_executor/layers/fused_moe/config.py` modified +56/-17 (73 lines); hunks: -9,6 +9,10; -30,7 +34,7 @@ def _get_config_dtype_str(; symbols: _get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4, ocp_mx_scheme, touching `_get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: _quant_dequant_mxfp6, _quant_dequant_mxfp6_fake, _dequant_mxfp6, _dequant_mxfp6_fake
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45 (140 lines); hunks: -1,22 +1,34; -96,23 +108,57 @@ def gemm_with_dynamic_quant_fake(; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant_fake, QuarkW4A4MXFP4, QuarkOCP_MX
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21 (90 lines); hunks: -42,6 +42,8; -1323,7 +1325,7 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `vllm/model_executor/layers/fused_moe/config.py` modified +56/-17 (73 lines); hunks: -9,6 +9,10; -30,7 +34,7 @@ def _get_config_dtype_str(; symbols: _get_config_dtype_str, use_int4_w4a16, use_mxfp4_w4a4, ocp_mx_scheme
  - `vllm/model_executor/layers/fused_moe/utils.py` modified +60/-13 (73 lines); hunks: -16,11 +16,15; -106,12 +110,14 @@ def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) ->...; symbols: _resize_cache, _fp4_quantize, _nvfp4_quantize, _fp8_quantize
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py` added +142/-0; `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` renamed +95/-45; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +69/-21; `vllm/model_executor/layers/fused_moe/config.py` modified +56/-17; `vllm/model_executor/layers/fused_moe/utils.py` modified +60/-13; `vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py` added +54/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_ocp_mx_moe.py`, `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26364 - [ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py`

- Link: https://github.com/vllm-project/vllm/pull/26364
- Status/date: merged / 2025-10-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`; associated commits `a38c1bfe09f4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py`"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `tests/quantization/test_quark.py`; technical summary: Covers "[ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py`"; the main implementation surface is `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/quantization/test_quark.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7.
- Code diff details:
  - `tests/quantization/test_quark.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7
- Key code excerpts:

```diff
diff -- tests/quantization/test_quark.py
@@ -4,7 +4,7 @@
-See also `tests/kernels/moe/test_mxfp4_moe.py`.
+See also `tests/kernels/moe/test_ocp_mx_moe.py`.
```

- Reviewed files:
  - tests: `tests/quantization/test_quark.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26739 - [torch.compile] Unwrap fused_marlin_moe custom op

- Link: https://github.com/vllm-project/vllm/pull/26739
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `8ae169286f93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +22/-52, 228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile] Unwrap fused_marlin_moe custom op"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[torch.compile] Unwrap fused_marlin_moe custom op"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -20,6 +20,7; -402,7 +403,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -20,6 +20,7; -402,7 +403,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -20,6 +20,7 @@
+from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
@@ -402,7 +403,7 @@ def apply(
-            return torch.ops.vllm.fused_marlin_moe(
+            return fused_marlin_moe(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26545 - [ROCM] MoE fp4 CK kernel

- Link: https://github.com/vllm-project/vllm/pull/26545
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `0925b28a8e92`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-24, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCM] MoE fp4 CK kernel"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCM] MoE fp4 CK kernel"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24 (92 lines); hunks: -23,6 +23,7; -472,22 +473,22 @@ def __init__(; symbols: __init__, get_packed_dim, create_weights, process_weights_after_loading, touching `__init__, get_packed_dim, create_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24 (92 lines); hunks: -23,6 +23,7; -472,22 +473,22 @@ def __init__(; symbols: __init__, get_packed_dim, create_weights, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +68/-24
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27029 - [Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic

- Link: https://github.com/vllm-project/vllm/pull/27029
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `e33ee23ee3cd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +14/-3, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -341,7 +341,8 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, touching `get_fused_moe_quant_config, apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -341,7 +341,8 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -341,7 +341,8 @@ def get_fused_moe_quant_config(
-            per_act_token_quant=self.weight_qscheme == "per_channel",
+            per_act_token_quant=self.input_qscheme == "per_channel",
+            per_out_ch_quant=self.weight_qscheme == "per_channel",
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/utils/w8a8_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27123 - [Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses.

- Link: https://github.com/vllm-project/vllm/pull/27123
- Status/date: merged / 2025-11-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `938772af03ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +271/-311, 1049 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses."; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses."; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28 (53 lines); hunks: -310,7 +310,6 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; -322,17 +321,11 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply, touching `process_weights_after_loading, get_fused_moe_quant_config, apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28 (53 lines); hunks: -310,7 +310,6 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; -322,17 +321,11 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config, apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +25/-28
- Risk and verification: Runtime changes concentrate in `vllm/distributed/device_communicators/base_device_communicator.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28322 - [CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness

- Link: https://github.com/vllm-project/vllm/pull/28322
- Status/date: merged / 2025-11-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_mixtral.py`; associated commits `6b2b9fd934af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-11, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `tests/lora/test_mixtral.py`; technical summary: Covers "[CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness"; the main implementation surface is `tests/lora/test_mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/lora/test_mixtral.py` modified +18/-11 (29 lines); hunks: -56,15 +56,22 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; symbols: test_mixtral_lora, check_outputs, touching `test_mixtral_lora, check_outputs`.
- Code diff details:
  - `tests/lora/test_mixtral.py` modified +18/-11 (29 lines); hunks: -56,15 +56,22 @@ def test_mixtral_lora(mixtral_lora_files, tp_size):; symbols: test_mixtral_lora, check_outputs
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/lora/test_mixtral.py` modified +18/-11
- Risk and verification: The diff ships test coverage in `tests/lora/test_mixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27474 - [Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model

- Link: https://github.com/vllm-project/vllm/pull/27474
- Status/date: merged / 2025-11-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `b06b9470ca88`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0 (12 lines); hunks: -458,6 +458,7 @@ def __init__(; -581,6 +582,17 @@ def process_weights_after_loading(self, layer):; symbols: __init__, process_weights_after_loading, get_fused_moe_quant_config, touching `__init__, process_weights_after_loading, get_fused_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0 (12 lines); hunks: -458,6 +458,7 @@ def __init__(; -581,6 +582,17 @@ def process_weights_after_loading(self, layer):; symbols: __init__, process_weights_after_loading, get_fused_moe_quant_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24239 - [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model

- Link: https://github.com/vllm-project/vllm/pull/24239
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs/features/quantization/quark.md`, `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `05576df85c52`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +127/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`, `docs/features/quantization/quark.md`; technical summary: Covers "[ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`, `docs/features/quantization/quark.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7 (32 lines); hunks: -114,7 +114,14 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":; -124,10 +131,15 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConf...; symbols: from_config, _find_matched_config, _matches_pattern, touching `from_config, _find_matched_config, _matches_pattern`; `docs/features/quantization/quark.md` modified +33/-1 (34 lines); hunks: -281,4 +281,36 @@ python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B....
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7 (32 lines); hunks: -114,7 +114,14 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":; -124,10 +131,15 @@ def from_config(cls, config: dict[str, Any]) -> "QuarkConf...; symbols: from_config, _find_matched_config, _matches_pattern
  - `docs/features/quantization/quark.md` modified +33/-1 (34 lines); hunks: -281,4 +281,36 @@ python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B...
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +25/-7
  - docs: `docs/features/quantization/quark.md` modified +33/-1
- Risk and verification: The diff ships test coverage in `tests/quantization/test_mixed_precision.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28275 - [Misc] add ignore mapper for quark quantization

- Link: https://github.com/vllm-project/vllm/pull/28275
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `0b25498990f0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-3, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] add ignore mapper for quark quantization"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[Misc] add ignore mapper for quark quantization"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3 (12 lines); hunks: -2,7 +2,7; -34,6 +34,9; symbols: __init__, get_linear_method, get_quant_method, apply_vllm_mapper, touching `__init__, get_linear_method, get_quant_method`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3 (12 lines); hunks: -2,7 +2,7; -34,6 +34,9; symbols: __init__, get_linear_method, get_quant_method, apply_vllm_mapper
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28638 - [ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss

- Link: https://github.com/vllm-project/vllm/pull/28638
- Status/date: merged / 2025-11-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `d0a73620cc85`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +30/-5, 70 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5 (35 lines); hunks: -32,6 +32,7; -57,7 +58,6 @@ def __init__(; symbols: __init__, get_linear_method, get_min_capability, get_name, touching `__init__, get_linear_method, get_min_capability`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5 (35 lines); hunks: -32,6 +32,7; -57,7 +58,6 @@ def __init__(; symbols: __init__, get_linear_method, get_min_capability, get_name
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +30/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29067 - [MoE][Refactor] Make select_experts a non-static method

- Link: https://github.com/vllm-project/vllm/pull/29067
- Status/date: merged / 2025-11-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `8f066146c395`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +163/-472, 1200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE][Refactor] Make select_experts a non-static method"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE][Refactor] Make select_experts a non-static method"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34 (38 lines); hunks: -334,7 +334,7 @@ def get_fused_moe_quant_config(; -355,24 +355,9 @@ def apply(; symbols: get_fused_moe_quant_config, apply, allow_inplace, touching `get_fused_moe_quant_config, apply, allow_inplace`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34 (38 lines); hunks: -334,7 +334,7 @@ def get_fused_moe_quant_config(; -355,24 +355,9 @@ def apply(; symbols: get_fused_moe_quant_config, apply, allow_inplace
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-34
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_flashinfer.py`, `tests/test_routing_simulator.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29773 - [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel

- Link: https://github.com/vllm-project/vllm/pull/29773
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `c6df05ebb499`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -633,6 +633,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -633,6 +633,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -633,6 +633,7 @@ def apply(
+                expert_map=expert_map,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29066 - [MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply

- Link: https://github.com/vllm-project/vllm/pull/29066
- Status/date: merged / 2025-12-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `00e5cbb96789`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +318/-872, 2081 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52 (72 lines); hunks: -1,7 +1,6; -337,23 +336,6 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52 (72 lines); hunks: -1,7 +1,6; -337,23 +336,6 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-52
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30071 - [Quantization] Support Quark int4-fp8 w4a8 for MoE

- Link: https://github.com/vllm-project/vllm/pull/30071
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `0c738b58bc0e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +201/-2, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Support Quark int4-fp8 w4a8 for MoE"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[Quantization] Support Quark int4-fp8 w4a8 for MoE"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2 (160 lines); hunks: -63,8 +63,9 @@ def get_moe_method(; -396,6 +397,161 @@ def apply(; symbols: get_moe_method, apply, QuarkW4A8Fp8MoEMethod, __init__, touching `get_moe_method, apply, QuarkW4A8Fp8MoEMethod`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0 (43 lines); hunks: -218,6 +218,49 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8, touching `_check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2 (160 lines); hunks: -63,8 +63,9 @@ def get_moe_method(; -396,6 +397,161 @@ def apply(; symbols: get_moe_method, apply, QuarkW4A8Fp8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0 (43 lines); hunks: -218,6 +218,49 @@ def _check_scheme_supported(self, min_capability: int, erro...; symbols: _check_scheme_supported, _is_fp8_w4a8, _is_fp8_w8a8
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +158/-2; `vllm/model_executor/layers/quantization/quark/quark.py` modified +43/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28891 - [MoE Refactor][5/N] Isolate zero expert to LongCatFlash

- Link: https://github.com/vllm-project/vllm/pull/28891
- Status/date: merged / 2025-12-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `54c892438479`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +263/-108, 709 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor][5/N] Isolate zero expert to LongCatFlash"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor][5/N] Isolate zero expert to LongCatFlash"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -338,7 +338,7 @@ def apply(; -530,7 +530,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -338,7 +338,7 @@ def apply(; -530,7 +530,7 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/test_routing_simulator.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31499 - [MoE Refactor][12/N] Marlin Fp8 MoE Pure Function

- Link: https://github.com/vllm-project/vllm/pull/31499
- Status/date: merged / 2025-12-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `9152a30d8f6f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +92/-76, 270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor][12/N] Marlin Fp8 MoE Pure Function"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor][12/N] Marlin Fp8 MoE Pure Function"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4 (24 lines); hunks: -315,10 +315,26 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config, touching `process_weights_after_loading, get_fused_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4 (24 lines); hunks: -315,10 +315,26 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, get_fused_moe_quant_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +20/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31759 - [MoE Refactor] Add Temporary Integration Tests - H100/B200

- Link: https://github.com/vllm-project/vllm/pull/31759
- Status/date: merged / 2026-01-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml`; associated commits `d3e477c01342`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 30 files, +247/-0, 280 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Add Temporary Integration Tests - H100/B200"; model line: Mixtral Quark INT4/FP8 MoE; category: docs/tests/CI; main diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml`; technical summary: Covers "[MoE Refactor] Add Temporary Integration Tests - H100/B200"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0 (9 lines); hunks: -0,0 +1,9
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` added +9/-0; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-triton.yaml` added +5/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-marlin.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31415 - [MoE Refactor][15/N] Apply Refactor to Fp8

- Link: https://github.com/vllm-project/vllm/pull/31415
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `5dcd7ef1f219`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +1446/-1535, 4055 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor][15/N] Apply Refactor to Fp8"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor][15/N] Apply Refactor to Fp8"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4 (7 lines); hunks: -22,7 +22,7; -315,16 +315,15 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading, touching `process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4 (7 lines); hunks: -22,7 +22,7; -315,16 +315,15 @@ def process_weights_after_loading(self, layer: torch.nn.Mo...; symbols: process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-4
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm-deepep-ht.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm-deepep-ll.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30519 - [Misc][Refactor] Add FusedMoERouter object

- Link: https://github.com/vllm-project/vllm/pull/30519
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `e74698c27ad5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +165/-36, 744 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][Refactor] Add FusedMoERouter object"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Misc][Refactor] Add FusedMoERouter object"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2 (8 lines); hunks: -13,6 +13,7; -350,10 +351,11 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, allow_inplace, touching `get_fused_moe_quant_config, apply, allow_inplace`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2 (8 lines); hunks: -13,6 +13,7; -350,10 +351,11 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, allow_inplace
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `tests/test_routing_simulator.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31827 - [MoE Refactor][17/N] Apply Refactor to Bf16

- Link: https://github.com/vllm-project/vllm/pull/31827
- Status/date: merged / 2026-01-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml`; associated commits `31c29257c852`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +257/-87, 445 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor][17/N] Apply Refactor to Bf16"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml`; technical summary: Covers "[MoE Refactor][17/N] Apply Refactor to Bf16"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0 (7 lines); hunks: -0,0 +1,7; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0 (7 lines); hunks: -0,0 +1,7
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0 (5 lines); hunks: -0,0 +1,5
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` added +7/-0; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml` added +5/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-BF16-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/config-b200.txt`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-triton.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30623 - [MoE Refactor] Separate Router into OO Classes

- Link: https://github.com/vllm-project/vllm/pull/30623
- Status/date: merged / 2026-01-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `327a02d8db86`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 45 files, +1754/-688, 2998 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Separate Router into OO Classes"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Separate Router into OO Classes"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -548,7 +548,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -548,7 +548,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -548,7 +548,7 @@ def apply(
-        topk_weights, topk_ids = layer.select_experts(
+        topk_weights, topk_ids = router.select_experts(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_cutlass_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27814 - [Refactor] Make FP8 Linear Ops use kernel abstraction

- Link: https://github.com/vllm-project/vllm/pull/27814
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`; associated commits `148117ea2e68`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 30 files, +1458/-1029, 3752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Make FP8 Linear Ops use kernel abstraction"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`; technical summary: Covers "[Refactor] Make FP8 Linear Ops use kernel abstraction"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19 (47 lines); hunks: -7,10 +7,18; -23,6 +31,8; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading, touching `QuarkW8A8Fp8, __init__, process_weights_after_loading`; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22 (32 lines); hunks: -7,8 +7,7; -22,8 +21,6; symbols: QuarkW8A8Int8, __init__, create_weights, touching `QuarkW8A8Int8, __init__, create_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19 (47 lines); hunks: -7,10 +7,18; -23,6 +31,8; symbols: QuarkW8A8Fp8, __init__, process_weights_after_loading
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22 (32 lines); hunks: -7,8 +7,7; -22,8 +21,6; symbols: QuarkW8A8Int8, __init__, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +28/-19; `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py` modified +10/-22
- Risk and verification: The diff ships test coverage in `tests/compile/distributed/test_fusion_all_reduce.py`, `tests/compile/distributed/test_sequence_parallelism.py`, `tests/compile/test_functionalization.py`, `tests/compile/test_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31996 - [MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE`

- Link: https://github.com/vllm-project/vllm/pull/31996
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `dc917cceb877`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +495/-530, 1822 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE`"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE`"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23 (29 lines); hunks: -13,7 +13,6; -351,15 +350,10 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply, touching `get_fused_moe_quant_config, apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23 (29 lines); hunks: -13,7 +13,6; -351,15 +350,10 @@ def get_fused_moe_quant_config(; symbols: get_fused_moe_quant_config, apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +6/-23
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32954 - [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe

- Link: https://github.com/vllm-project/vllm/pull/32954
- Status/date: merged / 2026-01-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +290/-17, 452 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`; technical summary: Covers "[NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe"; the main implementation surface is `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0 (103 lines); hunks: -77,6 +77,18 @@ def _supports_routing_method(; -115,6 +127,34 @@ def _make_reason(reason: str) -> str:; symbols: _supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config, _make_reason, touching `_supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config`; `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11 (85 lines); hunks: -6,6 +6,7; -32,6 +33,9; symbols: UnquantizedFusedMoEMethod, __init__, _select_monolithic, forward_native, touching `UnquantizedFusedMoEMethod, __init__, _select_monolithic`; `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0 (75 lines); hunks: -214,6 +214,81 @@ def is_flashinfer_supporting_global_sf(backend: FlashinferM...; symbols: is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi, touching `is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi`; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6 (41 lines); hunks: -14,19 +14,23; -40,6 +44,7 @@ class UnquantizedMoeBackend(Enum):; symbols: UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend, touching `UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0 (103 lines); hunks: -77,6 +77,18 @@ def _supports_routing_method(; -115,6 +127,34 @@ def _make_reason(reason: str) -> str:; symbols: _supports_routing_method, _supports_routing_method_bf16, _supports_parallel_config, _make_reason
  - `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11 (85 lines); hunks: -6,6 +6,7; -32,6 +33,9; symbols: UnquantizedFusedMoEMethod, __init__, _select_monolithic, forward_native
  - `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0 (75 lines); hunks: -214,6 +214,81 @@ def is_flashinfer_supporting_global_sf(backend: FlashinferM...; symbols: is_flashinfer_supporting_global_sf, convert_moe_weights_to_flashinfer_trtllm_block_layout, align_fp8_moe_weights_for_fi
  - `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6 (41 lines); hunks: -14,19 +14,23; -40,6 +44,7 @@ class UnquantizedMoeBackend(Enum):; symbols: UnquantizedMoeBackend, select_unquantized_moe_backend, _make_log_backend
  - `vllm/utils/flashinfer.py` modified +3/-0 (3 lines); hunks: -105,6 +105,9 @@ def wrapper(*args, **kwargs):; symbols: wrapper
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +103/-0; `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` modified +74/-11; `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +75/-0; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +35/-6; `vllm/utils/flashinfer.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`, `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33375 - [Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor

- Link: https://github.com/vllm-project/vllm/pull/33375
- Status/date: merged / 2026-02-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `a57c8228ffb3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 37 files, +132/-109, 869 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6 (9 lines); hunks: -388,6 +388,7 @@ def apply(; -398,7 +399,7 @@ def apply(; symbols: apply, get_fused_moe_quant_config, allow_inplace, touching `apply, get_fused_moe_quant_config, allow_inplace`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6 (9 lines); hunks: -388,6 +388,7 @@ def apply(; -398,7 +399,7 @@ def apply(; symbols: apply, get_fused_moe_quant_config, allow_inplace
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_cutlass_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29008 - [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations

- Link: https://github.com/vllm-project/vllm/pull/29008
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `b129136c7a73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1094/-213, 1860 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "[ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54 (352 lines); hunks: -8,6 +8,7; -18,9 +19,15; symbols: QuarkMoEMethod, __init__, get_moe_method, touching `QuarkMoEMethod, __init__, get_moe_method`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23 (70 lines); hunks: -320,38 +320,45 @@ def _is_static_tensor_w8a8(; -360,14 +367,31 @@ def _is_ocp_mx(; symbols: _is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x, is_mxfp4_quant, touching `_is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54 (352 lines); hunks: -8,6 +8,7; -18,9 +19,15; symbols: QuarkMoEMethod, __init__, get_moe_method
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23 (70 lines); hunks: -320,38 +320,45 @@ def _is_static_tensor_w8a8(; -360,14 +367,31 @@ def _is_ocp_mx(; symbols: _is_static_tensor_w8a8, _is_ocp_mx, _is_w_ocp_mx_a_x, is_mxfp4_quant
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +298/-54; `vllm/model_executor/layers/quantization/quark/quark.py` modified +47/-23
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/models/quantization/test_gpt_oss.py`, `tests/models/quantization/test_gpt_oss_attn_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32344 - [MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner

- Link: https://github.com/vllm-project/vllm/pull/32344
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `d1481ba78323`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +913/-753, 2246 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0 (3 lines); hunks: -419,6 +419,7 @@ def apply(; -607,6 +608,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0 (3 lines); hunks: -419,6 +419,7 @@ def apply(; -607,6 +608,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -419,6 +419,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
@@ -607,6 +608,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
@@ -977,6 +979,7 @@ def apply(
+        shared_experts_input: torch.Tensor | None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33715 - [NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE

- Link: https://github.com/vllm-project/vllm/pull/33715
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`; associated commits `275e0d2a993b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +296/-1, 337 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`; technical summary: Covers "[NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0 (1 lines); hunks: -5,3 +5,4 @@ num_fewshot: 5.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0 (1 lines); hunks: -5,3 +5,4 @@ num_fewshot: 5
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml
@@ -5,3 +5,4 @@ num_fewshot: 5
+  VLLM_FLASHINFER_MOE_BACKEND: "throughput"
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/kernels/moe/test_flashinfer.py`, `tests/kernels/moe/test_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33843 - [Refactor] Replace `activation: str` with `MoEActivation` enum

- Link: https://github.com/vllm-project/vllm/pull/33843
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `ff1f83b056ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 48 files, +474/-282, 2289 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Replace `activation: str` with `MoEActivation` enum"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Refactor] Replace `activation: str` with `MoEActivation` enum"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -15,6 +15,7; -438,7 +439,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -15,6 +15,7; -438,7 +439,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -15,6 +15,7 @@
+    MoEActivation,
@@ -438,7 +439,7 @@ def apply(
-            assert layer.activation == "silu", (
+            assert layer.activation == MoEActivation.SILU, (
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_cpu_fused_moe.py`, `tests/kernels/moe/test_cutlass_moe.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34192 - [ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter

- Link: https://github.com/vllm-project/vllm/pull/34192
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `8a798be929d6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0 (8 lines); hunks: -933,7 +933,15 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, get_fused_moe_quant_config, touching `process_weights_after_loading, get_fused_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0 (8 lines); hunks: -933,7 +933,15 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, get_fused_moe_quant_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34386 - [Quark] Fix MoE fp8 activation scale handling on mi300

- Link: https://github.com/vllm-project/vllm/pull/34386
- Status/date: merged / 2026-02-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `d9e62c03eb98`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quark] Fix MoE fp8 activation scale handling on mi300"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Quark] Fix MoE fp8 activation scale handling on mi300"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -858,7 +858,7 @@ def create_weights(; -883,14 +883,14 @@ def process_weights_after_loading(self, layer):; symbols: create_weights, process_weights_after_loading, touching `create_weights, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -858,7 +858,7 @@ def create_weights(; -883,14 +883,14 @@ def process_weights_after_loading(self, layer):; symbols: create_weights, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33807 - [UX] Add `--moe-backend` arg for explicit kernel selection

- Link: https://github.com/vllm-project/vllm/pull/33807
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`; associated commits `de527e1cec82`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 37 files, +260/-140, 720 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[UX] Add `--moe-backend` arg for explicit kernel selection"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`; technical summary: Covers "[UX] Add `--moe-backend` arg for explicit kernel selection"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -2,7 +2,4 @@ model_name: "mistralai/Mixtral-8x7B-v0.1"; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -3,7 +3,4.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -2,7 +2,4 @@ model_name: "mistralai/Mixtral-8x7B-v0.1"
  - `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4 (5 lines); hunks: -3,7 +3,4
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml` modified +1/-4; `tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-Fp8-AutoFp8-fi-cutlass.yaml` modified +1/-4
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Qwen3-Next-80B-A3B-NVFP4-EP2.yaml`, `tests/evals/gsm8k/configs/Qwen3-Next-FP8-EP2.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Llama-4-Scout-Fp8-ModelOpt-triton.yaml`, `tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-NvFp4-CT-fi-cutedsl-deepep-ll.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30357 - [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales

- Link: https://github.com/vllm-project/vllm/pull/30357
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `01923eec7092`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +315/-37, 494 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29 (202 lines); hunks: -5,8 +5,8; -32,6 +32,7; symbols: QuarkMoEMethod, get_moe_method, __init__, create_weights, touching `QuarkMoEMethod, get_moe_method, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29 (202 lines); hunks: -5,8 +5,8; -32,6 +32,7; symbols: QuarkMoEMethod, get_moe_method, __init__, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +173/-29
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35658 - [ROCm] add amd-quark package in requirements for rocm to use quantized models

- Link: https://github.com/vllm-project/vllm/pull/35658
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`; associated commits `f26650d649aa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +24/-6, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] add amd-quark package in requirements for rocm to use quantized models"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `tests/quantization/test_quark.py`; technical summary: Covers "[ROCm] add amd-quark package in requirements for rocm to use quantized models"; the main implementation surface is `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark, touching `get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness`.
- Code diff details:
  - `tests/quantization/test_quark.py` modified +20/-5 (25 lines); hunks: -26,9 +26,12; -200,7 +203,10 @@ def get_model_args(; symbols: get_model_args, test_ocp_mx_wikitext_correctness, test_mxfp4_gsm8k_correctness, test_mxfp4_fused_qdq_match_quark
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
```

- Reviewed files:
  - tests: `tests/quantization/test_quark.py` modified +20/-5
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35893 - [ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported

- Link: https://github.com/vllm-project/vllm/pull/35893
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `5dc3538736e4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +63/-1, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1 (31 lines); hunks: -32,7 +32,10; -732,6 +735,32 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1 (31 lines); hunks: -32,7 +32,10; -732,6 +735,32 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +30/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/mxfp4.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35316 - [ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer

- Link: https://github.com/vllm-project/vllm/pull/35316
- Status/date: merged / 2026-03-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`; associated commits `6341d4304351`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +311/-1, 362 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`; technical summary: Covers "[ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: QuarkW4A8_MXFP4_FP8, __init__, get_min_capability, get_packed_dim, touching `QuarkW4A8_MXFP4_FP8, __init__, get_min_capability`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0 (32 lines); hunks: -26,6 +26,7; -350,6 +351,31 @@ def _is_static_tensor_w8a8(; symbols: _is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x, _get_scheme_from_config, touching `_is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x`; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1 (9 lines); hunks: -3,7 +3,14.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: QuarkW4A8_MXFP4_FP8, __init__, get_min_capability, get_packed_dim
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0 (32 lines); hunks: -26,6 +26,7; -350,6 +351,31 @@ def _is_static_tensor_w8a8(; symbols: _is_static_tensor_w8a8, _is_w4a8_mxfp4_fp8, _is_w_ocp_mx_a_x, _get_scheme_from_config
  - `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1 (9 lines); hunks: -3,7 +3,14
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py` added +218/-0; `vllm/model_executor/layers/quantization/quark/quark.py` modified +32/-0; `vllm/model_executor/layers/quantization/quark/schemes/__init__.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/schemes/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32779 - Fix infinite recursive search issue in quark.py

- Link: https://github.com/vllm-project/vllm/pull/32779
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark.py`; associated commits `24b4272a8ca6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-4, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix infinite recursive search issue in quark.py"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark.py`; technical summary: Covers "Fix infinite recursive search issue in quark.py"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4 (15 lines); hunks: -467,10 +467,17 @@ def _find_matched_config(; symbols: _find_matched_config, touching `_find_matched_config`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4 (15 lines); hunks: -467,10 +467,17 @@ def _find_matched_config(; symbols: _find_matched_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark.py` modified +11/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36232 - [ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization

- Link: https://github.com/vllm-project/vllm/pull/36232
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`; associated commits `44eea10f6846`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-5, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4 (21 lines); hunks: -176,7 +176,7 @@ class QuarkOCP_MX(QuarkScheme):; -185,7 +185,13 @@ def __init__(; symbols: QuarkOCP_MX, __init__, touching `QuarkOCP_MX, __init__`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def get_moe_method(; symbols: get_moe_method, touching `get_moe_method`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4 (21 lines); hunks: -176,7 +176,7 @@ class QuarkOCP_MX(QuarkScheme):; -185,7 +185,13 @@ def __init__(; symbols: QuarkOCP_MX, __init__
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def get_moe_method(; symbols: get_moe_method
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +17/-4; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37128 - [MoE Refactor] Mxfp4 oracle rebased

- Link: https://github.com/vllm-project/vllm/pull/37128
- Status/date: merged / 2026-03-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `87bd91892f8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1695/-1369, 3508 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Mxfp4 oracle rebased"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Mxfp4 oracle rebased"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -25,9 +25,9; -698,9 +698,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -25,9 +25,9; -698,9 +698,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_ocp_mx_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36100 - [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs

- Link: https://github.com/vllm-project/vllm/pull/36100
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`; associated commits `e99fb98867c2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +16/-26, 122 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20 (25 lines); hunks: -3,13 +3,12; -37,22 +36,6; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__, touching `is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__`; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -765,7 +765,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20 (25 lines); hunks: -3,13 +3,12; -37,22 +36,6; symbols: is_rocm_aiter_fp4_asm_gemm_enabled, gemm_with_dynamic_quant, __init__
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -765,7 +765,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` modified +5/-20; `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37833 - [ROCm] Fix MoE kernel test failures on gfx950

- Link: https://github.com/vllm-project/vllm/pull/37833
- Status/date: merged / 2026-03-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `7d6917bef552`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +480/-88, 928 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Fix MoE kernel test failures on gfx950"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm] Fix MoE kernel test failures on gfx950"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0 (15 lines); hunks: -741,11 +741,14 @@ def __init__(; -819,6 +822,18 @@ def create_weights(; symbols: __init__, create_weights, touching `__init__, create_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0 (15 lines); hunks: -741,11 +741,14 @@ def __init__(; -819,6 +822,18 @@ def create_weights(; symbols: __init__, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +15/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_kernel_combinations.py`, `tests/kernels/moe/test_routing.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34285 - [Refactor] Move FusedMoE hidden_size roundup to quant_method

- Link: https://github.com/vllm-project/vllm/pull/34285
- Status/date: merged / 2026-03-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `0ae89f18fd75`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +204/-222, 868 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Move FusedMoE hidden_size roundup to quant_method"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Refactor] Move FusedMoE hidden_size roundup to quant_method"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79 (130 lines); hunks: -18,6 +18,7; -27,13 +28,13; symbols: create_weights, apply, touching `create_weights, apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79 (130 lines); hunks: -18,6 +18,7; -27,13 +28,13; symbols: create_weights, apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +51/-79
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`, `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35153 - [MoE Refactor] Make SharedExperts class for use with DefaultMoERunner

- Link: https://github.com/vllm-project/vllm/pull/35153
- Status/date: merged / 2026-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `7cf56a59a267`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 34 files, +556/-397, 1816 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Make SharedExperts class for use with DefaultMoERunner"; model line: Mixtral Quark INT4/FP8 MoE; category: model implementation change; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Make SharedExperts class for use with DefaultMoERunner"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -444,7 +444,7 @@ def apply(; -634,7 +634,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3 (6 lines); hunks: -444,7 +444,7 @@ def apply(; -634,7 +634,7 @@ def apply(; symbols: apply
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38774 - [ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle

- Link: https://github.com/vllm-project/vllm/pull/38774
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `103f0de565f2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +170/-15, 316 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`; technical summary: Covers "[ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7 (153 lines); hunks: -5,6 +5,7; -27,7 +28,11; symbols: __init__, process_weights_after_loading, _setup_kernel_via_oracle, touching `__init__, process_weights_after_loading, _setup_kernel_via_oracle`; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7 (153 lines); hunks: -5,6 +5,7; -27,7 +28,11; symbols: __init__, process_weights_after_loading, _setup_kernel_via_oracle
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0 (0 lines)
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +146/-7
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` added +8/-0; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` added +6/-0; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` renamed +0/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`, `tests/evals/gpt_oss/configs/models-gfx950.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38504 - [Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path

- Link: https://github.com/vllm-project/vllm/pull/38504
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`; associated commits `2df2c85be494`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +84/-216, 444 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`; technical summary: Covers "[Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path"; the main implementation surface is `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2 (4 lines); hunks: -3,6 +3,6; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1 (2 lines); hunks: -3,4 +3,4; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1 (2 lines); hunks: -3,6 +3,6.
- Code diff details:
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2 (4 lines); hunks: -3,6 +3,6
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1 (2 lines); hunks: -3,4 +3,4
  - `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1 (2 lines); hunks: -3,6 +3,6
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml` modified +2/-2; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml` modified +1/-1; `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-baseline.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-aiter.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-bf16-triton.yaml`, `tests/evals/gpt_oss/configs/gpt-oss-20b-rocm-quark-mxfp4-fp8-triton.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33892 - [W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections.

- Link: https://github.com/vllm-project/vllm/pull/33892
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; associated commits `2e9034c998e2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +1710/-904, 3623 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections."; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`; technical summary: Covers "[W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections."; the main implementation surface is `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1 (6 lines); hunks: -7,6 +7,7; -57,6 +58,7 @@ def __init__(; symbols: __init__, get_min_capability, create_weights, touching `__init__, get_min_capability, create_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1 (6 lines); hunks: -7,6 +7,7; -57,6 +58,7 @@ def __init__(; symbols: __init__, get_min_capability, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `tests/compile/passes/distributed/test_fusion_all_reduce.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`, `tests/compile/passes/test_functionalization.py`, `tests/compile/passes/test_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36320 - [Quantization] Support Quark W8A8 INT8 MoE inference

- Link: https://github.com/vllm-project/vllm/pull/36320
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `tests/quantization/test_quark.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `827268e98d92`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +360/-2, 411 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] Support Quark W8A8 INT8 MoE inference"; model line: Mixtral Quark INT4/FP8 MoE; category: model support/runtime entry; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`; technical summary: Covers "[Quantization] Support Quark W8A8 INT8 MoE inference"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/quark.py`, `tests/quantization/test_quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0 (282 lines); hunks: -109,6 +109,12 @@ def get_moe_method(; -505,6 +511,282 @@ def apply(; symbols: get_moe_method, apply, QuarkW8A8Int8MoEMethod, __init__, touching `get_moe_method, apply, QuarkW8A8Int8MoEMethod`; `vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0 (38 lines); hunks: -389,6 +389,37 @@ def _is_w4a8_mxfp4_fp8(; -556,6 +587,13 @@ def _get_scheme_from_config(; symbols: _is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x, _get_scheme_from_config, touching `_is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x`; `tests/quantization/test_quark.py` modified +31/-0 (31 lines); hunks: -22,6 +22,9; -126,6 +129,34 @@ def check_model(model):; symbols: check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity, touching `check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0 (282 lines); hunks: -109,6 +109,12 @@ def get_moe_method(; -505,6 +511,282 @@ def apply(; symbols: get_moe_method, apply, QuarkW8A8Int8MoEMethod, __init__
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0 (38 lines); hunks: -389,6 +389,37 @@ def _is_w4a8_mxfp4_fp8(; -556,6 +587,13 @@ def _get_scheme_from_config(; symbols: _is_w4a8_mxfp4_fp8, _is_dynamic_per_token_w8a8, _is_w_ocp_mx_a_x, _get_scheme_from_config
  - `tests/quantization/test_quark.py` modified +31/-0 (31 lines); hunks: -22,6 +22,9; -126,6 +129,34 @@ def check_model(model):; symbols: check_model, test_quark_int8_w8a8_moe, test_quark_fp8_parity
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +282/-0; `vllm/model_executor/layers/quantization/quark/quark.py` modified +38/-0
  - tests: `tests/quantization/test_quark.py` modified +31/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39604 - [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod"

- Link: https://github.com/vllm-project/vllm/pull/39604
- Status/date: merged / 2026-04-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `739e5945dc4b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +129/-34, 431 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod""; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod""; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4 (8 lines); hunks: -30,11 +30,11; -995,7 +995,7 @@ def __init__(; symbols: __init__, _setup_kernel_via_oracle, touching `__init__, _setup_kernel_via_oracle`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4 (8 lines); hunks: -30,11 +30,11; -995,7 +995,7 @@ def __init__(; symbols: __init__, _setup_kernel_via_oracle
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39107 - [MoE Refactor] Remove MoE DP chunking

- Link: https://github.com/vllm-project/vllm/pull/39107
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `e1e318af010b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +76/-389, 782 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove MoE DP chunking"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE Refactor] Remove MoE DP chunking"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, touching `process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1502,9 +1502,9 @@ def process_weights_after_loading(self, layer):
-        # only apply to  batched mode
+        # only apply to batched mode
-            num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
+            num_warps = 4 if self.moe.max_num_tokens <= 512 else 8
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`, `tests/kernels/moe/test_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39007 - [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/

- Link: https://github.com/vllm-project/vllm/pull/39007
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `1a9353bb02e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +16/-12, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -1591,7 +1591,7 @@ def apply_monolithic(; symbols: apply_monolithic, touching `apply_monolithic`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1 (2 lines); hunks: -1591,7 +1591,7 @@ def apply_monolithic(; symbols: apply_monolithic
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1591,7 +1591,7 @@ def apply_monolithic(
-        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
+        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gpt_oss_triton_kernels.py`, `tests/kernels/moe/test_modular_oai_triton_moe.py`, `tests/kernels/quantization/test_mxfp4_triton_ep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `726efe177bf2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -132,7 +132,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +0/-1 (1 lines); hunks: -132,7 +132,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mixtral.py
@@ -132,7 +132,6 @@ def __init__(
-            reduce_results=True,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35737 - [NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation

- Link: https://github.com/vllm-project/vllm/pull/35737
- Status/date: merged / 2026-04-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `d622e27d2be9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +601/-121, 1072 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54 (97 lines); hunks: -30,6 +30,7; -986,6 +987,8 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, process_weights_after_loading, touching `__init__, maybe_roundup_sizes, process_weights_after_loading`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54 (97 lines); hunks: -30,6 +30,7; -986,6 +987,8 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +43/-54
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/models-mi3xx.txt`, `tests/models/quantization/test_nvfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mixtral.py`; associated commits `1c2c1eb8b9fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: Mixtral Quark INT4/FP8 MoE; category: performance/backend optimization; main diff: `vllm/model_executor/models/mixtral.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/models/mixtral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mixtral.py` modified +5/-2 (7 lines); hunks: -40,7 +40,10; -364,7 +367,7 @@ def forward(; symbols: forward, get_expert_mapping, touching `forward, get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +5/-2 (7 lines); hunks: -40,7 +40,10; -364,7 +367,7 @@ def forward(; symbols: forward, get_expert_mapping
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/mixtral.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41175 - [ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware

- Link: https://github.com/vllm-project/vllm/pull/41175
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `944e138bcf39`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-7, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-6 (8 lines); hunks: -1025,12 +1025,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-6 (8 lines); hunks: -1025,12 +1025,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1025,12 +1025,8 @@ def __init__(
-        # TODO(aiter): extend once rocm_aiter_fused_experts gains dispatch
-        # for the other OCP MX schemes. Today its CK MoE kernel only has an
-        # entry for `w_mxfp4` (w4a16); mixed schemes like `w_mxfp4_a_mxfp6_*`
-        # fall through to QuantMethod.NO and raise "Unsupported kernel config
-        # for moe heuristic dispatch".
-        _AITER_NATIVE_OCP_MX_SCHEMES = ("w_mxfp4",)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +2/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41165 - [ROCm][Bugfix][GPTOSS]: fix input_ids and expert_map args for quark w4a8 gptoss

- Link: https://github.com/vllm-project/vllm/pull/41165
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/quantization/quark/quark_moe.py`; associated commits `3795d7acf431`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Bugfix][GPTOSS]: fix input_ids and expert_map args for quark w4a8 gptoss"; model line: Mixtral Quark INT4/FP8 MoE; category: bug fix; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`; technical summary: Covers "[ROCm][Bugfix][GPTOSS]: fix input_ids and expert_map args for quark w4a8 gptoss"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -1577,14 +1577,14 @@ def is_monolithic(self) -> bool:; -1601,7 +1601,7 @@ def apply_monolithic(; symbols: is_monolithic, apply_monolithic, touching `is_monolithic, apply_monolithic`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5 (10 lines); hunks: -1577,14 +1577,14 @@ def is_monolithic(self) -> bool:; -1601,7 +1601,7 @@ def apply_monolithic(; symbols: is_monolithic, apply_monolithic
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -1577,14 +1577,14 @@ def is_monolithic(self) -> bool:
-        layer: torch.nn.Module,
+        layer: FusedMoE,
-        expert_map: torch.Tensor | None = None,
-    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
+        input_ids: torch.Tensor | None = None,
+    ) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
