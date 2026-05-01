# sglang Kimi K2/K2.5/Linear/VL Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `sgl-project/sglang` trace worktree commit `4197c55968`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` | [#23394](https://github.com/sgl-project/sglang/pull/23394) |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-Linear.mdx` | no direct PR-number commit |
| `docs_new/docs/basic_usage/kimi_k2_5.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/kimi-k2-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/kimi-linear-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/configs/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789) |
| `python/sglang/srt/configs/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469) |
| `python/sglang/srt/configs/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/configs/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/function_call/kimik2_detector.py` | [#7940](https://github.com/sgl-project/sglang/pull/7940), [#8043](https://github.com/sgl-project/sglang/pull/8043), [#8968](https://github.com/sgl-project/sglang/pull/8968), [#10972](https://github.com/sgl-project/sglang/pull/10972), [#19120](https://github.com/sgl-project/sglang/pull/19120), [#19552](https://github.com/sgl-project/sglang/pull/19552) |
| `python/sglang/srt/models/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#18370](https://github.com/sgl-project/sglang/pull/18370), [#18434](https://github.com/sgl-project/sglang/pull/18434), [#18440](https://github.com/sgl-project/sglang/pull/18440), [#18689](https://github.com/sgl-project/sglang/pull/18689), [#19331](https://github.com/sgl-project/sglang/pull/19331), [#19689](https://github.com/sgl-project/sglang/pull/19689), [#19959](https://github.com/sgl-project/sglang/pull/19959), [#20747](https://github.com/sgl-project/sglang/pull/20747), [#21004](https://github.com/sgl-project/sglang/pull/21004), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22858](https://github.com/sgl-project/sglang/pull/22858), ... (13 total) |
| `python/sglang/srt/models/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469), [#12660](https://github.com/sgl-project/sglang/pull/12660), [#14337](https://github.com/sgl-project/sglang/pull/14337), [#17160](https://github.com/sgl-project/sglang/pull/17160), [#17506](https://github.com/sgl-project/sglang/pull/17506), [#17731](https://github.com/sgl-project/sglang/pull/17731), [#18849](https://github.com/sgl-project/sglang/pull/18849), [#20396](https://github.com/sgl-project/sglang/pull/20396) |
| `python/sglang/srt/models/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383), [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `python/sglang/srt/models/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/multimodal/processors/kimi_common.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `python/sglang/srt/multimodal/processors/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22368](https://github.com/sgl-project/sglang/pull/22368), [#22490](https://github.com/sgl-project/sglang/pull/22490), [#22858](https://github.com/sgl-project/sglang/pull/22858), [#23501](https://github.com/sgl-project/sglang/pull/23501) |
| `python/sglang/srt/multimodal/processors/kimi_vl.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `test/manual/models/test_kimi_k2_models.py` | no direct PR-number commit |
| `test/registered/8-gpu-models/test_kimi_k25.py` | [#19802](https://github.com/sgl-project/sglang/pull/19802), [#21391](https://github.com/sgl-project/sglang/pull/21391), [#21898](https://github.com/sgl-project/sglang/pull/21898) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` | [#21213](https://github.com/sgl-project/sglang/pull/21213) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/test_kimi_k25_mxfp4.py` | [#21213](https://github.com/sgl-project/sglang/pull/21213), [#22188](https://github.com/sgl-project/sglang/pull/22188) |
| `test/registered/amd/test_kimi_k2_instruct.py` | [#17656](https://github.com/sgl-project/sglang/pull/17656) |
| `test/registered/ascend/vlm_models/test_npu_kimi_vl_a3b_instruct.py` | no direct PR-number commit |
| `test/registered/function_call/test_kimik2_detector.py` | [#19552](https://github.com/sgl-project/sglang/pull/19552) |
| `test/registered/gb300/test_kimi_k25.py` | no direct PR-number commit |
| `test/registered/gb300/test_kimi_k25_nvfp4.py` | no direct PR-number commit |
| `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` | [#22381](https://github.com/sgl-project/sglang/pull/22381) |
| `test/registered/models/test_kimi_linear_models.py` | no direct PR-number commit |
| `test/registered/stress/test_stress_kimi_k2.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 41
- Extra PRs preserved from existing docs: 43
- Total PRs in this document: 84
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-18 | [#5440](https://github.com/sgl-project/sglang/pull/5440) | merged | Sgl kernel fused_moe_gate support n_shared_experts | `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py` |
| 2025-04-30 | [#5383](https://github.com/sgl-project/sglang/pull/5383) | merged | [Feature] add support kimi vl model | `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` |
| 2025-07-11 | [#7940](https://github.com/sgl-project/sglang/pull/7940) | merged | Support Kimi K2 | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-07-14 | [#8007](https://github.com/sgl-project/sglang/pull/8007) | open | [Kimi K2] num_experts extends to 384 | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu` |
| 2025-07-14 | [#8021](https://github.com/sgl-project/sglang/pull/8021) | merged | perf: add kimi k2 fused_moe tuning config for h30_3e | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-15 | [#8047](https://github.com/sgl-project/sglang/pull/8047) | merged | H20 tune config for Kimi | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8176](https://github.com/sgl-project/sglang/pull/8176) | merged | feat: add h200 tp 16 kimi k2 moe config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8178](https://github.com/sgl-project/sglang/pull/8178) | merged | feat: add b200 tp 16 kimi k2 moe config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8183](https://github.com/sgl-project/sglang/pull/8183) | merged | feat: add h200 tp 16 kimi k2 moe config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-24 | [#8043](https://github.com/sgl-project/sglang/pull/8043) | merged | feat(function call): complete utility method for KimiK2Detector and enhance documentation | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-08-01 | [#8013](https://github.com/sgl-project/sglang/pull/8013) | merged | [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384 | `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` |
| 2025-08-08 | [#8968](https://github.com/sgl-project/sglang/pull/8968) | merged | Fix kimi k2 function call format | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-08-09 | [#9010](https://github.com/sgl-project/sglang/pull/9010) | merged | [perf] add kimi-k2 b200 fused moe config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-08-26 | [#9606](https://github.com/sgl-project/sglang/pull/9606) | merged | Fix kimi k2 function calling format | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py` |
| 2025-09-26 | [#10612](https://github.com/sgl-project/sglang/pull/10612) | merged | Replace the Kimi-K2 generated tool call idx with history tool call count | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py` |
| 2025-10-01 | [#10972](https://github.com/sgl-project/sglang/pull/10972) | merged | fix: KimiK2Detector Improve tool call ID parsing with regex | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-10-31 | [#12469](https://github.com/sgl-project/sglang/pull/12469) | merged | Support Kimi Linear | `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py` |
| 2025-11-11 | [#12660](https://github.com/sgl-project/sglang/pull/12660) | merged | overlap shared + routed expert computation in kimi linear | `python/sglang/srt/models/kimi_linear.py` |
| 2025-11-13 | [#13150](https://github.com/sgl-project/sglang/pull/13150) | merged | Opt kimi_k2_thinking biased topk module | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-15 | [#13287](https://github.com/sgl-project/sglang/pull/13287) | merged | [opt kimi k2 1 / n] Add kimi k2 moe fused gate | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` |
| 2025-11-16 | [#13332](https://github.com/sgl-project/sglang/pull/13332) | merged | [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-18 | [#13374](https://github.com/sgl-project/sglang/pull/13374) | merged | [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2025-11-21 | [#13596](https://github.com/sgl-project/sglang/pull/13596) | merged | [kimi k2 thinking] Avoid useless torch.zeros_ | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py` |
| 2025-11-21 | [#13587](https://github.com/sgl-project/sglang/pull/13587) | merged | [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size | `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2025-11-21 | [#13466](https://github.com/sgl-project/sglang/pull/13466) | merged | [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking) | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-22 | [#9405](https://github.com/sgl-project/sglang/pull/9405) | merged | Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold) | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-22 | [#12759](https://github.com/sgl-project/sglang/pull/12759) | merged | [Ascend] support Kimi-K2-Thinking | `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/kimi_linear.py` |
| 2025-12-07 | [#13725](https://github.com/sgl-project/sglang/pull/13725) | merged | Add Expert Parallelism (EP) support for kimi-k2-thinking | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-12-16 | [#15100](https://github.com/sgl-project/sglang/pull/15100) | merged | Support piecewise cuda graph for fused marlin moe | `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py` |
| 2025-12-18 | [#15306](https://github.com/sgl-project/sglang/pull/15306) | merged | Fix warp illegal instruction in kimi k2 thinking PCG | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2026-01-19 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2026-01-19 | [#17325](https://github.com/sgl-project/sglang/pull/17325) | merged | Fix kernel selection in biased_grouped_topk_gpu | `python/sglang/srt/layers/moe/topk.py` |
| 2026-01-20 | [#17160](https://github.com/sgl-project/sglang/pull/17160) | merged | [Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code | `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-24 | [#17506](https://github.com/sgl-project/sglang/pull/17506) | merged | [Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention | `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-26 | [#17731](https://github.com/sgl-project/sglang/pull/17731) | merged | [Kimi-Linear] Remove duplicated code in kimi-linear | `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-26 | [#17656](https://github.com/sgl-project/sglang/pull/17656) | merged | [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases | `test/registered/amd/test_kimi_k2_instruct.py` |
| 2026-01-27 | [#17789](https://github.com/sgl-project/sglang/pull/17789) | merged | Support Kimi-K2.5 model | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-01-28 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` |
| 2026-01-30 | [#17624](https://github.com/sgl-project/sglang/pull/17624) | merged | [BUGFIX] Fix dp size > 1 for qwen3 vl model | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py` |
| 2026-02-02 | [#17991](https://github.com/sgl-project/sglang/pull/17991) | merged | Fix: Avoid Double Reduce in VLM DP Attention | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py` |
| 2026-02-04 | [#17895](https://github.com/sgl-project/sglang/pull/17895) | merged | [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes | `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` |
| 2026-02-05 | [#18064](https://github.com/sgl-project/sglang/pull/18064) | merged | fix kimi k2.5's moe gemm config init | `python/sglang/srt/managers/scheduler.py` |
| 2026-02-08 | [#18370](https://github.com/sgl-project/sglang/pull/18370) | merged | [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-08 | [#18440](https://github.com/sgl-project/sglang/pull/18440) | merged | [Kimi-K2.5] Fix missing `quant_config` in `KimiK25` | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-11 | [#18269](https://github.com/sgl-project/sglang/pull/18269) | merged | [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test | `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` |
| 2026-02-17 | [#18849](https://github.com/sgl-project/sglang/pull/18849) | merged | [PCG] support piecewise cuda graph for kimi-linear model | `python/sglang/srt/models/kimi_linear.py` |
| 2026-02-18 | [#18689](https://github.com/sgl-project/sglang/pull/18689) | merged | Add DP ViT support for Kimi K2.5 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-21 | [#19120](https://github.com/sgl-project/sglang/pull/19120) | merged | fix KimiK2Detector regex patterns with re.DOTALL | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-02-25 | [#18434](https://github.com/sgl-project/sglang/pull/18434) | merged | [Fix] Kimi K2.5 support pp | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-26 | [#19181](https://github.com/sgl-project/sglang/pull/19181) | merged | [Kernel Slimming] Migrate marlin moe kernel to JIT | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` |
| 2026-02-26 | [#19331](https://github.com/sgl-project/sglang/pull/19331) | merged | [NPU] support Kimi-K2.5 on NPU | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-26 | [#19228](https://github.com/sgl-project/sglang/pull/19228) | merged | [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` |
| 2026-03-02 | [#19703](https://github.com/sgl-project/sglang/pull/19703) | open | [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` |
| 2026-03-03 | [#19689](https://github.com/sgl-project/sglang/pull/19689) | merged | feat: support Kimi K2.5 for Eagle3 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-07 | [#19959](https://github.com/sgl-project/sglang/pull/19959) | merged | Fix Kimi K2.5 PP layer range exposure for PD disaggregation | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-07 | [#19802](https://github.com/sgl-project/sglang/pull/19802) | merged | [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2 | `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-03-17 | [#20747](https://github.com/sgl-project/sglang/pull/20747) | merged | fix piecewise cuda graph support for Kimi-K2.5 model | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-19 | [#19552](https://github.com/sgl-project/sglang/pull/19552) | merged | [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-03-20 | [#20396](https://github.com/sgl-project/sglang/pull/20396) | merged | perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path | `python/sglang/srt/models/kimi_linear.py` |
| 2026-03-26 | [#21004](https://github.com/sgl-project/sglang/pull/21004) | merged | [Fix] Add EPLB rebalance support for Kimi K2.5 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-26 | [#21391](https://github.com/sgl-project/sglang/pull/21391) | merged | Fix Kimi K2.5 dp attention+ spec decoding launch crash | `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-03-31 | [#21741](https://github.com/sgl-project/sglang/pull/21741) | open | [1/N] feat: support compressed-tensors w4afp8 MoE | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-04-02 | [#21898](https://github.com/sgl-project/sglang/pull/21898) | merged | [CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8 | `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-04-05 | [#21213](https://github.com/sgl-project/sglang/pull/21213) | merged | [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5… | `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-04-06 | [#22208](https://github.com/sgl-project/sglang/pull/22208) | open | [AMD] Optimize fused MoE kernel config for small-M decode on gfx950 | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2026-04-07 | [#22188](https://github.com/sgl-project/sglang/pull/22188) | merged | [AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1) | `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-04-10 | [#22269](https://github.com/sgl-project/sglang/pull/22269) | merged | [EPD][VLM] Support Kimi K25 EPD | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-10 | [#22488](https://github.com/sgl-project/sglang/pull/22488) | open | Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` |
| 2026-04-10 | [#22381](https://github.com/sgl-project/sglang/pull/22381) | merged | [Lora] Lora kimi support | `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` |
| 2026-04-10 | [#22496](https://github.com/sgl-project/sglang/pull/22496) | open | [Feature] kimi k25 w4a16 support deepep low latency | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-11 | [#22368](https://github.com/sgl-project/sglang/pull/22368) | merged | [VLM] GPU Image Preprocessing for Kimi-K2.5 | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-14 | [#22806](https://github.com/sgl-project/sglang/pull/22806) | open | feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading | `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-04-15 | [#22858](https://github.com/sgl-project/sglang/pull/22858) | merged | [VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5 | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-16 | [#22490](https://github.com/sgl-project/sglang/pull/22490) | merged | [EPD][VLM] Support Kimi VL EPD | `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py` |
| 2026-04-16 | [#13789](https://github.com/sgl-project/sglang/pull/13789) | closed | [DeepEP Support] Support kimi-k2-thinking deepep | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2026-04-21 | [#23186](https://github.com/sgl-project/sglang/pull/23186) | merged | [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-04-21 | [#23381](https://github.com/sgl-project/sglang/pull/23381) | open | [AMD] Add MI355X Kimi-K2.6 tuning artifacts | `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py` |
| 2026-04-21 | [#23394](https://github.com/sgl-project/sglang/pull/23394) | merged | [docs] sync kimi-k2.6 from sgl-cookbook | `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` |
| 2026-04-23 | [#23563](https://github.com/sgl-project/sglang/pull/23563) | open | [Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend | `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` |
| 2026-04-27 | [#23408](https://github.com/sgl-project/sglang/pull/23408) | merged | [AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping | `python/sglang/srt/models/kimi_k25.py` |
| 2026-04-27 | [#23848](https://github.com/sgl-project/sglang/pull/23848) | open | [AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` |
| 2026-04-27 | [#23501](https://github.com/sgl-project/sglang/pull/23501) | merged | [VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-30 | [#22964](https://github.com/sgl-project/sglang/pull/22964) | closed | [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output | `python/sglang/srt/multimodal/processors/kimi_k25.py` |

## Per-PR Diff Audit Cards

### PR #5440 - Sgl kernel fused_moe_gate support n_shared_experts

- Link: https://github.com/sgl-project/sglang/pull/5440
- Status/date: merged / 2025-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +140/-38, 351 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Sgl kernel fused_moe_gate support n_shared_experts"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py`; technical summary: Covers "Sgl kernel fused_moe_gate support n_shared_experts"; the main implementation surface is `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28 (109 lines); hunks: -57,6 +57,8 @@ __device__ void moe_fused_gate_impl(; -65,6 +67,9 @@ __device__ void moe_fused_gate_impl(; `sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5 (36 lines); hunks: -19,20 +19,24; -43,8 +47,30 @@ def test_moe_fused_gate_combined(seq_length, dtype, params):; symbols: test_moe_fused_gate_combined, touching `test_moe_fused_gate_combined`; `sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2 (20 lines); hunks: -34,13 +34,29 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate, touching `topk_softmax, moe_fused_gate`; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2 (10 lines); hunks: -200,8 +200,14 @@ void topk_softmax(.
- Code diff details:
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28 (109 lines); hunks: -57,6 +57,8 @@ __device__ void moe_fused_gate_impl(; -65,6 +67,9 @@ __device__ void moe_fused_gate_impl(
  - `sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5 (36 lines); hunks: -19,20 +19,24; -43,8 +47,30 @@ def test_moe_fused_gate_combined(seq_length, dtype, params):; symbols: test_moe_fused_gate_combined
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2 (20 lines); hunks: -34,13 +34,29 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2 (10 lines); hunks: -200,8 +200,14 @@ void topk_softmax(
  - `sgl-kernel/csrc/common_extension.cc` modified +2/-1 (3 lines); hunks: -146,7 +146,8 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/moe_fused_gate.cu
@@ -57,6 +57,8 @@ __device__ void moe_fused_gate_impl(
+    int64_t n_share_experts_fusion,
+    double routed_scaling_factor,
@@ -65,6 +67,9 @@ __device__ void moe_fused_gate_impl(
+  // Calculate topk_excluding_share_expert_fusion from topk
+  int64_t topk_excluding_share_expert_fusion = topk - (n_share_experts_fusion > 0 ? 1 : 0);
@@ -163,7 +168,7 @@ __device__ void moe_fused_gate_impl(
diff -- sgl-kernel/tests/test_moe_fused_gate.py
@@ -19,20 +19,24 @@
-def test_moe_fused_gate_combined(seq_length, dtype, params):
+@pytest.mark.parametrize("n_share_experts_fusion", [0, 1, 8, 16])
+def test_moe_fused_gate_combined(seq_length, dtype, params, n_share_experts_fusion):
+    topk = topk + min(1, n_share_experts_fusion)
+        n_share_experts_fusion=n_share_experts_fusion,
+        routed_scaling_factor=2.5,
diff -- sgl-kernel/python/sgl_kernel/moe.py
@@ -34,13 +34,29 @@ def topk_softmax(
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28; `sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2; `sgl-kernel/csrc/common_extension.cc` modified +2/-1
  - tests: `sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_moe_fused_gate.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5383 - [Feature] add support kimi vl model

- Link: https://github.com/sgl-project/sglang/pull/5383
- Status/date: merged / 2025-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/kimi_vl.py`, `python/sglang/srt/configs/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`; associated commits `8fefdd32c7c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1189/-11, 1316 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] add support kimi vl model"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py`; technical summary: Covers "[Feature] add support kimi vl model"; the main implementation surface is `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope, touching `multihead_attention, sdpa_attention, _apply_rope_input_validation`; `python/sglang/srt/models/kimi_vl.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward, touching `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`; `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: KimiVLImageProcessor, __init__, process_mm_data_async, touching `KimiVLImageProcessor, __init__, process_mm_data_async`; `python/sglang/srt/configs/kimi_vl.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: KimiVLConfig, __init__, touching `KimiVLConfig, __init__`.
- Code diff details:
  - `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `python/sglang/srt/models/kimi_vl.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: KimiVLImageProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/configs/kimi_vl.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: KimiVLConfig, __init__
  - `python/sglang/srt/configs/kimi_vl_moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_vl_moonvit.py
@@ -0,0 +1,639 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# This file is meant to be used in kimi_vl.py only
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
diff -- python/sglang/srt/models/kimi_vl.py
@@ -0,0 +1,308 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
+# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
diff -- python/sglang/srt/managers/multimodal_processors/kimi_vl.py
@@ -0,0 +1,73 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0; `python/sglang/srt/models/kimi_vl.py` added +308/-0; `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0; `python/sglang/srt/configs/kimi_vl.py` added +38/-0; `python/sglang/srt/configs/kimi_vl_moonvit.py` added +32/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7940 - Support Kimi K2

- Link: https://github.com/sgl-project/sglang/pull/7940
- Status/date: merged / 2025-07-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`; associated commits `615553079dc1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +480/-3, 568 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Kimi K2"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "Support Kimi K2"; the main implementation surface is `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: KimiK2Detector, __init__, has_tool_call, detect_and_parse, touching `KimiK2Detector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: KimiK2Detector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -0,0 +1,220 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8007 - [Kimi K2] num_experts extends to 384

- Link: https://github.com/sgl-project/sglang/pull/8007
- Status/date: open / 2025-07-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +30/-4, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi K2] num_experts extends to 384"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`; technical summary: Covers "[Kimi K2] num_experts extends to 384"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +5/-1 (6 lines); hunks: -45,6 +45,10; -321,7 +325,7 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, touching `biased_grouped_topk_gpu`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -246,7 +246,7 @@ def forward(self, hidden_states):; -2113,7 +2113,7 @@ def determine_num_fused_shared_experts(; symbols: forward, determine_num_fused_shared_experts, touching `forward, determine_num_fused_shared_experts`; `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1 (15 lines); hunks: -39,7 +39,9 @@ __device__ inline bool cmp_eq(const T& a, const T& b) {; -417,6 +419,17 @@ std::vector moe_fused_gate(; `sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0 (9 lines); hunks: -466,6 +466,9 @@ topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gati...; -520,6 +523,9 @@ topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gati....
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +5/-1 (6 lines); hunks: -45,6 +45,10; -321,7 +325,7 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -246,7 +246,7 @@ def forward(self, hidden_states):; -2113,7 +2113,7 @@ def determine_num_fused_shared_experts(; symbols: forward, determine_num_fused_shared_experts
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1 (15 lines); hunks: -39,7 +39,9 @@ __device__ inline bool cmp_eq(const T& a, const T& b) {; -417,6 +419,17 @@ std::vector moe_fused_gate(
  - `sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0 (9 lines); hunks: -466,6 +466,9 @@ topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gati...; -520,6 +523,9 @@ topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gati...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -45,6 +45,10 @@
+# Maximum VPT (Values Per Thread) supported by moe_fused_gate kernel
+# This should match MAX_VPT in moe_fused_gate.cu
+MAX_VPT_SUPPORTED = 384
@@ -321,7 +325,7 @@ def biased_grouped_topk_gpu(
-        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this asserti
+        <= MAX_VPT_SUPPORTED  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT now.
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -246,7 +246,7 @@ def forward(self, hidden_states):
-            and self.weight.shape[0] == 256
+            and self.weight.shape[0] in [256, 384]
@@ -2113,7 +2113,7 @@ def determine_num_fused_shared_experts(
-            or self.config.n_routed_experts != 256
+            or self.config.n_routed_experts not in [256, 384]
diff -- sgl-kernel/csrc/moe/moe_fused_gate.cu
@@ -39,7 +39,9 @@ __device__ inline bool cmp_eq(const T& a, const T& b) {
-static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +5/-1; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
  - other: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1; `sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8021 - perf: add kimi k2 fused_moe tuning config for h30_3e

- Link: https://github.com/sgl-project/sglang/pull/8021
- Status/date: merged / 2025-07-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf: add kimi k2 fused_moe tuning config for h30_3e"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "perf: add kimi k2 fused_moe tuning config for h30_3e"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8047 - H20 tune config for Kimi

- Link: https://github.com/sgl-project/sglang/pull/8047
- Status/date: merged / 2025-07-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "H20 tune config for Kimi"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "H20 tune config for Kimi"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8176 - feat: add h200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8176
- Status/date: merged / 2025-07-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add h200 tp 16 kimi k2 moe config"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "feat: add h200 tp 16 kimi k2 moe config"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8178 - feat: add b200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8178
- Status/date: merged / 2025-07-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add b200 tp 16 kimi k2 moe config"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "feat: add b200 tp 16 kimi k2 moe config"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8183 - feat: add h200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8183
- Status/date: merged / 2025-07-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add h200 tp 16 kimi k2 moe config"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "feat: add h200 tp 16 kimi k2 moe config"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8043 - feat(function call): complete utility method for KimiK2Detector and enhance documentation

- Link: https://github.com/sgl-project/sglang/pull/8043
- Status/date: merged / 2025-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`; associated commits `01079e174ff8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +205/-56, 404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(function call): complete utility method for KimiK2Detector and enhance documentation"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "feat(function call): complete utility method for KimiK2Detector and enhance documentation"; the main implementation surface is `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunks: -18,16 +18,21; -114,11 +119,7 @@ def parse_streaming_increment(; symbols: KimiK2Detector, __init__, parse_streaming_increment, touching `KimiK2Detector, __init__, parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunks: -18,16 +18,21; -114,11 +119,7 @@ def parse_streaming_increment(; symbols: KimiK2Detector, __init__, parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -18,16 +18,21 @@
+    """
+    Detector for Kimi K2 model function call format.
+    Format Structure:
+    '''
+    <|tool_calls_section_begin|>
+    <|tool_call_begin|>functions.{func_name}:{index} <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8013 - [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384

- Link: https://github.com/sgl-project/sglang/pull/8013
- Status/date: merged / 2025-08-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +188/-30, 318 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`; technical summary: Covers "[Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384"; the main implementation surface is `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunks: -25,6 +25,10; -91,12 +95,24 @@ void dsv3_router_gemm(; `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunks: -185,6 +185,7 @@ void invokeRouterGemmBf16Output(__nv_bfloat16* output, T con...; -232,3 +233,52 @@ template void invokeRouterGemmBf16Output (; `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunks: -184,6 +184,7 @@ void invokeRouterGemmFloatOutput(float* output, T const* mat...; -231,3 +232,52 @@ template void invokeRouterGemmFloatOutput (; `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunks: -13,29 +13,41; -55,29 +67,41 @@ def tflops(t_ms):; symbols: benchmark_bf16_output, runner, tflops, benchmark_float_output, touching `benchmark_bf16_output, runner, tflops`.
- Code diff details:
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunks: -25,6 +25,10; -91,12 +95,24 @@ void dsv3_router_gemm(
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunks: -185,6 +185,7 @@ void invokeRouterGemmBf16Output(__nv_bfloat16* output, T con...; -232,3 +233,52 @@ template void invokeRouterGemmBf16Output (
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunks: -184,6 +184,7 @@ void invokeRouterGemmFloatOutput(float* output, T const* mat...; -231,3 +232,52 @@ template void invokeRouterGemmFloatOutput (
  - `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunks: -13,29 +13,41; -55,29 +67,41 @@ def tflops(t_ms):; symbols: benchmark_bf16_output, runner, tflops, benchmark_float_output
  - `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2 (4 lines); hunks: -5,8 +5,8; symbols: test_dsv3_router_gemm
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu
@@ -25,6 +25,10 @@
+static constexpr int DEFAULT_NUM_EXPERTS = 256;
+static constexpr int KIMI_K2_NUM_EXPERTS = 384;
+static constexpr int DEFAULT_HIDDEN_DIM = 7168;
@@ -91,12 +95,24 @@ void dsv3_router_gemm(
-  constexpr int num_experts = 256;
-  constexpr int hidden_dim = 7168;
diff -- sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu
@@ -185,6 +185,7 @@ void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a, T const*
+// Template instantiations for DEFAULT_NUM_EXPERTS experts
@@ -232,3 +233,52 @@ template void invokeRouterGemmBf16Output<__nv_bfloat16, 15, 256, 7168>(
+// Template instantiations for KIMI_K2_NUM_EXPERTS experts
+template void invokeRouterGemmBf16Output<__nv_bfloat16, 1, 384, 7168>(
+    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
+template void invokeRouterGemmBf16Output<__nv_bfloat16, 2, 384, 7168>(
diff -- sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu
@@ -184,6 +184,7 @@ void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,
```

- Reviewed files:
  - other: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16; `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0; `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0; `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12
  - tests: `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_dsv3_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8968 - Fix kimi k2 function call format

- Link: https://github.com/sgl-project/sglang/pull/8968
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`; associated commits `91e2f902db0e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix kimi k2 function call format"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "Fix kimi k2 function call format"; the main implementation surface is `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3 (6 lines); hunks: -24,7 +24,7 @@ class KimiK2Detector(BaseFormatDetector):; -219,7 +219,7 @@ def structure_info(self) -> _GetInfoFunc:; symbols: KimiK2Detector, structure_info, get_info, build_ebnf, touching `KimiK2Detector, structure_info, get_info`.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3 (6 lines); hunks: -24,7 +24,7 @@ class KimiK2Detector(BaseFormatDetector):; -219,7 +219,7 @@ def structure_info(self) -> _GetInfoFunc:; symbols: KimiK2Detector, structure_info, get_info, build_ebnf
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -24,7 +24,7 @@ class KimiK2Detector(BaseFormatDetector):
-    <|tool_call_begin|>functions.{func_name}:{index} <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
+    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
@@ -219,7 +219,7 @@ def structure_info(self) -> _GetInfoFunc:
-                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0 <|tool_call_argument_begin|>",
+                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
@@ -240,6 +240,6 @@ def build_ebnf(self, tools: List[Tool]) -> str:
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/kimik2_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9010 - [perf] add kimi-k2 b200 fused moe config

- Link: https://github.com/sgl-project/sglang/pull/9010
- Status/date: merged / 2025-08-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] add kimi-k2 b200 fused moe config"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "[perf] add kimi-k2 b200 fused moe config"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9606 - Fix kimi k2 function calling format

- Link: https://github.com/sgl-project/sglang/pull/9606
- Status/date: merged / 2025-08-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +117/-9, 155 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix kimi k2 function calling format"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`; technical summary: Covers "Fix kimi k2 function calling format"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunks: -835,15 +835,23 @@ def _process_tool_calls(; -954,7 +962,11 @@ async def _process_tool_call_stream(; symbols: _process_tool_calls, _process_tool_call_stream, touching `_process_tool_calls, _process_tool_call_stream`; `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunks: -6,6 +6,8; -325,6 +327,100 @@ async def test_unstreamed_tool_args_no_parser_data(self):; symbols: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk, touching `test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunks: -835,15 +835,23 @@ def _process_tool_calls(; -954,7 +962,11 @@ async def _process_tool_call_stream(; symbols: _process_tool_calls, _process_tool_call_stream
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunks: -6,6 +6,8; -325,6 +327,100 @@ async def test_unstreamed_tool_args_no_parser_data(self):; symbols: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -835,15 +835,23 @@ def _process_tool_calls(
-                tool_calls = [
-                    ToolCall(
-                        id=f"call_{uuid.uuid4().hex[:24]}",
-                        function=FunctionResponse(
-                            name=call_info.name, arguments=call_info.parameters
-                        ),
diff -- test/srt/openai_server/basic/test_serving_chat.py
@@ -6,6 +6,8 @@
+import asyncio
+import json
@@ -325,6 +327,100 @@ async def test_unstreamed_tool_args_no_parser_data(self):
+    # ------------- kimi_k2 tool_call_id formatting -------------
+    def test_kimi_k2_non_streaming_tool_call_id_format(self):
+        """Ensure non-streaming tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9
  - tests: `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/basic/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10612 - Replace the Kimi-K2 generated tool call idx with history tool call count

- Link: https://github.com/sgl-project/sglang/pull/10612
- Status/date: merged / 2025-09-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +226/-15, 303 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Replace the Kimi-K2 generated tool call idx with history tool call count"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`; technical summary: Covers "Replace the Kimi-K2 generated tool call idx with history tool call count"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunks: -33,6 +33,7; -749,8 +750,9 @@ def _build_chat_response(; symbols: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls, touching `_build_chat_response, _process_response_logprobs, _process_tool_call_id`; `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunks: -420,6 +420,181 @@ async def collect_first_tool_chunk():; symbols: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history, touching `collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunks: -33,6 +33,7; -749,8 +750,9 @@ def _build_chat_response(; symbols: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunks: -420,6 +420,181 @@ async def collect_first_tool_chunk():; symbols: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -33,6 +33,7 @@
+from sglang.srt.function_call.core_types import ToolCallItem
@@ -749,8 +750,9 @@ def _build_chat_response(
+                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
-                    text, request.tools, finish_reason
+                    text, request.tools, finish_reason, history_tool_calls_cnt
@@ -840,11 +842,32 @@ def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs
diff -- test/srt/openai_server/basic/test_serving_chat.py
@@ -420,6 +420,181 @@ async def collect_first_tool_chunk():
+    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
+        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""
+        # Force kimi_k2 parser
+        self.chat.tool_call_parser = "kimi_k2"
+        # Prepare request with tool calls history
+        req = ChatCompletionRequest(
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15
  - tests: `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0
- Risk and verification: The diff ships test coverage in `test/srt/openai_server/basic/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10972 - fix: KimiK2Detector Improve tool call ID parsing with regex

- Link: https://github.com/sgl-project/sglang/pull/10972
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`; associated commits `1193f13181a2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-4, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: KimiK2Detector Improve tool call ID parsing with regex"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "fix: KimiK2Detector Improve tool call ID parsing with regex"; the main implementation surface is `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunks: -50,6 +50,11 @@ def __init__(self):; -76,14 +81,18 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment, touching `__init__, has_tool_call, detect_and_parse`.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunks: -50,6 +50,11 @@ def __init__(self):; -76,14 +81,18 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -50,6 +50,11 @@ def __init__(self):
+        # Robust parser for ids like "functions.search:0" or fallback "search:0"
+        self.tool_call_id_regex = re.compile(
+            r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$"
+        )
@@ -76,14 +81,18 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
-                function_name = function_id.split(".")[1].split(":")[0]
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/kimik2_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12469 - Support Kimi Linear

- Link: https://github.com/sgl-project/sglang/pull/12469
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/kimi_linear.py`, `python/sglang/srt/models/kimi_linear.py`; associated commits `a4bf5c6ad25d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +2847/-112, 3404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Kimi Linear"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py`; technical summary: Covers "Support Kimi Linear"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: KimiMoE, __init__, forward, KimiDeltaAttention, touching `KimiMoE, __init__, forward`; `python/sglang/srt/configs/kimi_linear.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: KimiLinearConfig, __init__, is_mla, is_moe, touching `KimiLinearConfig, __init__, is_mla`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: KimiMoE, __init__, forward, KimiDeltaAttention
  - `python/sglang/srt/configs/kimi_linear.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -0,0 +1,678 @@
+# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py
+from collections.abc import Iterable
+from typing import Optional
+import torch
+from einops import rearrange
+from torch import nn
diff -- python/sglang/srt/configs/kimi_linear.py
@@ -0,0 +1,160 @@
+# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/transformers_utils/configs/kimi_linear.py
+from transformers.configuration_utils import PretrainedConfig
+from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
+from sglang.srt.layers.dp_attention import get_attention_tp_size
+class KimiLinearConfig(PretrainedConfig):
+    model_type = "kimi_linear"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` added +678/-0; `python/sglang/srt/configs/kimi_linear.py` added +160/-0
- Risk and verification: The diff ships test coverage in `test/srt/models/test_kimi_linear_models.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12660 - overlap shared + routed expert computation in kimi linear

- Link: https://github.com/sgl-project/sglang/pull/12660
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `cc2e36c352e8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +37/-5, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "overlap shared + routed expert computation in kimi linear"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "overlap shared + routed expert computation in kimi linear"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +37/-5 (42 lines); hunks: -32,6 +32,7; -52,6 +53,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +37/-5 (42 lines); hunks: -32,6 +32,7; -52,6 +53,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -32,6 +32,7 @@
+from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
@@ -52,6 +53,7 @@ def __init__(
+        alt_stream: Optional[torch.cuda.Stream] = None,
@@ -63,6 +65,7 @@ def __init__(
+        self.alt_stream = alt_stream
@@ -120,11 +123,34 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +37/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13150 - Opt kimi_k2_thinking biased topk module

- Link: https://github.com/sgl-project/sglang/pull/13150
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +71/-14, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Opt kimi_k2_thinking biased topk module"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`; technical summary: Covers "Opt kimi_k2_thinking biased topk module"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunks: -600,6 +600,48 @@ def grouped_topk_cpu(; -760,20 +802,35 @@ def biased_grouped_topk_gpu(; symbols: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu, touching `grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunks: -600,6 +600,48 @@ def grouped_topk_cpu(; -760,20 +802,35 @@ def biased_grouped_topk_gpu(; symbols: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -600,6 +600,48 @@ def grouped_topk_cpu(
+@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
+def kimi_k2_biased_topk_impl(
+    hidden_states: torch.Tensor,
+    gating_output: torch.Tensor,
+    correction_bias: torch.Tensor,
+    topk: int,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +71/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13287 - [opt kimi k2 1 / n] Add kimi k2 moe fused gate

- Link: https://github.com/sgl-project/sglang/pull/13287
- Status/date: merged / 2025-11-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +646/-0, 684 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[opt kimi k2 1 / n] Add kimi k2 moe fused gate"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`; technical summary: Covers "[opt kimi k2 1 / n] Add kimi k2 moe fused gate"; the main implementation surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunks: -0,0 +1,354; `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case, touching `test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case`; `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark, touching `kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark`; `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunks: -111,6 +111,41 @@ def moe_fused_gate(; symbols: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm, touching `moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm`.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunks: -0,0 +1,354
  - `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case
  - `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunks: -111,6 +111,41 @@ def moe_fused_gate(; symbols: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: -331,6 +331,14 @@ std::vector moe_fused_gate(
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu
@@ -0,0 +1,354 @@
+#include <ATen/cuda/CUDAContext.h>
+#include <cuda_runtime.h>
+#include <cutlass/array.h>
+#include <cutlass/cutlass.h>
+#include <cutlass/numeric_types.h>
+#include <torch/all.h>
diff -- sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py
@@ -0,0 +1,124 @@
+import pytest
+import torch
+from sgl_kernel import kimi_k2_moe_fused_gate
+from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl
+@pytest.mark.parametrize(
+    "seq_length",
diff -- sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py
@@ -0,0 +1,117 @@
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0; `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0; `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0; `sgl-kernel/csrc/common_extension.cc` modified +6/-0; `sgl-kernel/CMakeLists.txt` modified +1/-0
  - tests: `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13332 - [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate

- Link: https://github.com/sgl-project/sglang/pull/13332
- Status/date: merged / 2025-11-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-9, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`; technical summary: Covers "[opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunks: -72,7 +72,7; -817,16 +817,13 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, touching `biased_grouped_topk_gpu`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunks: -72,7 +72,7; -817,16 +817,13 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -72,7 +72,7 @@
-    from sgl_kernel import moe_fused_gate
+    from sgl_kernel import kimi_k2_moe_fused_gate, moe_fused_gate
@@ -817,16 +817,13 @@ def biased_grouped_topk_gpu(
-        if num_experts == 384 and num_expert_group == 1:
-            return kimi_k2_biased_topk_impl(
-                hidden_states,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +6/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13374 - [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel

- Link: https://github.com/sgl-project/sglang/pull/13374
- Status/date: merged / 2025-11-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +130/-173, 400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; technical summary: Covers "[opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel"; the main implementation surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunks: -1,15 +1,9; -21,149 +15,144 @@ static constexpr int SMALL_TOKEN_THRESHOLD = 512;.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunks: -1,15 +1,9; -21,149 +15,144 @@ static constexpr int SMALL_TOKEN_THRESHOLD = 512;
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu
@@ -1,15 +1,9 @@
-#include <cutlass/array.h>
-#include <cutlass/cutlass.h>
-#include <cutlass/numeric_types.h>
-using bfloat16_t = cutlass::bfloat16_t;
-using float16_t = cutlass::half_t;
@@ -21,149 +15,144 @@ static constexpr int SMALL_TOKEN_THRESHOLD = 512;
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #13596 - [kimi k2 thinking] Avoid useless torch.zeros_

- Link: https://github.com/sgl-project/sglang/pull/13596
- Status/date: merged / 2025-11-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +252/-256, 598 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[kimi k2 thinking] Avoid useless torch.zeros_"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py`; technical summary: Covers "[kimi k2 thinking] Avoid useless torch.zeros_"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake, touching `get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunks: -7,13 +7,6; -56,9 +49,6; symbols: apply, touching `apply`; `python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunks: -52,12 +52,7; -835,6 +830,9 @@ def apply(; symbols: apply, touching `apply`; `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunks: -55,7 +55,7; -1059,14 +1059,14 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunks: -7,13 +7,6; -56,9 +49,6; symbols: apply
  - `python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunks: -52,12 +52,7; -835,6 +830,9 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunks: -55,7 +55,7; -1059,14 +1059,14 @@ def apply(; symbols: apply
  - `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232 (232 lines); hunks: -1,18 +1,6; -67,223 +55,3 @@ def moe_wna16_marlin_gemm(; symbols: get_scalar_type, moe_wna16_marlin_gemm, fused_marlin_moe, fused_marlin_moe_fake
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
@@ -0,0 +1,239 @@
+import functools
+from typing import Optional
+import torch
+from sglang.srt.utils import is_cuda
+_is_cuda = is_cuda()
+if _is_cuda:
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -7,13 +7,6 @@
-try:
-    from sgl_kernel import fused_marlin_moe
-    FUSED_MARLIN_MOE_AVAILABLE = True
-except ImportError:
-    FUSED_MARLIN_MOE_AVAILABLE = False
@@ -56,9 +49,6 @@
diff -- python/sglang/srt/layers/quantization/awq.py
@@ -52,12 +52,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12; `python/sglang/srt/layers/quantization/awq.py` modified +4/-6; `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4
  - other: `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232; `sgl-kernel/python/sgl_kernel/__init__.py` modified +1/-1
  - tests: `python/sglang/test/test_marlin_moe.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_marlin_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13587 - [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size

- Link: https://github.com/sgl-project/sglang/pull/13587
- Status/date: merged / 2025-11-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-6, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; technical summary: Covers "[opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunks: -69,11 +69,6 @@ def moe_align_block_size(; -82,6 +77,6 @@ def moe_align_block_size(; symbols: moe_align_block_size, touching `moe_align_block_size`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunks: -69,11 +69,6 @@ def moe_align_block_size(; -82,6 +77,6 @@ def moe_align_block_size(; symbols: moe_align_block_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
@@ -69,11 +69,6 @@ def moe_align_block_size(
-    # Threshold based on benchmark results
-    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
-    if not fuse_sorted_ids_padding:
-        sorted_ids.fill_(topk_ids.numel())
@@ -82,6 +77,6 @@ def moe_align_block_size(
-        fuse_sorted_ids_padding,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13466 - [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)

- Link: https://github.com/sgl-project/sglang/pull/13466
- Status/date: merged / 2025-11-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`; technical summary: Covers "[Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunks: -74,6 +74,29; symbols: _kimi_k2_moe_fused_gate, touching `_kimi_k2_moe_fused_gate`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunks: -74,6 +74,29; symbols: _kimi_k2_moe_fused_gate
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -74,6 +74,29 @@
+    @torch.library.register_fake("sgl_kernel::kimi_k2_moe_fused_gate")
+    def _kimi_k2_moe_fused_gate(
+        input_tensor,
+        bias,
+        topk,
+        renormalize,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +23/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9405 - Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)

- Link: https://github.com/sgl-project/sglang/pull/9405
- Status/date: merged / 2025-11-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -787,12 +787,13 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -787,12 +787,13 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -787,12 +787,13 @@ def forward(
-            DUAL_STREAM_TOKEN_THRESHOLD = 1024
+            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
-                and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
+                and get_is_capture_mode()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12759 - [Ascend] support Kimi-K2-Thinking

- Link: https://github.com/sgl-project/sglang/pull/12759
- Status/date: merged / 2025-11-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +549/-170, 871 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Ascend] support Kimi-K2-Thinking"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[Ascend] support Kimi-K2-Thinking"; the main implementation surface is `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunks: -1,9 +1,11; -21,6 +23,9; symbols: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for, touching `npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunks: -35,12 +35,12; -314,87 +314,44 @@ def forward_npu(; symbols: forward_npu, _forward_normal, _forward_ll, npu_fused_moe_without_routing_weights_bf16, touching `forward_npu, _forward_normal, _forward_ll`; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -3979,6 +3979,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -4006,7 +4008,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, touching `load_weights`; `python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunks: -217,7 +217,7 @@ def add_chunked_prefix_cache_attention_backend(backend_name):; symbols: add_chunked_prefix_cache_attention_backend, touching `add_chunked_prefix_cache_attention_backend`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunks: -1,9 +1,11; -21,6 +23,9; symbols: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunks: -35,12 +35,12; -314,87 +314,44 @@ def forward_npu(; symbols: forward_npu, _forward_normal, _forward_ll, npu_fused_moe_without_routing_weights_bf16
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -3979,6 +3979,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -4006,7 +4008,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunks: -217,7 +217,7 @@ def add_chunked_prefix_cache_attention_backend(backend_name):; symbols: add_chunked_prefix_cache_attention_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/w8a8_int8.py
@@ -1,9 +1,11 @@
+import logging
+from compressed_tensors.quantization import QuantizationStrategy
@@ -21,6 +23,9 @@
+from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
+    CompressedTensorsConfig,
+)
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -35,12 +35,12 @@
-if not (_is_npu or _is_hip):
-    pass
+elif _is_npu:
+    import torch_npu
@@ -314,87 +314,44 @@ def forward_npu(
-        import torch_npu
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -3979,6 +3979,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0; `python/sglang/srt/model_executor/model_runner.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- Link: https://github.com/sgl-project/sglang/pull/14337
- Status/date: merged / 2025-12-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `6d5d76ad97dd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +0/-8, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13725 - Add Expert Parallelism (EP) support for kimi-k2-thinking

- Link: https://github.com/sgl-project/sglang/pull/13725
- Status/date: merged / 2025-12-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Expert Parallelism (EP) support for kimi-k2-thinking"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; technical summary: Covers "Add Expert Parallelism (EP) support for kimi-k2-thinking"; the main implementation surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunks: -634,6 +634,16 @@ def apply(; -643,6 +653,8 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunks: -634,6 +634,16 @@ def apply(; -643,6 +653,8 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -634,6 +634,16 @@ def apply(
+        # Get expert_map for EP support
+        expert_map = None
+        global_num_experts = -1
+        if hasattr(layer, "dispatcher") and hasattr(
+            layer.dispatcher, "local_expert_mapping"
+        ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15100 - Support piecewise cuda graph for fused marlin moe

- Link: https://github.com/sgl-project/sglang/pull/15100
- Status/date: merged / 2025-12-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +55/-36, 159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support piecewise cuda graph for fused marlin moe"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py`; technical summary: Covers "Support piecewise cuda graph for fused marlin moe"; the main implementation surface is `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunks: -1099,32 +1099,3 @@ def _(b_q_weight, perm, size_k, size_n, num_bits):; symbols: _, touching `_`; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunks: -2,7 +2,7; -41,7 +41,7 @@ def fused_marlin_moe(; symbols: fused_marlin_moe, fused_marlin_moe_fake, touching `fused_marlin_moe, fused_marlin_moe_fake`; `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunks: -80,7 +80,9 @@ def fused_experts_none_to_marlin(; -97,7 +99,7 @@ def fused_experts_none_to_marlin(; symbols: fused_experts_none_to_marlin, touching `fused_experts_none_to_marlin`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunks: -943,7 +943,7 @@ def apply(; -967,7 +967,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunks: -1099,32 +1099,3 @@ def _(b_q_weight, perm, size_k, size_n, num_bits):; symbols: _
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunks: -2,7 +2,7; -41,7 +41,7 @@ def fused_marlin_moe(; symbols: fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunks: -80,7 +80,9 @@ def fused_experts_none_to_marlin(; -97,7 +99,7 @@ def fused_experts_none_to_marlin(; symbols: fused_experts_none_to_marlin
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunks: -943,7 +943,7 @@ def apply(; -967,7 +967,7 @@ def apply(; symbols: apply
  - `test/srt/test_piecewise_cuda_graph.py` modified +35/-0 (35 lines); hunks: -214,6 +214,41 @@ def test_mgsm_accuracy(self):; symbols: test_mgsm_accuracy, TestPiecewiseCudaGraphGPTQ, setUpClass, tearDownClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/gptq.py
@@ -1099,32 +1099,3 @@ def _(b_q_weight, perm, size_k, size_n, num_bits):
-    @register_fake_if_exists("sgl_kernel::moe_wna16_marlin_gemm")
-    def _(
-        a,
-        c,
-        b_q_weight,
-        b_scales,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
@@ -2,7 +2,7 @@
-from sglang.srt.utils import is_cuda
+from sglang.srt.utils import direct_register_custom_op, is_cuda
@@ -41,7 +41,7 @@ def fused_marlin_moe(
-    routed_scaling_factor: float = None,
+    routed_scaling_factor: Optional[float] = None,
@@ -225,15 +225,26 @@ def fused_marlin_moe_fake(
diff -- python/sglang/srt/layers/moe/moe_runner/marlin.py
@@ -80,7 +80,9 @@ def fused_experts_none_to_marlin(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3; `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2
  - tests: `test/srt/test_piecewise_cuda_graph.py` modified +35/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_piecewise_cuda_graph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15306 - Fix warp illegal instruction in kimi k2 thinking PCG

- Link: https://github.com/sgl-project/sglang/pull/15306
- Status/date: merged / 2025-12-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-4, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix warp illegal instruction in kimi k2 thinking PCG"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; technical summary: Covers "Fix warp illegal instruction in kimi k2 thinking PCG"; the main implementation surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunks: -126,6 +126,9 @@ __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; -219,11 +222,16 @@ __global__ void kimi_k2_moe_fused_gate_kernel(.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunks: -126,6 +126,9 @@ __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; -219,11 +222,16 @@ __global__ void kimi_k2_moe_fused_gate_kernel(
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu
@@ -126,6 +126,9 @@ __global__ void kimi_k2_moe_fused_gate_kernel_small_token(
+      } else {
+        output_ptr[row_idx * topk + k] = 0.0f;
+        indices_ptr[row_idx * topk + k] = 0;
@@ -219,11 +222,16 @@ __global__ void kimi_k2_moe_fused_gate_kernel(
-    if (lane_id == 0 && max_expert != -1) {
+    if (lane_id == 0) {
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #15347 - Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

- Link: https://github.com/sgl-project/sglang/pull/15347
- Status/date: merged / 2026-01-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +165/-12, 215 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`; technical summary: Covers "Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, touching `biased_grouped_topk_gpu`; `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek, touching `test_fused_topk_deepseek`; `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed, touching `test_a_gsm8k, test_bs_1_speed`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
  - `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek
  - `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -75,6 +75,11 @@
+    try:
+        from flashinfer.fused_moe import fused_topk_deepseek
+    except ImportError:
+        fused_topk_deepseek = None
@@ -732,12 +737,68 @@ def biased_grouped_topk_gpu(
-    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
diff -- test/registered/kernels/test_fused_topk_deepseek.py
@@ -0,0 +1,97 @@
+import pytest
+import torch
+from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu, biased_grouped_topk_impl
+from sglang.test.ci.ci_register import register_cuda_ci
+register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)
+@pytest.mark.parametrize(
diff -- test/srt/test_deepseek_v3_mtp.py
@@ -82,10 +82,7 @@ def test_a_gsm8k(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +66/-4
  - tests: `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0; `test/srt/test_deepseek_v3_mtp.py` modified +2/-8
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17325 - Fix kernel selection in biased_grouped_topk_gpu

- Link: https://github.com/sgl-project/sglang/pull/17325
- Status/date: merged / 2026-01-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix kernel selection in biased_grouped_topk_gpu"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/layers/moe/topk.py`; technical summary: Covers "Fix kernel selection in biased_grouped_topk_gpu"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunks: -795,7 +795,6 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, touching `biased_grouped_topk_gpu`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunks: -795,7 +795,6 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -795,7 +795,6 @@ def biased_grouped_topk_gpu(
-        and num_fused_shared_experts == 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17160 - [Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code

- Link: https://github.com/sgl-project/sglang/pull/17160
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `e6b7c04947ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-42, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +13/-9 (22 lines); hunks: -15,7 +15,7; -314,6 +314,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +13/-9 (22 lines); hunks: -15,7 +15,7; -314,6 +314,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -15,7 +15,7 @@
-from sglang.srt.layers.attention.fla.kda import FusedRMSNormGated
+from sglang.srt.layers.attention.fla.kda import FusedRMSNormGated, fused_kda_gate
@@ -314,6 +314,14 @@ def forward(
+        beta = self.b_proj(hidden_states)[0].float().sigmoid()
+        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
+        forget_gate = fused_kda_gate(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +13/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17506 - [Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention

- Link: https://github.com/sgl-project/sglang/pull/17506
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `0c8165ffbd1b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +95/-90, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +42/-37 (79 lines); hunks: -16,6 +16,7; -27,6 +28,7; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +42/-37 (79 lines); hunks: -16,6 +16,7; -27,6 +28,7; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -16,6 +16,7 @@
+from sglang.srt.layers.dp_attention import get_attention_tp_size
@@ -27,6 +28,7 @@
+from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
@@ -171,10 +173,15 @@ def __init__(
+        self.attn_tp_size = get_attention_tp_size()
+        self.num_k_heads = config.linear_attn_config["num_heads"]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +42/-37
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17731 - [Kimi-Linear] Remove duplicated code in kimi-linear

- Link: https://github.com/sgl-project/sglang/pull/17731
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `1e8db1829096`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-Linear] Remove duplicated code in kimi-linear"; model line: Kimi K2/K2.5/Linear/VL; category: model implementation change; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "[Kimi-Linear] Remove duplicated code in kimi-linear"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +0/-1 (1 lines); hunks: -340,7 +340,6 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-1 (1 lines); hunks: -340,7 +340,6 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -340,7 +340,6 @@ def forward(
-        beta = self.b_proj(hidden_states)[0].float()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17656 - [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases

- Link: https://github.com/sgl-project/sglang/pull/17656
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_kimi_k2_instruct.py`; associated commits `738b1ac988c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +97/-2, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `test/registered/amd/test_kimi_k2_instruct.py`; technical summary: Covers "[AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases"; the main implementation surface is `test/registered/amd/test_kimi_k2_instruct.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k, touching `TestKimiK2Instruct0905, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/amd/test_kimi_k2_instruct.py
@@ -0,0 +1,95 @@
+import os
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
```

- Reviewed files:
  - tests: `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_kimi_k2_instruct.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17789 - Support Kimi-K2.5 model

- Link: https://github.com/sgl-project/sglang/pull/17789
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; associated commits `479ab7a4e7e4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +1053/-12, 1193 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Kimi-K2.5 model"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "Support Kimi-K2.5 model"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunks: -0,0 +1,744; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__, touching `apply_rope, tpool_patch_merger, MoonViTEncoderLayer`; `python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size, touching `KimiK25VisionConfig, __init__, KimiK25Config`; `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunks: -0,0 +1,88; symbols: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items, touching `KimiK2_5VLImageProcessor, __init__, process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunks: -0,0 +1,744; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__
  - `python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunks: -0,0 +1,88; symbols: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -0,0 +1,744 @@
+import logging
+from copy import deepcopy
+from typing import Iterable, List, Optional, Sequence, Tuple
+import numpy as np
+import torch
+import torch.nn.functional as F
diff -- python/sglang/srt/configs/kimi_k25.py
@@ -0,0 +1,171 @@
+"""
+Kimi K25 Model Configuration.
+"""
+from transformers import DeepseekV3Config
+from transformers.configuration_utils import PretrainedConfig
+class KimiK25VisionConfig(PretrainedConfig):
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -0,0 +1,88 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` added +744/-0; `python/sglang/srt/configs/kimi_k25.py` added +171/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17523
- Status/date: merged / 2026-01-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +1540/-43, 1823 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`; technical summary: Covers "[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI"; the main implementation surface is `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: -25,18 +25,21 @@ on:; -248,35 +251,6 @@ jobs:; `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch, touching `generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass`; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32TPMTP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: -25,18 +25,21 @@ on:; -248,35 +251,6 @@ jobs:
  - `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py
@@ -0,0 +1,248 @@
+"""AMD DeepSeek-V3.2 GSM8K Completion Evaluation Test (8-GPU)
+Tests DeepSeek-V3.2 with basic configuration using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-deepseek-v32 suite
+"""
+import ast
diff -- .github/workflows/nightly-test-amd.yml
@@ -25,18 +25,21 @@ on:
-          - 'nightly-accuracy-8-gpu-deepseek-r1'
+          - 'nightly-8-gpu-deepseek-v32'
+          - 'nightly-8-gpu-deepseek-v32-mtp'
+          - 'nightly-8-gpu-kimi-k2'
+          - 'nightly-accuracy-8-gpu-mi35x-deepseek-v32-mtp'
@@ -248,35 +251,6 @@ jobs:
diff -- test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py
@@ -0,0 +1,149 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0; `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0; `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0; `test/registered/amd/perf/test_deepseek_v32_basic_perf_amd.py` added +142/-0; `test/registered/amd/accuracy/test_deepseek_v32_tc_eval_amd.py` added +123/-0
  - ci: `.github/workflows/nightly-test-amd.yml` modified +158/-35
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- Link: https://github.com/sgl-project/sglang/pull/17624
- Status/date: merged / 2026-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +48/-19, 185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] Fix dp size > 1 for qwen3 vl model"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`; technical summary: Covers "[BUGFIX] Fix dp size > 1 for qwen3 vl model"; the main implementation surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #17991 - Fix: Avoid Double Reduce in VLM DP Attention

- Link: https://github.com/sgl-project/sglang/pull/17991
- Status/date: merged / 2026-02-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +51/-12, 132 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix: Avoid Double Reduce in VLM DP Attention"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py`; technical summary: Covers "Fix: Avoid Double Reduce in VLM DP Attention"; the main implementation surface is `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunks: -13,11 +13,7; -687,7 +683,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunks: -39,6 +39,8; -126,6 +128,7 @@ def __init__(; symbols: apply_rope, __init__, forward, touching `apply_rope, __init__, forward`; `test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass, touching `test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass`; `test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunks: -187,8 +187,6 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass, touching `test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunks: -13,11 +13,7; -687,7 +683,6 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunks: -39,6 +39,8; -126,6 +128,7 @@ def __init__(; symbols: apply_rope, __init__, forward
  - `test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass
  - `test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunks: -187,8 +187,6 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/vision.py
@@ -13,11 +13,7 @@
-from sglang.srt.layers.dp_attention import (
-    get_attention_tp_group,
-    get_attention_tp_rank,
-    get_attention_tp_size,
-)
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
diff -- python/sglang/srt/models/kimi_k25.py
@@ -39,6 +39,8 @@
+from sglang.srt.layers.dp_attention import is_dp_attention_enabled
@@ -126,6 +128,7 @@ def __init__(
+            use_dp_attention_reduce=is_dp_attention_enabled(),
diff -- test/registered/distributed/test_dp_attention_large.py
@@ -3,6 +3,7 @@
+from sglang.lang.chat_template import get_chat_template_by_model_path
@@ -11,6 +12,7 @@
+    DEFAULT_IMAGE_URL,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +1/-10; `python/sglang/srt/models/kimi_k25.py` modified +3/-0
  - tests: `test/registered/distributed/test_dp_attention_large.py` modified +47/-0; `test/registered/distributed/test_dp_attention.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_dp_attention.py`, `test/registered/distributed/test_dp_attention_large.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17895 - [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes

- Link: https://github.com/sgl-project/sglang/pull/17895
- Status/date: merged / 2026-02-04
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`; associated commits `6fd878b41df0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 34 files, +184/-14, 414 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`; technical summary: Covers "[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy, touching `TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy`; `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py
@@ -0,0 +1,105 @@
+"""MI35x Kimi-K2 GSM8K Completion Evaluation Test (8-GPU)
+Tests moonshotai/Kimi-K2-Instruct-0905 with GSM8K few-shot benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-kimi-k2 suite
+"""
+import os
+import unittest
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18064 - fix kimi k2.5's moe gemm config init

- Link: https://github.com/sgl-project/sglang/pull/18064
- Status/date: merged / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix kimi k2.5's moe gemm config init"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/managers/scheduler.py`; technical summary: Covers "fix kimi k2.5's moe gemm config init"; the main implementation surface is `python/sglang/srt/managers/scheduler.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunks: -485,7 +485,12 @@ def init_tokenizer(self):; symbols: init_tokenizer, init_moe_gemm_config, touching `init_tokenizer, init_moe_gemm_config`.
- Code diff details:
  - `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunks: -485,7 +485,12 @@ def init_tokenizer(self):; symbols: init_tokenizer, init_moe_gemm_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/managers/scheduler.py
@@ -485,7 +485,12 @@ def init_tokenizer(self):
-        if hasattr(self.model_config.hf_config, "num_experts_per_tok"):
+        # For the MM models, check the text_config for MoE settings
+        config_to_check = getattr(
+            self.model_config.hf_config, "text_config", self.model_config.hf_config
+        )
+        if hasattr(config_to_check, "num_experts_per_tok"):
```

- Reviewed files:
  - runtime: `python/sglang/srt/managers/scheduler.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/scheduler.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18370 - [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list

- Link: https://github.com/sgl-project/sglang/pull/18370
- Status/date: merged / 2026-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `7b8365931085`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +30/-1, 66 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunks: -34,6 +34,7; -643,6 +644,15 @@ def vision_tower_forward_auto(; symbols: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward, touching `vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunks: -34,6 +34,7; -643,6 +644,15 @@ def vision_tower_forward_auto(; symbols: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -34,6 +34,7 @@
+from sglang.srt.models.utils import WeightsMapper
@@ -643,6 +644,15 @@ def vision_tower_forward_auto(
+    # Support nvidia/Kimi-K2.5-NVFP4 naming: language_model.layers.*.
+    # Ref: HF config.json for nvidia/Kimi-K2.5-NVFP4
+    # https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/config.json
+    hf_to_sglang_mapper = WeightsMapper(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +13/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18440 - [Kimi-K2.5] Fix missing `quant_config` in `KimiK25`

- Link: https://github.com/sgl-project/sglang/pull/18440
- Status/date: merged / 2026-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `071bf2ce094c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi-K2.5] Fix missing `quant_config` in `KimiK25`"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[Kimi-K2.5] Fix missing `quant_config` in `KimiK25`"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunks: -662,6 +662,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunks: -662,6 +662,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -662,6 +662,7 @@ def __init__(
+        self.quant_config = quant_config
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18269 - [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18269
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`; associated commits `d84d2063d32a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +250/-10, 318 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`; technical summary: Covers "[AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy, touching `TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy`; `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy, touching `TestKimiK25EvalAMD, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py
@@ -0,0 +1,106 @@
+"""MI35x Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests moonshotai/Kimi-K2.5 with GSM8K few-shot benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-kimi-k25 suite
+"""
+import os
+import unittest
diff -- test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py
@@ -0,0 +1,104 @@
+"""AMD Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU)
+Tests moonshotai/Kimi-K2.5 with GSM8K few-shot benchmark on MI325.
+Registry: nightly-amd-accuracy-8-gpu-kimi-k25 suite
+"""
+import os
+import unittest
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18849 - [PCG] support piecewise cuda graph for kimi-linear model

- Link: https://github.com/sgl-project/sglang/pull/18849
- Status/date: merged / 2026-02-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `bf5238835459`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +157/-71, 423 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PCG] support piecewise cuda graph for kimi-linear model"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "[PCG] support piecewise cuda graph for kimi-linear model"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +61/-42 (103 lines); hunks: -16,12 +16,13; -194,48 +195,46 @@ def __init__(; symbols: __init__, forward_qkvbfg, forward_qkvbfg_fused, touching `__init__, forward_qkvbfg, forward_qkvbfg_fused`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +61/-42 (103 lines); hunks: -16,12 +16,13; -194,48 +195,46 @@ def __init__(; symbols: __init__, forward_qkvbfg, forward_qkvbfg_fused
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -16,12 +16,13 @@
-from sglang.srt.layers.dp_attention import get_attention_tp_size
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
+    QKVParallelLinear,
@@ -194,48 +195,46 @@ def __init__(
+            # Fuse: q, k, v, beta (column parallel) + f_a, g_a (replicated)
-            self.fused_qkvbfg_proj = MergedColumnParallelRepeatedLinear(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +61/-42
- Risk and verification: The diff ships test coverage in `test/registered/models/test_kimi_linear_models_pcg.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18689 - Add DP ViT support for Kimi K2.5

- Link: https://github.com/sgl-project/sglang/pull/18689
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `5a7ae059e37f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-4, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DP ViT support for Kimi K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "Add DP ViT support for Kimi K2.5"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunks: -35,6 +35,8; -475,9 +477,10 @@ class MoonViT3dPretrainedModel(nn.Module):; symbols: MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector, touching `MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunks: -35,6 +35,8; -475,9 +477,10 @@ class MoonViT3dPretrainedModel(nn.Module):; symbols: MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -35,6 +35,8 @@
+from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
+from sglang.srt.server_args import get_global_server_args
@@ -475,9 +477,10 @@ class MoonViT3dPretrainedModel(nn.Module):
-    def __init__(self, config, *inputs, **kwargs):
+    def __init__(self, config, *inputs, use_data_parallel: bool = False, **kwargs):
+        self.config = config
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +20/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19120 - fix KimiK2Detector regex patterns with re.DOTALL

- Link: https://github.com/sgl-project/sglang/pull/19120
- Status/date: merged / 2026-02-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`; associated commits `677b66af805d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-3, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix KimiK2Detector regex patterns with re.DOTALL"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "fix KimiK2Detector regex patterns with re.DOTALL"; the main implementation surface is `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunks: -40,11 +40,13 @@ def __init__(self):; -87,7 +89,7 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: __init__, detect_and_parse, touching `__init__, detect_and_parse`.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunks: -40,11 +40,13 @@ def __init__(self):; -87,7 +89,7 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: __init__, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -40,11 +40,13 @@ def __init__(self):
-            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>"
+            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>",
+            re.DOTALL,
-            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)"
+            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)",
+            re.DOTALL,
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/kimik2_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18434 - [Fix] Kimi K2.5 support pp

- Link: https://github.com/sgl-project/sglang/pull/18434
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `4a3a787f1e1f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-13, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Kimi K2.5 support pp"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[Fix] Kimi K2.5 support pp"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunks: -30,7 +30,7; -722,6 +722,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunks: -30,7 +30,7; -722,6 +722,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -30,7 +30,7 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
@@ -722,6 +722,7 @@ def forward(
+        pp_proxy_tensors: Optional[PPProxyTensors] = None,
@@ -731,6 +732,7 @@ def forward(
+            pp_proxy_tensors=pp_proxy_tensors,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19181 - [Kernel Slimming] Migrate marlin moe kernel to JIT

- Link: https://github.com/sgl-project/sglang/pull/19181
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +3780/-4, 3825 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel Slimming] Migrate marlin moe kernel to JIT"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`; technical summary: Covers "[Kernel Slimming] Migrate marlin moe kernel to JIT"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4 (10 lines); hunks: -10,6 +10,8; -142,7 +144,7 @@ def fused_marlin_moe(; symbols: get_scalar_type, fused_marlin_moe, touching `get_scalar_type, fused_marlin_moe`; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunks: -0,0 +1,1896; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunks: -0,0 +1,1089; `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm, touching `stack_and_dev, _get_scalar_type, _setup_moe_weights`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4 (10 lines); hunks: -10,6 +10,8; -142,7 +144,7 @@ def fused_marlin_moe(; symbols: get_scalar_type, fused_marlin_moe
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunks: -0,0 +1,1896
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunks: -0,0 +1,1089
  - `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm
  - `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: stack_and_dev, _make_inputs, _run_jit, _run_aot
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
@@ -10,6 +10,8 @@
+    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
@@ -142,7 +144,7 @@ def fused_marlin_moe(
-    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
+    intermediate_cache1 = moe_wna16_marlin_gemm(
@@ -161,7 +163,7 @@ def fused_marlin_moe(
-        b_q_type_id=scalar_type1.id,
diff -- python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h
@@ -0,0 +1,1896 @@
+/*
+ * Modified by Neural Magic
+ * Copyright (C) Marlin.2024 Elias Frantar
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
diff -- python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh
@@ -0,0 +1,1089 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0; `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0; `python/sglang/jit_kernel/moe_wna16_marlin.py` added +172/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel.h` added +37/-0
  - tests: `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19331 - [NPU] support Kimi-K2.5 on NPU

- Link: https://github.com/sgl-project/sglang/pull/19331
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `86eb80007e78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +23/-3, 80 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] support Kimi-K2.5 on NPU"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[NPU] support Kimi-K2.5 on NPU"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunks: -9,6 +9,7; -37,13 +38,15; symbols: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights, touching `apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunks: -9,6 +9,7; -37,13 +38,15; symbols: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -9,6 +9,7 @@
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -37,13 +38,15 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_npu
+_is_npu = is_npu()
@@ -197,7 +200,7 @@ def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +14/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19228 - [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning

- Link: https://github.com/sgl-project/sglang/pull/19228
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +486/-23, 892 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`; technical summary: Covers "[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunks: -32,6 +32,10; -132,6 +136,7 @@ def benchmark_config(; symbols: benchmark_config, get_kernel_wrapper, touching `benchmark_config, get_kernel_wrapper`; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunks: -28,6 +28,10; -44,6 +48,7 @@ def benchmark_config(; symbols: benchmark_config, run, touching `benchmark_config, run`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunks: -32,6 +32,10; -132,6 +136,7 @@ def benchmark_config(; symbols: benchmark_config, get_kernel_wrapper
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunks: -28,6 +28,10; -44,6 +48,7 @@ def benchmark_config(; symbols: benchmark_config, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5 (28 lines); hunks: -38,6 +38,10 @@ def get_model_config(; -46,11 +50,19 @@ def get_model_config(; symbols: get_model_config, get_config_filename
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 16,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 16,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 1,
diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py
@@ -32,6 +32,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19703 - [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT

- Link: https://github.com/sgl-project/sglang/pull/19703
- Status/date: open / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +576/-1, 588 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`; technical summary: Covers "[JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7; `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunks: -0,0 +1,317; `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: check_correctness, benchmark, fn, touching `check_correctness, benchmark, fn`; `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts, touching `_reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7
  - `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: check_correctness, benchmark, fn
  - `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts
  - `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: _jit_kimi_k2_moe_fused_gate_module, _kimi_k2_moe_fused_gate_op, kimi_k2_moe_fused_gate
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -84,7 +84,7 @@
-        from sgl_kernel import kimi_k2_moe_fused_gate
+        from sglang.jit_kernel.kimi_k2_moe_fused_gate import kimi_k2_moe_fused_gate
diff -- python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh
@@ -0,0 +1,317 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/utils.cuh>
+#include <dlpack/dlpack.h>
+#include <tvm/ffi/container/tensor.h>
+#include <cfloat>
diff -- python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py
@@ -0,0 +1,111 @@
+import itertools
+import torch
+import triton
+import triton.testing
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +1/-1; `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0; `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0; `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0
  - tests: `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19689 - feat: support Kimi K2.5 for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/19689
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `85f7a0aa3077`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-0, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support Kimi K2.5 for Eagle3"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "feat: support Kimi K2.5 for Eagle3"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,5 +786,34 @@ def get_model_config_for_expert_location(cls, config: KimiK...; symbols: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head, touching `get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,5 +786,34 @@ def get_model_config_for_expert_location(cls, config: KimiK...; symbols: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -786,5 +786,34 @@ def get_model_config_for_expert_location(cls, config: KimiK25Config):
+    def set_eagle3_layers_to_capture(
+        self, layer_ids: Optional[List[int]] = None
+    ) -> None:
+        """Set the layers to capture for EAGLE3 speculative decoding."""
+        if not hasattr(self.language_model, "set_eagle3_layers_to_capture"):
+            raise AttributeError(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +29/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19959 - Fix Kimi K2.5 PP layer range exposure for PD disaggregation

- Link: https://github.com/sgl-project/sglang/pull/19959
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `069d4c577b39`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Kimi K2.5 PP layer range exposure for PD disaggregation"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "Fix Kimi K2.5 PP layer range exposure for PD disaggregation"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunks: -719,6 +719,14 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, start_layer, end_layer, forward, touching `pad_input_ids, start_layer, end_layer`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunks: -719,6 +719,14 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, start_layer, end_layer, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -719,6 +719,14 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
+    @property
+    def start_layer(self) -> int:
+        return self.language_model.start_layer
+    @property
+    def end_layer(self) -> int:
+        return self.language_model.end_layer
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19802 - [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2

- Link: https://github.com/sgl-project/sglang/pull/19802
- Status/date: merged / 2026-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_kimi_k25.py`; associated commits `011806c41999`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +72/-53, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_kimi_k25.py`; technical summary: Covers "[Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2"; the main implementation surface is `test/registered/8-gpu-models/test_kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: TestKimiK25, for, test_kimi_k25, touching `TestKimiK25, for, test_kimi_k25`.
- Code diff details:
  - `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: TestKimiK25, for, test_kimi_k25
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_kimi_k25.py
@@ -0,0 +1,72 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_kimi_k2.py`, `test/registered/8-gpu-models/test_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20747 - fix piecewise cuda graph support for Kimi-K2.5 model

- Link: https://github.com/sgl-project/sglang/pull/20747
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `24a27d532084`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix piecewise cuda graph support for Kimi-K2.5 model"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "fix piecewise cuda graph support for Kimi-K2.5 model"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunks: -716,6 +716,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunks: -716,6 +716,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -716,6 +716,8 @@ def __init__(
+        self.model = self.language_model.model
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19552 - [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection

- Link: https://github.com/sgl-project/sglang/pull/19552
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/kimik2_detector.py`, `test/registered/function_call/test_kimik2_detector.py`; associated commits `c562e0d13ba9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +700/-19, 799 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feat] Enhance Kimi-K2/K2.5 function call and reasoning detection"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`; technical summary: Covers "[feat] Enhance Kimi-K2/K2.5 function call and reasoning detection"; the main implementation surface is `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunks: -0,0 +1,667; symbols: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp, touching `_make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic`; `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunks: -15,10 +15,25; -38,22 +53,24 @@ def __init__(self):; symbols: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call, touching `_strip_special_tokens, KimiK2Detector, __init__`.
- Code diff details:
  - `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunks: -0,0 +1,667; symbols: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunks: -15,10 +15,25; -38,22 +53,24 @@ def __init__(self):; symbols: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call
- Key code excerpts:

```diff
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -0,0 +1,667 @@
+import json
+import unittest
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.kimik2_detector import (
+    KimiK2Detector as KimiK2FuncDetector,
+)
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -15,10 +15,25 @@
+_KIMI_K2_SPECIAL_TOKENS = [
+    "<|tool_calls_section_begin|>",
+    "<|tool_calls_section_end|>",
+    "<|tool_call_begin|>",
+    "<|tool_call_end|>",
+    "<|tool_call_argument_begin|>",
```

- Reviewed files:
  - tests: `test/registered/function_call/test_kimik2_detector.py` added +667/-0
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_kimik2_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20396 - perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path

- Link: https://github.com/sgl-project/sglang/pull/20396
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_linear.py`; associated commits `db995fba4790`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-10, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/kimi_linear.py`; technical summary: Covers "perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path"; the main implementation surface is `python/sglang/srt/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -4,7 +4,6; -399,9 +398,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -4,7 +4,6; -399,9 +398,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -4,7 +4,6 @@
-from einops import rearrange
@@ -399,9 +398,11 @@ def forward(
-        norm_gate = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
+        norm_gate = g_proj_states.unflatten(
+            -1, (-1, self.head_dim)
+        )  # ... (h d) -> ... h d
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/linear/kda_backend.py`, `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21004 - [Fix] Add EPLB rebalance support for Kimi K2.5

- Link: https://github.com/sgl-project/sglang/pull/21004
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `01ccdb91b162`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Add EPLB rebalance support for Kimi K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[Fix] Add EPLB rebalance support for Kimi K2.5"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunks: -767,6 +767,10 @@ def start_layer(self) -> int:; symbols: start_layer, end_layer, routed_experts_weights_of_layer, forward, touching `start_layer, end_layer, routed_experts_weights_of_layer`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunks: -767,6 +767,10 @@ def start_layer(self) -> int:; symbols: start_layer, end_layer, routed_experts_weights_of_layer, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -767,6 +767,10 @@ def start_layer(self) -> int:
+    @property
+    def routed_experts_weights_of_layer(self):
+        return self.language_model._routed_experts_weights_of_layer.value
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21391 - Fix Kimi K2.5 dp attention+ spec decoding launch crash

- Link: https://github.com/sgl-project/sglang/pull/21391
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_kimi_k25.py`; associated commits `8c3ccef2d94e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +23/-2, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Kimi K2.5 dp attention+ spec decoding launch crash"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/8-gpu-models/test_kimi_k25.py`; technical summary: Covers "Fix Kimi K2.5 dp attention+ spec decoding launch crash"; the main implementation surface is `test/registered/8-gpu-models/test_kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25, touching `test_kimi_k25`.
- Code diff details:
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_kimi_k25.py
@@ -38,11 +38,15 @@ def test_kimi_k25(self):
-            "--mem-frac=0.85",
+        dp_attn_args = [
+            "--dp=8",
+            "--enable-dp-attention",
+        ]
@@ -56,6 +60,12 @@ def test_kimi_k25(self):
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21741 - [1/N] feat: support compressed-tensors w4afp8 MoE

- Link: https://github.com/sgl-project/sglang/pull/21741
- Status/date: open / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1657/-37, 1828 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] feat: support compressed-tensors w4afp8 MoE"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; technical summary: Covers "[1/N] feat: support compressed-tensors w4afp8 MoE"; the main implementation surface is `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability, touching `_unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__`; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0 (62 lines); hunks: -429,6 +429,68 @@ def silu_and_mul_masked_post_quant_fwd(; symbols: silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe, silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe, touching `silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8 (35 lines); hunks: -43,6 +43,7; -304,15 +305,16 @@ def _quantization_scheme_map_from_config(; symbols: _quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8, _is_static_tensor_w8a8, touching `_quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6 (25 lines); hunks: -13,11 +13,11; -29,6 +29,7; symbols: cutlass_w4a8_moe, touching `cutlass_w4a8_moe`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0 (62 lines); hunks: -429,6 +429,68 @@ def silu_and_mul_masked_post_quant_fwd(; symbols: silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe, silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8 (35 lines); hunks: -43,6 +43,7; -304,15 +305,16 @@ def _quantization_scheme_map_from_config(; symbols: _quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8, _is_static_tensor_w8a8
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6 (25 lines); hunks: -13,11 +13,11; -29,6 +29,7; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -41,4 +42,5
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py
@@ -0,0 +1,315 @@
+"""W4AFP8 MoE scheme: INT4 group-quantized weights + FP8 dynamic activations.
+Loads INT4 weights from compressed-tensors pack-quantized format,
+converts to CUTLASS W4A8 layout, and runs CUTLASS grouped GEMM
+with dynamic FP8 activation quantization.
+"""
+from __future__ import annotations
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -429,6 +429,68 @@ def silu_and_mul_masked_post_quant_fwd(
+@triton.jit
+def silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe(
+    input_ptr,
+    scale_ptr,
+    num_tokens_tensor_ptr,
+    intermediate_size,
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
@@ -43,6 +43,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py` modified +2/-0; `python/sglang/srt/layers/quantization/compressed_tensors/utils.py` modified +1/-0
  - other: `benchmark/kernels/quantization/bench_w4a8_moe_decode.py` added +887/-0
  - tests: `python/sglang/test/test_cutlass_w4a8_moe.py` modified +66/-23
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_per_tensor_absmax_fp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21898 - [CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8

- Link: https://github.com/sgl-project/sglang/pull/21898
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_kimi_k25.py`; associated commits `648632b6c41f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-23, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/8-gpu-models/test_kimi_k25.py`; technical summary: Covers "[CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8"; the main implementation surface is `test/registered/8-gpu-models/test_kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23 (27 lines); hunks: -10,19 +10,13; -31,13 +25,6 @@ def test_kimi_k25(self):; symbols: TestKimiK25, for, test_kimi_k25, touching `TestKimiK25, for, test_kimi_k25`.
- Code diff details:
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23 (27 lines); hunks: -10,19 +10,13; -31,13 +25,6 @@ def test_kimi_k25(self):; symbols: TestKimiK25, for, test_kimi_k25
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_kimi_k25.py
@@ -10,19 +10,13 @@
-EAGLE3_DRAFT_MODEL_PATH = "AQ-MedAI/Kimi-K25-eagle3"
-    Two variants:
-    - basic: TP=8 + tool/reasoning parsers
-    - eagle3: TP=8 + EAGLE3 speculative decoding with draft model
-    Each variant runs BOTH:
-    - Performance test (using NightlyBenchmarkRunner)
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21213 - [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…

- Link: https://github.com/sgl-project/sglang/pull/21213
- Status/date: merged / 2026-04-05
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; associated commits `dd49127fe612`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +81/-83, 319 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; technical summary: Covers "[AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models, touching `ModelConfig, get_kimi_k25_mxfp4_models`; `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass, touching `TestKimiK25MXFP4, setUpClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py
@@ -1,4 +1,4 @@
-"""MI35x Kimi-K2.5-MXFP4 aiter MLA backend accuracy tests (4-GPU)
+"""MI35x Kimi-K2.5-MXFP4 aiter MLA backend accuracy tests (8-GPU)
@@ -7,13 +7,6 @@
-NOTE: TP must be <= 4 for Kimi-K2.5 with the aiter MLA kernel.
-Kimi-K2.5 has num_attention_heads=64; with tp_size=8 that gives
-64/8 = 8 heads per GPU, but the aiter ASM MLA kernel requires
diff -- test/registered/amd/test_kimi_k25_mxfp4.py
@@ -1,14 +1,8 @@
-"""Kimi-K2.5-MXFP4 aiter MLA backend test (4-GPU, FP8 KV cache)
+"""Kimi-K2.5-MXFP4 aiter MLA backend test (8-GPU, FP8 KV cache)
-NOTE: TP must be <= 4 for Kimi-K2.5 with the aiter MLA kernel.
-Kimi-K2.5 has num_attention_heads=64; with tp_size=8 that gives
-64/8 = 8 heads per GPU, but the aiter ASM MLA kernel requires
-heads_per_gpu % 16 == 0. With tp_size=4: 64/4 = 16 heads, which
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12; `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22208 - [AMD] Optimize fused MoE kernel config for small-M decode on gfx950

- Link: https://github.com/sgl-project/sglang/pull/22208
- Status/date: open / 2026-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-6, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Optimize fused MoE kernel config for small-M decode on gfx950"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; technical summary: Covers "[AMD] Optimize fused MoE kernel config for small-M decode on gfx950"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunks: -191,12 +191,26 @@ def get_default_config(; symbols: get_default_config, touching `get_default_config`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunks: -191,12 +191,26 @@ def get_default_config(; symbols: get_default_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py
@@ -191,12 +191,26 @@ def get_default_config(
-            config = {
-                "BLOCK_SIZE_M": 16,
-                "BLOCK_SIZE_N": 32,
-                "BLOCK_SIZE_K": 64,
-                "GROUP_SIZE_M": 1,
-            }
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22188 - [AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)

- Link: https://github.com/sgl-project/sglang/pull/22188
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_kimi_k25_mxfp4.py`; associated commits `e14876742a08`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `test/registered/amd/test_kimi_k25_mxfp4.py`; technical summary: Covers "[AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)"; the main implementation surface is `test/registered/amd/test_kimi_k25_mxfp4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0 (3 lines); hunks: -27,6 +27,7; -36,6 +37,8 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0 (3 lines); hunks: -27,6 +27,7; -36,6 +37,8 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/test_kimi_k25_mxfp4.py
@@ -27,6 +27,7 @@
+KIMI_K25_MXFP4_REVISION = "b071bc6f8eb042e093e14f3b8bdbad71c18e09d3"
@@ -36,6 +37,8 @@ def setUpClass(cls):
+            "--revision",
+            KIMI_K25_MXFP4_REVISION,
```

- Reviewed files:
  - tests: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_kimi_k25_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22269 - [EPD][VLM] Support Kimi K25 EPD

- Link: https://github.com/sgl-project/sglang/pull/22269
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; associated commits `42ffb168b311`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +166/-42, 348 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[EPD][VLM] Support Kimi K25 EPD"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "[EPD][VLM] Support Kimi K25 EPD"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +48/-35 (83 lines); hunks: -708,33 +708,32 @@ def __init__(; -761,15 +760,22 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: M...; symbols: __init__, get_image_feature, pad_input_ids, start_layer, touching `__init__, get_image_feature, pad_input_ids`; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0 (65 lines); hunks: -4,6 +4,7; -55,6 +56,70 @@ async def process_mm_data_async(; symbols: process_mm_data_async, _num_image_tokens_from_grid, get_mm_data, _process_and_collect_mm_items, touching `process_mm_data_async, _num_image_tokens_from_grid, get_mm_data`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +48/-35 (83 lines); hunks: -708,33 +708,32 @@ def __init__(; -761,15 +760,22 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: M...; symbols: __init__, get_image_feature, pad_input_ids, start_layer
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0 (65 lines); hunks: -4,6 +4,7; -55,6 +56,70 @@ async def process_mm_data_async(; symbols: process_mm_data_async, _num_image_tokens_from_grid, get_mm_data, _process_and_collect_mm_items
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -708,33 +708,32 @@ def __init__(
-        self.language_model = DeepseekV3ForCausalLM(
-            config.text_config,
-            quant_config,
-            prefix=(
-                "language_model" if isinstance(quant_config, ModelSlimConfig) else ""
-            ),
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -4,6 +4,7 @@
+    Modality,
@@ -55,6 +56,70 @@ async def process_mm_data_async(
+    def _num_image_tokens_from_grid(self, grid_thw: torch.Tensor) -> int:
+        # Kimi-K2.5 applies temporal pooling and spatial 2D merge in vision tower.
+        # The output sequence length per image is h*w/(merge_h*merge_w).
+        merge_h, merge_w = self.hf_config.vision_config.merge_kernel_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +48/-35; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22488 - Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation

- Link: https://github.com/sgl-project/sglang/pull/22488
- Status/date: open / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +794/-53, 890 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`; technical summary: Covers "Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -65,7 +65,6; -120,9 +119,11 @@ def fused_topk_deepseek(; symbols: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped, touching `fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess`; `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344; `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts, touching `_reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped`; `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped, touching `_jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -65,7 +65,6; -120,9 +119,11 @@ def fused_topk_deepseek(; symbols: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped
  - `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344
  - `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts
  - `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -65,7 +65,6 @@
-from sglang.srt.utils.patch_torch import register_fake_if_exists
@@ -120,9 +119,11 @@ def fused_topk_deepseek(
-        from sgl_kernel import kimi_k2_moe_fused_gate
-    except ImportError as e:
-        pass
+        from sglang.jit_kernel.moe_fused_gate_ungrouped import (
diff -- python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu
@@ -0,0 +1,344 @@
+/* Copyright 2025 SGLang Team. All Rights Reserved.
+Licensed under the Apache License, Version 2.0 (the "License");
+you may not use this file except in compliance with the License.
+You may obtain a copy of the License at
+    http://www.apache.org/licenses/LICENSE-2.0
+Unless required by applicable law or agreed to in writing, software
diff -- python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py
@@ -0,0 +1,276 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +94/-53; `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0; `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0
  - tests: `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22381 - [Lora] Lora kimi support

- Link: https://github.com/sgl-project/sglang/pull/22381
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`; associated commits `6d79c6099545`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +188/-12, 248 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Lora] Lora kimi support"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`; technical summary: Covers "[Lora] Lora kimi support"; the main implementation surface is `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy, touching `kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff`.
- Code diff details:
  - `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy
- Key code excerpts:

```diff
diff -- test/registered/lora/test_lora_kimi_k25_logprob_diff.py
@@ -0,0 +1,150 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - tests: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0
- Risk and verification: The diff ships test coverage in `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22496 - [Feature] kimi k25 w4a16 support deepep low latency

- Link: https://github.com/sgl-project/sglang/pull/22496
- Status/date: open / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +4882/-25, 5138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] kimi k25 w4a16 support deepep low latency"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; technical summary: Covers "[Feature] kimi k25 w4a16 support deepep low latency"; the main implementation surface is `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunks: -39,15 +39,222; -355,6 +562,461 @@ def create_moe_runner(; symbols: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd, touching `_get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd`; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3 (59 lines); hunks: -56,7 +56,7; -386,6 +386,7 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core, combine_a, touching `dispatch_a, _dispatch_core, combine_a`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0 (44 lines); hunks: -10,6 +10,7; -37,6 +38,7; symbols: __init__, run_moe_core, get_moe_impl_class, touching `__init__, run_moe_core, get_moe_impl_class`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0 (14 lines); hunks: -1041,3 +1041,17 @@ def apply_without_routing_weights(; symbols: apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll, touching `apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunks: -39,15 +39,222; -355,6 +562,461 @@ def create_moe_runner(; symbols: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3 (59 lines); hunks: -56,7 +56,7; -386,6 +386,7 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core, combine_a
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0 (44 lines); hunks: -10,6 +10,7; -37,6 +38,7; symbols: __init__, run_moe_core, get_moe_impl_class
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0 (14 lines); hunks: -1041,3 +1041,17 @@ def apply_without_routing_weights(; symbols: apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0 (1948 lines); hunks: -0,0 +1,1948
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py
@@ -39,15 +39,222 @@
+_LOW_LATENCY_PROFILE_LOG = get_bool_env_var("SGLANG_DEEPEP_LOW_LATENCY_PROFILE_LOG")
+_DEEPEP_LL_GRAPH_DEBUG = get_bool_env_var("SGLANG_DEEPEP_LL_GRAPH_DEBUG")
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+logger = logging.getLogger(__name__)
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
-logger = logging.getLogger(__name__)
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -56,7 +56,7 @@
+_LOW_LATENCY_PROFILE_LOG = get_bool_env_var("SGLANG_DEEPEP_LOW_LATENCY_PROFILE_LOG")
@@ -386,6 +386,7 @@ def dispatch_a(
+            and get_moe_runner_backend().is_deep_gemm()
@@ -466,7 +467,12 @@ def _dispatch_core(
-            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
+            expert_alignment=(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -10,6 +10,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +1264/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/elementwise/mask_silu_and_mul.cuh`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel_direct.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22368 - [VLM] GPU Image Preprocessing for Kimi-K2.5

- Link: https://github.com/sgl-project/sglang/pull/22368
- Status/date: merged / 2026-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/kimi_k25.py`; associated commits `16f306fd85b6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +344/-48, 438 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] GPU Image Preprocessing for Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "[VLM] GPU Image Preprocessing for Kimi-K2.5"; the main implementation surface is `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41 (370 lines); hunks: -1,7 +1,12; -16,11 +21,317; symbols: navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw, _process_single_image, touching `navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41 (370 lines); hunks: -1,7 +1,12; -16,11 +21,317; symbols: navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw, _process_single_image
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -1,7 +1,12 @@
+import math
-from typing import Dict, List, Tuple, Union
+from collections import defaultdict
+from typing import Dict, List, Union
+import numpy as np
+import torch.nn.functional as F
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41
- Risk and verification: Runtime changes concentrate in `python/sglang/benchmark/datasets/image.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22806 - feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading

- Link: https://github.com/sgl-project/sglang/pull/22806
- Status/date: open / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +548/-9, 619 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading"; the main implementation surface is `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunks: -33,7 +33,11; -75,7 +79,7 @@ def get_config_filenames(cls) -> List[str]:; symbols: W4AFp8Config, for, __init__, get_config_filenames, touching `W4AFp8Config, for, __init__`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunks: -123,13 +123,24 @@ def do_load_weights(; symbols: do_load_weights, touching `do_load_weights`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunks: -1124,17 +1124,28 @@ def make_expert_params_mapping_fused_mxfp4(; symbols: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args, touching `make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args`; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ def override_quantization_method(self, *args, **kwargs):; -71,6 +71,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method, touching `override_quantization_method`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunks: -33,7 +33,11; -75,7 +79,7 @@ def get_config_filenames(cls) -> List[str]:; symbols: W4AFp8Config, for, __init__, get_config_filenames
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunks: -123,13 +123,24 @@ def do_load_weights(; symbols: do_load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunks: -1124,17 +1124,28 @@ def make_expert_params_mapping_fused_mxfp4(; symbols: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ def override_quantization_method(self, *args, **kwargs):; -71,6 +71,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0 (363 lines); hunks: -0,0 +1,363; symbols: _make_kimi_quant_config, TestKimiW4AFp8ConfigFromConfig, method, test_basic_parsing
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -33,7 +33,11 @@
-    """Config class for MIXED_PRECISION W4AFp8."""
+    """Config class for MIXED_PRECISION W4AFp8.
+    This is the base W4AFP8 config for DeepSeek-style checkpoints.
+    For Kimi K2.5 checkpoints, see KimiW4AFp8Config below.
+    """
@@ -75,7 +79,7 @@ def get_config_filenames(cls) -> List[str]:
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -123,13 +123,24 @@ def do_load_weights(
-        # Params for special naming rules in mixed-precision models, for example:
-        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
-        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.
-        if self.quant_config and self.quant_config.get_name() == "w4afp8":
+        # Params for input_scale in W4AFP8 quantized models.
+        # Supports both w1/w2/w3 naming (DeepSeek official checkpoints)
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -1124,17 +1124,28 @@ def make_expert_params_mapping_fused_mxfp4(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1
  - tests: `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_kimi_w4afp8_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22858 - [VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5

- Link: https://github.com/sgl-project/sglang/pull/22858
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; associated commits `8686f42acb3e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-64, 113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "[VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +6/-63 (69 lines); hunks: -42,7 +42,6; -622,59 +621,6 @@ def mm_projection_auto(; symbols: mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration, get_image_feature, touching `mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration`; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -285,10 +285,14 @@ def _gpu_call(self, text, images):; symbols: _gpu_call, _cpu_call, touching `_gpu_call, _cpu_call`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +6/-63 (69 lines); hunks: -42,7 +42,6; -622,59 +621,6 @@ def mm_projection_auto(; symbols: mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration, get_image_feature
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -285,10 +285,14 @@ def _gpu_call(self, text, images):; symbols: _gpu_call, _cpu_call
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -42,7 +42,6 @@
-KIMIV_VT_INFER_MAX_PATCH_NUM = 16328
@@ -622,59 +621,6 @@ def mm_projection_auto(
-@torch.inference_mode()
-def vision_tower_forward_auto(
-    vision_tower: torch.nn.Module,
-    pixel_values: torch.Tensor,
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -285,10 +285,14 @@ def _gpu_call(self, text, images):
+        grid_thws = grid_thws.cpu()
-            "grid_thws": grid_thws,
+            # Use SGL-standard key so get_new_expanded_mm_items() can split
+            # per-image for cache granularity (it looks up 'image_grid_thw').
+            "image_grid_thw": grid_thws,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +6/-63; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22490 - [EPD][VLM] Support Kimi VL EPD

- Link: https://github.com/sgl-project/sglang/pull/22490
- Status/date: merged / 2026-04-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_vl.py`; associated commits `e7ad7c587a35`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +268/-102, 520 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[EPD][VLM] Support Kimi VL EPD"; model line: Kimi K2/K2.5/Linear/VL; category: model support/runtime entry; main diff: `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py`; technical summary: Covers "[EPD][VLM] Support Kimi VL EPD"; the main implementation surface is `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: KimiGridMMDataMixin, to, _num_image_tokens_from_grid, _build_kimi_mm_data_from_grids, touching `KimiGridMMDataMixin, to, _num_image_tokens_from_grid`; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63 (70 lines); hunks: -9,8 +9,6; -20,6 +18,7; symbols: _get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async, _num_image_tokens_from_grid, touching `_get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async`; `python/sglang/srt/models/kimi_vl.py` modified +23/-8 (31 lines); hunks: -128,13 +128,16 @@ def __init__(; -215,6 +218,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, get_image_feature, load_weights, touching `__init__, get_image_feature, load_weights`; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1 (12 lines); hunks: -9,10 +9,11; -48,3 +49,12 @@ async def process_mm_data_async(; symbols: KimiVLImageProcessor, process_mm_data_async, get_mm_data, touching `KimiVLImageProcessor, process_mm_data_async, get_mm_data`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: KimiGridMMDataMixin, to, _num_image_tokens_from_grid, _build_kimi_mm_data_from_grids
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63 (70 lines); hunks: -9,8 +9,6; -20,6 +18,7; symbols: _get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async, _num_image_tokens_from_grid
  - `python/sglang/srt/models/kimi_vl.py` modified +23/-8 (31 lines); hunks: -128,13 +128,16 @@ def __init__(; -215,6 +218,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, get_image_feature, load_weights
  - `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1 (12 lines); hunks: -9,10 +9,11; -48,3 +49,12 @@ async def process_mm_data_async(; symbols: KimiVLImageProcessor, process_mm_data_async, get_mm_data
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_common.py
@@ -0,0 +1,113 @@
+"""Kimi-specific grid-based multimodal data helpers.
+Shared by KimiVLImageProcessor and KimiK2_5VLImageProcessor.
+"""
+from typing import Union
+import numpy as np
+import torch
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -9,8 +9,6 @@
-    Modality,
-    MultimodalDataItem,
@@ -20,6 +18,7 @@
+from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
@@ -329,7 +328,7 @@ def _get_gpu_norm_tensors(self, device="cuda"):
-class KimiK2_5VLImageProcessor(SGLangBaseProcessor):
diff -- python/sglang/srt/models/kimi_vl.py
@@ -128,13 +128,16 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63; `python/sglang/srt/models/kimi_vl.py` modified +23/-8; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/models/kimi_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13789 - [DeepEP Support] Support kimi-k2-thinking deepep

- Link: https://github.com/sgl-project/sglang/pull/13789
- Status/date: closed / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +674/-0, 753 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepEP Support] Support kimi-k2-thinking deepep"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; technical summary: Covers "[DeepEP Support] Support kimi-k2-thinking deepep"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunks: -231,3 +231,211 @@ def fused_marlin_moe_fake(; symbols: fused_marlin_moe_fake, batched_fused_marlin_moe, touching `fused_marlin_moe_fake, batched_fused_marlin_moe`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunks: -652,3 +652,153 @@ def apply(; symbols: apply, apply_deepep_normal, apply_deepep_ll, touching `apply, apply_deepep_normal, apply_deepep_ll`; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunks: -80,3 +80,91 @@ def moe_align_block_size(; symbols: moe_align_block_size, batched_moe_align_block_size, touching `moe_align_block_size, batched_moe_align_block_size`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunks: -198,6 +198,8 @@ def run_moe_core(; -208,6 +210,8 @@ def run_moe_core(; symbols: run_moe_core, combine, _is_marlin_moe, forward_marlin_moe, touching `run_moe_core, combine, _is_marlin_moe`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunks: -231,3 +231,211 @@ def fused_marlin_moe_fake(; symbols: fused_marlin_moe_fake, batched_fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunks: -652,3 +652,153 @@ def apply(; symbols: apply, apply_deepep_normal, apply_deepep_ll
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunks: -80,3 +80,91 @@ def moe_align_block_size(; symbols: moe_align_block_size, batched_moe_align_block_size
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunks: -198,6 +198,8 @@ def run_moe_core(; -208,6 +210,8 @@ def run_moe_core(; symbols: run_moe_core, combine, _is_marlin_moe, forward_marlin_moe
  - `python/sglang/srt/layers/quantization/marlin_utils.py` modified +9/-0 (9 lines); hunks: -257,6 +257,15 @@ def check_moe_marlin_supports_layer(layer: FusedMoE, group_...; symbols: check_moe_marlin_supports_layer, marlin_moe_intermediate_size, marlin_make_workspace
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
@@ -231,3 +231,211 @@ def fused_marlin_moe_fake(
+def batched_fused_marlin_moe(
+    hidden_states: torch.Tensor,
+    expert_num_tokens: torch.Tensor,
+    w1: torch.Tensor,
+    w2: torch.Tensor,
+    w1_scale: torch.Tensor,
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -652,3 +652,153 @@ def apply(
+    def apply_deepep_normal(
+        self,
+        layer: torch.nn.Module,
+        dispatch_output,
+    ) -> torch.Tensor:
+        """Apply MoE computation for DeepEP normal mode.
diff -- python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
@@ -80,3 +80,91 @@ def moe_align_block_size(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0; `python/sglang/srt/layers/quantization/marlin_utils.py` modified +9/-0
  - other: `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +140/-0; `sgl-kernel/python/sgl_kernel/moe.py` modified +29/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23186 - [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4

- Link: https://github.com/sgl-project/sglang/pull/23186
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunks: -60,6 +60,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; -160,6 +163,15 @@ def forward_absorb_prepare(; symbols: bmm_fp8, forward_absorb_prepare, touching `bmm_fp8, forward_absorb_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunks: -60,6 +60,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; -160,6 +163,15 @@ def forward_absorb_prepare(; symbols: bmm_fp8, forward_absorb_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -60,6 +60,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
+    from aiter.ops.fused_qk_norm_rope_cache_quant import (
+        fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
+    )
@@ -160,6 +163,15 @@ def forward_absorb_prepare(
+                    elif _use_aiter:
+                        q, k_nope = fused_qk_rmsnorm_bf16(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23381 - [AMD] Add MI355X Kimi-K2.6 tuning artifacts

- Link: https://github.com/sgl-project/sglang/pull/23381
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +126/-2, 150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add MI355X Kimi-K2.6 tuning artifacts"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py`; technical summary: Covers "[AMD] Add MI355X Kimi-K2.6 tuning artifacts"; the main implementation surface is `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0 (119 lines); hunks: -0,0 +1,119; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -141,7 +141,8 @@ def do_load_weights(; symbols: do_load_weights, touching `do_load_weights`; `python/sglang/srt/environ.py` modified +4/-0 (4 lines); hunks: -164,6 +164,10 @@ class Envs:; symbols: Envs, touching `Envs`; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +1/-1 (2 lines); hunks: -242,7 +242,7 @@ def __init__(self, seed: int, server_args: ServerArgs) -> None:; symbols: __init__, benchmark, touching `__init__, benchmark`.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0 (119 lines); hunks: -0,0 +1,119
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -141,7 +141,8 @@ def do_load_weights(; symbols: do_load_weights
  - `python/sglang/srt/environ.py` modified +4/-0 (4 lines); hunks: -164,6 +164,10 @@ class Envs:; symbols: Envs
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +1/-1 (2 lines); hunks: -242,7 +242,7 @@ def __init__(self, seed: int, server_args: ServerArgs) -> None:; symbols: __init__, benchmark
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json
@@ -0,0 +1,119 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 16,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8,
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -141,7 +141,8 @@ def do_load_weights(
-        with concurrent.futures.ThreadPoolExecutor() as executor:
+        max_workers = envs.SGLANG_DEEPSEEK_LOAD_MAX_WORKERS.get()
+        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
diff -- python/sglang/srt/environ.py
@@ -164,6 +164,10 @@ class Envs:
+    # None => fall back to ThreadPoolExecutor's default (min(32, cpu_count() + 4)).
+    # Lower this (e.g. to 4) for very large MoE checkpoints where unbounded
+    # parallelism causes host-memory or file-descriptor pressure.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1; `python/sglang/srt/environ.py` modified +4/-0
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23394 - [docs] sync kimi-k2.6 from sgl-cookbook

- Link: https://github.com/sgl-project/sglang/pull/23394
- Status/date: merged / 2026-04-21
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`; associated commits `d20ae9ceaa14`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-2, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] sync kimi-k2.6 from sgl-cookbook"; model line: Kimi K2/K2.5/Linear/VL; category: docs/tests/CI; main diff: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`; technical summary: Covers "[docs] sync kimi-k2.6 from sgl-cookbook"; the main implementation surface is `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2 (36 lines); hunks: -693,10 +693,42 @@ python3 eval.py ocrbench \.
- Code diff details:
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2 (36 lines); hunks: -693,10 +693,42 @@ python3 eval.py ocrbench \
- Key code excerpts:

```diff
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx
@@ -693,10 +693,42 @@ python3 eval.py ocrbench \
-'''text Output
-Pending update...
+- Dataset: [MMMU Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) standard 10-option subset (1,730 questions with images)
+- Evaluation Tool: [Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier) (inspect-ai based)
+- Settings: max_tokens=32,768, thinking mode (default), max_connections=256
+> **Important**: Kimi-K2.6 is a reasoning model. Setting `max_tokens` too low (e.g., 4096) causes the thinking process to consume the entire token budget, leaving no tokens for th
```

- Reviewed files:
  - docs: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23563 - [Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend

- Link: https://github.com/sgl-project/sglang/pull/23563
- Status/date: open / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +60/-3, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`; technical summary: Covers "[Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend"; the main implementation surface is `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2 (35 lines); hunks: -1,5 +1,6; -37,6 +38,15 @@ export const KimiK26Deployment = () => {; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0 (23 lines); hunks: -16,6 +16,7 @@ tag: NEW; -469,6 +470,28 @@ Let me search for this product and similar items for you.; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0 (3 lines); hunks: -453,6 +453,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \; -472,6 +473,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \; `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1 (2 lines); hunks: -195,7 +195,7 @@ export const KimiK25Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2 (35 lines); hunks: -1,5 +1,6; -37,6 +38,15 @@ export const KimiK26Deployment = () => {
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0 (23 lines); hunks: -16,6 +16,7 @@ tag: NEW; -469,6 +470,28 @@ Let me search for this product and similar items for you.
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0 (3 lines); hunks: -453,6 +453,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \; -472,6 +473,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  - `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1 (2 lines); hunks: -195,7 +195,7 @@ export const KimiK25Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx
@@ -1,5 +1,6 @@
+  // Speculative decoding is only supported on H200 and B300.
@@ -37,6 +38,15 @@ export const KimiK26Deployment = () => {
+    speculative: {
+      name: 'speculative',
+      title: 'Speculative Decoding',
+      condition: (values) => values.hardware === 'h200' || values.hardware === 'b300',
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx
@@ -16,6 +16,7 @@ tag: NEW
+- **Speculative Decoding**: EAGLE-based speculative decoding support for lower latency.
@@ -469,6 +470,28 @@ Let me search for this product and similar items for you.
+#### 4.2.5 Speculative Decoding
+**Nvidia**
+Deploy Kimi-K2.6 with the following command (H200/B200, all features enabled):
+'''shell Command
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx
@@ -453,6 +453,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0; `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23408 - [AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping

- Link: https://github.com/sgl-project/sglang/pull/23408
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/kimi_k25.py`; associated commits `d49561b8ae9e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-2, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/models/kimi_k25.py`; technical summary: Covers "[AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -28,6 +28,7; -661,7 +662,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -28,6 +28,7; -661,7 +662,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -28,6 +28,7 @@
+from sglang.srt.layers.quantization.quark.quark import QuarkConfig
@@ -661,7 +662,7 @@ def __init__(
-                    if isinstance(quant_config, ModelSlimConfig)
+                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23848 - [AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/23848
- Status/date: open / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +546/-28, 710 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x"; model line: Kimi K2/K2.5/Linear/VL; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`; technical summary: Covers "[AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x"; the main implementation surface is `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass, test_bench_kimi_k26, touching `generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass`; `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass, test_bench_kimi_k26, touching `generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass`; `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy, touching `TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy`; `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: TestKimiK26EvalAMD, setUpClass, tearDownClass, test_kimi_k26_gsm8k_accuracy, touching `TestKimiK26EvalAMD, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass, test_bench_kimi_k26
  - `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass, test_bench_kimi_k26
  - `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: TestKimiK26EvalAMD, setUpClass, tearDownClass, test_kimi_k26_gsm8k_accuracy
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +14/-14 (28 lines); hunks: -40,7 +40,7 @@ on:; -57,7 +57,7 @@ on:
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py
@@ -0,0 +1,152 @@
+"""MI35x Nightly performance benchmark for Kimi-K2.6 model.
+This test benchmarks moonshotai/Kimi-K2.6 with TP=8 on MI35x.
+Kimi-K2.6 shares the same architecture as Kimi-K2.5 (per the model card the
+deployment method is directly reused), so the AMD server arguments match the
+existing Kimi-K2.5 MI35x accuracy test (mixed aiter prefill + triton decode).
+The model path can be configured via KIMI_K26_MODEL_PATH environment variable.
diff -- test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py
@@ -0,0 +1,148 @@
+"""AMD Nightly performance benchmark for Kimi-K2.6 model.
+This test benchmarks moonshotai/Kimi-K2.6 with TP=8 on MI325/MI300X.
+Kimi-K2.6 shares the same architecture as Kimi-K2.5 (per the model card the
+deployment method is directly reused), so the AMD server arguments match the
+existing Kimi-K2.5 MI30x accuracy test (mixed aiter prefill + triton decode).
+The model path can be configured via KIMI_K26_MODEL_PATH environment variable.
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py
@@ -0,0 +1,110 @@
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0; `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0; `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +14/-14; `.github/workflows/nightly-test-amd.yml` modified +14/-14
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23501 - [VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw

- Link: https://github.com/sgl-project/sglang/pull/23501
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/multimodal/processors/kimi_k25.py`; associated commits `f34c20af86af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "[VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw"; the main implementation surface is `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -312,7 +312,11 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors, touching `_cpu_call, _get_gpu_norm_tensors`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -312,7 +312,11 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -312,7 +312,11 @@ def _cpu_call(self, text, images, **kwargs):
-        return self._hf_processor(text=[input_text], **kwargs)
+        out = self._hf_processor(text=[input_text], **kwargs)
+        grid_thws = out.pop("grid_thws", None)
+        if grid_thws is not None:
+            out["image_grid_thw"] = grid_thws
+        return out
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22964 - [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output

- Link: https://github.com/sgl-project/sglang/pull/22964
- Status/date: closed / 2026-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output"; model line: Kimi K2/K2.5/Linear/VL; category: bug fix; main diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`; technical summary: Covers "[fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output"; the main implementation surface is `python/sglang/srt/multimodal/processors/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunks: -312,7 +312,12 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors, touching `_cpu_call, _get_gpu_norm_tensors`.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunks: -312,7 +312,12 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors
- Key code excerpts:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -312,7 +312,12 @@ def _cpu_call(self, text, images, **kwargs):
-        return self._hf_processor(text=[input_text], **kwargs)
+        hf_processor_output = self._hf_processor(text=[input_text], **kwargs)
+        if "grid_thws" in hf_processor_output:
+            hf_processor_output["image_grid_thw"] = hf_processor_output.pop(
+                "grid_thws", None
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/multimodal/processors/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
