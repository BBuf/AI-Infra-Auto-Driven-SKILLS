# sglang Kimi K2/K2.5/Linear/VL 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` | [#23394](https://github.com/sgl-project/sglang/pull/23394) |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-Linear.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/kimi_k2_5.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/kimi-k2-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/kimi-linear-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789) |
| `python/sglang/srt/configs/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469) |
| `python/sglang/srt/configs/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/configs/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/function_call/kimik2_detector.py` | [#7940](https://github.com/sgl-project/sglang/pull/7940), [#8043](https://github.com/sgl-project/sglang/pull/8043), [#8968](https://github.com/sgl-project/sglang/pull/8968), [#10972](https://github.com/sgl-project/sglang/pull/10972), [#19120](https://github.com/sgl-project/sglang/pull/19120), [#19552](https://github.com/sgl-project/sglang/pull/19552) |
| `python/sglang/srt/models/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#18370](https://github.com/sgl-project/sglang/pull/18370), [#18434](https://github.com/sgl-project/sglang/pull/18434), [#18440](https://github.com/sgl-project/sglang/pull/18440), [#18689](https://github.com/sgl-project/sglang/pull/18689), [#19331](https://github.com/sgl-project/sglang/pull/19331), [#19689](https://github.com/sgl-project/sglang/pull/19689), [#19959](https://github.com/sgl-project/sglang/pull/19959), [#20747](https://github.com/sgl-project/sglang/pull/20747), [#21004](https://github.com/sgl-project/sglang/pull/21004), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22858](https://github.com/sgl-project/sglang/pull/22858) |
| `python/sglang/srt/models/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469), [#12660](https://github.com/sgl-project/sglang/pull/12660), [#14337](https://github.com/sgl-project/sglang/pull/14337), [#17160](https://github.com/sgl-project/sglang/pull/17160), [#17506](https://github.com/sgl-project/sglang/pull/17506), [#17731](https://github.com/sgl-project/sglang/pull/17731), [#18849](https://github.com/sgl-project/sglang/pull/18849), [#20396](https://github.com/sgl-project/sglang/pull/20396) |
| `python/sglang/srt/models/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383), [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `python/sglang/srt/models/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/multimodal/processors/kimi_common.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `python/sglang/srt/multimodal/processors/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22368](https://github.com/sgl-project/sglang/pull/22368), [#22490](https://github.com/sgl-project/sglang/pull/22490), [#22858](https://github.com/sgl-project/sglang/pull/22858) |
| `python/sglang/srt/multimodal/processors/kimi_vl.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490) |
| `test/manual/models/test_kimi_k2_models.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_kimi_k25.py` | [#19802](https://github.com/sgl-project/sglang/pull/19802), [#21391](https://github.com/sgl-project/sglang/pull/21391), [#21898](https://github.com/sgl-project/sglang/pull/21898) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` | [#21213](https://github.com/sgl-project/sglang/pull/21213) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/test_kimi_k25_mxfp4.py` | [#21213](https://github.com/sgl-project/sglang/pull/21213), [#22188](https://github.com/sgl-project/sglang/pull/22188) |
| `test/registered/amd/test_kimi_k2_instruct.py` | [#17656](https://github.com/sgl-project/sglang/pull/17656) |
| `test/registered/ascend/vlm_models/test_npu_kimi_vl_a3b_instruct.py` | 无直接 PR 号提交 |
| `test/registered/function_call/test_kimik2_detector.py` | [#19552](https://github.com/sgl-project/sglang/pull/19552) |
| `test/registered/gb300/test_kimi_k25.py` | 无直接 PR 号提交 |
| `test/registered/gb300/test_kimi_k25_nvfp4.py` | 无直接 PR 号提交 |
| `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` | [#22381](https://github.com/sgl-project/sglang/pull/22381) |
| `test/registered/models/test_kimi_linear_models.py` | 无直接 PR 号提交 |
| `test/registered/stress/test_stress_kimi_k2.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 39
- 原文档显式引用补充 PR 数: 40
- 当前文档总 PR 数: 79
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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
| 2026-04-16 | [#22964](https://github.com/sgl-project/sglang/pull/22964) | open | [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-16 | [#13789](https://github.com/sgl-project/sglang/pull/13789) | closed | [DeepEP Support] Support kimi-k2-thinking deepep | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2026-04-21 | [#23186](https://github.com/sgl-project/sglang/pull/23186) | merged | [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-04-21 | [#23394](https://github.com/sgl-project/sglang/pull/23394) | merged | [docs] sync kimi-k2.6 from sgl-cookbook | `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` |

## 逐 PR diff 审计卡

### PR #5440 - Sgl kernel fused_moe_gate support n_shared_experts

- 链接: https://github.com/sgl-project/sglang/pull/5440
- 状态/时间: merged / 2025-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+140/-38，可读 patch 351 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Sgl kernel fused_moe_gate support n_shared_experts」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py`；PR 正文未提供可用摘要。
- 实现要点: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28 (109 lines); hunks: -57,6 +57,8 @@ __device__ void moe_fused_gate_impl(; -65,6 +67,9 @@ __device__ void moe_fused_gate_impl(；`sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5 (36 lines); hunks: -19,20 +19,24; -43,8 +47,30 @@ def test_moe_fused_gate_combined(seq_length, dtype, params):; symbols: test_moe_fused_gate_combined，涉及 `test_moe_fused_gate_combined`；`sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2 (20 lines); hunks: -34,13 +34,29 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate，涉及 `topk_softmax, moe_fused_gate`；`sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2 (10 lines); hunks: -200,8 +200,14 @@ void topk_softmax(。
- 代码 diff 细节:
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28 (109 lines); hunks: -57,6 +57,8 @@ __device__ void moe_fused_gate_impl(; -65,6 +67,9 @@ __device__ void moe_fused_gate_impl(
  - `sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5 (36 lines); hunks: -19,20 +19,24; -43,8 +47,30 @@ def test_moe_fused_gate_combined(seq_length, dtype, params):; symbols: test_moe_fused_gate_combined
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2 (20 lines); hunks: -34,13 +34,29 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2 (10 lines); hunks: -200,8 +200,14 @@ void topk_softmax(
  - `sgl-kernel/csrc/common_extension.cc` modified +2/-1 (3 lines); hunks: -146,7 +146,8 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- 关键代码摘录:

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

- 已读文件:
  - other: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +81/-28; `sgl-kernel/python/sgl_kernel/moe.py` modified +18/-2; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-2; `sgl-kernel/csrc/common_extension.cc` modified +2/-1
  - tests: `sgl-kernel/tests/test_moe_fused_gate.py` modified +31/-5
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_moe_fused_gate.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5383 - [Feature] add support kimi vl model

- 链接: https://github.com/sgl-project/sglang/pull/5383
- 状态/时间: merged / 2025-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/kimi_vl.py`, `python/sglang/srt/configs/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`；关联提交 `8fefdd32c7c3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1189/-11，可读 patch 1316 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] add support kimi vl model」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py`；PR 正文摘要: add support kimi vl model：https://github.com/sgl-project/sglang/issues/5314 i just tested the code locally. it works fine! Prompt: What's in this image? image assistant Generate...。
- 实现要点: `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope，涉及 `multihead_attention, sdpa_attention, _apply_rope_input_validation`；`python/sglang/srt/models/kimi_vl.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward，涉及 `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`；`python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: KimiVLImageProcessor, __init__, process_mm_data_async，涉及 `KimiVLImageProcessor, __init__, process_mm_data_async`；`python/sglang/srt/configs/kimi_vl.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: KimiVLConfig, __init__，涉及 `KimiVLConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `python/sglang/srt/models/kimi_vl.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: KimiVLImageProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/configs/kimi_vl.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: KimiVLConfig, __init__
  - `python/sglang/srt/configs/kimi_vl_moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_vl_moonvit.py` added +639/-0; `python/sglang/srt/models/kimi_vl.py` added +308/-0; `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` added +73/-0; `python/sglang/srt/configs/kimi_vl.py` added +38/-0; `python/sglang/srt/configs/kimi_vl_moonvit.py` added +32/-0
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7940 - Support Kimi K2

- 链接: https://github.com/sgl-project/sglang/pull/7940
- 状态/时间: merged / 2025-07-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`；关联提交 `615553079dc1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+480/-3，可读 patch 568 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Kimi K2」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: Support Kimi K2 add Kimi style tool call parser read generation_config.json and update eos_ids。
- 实现要点: `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: KimiK2Detector, __init__, has_tool_call, detect_and_parse，涉及 `KimiK2Detector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: KimiK2Detector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` added +220/-0
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8007 - [Kimi K2] num_experts extends to 384

- 链接: https://github.com/sgl-project/sglang/pull/8007
- 状态/时间: open / 2025-07-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+30/-4，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K2] num_experts extends to 384」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`；PR 正文摘要: Kimi K2 has `"n_routed_experts": 384` change the code , to avoid go to `"Unexpected num_experts: "` Test Result for CUDA code on A800: ``` python sgl-kernel/benchmark/bench_moe_...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +5/-1 (6 lines); hunks: -45,6 +45,10; -321,7 +325,7 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu，涉及 `biased_grouped_topk_gpu`；`python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -246,7 +246,7 @@ def forward(self, hidden_states):; -2113,7 +2113,7 @@ def determine_num_fused_shared_experts(; symbols: forward, determine_num_fused_shared_experts，涉及 `forward, determine_num_fused_shared_experts`；`sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1 (15 lines); hunks: -39,7 +39,9 @@ __device__ inline bool cmp_eq(const T& a, const T& b) {; -417,6 +419,17 @@ std::vector moe_fused_gate(；`sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0 (9 lines); hunks: -466,6 +466,9 @@ topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gati...; -520,6 +523,9 @@ topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gati...。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +5/-1 (6 lines); hunks: -45,6 +45,10; -321,7 +325,7 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -246,7 +246,7 @@ def forward(self, hidden_states):; -2113,7 +2113,7 @@ def determine_num_fused_shared_experts(; symbols: forward, determine_num_fused_shared_experts
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1 (15 lines); hunks: -39,7 +39,9 @@ __device__ inline bool cmp_eq(const T& a, const T& b) {; -417,6 +419,17 @@ std::vector moe_fused_gate(
  - `sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0 (9 lines); hunks: -466,6 +466,9 @@ topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gati...; -520,6 +523,9 @@ topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gati...
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +5/-1; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
  - other: `sgl-kernel/csrc/moe/moe_fused_gate.cu` modified +14/-1; `sgl-kernel/csrc/cpu/topk.cpp` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8021 - perf: add kimi k2 fused_moe tuning config for h30_3e

- 链接: https://github.com/sgl-project/sglang/pull/8021
- 状态/时间: merged / 2025-07-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf: add kimi k2 fused_moe tuning config for h30_3e」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文摘要: Add h20_3e fused MoE tuning config for Kimi K2 (E=384)。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8047 - H20 tune config for Kimi

- 链接: https://github.com/sgl-project/sglang/pull/8047
- 状态/时间: merged / 2025-07-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「H20 tune config for Kimi」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文摘要: add tune config for kimi on H20 , performance improvement will be update soon.. Performance * origin Kimi Performance * Performance with our config。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8176 - feat: add h200 tp 16 kimi k2 moe config

- 链接: https://github.com/sgl-project/sglang/pull/8176
- 状态/时间: merged / 2025-07-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add h200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8178 - feat: add b200 tp 16 kimi k2 moe config

- 链接: https://github.com/sgl-project/sglang/pull/8178
- 状态/时间: merged / 2025-07-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add b200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8183 - feat: add h200 tp 16 kimi k2 moe config

- 链接: https://github.com/sgl-project/sglang/pull/8183
- 状态/时间: merged / 2025-07-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add h200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8043 - feat(function call): complete utility method for KimiK2Detector and enhance documentation

- 链接: https://github.com/sgl-project/sglang/pull/8043
- 状态/时间: merged / 2025-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`；关联提交 `01079e174ff8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+205/-56，可读 patch 404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(function call): complete utility method for KimiK2Detector and enhance documentation」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: This PR tries to implement the missing utility methods for `KimiK2Detector` and improves the the overall function call module by adding thorough documentation and eliminating th...。
- 实现要点: `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunks: -18,16 +18,21; -114,11 +119,7 @@ def parse_streaming_increment(; symbols: KimiK2Detector, __init__, parse_streaming_increment，涉及 `KimiK2Detector, __init__, parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunks: -18,16 +18,21; -114,11 +119,7 @@ def parse_streaming_increment(; symbols: KimiK2Detector, __init__, parse_streaming_increment
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8013 - [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384

- 链接: https://github.com/sgl-project/sglang/pull/8013
- 状态/时间: merged / 2025-08-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+188/-30，可读 patch 318 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`；PR 正文摘要: Follow up of PR https://github.com/sgl-project/sglang/pull/8007 @yhyang201 mentioned that the `router_gemm` should be modified to support Kimi K2 in DeepSeek architecture but nu...。
- 实现要点: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunks: -25,6 +25,10; -91,12 +95,24 @@ void dsv3_router_gemm(；`sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunks: -185,6 +185,7 @@ void invokeRouterGemmBf16Output(__nv_bfloat16* output, T con...; -232,3 +233,52 @@ template void invokeRouterGemmBf16Output (；`sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunks: -184,6 +184,7 @@ void invokeRouterGemmFloatOutput(float* output, T const* mat...; -231,3 +232,52 @@ template void invokeRouterGemmFloatOutput (；`sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunks: -13,29 +13,41; -55,29 +67,41 @@ def tflops(t_ms):; symbols: benchmark_bf16_output, runner, tflops, benchmark_float_output，涉及 `benchmark_bf16_output, runner, tflops`。
- 代码 diff 细节:
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunks: -25,6 +25,10; -91,12 +95,24 @@ void dsv3_router_gemm(
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunks: -185,6 +185,7 @@ void invokeRouterGemmBf16Output(__nv_bfloat16* output, T con...; -232,3 +233,52 @@ template void invokeRouterGemmBf16Output (
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunks: -184,6 +184,7 @@ void invokeRouterGemmFloatOutput(float* output, T const* mat...; -231,3 +232,52 @@ template void invokeRouterGemmFloatOutput (
  - `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunks: -13,29 +13,41; -55,29 +67,41 @@ def tflops(t_ms):; symbols: benchmark_bf16_output, runner, tflops, benchmark_float_output
  - `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2 (4 lines); hunks: -5,8 +5,8; symbols: test_dsv3_router_gemm
- 关键代码摘录:

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

- 已读文件:
  - other: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16; `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0; `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0; `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12
  - tests: `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_dsv3_router_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8968 - Fix kimi k2 function call format

- 链接: https://github.com/sgl-project/sglang/pull/8968
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`；关联提交 `91e2f902db0e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix kimi k2 function call format」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: Remove the space after `functions.{func_name}:{index}` in the template. Otherwise, xgrammar will mask all tokens as invalid, then the probability becomes NaN, and the engine wil...。
- 实现要点: `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3 (6 lines); hunks: -24,7 +24,7 @@ class KimiK2Detector(BaseFormatDetector):; -219,7 +219,7 @@ def structure_info(self) -> _GetInfoFunc:; symbols: KimiK2Detector, structure_info, get_info, build_ebnf，涉及 `KimiK2Detector, structure_info, get_info`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3 (6 lines); hunks: -24,7 +24,7 @@ class KimiK2Detector(BaseFormatDetector):; -219,7 +219,7 @@ def structure_info(self) -> _GetInfoFunc:; symbols: KimiK2Detector, structure_info, get_info, build_ebnf
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/kimik2_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9010 - [perf] add kimi-k2 b200 fused moe config

- 链接: https://github.com/sgl-project/sglang/pull/9010
- 状态/时间: merged / 2025-08-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[perf] add kimi-k2 b200 fused moe config」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；PR 正文摘要: Benchmark & Profiling。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9606 - Fix kimi k2 function calling format

- 链接: https://github.com/sgl-project/sglang/pull/9606
- 状态/时间: merged / 2025-08-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+117/-9，可读 patch 155 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix kimi k2 function calling format」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`；PR 正文摘要: https://github.com/sgl-project/sglang/issues/9575。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunks: -835,15 +835,23 @@ def _process_tool_calls(; -954,7 +962,11 @@ async def _process_tool_call_stream(; symbols: _process_tool_calls, _process_tool_call_stream，涉及 `_process_tool_calls, _process_tool_call_stream`；`test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunks: -6,6 +6,8; -325,6 +327,100 @@ async def test_unstreamed_tool_args_no_parser_data(self):; symbols: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk，涉及 `test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunks: -835,15 +835,23 @@ def _process_tool_calls(; -954,7 +962,11 @@ async def _process_tool_call_stream(; symbols: _process_tool_calls, _process_tool_call_stream
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunks: -6,6 +6,8; -325,6 +327,100 @@ async def test_unstreamed_tool_args_no_parser_data(self):; symbols: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9
  - tests: `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0
- 验证与风险: diff 自带测试面 `test/srt/openai_server/basic/test_serving_chat.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10612 - Replace the Kimi-K2 generated tool call idx with history tool call count

- 链接: https://github.com/sgl-project/sglang/pull/10612
- 状态/时间: merged / 2025-09-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+226/-15，可读 patch 303 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Replace the Kimi-K2 generated tool call idx with history tool call count」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`；PR 正文摘要: Related to #10600 Replace the Kimi-K2 generated tool call idx with history tool call count, make sure the tool calls indexes in history are ordered. Add a `_get_history_tool_cal...。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunks: -33,6 +33,7; -749,8 +750,9 @@ def _build_chat_response(; symbols: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls，涉及 `_build_chat_response, _process_response_logprobs, _process_tool_call_id`；`test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunks: -420,6 +420,181 @@ async def collect_first_tool_chunk():; symbols: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history，涉及 `collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunks: -33,6 +33,7; -749,8 +750,9 @@ def _build_chat_response(; symbols: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunks: -420,6 +420,181 @@ async def collect_first_tool_chunk():; symbols: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15
  - tests: `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0
- 验证与风险: diff 自带测试面 `test/srt/openai_server/basic/test_serving_chat.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10972 - fix: KimiK2Detector Improve tool call ID parsing with regex

- 链接: https://github.com/sgl-project/sglang/pull/10972
- 状态/时间: merged / 2025-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`；关联提交 `1193f13181a2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-4，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: KimiK2Detector Improve tool call ID parsing with regex」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: Enhance the tool_call_id parser, make it more robust. For random scenarios, like samples provide in https://github.com/MoonshotAI/K2-Vendor-Verfier , this change could make the...。
- 实现要点: `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunks: -50,6 +50,11 @@ def __init__(self):; -76,14 +81,18 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment，涉及 `__init__, has_tool_call, detect_and_parse`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunks: -50,6 +50,11 @@ def __init__(self):; -76,14 +81,18 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/kimik2_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12469 - Support Kimi Linear

- 链接: https://github.com/sgl-project/sglang/pull/12469
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/kimi_linear.py`, `python/sglang/srt/models/kimi_linear.py`；关联提交 `a4bf5c6ad25d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+2847/-112，可读 patch 3404 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Kimi Linear」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py`；PR 正文摘要: Support Kimi Linear model (https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct). Major work is done by @yizhang2077 . Thanks @zhiyuan1i for valuable discussion.。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: KimiMoE, __init__, forward, KimiDeltaAttention，涉及 `KimiMoE, __init__, forward`；`python/sglang/srt/configs/kimi_linear.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: KimiLinearConfig, __init__, is_mla, is_moe，涉及 `KimiLinearConfig, __init__, is_mla`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: KimiMoE, __init__, forward, KimiDeltaAttention
  - `python/sglang/srt/configs/kimi_linear.py` added +160/-0 (160 lines); hunks: -0,0 +1,160; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` added +678/-0; `python/sglang/srt/configs/kimi_linear.py` added +160/-0
- 验证与风险: diff 自带测试面 `test/srt/models/test_kimi_linear_models.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12660 - overlap shared + routed expert computation in kimi linear

- 链接: https://github.com/sgl-project/sglang/pull/12660
- 状态/时间: merged / 2025-11-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `cc2e36c352e8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+37/-5，可读 patch 101 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「overlap shared + routed expert computation in kimi linear」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: Before: `python3 -m sglang.test.send_one` After: Around 6%.。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +37/-5 (42 lines); hunks: -32,6 +32,7; -52,6 +53,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +37/-5 (42 lines); hunks: -32,6 +32,7; -52,6 +53,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +37/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13150 - Opt kimi_k2_thinking biased topk module

- 链接: https://github.com/sgl-project/sglang/pull/13150
- 状态/时间: merged / 2025-11-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+71/-14，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Opt kimi_k2_thinking biased topk module」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；PR 正文摘要: launch server Acc profiler main: pr: **33us->15us** . End2End benchmark。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunks: -600,6 +600,48 @@ def grouped_topk_cpu(; -760,20 +802,35 @@ def biased_grouped_topk_gpu(; symbols: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu，涉及 `grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunks: -600,6 +600,48 @@ def grouped_topk_cpu(; -760,20 +802,35 @@ def biased_grouped_topk_gpu(; symbols: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +71/-14
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13287 - [opt kimi k2 1 / n] Add kimi k2 moe fused gate

- 链接: https://github.com/sgl-project/sglang/pull/13287
- 状态/时间: merged / 2025-11-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+646/-0，可读 patch 684 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[opt kimi k2 1 / n] Add kimi k2 moe fused gate」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`；PR 正文摘要: Kimi K2 Acc test main branch pr Kernel benchmark Kimi K2 Profile main: pr: 14us->9us. End2End benchmark。
- 实现要点: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunks: -0,0 +1,354；`sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case，涉及 `test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case`；`sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark，涉及 `kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark`；`sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunks: -111,6 +111,41 @@ def moe_fused_gate(; symbols: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm，涉及 `moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm`。
- 代码 diff 细节:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunks: -0,0 +1,354
  - `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case
  - `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunks: -111,6 +111,41 @@ def moe_fused_gate(; symbols: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: -331,6 +331,14 @@ std::vector moe_fused_gate(
- 关键代码摘录:

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

- 已读文件:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0; `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0; `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0; `sgl-kernel/csrc/common_extension.cc` modified +6/-0; `sgl-kernel/CMakeLists.txt` modified +1/-0
  - tests: `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0
- 验证与风险: diff 自带测试面 `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13332 - [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate

- 链接: https://github.com/sgl-project/sglang/pull/13332
- 状态/时间: merged / 2025-11-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-9，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；PR 正文摘要: Follow https://github.com/sgl-project/sglang/pull/13287。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunks: -72,7 +72,7; -817,16 +817,13 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu，涉及 `biased_grouped_topk_gpu`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunks: -72,7 +72,7; -817,16 +817,13 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +6/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13374 - [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel

- 链接: https://github.com/sgl-project/sglang/pull/13374
- 状态/时间: merged / 2025-11-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+130/-173，可读 patch 400 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；PR 正文摘要: Kimi K2 Acc test main branch pr Kernel benchmark main branch(https://github.com/sgl-project/sglang/pull/13287) pr Kimi K2 Profile main branch(https://github.com/sgl-project/sgla...。
- 实现要点: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunks: -1,15 +1,9; -21,149 +15,144 @@ static constexpr int SMALL_TOKEN_THRESHOLD = 512;。
- 代码 diff 细节:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunks: -1,15 +1,9; -21,149 +15,144 @@ static constexpr int SMALL_TOKEN_THRESHOLD = 512;
- 关键代码摘录:

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

- 已读文件:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #13596 - [kimi k2 thinking] Avoid useless torch.zeros_

- 链接: https://github.com/sgl-project/sglang/pull/13596
- 状态/时间: merged / 2025-11-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+252/-256，可读 patch 598 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[kimi k2 thinking] Avoid useless torch.zeros_」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py`；PR 正文摘要: main: pr: 3.5us->2us.。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake，涉及 `get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunks: -7,13 +7,6; -56,9 +49,6; symbols: apply，涉及 `apply`；`python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunks: -52,12 +52,7; -835,6 +830,9 @@ def apply(; symbols: apply，涉及 `apply`；`python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunks: -55,7 +55,7; -1059,14 +1059,14 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunks: -0,0 +1,239; symbols: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunks: -7,13 +7,6; -56,9 +49,6; symbols: apply
  - `python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunks: -52,12 +52,7; -835,6 +830,9 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunks: -55,7 +55,7; -1059,14 +1059,14 @@ def apply(; symbols: apply
  - `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232 (232 lines); hunks: -1,18 +1,6; -67,223 +55,3 @@ def moe_wna16_marlin_gemm(; symbols: get_scalar_type, moe_wna16_marlin_gemm, fused_marlin_moe, fused_marlin_moe_fake
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12; `python/sglang/srt/layers/quantization/awq.py` modified +4/-6; `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4
  - other: `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232; `sgl-kernel/python/sgl_kernel/__init__.py` modified +1/-1
  - tests: `python/sglang/test/test_marlin_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/test_marlin_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13587 - [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size

- 链接: https://github.com/sgl-project/sglang/pull/13587
- 状态/时间: merged / 2025-11-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-6，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；PR 正文摘要: main: pr: 7.8us->6.5us.。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunks: -69,11 +69,6 @@ def moe_align_block_size(; -82,6 +77,6 @@ def moe_align_block_size(; symbols: moe_align_block_size，涉及 `moe_align_block_size`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunks: -69,11 +69,6 @@ def moe_align_block_size(; -82,6 +77,6 @@ def moe_align_block_size(; symbols: moe_align_block_size
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13466 - [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)

- 链接: https://github.com/sgl-project/sglang/pull/13466
- 状态/时间: merged / 2025-11-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；PR 正文摘要: Currently, this only supports `moonshotai/Kimi-K2-Instruct-0905` (there are some additional complication w/ some torch.cuda non tensor ops in the prefill stage that I will remov...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunks: -74,6 +74,29; symbols: _kimi_k2_moe_fused_gate，涉及 `_kimi_k2_moe_fused_gate`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunks: -74,6 +74,29; symbols: _kimi_k2_moe_fused_gate
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +23/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9405 - Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)

- 链接: https://github.com/sgl-project/sglang/pull/9405
- 状态/时间: merged / 2025-11-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v2.py`；PR 正文摘要: The token threshold was hardcoded to 1024, add an env var to make it more flexible.。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -787,12 +787,13 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: -787,12 +787,13 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -787,12 +787,13 @@ def forward(
-            DUAL_STREAM_TOKEN_THRESHOLD = 1024
+            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
-                and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
+                and get_is_capture_mode()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12759 - [Ascend] support Kimi-K2-Thinking

- 链接: https://github.com/sgl-project/sglang/pull/12759
- 状态/时间: merged / 2025-11-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+549/-170，可读 patch 871 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend] support Kimi-K2-Thinking」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`；PR 正文摘要: Kimi-K2-Think model Day 0 support on SGLang on Ascend NPU backend. Ascend has completed the adaptation of the INT4 Weight (A16W4, pergroup=32) quantization format for the K2 Thi...。
- 实现要点: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunks: -1,9 +1,11; -21,6 +23,9; symbols: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for，涉及 `npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunks: -35,12 +35,12; -314,87 +314,44 @@ def forward_npu(; symbols: forward_npu, _forward_normal, _forward_ll, npu_fused_moe_without_routing_weights_bf16，涉及 `forward_npu, _forward_normal, _forward_ll`；`python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -3979,6 +3979,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -4006,7 +4008,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunks: -217,7 +217,7 @@ def add_chunked_prefix_cache_attention_backend(backend_name):; symbols: add_chunked_prefix_cache_attention_backend，涉及 `add_chunked_prefix_cache_attention_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunks: -1,9 +1,11; -21,6 +23,9; symbols: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunks: -35,12 +35,12; -314,87 +314,44 @@ def forward_npu(; symbols: forward_npu, _forward_normal, _forward_ll, npu_fused_moe_without_routing_weights_bf16
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -3979,6 +3979,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -4006,7 +4008,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunks: -217,7 +217,7 @@ def add_chunked_prefix_cache_attention_backend(backend_name):; symbols: add_chunked_prefix_cache_attention_backend
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0; `python/sglang/srt/model_executor/model_runner.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- 链接: https://github.com/sgl-project/sglang/pull/14337
- 状态/时间: merged / 2025-12-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `6d5d76ad97dd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+0/-8，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: in an earlier pr of llama 4, i found the same conclusion as https://github.com/sgl-project/sglang/pull/9405 (it is also done in glm now). TODO there are still some in GDN and NS...。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13725 - Add Expert Parallelism (EP) support for kimi-k2-thinking

- 链接: https://github.com/sgl-project/sglang/pull/13725
- 状态/时间: merged / 2025-12-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Expert Parallelism (EP) support for kimi-k2-thinking」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；PR 正文摘要: Result:。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunks: -634,6 +634,16 @@ def apply(; -643,6 +653,8 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunks: -634,6 +634,16 @@ def apply(; -643,6 +653,8 @@ def apply(; symbols: apply
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15100 - Support piecewise cuda graph for fused marlin moe

- 链接: https://github.com/sgl-project/sglang/pull/15100
- 状态/时间: merged / 2025-12-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+55/-36，可读 patch 159 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support piecewise cuda graph for fused marlin moe」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py`；PR 正文摘要: Support piecewise cuda graph for models using `fused_marlin_moe` like moe models with gptq/awq quantization, kimi-k2-thinking model.。
- 实现要点: `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunks: -1099,32 +1099,3 @@ def _(b_q_weight, perm, size_k, size_n, num_bits):; symbols: _，涉及 `_`；`python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunks: -2,7 +2,7; -41,7 +41,7 @@ def fused_marlin_moe(; symbols: fused_marlin_moe, fused_marlin_moe_fake，涉及 `fused_marlin_moe, fused_marlin_moe_fake`；`python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunks: -80,7 +80,9 @@ def fused_experts_none_to_marlin(; -97,7 +99,7 @@ def fused_experts_none_to_marlin(; symbols: fused_experts_none_to_marlin，涉及 `fused_experts_none_to_marlin`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunks: -943,7 +943,7 @@ def apply(; -967,7 +967,7 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunks: -1099,32 +1099,3 @@ def _(b_q_weight, perm, size_k, size_n, num_bits):; symbols: _
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunks: -2,7 +2,7; -41,7 +41,7 @@ def fused_marlin_moe(; symbols: fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunks: -80,7 +80,9 @@ def fused_experts_none_to_marlin(; -97,7 +99,7 @@ def fused_experts_none_to_marlin(; symbols: fused_experts_none_to_marlin
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunks: -943,7 +943,7 @@ def apply(; -967,7 +967,7 @@ def apply(; symbols: apply
  - `test/srt/test_piecewise_cuda_graph.py` modified +35/-0 (35 lines); hunks: -214,6 +214,41 @@ def test_mgsm_accuracy(self):; symbols: test_mgsm_accuracy, TestPiecewiseCudaGraphGPTQ, setUpClass, tearDownClass
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29; `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3; `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2
  - tests: `test/srt/test_piecewise_cuda_graph.py` modified +35/-0
- 验证与风险: diff 自带测试面 `test/srt/test_piecewise_cuda_graph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15306 - Fix warp illegal instruction in kimi k2 thinking PCG

- 链接: https://github.com/sgl-project/sglang/pull/15306
- 状态/时间: merged / 2025-12-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix warp illegal instruction in kimi k2 thinking PCG」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；PR 正文摘要: The `kimi_k2_moe_fused_gate` kernels fail to initialize `indices_ptr` when no valid expert is found (when `expert_id` is out of bounds or `max_expert == -1`), leaving uninitiali...。
- 实现要点: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunks: -126,6 +126,9 @@ __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; -219,11 +222,16 @@ __global__ void kimi_k2_moe_fused_gate_kernel(。
- 代码 diff 细节:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunks: -126,6 +126,9 @@ __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; -219,11 +222,16 @@ __global__ void kimi_k2_moe_fused_gate_kernel(
- 关键代码摘录:

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

- 已读文件:
  - other: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #15347 - Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

- 链接: https://github.com/sgl-project/sglang/pull/15347
- 状态/时间: merged / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+165/-12，可读 patch 215 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`；PR 正文摘要: flashinfer has an optimized routing kernel for DeepSeek V3: https://github.com/flashinfer-ai/flashinfer/pull/2099 The API was renamed to `fused_topk_deepseek` here: https://gith...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu，涉及 `biased_grouped_topk_gpu`；`test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek，涉及 `test_fused_topk_deepseek`；`test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed，涉及 `test_a_gsm8k, test_bs_1_speed`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
  - `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek
  - `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +66/-4
  - tests: `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0; `test/srt/test_deepseek_v3_mtp.py` modified +2/-8
- 验证与风险: diff 自带测试面 `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17325 - Fix kernel selection in biased_grouped_topk_gpu

- 链接: https://github.com/sgl-project/sglang/pull/17325
- 状态/时间: merged / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix kernel selection in biased_grouped_topk_gpu」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/topk.py`；PR 正文摘要: The has supported the case since PR https://github.com/sgl-project/sglang/pull/5440, but PR https://github.com/sgl-project/sglang/pull/15347 incorrectly added a check for , whic...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunks: -795,7 +795,6 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu，涉及 `biased_grouped_topk_gpu`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunks: -795,7 +795,6 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -795,7 +795,6 @@ def biased_grouped_topk_gpu(
-        and num_fused_shared_experts == 0
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17160 - [Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code

- 链接: https://github.com/sgl-project/sglang/pull/17160
- 状态/时间: merged / 2026-01-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `e6b7c04947ee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-42，可读 patch 129 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: Refactor Kimi-Linear gate ReplicatedLinear / ColumnParallelLinear project and fused_kda_gate out of attention forward in order to avoid duplicated code in both KDA attention for...。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +13/-9 (22 lines); hunks: -15,7 +15,7; -314,6 +314,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +13/-9 (22 lines); hunks: -15,7 +15,7; -314,6 +314,14 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +13/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17506 - [Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention

- 链接: https://github.com/sgl-project/sglang/pull/17506
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `0c8165ffbd1b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+95/-90，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: Refactor Kimi-Linear based on new RadixLinearAttention. Result is correct. gsm8k Server: Client:。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +42/-37 (79 lines); hunks: -16,6 +16,7; -27,6 +28,7; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +42/-37 (79 lines); hunks: -16,6 +16,7; -27,6 +28,7; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +42/-37
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17731 - [Kimi-Linear] Remove duplicated code in kimi-linear

- 链接: https://github.com/sgl-project/sglang/pull/17731
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `1e8db1829096`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-Linear] Remove duplicated code in kimi-linear」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: There's a duplicated code in Kimi-Linear due to PR rebase. Removed it.。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +0/-1 (1 lines); hunks: -340,7 +340,6 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-1 (1 lines); hunks: -340,7 +340,6 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -340,7 +340,6 @@ def forward(
-        beta = self.b_proj(hidden_states)[0].float()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17656 - [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases

- 链接: https://github.com/sgl-project/sglang/pull/17656
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k2_instruct.py`；关联提交 `738b1ac988c3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+97/-2，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/amd/test_kimi_k2_instruct.py`；PR 正文摘要: Add kimi_k2_instruct model testcases to increase the test coverage on AMD GPUs. Change CI job partition from 2 to 3. Add 1 test script : test/registered/amd/test_kimi_k2_instruc...。
- 实现要点: `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestKimiK2Instruct0905, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_kimi_k2_instruct.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17789 - Support Kimi-K2.5 model

- 链接: https://github.com/sgl-project/sglang/pull/17789
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `479ab7a4e7e4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1053/-12，可读 patch 1193 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Kimi-K2.5 model」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunks: -0,0 +1,744; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__，涉及 `apply_rope, tpool_patch_merger, MoonViTEncoderLayer`；`python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size，涉及 `KimiK25VisionConfig, __init__, KimiK25Config`；`python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunks: -0,0 +1,88; symbols: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items，涉及 `KimiK2_5VLImageProcessor, __init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunks: -0,0 +1,744; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__
  - `python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunks: -0,0 +1,171; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunks: -0,0 +1,88; symbols: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` added +744/-0; `python/sglang/srt/configs/kimi_k25.py` added +171/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/configs/model_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- 链接: https://github.com/sgl-project/sglang/pull/17523
- 状态/时间: merged / 2026-01-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+1540/-43，可读 patch 1823 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`；PR 正文摘要: 1. Add **Kimi-K2**, **DeepSeek-V3.2** accuracy and performance tests for MI325 (MI30x) platform, update Mi35x tests, consolidate test jobs, and fix various CI failures. 2. Total...。
- 实现要点: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: ModelConfig, __post_init__, get_display_name, get_one_example，涉及 `ModelConfig, __post_init__, get_display_name`；`.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: -25,18 +25,21 @@ on:; -248,35 +251,6 @@ jobs:；`test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch，涉及 `generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass`；`test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestDeepseekV32TPMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: -25,18 +25,21 @@ on:; -248,35 +251,6 @@ jobs:
  - `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: -0,0 +1,149; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0; `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0; `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0; `test/registered/amd/perf/test_deepseek_v32_basic_perf_amd.py` added +142/-0; `test/registered/amd/accuracy/test_deepseek_v32_tc_eval_amd.py` added +123/-0
  - ci: `.github/workflows/nightly-test-amd.yml` modified +158/-35
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- 链接: https://github.com/sgl-project/sglang/pull/17624
- 状态/时间: merged / 2026-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+48/-19，可读 patch 185 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] Fix dp size > 1 for qwen3 vl model」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`；PR 正文摘要: Question In my local tests, the Qwen3-VL service fails to start when both `--mm-enable-dp-encoder` and `--enable-dp-attention` are enabled. I came across the related PR: pr17157...。
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

### PR #17991 - Fix: Avoid Double Reduce in VLM DP Attention

- 链接: https://github.com/sgl-project/sglang/pull/17991
- 状态/时间: merged / 2026-02-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+51/-12，可读 patch 132 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: Avoid Double Reduce in VLM DP Attention」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py`；PR 正文摘要: This PR fixes a redundant reduce operation in VLM DP Attention. Here’s the context: 1. After PR1 (https://github.com/sgl-project/sglang/pull/17624/changes) was opened, we introd...。
- 实现要点: `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunks: -13,11 +13,7; -687,7 +683,6 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunks: -39,6 +39,8; -126,6 +128,7 @@ def __init__(; symbols: apply_rope, __init__, forward，涉及 `apply_rope, __init__, forward`；`test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass，涉及 `test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass`；`test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunks: -187,8 +187,6 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass，涉及 `test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunks: -13,11 +13,7; -687,7 +683,6 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunks: -39,6 +39,8; -126,6 +128,7 @@ def __init__(; symbols: apply_rope, __init__, forward
  - `test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunks: -3,6 +3,7; -11,6 +12,7; symbols: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass
  - `test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunks: -187,8 +187,6 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/vision.py` modified +1/-10; `python/sglang/srt/models/kimi_k25.py` modified +3/-0
  - tests: `test/registered/distributed/test_dp_attention_large.py` modified +47/-0; `test/registered/distributed/test_dp_attention.py` modified +0/-2
- 验证与风险: diff 自带测试面 `test/registered/distributed/test_dp_attention.py`, `test/registered/distributed/test_dp_attention_large.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17895 - [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes

- 链接: https://github.com/sgl-project/sglang/pull/17895
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`；关联提交 `6fd878b41df0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+184/-14，可读 patch 414 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`；PR 正文摘要: Add kimi mi35x nightly test, Test Suite Restructuring and fix gpt-oss accuracy issue. Please help to review: @yctseng0211 @bingxche. Nightly: https://github.com/sgl-project/sgla...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy，涉及 `TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy`；`test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` renamed +0/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18064 - fix kimi k2.5's moe gemm config init

- 链接: https://github.com/sgl-project/sglang/pull/18064
- 状态/时间: merged / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix kimi k2.5's moe gemm config init」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/managers/scheduler.py`；PR 正文摘要: ` --moe-runner-backend flashinfer_trtllm` doesn't work for kimi k2.5 - b200 tp8 before this: : ~80 TPS after this: ~180 TPS。
- 实现要点: `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunks: -485,7 +485,12 @@ def init_tokenizer(self):; symbols: init_tokenizer, init_moe_gemm_config，涉及 `init_tokenizer, init_moe_gemm_config`。
- 代码 diff 细节:
  - `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunks: -485,7 +485,12 @@ def init_tokenizer(self):; symbols: init_tokenizer, init_moe_gemm_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/managers/scheduler.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/scheduler.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18370 - [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list

- 链接: https://github.com/sgl-project/sglang/pull/18370
- 状态/时间: merged / 2026-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `7b8365931085`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+30/-1，可读 patch 66 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: Make `nvidia/Kimi-K2.5-NVFP4` load cleanly in SGLang. The checkpoint’s naming and `hf_quant_config.json` exclude list use `language_model.layers.*`, which doesn’t match SGLang’s...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunks: -34,6 +34,7; -643,6 +644,15 @@ def vision_tower_forward_auto(; symbols: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward，涉及 `vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunks: -34,6 +34,7; -643,6 +644,15 @@ def vision_tower_forward_auto(; symbols: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +13/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18440 - [Kimi-K2.5] Fix missing `quant_config` in `KimiK25`

- 链接: https://github.com/sgl-project/sglang/pull/18440
- 状态/时间: merged / 2026-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `071bf2ce094c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-K2.5] Fix missing `quant_config` in `KimiK25`」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: Fix missing `quant_config` in `KimiK25` causing `nvidia/Kimi-K2.5-NVFP4` to use bf16 KV cache instead of fp8.。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunks: -662,6 +662,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunks: -662,6 +662,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -662,6 +662,7 @@ def __init__(
+        self.quant_config = quant_config
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18269 - [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18269
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`；关联提交 `d84d2063d32a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+250/-10，可读 patch 318 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`；PR 正文摘要: **1. Fix Janus-Pro crash (regression from #18503)**, 2/9 nightly: ERROR: test_mmmu_vlm_models (__main__.TestNightlyVLMMmmuEvalAMD) (model='deepseek-ai/Janus-Pro-7B') - `LogitsPr...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy，涉及 `TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy`；`test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy，涉及 `TestKimiK25EvalAMD, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18849 - [PCG] support piecewise cuda graph for kimi-linear model

- 链接: https://github.com/sgl-project/sglang/pull/18849
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `bf5238835459`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+157/-71，可读 patch 423 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] support piecewise cuda graph for kimi-linear model」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: 1. refactor `kimi-linear` to do fused `qkv_proj` and unify `ralix_linear_attention` interface for mixed_qkv input. 2. support piecewise cuda graph for `kimi-linear` model, which...。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +61/-42 (103 lines); hunks: -16,12 +16,13; -194,48 +195,46 @@ def __init__(; symbols: __init__, forward_qkvbfg, forward_qkvbfg_fused，涉及 `__init__, forward_qkvbfg, forward_qkvbfg_fused`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +61/-42 (103 lines); hunks: -16,12 +16,13; -194,48 +195,46 @@ def __init__(; symbols: __init__, forward_qkvbfg, forward_qkvbfg_fused
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +61/-42
- 验证与风险: diff 自带测试面 `test/registered/models/test_kimi_linear_models_pcg.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18689 - Add DP ViT support for Kimi K2.5

- 链接: https://github.com/sgl-project/sglang/pull/18689
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `5a7ae059e37f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-4，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DP ViT support for Kimi K2.5」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: Add DP ViT support for Kimi K2.5 launch: eval: Results (OCRBench):。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunks: -35,6 +35,8; -475,9 +477,10 @@ class MoonViT3dPretrainedModel(nn.Module):; symbols: MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector，涉及 `MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunks: -35,6 +35,8; -475,9 +477,10 @@ class MoonViT3dPretrainedModel(nn.Module):; symbols: MoonViT3dPretrainedModel, __init__, K2VLMultiModalProjector
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +20/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19120 - fix KimiK2Detector regex patterns with re.DOTALL

- 链接: https://github.com/sgl-project/sglang/pull/19120
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`；关联提交 `677b66af805d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-3，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix KimiK2Detector regex patterns with re.DOTALL」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: Enhance the tool call detection for kimi_k2 tool call parser to parse the multi-line tool arguments that contains `\n`. Contribute to #18086。
- 实现要点: `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunks: -40,11 +40,13 @@ def __init__(self):; -87,7 +89,7 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: __init__, detect_and_parse，涉及 `__init__, detect_and_parse`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunks: -40,11 +40,13 @@ def __init__(self):; -87,7 +89,7 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> St...; symbols: __init__, detect_and_parse
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/kimik2_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18434 - [Fix] Kimi K2.5 support pp

- 链接: https://github.com/sgl-project/sglang/pull/18434
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `4a3a787f1e1f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-13，可读 patch 62 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Kimi K2.5 support pp」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: fix 18433。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunks: -30,7 +30,7; -722,6 +722,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunks: -30,7 +30,7; -722,6 +722,7 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19181 - [Kernel Slimming] Migrate marlin moe kernel to JIT

- 链接: https://github.com/sgl-project/sglang/pull/19181
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+3780/-4，可读 patch 3825 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel Slimming] Migrate marlin moe kernel to JIT」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`；PR 正文摘要: See https://github.com/sgl-project/sglang/issues/17865 New files: - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` — JIT-compiled CUDA kernel ported from `...。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4 (10 lines); hunks: -10,6 +10,8; -142,7 +144,7 @@ def fused_marlin_moe(; symbols: get_scalar_type, fused_marlin_moe，涉及 `get_scalar_type, fused_marlin_moe`；`python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunks: -0,0 +1,1896；`python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunks: -0,0 +1,1089；`python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm，涉及 `stack_and_dev, _get_scalar_type, _setup_moe_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4 (10 lines); hunks: -10,6 +10,8; -142,7 +144,7 @@ def fused_marlin_moe(; symbols: get_scalar_type, fused_marlin_moe
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunks: -0,0 +1,1896
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunks: -0,0 +1,1089
  - `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm
  - `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: stack_and_dev, _make_inputs, _run_jit, _run_aot
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +6/-4; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0; `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0; `python/sglang/jit_kernel/moe_wna16_marlin.py` added +172/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel.h` added +37/-0
  - tests: `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19331 - [NPU] support Kimi-K2.5 on NPU

- 链接: https://github.com/sgl-project/sglang/pull/19331
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `86eb80007e78`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+23/-3，可读 patch 80 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] support Kimi-K2.5 on NPU」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: support Kimi-K2.5 on Ascend gsm8k test: image example:。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunks: -9,6 +9,7; -37,13 +38,15; symbols: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights，涉及 `apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunks: -9,6 +9,7; -37,13 +38,15; symbols: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +14/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19228 - [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning

- 链接: https://github.com/sgl-project/sglang/pull/19228
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+486/-23，可读 patch 892 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`；PR 正文摘要: Kimi K2.5 fused_moe_triton use default config so the performance is poor. 1. optimize Kimi K2.5 fused_moe_triton performance by tuning 2. fix fused_moe_triton/tuning_fused_moe_t...。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunks: -32,6 +32,10; -132,6 +136,7 @@ def benchmark_config(; symbols: benchmark_config, get_kernel_wrapper，涉及 `benchmark_config, get_kernel_wrapper`；`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunks: -28,6 +28,10; -44,6 +48,7 @@ def benchmark_config(; symbols: benchmark_config, run，涉及 `benchmark_config, run`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunks: -32,6 +32,10; -132,6 +136,7 @@ def benchmark_config(; symbols: benchmark_config, get_kernel_wrapper
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunks: -28,6 +28,10; -44,6 +48,7 @@ def benchmark_config(; symbols: benchmark_config, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5 (28 lines); hunks: -38,6 +38,10 @@ def get_model_config(; -46,11 +50,19 @@ def get_model_config(; symbols: get_model_config, get_config_filename
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6; `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19703 - [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT

- 链接: https://github.com/sgl-project/sglang/pull/19703
- 状态/时间: open / 2026-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+576/-1，可读 patch 588 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`；PR 正文摘要: Migrate `kimi_k2_moe_fused_gate` kernel to JIT compilation (#17865). Under `python/sglang/jit_kernel/`: - `csrc/moe/kimi_k2_moe_fused_gate.cuh` — CUDA kernel (ported from `sgl-k...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7；`python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunks: -0,0 +1,317；`python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: check_correctness, benchmark, fn，涉及 `check_correctness, benchmark, fn`；`python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts，涉及 `_reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -84,7 +84,7
  - `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: check_correctness, benchmark, fn
  - `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts
  - `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: _jit_kimi_k2_moe_fused_gate_module, _kimi_k2_moe_fused_gate_op, kimi_k2_moe_fused_gate
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +1/-1; `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0; `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0; `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0
  - tests: `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19689 - feat: support Kimi K2.5 for Eagle3

- 链接: https://github.com/sgl-project/sglang/pull/19689
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `85f7a0aa3077`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-0，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: support Kimi K2.5 for Eagle3」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: feat: Integrate Kimi K2.5 with Eagle3 and benchmark performance on H200 This commit introduces initial support for the Kimi K2.5 model on the Eagle3 platform. The primary goal i...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,5 +786,34 @@ def get_model_config_for_expert_location(cls, config: KimiK...; symbols: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head，涉及 `get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,5 +786,34 @@ def get_model_config_for_expert_location(cls, config: KimiK...; symbols: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19959 - Fix Kimi K2.5 PP layer range exposure for PD disaggregation

- 链接: https://github.com/sgl-project/sglang/pull/19959
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `069d4c577b39`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Kimi K2.5 PP layer range exposure for PD disaggregation」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: - Kimi K2.5 shows repeated outputs in **PP>1 + PD**. Root cause: the wrapper does not expose `start_layer/end_layer`, so each PP rank initializes KV cache with `start_layer=0`....。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunks: -719,6 +719,14 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, start_layer, end_layer, forward，涉及 `pad_input_ids, start_layer, end_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunks: -719,6 +719,14 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: Mu...; symbols: pad_input_ids, start_layer, end_layer, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19802 - [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2

- 链接: https://github.com/sgl-project/sglang/pull/19802
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_kimi_k25.py`；关联提交 `011806c41999`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+72/-53，可读 patch 127 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`；PR 正文摘要: - Add unified nightly test for Kimi K2.5 with two variants using `run_combined_tests` framework: - **TP8**: Base Kimi K2.5 with tool/reasoning parsers (from #19218) - **TP8+MTP*...。
- 实现要点: `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: TestKimiK25, for, test_kimi_k25，涉及 `TestKimiK25, for, test_kimi_k25`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: TestKimiK25, for, test_kimi_k25
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_kimi_k2.py`, `test/registered/8-gpu-models/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20747 - fix piecewise cuda graph support for Kimi-K2.5 model

- 链接: https://github.com/sgl-project/sglang/pull/20747
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `24a27d532084`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix piecewise cuda graph support for Kimi-K2.5 model」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: - Add `self.model` alias pointing to `self.language_model.model` in `KimiK25ForConditionalGeneration`, following the same pattern as InternVL. This allows the piecewise CUDA gra...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunks: -716,6 +716,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunks: -716,6 +716,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -716,6 +716,8 @@ def __init__(
+        self.model = self.language_model.model
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19552 - [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection

- 链接: https://github.com/sgl-project/sglang/pull/19552
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`, `test/registered/function_call/test_kimik2_detector.py`；关联提交 `c562e0d13ba9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+700/-19，可读 patch 799 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] Enhance Kimi-K2/K2.5 function call and reasoning detection」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`；PR 正文摘要: Related: #18086 fix 18086 Kimi-K2.5 models exhibit a special behavior where they output tool call markers ( ) directly inside blocks without first closing with . Since Kimi-K2 r...。
- 实现要点: `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunks: -0,0 +1,667; symbols: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp，涉及 `_make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic`；`python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunks: -15,10 +15,25; -38,22 +53,24 @@ def __init__(self):; symbols: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call，涉及 `_strip_special_tokens, KimiK2Detector, __init__`。
- 代码 diff 细节:
  - `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunks: -0,0 +1,667; symbols: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunks: -15,10 +15,25; -38,22 +53,24 @@ def __init__(self):; symbols: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/function_call/test_kimik2_detector.py` added +667/-0
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_kimik2_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20396 - perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path

- 链接: https://github.com/sgl-project/sglang/pull/20396
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `db995fba4790`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-10，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_linear.py`；PR 正文摘要: `einops.rearrange` adds Python-level overhead (pattern parsing, backend dispatch, shape validation) on every call. In the Kimi-Linear-48B model's KimiDeltaAttention hot path, th...。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -4,7 +4,6; -399,9 +398,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -4,7 +4,6; -399,9 +398,11 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/linear/kda_backend.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21004 - [Fix] Add EPLB rebalance support for Kimi K2.5

- 链接: https://github.com/sgl-project/sglang/pull/21004
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `01ccdb91b162`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Add EPLB rebalance support for Kimi K2.5」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；PR 正文摘要: Add routed_experts_weights_of_layer property to KimiK25ForConditionalGeneration to enable EPLB (Expert Parallel Load Balancing) rebalance support for Kimi K2.5 models. EPLB Reba...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunks: -767,6 +767,10 @@ def start_layer(self) -> int:; symbols: start_layer, end_layer, routed_experts_weights_of_layer, forward，涉及 `start_layer, end_layer, routed_experts_weights_of_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunks: -767,6 +767,10 @@ def start_layer(self) -> int:; symbols: start_layer, end_layer, routed_experts_weights_of_layer, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -767,6 +767,10 @@ def start_layer(self) -> int:
+    @property
+    def routed_experts_weights_of_layer(self):
+        return self.language_model._routed_experts_weights_of_layer.value
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21391 - Fix Kimi K2.5 dp attention+ spec decoding launch crash

- 链接: https://github.com/sgl-project/sglang/pull/21391
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_kimi_k25.py`；关联提交 `8c3ccef2d94e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+23/-2，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Kimi K2.5 dp attention+ spec decoding launch crash」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`；PR 正文摘要: Closes #21336 This issue is if there are mm input, draft embedding cannot handle it (will have out of boundary issue) Follows the same pattern in https://github.com/sgl-project/...。
- 实现要点: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25，涉及 `test_kimi_k25`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21741 - [1/N] feat: support compressed-tensors w4afp8 MoE

- 链接: https://github.com/sgl-project/sglang/pull/21741
- 状态/时间: open / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1664/-40，可读 patch 1845 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] feat: support compressed-tensors w4afp8 MoE」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；PR 正文摘要: Add support for loading compressed-tensors W4AFP8 MoE checkpoints in SGLang's `compressed_tensors` path. This enables INT4 group-quantized MoE weights with dynamic FP8 activatio...。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability，涉及 `_unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__`；`python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0 (62 lines); hunks: -429,6 +429,68 @@ def silu_and_mul_masked_post_quant_fwd(; symbols: silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe, silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe，涉及 `silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8 (35 lines); hunks: -43,6 +43,7; -304,15 +305,16 @@ def _quantization_scheme_map_from_config(; symbols: _quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8, _is_static_tensor_w8a8，涉及 `_quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8`；`python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6 (25 lines); hunks: -13,11 +13,11; -29,6 +29,7; symbols: cutlass_w4a8_moe，涉及 `cutlass_w4a8_moe`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0 (62 lines); hunks: -429,6 +429,68 @@ def silu_and_mul_masked_post_quant_fwd(; symbols: silu_and_mul_masked_post_quant_fwd, silu_mul_dynamic_scale_triton_kernel_for_cutlass_moe, silu_mul_dynamic_tensorwise_quant_for_cutlass_moe, silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8 (35 lines); hunks: -43,6 +43,7; -304,15 +305,16 @@ def _quantization_scheme_map_from_config(; symbols: _quantization_scheme_map_from_config, _is_dynamic_token_w4a8, _is_w4afp8, _is_static_tensor_w8a8
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6 (25 lines); hunks: -13,11 +13,11; -29,6 +29,7; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -41,4 +42,5
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +62/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +27/-8; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +19/-6; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py` modified +2/-0; `python/sglang/srt/layers/quantization/compressed_tensors/utils.py` modified +1/-0
  - other: `benchmark/kernels/quantization/bench_w4a8_moe_decode.py` added +887/-0
  - tests: `python/sglang/test/test_cutlass_w4a8_moe.py` modified +66/-23
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_per_tensor_absmax_fp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21898 - [CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8

- 链接: https://github.com/sgl-project/sglang/pull/21898
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_kimi_k25.py`；关联提交 `648632b6c41f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-23，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`；PR 正文摘要: - Remove Kimi K2.5 MTP variants (TP8+MTP and TP8+DP8+MTP) that crash in nightly tests - Keep TP8 base variant and add TP8+DP8 variant (without EAGLE3 speculative decoding) The T...。
- 实现要点: `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23 (27 lines); hunks: -10,19 +10,13; -31,13 +25,6 @@ def test_kimi_k25(self):; symbols: TestKimiK25, for, test_kimi_k25，涉及 `TestKimiK25, for, test_kimi_k25`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23 (27 lines); hunks: -10,19 +10,13; -31,13 +25,6 @@ def test_kimi_k25(self):; symbols: TestKimiK25, for, test_kimi_k25
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` modified +4/-23
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21213 - [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…

- 链接: https://github.com/sgl-project/sglang/pull/21213
- 状态/时间: merged / 2026-04-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`；关联提交 `dd49127fe612`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+81/-83，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`；PR 正文摘要: 1. Support AITER MLA for num_heads < 16 (e.g., TP=8 with Kimi K2.5 giving 8 heads/rank). Uses head-repeat to expand to 16 heads before calling the AITER MLA decode kernel, then...。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models，涉及 `ModelConfig, get_kimi_k25_mxfp4_models`；`test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass，涉及 `TestKimiK25MXFP4, setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12; `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22208 - [AMD] Optimize fused MoE kernel config for small-M decode on gfx950

- 链接: https://github.com/sgl-project/sglang/pull/22208
- 状态/时间: open / 2026-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-6，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize fused MoE kernel config for small-M decode on gfx950」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；PR 正文摘要: - Adds an AMD-optimized Triton fused MoE kernel config for very small M (batch=1-2 decode) on gfx950 (MI300X/MI355X). - When `_is_hip and M 2, NVIDIA, Marlin). During decode, ea...。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunks: -191,12 +191,26 @@ def get_default_config(; symbols: get_default_config，涉及 `get_default_config`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunks: -191,12 +191,26 @@ def get_default_config(; symbols: get_default_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22188 - [AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)

- 链接: https://github.com/sgl-project/sglang/pull/22188
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k25_mxfp4.py`；关联提交 `e14876742a08`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/test_kimi_k25_mxfp4.py`；PR 正文摘要: Pin amd/Kimi-K2.5-MXFP4 model revision in test_kimi_k25_mxfp4.py to fix CI weight loading failure. The HuggingFace model was updated (commit 94d8c1bd) to quantize additional lay...。
- 实现要点: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0 (3 lines); hunks: -27,6 +27,7; -36,6 +37,8 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0 (3 lines); hunks: -27,6 +27,7; -36,6 +37,8 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_kimi_k25_mxfp4.py
@@ -27,6 +27,7 @@
+KIMI_K25_MXFP4_REVISION = "b071bc6f8eb042e093e14f3b8bdbad71c18e09d3"
@@ -36,6 +37,8 @@ def setUpClass(cls):
+            "--revision",
+            KIMI_K25_MXFP4_REVISION,
```

- 已读文件:
  - tests: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_kimi_k25_mxfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22269 - [EPD][VLM] Support Kimi K25 EPD

- 链接: https://github.com/sgl-project/sglang/pull/22269
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `42ffb168b311`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+166/-42，可读 patch 348 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[EPD][VLM] Support Kimi K25 EPD」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；PR 正文摘要: Encoder–Prefill–Decode (EPD) disaggregation lets vision encoding run on dedicated encoder instances while language prefill/decode runs elsewhere. **Kimi K2.5** (`KimiK25ForCondi...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +48/-35 (83 lines); hunks: -708,33 +708,32 @@ def __init__(; -761,15 +760,22 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: M...; symbols: __init__, get_image_feature, pad_input_ids, start_layer，涉及 `__init__, get_image_feature, pad_input_ids`；`python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0 (65 lines); hunks: -4,6 +4,7; -55,6 +56,70 @@ async def process_mm_data_async(; symbols: process_mm_data_async, _num_image_tokens_from_grid, get_mm_data, _process_and_collect_mm_items，涉及 `process_mm_data_async, _num_image_tokens_from_grid, get_mm_data`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +48/-35 (83 lines); hunks: -708,33 +708,32 @@ def __init__(; -761,15 +760,22 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: M...; symbols: __init__, get_image_feature, pad_input_ids, start_layer
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0 (65 lines); hunks: -4,6 +4,7; -55,6 +56,70 @@ async def process_mm_data_async(; symbols: process_mm_data_async, _num_image_tokens_from_grid, get_mm_data, _process_and_collect_mm_items
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +48/-35; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +65/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22488 - Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation

- 链接: https://github.com/sgl-project/sglang/pull/22488
- 状态/时间: open / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+794/-53，可读 patch 890 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`；PR 正文摘要: GLM-5 uses a MoE architecture with 256 experts and `num_expert_group=1`, which shares the same ungrouped gating pattern as Kimi K2 (384 experts). However, the previous PyTorch i...。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -65,7 +65,6; -120,9 +119,11 @@ def fused_topk_deepseek(; symbols: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped，涉及 `fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess`；`python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344；`python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts，涉及 `_reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped`；`python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped，涉及 `_jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -65,7 +65,6; -120,9 +119,11 @@ def fused_topk_deepseek(; symbols: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped
  - `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344
  - `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts
  - `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +94/-53; `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0; `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0
  - tests: `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22381 - [Lora] Lora kimi support

- 链接: https://github.com/sgl-project/sglang/pull/22381
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`；关联提交 `6d79c6099545`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+188/-12，可读 patch 248 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Lora] Lora kimi support」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`；PR 正文未提供可用摘要。
- 实现要点: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff`。
- 代码 diff 细节:
  - `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0
- 验证与风险: diff 自带测试面 `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22496 - [Feature] kimi k25 w4a16 support deepep low latency

- 链接: https://github.com/sgl-project/sglang/pull/22496
- 状态/时间: open / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+4882/-25，可读 patch 5138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] kimi k25 w4a16 support deepep low latency」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunks: -39,15 +39,222; -355,6 +562,461 @@ def create_moe_runner(; symbols: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd，涉及 `_get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3 (59 lines); hunks: -56,7 +56,7; -386,6 +386,7 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core, combine_a，涉及 `dispatch_a, _dispatch_core, combine_a`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0 (44 lines); hunks: -10,6 +10,7; -37,6 +38,7; symbols: __init__, run_moe_core, get_moe_impl_class，涉及 `__init__, run_moe_core, get_moe_impl_class`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0 (14 lines); hunks: -1041,3 +1041,17 @@ def apply_without_routing_weights(; symbols: apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll，涉及 `apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunks: -39,15 +39,222; -355,6 +562,461 @@ def create_moe_runner(; symbols: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3 (59 lines); hunks: -56,7 +56,7; -386,6 +386,7 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core, combine_a
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0 (44 lines); hunks: -10,6 +10,7; -37,6 +38,7; symbols: __init__, run_moe_core, get_moe_impl_class
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0 (14 lines); hunks: -1041,3 +1041,17 @@ def apply_without_routing_weights(; symbols: apply_without_routing_weights, apply_deepep_normal, apply_deepep_ll
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0 (1948 lines); hunks: -0,0 +1,1948
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +56/-3; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +44/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +14/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0; `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +1264/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/elementwise/mask_silu_and_mul.cuh`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel_direct.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22368 - [VLM] GPU Image Preprocessing for Kimi-K2.5

- 链接: https://github.com/sgl-project/sglang/pull/22368
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `16f306fd85b6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+344/-48，可读 patch 438 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] GPU Image Preprocessing for Kimi-K2.5」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；PR 正文摘要: H200x8 std is 0.747 from https://www.kimi.com/blog/kimi-vendor-verifier SGLANG_USE_CUDA_IPC_TRANSPORT=1 python -m sglang.launch_server --model-path moonshotai/Kimi-K2.5 --tp 8 -...。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41 (370 lines); hunks: -1,7 +1,12; -16,11 +21,317; symbols: navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw, _process_single_image，涉及 `navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41 (370 lines); hunks: -1,7 +1,12; -16,11 +21,317; symbols: navit_resize_config, _get_image_dimensions, _pil_to_cuda_chw, _process_single_image
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +329/-41
- 验证与风险: runtime 路径改动集中在 `python/sglang/benchmark/datasets/image.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22806 - feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading

- 链接: https://github.com/sgl-project/sglang/pull/22806
- 状态/时间: open / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+548/-9，可读 patch 619 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；PR 正文摘要: The existing `W4AFp8Config` is designed for DeepSeek-style W4AFP8 checkpoints with hardcoded defaults (e.g., `moe_activation_scheme="static"`, empty `ignored_layers`). Kimi K2.5...。
- 实现要点: `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunks: -33,7 +33,11; -75,7 +79,7 @@ def get_config_filenames(cls) -> List[str]:; symbols: W4AFp8Config, for, __init__, get_config_filenames，涉及 `W4AFp8Config, for, __init__`；`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunks: -123,13 +123,24 @@ def do_load_weights(; symbols: do_load_weights，涉及 `do_load_weights`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunks: -1124,17 +1124,28 @@ def make_expert_params_mapping_fused_mxfp4(; symbols: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args，涉及 `make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args`；`python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ def override_quantization_method(self, *args, **kwargs):; -71,6 +71,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method，涉及 `override_quantization_method`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunks: -33,7 +33,11; -75,7 +79,7 @@ def get_config_filenames(cls) -> List[str]:; symbols: W4AFp8Config, for, __init__, get_config_filenames
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunks: -123,13 +123,24 @@ def do_load_weights(; symbols: do_load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunks: -1124,17 +1124,28 @@ def make_expert_params_mapping_fused_mxfp4(; symbols: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ def override_quantization_method(self, *args, **kwargs):; -71,6 +71,7 @@ def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method
  - `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0 (363 lines); hunks: -0,0 +1,363; symbols: _make_kimi_quant_config, TestKimiW4AFp8ConfigFromConfig, method, test_basic_parsing
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1
  - tests: `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_kimi_w4afp8_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22858 - [VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5

- 链接: https://github.com/sgl-project/sglang/pull/22858
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `8686f42acb3e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-64，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；PR 正文摘要: …DA context on device 0 in every TP rank Fixes redundant CUDA context allocation on device 0 across all TP ranks when running Kimi-K2.5 with the GPU image preprocessing fast pat...。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +6/-63 (69 lines); hunks: -42,7 +42,6; -622,59 +621,6 @@ def mm_projection_auto(; symbols: mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration, get_image_feature，涉及 `mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration`；`python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -285,10 +285,14 @@ def _gpu_call(self, text, images):; symbols: _gpu_call, _cpu_call，涉及 `_gpu_call, _cpu_call`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +6/-63 (69 lines); hunks: -42,7 +42,6; -622,59 +621,6 @@ def mm_projection_auto(; symbols: mm_projection_auto, vision_tower_forward_auto, KimiK25ForConditionalGeneration, get_image_feature
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -285,10 +285,14 @@ def _gpu_call(self, text, images):; symbols: _gpu_call, _cpu_call
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +6/-63; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22490 - [EPD][VLM] Support Kimi VL EPD

- 链接: https://github.com/sgl-project/sglang/pull/22490
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_vl.py`；关联提交 `e7ad7c587a35`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+268/-102，可读 patch 520 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[EPD][VLM] Support Kimi VL EPD」；模型线: Kimi K2/K2.5/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py`；PR 正文摘要: Kimi VL (kimi-vl-a3b) shares the same MoonViT vision tower as Kimi K2.5 but was not yet supported under the EPD (Encode-Prefill-Decode) disaggregation pipeline. This PR extends...。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: KimiGridMMDataMixin, to, _num_image_tokens_from_grid, _build_kimi_mm_data_from_grids，涉及 `KimiGridMMDataMixin, to, _num_image_tokens_from_grid`；`python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63 (70 lines); hunks: -9,8 +9,6; -20,6 +18,7; symbols: _get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async, _num_image_tokens_from_grid，涉及 `_get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async`；`python/sglang/srt/models/kimi_vl.py` modified +23/-8 (31 lines); hunks: -128,13 +128,16 @@ def __init__(; -215,6 +218,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, get_image_feature, load_weights，涉及 `__init__, get_image_feature, load_weights`；`python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1 (12 lines); hunks: -9,10 +9,11; -48,3 +49,12 @@ async def process_mm_data_async(; symbols: KimiVLImageProcessor, process_mm_data_async, get_mm_data，涉及 `KimiVLImageProcessor, process_mm_data_async, get_mm_data`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: KimiGridMMDataMixin, to, _num_image_tokens_from_grid, _build_kimi_mm_data_from_grids
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63 (70 lines); hunks: -9,8 +9,6; -20,6 +18,7; symbols: _get_gpu_norm_tensors, KimiK2_5VLImageProcessor, process_mm_data_async, _num_image_tokens_from_grid
  - `python/sglang/srt/models/kimi_vl.py` modified +23/-8 (31 lines); hunks: -128,13 +128,16 @@ def __init__(; -215,6 +218,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, get_image_feature, load_weights
  - `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1 (12 lines); hunks: -9,10 +9,11; -48,3 +49,12 @@ async def process_mm_data_async(; symbols: KimiVLImageProcessor, process_mm_data_async, get_mm_data
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_common.py` added +113/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +7/-63; `python/sglang/srt/models/kimi_vl.py` modified +23/-8; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +11/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/models/kimi_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22964 - [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output

- 链接: https://github.com/sgl-project/sglang/pull/22964
- 状态/时间: open / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；PR 正文摘要: When I run Kimi-K2.5 and curl with image, I get error `AttributeError: 'MultimodalDataItem' object has no attribute 'image_grid_thw'`. In this PR#22858, the return key in the `_...。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunks: -312,7 +312,12 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors，涉及 `_cpu_call, _get_gpu_norm_tensors`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunks: -312,7 +312,12 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13789 - [DeepEP Support] Support kimi-k2-thinking deepep

- 链接: https://github.com/sgl-project/sglang/pull/13789
- 状态/时间: closed / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+674/-0，可读 patch 753 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepEP Support] Support kimi-k2-thinking deepep」；模型线: Kimi K2/K2.5/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；PR 正文摘要: It will crash with IMA when launch server now, need more debug or advice.。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunks: -231,3 +231,211 @@ def fused_marlin_moe_fake(; symbols: fused_marlin_moe_fake, batched_fused_marlin_moe，涉及 `fused_marlin_moe_fake, batched_fused_marlin_moe`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunks: -652,3 +652,153 @@ def apply(; symbols: apply, apply_deepep_normal, apply_deepep_ll，涉及 `apply, apply_deepep_normal, apply_deepep_ll`；`python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunks: -80,3 +80,91 @@ def moe_align_block_size(; symbols: moe_align_block_size, batched_moe_align_block_size，涉及 `moe_align_block_size, batched_moe_align_block_size`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunks: -198,6 +198,8 @@ def run_moe_core(; -208,6 +210,8 @@ def run_moe_core(; symbols: run_moe_core, combine, _is_marlin_moe, forward_marlin_moe，涉及 `run_moe_core, combine, _is_marlin_moe`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunks: -231,3 +231,211 @@ def fused_marlin_moe_fake(; symbols: fused_marlin_moe_fake, batched_fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunks: -652,3 +652,153 @@ def apply(; symbols: apply, apply_deepep_normal, apply_deepep_ll
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunks: -80,3 +80,91 @@ def moe_align_block_size(; symbols: moe_align_block_size, batched_moe_align_block_size
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunks: -198,6 +198,8 @@ def run_moe_core(; -208,6 +210,8 @@ def run_moe_core(; symbols: run_moe_core, combine, _is_marlin_moe, forward_marlin_moe
  - `python/sglang/srt/layers/quantization/marlin_utils.py` modified +9/-0 (9 lines); hunks: -257,6 +257,15 @@ def check_moe_marlin_supports_layer(layer: FusedMoE, group_...; symbols: check_moe_marlin_supports_layer, marlin_moe_intermediate_size, marlin_make_workspace
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0; `python/sglang/srt/layers/quantization/marlin_utils.py` modified +9/-0
  - other: `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +140/-0; `sgl-kernel/python/sgl_kernel/moe.py` modified +29/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23186 - [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4

- 链接: https://github.com/sgl-project/sglang/pull/23186
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4」；模型线: Kimi K2/K2.5/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`；PR 正文摘要: On ROCm targets with a BF16 DeepSeek model, the MLA QK layernorm path falls through to the PyTorch sequential fallback instead of using aiter's fused kernel. The BF16 fused bran...。
- 实现要点: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunks: -60,6 +60,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; -160,6 +163,15 @@ def forward_absorb_prepare(; symbols: bmm_fp8, forward_absorb_prepare，涉及 `bmm_fp8, forward_absorb_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunks: -60,6 +60,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; -160,6 +163,15 @@ def forward_absorb_prepare(; symbols: bmm_fp8, forward_absorb_prepare
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23394 - [docs] sync kimi-k2.6 from sgl-cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23394
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`；关联提交 `d20ae9ceaa14`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-2，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] sync kimi-k2.6 from sgl-cookbook」；模型线: Kimi K2/K2.5/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`；PR 正文未提供可用摘要。
- 实现要点: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2 (36 lines); hunks: -693,10 +693,42 @@ python3 eval.py ocrbench \。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2 (36 lines); hunks: -693,10 +693,42 @@ python3 eval.py ocrbench \
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +34/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
