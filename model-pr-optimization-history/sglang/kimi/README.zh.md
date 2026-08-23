# sglang Kimi K2/K2.5/K3/Linear/VL 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` | 无直接 PR 号提交 |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` | 无直接 PR 号提交 |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx` | 无直接 PR 号提交 |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-K2.mdx` | 无直接 PR 号提交 |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` | [#33997](https://github.com/sgl-project/sglang/pull/33997), [#34886](https://github.com/sgl-project/sglang/pull/34886), [#35168](https://github.com/sgl-project/sglang/pull/35168) |
| `docs/cookbook/autoregressive/Moonshotai/Kimi-Linear.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/kimi_k2_6.mdx` | 无直接 PR 号提交 |
| `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/kimi_k2_6.mdx` | 无直接 PR 号提交 |
| `docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/kimi-k2-deployment.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/kimi-k25-deployment.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/kimi-k26-deployment.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/autoregressive/kimi-linear-deployment.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` | [#33612](https://github.com/sgl-project/sglang/pull/33612), [#33997](https://github.com/sgl-project/sglang/pull/33997), [#34444](https://github.com/sgl-project/sglang/pull/34444), [#35168](https://github.com/sgl-project/sglang/pull/35168) |
| `python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33997](https://github.com/sgl-project/sglang/pull/33997) |
| `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ag.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/comm/ptx_sys.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/comm/sp_collective.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh` | [#32541](https://github.com/sgl-project/sglang/pull/32541), [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33764](https://github.com/sgl-project/sglang/pull/33764) |
| `python/sglang/kernels/jit/csrc/trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/kimi_k3/__init__.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33465](https://github.com/sgl-project/sglang/pull/33465) |
| `python/sglang/kernels/ops/kimi_k3/activation.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33764](https://github.com/sgl-project/sglang/pull/33764) |
| `python/sglang/kernels/ops/kimi_k3/all_reduce.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/attn_res.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py` | [#33599](https://github.com/sgl-project/sglang/pull/33599) |
| `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/gemm_ag.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/gemm_ar.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890), [#33465](https://github.com/sgl-project/sglang/pull/33465) |
| `python/sglang/kernels/ops/kimi_k3/moe.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/kimi_k3/sp_collective.py` | [#32890](https://github.com/sgl-project/sglang/pull/32890) |
| `python/sglang/kernels/ops/moe/trtllm_lora_temp/kimi_k2_moe_fused_gate.py` | 无直接 PR 号提交 |
| `python/sglang/srt/arg_groups/kimi_k3_hook.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541) |
| `python/sglang/srt/configs/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789) |
| `python/sglang/srt/configs/kimi_k3.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541) |
| `python/sglang/srt/configs/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469), [#32541](https://github.com/sgl-project/sglang/pull/32541) |
| `python/sglang/srt/configs/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/configs/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383) |
| `python/sglang/srt/function_call/kimik2_detector.py` | [#7940](https://github.com/sgl-project/sglang/pull/7940), [#8043](https://github.com/sgl-project/sglang/pull/8043), [#8968](https://github.com/sgl-project/sglang/pull/8968), [#10972](https://github.com/sgl-project/sglang/pull/10972), [#19120](https://github.com/sgl-project/sglang/pull/19120), [#19552](https://github.com/sgl-project/sglang/pull/19552), [#23950](https://github.com/sgl-project/sglang/pull/23950), [#25071](https://github.com/sgl-project/sglang/pull/25071), [#33025](https://github.com/sgl-project/sglang/pull/33025) |
| `python/sglang/srt/function_call/kimik3_detector.py` | [#33025](https://github.com/sgl-project/sglang/pull/33025), [#34881](https://github.com/sgl-project/sglang/pull/34881) |
| `python/sglang/srt/function_call/kimik3_format.py` | [#33025](https://github.com/sgl-project/sglang/pull/33025) |
| `python/sglang/srt/function_call/kimik3_structural_tag.py` | [#33025](https://github.com/sgl-project/sglang/pull/33025), [#34778](https://github.com/sgl-project/sglang/pull/34778) |
| `python/sglang/srt/models/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#18370](https://github.com/sgl-project/sglang/pull/18370), [#18434](https://github.com/sgl-project/sglang/pull/18434), [#18440](https://github.com/sgl-project/sglang/pull/18440), [#18689](https://github.com/sgl-project/sglang/pull/18689), [#19331](https://github.com/sgl-project/sglang/pull/19331), [#19689](https://github.com/sgl-project/sglang/pull/19689), [#19959](https://github.com/sgl-project/sglang/pull/19959), [#20747](https://github.com/sgl-project/sglang/pull/20747), [#21004](https://github.com/sgl-project/sglang/pull/21004), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22858](https://github.com/sgl-project/sglang/pull/22858), ... (20 total) |
| `python/sglang/srt/models/kimi_k25_eagle3.py` | [#24826](https://github.com/sgl-project/sglang/pull/24826), [#25033](https://github.com/sgl-project/sglang/pull/25033), [#26506](https://github.com/sgl-project/sglang/pull/26506), [#27647](https://github.com/sgl-project/sglang/pull/27647), [#29223](https://github.com/sgl-project/sglang/pull/29223) |
| `python/sglang/srt/models/kimi_k3.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541), [#33465](https://github.com/sgl-project/sglang/pull/33465), [#33623](https://github.com/sgl-project/sglang/pull/33623), [#33764](https://github.com/sgl-project/sglang/pull/33764), [#33921](https://github.com/sgl-project/sglang/pull/33921), [#34404](https://github.com/sgl-project/sglang/pull/34404), [#34642](https://github.com/sgl-project/sglang/pull/34642), [#34766](https://github.com/sgl-project/sglang/pull/34766), [#34883](https://github.com/sgl-project/sglang/pull/34883), [#35077](https://github.com/sgl-project/sglang/pull/35077) |
| `python/sglang/srt/models/kimi_k3_vl.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541) |
| `python/sglang/srt/models/kimi_linear.py` | [#12469](https://github.com/sgl-project/sglang/pull/12469), [#12660](https://github.com/sgl-project/sglang/pull/12660), [#14337](https://github.com/sgl-project/sglang/pull/14337), [#17160](https://github.com/sgl-project/sglang/pull/17160), [#17506](https://github.com/sgl-project/sglang/pull/17506), [#17731](https://github.com/sgl-project/sglang/pull/17731), [#18849](https://github.com/sgl-project/sglang/pull/18849), [#20396](https://github.com/sgl-project/sglang/pull/20396), [#32262](https://github.com/sgl-project/sglang/pull/32262), [#32541](https://github.com/sgl-project/sglang/pull/32541), [#32612](https://github.com/sgl-project/sglang/pull/32612), [#34546](https://github.com/sgl-project/sglang/pull/34546) |
| `python/sglang/srt/models/kimi_vl.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383), [#22490](https://github.com/sgl-project/sglang/pull/22490), [#30869](https://github.com/sgl-project/sglang/pull/30869), [#32118](https://github.com/sgl-project/sglang/pull/32118) |
| `python/sglang/srt/models/kimi_vl_moonvit.py` | [#5383](https://github.com/sgl-project/sglang/pull/5383), [#30869](https://github.com/sgl-project/sglang/pull/30869), [#32118](https://github.com/sgl-project/sglang/pull/32118), [#32541](https://github.com/sgl-project/sglang/pull/32541), [#33349](https://github.com/sgl-project/sglang/pull/33349) |
| `python/sglang/srt/multimodal/kimi_k3_image_processing.py` | [#33921](https://github.com/sgl-project/sglang/pull/33921), [#34404](https://github.com/sgl-project/sglang/pull/34404), [#34766](https://github.com/sgl-project/sglang/pull/34766) |
| `python/sglang/srt/multimodal/kimi_k3_vit_cuda_graph_runner.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541) |
| `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py` | [#34404](https://github.com/sgl-project/sglang/pull/34404) |
| `python/sglang/srt/multimodal/processors/kimi_common.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490), [#33349](https://github.com/sgl-project/sglang/pull/33349) |
| `python/sglang/srt/multimodal/processors/kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#22269](https://github.com/sgl-project/sglang/pull/22269), [#22368](https://github.com/sgl-project/sglang/pull/22368), [#22490](https://github.com/sgl-project/sglang/pull/22490), [#22858](https://github.com/sgl-project/sglang/pull/22858), [#23501](https://github.com/sgl-project/sglang/pull/23501), [#28647](https://github.com/sgl-project/sglang/pull/28647), [#31227](https://github.com/sgl-project/sglang/pull/31227), [#32541](https://github.com/sgl-project/sglang/pull/32541), [#33349](https://github.com/sgl-project/sglang/pull/33349) |
| `python/sglang/srt/multimodal/processors/kimi_k3.py` | [#32541](https://github.com/sgl-project/sglang/pull/32541), [#33921](https://github.com/sgl-project/sglang/pull/33921), [#34163](https://github.com/sgl-project/sglang/pull/34163), [#34404](https://github.com/sgl-project/sglang/pull/34404), [#34766](https://github.com/sgl-project/sglang/pull/34766) |
| `python/sglang/srt/multimodal/processors/kimi_vl.py` | [#22490](https://github.com/sgl-project/sglang/pull/22490), [#33349](https://github.com/sgl-project/sglang/pull/33349) |
| `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml` | [#29855](https://github.com/sgl-project/sglang/pull/29855) |
| `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml` | [#29855](https://github.com/sgl-project/sglang/pull/29855) |
| `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml` | [#33333](https://github.com/sgl-project/sglang/pull/33333) |
| `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml` | [#33333](https://github.com/sgl-project/sglang/pull/33333) |
| `test/manual/models/test_kimi_k2_models.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_kimi_k25.py` | [#19802](https://github.com/sgl-project/sglang/pull/19802), [#21391](https://github.com/sgl-project/sglang/pull/21391), [#21898](https://github.com/sgl-project/sglang/pull/21898) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` | [#23848](https://github.com/sgl-project/sglang/pull/23848) |
| `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` | [#18269](https://github.com/sgl-project/sglang/pull/18269) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` | [#21213](https://github.com/sgl-project/sglang/pull/21213) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` | [#23848](https://github.com/sgl-project/sglang/pull/23848) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py` | [#31105](https://github.com/sgl-project/sglang/pull/31105) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py` | [#32568](https://github.com/sgl-project/sglang/pull/32568) |
| `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` | [#23848](https://github.com/sgl-project/sglang/pull/23848) |
| ... | 36 more files omitted from table; all were used for git tracing. |

## PR 覆盖总览

- git 追溯 PR 数: 98
- 原文档显式引用补充 PR 数: 70
- 当前文档总 PR 数: 168
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-18 | [#5440](https://github.com/sgl-project/sglang/pull/5440) | merged | Sgl kernel fused_moe_gate support n_shared_experts | `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py` |
| 2025-04-30 | [#5383](https://github.com/sgl-project/sglang/pull/5383) | merged | [Feature] add support kimi vl model | `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py` |
| 2025-07-11 | [#7940](https://github.com/sgl-project/sglang/pull/7940) | merged | Support Kimi K2 | `python/sglang/srt/function_call/kimik2_detector.py` |
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
| 2026-02-11 | [#18269](https://github.com/sgl-project/sglang/pull/18269) | merged | [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test | `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `python/sglang/srt/models/deepseek_janus_pro.py` |
| 2026-02-17 | [#18849](https://github.com/sgl-project/sglang/pull/18849) | merged | [PCG] support piecewise cuda graph for kimi-linear model | `python/sglang/srt/models/kimi_linear.py` |
| 2026-02-18 | [#18689](https://github.com/sgl-project/sglang/pull/18689) | merged | Add DP ViT support for Kimi K2.5 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-21 | [#19120](https://github.com/sgl-project/sglang/pull/19120) | merged | fix KimiK2Detector regex patterns with re.DOTALL | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-02-25 | [#18434](https://github.com/sgl-project/sglang/pull/18434) | merged | [Fix] Kimi K2.5 support pp | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-26 | [#19181](https://github.com/sgl-project/sglang/pull/19181) | merged | [Kernel Slimming] Migrate marlin moe kernel to JIT | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` |
| 2026-02-26 | [#19331](https://github.com/sgl-project/sglang/pull/19331) | merged | [NPU] support Kimi-K2.5 on NPU | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-26 | [#19228](https://github.com/sgl-project/sglang/pull/19228) | merged | [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` |
| 2026-03-03 | [#19689](https://github.com/sgl-project/sglang/pull/19689) | merged | feat: support Kimi K2.5 for Eagle3 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-07 | [#19959](https://github.com/sgl-project/sglang/pull/19959) | merged | Fix Kimi K2.5 PP layer range exposure for PD disaggregation | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-07 | [#19802](https://github.com/sgl-project/sglang/pull/19802) | merged | [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2 | `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-03-17 | [#20747](https://github.com/sgl-project/sglang/pull/20747) | merged | fix piecewise cuda graph support for Kimi-K2.5 model | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-19 | [#19552](https://github.com/sgl-project/sglang/pull/19552) | merged | [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-03-20 | [#20396](https://github.com/sgl-project/sglang/pull/20396) | merged | perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path | `python/sglang/srt/models/kimi_linear.py` |
| 2026-03-26 | [#21004](https://github.com/sgl-project/sglang/pull/21004) | merged | [Fix] Add EPLB rebalance support for Kimi K2.5 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-26 | [#21391](https://github.com/sgl-project/sglang/pull/21391) | merged | Fix Kimi K2.5 dp attention+ spec decoding launch crash | `test/registered/8-gpu-models/test_kimi_k25.py`, `python/sglang/srt/models/llama_eagle3.py` |
| 2026-04-02 | [#21898](https://github.com/sgl-project/sglang/pull/21898) | merged | [CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8 | `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-04-05 | [#21213](https://github.com/sgl-project/sglang/pull/21213) | merged | [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5… | `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-04-07 | [#22188](https://github.com/sgl-project/sglang/pull/22188) | merged | [AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1) | `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-04-10 | [#22269](https://github.com/sgl-project/sglang/pull/22269) | merged | [EPD][VLM] Support Kimi K25 EPD | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-10 | [#22381](https://github.com/sgl-project/sglang/pull/22381) | merged | [Lora] Lora kimi support | `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-04-11 | [#22368](https://github.com/sgl-project/sglang/pull/22368) | merged | [VLM] GPU Image Preprocessing for Kimi-K2.5 | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-14 | [#22806](https://github.com/sgl-project/sglang/pull/22806) | open | feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading | `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-04-15 | [#22858](https://github.com/sgl-project/sglang/pull/22858) | merged | [VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5 | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-16 | [#22490](https://github.com/sgl-project/sglang/pull/22490) | merged | [EPD][VLM] Support Kimi VL EPD | `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py` |
| 2026-04-16 | [#13789](https://github.com/sgl-project/sglang/pull/13789) | closed | [DeepEP Support] Support kimi-k2-thinking deepep | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2026-04-21 | [#23186](https://github.com/sgl-project/sglang/pull/23186) | merged | [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-04-21 | [#23381](https://github.com/sgl-project/sglang/pull/23381) | open | [AMD] Add MI355X Kimi-K2.6 tuning artifacts | `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py` |
| 2026-04-21 | [#23394](https://github.com/sgl-project/sglang/pull/23394) | merged | [docs] sync kimi-k2.6 from sgl-cookbook | `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` |
| 2026-04-27 | [#23408](https://github.com/sgl-project/sglang/pull/23408) | merged | [AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping | `python/sglang/srt/models/kimi_k25.py` |
| 2026-04-27 | [#23501](https://github.com/sgl-project/sglang/pull/23501) | merged | [VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-30 | [#22964](https://github.com/sgl-project/sglang/pull/22964) | closed | [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-05-05 | [#23848](https://github.com/sgl-project/sglang/pull/23848) | merged | [AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x | `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` |
| 2026-05-05 | [#24441](https://github.com/sgl-project/sglang/pull/24441) | merged | [Docs] Add B200, GB200, GB300 NVIDIA hardware platform support for Kimi-K2.6 | `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` |
| 2026-05-07 | [#23950](https://github.com/sgl-project/sglang/pull/23950) | merged | fix(function_call): handle Kimi-K2.5 bare numeric tool call IDs | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-05-10 | [#24826](https://github.com/sgl-project/sglang/pull/24826) | merged | [spec decoding] support kimi-k2.5-eagle3-mla | `python/sglang/srt/models/kimi_k25_eagle3.py` |
| 2026-05-12 | [#25033](https://github.com/sgl-project/sglang/pull/25033) | merged | Fix kimi k2.5 mla eagle + dp attention | `python/sglang/srt/models/kimi_k25_eagle3.py` |
| 2026-05-15 | [#25265](https://github.com/sgl-project/sglang/pull/25265) | merged | [perf] fix kimi tokenizer to improve ttft | `python/sglang/srt/managers/tokenizer_manager.py` |
| 2026-05-15 | [#23563](https://github.com/sgl-project/sglang/pull/23563) | closed | [Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend | `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` |
| 2026-05-18 | [#25390](https://github.com/sgl-project/sglang/pull/25390) | merged | [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model. | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-05-19 | [#25269](https://github.com/sgl-project/sglang/pull/25269) | merged | [NPU][Docs] Add Kimi-K2.5-W4A8 instance doc on NPU | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` |
| 2026-05-19 | [#25740](https://github.com/sgl-project/sglang/pull/25740) | merged | [AMD] Bump amd/Kimi-K2.5-MXFP4 revision to align with shared-experts fusion | `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-05-20 | [#25831](https://github.com/sgl-project/sglang/pull/25831) | merged | [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-05-25 | [#26149](https://github.com/sgl-project/sglang/pull/26149) | merged | [VLM] feat: accept grid_thws from preprocessed metadata for kimi | `python/sglang/srt/models/kimi_k25.py` |
| 2026-05-27 | [#26511](https://github.com/sgl-project/sglang/pull/26511) | merged | Update kimi k25 launch command in cookbook | `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` |
| 2026-05-28 | [#24649](https://github.com/sgl-project/sglang/pull/24649) | merged | [Xeon] CPU CI enhancement for Intel Xeon platforms | `test/registered/unit/models/test_llava.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/function_call/test_kimik2_detector.py` |
| 2026-05-28 | [#26382](https://github.com/sgl-project/sglang/pull/26382) | merged | Enable Kimi-K2.5 piecewise CUDA graph | `python/sglang/srt/models/kimi_k25.py` |
| 2026-05-28 | [#26506](https://github.com/sgl-project/sglang/pull/26506) | merged | [spec decoding] support kimi-k2.6-eagle3.1-mla draft | `python/sglang/srt/models/kimi_k25_eagle3.py` |
| 2026-05-29 | [#26353](https://github.com/sgl-project/sglang/pull/26353) | merged | NPU Nightly Pipeline Skip Test Case Adaptation and Recovery Testing | `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py`, `test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py` |
| 2026-05-29 | [#26257](https://github.com/sgl-project/sglang/pull/26257) | merged | [XPU] Fix Device Assignment | `python/sglang/srt/models/minicpmv.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/minicpmo.py` |
| 2026-05-29 | [#25676](https://github.com/sgl-project/sglang/pull/25676) | merged | Upgrade xgrammar to 0.2.1 | `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py` |
| 2026-05-29 | [#26744](https://github.com/sgl-project/sglang/pull/26744) | merged | [RL] Forward Kimi K2.5 weight hooks to language model | `python/sglang/srt/models/kimi_k25.py` |
| 2026-06-01 | [#26555](https://github.com/sgl-project/sglang/pull/26555) | merged | [RL+VLM] Avoid retokenization drift for pre-tokenized (token-id) VLM requests | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `test/registered/vlm/test_token_id_retokenize_e2e.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-03 | [#24762](https://github.com/sgl-project/sglang/pull/24762) | merged | [AMD] fix(triton-mla): cap max_kv_splits at 256 on gfx942 (Kimi-K2.6 hang) | `test/registered/amd/test_kimi_k2_instruct.py`, `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/utils/common.py` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-04 | [#22488](https://github.com/sgl-project/sglang/pull/22488) | closed | Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` |
| 2026-06-09 | [#27647](https://github.com/sgl-project/sglang/pull/27647) | merged | [sgl] Fix kimi-k2.5 EAGLE3 MLA draft embeds for batched MM prefill | `python/sglang/srt/models/kimi_k25_eagle3.py` |
| 2026-06-10 | [#8007](https://github.com/sgl-project/sglang/pull/8007) | closed | [Kimi K2] num_experts extends to 384 | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-10 | [#27714](https://github.com/sgl-project/sglang/pull/27714) | merged | [Docs] Add Kimi-K2.6 NVFP4 and update Kimi-K2.5 cookbook guidance | `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` |
| 2026-06-12 | [#28064](https://github.com/sgl-project/sglang/pull/28064) | merged | [Docs] Add Kimi K2.7 Code cookbook | `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`, `docs_new/cookbook/intro copy.mdx` |
| 2026-06-16 | [#28467](https://github.com/sgl-project/sglang/pull/28467) | merged | [ci] add kimi nvfp4 nightly tests | `test/registered/quant/test_kimi_k25_nvfp4_eagle.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28201](https://github.com/sgl-project/sglang/pull/28201) | merged | [Docs] Add fp8 kv cache for tokenspeed mla docs | `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-19 | [#28536](https://github.com/sgl-project/sglang/pull/28536) | merged | ci: run GB300 nightly suite in the standard Nvidia nightly workflow | `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py` |
| 2026-06-22 | [#28647](https://github.com/sgl-project/sglang/pull/28647) | merged | Fix Kimi-VL GPU image preprocessing crash on non-RGB images | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-06-23 | [#22496](https://github.com/sgl-project/sglang/pull/22496) | closed | [Feature] kimi k25 w4a16 support deepep low latency | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-06-24 | [#27833](https://github.com/sgl-project/sglang/pull/27833) | merged | [AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5 | `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` |
| 2026-06-24 | [#25071](https://github.com/sgl-project/sglang/pull/25071) | merged | kimik2_detector fix the normal text detection before tool call. | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-06-24 | [#28623](https://github.com/sgl-project/sglang/pull/28623) | merged | [CI] reduce CPU CI scope with base-c suite | `test/registered/function_call/test_kimik2_detector.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/unit/entrypoints/openai/test_serving_embedding.py` |
| 2026-06-25 | [#28103](https://github.com/sgl-project/sglang/pull/28103) | merged | Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test | `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml` |
| 2026-06-28 | [#29223](https://github.com/sgl-project/sglang/pull/29223) | merged | (perf): Shard Kimi-K2.5 Eagle3 draft fc + symm-mem AG | `python/sglang/srt/models/kimi_k25_eagle3.py` |
| 2026-07-05 | [#29855](https://github.com/sgl-project/sglang/pull/29855) | merged | [AMD][DI][CI] 3/N Add Kimi K2.6 FP8 MI355X 1P1D nightly recipes | `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml` |
| 2026-07-08 | [#29218](https://github.com/sgl-project/sglang/pull/29218) | merged | [Spec] DFlash: support pure-MLA targets with an fp8 KV cache (Kimi-K2.x-NVFP4) | `python/sglang/srt/model_executor/model_runner.py`, `test/registered/quant/test_kimi_k26_nvfp4_dflash.py`, `python/sglang/srt/speculative/dflash_worker_v2.py` |
| 2026-07-12 | [#30878](https://github.com/sgl-project/sglang/pull/30878) | merged | perf: reuse MoonViT FA3 max-seqlen metadata | `python/sglang/srt/models/kimi_k25.py` |
| 2026-07-14 | [#30869](https://github.com/sgl-project/sglang/pull/30869) | merged | fix: fix Kimi-VL encoder parallelism | `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `test/registered/unit/models/test_kimi_vl_moonvit.py` |
| 2026-07-16 | [#31227](https://github.com/sgl-project/sglang/pull/31227) | merged | perf: shard Kimi DP image feature transport | `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-07-17 | [#21741](https://github.com/sgl-project/sglang/pull/21741) | closed | [1/N] feat: support compressed-tensors w4afp8 MoE | `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-07-17 | [#31514](https://github.com/sgl-project/sglang/pull/31514) | merged | [DCP] Enable decode context parallel for Kimi K2.5 NVFP4 | `python/sglang/srt/models/kimi_k25.py` |
| 2026-07-19 | [#31474](https://github.com/sgl-project/sglang/pull/31474) | merged | Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear | `test/registered/models_e2e/test_kimi_linear_models.py`, `python/sglang/srt/layers/attention/linear/kda_backend.py`, `python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py` |
| 2026-07-24 | [#32262](https://github.com/sgl-project/sglang/pull/32262) | merged | [Bugfix] Fix Kimi-Linear state transfer across heterogeneous TP | `python/sglang/srt/models/kimi_linear.py`, `test/registered/disaggregation/test_disaggregation_kimi_linear.py` |
| 2026-07-27 | [#32542](https://github.com/sgl-project/sglang/pull/32542) | merged | docs(cookbook): add the Kimi-K3 serving cookbook | `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` |
| 2026-07-27 | [#32547](https://github.com/sgl-project/sglang/pull/32547) | merged | docs: point Kimi-K3 references to public branch | `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` |
| 2026-07-29 | [#32612](https://github.com/sgl-project/sglang/pull/32612) | merged | Support DCP for Kimi Linear model | `python/sglang/srt/models/kimi_linear.py`, `test/registered/dcp/test_kimi_linear_dcp4.py` |
| 2026-07-29 | [#32118](https://github.com/sgl-project/sglang/pull/32118) | merged | Fix nightly CI: NVFP4 cuda-graph crash, NVILA batching, CuTe paged-KV zero-size, Kimi-VL OOM | `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `python/sglang/srt/models/kimi_vl.py` |
| 2026-07-31 | [#32972](https://github.com/sgl-project/sglang/pull/32972) | merged | [unified-memory] Let Kimi-Linear use the paged MLA attention backends | `test/registered/models_e2e/test_kimi_linear_unified_memory.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py` |
| 2026-07-31 | [#32837](https://github.com/sgl-project/sglang/pull/32837) | merged | feat: support Kimi Linear PD disaggregation with DCP | `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/nixl/conn.py` |
| 2026-08-01 | [#32828](https://github.com/sgl-project/sglang/pull/32828) | merged | [Kimi] Support DCP + DSpark (ported from kimi-k3 branch) | `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/model_executor/pool_configurator.py` |
| 2026-08-01 | [#32890](https://github.com/sgl-project/sglang/pull/32890) | merged | feat(kernels): port standalone Kimi K3 kernels | `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh` |
| 2026-08-01 | [#33025](https://github.com/sgl-project/sglang/pull/33025) | merged | [Kimi K3] Add reasoning, tool-call, and OpenAI serving support | `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_detector.py` |
| 2026-08-04 | [#33333](https://github.com/sgl-project/sglang/pull/33333) | merged | [AMD][DI][CI] 6/N Add Kimi-K2.6 MXFP4 wide-EP16 2P1D nightly recipes | `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml` |
| 2026-08-04 | [#33349](https://github.com/sgl-project/sglang/pull/33349) | merged | [Perf] Speed up the Kimi-K2.5 vision path and match PIL bicubic in the GPU resize | `python/sglang/srt/multimodal/processors/kimi_k25.py`, `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-08-04 | [#32541](https://github.com/sgl-project/sglang/pull/32541) | merged | [Kimi] Support kimi-k3 | `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/models/kimi_k3_vl.py`, `test/registered/unit/models/test_kimi_k3_vision.py` |
| 2026-08-04 | [#33609](https://github.com/sgl-project/sglang/pull/33609) | merged | [Tiny] Little enhancement of Kimi-K3 test | `test/registered/models_e2e/test_kimi_k3_b300.py` |
| 2026-08-04 | [#33612](https://github.com/sgl-project/sglang/pull/33612) | merged | Remove custom all-reduce disable from Kimi-K3 B300 recipe | `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` |
| 2026-08-05 | [#33599](https://github.com/sgl-project/sglang/pull/33599) | merged | [AMD] Fuse Kimi-K3 attn-residual aggregation | `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py` |
| 2026-08-08 | [#33764](https://github.com/sgl-project/sglang/pull/33764) | merged | Fix the router GEMM inaccuracy when using _front_w in Kimi-K3 | `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/kernels/ops/kimi_k3/activation.py` |
| 2026-08-08 | [#34089](https://github.com/sgl-project/sglang/pull/34089) | merged | [CI] Add Kimi-K3 low-latency performance check | `test/registered/models_e2e/test_kimi_k3_b300.py` |
| 2026-08-09 | [#34145](https://github.com/sgl-project/sglang/pull/34145) | merged | [CI] Gate Kimi-K3 acceptance length on the GSM8K average | `test/registered/models_e2e/test_kimi_k3_b300.py` |
| 2026-08-09 | [#33921](https://github.com/sgl-project/sglang/pull/33921) | merged | [Kimi K3] Preprocess CPU-transport images on the vision owner | `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py` |
| 2026-08-10 | [#19703](https://github.com/sgl-project/sglang/pull/19703) | closed | [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` |
| 2026-08-10 | [#34163](https://github.com/sgl-project/sglang/pull/34163) | merged | fix(vlm): preserve Kimi-K3 GPU JPEG accuracy | `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` |
| 2026-08-11 | [#22208](https://github.com/sgl-project/sglang/pull/22208) | closed | [AMD] Optimize fused MoE kernel config for small-M decode on gfx950 | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2026-08-11 | [#31105](https://github.com/sgl-project/sglang/pull/31105) | merged | [ROCm/gfx95] Fix fp8 per-channel attention for Kimi-K2.7-code-mxfp4 o… | `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py` |
| 2026-08-11 | [#34444](https://github.com/sgl-project/sglang/pull/34444) | merged | docs(cookbook): Kimi-K3 — drop --enable-symm-mem from the GB cells | `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` |
| 2026-08-12 | [#33997](https://github.com/sgl-project/sglang/pull/33997) | merged | Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds | `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` |
| 2026-08-12 | [#33465](https://github.com/sgl-project/sglang/pull/33465) | merged | [Kimi-K3][NPU] Support Kimi-K3 on NPU | `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/ops/kimi_k3/__init__.py`, `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py` |
| 2026-08-12 | [#33623](https://github.com/sgl-project/sglang/pull/33623) | merged | [Kimi K3] Fuse MLA gate projection into QKV-A GEMM | `python/sglang/srt/models/kimi_k3.py` |
| 2026-08-13 | [#34642](https://github.com/sgl-project/sglang/pull/34642) | merged | Revert "[Kimi K3] Fuse MLA gate projection into QKV-A GEMM" | `python/sglang/srt/models/kimi_k3.py` |
| 2026-08-13 | [#34766](https://github.com/sgl-project/sglang/pull/34766) | merged | [Fix] Carry the backend on Kimi-K3 deferred preprocessing configs | `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py` |
| 2026-08-13 | [#34779](https://github.com/sgl-project/sglang/pull/34779) | merged | [CI] Disable the prefill CUDA graph on the P worker of test_kimi_linear_pd_dcp4 | `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` |
| 2026-08-14 | [#34886](https://github.com/sgl-project/sglang/pull/34886) | merged | [Docs] Update Kimi-K3 installation options | `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` |
| 2026-08-14 | [#34778](https://github.com/sgl-project/sglang/pull/34778) | merged | [Fix] Work around xgrammar 0.2.1 negative integer minimum in Kimi-K3 structural tags | `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py` |
| 2026-08-15 | [#34883](https://github.com/sgl-project/sglang/pull/34883) | merged | [Kimi-K3] Use explicit SiTU activation for MegaMoE | `python/sglang/srt/models/kimi_k3.py`, `test/registered/models_e2e/test_kimi_k3_b300.py` |
| 2026-08-16 | [#34404](https://github.com/sgl-project/sglang/pull/34404) | merged | [VLM] Cache Kimi-K3 per-image processor artifacts | `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py` |
| 2026-08-17 | [#35168](https://github.com/sgl-project/sglang/pull/35168) | merged | docs: add NVFP4 quantization option to Kimi-K3 deploy panel | `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` |
| 2026-08-17 | [#32568](https://github.com/sgl-project/sglang/pull/32568) | merged | [AMD] Add Kimi-K3 8-GPU MI35x nightly accuracy CI | `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py` |
| 2026-08-18 | [#34985](https://github.com/sgl-project/sglang/pull/34985) | merged | [AMD] Add the Kimi-K3 MI35x perf benchmarks in nightly | `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py` |
| 2026-08-18 | [#34881](https://github.com/sgl-project/sglang/pull/34881) | merged | Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation | `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`, `python/sglang/srt/function_call/kimik3_detector.py` |
| 2026-08-19 | [#35077](https://github.com/sgl-project/sglang/pull/35077) | merged | [Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint | `python/sglang/srt/models/kimi_k3.py`, `test/registered/unit/models/test_kimi_k3_bfa_overlap.py` |
| 2026-08-20 | [#34546](https://github.com/sgl-project/sglang/pull/34546) | merged | [XPU] Fix/kimi linear xpu | `python/sglang/srt/models/kimi_linear.py` |

## 逐 PR diff 审计卡

### PR #5440 - Sgl kernel fused_moe_gate support n_shared_experts

- 链接: https://github.com/sgl-project/sglang/pull/5440
- 状态/时间: merged / 2025-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+140/-38，可读 patch 351 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Sgl kernel fused_moe_gate support n_shared_experts」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py`；技术摘要: 覆盖「Sgl kernel fused_moe_gate support n_shared_experts」；主要实现面是 `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/tests/test_moe_fused_gate.py`, `sgl-kernel/python/sgl_kernel/moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Feature] add support kimi vl model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py`；技术摘要: 覆盖「[Feature] add support kimi vl model」；主要实现面是 `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/managers/multimodal_processors/kimi_vl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Support Kimi K2」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「Support Kimi K2」；主要实现面是 `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #8021 - perf: add kimi k2 fused_moe tuning config for h30_3e

- 链接: https://github.com/sgl-project/sglang/pull/8021
- 状态/时间: merged / 2025-07-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf: add kimi k2 fused_moe tuning config for h30_3e」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「perf: add kimi k2 fused_moe tuning config for h30_3e」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「H20 tune config for Kimi」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「H20 tune config for Kimi」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat: add h200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「feat: add h200 tp 16 kimi k2 moe config」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat: add b200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「feat: add b200 tp 16 kimi k2 moe config」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat: add h200 tp 16 kimi k2 moe config」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「feat: add h200 tp 16 kimi k2 moe config」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat(function call): complete utility method for KimiK2Detector and enhance documentation」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「feat(function call): complete utility method for KimiK2Detector and enhance documentation」；主要实现面是 `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`；技术摘要: 覆盖「[Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384」；主要实现面是 `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix kimi k2 function call format」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「Fix kimi k2 function call format」；主要实现面是 `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[perf] add kimi-k2 b200 fused moe config」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「[perf] add kimi-k2 b200 fused moe config」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix kimi k2 function calling format」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`；技术摘要: 覆盖「Fix kimi k2 function calling format」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Replace the Kimi-K2 generated tool call idx with history tool call count」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`；技术摘要: 覆盖「Replace the Kimi-K2 generated tool call idx with history tool call count」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/srt/openai_server/basic/test_serving_chat.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「fix: KimiK2Detector Improve tool call ID parsing with regex」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「fix: KimiK2Detector Improve tool call ID parsing with regex」；主要实现面是 `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Support Kimi Linear」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py`；技术摘要: 覆盖「Support Kimi Linear」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/configs/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「overlap shared + routed expert computation in kimi linear」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「overlap shared + routed expert computation in kimi linear」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Opt kimi_k2_thinking biased topk module」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；技术摘要: 覆盖「Opt kimi_k2_thinking biased topk module」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[opt kimi k2 1 / n] Add kimi k2 moe fused gate」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`；技术摘要: 覆盖「[opt kimi k2 1 / n] Add kimi k2 moe fused gate」；主要实现面是 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；技术摘要: 覆盖「[opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；技术摘要: 覆盖「[opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel」；主要实现面是 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[kimi k2 thinking] Avoid useless torch.zeros_」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py`；技术摘要: 覆盖「[kimi k2 thinking] Avoid useless torch.zeros_」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/awq.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；技术摘要: 覆盖「[opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`；技术摘要: 覆盖「[Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「Use dual stream for DS MoE whenever cuda graph is used (instead of with token threshold)」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Ascend] support Kimi-K2-Thinking」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「[Ascend] support Kimi-K2-Thinking」；主要实现面是 `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Add Expert Parallelism (EP) support for kimi-k2-thinking」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；技术摘要: 覆盖「Add Expert Parallelism (EP) support for kimi-k2-thinking」；主要实现面是 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Support piecewise cuda graph for fused marlin moe」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py`；技术摘要: 覆盖「Support piecewise cuda graph for fused marlin moe」；主要实现面是 `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/moe/moe_runner/marlin.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix warp illegal instruction in kimi k2 thinking PCG」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；技术摘要: 覆盖「Fix warp illegal instruction in kimi k2 thinking PCG」；主要实现面是 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`；技术摘要: 覆盖「Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix kernel selection in biased_grouped_topk_gpu」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/topk.py`；技术摘要: 覆盖「Fix kernel selection in biased_grouped_topk_gpu」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[Kimi-Linear] Refactor kimi-linear gate calculation to avoid duplicated code」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[Kimi-Linear] Refactor Kimi-Linear to support RadixLinearAttention」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi-Linear] Remove duplicated code in kimi-linear」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[Kimi-Linear] Remove duplicated code in kimi-linear」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/amd/test_kimi_k2_instruct.py`；技术摘要: 覆盖「[AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases」；主要实现面是 `test/registered/amd/test_kimi_k2_instruct.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Support Kimi-K2.5 model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「Support Kimi-K2.5 model」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`；技术摘要: 覆盖「[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI」；主要实现面是 `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[BUGFIX] Fix dp size > 1 for qwen3 vl model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`；技术摘要: 覆盖「[BUGFIX] Fix dp size > 1 for qwen3 vl model」；主要实现面是 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix: Avoid Double Reduce in VLM DP Attention」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py`；技术摘要: 覆盖「Fix: Avoid Double Reduce in VLM DP Attention」；主要实现面是 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`, `test/registered/distributed/test_dp_attention_large.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`；技术摘要: 覆盖「[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k2_eval_amd.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「fix kimi k2.5's moe gemm config init」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/managers/scheduler.py`；技术摘要: 覆盖「fix kimi k2.5's moe gemm config init」；主要实现面是 `python/sglang/srt/managers/scheduler.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kimi-K2.5] Fix missing `quant_config` in `KimiK25`」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[Kimi-K2.5] Fix missing `quant_config` in `KimiK25`」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `python/sglang/srt/models/deepseek_janus_pro.py`；技术摘要: 覆盖「[AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `python/sglang/srt/models/deepseek_janus_pro.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy，涉及 `TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy`；`test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy，涉及 `TestKimiK25EvalAMD, setUpClass, tearDownClass`；`python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1955,7 +1955,7 @@ def __init__(; symbols: __init__, get_image_feature，涉及 `__init__, get_image_feature`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy
  - `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1955,7 +1955,7 @@ def __init__(; symbols: __init__, get_image_feature
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
diff -- python/sglang/srt/models/deepseek_janus_pro.py
@@ -1955,7 +1955,7 @@ def __init__(
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0
  - runtime: `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18849 - [PCG] support piecewise cuda graph for kimi-linear model

- 链接: https://github.com/sgl-project/sglang/pull/18849
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `bf5238835459`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+157/-71，可读 patch 423 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] support piecewise cuda graph for kimi-linear model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[PCG] support piecewise cuda graph for kimi-linear model」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Add DP ViT support for Kimi K2.5」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「Add DP ViT support for Kimi K2.5」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「fix KimiK2Detector regex patterns with re.DOTALL」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「fix KimiK2Detector regex patterns with re.DOTALL」；主要实现面是 `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Fix] Kimi K2.5 support pp」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[Fix] Kimi K2.5 support pp」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kernel Slimming] Migrate marlin moe kernel to JIT」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`；技术摘要: 覆盖「[Kernel Slimming] Migrate marlin moe kernel to JIT」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[NPU] support Kimi-K2.5 on NPU」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[NPU] support Kimi-K2.5 on NPU」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`；技术摘要: 覆盖「[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #19689 - feat: support Kimi K2.5 for Eagle3

- 链接: https://github.com/sgl-project/sglang/pull/19689
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `85f7a0aa3077`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-0，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: support Kimi K2.5 for Eagle3」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「feat: support Kimi K2.5 for Eagle3」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix Kimi K2.5 PP layer range exposure for PD disaggregation」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「Fix Kimi K2.5 PP layer range exposure for PD disaggregation」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`；技术摘要: 覆盖「[Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2」；主要实现面是 `test/registered/8-gpu-models/test_kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「fix piecewise cuda graph support for Kimi-K2.5 model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「fix piecewise cuda graph support for Kimi-K2.5 model」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[feat] Enhance Kimi-K2/K2.5 function call and reasoning detection」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「[feat] Enhance Kimi-K2/K2.5 function call and reasoning detection」；主要实现面是 `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「perf(kimi_linear): replace einops rearrange with native torch ops in Kimi-Linear KDA path」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Fix] Add EPLB rebalance support for Kimi K2.5」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[Fix] Add EPLB rebalance support for Kimi K2.5」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix Kimi K2.5 dp attention+ spec decoding launch crash」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`, `python/sglang/srt/models/llama_eagle3.py`；技术摘要: 覆盖「Fix Kimi K2.5 dp attention+ spec decoding launch crash」；主要实现面是 `test/registered/8-gpu-models/test_kimi_k25.py`, `python/sglang/srt/models/llama_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25，涉及 `test_kimi_k25`；`python/sglang/srt/models/llama_eagle3.py` modified +12/-1 (13 lines); hunks: -150,7 +150,18 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: -38,11 +38,15 @@ def test_kimi_k25(self):; -56,6 +60,12 @@ def test_kimi_k25(self):; symbols: test_kimi_k25
  - `python/sglang/srt/models/llama_eagle3.py` modified +12/-1 (13 lines); hunks: -150,7 +150,18 @@ def forward(; symbols: forward
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
diff -- python/sglang/srt/models/llama_eagle3.py
@@ -150,7 +150,18 @@ def forward(
-            embeds = self.embed_tokens(input_ids)
+            embeds = forward_batch.mm_input_embeds
+            if (
+                forward_batch.forward_mode.is_extend()
+                and forward_batch.contains_mm_inputs()
+                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1
  - runtime: `python/sglang/srt/models/llama_eagle3.py` modified +12/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21898 - [CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8

- 链接: https://github.com/sgl-project/sglang/pull/21898
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_kimi_k25.py`；关联提交 `648632b6c41f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-23，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_kimi_k25.py`；技术摘要: 覆盖「[CI] Remove crashing Kimi K2.5 EAGLE3/MTP variants, keep TP8 and TP8+DP8」；主要实现面是 `test/registered/8-gpu-models/test_kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`；技术摘要: 覆盖「[AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5…」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models，涉及 `ModelConfig, get_kimi_k25_mxfp4_models`；`test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass，涉及 `TestKimiK25MXFP4, setUpClass`；`python/sglang/srt/layers/attention/aiter_backend.py` modified +76/-62 (138 lines); hunks: -234,13 +234,25 @@ def __init__(; -254,7 +266,7 @@ def __init__(; symbols: __init__, make_mla_decode_meta_data_buffer, make_mla_meta_data，涉及 `__init__, make_mla_decode_meta_data_buffer, make_mla_meta_data`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12 (15 lines); hunks: -1,4 +1,4; -7,13 +7,6; symbols: ModelConfig, get_kimi_k25_mxfp4_models
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9 (11 lines); hunks: -1,14 +1,8; -41,10 +35,9 @@ class TestKimiK25MXFP4(CustomTestCase):; symbols: TestKimiK25MXFP4, setUpClass
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +76/-62 (138 lines); hunks: -234,13 +234,25 @@ def __init__(; -254,7 +266,7 @@ def __init__(; symbols: __init__, make_mla_decode_meta_data_buffer, make_mla_meta_data
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
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -234,13 +234,25 @@ def __init__(
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` modified +3/-12; `test/registered/amd/test_kimi_k25_mxfp4.py` modified +2/-9
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +76/-62
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22188 - [AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)

- 链接: https://github.com/sgl-project/sglang/pull/22188
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k25_mxfp4.py`；关联提交 `e14876742a08`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/test_kimi_k25_mxfp4.py`；技术摘要: 覆盖「[AMD] Fix test_kimi_k25_mxfp4.py : stage-c-test-large-8-gpu-amd-mi35x (linux-mi35x-gpu-8, 1)」；主要实现面是 `test/registered/amd/test_kimi_k25_mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[EPD][VLM] Support Kimi K25 EPD」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「[EPD][VLM] Support Kimi K25 EPD」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #22381 - [Lora] Lora kimi support

- 链接: https://github.com/sgl-project/sglang/pull/22381
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`；关联提交 `6d79c6099545`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+188/-12，可读 patch 248 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Lora] Lora kimi support」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；技术摘要: 覆盖「[Lora] Lora kimi support」；主要实现面是 `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy，涉及 `kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +13/-10 (23 lines); hunks: -448,25 +448,28 @@ def create_moe_runner(; symbols: create_moe_runner, apply_weights, get_triton_quant_info，涉及 `create_moe_runner, apply_weights, get_triton_quant_info`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +13/-0 (13 lines); hunks: -682,6 +682,16 @@ def get_moe_scheme(; -997,6 +1007,9 @@ def create_moe_runner(; symbols: get_moe_scheme, create_moe_runner, get_triton_quant_info, apply，涉及 `get_moe_scheme, create_moe_runner, get_triton_quant_info`；`python/sglang/srt/lora/layers.py` modified +8/-1 (9 lines); hunks: -809,10 +809,17 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: kl_v2, get_prompt_logprobs, TestLoRAKimiK25LogprobDiff, test_lora_kimi_k25_logprob_accuracy
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +13/-10 (23 lines); hunks: -448,25 +448,28 @@ def create_moe_runner(; symbols: create_moe_runner, apply_weights, get_triton_quant_info
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +13/-0 (13 lines); hunks: -682,6 +682,16 @@ def get_moe_scheme(; -997,6 +1007,9 @@ def create_moe_runner(; symbols: get_moe_scheme, create_moe_runner, get_triton_quant_info, apply
  - `python/sglang/srt/lora/layers.py` modified +8/-1 (9 lines); hunks: -809,10 +809,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/lora/lora_manager.py` modified +4/-1 (5 lines); hunks: -66,7 +66,10 @@ def __init__(; symbols: __init__
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
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py
@@ -448,25 +448,28 @@ def create_moe_runner(
-    def apply_weights(
-        self,
-        layer: torch.nn.Module,
-        dispatch_output: "StandardDispatchOutput",
-    ) -> "CombineInput":
+    def get_triton_quant_info(self, layer):
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
@@ -682,6 +682,16 @@ def get_moe_scheme(
```

- 已读文件:
  - tests: `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` added +150/-0
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +13/-10; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +13/-0; `python/sglang/srt/lora/layers.py` modified +8/-1; `python/sglang/srt/lora/lora_manager.py` modified +4/-1
- 验证与风险: diff 自带测试面 `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22368 - [VLM] GPU Image Preprocessing for Kimi-K2.5

- 链接: https://github.com/sgl-project/sglang/pull/22368
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `16f306fd85b6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+344/-48，可读 patch 438 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] GPU Image Preprocessing for Kimi-K2.5」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「[VLM] GPU Image Preprocessing for Kimi-K2.5」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；技术摘要: 覆盖「feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading」；主要实现面是 `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「[VLM] Enable per-image ViT cache and avoid TP CUDA context creation for Kimi-K2.5」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[EPD][VLM] Support Kimi VL EPD」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py`；技术摘要: 覆盖「[EPD][VLM] Support Kimi VL EPD」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_vl.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #13789 - [DeepEP Support] Support kimi-k2-thinking deepep

- 链接: https://github.com/sgl-project/sglang/pull/13789
- 状态/时间: closed / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+674/-0，可读 patch 753 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepEP Support] Support kimi-k2-thinking deepep」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；技术摘要: 覆盖「[DeepEP Support] Support kimi-k2-thinking deepep」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`；技术摘要: 覆盖「[AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4」；主要实现面是 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23381 - [AMD] Add MI355X Kimi-K2.6 tuning artifacts

- 链接: https://github.com/sgl-project/sglang/pull/23381
- 状态/时间: open / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+133/-5，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add MI355X Kimi-K2.6 tuning artifacts」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py`；技术摘要: 覆盖「[AMD] Add MI355X Kimi-K2.6 tuning artifacts」；主要实现面是 `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/environ.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0 (119 lines); hunks: -0,0 +1,119；`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -151,7 +151,8 @@ def do_load_weights(; symbols: do_load_weights，涉及 `do_load_weights`；`python/sglang/srt/environ.py` modified +5/-1 (6 lines); hunks: -206,6 +206,10 @@ class Envs:; -992,7 +996,7 @@ def assert_throws(message_matcher: str):; symbols: Envs, assert_throws，涉及 `Envs, assert_throws`；`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-3 (5 lines); hunks: -242,14 +242,13 @@ def run():; -419,7 +418,7 @@ def _distribute(method: str, inputs: List[Any]) -> List[Any]:; symbols: run, BenchmarkWorker, __init__, benchmark，涉及 `run, BenchmarkWorker, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0 (119 lines); hunks: -0,0 +1,119
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -151,7 +151,8 @@ def do_load_weights(; symbols: do_load_weights
  - `python/sglang/srt/environ.py` modified +5/-1 (6 lines); hunks: -206,6 +206,10 @@ class Envs:; -992,7 +996,7 @@ def assert_throws(message_matcher: str):; symbols: Envs, assert_throws
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-3 (5 lines); hunks: -242,14 +242,13 @@ def run():; -419,7 +418,7 @@ def _distribute(method: str, inputs: List[Any]) -> List[Any]:; symbols: run, BenchmarkWorker, __init__, benchmark
  - `docs_new/docs/references/environment_variables.mdx` modified +5/-0 (5 lines); hunks: -83,6 +83,11 @@ SGLang supports various environment variables that can be use...
- 关键代码摘录:

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
@@ -151,7 +151,8 @@ def do_load_weights(
-        with concurrent.futures.ThreadPoolExecutor() as executor:
+        max_workers = envs.SGLANG_DEEPSEEK_LOAD_MAX_WORKERS.get()
+        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
diff -- python/sglang/srt/environ.py
@@ -206,6 +206,10 @@ class Envs:
+    # None => fall back to ThreadPoolExecutor's default worker count.
+    # Lower this (e.g. to 4) for very large MoE checkpoints where the default
+    # creates too much aggregate host I/O pressure across ranks.
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json` added +119/-0; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1; `python/sglang/srt/environ.py` modified +5/-1
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-3
  - docs: `docs_new/docs/references/environment_variables.mdx` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23394 - [docs] sync kimi-k2.6 from sgl-cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23394
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-2，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] sync kimi-k2.6 from sgl-cookbook」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`；技术摘要: 覆盖「[docs] sync kimi-k2.6 from sgl-cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23408 - [AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping

- 链接: https://github.com/sgl-project/sglang/pull/23408
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `d49561b8ae9e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-2，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[AMD] Fix Kimi-K2.6 Quark MXFP4 loading prefix and packed module mapping」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -28,6 +28,7; -661,7 +662,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -28,6 +28,7; -661,7 +662,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -28,6 +28,7 @@
+from sglang.srt.layers.quantization.quark.quark import QuarkConfig
@@ -661,7 +662,7 @@ def __init__(
-                    if isinstance(quant_config, ModelSlimConfig)
+                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23501 - [VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw

- 链接: https://github.com/sgl-project/sglang/pull/23501
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `f34c20af86af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「[VLM] Fix Kimi-K2.5 CPU path: rename grid_thws -> image_grid_thw」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -312,7 +312,11 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors，涉及 `_cpu_call, _get_gpu_norm_tensors`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1 (6 lines); hunks: -312,7 +312,11 @@ def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22964 - [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output

- 链接: https://github.com/sgl-project/sglang/pull/22964
- 状态/时间: closed / 2026-04-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「[fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23848 - [AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x

- 链接: https://github.com/sgl-project/sglang/pull/23848
- 状态/时间: merged / 2026-05-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`；关联提交 `244531bc4f3b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+546/-28，可读 patch 710 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`；技术摘要: 覆盖「[AMD] Add Kimi-K2.6 in nightly tests for MI30x and MI35x」；主要实现面是 `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass, test_bench_kimi_k26，涉及 `generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass`；`test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass, test_bench_kimi_k26，涉及 `generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass`；`test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy，涉及 `TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy`；`test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: TestKimiK26EvalAMD, setUpClass, tearDownClass, test_kimi_k26_gsm8k_accuracy，涉及 `TestKimiK26EvalAMD, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: generate_simple_markdown_report, TestNightlyKimiK26PerformanceMI35x, setUpClass, test_bench_kimi_k26
  - `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: generate_simple_markdown_report, TestNightlyKimiK26Performance, setUpClass, test_bench_kimi_k26
  - `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestKimiK26EvalMI35x, setUpClass, test_kimi_k26_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0 (108 lines); hunks: -0,0 +1,108; symbols: TestKimiK26EvalAMD, setUpClass, tearDownClass, test_kimi_k26_gsm8k_accuracy
- 关键代码摘录:

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

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py` added +152/-0; `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py` added +148/-0; `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py` added +110/-0; `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py` added +108/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi30x/test_kimi_k26_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_kimi_k26_perf_amd.py`, `test/registered/amd/perf/mi35x/test_kimi_k26_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24441 - [Docs] Add B200, GB200, GB300 NVIDIA hardware platform support for Kimi-K2.6

- 链接: https://github.com/sgl-project/sglang/pull/24441
- 状态/时间: merged / 2026-05-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-1，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add B200, GB200, GB300 NVIDIA hardware platform support for Kimi-K2.6」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`；技术摘要: 覆盖「[Docs] Add B200, GB200, GB300 NVIDIA hardware platform support for Kimi-K2.6」；主要实现面是 `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +6/-0 (6 lines); hunks: -6,7 +6,10 @@ export const KimiK26Deployment = () => {; -41,7 +44,10 @@ export const KimiK26Deployment = () => {；`docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +1/-1 (2 lines); hunks: -86,7 +86,7 @@ import { KimiK26Deployment } from '/src/snippets/autoregressiv...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +6/-0 (6 lines); hunks: -6,7 +6,10 @@ export const KimiK26Deployment = () => {; -41,7 +44,10 @@ export const KimiK26Deployment = () => {
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +1/-1 (2 lines); hunks: -86,7 +86,7 @@ import { KimiK26Deployment } from '/src/snippets/autoregressiv...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx
@@ -6,7 +6,10 @@ export const KimiK26Deployment = () => {
+        { id: 'b200', label: 'B200', default: false },
+        { id: 'gb200', label: 'GB200', default: false },
+        { id: 'gb300', label: 'GB300', default: false },
@@ -41,7 +44,10 @@ export const KimiK26Deployment = () => {
+    b200: { tp: 8 },
+    gb200: { tp: 4 },
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx
@@ -86,7 +86,7 @@ import { KimiK26Deployment } from '/src/snippets/autoregressive/kimi-k26-deploym
-- **Memory**: Requires GPUs with ≥140GB each. Supported platforms: H200 (8×, TP=8), B300 (8×, TP=8), MI300X/MI325X (4×, TP=4), MI350X/MI355X (4×, TP=4). Use `--context-length 1280
+- **Memory**: Requires GPUs with ≥140GB each. Supported platforms: H200 (8×, TP=8), B200 (8×, TP=8), B300 (8×, TP=8), GB200 (4×, TP=4), GB300 (4×, TP=4), MI300X/MI325X (4×, TP=4),
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +6/-0; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23950 - fix(function_call): handle Kimi-K2.5 bare numeric tool call IDs

- 链接: https://github.com/sgl-project/sglang/pull/23950
- 状态/时间: merged / 2026-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`, `test/registered/function_call/test_kimik2_detector.py`；关联提交 `af2a2ac61839`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+244/-22，可读 patch 358 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(function_call): handle Kimi-K2.5 bare numeric tool call IDs」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「fix(function_call): handle Kimi-K2.5 bare numeric tool call IDs」；主要实现面是 `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_kimik2_detector.py` modified +143/-0 (143 lines); hunks: -663,5 +663,148 @@ def test_e2e_multiple_tool_calls_without_think_close(self):; symbols: test_e2e_multiple_tool_calls_without_think_close, TestKimiK2BareCounterParsing, setUp, test_standard_format_with_functions_prefix，涉及 `test_e2e_multiple_tool_calls_without_think_close, TestKimiK2BareCounterParsing, setUp`；`python/sglang/srt/function_call/kimik2_detector.py` modified +101/-22 (123 lines); hunks: -35,13 +35,18 @@ class KimiK2Detector(BaseFormatDetector):; -55,23 +60,96 @@ def __init__(self):; symbols: KimiK2Detector, __init__, _parse_tool_call_id, _infer_tool_name，涉及 `KimiK2Detector, __init__, _parse_tool_call_id`。
- 代码 diff 细节:
  - `test/registered/function_call/test_kimik2_detector.py` modified +143/-0 (143 lines); hunks: -663,5 +663,148 @@ def test_e2e_multiple_tool_calls_without_think_close(self):; symbols: test_e2e_multiple_tool_calls_without_think_close, TestKimiK2BareCounterParsing, setUp, test_standard_format_with_functions_prefix
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +101/-22 (123 lines); hunks: -35,13 +35,18 @@ class KimiK2Detector(BaseFormatDetector):; -55,23 +60,96 @@ def __init__(self):; symbols: KimiK2Detector, __init__, _parse_tool_call_id, _infer_tool_name
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -663,5 +663,148 @@ def test_e2e_multiple_tool_calls_without_think_close(self):
+# ============================================================
+# Part 3: Bare-counter tool call ID parsing
+# ============================================================
+class TestKimiK2BareCounterParsing(unittest.TestCase):
+    """Tests for bare numeric tool_call_id format (e.g., '3' instead of 'functions.ReadFile:0')."""
+    def setUp(self):
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -35,13 +35,18 @@ class KimiK2Detector(BaseFormatDetector):
-    Format Structure:
+    Format Structure (standard):
+    Format Structure (bare counter — model omits function name):
+    '''
+    <|tool_call_begin|>{counter}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
+    '''
```

- 已读文件:
  - tests: `test/registered/function_call/test_kimik2_detector.py` modified +143/-0
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +101/-22
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_kimik2_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24826 - [spec decoding] support kimi-k2.5-eagle3-mla

- 链接: https://github.com/sgl-project/sglang/pull/24826
- 状态/时间: merged / 2026-05-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25_eagle3.py`；关联提交 `a87fb399deaa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+465/-0，可读 patch 480 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[spec decoding] support kimi-k2.5-eagle3-mla」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25_eagle3.py`；技术摘要: 覆盖「[spec decoding] support kimi-k2.5-eagle3-mla」；主要实现面是 `python/sglang/srt/models/kimi_k25_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25_eagle3.py` added +458/-0 (458 lines); hunks: -0,0 +1,458; symbols: _get_eagle_aux_layer_count, Eagle3MLADecoderLayer, __init__, forward，涉及 `_get_eagle_aux_layer_count, Eagle3MLADecoderLayer, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25_eagle3.py` added +458/-0 (458 lines); hunks: -0,0 +1,458; symbols: _get_eagle_aux_layer_count, Eagle3MLADecoderLayer, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25_eagle3.py
@@ -0,0 +1,458 @@
+"""EAGLE3 draft model with MLA attention for Kimi-K2.5.
+The ``kimi-k2.5-eagle3-mla`` checkpoint pairs an EAGLE3 layout
+(concatenated [embed_norm, hidden_norm] pre-attention input, fc projection
+over the concatenated multi-layer aux hidden states, single decoder layer,
+dense MLP) with DeepSeek-V2 multi-latent attention. Sharing the MLA layout
+with the Kimi-K2.5 target keeps the draft KV cache small.
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25_eagle3.py` added +458/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/kimi_k25_eagle3.py`, `python/sglang/srt/utils/hf_transformers/common.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25033 - Fix kimi k2.5 mla eagle + dp attention

- 链接: https://github.com/sgl-project/sglang/pull/25033
- 状态/时间: merged / 2026-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25_eagle3.py`；关联提交 `cfc41d5b15fe`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+15/-1，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix kimi k2.5 mla eagle + dp attention」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25_eagle3.py`；技术摘要: 覆盖「Fix kimi k2.5 mla eagle + dp attention」；主要实现面是 `python/sglang/srt/models/kimi_k25_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-1 (16 lines); hunks: -223,7 +223,21 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-1 (16 lines); hunks: -223,7 +223,21 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25_eagle3.py
@@ -223,7 +223,21 @@ def forward(
-            embeds = self.embed_tokens(input_ids)
+            # MM positions in input_ids hold MM_PAD_SHIFT_VALUE+hash sentinels (far above
+            # vocab_size). Use target-produced mm_input_embeds for these positions and
+            # only call embed_tokens on the appended next-token to avoid embed OOB.
+            embeds = forward_batch.mm_input_embeds
+            if (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25_eagle3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25265 - [perf] fix kimi tokenizer to improve ttft

- 链接: https://github.com/sgl-project/sglang/pull/25265
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-3，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[perf] fix kimi tokenizer to improve ttft」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/managers/tokenizer_manager.py`；技术摘要: 覆盖「[perf] fix kimi tokenizer to improve ttft」；主要实现面是 `python/sglang/srt/managers/tokenizer_manager.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/managers/tokenizer_manager.py` modified +10/-3 (13 lines); hunks: -689,9 +689,16 @@ async def _tokenize_texts(; symbols: _tokenize_texts，涉及 `_tokenize_texts`。
- 代码 diff 细节:
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +10/-3 (13 lines); hunks: -689,9 +689,16 @@ async def _tokenize_texts(; symbols: _tokenize_texts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -689,9 +689,16 @@ async def _tokenize_texts(
-            encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
-            input_ids = encoded["input_ids"]
-            token_type_ids = encoded.get("token_type_ids") if is_cross_encoder else None
+            if not is_cross_encoder and (not getattr(self.tokenizer, "is_fast", False)):
+                input_ids = [self.tokenizer.encode(t) for t in tokenizer_input]
+                token_type_ids = None
```

- 已读文件:
  - runtime: `python/sglang/srt/managers/tokenizer_manager.py` modified +10/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/tokenizer_manager.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23563 - [Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend

- 链接: https://github.com/sgl-project/sglang/pull/23563
- 状态/时间: closed / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+60/-3，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`；技术摘要: 覆盖「[Cookbook] Add Kimi K2.6 speculative decoding + fix draft attention backend」；主要实现面是 `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2 (35 lines); hunks: -1,5 +1,6; -37,6 +38,15 @@ export const KimiK26Deployment = () => {；`docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0 (23 lines); hunks: -16,6 +16,7 @@ tag: NEW; -469,6 +470,28 @@ Let me search for this product and similar items for you.；`docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0 (3 lines); hunks: -453,6 +453,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \; -472,6 +473,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \；`docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1 (2 lines); hunks: -195,7 +195,7 @@ export const KimiK25Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2 (35 lines); hunks: -1,5 +1,6; -37,6 +38,15 @@ export const KimiK26Deployment = () => {
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0 (23 lines); hunks: -16,6 +16,7 @@ tag: NEW; -469,6 +470,28 @@ Let me search for this product and similar items for you.
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0 (3 lines); hunks: -453,6 +453,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \; -472,6 +473,7 @@ SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  - `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1 (2 lines); hunks: -195,7 +195,7 @@ export const KimiK25Deployment = () => {
- 关键代码摘录:

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

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +33/-2; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +23/-0; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +3/-0; `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25390 - [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.

- 链接: https://github.com/sgl-project/sglang/pull/25390
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-2，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`；技术摘要: 覆盖「[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`；`python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method，涉及 `get_name, apply_weight_name_mapper, get_quant_method`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2355,6 +2355,12 @@ def __init__(
+        # Quant configs like Quark may rely on the model to provide fused-module
+        # mappings so exclusion checks can unfuse derived names back to the
+        # checkpoint's source layer names.
+        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
+            quant_config.packed_modules_mapping = self.packed_modules_mapping
@@ -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -71,7 +71,13 @@ def get_name(self) -> str:
-        self.exclude_layers = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        expanded = []
+        for name in mapped:
+            expanded.append(name)
+            if name.startswith("language_model."):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25269 - [NPU][Docs] Add Kimi-K2.5-W4A8 instance doc on NPU

- 链接: https://github.com/sgl-project/sglang/pull/25269
- 状态/时间: merged / 2026-05-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+314/-0，可读 patch 315 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU][Docs] Add Kimi-K2.5-W4A8 instance doc on NPU」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx`；技术摘要: 覆盖「[NPU][Docs] Add Kimi-K2.5-W4A8 instance doc on NPU」；主要实现面是 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` added +314/-0 (314 lines); hunks: -0,0 +1,314。
- 代码 diff 细节:
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` added +314/-0 (314 lines); hunks: -0,0 +1,314
- 关键代码摘录:

```diff
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx
@@ -0,0 +1,314 @@
+---
+title: "Kimi K2.5 examples"
+metatags:
+  description: "Documentation for Kimi K2.5 examples"
+---
+## Introduction
```

- 已读文件:
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx` added +314/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_kimi_k2.5_examples.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #25740 - [AMD] Bump amd/Kimi-K2.5-MXFP4 revision to align with shared-experts fusion

- 链接: https://github.com/sgl-project/sglang/pull/25740
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k25_mxfp4.py`；关联提交 `7c3f614e2352`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-1，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Bump amd/Kimi-K2.5-MXFP4 revision to align with shared-experts fusion」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_kimi_k25_mxfp4.py`；技术摘要: 覆盖「[AMD] Bump amd/Kimi-K2.5-MXFP4 revision to align with shared-experts fusion」；主要实现面是 `test/registered/amd/test_kimi_k25_mxfp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +7/-1 (8 lines); hunks: -27,7 +27,13。
- 代码 diff 细节:
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +7/-1 (8 lines); hunks: -27,7 +27,13
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_kimi_k25_mxfp4.py
@@ -27,7 +27,13 @@
-KIMI_K25_MXFP4_REVISION = "b071bc6f8eb042e093e14f3b8bdbad71c18e09d3"
+# Bumped from b071bc6f -> 419004c8 (HF main HEAD as of 2026-05-18). The pinned
+# b071bc6f revision keeps shared_experts unquantized (bf16), which is
+# incompatible with the shared-experts fusion path enabled for Kimi-K2.5
+# (n_routed_experts=384) in #25390. Revisions from 94d8c1bd onward quantize
+# shared_experts to MXFP4 so the fusion can copy weights between routed and
```

- 已读文件:
  - tests: `test/registered/amd/test_kimi_k25_mxfp4.py` modified +7/-1
- 验证与风险: diff 自带测试面 `test/registered/amd/test_kimi_k25_mxfp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25831 - [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests

- 链接: https://github.com/sgl-project/sglang/pull/25831
- 状态/时间: merged / 2026-05-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 36 个文件，+572/-639，可读 patch 1504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`；技术摘要: 覆盖「[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests」；主要实现面是 `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- 链接: https://github.com/sgl-project/sglang/pull/24751
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+45/-44，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #26149 - [VLM] feat: accept grid_thws from preprocessed metadata for kimi

- 链接: https://github.com/sgl-project/sglang/pull/26149
- 状态/时间: merged / 2026-05-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `64e2b54a8f45`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] feat: accept grid_thws from preprocessed metadata for kimi」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[VLM] feat: accept grid_thws from preprocessed metadata for kimi」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +7/-3 (10 lines); hunks: -680,9 +680,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]...; symbols: get_image_feature，涉及 `get_image_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +7/-3 (10 lines); hunks: -680,9 +680,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]...; symbols: get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -680,9 +680,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
-        grid_thws = torch.concat([item.image_grid_thw for item in items], dim=0).to(
-            device
-        )
+        image_grid_thws = []
+        for item in items:
+            grid_thw = item.model_specific_data.get("image_grid_thw")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +7/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26511 - Update kimi k25 launch command in cookbook

- 链接: https://github.com/sgl-project/sglang/pull/26511
- 状态/时间: merged / 2026-05-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update kimi k25 launch command in cookbook」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`；技术摘要: 覆盖「Update kimi k25 launch command in cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +6/-1 (7 lines); hunks: -195,7 +195,12 @@ export const KimiK25Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +6/-1 (7 lines); hunks: -195,7 +195,12 @@ export const KimiK25Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx
@@ -195,7 +195,12 @@ export const KimiK25Deployment = () => {
-      cmd += ' \\\n  --speculative-algorithm EAGLE3 \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4 \\\n  --speculative-dra
+      cmd += ' \\\n  --speculative-algorithm EAGLE3 \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4 \\\n  --speculative-dra
+    }
+    // Blackwell (B300): tokenspeed MLA attention backend
+    if (hardware === 'b300') {
+      cmd += ' \\\n  --attention-backend tokenspeed_mla';
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +6/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24649 - [Xeon] CPU CI enhancement for Intel Xeon platforms

- 链接: https://github.com/sgl-project/sglang/pull/24649
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 88 个文件，+192/-31，可读 patch 969 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Xeon] CPU CI enhancement for Intel Xeon platforms」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/unit/models/test_llava.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/function_call/test_kimik2_detector.py`；技术摘要: 覆盖「[Xeon] CPU CI enhancement for Intel Xeon platforms」；主要实现面是 `test/registered/unit/models/test_llava.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/function_call/test_kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_llava.py` modified +6/-1 (7 lines); hunks: -2,11 +2,16; symbols: PixtralVisionConfig，涉及 `PixtralVisionConfig`；`test/registered/models/test_transformers_backend_eval.py` modified +2/-1 (3 lines); hunks: -3,11 +3,12; symbols: TestTransformersBackendEval，涉及 `TestTransformersBackendEval`；`test/registered/function_call/test_kimik2_detector.py` modified +1/-0 (1 lines); hunks: -12,6 +12,7; symbols: _make_tool，涉及 `_make_tool`；`test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-0 (1 lines); hunks: -57,6 +57,7 @@ def find_spec(self, fullname, path, target=None):; symbols: find_spec，涉及 `find_spec`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_llava.py` modified +6/-1 (7 lines); hunks: -2,11 +2,16; symbols: PixtralVisionConfig
  - `test/registered/models/test_transformers_backend_eval.py` modified +2/-1 (3 lines); hunks: -3,11 +3,12; symbols: TestTransformersBackendEval
  - `test/registered/function_call/test_kimik2_detector.py` modified +1/-0 (1 lines); hunks: -12,6 +12,7; symbols: _make_tool
  - `test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-0 (1 lines); hunks: -57,6 +57,7 @@ def find_spec(self, fullname, path, target=None):; symbols: find_spec
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +1/-0 (1 lines); hunks: -32,6 +32,7; symbols: TestPythonicDetector
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_llava.py
@@ -2,11 +2,16 @@
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+from sglang.test.ci.ci_register import (
+    register_amd_ci,
+    register_cpu_ci,
+    register_cuda_ci,
+)
diff -- test/registered/models/test_transformers_backend_eval.py
@@ -3,11 +3,12 @@
-from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
+register_cpu_ci(est_time=320, suite="base-b-test-cpu")
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -12,6 +12,7 @@
+register_cpu_ci(est_time=7, suite="base-b-test-cpu")
diff -- test/registered/unit/entrypoints/openai/test_serving_embedding.py
@@ -57,6 +57,7 @@ def find_spec(self, fullname, path, target=None):
```

- 已读文件:
  - tests: `test/registered/unit/models/test_llava.py` modified +6/-1; `test/registered/models/test_transformers_backend_eval.py` modified +2/-1; `test/registered/function_call/test_kimik2_detector.py` modified +1/-0; `test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-0; `test/registered/unit/function_call/test_function_call_parser.py` modified +1/-0; `test/registered/unit/function_call/test_json_schema_constraint.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/registered/bench_fn/test_benchmark_datasets_api.py`, `test/registered/debug_utils/comparator/aligner/entrypoint/test_executor.py`, `test/registered/debug_utils/comparator/aligner/entrypoint/test_planner.py`, `test/registered/debug_utils/comparator/aligner/reorderer/test_executor.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26382 - Enable Kimi-K2.5 piecewise CUDA graph

- 链接: https://github.com/sgl-project/sglang/pull/26382
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `e60f799b4019`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-0，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Kimi-K2.5 piecewise CUDA graph」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「Enable Kimi-K2.5 piecewise CUDA graph」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +13/-0 (13 lines); hunks: -674,6 +674,19 @@ def __init__(; symbols: __init__, model, satisfies, __setattr__，涉及 `__init__, model, satisfies`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +13/-0 (13 lines); hunks: -674,6 +674,19 @@ def __init__(; symbols: __init__, model, satisfies, __setattr__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -674,6 +674,19 @@ def __init__(
+    @property
+    def model(self):
+        # Alias .model to .language_model so this class satisfies the piecewise
+        # CUDA graph gate, which checks `hasattr(model, "model")`.
+        return self.language_model
+    def __setattr__(self, name, value):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +13/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26506 - [spec decoding] support kimi-k2.6-eagle3.1-mla draft

- 链接: https://github.com/sgl-project/sglang/pull/26506
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25_eagle3.py`；关联提交 `93445e6359f8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+38/-5，可读 patch 85 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[spec decoding] support kimi-k2.6-eagle3.1-mla draft」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k25_eagle3.py`；技术摘要: 覆盖「[spec decoding] support kimi-k2.6-eagle3.1-mla draft」；主要实现面是 `python/sglang/srt/models/kimi_k25_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +38/-5 (43 lines); hunks: -1,10 +1,18; -196,13 +204,28 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25_eagle3.py` modified +38/-5 (43 lines); hunks: -1,10 +1,18; -196,13 +204,28 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25_eagle3.py
@@ -1,10 +1,18 @@
-"""EAGLE3 draft model with MLA attention for Kimi-K2.5.
+"""EAGLE3 / EAGLE3.1 draft model with MLA attention for Kimi-K2.x.
-with the Kimi-K2.5 target keeps the draft KV cache small.
+with the Kimi-K2.x target keeps the draft KV cache small.
+The eagle3.1 variant (e.g. ``kimi-k2.6-eagle3.1-mla``) adds two optional
+config flags on top of the same layout:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +38/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25_eagle3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26353 - NPU Nightly Pipeline Skip Test Case Adaptation and Recovery Testing

- 链接: https://github.com/sgl-project/sglang/pull/26353
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+151/-118，可读 patch 487 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「NPU Nightly Pipeline Skip Test Case Adaptation and Recovery Testing」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py`, `test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py`；技术摘要: 覆盖「NPU Nightly Pipeline Skip Test Case Adaptation and Recovery Testing」；主要实现面是 `test/registered/ascend/interface/test_npu_openai_function_calling.py`, `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py`, `test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ascend/interface/test_npu_openai_function_calling.py` modified +22/-25 (47 lines); hunks: -18,7 +18,6; -429,8 +428,10 @@ def test_function_call_strict(self):; symbols: test_function_call_strict, test_function_call_required, test_function_call_specific，涉及 `test_function_call_strict, test_function_call_required, test_function_call_specific`；`test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestNoChunkedPrefill, setUpClass, tearDownClass, test_mmlu，涉及 `TestNoChunkedPrefill, setUpClass, tearDownClass`；`test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py` removed +0/-39 (39 lines); hunks: -1,39 +0,0; symbols: TestNoChunkedPrefill, test_no_chunked_prefill, test_no_chunked_prefill_without_radix_cache，涉及 `TestNoChunkedPrefill, test_no_chunked_prefill, test_no_chunked_prefill_without_radix_cache`；`test/registered/ascend/vlm_models/test_npu_kimi_vl_a3b_instruct.py` modified +14/-18 (32 lines); hunks: -1,32 +1,28; symbols: TestKimiVLA3BInstruct, test_vlm_mmmu_benchmark，涉及 `TestKimiVLA3BInstruct, test_vlm_mmmu_benchmark`。
- 代码 diff 细节:
  - `test/registered/ascend/interface/test_npu_openai_function_calling.py` modified +22/-25 (47 lines); hunks: -18,7 +18,6; -429,8 +428,10 @@ def test_function_call_strict(self):; symbols: test_function_call_strict, test_function_call_required, test_function_call_specific
  - `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: TestNoChunkedPrefill, setUpClass, tearDownClass, test_mmlu
  - `test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py` removed +0/-39 (39 lines); hunks: -1,39 +0,0; symbols: TestNoChunkedPrefill, test_no_chunked_prefill, test_no_chunked_prefill_without_radix_cache
  - `test/registered/ascend/vlm_models/test_npu_kimi_vl_a3b_instruct.py` modified +14/-18 (32 lines); hunks: -1,32 +1,28; symbols: TestKimiVLA3BInstruct, test_vlm_mmmu_benchmark
  - `test/registered/ascend/vlm_models/test_npu_llama_3_2_11b_vision_instruct.py` modified +11/-9 (20 lines); hunks: -1,20 +1,22; symbols: TestLlama3211BVisionInstruct
- 关键代码摘录:

```diff
diff -- test/registered/ascend/interface/test_npu_openai_function_calling.py
@@ -18,7 +18,6 @@
-    disabled="https://github.com/Ascend/sglang/issues/39",
@@ -429,8 +428,10 @@ def test_function_call_strict(self):
-        Test: Whether tool_choice: "required" works as expected
-        - When tool_choice == "required", the model should return one or more tool_calls.
+        Test: Whether tool_choice: "required" works as expected.
+        - When tool_choice == "required", the model MUST return one or more tool_calls.
diff -- test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py
@@ -0,0 +1,74 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
+from sglang.test.ci.ci_register import register_npu_ci
+from sglang.test.run_eval import run_eval
diff -- test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py
@@ -1,39 +0,0 @@
```

- 已读文件:
  - tests: `test/registered/ascend/interface/test_npu_openai_function_calling.py` modified +22/-25; `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py` added +74/-0; `test/registered/ascend/basic_function/parameter/test_npu_no_chunked_prefill.py` removed +0/-39; `test/registered/ascend/vlm_models/test_npu_kimi_vl_a3b_instruct.py` modified +14/-18; `test/registered/ascend/vlm_models/test_npu_llama_3_2_11b_vision_instruct.py` modified +11/-9; `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_next.py` modified +13/-3
- 验证与风险: diff 自带测试面 `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mla.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_ttft_mha.py`, `test/registered/ascend/basic_function/memory_and_scheduling/test_npu_no_chunked_prefill.py`, `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26257 - [XPU] Fix Device Assignment

- 链接: https://github.com/sgl-project/sglang/pull/26257
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+13/-12，可读 patch 123 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Fix Device Assignment」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/minicpmv.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/minicpmo.py`；技术摘要: 覆盖「[XPU] Fix Device Assignment」；主要实现面是 `python/sglang/srt/models/minicpmv.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/minicpmo.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/minicpmv.py` modified +5/-5 (10 lines); hunks: -68,7 +68,7; -936,7 +936,7 @@ def init_resampler(; symbols: init_resampler, get_vision_embedding，涉及 `init_resampler, get_vision_embedding`；`python/sglang/srt/models/kimi_vl_moonvit.py` modified +3/-3 (6 lines); hunks: -64,7 +64,7; -300,15 +300,15 @@ class Rope2DPosEmb(nn.Module):; symbols: Rope2DPosEmb, __init__, extra_repr，涉及 `Rope2DPosEmb, __init__, extra_repr`；`python/sglang/srt/models/minicpmo.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -1514,7 +1514,7 @@ def init_resampler(; symbols: init_resampler, pad_input_ids，涉及 `init_resampler, pad_input_ids`；`python/sglang/srt/models/transformers.py` modified +2/-1 (3 lines); hunks: -68,6 +68,7; -669,7 +670,7 @@ def _init_parameters(self, module: nn.Module):; symbols: _init_parameters，涉及 `_init_parameters`。
- 代码 diff 细节:
  - `python/sglang/srt/models/minicpmv.py` modified +5/-5 (10 lines); hunks: -68,7 +68,7; -936,7 +936,7 @@ def init_resampler(; symbols: init_resampler, get_vision_embedding
  - `python/sglang/srt/models/kimi_vl_moonvit.py` modified +3/-3 (6 lines); hunks: -64,7 +64,7; -300,15 +300,15 @@ class Rope2DPosEmb(nn.Module):; symbols: Rope2DPosEmb, __init__, extra_repr
  - `python/sglang/srt/models/minicpmo.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -1514,7 +1514,7 @@ def init_resampler(; symbols: init_resampler, pad_input_ids
  - `python/sglang/srt/models/transformers.py` modified +2/-1 (3 lines); hunks: -68,6 +68,7; -669,7 +670,7 @@ def _init_parameters(self, module: nn.Module):; symbols: _init_parameters
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +1/-0 (1 lines); hunks: -419,6 +419,7 @@ def forward_xpu(; symbols: forward_xpu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/minicpmv.py
@@ -68,7 +68,7 @@
-from sglang.srt.utils import add_prefix, flatten_nested_list
+from sglang.srt.utils import add_prefix, flatten_nested_list, get_device
@@ -936,7 +936,7 @@ def init_resampler(
-        return resampler.to(device="cuda", dtype=torch.get_default_dtype())
+        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())
@@ -1102,7 +1102,7 @@ def init_resampler(
diff -- python/sglang/srt/models/kimi_vl_moonvit.py
@@ -64,7 +64,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, get_device
@@ -300,15 +300,15 @@ class Rope2DPosEmb(nn.Module):
-        self, dim: int, max_height: int, max_width: int, theta_base=10000, device="cuda"
+        self, dim: int, max_height: int, max_width: int, theta_base=10000, device=None
-        self.device = device
diff -- python/sglang/srt/models/minicpmo.py
@@ -54,7 +54,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/minicpmv.py` modified +5/-5; `python/sglang/srt/models/kimi_vl_moonvit.py` modified +3/-3; `python/sglang/srt/models/minicpmo.py` modified +2/-2; `python/sglang/srt/models/transformers.py` modified +2/-1; `python/sglang/srt/layers/rotary_embedding/base.py` modified +1/-0; `python/sglang/srt/multimodal/processors/transformers_auto.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding/base.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/models/minicpmo.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25676 - Upgrade xgrammar to 0.2.1

- 链接: https://github.com/sgl-project/sglang/pull/25676
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+409/-174，可读 patch 834 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade xgrammar to 0.2.1」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`；技术摘要: 覆盖「Upgrade xgrammar to 0.2.1」；主要实现面是 `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/unit/function_call/test_function_call_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/entrypoints/openai/test_serving_chat.py` modified +284/-0 (284 lines); hunks: -147,6 +147,241 @@ def test_convert_to_internal_request_single(self):; -225,6 +460,55 @@ def test_jinja_tool_schema_fallback_to_flat_function(self):; symbols: test_convert_to_internal_request_single, test_kimi_tool_call_keeps_default_reasoning, test_kimi_tool_call_keeps_explicit_reasoning, test_kimi_tool_call_respects_explicit_reasoning_disable，涉及 `test_convert_to_internal_request_single, test_kimi_tool_call_keeps_default_reasoning, test_kimi_tool_call_keeps_explicit_reasoning`；`python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-115 (118 lines); hunks: -1,11 +1,10; -15,30 +14,8; symbols: DeepSeekV32Detector, structure_info, get_structural_tag, _invoke_tag，涉及 `DeepSeekV32Detector, structure_info, get_structural_tag`；`test/registered/unit/function_call/test_function_call_parser.py` modified +52/-30 (82 lines); hunks: -1642,12 +1642,17 @@ def test_streaming_no_parameters_with_whitespace(self):; -2087,12 +2092,17 @@ def test_streaming_no_parameters_with_whitespace(self):; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, _make_parser，涉及 `test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, _make_parser`；`python/sglang/srt/function_call/kimik2_detector.py` modified +51/-12 (63 lines); hunks: -1,10 +1,14; -23,6 +27,8; symbols: _strip_special_tokens, get_info, get_structural_tag, get_structural_tag_name，涉及 `_strip_special_tokens, get_info, get_structural_tag`。
- 代码 diff 细节:
  - `test/registered/unit/entrypoints/openai/test_serving_chat.py` modified +284/-0 (284 lines); hunks: -147,6 +147,241 @@ def test_convert_to_internal_request_single(self):; -225,6 +460,55 @@ def test_jinja_tool_schema_fallback_to_flat_function(self):; symbols: test_convert_to_internal_request_single, test_kimi_tool_call_keeps_default_reasoning, test_kimi_tool_call_keeps_explicit_reasoning, test_kimi_tool_call_respects_explicit_reasoning_disable
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-115 (118 lines); hunks: -1,11 +1,10; -15,30 +14,8; symbols: DeepSeekV32Detector, structure_info, get_structural_tag, _invoke_tag
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +52/-30 (82 lines); hunks: -1642,12 +1642,17 @@ def test_streaming_no_parameters_with_whitespace(self):; -2087,12 +2092,17 @@ def test_streaming_no_parameters_with_whitespace(self):; symbols: test_streaming_no_parameters_with_whitespace, test_get_model_structural_tag, _make_parser
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +51/-12 (63 lines); hunks: -1,10 +1,14; -23,6 +27,8; symbols: _strip_special_tokens, get_info, get_structural_tag, get_structural_tag_name
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +4/-2 (6 lines); hunks: -492,6 +492,8 @@ def _convert_to_internal_request(; -515,7 +517,7 @@ def _convert_to_internal_request(; symbols: _convert_to_internal_request, _process_messages
- 关键代码摘录:

```diff
diff -- test/registered/unit/entrypoints/openai/test_serving_chat.py
@@ -147,6 +147,241 @@ def test_convert_to_internal_request_single(self):
+    def test_kimi_tool_call_keeps_default_reasoning(self):
+        self.template_manager.reasoning_config = ReasoningToggleConfig(
+            toggle_param="thinking", default_enabled=True
+        )
+        self.tm.server_args.reasoning_parser = "kimi_k2"
+        self.tm.server_args.tool_call_parser = "kimi_k2"
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -1,11 +1,10 @@
-from typing import List, Literal, Optional, Union
-from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
+from sglang.srt.entrypoints.openai.protocol import Tool
@@ -15,30 +14,8 @@
-try:
-    from xgrammar import StructuralTag
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -1642,12 +1642,17 @@ def test_streaming_no_parameters_with_whitespace(self):
```

- 已读文件:
  - tests: `test/registered/unit/entrypoints/openai/test_serving_chat.py` modified +284/-0; `test/registered/unit/function_call/test_function_call_parser.py` modified +52/-30
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-115; `python/sglang/srt/function_call/kimik2_detector.py` modified +51/-12; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +4/-2
  - ci: `.github/workflows/nightly-test-npu.yml` modified +5/-5; `.github/workflows/full-test-npu.yml` modified +4/-4
  - other: `3rdparty/amd/wheel/sglang/pyproject.toml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `test/registered/unit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26744 - [RL] Forward Kimi K2.5 weight hooks to language model

- 链接: https://github.com/sgl-project/sglang/pull/26744
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `6ea69efb7f65`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-0，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[RL] Forward Kimi K2.5 weight hooks to language model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[RL] Forward Kimi K2.5 weight hooks to language model」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +18/-0 (18 lines); hunks: -804,6 +804,24 @@ def stream_language_weights():; symbols: stream_language_weights, post_load_weights, stacked_params_mapping, expert_params_mapping，涉及 `stream_language_weights, post_load_weights, stacked_params_mapping`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +18/-0 (18 lines); hunks: -804,6 +804,24 @@ def stream_language_weights():; symbols: stream_language_weights, post_load_weights, stacked_params_mapping, expert_params_mapping
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -804,6 +804,24 @@ def stream_language_weights():
+    def post_load_weights(self):
+        if self.language_model is not None:
+            self.language_model.post_load_weights()
+    @property
+    def stacked_params_mapping(self):
+        return getattr(self.language_model, "stacked_params_mapping", [])
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +18/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26555 - [RL+VLM] Avoid retokenization drift for pre-tokenized (token-id) VLM requests

- 链接: https://github.com/sgl-project/sglang/pull/26555
- 状态/时间: merged / 2026-06-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+227/-0，可读 patch 256 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[RL+VLM] Avoid retokenization drift for pre-tokenized (token-id) VLM requests」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `test/registered/vlm/test_token_id_retokenize_e2e.py`；技术摘要: 覆盖「[RL+VLM] Avoid retokenization drift for pre-tokenized (token-id) VLM requests」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `test/registered/vlm/test_token_id_retokenize_e2e.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +94/-0 (94 lines); hunks: -1219,6 +1219,58 @@ def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):; -1268,6 +1320,48 @@ def process_and_combine_mm_data(; symbols: _wrap_tensor_for_cuda_ipc, resolve_image_token_counts, _expand_input_ids, process_and_combine_mm_data，涉及 `_wrap_tensor_for_cuda_ipc, resolve_image_token_counts, _expand_input_ids`；`python/sglang/srt/multimodal/processors/kimi_common.py` modified +15/-0 (15 lines); hunks: -23,6 +23,21 @@ class KimiGridMMDataMixin:; symbols: KimiGridMMDataMixin, resolve_image_token_counts, _num_image_tokens_from_grid，涉及 `KimiGridMMDataMixin, resolve_image_token_counts, _num_image_tokens_from_grid`；`test/registered/vlm/test_token_id_retokenize_e2e.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: _data_uri, _build_drift_prompt, enc, _prompt_tokens，涉及 `_data_uri, _build_drift_prompt, enc`；`python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -539,6 +539,9 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +94/-0 (94 lines); hunks: -1219,6 +1219,58 @@ def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):; -1268,6 +1320,48 @@ def process_and_combine_mm_data(; symbols: _wrap_tensor_for_cuda_ipc, resolve_image_token_counts, _expand_input_ids, process_and_combine_mm_data
  - `python/sglang/srt/multimodal/processors/kimi_common.py` modified +15/-0 (15 lines); hunks: -23,6 +23,21 @@ class KimiGridMMDataMixin:; symbols: KimiGridMMDataMixin, resolve_image_token_counts, _num_image_tokens_from_grid
  - `test/registered/vlm/test_token_id_retokenize_e2e.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: _data_uri, _build_drift_prompt, enc, _prompt_tokens
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -539,6 +539,9 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1219,6 +1219,58 @@ def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):
+    def resolve_image_token_counts(self, images: List) -> List[int]:
+        """Per-image expanded token counts, computed without re-tokenizing.
+        Default implementation uses the transformers in-tree convention
+        ``_get_num_multimodal_tokens(image_sizes=...)`` (present on the in-tree
+        VLM processors, e.g. Qwen-VL, Gemma3, GLM4V). Models whose processor
+        does not implement it (e.g. Kimi) override this method.
diff -- python/sglang/srt/multimodal/processors/kimi_common.py
@@ -23,6 +23,21 @@ class KimiGridMMDataMixin:
+    def resolve_image_token_counts(self, images):
+        """Kimi's processor is remote-code and does not implement the
+        transformers ``_get_num_multimodal_tokens`` convention; use its
+        ``media_tokens_calculator`` instead.
+        """
+        assert images is not None
diff -- test/registered/vlm/test_token_id_retokenize_e2e.py
@@ -0,0 +1,115 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +94/-0; `python/sglang/srt/multimodal/processors/kimi_common.py` modified +15/-0; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `test/registered/vlm/test_token_id_retokenize_e2e.py` added +115/-0
- 验证与风险: diff 自带测试面 `test/registered/vlm/test_token_id_retokenize_e2e.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- 链接: https://github.com/sgl-project/sglang/pull/25813
- 状态/时间: merged / 2026-06-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+1262/-2154，可读 patch 4187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): port popular model usage guides into cookbook pages」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`；技术摘要: 覆盖「docs(cookbook): port popular model usage guides into cookbook pages」；主要实现面是 `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #24762 - [AMD] fix(triton-mla): cap max_kv_splits at 256 on gfx942 (Kimi-K2.6 hang)

- 链接: https://github.com/sgl-project/sglang/pull/24762
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k2_instruct.py`；关联提交 `8e77af1afcee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+27/-4，可读 patch 84 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] fix(triton-mla): cap max_kv_splits at 256 on gfx942 (Kimi-K2.6 hang)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/test_kimi_k2_instruct.py`, `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/utils/common.py`；技术摘要: 覆盖「[AMD] fix(triton-mla): cap max_kv_splits at 256 on gfx942 (Kimi-K2.6 hang)」；主要实现面是 `test/registered/amd/test_kimi_k2_instruct.py`, `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/utils/common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_kimi_k2_instruct.py` modified +1/-1 (2 lines); hunks: -63,7 +63,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k，涉及 `test_a_gsm8k`；`python/sglang/srt/layers/attention/triton_backend.py` modified +11/-0 (11 lines); hunks: -20,10 +20,12; -161,6 +163,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/utils/common.py` modified +12/-0 (12 lines); hunks: -3620,6 +3620,18 @@ def is_gfx95_supported():; symbols: is_gfx95_supported, is_gfx942_supported, get_hip_version，涉及 `is_gfx95_supported, is_gfx942_supported, get_hip_version`。
- 代码 diff 细节:
  - `test/registered/amd/test_kimi_k2_instruct.py` modified +1/-1 (2 lines); hunks: -63,7 +63,7 @@ def test_a_gsm8k(; symbols: test_a_gsm8k
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +11/-0 (11 lines); hunks: -20,10 +20,12; -161,6 +163,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/common.py` modified +12/-0 (12 lines); hunks: -3620,6 +3620,18 @@ def is_gfx95_supported():; symbols: is_gfx95_supported, is_gfx942_supported, get_hip_version
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_kimi_k2_instruct.py
@@ -63,7 +63,7 @@ def test_a_gsm8k(
-            parallel=1319,
+            parallel=512,
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -20,10 +20,12 @@
+    is_gfx942_supported,
+_is_gfx942 = is_gfx942_supported()
@@ -161,6 +163,15 @@ def __init__(
+            if _is_gfx942:
+                # gfx942 (MI300X / MI325X) has 304 CUs, so #20479's next_power_of_2(sm_count)
+                # rounds up to 512 — twice MI355X's natural cap of 256 — and the persistent
diff -- python/sglang/srt/utils/common.py
@@ -3620,6 +3620,18 @@ def is_gfx95_supported():
+@lru_cache(maxsize=1)
+def is_gfx942_supported():
+    """
+    Returns whether the current platform is AMD CDNA3 (gfx942 — MI300X / MI325X).
```

- 已读文件:
  - tests: `test/registered/amd/test_kimi_k2_instruct.py` modified +1/-1
  - runtime: `python/sglang/srt/layers/attention/triton_backend.py` modified +11/-0; `python/sglang/srt/utils/common.py` modified +12/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_kimi_k2_instruct.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/27001
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+11/-471，可读 patch 936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`；技术摘要: 覆盖「[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests」；主要实现面是 `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #22488 - Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation

- 链接: https://github.com/sgl-project/sglang/pull/22488
- 状态/时间: closed / 2026-06-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+794/-53，可读 patch 890 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`；技术摘要: 覆盖「Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -111,7 +111,6 @@ def routing(; -167,9 +166,11 @@ def fused_topk_deepseek(; symbols: routing, fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess，涉及 `routing, fused_topk_deepseek, biased_grouped_topk_impl`；`python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344；`python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts，涉及 `_reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped`；`python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped，涉及 `_jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: -111,7 +111,6 @@ def routing(; -167,9 +166,11 @@ def fused_topk_deepseek(; symbols: routing, fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess
  - `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: -0,0 +1,344
  - `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts
  - `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -111,7 +111,6 @@ def routing(
-from sglang.srt.utils.patch_torch import register_fake_if_exists
@@ -167,9 +166,11 @@ def fused_topk_deepseek(
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

### PR #27647 - [sgl] Fix kimi-k2.5 EAGLE3 MLA draft embeds for batched MM prefill

- 链接: https://github.com/sgl-project/sglang/pull/27647
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25_eagle3.py`；关联提交 `8ae328e5f042`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-3，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[sgl] Fix kimi-k2.5 EAGLE3 MLA draft embeds for batched MM prefill」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k25_eagle3.py`；技术摘要: 覆盖「[sgl] Fix kimi-k2.5 EAGLE3 MLA draft embeds for batched MM prefill」；主要实现面是 `python/sglang/srt/models/kimi_k25_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +4/-3 (7 lines); hunks: -261,9 +261,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25_eagle3.py` modified +4/-3 (7 lines); hunks: -261,9 +261,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25_eagle3.py
@@ -261,9 +261,10 @@ def forward(
-                embeds = torch.cat(
-                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
-                )
+                last_indices = (
+                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
+                ).long()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/kimi_k25_eagle3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8007 - [Kimi K2] num_experts extends to 384

- 链接: https://github.com/sgl-project/sglang/pull/8007
- 状态/时间: closed / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+30/-4，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K2] num_experts extends to 384」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`；技术摘要: 覆盖「[Kimi K2] num_experts extends to 384」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- 链接: https://github.com/sgl-project/sglang/pull/23906
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 160 个文件，+5197/-3068，可读 patch 12233 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Cuda Graph Runner/Backend Refactor」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`；技术摘要: 覆盖「[Refactor] Cuda Graph Runner/Backend Refactor」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27714 - [Docs] Add Kimi-K2.6 NVFP4 and update Kimi-K2.5 cookbook guidance

- 链接: https://github.com/sgl-project/sglang/pull/27714
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+198/-38，可读 patch 468 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add Kimi-K2.6 NVFP4 and update Kimi-K2.5 cookbook guidance」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`；技术摘要: 覆盖「[Docs] Add Kimi-K2.6 NVFP4 and update Kimi-K2.5 cookbook guidance」；主要实现面是 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +88/-2 (90 lines); hunks: -66,7 +66,12 @@ tag: NEW; -86,7 +91,8 @@ import { KimiK26Deployment } from '/src/snippets/autoregressiv...；`docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +81/-7 (88 lines); hunks: -1,21 +1,39; -40,20 +58,32 @@ export const KimiK26Deployment = () => {；`docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +18/-20 (38 lines); hunks: -4,20 +4,22 @@ export const KimiK25Deployment = () => {; -29,10 +31,10 @@ export const KimiK25Deployment = () => {；`docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +11/-9 (20 lines); hunks: -37,7 +37,7 @@ import { KimiK25Deployment } from '/src/snippets/autoregressiv...; -440,10 +440,10 @@ Let me search for this product and similar items:。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +88/-2 (90 lines); hunks: -66,7 +66,12 @@ tag: NEW; -86,7 +91,8 @@ import { KimiK26Deployment } from '/src/snippets/autoregressiv...
  - `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +81/-7 (88 lines); hunks: -1,21 +1,39; -40,20 +58,32 @@ export const KimiK26Deployment = () => {
  - `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +18/-20 (38 lines); hunks: -4,20 +4,22 @@ export const KimiK25Deployment = () => {; -29,10 +31,10 @@ export const KimiK25Deployment = () => {
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +11/-9 (20 lines); hunks: -37,7 +37,7 @@ import { KimiK25Deployment } from '/src/snippets/autoregressiv...; -440,10 +440,10 @@ Let me search for this product and similar items:
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx
@@ -66,7 +66,12 @@ tag: NEW
-**License:** Modified MIT
+**Available Models:**
+- **INT4 (native checkpoint)**: [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)
+- **NVFP4 (4-bit quantized, NVIDIA Blackwell)**: [nvidia/Kimi-K2.6-NVFP4](https://huggingface.co/nvidia/Kimi-K2.6-NVFP4)
+**License:** Modified MIT for the native checkpoint. The NVIDIA NVFP4 checkpoint is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-
@@ -86,7 +91,8 @@ import { KimiK26Deployment } from '/src/snippets/autoregressive/kimi-k26-deploym
diff -- docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx
@@ -1,21 +1,39 @@
+  //
+  // INT4:
+  //   H200/B300: tp=8
+  //   GB300/AMD: tp=4
+  //
+  // NVFP4:
diff -- docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx
@@ -4,20 +4,22 @@ export const KimiK25Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +88/-2; `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +81/-7; `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +18/-20; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx` modified +11/-9
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.5.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28064 - [Docs] Add Kimi K2.7 Code cookbook

- 链接: https://github.com/sgl-project/sglang/pull/28064
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+772/-3，可读 patch 805 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add Kimi K2.7 Code cookbook」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`, `docs_new/cookbook/intro copy.mdx`；技术摘要: 覆盖「[Docs] Add Kimi K2.7 Code cookbook」；主要实现面是 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`, `docs_new/cookbook/intro copy.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx` added +557/-0 (557 lines); hunks: -0,0 +1,557；`docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` added +211/-0 (211 lines); hunks: -0,0 +1,211；`docs_new/cookbook/intro copy.mdx` modified +2/-1 (3 lines); hunks: -67,7 +67,8 @@ Each recipe provides step-by-step instructions to help you qui...；`docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ metatags:。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx` added +557/-0 (557 lines); hunks: -0,0 +1,557
  - `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` added +211/-0 (211 lines); hunks: -0,0 +1,211
  - `docs_new/cookbook/intro copy.mdx` modified +2/-1 (3 lines); hunks: -67,7 +67,8 @@ Each recipe provides step-by-step instructions to help you qui...
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: -52,7 +52,7 @@ metatags:
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +0/-1 (1 lines); hunks: -2,7 +2,6
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx
@@ -0,0 +1,557 @@
+---
+title: Kimi-K2.7-Code
+description: "Deploy Kimi-K2.7-Code with SGLang for coding-focused agentic workflows, thinking output, tool calling, and multimodal input."
+metatags:
+    description: "Deploy Kimi-K2.7-Code native multimodal agentic model with SGLang - reasoning, tool calling, and multimodal capabilities."
+tag: NEW
diff -- docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx
@@ -0,0 +1,211 @@
+export const KimiK27CodeDeployment = () => {
+  // Kimi-K2.7-Code reuses the Kimi-K2.6 architecture and deployment layout.
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
diff -- docs_new/cookbook/intro copy.mdx
@@ -67,7 +67,8 @@ Each recipe provides step-by-step instructions to help you quickly implement SGL
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx` added +557/-0; `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` added +211/-0; `docs_new/cookbook/intro copy.mdx` modified +2/-1; `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx` modified +0/-1; `docs_new/docs.json` modified +1/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.6.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28467 - [ci] add kimi nvfp4 nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/28467
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+68/-0，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ci] add kimi nvfp4 nightly tests」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/quant/test_kimi_k25_nvfp4_eagle.py`；技术摘要: 覆盖「[ci] add kimi nvfp4 nightly tests」；主要实现面是 `test/registered/quant/test_kimi_k25_nvfp4_eagle.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/quant/test_kimi_k25_nvfp4_eagle.py` added +68/-0 (68 lines); hunks: -0,0 +1,68; symbols: TestKimiK25Nvfp4Eagle, test_kimi_k25_nvfp4_eagle，涉及 `TestKimiK25Nvfp4Eagle, test_kimi_k25_nvfp4_eagle`。
- 代码 diff 细节:
  - `test/registered/quant/test_kimi_k25_nvfp4_eagle.py` added +68/-0 (68 lines); hunks: -0,0 +1,68; symbols: TestKimiK25Nvfp4Eagle, test_kimi_k25_nvfp4_eagle
- 关键代码摘录:

```diff
diff -- test/registered/quant/test_kimi_k25_nvfp4_eagle.py
@@ -0,0 +1,68 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- 已读文件:
  - tests: `test/registered/quant/test_kimi_k25_nvfp4_eagle.py` added +68/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_kimi_k25_nvfp4_eagle.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28201 - [Docs] Add fp8 kv cache for tokenspeed mla docs

- 链接: https://github.com/sgl-project/sglang/pull/28201
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+13/-7，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Add fp8 kv cache for tokenspeed mla docs」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`；技术摘要: 覆盖「[Docs] Add fp8 kv cache for tokenspeed mla docs」；主要实现面是 `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +5/-3 (8 lines); hunks: -196,13 +196,15 @@ export const KimiK25Deployment = () => {；`docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +4/-2 (6 lines); hunks: -194,11 +194,13 @@ export const KimiK26Deployment = () => {；`docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` modified +4/-2 (6 lines); hunks: -144,11 +144,13 @@ export const KimiK27CodeDeployment = () => {。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +5/-3 (8 lines); hunks: -196,13 +196,15 @@ export const KimiK25Deployment = () => {
  - `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +4/-2 (6 lines); hunks: -194,11 +194,13 @@ export const KimiK26Deployment = () => {
  - `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` modified +4/-2 (6 lines); hunks: -144,11 +144,13 @@ export const KimiK27CodeDeployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx
@@ -196,13 +196,15 @@ export const KimiK25Deployment = () => {
+    const usesTokenspeedMla = hardware === 'b300' || hardware === 'gb300';
-    if (hardware === 'b300' || hardware === 'gb300') {
+    if (usesTokenspeedMla) {
-    // AMD: FP8 KV cache for memory efficiency
-    if (isAMD) {
+    // FP8 KV cache for AMD memory efficiency and tokenspeed MLA compatibility
diff -- docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx
@@ -194,11 +194,13 @@ export const KimiK26Deployment = () => {
-    if (hardware === 'b300' || hardware === 'gb300') {
+    const usesTokenspeedMla = hardware === 'b300' || hardware === 'gb300';
+    if (usesTokenspeedMla) {
-    if (isAMD) {
+    if (isAMD || usesTokenspeedMla) {
diff -- docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx
@@ -144,11 +144,13 @@ export const KimiK27CodeDeployment = () => {
-    if (hardware === 'b300' || hardware === 'gb300') {
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx` modified +5/-3; `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx` modified +4/-2; `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx` modified +4/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/src/snippets/autoregressive/kimi-k25-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k26-deployment.jsx`, `docs_new/src/snippets/autoregressive/kimi-k27-code-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #28697 - [docs] Add B300 cookbook deployment options

- 链接: https://github.com/sgl-project/sglang/pull/28697
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+503/-69，可读 patch 1291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[docs] Add B300 cookbook deployment options」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`；技术摘要: 覆盖「[docs] Add B300 cookbook deployment options」；主要实现面是 `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`；技术摘要: 覆盖「ci: run GB300 nightly suite in the standard Nvidia nightly workflow」；主要实现面是 `test/registered/gb300/test_deepseek_v32_nvfp4.py`, `test/registered/gb300/test_deepseek_v32.py`, `test/registered/gb300/test_qwen35_fp8.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #28647 - Fix Kimi-VL GPU image preprocessing crash on non-RGB images

- 链接: https://github.com/sgl-project/sglang/pull/28647
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/kimi_k25.py`；关联提交 `4f60378ff539`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+30/-0，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Kimi-VL GPU image preprocessing crash on non-RGB images」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`；技术摘要: 覆盖「Fix Kimi-VL GPU image preprocessing crash on non-RGB images」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +30/-0 (30 lines); hunks: -82,6 +82,32 @@ def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:; -92,6 +118,8 @@ def _process_single_image(; symbols: _pil_to_cuda_chw, _ensure_chw_rgb, _process_single_image, _gpu_preprocess_images，涉及 `_pil_to_cuda_chw, _ensure_chw_rgb, _process_single_image`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +30/-0 (30 lines); hunks: -82,6 +82,32 @@ def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:; -92,6 +118,8 @@ def _process_single_image(; symbols: _pil_to_cuda_chw, _ensure_chw_rgb, _process_single_image, _gpu_preprocess_images
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -82,6 +82,32 @@ def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:
+def _ensure_chw_rgb(image: torch.Tensor) -> torch.Tensor:
+    """Coerce an already-decoded (C, H, W) image tensor to 3-channel RGB.
+    PIL inputs are RGB-normalized by _pil_to_cuda_chw, but pre-decoded
+    tensor inputs (e.g. nvJPEG / cached CUDA tensors) keep their native
+    channel count. Grayscale (1ch) or RGBA (4ch) images then break the
+    downstream torch.cat over a batch of images, which requires a
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +30/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22496 - [Feature] kimi k25 w4a16 support deepep low latency

- 链接: https://github.com/sgl-project/sglang/pull/22496
- 状态/时间: closed / 2026-06-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+4882/-25，可读 patch 5138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] kimi k25 w4a16 support deepep low latency」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；技术摘要: 覆盖「[Feature] kimi k25 w4a16 support deepep low latency」；主要实现面是 `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #27833 - [AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5

- 链接: https://github.com/sgl-project/sglang/pull/27833
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`；关联提交 `20b2817bdfcc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+187/-0，可读 patch 202 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`；技术摘要: 覆盖「[AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5」；主要实现面是 `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x, setUpClass，涉及 `CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x`；`python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0 (8 lines); hunks: -1,6 +1,9; -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):; symbols: handle_attention_tokenspeed_mla, handle_attention_aiter，涉及 `handle_attention_tokenspeed_mla, handle_attention_aiter`。
- 代码 diff 细节:
  - `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x, setUpClass
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0 (8 lines); hunks: -1,6 +1,9; -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):; symbols: handle_attention_tokenspeed_mla, handle_attention_aiter
- 关键代码摘录:

```diff
diff -- test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py
@@ -0,0 +1,179 @@
+"""Kimi-K2.5-MXFP4 aiter breakable CUDA-graph (BCG) capture accuracy test
+(MI35x, PR-CI)
+Exercises the AMD breakable (BCG) CUDA-graph prefill capture path on a
+deepseek-family (Kimi-K2.5) aiter model so the code added in this PR actually
+runs in PR CI:
+  * runner_backend/breakable_cuda_graph_backend.py
diff -- python/sglang/srt/models/deepseek_common/attention_backend_handler.py
@@ -1,6 +1,9 @@
+from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
+    is_in_breakable_cuda_graph,
+)
@@ -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):
+    # During PCG/BCG capture on ROCm, aiter fp8 MLA prefill has no capture
+    # kernels; route through the MHA path (radix_attention swaps attn_mqa for
```

- 已读文件:
  - tests: `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0
  - runtime: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25071 - kimik2_detector fix the normal text detection before tool call.

- 链接: https://github.com/sgl-project/sglang/pull/25071
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`, `test/registered/function_call/test_kimik2_detector.py`；关联提交 `5f767364279c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+774/-101，可读 patch 960 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「kimik2_detector fix the normal text detection before tool call.」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`；技术摘要: 覆盖「kimik2_detector fix the normal text detection before tool call.」；主要实现面是 `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_kimik2_detector.py` modified +587/-0 (587 lines); hunks: -111,6 +111,32 @@ def test_multiple_tool_calls(self):; -183,6 +209,15 @@ def test_hyphenated_name_streaming(self):; symbols: test_multiple_tool_calls, test_non_streaming_tool_index_is_local, test_normal_text_before_tool_call, test_hyphenated_name_streaming，涉及 `test_multiple_tool_calls, test_non_streaming_tool_index_is_local, test_normal_text_before_tool_call`；`python/sglang/srt/function_call/kimik2_detector.py` modified +187/-101 (288 lines); hunks: -15,7 +15,6; -177,9 +176,15 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment, _reset_inflight_call_state，涉及 `detect_and_parse, parse_streaming_increment, _reset_inflight_call_state`。
- 代码 diff 细节:
  - `test/registered/function_call/test_kimik2_detector.py` modified +587/-0 (587 lines); hunks: -111,6 +111,32 @@ def test_multiple_tool_calls(self):; -183,6 +209,15 @@ def test_hyphenated_name_streaming(self):; symbols: test_multiple_tool_calls, test_non_streaming_tool_index_is_local, test_normal_text_before_tool_call, test_hyphenated_name_streaming
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +187/-101 (288 lines); hunks: -15,7 +15,6; -177,9 +176,15 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment, _reset_inflight_call_state
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -111,6 +111,32 @@ def test_multiple_tool_calls(self):
+    def test_non_streaming_tool_index_is_local(self):
+        """tool_index is the per-response 0-based position, not the model's :N suffix.
+        The model may emit conversation-level ``:N`` counters (e.g. ``:5``, ``:6``)
+        in a multi-turn conversation. The non-streaming parser must enumerate
+        parsed calls locally (0, 1, ...) so that
+        ``serving_chat._process_tool_call_id()`` can offset them by
diff -- python/sglang/srt/function_call/kimik2_detector.py
@@ -15,7 +15,6 @@
-from sglang.srt.function_call.utils import _is_complete_json
@@ -177,9 +176,15 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
+            # ``tool_index`` is the per-response 0-based position of the call
+            # (OpenAI spec); enumerate parsed calls locally and ignore the
+            # model's ``:N`` suffix, which is a conversation-level counter.
+            # ``serving_chat._process_tool_call_id()`` later offsets these by
```

- 已读文件:
  - tests: `test/registered/function_call/test_kimik2_detector.py` modified +587/-0
  - runtime: `python/sglang/srt/function_call/kimik2_detector.py` modified +187/-101
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_kimik2_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28623 - [CI] reduce CPU CI scope with base-c suite

- 链接: https://github.com/sgl-project/sglang/pull/28623
- 状态/时间: merged / 2026-06-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 91 个文件，+96/-91，可读 patch 824 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] reduce CPU CI scope with base-c suite」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/function_call/test_kimik2_detector.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/unit/entrypoints/openai/test_serving_embedding.py`；技术摘要: 覆盖「[CI] reduce CPU CI scope with base-c suite」；主要实现面是 `test/registered/function_call/test_kimik2_detector.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/unit/entrypoints/openai/test_serving_embedding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7; symbols: _make_tool，涉及 `_make_tool`；`test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7; symbols: TestTransformersBackendEval，涉及 `TestTransformersBackendEval`；`test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ def find_spec(self, fullname, path, target=None):; symbols: find_spec，涉及 `find_spec`；`test/registered/unit/function_call/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7; symbols: TestPythonicDetector，涉及 `TestPythonicDetector`。
- 代码 diff 细节:
  - `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7; symbols: _make_tool
  - `test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7; symbols: TestTransformersBackendEval
  - `test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ def find_spec(self, fullname, path, target=None):; symbols: find_spec
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7; symbols: TestPythonicDetector
  - `test/registered/unit/function_call/test_json_schema_constraint.py` modified +1/-1 (2 lines); hunks: -19,7 +19,7; symbols: TestJsonSchemaConstraint
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -12,7 +12,7 @@
-register_cpu_ci(est_time=7, suite="base-b-test-cpu")
+register_cpu_ci(est_time=7, suite="base-c-test-cpu")
diff -- test/registered/models/test_transformers_backend_eval.py
@@ -13,7 +13,7 @@
-register_cpu_ci(est_time=320, suite="base-b-test-cpu")
+register_cpu_ci(est_time=320, suite="base-c-test-cpu")
diff -- test/registered/unit/entrypoints/openai/test_serving_embedding.py
@@ -57,7 +57,7 @@ def find_spec(self, fullname, path, target=None):
-register_cpu_ci(est_time=8, suite="base-b-test-cpu")
+register_cpu_ci(est_time=8, suite="base-c-test-cpu")
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -32,7 +32,7 @@
-register_cpu_ci(est_time=61, suite="base-b-test-cpu")
+register_cpu_ci(est_time=61, suite="base-c-test-cpu")
diff -- test/registered/unit/function_call/test_json_schema_constraint.py
@@ -19,7 +19,7 @@
```

- 已读文件:
  - tests: `test/registered/function_call/test_kimik2_detector.py` modified +1/-1; `test/registered/models/test_transformers_backend_eval.py` modified +1/-1; `test/registered/unit/entrypoints/openai/test_serving_embedding.py` modified +1/-1; `test/registered/unit/function_call/test_function_call_parser.py` modified +1/-1; `test/registered/unit/function_call/test_json_schema_constraint.py` modified +1/-1; `test/registered/unit/function_call/test_parallel_tool_calls.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/bench_fn/test_benchmark_datasets_api.py`, `test/registered/debug_utils/comparator/aligner/entrypoint/test_executor.py`, `test/registered/debug_utils/comparator/aligner/entrypoint/test_planner.py`, `test/registered/debug_utils/comparator/aligner/reorderer/test_executor.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28103 - Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/28103
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+218/-19，可读 patch 334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`；技术摘要: 覆盖「Add DeepSeek V4 Pro GB300 nightly and expand Kimi K25 nightly test」；主要实现面是 `test/registered/gb300/test_deepseek_v4_pro_fp4.py`, `test/registered/gb300/test_kimi_k25_nvfp4.py`, `.github/workflows/nightly-test-nvidia.yml`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #29223 - (perf): Shard Kimi-K2.5 Eagle3 draft fc + symm-mem AG

- 链接: https://github.com/sgl-project/sglang/pull/29223
- 状态/时间: merged / 2026-06-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25_eagle3.py`；关联提交 `da802ddcafe5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+885/-7，可读 patch 942 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「(perf): Shard Kimi-K2.5 Eagle3 draft fc + symm-mem AG」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25_eagle3.py`；技术摘要: 覆盖「(perf): Shard Kimi-K2.5 Eagle3 draft fc + symm-mem AG」；主要实现面是 `python/sglang/srt/models/kimi_k25_eagle3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-3 (18 lines); hunks: -25,9 +25,10; -207,10 +208,20 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-3 (18 lines); hunks: -25,9 +25,10; -207,10 +208,20 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25_eagle3.py
@@ -25,9 +25,10 @@
+from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
-from sglang.srt.layers.linear import ReplicatedLinear
+from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
@@ -207,10 +208,20 @@ def __init__(
-        self.fc = nn.Linear(
+        self.fc = ColumnParallelLinear(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25_eagle3.py` modified +15/-3
- 验证与风险: diff 自带测试面 `test/registered/jit/benchmark/bench_symm_mem_all_gather.py`, `test/registered/jit/test_symm_mem_all_gather.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29855 - [AMD][DI][CI] 3/N Add Kimi K2.6 FP8 MI355X 1P1D nightly recipes

- 链接: https://github.com/sgl-project/sglang/pull/29855
- 状态/时间: merged / 2026-07-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml`；关联提交 `67361ff91b5f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+303/-22，可读 patch 414 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DI][CI] 3/N Add Kimi K2.6 FP8 MI355X 1P1D nightly recipes」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml`；技术摘要: 覆盖「[AMD][DI][CI] 3/N Add Kimi K2.6 FP8 MI355X 1P1D nightly recipes」；主要实现面是 `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml` added +85/-0 (85 lines); hunks: -0,0 +1,85；`scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75。
- 代码 diff 细节:
  - `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml` added +85/-0 (85 lines); hunks: -0,0 +1,85
  - `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml` added +75/-0 (75 lines); hunks: -0,0 +1,75
- 关键代码摘录:

```diff
diff -- scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml
@@ -0,0 +1,85 @@
+# MI355X Kimi-K2.6 (FP8) 2-node 1P1D disaggregation recipe (base + EAGLE3 MTP).
+#
+# Self-contained (no inheritance): same as 1p1d.yaml plus the `mtp:` block. Uses
+# EAGLE3 speculative decoding with an EXTERNAL draft checkpoint (unlike DSV4's
+# built-in EAGLE NextN head): the launcher resolves mtp.draft_model_path through
+# the HF-cache snapshot logic and appends --speculative-draft-model-path. The
diff -- scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml
@@ -0,0 +1,75 @@
+# MI355X Kimi-K2.6 (FP8) 2-node 1P1D disaggregation recipe (base).
+#
+# All Kimi-specific config lives in this recipe's `model:` block (docker env +
+# sglang server args) and `runtime` (split attention backends); nothing about
+# Kimi is hardcoded in launch_mi355x.sh. Mirrors the single-node registered test
+# test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py (TP8, split
```

- 已读文件:
  - other: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d-mtp.yaml` added +85/-0; `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/1p1d.yaml` added +75/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #29218 - [Spec] DFlash: support pure-MLA targets with an fp8 KV cache (Kimi-K2.x-NVFP4)

- 链接: https://github.com/sgl-project/sglang/pull/29218
- 状态/时间: merged / 2026-07-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+96/-8，可读 patch 142 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec] DFlash: support pure-MLA targets with an fp8 KV cache (Kimi-K2.x-NVFP4)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `test/registered/quant/test_kimi_k26_nvfp4_dflash.py`, `python/sglang/srt/speculative/dflash_worker_v2.py`；技术摘要: 覆盖「[Spec] DFlash: support pure-MLA targets with an fp8 KV cache (Kimi-K2.x-NVFP4)」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `test/registered/quant/test_kimi_k26_nvfp4_dflash.py`, `python/sglang/srt/speculative/dflash_worker_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +16/-0 (16 lines); hunks: -2483,6 +2483,22 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype, init_cublas，涉及 `configure_kv_cache_dtype, init_cublas`；`test/registered/quant/test_kimi_k26_nvfp4_dflash.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestKimiK26Nvfp4Dflash, test_kimi_k26_nvfp4_dflash，涉及 `TestKimiK26Nvfp4Dflash, test_kimi_k26_nvfp4_dflash`；`python/sglang/srt/speculative/dflash_worker_v2.py` modified +10/-8 (18 lines); hunks: -117,6 +117,12 @@ def __init__(; -1099,9 +1105,9 @@ def _update_target_mamba_state_after_verify(; symbols: __init__, _update_target_mamba_state_after_verify, forward_batch_generation，涉及 `__init__, _update_target_mamba_state_after_verify, forward_batch_generation`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-0 (16 lines); hunks: -2483,6 +2483,22 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype, init_cublas
  - `test/registered/quant/test_kimi_k26_nvfp4_dflash.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestKimiK26Nvfp4Dflash, test_kimi_k26_nvfp4_dflash
  - `python/sglang/srt/speculative/dflash_worker_v2.py` modified +10/-8 (18 lines); hunks: -117,6 +117,12 @@ def __init__(; -1099,9 +1105,9 @@ def _update_target_mamba_state_after_verify(; symbols: __init__, _update_target_mamba_state_after_verify, forward_batch_generation
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2483,6 +2483,22 @@ def configure_kv_cache_dtype(self):
+        # DFLASH: fa4 draft attention can't read the target's fp8 KV (needs K.dtype == Q.dtype),
+        # so give the fa4 draft its own compute-dtype KV. fp8-capable backends keep the target dtype.
+        if (
+            self.is_draft_worker
+            and self.spec_algorithm.is_dflash()
+            and self.server_args.speculative_draft_attention_backend == "fa4"
diff -- test/registered/quant/test_kimi_k26_nvfp4_dflash.py
@@ -0,0 +1,70 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
diff -- python/sglang/srt/speculative/dflash_worker_v2.py
@@ -117,6 +117,12 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +16/-0; `python/sglang/srt/speculative/dflash_worker_v2.py` modified +10/-8
  - tests: `test/registered/quant/test_kimi_k26_nvfp4_dflash.py` added +70/-0
- 验证与风险: diff 自带测试面 `test/registered/quant/test_kimi_k26_nvfp4_dflash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30878 - perf: reuse MoonViT FA3 max-seqlen metadata

- 链接: https://github.com/sgl-project/sglang/pull/30878
- 状态/时间: merged / 2026-07-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `bce3fc987d89`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+97/-3，可读 patch 122 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf: reuse MoonViT FA3 max-seqlen metadata」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「perf: reuse MoonViT FA3 max-seqlen metadata」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +5/-1 (6 lines); hunks: -150,6 +150,7 @@ def forward(; -469,7 +470,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +5/-1 (6 lines); hunks: -150,6 +150,7 @@ def forward(; -469,7 +470,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -150,6 +150,7 @@ def forward(
+            max_seqlen=max_seqlen,
@@ -469,7 +470,10 @@ def forward(
-        max_seqlen = lengths.max()
+        # FlashAttention needs a host integer.  Compute it once per MoonViT
+        # forward and pass it to every encoder block instead of synchronizing
+        # once per block inside the attention backend.
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +5/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/attention/test_vision_max_seqlen.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30869 - fix: fix Kimi-VL encoder parallelism

- 链接: https://github.com/sgl-project/sglang/pull/30869
- 状态/时间: merged / 2026-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `test/registered/unit/models/test_kimi_vl_moonvit.py`；关联提交 `33f83011e0b7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+596/-71，可读 patch 992 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: fix Kimi-VL encoder parallelism」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `test/registered/unit/models/test_kimi_vl_moonvit.py`；技术摘要: 覆盖「fix: fix Kimi-VL encoder parallelism」；主要实现面是 `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `test/registered/unit/models/test_kimi_vl_moonvit.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_vl_moonvit.py` modified +175/-32 (207 lines); hunks: -61,11 +61,19; -74,6 +82,7 @@ def multihead_attention(; symbols: multihead_attention, sdpa_attention, __init__，涉及 `multihead_attention, sdpa_attention, __init__`；`test/registered/unit/models/test_kimi_vl.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _VisionTower, __init__, __call__, _Projector，涉及 `_VisionTower, __init__, __call__`；`test/registered/unit/models/test_kimi_vl_moonvit.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: test_learnable_2d_pos_emb_caches_inference_interpolation, counting_interpolate, test_learnable_2d_pos_emb_does_not_cache_training_interpolation, test_learnable_2d_pos_emb_evicts_oldest_inference_cache_entry，涉及 `test_learnable_2d_pos_emb_caches_inference_interpolation, counting_interpolate, test_learnable_2d_pos_emb_does_not_cache_training_interpolation`；`python/sglang/srt/models/kimi_vl.py` modified +30/-11 (41 lines); hunks: -43,14 +43,14; -73,6 +73,8; symbols: __init__, get_image_feature, pad_input_ids, load_weights，涉及 `__init__, get_image_feature, pad_input_ids`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_vl_moonvit.py` modified +175/-32 (207 lines); hunks: -61,11 +61,19; -74,6 +82,7 @@ def multihead_attention(; symbols: multihead_attention, sdpa_attention, __init__
  - `test/registered/unit/models/test_kimi_vl.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _VisionTower, __init__, __call__, _Projector
  - `test/registered/unit/models/test_kimi_vl_moonvit.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: test_learnable_2d_pos_emb_caches_inference_interpolation, counting_interpolate, test_learnable_2d_pos_emb_does_not_cache_training_interpolation, test_learnable_2d_pos_emb_evicts_oldest_inference_cache_entry
  - `python/sglang/srt/models/kimi_vl.py` modified +30/-11 (41 lines); hunks: -43,14 +43,14; -73,6 +73,8; symbols: __init__, get_image_feature, pad_input_ids, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_vl_moonvit.py
@@ -61,11 +61,19 @@
-from sglang.srt.layers.linear import ReplicatedLinear
+from sglang.srt.layers.linear import (
+    ColumnParallelLinear,
+    QKVParallelLinear,
+    ReplicatedLinear,
+    RowParallelLinear,
diff -- test/registered/unit/models/test_kimi_vl.py
@@ -0,0 +1,159 @@
+"""CPU-only coverage for Kimi-VL encoder parallelism wiring."""
+from types import SimpleNamespace
+from unittest.mock import patch
+import pytest
+import torch
+import torch.nn as nn
diff -- test/registered/unit/models/test_kimi_vl_moonvit.py
@@ -0,0 +1,64 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_vl_moonvit.py` modified +175/-32; `python/sglang/srt/models/kimi_vl.py` modified +30/-11
  - tests: `test/registered/unit/models/test_kimi_vl.py` added +159/-0; `test/registered/unit/models/test_kimi_vl_moonvit.py` added +64/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_kimi_vl.py`, `test/registered/unit/models/test_kimi_vl_moonvit.py`, `test/registered/unit/multimodal/test_vit_cuda_graph_runner.py`, `test/registered/unit/utils/test_hf_transformers.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31227 - perf: shard Kimi DP image feature transport

- 链接: https://github.com/sgl-project/sglang/pull/31227
- 状态/时间: merged / 2026-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `test/registered/unit/models/test_kimi_k25.py`；关联提交 `7d0fd5101d04`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+972/-99，可读 patch 1452 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf: shard Kimi DP image feature transport」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「perf: shard Kimi DP image feature transport」；主要实现面是 `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_kimi_k25.py` added +241/-0 (241 lines); hunks: -0,0 +1,241; symbols: _MoonViT3dTower, __init__, __call__, _Projector，涉及 `_MoonViT3dTower, __init__, __call__`；`python/sglang/srt/multimodal/processors/kimi_k25.py` modified +68/-11 (79 lines); hunks: -19,6 +19,9; -143,6 +146,49 @@ def _process_single_image(; symbols: _process_single_image, _resize_images_by_source_shape, _gpu_preprocess_images, process_mm_data_async，涉及 `_process_single_image, _resize_images_by_source_shape, _gpu_preprocess_images`；`python/sglang/srt/models/kimi_k25.py` modified +65/-12 (77 lines); hunks: -30,8 +30,11; -142,6 +145,7 @@ def forward(; symbols: forward, __setattr__，涉及 `forward, __setattr__`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_kimi_k25.py` added +241/-0 (241 lines); hunks: -0,0 +1,241; symbols: _MoonViT3dTower, __init__, __call__, _Projector
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +68/-11 (79 lines); hunks: -19,6 +19,9; -143,6 +146,49 @@ def _process_single_image(; symbols: _process_single_image, _resize_images_by_source_shape, _gpu_preprocess_images, process_mm_data_async
  - `python/sglang/srt/models/kimi_k25.py` modified +65/-12 (77 lines); hunks: -30,8 +30,11; -142,6 +145,7 @@ def forward(; symbols: forward, __setattr__
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_kimi_k25.py
@@ -0,0 +1,241 @@
+"""CPU coverage for Kimi-K2.5/K2.7 encoder-DP wiring."""
+from types import SimpleNamespace
+from unittest.mock import Mock, patch
+import pytest
+import torch
+import torch.nn as nn
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -19,6 +19,9 @@
+from sglang.srt.utils.cuda_ipc_transport_utils import (
+    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
+)
@@ -143,6 +146,49 @@ def _process_single_image(
+def _resize_images_by_source_shape(
+    indexed_images: list[tuple[int, torch.Tensor]],
diff -- python/sglang/srt/models/kimi_k25.py
@@ -30,8 +30,11 @@
```

- 已读文件:
  - tests: `test/registered/unit/models/test_kimi_k25.py` added +241/-0
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +68/-11; `python/sglang/srt/models/kimi_k25.py` modified +65/-12
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/attention/test_vision_max_seqlen.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/multimodal/test_cuda_ipc_pool_budget.py`, `test/registered/unit/multimodal/test_cuda_ipc_transport.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21741 - [1/N] feat: support compressed-tensors w4afp8 MoE

- 链接: https://github.com/sgl-project/sglang/pull/21741
- 状态/时间: closed / 2026-07-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1657/-37，可读 patch 1828 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] feat: support compressed-tensors w4afp8 MoE」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；技术摘要: 覆盖「[1/N] feat: support compressed-tensors w4afp8 MoE」；主要实现面是 `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #31514 - [DCP] Enable decode context parallel for Kimi K2.5 NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/31514
- 状态/时间: merged / 2026-07-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`；关联提交 `ead703815ef3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+30/-1，可读 patch 45 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DCP] Enable decode context parallel for Kimi K2.5 NVFP4」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[DCP] Enable decode context parallel for Kimi K2.5 NVFP4」；主要实现面是 `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,6 +786,35 @@ def routed_experts_weights_of_layer(self):; symbols: routed_experts_weights_of_layer, prepare_context_parallel_metadata_for_dcp, forward，涉及 `routed_experts_weights_of_layer, prepare_context_parallel_metadata_for_dcp, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: -786,6 +786,35 @@ def routed_experts_weights_of_layer(self):; symbols: routed_experts_weights_of_layer, prepare_context_parallel_metadata_for_dcp, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -786,6 +786,35 @@ def routed_experts_weights_of_layer(self):
+    def prepare_context_parallel_metadata_for_dcp(
+        self,
+        seq_lens: torch.Tensor,
+        extend_prefix_lens: torch.Tensor,
+        extend_prefix_lens_cpu: torch.Tensor,
+        extend_seq_lens: torch.Tensor,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/dcp/comm.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31474 - Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear

- 链接: https://github.com/sgl-project/sglang/pull/31474
- 状态/时间: merged / 2026-07-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_kimi_linear_models.py`；关联提交 `a03ca46a2847`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+153/-19，可读 patch 453 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/models_e2e/test_kimi_linear_models.py`, `python/sglang/srt/layers/attention/linear/kda_backend.py`, `python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py`；技术摘要: 覆盖「Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear」；主要实现面是 `test/registered/models_e2e/test_kimi_linear_models.py`, `python/sglang/srt/layers/attention/linear/kda_backend.py`, `python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_kimi_linear_models.py` modified +35/-2 (37 lines); hunks: -3,21 +3,27; -45,5 +51,32 @@ def test_gsm8k(self):; symbols: TestKimiLinear, setUpClass, test_gsm8k, TestKimiLinearExtraBuffer，涉及 `TestKimiLinear, setUpClass, test_gsm8k`；`python/sglang/srt/layers/attention/linear/kda_backend.py` modified +43/-2 (45 lines); hunks: -254,6 +254,12 @@ class KDAAttnBackend(MambaAttnBackendBase):; -274,9 +280,22 @@ def __init__(self, model_runner: ModelRunner):; symbols: KDAAttnBackend, __init__, init_forward_metadata, forward_decode，涉及 `KDAAttnBackend, __init__, init_forward_metadata`；`python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py` modified +7/-2 (9 lines); hunks: -39,13 +39,15 @@ def _triton_fallback(; -62,6 +64,7 @@ def _triton_fallback(; symbols: _triton_fallback, extend，涉及 `_triton_fallback, extend`；`python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py` modified +7/-0 (7 lines); hunks: -105,6 +105,13 @@ def extend(; symbols: extend，涉及 `extend`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_kimi_linear_models.py` modified +35/-2 (37 lines); hunks: -3,21 +3,27; -45,5 +51,32 @@ def test_gsm8k(self):; symbols: TestKimiLinear, setUpClass, test_gsm8k, TestKimiLinearExtraBuffer
  - `python/sglang/srt/layers/attention/linear/kda_backend.py` modified +43/-2 (45 lines); hunks: -254,6 +254,12 @@ class KDAAttnBackend(MambaAttnBackendBase):; -274,9 +280,22 @@ def __init__(self, model_runner: ModelRunner):; symbols: KDAAttnBackend, __init__, init_forward_metadata, forward_decode
  - `python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py` modified +7/-2 (9 lines); hunks: -39,13 +39,15 @@ def _triton_fallback(; -62,6 +64,7 @@ def _triton_fallback(; symbols: _triton_fallback, extend
  - `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py` modified +7/-0 (7 lines); hunks: -105,6 +105,13 @@ def extend(; symbols: extend
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +4/-0 (4 lines); hunks: -820,6 +820,10 @@ def __init__(; symbols: __init__, data_type, _is_full_attn
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_kimi_linear_models.py
@@ -3,21 +3,27 @@
+from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
+from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
+from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
+from sglang.test.server_fixtures.default_fixture import DefaultServerBase
-register_cuda_ci(est_time=178, stage="base-b", runner_config="2-gpu-large")
+register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")
diff -- python/sglang/srt/layers/attention/linear/kda_backend.py
@@ -254,6 +254,12 @@ class KDAAttnBackend(MambaAttnBackendBase):
+        # mamba_cache.conv is [..., kernel-1, dim] while conv_states_shape expects the window length (kernel-1) at shape[-1], hence the transpose.
+        self.conv_states_shape = (
+            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
+            .transpose(-1, -2)
+            .shape
+        )
diff -- python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py
@@ -39,13 +39,15 @@ def _triton_fallback(
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_kimi_linear_models.py` modified +35/-2
  - runtime: `python/sglang/srt/layers/attention/linear/kda_backend.py` modified +43/-2; `python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py` modified +7/-2; `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py` modified +7/-0; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +4/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +3/-1; `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py` modified +2/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/kl_divergence_kit.py`, `python/sglang/test/kl_test_utils.py`, `test/registered/models_e2e/test_kimi_linear_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32262 - [Bugfix] Fix Kimi-Linear state transfer across heterogeneous TP

- 链接: https://github.com/sgl-project/sglang/pull/32262
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`, `test/registered/disaggregation/test_disaggregation_kimi_linear.py`；关联提交 `2428f5614561`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+306/-130，可读 patch 769 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Kimi-Linear state transfer across heterogeneous TP」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_linear.py`, `test/registered/disaggregation/test_disaggregation_kimi_linear.py`；技术摘要: 覆盖「[Bugfix] Fix Kimi-Linear state transfer across heterogeneous TP」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`, `test/registered/disaggregation/test_disaggregation_kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +1/-1 (2 lines); hunks: -185,7 +185,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/registered/disaggregation/test_disaggregation_kimi_linear.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiLinearHeterogeneousTPDisaggregation, setUpClass, generate, test_logprob_parity，涉及 `TestKimiLinearHeterogeneousTPDisaggregation, setUpClass, generate`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +1/-1 (2 lines); hunks: -185,7 +185,7 @@ def __init__(; symbols: __init__
  - `test/registered/disaggregation/test_disaggregation_kimi_linear.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiLinearHeterogeneousTPDisaggregation, setUpClass, generate, test_logprob_parity
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -185,7 +185,7 @@ def __init__(
-        self.head_v_dim = config.v_head_dim
+        self.head_v_dim = config.linear_attn_config["head_dim"]
diff -- test/registered/disaggregation/test_disaggregation_kimi_linear.py
@@ -0,0 +1,105 @@
+import time
+import unittest
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.server_fixtures.disaggregation_fixture import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +1/-1
  - tests: `test/registered/disaggregation/test_disaggregation_kimi_linear.py` added +105/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/disaggregation_fixture.py`, `test/registered/disaggregation/test_disaggregation_kimi_linear.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32542 - docs(cookbook): add the Kimi-K3 serving cookbook

- 链接: https://github.com/sgl-project/sglang/pull/32542
- 状态/时间: merged / 2026-07-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+3650/-275，可读 patch 4524 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): add the Kimi-K3 serving cookbook」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`；技术摘要: 覆盖「docs(cookbook): add the Kimi-K3 serving cookbook」；主要实现面是 `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx`, `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` added +1767/-0 (1767 lines); hunks: -0,0 +1,1767；`docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx` added +23/-0 (23 lines); hunks: -0,0 +1,23；`docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ export const config = {；`docs_new/src/snippets/configs/meituan-longcat/longcat-2.0.jsx` modified +1/-1 (2 lines); hunks: -8,7 +8,7 @@ export const config = {。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` added +1767/-0 (1767 lines); hunks: -0,0 +1,1767
  - `docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx` added +23/-0 (23 lines); hunks: -0,0 +1,23
  - `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ export const config = {
  - `docs_new/src/snippets/configs/meituan-longcat/longcat-2.0.jsx` modified +1/-1 (2 lines); hunks: -8,7 +8,7 @@ export const config = {
  - `docs_new/src/snippets/_playground.jsx` modified +351/-182 (533 lines); hunks: -10,14 +10,25; -34,7 +45,60 @@ export const Playground = ({ config }) => {
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -0,0 +1,1767 @@
+// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
+// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
+//
+// Recipes transcribed from the K3 serving benchmark scripts
+// (benchmark/H200/script/v1/launch-k3.sh, benchmark/B300/script/v1/launch-k3.sh)
+// and the B200 2×8 / GB200 4×4 / H100 4×8 / MI35x 1×8 reference launches.
diff -- docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx
@@ -0,0 +1,23 @@
+export const benchmarks = [
+  { match: { hw: "b300",  pdMode: "unified", strategy: "balanced"    } },
+  { match: { hw: "b300",  pdMode: "unified", strategy: "low-latency" } },
+  { match: { hw: "b300",  pdMode: "unified", strategy: "high-throughput"    } },
+  { match: { hw: "b200",  pdMode: "unified", strategy: "low-latency" } },
+  { match: { hw: "b200",  pdMode: "unified", strategy: "balanced"    } },
diff -- docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx
@@ -17,7 +17,7 @@ export const config = {
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` added +1767/-0; `docs_new/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx` added +23/-0; `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` modified +1/-1; `docs_new/src/snippets/configs/meituan-longcat/longcat-2.0.jsx` modified +1/-1; `docs_new/src/snippets/_playground.jsx` modified +351/-182; `docs_new/src/snippets/_deployment.jsx` modified +362/-86
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K2.7-Code.mdx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #32547 - docs: point Kimi-K3 references to public branch

- 链接: https://github.com/sgl-project/sglang/pull/32547
- 状态/时间: merged / 2026-07-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-4，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: point Kimi-K3 references to public branch」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`；技术摘要: 覆盖「docs: point Kimi-K3 references to public branch」；主要实现面是 `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +1/-1 (2 lines); hunks: -5,7 +5,7；`docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +5/-3 (8 lines); hunks: -30,7 +30,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -96,8 +96,8 @@ K3 **always runs with thinking enabled**, with reasoning depth...。
- 代码 diff 细节:
  - `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +1/-1 (2 lines); hunks: -5,7 +5,7
  - `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +5/-3 (8 lines); hunks: -30,7 +30,7 @@ For how to launch the image, see [Install → Method 3: Using Do...; -96,8 +96,8 @@ K3 **always runs with thinking enabled**, with reasoning depth...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -5,7 +5,7 @@
-// experts + 1 shared. Served today from the DarkSharpness/sglang-kimi fork.
+// experts + 1 shared. Served today from the public sgl-project/sglang kimi-k3 branch.
diff -- docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
@@ -30,7 +30,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
-If you do not want to use a Docker image, reproduce the dependency installation steps from the [CUDA 13 Dockerfile](https://github.com/DarkSharpness/sglang-kimi/blob/kimi-k3/docke
+If you do not want to use a Docker image, reproduce the dependency installation steps from the [CUDA 13 Dockerfile](https://github.com/sgl-project/sglang/blob/kimi-k3/docker/kimi_
@@ -96,8 +96,8 @@ K3 **always runs with thinking enabled**, with reasoning depth controlled by `re
-are scheduled to release by July 27, 2026**. The recipes on this page were validated on the
-[`DarkSharpness/sglang-kimi`](https://github.com/DarkSharpness/sglang-kimi) fork — the HuggingFace
+are scheduled to release by July 27, 2026**. The recipes on this page were validated on the public
```

- 已读文件:
  - docs: `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +1/-1; `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +5/-3
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #32612 - Support DCP for Kimi Linear model

- 链接: https://github.com/sgl-project/sglang/pull/32612
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`, `test/registered/dcp/test_kimi_linear_dcp4.py`；关联提交 `ef6c07008b5e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1331/-86，可读 patch 1802 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DCP for Kimi Linear model」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_linear.py`, `test/registered/dcp/test_kimi_linear_dcp4.py`；技术摘要: 覆盖「Support DCP for Kimi Linear model」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`, `test/registered/dcp/test_kimi_linear_dcp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +82/-11 (93 lines); hunks: -16,6 +16,7; -52,6 +53,20; symbols: _get_kda_local_num_heads, _materialize_residual_stream, KimiMoE, __init__，涉及 `_get_kda_local_num_heads, _materialize_residual_stream, KimiMoE`；`test/registered/dcp/test_kimi_linear_dcp4.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: _has_four_blackwell_gpus, TestKimiLinearDCP4, setUpClass, tearDownClass，涉及 `_has_four_blackwell_gpus, TestKimiLinearDCP4, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +82/-11 (93 lines); hunks: -16,6 +16,7; -52,6 +53,20; symbols: _get_kda_local_num_heads, _materialize_residual_stream, KimiMoE, __init__
  - `test/registered/dcp/test_kimi_linear_dcp4.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: _has_four_blackwell_gpus, TestKimiLinearDCP4, setUpClass, tearDownClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -16,6 +16,7 @@
+from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
@@ -52,6 +53,20 @@
+def _get_kda_local_num_heads(num_heads: int, tp_size: int) -> int:
+    if num_heads % tp_size != 0:
+        raise ValueError(
+            f"KDA num_heads ({num_heads}) must be divisible by global tp_size ({tp_size})"
diff -- test/registered/dcp/test_kimi_linear_dcp4.py
@@ -0,0 +1,129 @@
+"""Four-Blackwell acceptance coverage for Kimi Linear TokenSpeed MLA DCP.
+The captured-shape and eager-shape requests deliberately straddle
+``--cuda-graph-max-bs-decode=64``.  This guards both the regular CUDA graph
+decode path and the full-capacity eager DCP LSE scratch-buffer path.
+"""
+import unittest
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +82/-11
  - tests: `test/registered/dcp/test_kimi_linear_dcp4.py` added +129/-0
- 验证与风险: diff 自带测试面 `test/registered/dcp/test_dcp_layout_unit.py`, `test/registered/dcp/test_kimi_linear_dcp4.py`, `test/registered/unit/test_model_overrides.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32118 - Fix nightly CI: NVFP4 cuda-graph crash, NVILA batching, CuTe paged-KV zero-size, Kimi-VL OOM

- 链接: https://github.com/sgl-project/sglang/pull/32118
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_vl.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`；关联提交 `7dcebca25599`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+48/-221，可读 patch 441 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix nightly CI: NVFP4 cuda-graph crash, NVILA batching, CuTe paged-KV zero-size, Kimi-VL OOM」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `python/sglang/srt/models/kimi_vl.py`；技术摘要: 覆盖「Fix nightly CI: NVFP4 cuda-graph crash, NVILA batching, CuTe paged-KV zero-size, Kimi-VL OOM」；主要实现面是 `python/sglang/srt/models/kimi_vl_moonvit.py`, `test/registered/unit/models/test_kimi_vl.py`, `python/sglang/srt/models/kimi_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-198 (228 lines); hunks: -52,18 +52,15; -75,112 +72,6; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope，涉及 `multihead_attention, sdpa_attention, _apply_rope_input_validation`；`test/registered/unit/models/test_kimi_vl.py` modified +5/-18 (23 lines); hunks: -14,9 +14,9; -80,17 +80,16 @@ class TestKimiVLEncoderParallelism(CustomTestCase):; symbols: TestKimiVLEncoderParallelism, test_moonvit_uses_tensor_parallel_layers, test_encoder_dp_tp1_concatenates_moonvit_image_outputs, test_moonvit_attention_accepts_precomputed_max_seqlen，涉及 `TestKimiVLEncoderParallelism, test_moonvit_uses_tensor_parallel_layers, test_encoder_dp_tp1_concatenates_moonvit_image_outputs`；`python/sglang/srt/models/kimi_vl.py` modified +6/-1 (7 lines); hunks: -131,7 +131,6 @@ def __init__(; -263,6 +262,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-198 (228 lines); hunks: -52,18 +52,15; -75,112 +72,6; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `test/registered/unit/models/test_kimi_vl.py` modified +5/-18 (23 lines); hunks: -14,9 +14,9; -80,17 +80,16 @@ class TestKimiVLEncoderParallelism(CustomTestCase):; symbols: TestKimiVLEncoderParallelism, test_moonvit_uses_tensor_parallel_layers, test_encoder_dp_tp1_concatenates_moonvit_image_outputs, test_moonvit_attention_accepts_precomputed_max_seqlen
  - `python/sglang/srt/models/kimi_vl.py` modified +6/-1 (7 lines); hunks: -131,7 +131,6 @@ def __init__(; -263,6 +262,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_vl_moonvit.py
@@ -52,18 +52,15 @@
-from sglang.kernel_api_logging import debug_kernel_api
-try:
-    from flash_attn.flash_attn_interface import flash_attn_varlen_func
-except ImportError:
-    flash_attn_varlen_func = None
+from sglang.srt.layers.attention.vision import (
diff -- test/registered/unit/models/test_kimi_vl.py
@@ -14,9 +14,9 @@
-from sglang.srt.models.kimi_vl_moonvit import MoonVitEncoderLayer, multihead_attention
+from sglang.srt.models.kimi_vl_moonvit import MoonVitEncoderLayer
-from sglang.srt.runtime_context import get_parallel
+from sglang.srt.runtime_context import get_context, get_parallel
@@ -80,17 +80,16 @@ class TestKimiVLEncoderParallelism(CustomTestCase):
-        ):
diff -- python/sglang/srt/models/kimi_vl.py
@@ -131,7 +131,6 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-198; `python/sglang/srt/models/kimi_vl.py` modified +6/-1
  - tests: `test/registered/unit/models/test_kimi_vl.py` modified +5/-18
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_kimi_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32972 - [unified-memory] Let Kimi-Linear use the paged MLA attention backends

- 链接: https://github.com/sgl-project/sglang/pull/32972
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_kimi_linear_unified_memory.py`；关联提交 `33c27d8e7f4f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+759/-11，可读 patch 901 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[unified-memory] Let Kimi-Linear use the paged MLA attention backends」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_kimi_linear_unified_memory.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`；技术摘要: 覆盖「[unified-memory] Let Kimi-Linear use the paged MLA attention backends」；主要实现面是 `test/registered/models_e2e/test_kimi_linear_unified_memory.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_kimi_linear_unified_memory.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestKimiLinearUnifiedMemory，涉及 `TestKimiLinearUnifiedMemory`；`python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +76/-0 (76 lines); hunks: -66,6 +66,51; -675,6 +720,10 @@ def __init__(self, model_runner: ModelRunner, attn_backend:...; symbols: UnifiedMLAHooks, unified_mla_hooks, DecodeMetadata, __init__，涉及 `UnifiedMLAHooks, unified_mla_hooks, DecodeMetadata`；`python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +64/-3 (67 lines); hunks: -34,6 +34,7; -245,6 +246,23 @@ def __init__(; symbols: __init__, _calc_padded_blocks, _create_block_kv_indices, init_cuda_graph_state，涉及 `__init__, _calc_padded_blocks, _create_block_kv_indices`；`python/sglang/srt/server_args.py` modified +24/-4 (28 lines); hunks: -7705,13 +7705,33 @@ def _handle_page_major_kv_layout(self):; symbols: _handle_page_major_kv_layout，涉及 `_handle_page_major_kv_layout`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_kimi_linear_unified_memory.py` added +77/-0 (77 lines); hunks: -0,0 +1,77; symbols: TestKimiLinearUnifiedMemory
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +76/-0 (76 lines); hunks: -66,6 +66,51; -675,6 +720,10 @@ def __init__(self, model_runner: ModelRunner, attn_backend:...; symbols: UnifiedMLAHooks, unified_mla_hooks, DecodeMetadata, __init__
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +64/-3 (67 lines); hunks: -34,6 +34,7; -245,6 +246,23 @@ def __init__(; symbols: __init__, _calc_padded_blocks, _create_block_kv_indices, init_cuda_graph_state
  - `python/sglang/srt/server_args.py` modified +24/-4 (28 lines); hunks: -7705,13 +7705,33 @@ def _handle_page_major_kv_layout(self):; symbols: _handle_page_major_kv_layout
  - `python/sglang/kernels/ops/kvcache/kv_indices.py` modified +15/-1 (16 lines); hunks: -105,6 +105,15 @@ def create_flashmla_kv_indices_triton(; -145,8 +154,13 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_kimi_linear_unified_memory.py
@@ -0,0 +1,77 @@
+"""Kimi-Linear (MLA full attention + KDA linear attention) served from the
+unified memory pool.
+`--enable-unified-memory` replaces the statically-partitioned hybrid pools with
+one byte buffer split dynamically between the full-attention KV sub-pool and the
+per-request KDA state sub-pool. For an MLA model the full side is exposed as
+DENSE per-layer views (`build_dense_mla_views`) and every loc the kernels see is
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -66,6 +66,51 @@
+@dataclass(frozen=True)
+class UnifiedMLAHooks:
+    """Allocator hooks the paged MLA backends need under the unified memory pool.
+    All-``None``/1/``False`` for the statically-partitioned pool, where
+    ``req_to_token`` already holds physical ids.
+    """
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -34,6 +34,7 @@
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_kimi_linear_unified_memory.py` added +77/-0
  - runtime: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +76/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +64/-3; `python/sglang/srt/server_args.py` modified +24/-4; `python/sglang/kernels/ops/kvcache/kv_indices.py` modified +15/-1; `python/sglang/srt/mem_cache/memory_pool.py` modified +8/-3; `python/sglang/srt/mem_cache/multi_ended_allocator.py` modified +9/-0
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_kimi_linear_unified_memory.py`, `test/registered/unit/mem_cache/test_unified_mla_dense_block_table.py`, `test/registered/unit/server_args/test_page_major_backend_allowlist.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32837 - feat: support Kimi Linear PD disaggregation with DCP

- 链接: https://github.com/sgl-project/sglang/pull/32837
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`；关联提交 `2573190b9377`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1084/-48，可读 patch 1566 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: support Kimi Linear PD disaggregation with DCP」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/nixl/conn.py`；技术摘要: 覆盖「feat: support Kimi Linear PD disaggregation with DCP」；主要实现面是 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/nixl/conn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` added +488/-0 (488 lines); hunks: -0,0 +1,488; symbols: _has_eight_blackwell_gpus, TestKimiLinearPDDCP4, setUpClass, _monolithic_reference_args，涉及 `_has_eight_blackwell_gpus, TestKimiLinearPDDCP4, setUpClass`；`python/sglang/srt/disaggregation/mooncake/conn.py` modified +179/-16 (195 lines); hunks: -34,6 +34,7; -45,6 +46,7; symbols: KVArgsRegisterInfo, from_zmq, send_kvcache, send_kvcache_dcp，涉及 `KVArgsRegisterInfo, from_zmq, send_kvcache`；`python/sglang/srt/disaggregation/nixl/conn.py` modified +158/-24 (182 lines); hunks: -32,6 +32,7; -40,6 +41,7; symbols: TransferInfo, is_dummy, from_zmq, KVArgsRegisterInfo，涉及 `TransferInfo, is_dummy, from_zmq`；`python/sglang/srt/disaggregation/common/utils.py` modified +69/-0 (69 lines); hunks: -25,6 +25,7 @@ class TransferKVChunk:; -127,3 +128,71 @@ def group_concurrent_contiguous(; symbols: TransferKVChunk, group_concurrent_contiguous, DCPTokenTransferPlan, build_dcp_token_transfer_plan，涉及 `TransferKVChunk, group_concurrent_contiguous, DCPTokenTransferPlan`。
- 代码 diff 细节:
  - `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` added +488/-0 (488 lines); hunks: -0,0 +1,488; symbols: _has_eight_blackwell_gpus, TestKimiLinearPDDCP4, setUpClass, _monolithic_reference_args
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +179/-16 (195 lines); hunks: -34,6 +34,7; -45,6 +46,7; symbols: KVArgsRegisterInfo, from_zmq, send_kvcache, send_kvcache_dcp
  - `python/sglang/srt/disaggregation/nixl/conn.py` modified +158/-24 (182 lines); hunks: -32,6 +32,7; -40,6 +41,7; symbols: TransferInfo, is_dummy, from_zmq, KVArgsRegisterInfo
  - `python/sglang/srt/disaggregation/common/utils.py` modified +69/-0 (69 lines); hunks: -25,6 +25,7 @@ class TransferKVChunk:; -127,3 +128,71 @@ def group_concurrent_contiguous(; symbols: TransferKVChunk, group_concurrent_contiguous, DCPTokenTransferPlan, build_dcp_token_transfer_plan
  - `python/sglang/srt/disaggregation/common/conn.py` modified +57/-5 (62 lines); hunks: -163,10 +163,13 @@ def __init__(; -268,6 +271,39 @@ def __init__(; symbols: __init__, requires_dcp_relayout, prepare_dcp_token_item_lens, check_status
- 关键代码摘录:

```diff
diff -- test/registered/disaggregation/test_kimi_linear_pd_dcp4.py
@@ -0,0 +1,488 @@
+import math
+import os
+import time
+import unittest
+import requests
+import torch
diff -- python/sglang/srt/disaggregation/mooncake/conn.py
@@ -34,6 +34,7 @@
+    build_dcp_token_transfer_plan,
@@ -45,6 +46,7 @@
+    resolve_dcp_dst_entry_indices,
@@ -131,6 +133,10 @@ class KVArgsRegisterInfo:
+    dst_dcp_size: int = 1
+    dst_dcp_rank: int = 0
diff -- python/sglang/srt/disaggregation/nixl/conn.py
@@ -32,6 +32,7 @@
```

- 已读文件:
  - tests: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` added +488/-0
  - runtime: `python/sglang/srt/disaggregation/mooncake/conn.py` modified +179/-16; `python/sglang/srt/disaggregation/nixl/conn.py` modified +158/-24; `python/sglang/srt/disaggregation/common/utils.py` modified +69/-0; `python/sglang/srt/disaggregation/common/conn.py` modified +57/-5; `python/sglang/srt/arg_groups/pd_disaggregation_hook.py` modified +25/-0; `python/sglang/srt/disaggregation/utils.py` modified +25/-0
- 验证与风险: diff 自带测试面 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`, `test/registered/unit/disaggregation/test_nixl_backend_basic.py`, `test/registered/unit/server_args/test_server_args.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32828 - [Kimi] Support DCP + DSpark (ported from kimi-k3 branch)

- 链接: https://github.com/sgl-project/sglang/pull/32828
- 状态/时间: merged / 2026-08-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`；关联提交 `1496bfee93bc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+445/-17，可读 patch 586 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi] Support DCP + DSpark (ported from kimi-k3 branch)」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/model_executor/pool_configurator.py`；技术摘要: 覆盖「[Kimi] Support DCP + DSpark (ported from kimi-k3 branch)」；主要实现面是 `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/model_executor/pool_configurator.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/dcp/test_kimi_linear_dcp_dspark4.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _has_four_blackwell_gpus, _write_dummy_qwen3_dspark_draft, _wait_for_port_release, TestKimiLinearDCPDSpark4，涉及 `_has_four_blackwell_gpus, _write_dummy_qwen3_dspark_draft, _wait_for_port_release`；`python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +15/-3 (18 lines); hunks: -80,14 +80,21; -133,7 +140,12 @@ def __init__(; symbols: _get_tokenspeed_workspace, __init__，涉及 `_get_tokenspeed_workspace, __init__`；`python/sglang/srt/model_executor/pool_configurator.py` modified +1/-1 (2 lines); hunks: -177,7 +177,7 @@ def __init__(self, kvc: KVCacheConfigurator):; symbols: __init__, _compute_cell_size，涉及 `__init__, _compute_cell_size`；`python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` modified +64/-0 (64 lines); hunks: -5,6 +5,7; -57,6 +58,7; symbols: __init__, alloc_memory_pool, init_attention_backends, init_cuda_graphs，涉及 `__init__, alloc_memory_pool, init_attention_backends`。
- 代码 diff 细节:
  - `test/registered/dcp/test_kimi_linear_dcp_dspark4.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _has_four_blackwell_gpus, _write_dummy_qwen3_dspark_draft, _wait_for_port_release, TestKimiLinearDCPDSpark4
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +15/-3 (18 lines); hunks: -80,14 +80,21; -133,7 +140,12 @@ def __init__(; symbols: _get_tokenspeed_workspace, __init__
  - `python/sglang/srt/model_executor/pool_configurator.py` modified +1/-1 (2 lines); hunks: -177,7 +177,7 @@ def __init__(self, kvc: KVCacheConfigurator):; symbols: __init__, _compute_cell_size
  - `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` modified +64/-0 (64 lines); hunks: -5,6 +5,7; -57,6 +58,7; symbols: __init__, alloc_memory_pool, init_attention_backends, init_cuda_graphs
  - `python/sglang/srt/mem_cache/common.py` modified +6/-13 (19 lines); hunks: -16,12 +16,7; -183,8 +178,8 @@ def release_kv_cache(req: Req, tree_cache: BasePrefixCache,...; symbols: release_kv_cache, _release_overallocated_kv_indices, available_and_evictable_str
- 关键代码摘录:

```diff
diff -- test/registered/dcp/test_kimi_linear_dcp_dspark4.py
@@ -0,0 +1,222 @@
+import json
+import socket
+import tempfile
+import time
+import unittest
+from pathlib import Path
diff -- python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
@@ -80,14 +80,21 @@
-    device: torch.device, num_heads: int, kv_lora_rank: int
+    device: torch.device,
+    num_heads: int,
+    kv_lora_rank: int,
+    max_q_len: int = _TOKENSPEED_MAX_Q_LEN,
+    # DCP target verification gathers Q to the full head count before launching
diff -- python/sglang/srt/model_executor/pool_configurator.py
@@ -177,7 +177,7 @@ def __init__(self, kvc: KVCacheConfigurator):
```

- 已读文件:
  - tests: `test/registered/dcp/test_kimi_linear_dcp_dspark4.py` added +222/-0
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +15/-3; `python/sglang/srt/model_executor/pool_configurator.py` modified +1/-1; `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` modified +64/-0; `python/sglang/srt/mem_cache/common.py` modified +6/-13; `python/sglang/srt/mem_cache/kv_cache_configurator.py` modified +10/-0
- 验证与风险: diff 自带测试面 `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `test/registered/dcp/test_tokenspeed_mla_dcp_metadata.py`, `test/registered/unit/mem_cache/test_paged_free_segment.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32890 - feat(kernels): port standalone Kimi K3 kernels

- 链接: https://github.com/sgl-project/sglang/pull/32890
- 状态/时间: merged / 2026-08-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ag.cuh`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ptx_sys.cuh` 等 23 个文件；关联提交 `fb207b72b02a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 84 个文件，+22871/-155，可读 patch 19934 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(kernels): port standalone Kimi K3 kernels」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型实现调整；主要 diff: `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh`；技术摘要: 覆盖「feat(kernels): port standalone Kimi K3 kernels」；主要实现面是 `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json` added +271/-0 (271 lines); hunks: -0,0 +1,271；`python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json` added +235/-0 (235 lines); hunks: -0,0 +1,235；`python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh` added +1569/-0 (1569 lines); hunks: -0,0 +1,1569; symbols: Major, F16Type, DType, Comm，涉及 `Major, F16Type, DType`；`python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py` added +1132/-0 (1132 lines); hunks: -0,0 +1,1132; symbols: _stream_state, _p2_lanes, _qk_smem, _state_tile_v，涉及 `_stream_state, _p2_lanes, _qk_smem`。
- 代码 diff 细节:
  - `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json` added +271/-0 (271 lines); hunks: -0,0 +1,271
  - `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json` added +235/-0 (235 lines); hunks: -0,0 +1,235
  - `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh` added +1569/-0 (1569 lines); hunks: -0,0 +1,1569; symbols: Major, F16Type, DType, Comm
  - `python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py` added +1132/-0 (1132 lines); hunks: -0,0 +1,1132; symbols: _stream_state, _p2_lanes, _qk_smem, _state_tile_v
  - `python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh` added +946/-0 (946 lines); hunks: -0,0 +1,946
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json
@@ -0,0 +1,271 @@
+{
+  "source": {
+    "code_commits": [
+      "c4ad2e84b",
+      "5bf69bd32"
+    ],
diff -- python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json
@@ -0,0 +1,235 @@
+{
+  "source": {
+    "code_commits": [
+      "8acabdbb7",
+      "e6bbf14d7",
+      "a74213e8c",
diff -- python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh
@@ -0,0 +1,1569 @@
```

- 已读文件:
  - runtime: `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json` added +271/-0; `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json` added +235/-0; `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh` added +1569/-0; `python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py` added +1132/-0; `python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh` added +946/-0; `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` added +908/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_kda_fused_decode.py`, `test/registered/kernels/ops/attention/test_kda_prefill.py`, `test/registered/kernels/ops/kimi_k3/test_collectives.py`, `test/registered/kernels/ops/kimi_k3/test_compute.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33025 - [Kimi K3] Add reasoning, tool-call, and OpenAI serving support

- 链接: https://github.com/sgl-project/sglang/pull/33025
- 状态/时间: merged / 2026-08-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/kimik3_detector.py`, `python/sglang/srt/function_call/kimik3_format.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`, `test/registered/function_call/test_kimik3_detector.py` 等 7 个文件；关联提交 `e2cf21b9e561`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+3439/-198，可读 patch 4734 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K3] Add reasoning, tool-call, and OpenAI serving support」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_detector.py`；技术摘要: 覆盖「[Kimi K3] Add reasoning, tool-call, and OpenAI serving support」；主要实现面是 `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_kimik3_structural_tag.py` added +810/-0 (810 lines); hunks: -0,0 +1,810; symbols: _tool, _argument, _call, _tools_section，涉及 `_tool, _argument, _call`；`python/sglang/srt/function_call/kimik3_structural_tag.py` added +598/-0 (598 lines); hunks: -0,0 +1,598; symbols: _escape_attr, _json_type, _matches_json_type, _resolve_local_ref，涉及 `_escape_attr, _json_type, _matches_json_type`；`python/sglang/srt/function_call/kimik3_detector.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _unescape_attr, _parse_attrs, KimiK3Detector, __init__，涉及 `_unescape_attr, _parse_attrs, KimiK3Detector`；`test/registered/function_call/test_kimik3_detector.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: _make_tool, _call_block, _chunks, _stream，涉及 `_make_tool, _call_block, _chunks`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_kimik3_structural_tag.py` added +810/-0 (810 lines); hunks: -0,0 +1,810; symbols: _tool, _argument, _call, _tools_section
  - `python/sglang/srt/function_call/kimik3_structural_tag.py` added +598/-0 (598 lines); hunks: -0,0 +1,598; symbols: _escape_attr, _json_type, _matches_json_type, _resolve_local_ref
  - `python/sglang/srt/function_call/kimik3_detector.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _unescape_attr, _parse_attrs, KimiK3Detector, __init__
  - `test/registered/function_call/test_kimik3_detector.py` added +221/-0 (221 lines); hunks: -0,0 +1,221; symbols: _make_tool, _call_block, _chunks, _stream
  - `test/registered/unit/parser/test_kimik3_reasoning_parser.py` added +162/-0 (162 lines); hunks: -0,0 +1,162; symbols: _stream, _chunks, test_non_stream_reasoning_channels, test_non_stream_tools_channel_passthrough
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_kimik3_structural_tag.py
@@ -0,0 +1,810 @@
+import json
+import sys
+import pytest
+import xgrammar as xgr
+from xgrammar.testing import _is_grammar_accept_string
+from sglang.srt.entrypoints.openai.protocol import (
diff -- python/sglang/srt/function_call/kimik3_structural_tag.py
@@ -0,0 +1,598 @@
+from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
+from xgrammar import StructuralTag
+from xgrammar.structural_tag import (
+    AnyTextFormat,
+    AnyTokensFormat,
+    ConstStringFormat,
diff -- python/sglang/srt/function_call/kimik3_detector.py
@@ -0,0 +1,234 @@
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_kimik3_structural_tag.py` added +810/-0; `test/registered/function_call/test_kimik3_detector.py` added +221/-0; `test/registered/unit/parser/test_kimik3_reasoning_parser.py` added +162/-0
  - runtime: `python/sglang/srt/function_call/kimik3_structural_tag.py` added +598/-0; `python/sglang/srt/function_call/kimik3_detector.py` added +234/-0; `python/sglang/srt/function_call/kimik3_format.py` added +55/-0; `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-1
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/constrained/test_base_grammar_backend.py`, `test/registered/unit/constrained/test_grammar_manager.py`, `test/registered/unit/constrained/test_reasoner_grammar_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33333 - [AMD][DI][CI] 6/N Add Kimi-K2.6 MXFP4 wide-EP16 2P1D nightly recipes

- 链接: https://github.com/sgl-project/sglang/pull/33333
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml`；关联提交 `a84e70eb1e66`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+633/-63，可读 patch 903 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD][DI][CI] 6/N Add Kimi-K2.6 MXFP4 wide-EP16 2P1D nightly recipes」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml`；技术摘要: 覆盖「[AMD][DI][CI] 6/N Add Kimi-K2.6 MXFP4 wide-EP16 2P1D nightly recipes」；主要实现面是 `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml`, `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml` added +122/-0 (122 lines); hunks: -0,0 +1,122；`scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml` added +109/-0 (109 lines); hunks: -0,0 +1,109。
- 代码 diff 细节:
  - `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml` added +122/-0 (122 lines); hunks: -0,0 +1,122
  - `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml` added +109/-0 (109 lines); hunks: -0,0 +1,109
- 关键代码摘录:

```diff
diff -- scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml
@@ -0,0 +1,122 @@
+# MI355X Kimi-K2.6 (MXFP4 experts, FP8 KV) 4-node 2P1D disaggregation recipe (MTP) — narrow-prefill EP8
+# + wide-decode EP16 (mirrors the DSV4-Pro Oren config: wide EP only helps decode).
+#
+# Two prefill engines (EP8, one node each; the router fans requests across both) +
+# one decode engine (EP16) spanning 2 nodes. Still one logical P/D pair per role
+# group, 4 nodes total. nodes-per-engine = ceil(TP/8): prefill 8->1, decode 16->2,
diff -- scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml
@@ -0,0 +1,109 @@
+# MI355X Kimi-K2.6 (MXFP4 experts, FP8 KV) 4-node 2P1D disaggregation recipe — narrow-prefill EP8
+# + wide-decode EP16 (mirrors the DSV4-Pro Oren config: wide EP only helps decode).
+#
+# Two prefill engines (EP8, one node each; the router fans requests across both) +
+# one decode engine (EP16) spanning 2 nodes. Still one logical P/D pair per role
+# group, 4 nodes total. nodes-per-engine = ceil(TP/8): prefill 8->1, decode 16->2,
```

- 已读文件:
  - other: `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mtp-mxfp4.yaml` added +122/-0; `scripts/ci/slurm/recipes/mi355x-fp8/kimik26/1k1k/2p1d-ep16-mxfp4.yaml` added +109/-0
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #33349 - [Perf] Speed up the Kimi-K2.5 vision path and match PIL bicubic in the GPU resize

- 链接: https://github.com/sgl-project/sglang/pull/33349
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/kimi_vl_moonvit.py`, `python/sglang/srt/multimodal/processors/kimi_common.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_vl.py` 等 7 个文件；关联提交 `afc868517b7f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+760/-180，可读 patch 1368 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Speed up the Kimi-K2.5 vision path and match PIL bicubic in the GPU resize」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k25.py`, `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`；技术摘要: 覆盖「[Perf] Speed up the Kimi-K2.5 vision path and match PIL bicubic in the GPU resize」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k25.py`, `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +219/-110 (329 lines); hunks: -8,6 +8,7; -79,6 +80,34 @@ def _get_image_dimensions(image: Union[torch.Tensor, Image.Im...; symbols: _get_image_dimensions, _expand_image_token_ids, _pil_to_cuda_chw, _ensure_chw_rgb，涉及 `_get_image_dimensions, _expand_image_token_ids, _pil_to_cuda_chw`；`test/registered/unit/models/test_kimi_k25.py` modified +242/-7 (249 lines); hunks: -3,20 +3,32; -63,14 +75,12 @@ def _image_item(feature, grid_thw):; symbols: _image_item, test_kimi_gpu_preprocess_batches_only_source_compatible_images, record_interpolate, test_kimi_resize_tracks_the_checkpoint_processors_pil_bicubic，涉及 `_image_item, test_kimi_gpu_preprocess_batches_only_source_compatible_images, record_interpolate`；`python/sglang/srt/models/kimi_k25.py` modified +55/-56 (111 lines); hunks: -8,6 +8,11; -33,32 +38,46; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__，涉及 `apply_rope, tpool_patch_merger, MoonViTEncoderLayer`；`python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-1 (31 lines); hunks: -1,7 +1,7; -566,6 +566,35 @@ def patch_merger(; symbols: patch_merger, tpool_patch_merger, MoonVitVLProjector, __init__，涉及 `patch_merger, tpool_patch_merger, MoonVitVLProjector`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +219/-110 (329 lines); hunks: -8,6 +8,7; -79,6 +80,34 @@ def _get_image_dimensions(image: Union[torch.Tensor, Image.Im...; symbols: _get_image_dimensions, _expand_image_token_ids, _pil_to_cuda_chw, _ensure_chw_rgb
  - `test/registered/unit/models/test_kimi_k25.py` modified +242/-7 (249 lines); hunks: -3,20 +3,32; -63,14 +75,12 @@ def _image_item(feature, grid_thw):; symbols: _image_item, test_kimi_gpu_preprocess_batches_only_source_compatible_images, record_interpolate, test_kimi_resize_tracks_the_checkpoint_processors_pil_bicubic
  - `python/sglang/srt/models/kimi_k25.py` modified +55/-56 (111 lines); hunks: -8,6 +8,11; -33,32 +38,46; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__
  - `python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-1 (31 lines); hunks: -1,7 +1,7; -566,6 +566,35 @@ def patch_merger(; symbols: patch_merger, tpool_patch_merger, MoonVitVLProjector, __init__
  - `python/sglang/srt/multimodal/processors/kimi_common.py` modified +17/-1 (18 lines); hunks: -3,7 +3,7; -38,6 +38,22 @@ def resolve_image_token_counts(self, images):; symbols: resolve_image_token_counts, count_image_placeholders, _num_image_tokens_from_grid
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k25.py
@@ -8,6 +8,7 @@
+from sglang.kernels.ops.mm.process import normalize_and_patchify
@@ -79,6 +80,34 @@ def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int,
+def _expand_image_token_ids(
+    input_ids: Union[List[int], torch.Tensor],
+    image_token_id: int,
+    image_token_counts: List[int],
diff -- test/registered/unit/models/test_kimi_k25.py
@@ -3,20 +3,32 @@
+import numpy as np
+from PIL import Image
-from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
+from sglang.srt.models.kimi_k25 import (
+    KimiK25ForConditionalGeneration,
+    mm_projection_auto,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -8,6 +8,11 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +219/-110; `python/sglang/srt/models/kimi_k25.py` modified +55/-56; `python/sglang/srt/models/kimi_vl_moonvit.py` modified +30/-1; `python/sglang/srt/multimodal/processors/kimi_common.py` modified +17/-1; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +6/-0
  - tests: `test/registered/unit/models/test_kimi_k25.py` modified +242/-7; `test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py` modified +29/-0
- 验证与风险: diff 自带测试面 `test/manual/vlm/verify_k25_equivalence.py`, `test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py`, `test/registered/unit/models/test_kimi_k25.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32541 - [Kimi] Support kimi-k3

- 链接: https://github.com/sgl-project/sglang/pull/32541
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/srt/arg_groups/kimi_k3_hook.py`, `python/sglang/srt/configs/kimi_k3.py`, `python/sglang/srt/configs/kimi_linear.py`, `python/sglang/srt/models/kimi_k3.py` 等 21 个文件；关联提交 `abddb1c7e9d6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 139 个文件，+15414/-911，可读 patch 16758 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi] Support kimi-k3」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/models/kimi_k3_vl.py`, `test/registered/unit/models/test_kimi_k3_vision.py`；技术摘要: 覆盖「[Kimi] Support kimi-k3」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/models/kimi_k3_vl.py`, `test/registered/unit/models/test_kimi_k3_vision.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` added +3203/-0 (3203 lines)；`python/sglang/srt/models/kimi_k3_vl.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _resolve_grid_thw_list, _get_mm_attention_backend, _is_fa4_available, _resolve_mm_attention_backend，涉及 `_resolve_grid_thw_list, _get_mm_attention_backend, _is_fa4_available`；`test/registered/unit/models/test_kimi_k3_vision.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: test_kimi_k3_resolves_shape_aware_attention_backend, test_kimi_k3_skips_attention_precompile_on_cpu, test_kimi_k3_sdpa_reuses_prepared_segment_bounds, SeqlensThatMustNotSync，涉及 `test_kimi_k3_resolves_shape_aware_attention_backend, test_kimi_k3_skips_attention_precompile_on_cpu, test_kimi_k3_sdpa_reuses_prepared_segment_bounds`；`python/sglang/srt/multimodal/processors/kimi_k3.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _encode_k3_special_tokens, _expand_k3_image_prompt_token_ids, _expand_k3_image_prompt_text, _k3_to_cuda_chw，涉及 `_encode_k3_special_tokens, _expand_k3_image_prompt_token_ids, _expand_k3_image_prompt_text`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` added +3203/-0 (3203 lines)
  - `python/sglang/srt/models/kimi_k3_vl.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _resolve_grid_thw_list, _get_mm_attention_backend, _is_fa4_available, _resolve_mm_attention_backend
  - `test/registered/unit/models/test_kimi_k3_vision.py` added +524/-0 (524 lines); hunks: -0,0 +1,524; symbols: test_kimi_k3_resolves_shape_aware_attention_backend, test_kimi_k3_skips_attention_precompile_on_cpu, test_kimi_k3_sdpa_reuses_prepared_segment_bounds, SeqlensThatMustNotSync
  - `python/sglang/srt/multimodal/processors/kimi_k3.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _encode_k3_special_tokens, _expand_k3_image_prompt_token_ids, _expand_k3_image_prompt_text, _k3_to_cuda_chw
  - `test/registered/unit/models/test_kimi_k25.py` modified +226/-1 (227 lines); hunks: -1,7 +1,8; -24,8 +25,15; symbols: test_kimi_lazy_ipc_feature_acknowledges_all_tp_consumers, _Tokenizer, encode, _HFProcessor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3_vl.py
@@ -0,0 +1,937 @@
+"""Kimi K3 vision tower (MoonViT3d) and projector.
+Faithful port of the checkpoint reference implementation
+(modeling_kimi_k3.py). Dedicated to K3 — do not share with Kimi K2.5:
+K3 uses qkv_hidden_size != hidden_size (head_dim = qkv_hidden_size //
+num_heads), RMSNorm encoder norms, bias-free linears, and the
+PatchMergerMLPV2 projector (no pre-norm, post RMSNorm), all of which
diff -- test/registered/unit/models/test_kimi_k3_vision.py
@@ -0,0 +1,524 @@
+from contextlib import nullcontext
+from types import SimpleNamespace
+import pytest
+import torch
+import torch.nn.functional as F
+from sglang.srt.layers.attention.vision import (
diff -- python/sglang/srt/multimodal/processors/kimi_k3.py
@@ -0,0 +1,438 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` added +3203/-0; `python/sglang/srt/models/kimi_k3_vl.py` added +937/-0; `python/sglang/srt/multimodal/processors/kimi_k3.py` added +438/-0; `python/sglang/srt/multimodal/kimi_k3_vit_cuda_graph_runner.py` added +211/-0; `python/sglang/srt/configs/kimi_k3.py` added +124/-0
  - tests: `test/registered/unit/models/test_kimi_k3_vision.py` added +524/-0; `test/registered/unit/models/test_kimi_k25.py` modified +226/-1; `test/registered/unit/multimodal/test_kimi_k3_gpu_preprocess.py` added +114/-0
- 验证与风险: diff 自带测试面 `test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py`, `test/registered/dcp/test_kimi_linear_dcp4.py`, `test/registered/dcp/test_kimi_linear_dcp_dspark4.py`, `test/registered/kernels/ops/gemm/test_cutedsl_bf16_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33609 - [Tiny] Little enhancement of Kimi-K3 test

- 链接: https://github.com/sgl-project/sglang/pull/33609
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_kimi_k3_b300.py`；关联提交 `6808c6d57165`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tiny] Little enhancement of Kimi-K3 test」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/models_e2e/test_kimi_k3_b300.py`；技术摘要: 覆盖「[Tiny] Little enhancement of Kimi-K3 test」；主要实现面是 `test/registered/models_e2e/test_kimi_k3_b300.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +2/-2 (4 lines); hunks: -16,7 +16,7; -66,6 +66,7 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_kimi_k3_b300.py` modified +2/-2 (4 lines); hunks: -16,7 +16,7; -66,6 +66,7 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_kimi_k3_b300.py
@@ -16,7 +16,7 @@
-register_cuda_ci(est_time=1800, stage="base-c", runner_config="8-gpu-b300")
+register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")
@@ -66,6 +66,7 @@ def setUpClass(cls):
+                "--enable-linear-replayssm-spec",
@@ -94,7 +95,6 @@ def setUpClass(cls):
-                "--disable-custom-all-reduce",
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_kimi_k3_b300.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33612 - Remove custom all-reduce disable from Kimi-K3 B300 recipe

- 链接: https://github.com/sgl-project/sglang/pull/33612
- 状态/时间: merged / 2026-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；关联提交 `87ed82ff7e09`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove custom all-reduce disable from Kimi-K3 B300 recipe」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；技术摘要: 覆盖「Remove custom all-reduce disable from Kimi-K3 B300 recipe」；主要实现面是 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-1 (1 lines); hunks: -817,7 +817,6 @@ export const config = {。
- 代码 diff 细节:
  - `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-1 (1 lines); hunks: -817,7 +817,6 @@ export const config = {
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -817,7 +817,6 @@ export const config = {
-        "--disable-custom-all-reduce",
```

- 已读文件:
  - docs: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #33599 - [AMD] Fuse Kimi-K3 attn-residual aggregation

- 链接: https://github.com/sgl-project/sglang/pull/33599
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py`；关联提交 `1478cdec9fcf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+295/-1，可读 patch 358 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fuse Kimi-K3 attn-residual aggregation」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py`；技术摘要: 覆盖「[AMD] Fuse Kimi-K3 attn-residual aggregation」；主要实现面是 `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: supports_attn_res_hip, _tile_size, _agg_kernel, attn_res_hip，涉及 `supports_attn_res_hip, _tile_size, _agg_kernel`。
- 代码 diff 细节:
  - `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: supports_attn_res_hip, _tile_size, _agg_kernel, attn_res_hip
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/ops/kimi_k3/attn_res_hip.py
@@ -0,0 +1,212 @@
+"""Triton attention-residual aggregation for Kimi-K3 on ROCm.
+The HIP counterpart of attn_res.py: same aggregation point (score the bank rows
+against the current prefix, softmax, weighted sum, output RMSNorm), one launch,
+but built for a GPU with no TMA and no tcgen05. See _agg_kernel for why the
+shape differs so much from the SM100 kernel's.
+"""
```

- 已读文件:
  - runtime: `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py` added +212/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/ops/kimi_k3/attn_res_hip.py`, `python/sglang/srt/layers/attn_residual.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33764 - Fix the router GEMM inaccuracy when using _front_w in Kimi-K3

- 链接: https://github.com/sgl-project/sglang/pull/33764
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/kernels/ops/kimi_k3/activation.py`, `python/sglang/srt/models/kimi_k3.py`；关联提交 `3fbb5330c7e5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+216/-59，可读 patch 579 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix the router GEMM inaccuracy when using _front_w in Kimi-K3」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/kernels/ops/kimi_k3/activation.py`；技术摘要: 覆盖「Fix the router GEMM inaccuracy when using _front_w in Kimi-K3」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/kernels/ops/kimi_k3/activation.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +53/-18 (71 lines); hunks: -137,11 +137,16 @@ def _k3_bf16_gemm(; -155,15 +160,11 @@ def _k3_bf16_gemm(; symbols: _k3_bf16_gemm, __init__, _merge_front_weights, _eligible_for_fused_front，涉及 `_k3_bf16_gemm, __init__, _merge_front_weights`；`python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh` modified +17/-12 (29 lines); hunks: -64,11 +64,13 @@ struct SituAndMulParams {; -94,23 +96,24 @@ __global__ void situ_and_mul_kernel(const __grid_constant__...；`python/sglang/kernels/ops/kimi_k3/activation.py` modified +10/-8 (18 lines); hunks: -32,9 +32,9 @@ def _fast_math_flags() -> list[str]:; -50,22 +50,24 @@ def situ_and_mul(; symbols: _fast_math_flags, _jit_situ_and_mul_module, situ_and_mul，涉及 `_fast_math_flags, _jit_situ_and_mul_module, situ_and_mul`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +53/-18 (71 lines); hunks: -137,11 +137,16 @@ def _k3_bf16_gemm(; -155,15 +160,11 @@ def _k3_bf16_gemm(; symbols: _k3_bf16_gemm, __init__, _merge_front_weights, _eligible_for_fused_front
  - `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh` modified +17/-12 (29 lines); hunks: -64,11 +64,13 @@ struct SituAndMulParams {; -94,23 +96,24 @@ __global__ void situ_and_mul_kernel(const __grid_constant__...
  - `python/sglang/kernels/ops/kimi_k3/activation.py` modified +10/-8 (18 lines); hunks: -32,9 +32,9 @@ def _fast_math_flags() -> list[str]:; -50,22 +50,24 @@ def situ_and_mul(; symbols: _fast_math_flags, _jit_situ_and_mul_module, situ_and_mul
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -137,11 +137,16 @@ def _k3_bf16_gemm(
+    out_dtype: Optional[torch.dtype] = None,
+    if out is None and out_dtype is not None and out_dtype != x.dtype:
+        out = torch.empty(
+            (x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device
+        )
@@ -155,15 +160,11 @@ def _k3_bf16_gemm(
diff -- python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh
@@ -64,11 +64,13 @@ struct SituAndMulParams {
-template <typename T, bool kHasLinearBeta, bool kUsePDL>
+template <typename TIn, typename TOut, bool kHasLinearBeta, bool kUsePDL>
-  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
-  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
+  constexpr auto kWidest = sizeof(TIn) > sizeof(TOut) ? sizeof(TIn) : sizeof(TOut);
+  constexpr auto kVecSize = kMaxVecBytes / kWidest;
diff -- python/sglang/kernels/ops/kimi_k3/activation.py
@@ -32,9 +32,9 @@ def _fast_math_flags() -> list[str]:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +53/-18; `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh` modified +17/-12; `python/sglang/kernels/ops/kimi_k3/activation.py` modified +10/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh`, `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`, `python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34089 - [CI] Add Kimi-K3 low-latency performance check

- 链接: https://github.com/sgl-project/sglang/pull/34089
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_kimi_k3_b300.py`；关联提交 `c2d90db1e311`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Kimi-K3 low-latency performance check」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/models_e2e/test_kimi_k3_b300.py`；技术摘要: 覆盖「[CI] Add Kimi-K3 low-latency performance check」；主要实现面是 `test/registered/models_e2e/test_kimi_k3_b300.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +6/-2 (8 lines); hunks: -1,14 +1,16; -33,11 +35,13 @@ def _stop_server(process):; symbols: _stop_server, TestKimiK3B300LowLatency, setUpClass，涉及 `_stop_server, TestKimiK3B300LowLatency, setUpClass`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_kimi_k3_b300.py` modified +6/-2 (8 lines); hunks: -1,14 +1,16; -33,11 +35,13 @@ def _stop_server(process):; symbols: _stop_server, TestKimiK3B300LowLatency, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_kimi_k3_b300.py
@@ -1,14 +1,16 @@
-eight B300 GPUs. Each server must preserve basic model quality on GSM8K.
+eight B300 GPUs. Each server must preserve basic model quality on GSM8K, and
+the Low Latency recipe must also preserve single-request decode performance.
+from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
@@ -33,11 +35,13 @@ def _stop_server(process):
-class TestKimiK3B300LowLatency(GSM8KMixin, CustomTestCase):
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +6/-2
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_kimi_k3_b300.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34145 - [CI] Gate Kimi-K3 acceptance length on the GSM8K average

- 链接: https://github.com/sgl-project/sglang/pull/34145
- 状态/时间: merged / 2026-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models_e2e/test_kimi_k3_b300.py`；关联提交 `1ebd6fab6c6c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Gate Kimi-K3 acceptance length on the GSM8K average」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/models_e2e/test_kimi_k3_b300.py`；技术摘要: 覆盖「[CI] Gate Kimi-K3 acceptance length on the GSM8K average」；主要实现面是 `test/registered/models_e2e/test_kimi_k3_b300.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +9/-2 (11 lines); hunks: -40,8 +40,15 @@ class TestKimiK3B300LowLatency(GSM8KMixin, SpecDecodingMixin,...; symbols: TestKimiK3B300LowLatency, setUpClass，涉及 `TestKimiK3B300LowLatency, setUpClass`。
- 代码 diff 细节:
  - `test/registered/models_e2e/test_kimi_k3_b300.py` modified +9/-2 (11 lines); hunks: -40,8 +40,15 @@ class TestKimiK3B300LowLatency(GSM8KMixin, SpecDecodingMixin,...; symbols: TestKimiK3B300LowLatency, setUpClass
- 关键代码摘录:

```diff
diff -- test/registered/models_e2e/test_kimi_k3_b300.py
@@ -40,8 +40,15 @@ class TestKimiK3B300LowLatency(GSM8KMixin, SpecDecodingMixin, CustomTestCase):
-    accept_length_thres = 6.6
-    bs_1_speed_thres = 440
+    # Gated on GSM8K rather than on test_bs_1_speed below: a 200-question
+    # average holds steady when a numerics change moves where the single
+    # greedy prompt hits EOS.
+    gsm8k_accept_length_thres = 4.5
```

- 已读文件:
  - tests: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +9/-2
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_kimi_k3_b300.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33921 - [Kimi K3] Preprocess CPU-transport images on the vision owner

- 链接: https://github.com/sgl-project/sglang/pull/33921
- 状态/时间: merged / 2026-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py`；关联提交 `22e003580bc0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+470/-82，可读 patch 684 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K3] Preprocess CPU-transport images on the vision owner」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py`；技术摘要: 覆盖「[Kimi K3] Preprocess CPU-transport images on the vision owner」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +130/-55 (185 lines); hunks: -15,8 +15,21; -28,8 +41,10; symbols: _expand_k3_image_prompt_text, _k3_to_cuda_chw, _chessboard_background, _fill_transparent_bg，涉及 `_expand_k3_image_prompt_text, _k3_to_cuda_chw, _chessboard_background`；`test/registered/unit/models/test_kimi_k25.py` modified +134/-0 (134 lines); hunks: -658,6 +658,138 @@ def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():; -676,6 +808,8 @@ def test_kimi_k3_rejects_silently_dropped_images():; symbols: test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing, test_kimi_k3_defers_only_when_raw_transport_is_smaller, test_kimi_k3_does_not_defer_non_uint8_tensor_preprocessing，涉及 `test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing, test_kimi_k3_defers_only_when_raw_transport_is_smaller`；`test/registered/unit/models/test_kimi_k3_vision.py` modified +78/-6 (84 lines); hunks: -167,7 +167,7 @@ def fail_position_recompute(*_args, **_kwargs):; -424,6 +424,7 @@ def test_flashinfer_cudnn_metadata_uses_bucketed_element_ind...; symbols: fail_position_recompute, test_kimi_k3_dp_helper_passes_host_grid_list_to_capable_tower, test_flashinfer_cudnn_metadata_uses_bucketed_element_indptrs, _K3TowerStub，涉及 `fail_position_recompute, test_kimi_k3_dp_helper_passes_host_grid_list_to_capable_tower, test_flashinfer_cudnn_metadata_uses_bucketed_element_indptrs`；`python/sglang/srt/multimodal/kimi_k3_image_processing.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: to_chw_uint8, fill_transparent_bg, normalization_tensors，涉及 `to_chw_uint8, fill_transparent_bg, normalization_tensors`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +130/-55 (185 lines); hunks: -15,8 +15,21; -28,8 +41,10; symbols: _expand_k3_image_prompt_text, _k3_to_cuda_chw, _chessboard_background, _fill_transparent_bg
  - `test/registered/unit/models/test_kimi_k25.py` modified +134/-0 (134 lines); hunks: -658,6 +658,138 @@ def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():; -676,6 +808,8 @@ def test_kimi_k3_rejects_silently_dropped_images():; symbols: test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing, test_kimi_k3_defers_only_when_raw_transport_is_smaller, test_kimi_k3_does_not_defer_non_uint8_tensor_preprocessing
  - `test/registered/unit/models/test_kimi_k3_vision.py` modified +78/-6 (84 lines); hunks: -167,7 +167,7 @@ def fail_position_recompute(*_args, **_kwargs):; -424,6 +424,7 @@ def test_flashinfer_cudnn_metadata_uses_bucketed_element_ind...; symbols: fail_position_recompute, test_kimi_k3_dp_helper_passes_host_grid_list_to_capable_tower, test_flashinfer_cudnn_metadata_uses_bucketed_element_indptrs, _K3TowerStub
  - `python/sglang/srt/multimodal/kimi_k3_image_processing.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: to_chw_uint8, fill_transparent_bg, normalization_tensors
  - `python/sglang/srt/models/kimi_k3.py` modified +46/-21 (67 lines); hunks: -104,6 +104,12; -3038,40 +3044,59 @@ def get_image_feature(self, items: List[MultimodalDataIt...; symbols: get_image_feature, materialize_item_features
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k3.py
@@ -15,8 +15,21 @@
-from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalProcessorOutput,
+)
diff -- test/registered/unit/models/test_kimi_k25.py
@@ -658,6 +658,138 @@ def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():
+def test_kimi_k3_cpu_transport_defers_gpu_preprocessing():
+    from sglang.srt.multimodal.kimi_k3_image_processing import (
+        DEFERRED_PREPROCESSING_KEY,
+    )
+    processor = object.__new__(KimiK3ImageProcessor)
+    processor.mm_tokens = SimpleNamespace(image_token_id=99)
diff -- test/registered/unit/models/test_kimi_k3_vision.py
@@ -167,7 +167,7 @@ def fail_position_recompute(*_args, **_kwargs):
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +130/-55; `python/sglang/srt/multimodal/kimi_k3_image_processing.py` added +81/-0; `python/sglang/srt/models/kimi_k3.py` modified +46/-21
  - tests: `test/registered/unit/models/test_kimi_k25.py` modified +134/-0; `test/registered/unit/models/test_kimi_k3_vision.py` modified +78/-6
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19703 - [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT

- 链接: https://github.com/sgl-project/sglang/pull/19703
- 状态/时间: closed / 2026-08-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+576/-1，可读 patch 588 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`；技术摘要: 覆盖「[JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT」；主要实现面是 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #34163 - fix(vlm): preserve Kimi-K3 GPU JPEG accuracy

- 链接: https://github.com/sgl-project/sglang/pull/34163
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`；关联提交 `c20e99bd22d8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+258/-9，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(vlm): preserve Kimi-K3 GPU JPEG accuracy」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`；技术摘要: 覆盖「fix(vlm): preserve Kimi-K3 GPU JPEG accuracy」；主要实现面是 `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +4/-1 (5 lines); hunks: -282,7 +282,10 @@ def prepare_deferred(self, text, images, original_input_ids...; symbols: prepare_deferred, KimiK3ImageProcessor，涉及 `prepare_deferred, KimiK3ImageProcessor`；`test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` modified +22/-1 (23 lines); hunks: -6,7 +6,7; -138,6 +138,27 @@ def image_processor(*, images, **kwargs):; symbols: image_processor, test_kimi_k3_epd_selects_matching_jpeg_decode_mode, test_kimi_k3_epd_aggregates_original_image_sizes_in_part_order，涉及 `image_processor, test_kimi_k3_epd_selects_matching_jpeg_decode_mode, test_kimi_k3_epd_aggregates_original_image_sizes_in_part_order`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +4/-1 (5 lines); hunks: -282,7 +282,10 @@ def prepare_deferred(self, text, images, original_input_ids...; symbols: prepare_deferred, KimiK3ImageProcessor
  - `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` modified +22/-1 (23 lines); hunks: -6,7 +6,7; -138,6 +138,27 @@ def image_processor(*, images, **kwargs):; symbols: image_processor, test_kimi_k3_epd_selects_matching_jpeg_decode_mode, test_kimi_k3_epd_aggregates_original_image_sizes_in_part_order
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/kimi_k3.py
@@ -282,7 +282,10 @@ def prepare_deferred(self, text, images, original_input_ids=None):
-    gpu_image_decode = True
+    # K3 accuracy is sensitive to the chroma upsampling used for common 4:2:0
+    # JPEG inputs. This mode uses interpolated nvJPEG upsampling when the K3
+    # image dependency is installed and otherwise falls back to PIL.
+    gpu_image_decode = "nvjpeg_fancy"
diff -- test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py
@@ -6,7 +6,7 @@
-from unittest.mock import AsyncMock
+from unittest.mock import AsyncMock, patch
@@ -138,6 +138,27 @@ def image_processor(*, images, **kwargs):
+@pytest.mark.parametrize(
+    ("use_image_processor_gpu", "expected_decode_mode"),
+    [(False, False), (True, "nvjpeg_fancy")],
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +4/-1
  - tests: `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` modified +22/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`, `test/registered/unit/multimodal/test_base_processor_image_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22208 - [AMD] Optimize fused MoE kernel config for small-M decode on gfx950

- 链接: https://github.com/sgl-project/sglang/pull/22208
- 状态/时间: closed / 2026-08-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-6，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize fused MoE kernel config for small-M decode on gfx950」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；技术摘要: 覆盖「[AMD] Optimize fused MoE kernel config for small-M decode on gfx950」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #31105 - [ROCm/gfx95] Fix fp8 per-channel attention for Kimi-K2.7-code-mxfp4 o…

- 链接: https://github.com/sgl-project/sglang/pull/31105
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py`；关联提交 `e74ea5b1d709`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+314/-15，可读 patch 443 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm/gfx95] Fix fp8 per-channel attention for Kimi-K2.7-code-mxfp4 o…」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`；技术摘要: 覆盖「[ROCm/gfx95] Fix fp8 per-channel attention for Kimi-K2.7-code-mxfp4 o…」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: ModelConfig, __post_init__, get_display_name, get_kimi_k27_code_mxfp4_models，涉及 `ModelConfig, __post_init__, get_display_name`；`python/sglang/srt/models/deepseek_v2.py` modified +24/-5 (29 lines); hunks: -179,6 +179,7; -2402,17 +2403,35 @@ def __init__(; symbols: __init__, _detect_gfx95_quant_format, _resolve_gfx95_quant_format, _is_layer_sparse，涉及 `__init__, _detect_gfx95_quant_format, _resolve_gfx95_quant_format`；`python/sglang/srt/models/deepseek_common/utils.py` modified +17/-0 (17 lines); hunks: -72,6 +72,23; symbols: _is_block_scale_fp8, awq_dequantize_func，涉及 `_is_block_scale_fp8, awq_dequantize_func`；`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py` modified +5/-7 (12 lines); hunks: -28,6 +28,7; -74,10 +75,7 @@ def forward_normal_rocm_prepare(; symbols: forward_normal_rocm_prepare，涉及 `forward_normal_rocm_prepare`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py` added +184/-0 (184 lines); hunks: -0,0 +1,184; symbols: ModelConfig, __post_init__, get_display_name, get_kimi_k27_code_mxfp4_models
  - `python/sglang/srt/models/deepseek_v2.py` modified +24/-5 (29 lines); hunks: -179,6 +179,7; -2402,17 +2403,35 @@ def __init__(; symbols: __init__, _detect_gfx95_quant_format, _resolve_gfx95_quant_format, _is_layer_sparse
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +17/-0 (17 lines); hunks: -72,6 +72,23; symbols: _is_block_scale_fp8, awq_dequantize_func
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py` modified +5/-7 (12 lines); hunks: -28,6 +28,7; -74,10 +75,7 @@ def forward_normal_rocm_prepare(; symbols: forward_normal_rocm_prepare
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` modified +4/-3 (7 lines); hunks: -53,6 +53,7; -224,7 +225,7 @@ def rocm_absorb_v_bmm(; symbols: rocm_absorb_v_bmm, forward_absorb_rocm_prepare
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py
@@ -0,0 +1,184 @@
+"""MI35x Kimi-K2.7-Code-MXFP4 aiter MLA backend accuracy tests (4-GPU)
+Tests Kimi-K2.7-Code-MXFP4 with the aiter unified attention backend on MI35x.
+This model uses mixed quantization: mxfp4 for MoE layers and fp8 per-channel
+for attention projections (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj,
+o_proj). The per-channel fp8 detection fix ensures the correct kernel path is
+selected for each layer type.
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -179,6 +179,7 @@
+    _is_block_scale_fp8,
@@ -2402,17 +2403,35 @@ def __init__(
-        weight = getattr(
-            getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None), "weight", None
-        )
+        proj = getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
diff -- python/sglang/srt/models/deepseek_common/utils.py
@@ -72,6 +72,23 @@
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py` added +184/-0
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +24/-5; `python/sglang/srt/models/deepseek_common/utils.py` modified +17/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py` modified +5/-7; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` modified +4/-3; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +5/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py`, `test/registered/amd/test_fp8_per_channel_detection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34444 - docs(cookbook): Kimi-K3 — drop --enable-symm-mem from the GB cells

- 链接: https://github.com/sgl-project/sglang/pull/34444
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；关联提交 `59450c4f186f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-4，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs(cookbook): Kimi-K3 — drop --enable-symm-mem from the GB cells」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；技术摘要: 覆盖「docs(cookbook): Kimi-K3 — drop --enable-symm-mem from the GB cells」；主要实现面是 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-4 (4 lines); hunks: -1231,7 +1231,6 @@ export const config = {; -1296,7 +1295,6 @@ export const config = {。
- 代码 diff 细节:
  - `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-4 (4 lines); hunks: -1231,7 +1231,6 @@ export const config = {; -1296,7 +1295,6 @@ export const config = {
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -1231,7 +1231,6 @@ export const config = {
-        "--enable-symm-mem",
@@ -1296,7 +1295,6 @@ export const config = {
-        "--enable-symm-mem",
@@ -1425,7 +1423,6 @@ export const config = {
-        "--enable-symm-mem",
@@ -2006,7 +2003,6 @@ export const config = {
```

- 已读文件:
  - docs: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +0/-4
- 验证与风险: 该 PR 主要落在文档/示例 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #33997 - Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds

- 链接: https://github.com/sgl-project/sglang/pull/33997
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`；关联提交 `00e57d74f07b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+84/-6496，可读 patch 1315 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`；技术摘要: 覆盖「Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds」；主要实现面是 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +3/-7 (10 lines); hunks: -458,10 +458,8 @@ export const config = {; -931,10 +929,8 @@ export const config = {；`docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +2/-11 (13 lines); hunks: -127,16 +127,7 @@ Capacity levers, all in the Playground. Each trades precisi...; -395,7 +386,7 @@ Both presets are one click away in the Playground above: pic...；`python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ __global__ __launch_bounds__(1024, 1) void all_reduce_push_r...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +3/-7 (10 lines); hunks: -458,10 +458,8 @@ export const config = {; -931,10 +929,8 @@ export const config = {
  - `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +2/-11 (13 lines); hunks: -127,16 +127,7 @@ Capacity levers, all in the Playground. Each trades precisi...; -395,7 +386,7 @@ Both presets are one click away in the Playground above: pic...
  - `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ __global__ __launch_bounds__(1024, 1) void all_reduce_push_r...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -458,10 +458,8 @@ export const config = {
-          // Blackwell-only: runs the prebuilt trtllm-gen SiTU cubins; needs the
-          // downloadable SiTU cubin pool unpacked and pointed to by the env var.
+          // Blackwell-only: runs FlashInfer's official trtllm-gen SiTU kernels.
-            env: ["SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=/path/to/trtllm_gen_moe_cubin_pool"],
@@ -931,10 +929,8 @@ export const config = {
-        // Both pinned to the brought-up shape rather than left to the auto
diff -- docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
@@ -127,16 +127,7 @@ Capacity levers, all in the Playground. Each trades precision or cache behavior
-**MoE runner.** Leave `--moe-runner-backend` unset on Blackwell and it resolves to FlashInfer MXFP4 (W4A8, prebuilt trtllm-gen SiTU kernels) when the cubin pool is installed, Marl
-'''bash
-wget https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/trtllm_gen_moe_cubin_pool_20260617_v0613rc1.zip
-sudo mkdir -p /opt/trtllm_gen_moe_cubin_pool
-sudo unzip -q trtllm_gen_moe_cubin_pool_20260617_v0613rc1.zip -d /opt/trtllm_gen_moe_cubin_pool
-export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=/opt/trtllm_gen_moe_cubin_pool/trtllm_gen_moe_cubin_pool_20260617_v0613rc1
diff -- python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh
@@ -152,7 +152,7 @@ __global__ __launch_bounds__(1024, 1) void all_reduce_push_res_kernel(const __gr
```

- 已读文件:
  - docs: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +3/-7; `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +2/-11
  - runtime: `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/sampling/test_penalty.py`, `test/registered/unit/mem_cache/test_unified_mamba_views.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33465 - [Kimi-K3][NPU] Support Kimi-K3 on NPU

- 链接: https://github.com/sgl-project/sglang/pull/33465
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/kimi_k3/__init__.py`, `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py`, `python/sglang/srt/models/kimi_k3.py`；关联提交 `197832bcf536`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+1294/-129，可读 patch 2156 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-K3][NPU] Support Kimi-K3 on NPU」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/ops/kimi_k3/__init__.py`, `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py`；技术摘要: 覆盖「[Kimi-K3][NPU] Support Kimi-K3 on NPU」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/kernels/ops/kimi_k3/__init__.py`, `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +146/-29 (175 lines); hunks: -38,6 +38,8; -117,7 +119,7; symbols: _cdiv, _merge_weights_as_views, _dp_local_buffer_group, __init__，涉及 `_cdiv, _merge_weights_as_views, _dp_local_buffer_group`；`python/sglang/kernels/ops/kimi_k3/__init__.py` modified +11/-6 (17 lines); hunks: -2,9 +2,13; -65,12 +69,13 @@ def kimi_k3_tiny_gemm(; symbols: kimi_k3_tiny_gemm，涉及 `kimi_k3_tiny_gemm`；`python/sglang/kernels/ops/kimi_k3/mla_output_gate.py` modified +4/-1 (5 lines); hunks: -12,11 +12,13; -33,7 +35,8 @@ def _jit_mla_output_gate_module() -> Module:; symbols: _jit_mla_output_gate_module, covered，涉及 `_jit_mla_output_gate_module, covered`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +146/-29 (175 lines); hunks: -38,6 +38,8; -117,7 +119,7; symbols: _cdiv, _merge_weights_as_views, _dp_local_buffer_group, __init__
  - `python/sglang/kernels/ops/kimi_k3/__init__.py` modified +11/-6 (17 lines); hunks: -2,9 +2,13; -65,12 +69,13 @@ def kimi_k3_tiny_gemm(; symbols: kimi_k3_tiny_gemm
  - `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py` modified +4/-1 (5 lines); hunks: -12,11 +12,13; -33,7 +35,8 @@ def _jit_mla_output_gate_module() -> Module:; symbols: _jit_mla_output_gate_module, covered
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -38,6 +38,8 @@
+    attn_tp_all_gather_into_tensor,
+    attn_tp_reduce_scatter_tensor,
@@ -117,7 +119,7 @@
-from sglang.srt.utils import is_blackwell_supported, is_hip, make_layers
+from sglang.srt.utils import is_blackwell_supported, is_hip, is_npu, make_layers
@@ -129,7 +131,10 @@
diff -- python/sglang/kernels/ops/kimi_k3/__init__.py
@@ -2,9 +2,13 @@
+from sglang.srt.utils import is_npu
+_is_npu = is_npu()
@@ -65,12 +69,13 @@ def kimi_k3_tiny_gemm(
-    if max_num_tokens := _K3_N_GEMM_DISPATCH_MAP.get((n, k)):
-        if 0 < m <= max_num_tokens:
-            return tiny_n_gemm_bf16(x, w)
diff -- python/sglang/kernels/ops/kimi_k3/mla_output_gate.py
@@ -12,11 +12,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +146/-29; `python/sglang/kernels/ops/kimi_k3/__init__.py` modified +11/-6; `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/ops/elementwise/add3.py`, `python/sglang/kernels/ops/kimi_k3/__init__.py`, `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33623 - [Kimi K3] Fuse MLA gate projection into QKV-A GEMM

- 链接: https://github.com/sgl-project/sglang/pull/33623
- 状态/时间: merged / 2026-08-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`；关联提交 `b501311fa147`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+181/-4，可读 patch 249 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi K3] Fuse MLA gate projection into QKV-A GEMM」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k3.py`；技术摘要: 覆盖「[Kimi K3] Fuse MLA gate projection into QKV-A GEMM」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +62/-4 (66 lines); hunks: -15,6 +15,10; -71,6 +75,7; symbols: _symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights，涉及 `_symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +62/-4 (66 lines); hunks: -15,6 +15,10; -71,6 +75,7; symbols: _symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -15,6 +15,10 @@
+from sglang.kernels.ops.gemm.fused_a_gemm import (
+    dsv3_fused_a_gemm,
+    fused_a_gemm_weight_eligible,
+)
@@ -71,6 +75,7 @@
+from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +62/-4
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_kimi_k3_mla_gate_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34642 - Revert "[Kimi K3] Fuse MLA gate projection into QKV-A GEMM"

- 链接: https://github.com/sgl-project/sglang/pull/34642
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`；关联提交 `aefe2d0207f0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-181，可读 patch 249 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[Kimi K3] Fuse MLA gate projection into QKV-A GEMM"」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/kimi_k3.py`；技术摘要: 覆盖「Revert "[Kimi K3] Fuse MLA gate projection into QKV-A GEMM"」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +4/-62 (66 lines); hunks: -15,10 +15,6; -77,7 +73,6; symbols: _symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights，涉及 `_symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +4/-62 (66 lines); hunks: -15,10 +15,6; -77,7 +73,6; symbols: _symm_o_proj_forward, _gated_o_proj_forward, _merge_qkv_a_g_proj_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -15,10 +15,6 @@
-from sglang.kernels.ops.gemm.fused_a_gemm import (
-    dsv3_fused_a_gemm,
-    fused_a_gemm_weight_eligible,
-)
@@ -77,7 +73,6 @@
-from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +4/-62
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_kimi_k3_mla_gate_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34766 - [Fix] Carry the backend on Kimi-K3 deferred preprocessing configs

- 链接: https://github.com/sgl-project/sglang/pull/34766
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`, `test/registered/unit/models/test_kimi_k25.py` 等 6 个文件；关联提交 `8554d9a5bc9d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+80/-51，可读 patch 284 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Carry the backend on Kimi-K3 deferred preprocessing configs」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`；技术摘要: 覆盖「[Fix] Carry the backend on Kimi-K3 deferred preprocessing configs」；主要实现面是 `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/models/test_kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/kimi_k3_image_processing.py` modified +28/-15 (43 lines); hunks: -1,6 +1,7; -9,6 +10,22; symbols: KimiK3DeferredPreprocessing, prepare_kimi_k3_encoder_inputs, materialize_kimi_k3_cpu_features，涉及 `KimiK3DeferredPreprocessing, prepare_kimi_k3_encoder_inputs, materialize_kimi_k3_cpu_features`；`python/sglang/srt/multimodal/processors/kimi_k3.py` modified +20/-11 (31 lines); hunks: -8,6 +8,7; -23,6 +24,7; symbols: prepare_deferred, KimiK3ImageProcessor, _should_defer_gpu_preprocessing, _build_deferred_output，涉及 `prepare_deferred, KimiK3ImageProcessor, _should_defer_gpu_preprocessing`；`test/registered/unit/models/test_kimi_k25.py` modified +16/-9 (25 lines); hunks: -1,6 +1,7; -690,6 +691,7 @@ def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():; symbols: test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing，涉及 `test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing`；`test/registered/unit/models/test_kimi_k3_vision.py` modified +8/-8 (16 lines); hunks: -509,6 +509,7 @@ def test_kimi_k3_preprocesses_only_dp_owner_images(monkeypat...; -517,20 +518,19 @@ def test_kimi_k3_preprocesses_only_dp_owner_images(monkeyp...; symbols: test_kimi_k3_preprocesses_only_dp_owner_images，涉及 `test_kimi_k3_preprocesses_only_dp_owner_images`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/kimi_k3_image_processing.py` modified +28/-15 (43 lines); hunks: -1,6 +1,7; -9,6 +10,22; symbols: KimiK3DeferredPreprocessing, prepare_kimi_k3_encoder_inputs, materialize_kimi_k3_cpu_features
  - `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +20/-11 (31 lines); hunks: -8,6 +8,7; -23,6 +24,7; symbols: prepare_deferred, KimiK3ImageProcessor, _should_defer_gpu_preprocessing, _build_deferred_output
  - `test/registered/unit/models/test_kimi_k25.py` modified +16/-9 (25 lines); hunks: -1,6 +1,7; -690,6 +691,7 @@ def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():; symbols: test_kimi_k3_epd_rebuild_uses_the_same_media_contract, test_kimi_k3_cpu_transport_defers_gpu_preprocessing
  - `test/registered/unit/models/test_kimi_k3_vision.py` modified +8/-8 (16 lines); hunks: -509,6 +509,7 @@ def test_kimi_k3_preprocesses_only_dp_owner_images(monkeypat...; -517,20 +518,19 @@ def test_kimi_k3_preprocesses_only_dp_owner_images(monkeyp...; symbols: test_kimi_k3_preprocesses_only_dp_owner_images
  - `python/sglang/srt/models/kimi_k3.py` modified +5/-5 (10 lines); hunks: -3276,8 +3276,8 @@ def materialize_item_features(image_indices: List[int]) ->...; -3287,17 +3287,17 @@ def materialize_item_features(image_indices: List[int])...; symbols: materialize_item_features
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/kimi_k3_image_processing.py
@@ -1,6 +1,7 @@
-from typing import Union
+from dataclasses import dataclass
+from typing import Literal, Optional, Union
@@ -9,6 +10,22 @@
+@dataclass(frozen=True)
+class KimiK3DeferredPreprocessing:
diff -- python/sglang/srt/multimodal/processors/kimi_k3.py
@@ -8,6 +8,7 @@
+import functools
@@ -23,6 +24,7 @@
+    KimiK3DeferredPreprocessing,
@@ -272,12 +274,16 @@ def prepare_deferred(self, text, images, original_input_ids=None):
-        deferred_config = {
-            "image_mean": list(self._image_mean),
diff -- test/registered/unit/models/test_kimi_k25.py
@@ -1,6 +1,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/kimi_k3_image_processing.py` modified +28/-15; `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +20/-11; `python/sglang/srt/models/kimi_k3.py` modified +5/-5
  - tests: `test/registered/unit/models/test_kimi_k25.py` modified +16/-9; `test/registered/unit/models/test_kimi_k3_vision.py` modified +8/-8; `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/models/test_kimi_k3_vision.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34779 - [CI] Disable the prefill CUDA graph on the P worker of test_kimi_linear_pd_dcp4

- 链接: https://github.com/sgl-project/sglang/pull/34779
- 状态/时间: merged / 2026-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`；关联提交 `f3beb2c52903`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Disable the prefill CUDA graph on the P worker of test_kimi_linear_pd_dcp4」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`；技术摘要: 覆盖「[CI] Disable the prefill CUDA graph on the P worker of test_kimi_linear_pd_dcp4」；主要实现面是 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` modified +2/-0 (2 lines); hunks: -306,6 +306,8 @@ def start_prefill(cls):; symbols: start_prefill，涉及 `start_prefill`。
- 代码 diff 细节:
  - `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` modified +2/-0 (2 lines); hunks: -306,6 +306,8 @@ def start_prefill(cls):; symbols: start_prefill
- 关键代码摘录:

```diff
diff -- test/registered/disaggregation/test_kimi_linear_pd_dcp4.py
@@ -306,6 +306,8 @@ def start_prefill(cls):
+            "--cuda-graph-backend-prefill",
+            "disabled",
```

- 已读文件:
  - tests: `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/disaggregation/test_kimi_linear_pd_dcp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34886 - [Docs] Update Kimi-K3 installation options

- 链接: https://github.com/sgl-project/sglang/pull/34886
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`；关联提交 `8b4faa33365b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-8，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Update Kimi-K3 installation options」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`；技术摘要: 覆盖「[Docs] Update Kimi-K3 installation options」；主要实现面是 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +14/-8 (22 lines); hunks: -10,28 +10,34 @@ tag: NEW。
- 代码 diff 细节:
  - `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +14/-8 (22 lines); hunks: -10,28 +10,34 @@ tag: NEW
- 关键代码摘录:

```diff
diff -- docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
@@ -10,28 +10,34 @@ tag: NEW
-For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install).
+For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle i
-<Tab title="Docker">
+<Tab title="Python (pip / uv)">
-docker pull lmsysorg/sglang:kimi-k3 # CUDA13
-docker pull lmsysorg/sglang:kimi-k3-cu12 # CUDA12
```

- 已读文件:
  - docs: `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +14/-8
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34778 - [Fix] Work around xgrammar 0.2.1 negative integer minimum in Kimi-K3 structural tags

- 链接: https://github.com/sgl-project/sglang/pull/34778
- 状态/时间: merged / 2026-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik3_structural_tag.py`, `test/registered/unit/function_call/test_kimik3_structural_tag.py`；关联提交 `3f64f14360d0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+60/-0，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Work around xgrammar 0.2.1 negative integer minimum in Kimi-K3 structural tags」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`；技术摘要: 覆盖「[Fix] Work around xgrammar 0.2.1 negative integer minimum in Kimi-K3 structural tags」；主要实现面是 `test/registered/unit/function_call/test_kimik3_structural_tag.py`, `python/sglang/srt/function_call/kimik3_structural_tag.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_kimik3_structural_tag.py` modified +33/-0 (33 lines); hunks: -379,6 +379,39 @@ def test_strict_schema_handles_number_enums_and_all_of_inte...; symbols: test_strict_schema_handles_number_enums_and_all_of_integer_constraints, test_strict_schema_handles_one_sided_negative_integer_minimum, test_strict_schema_preserves_additional_properties_default，涉及 `test_strict_schema_handles_number_enums_and_all_of_integer_constraints, test_strict_schema_handles_one_sided_negative_integer_minimum, test_strict_schema_preserves_additional_properties_default`；`python/sglang/srt/function_call/kimik3_structural_tag.py` modified +27/-0 (27 lines); hunks: -274,6 +274,33 @@ def _value_format(; symbols: _value_format，涉及 `_value_format`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_kimik3_structural_tag.py` modified +33/-0 (33 lines); hunks: -379,6 +379,39 @@ def test_strict_schema_handles_number_enums_and_all_of_inte...; symbols: test_strict_schema_handles_number_enums_and_all_of_integer_constraints, test_strict_schema_handles_one_sided_negative_integer_minimum, test_strict_schema_preserves_additional_properties_default
  - `python/sglang/srt/function_call/kimik3_structural_tag.py` modified +27/-0 (27 lines); hunks: -274,6 +274,33 @@ def _value_format(; symbols: _value_format
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_kimik3_structural_tag.py
@@ -379,6 +379,39 @@ def test_strict_schema_handles_number_enums_and_all_of_integer_constraints():
+def test_strict_schema_handles_one_sided_negative_integer_minimum():
+    tool = Tool(
+        type="function",
+        function=Function(
+            name="submit",
+            strict=True,
diff -- python/sglang/srt/function_call/kimik3_structural_tag.py
@@ -274,6 +274,33 @@ def _value_format(
+    # XGrammar 0.2.1 miscompiles a one-sided negative integer lower bound:
+    # {"type": "integer", "minimum": -N} accepts the incomplete value "-"
+    # and rejects every valid negative integer. Splitting the range at zero
+    # avoids that converter bug without weakening the schema.
+    if (
+        json_type == "integer"
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_kimik3_structural_tag.py` modified +33/-0
  - runtime: `python/sglang/srt/function_call/kimik3_structural_tag.py` modified +27/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_kimik3_structural_tag.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34883 - [Kimi-K3] Use explicit SiTU activation for MegaMoE

- 链接: https://github.com/sgl-project/sglang/pull/34883
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`, `test/registered/models_e2e/test_kimi_k3_b300.py`；关联提交 `7769f54febc9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+68/-24，可读 patch 194 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kimi-K3] Use explicit SiTU activation for MegaMoE」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/kimi_k3.py`, `test/registered/models_e2e/test_kimi_k3_b300.py`；技术摘要: 覆盖「[Kimi-K3] Use explicit SiTU activation for MegaMoE」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`, `test/registered/models_e2e/test_kimi_k3_b300.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +4/-16 (20 lines); hunks: -141,14 +141,6 @@ def _cdiv(a: int, b: int) -> int:; -487,9 +479,8 @@ def __init__(; symbols: _cdiv, _k3_bf16_gemm, __init__, _forward_mega_experts，涉及 `_cdiv, _k3_bf16_gemm, __init__`；`test/registered/models_e2e/test_kimi_k3_b300.py` modified +54/-2 (56 lines); hunks: -1,6 +1,6; -18,13 +18,17; symbols: tearDownClass, TestKimiK3B300MegaMoE, setUpClass，涉及 `tearDownClass, TestKimiK3B300MegaMoE, setUpClass`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +4/-16 (20 lines); hunks: -141,14 +141,6 @@ def _cdiv(a: int, b: int) -> int:; -487,9 +479,8 @@ def __init__(; symbols: _cdiv, _k3_bf16_gemm, __init__, _forward_mega_experts
  - `test/registered/models_e2e/test_kimi_k3_b300.py` modified +54/-2 (56 lines); hunks: -1,6 +1,6; -18,13 +18,17; symbols: tearDownClass, TestKimiK3B300MegaMoE, setUpClass
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -141,14 +141,6 @@ def _cdiv(a: int, b: int) -> int:
-# MegaMoE SiTU sentinel: DeepGEMM 0.1.5.post1+ selects the K3 SiTU
-# activation when activation_clamp == 0.03125 (2^-5: exactly representable and
-# unused by any legitimate swiglu clamp; the host asserts clamp >= 0 so a
-# negative sentinel is impossible). beta=4.0 / linear_beta=25.0 are baked into
-# the DeepGEMM kernel.
-_K3_MEGA_SITU_SENTINEL_CLAMP = 0.03125
diff -- test/registered/models_e2e/test_kimi_k3_b300.py
@@ -1,6 +1,6 @@
-Runs the Low Latency DSPARK recipe and the Balanced DCP/HiCache recipe on
+Runs the Low Latency DSPARK, Balanced DCP/HiCache, and MegaMoE recipes on
@@ -18,13 +18,17 @@
-register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")
+register_cuda_ci(est_time=1200, stage="base-c", runner_config="8-gpu-b300")
+MEGAMOE_URL = "http://0.0.0.0:30000"
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +4/-16
  - tests: `test/registered/models_e2e/test_kimi_k3_b300.py` modified +54/-2
- 验证与风险: diff 自带测试面 `test/registered/models_e2e/test_kimi_k3_b300.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34404 - [VLM] Cache Kimi-K3 per-image processor artifacts

- 链接: https://github.com/sgl-project/sglang/pull/34404
- 状态/时间: merged / 2026-08-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`, `python/sglang/srt/multimodal/kimi_k3_image_processing.py`, `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` 等 6 个文件；关联提交 `3d3194f6c311`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+2596/-255，可读 patch 3784 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Cache Kimi-K3 per-image processor artifacts」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py`；技术摘要: 覆盖「[VLM] Cache Kimi-K3 per-image processor artifacts」；主要实现面是 `test/registered/unit/models/test_kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k3.py`, `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/models/test_kimi_k25.py` modified +558/-19 (577 lines); hunks: -1,7 +1,14; -18,11 +25,26; symbols: test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1, test_dp_helper_can_lazily_load_kimi_features_on_tp1, test_kimi_non_dp_keeps_grid_thws_on_the_host, __init__，涉及 `test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1, test_dp_helper_can_lazily_load_kimi_features_on_tp1, test_kimi_non_dp_keeps_grid_thws_on_the_host`；`python/sglang/srt/multimodal/processors/kimi_k3.py` modified +304/-51 (355 lines); hunks: -9,8 +9,9; -22,6 +23,7; symbols: _k3_to_cuda_chw, KimiK3GPUProcessorWrapper, __init__, preprocess_fingerprint_payload，涉及 `_k3_to_cuda_chw, KimiK3GPUProcessorWrapper, __init__`；`python/sglang/srt/multimodal/media_artifacts/kimi_k3.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: KimiK3MediaProcessorConfigProvider, KimiK3PreprocessConfig, from_media_processor, KimiK3ResizeConfig，涉及 `KimiK3MediaProcessorConfigProvider, KimiK3PreprocessConfig, from_media_processor`；`python/sglang/srt/models/kimi_k3.py` modified +76/-41 (117 lines); hunks: -106,6 +106,7; -3075,6 +3076,10 @@ class KimiK3ForConditionalGeneration(nn.Module):; symbols: KimiK3ForConditionalGeneration, materialize_item_features，涉及 `KimiK3ForConditionalGeneration, materialize_item_features`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_kimi_k25.py` modified +558/-19 (577 lines); hunks: -1,7 +1,14; -18,11 +25,26; symbols: test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1, test_dp_helper_can_lazily_load_kimi_features_on_tp1, test_kimi_non_dp_keeps_grid_thws_on_the_host, __init__
  - `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +304/-51 (355 lines); hunks: -9,8 +9,9; -22,6 +23,7; symbols: _k3_to_cuda_chw, KimiK3GPUProcessorWrapper, __init__, preprocess_fingerprint_payload
  - `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py` added +140/-0 (140 lines); hunks: -0,0 +1,140; symbols: KimiK3MediaProcessorConfigProvider, KimiK3PreprocessConfig, from_media_processor, KimiK3ResizeConfig
  - `python/sglang/srt/models/kimi_k3.py` modified +76/-41 (117 lines); hunks: -106,6 +106,7; -3075,6 +3076,10 @@ class KimiK3ForConditionalGeneration(nn.Module):; symbols: KimiK3ForConditionalGeneration, materialize_item_features
  - `python/sglang/srt/multimodal/kimi_k3_image_processing.py` modified +21/-8 (29 lines); hunks: -46,7 +46,13 @@ def prepare_kimi_k3_encoder_inputs(; -69,12 +75,16 @@ def prepare_kimi_k3_encoder_inputs(; symbols: prepare_kimi_k3_encoder_inputs
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_kimi_k25.py
@@ -1,7 +1,14 @@
+import base64
+import io
+import pickle
+import tempfile
+from concurrent.futures import ThreadPoolExecutor
+from dataclasses import replace
diff -- python/sglang/srt/multimodal/processors/kimi_k3.py
@@ -9,8 +9,9 @@
+import math
-from typing import Dict, List, Union
+from typing import Dict, List, Optional, Union
@@ -22,6 +23,7 @@
+from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
@@ -32,6 +34,15 @@
diff -- python/sglang/srt/multimodal/media_artifacts/kimi_k3.py
@@ -0,0 +1,140 @@
```

- 已读文件:
  - tests: `test/registered/unit/models/test_kimi_k25.py` modified +558/-19; `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py` modified +83/-0
  - runtime: `python/sglang/srt/multimodal/processors/kimi_k3.py` modified +304/-51; `python/sglang/srt/multimodal/media_artifacts/kimi_k3.py` added +140/-0; `python/sglang/srt/models/kimi_k3.py` modified +76/-41; `python/sglang/srt/multimodal/kimi_k3_image_processing.py` modified +21/-8
- 验证与风险: diff 自带测试面 `test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py`, `test/registered/unit/managers/test_mm_process_config.py`, `test/registered/unit/models/test_kimi_k25.py`, `test/registered/unit/multimodal/rust/qwen/_fixtures.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35168 - docs: add NVFP4 quantization option to Kimi-K3 deploy panel

- 链接: https://github.com/sgl-project/sglang/pull/35168
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；关联提交 `861eca8e2519`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+41/-1，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: add NVFP4 quantization option to Kimi-K3 deploy panel」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`；技术摘要: 覆盖「docs: add NVFP4 quantization option to Kimi-K3 deploy panel」；主要实现面是 `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`, `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +40/-0 (40 lines); hunks: -160,6 +160,39 @@ export const config = {; -333,6 +366,7 @@ export const config = {；`docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +1/-1 (2 lines); hunks: -40,7 +40,7 @@ For how to launch the image, see [Install → Method 3: Using Do...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +40/-0 (40 lines); hunks: -160,6 +160,39 @@ export const config = {; -333,6 +366,7 @@ export const config = {
  - `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +1/-1 (2 lines); hunks: -40,7 +40,7 @@ For how to launch the image, see [Install → Method 3: Using Do...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/moonshotai/kimi-k3.jsx
@@ -160,6 +160,39 @@ export const config = {
+    {
+      // Checkpoint choice, orthogonal to the cell grid: MXFP4 is the shipping
+      // default, NVFP4 is NVIDIA's ModelOpt mixed checkpoint (NVFP4 SiTU routed
+      // experts + FP8_PB_WO 128x128 block-FP8 attention). NVFP4 swaps the model
+      // slug (modelNames) and pins the TRT-LLM MoE runner — the auto resolution
+      // never engages TRT-LLM deferred finalize and the NVFP4 MoE raises
diff -- docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
@@ -40,7 +40,7 @@ For how to launch the image, see [Install → Method 3: Using Docker](../../../d
-Pick your hardware, then the deployment shape and operating point. Node count follows the hardware recipe (B200 2×8, GB200 4×4, H100 4×8, B300 1×8, H200 2×8 — 4×8 on Unified High-
+Pick your hardware, then the deployment shape and operating point. Node count follows the hardware recipe (B200 2×8, GB200 4×4, H100 4×8, B300 1×8, H200 2×8 — 4×8 on Unified High-
```

- 已读文件:
  - docs: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` modified +40/-0; `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`, `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #32568 - [AMD] Add Kimi-K3 8-GPU MI35x nightly accuracy CI

- 链接: https://github.com/sgl-project/sglang/pull/32568
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py`；关联提交 `816ea65058d2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+185/-10，可读 patch 244 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add Kimi-K3 8-GPU MI35x nightly accuracy CI」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py`；技术摘要: 覆盖「[AMD] Add Kimi-K3 8-GPU MI35x nightly accuracy CI」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestKimiK3EvalMI35x, setUpClass, test_kimi_k3_gsm8k_accuracy，涉及 `TestKimiK3EvalMI35x, setUpClass, test_kimi_k3_gsm8k_accuracy`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestKimiK3EvalMI35x, setUpClass, test_kimi_k3_gsm8k_accuracy
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py
@@ -0,0 +1,163 @@
+"""MI35x Kimi-K3 GSM8K Completion Evaluation Test (8-GPU)
+Tests moonshotai/Kimi-K3 with GSM8K few-shot benchmark on MI35x.
+Server arguments follow the Day-0 recipe in the AMD tracking issue
+(sgl-project/sglang#32548) for the non-speculative config: TP8 with the
+Triton attention backend, the AITER FlyDSL A8W4 SiTU MoE path, and the radix
+cache disabled.
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py` added +163/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_kimi_k3_eval_mi35x.py`, `test/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34985 - [AMD] Add the Kimi-K3 MI35x perf benchmarks in nightly

- 链接: https://github.com/sgl-project/sglang/pull/34985
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py`；关联提交 `8ea5229d421d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+155/-0，可读 patch 163 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add the Kimi-K3 MI35x perf benchmarks in nightly」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 性能/后端优化；主要 diff: `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py`；技术摘要: 覆盖「[AMD] Add the Kimi-K3 MI35x perf benchmarks in nightly」；主要实现面是 `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: TestNightlyKimiK3PerformanceMI35x, setUpClass, test_bench_kimi_k3，涉及 `TestNightlyKimiK3PerformanceMI35x, setUpClass, test_bench_kimi_k3`。
- 代码 diff 细节:
  - `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: TestNightlyKimiK3PerformanceMI35x, setUpClass, test_bench_kimi_k3
- 关键代码摘录:

```diff
diff -- test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py
@@ -0,0 +1,138 @@
+"""MI35x nightly performance benchmark for Kimi-K3 (8-GPU).
+Benchmarks moonshotai/Kimi-K3 at TP8 on MI35x using the same non-speculative
+Day-0 recipe as the accuracy test (sgl-project/sglang#32548), so the two are
+directly comparable and a perf regression cannot be confused with a config
+difference.
+This runs as the step after the eval inside nightly-8-gpu-mi35x-kimi-k3-rocm720
```

- 已读文件:
  - tests: `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py` added +138/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34881 - Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation

- 链接: https://github.com/sgl-project/sglang/pull/34881
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/kimik3_detector.py`, `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`；关联提交 `307a90f6d3c7`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+374/-12，可读 patch 529 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 文档/测试/CI；主要 diff: `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`, `python/sglang/srt/function_call/kimik3_detector.py`；技术摘要: 覆盖「Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation」；主要实现面是 `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`, `python/sglang/srt/function_call/kimik3_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_kimik3_detector.py` modified +32/-0 (32 lines); hunks: -208,6 +208,38 @@ def test_streaming_bookkeeping_for_serving_layer() -> None:; symbols: test_streaming_bookkeeping_for_serving_layer, test_stream_end_reports_truncated_tools_section, test_stream_end_releases_held_back_text, test_stream_end_drops_truncated_marker，涉及 `test_streaming_bookkeeping_for_serving_layer, test_stream_end_reports_truncated_tools_section, test_stream_end_releases_held_back_text`；`test/registered/unit/parser/test_kimik3_reasoning_parser.py` modified +32/-0 (32 lines); hunks: -154,6 +154,38 @@ def test_streaming_recovers_missing_think_separator() -> None:; symbols: test_streaming_recovers_missing_think_separator, test_non_stream_tools_channel_before_think_close_is_not_reasoning, test_streaming_tools_channel_before_think_close, test_reasoning_parser_registration，涉及 `test_streaming_recovers_missing_think_separator, test_non_stream_tools_channel_before_think_close_is_not_reasoning, test_streaming_tools_channel_before_think_close`；`python/sglang/srt/function_call/kimik3_detector.py` modified +15/-0 (15 lines); hunks: -19,6 +19,7; -217,6 +218,20 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, finish, _emit_normal_text，涉及 `parse_streaming_increment, finish, _emit_normal_text`。
- 代码 diff 细节:
  - `test/registered/function_call/test_kimik3_detector.py` modified +32/-0 (32 lines); hunks: -208,6 +208,38 @@ def test_streaming_bookkeeping_for_serving_layer() -> None:; symbols: test_streaming_bookkeeping_for_serving_layer, test_stream_end_reports_truncated_tools_section, test_stream_end_releases_held_back_text, test_stream_end_drops_truncated_marker
  - `test/registered/unit/parser/test_kimik3_reasoning_parser.py` modified +32/-0 (32 lines); hunks: -154,6 +154,38 @@ def test_streaming_recovers_missing_think_separator() -> None:; symbols: test_streaming_recovers_missing_think_separator, test_non_stream_tools_channel_before_think_close_is_not_reasoning, test_streaming_tools_channel_before_think_close, test_reasoning_parser_registration
  - `python/sglang/srt/function_call/kimik3_detector.py` modified +15/-0 (15 lines); hunks: -19,6 +19,7; -217,6 +218,20 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, finish, _emit_normal_text
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_kimik3_detector.py
@@ -208,6 +208,38 @@ def test_streaming_bookkeeping_for_serving_layer() -> None:
+def test_stream_end_reports_truncated_tools_section(caplog) -> None:
+    """A tools section cut off before its closing tag used to vanish at
+    end-of-stream: no call, no text, no log. It must at least be reported."""
+    detector = KimiK3Detector()
+    tools = [_make_tool("python")]
+    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
diff -- test/registered/unit/parser/test_kimik3_reasoning_parser.py
@@ -154,6 +154,38 @@ def test_streaming_recovers_missing_think_separator() -> None:
+_TOOLS_CHANNEL = (
+    f'{TOOLS_OPEN}<|open|>call tool="python" index="1"<|sep|>'
+    "<|close|>call<|sep|>"
+    f"{TOOLS_CLOSE}"
+)
+def test_non_stream_tools_channel_before_think_close_is_not_reasoning() -> None:
diff -- python/sglang/srt/function_call/kimik3_detector.py
@@ -19,6 +19,7 @@
```

- 已读文件:
  - tests: `test/registered/function_call/test_kimik3_detector.py` modified +32/-0; `test/registered/unit/parser/test_kimik3_reasoning_parser.py` modified +32/-0
  - runtime: `python/sglang/srt/function_call/kimik3_detector.py` modified +15/-0
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_kimik3_detector.py`, `test/registered/unit/entrypoints/openai/test_serving_chat.py`, `test/registered/unit/entrypoints/openai/test_serving_responses.py`, `test/registered/unit/parser/test_kimik3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35077 - [Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint

- 链接: https://github.com/sgl-project/sglang/pull/35077
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_k3.py`, `test/registered/unit/models/test_kimi_k3_bfa_overlap.py`；关联提交 `5f128395910d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+205/-38，可读 patch 506 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_k3.py`, `test/registered/unit/models/test_kimi_k3_bfa_overlap.py`；技术摘要: 覆盖「[Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint」；主要实现面是 `python/sglang/srt/models/kimi_k3.py`, `test/registered/unit/models/test_kimi_k3_bfa_overlap.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_k3.py` modified +82/-23 (105 lines); hunks: -73,6 +73,7; -142,6 +143,35 @@ def _cdiv(a: int, b: int) -> int:; symbols: _cdiv, _uses_modelopt_fp8_pb_wo, _maybe_map_fp8_pb_scale_name, _get_k3_dense_weight，涉及 `_cdiv, _uses_modelopt_fp8_pb_wo, _maybe_map_fp8_pb_scale_name`；`test/registered/unit/models/test_kimi_k3_bfa_overlap.py` modified +38/-2 (40 lines); hunks: -8,7 +8,10; -39,9 +42,9 @@ def fused_qkvg_proj(x):; symbols: fused_qkvg_proj, test_eager_stream_branch_not_taken, test_block_fp8_weight_is_dequantized_for_tiny_gemm, test_per_tensor_fp8_weight_is_not_block_dequantized，涉及 `fused_qkvg_proj, test_eager_stream_branch_not_taken, test_block_fp8_weight_is_dequantized_for_tiny_gemm`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_k3.py` modified +82/-23 (105 lines); hunks: -73,6 +73,7; -142,6 +143,35 @@ def _cdiv(a: int, b: int) -> int:; symbols: _cdiv, _uses_modelopt_fp8_pb_wo, _maybe_map_fp8_pb_scale_name, _get_k3_dense_weight
  - `test/registered/unit/models/test_kimi_k3_bfa_overlap.py` modified +38/-2 (40 lines); hunks: -8,7 +8,10; -39,9 +42,9 @@ def fused_qkvg_proj(x):; symbols: fused_qkvg_proj, test_eager_stream_branch_not_taken, test_block_fp8_weight_is_dequantized_for_tiny_gemm, test_per_tensor_fp8_weight_is_not_block_dequantized
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_k3.py
@@ -73,6 +73,7 @@
+from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
@@ -142,6 +143,35 @@ def _cdiv(a: int, b: int) -> int:
+def _uses_modelopt_fp8_pb_wo(
+    quant_config: Optional[QuantizationConfig], prefix: str
+) -> bool:
+    resolver = getattr(quant_config, "_resolve_quant_algo", None)
diff -- test/registered/unit/models/test_kimi_k3_bfa_overlap.py
@@ -8,7 +8,10 @@
-from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
+from sglang.srt.models.kimi_k3 import (
+    KimiK3DeltaAttention,
+    _get_k3_dense_weight,
+)
@@ -39,9 +42,9 @@ def fused_qkvg_proj(x):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_k3.py` modified +82/-23
  - tests: `test/registered/unit/models/test_kimi_k3_bfa_overlap.py` modified +38/-2
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py`, `test/registered/unit/model_loader/test_modelopt_loader.py`, `test/registered/unit/models/test_kimi_k3_bfa_overlap.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34546 - [XPU] Fix/kimi linear xpu

- 链接: https://github.com/sgl-project/sglang/pull/34546
- 状态/时间: merged / 2026-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/kimi_linear.py`；关联提交 `a5a9d66bafa9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+9/-5，可读 patch 49 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Fix/kimi linear xpu」；模型线: Kimi K2/K2.5/K3/Linear/VL；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[XPU] Fix/kimi linear xpu」；主要实现面是 `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/kimi_linear.py` modified +2/-2 (4 lines); hunks: -49,7 +49,7; -550,7 +550,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/kimi_linear.py` modified +2/-2 (4 lines); hunks: -49,7 +49,7; -550,7 +550,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/kimi_linear.py
@@ -49,7 +49,7 @@
-from sglang.srt.utils import make_layers
+from sglang.srt.utils import is_xpu, make_layers
@@ -550,7 +550,7 @@ def __init__(
-        self.alt_stream = get_stream("alt")
+        self.alt_stream = None if is_xpu() else get_stream("alt")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/kimi_linear.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
