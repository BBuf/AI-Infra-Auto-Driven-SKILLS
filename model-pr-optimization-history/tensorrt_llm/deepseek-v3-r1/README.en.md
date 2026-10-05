# TensorRT-LLM DeepSeek V3/R1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/CMakeLists.txt` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) |
| `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560), [#9799](https://github.com/NVIDIA/TensorRT-LLM/pull/9799) |
| `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) |
| `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560), [#9799](https://github.com/NVIDIA/TensorRT-LLM/pull/9799) |
| `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) |
| `cpp/tensorrt_llm/thop/dsv3FusedAGemmOp.cpp` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) |
| `cpp/tensorrt_llm/thop/dsv3RopeOp.cpp` | no direct PR-number commit |
| `cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) |
| `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` | [#3232](https://github.com/NVIDIA/TensorRT-LLM/pull/3232), [#5235](https://github.com/NVIDIA/TensorRT-LLM/pull/5235), [#5600](https://github.com/NVIDIA/TensorRT-LLM/pull/5600), [#5796](https://github.com/NVIDIA/TensorRT-LLM/pull/5796) |
| `docs/source/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md` | no direct PR-number commit |
| `docs/source/blogs/tech_blog/blog02_DeepSeek_R1_MTP_Implementation_and_Optimization.md` | no direct PR-number commit |
| `docs/source/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md` | no direct PR-number commit |
| `docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md` | no direct PR-number commit |
| `examples/configs/curated/deepseek-r1-deepgemm.yaml` | no direct PR-number commit |
| `examples/configs/curated/deepseek-r1-latency.yaml` | no direct PR-number commit |
| `examples/configs/curated/deepseek-r1-throughput.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp4_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp4_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc1024.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc128.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc2048.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc256.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc4.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc512.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc64.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k1k_tp8_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc1024.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc128.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc2048.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc256.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc4.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc512.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc64.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/1k8k_tp8_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp4_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp4_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp4_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc1024.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc128.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc2048.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc256.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc4.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc512.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc64.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/B200/8k1k_tp8_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc1024.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc128.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc2048.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc256.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc4.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc512.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc64.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k1k_tp8_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc1.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc128.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc16.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc2.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc256.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc32.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc4.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc512.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc64.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/1k8k_tp8_conc8.yaml` | no direct PR-number commit |
| `examples/configs/database/deepseek-ai/DeepSeek-R1-0528/H200/8k1k_tp8_conc1.yaml` | no direct PR-number commit |
| ... | 170 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 41
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 41
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-10 | [#3232](https://github.com/NVIDIA/TensorRT-LLM/pull/3232) | merged | doc: Add DeepSeek-R1 perf doc | `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` |
| 2025-04-16 | [#3572](https://github.com/NVIDIA/TensorRT-LLM/pull/3572) | merged | chore: Add comments to modifications that fix TP size of DeepSeek-V3/R1 when using more than 16 GPUs | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-04-28 | [#3910](https://github.com/NVIDIA/TensorRT-LLM/pull/3910) | merged | Add docs about DeepSeek-R1 long context support. | `examples/models/core/deepseek_v3/README.md` |
| 2025-04-30 | [#3829](https://github.com/NVIDIA/TensorRT-LLM/pull/3829) | merged | refactor: Clean up allreduce module for Deepseek V3 model | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-05-06 | [#3354](https://github.com/NVIDIA/TensorRT-LLM/pull/3354) | merged | feat: add deepseek-r1 reasoning parser to trtllm-serve | `examples/serve/deepseek_r1_reasoning_parser.sh`, `tensorrt_llm/llmapi/reasoning_parser.py`, `tensorrt_llm/serve/postprocess_handlers.py` |
| 2025-05-14 | [#4123](https://github.com/NVIDIA/TensorRT-LLM/pull/4123) | merged | [TRTLLM-3330][feat] Support DeepSeek-R1 W4A8 on Hopper | `examples/models/core/deepseek_v3/README.md`, `tensorrt_llm/_torch/models/modeling_utils.py`, `tensorrt_llm/layers/moe.py` |
| 2025-05-15 | [#4338](https://github.com/NVIDIA/TensorRT-LLM/pull/4338) | merged | [doc] Add tensorrtllm_backend serving documentation in the Deepseek-V3 README | `examples/models/core/deepseek_v3/README.md` |
| 2025-05-19 | [#3952](https://github.com/NVIDIA/TensorRT-LLM/pull/3952) | merged | [https://nvbugs/5123103][fix] Fix torch compile for DeepSeekV3 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-06-11 | [#4522](https://github.com/NVIDIA/TensorRT-LLM/pull/4522) | merged | test: conditional disagg and cache aware balancing for deepseek v3 | `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` |
| 2025-06-14 | [#4560](https://github.com/NVIDIA/TensorRT-LLM/pull/4560) | merged | Feat/ds r1 min latency opt round3, add router gemm, fused a gemm, PDL | `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` |
| 2025-06-16 | [#5235](https://github.com/NVIDIA/TensorRT-LLM/pull/5235) | merged | Update DeepSeek R1 perf numbers to latest release/0.20 results | `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` |
| 2025-06-30 | [#5600](https://github.com/NVIDIA/TensorRT-LLM/pull/5600) | merged | doc: Minor update to DeepSeek R1 best practice | `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` |
| 2025-07-08 | [#5796](https://github.com/NVIDIA/TensorRT-LLM/pull/5796) | merged | doc: update cuda_graph_config usage part in DS R1 docs | `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` |
| 2025-08-01 | [#6486](https://github.com/NVIDIA/TensorRT-LLM/pull/6486) | merged | Deepseek R1 FP8 Support on Blackwell | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-08-13 | [#6200](https://github.com/NVIDIA/TensorRT-LLM/pull/6200) | merged | [https://nvbugs/5378031] [feat] Hopper W4A8 MoE supports ModelOpt ckpt for PyT backend | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-08-14 | [#6698](https://github.com/NVIDIA/TensorRT-LLM/pull/6698) | merged | [TRTLLM-6853][feat] refactor deepseekv3 model | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-08-27 | [#7238](https://github.com/NVIDIA/TensorRT-LLM/pull/7238) | merged | [None][fix] Remove and fuse some element-wise ops in the ds-r1-fp8 model | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-08-28 | [#6886](https://github.com/NVIDIA/TensorRT-LLM/pull/6886) | merged | [https://nvbugs/5445466][fix] Bypass MLP TP split for MNNVL in DeepSeek V3 to avoid hanging. | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-09-06 | [#7123](https://github.com/NVIDIA/TensorRT-LLM/pull/7123) | merged | [None][fix] DeepSeek-R1 W4A8 weight loading issue; fixes regression from #6200 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-10-20 | [#7761](https://github.com/NVIDIA/TensorRT-LLM/pull/7761) | merged | [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for 384 experts (MoE TRTLLM backend) | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-10-24 | [#8405](https://github.com/NVIDIA/TensorRT-LLM/pull/8405) | merged | [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache | `tensorrt_llm/_torch/configs/deepseek_v3.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-11-14 | [#9141](https://github.com/NVIDIA/TensorRT-LLM/pull/9141) | merged | [None][doc] Add DeepSeek-V3.2-Exp document | `examples/models/core/deepseek_v3/README.md` |
| 2025-11-18 | [#9217](https://github.com/NVIDIA/TensorRT-LLM/pull/9217) | merged | [None][chore] fix a deepseekv3 error when debug mode is on | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-11-19 | [#9231](https://github.com/NVIDIA/TensorRT-LLM/pull/9231) | merged | [None][doc] Update DS-R1 example doc | `examples/models/core/deepseek_v3/README.md` |
| 2025-12-02 | [#9383](https://github.com/NVIDIA/TensorRT-LLM/pull/9383) | merged | [None][feat] Add support for KVCache reuse for DSv32 | `examples/models/core/deepseek_v3/README.md`, `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`, `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` |
| 2025-12-08 | [#9661](https://github.com/NVIDIA/TensorRT-LLM/pull/9661) | merged | [TRTLLM-9506][fix] Fix AR for DeepSeek-R1 2 model path | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-12-09 | [#9799](https://github.com/NVIDIA/TensorRT-LLM/pull/9799) | merged | [None][fix] Fix PDL in TRTLLM MOE for dsv3 | `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` |
| 2025-12-19 | [#10010](https://github.com/NVIDIA/TensorRT-LLM/pull/10010) | merged | [TRTLLM-9604][feat] DS R1 & V3.1 tool parser | `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py`, `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py`, `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja` |
| 2025-12-23 | [#10126](https://github.com/NVIDIA/TensorRT-LLM/pull/10126) | merged | [TRTLLM-9677][feat] Support DeepSeek-V3.2 tool parser | `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` |
| 2026-03-04 | [#11215](https://github.com/NVIDIA/TensorRT-LLM/pull/11215) | merged | [None][feat] Support mix quantization between shared experts and routed experts for dsv3 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-03-06 | [#11507](https://github.com/NVIDIA/TensorRT-LLM/pull/11507) | merged | [TRTLLM-11057][feat] Add Helix CP support for DSV3.2 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-05-18 | [#12530](https://github.com/NVIDIA/TensorRT-LLM/pull/12530) | merged | [https://nvbugs/5879577][fix] Fix KeyError in DeepSeekV3Lite FP8 MTP weight loading | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-05-18 | [#13882](https://github.com/NVIDIA/TensorRT-LLM/pull/13882) | merged | [None][test] Add DSR1 B200 DISAGG to CI Perf Test | `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml` |
| 2026-06-03 | [#14856](https://github.com/NVIDIA/TensorRT-LLM/pull/14856) | merged | [None][test] Update DSV32 32k4k config to avoid timeout issue | `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` |
| 2026-06-10 | [#15205](https://github.com/NVIDIA/TensorRT-LLM/pull/15205) | merged | [None][test] Increase kv_transfer_timeout_ms for b200 deepseek-r1 disagg gen_only perf test | `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` |
| 2026-07-17 | [#16528](https://github.com/NVIDIA/TensorRT-LLM/pull/16528) | merged | [https://nvbugs/6329052][fix] Change the config for DSV3 disagg conditional test | `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` |
| 2026-07-24 | [#16669](https://github.com/NVIDIA/TensorRT-LLM/pull/16669) | merged | [TRTLLM-13948][test] Migrate DeepSeek R1/V3.2 disagg perf cases to transceiver V2 | `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` |
| 2026-08-03 | [#16908](https://github.com/NVIDIA/TensorRT-LLM/pull/16908) | merged | [TRTLLM-13948][feat] Set DeepSeekV3 to use Python KV-cache transceiver V2 by default | `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` |
| 2026-08-07 | [#17397](https://github.com/NVIDIA/TensorRT-LLM/pull/17397) | merged | [https://nvbugs/6523520][fix] Halve gb200 r1-fp4 128k8k con128 multi_round to fit perf-sanity budget | `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` |
| 2026-08-12 | [#17427](https://github.com/NVIDIA/TensorRT-LLM/pull/17427) | merged | [https://nvbugs/6472256][fix] Fix disagg stress cluster flapping and DeepSeek R1 FP4 ctx OOM; add aiperf error-rate gate | `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml` |
| 2026-09-21 | [#19332](https://github.com/NVIDIA/TensorRT-LLM/pull/19332) | merged | [None][test] Add coverage for DeepseekV32ForCausalLM | `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` |

## Per-PR Diff Audit Cards

### PR #3232 - doc: Add DeepSeek-R1 perf doc

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3232
- Status/date: merged / 2025-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; associated commits `67949f7c3901`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +384/-123, 599 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` added +324/-0 (324 lines); hunks: -0,0 +1,324.
- Code diff details:
  - `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` added +324/-0 (324 lines); hunks: -0,0 +1,324
- Key code excerpts:

```diff
diff -- docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
@@ -0,0 +1,324 @@
+# How to get best performance on DeepSeek-R1 in TensorRT-LLM
+NVIDIA has announced world-record DeepSeek-R1 inference performance at NVIDIA GTC 2025. A single NVIDIA DGX system with eight NVIDIA Blackwell GPUs can achieve over 250 tokens per
+In this blog, we share the configurations and procedures about how to reproduce the number on both B200 and H200 with PyTorch workflow.
+## Prerequisites: Install TensorRT-LLM and download models
+This section can be skipped if you already have TensorRT-LLM installed and have already downloaded the DeepSeek R1 model checkpoint.
+#### 1. Download TensorRT-LLM
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` added +324/-0
- Risk and verification: This is mostly docs/examples in `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`, `examples/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #3572 - chore: Add comments to modifications that fix TP size of DeepSeek-V3/R1 when using more than 16 GPUs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3572
- Status/date: merged / 2025-04-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `efabf6b44374`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +12/-2, 38 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +12/-2 (14 lines); hunks: -287,14 +287,19 @@ def __init__(self,; -432,10 +437,15 @@ def __init__(self, model_config: ModelConfig[PretrainedCon...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +12/-2 (14 lines); hunks: -287,14 +287,19 @@ def __init__(self,; -432,10 +437,15 @@ def __init__(self, model_config: ModelConfig[PretrainedCon...; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -287,14 +287,19 @@ def __init__(self,
+        # The block scale size is 128, which requires shared_expert_intermediate_size to be divisible by 128.
+        assert shared_expert_intermediate_size % 128 == 0
+            # If using attention DP, the shared experts also use DP instead of TP.
-            assert shared_expert_intermediate_size % 128 == 0
+            # Due to the restriction of block scale size (i.e., 128), the supported TP sizes only include 1, 2, 4, 8, and 16.
+            # The math.gcd operation ensures that shared_tp_size falls in the supported TP sizes.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3910 - Add docs about DeepSeek-R1 long context support.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3910
- Status/date: merged / 2025-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `3617e948fda3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +64/-1, 79 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +64/-1 (65 lines); hunks: -21,7 +21,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...; -88,6 +89,68 @@ python quickstart_advanced.py --model_dir --spec_decode_algo MT.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +64/-1 (65 lines); hunks: -21,7 +21,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...; -88,6 +89,68 @@ python quickstart_advanced.py --model_dir --spec_decode_algo MT
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -21,7 +21,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/
-    - [Run evaluation on GPQA dataset](#run-evaluation-on-gpqa-dataset)
+    - [Long context support](#long-context-support)
+  - [Evaluation](#evaluation)
@@ -88,6 +89,68 @@ python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MT
+### Long context support
+DeepSeek-V3 model can support up to 128k context length. The following shows how to benchmark 64k and 128k input_seq_length using trtllm-bench on B200.
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +64/-1
- Risk and verification: This is mostly docs/examples in `examples/models/core/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #3829 - refactor: Clean up allreduce module for Deepseek V3 model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3829
- Status/date: merged / 2025-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `9cc5922a0bd3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +371/-365, 957 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +65/-145 (210 lines); hunks: -45,7 +45,7; -392,7 +392,7 @@ def __init__(self,; symbols: __init__, _compute_routed_output, _compute_mlp_tp_size, forward, touching `__init__, _compute_routed_output, _compute_mlp_tp_size`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +65/-145 (210 lines); hunks: -45,7 +45,7; -392,7 +392,7 @@ def __init__(self,; symbols: __init__, _compute_routed_output, _compute_mlp_tp_size, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -45,7 +45,7 @@
-                           DeepseekAllReduce, allgather)
+                           MoEAllReduce, allgather)
@@ -392,7 +392,7 @@ def __init__(self,
-        self.all_reduce = AllReduce(self.mapping)
+        self.allreduce = AllReduce(self.mapping)
@@ -516,7 +516,7 @@ def _compute_routed_output():
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +65/-145
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/multi_gpu/test_allreduce.py`, `tests/unittest/_torch/multi_gpu/test_deepseek_allreduce.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #3354 - feat: add deepseek-r1 reasoning parser to trtllm-serve

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3354
- Status/date: merged / 2025-05-06
- Trace source: `git log --name-only -- <model-files>` found it through `examples/serve/deepseek_r1_reasoning_parser.sh`; associated commits `e84dc6b3c7bc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +352/-6, 499 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/serve/deepseek_r1_reasoning_parser.sh` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tensorrt_llm/llmapi/reasoning_parser.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: ReasoningParserResult, __init__, BaseReasoningParser, parse, touching `ReasoningParserResult, __init__, BaseReasoningParser`; `tensorrt_llm/serve/postprocess_handlers.py` modified +46/-5 (51 lines); hunks: -1,10 +1,12; -36,6 +38,9 @@ class ChatPostprocArgs(PostprocArgs):; symbols: ChatPostprocArgs, from_request, create_logprobs, apply_reasoning_parser, touching `ChatPostprocArgs, from_request, create_logprobs`; `tensorrt_llm/commands/serve.py` modified +12/-1 (13 lines); hunks: -15,6 +15,7; -33,6 +34,7 @@ def get_llm_args(model: str,; symbols: get_llm_args, launch_server, serve, touching `get_llm_args, launch_server, serve`.
- Code diff details:
  - `examples/serve/deepseek_r1_reasoning_parser.sh` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tensorrt_llm/llmapi/reasoning_parser.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: ReasoningParserResult, __init__, BaseReasoningParser, parse
  - `tensorrt_llm/serve/postprocess_handlers.py` modified +46/-5 (51 lines); hunks: -1,10 +1,12; -36,6 +38,9 @@ class ChatPostprocArgs(PostprocArgs):; symbols: ChatPostprocArgs, from_request, create_logprobs, apply_reasoning_parser
  - `tensorrt_llm/commands/serve.py` modified +12/-1 (13 lines); hunks: -15,6 +15,7; -33,6 +34,7 @@ def get_llm_args(model: str,; symbols: get_llm_args, launch_server, serve
  - `tensorrt_llm/llmapi/llm_args.py` modified +5/-0 (5 lines); hunks: -921,6 +921,11 @@ class LlmArgs(BaseModel):; symbols: LlmArgs
- Key code excerpts:

```diff
diff -- examples/serve/deepseek_r1_reasoning_parser.sh
@@ -0,0 +1,11 @@
+#! /usr/bin/env bash
+trtllm-serve \
+    deepseek-ai/DeepSeek-R1 \
+    --host localhost --port 8000 \
+    --backend pytorch \
+    --max_batch_size 161 --max_num_tokens 1160 \
diff -- tensorrt_llm/llmapi/reasoning_parser.py
@@ -0,0 +1,95 @@
+from abc import ABC, abstractmethod
+from dataclasses import dataclass
+from typing import Dict, Optional
+@dataclass
+class ReasoningParserResult:
+    def __init__(self,
diff -- tensorrt_llm/serve/postprocess_handlers.py
@@ -1,10 +1,12 @@
```

- Extracted files (not manually reviewed):
  - docs: `examples/serve/deepseek_r1_reasoning_parser.sh` added +11/-0
  - runtime: `tensorrt_llm/llmapi/reasoning_parser.py` added +95/-0; `tensorrt_llm/serve/postprocess_handlers.py` modified +46/-5; `tensorrt_llm/commands/serve.py` modified +12/-1; `tensorrt_llm/llmapi/llm_args.py` modified +5/-0; `tensorrt_llm/serve/openai_protocol.py` modified +2/-0; `tensorrt_llm/serve/openai_server.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/test_e2e.py`, `tests/integration/test_lists/test-db/l0_a10.yml`, `tests/unittest/llmapi/apps/_test_openai_reasoning.py`, `tests/unittest/llmapi/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4123 - [TRTLLM-3330][feat] Support DeepSeek-R1 W4A8 on Hopper

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4123
- Status/date: merged / 2025-05-14
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `20b42912cef7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 21 files, +1235/-120, 1826 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +77/-8 (85 lines); hunks: -32,21 +32,22 @@ Please refer to [this guide](https://nvidia.github.io/Tensor...; -566,6 +567,74 @@ pytorch_backend_config:; `tensorrt_llm/_torch/models/modeling_utils.py` modified +8/-2 (10 lines); hunks: -418,7 +418,7 @@ def __post_init__(self):; -438,7 +438,13 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`; `tensorrt_llm/layers/moe.py` modified +2/-2 (4 lines); hunks: -500,8 +500,8 @@ def __init__(self, in_features: int, out_features: int,; symbols: __init__, touching `__init__`; `tensorrt_llm/_torch/modules/fused_moe.py` modified +244/-3 (247 lines); hunks: -8,6 +8,7; -398,7 +399,9 @@ def _check_configs(self):; symbols: _check_configs, setup_quant_scales, is_trtllm, get_quant_scales, touching `_check_configs, setup_quant_scales, is_trtllm`.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +77/-8 (85 lines); hunks: -32,21 +32,22 @@ Please refer to [this guide](https://nvidia.github.io/Tensor...; -566,6 +567,74 @@ pytorch_backend_config:
  - `tensorrt_llm/_torch/models/modeling_utils.py` modified +8/-2 (10 lines); hunks: -418,7 +418,7 @@ def __post_init__(self):; -438,7 +438,13 @@ def __post_init__(self):; symbols: __post_init__
  - `tensorrt_llm/layers/moe.py` modified +2/-2 (4 lines); hunks: -500,8 +500,8 @@ def __init__(self, in_features: int, out_features: int,; symbols: __init__
  - `tensorrt_llm/_torch/modules/fused_moe.py` modified +244/-3 (247 lines); hunks: -8,6 +8,7; -398,7 +399,9 @@ def _check_configs(self):; symbols: _check_configs, setup_quant_scales, is_trtllm, get_quant_scales
  - `cpp/tensorrt_llm/kernels/cutlass_kernels/python/generate_kernels.py` modified +102/-35 (137 lines); hunks: -206,38 +206,47 @@ def instantiate_operation_tma_warp_specialized(operation):; -510,9 +519,57 @@ def generate_sm90_grouped_gemm_operations(is_arch_enabled):; symbols: instantiate_operation_tma_warp_specialized, generate_sm90_grouped_gemm_operations, generate_sm90_mixed_type_grouped_gemm_operations, generate_sm90_operations
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -32,21 +32,22 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/
+    - [W4AFP8](#w4afp8)
-The minimum hardware requirements for running DeepSeek V3/R1 FP8&FP4 are listed as follows.
+The minimum hardware requirements for running DeepSeek V3/R1 at FP8/FP4/W4A8 are listed as follows.
-| GPU  | DeepSeek-V3/R1 FP8 | DeepSeek-V3/R1 FP4 |
-| -------- | ------- | -- |
-| H100 80GB | 16 | N/A |
diff -- tensorrt_llm/_torch/models/modeling_utils.py
@@ -418,7 +418,7 @@ def __post_init__(self):
-                            if prefix_name + '.gate_proj' in n:
+                            if prefix_name + '.gate_proj' in n or prefix_name + '.gate_up_proj' in n:
@@ -438,7 +438,13 @@ def __post_init__(self):
-                # TODO: support MLA
+                elif hasattr(module, 'fused_a'):
+                    # DeepseekV3Attention
diff -- tensorrt_llm/layers/moe.py
@@ -500,8 +500,8 @@ def __init__(self, in_features: int, out_features: int,
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +77/-8
  - runtime: `tensorrt_llm/_torch/models/modeling_utils.py` modified +8/-2; `tensorrt_llm/layers/moe.py` modified +2/-2; `tensorrt_llm/_torch/modules/fused_moe.py` modified +244/-3; `cpp/tensorrt_llm/kernels/cutlass_kernels/python/generate_kernels.py` modified +102/-35; `cpp/tensorrt_llm/kernels/preQuantScaleKernel.cu` modified +123/-0; `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp` modified +74/-23
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modules/test_fused_moe.py`, `tests/unittest/trt/quantization/test_moe_weight_only_groupwise_quant_matmul.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4338 - [doc] Add tensorrtllm_backend serving documentation in the Deepseek-V3 README

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4338
- Status/date: merged / 2025-05-15
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `efe0972efb71`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +24/-0, 45 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +24/-0 (24 lines); hunks: -24,6 +24,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...; -226,6 +228,7 @@ trtllm-eval --model \.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +24/-0 (24 lines); hunks: -24,6 +24,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...; -226,6 +228,7 @@ trtllm-eval --model \
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -24,6 +24,8 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/
+    - [Use trtllm-serve](#use-trtllm-serve)
+    - [Use tensorrtllm_backend for triton inference server (Experimental)](#use-tensorrtllm_backend-for-triton-inference-server-experimental)
@@ -226,6 +228,7 @@ trtllm-eval --model  <YOUR_MODEL_DIR> \
+### Use trtllm-serve
@@ -278,6 +281,27 @@ curl http://localhost:8000/v1/completions \
+### Use tensorrtllm_backend for triton inference server (Experimental)
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +24/-0
- Risk and verification: This is mostly docs/examples in `examples/models/core/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #3952 - [https://nvbugs/5123103][fix] Fix torch compile for DeepSeekV3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/3952
- Status/date: merged / 2025-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `58e405624aa9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 21 files, +439/-231, 1192 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-7 (14 lines); hunks: -633,19 +633,19 @@ def _compute_mlp_tp_size(self, intermediate_size: int,; symbols: _compute_mlp_tp_size, _enable_min_latency_mode, touching `_compute_mlp_tp_size, _enable_min_latency_mode`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-7 (14 lines); hunks: -633,19 +633,19 @@ def _compute_mlp_tp_size(self, intermediate_size: int,; symbols: _compute_mlp_tp_size, _enable_min_latency_mode
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -633,19 +633,19 @@ def _compute_mlp_tp_size(self, intermediate_size: int,
-            mlp_tp_size = math.gcd(
-                math.gcd(
-                    intermediate_size // block_size,
-                    self.mapping.tp_size,
-                ),
-                self.mapping.gpus_per_node,  # Avoid costly inter-node TP
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-7
- Risk and verification: The diff ships test coverage in `tests/integration/defs/.test_durations`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/examples_test_list.txt`, `tests/integration/test_lists/qa/llm_release_rtx_pro_6000.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4522 - test: conditional disagg and cache aware balancing for deepseek v3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4522
- Status/date: merged / 2025-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml`; associated commits `580a92521e94`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +233/-16, 374 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml` added +35/-0 (35 lines); hunks: -0,0 +1,35; `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml` added +34/-0 (34 lines); hunks: -0,0 +1,34; `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` added +31/-0 (31 lines); hunks: -0,0 +1,31.
- Code diff details:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml` added +35/-0 (35 lines); hunks: -0,0 +1,35
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml` added +34/-0 (34 lines); hunks: -0,0 +1,34
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` added +31/-0 (31 lines); hunks: -0,0 +1,31
- Key code excerpts:

```diff
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml
@@ -0,0 +1,35 @@
+hostname: localhost
+port: 8000
+model: DeepSeek-V3-Lite/bf16
+backend: "pytorch"
+use_cuda_graph: False
+disable_overlap_scheduler: True
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml
@@ -0,0 +1,34 @@
+hostname: localhost
+port: 8000
+model: DeepSeek-V3-Lite/bf16
+backend: "pytorch"
+free_gpu_memory_fraction: 0.15
+conditional_disagg_config:
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml
@@ -0,0 +1,31 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml` added +35/-0; `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml` added +34/-0; `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` added +31/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml`, `tests/integration/defs/disaggregated/test_disaggregated.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4560 - Feat/ds r1 min latency opt round3, add router gemm, fused a gemm, PDL

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/4560
- Status/date: merged / 2025-06-14
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/CMakeLists.txt`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h` and 8 files; associated commits `b99c5ce8c19f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 22 files, +1469/-42, 1859 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +73/-9 (82 lines); hunks: -41,23 +41,25; -149,6 +151,57 @@ def forward(self, hidden_states: torch.Tensor, lm_head: Lin...; symbols: forward, DeepseekV3Linear, __init__, apply_linear, touching `forward, DeepseekV3Linear, __init__`; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` added +695/-0 (695 lines); hunks: -0,0 +1,695; symbols: Type, touching `Type`; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` added +241/-0 (241 lines); hunks: -0,0 +1,241; `cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` added +117/-0 (117 lines); hunks: -0,0 +1,117.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +73/-9 (82 lines); hunks: -41,23 +41,25; -149,6 +151,57 @@ def forward(self, hidden_states: torch.Tensor, lm_head: Lin...; symbols: forward, DeepseekV3Linear, __init__, apply_linear
  - `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` added +695/-0 (695 lines); hunks: -0,0 +1,695; symbols: Type
  - `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` added +241/-0 (241 lines); hunks: -0,0 +1,241
  - `cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` added +117/-0 (117 lines); hunks: -0,0 +1,117
  - `cpp/tensorrt_llm/thop/dsv3FusedAGemmOp.cpp` added +96/-0 (96 lines); hunks: -0,0 +1,96
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -41,23 +41,25 @@
+from tensorrt_llm.mapping import Mapping
-from ..models.modeling_utils import ModelConfig
+from ..models.modeling_utils import ModelConfig, QuantConfig
-from ..modules.linear import Linear
+from ..modules.linear import Linear, TensorParallelMode, WeightsLoadingConfig
+from ..peft.lora.layer import LoraLayer
diff -- cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
@@ -0,0 +1,695 @@
+/*
+ * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
+ * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
diff -- cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
@@ -0,0 +1,241 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +73/-9; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` added +695/-0; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` added +241/-0; `cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp` added +117/-0; `cpp/tensorrt_llm/thop/dsv3FusedAGemmOp.cpp` added +96/-0; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h` added +31/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/thop/test_dsv3_fused_a_gemm.py`, `tests/unittest/_torch/thop/test_dsv3_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5235 - Update DeepSeek R1 perf numbers to latest release/0.20 results

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5235
- Status/date: merged / 2025-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; associated commits `03f1a6a3d85a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +87/-23, 139 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +87/-23 (110 lines); hunks: -18,7 +18,10 @@ In this blog, we share the configurations and procedures abou...; -181,9 +184,68 @@ Total Token Throughput (tokens/sec): 414.0461.
- Code diff details:
  - `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +87/-23 (110 lines); hunks: -18,7 +18,10 @@ In this blog, we share the configurations and procedures abou...; -181,9 +184,68 @@ Total Token Throughput (tokens/sec): 414.0461
- Key code excerpts:

```diff
diff -- docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
@@ -18,7 +18,10 @@ In this blog, we share the configurations and procedures about how to reproduce
-    - [B200 max-throughput](#b200-max-throughput)
+    - [B200 max-throughput with FP8 KV](#b200-max-throughput-for-r1-0528-with-fp8-kv-cache)
+      - [Benchmark](#benchmark)
+      - [Expected Result Format](#expected-result-format)
+    - [B200 max-throughput with FP16 KV](#b200-max-throughput-for-r1-with-fp16-kv-cache)
@@ -181,9 +184,68 @@ Total Token Throughput (tokens/sec):              414.0461
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +87/-23
- Risk and verification: This is mostly docs/examples in `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #5600 - doc: Minor update to DeepSeek R1 best practice

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5600
- Status/date: merged / 2025-06-30
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; associated commits `2ce200fbbb66`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-9, 39 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +9/-9 (18 lines); hunks: -18,19 +18,19 @@ In this blog, we share the configurations and procedures abo...; -421,9 +421,9 @@ Generally, you should make sure that `max_batch_size` is not....
- Code diff details:
  - `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +9/-9 (18 lines); hunks: -18,19 +18,19 @@ In this blog, we share the configurations and procedures abo...; -421,9 +421,9 @@ Generally, you should make sure that `max_batch_size` is not...
- Key code excerpts:

```diff
diff -- docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
@@ -18,19 +18,19 @@ In this blog, we share the configurations and procedures about how to reproduce
-    - [B200 max-throughput with FP8 KV](#b200-max-throughput-for-r1-0528-with-fp8-kv-cache)
+    - [B200 max-throughput for R1-0528 with FP8 KV cache](#b200-max-throughput-for-r1-0528-with-fp8-kv-cache)
-    - [B200 max-throughput with FP16 KV](#b200-max-throughput-for-r1-with-fp16-kv-cache)
-      - [Benchmark](#benchmark)
-      - [Expected Result Format](#expected-result-format)
-    - [H200 min-latency](#h200-min-latency)
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +9/-9
- Risk and verification: This is mostly docs/examples in `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #5796 - doc: update cuda_graph_config usage part in DS R1 docs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/5796
- Status/date: merged / 2025-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`; associated commits `c8fa08da5cac`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +41/-32, 108 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +27/-27 (54 lines); hunks: -195,20 +195,20 @@ We are seeing meaningful speedup using FP8 KV cache, thus...; -262,19 +262,19 @@ python ${YOUR_WORK_PATH}/benchmarks/cpp/prepare_dataset.py \.
- Code diff details:
  - `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +27/-27 (54 lines); hunks: -195,20 +195,20 @@ We are seeing meaningful speedup using FP8 KV cache, thus...; -262,19 +262,19 @@ python ${YOUR_WORK_PATH}/benchmarks/cpp/prepare_dataset.py \
- Key code excerpts:

```diff
diff -- docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
@@ -195,20 +195,20 @@ We are seeing meaningful speedup using FP8 KV cache, thus refreshing the numbers
-use_cuda_graph: true
-cuda_graph_padding_enabled: true
-cuda_graph_batch_sizes:
-- 896
-- 512
-- 256
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md` modified +27/-27
- Risk and verification: The diff ships test coverage in `tests/integration/defs/perf/pytorch_model_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6486 - Deepseek R1 FP8 Support on Blackwell

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6486
- Status/date: merged / 2025-08-01
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `7bb0a78631de`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 18 files, +1487/-48, 1830 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +52/-7 (59 lines); hunks: -38,12 +38,15; -1244,7 +1247,7 @@ def load_kv_b_proj_and_k_b_proj_trans(module_name: str,; symbols: load_kv_b_proj_and_k_b_proj_trans, check_weight_dtype, load_kv_b_proj_and_k_b_proj_trans_dequant, touching `load_kv_b_proj_and_k_b_proj_trans, check_weight_dtype, load_kv_b_proj_and_k_b_proj_trans_dequant`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +52/-7 (59 lines); hunks: -38,12 +38,15; -1244,7 +1247,7 @@ def load_kv_b_proj_and_k_b_proj_trans(module_name: str,; symbols: load_kv_b_proj_and_k_b_proj_trans, check_weight_dtype, load_kv_b_proj_and_k_b_proj_trans_dequant
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -38,12 +38,15 @@
+from tensorrt_llm import logger
+from tensorrt_llm.quantization.utils.fp8_utils import (
+    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)
@@ -1244,7 +1247,7 @@ def load_kv_b_proj_and_k_b_proj_trans(module_name: str,
-            k_nope_weight_trans = k_nope_weight.transpose(2, 1)
+            k_nope_weight_trans = k_nope_weight.transpose(2, 1).contiguous()
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +52/-7
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/helpers.py`, `tests/unittest/_torch/modules/test_fused_moe.py`, `tests/unittest/_torch/thop/test_fp8_block_scale_gemm.py`, `tests/unittest/test_pip_install.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6200 - [https://nvbugs/5378031] [feat] Hopper W4A8 MoE supports ModelOpt ckpt for PyT backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6200
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `12c66f7610c5`, `2198587b35e5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +365/-81, 709 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-2 (11 lines); hunks: -57,7 +57,8; -454,7 +455,13 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-2 (11 lines); hunks: -57,7 +57,8; -454,7 +455,13 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -57,7 +57,8 @@
-from ..modules.fused_moe import (DeepSeekV3MoeRoutingMethod, TRTLLMGenFusedMoE,
+from ..modules.fused_moe import (DeepSeekV3MoeRoutingMethod,
+                                 MoEWeightLoadingMode, TRTLLMGenFusedMoE,
@@ -454,7 +455,13 @@ def __init__(self,
-            layer_idx=layer_idx)
+            layer_idx=layer_idx,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-2
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_dgx_h100.yml`, `tests/unittest/_torch/modules/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6698 - [TRTLLM-6853][feat] refactor deepseekv3 model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6698
- Status/date: merged / 2025-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `4aed7a7d1937`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +100/-107, 407 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +29/-68 (97 lines); hunks: -53,7 +53,6; -65,10 +64,10; symbols: compute_routed_output, forward, _compute_shared_output, forward_MoE, touching `compute_routed_output, forward, _compute_shared_output`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +29/-68 (97 lines); hunks: -53,7 +53,6; -65,10 +64,10; symbols: compute_routed_output, forward, _compute_shared_output, forward_MoE
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -53,7 +53,6 @@
-from ..models.modeling_utils import ModelConfig, QuantConfig
@@ -65,10 +64,10 @@
-from ..speculative import MTPEagleWorker, MTPSpecMetadata, MTPWorker
+from ..speculative import MTPSpecMetadata, SpecMetadata
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             EagerFusionConfig, filter_weights,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +29/-68
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tensorrt_llm/_torch/models/modeling_speculative.py`, `tensorrt_llm/_torch/speculative/interface.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7238 - [None][fix] Remove and fuse some element-wise ops in the ds-r1-fp8 model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7238
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `e12868bc00a8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +82/-50, 174 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +8/-2 (10 lines); hunks: -279,10 +279,16 @@ def __init__(; symbols: __init__, noaux_tc, get_scores, touching `__init__, noaux_tc, get_scores`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +8/-2 (10 lines); hunks: -279,10 +279,16 @@ def __init__(; symbols: __init__, noaux_tc, get_scores
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -279,10 +279,16 @@ def __init__(
-    def noaux_tc(self, logits, e_score_correction_bias):
-        n_group = self.n_group
+    @torch.compile(options={"max-autotune": True})
+    def get_scores(self, logits, e_score_correction_bias):
+        return scores, scores_with_bias
+    def noaux_tc(self, logits, e_score_correction_bias):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tensorrt_llm/_torch/modules/attention.py`, `tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6886 - [https://nvbugs/5445466][fix] Bypass MLP TP split for MNNVL in DeepSeek V3 to avoid hanging.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/6886
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `b093d94d3456`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +27/-11, 87 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +15/-9 (24 lines); hunks: -611,7 +611,8 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; -642,6 +643,10 @@ def __init__(self, model_config: ModelConfig[PretrainedConf...; symbols: __init__, _get_decoder_layer_quant_config, _compute_mlp_tp_size, touching `__init__, _get_decoder_layer_quant_config, _compute_mlp_tp_size`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +15/-9 (24 lines); hunks: -611,7 +611,8 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; -642,6 +643,10 @@ def __init__(self, model_config: ModelConfig[PretrainedConf...; symbols: __init__, _get_decoder_layer_quant_config, _compute_mlp_tp_size
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -611,7 +611,8 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],
-        config = model_config.pretrained_config
+        self.config = model_config.pretrained_config
+        config = self.config
@@ -642,6 +643,10 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],
+        self.allreduce = AllReduce(mapping=model_config.mapping,
+                                   strategy=model_config.allreduce_strategy,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +15/-9
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/distributed/ops.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7123 - [None][fix] DeepSeek-R1 W4A8 weight loading issue; fixes regression from #6200

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7123
- Status/date: merged / 2025-09-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `12c66f7610c5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +18/-4, 50 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +18/-4 (22 lines); hunks: -45,6 +45,7; -456,10 +457,13 @@ def __init__(self,; symbols: __init__, _compute_shared_expert_tp_size, _get_experts_quant_config, compute_routed_output, touching `__init__, _compute_shared_expert_tp_size, _get_experts_quant_config`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +18/-4 (22 lines); hunks: -45,6 +45,7; -456,10 +457,13 @@ def __init__(self,; symbols: __init__, _compute_shared_expert_tp_size, _get_experts_quant_config, compute_routed_output
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -45,6 +45,7 @@
+from tensorrt_llm.quantization.mode import QuantAlgo
@@ -456,10 +457,13 @@ def __init__(self,
-            weight_loading_mode=(MoEWeightLoadingMode.W4A8_CUSTOM
-                                 if model_config.quant_config.quant_mode.
-                                 is_int4_weight_only_per_group() else
-                                 MoEWeightLoadingMode.VANILLA))
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7761 - [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for 384 experts (MoE TRTLLM backend)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7761
- Status/date: merged / 2025-10-20
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `c8b9998acb8b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +1013/-854, 2873 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +22/-5 (27 lines); hunks: -569,9 +569,6 @@ def get_scores(logits, e_score_correction_bias):; -580,7 +577,27 @@ def noaux_tc(self, logits, e_score_correction_bias):; symbols: get_scores, noaux_tc, apply, touching `get_scores, noaux_tc, apply`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +22/-5 (27 lines); hunks: -569,9 +569,6 @@ def get_scores(logits, e_score_correction_bias):; -580,7 +577,27 @@ def noaux_tc(self, logits, e_score_correction_bias):; symbols: get_scores, noaux_tc, apply
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -569,9 +569,6 @@ def get_scores(logits, e_score_correction_bias):
-        scores, scores_with_bias = Deepseekv3RoutingImpl.get_scores(
-            logits, e_score_correction_bias)
-        scores_shape = list(scores_with_bias.shape)
@@ -580,7 +577,27 @@ def noaux_tc(self, logits, e_score_correction_bias):
+        _, num_experts = logits.shape
+        if self.n_group > 1:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +22/-5
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/kernels/routing/routingDeepSeekTest.cpp`, `cpp/tests/unit_tests/kernels/routing/routingRenormalizeTest.cpp`, `tests/unittest/_torch/thop/parallel/test_moe.py`, `tests/unittest/_torch/thop/parallel/test_noaux_tc.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8405 - [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8405
- Status/date: merged / 2025-10-24
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/configs/deepseek_v3.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `e47c787dd702`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 43 files, +4914/-153, 5772 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: DeepseekV3Config, __init__, touching `DeepseekV3Config, __init__`; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0 (1 lines); hunks: -1468,6 +1468,7 @@ def forward(; symbols: forward, DeepseekV3ForCausalLM, touching `forward, DeepseekV3ForCausalLM`.
- Code diff details:
  - `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: DeepseekV3Config, __init__
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0 (1 lines); hunks: -1468,6 +1468,7 @@ def forward(; symbols: forward, DeepseekV3ForCausalLM
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/configs/deepseek_v3.py
@@ -0,0 +1,103 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers.configuration_utils import PretrainedConfig
+from transformers.utils import logging
+logger = logging.get_logger(__name__)
+# This is a temporary workaround for DeepSeek-V3.2 model as HF does not support it yet
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -1468,6 +1468,7 @@ def forward(
+@register_auto_model("DeepseekV32ForCausalLM")
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gpqa_diamond.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9141 - [None][doc] Add DeepSeek-V3.2-Exp document

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9141
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `25bd2e691790`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +21/-10, 88 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +21/-10 (31 lines); hunks: -1,7 +1,8; -14,7 +15,7 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT....
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +21/-10 (31 lines); hunks: -1,7 +1,8; -14,7 +15,7 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -1,7 +1,8 @@
-# DeepSeek‑V3 and DeepSeek-R1
+# DeepSeek‑V3, DeepSeek-R1, and DeepSeek-V3.2-Exp
+This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1/DeepSeek-V3.2-Exp models using NVIDIA's TensorRT LLM framework with the PyTorch backend.
+**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT LLM. DeepSeek-V3.2-Exp features DeepSeek Spa
-This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1 models using NVIDIA's TensorRT LLM framework with the PyTorch backend.
-**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT-LLM, for brevity we only provide one model e
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +21/-10
- Risk and verification: This is mostly docs/examples in `examples/models/core/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9217 - [None][chore] fix a deepseekv3 error when debug mode is on

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9217
- Status/date: merged / 2025-11-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `07343bb11c67`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +5/-5, 24 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +5/-5 (10 lines); hunks: -640,18 +640,18 @@ def __init__(; symbols: __init__, get_scores, noaux_tc, touching `__init__, get_scores, noaux_tc`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +5/-5 (10 lines); hunks: -640,18 +640,18 @@ def __init__(; symbols: __init__, get_scores, noaux_tc
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -640,18 +640,18 @@ def __init__(
-        return scores, scores_with_bias
-    def noaux_tc(self, logits, e_score_correction_bias):
-        n_group = self.n_group
+        return scores, scores_with_bias
+    def noaux_tc(self, logits, e_score_correction_bias):
+        n_group = self.n_group
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9231 - [None][doc] Update DS-R1 example doc

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9231
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `255e4ea9f00f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +20/-2, 78 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +20/-2 (22 lines); hunks: -245,6 +245,7 @@ cuda_graph_config:; -256,22 +257,34 @@ cat >./extra-llm-api-config.yml <<EOF.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +20/-2 (22 lines); hunks: -245,6 +245,7 @@ cuda_graph_config:; -256,22 +257,34 @@ cat >./extra-llm-api-config.yml <<EOF
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -245,6 +245,7 @@ cuda_graph_config:
+    enable_block_reuse: false
@@ -256,22 +257,34 @@ cat >./extra-llm-api-config.yml <<EOF
+  - 2048
+  - 384
+  - 192
+  - 160
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +20/-2
- Risk and verification: This is mostly docs/examples in `examples/models/core/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9383 - [None][feat] Add support for KVCache reuse for DSv32

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9383
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `356a52edf56e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +14/-38, 146 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +0/-9 (9 lines); hunks: -881,12 +881,3 @@ python quickstart_advanced.py --model_dir --enable_chunked_...; `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8 (12 lines); hunks: -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; symbols: prepare, __init__, touching `prepare, __init__`; `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8 (9 lines); hunks: -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm); `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0 (4 lines); hunks: -183,6 +183,10 @@ class BlockRange; symbols: BlockRange, touching `BlockRange`.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +0/-9 (9 lines); hunks: -881,12 +881,3 @@ python quickstart_advanced.py --model_dir --enable_chunked_...
  - `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8 (12 lines); hunks: -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; symbols: prepare, __init__
  - `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8 (9 lines); hunks: -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm)
  - `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0 (4 lines); hunks: -183,6 +183,10 @@ class BlockRange; symbols: BlockRange
  - `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` modified +1/-1 (2 lines); hunks: -806,7 +806,7 @@ class CacheReceiver::Impl; symbols: CacheReceiver
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -881,12 +881,3 @@ python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --enable_chunked_pref
-## Known Issues
-- Support for KV Cache Reuse and Chunked Prefill in DeepSeek-V3.2-Exp is currently under development. When running `quickstart_advanced.py`, please include `--disable_kv_cache_reu
-'''
-kv_cache_config:
-    enable_block_reuse: false
-    tokens_per_block: 64
diff -- tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):
-                if len(chunk_groups) > 1:
+                if len(chunk_groups
+                       ) > 1 or metadata.enable_context_mla_with_cached_kv:
-                    # Single chunk - use non-chunked fallback path
@@ -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):
-        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
diff -- cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
@@ -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm)
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +0/-9
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8; `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8; `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0; `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9661 - [TRTLLM-9506][fix] Fix AR for DeepSeek-R1 2 model path

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9661
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `1c7b7cdd4755`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +12/-4, 61 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-4 (13 lines); hunks: -1331,13 +1331,13 @@ def _run_MoE(hidden_states, hidden_states_fp4, do_finali...; -1455,6 +1455,7 @@ def forward(; symbols: _run_MoE, forward, norm_hidden, touching `_run_MoE, forward, norm_hidden`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-4 (13 lines); hunks: -1331,13 +1331,13 @@ def _run_MoE(hidden_states, hidden_states_fp4, do_finali...; -1455,6 +1455,7 @@ def forward(; symbols: _run_MoE, forward, norm_hidden
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -1331,13 +1331,13 @@ def _run_MoE(hidden_states, hidden_states_fp4, do_finalize):
-            if spec_metadata is not None and spec_metadata.is_layer_capture(
-                    self.layer_idx):
-                spec_metadata.maybe_capture_hidden_states(
-                    self.layer_idx, hidden_states, residual)
+            if spec_metadata is not None and spec_metadata.is_layer_capture(
+                    self.layer_idx):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tensorrt_llm/_torch/models/modeling_speculative.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9799 - [None][fix] Fix PDL in TRTLLM MOE for dsv3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9799
- Status/date: merged / 2025-12-09
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu`; associated commits `1c4dacb19a52`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +7/-7, 56 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` modified +3/-3 (6 lines); hunks: -296,7 +296,7 @@ public:; -601,8 +601,8 @@ __global__ __launch_bounds__(256, 1) void fused_a_gemm_kernel(; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` modified +2/-2 (4 lines); hunks: -74,7 +74,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel(f...; -167,7 +167,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel....
- Code diff details:
  - `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` modified +3/-3 (6 lines); hunks: -296,7 +296,7 @@ public:; -601,8 +601,8 @@ __global__ __launch_bounds__(256, 1) void fused_a_gemm_kernel(
  - `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` modified +2/-2 (4 lines); hunks: -74,7 +74,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel(f...; -167,7 +167,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel...
- Key code excerpts:

```diff
diff -- cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
@@ -296,7 +296,7 @@ public:
-        asm volatile("griddepcontrol.wait;");
+        cudaGridDependencySynchronize();
@@ -601,8 +601,8 @@ __global__ __launch_bounds__(256, 1) void fused_a_gemm_kernel(
-    asm volatile("griddepcontrol.wait;");
-    asm volatile("griddepcontrol.launch_dependents;");
+    cudaGridDependencySynchronize();
diff -- cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
@@ -74,7 +74,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel(float* out, T const
-    asm volatile("griddepcontrol.wait;");
+    cudaGridDependencySynchronize();
@@ -167,7 +167,7 @@ __global__ __launch_bounds__(128, 1) void router_gemm_kernel(float* out, T const
-    asm volatile("griddepcontrol.launch_dependents;");
+    cudaTriggerProgrammaticLaunchCompletion();
```

- Extracted files (not manually reviewed):
  - runtime: `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` modified +3/-3; `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` modified +2/-2
- Risk and verification: Runtime changes concentrate in `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu`, `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu`, `cpp/tensorrt_llm/kernels/noAuxTcKernels.cu`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10010 - [TRTLLM-9604][feat] DS R1 & V3.1 tool parser

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10010
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja`, `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py`, `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py`; associated commits `ac03915dc382`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +767/-34, 842 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: DeepSeekV3Parser, __init__, has_tool_call, detect_and_parse, touching `DeepSeekV3Parser, __init__, has_tool_call`; `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: DeepSeekV31Parser, __init__, has_tool_call, detect_and_parse, touching `DeepSeekV31Parser, __init__, has_tool_call`; `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py` added +207/-0 (207 lines); hunks: -0,0 +1,207; symbols: DeepSeekV3Parser, __init__, has_tool_call, detect_and_parse
  - `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: DeepSeekV31Parser, __init__, has_tool_call, detect_and_parse
  - `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/deepseekv3_parser.py
@@ -0,0 +1,207 @@
+# Adapted from https://github.com/sgl-project/sglang/blob/94e1251131ca27260cb0e8938aeb7b4a4e630b19/python/sglang/srt/function_call/deepseekv3_detector.py
+import json
+import re
+from typing import List
+from tensorrt_llm.logger import logger
+from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
diff -- tensorrt_llm/serve/tool_parser/deepseekv31_parser.py
@@ -0,0 +1,203 @@
+# Adapted from https://github.com/sgl-project/sglang/blob/94e1251131ca27260cb0e8938aeb7b4a4e630b19/python/sglang/srt/function_call/deepseekv31_detector.py
+import json
+import re
+from typing import List
+from tensorrt_llm.logger import logger
+from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
diff -- examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py` added +207/-0; `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py` added +203/-0
  - docs: `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja` added +91/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/llmapi/apps/test_tool_parsers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10126 - [TRTLLM-9677][feat] Support DeepSeek-V3.2 tool parser

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10126
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py`; associated commits `0d2500c631d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +444/-0, 480 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: DeepSeekV32Parser, __init__, has_tool_call, _parse_parameters_from_xml, touching `DeepSeekV32Parser, __init__, has_tool_call`.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: DeepSeekV32Parser, __init__, has_tool_call, _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/deepseekv32_parser.py
@@ -0,0 +1,296 @@
+# Adapted from https://github.com/sgl-project/sglang/blob/0071fe9c407ad59f2803cc319e1bcaa3ac2021f1/python/sglang/srt/function_call/deepseekv32_detector.py
+import json
+import re
+from typing import List
+from tensorrt_llm.logger import logger
+from ..openai_protocol import ChatCompletionToolsParam as Tool
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/llmapi/apps/test_tool_parsers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11215 - [None][feat] Support mix quantization between shared experts and routed experts for dsv3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11215
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `72091b37e797`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +54/-7, 94 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +54/-7 (61 lines); hunks: -968,10 +968,19 @@ def __init__(self,; -981,11 +990,14 @@ def __init__(self,; symbols: __init__, _create_ideal_expert_load_balanced_logits, _get_shared_experts_quant_config, compute_routed_output, touching `__init__, _create_ideal_expert_load_balanced_logits, _get_shared_experts_quant_config`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +54/-7 (61 lines); hunks: -968,10 +968,19 @@ def __init__(self,; -981,11 +990,14 @@ def __init__(self,; symbols: __init__, _create_ideal_expert_load_balanced_logits, _get_shared_experts_quant_config, compute_routed_output
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -968,10 +968,19 @@ def __init__(self,
-        # FIXME: incompatible with mixed quantization mode (including excluding modules from quantization)
+        shared_quant_config = self._get_shared_experts_quant_config(
+            model_config, layer_idx)
+        shared_model_config = model_config
+        if shared_quant_config is not model_config.quant_config:
+            shared_model_config = copy.copy(model_config)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +54/-7
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11507 - [TRTLLM-11057][feat] Add Helix CP support for DSV3.2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11507
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `ac8bc6ed1112`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +178/-26, 363 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11 (24 lines); hunks: -764,6 +764,7 @@ def __init__(; -789,6 +790,7 @@ def __init__(; symbols: __init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits, touching `__init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11 (24 lines); hunks: -764,6 +764,7 @@ def __init__(; -789,6 +790,7 @@ def __init__(; symbols: __init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -764,6 +764,7 @@ def __init__(
+        mapping_with_cp: Optional[Mapping] = None,
@@ -789,6 +790,7 @@ def __init__(
+                         mapping_with_cp=mapping_with_cp,
@@ -1008,7 +1010,7 @@ def __init__(self,
-                dtype=dtype)
+                dtype=torch.float32)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/multi_gpu/cacheTransceiverTest.cpp`, `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12530 - [https://nvbugs/5879577][fix] Fix KeyError in DeepSeekV3Lite FP8 MTP weight loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12530
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `0620bf4c9172`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +27/-7, 105 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +27/-6 (33 lines); hunks: -334,6 +334,24 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; -343,14 +361,17 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; symbols: split_kv_b_proj, detect_shared_mtp_weights, touching `split_kv_b_proj, detect_shared_mtp_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +27/-6 (33 lines); hunks: -334,6 +334,24 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; -343,14 +361,17 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; symbols: split_kv_b_proj, detect_shared_mtp_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -334,6 +334,24 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,
+        def detect_shared_mtp_weights() -> bool:
+            # Detect if MTP layers share checkpoint weights (model requests more
+            # MTP layers than the checkpoint provides). In this case, multiple
+            # model MTP layers map to the same checkpoint layer via modulo, and
+            # mark_consumed must be skipped to avoid deleting weights that later
+            # MTP layers still need.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +27/-6
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13882 - [None][test] Add DSR1 B200 DISAGG to CI Perf Test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13882
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml`; associated commits `f819383a442b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +164/-14, 239 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108.
- Code diff details:
  - `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml` added +108/-0 (108 lines); hunks: -0,0 +1,108
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml
@@ -0,0 +1,108 @@
+metadata:
+  model_name: deepseek_r1_0528_fp4_v2
+  supported_gpus:
+  - B200
+hardware:
+  gpus_per_node: 8
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml` added +108/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200_multi_nodes_perf_sanity_node2_gpu16.yml`, `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/aggregated/deepseek_r1_fp4_v2_2_nodes_blackwell.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14856 - [None][test] Update DSV32 32k4k config to avoid timeout issue

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14856
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`; associated commits `aa4276d473ac`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +18/-9, 157 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -69,6 +69,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -69,6 +69,7 @@ worker_config:; `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -80,6 +80,7 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -80,6 +80,7 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -69,6 +69,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -69,6 +69,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -80,6 +80,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1 (4 lines); hunks: -17,7 +17,7 @@ slurm:; -80,6 +80,7 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml
@@ -17,7 +17,7 @@ slurm:
-  multi_round: 5
+  multi_round: 2
@@ -69,6 +69,7 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
@@ -105,5 +106,6 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml
@@ -17,7 +17,7 @@ slurm:
-  multi_round: 5
+  multi_round: 2
@@ -69,6 +69,7 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
@@ -105,5 +106,6 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
diff -- tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml
@@ -17,7 +17,7 @@ slurm:
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1; `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1; `tests/scripts/perf/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con2048_ctx1_dep4_gen1_dep32_eplb288_mtp1_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15205 - [None][test] Increase kv_transfer_timeout_ms for b200 deepseek-r1 disagg gen_only perf test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15205
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`; associated commits `2d196f7cdfe8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-1, 23 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml
@@ -63,6 +63,7 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
@@ -90,5 +91,6 @@ worker_config:
+      kv_transfer_timeout_ms: 600000
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16528 - [https://nvbugs/6329052][fix] Change the config for DSV3 disagg conditional test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16528
- Status/date: merged / 2026-07-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml`; associated commits `74739166e29e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +3/-2, 29 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` modified +3/-0 (3 lines); hunks: -2,9 +2,12 @@ hostname: localhost.
- Code diff details:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` modified +3/-0 (3 lines); hunks: -2,9 +2,12 @@ hostname: localhost
- Key code excerpts:

```diff
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml
@@ -2,9 +2,12 @@ hostname: localhost
+attn_backend: FLASHINFER
+model_kwargs:
+  num_hidden_layers: 4
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16669 - [TRTLLM-13948][test] Migrate DeepSeek R1/V3.2 disagg perf cases to transceiver V2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16669
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` and 60 files; associated commits `641bbf28ff29`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 76 files, +152/-0, 1136 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -88,5 +89,6 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -91,6 +92,7 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -88,5 +89,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -91,6 +92,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -61,6 +61,7 @@ worker_config:; -88,5 +89,6 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -61,6 +61,7 @@ worker_config:
+      transceiver_runtime: PYTHON
@@ -88,5 +89,6 @@ worker_config:
+      transceiver_runtime: PYTHON
diff -- tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml
@@ -63,6 +63,7 @@ worker_config:
+      transceiver_runtime: PYTHON
@@ -90,5 +91,6 @@ worker_config:
+      transceiver_runtime: PYTHON
diff -- tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -63,6 +63,7 @@ worker_config:
+      transceiver_runtime: PYTHON
@@ -90,5 +91,6 @@ worker_config:
+      transceiver_runtime: PYTHON
diff -- tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml
@@ -63,6 +63,7 @@ worker_config:
+      transceiver_runtime: PYTHON
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con2048_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_1k1k_con256_ctx1_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/b200_deepseek-r1-fp4_8k1k_con1536_ctx1_dep4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16908 - [TRTLLM-13948][feat] Set DeepSeekV3 to use Python KV-cache transceiver V2 by default

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16908
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`; associated commits `aa535e5d41b5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +32/-49, 170 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-18 (25 lines); hunks: -1897,25 +1897,14 @@ class DeepseekV3ForCausalLM(SpecDecOneEngineForCausalLM[...; symbols: DeepseekV3ForCausalLM, get_preferred_transceiver_runtime, is, __init__, touching `DeepseekV3ForCausalLM, get_preferred_transceiver_runtime, is`; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -64,6 +64,7 @@ worker_config:; -91,6 +92,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -75,6 +75,7 @@ worker_config:; -102,6 +103,7 @@ worker_config:.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-18 (25 lines); hunks: -1897,25 +1897,14 @@ class DeepseekV3ForCausalLM(SpecDecOneEngineForCausalLM[...; symbols: DeepseekV3ForCausalLM, get_preferred_transceiver_runtime, is, __init__
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -64,6 +64,7 @@ worker_config:; -91,6 +92,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -90,5 +91,6 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -75,6 +75,7 @@ worker_config:; -102,6 +103,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -74,6 +74,7 @@ worker_config:; -101,5 +102,6 @@ worker_config:
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -1897,25 +1897,14 @@ class DeepseekV3ForCausalLM(SpecDecOneEngineForCausalLM[DeepseekV3Model,
-        """GLM-5 family checkpoints default to the Python (v2) KV-cache transceiver.
-        This implementation class is shared by DeepSeek-V3/V3.2 and the GLM-5 family — both
-        GLM-5 and GLM-5.2 declare ``GlmMoeDsaForCausalLM`` / ``glm_moe_dsa`` — so the preference
-        is differentiated per checkpoint: only GLM checkpoints opt into the Python transceiver.
-        The MLA backbone transfers a large latent KV, which the Python transceiver handles better
-        in disaggregated serving. This is only adopted when the user leaves
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml
@@ -64,6 +64,7 @@ worker_config:
+      transceiver_runtime: CPP
@@ -91,6 +92,7 @@ worker_config:
+      transceiver_runtime: CPP
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml
@@ -63,6 +63,7 @@ worker_config:
+      transceiver_runtime: CPP
@@ -90,5 +91,6 @@ worker_config:
+      transceiver_runtime: CPP
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-18
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-r1-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17397 - [https://nvbugs/6523520][fix] Halve gb200 r1-fp4 128k8k con128 multi_round to fit perf-sanity budget

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17397
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`; associated commits `33c6270c355b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +1/-3, 25 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ slurm:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ slurm:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml
@@ -17,7 +17,7 @@ slurm:
-  multi_round: 2
+  multi_round: 1
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17427 - [https://nvbugs/6472256][fix] Fix disagg stress cluster flapping and DeepSeek R1 FP4 ctx OOM; add aiperf error-rate gate

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17427
- Status/date: merged / 2026-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml`; associated commits `66758429f7bb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +455/-20, 630 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml` modified +7/-0 (7 lines); hunks: -18,6 +18,13 @@ context_servers:; `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml` modified +6/-0 (6 lines); hunks: -22,6 +22,12 @@ context_servers:.
- Code diff details:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml` modified +7/-0 (7 lines); hunks: -18,6 +18,13 @@ context_servers:
  - `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml` modified +6/-0 (6 lines); hunks: -22,6 +22,12 @@ context_servers:
- Key code excerpts:

```diff
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml
@@ -18,6 +18,13 @@ context_servers:
+    # Chunk the MoE forward at half of max_num_tokens: the TRTLLM-Gen FP4
+    # MoE workspace for a full 16640-token chunk is a 7-8 GiB transient,
+    # which exceeds the headroom left after weights (~107.5 GiB/GPU) and the
+    # KV pool on 192GB B200 and intermittently OOMs the ctx worker under
+    # 512-concurrency 8k prefill load (memory estimation only observes
+    # ~10.8 GiB dynamic peak, so the KV pool leaves no slack for it).
diff -- tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml
@@ -22,6 +22,12 @@ context_servers:
+    # Chunk the MoE forward at half of max_num_tokens, mirroring the non-MTP
+    # config: the ctx sizing here is identical (16640 tokens, 0.8 KV fraction,
+    # TRTLLM-Gen FP4 MoE), so a full-size chunk carries the same 7-8 GiB
+    # workspace transient that OOMs the ctx worker on 192GB B200 under
+    # 512-concurrency 8k prefill load.
+    max_num_tokens: 8320
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml` modified +7/-0; `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml` modified +6/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/disaggregated/test_aiperf_gate.py`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp4_gentp4_deepseek_r1_v2_fp4_tllm_mtp.yaml`, `tests/integration/defs/disaggregated/test_disaggregated.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19332 - [None][test] Add coverage for DeepseekV32ForCausalLM

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19332
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py`; associated commits `a7eae4beee5b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +202/-0, 210 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: test_deepseek_v32_context_forward, fresh_metadata, touching `test_deepseek_v32_context_forward, fresh_metadata`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: test_deepseek_v32_context_forward, fresh_metadata
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv32.py
@@ -0,0 +1,201 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
