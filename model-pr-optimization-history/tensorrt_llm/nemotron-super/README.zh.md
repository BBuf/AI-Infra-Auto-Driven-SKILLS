# TensorRT-LLM Nemotron Super 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md` | [#14964](https://github.com/NVIDIA/TensorRT-LLM/pull/14964) |
| `examples/configs/curated/nemotron-3-super-throughput.yaml` | [#12129](https://github.com/NVIDIA/TensorRT-LLM/pull/12129) |
| `examples/configs/curated/nemotron-3-ultra-throughput.yaml` | [#14964](https://github.com/NVIDIA/TensorRT-LLM/pull/14964) |
| `examples/configs/curated/nemotron-super-marlin.yaml` | 无直接 PR 号提交 |
| `examples/models/core/nemotron/README_nano-v2-vl.md` | 无直接 PR 号提交 |
| `examples/models/core/nemotron/README_nemotron_super_v3.md` | [#12129](https://github.com/NVIDIA/TensorRT-LLM/pull/12129), [#12215](https://github.com/NVIDIA/TensorRT-LLM/pull/12215), [#14964](https://github.com/NVIDIA/TensorRT-LLM/pull/14964) |
| `examples/models/core/nemotron_nas/README.md` | [#3632](https://github.com/NVIDIA/TensorRT-LLM/pull/3632) |
| `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` | [#7589](https://github.com/NVIDIA/TensorRT-LLM/pull/7589), [#10118](https://github.com/NVIDIA/TensorRT-LLM/pull/10118), [#10347](https://github.com/NVIDIA/TensorRT-LLM/pull/10347), [#10754](https://github.com/NVIDIA/TensorRT-LLM/pull/10754), [#11405](https://github.com/NVIDIA/TensorRT-LLM/pull/11405), [#11601](https://github.com/NVIDIA/TensorRT-LLM/pull/11601), [#15573](https://github.com/NVIDIA/TensorRT-LLM/pull/15573), [#16833](https://github.com/NVIDIA/TensorRT-LLM/pull/16833), [#19597](https://github.com/NVIDIA/TensorRT-LLM/pull/19597) |
| `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py` | [#13968](https://github.com/NVIDIA/TensorRT-LLM/pull/13968) |
| `tensorrt_llm/_torch/models/modeling_nemotron.py` | 无直接 PR 号提交 |
| `tensorrt_llm/_torch/models/modeling_nemotron_h.py` | [#3430](https://github.com/NVIDIA/TensorRT-LLM/pull/3430), [#3646](https://github.com/NVIDIA/TensorRT-LLM/pull/3646), [#4494](https://github.com/NVIDIA/TensorRT-LLM/pull/4494), [#6334](https://github.com/NVIDIA/TensorRT-LLM/pull/6334), [#6866](https://github.com/NVIDIA/TensorRT-LLM/pull/6866), [#7589](https://github.com/NVIDIA/TensorRT-LLM/pull/7589), [#8697](https://github.com/NVIDIA/TensorRT-LLM/pull/8697), [#10118](https://github.com/NVIDIA/TensorRT-LLM/pull/10118), [#10347](https://github.com/NVIDIA/TensorRT-LLM/pull/10347), [#10754](https://github.com/NVIDIA/TensorRT-LLM/pull/10754), [#11131](https://github.com/NVIDIA/TensorRT-LLM/pull/11131), [#11273](https://github.com/NVIDIA/TensorRT-LLM/pull/11273), ... (26 total) |
| `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` | [#19146](https://github.com/NVIDIA/TensorRT-LLM/pull/19146), [#19151](https://github.com/NVIDIA/TensorRT-LLM/pull/19151), [#19597](https://github.com/NVIDIA/TensorRT-LLM/pull/19597), [#19648](https://github.com/NVIDIA/TensorRT-LLM/pull/19648), [#19702](https://github.com/NVIDIA/TensorRT-LLM/pull/19702) |
| `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` | [#3447](https://github.com/NVIDIA/TensorRT-LLM/pull/3447), [#4180](https://github.com/NVIDIA/TensorRT-LLM/pull/4180), [#4906](https://github.com/NVIDIA/TensorRT-LLM/pull/4906), [#6455](https://github.com/NVIDIA/TensorRT-LLM/pull/6455), [#8985](https://github.com/NVIDIA/TensorRT-LLM/pull/8985) |
| `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml` | [#19340](https://github.com/NVIDIA/TensorRT-LLM/pull/19340) |
| `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml` | [#13837](https://github.com/NVIDIA/TensorRT-LLM/pull/13837) |
| `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml` | [#13837](https://github.com/NVIDIA/TensorRT-LLM/pull/13837) |
| `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml` | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) |
| `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) |
| `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con12_ctx1_dep4_gen6_tep4_eplb0_mtp6_ccb-NIXL.yaml` | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) |
| `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) |
| `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con9832_ctx1_dep4_gen8_dep8_eplb0_mtp3_ccb-NIXL.yaml` | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) |
| `tests/scripts/perf-sanity/disaggregated/h200_nemotron-super-fp8_8k1k_con64_ctx1_tp2_gen1_tp2_eplb0_mtp0_ccb-UCX.yaml` | 无直接 PR 号提交 |
| `tests/scripts/perf/disaggregated/vr200_nemotron-ultra-v3-fp4_50k2k_con12_ctx1_dep4_gen6_tep4_eplb0_mtp6_ccb-NIXL.yaml` | 无直接 PR 号提交 |
| `tests/scripts/perf/disaggregated/vr200_nemotron-ultra-v3-fp4_50k2k_con178_ctx5_dep4_gen1_dep4_eplb0_mtp6_ccb-NIXL.yaml` | 无直接 PR 号提交 |
| `tests/scripts/perf/disaggregated/vr200_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` | 无直接 PR 号提交 |
| `tests/scripts/perf/disaggregated/vr200_nemotron-ultra-v3-fp4_8k64k_con64_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | 无直接 PR 号提交 |
| `tests/unittest/_torch/modeling/test_modeling_nemotron.py` | 无直接 PR 号提交 |
| `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` | [#3430](https://github.com/NVIDIA/TensorRT-LLM/pull/3430), [#4444](https://github.com/NVIDIA/TensorRT-LLM/pull/4444), [#4954](https://github.com/NVIDIA/TensorRT-LLM/pull/4954), [#5097](https://github.com/NVIDIA/TensorRT-LLM/pull/5097), [#5646](https://github.com/NVIDIA/TensorRT-LLM/pull/5646), [#6334](https://github.com/NVIDIA/TensorRT-LLM/pull/6334), [#6485](https://github.com/NVIDIA/TensorRT-LLM/pull/6485), [#6996](https://github.com/NVIDIA/TensorRT-LLM/pull/6996), [#9993](https://github.com/NVIDIA/TensorRT-LLM/pull/9993), [#12980](https://github.com/NVIDIA/TensorRT-LLM/pull/12980), [#17601](https://github.com/NVIDIA/TensorRT-LLM/pull/17601), [#18888](https://github.com/NVIDIA/TensorRT-LLM/pull/18888), ... (14 total) |
| `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` | [#19558](https://github.com/NVIDIA/TensorRT-LLM/pull/19558), [#19597](https://github.com/NVIDIA/TensorRT-LLM/pull/19597) |
| `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` | [#19146](https://github.com/NVIDIA/TensorRT-LLM/pull/19146), [#19597](https://github.com/NVIDIA/TensorRT-LLM/pull/19597), [#19702](https://github.com/NVIDIA/TensorRT-LLM/pull/19702), [#19741](https://github.com/NVIDIA/TensorRT-LLM/pull/19741) |
| `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` | [#3447](https://github.com/NVIDIA/TensorRT-LLM/pull/3447), [#5202](https://github.com/NVIDIA/TensorRT-LLM/pull/5202), [#17858](https://github.com/NVIDIA/TensorRT-LLM/pull/17858) |
| `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py` | [#19168](https://github.com/NVIDIA/TensorRT-LLM/pull/19168) |
| `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` | [#19146](https://github.com/NVIDIA/TensorRT-LLM/pull/19146), [#19648](https://github.com/NVIDIA/TensorRT-LLM/pull/19648), [#19702](https://github.com/NVIDIA/TensorRT-LLM/pull/19702) |
| `tests/unittest/_torch/models/test_nemotron_h_puzzle.py` | 无直接 PR 号提交 |
| `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py` | [#19151](https://github.com/NVIDIA/TensorRT-LLM/pull/19151) |
| `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py` | [#19151](https://github.com/NVIDIA/TensorRT-LLM/pull/19151) |
| `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` | [#12154](https://github.com/NVIDIA/TensorRT-LLM/pull/12154), [#19151](https://github.com/NVIDIA/TensorRT-LLM/pull/19151) |
| `tests/unittest/_torch/multimodal/test_nemotron_h_multimodal_encoder_groups.py` | [#19146](https://github.com/NVIDIA/TensorRT-LLM/pull/19146) |

## PR 覆盖总览

- git 追溯 PR 数: 60
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 60
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-10 | [#3447](https://github.com/NVIDIA/TensorRT-LLM/pull/3447) | merged | chore: Rename nvsmall to nemotron nas | `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` |
| 2025-04-16 | [#3430](https://github.com/NVIDIA/TensorRT-LLM/pull/3430) | merged | feat: Nemotron-H model support | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-04-17 | [#3646](https://github.com/NVIDIA/TensorRT-LLM/pull/3646) | merged | Fix rotary_emb param in NemotronH attention | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2025-04-20 | [#3632](https://github.com/NVIDIA/TensorRT-LLM/pull/3632) | merged | Update Nemotron Super and Ultra in Supported Models and add an example | `examples/models/core/nemotron_nas/README.md`, `examples/pytorch/README.md` |
| 2025-05-18 | [#4180](https://github.com/NVIDIA/TensorRT-LLM/pull/4180) | merged | add changes for fp8, nemotron-nas, API | `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` |
| 2025-05-20 | [#4444](https://github.com/NVIDIA/TensorRT-LLM/pull/4444) | merged | [TRTLLM-5085][fix] Nemotron H correctness test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-06-01 | [#4494](https://github.com/NVIDIA/TensorRT-LLM/pull/4494) | merged | [TRTLLM-4783][feat] Mamba2 kernel updates for Nemotron-H | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2025-06-10 | [#4954](https://github.com/NVIDIA/TensorRT-LLM/pull/4954) | merged | [nvbug 5325284][fix] Increase Nemotron-H warmup request robustness | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`, `tensorrt_llm/_torch/pyexecutor/resource_manager.py` |
| 2025-06-12 | [#5097](https://github.com/NVIDIA/TensorRT-LLM/pull/5097) | merged | [test] Use LLM API for Nemotron-H correctness test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-06-15 | [#5202](https://github.com/NVIDIA/TensorRT-LLM/pull/5202) | merged | [fix][test] Speedup Nemotron NAS unittests | `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` |
| 2025-06-29 | [#4906](https://github.com/NVIDIA/TensorRT-LLM/pull/4906) | merged | feat: Add support for YARN in NemotronNAS models | `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` |
| 2025-07-03 | [#5646](https://github.com/NVIDIA/TensorRT-LLM/pull/5646) | merged | [TRTLLM-4923][feat] Enable CUDA graphs for Nemotron-H | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`, `tensorrt_llm/_torch/pyexecutor/resource_manager.py` |
| 2025-07-30 | [#6455](https://github.com/NVIDIA/TensorRT-LLM/pull/6455) | merged | [Perf]: Add residual, norm for nemotron_nas models | `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` |
| 2025-07-31 | [#6485](https://github.com/NVIDIA/TensorRT-LLM/pull/6485) | merged | [https://nvbugs/5404046][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-08-13 | [#6866](https://github.com/NVIDIA/TensorRT-LLM/pull/6866) | merged | [None][feat] Support running heterogeneous model execution for Nemotron-H | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2025-08-19 | [#6996](https://github.com/NVIDIA/TensorRT-LLM/pull/6996) | merged | [https://nvbugs/5458874][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-08-22 | [#6334](https://github.com/NVIDIA/TensorRT-LLM/pull/6334) | merged | [TRTLLM-4921][feat] Enable chunked prefill for Nemotron-H | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-09-09 | [#7589](https://github.com/NVIDIA/TensorRT-LLM/pull/7589) | merged | [None][fix] enable NvFP4/FP8 quantization for Nemotron-H architecture | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2025-10-28 | [#8697](https://github.com/NVIDIA/TensorRT-LLM/pull/8697) | merged | [None][fix] Properly raise error for nemotron H models | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2025-11-19 | [#8985](https://github.com/NVIDIA/TensorRT-LLM/pull/8985) | merged | [None][feat] add specdec to nemotron nas | `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` |
| 2025-12-18 | [#9993](https://github.com/NVIDIA/TensorRT-LLM/pull/9993) | merged | [https://nvbugs/5721644][fix] Update tests for nemotron_h | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2025-12-26 | [#10118](https://github.com/NVIDIA/TensorRT-LLM/pull/10118) | merged | [None][feat] Support multi-gpu running for nemotron-v3-nano and super | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-01-13 | [#10347](https://github.com/NVIDIA/TensorRT-LLM/pull/10347) | merged | [TRTLLM-10060][feat] Enable attention dp for Nemotron Super v3. | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2026-01-26 | [#10754](https://github.com/NVIDIA/TensorRT-LLM/pull/10754) | merged | [TRTLLM-10062][feat] Enable MTP for Nemotron Super | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2026-02-02 | [#11131](https://github.com/NVIDIA/TensorRT-LLM/pull/11131) | merged | [None][feat] Nemotron H: Eagle3 support | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-02-11 | [#11406](https://github.com/NVIDIA/TensorRT-LLM/pull/11406) | merged | [None][chore] Merge residual+hidden into layer norm at the end of each NemotronH MTP, and remove a % operation | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-02-12 | [#11273](https://github.com/NVIDIA/TensorRT-LLM/pull/11273) | merged | [None][feat] Optimize super-v3 nvfp4 for better perf | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-02-13 | [#11405](https://github.com/NVIDIA/TensorRT-LLM/pull/11405) | merged | [TRTLLM-10329][feat] Fix weight loading for Nemotron 3 models on DGX Spark | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2026-02-23 | [#11601](https://github.com/NVIDIA/TensorRT-LLM/pull/11601) | merged | [None][fix] Nemotron H fp4 and MTP | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2026-03-05 | [#11807](https://github.com/NVIDIA/TensorRT-LLM/pull/11807) | merged | [None][fix] Fix nemotron super MTP crash on SM90 | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-03-11 | [#11972](https://github.com/NVIDIA/TensorRT-LLM/pull/11972) | merged | [None][feat] Mamba optimization and mixed quantization support for nemotron-h | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-03-12 | [#12129](https://github.com/NVIDIA/TensorRT-LLM/pull/12129) | merged | [TRTLLM-10244][doc] Add deployment guide for Nemotron 3 Super | `examples/models/core/nemotron/README_nemotron_super_v3.md`, `examples/configs/curated/nemotron-3-super-throughput.yaml` |
| 2026-03-17 | [#12215](https://github.com/NVIDIA/TensorRT-LLM/pull/12215) | merged | [None][docs] Update nemotron 3 super deployment to include tool calling and reasoning parser | `examples/models/core/nemotron/README_nemotron_super_v3.md` |
| 2026-03-24 | [#12410](https://github.com/NVIDIA/TensorRT-LLM/pull/12410) | merged | [None][feat] Fuse all_reduce with norm for nemotron_h models | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-04-02 | [#12154](https://github.com/NVIDIA/TensorRT-LLM/pull/12154) | merged | [TRTLLM-10232][feat] Support LoRA adapter for nemotron-h models | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` |
| 2026-04-05 | [#12620](https://github.com/NVIDIA/TensorRT-LLM/pull/12620) | merged | [None][fix] Update codes to support nemotron-h corner cases | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-04-17 | [#12980](https://github.com/NVIDIA/TensorRT-LLM/pull/12980) | merged | [https://nvbugs/5626259][fix] Enable nemotron-h chunk prefill test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2026-05-01 | [#13160](https://github.com/NVIDIA/TensorRT-LLM/pull/13160) | merged | [None][chore] improve gemm perf for nemotron in spark | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-05-13 | [#13837](https://github.com/NVIDIA/TensorRT-LLM/pull/13837) | merged | [None][test] Add func and perf case of nemotron-3-Nano-Omni model on DGX-Spark | `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml`, `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml` |
| 2026-05-18 | [#13968](https://github.com/NVIDIA/TensorRT-LLM/pull/13968) | merged | [None][fix] Fix bugs related with nemotron-nas model | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py` |
| 2026-06-01 | [#14775](https://github.com/NVIDIA/TensorRT-LLM/pull/14775) | merged | [TRTLLM-12288][feat] Support Nemotron-H nvfp4 ckpt on Hopper | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-06-07 | [#14964](https://github.com/NVIDIA/TensorRT-LLM/pull/14964) | merged | [TRTLLM-13177][doc] Add Nemotron 3 Ultra doc | `examples/configs/curated/nemotron-3-ultra-throughput.yaml`, `examples/models/core/nemotron/README_nemotron_super_v3.md`, `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md` |
| 2026-06-24 | [#15294](https://github.com/NVIDIA/TensorRT-LLM/pull/15294) | merged | [https://nvbugs/6264844][fix] Fix wrong NCCL fallback in nemotron-h | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-07-01 | [#15573](https://github.com/NVIDIA/TensorRT-LLM/pull/15573) | merged | [None][feat] Support update weight for nemotron-h | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-07-22 | [#15582](https://github.com/NVIDIA/TensorRT-LLM/pull/15582) | merged | [None][feat] Support Nemotron dynamic-tree MTP decoding | `tensorrt_llm/_torch/models/modeling_nemotron_h.py` |
| 2026-07-28 | [#16833](https://github.com/NVIDIA/TensorRT-LLM/pull/16833) | merged | [None][fix] Fix nemotron-h quant and loading config | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` |
| 2026-08-15 | [#17609](https://github.com/NVIDIA/TensorRT-LLM/pull/17609) | merged | [None][test] Nemotron-Ultra-V3 perf-sanity cases (GB300); de-enroll DeepSeek-V3.2, Kimi-K2.5 & Llama cases | `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` |
| 2026-08-17 | [#17601](https://github.com/NVIDIA/TensorRT-LLM/pull/17601) | merged | [TRTLLM-15100][test] Prune Nemotron-H functional and un… | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2026-09-02 | [#17858](https://github.com/NVIDIA/TensorRT-LLM/pull/17858) | merged | [TRTLLM-15040][test] Prune legacy Llama and Nemotron tests | `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` |
| 2026-09-14 | [#18888](https://github.com/NVIDIA/TensorRT-LLM/pull/18888) | merged | [TRTLLM-15936][fix] Enable breakable prefill CUDA graphs (BCG) for Nemotron-H hybrid models | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`, `tensorrt_llm/_torch/compilation/utils.py` |
| 2026-09-15 | [#19168](https://github.com/NVIDIA/TensorRT-LLM/pull/19168) | merged | [https://nvbugs/6708111][fix] Load Nemotron-H saved by transformers>=5.13 | `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py`, `tensorrt_llm/tokenizer/tokenizer.py`, `tensorrt_llm/_torch/pyexecutor/config_utils.py` |
| 2026-09-17 | [#19146](https://github.com/NVIDIA/TensorRT-LLM/pull/19146) | merged | [None][chore] Name the Nemotron multimodal module for the family it serves instead of one of its three models | `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` |
| 2026-09-18 | [#19340](https://github.com/NVIDIA/TensorRT-LLM/pull/19340) | merged | [None][test] Add nemotron_3.5_lightning_30b_nvfp4 and nemotron_3.5_lightning_30b_bf16 func and perf cases on Spark | `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml` |
| 2026-09-20 | [#19335](https://github.com/NVIDIA/TensorRT-LLM/pull/19335) | merged | [https://nvbugs/6777501][fix] Fix nemotron breakable cuda graph test parity check | `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` |
| 2026-09-22 | [#19151](https://github.com/NVIDIA/TensorRT-LLM/pull/19151) | merged | [TRTLLM-15820][feat] Expose Nemotron-H VL LoRA configuration hook | `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py` |
| 2026-09-26 | [#19558](https://github.com/NVIDIA/TensorRT-LLM/pull/19558) | merged | [None][fix] Fix quantized MTP head loading for Nemotron H | `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` |
| 2026-09-29 | [#19648](https://github.com/NVIDIA/TensorRT-LLM/pull/19648) | merged | [TRTLLM-15824][feat] Enable EVS for Nemotron Super 3.5 VL | `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` |
| 2026-10-01 | [#19741](https://github.com/NVIDIA/TensorRT-LLM/pull/19741) | merged | [https://nvbugs/6625695][fix] Compare teacher-forced logits in the Nemotron VL video batch-equivalence test | `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` |
| 2026-10-02 | [#19702](https://github.com/NVIDIA/TensorRT-LLM/pull/19702) | merged | [TRTLLM-15824][feat] Enable EVS handoff for Nemotron-H in EPD | `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` |
| 2026-10-02 | [#19597](https://github.com/NVIDIA/TensorRT-LLM/pull/19597) | merged | [None][feat] Load quantized Nemotron-H MTP replacements | `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` |

## 逐 PR diff 审计卡

### PR #3447 - chore: Rename nvsmall to nemotron nas

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/3447
- 状态/时间: merged / 2025-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py`；关联提交 `a6a2ae6cc144`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+78/-75，可读 patch 366 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` renamed +11/-11 (22 lines); hunks: -21,7 +21,7; -69,7 +69,7 @@ def _create_linear_from_configs(model_config: ModelConfig[Pret...; symbols: NVSmallRotaryEmbedding, NemotronNASRotaryEmbedding, __init__, _create_linear_from_configs，涉及 `NVSmallRotaryEmbedding, NemotronNASRotaryEmbedding, __init__`；`tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` renamed +60/-58 (118 lines); hunks: -13,12 +13,13; -224,7 +225,8 @@ def __repr__(self) -> str:; symbols: __repr__, reduce_nvsmall_config, reduce_nemotron_nas_config, TestNVSmall，涉及 `__repr__, reduce_nvsmall_config, reduce_nemotron_nas_config`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` renamed +11/-11 (22 lines); hunks: -21,7 +21,7; -69,7 +69,7 @@ def _create_linear_from_configs(model_config: ModelConfig[Pret...; symbols: NVSmallRotaryEmbedding, NemotronNASRotaryEmbedding, __init__, _create_linear_from_configs
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` renamed +60/-58 (118 lines); hunks: -13,12 +13,13; -224,7 +225,8 @@ def __repr__(self) -> str:; symbols: __repr__, reduce_nvsmall_config, reduce_nemotron_nas_config, TestNVSmall
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -21,7 +21,7 @@
-class NVSmallRotaryEmbedding(RotaryEmbedding):
+class NemotronNASRotaryEmbedding(RotaryEmbedding):
@@ -69,7 +69,7 @@ def _create_linear_from_configs(model_config: ModelConfig[PretrainedConfig],
-class NVSmallAttention(Attention):
+class NemotronNASAttention(Attention):
@@ -88,7 +88,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py
@@ -13,12 +13,13 @@
-from tensorrt_llm._torch.models.modeling_nvsmall import NVSmallForCausalLM
+from tensorrt_llm._torch.models.modeling_nemotron_nas import \
+    NemotronNASForCausalLM
-NVSMALL_MINI_CONFIG = {
+NEMOTRON_NAS_MINI_CONFIG = {
@@ -224,7 +225,8 @@ def __repr__(self) -> str:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` renamed +11/-11
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` renamed +60/-58
- 验证与风险: diff 自带测试面 `tests/integration/defs/.test_durations`, `tests/integration/test_lists/test-db/l0_a30.yml`, `tests/integration/test_lists/test-db/l0_l40s.yml`, `tests/unittest/_torch/auto_deploy/integration/test_ad_build.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #3430 - feat: Nemotron-H model support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/3430
- 状态/时间: merged / 2025-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `0bda1f9780d8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+3211/-427，可读 patch 3783 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: split, relu2, NemotronHConfig, MLPLayer，涉及 `split, relu2, NemotronHConfig`；`tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` renamed +48/-37 (85 lines); hunks: -7,14 +7,19; -30,52 +35,46; symbols: TestMambaHybrid, TestNemotronH, test_mamba_hybrid_sanity, test_nemotron_h_sanity，涉及 `TestMambaHybrid, TestNemotronH, test_mamba_hybrid_sanity`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: split, relu2, NemotronHConfig, MLPLayer
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` renamed +48/-37 (85 lines); hunks: -7,14 +7,19; -30,52 +35,46; symbols: TestMambaHybrid, TestNemotronH, test_mamba_hybrid_sanity, test_nemotron_h_sanity
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -0,0 +1,331 @@
+from typing import Dict, Optional
+import torch
+from torch import nn
+from torch.nn import functional as F
+try:
+    from transformer_engine.pytorch import RMSNorm
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -7,14 +7,19 @@
-from tensorrt_llm._torch.models.modeling_mamba_hybrid import (
-    MambaHybridConfig, MambaHybridForCausalLM)
-from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
+# isort: off
+from tensorrt_llm._torch.models.modeling_nemotron_h import (NemotronHConfig,
+                                                            NemotronHForCausalLM
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` added +331/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` renamed +48/-37
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #3646 - Fix rotary_emb param in NemotronH attention

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/3646
- 状态/时间: merged / 2025-04-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `a06bff505201`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +0/-1 (1 lines); hunks: -77,7 +77,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +0/-1 (1 lines); hunks: -77,7 +77,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -77,7 +77,6 @@ def __init__(
-            rotary_emb=None,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #3632 - Update Nemotron Super and Ultra in Supported Models and add an example

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/3632
- 状态/时间: merged / 2025-04-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/nemotron_nas/README.md`；关联提交 `f7c2eb4fa2dc`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+9/-2，可读 patch 39 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/nemotron_nas/README.md` modified +3/-0 (3 lines); hunks: -16,6 +16,9 @@ The TensorRT-LLM Nemotron-NAS implementation can be found in [...；`examples/pytorch/README.md` modified +6/-2 (8 lines); hunks: -23,6 +23,9 @@ python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B...; -42,7 +45,6 @@ python3 quickstart_multimodal.py --model_dir Efficient-Large-M...。
- 代码 diff 细节:
  - `examples/models/core/nemotron_nas/README.md` modified +3/-0 (3 lines); hunks: -16,6 +16,9 @@ The TensorRT-LLM Nemotron-NAS implementation can be found in [...
  - `examples/pytorch/README.md` modified +6/-2 (8 lines); hunks: -23,6 +23,9 @@ python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B...; -42,7 +45,6 @@ python3 quickstart_multimodal.py --model_dir Efficient-Large-M...
- 关键代码摘录:

```diff
diff -- examples/models/core/nemotron_nas/README.md
@@ -16,6 +16,9 @@ The TensorRT-LLM Nemotron-NAS implementation can be found in [tensorrt_llm/model
+The recommended flow for using Nemotron-NAS models is through TRTLLM's PyTorch-based flow.
+An example of how to run `Nemotron-NAS` models through the PyTorch workflow can be found in the [PyTorch quickstart example](../../../pytorch/README.md).
diff -- examples/pytorch/README.md
@@ -23,6 +23,9 @@ python3 quickstart_advanced.py --model_dir nvidia/Llama-3.1-8B-Instruct-FP8 --tp
+# BF16 + TP=8
+python3 quickstart_advanced.py --model_dir nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 --tp_size 8
@@ -42,7 +45,6 @@ python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --mo
-| `DeciLMForCausalLM` | Nemotron | `nvidia/Llama-3_1-Nemotron-51B-Instruct` | L |
@@ -52,7 +54,9 @@ python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --mo
-| `NemotronNASForCausalLM` | NemotronNAS | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | L |
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/nemotron_nas/README.md` modified +3/-0; `examples/pytorch/README.md` modified +6/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/models/core/nemotron_nas/README.md`, `examples/pytorch/README.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #4180 - add changes for fp8, nemotron-nas, API

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4180
- 状态/时间: merged / 2025-05-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；关联提交 `27afcb9928f2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+370/-85，可读 patch 745 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +41/-5 (46 lines); hunks: -1,10 +1,12; -75,8 +77,12 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +41/-5 (46 lines); hunks: -1,10 +1,12; -75,8 +77,12 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -1,10 +1,12 @@
-from typing import Any, Dict
+from typing import Any, Dict, Optional
+from tensorrt_llm.lora_manager import HfLoraLoader
+from tensorrt_llm.models.convert_utils import split_matrix_tp
@@ -75,8 +77,12 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],
+        lora_params: Optional[dict] = None,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +41/-5
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_dgx_h200.yml`, `tests/unittest/api_stability/references/llm.yaml`, `tests/unittest/api_stability/references_committed/llm.yaml`, `tests/unittest/llmapi/test_llm_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4444 - [TRTLLM-5085][fix] Nemotron H correctness test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4444
- 状态/时间: merged / 2025-05-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `7b09cd904d34`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+340/-109，可读 patch 496 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +340/-109 (449 lines); hunks: -1,5 +1,4; -13,80 +12,190; symbols: get_logprobs, _generate, generate, TestNemotronH，涉及 `get_logprobs, _generate, generate`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +340/-109 (449 lines); hunks: -1,5 +1,4; -13,80 +12,190; symbols: get_logprobs, _generate, generate, TestNemotronH
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,5 +1,4 @@
-from copy import deepcopy
@@ -13,80 +12,190 @@
+from transformers import AutoTokenizer, PreTrainedTokenizerBase
+from utils.llm_data import llm_models_root
+from utils.util import skip_gpu_memory_less_than
+from tensorrt_llm._torch.pyexecutor.model_engine import load_weights
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +340/-109
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4494 - [TRTLLM-4783][feat] Mamba2 kernel updates for Nemotron-H

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4494
- 状态/时间: merged / 2025-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `bf9cd11fd498`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+2562/-412，可读 patch 3575 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +28/-21 (49 lines); hunks: -1,22 +1,33; -112,26 +123,24 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +28/-21 (49 lines); hunks: -1,22 +1,33; -112,26 +123,24 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -1,22 +1,33 @@
+# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +28/-21
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/thop/test_causal_conv1d_op.py`, `tests/unittest/_torch/thop/test_mamba2_chunk_ss_update.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4954 - [nvbug 5325284][fix] Increase Nemotron-H warmup request robustness

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4954
- 状态/时间: merged / 2025-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `f121f13ddfa7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+286/-256，可读 patch 610 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +265/-233 (498 lines); hunks: -1,5 +1,3; -16,11 +14,14; symbols: get_logprobs, generate, TestNemotronH, test_nemotron_h_correctness，涉及 `get_logprobs, generate, TestNemotronH`；`tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +18/-22 (40 lines); hunks: -154,11 +154,6 @@ def forward(; -167,11 +162,18 @@ def forward(; symbols: forward，涉及 `forward`；`tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-1 (4 lines); hunks: -578,7 +578,9 @@ def __init__(; symbols: __init__, prepare_mamba_cache_blocks，涉及 `__init__, prepare_mamba_cache_blocks`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +265/-233 (498 lines); hunks: -1,5 +1,3; -16,11 +14,14; symbols: get_logprobs, generate, TestNemotronH, test_nemotron_h_correctness
  - `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +18/-22 (40 lines); hunks: -154,11 +154,6 @@ def forward(; -167,11 +162,18 @@ def forward(; symbols: forward
  - `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-1 (4 lines); hunks: -578,7 +578,9 @@ def __init__(; symbols: __init__, prepare_mamba_cache_blocks
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,5 +1,3 @@
-import unittest
@@ -16,11 +14,14 @@
+from tensorrt_llm._torch import LLM
-from tensorrt_llm.bindings.executor import KvCacheConfig
+from tensorrt_llm.bindings.executor import KvCacheConfig as KvCacheConfigCpp
+from tensorrt_llm.llmapi import KvCacheConfig
diff -- tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py
@@ -154,11 +154,6 @@ def forward(
-        # warm up does not prepare resources, there are two warmup requests
-        is_warmup = attn_metadata.kv_cache_manager is None or attn_metadata.request_ids == [
-            0
-        ]
@@ -167,11 +162,18 @@ def forward(
-        # handle warm up request
diff -- tensorrt_llm/_torch/pyexecutor/resource_manager.py
@@ -578,7 +578,9 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +265/-233
  - runtime: `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +18/-22; `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5097 - [test] Use LLM API for Nemotron-H correctness test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5097
- 状态/时间: merged / 2025-06-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `06d9f1e2f6c4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+135/-356，可读 patch 552 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +135/-356 (491 lines); hunks: -1,26 +1,10; -31,236 +15,43 @@ def get_logprobs(token_ids: torch.Tensor, logits: torch.Ten...; symbols: get_logprobs, _generate, extract_prefill_logprobs, generate，涉及 `get_logprobs, _generate, extract_prefill_logprobs`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +135/-356 (491 lines); hunks: -1,26 +1,10; -31,236 +15,43 @@ def get_logprobs(token_ids: torch.Tensor, logits: torch.Ten...; symbols: get_logprobs, _generate, extract_prefill_logprobs, generate
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,26 +1,10 @@
-import tensorrt_llm
-from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
-from tensorrt_llm._torch.metadata import KVCacheParams
-from tensorrt_llm._torch.model_config import ModelConfig
-# isort: off
-from tensorrt_llm._torch.models.modeling_nemotron_h import (NemotronHConfig,
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +135/-356
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5202 - [fix][test] Speedup Nemotron NAS unittests

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5202
- 状态/时间: merged / 2025-06-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py`；关联提交 `4eade3ae3349`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-8，可读 patch 40 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` modified +11/-8 (19 lines); hunks: -5,7 +5,8; -371,20 +372,22 @@ def test_nemotron_nas_allclose_to_hf(self, scenario: Scena...; symbols: test_nemotron_nas_allclose_to_hf, DeciLMForCausalLM，涉及 `test_nemotron_nas_allclose_to_hf, DeciLMForCausalLM`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` modified +11/-8 (19 lines); hunks: -5,7 +5,8; -371,20 +372,22 @@ def test_nemotron_nas_allclose_to_hf(self, scenario: Scena...; symbols: test_nemotron_nas_allclose_to_hf, DeciLMForCausalLM
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py
@@ -5,7 +5,8 @@
-from transformers import AutoConfig, AutoModelForCausalLM
+from transformers import AutoConfig
+from transformers.dynamic_module_utils import get_class_from_dynamic_module
@@ -371,20 +372,22 @@ def test_nemotron_nas_allclose_to_hf(self, scenario: Scenario) -> None:
+        nemotron_nas_ckpt = llm_models_root(
+        ) / "nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` modified +11/-8
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4906 - feat: Add support for YARN in NemotronNAS models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4906
- 状态/时间: merged / 2025-06-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；关联提交 `de9779900c4c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+14/-6，可读 patch 59 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +12/-3 (15 lines); hunks: -4,7 +4,7; -48,19 +48,28 @@ def _create_linear_from_configs(model_config: ModelConfig[Pr...; symbols: _create_linear_from_configs, NemotronNASAttention, __init__，涉及 `_create_linear_from_configs, NemotronNASAttention, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +12/-3 (15 lines); hunks: -4,7 +4,7; -48,19 +48,28 @@ def _create_linear_from_configs(model_config: ModelConfig[Pr...; symbols: _create_linear_from_configs, NemotronNASAttention, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -4,7 +4,7 @@
-from tensorrt_llm.functional import PositionEmbeddingType
+from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
@@ -48,19 +48,28 @@ def _create_linear_from_configs(model_config: ModelConfig[PretrainedConfig],
+    NON_NEOX_TYPES = ("mistral_yarn", "rope_llama4")
+        is_neox = getattr(model_config.pretrained_config,
+                          "position_embedding_type",
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +12/-3
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/attention_backend/interface.py`, `tensorrt_llm/_torch/attention_backend/trtllm.py`, `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5646 - [TRTLLM-4923][feat] Enable CUDA graphs for Nemotron-H

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5646
- 状态/时间: merged / 2025-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `7dbecf7272ba`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+89/-20，可读 patch 160 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +84/-7 (91 lines); hunks: -5,6 +5,7; -28,25 +29,36 @@ def extract_decode_logprobs(result: RequestOutput,; symbols: extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler，涉及 `extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_correctness`；`tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +2/-9 (11 lines); hunks: -163,15 +163,8 @@ def forward(; symbols: forward，涉及 `forward`；`tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-4 (7 lines); hunks: -593,7 +593,7 @@ def __init__(; -610,9 +610,8 @@ def prepare_mamba_cache_blocks(self, request_ids: List[int]):; symbols: __init__, prepare_mamba_cache_blocks, free_mamba_cache_blocks，涉及 `__init__, prepare_mamba_cache_blocks, free_mamba_cache_blocks`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +84/-7 (91 lines); hunks: -5,6 +5,7; -28,25 +29,36 @@ def extract_decode_logprobs(result: RequestOutput,; symbols: extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler
  - `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +2/-9 (11 lines); hunks: -163,15 +163,8 @@ def forward(; symbols: forward
  - `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-4 (7 lines); hunks: -593,7 +593,7 @@ def __init__(; -610,9 +610,8 @@ def prepare_mamba_cache_blocks(self, request_ids: List[int]):; symbols: __init__, prepare_mamba_cache_blocks, free_mamba_cache_blocks
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -5,6 +5,7 @@
+from tensorrt_llm.llmapi.llm_args import CudaGraphConfig
@@ -28,25 +29,36 @@ def extract_decode_logprobs(result: RequestOutput,
+def create_nemotron_h_llm(use_cuda_graph, disable_overlap_scheduler,
+                          max_batch_size):
+    """Create LLM with specific overlap scheduler setting"""
+    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
diff -- tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py
@@ -163,15 +163,8 @@ def forward(
-        state_indices = attn_metadata.kv_cache_manager.get_state_indices()
-        # warm up does not prepare resources, so no relevant state indices
-        is_warmup = state_indices.numel() == 0
-        if is_warmup:
-            # in this case, assume batch takes first indices in mamba cache
-            state_indices = torch.arange(num_prefills + num_decodes,
diff -- tensorrt_llm/_torch/pyexecutor/resource_manager.py
@@ -593,7 +593,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +84/-7
  - runtime: `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +2/-9; `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +3/-4
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6455 - [Perf]: Add residual, norm for nemotron_nas models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6455
- 状态/时间: merged / 2025-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；关联提交 `e67f4da9b5c0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+47/-7，可读 patch 84 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +47/-7 (54 lines); hunks: -149,29 +149,36 @@ def forward(; -225,6 +232,39 @@ def __init__(self, model_config):; symbols: forward, NemotronNASModel, __init__, NemotronNASForCausalLM，涉及 `forward, NemotronNASModel, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +47/-7 (54 lines); hunks: -149,29 +149,36 @@ def forward(; -225,6 +232,39 @@ def __init__(self, model_config):; symbols: forward, NemotronNASModel, __init__, NemotronNASForCausalLM
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -149,29 +149,36 @@ def forward(
+        residual: Optional[torch.Tensor] = None,
-            residual = hidden_states
-            hidden_states = self.input_layernorm(hidden_states)
+            if residual is None:
+                residual = hidden_states
+                hidden_states = self.input_layernorm(hidden_states)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +47/-7
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6485 - [https://nvbugs/5404046][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6485
- 状态/时间: merged / 2025-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `6d5da9f7c299`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+35/-22，可读 patch 89 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +35/-22 (57 lines); hunks: -1,4 +1,3; -238,15 +237,15 @@ def test_nemotron_h_correctness():; symbols: test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler，涉及 `test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +35/-22 (57 lines); hunks: -1,4 +1,3; -238,15 +237,15 @@ def test_nemotron_h_correctness():; symbols: test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,4 +1,3 @@
-import pytest
@@ -238,15 +237,15 @@ def test_nemotron_h_correctness():
-@pytest.mark.skip(reason="https://nvbugs/5404046")
-        "Tell me something I don't know about the future of AI",
-        "The president of the United States is",
-        "The capital of France is",
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +35/-22
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6866 - [None][feat] Support running heterogeneous model execution for Nemotron-H

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6866
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `bda42f8c3a3e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +9/-1 (10 lines); hunks: -63,8 +63,16 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +9/-1 (10 lines); hunks: -63,8 +63,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -63,8 +63,16 @@ def __init__(
+        if isinstance(config.intermediate_size, list):
+            if len(config.intermediate_size) == 1:
+                intermediate_size = config.intermediate_size[0]
+            else:
+                intermediate_size = config.intermediate_size[layer_idx]
+        else:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6996 - [https://nvbugs/5458874][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6996
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `f0bfb49219a8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+14/-4，可读 patch 35 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +14/-4 (18 lines); hunks: -247,7 +247,6 @@ def test_nemotron_h_correctness(mamba_ssm_cache_dtype):; -317,12 +316,23 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():; symbols: test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler，涉及 `test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +14/-4 (18 lines); hunks: -247,7 +247,6 @@ def test_nemotron_h_correctness(mamba_ssm_cache_dtype):; -317,12 +316,23 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():; symbols: test_nemotron_h_correctness, test_nemotron_h_cuda_graph_overlap_scheduler
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -247,7 +247,6 @@ def test_nemotron_h_correctness(mamba_ssm_cache_dtype):
-@pytest.mark.skip(reason="https://nvbugs/5458874")
@@ -317,12 +316,23 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():
+        # Similar comparison for with / without overlap scheduler, compare logits of first generation step (2nd generated token)
-            with_cg_no_overlap.outputs[0].generation_logits,
-            with_cg_with_overlap.outputs[0].generation_logits,
+            with_cg_no_overlap.outputs[0].generation_logits[1, :],
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +14/-4
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6334 - [TRTLLM-4921][feat] Enable chunked prefill for Nemotron-H

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6334
- 状态/时间: merged / 2025-08-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `c232ba8157eb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+540/-55，可读 patch 959 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +3/-1 (4 lines); hunks: -221,7 +221,9 @@ def forward(; symbols: forward，涉及 `forward`；`tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +64/-1 (65 lines); hunks: -33,7 +33,9 @@ def extract_decode_logprobs(result: RequestOutput,; -47,6 +49,8 @@ def create_nemotron_h_llm(use_cuda_graph,; symbols: extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_cuda_graph_overlap_scheduler, test_nemotron_h_chunked_prefill，涉及 `extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_cuda_graph_overlap_scheduler`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +3/-1 (4 lines); hunks: -221,7 +221,9 @@ def forward(; symbols: forward
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +64/-1 (65 lines); hunks: -33,7 +33,9 @@ def extract_decode_logprobs(result: RequestOutput,; -47,6 +49,8 @@ def create_nemotron_h_llm(use_cuda_graph,; symbols: extract_decode_logprobs, create_nemotron_h_llm, test_nemotron_h_cuda_graph_overlap_scheduler, test_nemotron_h_chunked_prefill
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -221,7 +221,9 @@ def forward(
-            self.mamba_metadata = Mamba2Metadata(attn_metadata.max_num_requests)
+            self.mamba_metadata = Mamba2Metadata(
+                attn_metadata.max_num_requests,
+                chunk_size=self.model_config.pretrained_config.chunk_size)
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -33,7 +33,9 @@ def extract_decode_logprobs(result: RequestOutput,
-                          mamba_ssm_cache_dtype=None):
+                          mamba_ssm_cache_dtype=None,
+                          enable_chunked_prefill=False,
+                          max_num_tokens=None):
@@ -47,6 +49,8 @@ def create_nemotron_h_llm(use_cuda_graph,
+        enable_chunked_prefill=enable_chunked_prefill,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +3/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +64/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`, `tests/unittest/_torch/thop/test_causal_conv1d_op.py`, `tests/unittest/_torch/thop/test_mamba2_chunk_ss_update.py`, `tests/unittest/utils/torch_ref.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7589 - [None][fix] enable NvFP4/FP8 quantization for Nemotron-H architecture

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7589
- 状态/时间: merged / 2025-09-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `6e712dd1cc2e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+44/-20，可读 patch 106 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1 (3 lines); hunks: -13,6 +13,7; -255,7 +256,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -34,7 +34,7 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights，涉及 `preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1 (3 lines); hunks: -13,6 +13,7; -255,7 +256,7 @@ def __init__(; symbols: __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -34,7 +34,7 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -13,6 +13,7 @@
+import re
@@ -255,7 +256,7 @@ def __init__(
-                k.replace('model.layers.backbone', 'model')
+                re.sub(r'(model\.layers\.)?backbone', 'model', k)
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -34,7 +34,7 @@ def preprocess_weights(self, weights: dict) -> dict:
-            if "_scale" in key and weights[name].dim() == 0:
+            if "_scale" in key:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1; `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/modeling_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8697 - [None][fix] Properly raise error for nemotron H models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8697
- 状态/时间: merged / 2025-10-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `cdc9e5e64566`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -161,7 +161,7 @@ def __init__(
-            ValueError(f"{layer_type} is not supported")
+            raise ValueError(f"{layer_type} is not supported")
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8985 - [None][feat] add specdec to nemotron nas

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8985
- 状态/时间: merged / 2025-11-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；关联提交 `a7c0b54ce7d4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+15/-8，可读 patch 71 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +15/-8 (23 lines); hunks: -17,8 +17,9; -117,6 +118,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: _ffn_mult_to_intermediate_size, __init__, forward，涉及 `_ffn_mult_to_intermediate_size, __init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +15/-8 (23 lines); hunks: -17,8 +17,9; -117,6 +118,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],; symbols: _ffn_mult_to_intermediate_size, __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -17,8 +17,9 @@
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             register_auto_model)
+from ..speculative import SpecMetadata
+from .modeling_speculative import SpecDecOneEngineForCausalLM
+from .modeling_utils import DecoderModel, register_auto_model
@@ -117,6 +118,7 @@ def __init__(self, model_config: ModelConfig[PretrainedConfig],
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_nas.py` modified +15/-8
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9993 - [https://nvbugs/5721644][fix] Update tests for nemotron_h

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9993
- 状态/时间: merged / 2025-12-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `601c29ca7349`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+123/-64，可读 patch 276 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +69/-60 (129 lines); hunks: -1,12 +1,12; -37,9 +37,15 @@ def create_nemotron_h_llm(model_folder,; symbols: create_nemotron_h_llm, test_nemotron_h_sanity, test_nemotron_h_correctness，涉及 `create_nemotron_h_llm, test_nemotron_h_sanity, test_nemotron_h_correctness`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +69/-60 (129 lines); hunks: -1,12 +1,12; -37,9 +37,15 @@ def create_nemotron_h_llm(model_folder,; symbols: create_nemotron_h_llm, test_nemotron_h_sanity, test_nemotron_h_correctness
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,12 +1,12 @@
-from utils.util import similar, skip_gpu_memory_less_than
+from utils.util import skip_fp8_pre_ada, skip_gpu_memory_less_than
-from tensorrt_llm.llmapi.llm_args import CudaGraphConfig
+from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, LoadFormat
@@ -37,9 +37,15 @@ def create_nemotron_h_llm(model_folder,
-                          max_num_tokens=8192):
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +69/-60
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10118 - [None][feat] Support multi-gpu running for nemotron-v3-nano and super

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10118
- 状态/时间: merged / 2025-12-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `14554ab3f33c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+122/-40，可读 patch 319 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +41/-32 (73 lines); hunks: -15,6 +15,30 @@ def preprocess_weights(self, weights: dict) -> dict:; -36,7 +60,12 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights, _split_mamba2_mixer_in_proj，涉及 `preprocess_weights, _split_mamba2_mixer_in_proj`；`tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +15/-2 (17 lines); hunks: -26,6 +26,7; -124,7 +125,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +41/-32 (73 lines); hunks: -15,6 +15,30 @@ def preprocess_weights(self, weights: dict) -> dict:; -36,7 +60,12 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights, _split_mamba2_mixer_in_proj
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +15/-2 (17 lines); hunks: -26,6 +26,7; -124,7 +125,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -15,6 +15,30 @@ def preprocess_weights(self, weights: dict) -> dict:
+        def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
+            # Special handling for Mamba2 mixer in_proj.weights and scales.
+            in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
+                w, [
+                    d_inner, d_inner, n_groups * d_state, n_groups * d_state,
+                    nheads
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -26,6 +26,7 @@
+from ..distributed import AllReduce
@@ -124,7 +125,7 @@ def __init__(
-        self.reduce_results = True
+        self.reduce_results = False
@@ -144,6 +145,7 @@ def __init__(
+        self.mapping = model_config.mapping
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +41/-32; `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +15/-2
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10347 - [TRTLLM-10060][feat] Enable attention dp for Nemotron Super v3.

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10347
- 状态/时间: merged / 2026-01-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `bdaee87895d1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+75/-30，可读 patch 254 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +38/-14 (52 lines); hunks: -32,7 +32,7; -85,8 +85,10 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7 @@ class NemotronHHfWeightMapper(HfWeightMapper):; symbols: NemotronHHfWeightMapper, preprocess_weights，涉及 `NemotronHHfWeightMapper, preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +38/-14 (52 lines); hunks: -32,7 +32,7; -85,8 +85,10 @@ def __init__(; symbols: __init__, forward
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7 @@ class NemotronHHfWeightMapper(HfWeightMapper):; symbols: NemotronHHfWeightMapper, preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -32,7 +32,7 @@
-from ..modules.linear import Linear
+from ..modules.linear import Linear, TensorParallelMode
@@ -85,8 +85,10 @@ def __init__(
+        reduce_output: bool = False,
@@ -97,6 +99,7 @@ def __init__(
+            reduce_output=reduce_output,
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -11,7 +11,7 @@ class NemotronHHfWeightMapper(HfWeightMapper):
-        tp_size = self.config.mapping.tp_size
+        tp_size = 1 if self.config.mapping.enable_attention_dp else self.config.mapping.tp_size
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +38/-14; `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10754 - [TRTLLM-10062][feat] Enable MTP for Nemotron Super

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10754
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `ff0dd6076e9e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+2244/-313，可读 patch 3309 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +294/-14 (308 lines); hunks: -14,7 +14,7; -37,9 +37,11; symbols: NemotronHConfig, forward, __init__，涉及 `NemotronHConfig, forward, __init__`；`tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +11/-0 (11 lines); hunks: -1,5 +1,6; -55,6 +56,16 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Ten...; symbols: _split_mamba2_mixer_in_proj，涉及 `_split_mamba2_mixer_in_proj`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +294/-14 (308 lines); hunks: -14,7 +14,7; -37,9 +37,11; symbols: NemotronHConfig, forward, __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +11/-0 (11 lines); hunks: -1,5 +1,6; -55,6 +56,16 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Ten...; symbols: _split_mamba2_mixer_in_proj
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -14,7 +14,7 @@
-from typing import Dict, Optional
+from typing import Dict, List, Optional
@@ -37,9 +37,11 @@
+from ..speculative import SpecMetadata
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             register_auto_model)
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -1,5 +1,6 @@
+import tensorrt_llm.logger as logger
@@ -55,6 +56,16 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
+            # MTP layers are stored as mtp.layers.0.xxx (sublayer 0, Attention) and mtp.layers.1.xxx (sublayer 1, MoE)
+            if "mtp.layers." in key:
+                import re
+                match = re.match(r'mtp\.layers\.(\d+)\.(.*)', key)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +294/-14; `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +11/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/_torch/thop/parallel/test_mamba2_chunk_ss_update.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11131 - [None][feat] Nemotron H: Eagle3 support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11131
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `3ef8a4639b19`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +4/-0 (4 lines); hunks: -361,6 +361,10 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +4/-0 (4 lines); hunks: -361,6 +361,10 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -361,6 +361,10 @@ def forward(
+        if spec_metadata is not None and spec_metadata.is_layer_capture(
+                self.layer_idx):
+            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
+                                                      hidden_states, None)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11406 - [None][chore] Merge residual+hidden into layer norm at the end of each NemotronH MTP, and remove a % operation

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11406
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `2d5ebb3fe87b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+6/-9，可读 patch 36 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-9 (15 lines); hunks: -665,11 +665,10 @@ def forward(; -697,9 +696,7 @@ def __init__(self,; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-9 (15 lines); hunks: -665,11 +665,10 @@ def forward(; -697,9 +696,7 @@ def __init__(self,; symbols: forward, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -665,11 +665,10 @@ def forward(
-            if residual is not None:
-                hidden_states = hidden_states + residual
-                residual = None
-            hidden_states = self.final_layernorm(hidden_states)
+            hidden_states, residual = self.final_layernorm(
+                hidden_states, residual)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-9
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11273 - [None][feat] Optimize super-v3 nvfp4 for better perf

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11273
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `421eb9e39c85`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 27 个文件，+2195/-206，可读 patch 3077 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +106/-36 (142 lines); hunks: -14,7 +14,6; -23,6 +22,7; symbols: __init__, forward, _compute_shared_output，涉及 `__init__, forward, _compute_shared_output`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +106/-36 (142 lines); hunks: -14,7 +14,6; -23,6 +22,7; symbols: __init__, forward, _compute_shared_output
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -14,7 +14,6 @@
-from typing import Dict, List, Optional
@@ -23,6 +22,7 @@
+from tensorrt_llm.logger import logger
@@ -37,7 +37,7 @@
-from ..utils import AuxStreamType, EventType
+from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +106/-36
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/mamba/test_causal_conv1d.py`, `tests/unittest/_torch/modules/mamba/test_fuse_elementwise_ops.py`, `tests/unittest/_torch/modules/mamba/test_mamba2_metadata.py`, `tests/unittest/_torch/modules/test_fused_activation_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11405 - [TRTLLM-10329][feat] Fix weight loading for Nemotron 3 models on DGX Spark

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11405
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`；关联提交 `19a3031ecb0b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+21/-2，可读 patch 65 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +3/-1 (4 lines); hunks: -80,7 +80,9 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:; symbols: _split_mamba2_mixer_in_proj，涉及 `_split_mamba2_mixer_in_proj`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +3/-1 (4 lines); hunks: -80,7 +80,9 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:; symbols: _split_mamba2_mixer_in_proj
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -80,7 +80,9 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
-                w = -torch.exp(w)
+                # Avoid extra temporaries: one fp32 cast, then in-place exp/neg.
+                w.exp_()
+                w.neg_()
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/modules/fused_moe/quantization.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11601 - [None][fix] Nemotron H fp4 and MTP

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11601
- 状态/时间: merged / 2026-02-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `c53b8fc2f150`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+8/-5，可读 patch 62 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-4 (6 lines); hunks: -281,10 +281,8 @@ def _compute_routed_output():; symbols: _compute_routed_output，涉及 `_compute_routed_output`；`tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -124,7 +124,7 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Te...; symbols: _split_mamba2_mixer_in_proj，涉及 `_split_mamba2_mixer_in_proj`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-4 (6 lines); hunks: -281,10 +281,8 @@ def _compute_routed_output():; symbols: _compute_routed_output
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1 (2 lines); hunks: -124,7 +124,7 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Te...; symbols: _split_mamba2_mixer_in_proj
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -281,10 +281,8 @@ def _compute_routed_output():
-            routed_hidden_states = hidden_states
-            if self.use_latent_moe:
-                routed_hidden_states = self.fc1_latent_proj(
-                    routed_hidden_states)
+            routed_hidden_states = self.fc1_latent_proj(
+                hidden_states_hp) if self.use_latent_moe else hidden_states
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -124,7 +124,7 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
-                        if "input_scale" in key or "weight_scale_2" in key:
+                        if "input_scale" in key or "weight_scale_2" in key or "input_quantizer" in key or "weight_quantizer" in key:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-4; `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/visual_gen/test_wan.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11807 - [None][fix] Fix nemotron super MTP crash on SM90

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11807
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `517ee94938f8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+177/-16，可读 patch 300 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +17/-10 (27 lines); hunks: -1,4 +1,4; -14,6 +14,7; symbols: forward, _compute_shared_output, __init__，涉及 `forward, _compute_shared_output, __init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +17/-10 (27 lines); hunks: -1,4 +1,4; -14,6 +14,7; symbols: forward, _compute_shared_output, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -1,4 +1,4 @@
-# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@@ -14,6 +14,7 @@
+from dataclasses import replace
@@ -268,7 +269,10 @@ def forward(
-        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +17/-10
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11972 - [None][feat] Mamba optimization and mixed quantization support for nemotron-h

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11972
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `73fca4e0bd85`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+186/-50，可读 patch 405 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +37/-5 (42 lines); hunks: -197,6 +197,19 @@ def __init__(; -206,7 +219,7 @@ def __init__(; symbols: __init__, forward, _compute_shared_output，涉及 `__init__, forward, _compute_shared_output`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +37/-5 (42 lines); hunks: -197,6 +197,19 @@ def __init__(; -206,7 +219,7 @@ def __init__(; symbols: __init__, forward, _compute_shared_output
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -197,6 +197,19 @@ def __init__(
+        # For MIXED_PRECISION models, the global quant_config has quant_algo=MIXED_PRECISION
+        # which maps to QuantMode(0) (no quant). This would cause the MoE backend to select
+        # UnquantizedFusedMoEMethod and allocate BF16 weight buffers, causing a shape mismatch
+        # when loading NVFP4/W4A8_NVFP4_FP8 quantized expert weights.
+        # Look up the per-expert quant config from quant_config_dict and use it for create_moe.
+        moe_model_config = model_config
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +37/-5
- 验证与风险: diff 自带测试面 `tests/unittest/api_stability/references/quant_config.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12129 - [TRTLLM-10244][doc] Add deployment guide for Nemotron 3 Super

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12129
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/configs/curated/nemotron-3-super-throughput.yaml`, `examples/models/core/nemotron/README_nemotron_super_v3.md`；关联提交 `a4e6745c24bb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+486/-1，可读 patch 518 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/nemotron/README_nemotron_super_v3.md` added +197/-0 (197 lines); hunks: -0,0 +1,197；`examples/configs/curated/nemotron-3-super-throughput.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13。
- 代码 diff 细节:
  - `examples/models/core/nemotron/README_nemotron_super_v3.md` added +197/-0 (197 lines); hunks: -0,0 +1,197
  - `examples/configs/curated/nemotron-3-super-throughput.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- examples/models/core/nemotron/README_nemotron_super_v3.md
@@ -0,0 +1,197 @@
+# Nemotron Super V3 model
+## Table of Contents
+- [Overview](#overview)
+- [Supported Hardware](#supported-hardware)
+- [Usage](#usage)
+  - [Online serving example](#online-serving-example)
diff -- examples/configs/curated/nemotron-3-super-throughput.yaml
@@ -0,0 +1,13 @@
+max_batch_size: 512
+max_num_tokens: 2048
+tensor_parallel_size: 4
+moe_expert_parallel_size: 4
+trust_remote_code: true
+enable_attention_dp: true
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/nemotron/README_nemotron_super_v3.md` added +197/-0; `examples/configs/curated/nemotron-3-super-throughput.yaml` added +13/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/deployment-guide/deployment-guide-for-nemotron-3-super-on-trtllm.md`, `docs/source/deployment-guide/index.rst`, `docs/source/models/supported-models.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #12215 - [None][docs] Update nemotron 3 super deployment to include tool calling and reasoning parser

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12215
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/models/core/nemotron/README_nemotron_super_v3.md`；关联提交 `20fc52c82d46`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+7/-1，可读 patch 36 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/models/core/nemotron/README_nemotron_super_v3.md` modified +6/-0 (6 lines); hunks: -70,6 +70,8 @@ EOF; -119,6 +121,8 @@ EOF。
- 代码 diff 细节:
  - `examples/models/core/nemotron/README_nemotron_super_v3.md` modified +6/-0 (6 lines); hunks: -70,6 +70,8 @@ EOF; -119,6 +121,8 @@ EOF
- 关键代码摘录:

```diff
diff -- examples/models/core/nemotron/README_nemotron_super_v3.md
@@ -70,6 +70,8 @@ EOF
+--reasoning_parser nano-v3 \
+--tool_parser qwen3_coder \
@@ -119,6 +121,8 @@ EOF
+--reasoning_parser nano-v3 \
+--tool_parser qwen3_coder \
@@ -155,6 +159,8 @@ EOF
```

- 提取文件（未人工审阅）:
  - docs: `examples/models/core/nemotron/README_nemotron_super_v3.md` modified +6/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/deployment-guide/deployment-guide-for-nemotron-3-super-on-trtllm.md`, `examples/models/core/nemotron/README_nemotron_super_v3.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #12410 - [None][feat] Fuse all_reduce with norm for nemotron_h models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12410
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `e5d843542999`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+124/-26，可读 patch 287 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +117/-24 (141 lines); hunks: -31,7 +31,7; -59,6 +59,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +117/-24 (141 lines); hunks: -31,7 +31,7; -59,6 +59,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -31,7 +31,7 @@
-from ..distributed import AllReduce
+from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
@@ -59,6 +59,7 @@ def __init__(
+        reduce_output: bool = True,
@@ -76,6 +77,7 @@ def __init__(
+            reduce_output=reduce_output,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +117/-24
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12154 - [TRTLLM-10232][feat] Support LoRA adapter for nemotron-h models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12154
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py`；关联提交 `dbb1c8cb4fda`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+545/-44，可读 patch 1048 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +83/-6 (89 lines); hunks: -29,6 +29,7; -42,6 +43,7; symbols: forward, TransformerLayer, __init__，涉及 `forward, TransformerLayer, __init__`；`tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` added +256/-0 (256 lines); hunks: -0,0 +1,256; symbols: _create_lora_adapter, _add, _layer, _get_lora_config，涉及 `_create_lora_adapter, _add, _layer`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +83/-6 (89 lines); hunks: -29,6 +29,7; -42,6 +43,7; symbols: forward, TransformerLayer, __init__
  - `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` added +256/-0 (256 lines); hunks: -0,0 +1,256; symbols: _create_lora_adapter, _add, _layer, _get_lora_config
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -29,6 +29,7 @@
+from tensorrt_llm.lora_helper import LoraConfig
@@ -42,6 +43,7 @@
+from ..peft.lora.layer import LoraLayer, LoraModuleType
@@ -85,9 +87,10 @@ def forward(
+        lora_params: dict | None = None,
-        return super().forward(hidden_states)
diff -- tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py
@@ -0,0 +1,256 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +83/-6
  - tests: `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` added +256/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12620 - [None][fix] Update codes to support nemotron-h corner cases

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12620
- 状态/时间: merged / 2026-04-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `56e6961b0a31`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+6/-5，可读 patch 33 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1 (3 lines); hunks: -720,7 +720,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1 (3 lines); hunks: -720,7 +720,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -720,7 +720,8 @@ def __init__(
-        if model_config.mapping.tp_size not in [1, 2, 4, 8]:
+        if (not model_config.mapping.enable_attention_dp
+                and model_config.mapping.tp_size not in [1, 2, 4, 8]):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `cpp/tensorrt_llm/kernels/xqaDispatcher.cpp`, `tensorrt_llm/_torch/attention_backend/trtllm_gen.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12980 - [https://nvbugs/5626259][fix] Enable nemotron-h chunk prefill test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12980
- 状态/时间: merged / 2026-04-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `d66abd7c7cb5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+10/-42，可读 patch 78 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +10/-41 (51 lines); hunks: -415,7 +415,6 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():; -427,51 +426,21 @@ def test_nemotron_h_chunked_prefill():; symbols: test_nemotron_h_cuda_graph_overlap_scheduler, test_nemotron_h_chunked_prefill，涉及 `test_nemotron_h_cuda_graph_overlap_scheduler, test_nemotron_h_chunked_prefill`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +10/-41 (51 lines); hunks: -415,7 +415,6 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():; -427,51 +426,21 @@ def test_nemotron_h_chunked_prefill():; symbols: test_nemotron_h_cuda_graph_overlap_scheduler, test_nemotron_h_chunked_prefill
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -415,7 +415,6 @@ def test_nemotron_h_cuda_graph_overlap_scheduler():
-@pytest.mark.skip(reason="https://nvbugs/5626259")
@@ -427,51 +426,21 @@ def test_nemotron_h_chunked_prefill():
-    sampling_config = SamplingParams(max_tokens=10,
-                                     temperature=0.0,
-                                     return_context_logits=True,
-                                     return_generation_logits=True)
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +10/-41
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13160 - [None][chore] improve gemm perf for nemotron in spark

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13160
- 状态/时间: merged / 2026-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `43e3070de4d4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+56/-3，可读 patch 260 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +23/-0 (23 lines); hunks: -28,6 +28,7; -62,6 +63,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +23/-0 (23 lines); hunks: -28,6 +28,7; -62,6 +63,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -28,6 +28,7 @@
+from tensorrt_llm._utils import get_sm_version
@@ -62,6 +63,7 @@ def __init__(
+        use_custom_cublas_mm: bool = False,
@@ -80,6 +82,7 @@ def __init__(
+            use_custom_cublas_mm=use_custom_cublas_mm,
@@ -100,6 +103,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +23/-0
- 验证与风险: runtime 路径改动集中在 `cpp/tensorrt_llm/thop/cublasScaledMMLut.h`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13837 - [None][test] Add func and perf case of nemotron-3-Nano-Omni model on DGX-Spark

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13837
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml`, `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml`；关联提交 `7b39eb1a1959`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+368/-8，可读 patch 458 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml` added +14/-0 (14 lines); hunks: -0,0 +1,14；`tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml` renamed +0/-4 (4 lines); hunks: -1,7 +1,3。
- 代码 diff 细节:
  - `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml` added +14/-0 (14 lines); hunks: -0,0 +1,14
  - `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml` renamed +0/-4 (4 lines); hunks: -1,7 +1,3
- 关键代码摘录:

```diff
diff -- tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml
@@ -0,0 +1,14 @@
+kv_cache_config:
+  enable_block_reuse: false
+  free_gpu_memory_fraction: 0.80
+  mamba_ssm_cache_dtype: float32
+moe_config:
+  backend: CUTLASS
diff -- tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml
@@ -1,7 +1,3 @@
-# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-# SPDX-License-Identifier: Apache-2.0
-#
-# Config for Nemotron 3 Super 120B NVFP4 on DGX Spark (1x GB10, 128GB unified memory).
```

- 提取文件（未人工审阅）:
  - tests: `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml` added +14/-0; `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml` renamed +0/-4
- 验证与风险: diff 自带测试面 `tests/integration/defs/examples/serve/test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml`, `tests/integration/defs/examples/serve/test_configs/Nemotron3_Super_120B_NVFP4.yml`, `tests/integration/defs/examples/serve/test_serve.py`, `tests/integration/defs/perf/pytorch_model_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13968 - [None][fix] Fix bugs related with nemotron-nas model

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13968
- 状态/时间: merged / 2026-05-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py`；关联提交 `70951837cd0b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+69/-7，可读 patch 119 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: NemotronNasHfWeightMapper, apply_callbacks, _layer_idx_from_breakdown, _duplicate_kv_weights_for_layer，涉及 `NemotronNasHfWeightMapper, apply_callbacks, _layer_idx_from_breakdown`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: NemotronNasHfWeightMapper, apply_callbacks, _layer_idx_from_breakdown, _duplicate_kv_weights_for_layer
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py
@@ -0,0 +1,64 @@
+from torch import nn
+from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
+from tensorrt_llm._torch.models.modeling_utils import register_mapper
+@register_mapper("HF", "DeciLMForCausalLM")
+class NemotronNasHfWeightMapper(HfWeightMapper):
+    """Weight mapper for Nemotron-NAS / DeciLM.
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py` added +64/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/llmapi/test_llm_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14775 - [TRTLLM-12288][feat] Support Nemotron-H nvfp4 ckpt on Hopper

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14775
- 状态/时间: merged / 2026-06-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `71a188ccd8c5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+745/-14，可读 patch 875 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +122/-7 (129 lines); hunks: -13,7 +13,9; -31,6 +33,7; symbols: __init__, forward, _force_moe_backend_for_w4a16_on_hopper, _use_w4a16_for_nvfp4_on_hopper，涉及 `__init__, forward, _force_moe_backend_for_w4a16_on_hopper`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +122/-7 (129 lines); hunks: -13,7 +13,9; -31,6 +33,7; symbols: __init__, forward, _force_moe_backend_for_w4a16_on_hopper, _use_w4a16_for_nvfp4_on_hopper
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -13,7 +13,9 @@
+import os
+from contextlib import contextmanager
@@ -31,6 +33,7 @@
+from tensorrt_llm.models.modeling_utils import QuantAlgo  # noqa: E402
@@ -39,7 +42,11 @@
-from ..modules.linear import Linear, TensorParallelMode
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +122/-7
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_h100.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14964 - [TRTLLM-13177][doc] Add Nemotron 3 Ultra doc

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14964
- 状态/时间: merged / 2026-06-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md`, `examples/configs/curated/nemotron-3-ultra-throughput.yaml`, `examples/models/core/nemotron/README_nemotron_super_v3.md`；关联提交 `428cc3eb68ac`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+128/-18，可读 patch 290 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `examples/configs/curated/nemotron-3-ultra-throughput.yaml` added +19/-0 (19 lines); hunks: -0,0 +1,19；`examples/models/core/nemotron/README_nemotron_super_v3.md` modified +1/-1 (2 lines); hunks: -200,4 +200,4 @@ Key options:；`docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md` renamed +80/-15 (95 lines); hunks: -1,8 +1,13; -14,6 +19,13 @@ This deployment guide provides step-by-step instructions for...。
- 代码 diff 细节:
  - `examples/configs/curated/nemotron-3-ultra-throughput.yaml` added +19/-0 (19 lines); hunks: -0,0 +1,19
  - `examples/models/core/nemotron/README_nemotron_super_v3.md` modified +1/-1 (2 lines); hunks: -200,4 +200,4 @@ Key options:
  - `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md` renamed +80/-15 (95 lines); hunks: -1,8 +1,13; -14,6 +19,13 @@ This deployment guide provides step-by-step instructions for...
- 关键代码摘录:

```diff
diff -- examples/configs/curated/nemotron-3-ultra-throughput.yaml
@@ -0,0 +1,19 @@
+max_batch_size: 256
+max_num_tokens: 2048
+tensor_parallel_size: 4
+moe_expert_parallel_size: 4
+trust_remote_code: true
+enable_attention_dp: true
diff -- examples/models/core/nemotron/README_nemotron_super_v3.md
@@ -200,4 +200,4 @@ Key options:
-* For detailed deployment instructions, see the [deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-nemotron-3-su
+* For detailed deployment instructions, see the [deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-nemotron-3-on
diff -- docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md
@@ -1,8 +1,13 @@
-# Deployment Guide for Nemotron v3 Super on TensorRT LLM - Blackwell & Hopper Hardware
+# Deployment Guide for Nemotron v3 (Ultra & Super) on TensorRT LLM - Blackwell & Hopper Hardware
-This deployment guide provides step-by-step instructions for running the NVIDIA Nemotron v3 Super 120B-A12B model using TensorRT LLM. Nemotron v3 Super is a hybrid architecture mo
+This deployment guide provides step-by-step instructions for running the NVIDIA Nemotron v3 family of models using TensorRT LLM. It covers two models:
```

- 提取文件（未人工审阅）:
  - docs: `examples/configs/curated/nemotron-3-ultra-throughput.yaml` added +19/-0; `examples/models/core/nemotron/README_nemotron_super_v3.md` modified +1/-1; `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md` renamed +80/-15
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/_static/config_db.json`, `docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md`, `docs/source/deployment-guide/index.rst`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #15294 - [https://nvbugs/6264844][fix] Fix wrong NCCL fallback in nemotron-h

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15294
- 状态/时间: merged / 2026-06-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `0ff7b4acba23`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 36 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-0 (6 lines); hunks: -277,9 +277,11 @@ def _moe(name):; -488,9 +490,11 @@ def __init__(; symbols: _moe, __init__, forward，涉及 `_moe, __init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-0 (6 lines); hunks: -277,9 +277,11 @@ def _moe(name):; -488,9 +490,11 @@ def __init__(; symbols: _moe, __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -277,9 +277,11 @@ def _moe(name):
+            # AllReduce needs dtype at construction to build fused MNNVL paths.
+                dtype=config.torch_dtype,
@@ -488,9 +490,11 @@ def __init__(
+            # AllReduce needs dtype at construction to build fused MNNVL paths.
+                dtype=config.torch_dtype,
@@ -717,9 +721,11 @@ def __init__(self, model_config: NemotronHModelConfig):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15573 - [None][feat] Support update weight for nemotron-h

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15573
- 状态/时间: merged / 2026-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `f397bb00265e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+243/-32，可读 patch 364 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +30/-2 (32 lines); hunks: -1,3 +1,5; -59,7 +61,6 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:; symbols: _split_mamba2_mixer_in_proj，涉及 `_split_mamba2_mixer_in_proj`；`tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-2 (9 lines); hunks: -972,9 +972,14 @@ def _is_moe(bc):; symbols: _is_moe, load_weights, get_model_defaults，涉及 `_is_moe, load_weights, get_model_defaults`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +30/-2 (32 lines); hunks: -1,3 +1,5; -59,7 +61,6 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:; symbols: _split_mamba2_mixer_in_proj
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-2 (9 lines); hunks: -972,9 +972,14 @@ def _is_moe(bc):; symbols: _is_moe, load_weights, get_model_defaults
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -1,3 +1,5 @@
+import re
@@ -59,7 +61,6 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
-                import re
@@ -126,7 +127,34 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
-                    if "up_proj" in key:
+                    # HF transformers 5.x exposes routed MoE experts as fused
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -972,9 +972,14 @@ def _is_moe(bc):
-    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
+    def load_weights(self,
+                     weights: dict,
+                     weight_mapper: BaseWeightMapper,
+                     allow_partial_loading: bool = False):
-        super().load_weights(weights=new_weights, weight_mapper=weight_mapper)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +30/-2; `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-2
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/ray_orchestrator/multi_gpu/test_llm_update_weights_multi_gpu.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15582 - [None][feat] Support Nemotron dynamic-tree MTP decoding

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15582
- 状态/时间: merged / 2026-07-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`；关联提交 `858fd17b79b2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+1514/-80，可读 patch 1979 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-0 (2 lines); hunks: -1275,6 +1275,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-0 (2 lines); hunks: -1275,6 +1275,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -1275,6 +1275,8 @@ def forward(
+                spec_metadata=spec_metadata,
+                mamba_metadata=attn_metadata.mamba_metadata,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/speculative/test_eagle3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16833 - [None][fix] Fix nemotron-h quant and loading config

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16833
- 状态/时间: merged / 2026-07-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`；关联提交 `becf773b5a70`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+116/-38，可读 patch 196 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +47/-5 (52 lines); hunks: -1,6 +1,8; -42,10 +44,14 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Te...; symbols: _split_mamba2_mixer_in_proj, _num_rows, _duplicate_kv_weights，涉及 `_split_mamba2_mixer_in_proj, _num_rows, _duplicate_kv_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +47/-5 (52 lines); hunks: -1,6 +1,8; -42,10 +44,14 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Te...; symbols: _split_mamba2_mixer_in_proj, _num_rows, _duplicate_kv_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -1,6 +1,8 @@
+from typing import Optional
+from torch import nn
@@ -42,10 +44,14 @@ def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
-        is_nvfp4 = self.config.quant_config.quant_algo == "NVFP4"
+        # Full in_proj out_features = concat([z, x, B, C, dt]). Only its
+        # per-output-row block scale spans this dim 0 and takes the same
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +47/-5
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17609 - [None][test] Nemotron-Ultra-V3 perf-sanity cases (GB300); de-enroll DeepSeek-V3.2, Kimi-K2.5 & Llama cases

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17609
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con12_ctx1_dep4_gen6_tep4_eplb0_mtp6_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con9832_ctx1_dep4_gen8_dep8_eplb0_mtp3_ccb-NIXL.yaml`；关联提交 `f8c7f55b2be0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+997/-120，可读 patch 1295 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml` added +155/-0 (155 lines); hunks: -0,0 +1,155；`tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +126/-0 (126 lines); hunks: -0,0 +1,126；`tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` added +120/-0 (120 lines); hunks: -0,0 +1,120；`tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con9832_ctx1_dep4_gen8_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +120/-0 (120 lines); hunks: -0,0 +1,120。
- 代码 diff 细节:
  - `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml` added +155/-0 (155 lines); hunks: -0,0 +1,155
  - `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +126/-0 (126 lines); hunks: -0,0 +1,126
  - `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` added +120/-0 (120 lines); hunks: -0,0 +1,120
  - `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con9832_ctx1_dep4_gen8_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +120/-0 (120 lines); hunks: -0,0 +1,120
  - `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con12_ctx1_dep4_gen6_tep4_eplb0_mtp6_ccb-NIXL.yaml` added +118/-0 (118 lines); hunks: -0,0 +1,118
- 关键代码摘录:

```diff
diff -- tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml
@@ -0,0 +1,155 @@
+# Nemotron-Ultra-V3 (mixed NVFP4/FP8) — aggregated 50k/2k perf-sanity points on GB300.
+# All three run on 4 GPUs as a single trtllm-serve instance (prefill+decode colocated).
+#
+#   con1   TEP4 bs1   MTP6      81.0 tok/s/GPU   498.0 tok/s/user   min latency
+#   con128 DEP4 bs32  MTP3     508.2 tok/s/GPU    19.7 tok/s/user   balanced
+#   con512 DEP4 bs128 MTP-off  539.6 tok/s/GPU     5.6 tok/s/user   max throughput
diff -- tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,126 @@
+# Nemotron-Ultra-V3 (mixed NVFP4 weights + FP8 KV + fp16 Mamba cache),
+# disaggregated E2E, 50K ISL / 2K OSL. MAX-THROUGHPUT point.
+# Measured on GB300 NVL72 (72 GPUs = 16 ctx x DEP4 + 1 gen x DEP8):
+#   concurrency 1197, out_tps 38904, tps/gpu 540.3, TPOT 24.69 ms (40 tok/s/user), TTFT 11.46 s.
+# CI enrolls this point as ctx_only ONLY (the 72-GPU / 18-node e2e + gen_only
+# topology is intentionally not created); the aggr ctx_only id reads
diff -- tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml
@@ -0,0 +1,120 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/scripts/perf-sanity/aggregated/gb300_nemotron_ultra_v3_fp4_grace_blackwell.yaml` added +155/-0; `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con1197_ctx16_dep4_gen1_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +126/-0; `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con1_ctx1_dep4_gen1_tep4_eplb0_mtp5_ccb-NIXL.yaml` added +120/-0; `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_8k64k_con9832_ctx1_dep4_gen8_dep8_eplb0_mtp3_ccb-NIXL.yaml` added +120/-0; `tests/scripts/perf-sanity/disaggregated/gb300_nemotron-ultra-v3-fp4_50k2k_con12_ctx1_dep4_gen6_tep4_eplb0_mtp6_ccb-NIXL.yaml` added +118/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/perf/_model_paths.py`, `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17601 - [TRTLLM-15100][test] Prune Nemotron-H functional and un…

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17601
- 状态/时间: merged / 2026-08-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `2c8522ef0479`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+92/-403，可读 patch 641 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +82/-311 (393 lines); hunks: -13,19 +13,17; -38,7 +36,8 @@ def create_nemotron_h_llm(model_folder,; symbols: get_logprobs, extract_prefill_logprobs, extract_decode_logprobs, create_nemotron_h_llm，涉及 `get_logprobs, extract_prefill_logprobs, extract_decode_logprobs`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +82/-311 (393 lines); hunks: -13,19 +13,17; -38,7 +36,8 @@ def create_nemotron_h_llm(model_folder,; symbols: get_logprobs, extract_prefill_logprobs, extract_decode_logprobs, create_nemotron_h_llm
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -13,19 +13,17 @@
-    assert index.device == raw_probs.device, f"index and raw_probs should be on the same device, but got index location: {index.device}, raw_probs location: {raw_probs.device}"
+    assert index.device == raw_probs.device, (
+        "index and raw_probs should be on the same device, "
+        f"but got index location: {index.device}, raw_probs location: {raw_probs.device}"
+    )
-def extract_prefill_logprobs(result: RequestOutput) -> torch.Tensor:
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +82/-311
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_autodeploy.py`, `tests/integration/defs/test_e2e.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17858 - [TRTLLM-15040][test] Prune legacy Llama and Nemotron tests

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/17858
- 状态/时间: merged / 2026-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py`；关联提交 `1ed2928d3da7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 62 个文件，+452/-3589，可读 patch 5061 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` removed +0/-525 (525 lines); hunks: -1,525 +0,0; symbols: Scenario, __repr__, reduce_nemotron_nas_config, TestNemotronNAS，涉及 `Scenario, __repr__, reduce_nemotron_nas_config`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` removed +0/-525 (525 lines); hunks: -1,525 +0,0; symbols: Scenario, __repr__, reduce_nemotron_nas_config, TestNemotronNAS
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py
@@ -1,525 +0,0 @@
-import unittest
-from copy import deepcopy
-from dataclasses import dataclass
-from typing import Any
-import torch
-import transformers
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_nas.py` removed +0/-525
- 验证与风险: diff 自带测试面 `tests/integration/defs/.test_durations`, `tests/integration/defs/.test_durations_aws_dfw`, `tests/integration/defs/accuracy/references/SlimPajama-6B.yaml`, `tests/integration/defs/accuracy/references/cnn_dailymail.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18888 - [TRTLLM-15936][fix] Enable breakable prefill CUDA graphs (BCG) for Nemotron-H hybrid models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18888
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `fa9813cc4d1d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+440/-43，可读 patch 602 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +188/-2 (190 lines); hunks: -1,12 +1,14; -216,3 +218,187 @@ def test_nemotron_h_chunked_prefill():; symbols: test_nemotron_h_chunked_prefill, _first_step_logprobs, _assert_mixed_batch_overlap, _run_nemotron_h_prefill_backend，涉及 `test_nemotron_h_chunked_prefill, _first_step_logprobs, _assert_mixed_batch_overlap`；`tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +231/-41 (272 lines); hunks: -15,6 +15,8; -30,7 +32,10; symbols: _extract_mamba2_extra_attrs, mamba2_custom_op_inplace, _flashinfer_selective_state_update_op, _，涉及 `_extract_mamba2_extra_attrs, mamba2_custom_op_inplace, _flashinfer_selective_state_update_op`；`tensorrt_llm/_torch/compilation/utils.py` modified +15/-0 (15 lines); hunks: -201,6 +201,21 @@ def inplace_info():; symbols: inplace_info，涉及 `inplace_info`；`tensorrt_llm/_torch/compilation/piecewise_optimizer.py` modified +1/-0 (1 lines); hunks: -26,6 +26,7 @@ def _piecewise_boundary_ops():; symbols: _piecewise_boundary_ops，涉及 `_piecewise_boundary_ops`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +188/-2 (190 lines); hunks: -1,12 +1,14; -216,3 +218,187 @@ def test_nemotron_h_chunked_prefill():; symbols: test_nemotron_h_chunked_prefill, _first_step_logprobs, _assert_mixed_batch_overlap, _run_nemotron_h_prefill_backend
  - `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +231/-41 (272 lines); hunks: -15,6 +15,8; -30,7 +32,10; symbols: _extract_mamba2_extra_attrs, mamba2_custom_op_inplace, _flashinfer_selective_state_update_op, _
  - `tensorrt_llm/_torch/compilation/utils.py` modified +15/-0 (15 lines); hunks: -201,6 +201,21 @@ def inplace_info():; symbols: inplace_info
  - `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` modified +1/-0 (1 lines); hunks: -26,6 +26,7 @@ def _piecewise_boundary_ops():; symbols: _piecewise_boundary_ops
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -1,12 +1,14 @@
-from utils.util import skip_fp8_pre_ada, skip_gpu_memory_less_than
+from utils.util import (skip_fp8_pre_ada, skip_gpu_memory_less_than,
+                        skip_single_gpu)
-from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, LoadFormat
+from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, LoadFormat,
+                                          PrefillCudaGraphBackend)
diff -- tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py
@@ -15,6 +15,8 @@
+import weakref
+from typing import Optional
@@ -30,7 +32,10 @@
+from ...pyexecutor.breakable_cuda_graph import (eager_on_graph,
+                                                is_in_breakable_cuda_graph)
+from ...utils import get_model_extra_attrs, is_torch_compiling
diff -- tensorrt_llm/_torch/compilation/utils.py
@@ -201,6 +201,21 @@ def inplace_info():
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +188/-2
  - runtime: `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` modified +231/-41; `tensorrt_llm/_torch/compilation/utils.py` modified +15/-0; `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_dgx_h100.yml`, `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19168 - [https://nvbugs/6708111][fix] Load Nemotron-H saved by transformers>=5.13

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19168
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py`；关联提交 `df569f4bbe90`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+176/-12，可读 patch 243 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: _load, TestLoader, test_renamed_checkpoint_regains_its_pattern, test_parsable_checkpoint_keeps_its_own_result，涉及 `_load, TestLoader, test_renamed_checkpoint_regains_its_pattern`；`tensorrt_llm/tokenizer/tokenizer.py` modified +9/-5 (14 lines); hunks: -299,18 +299,22 @@ def from_pretrained(cls, pretrained_model_dir: str, **kwar...; symbols: from_pretrained，涉及 `from_pretrained`；`tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +65/-2 (67 lines); hunks: -797,6 +797,49 @@ def __getitem__(self, key):; -946,8 +989,28 @@ def load_pretrained_config(model_name_or_path: str,; symbols: __getitem__, nemotron_h_legacy_layer_types, match_nemotron_h_layer_types, load_pretrained_config，涉及 `__getitem__, nemotron_h_legacy_layer_types, match_nemotron_h_layer_types`；`tensorrt_llm/_torch/speculative/utils.py` modified +13/-5 (18 lines); hunks: -14,6 +14,7; -259,23 +260,30 @@ def _merge_mtp_fields_from_speculative_model(spec_config,; symbols: _merge_mtp_fields_from_speculative_model，涉及 `_merge_mtp_fields_from_speculative_model`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: _load, TestLoader, test_renamed_checkpoint_regains_its_pattern, test_parsable_checkpoint_keeps_its_own_result
  - `tensorrt_llm/tokenizer/tokenizer.py` modified +9/-5 (14 lines); hunks: -299,18 +299,22 @@ def from_pretrained(cls, pretrained_model_dir: str, **kwar...; symbols: from_pretrained
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +65/-2 (67 lines); hunks: -797,6 +797,49 @@ def __getitem__(self, key):; -946,8 +989,28 @@ def load_pretrained_config(model_name_or_path: str,; symbols: __getitem__, nemotron_h_legacy_layer_types, match_nemotron_h_layer_types, load_pretrained_config
  - `tensorrt_llm/_torch/speculative/utils.py` modified +13/-5 (18 lines); hunks: -14,6 +14,7; -259,23 +260,30 @@ def _merge_mtp_fields_from_speculative_model(spec_config,; symbols: _merge_mtp_fields_from_speculative_model
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py
@@ -0,0 +1,89 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+"""Nemotron-H layer-vocabulary compatibility.
+transformers 5.13 renamed the per-layer vocabulary and stopped serializing
+``hybrid_override_pattern``, so a checkpoint re-saved under a newer transformers
+cannot be parsed by the pinned one. The pattern drives layer counts and the
diff -- tensorrt_llm/tokenizer/tokenizer.py
@@ -299,18 +299,22 @@ def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
-            # Two transformers 5.x regressions for model_types not registered
-            # in CONFIG_MAPPING_NAMES. PreTrainedTokenizerFast reads
-            # tokenizer.json directly and skips AutoConfig, so it sidesteps
-            # both:
+            # Three transformers 5.x regressions for model_types not
+            # registered in CONFIG_MAPPING_NAMES. PreTrainedTokenizerFast reads
diff -- tensorrt_llm/_torch/pyexecutor/config_utils.py
@@ -797,6 +797,49 @@ def __getitem__(self, key):
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py` added +89/-0
  - runtime: `tensorrt_llm/tokenizer/tokenizer.py` modified +9/-5; `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +65/-2; `tensorrt_llm/_torch/speculative/utils.py` modified +13/-5
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_nemotron_h_layer_vocabulary.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19146 - [None][chore] Name the Nemotron multimodal module for the family it serves instead of one of its three models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19146
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`, `tests/unittest/_torch/multimodal/test_nemotron_h_multimodal_encoder_groups.py`；关联提交 `64e3b82cceae`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+162/-147，可读 patch 1034 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` renamed +49/-41 (90 lines); hunks: -15,15 +15,15; -71,17 +71,17 @@ def _make_minimal_nano_model_config():; symbols: _make_minimal_nano_model_config, test_nemotron_nano_registers_native_multimodal_epd_components, advertises, _assert_nano_video_handoff，涉及 `_make_minimal_nano_model_config, test_nemotron_nano_registers_native_multimodal_epd_components, advertises`；`tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` renamed +45/-43 (88 lines); hunks: -585,7 +585,7 @@ def _normalize_vision_weights(weights: Mapping[str, torch.Te...; -906,7 +906,7 @@ def forward(; symbols: _normalize_vision_weights, NanoV2VLVisionEncoder, NemotronHVisionEncoder, __init__，涉及 `_normalize_vision_weights, NanoV2VLVisionEncoder, NemotronHVisionEncoder`；`tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` renamed +42/-38 (80 lines); hunks: -1,5 +1,5; -13,13 +13,13; symbols: test_compute_params_raises_on_unconvergeable, _make_processor, _make_nano_processor，涉及 `test_compute_params_raises_on_unconvergeable, _make_processor, _make_nano_processor`；`tests/unittest/_torch/multimodal/test_nemotron_h_multimodal_encoder_groups.py` renamed +5/-5 (10 lines); hunks: -1,6 +1,6; -14,7 +14,7; symbols: _marker_tensor, _model_with_stubs, _vision，涉及 `_marker_tensor, _model_with_stubs, _vision`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` renamed +49/-41 (90 lines); hunks: -15,15 +15,15; -71,17 +71,17 @@ def _make_minimal_nano_model_config():; symbols: _make_minimal_nano_model_config, test_nemotron_nano_registers_native_multimodal_epd_components, advertises, _assert_nano_video_handoff
  - `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` renamed +45/-43 (88 lines); hunks: -585,7 +585,7 @@ def _normalize_vision_weights(weights: Mapping[str, torch.Te...; -906,7 +906,7 @@ def forward(; symbols: _normalize_vision_weights, NanoV2VLVisionEncoder, NemotronHVisionEncoder, __init__
  - `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` renamed +42/-38 (80 lines); hunks: -1,5 +1,5; -13,13 +13,13; symbols: test_compute_params_raises_on_unconvergeable, _make_processor, _make_nano_processor
  - `tests/unittest/_torch/multimodal/test_nemotron_h_multimodal_encoder_groups.py` renamed +5/-5 (10 lines); hunks: -1,6 +1,6; -14,7 +14,7; symbols: _marker_tensor, _model_with_stubs, _vision
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7 @@ def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor...; symbols: get_logprobs, extract_decode_logprobs
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py
@@ -15,15 +15,15 @@
-from tensorrt_llm._torch.models import modeling_nemotron_nano as nemotron_nano
+from tensorrt_llm._torch.models import modeling_nemotron_h_multimodal as nemotron_h_multimodal
-from tensorrt_llm._torch.models.modeling_nemotron_nano import (
-    NanoV2VLInputProcessor,
-    NanoV2VLMultimodalEncoder,
-    NanoV2VLVisionEncoder,
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py
@@ -585,7 +585,7 @@ def _normalize_vision_weights(weights: Mapping[str, torch.Tensor]) -> Dict[str,
-class NanoV2VLVisionEncoder(transformers.PreTrainedModel):
+class NemotronHVisionEncoder(transformers.PreTrainedModel):
@@ -906,7 +906,7 @@ def forward(
-                    "NanoV2VLVisionEncoder expects exactly one of image / video "
+                    "NemotronHVisionEncoder expects exactly one of image / video "
@@ -1076,13 +1076,13 @@ def _video_tubelet_geometry(self, t: int, T: int, ih: int, iw: int) -> Tuple[int
diff -- tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py
@@ -1,5 +1,5 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` renamed +49/-41; `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` renamed +42/-38; `tests/unittest/_torch/multimodal/test_nemotron_h_multimodal_encoder_groups.py` renamed +5/-5; `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +1/-1
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` renamed +45/-43
- 验证与风险: diff 自带测试面 `tests/integration/defs/.test_durations`, `tests/integration/test_lists/test-db/l0_l40s.yml`, `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/executor/kv_cache/test_mamba_cache_manager.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19340 - [None][test] Add nemotron_3.5_lightning_30b_nvfp4 and nemotron_3.5_lightning_30b_bf16 func and perf cases on Spark

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19340
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml`；关联提交 `55f7eca86161`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+272/-76，可读 patch 459 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml` added +21/-0 (21 lines); hunks: -0,0 +1,21。
- 代码 diff 细节:
  - `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml` added +21/-0 (21 lines); hunks: -0,0 +1,21
- 关键代码摘录:

```diff
diff -- tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml
@@ -0,0 +1,21 @@
+kv_cache_config:
+  dtype: fp8
+  enable_block_reuse: false
+  mamba_state_config:
+     periodic_snapshot_interval: 8192
+  free_gpu_memory_fraction: 0.8
```

- 提取文件（未人工审阅）:
  - tests: `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml` added +21/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/examples/serve/test_configs/Nemotron35_Lightning_30B.yml`, `tests/integration/defs/examples/serve/test_serve.py`, `tests/integration/defs/perf/_model_paths.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19335 - [https://nvbugs/6777501][fix] Fix nemotron breakable cuda graph test parity check

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19335
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；关联提交 `e1e562d3e63d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+39/-17，可读 patch 112 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +39/-12 (51 lines); hunks: -229,14 +229,22 @@ def test_nemotron_h_chunked_prefill():; -322,7 +330,7 @@ def _run_nemotron_h_prefill_backend(backend: PrefillCudaGrap...; symbols: test_nemotron_h_chunked_prefill, _run_nemotron_h_prefill_backend, test_nemotron_h_breakable_prefill_cuda_graph，涉及 `test_nemotron_h_chunked_prefill, _run_nemotron_h_prefill_backend, test_nemotron_h_breakable_prefill_cuda_graph`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +39/-12 (51 lines); hunks: -229,14 +229,22 @@ def test_nemotron_h_chunked_prefill():; -322,7 +330,7 @@ def _run_nemotron_h_prefill_backend(backend: PrefillCudaGrap...; symbols: test_nemotron_h_chunked_prefill, _run_nemotron_h_prefill_backend, test_nemotron_h_breakable_prefill_cuda_graph
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h.py
@@ -229,14 +229,22 @@ def test_nemotron_h_chunked_prefill():
+# Every prompt repeats one trained token. Ids 0-513 of the Nemotron-3 vocab are
+# added control tokens, most of them untrained <SPECIAL_n> placeholders; a prompt
+# built from one (the Qwen3.5 parity test's ids 23 and 31, which are ordinary
+# BPE tokens there) gives a flat next-token distribution whose leading
+# candidates tie within run-to-run noise (https://nvbugs/6777501). Id 17 is the
+# trained </tool_response> token. NemotronH disables KV block reuse by default,
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py` modified +39/-12
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19151 - [TRTLLM-15820][feat] Expose Nemotron-H VL LoRA configuration hook

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19151
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py`；关联提交 `c7b96cda9b86`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+646/-84，可读 patch 789 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +10/-0 (10 lines); hunks: -55,6 +55,7; -2954,6 +2955,15 @@ def post_config(self):; symbols: post_config, lora_config, _build_evs_adjusted_context_ids，涉及 `post_config, lora_config, _build_evs_adjusted_context_ids`；`tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: read_llm_config, layer_plan, projection_dims, _projections_for，涉及 `read_llm_config, layer_plan, projection_dims`；`tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: mini_checkpoint, test_read_llm_config_unwraps_the_vl_nesting, test_read_llm_config_accepts_a_flat_text_only_config, test_layer_plan_accepts_both_spellings，涉及 `mini_checkpoint, test_read_llm_config_unwraps_the_vl_nesting, test_read_llm_config_accepts_a_flat_text_only_config`；`tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` modified +11/-84 (95 lines); hunks: -14,99 +14,22; -139,7 +62,9 @@ def _run_generate(self, llm, lora_dir, prompts):; symbols: _create_lora_adapter, _add, _layer, _get_lora_config，涉及 `_create_lora_adapter, _add, _layer`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +10/-0 (10 lines); hunks: -55,6 +55,7; -2954,6 +2955,15 @@ def post_config(self):; symbols: post_config, lora_config, _build_evs_adjusted_context_ids
  - `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py` added +320/-0 (320 lines); hunks: -0,0 +1,320; symbols: read_llm_config, layer_plan, projection_dims, _projections_for
  - `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: mini_checkpoint, test_read_llm_config_unwraps_the_vl_nesting, test_read_llm_config_accepts_a_flat_text_only_config, test_layer_plan_accepts_both_spellings
  - `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` modified +11/-84 (95 lines); hunks: -14,99 +14,22; -139,7 +62,9 @@ def _run_generate(self, llm, lora_dir, prompts):; symbols: _create_lora_adapter, _add, _layer, _get_lora_config
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py
@@ -55,6 +55,7 @@
+from ..peft.lora.config import LoraConfig
@@ -2954,6 +2955,15 @@ def post_config(self):
+    @classmethod
+    def lora_config(cls, model_dir: str) -> LoraConfig:
+        """Return the decoder's LoRA target configuration.
+        Callers supply adapter paths and capacity through this configuration
diff -- tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py
@@ -0,0 +1,320 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py
@@ -0,0 +1,296 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +10/-0
  - tests: `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py` added +320/-0; `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py` added +296/-0; `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py` modified +11/-84
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/tests_lora_modules/nemotron_h_lora_utils.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron35_vl_lora_adapter.py`, `tests/unittest/_torch/modules/tests_lora_modules/test_nemotron_h_lora_sanity.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19558 - [None][fix] Fix quantized MTP head loading for Nemotron H

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19558
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py`；关联提交 `923c24fdd0be`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+174/-26，可读 patch 360 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +65/-18 (83 lines); hunks: -83,6 +83,18 @@ def _get_layer_moe_param(config, layer_idx: int, param_name:...; -171,12 +183,19 @@ def __init__(; symbols: _get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer, __init__，涉及 `_get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer`；`tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` modified +108/-8 (116 lines); hunks: -16,11 +16,16; -144,10 +149,71 @@ def fake_create_moe(**kwargs):; symbols: fake_create_moe, test_nemotron_h_mtp_overrides_quant_and_inherits_moe_backend, test_nemotron_h_moe_uses_module_prefix_for_mtp_sublayer_quant_config, test_remap_hf_quant_module_name，涉及 `fake_create_moe, test_nemotron_h_mtp_overrides_quant_and_inherits_moe_backend, test_nemotron_h_moe_uses_module_prefix_for_mtp_sublayer_quant_config`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +65/-18 (83 lines); hunks: -83,6 +83,18 @@ def _get_layer_moe_param(config, layer_idx: int, param_name:...; -171,12 +183,19 @@ def __init__(; symbols: _get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer, __init__
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` modified +108/-8 (116 lines); hunks: -16,11 +16,16; -144,10 +149,71 @@ def fake_create_moe(**kwargs):; symbols: fake_create_moe, test_nemotron_h_mtp_overrides_quant_and_inherits_moe_backend, test_nemotron_h_moe_uses_module_prefix_for_mtp_sublayer_quant_config, test_remap_hf_quant_module_name
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -83,6 +83,18 @@ def _get_layer_moe_param(config, layer_idx: int, param_name: str):
+def _remap_hf_quant_module_name(name: str, num_hidden_layers: int) -> str:
+    """Map an HF-checkpoint module name or glob onto the TRT-LLM module tree.
+    """
+    name = re.sub(r"(model\.layers\.)?backbone", "model", name)
+    mtp_root = f"model.layers.{num_hidden_layers}"
+    if name in ("mtp", "mtp*", "mtp.*"):
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py
@@ -16,11 +16,16 @@
+import pytest
-from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE, NemotronHMTP
+from tensorrt_llm._torch.models.modeling_nemotron_h import (
+    NemotronHMOE,
+    NemotronHMTP,
+    _remap_hf_quant_module_name,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +65/-18
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` modified +108/-8
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19648 - [TRTLLM-15824][feat] Enable EVS for Nemotron Super 3.5 VL

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19648
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`；关联提交 `313c4858e4a5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+239/-9，可读 patch 323 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +199/-0 (199 lines); hunks: -11,8 +11,10; -1294,6 +1296,203 @@ def test_placeholder_count_matches_tubelets(self, num_se...; symbols: test_placeholder_count_matches_tubelets, TestEvsVideoContextTokenId, _make_model, test_processor_and_model_resolve_same_video_id，涉及 `test_placeholder_count_matches_tubelets, TestEvsVideoContextTokenId, _make_model`；`tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +40/-9 (49 lines); hunks: -65,6 +65,7; -90,6 +91,14; symbols: _resolve_video_context_token_id, _compute_aspect_preserving_size, __init__，涉及 `_resolve_video_context_token_id, _compute_aspect_preserving_size, __init__`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +199/-0 (199 lines); hunks: -11,8 +11,10; -1294,6 +1296,203 @@ def test_placeholder_count_matches_tubelets(self, num_se...; symbols: test_placeholder_count_matches_tubelets, TestEvsVideoContextTokenId, _make_model, test_processor_and_model_resolve_same_video_id
  - `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +40/-9 (49 lines); hunks: -65,6 +65,7; -90,6 +91,14; symbols: _resolve_video_context_token_id, _compute_aspect_preserving_size, __init__
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py
@@ -11,8 +11,10 @@
+import transformers
+from tensorrt_llm._torch.models.modeling_multimodal_utils import fuse_input_embeds
@@ -1294,6 +1296,203 @@ def test_placeholder_count_matches_tubelets(self, num_seps):
+class TestEvsVideoContextTokenId:
+    @staticmethod
+    def _make_model(video_context_token_id: int | None) -> NemotronHMultimodalModel:
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py
@@ -65,6 +65,7 @@
+    filter_mm_token_from_input_ids,
@@ -90,6 +91,14 @@
+def _resolve_video_context_token_id(config: transformers.PretrainedConfig) -> int:
+    """Resolve the internal marker replaced by image tokens during EVS merge."""
+    token_id = getattr(config, "video_context_token_id", None)
+    # Tokenizer IDs are nonnegative. This marker stays in evs_ids and is
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +199/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +40/-9
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19741 - [https://nvbugs/6625695][fix] Compare teacher-forced logits in the Nemotron VL video batch-equivalence test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19741
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`；关联提交 `de1d696889ae`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+89/-55，可读 patch 186 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +89/-54 (143 lines); hunks: -40,13 +40,14; -576,74 +577,108 @@ def _build_inputs(prompts_subset, media_subset):; symbols: _build_inputs, _ForceTokenScript, __init__, __call__，涉及 `_build_inputs, _ForceTokenScript, __init__`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +89/-54 (143 lines); hunks: -40,13 +40,14; -576,74 +577,108 @@ def _build_inputs(prompts_subset, media_subset):; symbols: _build_inputs, _ForceTokenScript, __init__, __call__
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py
@@ -40,13 +40,14 @@
+from tensorrt_llm.llmapi.llm import RequestOutput
-from tensorrt_llm.sampling_params import SamplingParams
+from tensorrt_llm.sampling_params import LogitsProcessor, SamplingParams
@@ -576,74 +577,108 @@ def _build_inputs(prompts_subset, media_subset):
+class _ForceTokenScript(LogitsProcessor):
+    """Force greedy decoding onto `script`, leaving all other logits untouched.
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +89/-54
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19702 - [TRTLLM-15824][feat] Enable EVS handoff for Nemotron-H in EPD

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19702
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`；关联提交 `6096aa814d4a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+335/-50，可读 patch 545 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +166/-1 (167 lines); hunks: -19,6 +19,7; -32,7 +33,7; symbols: test_expansion_length_matches_get_num_tokens_per_video, TestNemotronEpdEvs, _processor, test_handoff_rebuilds_pruned_video_layout，涉及 `test_expansion_length_matches_get_num_tokens_per_video, TestNemotronEpdEvs, _processor`；`tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +63/-21 (84 lines); hunks: -1125,7 +1125,20 @@ def _encoder_fn(multimodal_params, modality):; -2519,13 +2532,55 @@ def build_disagg_prefill_multimodal_inputs(; symbols: _encoder_fn, build_disagg_prefill_multimodal_inputs, forward，涉及 `_encoder_fn, build_disagg_prefill_multimodal_inputs, forward`；`tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +1/-24 (25 lines); hunks: -128,6 +128,7 @@ def test_nemotron_nano_epd_handoff_preserves_non_contiguous_...; -198,30 +199,6 @@ def test_nemotron_nano_multimodal_encoder_load_by_worker_ro...; symbols: test_nemotron_nano_epd_handoff_preserves_non_contiguous_video_runs, test_nemotron_nano_multimodal_encoder_load_by_worker_role, test_nemotron_nano_rejects_evs_attached_video_embeddings, _spec_forward_stub，涉及 `test_nemotron_nano_epd_handoff_preserves_non_contiguous_video_runs, test_nemotron_nano_multimodal_encoder_load_by_worker_role, test_nemotron_nano_rejects_evs_attached_video_embeddings`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +166/-1 (167 lines); hunks: -19,6 +19,7; -32,7 +33,7; symbols: test_expansion_length_matches_get_num_tokens_per_video, TestNemotronEpdEvs, _processor, test_handoff_rebuilds_pruned_video_layout
  - `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +63/-21 (84 lines); hunks: -1125,7 +1125,20 @@ def _encoder_fn(multimodal_params, modality):; -2519,13 +2532,55 @@ def build_disagg_prefill_multimodal_inputs(; symbols: _encoder_fn, build_disagg_prefill_multimodal_inputs, forward
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +1/-24 (25 lines); hunks: -128,6 +128,7 @@ def test_nemotron_nano_epd_handoff_preserves_non_contiguous_...; -198,30 +199,6 @@ def test_nemotron_nano_multimodal_encoder_load_by_worker_ro...; symbols: test_nemotron_nano_epd_handoff_preserves_non_contiguous_video_runs, test_nemotron_nano_multimodal_encoder_load_by_worker_role, test_nemotron_nano_rejects_evs_attached_video_embeddings, _spec_forward_stub
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py
@@ -19,6 +19,7 @@
+    NemotronHMultimodalEncoder,
@@ -32,7 +33,7 @@
-from tensorrt_llm.inputs.multimodal_data import AudioData
+from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData
@@ -2472,3 +2473,167 @@ def test_expansion_length_matches_get_num_tokens_per_video(self):
+class TestNemotronEpdEvs:
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py
@@ -1125,7 +1125,20 @@ def _encoder_fn(multimodal_params, modality):
-            embeds, _ = super(NemotronHMultimodalEncoder, self).forward(views)
+            embeds, retained_counts = super(NemotronHMultimodalEncoder, self).forward(views)
+            if retained_counts is not None:
+                for param, counts in zip(multimodal_params, retained_counts, strict=True):
+                    if counts is None:
+                        continue
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py
@@ -128,6 +128,7 @@ def test_nemotron_nano_epd_handoff_preserves_non_contiguous_video_runs(
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py` modified +166/-1; `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +1/-24
  - runtime: `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +63/-21
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/executor/engine/test_runners.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_nemotron_h_multimodal_preprocessing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19597 - [None][feat] Load quantized Nemotron-H MTP replacements

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19597
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`；关联提交 `50356f3ee3d3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+1124/-172，可读 patch 1778 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +15/-7 (22 lines); hunks: -1,10 +1,12; -15,6 +17,17; symbols: NemotronHHfWeightMapper, map_mtp_module_name, preprocess_weights, _split_mamba2_mixer_in_proj，涉及 `NemotronHHfWeightMapper, map_mtp_module_name, preprocess_weights`；`tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-15 (22 lines); hunks: -14,7 +14,6; -86,13 +85,9 @@ def _get_layer_moe_param(config, layer_idx: int, param_name:...; symbols: _get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer, _moe，涉及 `_get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer`；`tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +15/-0 (15 lines); hunks: -233,6 +233,21 @@ def _spec_forward_stub():; symbols: _spec_forward_stub, test_nemotron_nano_delegates_draft_loading, test_nemotron_nano_forward_threads_spec_decoding_args，涉及 `_spec_forward_stub, test_nemotron_nano_delegates_draft_loading, test_nemotron_nano_forward_threads_spec_decoding_args`；`tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +14/-0 (14 lines); hunks: -26,6 +26,7; -2929,6 +2930,19 @@ def language_model(self) -> torch.nn.Module:; symbols: language_model, draft_config, draft_model, load_draft_weights，涉及 `language_model, draft_config, draft_model`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +15/-7 (22 lines); hunks: -1,10 +1,12; -15,6 +17,17; symbols: NemotronHHfWeightMapper, map_mtp_module_name, preprocess_weights, _split_mamba2_mixer_in_proj
  - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-15 (22 lines); hunks: -14,7 +14,6; -86,13 +85,9 @@ def _get_layer_moe_param(config, layer_idx: int, param_name:...; symbols: _get_layer_moe_param, _remap_hf_quant_module_name, MLPLayer, _moe
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +15/-0 (15 lines); hunks: -233,6 +233,21 @@ def _spec_forward_stub():; symbols: _spec_forward_stub, test_nemotron_nano_delegates_draft_loading, test_nemotron_nano_forward_threads_spec_decoding_args
  - `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +14/-0 (14 lines); hunks: -26,6 +26,7; -2929,6 +2930,19 @@ def language_model(self) -> torch.nn.Module:; symbols: language_model, draft_config, draft_model, load_draft_weights
  - `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` modified +158/-2 (160 lines); hunks: -13,6 +13,7; -26,6 +27,8; symbols: _make_nemotron_h_moe_config, fake_create_moe, test_nemotron_h_moe_uses_module_prefix_for_mtp_sublayer_quant_config, test_nemotron_h_mtp_quantized_head_inherits_checkpoint_quant_config
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py
@@ -1,10 +1,12 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
-import tensorrt_llm.logger as logger
@@ -15,6 +17,17 @@
+    @staticmethod
+    def map_mtp_module_name(name: str, num_hidden_layers: int) -> str:
diff -- tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -14,7 +14,6 @@
-import re
@@ -86,13 +85,9 @@ def _get_layer_moe_param(config, layer_idx: int, param_name: str):
-    name = re.sub(r"(model\.layers\.)?backbone", "model", name)
-    mtp_root = f"model.layers.{num_hidden_layers}"
-    if name in ("mtp", "mtp*", "mtp.*"):
-        return mtp_root
diff -- tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py
@@ -233,6 +233,21 @@ def _spec_forward_stub():
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` modified +15/-7; `tensorrt_llm/_torch/models/modeling_nemotron_h.py` modified +7/-15; `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py` modified +14/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py` modified +15/-0; `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py` modified +158/-2
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_moe_quant.py`, `tests/unittest/_torch/modeling/test_modeling_nemotron_h_multimodal.py`, `tests/unittest/_torch/modeling/test_modeling_speculative.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
