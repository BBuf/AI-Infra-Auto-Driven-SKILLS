# TensorRT-LLM DeepSeek V4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu` | [#16028](https://github.com/NVIDIA/TensorRT-LLM/pull/16028), [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723) |
| `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h` | [#16028](https://github.com/NVIDIA/TensorRT-LLM/pull/16028) |
| `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` | [#15390](https://github.com/NVIDIA/TensorRT-LLM/pull/15390), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#17273](https://github.com/NVIDIA/TensorRT-LLM/pull/17273) |
| `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` | [#15390](https://github.com/NVIDIA/TensorRT-LLM/pull/15390), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#17273](https://github.com/NVIDIA/TensorRT-LLM/pull/17273) |
| `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp` | [#16028](https://github.com/NVIDIA/TensorRT-LLM/pull/16028) |
| `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` | [#15390](https://github.com/NVIDIA/TensorRT-LLM/pull/15390), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#17273](https://github.com/NVIDIA/TensorRT-LLM/pull/17273) |
| `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md` | [#16539](https://github.com/NVIDIA/TensorRT-LLM/pull/16539) |
| `examples/configs/curated/deepseek-v4-pro-latency.yaml` | [#15919](https://github.com/NVIDIA/TensorRT-LLM/pull/15919) |
| `examples/configs/curated/deepseek-v4-pro-throughput.yaml` | [#15919](https://github.com/NVIDIA/TensorRT-LLM/pull/15919) |
| `examples/models/core/deepseek_v4/README.md` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414), [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/__init__.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#19305](https://github.com/NVIDIA/TensorRT-LLM/pull/19305), [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#19305](https://github.com/NVIDIA/TensorRT-LLM/pull/19305), [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#19305](https://github.com/NVIDIA/TensorRT-LLM/pull/19305) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/flash_mla.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/flashinfer.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/footer_scale_kv.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py` | [#19184](https://github.com/NVIDIA/TensorRT-LLM/pull/19184) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/metadata.py` | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py` | [#19139](https://github.com/NVIDIA/TensorRT-LLM/pull/19139) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/offload.py` | [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/params.py` | [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tensorrt_llm/_torch/configs/deepseekv4.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) |
| `tensorrt_llm/_torch/models/modeling_deepseekv4.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633), [#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224), [#16433](https://github.com/NVIDIA/TensorRT-LLM/pull/16433), [#16881](https://github.com/NVIDIA/TensorRT-LLM/pull/16881), [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940), [#18185](https://github.com/NVIDIA/TensorRT-LLM/pull/18185), [#19184](https://github.com/NVIDIA/TensorRT-LLM/pull/19184) |
| `tensorrt_llm/serve/tool_parser/deepseekv4_parser.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) |
| `tensorrt_llm/tokenizer/deepseek_v4/__init__.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) |
| `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) |
| `tests/integration/defs/examples/test_deepseek_v4_pro.py` | [#15710](https://github.com/NVIDIA/TensorRT-LLM/pull/15710), [#18306](https://github.com/NVIDIA/TensorRT-LLM/pull/18306) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots416.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots384.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots416.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep4_slots384.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep4_slots416.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep8_slots384.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep8_slots416.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` | [#18298](https://github.com/NVIDIA/TensorRT-LLM/pull/18298), [#18633](https://github.com/NVIDIA/TensorRT-LLM/pull/18633) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` | [#18298](https://github.com/NVIDIA/TensorRT-LLM/pull/18298), [#18633](https://github.com/NVIDIA/TensorRT-LLM/pull/18633) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540), [#17137](https://github.com/NVIDIA/TensorRT-LLM/pull/17137) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) |
| `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540), [#19424](https://github.com/NVIDIA/TensorRT-LLM/pull/19424) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots416.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots384.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots416.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep4_slots384.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep4_slots416.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep8_slots384.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep8_slots416.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` | [#18298](https://github.com/NVIDIA/TensorRT-LLM/pull/18298), [#18633](https://github.com/NVIDIA/TensorRT-LLM/pull/18633) |
| `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` | [#18298](https://github.com/NVIDIA/TensorRT-LLM/pull/18298), [#18633](https://github.com/NVIDIA/TensorRT-LLM/pull/18633) |
| `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) |
| `tests/scripts/perf/disaggregated/vr200_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/scripts/perf/disaggregated/vr200_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/unittest/_torch/attention/kernels/test_deepseek_v4_block_table.py` | no direct PR-number commit |
| `tests/unittest/_torch/attention/kernels/test_deepseek_v4_q_norm.py` | no direct PR-number commit |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py` | [#15379](https://github.com/NVIDIA/TensorRT-LLM/pull/15379) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` | [#15379](https://github.com/NVIDIA/TensorRT-LLM/pull/15379), [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` | [#15394](https://github.com/NVIDIA/TensorRT-LLM/pull/15394), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#16734](https://github.com/NVIDIA/TensorRT-LLM/pull/16734), [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940), [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py` | [#15379](https://github.com/NVIDIA/TensorRT-LLM/pull/15379) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` | [#15394](https://github.com/NVIDIA/TensorRT-LLM/pull/15394), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224), [#16466](https://github.com/NVIDIA/TensorRT-LLM/pull/16466), [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723), [#18783](https://github.com/NVIDIA/TensorRT-LLM/pull/18783), [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py` | [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py` | [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py` | [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` | [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#19139](https://github.com/NVIDIA/TensorRT-LLM/pull/19139), [#19184](https://github.com/NVIDIA/TensorRT-LLM/pull/19184) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py` | [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) |
| `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` | [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633) |
| `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414), [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633), [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717), [#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224), [#16433](https://github.com/NVIDIA/TensorRT-LLM/pull/16433), [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940), [#17273](https://github.com/NVIDIA/TensorRT-LLM/pull/17273), [#18185](https://github.com/NVIDIA/TensorRT-LLM/pull/18185), [#19184](https://github.com/NVIDIA/TensorRT-LLM/pull/19184) |
| `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` | [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633), [#16466](https://github.com/NVIDIA/TensorRT-LLM/pull/16466), [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940) |
| `tests/unittest/llmapi/test_deepseek_v4_tokenizer.py` | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) |

## PR Coverage Summary

- Git-traced PRs: 32
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 32
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-06-17 | [#15390](https://github.com/NVIDIA/TensorRT-LLM/pull/15390) | merged | [None][perf] DSv4 prep: attention fusion custom ops | `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` |
| 2026-06-24 | [#15379](https://github.com/NVIDIA/TensorRT-LLM/pull/15379) | merged | [None][feat] DSv4 prep: compressor and mHC primitives | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py`, `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` |
| 2026-06-25 | [#15394](https://github.com/NVIDIA/TensorRT-LLM/pull/15394) | merged | [None][feat] DSv4: sparse cache manager adapter | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` |
| 2026-06-27 | [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409) | merged | [None][feat] DSv4: sparse MLA attention backend | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` |
| 2026-06-28 | [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) | merged | [None][feat] DSv4: model, tokenizer, and integration coverage | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `examples/models/core/deepseek_v4/README.md`, `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py` |
| 2026-07-03 | [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633) | merged | [None][feat] DSv4 follow-up: runtime KV and cache foundations | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` |
| 2026-07-06 | [#15717](https://github.com/NVIDIA/TensorRT-LLM/pull/15717) | merged | [None][perf] DSv4 follow-up: sparse attention and model defaults | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` |
| 2026-07-09 | [#16028](https://github.com/NVIDIA/TensorRT-LLM/pull/16028) | merged | [None][perf] Port remaining DeepSeek V4 optimizations to main | `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu`, `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp`, `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h` |
| 2026-07-09 | [#15710](https://github.com/NVIDIA/TensorRT-LLM/pull/15710) | merged | [None][test] DSv4 PR-6 coverage and import safety | `tests/integration/defs/examples/test_deepseek_v4_pro.py`, `tensorrt_llm/_torch/attention_backend/flashinfer.py`, `tensorrt_llm/_torch/cuda_tile_utils.py` |
| 2026-07-16 | [#15919](https://github.com/NVIDIA/TensorRT-LLM/pull/15919) | merged | [None][feat] Add DeepSeek-V4-Pro curated configs | `examples/configs/curated/deepseek-v4-pro-latency.yaml`, `examples/configs/curated/deepseek-v4-pro-throughput.yaml` |
| 2026-07-17 | [#16539](https://github.com/NVIDIA/TensorRT-LLM/pull/16539) | merged | [None][doc] Add DeepSeek-V4 optimization tech blog | `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md` |
| 2026-07-20 | [#16466](https://github.com/NVIDIA/TensorRT-LLM/pull/16466) | merged | [None][fix] Fix DeepSeek V4 KV cache warmup handling and serveral other issues | `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` |
| 2026-07-20 | [#16540](https://github.com/NVIDIA/TensorRT-LLM/pull/16540) | merged | [None][test] Add DeepSeek-V4-Pro perf sanity cases on GB300 | `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` |
| 2026-07-21 | [#16611](https://github.com/NVIDIA/TensorRT-LLM/pull/16611) | merged | [None][test] Add deepseek v4 pro cases on the qa side | `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` |
| 2026-07-23 | [#16433](https://github.com/NVIDIA/TensorRT-LLM/pull/16433) | merged | [None][fix] Load DeepSeek V4 mixed-precision NVFP4 checkpoints | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` |
| 2026-07-24 | [#16734](https://github.com/NVIDIA/TensorRT-LLM/pull/16734) | merged | [None][perf] Avoid implicit device-scalar syncs in DeepSeek-V4 ctx sparse metadata | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` |
| 2026-07-28 | [#16881](https://github.com/NVIDIA/TensorRT-LLM/pull/16881) | merged | [https://nvbugs/6479863][fix] Use scalar SwiGLU limit for DeepSeek V4 FP8 MoE | `tensorrt_llm/_torch/models/modeling_deepseekv4.py` |
| 2026-08-18 | [#17273](https://github.com/NVIDIA/TensorRT-LLM/pull/17273) | merged | [TRTLLM-14597][perf] Fuse the DSv4 MLA prologue: kv_a_layernorm, q_nope FP8 quant and Q RoPE | `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` |
| 2026-08-24 | [#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224) | merged | [None][feat] Enable DeepSeek-V4 and DSA (DeepSeek-V3.2/GLM) serving on SM120 via FlashInfer sparse-MLA | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` |
| 2026-08-27 | [#18185](https://github.com/NVIDIA/TensorRT-LLM/pull/18185) | merged | [https://nvbugs/6644226][fix] Reduce DeepSeek V4 EPLB loading memory | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` |
| 2026-08-28 | [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940) | merged | [TRTLLM-14116][feat] Add DeepSeek-V4 Hopper support | `examples/models/core/deepseek_v4/README.md`, `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` |
| 2026-08-28 | [#18306](https://github.com/NVIDIA/TensorRT-LLM/pull/18306) | merged | [https://nvbugs/6670227][fix] Drop redundant min_tokens from DSv4-Pro token-boundary smoke | `tests/integration/defs/examples/test_deepseek_v4_pro.py` |
| 2026-08-28 | [#17137](https://github.com/NVIDIA/TensorRT-LLM/pull/17137) | merged | [https://nvbugs/6480621][test] Revert to 60-second KV transfer timeout for GB300 DeepSeek V4 Pro disaggregated perf-sanity | `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` |
| 2026-08-31 | [#18298](https://github.com/NVIDIA/TensorRT-LLM/pull/18298) | merged | [None][test] Add AgentX DeepSeek-V4-Pro-DSpark perf-sanity lanes on GB300 | `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` |
| 2026-09-05 | [#18633](https://github.com/NVIDIA/TensorRT-LLM/pull/18633) | merged | [None][test] Add the 40-GPU AgentX DeepSeek-V4-Pro-DSpark case to post-merge and switch both cases to the NVFP4 checkpoint | `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` |
| 2026-09-16 | [#18723](https://github.com/NVIDIA/TensorRT-LLM/pull/18723) | merged | [None][feat] Enable NVFP4 KV for DSV4 | `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` |
| 2026-09-16 | [#18783](https://github.com/NVIDIA/TensorRT-LLM/pull/18783) | merged | [None][feat] Add DeepSeek-V4 support to NVFP4 cold-page KV Cache Compression | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/nvfp4_quantization.py`, `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` |
| 2026-09-16 | [#19139](https://github.com/NVIDIA/TensorRT-LLM/pull/19139) | merged | [None][fix] Complete DeepSeek-V4 Rubin BF16 dispatch and optimize MLA KV expansion | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py` |
| 2026-09-19 | [#19184](https://github.com/NVIDIA/TensorRT-LLM/pull/19184) | merged | [None][feat] Rubin kernels & attention: DSV4/DSA, CuteDSL GEMM | `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` |
| 2026-09-22 | [#19424](https://github.com/NVIDIA/TensorRT-LLM/pull/19424) | merged | [None][test] Disable ctx overlap scheduler for deepseek-v4-pro con8 disagg config | `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` |
| 2026-09-22 | [#19305](https://github.com/NVIDIA/TensorRT-LLM/pull/19305) | merged | [None][feat] NVFP4 MLA residual switch for DeepSeek-V4 | `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` |
| 2026-10-01 | [#19502](https://github.com/NVIDIA/TensorRT-LLM/pull/19502) | merged | [TRTLLM-16283][feat] Prepare DeepSeek V4 attention for sparse KV offload | `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` |

## Per-PR Diff Audit Cards

### PR #15390 - [None][perf] DSv4 prep: attention fusion custom ops

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15390
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h`, `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp`; associated commits `5fe0a177d7d7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1286/-4, 1373 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` added +158/-0 (158 lines); hunks: -0,0 +1,158; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` added +71/-0 (71 lines); hunks: -0,0 +1,71; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` added +33/-0 (33 lines); hunks: -0,0 +1,33.
- Code diff details:
  - `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` added +158/-0 (158 lines); hunks: -0,0 +1,158
  - `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` added +71/-0 (71 lines); hunks: -0,0 +1,71
  - `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` added +33/-0 (33 lines); hunks: -0,0 +1,33
- Key code excerpts:

```diff
diff -- cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu
@@ -0,0 +1,158 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp
@@ -0,0 +1,71 @@
+/*
+ * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+ * SPDX-License-Identifier: Apache-2.0
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
diff -- cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h
@@ -0,0 +1,33 @@
```

- Extracted files (not manually reviewed):
  - runtime: `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` added +158/-0; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` added +71/-0; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` added +33/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/custom_ops/test_deepseek_v4_q_norm.py`, `tests/unittest/_torch/custom_ops/test_fused_inv_rope_fp8_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15379 - [None][feat] DSv4 prep: compressor and mHC primitives

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15379
- Status/date: merged / 2026-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py`; associated commits `29d08276dbe7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 25 files, +12774/-1, 10125 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` added +2710/-0 (2710 lines); `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: _create_wkv_gate, _bf16_path, _fp32_reference, seed, touching `_create_wkv_gate, _bf16_path, _fp32_reference`; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2; `tensorrt_llm/_torch/modules/mhc/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` added +2710/-0 (2710 lines)
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: _create_wkv_gate, _bf16_path, _fp32_reference, seed
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `tensorrt_llm/_torch/modules/mhc/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py
@@ -0,0 +1,86 @@
+#!/usr/bin/env python3
+# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Tests for the BF16 GEMM path in the DeepSeek-V4 Compressor wkv_gate.
+The wkv_gate weight now matches the upstream V4 checkpoint dtype (bf16). The
+GEMM runs in that dtype and the compressor kernels consume bf16 kv_score input.
diff -- tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py
@@ -0,0 +1,2 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
diff -- tensorrt_llm/_torch/modules/mhc/__init__.py
@@ -0,0 +1,2 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py
@@ -0,0 +1,2 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` added +2710/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py` added +86/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py` added +2/-0
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` added +2/-0; `tensorrt_llm/_torch/modules/mhc/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/__init__.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_tf32.py`, `tests/unittest/_torch/modules/test_mhc.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15394 - [None][feat] DSv4: sparse cache manager adapter

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15394
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`; associated commits `c7362be2bc6e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +4717/-15, 2098 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` added +2694/-0 (2694 lines); `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` added +949/-0 (949 lines); hunks: -0,0 +1,949; symbols: test_cache_size_estimation_uses_model_attention_layer_count, FakeModelConfig, get_num_attention_layers, _view_fp8_as_uint8, touching `test_cache_size_estimation_uses_model_attention_layer_count, FakeModelConfig, get_num_attention_layers`; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` added +880/-0 (880 lines); hunks: -0,0 +1,880; symbols: _estimate_bytes_per_token, _get_attn_bytes_per_token, DeepseekV4CacheManager, __init__, touching `_estimate_bytes_per_token, _get_attn_bytes_per_token, DeepseekV4CacheManager`; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +111/-13 (124 lines); hunks: -1,21 +1,119; symbols: DeepseekV4AttentionType, is_overlap_compressor, is_sparse_layer, is_compress_layer, touching `DeepseekV4AttentionType, is_overlap_compressor, is_sparse_layer`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` added +2694/-0 (2694 lines)
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` added +949/-0 (949 lines); hunks: -0,0 +1,949; symbols: test_cache_size_estimation_uses_model_attention_layer_count, FakeModelConfig, get_num_attention_layers, _view_fp8_as_uint8
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` added +880/-0 (880 lines); hunks: -0,0 +1,880; symbols: _estimate_bytes_per_token, _get_attn_bytes_per_token, DeepseekV4CacheManager, __init__
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +111/-13 (124 lines); hunks: -1,21 +1,119; symbols: DeepseekV4AttentionType, is_overlap_compressor, is_sparse_layer, is_compress_layer
  - `tensorrt_llm/llmapi/llm_args.py` modified +48/-0 (48 lines); hunks: -792,6 +792,53 @@ def _value(name: str, default=None):; -2907,6 +2954,7 @@ def supports_backend(self, backend: str) -> bool:; symbols: _value, DeepSeekV4SparseAttentionConfig, validate_index_head_dim, normalize_compress_ratios
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -0,0 +1,949 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py
@@ -0,0 +1,880 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py
@@ -1,21 +1,119 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` added +2694/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` added +949/-0
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` added +880/-0; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +111/-13; `tensorrt_llm/llmapi/llm_args.py` modified +48/-0; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` modified +17/-0; `tensorrt_llm/llmapi/__init__.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/api_stability/references/llm.yaml`, `tests/unittest/llmapi/test_llm_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15409 - [None][feat] DSv4: sparse MLA attention backend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15409
- Status/date: merged / 2026-06-27
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py`; associated commits `aaffa2f9fef3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 25 files, +10416/-2277, 7979 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` added +1664/-0 (1664 lines); hunks: -0,0 +1,1664; symbols: Scenario, RopeConfig, rotate_half, _rotate_k_pe_for_ctx, touching `Scenario, RopeConfig, rotate_half`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py` added +555/-0 (555 lines); hunks: -0,0 +1,555; symbols: Scenario, _create_cache_manager, _local_to_physical_idx, _build_swa_indices, touching `Scenario, _create_cache_manager, _local_to_physical_idx`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj, touching `calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj`; `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +1029/-639 (1668 lines).
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` added +1664/-0 (1664 lines); hunks: -0,0 +1,1664; symbols: Scenario, RopeConfig, rotate_half, _rotate_k_pe_for_ctx
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py` added +555/-0 (555 lines); hunks: -0,0 +1,555; symbols: Scenario, _create_cache_manager, _local_to_physical_idx, _build_swa_indices
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` added +308/-0 (308 lines); hunks: -0,0 +1,308; symbols: calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj
  - `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +1029/-639 (1668 lines)
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +1330/-0 (1330 lines); hunks: -13,7 +13,42; -117,3 +152,1298 @@ def get_token_bytes(; symbols: get_token_bytes, DeepSeekV4Params, DeepSeekV4MetadataParams, _sparse_config_value
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py
@@ -0,0 +1,1664 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py
@@ -0,0 +1,555 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py
@@ -0,0 +1,308 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` added +1664/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py` added +555/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` added +308/-0
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +1029/-639; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +1330/-0; `tensorrt_llm/_torch/modules/attention.py` modified +657/-210; `tensorrt_llm/_torch/attention_backend/sparse/kernel.py` modified +274/-0; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/__init__.py` modified +17/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/__init__.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15414 - [None][feat] DSv4: model, tokenizer, and integration coverage

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15414
- Status/date: merged / 2026-06-28
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v4/README.md`, `tensorrt_llm/_torch/configs/deepseekv4.py`, `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tensorrt_llm/serve/tool_parser/deepseekv4_parser.py`, `tensorrt_llm/tokenizer/deepseek_v4/__init__.py` and 8 files; associated commits `6f7c57c6c297`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 46 files, +7618/-73, 5730 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` added +2587/-0 (2587 lines); `examples/models/core/deepseek_v4/README.md` added +488/-0 (488 lines); hunks: -0,0 +1,488; symbols: main, touching `main`; `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py` added +457/-0 (457 lines); hunks: -0,0 +1,457; symbols: _message_content_to_text, _to_json, _tools_from_openai_format, _tool_calls_from_openai_format, touching `_message_content_to_text, _to_json, _tools_from_openai_format`; `tests/unittest/llmapi/test_deepseek_v4_tokenizer.py` added +449/-0 (449 lines); hunks: -0,0 +1,449; symbols: _DummyTokenizer, encode, test_deepseek_v4_chat_template_matches_reference_single_user_prompt, test_deepseek_v4_chat_template_tokenize_uses_rendered_prompt, touching `_DummyTokenizer, encode, test_deepseek_v4_chat_template_matches_reference_single_user_prompt`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` added +2587/-0 (2587 lines)
  - `examples/models/core/deepseek_v4/README.md` added +488/-0 (488 lines); hunks: -0,0 +1,488; symbols: main
  - `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py` added +457/-0 (457 lines); hunks: -0,0 +1,457; symbols: _message_content_to_text, _to_json, _tools_from_openai_format, _tool_calls_from_openai_format
  - `tests/unittest/llmapi/test_deepseek_v4_tokenizer.py` added +449/-0 (449 lines); hunks: -0,0 +1,449; symbols: _DummyTokenizer, encode, test_deepseek_v4_chat_template_matches_reference_single_user_prompt, test_deepseek_v4_chat_template_tokenize_uses_rendered_prompt
  - `tensorrt_llm/_torch/configs/deepseekv4.py` added +178/-0 (178 lines); hunks: -0,0 +1,178; symbols: DeepseekV4Config, __init__
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v4/README.md
@@ -0,0 +1,488 @@
+# DeepSeek-V4
+This guide walks you through the examples to run DeepSeek-V4 models using NVIDIA TensorRT LLM with
+the PyTorch backend.
+DeepSeek-V4 uses the `DeepseekV4ForCausalLM` architecture in TensorRT LLM. Compared with
+DeepSeek-V3/R1/V3.2, it has a separate model implementation and sparse attention path. Use the
+commands in this guide as starting points and tune the parallelism and memory settings for your
diff -- tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py
@@ -0,0 +1,457 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/llmapi/test_deepseek_v4_tokenizer.py
@@ -0,0 +1,449 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` added +2587/-0; `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py` added +457/-0; `tensorrt_llm/_torch/configs/deepseekv4.py` added +178/-0; `tensorrt_llm/serve/tool_parser/deepseekv4_parser.py` added +25/-0; `tensorrt_llm/tokenizer/deepseek_v4/__init__.py` added +18/-0; `tensorrt_llm/_torch/configs/__init__.py` modified +3/-1
  - docs: `examples/models/core/deepseek_v4/README.md` added +488/-0
  - tests: `tests/unittest/llmapi/test_deepseek_v4_tokenizer.py` added +449/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/conftest.py`, `tests/integration/test_lists/test-db/l0_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15633 - [None][feat] DSv4 follow-up: runtime KV and cache foundations

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15633
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` and 7 files; associated commits `0b65e4fd52f1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 73 files, +6284/-1247, 10706 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-1 (8 lines); hunks: -2453,7 +2453,13 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, get_model_defaults, __init__, touching `forward, DeepseekV4ForCausalLM, get_model_defaults`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +1065/-46 (1111 lines); hunks: -22,15 +22,26; -45,6 +56,7 @@ class FakeModelConfig:; symbols: FakeModelConfig, get_num_attention_layers, test_quota_from_max_tokens_models_context_swa_scratch, test_needed_resource_uses_context_swa_scratch_slope, touching `FakeModelConfig, get_num_attention_layers, test_quota_from_max_tokens_models_context_swa_scratch`; `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` added +963/-0 (963 lines); hunks: -0,0 +1,963; symbols: ThreadSafeDistributed, __init__, tp_size, pp_size, touching `ThreadSafeDistributed, __init__, tp_size`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +87/-15 (102 lines); hunks: -36,7 +36,9; -88,7 +90,9 @@ def __init__(; symbols: __init__, _create_deepseek_v4_cache_manager, normalize_is_prefill, touching `__init__, _create_deepseek_v4_cache_manager, normalize_is_prefill`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-1 (8 lines); hunks: -2453,7 +2453,13 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, get_model_defaults, __init__
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +1065/-46 (1111 lines); hunks: -22,15 +22,26; -45,6 +56,7 @@ class FakeModelConfig:; symbols: FakeModelConfig, get_num_attention_layers, test_quota_from_max_tokens_models_context_swa_scratch, test_needed_resource_uses_context_swa_scratch_slope
  - `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` added +963/-0 (963 lines); hunks: -0,0 +1,963; symbols: ThreadSafeDistributed, __init__, tp_size, pp_size
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +87/-15 (102 lines); hunks: -36,7 +36,9; -88,7 +90,9 @@ def __init__(; symbols: __init__, _create_deepseek_v4_cache_manager, normalize_is_prefill
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +33/-8 (41 lines); hunks: -148,13 +148,19 @@ def test_deepseek_v4_fused_hc_default_enabled(monkeypatch):; -733,6 +739,7 @@ def test_deepseek_v4_sanity():; symbols: test_deepseek_v4_fused_hc_default_enabled, test_deepseek_v4_model_defaults_keep_tokens_per_block, test_deepseek_v4_model_defaults, LlmArgs
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -2453,7 +2453,13 @@ def forward(
-        return {"kv_cache_config": {"tokens_per_block": 128}}
+        return {
+            "kv_cache_config": {
+                "tokens_per_block": 128,
+                "use_kv_cache_manager_v2": True,
+                "enable_swa_scratch_reuse": True,
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -22,15 +22,26 @@
+    DEEPSEEK_V4_SLIDING_ATTENTION,
+    compress_ratio_has_attention,
-from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
+from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
+from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
+from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
diff -- tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py
@@ -0,0 +1,963 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-1
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +1065/-46; `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` added +963/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +87/-15; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +33/-8; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py` modified +13/-1; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_indices_transform.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/executor/executorConfigTest.cpp`, `examples/disaggregated/slurm/cache_transceiver_test/run_cache_transceiver_test.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15717 - [None][perf] DSv4 follow-up: sparse attention and model defaults

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15717
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h`, `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` and 8 files; associated commits `496acab85c47`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 44 files, +2699/-713, 4508 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: _disable_runtime_sm_validation, test_indexer_k_dtype_default_is_fp4, test_indexer_k_dtype_accepts_fp4, test_indexer_k_dtype_rejects_non_128_head_dim, touching `_disable_runtime_sm_validation, test_indexer_k_dtype_default_is_fp4, test_indexer_k_dtype_accepts_fp4`; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +116/-0 (116 lines); hunks: -20,6 +20,7; -133,6 +134,100 @@ void dispatchDeepseekV4QNorm(; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +54/-21 (75 lines); hunks: -272,7 +272,6 @@ class TestDeepseekV4CacheManager:; -509,15 +508,16 @@ def _create_random_cache(; symbols: TestDeepseekV4CacheManager, _create_random_cache, _split_blockwise_buffer, _indexer_cache_layout, touching `TestDeepseekV4CacheManager, _create_random_cache, _split_blockwise_buffer`; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +67/-0 (67 lines); hunks: -56,16 +56,83 @@ torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHe....
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: _disable_runtime_sm_validation, test_indexer_k_dtype_default_is_fp4, test_indexer_k_dtype_accepts_fp4, test_indexer_k_dtype_rejects_non_128_head_dim
  - `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +116/-0 (116 lines); hunks: -20,6 +20,7; -133,6 +134,100 @@ void dispatchDeepseekV4QNorm(
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +54/-21 (75 lines); hunks: -272,7 +272,6 @@ class TestDeepseekV4CacheManager:; -509,15 +508,16 @@ def _create_random_cache(; symbols: TestDeepseekV4CacheManager, _create_random_cache, _split_blockwise_buffer, _indexer_cache_layout
  - `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +67/-0 (67 lines); hunks: -56,16 +56,83 @@ torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHe...
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +44/-0 (44 lines); hunks: -636,6 +636,50 @@ def test_deepseek_v4_sparse_ratios_prefer_checkpoint_defaul...; symbols: test_deepseek_v4_sparse_ratios_prefer_checkpoint_defaults, test_deepseek_v4_model_config_defaults_to_fp4_indexer, test_deepseek_v4_model_config_defaults_to_fp8_before_blackwell, test_deepseek_v4_sparse_ratios_keep_checkpoint_length_without_mtp
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py
@@ -0,0 +1,129 @@
+# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu
@@ -20,6 +20,7 @@
+#include <cuda_fp8.h>
@@ -133,6 +134,100 @@ void dispatchDeepseekV4QNorm(
+// Fused q-norm + FP8 quant of nope segment. Row layout [nope|rope]; writes FP8
+// nope (scaled by inv_rms * quant_scale_qkv) to `quant_q_nope` with per-row
+// stride `quantQNopeRowStrideBytes`, and bf16/fp16 rope to `q_pe_out`.
+// Requires kHeadDim==512, kNopeDim==448, kRopeDim==64 so each lane's
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -272,7 +272,6 @@ class TestDeepseekV4CacheManager:
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_fp4_indexer.py` added +129/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +54/-21; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +44/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +35/-5; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +5/-0
  - runtime: `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +116/-0; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +67/-0; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` modified +17/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16028 - [None][perf] Port remaining DeepSeek V4 optimizations to main

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16028
- Status/date: merged / 2026-07-09
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu`, `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h`, `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp`; associated commits `4899f00f32e7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +1301/-115, 1590 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu` added +372/-0 (372 lines); hunks: -0,0 +1,372; `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp` added +201/-0 (201 lines); hunks: -0,0 +1,201; `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h` added +43/-0 (43 lines); hunks: -0,0 +1,43.
- Code diff details:
  - `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu` added +372/-0 (372 lines); hunks: -0,0 +1,372
  - `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp` added +201/-0 (201 lines); hunks: -0,0 +1,201
  - `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h` added +43/-0 (43 lines); hunks: -0,0 +1,43
- Key code excerpts:

```diff
diff -- cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu
@@ -0,0 +1,372 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp
@@ -0,0 +1,201 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h
@@ -0,0 +1,43 @@
```

- Extracted files (not manually reviewed):
  - runtime: `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu` added +372/-0; `cpp/tensorrt_llm/thop/deepseekV4BlockTableOp.cpp` added +201/-0; `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.h` added +43/-0
- Risk and verification: The diff ships test coverage in `tests/microbenchmarks/dsv4_block_table_perf.py`, `tests/unittest/_torch/custom_ops/test_deepseek_v4_block_table.py`, `tests/unittest/api_stability/references/trtllm_serve_api.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15710 - [None][test] DSv4 PR-6 coverage and import safety

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15710
- Status/date: merged / 2026-07-09
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/examples/test_deepseek_v4_pro.py`; associated commits `7d600ecc1d23`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +520/-9, 612 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/examples/test_deepseek_v4_pro.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _deepseekv4_pro_agg_llm_kwargs, _get_llm_vocab_size, _make_token_id_prompt, _run_token_id_smoke_wave, touching `_deepseekv4_pro_agg_llm_kwargs, _get_llm_vocab_size, _make_token_id_prompt`; `tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +6/-3 (9 lines); hunks: -34,9 +34,12; `tensorrt_llm/_torch/cuda_tile_utils.py` modified +7/-1 (8 lines); hunks: -50,7 +50,13 @@ def ceil_div(a, b):; symbols: ceil_div, touching `ceil_div`; `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py` modified +2/-4 (6 lines); hunks: -1449,8 +1449,7 @@ def _on_stop_committing(self) -> None:; -1485,8 +1484,7 @@ def _unlock_stale_blocks(; symbols: _on_stop_committing, _unlock_stale_blocks, touching `_on_stop_committing, _unlock_stale_blocks`.
- Code diff details:
  - `tests/integration/defs/examples/test_deepseek_v4_pro.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: _deepseekv4_pro_agg_llm_kwargs, _get_llm_vocab_size, _make_token_id_prompt, _run_token_id_smoke_wave
  - `tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +6/-3 (9 lines); hunks: -34,9 +34,12
  - `tensorrt_llm/_torch/cuda_tile_utils.py` modified +7/-1 (8 lines); hunks: -50,7 +50,13 @@ def ceil_div(a, b):; symbols: ceil_div
  - `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py` modified +2/-4 (6 lines); hunks: -1449,8 +1449,7 @@ def _on_stop_committing(self) -> None:; -1485,8 +1484,7 @@ def _unlock_stale_blocks(; symbols: _on_stop_committing, _unlock_stale_blocks
  - `tensorrt_llm/_torch/attention_backend/triton_prefill.py` modified +3/-1 (4 lines); hunks: -33,7 +33,9; symbols: _get_block_sizes
- Key code excerpts:

```diff
diff -- tests/integration/defs/examples/test_deepseek_v4_pro.py
@@ -0,0 +1,151 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/attention_backend/flashinfer.py
@@ -34,9 +34,12 @@
-    capability = torch.cuda.get_device_capability()
-    arch_list = f"{capability[0]}.{capability[1]}"
-    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
+    # Guard on a visible GPU: with CUDA_VISIBLE_DEVICES="" (pure client) the
+    # capability query would force a CUDA context at import time.
+    if torch.cuda.is_available():
diff -- tensorrt_llm/_torch/cuda_tile_utils.py
@@ -50,7 +50,13 @@ def ceil_div(a, b):
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/examples/test_deepseek_v4_pro.py` added +151/-0
  - runtime: `tensorrt_llm/_torch/attention_backend/flashinfer.py` modified +6/-3; `tensorrt_llm/_torch/cuda_tile_utils.py` modified +7/-1; `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache.py` modified +2/-4; `tensorrt_llm/_torch/attention_backend/triton_prefill.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/accuracy_core.py`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15919 - [None][feat] Add DeepSeek-V4-Pro curated configs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15919
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `examples/configs/curated/deepseek-v4-pro-latency.yaml`, `examples/configs/curated/deepseek-v4-pro-throughput.yaml`; associated commits `44f05214ce16`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +90/-0, 96 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/configs/curated/deepseek-v4-pro-latency.yaml` added +45/-0 (45 lines); hunks: -0,0 +1,45; `examples/configs/curated/deepseek-v4-pro-throughput.yaml` added +35/-0 (35 lines); hunks: -0,0 +1,35.
- Code diff details:
  - `examples/configs/curated/deepseek-v4-pro-latency.yaml` added +45/-0 (45 lines); hunks: -0,0 +1,45
  - `examples/configs/curated/deepseek-v4-pro-throughput.yaml` added +35/-0 (35 lines); hunks: -0,0 +1,35
- Key code excerpts:

```diff
diff -- examples/configs/curated/deepseek-v4-pro-latency.yaml
@@ -0,0 +1,45 @@
+cuda_graph_config:
+  batch_sizes:
+  - 1
+  - 2
+  - 4
+  - 8
diff -- examples/configs/curated/deepseek-v4-pro-throughput.yaml
@@ -0,0 +1,35 @@
+attention_dp_config:
+  enable_balance: true
+cuda_graph_config:
+  batch_sizes:
+  - 1
+  - 2
```

- Extracted files (not manually reviewed):
  - docs: `examples/configs/curated/deepseek-v4-pro-latency.yaml` added +45/-0; `examples/configs/curated/deepseek-v4-pro-throughput.yaml` added +35/-0
- Risk and verification: This is mostly docs/examples in `examples/configs/curated/deepseek-v4-pro-latency.yaml`, `examples/configs/curated/deepseek-v4-pro-throughput.yaml`, `examples/configs/curated/lookup.yaml`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #16539 - [None][doc] Add DeepSeek-V4 optimization tech blog

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16539
- Status/date: merged / 2026-07-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md`; associated commits `d541ae56800e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +822/-0, 838 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md` added +488/-0 (488 lines); hunks: -0,0 +1,488.
- Code diff details:
  - `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md` added +488/-0 (488 lines); hunks: -0,0 +1,488
- Key code excerpts:

```diff
diff -- docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md
@@ -0,0 +1,488 @@
+<!--
+Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
+Licensed under the Apache License, Version 2.0 (the "License");
+you may not use this file except in compliance with the License.
+You may obtain a copy of the License at
+    http://www.apache.org/licenses/LICENSE-2.0
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md` added +488/-0
- Risk and verification: This is mostly docs/examples in `docs/source/blogs/media/tech_blog26_agentperf_closed_loop_workflow.svg`, `docs/source/blogs/media/tech_blog26_two_level_routing.svg`, `docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #16466 - [None][fix] Fix DeepSeek V4 KV cache warmup handling and serveral other issues

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16466
- Status/date: merged / 2026-07-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py`; associated commits `d94540b429bb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +103/-16, 225 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` modified +18/-14 (32 lines); hunks: -53,7 +53,6; -407,32 +406,31 @@ def _expected_valid_blocks(; symbols: _expected_valid_blocks, _split_blockwise_buffer, _read_cache_data, touching `_expected_valid_blocks, _split_blockwise_buffer, _read_cache_data`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +29/-0 (29 lines); hunks: -84,6 +84,35 @@ def get_num_attention_layers(self) -> int:; symbols: get_num_attention_layers, test_mtp_extra_tokens_are_in_context_capacity, test_quota_from_max_tokens_models_context_swa_scratch, touching `get_num_attention_layers, test_mtp_extra_tokens_are_in_context_capacity, test_quota_from_max_tokens_models_context_swa_scratch`; `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` modified +29/-1 (30 lines); hunks: -39,6 +39,27; -407,9 +428,12 @@ def capture(self,; symbols: _save_spec_decode_capture_state, _restore_spec_decode_capture_state, CUDAGraphRunnerConfig, capture, touching `_save_spec_decode_capture_state, _restore_spec_decode_capture_state, CUDAGraphRunnerConfig`; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` modified +3/-1 (4 lines); hunks: -888,7 +888,9 @@ def _add_layer(; symbols: _add_layer, touching `_add_layer`.
- Code diff details:
  - `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` modified +18/-14 (32 lines); hunks: -53,7 +53,6; -407,32 +406,31 @@ def _expected_valid_blocks(; symbols: _expected_valid_blocks, _split_blockwise_buffer, _read_cache_data
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +29/-0 (29 lines); hunks: -84,6 +84,35 @@ def get_num_attention_layers(self) -> int:; symbols: get_num_attention_layers, test_mtp_extra_tokens_are_in_context_capacity, test_quota_from_max_tokens_models_context_swa_scratch
  - `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` modified +29/-1 (30 lines); hunks: -39,6 +39,27; -407,9 +428,12 @@ def capture(self,; symbols: _save_spec_decode_capture_state, _restore_spec_decode_capture_state, CUDAGraphRunnerConfig, capture
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` modified +3/-1 (4 lines); hunks: -888,7 +888,9 @@ def _add_layer(; symbols: _add_layer
- Key code excerpts:

```diff
diff -- tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py
@@ -53,7 +53,6 @@
-INDEXER_QUANT_BLOCK_SIZE = 128
@@ -407,32 +406,31 @@ def _expected_valid_blocks(
-    index_head_dim: int = INDEX_HEAD_DIM,
-    quant_block_size: int = INDEXER_QUANT_BLOCK_SIZE,
+    data_size: int,
+    scale_size: int,
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -84,6 +84,35 @@ def get_num_attention_layers(self) -> int:
+def test_mtp_extra_tokens_are_in_context_capacity():
+    cache_manager = object.__new__(DeepseekV4CacheManager)
+    cache_manager.pp_layers = [0]
+    cache_manager._compress_ratios = [1]
+    cache_manager._get_attn_bytes_per_block = lambda _attn_type, _layer_idx: 1
+    cache_manager._get_window_size = lambda _compress_ratio, _attn_type: 128
diff -- tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py
@@ -39,6 +39,27 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` modified +18/-14; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +29/-0
  - runtime: `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` modified +29/-1; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/cache_manager.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/executor/test_pytorch_model_engine.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16540 - [None][test] Add DeepSeek-V4-Pro perf sanity cases on GB300

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16540
- Status/date: merged / 2026-07-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml`, `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml`, `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots416.yaml`, `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots384.yaml`, `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots416.yaml` and 13 files; associated commits `8bd00e4e3e56`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 23 files, +1316/-29, 909 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml` added +172/-0 (172 lines); hunks: -0,0 +1,172; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` added +107/-0 (107 lines); hunks: -0,0 +1,107; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml` added +172/-0 (172 lines); hunks: -0,0 +1,172
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` added +107/-0 (107 lines); hunks: -0,0 +1,107
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0 (105 lines); hunks: -0,0 +1,105
  - `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` added +65/-0 (65 lines)
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml
@@ -0,0 +1,172 @@
+metadata:
+  model_name: deepseek_v4_pro_fp4
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,107 @@
+metadata:
+  model_name: deepseek_v4_pro_fp4
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,105 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL.yaml` added +172/-0; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` added +107/-0; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL.yaml` added +105/-0; `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` added +65/-0; `tests/scripts/perf-sanity/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml` added +65/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/perf/_model_paths.py`, `tests/integration/defs/perf/test_perf_sanity.py`, `tests/integration/test_lists/test-db/l0_gb300_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_perf_sanity_ctx12_node1_gpu4_gen1_node2_gpu8.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16611 - [None][test] Add deepseek v4 pro cases on the qa side

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16611
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots416.yaml`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots384.yaml`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep32_slots416.yaml` and 12 files; associated commits `4fb31cb937ab`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +910/-0, 342 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0 (106 lines); hunks: -0,0 +1,106; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0 (106 lines); hunks: -0,0 +1,106; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` added +104/-0 (104 lines); hunks: -0,0 +1,104; `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` added +65/-0 (65 lines).
- Code diff details:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0 (106 lines); hunks: -0,0 +1,106
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0 (106 lines); hunks: -0,0 +1,106
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` added +104/-0 (104 lines); hunks: -0,0 +1,104
  - `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` added +65/-0 (65 lines)
  - `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml` added +65/-0 (65 lines)
- Key code excerpts:

```diff
diff -- tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,106 @@
+metadata:
+  model_name: deepseek_v4_pro_fp4
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,106 @@
+metadata:
+  model_name: deepseek_v4_pro_fp4
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml
@@ -0,0 +1,104 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con1229_ctx7_dep4_gen1_dep8_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx8_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` added +106/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con2_ctx1_dep4_gen5_tep4_eplb0_mtp3_ccb-NIXL.yaml` added +104/-0; `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml` added +65/-0; `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml` added +65/-0; `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots416.yaml` added +65/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_disagg.yml`, `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_ctx_ep4_384.yaml`, `tests/scripts/perf/disaggregated/deepseek-v4-pro-eplb/moe_load_balancer_gen_ep16_slots384.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16433 - [None][fix] Load DeepSeek V4 mixed-precision NVFP4 checkpoints

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16433
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; associated commits `f52f3feadd6e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +87/-0, 123 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +41/-0 (41 lines); hunks: -1,3 +1,6; -296,6 +299,43 @@ def _resolve_enable_fused_hc(config: PretrainedConfig) -> b...; symbols: _resolve_enable_fused_hc, _normalize_deepseek_v4_nvfp4_mixed_precision_config, _copy_deepseek_v4_fused_a_weight_scale, get_model_defaults, touching `_resolve_enable_fused_hc, _normalize_deepseek_v4_nvfp4_mixed_precision_config, _copy_deepseek_v4_fused_a_weight_scale`; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +46/-0 (46 lines); hunks: -1,3 +1,6; -33,6 +36,7; symbols: test_deepseek_v4_moe_auto_backend_on_blackwell, test_deepseek_v4_nvfp4_mixed_precision_config, test_deepseek_v4_routed_moe_quant_config_from_mxfp4_header, touching `test_deepseek_v4_moe_auto_backend_on_blackwell, test_deepseek_v4_nvfp4_mixed_precision_config, test_deepseek_v4_routed_moe_quant_config_from_mxfp4_header`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +41/-0 (41 lines); hunks: -1,3 +1,6; -296,6 +299,43 @@ def _resolve_enable_fused_hc(config: PretrainedConfig) -> b...; symbols: _resolve_enable_fused_hc, _normalize_deepseek_v4_nvfp4_mixed_precision_config, _copy_deepseek_v4_fused_a_weight_scale, get_model_defaults
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +46/-0 (46 lines); hunks: -1,3 +1,6; -33,6 +36,7; symbols: test_deepseek_v4_moe_auto_backend_on_blackwell, test_deepseek_v4_nvfp4_mixed_precision_config, test_deepseek_v4_routed_moe_quant_config_from_mxfp4_header
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -1,3 +1,6 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
@@ -296,6 +299,43 @@ def _resolve_enable_fused_hc(config: PretrainedConfig) -> bool:
+def _normalize_deepseek_v4_nvfp4_mixed_precision_config(
+    model_config: ModelConfig[PretrainedConfig],
+) -> ModelConfig[PretrainedConfig]:
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv4.py
@@ -1,3 +1,6 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
@@ -33,6 +36,7 @@
+    _normalize_deepseek_v4_nvfp4_mixed_precision_config,
@@ -442,6 +446,48 @@ def test_deepseek_v4_moe_auto_backend_on_blackwell(monkeypatch):
+def test_deepseek_v4_nvfp4_mixed_precision_config():
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +41/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +46/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16734 - [None][perf] Avoid implicit device-scalar syncs in DeepSeek-V4 ctx sparse metadata

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16734
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`; associated commits `29919947fcc5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +82/-4, 166 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +44/-0 (44 lines); hunks: -449,6 +449,50 @@ def test_mixed_context_generation_position_ids_follow_compa...; symbols: test_mixed_context_generation_position_ids_follow_compact_output, test_ctx_position_ids_host_sizes_match_device_scalar_fallback, _run, precompute_freqs_cis, touching `test_mixed_context_generation_position_ids_follow_compact_output, test_ctx_position_ids_host_sizes_match_device_scalar_fallback, _run`; `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +38/-4 (42 lines); hunks: -274,6 +274,7 @@ def __post_init__(self):; -736,12 +737,18 @@ def prepare(self):; symbols: __post_init__, prepare, prepare_compressed_kv_metadata, touching `__post_init__, prepare, prepare_compressed_kv_metadata`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +44/-0 (44 lines); hunks: -449,6 +449,50 @@ def test_mixed_context_generation_position_ids_follow_compa...; symbols: test_mixed_context_generation_position_ids_follow_compact_output, test_ctx_position_ids_host_sizes_match_device_scalar_fallback, _run, precompute_freqs_cis
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +38/-4 (42 lines); hunks: -274,6 +274,7 @@ def __post_init__(self):; -736,12 +737,18 @@ def prepare(self):; symbols: __post_init__, prepare, prepare_compressed_kv_metadata
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py
@@ -449,6 +449,50 @@ def test_mixed_context_generation_position_ids_follow_compact_output():
+@pytest.mark.parametrize(
+    "compress_ratio,cached_tokens,kv_lens",
+    [
+        pytest.param(1, [0], [4096], id="cr1_single"),
+        pytest.param(4, [0, 75, 4000], [4096, 4171, 8096], id="cr4_multi_boundary"),
+        pytest.param(128, [0], [64], id="cr128_empty_output"),
diff -- tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py
@@ -274,6 +274,7 @@ def __post_init__(self):
+        self._ctx_output_sizes: Optional[Dict[int, int]] = None
@@ -736,12 +737,18 @@ def prepare(self):
+        # Host-side per-ratio ctx compressed-token counts (Python ints), so
+        # _compute_ctx_compressed_position_ids never reads a device scalar
+        # (implicit D2H + stream sync) for its arange size / slice bound.
+        ctx_output_sizes: Optional[Dict[int, int]] = None
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +44/-0
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py` modified +38/-4
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16881 - [https://nvbugs/6479863][fix] Use scalar SwiGLU limit for DeepSeek V4 FP8 MoE

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16881
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv4.py`; associated commits `7a64f2660415`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-1, 19 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +11/-1 (12 lines); hunks: -1517,7 +1517,17 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +11/-1 (12 lines); hunks: -1517,7 +1517,17 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -1517,7 +1517,17 @@ def __init__(
-            if supports_swiglu_limit and not kernel_requires_bias_for_swiglu_limit:
+            # DeepSeek-V4 supplies a uniform scalar limit. The TRTLLM-Gen FP8
+            # path consumes it directly and rejects the redundant tensor.
+            requires_scalar_only_swiglu_limit = (
+                moe_cls is TRTLLMGenFusedMoE
+                and experts_quant_config.quant_mode.has_fp8_block_scales()
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_deepseekv4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17273 - [TRTLLM-14597][perf] Fuse the DSv4 MLA prologue: kv_a_layernorm, q_nope FP8 quant and Q RoPE

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17273
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu`, `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h`, `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; associated commits `0eeda343df8c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +2461/-222, 3464 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +468/-58 (526 lines); hunks: -1,28 +1,40; -39,6 +51,7; symbols: _source_calls, _write_safetensors_header, test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms, test_deepseek_v4_mla_q_b_layernorm_init_and_forward_shape, touching `_source_calls, _write_safetensors_header, test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms`; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +295/-38 (333 lines); hunks: -17,11 +17,13; -31,6 +33,23 @@ namespace; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +42/-3 (45 lines); hunks: -63,7 +63,9 @@ torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHead...; -112,11 +114,46 @@ void deepseekV4QNormFusedFp8(torch::Tensor q, torch::Tenso...; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` modified +9/-1 (10 lines); hunks: -18,6 +18,7; -41,9 +42,16 @@ void invokeDeepseekV4QNorm(.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +468/-58 (526 lines); hunks: -1,28 +1,40; -39,6 +51,7; symbols: _source_calls, _write_safetensors_header, test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms, test_deepseek_v4_mla_q_b_layernorm_init_and_forward_shape
  - `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +295/-38 (333 lines); hunks: -17,11 +17,13; -31,6 +33,23 @@ namespace
  - `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +42/-3 (45 lines); hunks: -63,7 +63,9 @@ torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHead...; -112,11 +114,46 @@ void deepseekV4QNormFusedFp8(torch::Tensor q, torch::Tenso...
  - `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` modified +9/-1 (10 lines); hunks: -18,6 +18,7; -41,9 +42,16 @@ void invokeDeepseekV4QNorm(
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv4.py
@@ -1,28 +1,40 @@
-import ast
-import textwrap
+from types import SimpleNamespace
+from unittest.mock import patch
+from torch import nn
+from utils.util import skip_pre_blackwell
diff -- cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu
@@ -17,11 +17,13 @@
+#include "tensorrt_llm/common/envUtils.h"
+#include <type_traits>
@@ -31,6 +33,23 @@ namespace
+// FP8 is one byte per element, so an N-element vector is an N-byte store.
+template <int BYTES>
+struct Fp8VecStore;
diff -- cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp
@@ -63,7 +63,9 @@ torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHeads, int64_t headDim
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +468/-58
  - runtime: `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.cu` modified +295/-38; `cpp/tensorrt_llm/thop/deepseekV4QNormOp.cpp` modified +42/-3; `cpp/tensorrt_llm/kernels/deepseekV4QNormKernel.h` modified +9/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/test_attention_mla.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16224 - [None][feat] Enable DeepSeek-V4 and DSA (DeepSeek-V3.2/GLM) serving on SM120 via FlashInfer sparse-MLA

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16224
- Status/date: merged / 2026-08-24
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; associated commits `2f22de218d5b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 38 files, +1667/-62, 2503 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +6/-5 (11 lines); hunks: -2544,12 +2544,13 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, get_model_defaults, get_preferred_kv_cache_manager_version, touching `forward, DeepseekV4ForCausalLM, get_model_defaults`; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +75/-18 (93 lines); hunks: -13,10 +13,11; -167,6 +168,20 @@ def test_deepseek_v4_kv_cache_defaults_and_v2_preference():; symbols: test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks, LlmArgs, test_deepseek_v4_weight_remap_for_mxfp4_routed_experts, touching `test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks, LlmArgs`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7 @@ def test_quota_from_max_tokens_models_context_swa_scratch():; -131,6 +132,7 @@ def test_needed_resource_uses_context_swa_scratch_slope():; symbols: test_quota_from_max_tokens_models_context_swa_scratch, test_needed_resource_uses_context_swa_scratch_slope, touching `test_quota_from_max_tokens_models_context_swa_scratch, test_needed_resource_uses_context_swa_scratch_slope`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +6/-5 (11 lines); hunks: -2544,12 +2544,13 @@ def forward(; symbols: forward, DeepseekV4ForCausalLM, get_model_defaults, get_preferred_kv_cache_manager_version
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +75/-18 (93 lines); hunks: -13,10 +13,11; -167,6 +168,20 @@ def test_deepseek_v4_kv_cache_defaults_and_v2_preference():; symbols: test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks, LlmArgs, test_deepseek_v4_weight_remap_for_mxfp4_routed_experts
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +2/-0 (2 lines); hunks: -97,6 +97,7 @@ def test_quota_from_max_tokens_models_context_swa_scratch():; -131,6 +132,7 @@ def test_needed_resource_uses_context_swa_scratch_slope():; symbols: test_quota_from_max_tokens_models_context_swa_scratch, test_needed_resource_uses_context_swa_scratch_slope
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -2544,12 +2544,13 @@ def forward(
-        return {
-            "kv_cache_config": {
-                "tokens_per_block": 128,
-                "enable_swa_scratch_reuse": True,
-            }
+        kv_cache_defaults = {
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv4.py
@@ -13,10 +13,11 @@
-from utils.util import skip_pre_blackwell
+from utils.util import getSMVersion, skip_blackwell_geforce, skip_pre_blackwell
+from tensorrt_llm._torch.attention_backend.fmha import FallbackFmha, FlashInferSparseMlaFmha
@@ -167,6 +168,20 @@ def test_deepseek_v4_kv_cache_defaults_and_v2_preference():
+def test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks() -> None:
+    class LlmArgs:
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -97,6 +97,7 @@ def test_quota_from_max_tokens_models_context_swa_scratch():
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +6/-5
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +75/-18; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_rtx_pro_6000.yml`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/attention/sparse/dsa/test_dsa_indexer.py`, `tests/unittest/_torch/attention/sparse/test_flashinfer_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18185 - [https://nvbugs/6644226][fix] Reduce DeepSeek V4 EPLB loading memory

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18185
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; associated commits `36138a2c8e03`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +87/-10, 210 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +42/-1 (43 lines); hunks: -45,6 +45,7; -61,6 +62,7; symbols: _load_weights_impl, split_kv_b_proj, pageout_previous_moe_layer, load_flat_hc_weights, touching `_load_weights_impl, split_kv_b_proj, pageout_previous_moe_layer`; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +45/-0 (45 lines); hunks: -45,6 +45,7; -206,6 +207,50 @@ def test_deepseek_v4_weight_remap_for_fp8_routed_experts():; symbols: test_deepseek_v4_weight_remap_for_fp8_routed_experts, test_deepseek_v4_eplb_weight_loader_pages_out_each_moe_layer, test_deepseek_v4_fused_a_weight_scale_rebuilds_fp8_shape, touching `test_deepseek_v4_weight_remap_for_fp8_routed_experts, test_deepseek_v4_eplb_weight_loader_pages_out_each_moe_layer, test_deepseek_v4_fused_a_weight_scale_rebuilds_fp8_shape`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +42/-1 (43 lines); hunks: -45,6 +45,7; -61,6 +62,7; symbols: _load_weights_impl, split_kv_b_proj, pageout_previous_moe_layer, load_flat_hc_weights
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +45/-0 (45 lines); hunks: -45,6 +45,7; -206,6 +207,50 @@ def test_deepseek_v4_weight_remap_for_fp8_routed_experts():; symbols: test_deepseek_v4_weight_remap_for_fp8_routed_experts, test_deepseek_v4_eplb_weight_loader_pages_out_each_moe_layer, test_deepseek_v4_fused_a_weight_scale_rebuilds_fp8_shape
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -45,6 +45,7 @@
+from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
@@ -61,6 +62,7 @@
+from ..mmap_utils import pageout_file_backed_regions
@@ -574,11 +576,12 @@ def _load_weights_impl(self, weights: Dict, skip_modules: List[str] = []):
-            weights = _remap_deepseek_v4_checkpoint_keys(
+            remapped_weights = _remap_deepseek_v4_checkpoint_keys(
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv4.py
@@ -45,6 +45,7 @@
+    DeepseekV4WeightLoader,
@@ -206,6 +207,50 @@ def test_deepseek_v4_weight_remap_for_fp8_routed_experts():
+def test_deepseek_v4_eplb_weight_loader_pages_out_each_moe_layer(monkeypatch):
+    model = torch.nn.Module()
+    model.model = torch.nn.Module()
+    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +42/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +45/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16940 - [TRTLLM-14116][feat] Add DeepSeek-V4 Hopper support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16940
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v4/README.md`, `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`, `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py`; associated commits `e68f58149ba0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1306/-248, 1960 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v4/README.md` modified +15/-15 (30 lines); hunks: -41,8 +41,7 @@ for how to build TensorRT LLM from source and start a TRT-LLM...; -56,15 +55,16 @@ below follows the model list published on the; `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +8/-1 (9 lines); hunks: -459,6 +459,11 @@ def _emit_or_collect(model_key: str, tensor: torch.Tensor):; -2548,7 +2553,9 @@ def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> d...; symbols: _emit_or_collect, get_model_defaults, touching `_emit_or_collect, get_model_defaults`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +25/-18 (43 lines); hunks: -18,6 +18,7; -924,24 +925,30 @@ def _create_deepseek_v4_cache_manager(self, compress_ratio...; symbols: _create_deepseek_v4_cache_manager, touching `_create_deepseek_v4_cache_manager`; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +27/-3 (30 lines); hunks: -156,7 +156,10 @@ def test_deepseek_v4_fused_hc_default_enabled(monkeypatch):; -168,7 +171,25 @@ def test_deepseek_v4_kv_cache_defaults_and_v2_preference():; symbols: test_deepseek_v4_fused_hc_default_enabled, test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks, touching `test_deepseek_v4_fused_hc_default_enabled, test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks`.
- Code diff details:
  - `examples/models/core/deepseek_v4/README.md` modified +15/-15 (30 lines); hunks: -41,8 +41,7 @@ for how to build TensorRT LLM from source and start a TRT-LLM...; -56,15 +55,16 @@ below follows the model list published on the
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +8/-1 (9 lines); hunks: -459,6 +459,11 @@ def _emit_or_collect(model_key: str, tensor: torch.Tensor):; -2548,7 +2553,9 @@ def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> d...; symbols: _emit_or_collect, get_model_defaults
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +25/-18 (43 lines); hunks: -18,6 +18,7; -924,24 +925,30 @@ def _create_deepseek_v4_cache_manager(self, compress_ratio...; symbols: _create_deepseek_v4_cache_manager
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +27/-3 (30 lines); hunks: -156,7 +156,10 @@ def test_deepseek_v4_fused_hc_default_enabled(monkeypatch):; -168,7 +171,25 @@ def test_deepseek_v4_kv_cache_defaults_and_v2_preference():; symbols: test_deepseek_v4_fused_hc_default_enabled, test_deepseek_v4_kv_cache_defaults_and_v2_preference, test_deepseek_v4_fp8_ds_mla_uses_256_token_blocks
  - `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` modified +3/-2 (5 lines); hunks: -43,15 +43,15; -78,6 +78,7 @@ def _create_deepseek_v4_manager(; symbols: _create_deepseek_v4_manager
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v4/README.md
@@ -41,8 +41,7 @@ for how to build TensorRT LLM from source and start a TRT-LLM Docker container.
-DeepSeek-V4 is only supported on Blackwell GPUs (`SM100+`) in the current PyTorch backend
-implementation. Pre-Blackwell GPUs are not supported for this model path.
+DeepSeek-V4 is supported on Hopper (`SM90`) and Blackwell (`SM100+`) GPUs in the PyTorch backend.
@@ -56,15 +55,16 @@ below follows the model list published on the
-maximum sequence length, and runtime batch size. For initial bring-up, an 8xB200 node is enough for
-Flash checkpoints and the FP4 + FP8 mixed DeepSeek-V4-Pro checkpoint. DeepSeek-V4-Pro-Base is larger
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -459,6 +459,11 @@ def _emit_or_collect(model_key: str, tensor: torch.Tensor):
+        # Hopper's W4A16 MXFP4 loader consumes the legacy suffix, while the
+        # Blackwell loaders consume the canonical suffix. Both keys reference
+        # the same packed E8M0 scale tensor.
+        if ".mlp.experts." in model_key and model_key.endswith(".weight_scale"):
+            out[f"{model_key}_inv"] = tensor
@@ -2548,7 +2553,9 @@ def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py
@@ -18,6 +18,7 @@
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v4/README.md` modified +15/-15
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +8/-1
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +25/-18; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +27/-3; `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py`, `tests/unittest/_torch/attention/sparse/test_sparse_mla_forward.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`, `tests/unittest/disaggregated/test_deepseek_v4_kv_transfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18306 - [https://nvbugs/6670227][fix] Drop redundant min_tokens from DSv4-Pro token-boundary smoke

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18306
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/examples/test_deepseek_v4_pro.py`; associated commits `40b9cbc2f4c7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +0/-2, 16 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/examples/test_deepseek_v4_pro.py` modified +0/-1 (1 lines); hunks: -121,7 +121,6 @@ def _run_token_id_smoke_wave(llm, prompt_lengths, max_tokens...; symbols: _run_token_id_smoke_wave, touching `_run_token_id_smoke_wave`.
- Code diff details:
  - `tests/integration/defs/examples/test_deepseek_v4_pro.py` modified +0/-1 (1 lines); hunks: -121,7 +121,6 @@ def _run_token_id_smoke_wave(llm, prompt_lengths, max_tokens...; symbols: _run_token_id_smoke_wave
- Key code excerpts:

```diff
diff -- tests/integration/defs/examples/test_deepseek_v4_pro.py
@@ -121,7 +121,6 @@ def _run_token_id_smoke_wave(llm, prompt_lengths, max_tokens, vocab_size):
-        min_tokens=max_tokens,
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/examples/test_deepseek_v4_pro.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/examples/test_deepseek_v4_pro.py`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17137 - [https://nvbugs/6480621][test] Revert to 60-second KV transfer timeout for GB300 DeepSeek V4 Pro disaggregated perf-sanity

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17137
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml`; associated commits `3c4dc51f9335`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +5/-2, 28 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` modified +5/-2 (7 lines); hunks: -41,6 +41,9 @@ environment:; -67,7 +70,7 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` modified +5/-2 (7 lines); hunks: -41,6 +41,9 @@ environment:; -67,7 +70,7 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml
@@ -41,6 +41,9 @@ environment:
+# Keep this stage as explicit coverage while the precheck is globally opt-in.
+cache_transceiver_precheck:
+  enabled: true
@@ -67,7 +70,7 @@ worker_config:
-      kv_transfer_timeout_ms: 600000
+      kv_transfer_timeout_ms: 60000
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml` modified +5/-2
- Risk and verification: The diff ships test coverage in `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18298 - [None][test] Add AgentX DeepSeek-V4-Pro-DSpark perf-sanity lanes on GB300

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18298
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`; associated commits `be3fecb42433`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +1568/-16, 1813 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0 (133 lines); hunks: -0,0 +1,133
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml
@@ -0,0 +1,133 @@
+metadata:
+  model_name: deepseek_v4_pro_dspark
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro-DSpark
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml
@@ -0,0 +1,133 @@
+metadata:
+  model_name: deepseek_v4_pro_dspark
+  precision: fp4
+  model_dir_name: DeepSeek-V4-Pro-DSpark
+  supported_gpus:
+  - GB300
diff -- tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml
@@ -0,0 +1,133 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` added +133/-0; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` added +133/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/.test_durations`, `tests/integration/defs/perf/README_test_perf_sanity.md`, `tests/integration/defs/perf/agentx_client.py`, `tests/integration/defs/perf/test_perf_sanity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18633 - [None][test] Add the 40-GPU AgentX DeepSeek-V4-Pro-DSpark case to post-merge and switch both cases to the NVFP4 checkpoint

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18633
- Status/date: merged / 2026-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml`; associated commits `9964d34d67ad`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +48/-12, 141 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -84,7 +84,7 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml
@@ -1,7 +1,7 @@
-  model_name: deepseek_v4_pro_dspark
+  model_name: deepseek_v4_pro_nvfp4_dspark
-  model_dir_name: DeepSeek-V4-Pro-DSpark
+  model_dir_name: DeepSeek-V4-Pro-nvfp4-DSpark
@@ -84,7 +84,7 @@ worker_config:
-      speculative_model: DeepSeek-V4-Pro-DSpark
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml
@@ -1,7 +1,7 @@
-  model_name: deepseek_v4_pro_dspark
+  model_name: deepseek_v4_pro_nvfp4_dspark
-  model_dir_name: DeepSeek-V4-Pro-DSpark
+  model_dir_name: DeepSeek-V4-Pro-nvfp4-DSpark
@@ -84,7 +84,7 @@ worker_config:
-      speculative_model: DeepSeek-V4-Pro-DSpark
diff -- tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml
@@ -1,7 +1,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3; `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL.yaml` modified +3/-3; `tests/scripts/perf/disaggregated/gb300_deepseek-v4-pro-dspark_agentx_con1456_ctx3_dep8_gen1_dep16_eplb0_dspark5_ccb-NIXL.yaml` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/integration/defs/.test_durations`, `tests/integration/defs/perf/_model_paths.py`, `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_perf_sanity_ctx3_node2_gpu8_gen1_node4_gpu16.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18723 - [None][feat] Enable NVFP4 KV for DSV4

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18723
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` and 9 files; associated commits `c8a88ab98985`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 29 files, +1790/-159, 2887 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +253/-40 (293 lines); hunks: -15,6 +15,7; -52,7 +53,7; symbols: get_attn_dim, get_token_bytes, _estimate_non_sliding_attn_size_per_token, touching `get_attn_dim, get_token_bytes, _estimate_non_sliding_attn_size_per_token`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +129/-6 (135 lines); hunks: -19,6 +19,7; -27,10 +28,13; symbols: DeepseekV4TrtllmAttention, uses_fp4_mla_attention, _configure_compress_quant_mode, update_quant_config, touching `DeepseekV4TrtllmAttention, uses_fp4_mla_attention, _configure_compress_quant_mode`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +23/-0 (23 lines); hunks: -44,6 +44,7 @@ class KVCacheDtype(IntEnum):; -52,8 +53,11 @@ class KVCacheDtype(IntEnum):; symbols: KVCacheDtype, resolve_kv_cache_dtype, Compressor, __init__, touching `KVCacheDtype, resolve_kv_cache_dtype, Compressor`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` modified +15/-0 (15 lines); hunks: -1963,13 +1963,16 @@ def test_fused_postprocess_scatter(; -2070,13 +2073,16 @@ def test_fused_postprocess_scatter_masked_batches(; symbols: test_fused_postprocess_scatter, test_fused_postprocess_scatter_masked_batches, test_fused_postprocess_scatter_fp8_pertensor, test_fused_postprocess_scatter_fp8_blockwise, touching `test_fused_postprocess_scatter, test_fused_postprocess_scatter_masked_batches, test_fused_postprocess_scatter_fp8_pertensor`.
- Code diff details:
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +253/-40 (293 lines); hunks: -15,6 +15,7; -52,7 +53,7; symbols: get_attn_dim, get_token_bytes, _estimate_non_sliding_attn_size_per_token
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +129/-6 (135 lines); hunks: -19,6 +19,7; -27,10 +28,13; symbols: DeepseekV4TrtllmAttention, uses_fp4_mla_attention, _configure_compress_quant_mode, update_quant_config
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +23/-0 (23 lines); hunks: -44,6 +44,7 @@ class KVCacheDtype(IntEnum):; -52,8 +53,11 @@ class KVCacheDtype(IntEnum):; symbols: KVCacheDtype, resolve_kv_cache_dtype, Compressor, __init__
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` modified +15/-0 (15 lines); hunks: -1963,13 +1963,16 @@ def test_fused_postprocess_scatter(; -2070,13 +2073,16 @@ def test_fused_postprocess_scatter_masked_batches(; symbols: test_fused_postprocess_scatter, test_fused_postprocess_scatter_masked_batches, test_fused_postprocess_scatter_fp8_pertensor, test_fused_postprocess_scatter_fp8_blockwise
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +12/-0 (12 lines); hunks: -1787,10 +1787,16 @@ def __init__(self, head_dim: int, tokens_per_block: int...; -1902,13 +1908,16 @@ def fake_postprocess_scatter(; symbols: __init__, get_buffers, get_compress_scale_buffers, _create_small_compressor
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py
@@ -15,6 +15,7 @@
+from math import gcd
@@ -52,7 +53,7 @@
-from .compressor import KVCacheDtype
+from .compressor import NVFP4_COMPRESS_RESIDUAL_DIM, KVCacheDtype
@@ -62,6 +63,10 @@
+COMPRESS_BLOCK_SCALE_ROLE = DataRole("deepseek_v4_compress_block_scale")
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py
@@ -19,6 +19,7 @@
+from tensorrt_llm._torch.attention.backends.fmha.manager import FmhaManager
@@ -27,10 +28,13 @@
+from tensorrt_llm.bindings import DataType
+from tensorrt_llm.quantization import QuantMode
+from ..dsa.backend import _get_nvfp4_mla_kv_cache_amax
-from .compressor import Compressor
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py
@@ -44,6 +44,7 @@ class KVCacheDtype(IntEnum):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +253/-40; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +129/-6; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +23/-0; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` modified +5/-3; `cpp/tensorrt_llm/kernels/deepseekV4BlockTable.cu` modified +2/-1
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_kernel.py` modified +15/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_compressor_module.py` modified +12/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_glm52.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_b300.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18783 - [None][feat] Add DeepSeek-V4 support to NVFP4 cold-page KV Cache Compression

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18783
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`; associated commits `d45727804501`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +1415/-115, 2139 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +203/-1 (204 lines); hunks: -15,6 +15,7; -33,19 +34,29; symbols: _create_deepseek_v4_cache_manager, test_nvfp4_cold_page_codec_accepts_real_csa_hca_lifecycle, test_nvfp4_cold_page_codec_migrates_real_csa_hca_through_host, touching `_create_deepseek_v4_cache_manager, test_nvfp4_cold_page_codec_accepts_real_csa_hca_lifecycle, test_nvfp4_cold_page_codec_migrates_real_csa_hca_through_host`; `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/nvfp4_quantization.py` modified +225/-31 (256 lines); hunks: -21,11 +21,15; -37,11 +41,32; symbols: _Nvfp4Scales, _Nvfp4LayerLayout, _load_modelopt_nvfp4_scales, Nvfp4ColdPageQuantizationCompression, touching `_Nvfp4Scales, _Nvfp4LayerLayout, _load_modelopt_nvfp4_scales`; `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` modified +171/-30 (201 lines); hunks: -57,21 +57,30 @@ constexpr std::size_t kKernelParameterLimitBytes = 32764;; -87,6 +96,8 @@ struct Nvfp4ColdPageKernelParams; symbols: Nvfp4ColdPageTransform, touching `Nvfp4ColdPageTransform`; `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +9/-1 (10 lines); hunks: -65,6 +65,8; -2808,8 +2810,14 @@ class KVCacheCompressionManager(BaseResourceManager):; symbols: KVCacheCompressionManager, __init__, touching `KVCacheCompressionManager, __init__`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +203/-1 (204 lines); hunks: -15,6 +15,7; -33,19 +34,29; symbols: _create_deepseek_v4_cache_manager, test_nvfp4_cold_page_codec_accepts_real_csa_hca_lifecycle, test_nvfp4_cold_page_codec_migrates_real_csa_hca_through_host
  - `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/nvfp4_quantization.py` modified +225/-31 (256 lines); hunks: -21,11 +21,15; -37,11 +41,32; symbols: _Nvfp4Scales, _Nvfp4LayerLayout, _load_modelopt_nvfp4_scales, Nvfp4ColdPageQuantizationCompression
  - `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` modified +171/-30 (201 lines); hunks: -57,21 +57,30 @@ constexpr std::size_t kKernelParameterLimitBytes = 32764;; -87,6 +96,8 @@ struct Nvfp4ColdPageKernelParams; symbols: Nvfp4ColdPageTransform
  - `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +9/-1 (10 lines); hunks: -65,6 +65,8; -2808,8 +2810,14 @@ class KVCacheCompressionManager(BaseResourceManager):; symbols: KVCacheCompressionManager, __init__
  - `tensorrt_llm/_torch/pyexecutor/_util.py` modified +4/-1 (5 lines); hunks: -3102,7 +3102,10 @@ def create_kv_cache_compression_manager(; symbols: create_kv_cache_compression_manager
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py
@@ -15,6 +15,7 @@
+from unittest.mock import patch
@@ -33,19 +34,29 @@
+from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
+    Nvfp4ColdPageQuantizationCompression,
+)
+from tensorrt_llm.bindings.internal import kv_cache_compression as native_kvcc
diff -- tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/nvfp4_quantization.py
@@ -21,11 +21,15 @@
+    from transformers import PretrainedConfig
+    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig
-_LayerScales = tuple[tuple[float, float], tuple[float, float]]
+_ScalePair = tuple[float, float]
+_LayerScales = dict[str, _ScalePair]
-_IDENTITY_NVFP4_SCALES: _LayerScales = ((1.0, 1.0), (1.0, 1.0))
diff -- cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu
@@ -57,21 +57,30 @@ constexpr std::size_t kKernelParameterLimitBytes = 32764;
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +203/-1
  - runtime: `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/nvfp4_quantization.py` modified +225/-31; `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` modified +171/-30; `tensorrt_llm/_torch/pyexecutor/resource_manager.py` modified +9/-1; `tensorrt_llm/_torch/pyexecutor/_util.py` modified +4/-1; `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention.py` modified +1/-2; `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.h` modified +1/-1
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/kernels/nvfp4ColdPageKernelsTest.cpp`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/executor/kv_cache/test_kv_cache_compression_manager.py`, `tests/unittest/_torch/kv_cache_compression/test_quantization_for_cold_page.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19139 - [None][fix] Complete DeepSeek-V4 Rubin BF16 dispatch and optimize MLA KV expansion

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19139
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`; associated commits `dc6d6631586c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +411/-45, 560 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +159/-0 (159 lines); hunks: -24,7 +24,9; -343,3 +345,160 @@ def test_deepseek_v4_o_proj(num_tokens: int, dtype_str: str):; symbols: test_deepseek_v4_o_proj, test_dsv4_q_b_dispatch, gemm, _output_projection, touching `test_deepseek_v4_o_proj, test_dsv4_q_b_dispatch, gemm`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py` modified +39/-28 (67 lines); hunks: -12,6 +12,10; -29,7 +33,26; symbols: _q_b_proj_cute_dsl_bf16, project_sparse_attn_output, forward_sparse_attn, _fused_q_fp8_quant_enabled, touching `_q_b_proj_cute_dsl_bf16, project_sparse_attn_output, forward_sparse_attn`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +159/-0 (159 lines); hunks: -24,7 +24,9; -343,3 +345,160 @@ def test_deepseek_v4_o_proj(num_tokens: int, dtype_str: str):; symbols: test_deepseek_v4_o_proj, test_dsv4_q_b_dispatch, gemm, _output_projection
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py` modified +39/-28 (67 lines); hunks: -12,6 +12,10; -29,7 +33,26; symbols: _q_b_proj_cute_dsl_bf16, project_sparse_attn_output, forward_sparse_attn, _fused_q_fp8_quant_enabled
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py
@@ -24,7 +24,9 @@
+from tensorrt_llm._torch import cute_dsl_utils
+from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import module as dsv4
@@ -343,3 +345,160 @@ def test_deepseek_v4_o_proj(num_tokens: int, dtype_str: str):
+@pytest.mark.cpu_only
+@pytest.mark.parametrize(
+    "sm,dsl,rubin,expected",
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py
@@ -12,6 +12,10 @@
+from tensorrt_llm._torch.cute_dsl_utils import (
+    IS_CUTLASS_DSL_AVAILABLE,
+    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
+)
@@ -29,7 +33,26 @@
-_q_b_proj_cute_dsl_import_ok: Optional[bool] = None
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +159/-0
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py` modified +39/-28
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tests/unittest/_torch/attention/test_attention_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19184 - [None][feat] Rubin kernels & attention: DSV4/DSA, CuteDSL GEMM

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19184
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py`, `tensorrt_llm/_torch/models/modeling_deepseekv4.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py`; associated commits `96d0c509e34d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 45 files, +4021/-924, 6828 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-2 (9 lines); hunks: -781,8 +781,8 @@ def load_o_a_proj(module_name: str, module) -> None:; -1598,6 +1598,7 @@ def __init__(; symbols: load_o_a_proj, __init__, touching `load_o_a_proj, __init__`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py` modified +35/-20 (55 lines); hunks: -4,6 +4,7; -129,11 +130,10 @@ def _apply_q_rope(self, q: torch.Tensor, position_ids: tor...; symbols: _apply_q_rope, _project_and_quantize_q, _is_fused_project_mxfp4_enabled, touching `_apply_q_rope, _project_and_quantize_q, _is_fused_project_mxfp4_enabled`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +31/-18 (49 lines); hunks: -30,7 +30,7; -66,6 +66,7 @@ def calculate_reference_deepseek_v4_o_proj(; symbols: calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj, touching `calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj`; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +3/-0 (3 lines); hunks: -697,6 +697,7 @@ def fake_decoder_layer_init(self, model_config, *_args, **_k...; -715,6 +716,8 @@ def fake_decoder_layer_init(self, model_config, *_args, **_k...; symbols: fake_decoder_layer_init, touching `fake_decoder_layer_init`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-2 (9 lines); hunks: -781,8 +781,8 @@ def load_o_a_proj(module_name: str, module) -> None:; -1598,6 +1598,7 @@ def __init__(; symbols: load_o_a_proj, __init__
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py` modified +35/-20 (55 lines); hunks: -4,6 +4,7; -129,11 +130,10 @@ def _apply_q_rope(self, q: torch.Tensor, position_ids: tor...; symbols: _apply_q_rope, _project_and_quantize_q, _is_fused_project_mxfp4_enabled
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +31/-18 (49 lines); hunks: -30,7 +30,7; -66,6 +66,7 @@ def calculate_reference_deepseek_v4_o_proj(; symbols: calculate_reference_deepseek_v4_o_proj, test_deepseek_v4_o_proj
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +3/-0 (3 lines); hunks: -697,6 +697,7 @@ def fake_decoder_layer_init(self, model_config, *_args, **_k...; -715,6 +716,8 @@ def fake_decoder_layer_init(self, model_config, *_args, **_k...; symbols: fake_decoder_layer_init
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv4.py
@@ -781,8 +781,8 @@ def load_o_a_proj(module_name: str, module) -> None:
-            # Skip the BF16 dequant when the destination is FP8 (the cute_dsl
-            # FP8 BMM path on SM100 consumes the native FP8 weight directly).
+            # Skip BF16 dequant when the architecture-specific CuTe DSL BMM
+            # consumes the native FP8 weight directly.
@@ -1598,6 +1598,7 @@ def __init__(
+            use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py
@@ -4,6 +4,7 @@
+import os
@@ -129,11 +130,10 @@ def _apply_q_rope(self, q: torch.Tensor, position_ids: torch.Tensor) -> torch.Te
-    def _project_and_quantize_q(
-        self, qr: torch.Tensor, position_ids: torch.Tensor
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
-        """Project and quantize Q, using the fused MXFP4 path when supported."""
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py
@@ -30,7 +30,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv4.py` modified +7/-2; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py` modified +35/-20
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py` modified +31/-18; `tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/common/attentionWorkspaceTest.cpp`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_o_proj.py`, `tests/unittest/_torch/attention/sparse/dsa/test_indexer_gvr_prior.py`, `tests/unittest/_torch/attention/sparse/dsa/test_metadata_topk_init.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19424 - [None][test] Disable ctx overlap scheduler for deepseek-v4-pro con8 disagg config

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19424
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml`; associated commits `83624baf69fa`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +1/-1 (2 lines); hunks: -103,5 +103,5 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +1/-1 (2 lines); hunks: -103,5 +103,5 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -103,5 +103,5 @@ worker_config:
-    disable_overlap_scheduler: false
+    disable_overlap_scheduler: true
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/scripts/perf-sanity/disaggregated/gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19305 - [None][feat] NVFP4 MLA residual switch for DeepSeek-V4

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19305
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py`; associated commits `45f2ef17aca4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +44/-21, 264 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +27/-15 (42 lines); hunks: -53,7 +53,7; -98,6 +98,7 @@ def get_token_bytes(; symbols: get_token_bytes, _estimate_non_sliding_attn_size_per_token, _get_attn_bytes_per_token, touching `get_token_bytes, _estimate_non_sliding_attn_size_per_token, _get_attn_bytes_per_token`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +12/-1 (13 lines); hunks: -1,6 +1,7; -59,6 +60,13 @@ class KVCacheDtype(IntEnum):; symbols: KVCacheDtype, get_nvfp4_compress_residual_dim, resolve_kv_cache_dtype, __init__, touching `KVCacheDtype, get_nvfp4_compress_residual_dim, resolve_kv_cache_dtype`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +5/-5 (10 lines); hunks: -34,7 +34,7; -344,7 +344,7 @@ def sparse_attn_predict(; symbols: sparse_attn_predict, touching `sparse_attn_predict`.
- Code diff details:
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +27/-15 (42 lines); hunks: -53,7 +53,7; -98,6 +98,7 @@ def get_token_bytes(; symbols: get_token_bytes, _estimate_non_sliding_attn_size_per_token, _get_attn_bytes_per_token
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +12/-1 (13 lines); hunks: -1,6 +1,7; -59,6 +60,13 @@ class KVCacheDtype(IntEnum):; symbols: KVCacheDtype, get_nvfp4_compress_residual_dim, resolve_kv_cache_dtype, __init__
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +5/-5 (10 lines); hunks: -34,7 +34,7; -344,7 +344,7 @@ def sparse_attn_predict(; symbols: sparse_attn_predict
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py
@@ -53,7 +53,7 @@
-from .compressor import NVFP4_COMPRESS_RESIDUAL_DIM, KVCacheDtype
+from .compressor import KVCacheDtype, get_nvfp4_compress_residual_dim
@@ -98,6 +98,7 @@ def get_token_bytes(
+    nvfp4_residual_dim: int | None = None,
@@ -142,7 +143,9 @@ def get_token_bytes(
-        storage_dim = attn_dim + NVFP4_COMPRESS_RESIDUAL_DIM
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py
@@ -1,6 +1,7 @@
+import os
@@ -59,6 +60,13 @@ class KVCacheDtype(IntEnum):
+def get_nvfp4_compress_residual_dim() -> int:
+    """Resolve the NVFP4 COMPRESS layout from the MLA residual startup switch."""
+    if os.environ.get("TRTLLM_NVFP4_MLA_RESIDUAL_QUANTIZATION", "1") == "1":
+        return NVFP4_COMPRESS_RESIDUAL_DIM
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py
@@ -34,7 +34,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +27/-15; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py` modified +12/-1; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/compressor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19502 - [TRTLLM-16283][feat] Prepare DeepSeek V4 attention for sparse KV offload

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19502
- Status/date: merged / 2026-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/metadata.py`, `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/offload.py` and 9 files; associated commits `cc7175930354`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +2917/-21, 3245 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py` added +936/-0 (936 lines); hunks: -0,0 +1,936; symbols: _metadata, test_sparse_offload_rejects_configuration_before_allocation, test_sparse_offload_disabled_allocates_nothing, test_sparse_offload_stage_without_local_csa_allocates_nothing, touching `_metadata, test_sparse_offload_rejects_configuration_before_allocation, test_sparse_offload_disabled_allocates_nothing`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: _cuda_tensor, _selection_reference, _merge_reference, test_select_history_pages_boundaries, touching `_cuda_tensor, _selection_reference, _merge_reference`; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` modified +467/-2 (469 lines); hunks: -55,6 +55,9 @@ def _deepseek_v4_local_to_global_kernel(; -92,6 +95,10 @@ def _deepseek_v4_local_to_global_kernel(; symbols: _deepseek_v4_local_to_global_kernel, deepseek_v4_local_to_global_indices, touching `_deepseek_v4_local_to_global_kernel, deepseek_v4_local_to_global_indices`; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +378/-1 (379 lines); hunks: -15,13 +15,16; -38,6 +41,10; symbols: _SparseRuntime, __init__, is_sparse, get_layer_group_id, touching `_SparseRuntime, __init__, is_sparse`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py` added +936/-0 (936 lines); hunks: -0,0 +1,936; symbols: _metadata, test_sparse_offload_rejects_configuration_before_allocation, test_sparse_offload_disabled_allocates_nothing, test_sparse_offload_stage_without_local_csa_allocates_nothing
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: _cuda_tensor, _selection_reference, _merge_reference, test_select_history_pages_boundaries
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` modified +467/-2 (469 lines); hunks: -55,6 +55,9 @@ def _deepseek_v4_local_to_global_kernel(; -92,6 +95,10 @@ def _deepseek_v4_local_to_global_kernel(; symbols: _deepseek_v4_local_to_global_kernel, deepseek_v4_local_to_global_indices
  - `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +378/-1 (379 lines); hunks: -15,13 +15,16; -38,6 +41,10; symbols: _SparseRuntime, __init__, is_sparse, get_layer_group_id
  - `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +357/-16 (373 lines); hunks: -16,18 +16,20; -46,6 +48,8; symbols: __init__, _add_layer, copy_batch_sliding_block_tables
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py
@@ -0,0 +1,936 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py
@@ -0,0 +1,484 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py
@@ -55,6 +55,9 @@ def _deepseek_v4_local_to_global_kernel(
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py` added +936/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py` added +484/-0; `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py` modified +378/-1
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` modified +467/-2; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py` modified +357/-16; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/metadata.py` modified +124/-0; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/offload.py` added +88/-0; `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_kernels.py`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_offload.py`, `tests/unittest/llmapi/test_llm_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
