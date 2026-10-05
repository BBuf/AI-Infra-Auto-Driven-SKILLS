# TensorRT-LLM MiniMax M2/M3 Series Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu` | [#12163](https://github.com/NVIDIA/TensorRT-LLM/pull/12163) |
| `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h` | [#12163](https://github.com/NVIDIA/TensorRT-LLM/pull/12163) |
| `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.cu` | [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318) |
| `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.h` | [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318) |
| `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` | [#17236](https://github.com/NVIDIA/TensorRT-LLM/pull/17236), [#18154](https://github.com/NVIDIA/TensorRT-LLM/pull/18154) |
| `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h` | [#17236](https://github.com/NVIDIA/TensorRT-LLM/pull/17236) |
| `cpp/tensorrt_llm/thop/minimaxM3Fp8IndexerOp.cpp` | [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318) |
| `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp` | [#17236](https://github.com/NVIDIA/TensorRT-LLM/pull/17236) |
| `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` | [#15587](https://github.com/NVIDIA/TensorRT-LLM/pull/15587), [#15687](https://github.com/NVIDIA/TensorRT-LLM/pull/15687), [#17238](https://github.com/NVIDIA/TensorRT-LLM/pull/17238), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `examples/configs/curated/minimax-m3-throughput.yaml` | [#15587](https://github.com/NVIDIA/TensorRT-LLM/pull/15587), [#15687](https://github.com/NVIDIA/TensorRT-LLM/pull/15687) |
| `examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `examples/visual_gen/models/minimax_h3.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/__init__.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872), [#19288](https://github.com/NVIDIA/TensorRT-LLM/pull/19288), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/common.py` | [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205), [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/__init__.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py` | [#18614](https://github.com/NVIDIA/TensorRT-LLM/pull/18614), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#18614](https://github.com/NVIDIA/TensorRT-LLM/pull/18614), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/paged_cache.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_availability.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` | [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205), [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#18614](https://github.com/NVIDIA/TensorRT-LLM/pull/18614), [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113), [#19288](https://github.com/NVIDIA/TensorRT-LLM/pull/19288), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422), [#19423](https://github.com/NVIDIA/TensorRT-LLM/pull/19423) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_backend.py` | [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113) |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_kernels.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_metadata.py` | [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py` | [#17842](https://github.com/NVIDIA/TensorRT-LLM/pull/17842) |
| `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py` | [#16468](https://github.com/NVIDIA/TensorRT-LLM/pull/16468) |
| `tensorrt_llm/_torch/models/modeling_minimaxm2.py` | [#10532](https://github.com/NVIDIA/TensorRT-LLM/pull/10532), [#12163](https://github.com/NVIDIA/TensorRT-LLM/pull/12163), [#12182](https://github.com/NVIDIA/TensorRT-LLM/pull/12182), [#14314](https://github.com/NVIDIA/TensorRT-LLM/pull/14314), [#17360](https://github.com/NVIDIA/TensorRT-LLM/pull/17360) |
| `tensorrt_llm/_torch/models/modeling_minimaxm3.py` | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292), [#15857](https://github.com/NVIDIA/TensorRT-LLM/pull/15857), [#15923](https://github.com/NVIDIA/TensorRT-LLM/pull/15923), [#15937](https://github.com/NVIDIA/TensorRT-LLM/pull/15937), [#16291](https://github.com/NVIDIA/TensorRT-LLM/pull/16291), [#16468](https://github.com/NVIDIA/TensorRT-LLM/pull/16468), [#16904](https://github.com/NVIDIA/TensorRT-LLM/pull/16904), [#17238](https://github.com/NVIDIA/TensorRT-LLM/pull/17238), [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318), [#17565](https://github.com/NVIDIA/TensorRT-LLM/pull/17565), [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205), [#18605](https://github.com/NVIDIA/TensorRT-LLM/pull/18605), ... (17 total) |
| `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292), [#16696](https://github.com/NVIDIA/TensorRT-LLM/pull/16696), [#18466](https://github.com/NVIDIA/TensorRT-LLM/pull/18466) |
| `tensorrt_llm/_torch/visual_gen/models/minimax_h3/__init__.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tensorrt_llm/serve/tool_parser/minimax_m2_parser.py` | no direct PR-number commit |
| `tensorrt_llm/serve/tool_parser/minimax_m3_parser.py` | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292) |
| `tests/integration/defs/examples/visual_gen/test_minimax_h3_e2e.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tests/microbenchmarks/minimax_all_reduce.py` | [#12163](https://github.com/NVIDIA/TensorRT-LLM/pull/12163) |
| `tests/microbenchmarks/minimax_m3_index_decode_score.py` | [#17842](https://github.com/NVIDIA/TensorRT-LLM/pull/17842), [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml` | [#18416](https://github.com/NVIDIA/TensorRT-LLM/pull/18416) |
| `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` | [#18416](https://github.com/NVIDIA/TensorRT-LLM/pull/18416) |
| `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` | [#18416](https://github.com/NVIDIA/TensorRT-LLM/pull/18416) |
| `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` | [#18416](https://github.com/NVIDIA/TensorRT-LLM/pull/18416) |
| `tests/scripts/perf/disaggregated/vr200_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/scripts/perf/disaggregated/vr200_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/scripts/perf/disaggregated/vr200_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/scripts/perf/disaggregated/vr200_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` | no direct PR-number commit |
| `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_dense_decode.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_index_decode_score.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_msa_selector.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) |
| `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_shared_draft_layers.py` | [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_sparse_attn_decode.py` | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` | [#16468](https://github.com/NVIDIA/TensorRT-LLM/pull/16468), [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tests/unittest/_torch/models/test_minimax_m3.py` | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292), [#16468](https://github.com/NVIDIA/TensorRT-LLM/pull/16468), [#16859](https://github.com/NVIDIA/TensorRT-LLM/pull/16859), [#16904](https://github.com/NVIDIA/TensorRT-LLM/pull/16904), [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318), [#17565](https://github.com/NVIDIA/TensorRT-LLM/pull/17565), [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205), [#18605](https://github.com/NVIDIA/TensorRT-LLM/pull/18605), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113), [#19298](https://github.com/NVIDIA/TensorRT-LLM/pull/19298), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422), [#19423](https://github.com/NVIDIA/TensorRT-LLM/pull/19423) |
| `tests/unittest/_torch/models/test_minimax_m3_vl.py` | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292), [#18466](https://github.com/NVIDIA/TensorRT-LLM/pull/18466) |
| `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py` | [#19423](https://github.com/NVIDIA/TensorRT-LLM/pull/19423) |
| `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` | [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205) |
| `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` | [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318), [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113) |
| `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_main_kv_insert.py` | [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205) |
| `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_nvfp4_horizontal_producer.py` | [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |
| `tests/unittest/_torch/visual_gen/test_minimax_h3_pipeline.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tests/unittest/_torch/visual_gen/test_minimax_h3_scheduler_packing.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tests/unittest/_torch/visual_gen/test_minimax_h3_transformer.py` | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) |
| `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` | [#16017](https://github.com/NVIDIA/TensorRT-LLM/pull/16017), [#17374](https://github.com/NVIDIA/TensorRT-LLM/pull/17374), [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) |

## PR Coverage Summary

- Git-traced PRs: 37
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 37
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-01-14 | [#10532](https://github.com/NVIDIA/TensorRT-LLM/pull/10532) | merged | [None][feat] MiniMax M2 support | `tensorrt_llm/_torch/models/modeling_minimaxm2.py` |
| 2026-03-17 | [#12182](https://github.com/NVIDIA/TensorRT-LLM/pull/12182) | merged | [https://nvbugs/5879588][fix] fix MiniMax model loading bugs | `tensorrt_llm/_torch/models/modeling_minimaxm2.py` |
| 2026-04-20 | [#12163](https://github.com/NVIDIA/TensorRT-LLM/pull/12163) | merged | [None][feat] Minimax RMS norm optimization | `tensorrt_llm/_torch/models/modeling_minimaxm2.py`, `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu`, `tests/microbenchmarks/minimax_all_reduce.py` |
| 2026-05-28 | [#14314](https://github.com/NVIDIA/TensorRT-LLM/pull/14314) | merged | [TRTLLM-12762][fix] Enable multi-node TP for MiniMax-M2 | `tensorrt_llm/_torch/models/modeling_minimaxm2.py` |
| 2026-06-19 | [#15292](https://github.com/NVIDIA/TensorRT-LLM/pull/15292) | merged | [None][feat] BREAKING: Add MiniMax-M3 PyTorch backend bring-up with API changes | `tests/unittest/_torch/models/test_minimax_m3_vl.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-06-26 | [#15587](https://github.com/NVIDIA/TensorRT-LLM/pull/15587) | merged | [None][doc] Add deploy guide for Minimax M3 | `examples/configs/curated/minimax-m3-throughput.yaml`, `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` |
| 2026-07-01 | [#15687](https://github.com/NVIDIA/TensorRT-LLM/pull/15687) | merged | [TRTLLM-13458][feat] Support Minimax M3 MXFP8 checkpoint | `examples/configs/curated/minimax-m3-throughput.yaml`, `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `tensorrt_llm/_torch/modules/fused_moe/routing.py` |
| 2026-07-03 | [#15857](https://github.com/NVIDIA/TensorRT-LLM/pull/15857) | merged | [TRTLLM-13458][feat] Support Minimax M3 NVFP4 checkpoint | `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-06 | [#15937](https://github.com/NVIDIA/TensorRT-LLM/pull/15937) | merged | [TRTLLM-13973][fix] Restrict MiniMax M3 dense SDPA backends | `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-07 | [#15923](https://github.com/NVIDIA/TensorRT-LLM/pull/15923) | merged | [TRTLLM-13968][fix] Enable MiniMax M3 piecewise CUDA graphs | `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-22 | [#16291](https://github.com/NVIDIA/TensorRT-LLM/pull/16291) | merged | [TRTLLM-14019][feat] Add MiniMax-M3 MSA sparse attention backend [revised] | `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-22 | [#16468](https://github.com/NVIDIA/TensorRT-LLM/pull/16468) | merged | [TRTLLM-14255][fix] migrate MiniMax M3 to loader v2 for TP8 support | `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-22 | [#16696](https://github.com/NVIDIA/TensorRT-LLM/pull/16696) | merged | [None][feat] Add BaseMultimodalDummyInputsBuilder to minimaxm3_vl | `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` |
| 2026-07-23 | [#16017](https://github.com/NVIDIA/TensorRT-LLM/pull/16017) | merged | [TRTLLM-13969][feat] Support MiniMax M3 for Disaggregated Serving | `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py`, `tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py`, `tensorrt_llm/_torch/disaggregation/native/peer.py` |
| 2026-07-29 | [#16904](https://github.com/NVIDIA/TensorRT-LLM/pull/16904) | merged | [None][perf] Fuse index-q/index-k projections in MinimaxM3 | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-07-29 | [#16859](https://github.com/NVIDIA/TensorRT-LLM/pull/16859) | merged | [None][perf] Fuse MiniMax-M3 MoE routing | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/modules/fused_moe/routing.py` |
| 2026-08-07 | [#17360](https://github.com/NVIDIA/TensorRT-LLM/pull/17360) | merged | [None][feat] Default MiniMax M2 to KV cache manager V2 | `tensorrt_llm/_torch/models/modeling_minimaxm2.py` |
| 2026-08-18 | [#17565](https://github.com/NVIDIA/TensorRT-LLM/pull/17565) | merged | [https://nvbugs/6373561][fix] Fix minimax M3 E2E test | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-08-23 | [#17236](https://github.com/NVIDIA/TensorRT-LLM/pull/17236) | merged | [None][perf] Optimize MiniMax-M3 MSA block selection | `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu`, `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp`, `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h` |
| 2026-08-24 | [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318) | merged | [None][perf] Use FP8 MiniMax-M3 MSA indexer QK | `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` |
| 2026-08-26 | [#17842](https://github.com/NVIDIA/TensorRT-LLM/pull/17842) | merged | [None][feat] Add the ported MiniMax-M3 decode kernels ahead of their wiring | `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py`, `tests/microbenchmarks/minimax_m3_index_decode_score.py` |
| 2026-08-27 | [#18154](https://github.com/NVIDIA/TensorRT-LLM/pull/18154) | merged | [None][perf] Histogram top-k for MiniMax-M3 block selector | `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` |
| 2026-08-31 | [#18416](https://github.com/NVIDIA/TensorRT-LLM/pull/18416) | merged | [None][test] Add MiniMax-M3 disaggregated perf recipes to QA multi-node list | `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` |
| 2026-09-01 | [#18466](https://github.com/NVIDIA/TensorRT-LLM/pull/18466) | merged | [None][fix] Support MiniMax-M3 vision meta init | `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`, `tests/unittest/_torch/models/test_minimax_m3_vl.py` |
| 2026-09-01 | [#17238](https://github.com/NVIDIA/TensorRT-LLM/pull/17238) | merged | [None][perf] Optimize MiniMax-M3 MXFP8 GEMMs | `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` |
| 2026-09-04 | [#17374](https://github.com/NVIDIA/TensorRT-LLM/pull/17374) | merged | [None][fix] Use HND mapping for MiniMax-M3 MSA KV cache | `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py`, `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/disaggregation/resource/page.py` |
| 2026-09-12 | [#18611](https://github.com/NVIDIA/TensorRT-LLM/pull/18611) | merged | [None][perf] Wire in custom decode kernels for MinimaxM3 | `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`, `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py` |
| 2026-09-12 | [#18733](https://github.com/NVIDIA/TensorRT-LLM/pull/18733) | merged | [TRTLLM-16194][feat] Add MiniMax H3 support | `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py`, `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py`, `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` |
| 2026-09-14 | [#18872](https://github.com/NVIDIA/TensorRT-LLM/pull/18872) | merged | [TRTLLM-14093][feat] Eagle3 support for MiniMax-M3 | `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` |
| 2026-09-15 | [#18614](https://github.com/NVIDIA/TensorRT-LLM/pull/18614) | merged | [None][perf] Fuse MiniMax-M3 MSA per-layer KV-cache writes into one kernel | `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py`, `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` |
| 2026-09-16 | [#18605](https://github.com/NVIDIA/TensorRT-LLM/pull/18605) | merged | [None][feat] Support MiniMax-M3 in MegaMoE CuTeDSL | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py` |
| 2026-09-16 | [#19298](https://github.com/NVIDIA/TensorRT-LLM/pull/19298) | merged | [None][fix] Align MiniMax-M3 composition test fixtures with MoE interfaces | `tests/unittest/_torch/models/test_minimax_m3.py` |
| 2026-09-18 | [#18205](https://github.com/NVIDIA/TensorRT-LLM/pull/18205) | merged | [None][perf] Fuse MiniMax-M3 QKV and index projection | `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` |
| 2026-09-21 | [#19288](https://github.com/NVIDIA/TensorRT-LLM/pull/19288) | merged | [None][test] Cover MiniMax-M3 C++ NIXL bounce | `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` |
| 2026-09-23 | [#19423](https://github.com/NVIDIA/TensorRT-LLM/pull/19423) | merged | [None][perf] Extend MiniMax-M3 piecewise CUDA graphs coverage | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py` |
| 2026-09-26 | [#19422](https://github.com/NVIDIA/TensorRT-LLM/pull/19422) | merged | [None][perf] Add MiniMax-M3 NVFP4 KV cache support | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` |
| 2026-09-29 | [#19113](https://github.com/NVIDIA/TensorRT-LLM/pull/19113) | merged | [None][fix] Guard MiniMax-M3 FP8 indexer against padded -1 cache slots | `tests/unittest/_torch/models/test_minimax_m3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` |

## Per-PR Diff Audit Cards

### PR #10532 - [None][feat] MiniMax M2 support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10532
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; associated commits `e7882d5c7423`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +417/-3, 501 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm2.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: MiniMaxM2MoE, __init__, load_weights, forward, touching `MiniMaxM2MoE, __init__, load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm2.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: MiniMaxM2MoE, __init__, load_weights, forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm2.py
@@ -0,0 +1,314 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm2.py` added +314/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12182 - [https://nvbugs/5879588][fix] fix MiniMax model loading bugs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12182
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; associated commits `822f401b9bd8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +24/-13, 70 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +24/-12 (36 lines); hunks: -40,6 +40,26; -60,30 +80,22 @@ def __init__(; symbols: _EScoreCorrectionBiasHolder, __init__, load_weights, MiniMaxM2MoE, touching `_EScoreCorrectionBiasHolder, __init__, load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +24/-12 (36 lines); hunks: -40,6 +40,26; -60,30 +80,22 @@ def __init__(; symbols: _EScoreCorrectionBiasHolder, __init__, load_weights, MiniMaxM2MoE
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm2.py
@@ -40,6 +40,26 @@
+class _EScoreCorrectionBiasHolder(nn.Module):
+    """Holds e_score_correction_bias so the generic weight loader visits it with a narrow
+    prefix (block_sparse_moe.e_score_correction_bias). This avoids mark_consumed deleting
+    the whole block_sparse_moe prefix before gate and experts.backend load (see #11119).
+    """
+    def __init__(self, num_experts: int):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +24/-12
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12163 - [None][feat] Minimax RMS norm optimization

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12163
- Status/date: merged / 2026-04-20
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu`, `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h`, `tensorrt_llm/_torch/models/modeling_minimaxm2.py`, `tests/microbenchmarks/minimax_all_reduce.py`; associated commits `a56a8d2435b6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +1721/-32, 1865 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +64/-26 (90 lines); hunks: -19,16 +19,18; -114,6 +116,38 @@ def forward(; symbols: forward, MiniMaxRMSNorm, __init__, load_weights, touching `forward, MiniMaxRMSNorm, __init__`; `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu` added +885/-0 (885 lines); hunks: -0,0 +1,885; symbols: IndexHelper, touching `IndexHelper`; `tests/microbenchmarks/minimax_all_reduce.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: profile_minimax_allreduce_rms, func, profile_minimax_allreduce_rms_qk, minimax_allreduce_benchmark, touching `profile_minimax_allreduce_rms, func, profile_minimax_allreduce_rms_qk`; `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h` added +83/-0 (83 lines); hunks: -0,0 +1,83.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +64/-26 (90 lines); hunks: -19,16 +19,18; -114,6 +116,38 @@ def forward(; symbols: forward, MiniMaxRMSNorm, __init__, load_weights
  - `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu` added +885/-0 (885 lines); hunks: -0,0 +1,885; symbols: IndexHelper
  - `tests/microbenchmarks/minimax_all_reduce.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: profile_minimax_allreduce_rms, func, profile_minimax_allreduce_rms_qk, minimax_allreduce_benchmark
  - `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h` added +83/-0 (83 lines); hunks: -0,0 +1,83
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm2.py
@@ -19,16 +19,18 @@
-from tensorrt_llm.functional import PositionEmbeddingType
+from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType
+from tensorrt_llm.mapping import Mapping
+from ..distributed import AllReduce, MiniMaxAllReduceRMS
-from ..modules.linear import Linear
+from ..modules.linear import Linear, TensorParallelMode, copy_weight, load_weight_shard
diff -- cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu
@@ -0,0 +1,885 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- tests/microbenchmarks/minimax_all_reduce.py
@@ -0,0 +1,297 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +64/-26; `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.cu` added +885/-0; `cpp/tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h` added +83/-0
  - tests: `tests/microbenchmarks/minimax_all_reduce.py` added +297/-0
- Risk and verification: The diff ships test coverage in `tests/microbenchmarks/minimax_all_reduce.py`, `tests/unittest/_torch/multi_gpu/test_allreduce.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14314 - [TRTLLM-12762][fix] Enable multi-node TP for MiniMax-M2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14314
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; associated commits `82679b58ab6c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +46/-2, 103 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +46/-2 (48 lines); hunks: -19,6 +19,7; -119,7 +120,13 @@ def forward(; symbols: forward, MiniMaxRMSNorm, __init__, load_weights, touching `forward, MiniMaxRMSNorm, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +46/-2 (48 lines); hunks: -19,6 +19,7; -119,7 +120,13 @@ def forward(; symbols: forward, MiniMaxRMSNorm, __init__, load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm2.py
@@ -19,6 +19,7 @@
+from tensorrt_llm._ipc_utils import can_access_peer
@@ -119,7 +120,13 @@ def forward(
-        self, *, hidden_size: int, eps: float, mapping: Mapping, dtype: torch.dtype = torch.bfloat16
+        self,
+        *,
+        hidden_size: int,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +46/-2
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15292 - [None][feat] BREAKING: Add MiniMax-M3 PyTorch backend bring-up with API changes

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15292
- Status/date: merged / 2026-06-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`, `tensorrt_llm/serve/tool_parser/minimax_m3_parser.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/models/test_minimax_m3_vl.py`; associated commits `2a18bd4e4051`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 30 files, +11125/-76, 11602 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3_vl.py` added +2178/-0 (2178 lines); hunks: -0,0 +1,2178; symbols: _checkpoint_path, _has_cuda, _make_vision_config_dict, test_clip_vision_config_from_dict_uses_checkpoint_fields, touching `_checkpoint_path, _has_cuda, _make_vision_config_dict`; `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: compute_visual_token_count, compute_visual_token_counts, expand_multimodal_placeholders, merge_multimodal_embeddings, touching `compute_visual_token_count, compute_visual_token_counts, expand_multimodal_placeholders`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` added +1641/-0 (1641 lines); hunks: -0,0 +1,1641; symbols: is_minimax_m3_vl_config, _wrap_dict_as_config, get_text_config, get_text_model_config, touching `is_minimax_m3_vl_config, _wrap_dict_as_config, get_text_config`; `tests/unittest/_torch/models/test_minimax_m3.py` added +973/-0 (973 lines); hunks: -0,0 +1,973; symbols: _make_text_config, _make_vl_config, _checkpoint_path, _has_cuda, touching `_make_text_config, _make_vl_config, _checkpoint_path`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3_vl.py` added +2178/-0 (2178 lines); hunks: -0,0 +1,2178; symbols: _checkpoint_path, _has_cuda, _make_vision_config_dict, test_clip_vision_config_from_dict_uses_checkpoint_fields
  - `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` added +2081/-0 (2081 lines); hunks: -0,0 +1,2081; symbols: compute_visual_token_count, compute_visual_token_counts, expand_multimodal_placeholders, merge_multimodal_embeddings
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` added +1641/-0 (1641 lines); hunks: -0,0 +1,1641; symbols: is_minimax_m3_vl_config, _wrap_dict_as_config, get_text_config, get_text_model_config
  - `tests/unittest/_torch/models/test_minimax_m3.py` added +973/-0 (973 lines); hunks: -0,0 +1,973; symbols: _make_text_config, _make_vl_config, _checkpoint_path, _has_cuda
  - `tensorrt_llm/serve/tool_parser/minimax_m3_parser.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _get_param_type, _coerce_value, MiniMaxM3ToolParser, __init__
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3_vl.py
@@ -0,0 +1,2178 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""MiniMax-M3 VL model surface weight accounting.
+This test file covers / acceptance item #1: the real MiniMax-M3
+VL checkpoint loads with nonempty ``vision_tower.*``,
+``multi_modal_projector.*``, and ``patch_merge_mlp.*`` weights, and the
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py
@@ -0,0 +1,2081 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""MiniMax-M3 VL vision-side modules and multimodal runtime wiring.
+Contains the vision tower (patch embed -> pre-LN -> 32 encoder layers
+with 3D RoPE attention -> projector -> patch merger), the placeholder
+expansion helpers (``image_grid_thw`` / ``video_grid_thw`` ->
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -0,0 +1,1641 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3_vl.py` added +2178/-0; `tests/unittest/_torch/models/test_minimax_m3.py` added +973/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` added +2081/-0; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` added +1641/-0; `tensorrt_llm/serve/tool_parser/minimax_m3_parser.py` added +222/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15587 - [None][doc] Add deploy guide for Minimax M3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15587
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `examples/configs/curated/minimax-m3-throughput.yaml`; associated commits `a6f29f500e41`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +370/-1, 409 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/configs/curated/minimax-m3-throughput.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15; `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` added +328/-0 (328 lines); hunks: -0,0 +1,328.
- Code diff details:
  - `examples/configs/curated/minimax-m3-throughput.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
  - `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` added +328/-0 (328 lines); hunks: -0,0 +1,328
- Key code excerpts:

```diff
diff -- examples/configs/curated/minimax-m3-throughput.yaml
@@ -0,0 +1,15 @@
+max_batch_size: 256
+max_num_tokens: 8192
+tensor_parallel_size: 8
+moe_expert_parallel_size: 8
+enable_attention_dp: true
+trust_remote_code: true
diff -- docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md
@@ -0,0 +1,328 @@
+# Deployment Guide for MiniMax-M3 on TensorRT LLM
+## Introduction
+This deployment guide provides step-by-step instructions for running the MiniMax-M3 model using TensorRT LLM. It covers the complete setup required; from accessing model weights a
+MiniMax-M3 is a Mixture-of-Experts (MoE) model that uses MiniMax block-sparse attention. The first few layers use dense attention with a dense MLP, while the remaining layers comb
+MiniMax-M3 is served in **BF16**; no FP8/NVFP4 serving path is supported at this time. The block-sparse attention path does **not** currently support KV cache reuse or Multi-Token
+This guide deploys MiniMax-M3 on **8x NVIDIA GB200 GPUs across 2 nodes** (4 GPUs per node) using Slurm and the `trtllm-llmapi-launch` multi-node launcher, with the MoE experts dis
```

- Extracted files (not manually reviewed):
  - docs: `examples/configs/curated/minimax-m3-throughput.yaml` added +15/-0; `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` added +328/-0
- Risk and verification: This is mostly docs/examples in `docs/source/_static/config_db.json`, `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `docs/source/deployment-guide/index.rst`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #15687 - [TRTLLM-13458][feat] Support Minimax M3 MXFP8 checkpoint

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15687
- Status/date: merged / 2026-07-01
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `examples/configs/curated/minimax-m3-throughput.yaml`; associated commits `9de285042260`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +98/-31, 280 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/configs/curated/minimax-m3-throughput.yaml` modified +10/-1 (11 lines); hunks: -1,13 +1,22; `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +29/-12 (41 lines); hunks: -6,15 +6,20 @@ This deployment guide provides step-by-step instructions for r...; -24,24 +29,36 @@ The guide is intended for developers and practitioners seeki...; `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +10/-3 (13 lines); hunks: -677,9 +677,16 @@ def __init__(; symbols: __init__, apply, touching `__init__, apply`.
- Code diff details:
  - `examples/configs/curated/minimax-m3-throughput.yaml` modified +10/-1 (11 lines); hunks: -1,13 +1,22
  - `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +29/-12 (41 lines); hunks: -6,15 +6,20 @@ This deployment guide provides step-by-step instructions for r...; -24,24 +29,36 @@ The guide is intended for developers and practitioners seeki...
  - `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +10/-3 (13 lines); hunks: -677,9 +677,16 @@ def __init__(; symbols: __init__, apply
- Key code excerpts:

```diff
diff -- examples/configs/curated/minimax-m3-throughput.yaml
@@ -1,13 +1,22 @@
+# Cap max_seq_len explicitly. Without this cap the warmup decode used for
+# CUDA-graph capture inherits the checkpoint's max_position_embeddings
+# (1M for MXFP8, 512K for BF16), which makes both the dense GQA expansion
+# in `_dense_forward` and the per-Q FP32 slab in `_sparse_gqa_masked`
+# allocate gigabyte-scale temporaries that exceed the caching allocator's
+# graph-safe path and fail capture with cudaErrorStreamCaptureUnsupported.
diff -- docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md
@@ -6,15 +6,20 @@ This deployment guide provides step-by-step instructions for running the MiniMax
-MiniMax-M3 is served in **BF16**; no FP8/NVFP4 serving path is supported at this time. The block-sparse attention path does **not** currently support KV cache reuse or Multi-Token
+TensorRT LLM supports two precisions for MiniMax-M3:
+* **BF16** — the official upstream checkpoint from MiniMaxAI.
+* **MXFP8** — an NVIDIA-published checkpoint that quantizes the MoE/Linear weights to MXFP8 while keeping activations and the KV cache in BF16. The weights occupy ~half the memory
+The block-sparse attention path does **not** currently support KV cache reuse or Multi-Token Prediction (MTP) in this release.
-* GPU: 8x NVIDIA GB200 GPUs across 2 nodes (4 GPUs per node). Tensor/expert parallelism of 8 (`--tp_size 8 --moe_expert_parallel_size 8`) spans all 8 GPUs, and the model is served
diff -- tensorrt_llm/_torch/modules/fused_moe/routing.py
@@ -677,9 +677,16 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - docs: `examples/configs/curated/minimax-m3-throughput.yaml` modified +10/-1; `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +29/-12
  - runtime: `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +10/-3
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15857 - [TRTLLM-13458][feat] Support Minimax M3 NVFP4 checkpoint

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15857
- Status/date: merged / 2026-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `a0e65c6e00b0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +157/-7, 277 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +21/-0 (21 lines); hunks: -31,6 +31,7; -352,6 +353,24 @@ class MiniMaxM3MoE(nn.Module):; symbols: MiniMaxM3MoE, _get_experts_quant_config, __init__, touching `MiniMaxM3MoE, _get_experts_quant_config, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +21/-0 (21 lines); hunks: -31,6 +31,7; -352,6 +353,24 @@ class MiniMaxM3MoE(nn.Module):; symbols: MiniMaxM3MoE, _get_experts_quant_config, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -31,6 +31,7 @@
+from tensorrt_llm.models.modeling_utils import QuantConfig
@@ -352,6 +353,24 @@ class MiniMaxM3MoE(nn.Module):
+    @staticmethod
+    def _get_experts_quant_config(model_config: "ModelConfig", layer_idx: int) -> QuantConfig:
+        """Return the per-layer quant config for the routed experts.
+        For MIXED_PRECISION checkpoints (MXFP8 base + NVFP4 experts),
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +21/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15937 - [TRTLLM-13973][fix] Restrict MiniMax M3 dense SDPA backends

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15937
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `1ae65a64abf4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +24/-16, 68 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +24/-16 (40 lines); hunks: -27,6 +27,7; -46,6 +47,11; symbols: _dense_forward, touching `_dense_forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +24/-16 (40 lines); hunks: -27,6 +27,7; -46,6 +47,11; symbols: _dense_forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -27,6 +27,7 @@
+from torch.nn.attention import SDPBackend, sdpa_kernel
@@ -46,6 +47,11 @@
+# Dense layers use SDPA with non-contiguous Q/K/V and a bool attn_mask.
+# Limit backends to memory-efficient and math; cuDNN SDPA fails for this layout,
+# and flash SDPA does not accept attn_mask.
+_DENSE_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +24/-16
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15923 - [TRTLLM-13968][fix] Enable MiniMax M3 piecewise CUDA graphs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15923
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `211482b36ac9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +315/-94, 759 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +146/-45 (191 lines); hunks: -45,7 +45,13; -579,6 +585,49 @@ def forward(self, hidden_states: torch.Tensor):; symbols: forward, _extract_minimax_m3_attention_extra_attrs, minimax_m3_attn_custom_op_inplace, MiniMaxM3Attention, touching `forward, _extract_minimax_m3_attention_extra_attrs, minimax_m3_attn_custom_op_inplace`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +146/-45 (191 lines); hunks: -45,7 +45,13; -579,6 +585,49 @@ def forward(self, hidden_states: torch.Tensor):; symbols: forward, _extract_minimax_m3_attention_extra_attrs, minimax_m3_attn_custom_op_inplace, MiniMaxM3Attention
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -45,7 +45,13 @@
-from ..utils import ActivationType, AuxStreamType, EventType
+from ..utils import (
+    ActivationType,
+    AuxStreamType,
+    EventType,
+    get_model_extra_attrs,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +146/-45
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16291 - [TRTLLM-14019][feat] Add MiniMax-M3 MSA sparse attention backend [revised]

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16291
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `45d1c111ece2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 41 files, +2836/-961, 4519 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +36/-18 (54 lines); hunks: -35,7 +35,17; -911,11 +921,6 @@ def _dense_attention_core(; symbols: _dense_attention_core, _attention_core, _sparse_forward, touching `_dense_attention_core, _attention_core, _sparse_forward`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +36/-18 (54 lines); hunks: -35,7 +35,17; -911,11 +921,6 @@ def _dense_attention_core(; symbols: _dense_attention_core, _attention_core, _sparse_forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -35,7 +35,17 @@
-from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
+from ..attention_backend.interface import (
+    AttentionForwardArgs,
+    PositionalEmbeddingParams,
+    RopeParams,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +36/-18
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16468 - [TRTLLM-14255][fix] migrate MiniMax M3 to loader v2 for TP8 support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16468
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `9e96f8b9e72b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +304/-33, 414 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: _make_mapper, _duplicate_heads, test_mapper_registration_and_mx_fallback, test_load_weights_exposes_weight_mapper, touching `_make_mapper, _duplicate_heads, test_mapper_registration_and_mx_fallback`; `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: MiniMaxM3HfWeightMapper, __init__, _duplicate_kv_weights, touching `MiniMaxM3HfWeightMapper, __init__, _duplicate_kv_weights`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +32/-23 (55 lines); hunks: -52,6 +52,8; -1453,20 +1455,6 @@ def forward(; symbols: forward, MiniMaxM3ForCausalLM, __init__, load_weights, touching `forward, MiniMaxM3ForCausalLM, __init__`; `tests/unittest/_torch/models/test_minimax_m3.py` modified +17/-10 (27 lines); hunks: -32,6 +32,9; -43,7 +46,7; symbols: test_minimax_m3_routing_method_default_scale_is_identity, test_text_norm_weights_real_loader_smoke, __init__, touching `test_minimax_m3_routing_method_default_scale_is_identity, test_text_norm_weights_real_loader_smoke, __init__`.
- Code diff details:
  - `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` added +198/-0 (198 lines); hunks: -0,0 +1,198; symbols: _make_mapper, _duplicate_heads, test_mapper_registration_and_mx_fallback, test_load_weights_exposes_weight_mapper
  - `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: MiniMaxM3HfWeightMapper, __init__, _duplicate_kv_weights
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +32/-23 (55 lines); hunks: -52,6 +52,8; -1453,20 +1455,6 @@ def forward(; symbols: forward, MiniMaxM3ForCausalLM, __init__, load_weights
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +17/-10 (27 lines); hunks: -32,6 +32,9; -43,7 +46,7; symbols: test_minimax_m3_routing_method_default_scale_is_identity, test_text_norm_weights_real_loader_smoke, __init__
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py
@@ -0,0 +1,198 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py
@@ -0,0 +1,57 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -52,6 +52,8 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` added +198/-0; `tests/unittest/_torch/models/test_minimax_m3.py` modified +17/-10
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/minimaxm3_weight_mapper.py` added +57/-0; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +32/-23
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16696 - [None][feat] Add BaseMultimodalDummyInputsBuilder to minimaxm3_vl

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16696
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`; associated commits `b294868a77bf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +7/-2, 17 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +7/-2 (9 lines); hunks: -2071,9 +2071,14 @@ def get_minimax_m3_vl_input_processor_cls():; symbols: get_minimax_m3_vl_input_processor_cls, _Registered, touching `get_minimax_m3_vl_input_processor_cls, _Registered`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +7/-2 (9 lines); hunks: -2071,9 +2071,14 @@ def get_minimax_m3_vl_input_processor_cls():; symbols: get_minimax_m3_vl_input_processor_cls, _Registered
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py
@@ -2071,9 +2071,14 @@ def get_minimax_m3_vl_input_processor_cls():
-    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
+    from tensorrt_llm.inputs.registry import (
+        BaseMultimodalDummyInputsBuilder,
+        BaseMultimodalInputProcessor,
+    )
-    class _Registered(MiniMaxM3VLInputProcessor, BaseMultimodalInputProcessor):
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16017 - [TRTLLM-13969][feat] Support MiniMax M3 for Disaggregated Serving

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16017
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py`; associated commits `8041493cf6c6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 36 files, +4549/-1519, 7432 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` added +647/-0 (647 lines); hunks: -0,0 +1,647; symbols: _FakeKVCache, get_base_page_indices, _ShortFakeKVCache, test_minimax_cache_indices_support_block_count_limit, touching `_FakeKVCache, get_base_page_indices, _ShortFakeKVCache`; `tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py` modified +421/-216 (637 lines); hunks: -1,3 +1,20; -12,107 +29,124; symbols: IdentityMapper, IntactMapper, regions, map, touching `IdentityMapper, IntactMapper, regions`; `tensorrt_llm/_torch/disaggregation/native/peer.py` modified +200/-89 (289 lines); hunks: -1,19 +1,38; -57,6 +76,33 @@ def register(self, peer_name: str, peer_rank: int, peer_ri: R...; symbols: register, peer_extractor, get_pool_mapping, exists, touching `register, peer_extractor, get_pool_mapping`; `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` modified +20/-194 (214 lines); hunks: -347,164 +347,6 @@ NixlTransferStatus::~NixlTransferStatus() noexcept; -693,12 +535,8 @@ void NixlTransferAgent::registerMemory(RegisterDescs const&....
- Code diff details:
  - `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` added +647/-0 (647 lines); hunks: -0,0 +1,647; symbols: _FakeKVCache, get_base_page_indices, _ShortFakeKVCache, test_minimax_cache_indices_support_block_count_limit
  - `tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py` modified +421/-216 (637 lines); hunks: -1,3 +1,20; -12,107 +29,124; symbols: IdentityMapper, IntactMapper, regions, map
  - `tensorrt_llm/_torch/disaggregation/native/peer.py` modified +200/-89 (289 lines); hunks: -1,19 +1,38; -57,6 +76,33 @@ def register(self, peer_name: str, peer_rank: int, peer_ri: R...; symbols: register, peer_extractor, get_pool_mapping, exists
  - `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` modified +20/-194 (214 lines); hunks: -347,164 +347,6 @@ NixlTransferStatus::~NixlTransferStatus() noexcept; -693,12 +535,8 @@ void NixlTransferAgent::registerMemory(RegisterDescs const&...
  - `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py` modified +85/-109 (194 lines); hunks: -1,5 +1,17; -17,17 +29,13; symbols: __init__, _extra_buffers_per_layer, get_disagg_role_mapper_kinds, _compute_num_total_slots
- Key code excerpts:

```diff
diff -- tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py
@@ -0,0 +1,647 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py
@@ -1,3 +1,20 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/disaggregation/native/peer.py
@@ -1,19 +1,38 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` added +647/-0
  - runtime: `tensorrt_llm/_torch/disaggregation/native/mixers/attention/peer.py` modified +421/-216; `tensorrt_llm/_torch/disaggregation/native/peer.py` modified +200/-89; `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` modified +20/-194; `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py` modified +85/-109; `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +157/-30; `tensorrt_llm/_torch/disaggregation/native/transfer.py` modified +101/-76
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/executor/CMakeLists.txt`, `cpp/tests/unit_tests/executor/coalesceTest.cpp`, `cpp/tests/unit_tests/executor/transferAgentTest.cpp`, `tests/integration/test_lists/test-db/l0_a10.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16904 - [None][perf] Fuse index-q/index-k projections in MinimaxM3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16904
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `4b9012b965bb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +96/-105, 301 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +34/-74 (108 lines); hunks: -279,17 +279,11 @@ def test_get_moe_layer_ids_length_mismatch_raises():; -404,6 +398,7 @@ def test_minimax_m3_attention_dense_construction_matches_con...; symbols: test_get_moe_layer_ids_length_mismatch_raises, _make_attention_test_config, test_minimax_m3_attention_dense_construction_matches_config, test_minimax_m3_attention_apply_qk_norm_matches_reference, touching `test_get_moe_layer_ids_length_mismatch_raises, _make_attention_test_config, test_minimax_m3_attention_dense_construction_matches_config`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +62/-31 (93 lines); hunks: -52,7 +52,14; -64,7 +71,13; symbols: MiniMaxM3Attention, __init__, _sparse_forward, forward, touching `MiniMaxM3Attention, __init__, _sparse_forward`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +34/-74 (108 lines); hunks: -279,17 +279,11 @@ def test_get_moe_layer_ids_length_mismatch_raises():; -404,6 +398,7 @@ def test_minimax_m3_attention_dense_construction_matches_con...; symbols: test_get_moe_layer_ids_length_mismatch_raises, _make_attention_test_config, test_minimax_m3_attention_dense_construction_matches_config, test_minimax_m3_attention_apply_qk_norm_matches_reference
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +62/-31 (93 lines); hunks: -52,7 +52,14; -64,7 +71,13; symbols: MiniMaxM3Attention, __init__, _sparse_forward, forward
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -279,17 +279,11 @@ def test_get_moe_layer_ids_length_mismatch_raises():
-#  * Sparse index branch: ``index_q_proj`` is column-parallel and
-#    projects to ``num_index_heads * sparse_index_dim``;
-#    ``index_k_proj`` is **replicated** (tp_mode is None) and projects
-#    to **only** ``sparse_index_dim`` (single K per token, broadcast
-#    across all index heads for block-selection scoring) — this is the
-#    SGLang reference contract, confirmed by the M3 checkpoint shape
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -52,7 +52,14 @@
-from ..modules.linear import Linear, TensorParallelMode, copy_weight, load_weight_shard
+from ..modules.linear import (
+    Linear,
+    TensorParallelMode,
+    WeightMode,
+    WeightsLoadingConfig,
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +34/-74
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +62/-31
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16859 - [None][perf] Fuse MiniMax-M3 MoE routing

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16859
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `960530bc277b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +136/-21, 267 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +79/-0 (79 lines); hunks: -764,6 +764,85 @@ def test_minimax_m3_routing_method_default_scale_is_identit...; symbols: test_minimax_m3_routing_method_default_scale_is_identity, test_minimax_m3_fused_routing_matches_reference, test_minimax_m3_fused_routing_cuda_graph_replay_tracks_inputs, test_text_norm_weights_real_loader_smoke, touching `test_minimax_m3_routing_method_default_scale_is_identity, test_minimax_m3_fused_routing_matches_reference, test_minimax_m3_fused_routing_cuda_graph_replay_tracks_inputs`; `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +3/-3 (6 lines); hunks: -377,8 +377,8 @@ class Step3p7MoeRoutingMethod(MiniMaxM2MoeRoutingMethod):; -903,7 +903,7 @@ def forward(; symbols: Step3p7MoeRoutingMethod, forward, touching `Step3p7MoeRoutingMethod, forward`; `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +21/-0 (21 lines); hunks: -676,12 +676,33 @@ def __init__(; symbols: __init__, apply, touching `__init__, apply`; `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` modified +9/-1 (10 lines); hunks: -47,7 +47,7; -649,6 +649,14 @@ def _extract_routing_params(self) -> RoutingParams:; symbols: _extract_routing_params, touching `_extract_routing_params`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +79/-0 (79 lines); hunks: -764,6 +764,85 @@ def test_minimax_m3_routing_method_default_scale_is_identit...; symbols: test_minimax_m3_routing_method_default_scale_is_identity, test_minimax_m3_fused_routing_matches_reference, test_minimax_m3_fused_routing_cuda_graph_replay_tracks_inputs, test_text_norm_weights_real_loader_smoke
  - `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +3/-3 (6 lines); hunks: -377,8 +377,8 @@ class Step3p7MoeRoutingMethod(MiniMaxM2MoeRoutingMethod):; -903,7 +903,7 @@ def forward(; symbols: Step3p7MoeRoutingMethod, forward
  - `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +21/-0 (21 lines); hunks: -676,12 +676,33 @@ def __init__(; symbols: __init__, apply
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` modified +9/-1 (10 lines); hunks: -47,7 +47,7; -649,6 +649,14 @@ def _extract_routing_params(self) -> RoutingParams:; symbols: _extract_routing_params
  - `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu` modified +4/-4 (8 lines); hunks: -174,9 +174,9 @@ void Runner::run(void* routingLogits, void* routingBias, int...; -189,7 +189,7 @@ void Runner::run(void* routingLogits, void* routingBias, int...
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -764,6 +764,85 @@ def test_minimax_m3_routing_method_default_scale_is_identity():
+@pytest.mark.gpu
+@pytest.mark.skipif(not _has_cuda(), reason="fused MiniMax-M3 routing requires CUDA")
+@pytest.mark.parametrize("num_tokens", [1, 64, 8192])
+def test_minimax_m3_fused_routing_matches_reference(
+    num_tokens: int, monkeypatch: pytest.MonkeyPatch
+) -> None:
diff -- tensorrt_llm/_torch/models/modeling_step3p7.py
@@ -377,8 +377,8 @@ class Step3p7MoeRoutingMethod(MiniMaxM2MoeRoutingMethod):
-    feeds the bias pointer to the kernel. The MiniMax2 C++ routing path
-    hard-codes ``routeScale = 1.0f`` (see ``runner.cu``), so
+    feeds the bias pointer to the kernel. The generic MiniMax2 metadata does
+    not supply a route scale and therefore defaults to ``1.0f``, so
@@ -903,7 +903,7 @@ def forward(
-        # TRTLLMGen MiniMax2 kernel hard-codes routeScale=1.0, so apply
diff -- tensorrt_llm/_torch/modules/fused_moe/routing.py
@@ -676,12 +676,33 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +79/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +3/-3; `tensorrt_llm/_torch/modules/fused_moe/routing.py` modified +21/-0; `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` modified +9/-1; `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/modules/moe/test_moe_module.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17360 - [None][feat] Default MiniMax M2 to KV cache manager V2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17360
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; associated commits `9deec3760619`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-1, 29 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +9/-1 (10 lines); hunks: -13,12 +13,15; -399,6 +402,11 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, get_model_defaults, __init__, touching `forward, MiniMaxM2ForCausalLM, get_model_defaults`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +9/-1 (10 lines); hunks: -13,12 +13,15; -399,6 +402,11 @@ def forward(; symbols: forward, MiniMaxM2ForCausalLM, get_model_defaults, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm2.py
@@ -13,12 +13,15 @@
-from typing import Dict, List, Optional
+from typing import TYPE_CHECKING, Dict, List, Optional
+if TYPE_CHECKING:
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -399,6 +402,11 @@ def forward(
+    @classmethod
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm2.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_minimaxm2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17565 - [https://nvbugs/6373561][fix] Fix minimax M3 E2E test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17565
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `24be2c11b88c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +100/-9, 177 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +38/-0 (38 lines); hunks: -37,9 +37,11; -53,6 +55,7; symbols: test_validate_sparse_attention_runtime_config_rejects_wrong_backend, test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_model_init_validates_sparse_attention_runtime_config, _make_text_config, touching `test_validate_sparse_attention_runtime_config_rejects_wrong_backend, test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_model_init_validates_sparse_attention_runtime_config`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +20/-0 (20 lines); hunks: -161,6 +161,25 @@ def get_text_model_config(; -1860,6 +1879,7 @@ class MiniMaxM3Model(DecoderModel):; symbols: get_text_model_config, _validate_sparse_attention_runtime_config, get_sparse_layer_ids, MiniMaxM3Model, touching `get_text_model_config, _validate_sparse_attention_runtime_config, get_sparse_layer_ids`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +38/-0 (38 lines); hunks: -37,9 +37,11; -53,6 +55,7; symbols: test_validate_sparse_attention_runtime_config_rejects_wrong_backend, test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_model_init_validates_sparse_attention_runtime_config, _make_text_config
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +20/-0 (20 lines); hunks: -161,6 +161,25 @@ def get_text_model_config(; -1860,6 +1879,7 @@ class MiniMaxM3Model(DecoderModel):; symbols: get_text_model_config, _validate_sparse_attention_runtime_config, get_sparse_layer_ids, MiniMaxM3Model
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -37,9 +37,11 @@
+    MiniMaxM3Model,
+    _validate_sparse_attention_runtime_config,
@@ -53,6 +55,7 @@
+from tensorrt_llm.llmapi import MiniMaxM3SparseAttentionConfig, RocketSparseAttentionConfig
@@ -65,6 +68,41 @@
+@pytest.mark.parametrize(
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -161,6 +161,25 @@ def get_text_model_config(
+def _validate_sparse_attention_runtime_config(
+    model_config: "ModelConfig[PretrainedConfig]",
+) -> None:
+    """Require the runtime backend that owns M3's metadata and index cache.
+    Both dense and sparse M3 layers use that cache manager, independent of
+    checkpoint precision or GPU architecture. Backend-specific constraints,
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +38/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +20/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/test_e2e.py`, `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/models/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17236 - [None][perf] Optimize MiniMax-M3 MSA block selection

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17236
- Status/date: merged / 2026-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu`, `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h`, `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp`; associated commits `ca728f8562e2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +1263/-30, 1367 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` added +567/-0 (567 lines); hunks: -0,0 +1,567; `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp` added +103/-0 (103 lines); hunks: -0,0 +1,103; `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h` added +36/-0 (36 lines); hunks: -0,0 +1,36.
- Code diff details:
  - `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` added +567/-0 (567 lines); hunks: -0,0 +1,567
  - `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp` added +103/-0 (103 lines); hunks: -0,0 +1,103
  - `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h` added +36/-0 (36 lines); hunks: -0,0 +1,36
- Key code excerpts:

```diff
diff -- cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu
@@ -0,0 +1,567 @@
+/*
+ * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+ * SPDX-License-Identifier: Apache-2.0
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
diff -- cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp
@@ -0,0 +1,103 @@
+/*
+ * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+ * SPDX-License-Identifier: Apache-2.0
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
diff -- cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h
@@ -0,0 +1,36 @@
```

- Extracted files (not manually reviewed):
  - runtime: `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` added +567/-0; `cpp/tensorrt_llm/thop/minimaxM3SelectBlocksOp.cpp` added +103/-0; `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.h` added +36/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_backend.py`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_selector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17318 - [None][perf] Use FP8 MiniMax-M3 MSA indexer QK

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17318
- Status/date: merged / 2026-08-24
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.cu`, `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.h`, `cpp/tensorrt_llm/thop/minimaxM3Fp8IndexerOp.cpp`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py` and 6 files; associated commits `5ee95a10372b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1202/-38, 1575 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +65/-1 (66 lines); hunks: -982,6 +982,66 @@ def _fused_qk_norm_rope(; -1403,7 +1463,7 @@ def _msa_attention_core(; symbols: _fused_qk_norm_rope, _fused_fp8_index_qk_norm_rope, _expect_fused_qk_norm_rope, _msa_attention_core, touching `_fused_qk_norm_rope, _fused_fp8_index_qk_norm_rope, _expect_fused_qk_norm_rope`; `tests/unittest/_torch/models/test_minimax_m3.py` modified +26/-0 (26 lines); hunks: -31,6 +31,7; -795,6 +796,31 @@ def test_minimax_m3_fused_qk_norm_rope_index_matches_separa...; symbols: test_minimax_m3_fused_qk_norm_rope_index_matches_separate, test_minimax_m3_fp8_indexer_rejects_different_qk_norm_epsilons, test_minimax_m3_fused_qk_norm_rope_fallbacks, touching `test_minimax_m3_fused_qk_norm_rope_index_matches_separate, test_minimax_m3_fp8_indexer_rejects_different_qk_norm_epsilons, test_minimax_m3_fused_qk_norm_rope_fallbacks`; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: _assert_fp8_close, _reference, _strided_cache, _run, touching `_assert_fp8_close, _reference, _strided_cache`; `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.cu` added +190/-0 (190 lines); hunks: -0,0 +1,190.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +65/-1 (66 lines); hunks: -982,6 +982,66 @@ def _fused_qk_norm_rope(; -1403,7 +1463,7 @@ def _msa_attention_core(; symbols: _fused_qk_norm_rope, _fused_fp8_index_qk_norm_rope, _expect_fused_qk_norm_rope, _msa_attention_core
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +26/-0 (26 lines); hunks: -31,6 +31,7; -795,6 +796,31 @@ def test_minimax_m3_fused_qk_norm_rope_index_matches_separa...; symbols: test_minimax_m3_fused_qk_norm_rope_index_matches_separate, test_minimax_m3_fp8_indexer_rejects_different_qk_norm_epsilons, test_minimax_m3_fused_qk_norm_rope_fallbacks
  - `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` added +349/-0 (349 lines); hunks: -0,0 +1,349; symbols: _assert_fp8_close, _reference, _strided_cache, _run
  - `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.cu` added +190/-0 (190 lines); hunks: -0,0 +1,190
  - `cpp/tensorrt_llm/thop/minimaxM3Fp8IndexerOp.cpp` added +137/-0 (137 lines); hunks: -0,0 +1,137
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -982,6 +982,66 @@ def _fused_qk_norm_rope(
+    def _fused_fp8_index_qk_norm_rope(
+        self,
+        idx_qk: torch.Tensor,
+        position_ids: Optional[torch.Tensor],
+        attn_metadata: AttentionMetadata,
+    ) -> Optional[torch.Tensor]:
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -31,6 +31,7 @@
+from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
@@ -795,6 +796,31 @@ def test_minimax_m3_fused_qk_norm_rope_index_matches_separate():
+@pytest.mark.cpu_only
+def test_minimax_m3_fp8_indexer_rejects_different_qk_norm_epsilons() -> None:
+    """The fused kernel has one epsilon, so Q/K norms must agree."""
+    attn = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
diff -- tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py
@@ -0,0 +1,349 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +65/-1; `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.cu` added +190/-0; `cpp/tensorrt_llm/thop/minimaxM3Fp8IndexerOp.cpp` added +137/-0; `cpp/tensorrt_llm/kernels/minimaxM3Fp8IndexerKernel.h` added +55/-0
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +26/-0; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` added +349/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_backend.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17842 - [None][feat] Add the ported MiniMax-M3 decode kernels ahead of their wiring

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17842
- Status/date: merged / 2026-08-26
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py`, `tests/microbenchmarks/minimax_m3_index_decode_score.py`; associated commits `0a373efd9f2b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +2781/-0, 2810 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py` added +429/-0 (429 lines); hunks: -0,0 +1,429; symbols: _fp8_to_f16_mma_fragments, IndexDecodeScoreKernel, __init__, _swizzle_elems, touching `_fp8_to_f16_mma_fragments, IndexDecodeScoreKernel, __init__`; `tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py` added +408/-0 (408 lines); hunks: -0,0 +1,408; symbols: _flat_page_table, _runner, _reference_decode_index_score, _make_inputs, touching `_flat_page_table, _runner, _reference_decode_index_score`; `tests/microbenchmarks/minimax_m3_index_decode_score.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _flat_page_table, _time_us, main, run_cutedsl, touching `_flat_page_table, _time_us, main`.
- Code diff details:
  - `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py` added +429/-0 (429 lines); hunks: -0,0 +1,429; symbols: _fp8_to_f16_mma_fragments, IndexDecodeScoreKernel, __init__, _swizzle_elems
  - `tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py` added +408/-0 (408 lines); hunks: -0,0 +1,408; symbols: _flat_page_table, _runner, _reference_decode_index_score, _make_inputs
  - `tests/microbenchmarks/minimax_m3_index_decode_score.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _flat_page_table, _time_us, main, run_cutedsl
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py
@@ -0,0 +1,429 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-License-Identifier: Apache-2.0
+# Vendored from vLLM (Apache-2.0):
+# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
+"""CuTe DSL MiniMax-M3 index decode block-scoring kernel (Blackwell SM100).
diff -- tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py
@@ -0,0 +1,408 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-License-Identifier: Apache-2.0
+# PyTorch oracle vendored from vLLM (Apache-2.0), _reference_decode_index_score:
+# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L188
+"""Correctness tests for the CuTe DSL MiniMax-M3 indexer decode scorer.
diff -- tests/microbenchmarks/minimax_m3_index_decode_score.py
@@ -0,0 +1,114 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/minimax_m3_index_decode_score.py` added +429/-0
  - tests: `tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py` added +408/-0; `tests/microbenchmarks/minimax_m3_index_decode_score.py` added +114/-0
- Risk and verification: The diff ships test coverage in `tests/microbenchmarks/minimax_m3_index_decode_score.py`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_dense_decode.py`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_index_decode_score.py`, `tests/unittest/_torch/attention/sparse/test_minimax_m3_sparse_attn_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18154 - [None][perf] Histogram top-k for MiniMax-M3 block selector

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18154
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu`; associated commits `682fa40d2711`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +702/-49, 848 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` modified +464/-48 (512 lines); hunks: -1,5 +1,6; -15,13 +16,20.
- Code diff details:
  - `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` modified +464/-48 (512 lines); hunks: -1,5 +1,6; -15,13 +16,20
- Key code excerpts:

```diff
diff -- cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu
@@ -1,5 +1,6 @@
+ * SPDX-FileCopyrightText: Copyright (c) 2021 NAVER Corp. Authored by CLOVA.
@@ -15,13 +16,20 @@
+// The histogram select in the second half of this file is derived from
+// TRT-LLM's own kernels/indexerTopK.cu (topKPerRowJob and its helpers), which
+// carries the NAVER/CLOVA copyright above; see the comment on that section for
+// what this port changes.
```

- Extracted files (not manually reviewed):
  - runtime: `cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu` modified +464/-48
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_selector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18416 - [None][test] Add MiniMax-M3 disaggregated perf recipes to QA multi-node list

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18416
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml`; associated commits `36808bd8123d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +805/-0, 816 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +201/-0 (201 lines); hunks: -0,0 +1,201; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml` added +200/-0 (200 lines); hunks: -0,0 +1,200; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0 (199 lines); hunks: -0,0 +1,199; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0 (199 lines); hunks: -0,0 +1,199.
- Code diff details:
  - `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +201/-0 (201 lines); hunks: -0,0 +1,201
  - `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml` added +200/-0 (200 lines); hunks: -0,0 +1,200
  - `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0 (199 lines); hunks: -0,0 +1,199
  - `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0 (199 lines); hunks: -0,0 +1,199
- Key code excerpts:

```diff
diff -- tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml
@@ -0,0 +1,201 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml
@@ -0,0 +1,200 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml
@@ -0,0 +1,199 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tp4_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +201/-0; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml` added +200/-0; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0; `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml` added +199/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep16_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con256_ctx4_tp4_gen1_dep8_eplb0_eagle3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_minimax-m3-fp4_8k1k_con30_ctx1_tep2_gen2_tp4_eplb0_eagle3_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18466 - [None][fix] Support MiniMax-M3 vision meta init

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18466
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`, `tests/unittest/_torch/models/test_minimax_m3_vl.py`; associated commits `6507185b02a1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +44/-3, 83 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +25/-3 (28 lines); hunks: -1181,6 +1181,22 @@ def forward(self, hidden_states: torch.Tensor) -> torch.T...; -1192,9 +1208,13 @@ def __init__(self, config: CLIPVisionConfig, dtype: torch...; symbols: forward, _MiniMaxVLCheckpointLayerNorm, reset_parameters, MiniMaxVLEncoderLayer, touching `forward, _MiniMaxVLCheckpointLayerNorm, reset_parameters`; `tests/unittest/_torch/models/test_minimax_m3_vl.py` modified +19/-0 (19 lines); hunks: -47,6 +47,7; -232,6 +233,24 @@ def test_minimax_vl_vision_model_param_shapes_match_checkpo...; symbols: test_minimax_vl_vision_model_param_shapes_match_checkpoint, test_minimax_vl_vision_model_supports_meta_init, touching `test_minimax_vl_vision_model_param_shapes_match_checkpoint, test_minimax_vl_vision_model_supports_meta_init`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +25/-3 (28 lines); hunks: -1181,6 +1181,22 @@ def forward(self, hidden_states: torch.Tensor) -> torch.T...; -1192,9 +1208,13 @@ def __init__(self, config: CLIPVisionConfig, dtype: torch...; symbols: forward, _MiniMaxVLCheckpointLayerNorm, reset_parameters, MiniMaxVLEncoderLayer
  - `tests/unittest/_torch/models/test_minimax_m3_vl.py` modified +19/-0 (19 lines); hunks: -47,6 +47,7; -232,6 +233,24 @@ def test_minimax_vl_vision_model_param_shapes_match_checkpo...; symbols: test_minimax_vl_vision_model_param_shapes_match_checkpoint, test_minimax_vl_vision_model_supports_meta_init
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py
@@ -1181,6 +1181,22 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+class _MiniMaxVLCheckpointLayerNorm(nn.LayerNorm):
+    """LayerNorm whose checkpoint-backed affine tensors support meta init.
+    ``MetaInitMode`` redirects empty parameter allocations to ``meta`` but
+    rejects the ``fill_`` calls made by ``LayerNorm.reset_parameters``. Every
+    affine tensor in the M3 vision tower is loaded from the checkpoint, so its
+    reset can be skipped only while the parameter is meta. Normal CPU and CUDA
diff -- tests/unittest/_torch/models/test_minimax_m3_vl.py
@@ -47,6 +47,7 @@
+from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
@@ -232,6 +233,24 @@ def test_minimax_vl_vision_model_param_shapes_match_checkpoint():
+def test_minimax_vl_vision_model_supports_meta_init():
+    config_dict = _make_vision_config_dict()
+    config_dict["num_hidden_layers"] = 2
+    cfg = CLIPVisionConfig.from_dict_or_obj(config_dict)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py` modified +25/-3
  - tests: `tests/unittest/_torch/models/test_minimax_m3_vl.py` modified +19/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/test_minimax_m3_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17238 - [None][perf] Optimize MiniMax-M3 MXFP8 GEMMs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17238
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `d147336b7a99`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +1727/-90, 2083 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +10/-2 (12 lines); hunks: -86,6 +86,7; -200,7 +201,9 @@ def get_sparse_layer_ids(text_config: PretrainedConfig) -> T...; symbols: get_sparse_layer_ids, get_sparse_disable_index_value_layer_ids, _forward_attention_core, __init__, touching `get_sparse_layer_ids, get_sparse_disable_index_value_layer_ids, _forward_attention_core`; `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +14/-0 (14 lines); hunks: -101,6 +101,20 @@ If you don't have access to the source code locally, you ca....
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +10/-2 (12 lines); hunks: -86,6 +86,7; -200,7 +201,9 @@ def get_sparse_layer_ids(text_config: PretrainedConfig) -> T...; symbols: get_sparse_layer_ids, get_sparse_disable_index_value_layer_ids, _forward_attention_core, __init__
  - `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +14/-0 (14 lines); hunks: -101,6 +101,20 @@ If you don't have access to the source code locally, you ca...
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -86,6 +86,7 @@
@@ -200,7 +201,9 @@ def get_sparse_layer_ids(text_config: PretrainedConfig) -> Tuple[List[int], List
-def get_sparse_disable_index_value_layer_ids(text_config: PretrainedConfig) -> List[int]:
+def get_sparse_disable_index_value_layer_ids(
+    text_config: PretrainedConfig,
+) -> List[int]:
@@ -1402,7 +1405,8 @@ def _forward_attention_core(
diff -- docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md
@@ -101,6 +101,20 @@ If you don't have access to the source code locally, you can manually create the
+For MXFP8 checkpoints, TensorRT LLM selects the GEMM backend automatically.
+`TRTLLM_MXFP8_GEMM_BACKEND` is an advanced override for debugging and
+performance experiments:
+* `trtllm` uses the native TensorRT LLM GEMM for eager execution and CUDA graphs.
+* `flashinfer` forces FlashInfer for eligible captured decode CUDA graphs; eager,
+  context/prefill, and piecewise CUDA-graph execution remain on the native GEMM.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +10/-2
  - docs: `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md` modified +14/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_b300.yml`, `tests/unittest/_torch/executor/test_pytorch_model_engine_warmup.py`, `tests/unittest/_torch/modules/test_mxfp8_linear.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17374 - [None][fix] Use HND mapping for MiniMax-M3 MSA KV cache

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17374
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py`; associated commits `8da2ed1961d7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +132/-34, 361 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` modified +105/-27 (132 lines); hunks: -26,6 +26,8; -93,13 +95,35 @@ def test_v2_disagg_role_mapper_kind_defaults() -> None:; symbols: test_v2_disagg_role_mapper_kind_defaults, test_minimax_disagg_role_mapper_kinds, fake_base_init, test_minimax_kv_pool_mapping_offset_ignores_layer_grouping_order, touching `test_v2_disagg_role_mapper_kind_defaults, test_minimax_disagg_role_mapper_kinds, fake_base_init`; `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py` modified +22/-4 (26 lines); hunks: -155,6 +155,8 @@ class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):; -171,6 +173,8 @@ def __init__(; symbols: MiniMaxM3KVCacheManagerV2, __init__, _extra_buffers_per_layer, get_disagg_role_mapper_kinds, touching `MiniMaxM3KVCacheManagerV2, __init__, _extra_buffers_per_layer`; `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ class MapperKind(IntEnum):; -70,6 +70,7 @@ class MapperKind(IntEnum):; symbols: MapperKind, exists, touching `MapperKind, exists`; `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +2/-1 (3 lines); hunks: -2111,7 +2111,8 @@ def get_disagg_role_mapper_kinds(self) -> dict[DataRole, M...; symbols: get_disagg_role_mapper_kinds, touching `get_disagg_role_mapper_kinds`.
- Code diff details:
  - `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` modified +105/-27 (132 lines); hunks: -26,6 +26,8; -93,13 +95,35 @@ def test_v2_disagg_role_mapper_kind_defaults() -> None:; symbols: test_v2_disagg_role_mapper_kind_defaults, test_minimax_disagg_role_mapper_kinds, fake_base_init, test_minimax_kv_pool_mapping_offset_ignores_layer_grouping_order
  - `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py` modified +22/-4 (26 lines); hunks: -155,6 +155,8 @@ class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):; -171,6 +173,8 @@ def __init__(; symbols: MiniMaxM3KVCacheManagerV2, __init__, _extra_buffers_per_layer, get_disagg_role_mapper_kinds
  - `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +2/-1 (3 lines); hunks: -40,7 +40,7 @@ class MapperKind(IntEnum):; -70,6 +70,7 @@ class MapperKind(IntEnum):; symbols: MapperKind, exists
  - `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +2/-1 (3 lines); hunks: -2111,7 +2111,8 @@ def get_disagg_role_mapper_kinds(self) -> dict[DataRole, M...; symbols: get_disagg_role_mapper_kinds
- Key code excerpts:

```diff
diff -- tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py
@@ -26,6 +26,8 @@
+from types import SimpleNamespace
+from typing import Any
@@ -93,13 +95,35 @@ def test_v2_disagg_role_mapper_kind_defaults() -> None:
-def test_minimax_disagg_role_mapper_kinds() -> None:
-    manager = object.__new__(MiniMaxM3KVCacheManagerV2)
+@pytest.mark.parametrize(
diff -- tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py
@@ -155,6 +155,8 @@ class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):
+    _main_kv_mapper_kind = MapperKind.NHD
@@ -171,6 +173,8 @@ def __init__(
+        implementation = getattr(sparse_attention_config, "implementation", "triton")
+        self._main_kv_mapper_kind = MapperKind.HND if implementation == "msa" else MapperKind.NHD
@@ -249,12 +253,16 @@ def _extra_buffers_per_layer(self, *, tokens_per_block):
-        """Declare MiniMax M3's token-major K/V and replicated index-K."""
diff -- tensorrt_llm/_torch/disaggregation/resource/page.py
@@ -40,7 +40,7 @@ class MapperKind(IntEnum):
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py` modified +105/-27
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/minimax_m3/cache_manager.py` modified +22/-4; `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +2/-1; `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/disaggregated/test_minimax_m3_kv_transfer.py`, `tests/unittest/disaggregated/test_pool_matching.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18611 - [None][perf] Wire in custom decode kernels for MinimaxM3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18611
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/__init__.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/common.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/__init__.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` and 16 files; associated commits `b7681d7f96ef`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 25 files, +2389/-794, 4024 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +595/-366 (961 lines); hunks: -5,7 +5,8; -18,37 +19,43; symbols: _cache_device, _worst_case_proxy_max_k_tiles, _MsaGraphSafePlan, MsaDecodeSpan, touching `_cache_device, _worst_case_proxy_max_k_tiles, _MsaGraphSafePlan`; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +673/-25 (698 lines); hunks: -9,6 +9,7; -19,15 +20,19; symbols: test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_import_preserves_cute_compile_option_selection, test_msa_metadata_rejects_undersized_max_score_buffer, _RecordingBuffers, touching `test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_import_preserves_cute_compile_option_selection, test_msa_metadata_rejects_undersized_max_score_buffer`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py` modified +172/-33 (205 lines); hunks: -18,7 +18,7; -155,6 +155,33 @@ def _proxy_max_score(; symbols: _proxy_max_score, _combined_topk_table, _group_max_reduce, select_blocks, touching `_proxy_max_score, _combined_topk_table, _group_max_reduce`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` renamed +130/-21 (151 lines); hunks: -3,9 +3,9; -21,7 +21,7; symbols: _counter_size, _device_index, _counter_buffer, _workspace, touching `_counter_size, _device_index, _counter_buffer`.
- Code diff details:
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +595/-366 (961 lines); hunks: -5,7 +5,8; -18,37 +19,43; symbols: _cache_device, _worst_case_proxy_max_k_tiles, _MsaGraphSafePlan, MsaDecodeSpan
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +673/-25 (698 lines); hunks: -9,6 +9,7; -19,15 +20,19; symbols: test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_import_preserves_cute_compile_option_selection, test_msa_metadata_rejects_undersized_max_score_buffer, _RecordingBuffers
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py` modified +172/-33 (205 lines); hunks: -18,7 +18,7; -155,6 +155,33 @@ def _proxy_max_score(; symbols: _proxy_max_score, _combined_topk_table, _group_max_reduce, select_blocks
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` renamed +130/-21 (151 lines); hunks: -3,9 +3,9; -21,7 +21,7; symbols: _counter_size, _device_index, _counter_buffer, _workspace
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +64/-34 (98 lines); hunks: -25,7 +25,7; -308,24 +308,15 @@ def get_index_v_buffer(self, layer_idx: int) -> Optional[t...; symbols: get_index_v_buffer, has_index_value, get_buffers, _kv_slot_geometry
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py
@@ -5,7 +5,8 @@
-  * The main sparse GQA runs through the registered MsaSparseGqaFmha.
+  * The main attention runs through the registered MsaPrefillFmha for context
+    rows and MsaDecodeFmha for generation rows.
@@ -18,37 +19,43 @@
-module scope. This is cycle-free because the fmha registry defers its
-MsaSparseGqaFmha import (see fmha/registry.py), so trtllm's import chain does
diff -- tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py
@@ -9,6 +9,7 @@
+import weakref
@@ -19,15 +20,19 @@
-from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import msa_paged_kv
+from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
+    MSA_REQUIRED_TOPK,
+    msa_paged_kv,
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py
@@ -18,7 +18,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +595/-366; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py` modified +172/-33; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` renamed +130/-21; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +64/-34; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` renamed +54/-2; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/paged_cache.py` added +49/-0
  - tests: `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +673/-25
- Risk and verification: The diff ships test coverage in `tests/microbenchmarks/minimax_m3_index_decode_score.py`, `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_dense_decode.py`, `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_index_decode_score.py`, `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_msa_selector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18733 - [TRTLLM-16194][feat] Add MiniMax H3 support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18733
- Status/date: merged / 2026-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml`, `examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml`, `examples/visual_gen/models/minimax_h3.py`, `tensorrt_llm/_torch/visual_gen/models/minimax_h3/__init__.py`, `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` and 11 files; associated commits `61154e9f3f8d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 28 files, +5367/-76, 5899 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py` added +877/-0 (877 lines); hunks: -0,0 +1,877; symbols: MiniMaxH3TransformerOutput, MiniMaxH3StaticContext, sequence_length, apply_minimax_h3_rotary_emb, touching `MiniMaxH3TransformerOutput, MiniMaxH3StaticContext, sequence_length`; `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py` added +824/-0 (824 lines); hunks: -0,0 +1,824; symbols: _component_skipped, _load_keyframe_image, _check_denoise_step, MiniMaxH3Pipeline, touching `_component_skipped, _load_keyframe_image, _check_denoise_step`; `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` added +418/-0 (418 lines); hunks: -0,0 +1,418; symbols: MiniMaxH3PackedSequence, resolve_canvas_size, align_num_frames, video_latent_num_frames, touching `MiniMaxH3PackedSequence, resolve_canvas_size, align_num_frames`; `examples/visual_gen/models/minimax_h3.py` added +68/-0 (68 lines); hunks: -0,0 +1,68; symbols: main, touching `main`.
- Code diff details:
  - `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py` added +877/-0 (877 lines); hunks: -0,0 +1,877; symbols: MiniMaxH3TransformerOutput, MiniMaxH3StaticContext, sequence_length, apply_minimax_h3_rotary_emb
  - `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py` added +824/-0 (824 lines); hunks: -0,0 +1,824; symbols: _component_skipped, _load_keyframe_image, _check_denoise_step, MiniMaxH3Pipeline
  - `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` added +418/-0 (418 lines); hunks: -0,0 +1,418; symbols: MiniMaxH3PackedSequence, resolve_canvas_size, align_num_frames, video_latent_num_frames
  - `examples/visual_gen/models/minimax_h3.py` added +68/-0 (68 lines); hunks: -0,0 +1,68; symbols: main
  - `tensorrt_llm/_torch/visual_gen/models/minimax_h3/__init__.py` added +22/-0 (22 lines); hunks: -0,0 +1,22
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py
@@ -0,0 +1,877 @@
+# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
diff -- tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py
@@ -0,0 +1,824 @@
+# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
diff -- tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py
@@ -0,0 +1,418 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py` added +877/-0; `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py` added +824/-0; `tensorrt_llm/_torch/visual_gen/models/minimax_h3/packing.py` added +418/-0; `tensorrt_llm/_torch/visual_gen/models/minimax_h3/__init__.py` added +22/-0; `tensorrt_llm/_torch/visual_gen/models/__init__.py` modified +2/-0
  - docs: `examples/visual_gen/models/minimax_h3.py` added +68/-0; `examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml` added +20/-0; `examples/visual_gen/configs/minimax-h3-fp8-blockwise-1gpu.yaml` added +20/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/examples/visual_gen/test_minimax_h3_e2e.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/unittest/_torch/visual_gen/test_cosmos3_edge.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18872 - [TRTLLM-14093][feat] Eagle3 support for MiniMax-M3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18872
- Status/date: merged / 2026-09-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_metadata.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_shared_draft_layers.py` and 6 files; associated commits `a8ac7e5bccb9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +1101/-149, 1777 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +59/-47 (106 lines); hunks: -64,16 +64,12; -1211,7 +1207,14 @@ def _sdpa_dense_attention_core(; symbols: _sdpa_dense_attention_core, forward, touching `_sdpa_dense_attention_core, forward`; `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +2/-0 (2 lines); hunks: -184,6 +184,8 @@ def test_load_weights_accepts_base_mapper_without_params_map...; symbols: test_load_weights_accepts_base_mapper_without_params_map, touching `test_load_weights_accepts_base_mapper_without_params_map`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +220/-23 (243 lines); hunks: -17,6 +17,11; -134,6 +139,14 @@ class MiniMaxM3MsaSparseAttentionMetadata(TrtllmAttentionMe...; symbols: MiniMaxM3MsaSparseAttentionMetadata, __post_init__, as_pinned_int32, touching `MiniMaxM3MsaSparseAttentionMetadata, __post_init__, as_pinned_int32`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +206/-9 (215 lines); hunks: -20,7 +20,8; -39,6 +40,10; symbols: set_index_v, shared_draft_layer_count, derive_shared_draft_layout, extend_attention_op_pools_for_shared_draft_layers, touching `set_index_v, shared_draft_layer_count, derive_shared_draft_layout`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +59/-47 (106 lines); hunks: -64,16 +64,12; -1211,7 +1207,14 @@ def _sdpa_dense_attention_core(; symbols: _sdpa_dense_attention_core, forward
  - `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +2/-0 (2 lines); hunks: -184,6 +184,8 @@ def test_load_weights_accepts_base_mapper_without_params_map...; symbols: test_load_weights_accepts_base_mapper_without_params_map
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +220/-23 (243 lines); hunks: -17,6 +17,11; -134,6 +139,14 @@ class MiniMaxM3MsaSparseAttentionMetadata(TrtllmAttentionMe...; symbols: MiniMaxM3MsaSparseAttentionMetadata, __post_init__, as_pinned_int32
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +206/-9 (215 lines); hunks: -20,7 +20,8; -39,6 +40,10; symbols: set_index_v, shared_draft_layer_count, derive_shared_draft_layout, extend_attention_op_pools_for_shared_draft_layers
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +197/-1 (198 lines); hunks: -192,9 +192,12 @@ def test_msa_metadata_rejects_undersized_max_score_buffer():; -417,6 +420,31 @@ def test_msa_paged_hnd_input_materializes_unaligned_outer_s...; symbols: test_msa_metadata_rejects_undersized_max_score_buffer, test_msa_paged_hnd_input_materializes_unaligned_outer_stride, test_per_token_valid_blocks_multi_token_decode, test_msa_index_k_uses_hnd_cache_view_and_writer
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -64,16 +64,12 @@
+from ..speculative import SpecMetadata
-from .modeling_utils import (
-    DecoderModel,
-    DecoderModelForCausalLM,
-    ModelConfig,
-    filter_weights,
diff -- tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py
@@ -184,6 +184,8 @@ def test_load_weights_accepts_base_mapper_without_params_map() -> None:
+    # Set by SpecDecOneEngineForCausalLM.__init__, which this test bypasses.
+    model.spec_config = None
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py
@@ -17,6 +17,11 @@
+  * With Eagle3 a decode row has 1 + draft_len query tokens. Slots and
+    valid-block counts are per token, and the decode kernels take that uniform
+    query length from msa_decode_span. on_update_kv_lens re-derives the
+    per-request lengths, the slots and the counts after the overlap scheduler
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +59/-47; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +220/-23; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +206/-9; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_metadata.py` modified +107/-65
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +2/-0; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +197/-1; `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_shared_draft_layers.py` added +175/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18614 - [None][perf] Fuse MiniMax-M3 MSA per-layer KV-cache writes into one kernel

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18614
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`; associated commits `86669b1a3b84`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +399/-8, 488 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +16/-2 (18 lines); hunks: -1398,20 +1398,34 @@ def _msa_attention_core(; symbols: _msa_attention_core, _sparse_forward, touching `_msa_attention_core, _sparse_forward`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _fused_paged_scatter_kernel, _row_stride_if_fusable, fused_write_layer_caches, touching `_fused_paged_scatter_kernel, _row_stride_if_fusable, fused_write_layer_caches`; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +143/-2 (145 lines); hunks: -4,8 +4,10; -20,10 +22,16; symbols: test_on_update_kv_lens_is_a_noop_without_speculative_decoding, test_fused_scatter_matches_reference, test_msa_attention_core_owns_the_cache_write, FakeBackend, touching `test_on_update_kv_lens_is_a_noop_without_speculative_decoding, test_fused_scatter_matches_reference, test_msa_attention_core_owns_the_cache_write`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +62/-3 (65 lines); hunks: -1208,20 +1208,74 @@ def support_fused_rope(cls) -> bool:; -1254,13 +1308,18 @@ def run_indexer(; symbols: support_fused_rope, write_layer_caches, run_indexer, touching `support_fused_rope, write_layer_caches, run_indexer`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +16/-2 (18 lines); hunks: -1398,20 +1398,34 @@ def _msa_attention_core(; symbols: _msa_attention_core, _sparse_forward
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py` added +173/-0 (173 lines); hunks: -0,0 +1,173; symbols: _fused_paged_scatter_kernel, _row_stride_if_fusable, fused_write_layer_caches
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +143/-2 (145 lines); hunks: -4,8 +4,10; -20,10 +22,16; symbols: test_on_update_kv_lens_is_a_noop_without_speculative_decoding, test_fused_scatter_matches_reference, test_msa_attention_core_owns_the_cache_write, FakeBackend
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +62/-3 (65 lines); hunks: -1208,20 +1208,74 @@ def support_fused_rope(cls) -> bool:; -1254,13 +1308,18 @@ def run_indexer(; symbols: support_fused_rope, write_layer_caches, run_indexer
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` modified +5/-1 (6 lines); hunks: -140,7 +140,11 @@ def write_msa_phase_kv(; symbols: write_msa_phase_kv
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -1398,20 +1398,34 @@ def _msa_attention_core(
+        This layer owns the cache write: write_layer_caches stores the
+        new-token K/V (and, on the bf16 indexer path, index-K) in one launch
+        before the indexer's proxy pass reads the index-K cache. forward()
+        then receives k=v=None, which is the backend's contract for "K/V are
+        already resident", so neither FMHA phase writes them again.
+            # On the FP8 indexer path idx_k is None: the fused producer already
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py
@@ -0,0 +1,173 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Fused paged-cache scatter for the MiniMax-M3 MSA backend.
+One Triton launch writes a layer's new-token main K, main V, and (sparse
+layers) index-K into their paged HND caches at the step's write slots.
+The legacy path costs three aten advanced-indexing writes per layer plus
diff -- tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py
@@ -4,8 +4,10 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +16/-2; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py` added +173/-0; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +62/-3; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` modified +5/-1
  - tests: `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +143/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18605 - [None][feat] Support MiniMax-M3 in MegaMoE CuTeDSL

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18605
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `65804bfcede1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +604/-32, 1069 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +144/-0 (144 lines); hunks: -38,9 +38,12; -52,10 +55,12; symbols: _M3CompositionGate, forward, _M3CompositionExperts, __init__, touching `_M3CompositionGate, forward, _M3CompositionExperts`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +25/-4 (29 lines); hunks: -63,6 +63,7; -77,6 +78,12; symbols: _moe_routed_output_is_global, __init__, _compute_shared_output, touching `_moe_routed_output_is_global, __init__, _compute_shared_output`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +144/-0 (144 lines); hunks: -38,9 +38,12; -52,10 +55,12; symbols: _M3CompositionGate, forward, _M3CompositionExperts, __init__
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +25/-4 (29 lines); hunks: -63,6 +63,7; -77,6 +78,12; symbols: _moe_routed_output_is_global, __init__, _compute_shared_output
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -38,9 +38,12 @@
+    MiniMaxM3DecoderLayer,
+    MiniMaxM3MoE,
+    _moe_routed_output_is_global,
@@ -52,10 +55,12 @@
+from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
+from tensorrt_llm._torch.utils import AuxStreamType, EventType
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -63,6 +63,7 @@
+from ..moe.fused_moe.interface import MoESchedulerKind
@@ -77,6 +78,12 @@
+def _moe_routed_output_is_global(experts: nn.Module) -> bool:
+    """Return whether the selected MoE backend performs fused communication."""
+    backend = getattr(experts, "backend", experts)
+    return getattr(backend, "scheduler_kind", None) == MoESchedulerKind.FUSED_COMM
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +144/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +25/-4
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/moe/moe_test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19298 - [None][fix] Align MiniMax-M3 composition test fixtures with MoE interfaces

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19298
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/models/test_minimax_m3.py`; associated commits `e19b5e3e7002`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +3/-1, 18 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def forward(; symbols: forward, _M3CompositionShared, touching `forward, _M3CompositionShared`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +2/-1 (3 lines); hunks: -92,7 +92,8 @@ def forward(; symbols: forward, _M3CompositionShared
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -92,7 +92,8 @@ def forward(
-    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+    def forward(self, hidden_states: torch.Tensor, lora_params: dict | None = None) -> torch.Tensor:
+        del lora_params
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/peft/test_moe_lora_model_path.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18205 - [None][perf] Fuse MiniMax-M3 QKV and index projection

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18205
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/common.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` and 6 files; associated commits `b069d2e4acec`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +2362/-63, 2920 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +534/-40 (574 lines); hunks: -47,6 +47,7; -84,6 +85,169 @@ def _moe_routed_output_is_global(experts: nn.Module) -> bool:; symbols: _moe_routed_output_is_global, MiniMaxM3QKVIndexerLinear, __init__, _shard_geometry, touching `_moe_routed_output_is_global, MiniMaxM3QKVIndexerLinear, __init__`; `tests/unittest/_torch/models/test_minimax_m3.py` modified +562/-10 (572 lines); hunks: -31,17 +31,28; -53,6 +64,7; symbols: test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_validate_fused_projection_requires_fp8_main_kv_cache, test_fused_qkv_index_projection_preserves_index_head_groups, test_minimax_m3_uses_one_engine_speculative_base, touching `test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_validate_fused_projection_requires_fp8_main_kv_cache, test_fused_qkv_index_projection_preserves_index_head_groups`; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: _rope_cache, _main_cache, _index_cache, _assert_fp8_within_one_ulp, touching `_rope_cache, _main_cache, _index_cache`; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_main_kv_insert.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: _reference, _strided_kv_cache, _inputs, _run, touching `_reference, _strided_kv_cache, _inputs`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +534/-40 (574 lines); hunks: -47,6 +47,7; -84,6 +85,169 @@ def _moe_routed_output_is_global(experts: nn.Module) -> bool:; symbols: _moe_routed_output_is_global, MiniMaxM3QKVIndexerLinear, __init__, _shard_geometry
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +562/-10 (572 lines); hunks: -31,17 +31,28; -53,6 +64,7; symbols: test_validate_sparse_attention_runtime_config_accepts_minimax_m3, test_validate_fused_projection_requires_fp8_main_kv_cache, test_fused_qkv_index_projection_preserves_index_head_groups, test_minimax_m3_uses_one_engine_speculative_base
  - `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: _rope_cache, _main_cache, _index_cache, _assert_fp8_within_one_ulp
  - `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_main_kv_insert.py` added +189/-0 (189 lines); hunks: -0,0 +1,189; symbols: _reference, _strided_kv_cache, _inputs, _run
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/common.py` modified +39/-1 (40 lines); hunks: -32,12 +32,39; -47,6 +74,7 @@ class MiniMaxM3SparseParams(SparseParams):; symbols: index_head_range, MiniMaxM3SparseParams, indices_block_size, _shard
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -47,6 +47,7 @@
+from ..attention.backends.sparse.minimax_m3.common import index_head_range
@@ -84,6 +85,169 @@ def _moe_routed_output_is_global(experts: nn.Module) -> bool:
+class MiniMaxM3QKVIndexerLinear(Linear):
+    """Five-way MiniMax-M3 projection with vLLM-compatible TP sharding.
+    Each rank emits ``[Q | K | V | index-Q | index-K]``. Q follows normal
+    attention head sharding, K/V/index-Q follow KV-head sharding (including
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -31,17 +31,28 @@
+import tensorrt_llm._torch.models.modeling_minimaxm3 as modeling_minimaxm3
+from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
+    MiniMaxM3SparseConfig,
+    MiniMaxM3SparseMetadataParams,
+    MiniMaxM3SparseParams,
+    index_head_range,
diff -- tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py
@@ -0,0 +1,357 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +534/-40; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/common.py` modified +39/-1; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +6/-5
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +562/-10; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_horizontal_producer.py` added +357/-0; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_main_kv_insert.py` added +189/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_cpu.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19288 - [None][test] Cover MiniMax-M3 C++ NIXL bounce

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19288
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`; associated commits `0921d98862a5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +387/-30, 581 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +89/-5 (94 lines); hunks: -32,7 +32,10; -158,7 +161,7 @@ def fake_base_init(self, *args, **kwargs) -> None:; symbols: fake_base_init, fake_get_index_k_buffer, test_index_k_views_are_fullgraph_safe, touching `fake_base_init, fake_get_index_k_buffer, test_index_k_views_are_fullgraph_safe`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +17/-16 (33 lines); hunks: -335,10 +335,22 @@ def __init__(; -477,24 +489,13 @@ def _torch_dtype_for_index_cache(self) -> torch.dtype:; symbols: __init__, _torch_dtype_for_index_cache, get_index_k_buffer, get_index_v_buffer, touching `__init__, _torch_dtype_for_index_cache, get_index_k_buffer`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +1/-1 (2 lines); hunks: -1071,7 +1071,7 @@ def _build_msa_fields(self) -> None:; symbols: _build_msa_fields, msa_idx_k_cache, msa_write_idx_k, touching `_build_msa_fields, msa_idx_k_cache, msa_write_idx_k`.
- Code diff details:
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +89/-5 (94 lines); hunks: -32,7 +32,10; -158,7 +161,7 @@ def fake_base_init(self, *args, **kwargs) -> None:; symbols: fake_base_init, fake_get_index_k_buffer, test_index_k_views_are_fullgraph_safe
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +17/-16 (33 lines); hunks: -335,10 +335,22 @@ def __init__(; -477,24 +489,13 @@ def _torch_dtype_for_index_cache(self) -> torch.dtype:; symbols: __init__, _torch_dtype_for_index_cache, get_index_k_buffer, get_index_v_buffer
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +1/-1 (2 lines); hunks: -1071,7 +1071,7 @@ def _build_msa_fields(self) -> None:; symbols: _build_msa_fields, msa_idx_k_cache, msa_write_idx_k
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py
@@ -32,7 +32,10 @@
-from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import MsaDecodeSpan
+from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
+    MiniMaxM3MsaSparseAttentionMetadata,
+    MsaDecodeSpan,
+)
@@ -158,7 +161,7 @@ def fake_base_init(self, *args, **kwargs) -> None:
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py
@@ -335,10 +335,22 @@ def __init__(
+        # Resolve the zero-copy index-K views eagerly. The V2 pool geometry
+        # stays fixed for this manager's lifetime, but resolving it calls
+        # nanobind methods that Dynamo cannot trace from the fused FP8 indexer.
+        kv_layout = self._main_kv_layout_name()
+        self._index_k_buffers: dict[int, Optional[torch.Tensor]] = {}
+            self._index_k_buffers[layer_idx] = super().get_index_k_buffer(
diff -- tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py
@@ -1071,7 +1071,7 @@ def _build_msa_fields(self) -> None:
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +89/-5
  - runtime: `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +17/-16; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/test_lists/test-db/l0_dgx_b200_m3_6gpu.yml`, `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19423 - [None][perf] Extend MiniMax-M3 piecewise CUDA graphs coverage

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19423
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`, `tensorrt_llm/_torch/models/modeling_minimaxm3.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py`; associated commits `e293b6ad4cc3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 24 files, +1883/-143, 2740 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +367/-32 (399 lines); hunks: -23,23 +23,37; -417,7 +431,8 @@ def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k,...; symbols: _dispatch_attention_backend, test_piecewise_projection_fake_preserves_padded_hidden_rows, test_piecewise_fused_projection_preserves_input_token_dimension, fake_boundary, touching `_dispatch_attention_backend, test_piecewise_projection_fake_preserves_padded_hidden_rows, test_piecewise_fused_projection_preserves_input_token_dimension`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +114/-24 (138 lines); hunks: -838,6 +838,54 @@ def _minimax_m3_qkv_index_proj_fake(; -852,11 +900,10 @@ def minimax_m3_attn_custom_op_inplace(; symbols: _minimax_m3_qkv_index_proj_fake, minimax_m3_fused_sparse_qkv_producer, _minimax_m3_fused_sparse_qkv_producer_fake, minimax_m3_attn_custom_op_inplace, touching `_minimax_m3_qkv_index_proj_fake, minimax_m3_fused_sparse_qkv_producer, _minimax_m3_fused_sparse_qkv_producer_fake`; `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py` added +196/-0 (196 lines); hunks: -0,0 +1,196; symbols: _run_empty_adp_rank, producer, test_minimax_m3_piecewise_empty_adp_rank_preserves_caches, touching `_run_empty_adp_rank, producer, test_minimax_m3_piecewise_empty_adp_rank_preserves_caches`; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +88/-5 (93 lines); hunks: -39,6 +39,54; -272,6 +320,36 @@ def test_msa_buffers_include_graph_stable_block_table():; symbols: test_msa_metadata_clears_padded_cache_slot_tail, test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_buffers_include_graph_stable_block_table, test_msa_buffers_stage_local_cache_views, touching `test_msa_metadata_clears_padded_cache_slot_tail, test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_buffers_include_graph_stable_block_table`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +367/-32 (399 lines); hunks: -23,23 +23,37; -417,7 +431,8 @@ def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k,...; symbols: _dispatch_attention_backend, test_piecewise_projection_fake_preserves_padded_hidden_rows, test_piecewise_fused_projection_preserves_input_token_dimension, fake_boundary
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +114/-24 (138 lines); hunks: -838,6 +838,54 @@ def _minimax_m3_qkv_index_proj_fake(; -852,11 +900,10 @@ def minimax_m3_attn_custom_op_inplace(; symbols: _minimax_m3_qkv_index_proj_fake, minimax_m3_fused_sparse_qkv_producer, _minimax_m3_fused_sparse_qkv_producer_fake, minimax_m3_attn_custom_op_inplace
  - `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py` added +196/-0 (196 lines); hunks: -0,0 +1,196; symbols: _run_empty_adp_rank, producer, test_minimax_m3_piecewise_empty_adp_rank_preserves_caches
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +88/-5 (93 lines); hunks: -39,6 +39,54; -272,6 +320,36 @@ def test_msa_buffers_include_graph_stable_block_table():; symbols: test_msa_metadata_clears_padded_cache_slot_tail, test_msa_package_availability_installs_cutlass_compatibility_aliases, test_msa_buffers_include_graph_stable_block_table, test_msa_buffers_stage_local_cache_views
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +17/-0 (17 lines); hunks: -152,6 +152,9 @@ class MiniMaxM3MsaSparseAttentionMetadata(TrtllmAttentionMet...; -399,6 +402,14 @@ def _create_msa_buffers(self) -> None:; symbols: MiniMaxM3MsaSparseAttentionMetadata, _create_msa_buffers, _build_msa_fields
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -23,23 +23,37 @@
+from unittest.mock import Mock
+from torch._dynamo.backends.common import aot_autograd
+from torch._functorch.aot_autograd import make_boxed_func
+from torch._higher_order_ops.auto_functionalize import auto_functionalized, auto_functionalized_v2
+from torch._subclasses.fake_tensor import FakeTensorMode
+from torch.fx.experimental.symbolic_shapes import ShapeEnv
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -838,6 +838,54 @@ def _minimax_m3_qkv_index_proj_fake(
+# Projection and cache insertion inspect runtime shapes and layouts. Keep those
+# checks opaque to avoid Dynamo specialization or graph breaks while PCG captures
+# the fused kernels; explicit mutable cache inputs keep their writes visible.
+@torch.library.custom_op(
+    "trtllm::minimax_m3_fused_sparse_qkv_producer",
+    mutates_args=("kv_cache", "index_k_cache"),
diff -- tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py
@@ -0,0 +1,196 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +367/-32; `tests/unittest/_torch/multi_gpu/test_minimax_m3_piecewise.py` added +196/-0; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +88/-5
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +114/-24; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +17/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/attention/kernels/parallel_hw_agnostic/test_fused_qk_norm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19422 - [None][perf] Add MiniMax-M3 NVFP4 KV cache support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19422
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` and 13 files; associated commits `fba7b54e6bdc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 45 files, +8799/-317, 6930 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +247/-7 (254 lines); hunks: -23,7 +23,7; -38,11 +38,15; symbols: test_nvfp4_cache_fused_layout_validation, test_nvfp4_reload_preserves_attention_scale_storage, load_scales, read_scales, touching `test_nvfp4_cache_fused_layout_validation, test_nvfp4_reload_preserves_attention_scale_storage, load_scales`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +170/-26 (196 lines); hunks: -31,11 +31,14; -228,6 +231,14 @@ def load_five_way_weights(self, shards: Dict[str, Dict]) ->...; symbols: load_five_way_weights, get_text_model_config, _nvfp4_cache_supports_fused_write, _validate_sparse_attention_runtime_config, touching `load_five_way_weights, get_text_model_config, _nvfp4_cache_supports_fused_write`; `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +3/-0 (3 lines); hunks: -184,6 +184,9 @@ def test_load_weights_accepts_base_mapper_without_params_map...; symbols: test_load_weights_accepts_base_mapper_without_params_map, touching `test_load_weights_accepts_base_mapper_without_params_map`; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py` modified +947/-85 (1032 lines); hunks: -11,15 +11,24; -50,11 +59,62; symbols: _pdl_enabled, _e2m1x2_to_f16x2, _dequant_nvfp4_rows, _gqa_sparse_decode_kernel, touching `_pdl_enabled, _e2m1x2_to_f16x2, _dequant_nvfp4_rows`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +247/-7 (254 lines); hunks: -23,7 +23,7; -38,11 +38,15; symbols: test_nvfp4_cache_fused_layout_validation, test_nvfp4_reload_preserves_attention_scale_storage, load_scales, read_scales
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +170/-26 (196 lines); hunks: -31,11 +31,14; -228,6 +231,14 @@ def load_five_way_weights(self, shards: Dict[str, Dict]) ->...; symbols: load_five_way_weights, get_text_model_config, _nvfp4_cache_supports_fused_write, _validate_sparse_attention_runtime_config
  - `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +3/-0 (3 lines); hunks: -184,6 +184,9 @@ def test_load_weights_accepts_base_mapper_without_params_map...; symbols: test_load_weights_accepts_base_mapper_without_params_map
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py` modified +947/-85 (1032 lines); hunks: -11,15 +11,24; -50,11 +59,62; symbols: _pdl_enabled, _e2m1x2_to_f16x2, _dequant_nvfp4_rows, _gqa_sparse_decode_kernel
  - `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_sparse_attn_decode.py` modified +632/-3 (635 lines); hunks: -10,19 +10,32; -32,7 +45,7; symbols: _make_inputs, _dequant_nvfp4_flat, _make_nvfp4_inputs, _run_nvfp4
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -23,7 +23,7 @@
-from unittest.mock import Mock
+from unittest.mock import Mock, create_autospec
@@ -38,11 +38,15 @@
+from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
+from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
+    MiniMaxM3KVCacheManagerV2,
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -31,11 +31,14 @@
+from tensorrt_llm.logger import logger
+from tensorrt_llm.quantization.mode import QuantAlgo
+from ..attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
@@ -228,6 +231,14 @@ def load_five_way_weights(self, shards: Dict[str, Dict]) -> None:
+        # KV calibration is per tensor, independent of the TP row partition.
+        for key in ("k_scale", "v_scale"):
diff -- tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py
@@ -184,6 +184,9 @@ def test_load_weights_accepts_base_mapper_without_params_map() -> None:
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +247/-7; `tests/unittest/_torch/models/checkpoints/hf/test_minimaxm3_weight_mapper.py` modified +3/-0; `tests/unittest/_torch/attention/sparse/msa/test_minimax_m3_sparse_attn_decode.py` modified +632/-3; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_nvfp4_horizontal_producer.py` added +438/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +170/-26; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py` modified +947/-85; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py` modified +563/-4; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py` modified +372/-4
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/gsm8k_inferencex.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19113 - [None][fix] Guard MiniMax-M3 FP8 indexer against padded -1 cache slots

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19113
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/paged_cache.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py`, `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` and 9 files; associated commits `7950afe4013b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +377/-21, 646 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_minimax_m3.py` modified +60/-0 (60 lines); hunks: -70,6 +70,7; -598,6 +599,7 @@ def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k,...; symbols: _dispatch_attention_backend, _has_cuda, test_attention_dispatch_clips_the_piecewise_token_pad, capture, touching `_dispatch_attention_backend, _has_cuda, test_attention_dispatch_clips_the_piecewise_token_pad`; `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +40/-10 (50 lines); hunks: -952,6 +952,40 @@ def _minimax_m3_fused_sparse_qkv_producer_fake(; -989,15 +1023,9 @@ def minimax_m3_attn_custom_op_inplace(; symbols: _minimax_m3_fused_sparse_qkv_producer_fake, _dispatch_attention_over_live_tokens, minimax_m3_attn_custom_op_inplace, _forward_attention_core, touching `_minimax_m3_fused_sparse_qkv_producer_fake, _dispatch_attention_over_live_tokens, minimax_m3_attn_custom_op_inplace`; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +103/-0 (103 lines); hunks: -76,6 +76,9 @@ def test_msa_metadata_clears_padded_cache_slot_tail(; -685,6 +688,8 @@ def get_index_k_buffer(self, layer_idx):; symbols: test_msa_metadata_clears_padded_cache_slot_tail, get_index_k_buffer, msa_write_idx_k, test_the_decode_span_of_a_mixed_step_is_its_generation_suffix, touching `test_msa_metadata_clears_padded_cache_slot_tail, get_index_k_buffer, msa_write_idx_k`; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` modified +80/-0 (80 lines); hunks: -78,6 +78,37 @@ def _strided_cache(num_pages: int, page_size: int = 128, stri...; -163,6 +194,55 @@ def test_minimax_m3_fp8_indexer_defensively_skips_invalid_d...; symbols: _strided_cache, _guarded_cache, _assert_all_zero, _run, touching `_strided_cache, _guarded_cache, _assert_all_zero`.
- Code diff details:
  - `tests/unittest/_torch/models/test_minimax_m3.py` modified +60/-0 (60 lines); hunks: -70,6 +70,7; -598,6 +599,7 @@ def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k,...; symbols: _dispatch_attention_backend, _has_cuda, test_attention_dispatch_clips_the_piecewise_token_pad, capture
  - `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +40/-10 (50 lines); hunks: -952,6 +952,40 @@ def _minimax_m3_fused_sparse_qkv_producer_fake(; -989,15 +1023,9 @@ def minimax_m3_attn_custom_op_inplace(; symbols: _minimax_m3_fused_sparse_qkv_producer_fake, _dispatch_attention_over_live_tokens, minimax_m3_attn_custom_op_inplace, _forward_attention_core
  - `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +103/-0 (103 lines); hunks: -76,6 +76,9 @@ def test_msa_metadata_clears_padded_cache_slot_tail(; -685,6 +688,8 @@ def get_index_k_buffer(self, layer_idx):; symbols: test_msa_metadata_clears_padded_cache_slot_tail, get_index_k_buffer, msa_write_idx_k, test_the_decode_span_of_a_mixed_step_is_its_generation_suffix
  - `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` modified +80/-0 (80 lines); hunks: -78,6 +78,37 @@ def _strided_cache(num_pages: int, page_size: int = 128, stri...; -163,6 +194,55 @@ def test_minimax_m3_fp8_indexer_defensively_skips_invalid_d...; symbols: _strided_cache, _guarded_cache, _assert_all_zero, _run
  - `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` modified +29/-2 (31 lines); hunks: -23,6 +23,21; -103,6 +118,7 @@ def write_msa_main_kv(; symbols: check_decode_span_shape, is_msa_layer, write_msa_main_kv, write_msa_phase_kv
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/test_minimax_m3.py
@@ -70,6 +70,7 @@
+    _dispatch_attention_over_live_tokens,
@@ -598,6 +599,7 @@ def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k, attn_metadata, outp
+    # The dispatch clips to the live tokens and leaves the pad rows as they came.
@@ -1141,6 +1143,64 @@ def _has_cuda() -> bool:
+def test_attention_dispatch_clips_the_piecewise_token_pad():
+    """Only the live tokens reach the attention core, and the output's pad rows
diff -- tensorrt_llm/_torch/models/modeling_minimaxm3.py
@@ -952,6 +952,40 @@ def _minimax_m3_fused_sparse_qkv_producer_fake(
+def _dispatch_attention_over_live_tokens(
+    attn_layer: "MiniMaxM3Attention",
+    q: torch.Tensor,
+    k: Optional[torch.Tensor],
+    v: Optional[torch.Tensor],
+    idx_q: Optional[torch.Tensor],
diff -- tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py
@@ -76,6 +76,9 @@ def test_msa_metadata_clears_padded_cache_slot_tail(
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_minimax_m3.py` modified +60/-0; `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py` modified +103/-0; `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py` modified +80/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_minimaxm3.py` modified +40/-10; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_utils.py` modified +29/-2; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py` modified +23/-2; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/paged_cache.py` modified +21/-0; `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/trtllm_gen_dense_decode.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/msa/test_msa_backend.py`, `tests/unittest/_torch/models/test_minimax_m3.py`, `tests/unittest/_torch/thop/parallel_hw_agnostic/test_minimax_m3_fp8_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
