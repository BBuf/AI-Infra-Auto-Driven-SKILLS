# vLLM DeepSeek V4.1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `benchmarks/kernels/benchmark_dsv41_mega_attn.py` | [#56935](https://github.com/vllm-project/vllm/pull/56935) |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-FP8-TP4-ROCm.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TEP4.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml` | 无直接 PR 号提交 |
| `tests/kernels/test_deepseek_v4_cpu_kernels.py` | 无直接 PR 号提交 |
| `tests/kernels/test_dsv41_mega_attn_layouts.py` | [#56935](https://github.com/vllm-project/vllm/pull/56935) |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | [#56893](https://github.com/vllm-project/vllm/pull/56893), [#56935](https://github.com/vllm-project/vllm/pull/56935) |
| `tests/models/multimodal/processing/test_deepseek_v4_vl.py` | 无直接 PR 号提交 |
| `tests/models/test_deepseek_v41_decoder_replay_layers.py` | [#58132](https://github.com/vllm-project/vllm/pull/58132) |
| `tests/models/test_deepseek_v41_replay_batch.py` | [#58132](https://github.com/vllm-project/vllm/pull/58132) |
| `tests/models/test_deepseek_v41_replay_start.py` | [#56227](https://github.com/vllm-project/vllm/pull/56227) |
| `tests/models/test_deepseek_v4_dspark_rocm.py` | 无直接 PR 号提交 |
| `tests/models/test_deepseek_v4_fi_moe_ep.py` | 无直接 PR 号提交 |
| `tests/models/test_deepseek_v4_mega_moe.py` | [#56214](https://github.com/vllm-project/vllm/pull/56214), [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56266](https://github.com/vllm-project/vllm/pull/56266), [#56568](https://github.com/vllm-project/vllm/pull/56568), [#56599](https://github.com/vllm-project/vllm/pull/56599), [#56741](https://github.com/vllm-project/vllm/pull/56741), [#57204](https://github.com/vllm-project/vllm/pull/57204), [#57604](https://github.com/vllm-project/vllm/pull/57604) |
| `tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py` | 无直接 PR 号提交 |
| `tests/models/test_deepseek_v4_rocm_wo_a.py` | 无直接 PR 号提交 |
| `tests/models/test_deepseek_v4_vl_rocm.py` | 无直接 PR 号提交 |
| `tests/parser/engine/test_deepseek_v4.py` | 无直接 PR 号提交 |
| `tests/parser/engine/test_deepseek_v41.py` | [#56208](https://github.com/vllm-project/vllm/pull/56208) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_4.json` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_5.json` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_2.txt` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_4.txt` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_5.txt` | 无直接 PR 号提交 |
| `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/tokenizers_/test_deepseek_v4.py` | 无直接 PR 号提交 |
| `tests/tokenizers_/test_deepseek_v41.py` | [#56208](https://github.com/vllm-project/vllm/pull/56208), [#56299](https://github.com/vllm-project/vllm/pull/56299), [#58316](https://github.com/vllm-project/vllm/pull/58316) |
| `tests/v1/attention/test_deepseek_v4_rocm_adaptive.py` | 无直接 PR 号提交 |
| `tests/v1/attention/test_deepseek_v4_swa_visible.py` | [#56227](https://github.com/vllm-project/vllm/pull/56227), [#57152](https://github.com/vllm-project/vllm/pull/57152) |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | [#56214](https://github.com/vllm-project/vllm/pull/56214), [#56227](https://github.com/vllm-project/vllm/pull/56227), [#56562](https://github.com/vllm-project/vllm/pull/56562), [#56741](https://github.com/vllm-project/vllm/pull/56741) |
| `vllm/models/deepseek_v4/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/amd/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/amd/dspark.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/amd/model.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/amd/mtp.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/amd/rocm.py` | [#56227](https://github.com/vllm-project/vllm/pull/56227), [#58983](https://github.com/vllm-project/vllm/pull/58983) |
| `vllm/models/deepseek_v4/attention.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/mm_preprocess.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | [#56254](https://github.com/vllm-project/vllm/pull/56254) |
| `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` | [#56228](https://github.com/vllm-project/vllm/pull/56228) |
| `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/rope.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/common/vision.py` | [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56625](https://github.com/vllm-project/vllm/pull/56625), [#58499](https://github.com/vllm-project/vllm/pull/58499) |
| `vllm/models/deepseek_v4/common/vl_model.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/compressor.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/cpu_compressor.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/cpu_mla.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/cpu_sparse.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/cpu_utils.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/dspark.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/model.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/cpu/mtp.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | [#56266](https://github.com/vllm-project/vllm/pull/56266) |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/model.py` | [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56266](https://github.com/vllm-project/vllm/pull/56266), [#56568](https://github.com/vllm-project/vllm/pull/56568), [#57204](https://github.com/vllm-project/vllm/pull/57204), [#57643](https://github.com/vllm-project/vllm/pull/57643) |
| `vllm/models/deepseek_v4/nvidia/mtp.py` | [#56266](https://github.com/vllm-project/vllm/pull/56266) |
| `vllm/models/deepseek_v4/nvidia/ops/__init__.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` | [#56893](https://github.com/vllm-project/vllm/pull/56893) |
| `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` | [#56228](https://github.com/vllm-project/vllm/pull/56228), [#56254](https://github.com/vllm-project/vllm/pull/56254) |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | [#56228](https://github.com/vllm-project/vllm/pull/56228), [#57428](https://github.com/vllm-project/vllm/pull/57428) |
| `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` | [#57604](https://github.com/vllm-project/vllm/pull/57604) |
| `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` | 无直接 PR 号提交 |
| `vllm/models/deepseek_v4/nvidia/vl_model.py` | 无直接 PR 号提交 |
| ... | 59 more files omitted from table; all were used for git tracing. |

## PR 覆盖总览

- git 追溯 PR 数: 51
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 51
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-09-10 | [#56208](https://github.com/vllm-project/vllm/pull/56208) | merged | [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends | `vllm/tokenizers/deepseek_v41_encoding.py`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py` |
| 2026-09-10 | [#56228](https://github.com/vllm-project/vllm/pull/56228) | merged | [Model] DeepSeek-V4.1-Flash Model Definitions | `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4_1/amd/model.py`, `vllm/models/deepseek_v4_1/amd/vl_model.py` |
| 2026-09-11 | [#56214](https://github.com/vllm-project/vllm/pull/56214) | merged | [Model] Support DeepSeek-V4.1-Flash | `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |
| 2026-09-12 | [#56554](https://github.com/vllm-project/vllm/pull/56554) | merged | [DSV4.1] Remove compressor-aware image sentinel token padding | `vllm/transformers_utils/configs/deepseek_v41.py` |
| 2026-09-12 | [#56562](https://github.com/vllm-project/vllm/pull/56562) | merged | [Perf] Fuse DSV4.1 input metadata preparation with Triton | `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/ops/metadata.py` |
| 2026-09-12 | [#56599](https://github.com/vllm-project/vllm/pull/56599) | merged | [CI] Update DeepSeek V4.1 MegaMoE routing test | `tests/models/test_deepseek_v4_mega_moe.py` |
| 2026-09-14 | [#56299](https://github.com/vllm-project/vllm/pull/56299) | merged | [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1 | `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py` |
| 2026-09-14 | [#56741](https://github.com/vllm-project/vllm/pull/56741) | merged | [Refactor] Normalize DeepSeek V4.1 model package naming | `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/nvidia/engram.py` |
| 2026-09-14 | [#56633](https://github.com/vllm-project/vllm/pull/56633) | merged | [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection | `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-14 | [#56513](https://github.com/vllm-project/vllm/pull/56513) | merged | [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-15 | [#56255](https://github.com/vllm-project/vllm/pull/56255) | closed | [DSv4.1] Integrate Mega-mHC from DeepGEMM | `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` |
| 2026-09-15 | [#56893](https://github.com/vllm-project/vllm/pull/56893) | merged | [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record) | `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py`, `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` |
| 2026-09-15 | [#56903](https://github.com/vllm-project/vllm/pull/56903) | merged | [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather | `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-15 | [#56962](https://github.com/vllm-project/vllm/pull/56962) | merged | [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1 (reopen of #56255) | `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py` |
| 2026-09-15 | [#56254](https://github.com/vllm-project/vllm/pull/56254) | merged | [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer | `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` |
| 2026-09-16 | [#56441](https://github.com/vllm-project/vllm/pull/56441) | merged | [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats | `vllm/models/deepseek_v41/nvidia/dspark.py` |
| 2026-09-16 | [#56568](https://github.com/vllm-project/vllm/pull/56568) | merged | [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-09-16 | [#56935](https://github.com/vllm-project/vllm/pull/56935) | merged | [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` |
| 2026-09-16 | [#57152](https://github.com/vllm-project/vllm/pull/57152) | merged | [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1 | `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flashmla.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` |
| 2026-09-16 | [#57204](https://github.com/vllm-project/vllm/pull/57204) | merged | [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py` |
| 2026-09-17 | [#56266](https://github.com/vllm-project/vllm/pull/56266) | merged | [DSv4.1] Integrate Mega-Gate from DeepGEMM | `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py` |
| 2026-09-17 | [#57432](https://github.com/vllm-project/vllm/pull/57432) | merged | [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` |
| 2026-09-18 | [#57604](https://github.com/vllm-project/vllm/pull/57604) | merged | [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers | `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` |
| 2026-09-18 | [#56227](https://github.com/vllm-project/vllm/pull/56227) | merged | [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash | `vllm/models/deepseek_v41/nvidia/model_state.py`, `tests/models/test_deepseek_v41_replay_start.py`, `vllm/models/deepseek_v41/attention.py` |
| 2026-09-20 | [#57434](https://github.com/vllm-project/vllm/pull/57434) | merged | [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-20 | [#56625](https://github.com/vllm-project/vllm/pull/56625) | merged | [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash | `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/amd/vl_model.py` |
| 2026-09-21 | [#57603](https://github.com/vllm-project/vllm/pull/57603) | merged | [Perf][DSV4.1] Overlap mHC coefficients for small TP batches | `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` |
| 2026-09-21 | [#57491](https://github.com/vllm-project/vllm/pull/57491) | merged | [ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-21 | [#57643](https://github.com/vllm-project/vllm/pull/57643) | merged | [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation | `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-21 | [#57874](https://github.com/vllm-project/vllm/pull/57874) | merged | [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs | `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-21 | [#57906](https://github.com/vllm-project/vllm/pull/57906) | merged | [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm | `vllm/models/deepseek_v41/attention.py` |
| 2026-09-22 | [#57428](https://github.com/vllm-project/vllm/pull/57428) | merged | [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter | `vllm/models/kimi_k3/nvidia/model.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-09-22 | [#57435](https://github.com/vllm-project/vllm/pull/57435) | merged | [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-24 | [#58456](https://github.com/vllm-project/vllm/pull/58456) | merged | [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-09-25 | [#57679](https://github.com/vllm-project/vllm/pull/57679) | merged | [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path | `vllm/models/deepseek_v41/common/ops/query_quant.py` |
| 2026-09-26 | [#58678](https://github.com/vllm-project/vllm/pull/58678) | merged | [Perf][DSv4.1] Shard the Engram wkv projection across TP ranks | `vllm/models/deepseek_v41/common/engram.py`, `tests/kernels/test_engram.py` |
| 2026-09-26 | [#57071](https://github.com/vllm-project/vllm/pull/57071) | merged | [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support | `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/quant_config.py`, `vllm/models/deepseek_v4/quant_config.py` |
| 2026-09-26 | [#58316](https://github.com/vllm-project/vllm/pull/58316) | merged | [Bugfix][Frontend][Rust Frontend] Update DeepSeek V4.1 Flash reasoning effort mappings | `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41_encoding.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` |
| 2026-09-26 | [#58499](https://github.com/vllm-project/vllm/pull/58499) | merged | [Bugfix][DSV4.1] Avoid host sync in ViT CUDA graph replay metadata | `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v4/common/vision.py` |
| 2026-09-26 | [#58586](https://github.com/vllm-project/vllm/pull/58586) | merged | [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary | `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py` |
| 2026-09-27 | [#58634](https://github.com/vllm-project/vllm/pull/58634) | merged | [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103 | `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py`, `vllm/models/deepseek_v41/nvidia/ops/o_proj.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` |
| 2026-09-27 | [#57407](https://github.com/vllm-project/vllm/pull/57407) | merged | [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash | `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py` |
| 2026-09-28 | [#58983](https://github.com/vllm-project/vllm/pull/58983) | merged | [ROCm][Refactor] Move DeepSeek-V4/V4.1 multi-stream overlap gate to ROCm platform | `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/platforms/rocm.py` |
| 2026-09-29 | [#58655](https://github.com/vllm-project/vllm/pull/58655) | merged | [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel | `vllm/models/deepseek_v41/amd/model.py` |
| 2026-09-29 | [#58671](https://github.com/vllm-project/vllm/pull/58671) | merged | [ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel | `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/config/attention.py` |
| 2026-09-29 | [#59119](https://github.com/vllm-project/vllm/pull/59119) | merged | [DSv4.1] Avoid runtime recompiles of _ring_slot_mapping_kernel | `vllm/models/deepseek_v41/compressor.py` |
| 2026-09-29 | [#58132](https://github.com/vllm-project/vllm/pull/58132) | merged | [Model] Decoder-side SWA bounded replay for DeepSeek-V4.1 | `vllm/models/deepseek_v41/nvidia/model_state.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `tests/models/test_deepseek_v41_replay_batch.py` |
| 2026-09-29 | [#57898](https://github.com/vllm-project/vllm/pull/57898) | merged | [Bugfix] Profile maximum DeepSeek V4.1 vision features | `vllm/models/deepseek_v41/common/mm_preprocess.py` |
| 2026-09-30 | [#58539](https://github.com/vllm-project/vllm/pull/58539) | merged | [ROCm][DSv4.1][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill | `vllm/models/deepseek_v41/amd/rocm.py` |
| 2026-10-01 | [#59327](https://github.com/vllm-project/vllm/pull/59327) | merged | [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups | `vllm/models/deepseek_v41/common/engram.py`, `vllm/models/deepseek_v41/nvidia/engram.py`, `vllm/models/deepseek_v41/nvidia/model.py` |
| 2026-10-05 | [#58560](https://github.com/vllm-project/vllm/pull/58560) | merged | [Bugfix][DSv4.1] Keep the compressor ring out of the null block | `vllm/models/deepseek_v41/compressor.py` |

## 逐 PR diff 审计卡

### PR #56208 - [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends

- 链接: https://github.com/vllm-project/vllm/pull/56208
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_deepseek_v41.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/parser/deepseek_v4.py` 等 10 个文件；关联提交 `b47b01cf3383`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 52 个文件，+2986/-113，可读 patch 3929 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0 (577 lines); hunks: -0,0 +1,577; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, merge_tool_messages，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`；`tests/tokenizers_/test_deepseek_v41.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: render, test_reference_encoder_fixtures, test_numeric_reasoning_effort, test_invalid_effort_is_a_request_error，涉及 `render, test_reference_encoder_fixtures, test_numeric_reasoning_effort`；`vllm/tokenizers/deepseek_v41.py` added +121/-0 (121 lines); hunks: -0,0 +1,121; symbols: _normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer, apply_chat_template，涉及 `_normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer`；`vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: DeepseekV41Config, __init__，涉及 `DeepseekV41Config, __init__`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0 (577 lines); hunks: -0,0 +1,577; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, merge_tool_messages
  - `tests/tokenizers_/test_deepseek_v41.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: render, test_reference_encoder_fixtures, test_numeric_reasoning_effort, test_invalid_effort_is_a_request_error
  - `vllm/tokenizers/deepseek_v41.py` added +121/-0 (121 lines); hunks: -0,0 +1,121; symbols: _normalize_messages, get_deepseek_v41_tokenizer, _DeepseekV41Tokenizer, apply_chat_template
  - `vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: DeepseekV41Config, __init__
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` added +38/-0 (38 lines); hunks: -0,0 +1,38
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v41_encoding.py
@@ -0,0 +1,577 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa
+"""Text encoding adapted from the DeepSeek V4.1 reference encoder.
+Message normalization and serving controls live in deepseek_v41.py.
+"""
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -0,0 +1,169 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+import json
+from pathlib import Path
+import pytest
diff -- vllm/tokenizers/deepseek_v41.py
@@ -0,0 +1,121 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/tokenizers/deepseek_v41_encoding.py` added +577/-0; `vllm/tokenizers/deepseek_v41.py` added +121/-0; `vllm/transformers_utils/configs/deepseek_v41.py` added +76/-0; `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py` added +6/-0
  - tests: `tests/tokenizers_/test_deepseek_v41.py` added +169/-0; `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` added +38/-0; `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` added +3/-0; `tests/parser/engine/test_deepseek_v41.py` added +146/-0
- 验证与风险: diff 自带测试面 `rust/src/chat/tests/roundtrip.rs`, `tests/parser/engine/test_deepseek_v41.py`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56228 - [Model] DeepSeek-V4.1-Flash Model Definitions

- 链接: https://github.com/vllm-project/vllm/pull/56228
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`, `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` 等 6 个文件；关联提交 `9b959b86577c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 47 个文件，+12902/-196，可读 patch 14210 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0 (1111 lines); hunks: -0,0 +1,1111; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, for，涉及 `DeepseekV4MoE, __init__, _select_dsv4_attn_cls`；`vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0 (1086 lines); hunks: -0,0 +1,1086; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, _use_sequence_parallel，涉及 `DeepseekV4MoE, __init__, _select_dsv4_attn_cls`；`vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str，涉及 `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`；`vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str，涉及 `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0 (1111 lines); hunks: -0,0 +1,1111; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, for
  - `vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0 (1086 lines); hunks: -0,0 +1,1086; symbols: DeepseekV4MoE, __init__, _select_dsv4_attn_cls, _use_sequence_parallel
  - `vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str
  - `vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0 (357 lines); hunks: -0,0 +1,357; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, get_placeholder_str
  - `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +105/-70 (175 lines); hunks: -32,37 +32,39 @@ class CompileKey:; -74,8 +76,8 @@ def kernel(; symbols: CompileKey, varies, kernel
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v4_1/nvidia/model.py
@@ -0,0 +1,1111 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4_1/amd/model.py
@@ -0,0 +1,1086 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/models/deepseek_v4_1/amd/vl_model.py
@@ -0,0 +1,357 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v4_1/nvidia/model.py` added +1111/-0; `vllm/models/deepseek_v4_1/amd/model.py` added +1086/-0; `vllm/models/deepseek_v4_1/amd/vl_model.py` added +357/-0; `vllm/models/deepseek_v4_1/nvidia/vl_model.py` added +357/-0; `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` modified +105/-70; `vllm/models/deepseek_v4/common/vision.py` modified +93/-2
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +47/-11
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`, `tests/parser/engine/trace_builder.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56214 - [Model] Support DeepSeek-V4.1-Flash

- 链接: https://github.com/vllm-project/vllm/pull/56214
- 状态/时间: merged / 2026-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`；关联提交 `e77daef89e18`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 49 个文件，+3996/-138，可读 patch 4994 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4 (215 lines); hunks: -18,14 +18,147; -169,15 +302,79 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing, test_deepseek_v41_fused_moe_uses_draft_counts_or_main_defaults，涉及 `v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing`；`tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1 (44 lines); hunks: -7,19 +7,61; symbols: test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios，涉及 `test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios`；`vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17 (26 lines); hunks: -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(; -1089,8 +1088,6 @@ def deepgemm_post_process_weight_scale_block(; symbols: w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block，涉及 `w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block`；`vllm/model_executor/layers/quantization/__init__.py` modified +15/-2 (17 lines); hunks: -115,7 +115,20 @@ def get_quantization_config(quantization: str) -> type[Quan...; -161,7 +174,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, is，涉及 `get_quantization_config, is`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4 (215 lines); hunks: -18,14 +18,147; -169,15 +302,79 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert...; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table, check_routing, test_deepseek_v41_fused_moe_uses_draft_counts_or_main_defaults
  - `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1 (44 lines); hunks: -7,19 +7,61; symbols: test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla, test_indexer_preserves_deepseek_v41_mla_block_size, test_indexer_warmup_normalizes_zero_compress_ratios
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17 (26 lines); hunks: -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(; -1089,8 +1088,6 @@ def deepgemm_post_process_weight_scale_block(; symbols: w8a8_triton_block_scaled_mm, deepgemm_post_process_weight_scale_block, deepgemm_post_process_fp8_weight_block
  - `vllm/model_executor/layers/quantization/__init__.py` modified +15/-2 (17 lines); hunks: -115,7 +115,20 @@ def get_quantization_config(quantization: str) -> type[Quan...; -161,7 +174,7 @@ def get_quantization_config(quantization: str) -> type[Quant...; symbols: get_quantization_config, is
  - `vllm/model_executor/layers/fusion/quant_activation.py` modified +11/-1 (12 lines); hunks: -8,7 +8,7; -34,6 +34,16 @@ class QuantizedActivation:; symbols: QuantizedActivation, weak_ref, expose_input_quant_key
- 关键代码摘录:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -18,14 +18,147 @@
+from vllm.models.deepseek_v4_1.common.mm_preprocess import (
+    IMAGE_PAD_ID,
+    IMAGE_SENTINEL_BASE_ID,
+)
+from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4MoE as DeepseekV41MoE
+from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
diff -- tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
@@ -7,19 +7,61 @@
+from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
+from vllm.models.deepseek_v4_1.sparse_mla import (
+    DeepseekV4SparseMLABackend as DeepseekV41SparseMLABackend,
+)
+    DeepseekV4IndexerBackend,
+    DeepseekV41IndexerBackend,
diff -- vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -937,14 +937,13 @@ def w8a8_triton_block_scaled_mm(
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +211/-4; `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +43/-1
  - runtime: `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +9/-17; `vllm/model_executor/layers/quantization/__init__.py` modified +15/-2; `vllm/model_executor/layers/fusion/quant_activation.py` modified +11/-1; `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` modified +8/-2; `vllm/model_executor/models/registry.py` modified +8/-0; `vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py` modified +6/-1
- 验证与风险: diff 自带测试面 `tests/fusion/test_quant_activation_contract.py`, `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/quantization/test_rocm_mxfp8_linear.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56554 - [DSV4.1] Remove compressor-aware image sentinel token padding

- 链接: https://github.com/vllm-project/vllm/pull/56554
- 状态/时间: merged / 2026-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/deepseek_v41.py`；关联提交 `30118ba27d1d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+10/-132，可读 patch 245 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6 (6 lines); hunks: -68,9 +68,3 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6 (6 lines); hunks: -68,9 +68,3 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/deepseek_v41.py
@@ -68,9 +68,3 @@ def __init__(
-        # The visibility span covers the image block [IMAGE_START,
-        # IMAGE_END]; the mm placeholder additionally carries a leading
-        # compressor-alignment pad of ``COMPRESS_PAD_TO - 1 - offset %
-        # COMPRESS_PAD_TO`` tokens (see common/mm_preprocess.py), with
-        # COMPRESS_PAD_TO = 2 for v4.1's ratio-2 compressors.
-        self.mm_prefix_span_leading_pad_modulus = 2 if vision_n_layers > 0 else 0
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-6
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4_1/amd/vl_model.py`, `vllm/models/deepseek_v4_1/common/mm_preprocess.py`, `vllm/models/deepseek_v4_1/nvidia/vl_model.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #56562 - [Perf] Fuse DSV4.1 input metadata preparation with Triton

- 链接: https://github.com/vllm-project/vllm/pull/56562
- 状态/时间: merged / 2026-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`；关联提交 `13e221f83084`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+274/-63，可读 patch 383 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0 (105 lines); hunks: -32,6 +32,111; symbols: test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla，涉及 `test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla`；`vllm/v1/attention/backends/mla/indexer.py` modified +69/-52 (121 lines); hunks: -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(; -1156,12 +1134,15 @@ def build(; symbols: _prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks, build，涉及 `_prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks`；`vllm/v1/attention/ops/metadata.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel，涉及 `_token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel`；`vllm/v1/attention/backend.py` modified +13/-11 (24 lines); hunks: -483,19 +483,21 @@ def token_to_req_indices(self, buffer: torch.Tensor) -> to...; symbols: token_to_req_indices，涉及 `token_to_req_indices`。
- 代码 diff 细节:
  - `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0 (105 lines); hunks: -32,6 +32,111; symbols: test_fused_indexer_decode_metadata, test_device_token_request_mapping, test_indexer_shares_uncompressed_block_size_with_deepseek_v4_mla
  - `vllm/v1/attention/backends/mla/indexer.py` modified +69/-52 (121 lines); hunks: -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(; -1156,12 +1134,15 @@ def build(; symbols: _prepare_global_decode_seq_lens, _build_varlen_decode_indices, _split_indexer_prefill_chunks, build
  - `vllm/v1/attention/ops/metadata.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: _token_request, _token_request_mapping_kernel, _indexer_decode_metadata_kernel
  - `vllm/v1/attention/backend.py` modified +13/-11 (24 lines); hunks: -483,19 +483,21 @@ def token_to_req_indices(self, buffer: torch.Tensor) -> to...; symbols: token_to_req_indices
- 关键代码摘录:

```diff
diff -- tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
@@ -32,6 +32,111 @@
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
+@pytest.mark.parametrize("query_lens", [[1], [6] * 16, [0, 2, 6, 0, 4], [0, 0, 0]])
+@pytest.mark.parametrize("padding", [0, 17])
+def test_fused_indexer_decode_metadata(query_lens, padding):
+    """Flatten device query boundaries and clear graph padding on every replay."""
+    from vllm.v1.attention.ops.metadata import _indexer_decode_metadata_kernel
diff -- vllm/v1/attention/backends/mla/indexer.py
@@ -960,28 +960,6 @@ def _prepare_global_decode_seq_lens(
-    def _build_varlen_decode_indices(
-        self,
-        decode_lens: torch.Tensor,
-        decode_lens_cpu: torch.Tensor,
-        num_decode_tokens: int,
-    ) -> torch.Tensor:
diff -- vllm/v1/attention/ops/metadata.py
@@ -0,0 +1,87 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` modified +105/-0
  - runtime: `vllm/v1/attention/backends/mla/indexer.py` modified +69/-52; `vllm/v1/attention/ops/metadata.py` added +87/-0; `vllm/v1/attention/backend.py` modified +13/-11
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56599 - [CI] Update DeepSeek V4.1 MegaMoE routing test

- 链接: https://github.com/vllm-project/vllm/pull/56599
- 状态/时间: merged / 2026-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`；关联提交 `1ee4be4dbc57`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-6，可读 patch 30 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6 (9 lines); hunks: -18,10 +18,7; -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(; symbols: test_deepseek_v41_moe_routes_without_hash_table，涉及 `test_deepseek_v41_moe_routes_without_hash_table`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6 (9 lines); hunks: -18,10 +18,7; -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(; symbols: test_deepseek_v41_moe_routes_without_hash_table
- 关键代码摘录:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -18,10 +18,7 @@
-from vllm.models.deepseek_v4_1.common.mm_preprocess import (
-    IMAGE_PAD_ID,
-    IMAGE_SENTINEL_BASE_ID,
-)
+from vllm.models.deepseek_v4_1.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
@@ -78,7 +75,7 @@ def test_deepseek_v41_moe_routes_without_hash_table(
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +3/-6
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56299 - [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1

- 链接: https://github.com/vllm-project/vllm/pull/56299
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41.py`；关联提交 `eb42686a30cd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+19/-1，可读 patch 34 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0 (18 lines); hunks: -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():; symbols: test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header，涉及 `test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header`；`vllm/tokenizers/deepseek_v41.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7 @@ def _normalize_messages(; symbols: _normalize_messages，涉及 `_normalize_messages`。
- 代码 diff 细节:
  - `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0 (18 lines); hunks: -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():; symbols: test_raw_text_parts_preserve_reference_separator, test_responses_text_parts_match_chat_text_parts, test_mid_system_gets_its_own_marker_and_generation_header
  - `vllm/tokenizers/deepseek_v41.py` modified +1/-1 (2 lines); hunks: -32,7 +32,7 @@ def _normalize_messages(; symbols: _normalize_messages
- 关键代码摘录:

```diff
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -99,6 +99,24 @@ def test_raw_text_parts_preserve_reference_separator():
+@pytest.mark.parametrize(
+    ("role", "responses_type"),
+    [("user", "input_text"), ("assistant", "output_text")],
+)
+def test_responses_text_parts_match_chat_text_parts(role, responses_type):
+    responses_message = {
diff -- vllm/tokenizers/deepseek_v41.py
@@ -32,7 +32,7 @@ def _normalize_messages(
-                if part_type == "text":
+                if part_type in ("text", "input_text", "output_text"):
```

- 提取文件（未人工审阅）:
  - tests: `tests/tokenizers_/test_deepseek_v41.py` modified +18/-0
  - runtime: `vllm/tokenizers/deepseek_v41.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/tokenizers_/test_deepseek_v41.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56741 - [Refactor] Normalize DeepSeek V4.1 model package naming

- 链接: https://github.com/vllm-project/vllm/pull/56741
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/models/deepseek_v41/__init__.py`, `vllm/models/deepseek_v41/amd/__init__.py`, `vllm/models/deepseek_v41/amd/dspark.py` 等 29 个文件；关联提交 `cf1584f373d0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 44 个文件，+71/-72，可读 patch 476 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/attention.py` renamed +7/-7 (14 lines); hunks: -26,7 +26,7; -49,8 +49,8; symbols: DeepseekV4Attention, __init__, forward, _can_fuse_query_quant，涉及 `DeepseekV4Attention, __init__, forward`；`vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3 (6 lines); hunks: -13,9 +13,9；`vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3 (6 lines); hunks: -23,16 +23,16；`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3 (6 lines); hunks: -13,12 +13,12。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/attention.py` renamed +7/-7 (14 lines); hunks: -26,7 +26,7; -49,8 +49,8; symbols: DeepseekV4Attention, __init__, forward, _can_fuse_query_quant
  - `vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3 (6 lines); hunks: -13,9 +13,9
  - `vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3 (6 lines); hunks: -23,16 +23,16
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3 (6 lines); hunks: -13,12 +13,12
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` renamed +3/-3 (6 lines); hunks: -10,13 +10,13
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -26,7 +26,7 @@
-from vllm.models.deepseek_v4_1.common.ops import (
+from vllm.models.deepseek_v41.common.ops import (
@@ -49,8 +49,8 @@
-from vllm.models.deepseek_v4_1.common.rope import build_deepseek_v4_rope
-from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
+from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,9 +13,9 @@
-from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
-from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
-from vllm.models.deepseek_v4_1.sparse_mla import (
+from vllm.models.deepseek_v41.attention import DeepseekV4Attention
+from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache
+from vllm.models.deepseek_v41.sparse_mla import (
diff -- vllm/models/deepseek_v41/nvidia/engram.py
@@ -23,16 +23,16 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/attention.py` renamed +7/-7; `vllm/models/deepseek_v41/amd/rocm.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/engram.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/flashmla.py` renamed +3/-3; `vllm/models/deepseek_v41/nvidia/model.py` renamed +3/-3
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_engram.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56633 - [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection

- 链接: https://github.com/vllm-project/vllm/pull/56633
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/dspark.py`, `vllm/models/deepseek_v41/nvidia/model.py`；关联提交 `5372e72a9884`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+837/-92，可读 patch 1316 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52 (178 lines); hunks: -20,6 +20,7; -214,17 +215,45 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1 (2 lines); hunks: -207,7 +207,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52 (178 lines); hunks: -20,6 +20,7; -214,17 +215,45 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1 (2 lines); hunks: -207,7 +207,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -20,6 +20,7 @@
+    mhc_fused_post_pre_delayed_tilelang,
@@ -214,17 +215,45 @@ def __init__(
+            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
+                mhc_fused_post_pre_splits,
+            )
-            for input_size, use_pre_mix in (
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -207,7 +207,7 @@ def forward(
-            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
+            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/model.py` modified +126/-52; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_mhc_jit_warmup.py`, `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56513 - [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection

- 链接: https://github.com/vllm-project/vllm/pull/56513
- 状态/时间: merged / 2026-09-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/model.py`；关联提交 `00972dfd7298`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+346/-64，可读 patch 633 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/model.py` modified +37/-17 (54 lines); hunks: -274,7 +274,7 @@ def forward(; -288,7 +288,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/model.py` modified +37/-17 (54 lines); hunks: -274,7 +274,7 @@ def forward(; -288,7 +288,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -274,7 +274,7 @@ def forward(
-                post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
+                residual, post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
@@ -288,7 +288,7 @@ def forward(
-                post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
+                residual, post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
@@ -301,18 +301,7 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +37/-17
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56255 - [DSv4.1] Integrate Mega-mHC from DeepGEMM

- 链接: https://github.com/vllm-project/vllm/pull/56255
- 状态/时间: closed / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`；关联提交 `bb5507741155`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+315/-43，可读 patch 566 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre，涉及 `is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`；`vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21 (62 lines); hunks: -76,6 +76,7; -334,30 +335,48 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21 (62 lines); hunks: -76,6 +76,7; -334,30 +335,48 @@ def forward(; symbols: forward
  - `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py
@@ -0,0 +1,144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import functools
+import torch
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_post_tilelang,
diff -- vllm/models/deepseek_v4_1/nvidia/model.py
@@ -76,6 +76,7 @@
+from .ops.mega_mhc import mhc_shifted_post_pre
@@ -334,30 +335,48 @@ def forward(
-            residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
+                residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
-            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
-                residual,
diff -- vllm/models/deepseek_v4_1/nvidia/ops/__init__.py
@@ -0,0 +1,2 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v4_1/nvidia/ops/mega_mhc.py` added +144/-0; `vllm/models/deepseek_v4_1/nvidia/model.py` modified +41/-21; `vllm/models/deepseek_v4_1/nvidia/ops/__init__.py` added +2/-0
- 验证与风险: diff 自带测试面 `tests/kernels/quantization/test_block_fp8.py`, `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56893 - [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record)

- 链接: https://github.com/vllm-project/vllm/pull/56893
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`, `vllm/models/deepseek_v41/amd/dspark.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py` 等 7 个文件；关联提交 `e6eb0d120c48`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+709/-110，可读 patch 1334 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8 (194 lines); hunks: -36,6 +36,18; -163,23 +175,81 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache，涉及 `quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache`；`vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6 (76 lines); hunks: -232,10 +232,13 @@ def rope_quant_insert(; -246,8 +249,15 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel, _rope_plain_insert_kernel，涉及 `rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel`；`vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23 (62 lines); hunks: -23,10 +23,11; -35,17 +36,20 @@ class DequantGatherKCacheKernel(; symbols: DequantGatherKCacheKernel, CompileKey, kernel, load_g2s，涉及 `DequantGatherKCacheKernel, CompileKey, kernel`；`vllm/models/deepseek_v41/attention.py` modified +34/-9 (43 lines); hunks: -51,6 +51,7; -109,6 +110,16 @@ def _fill_short_context_topk_indices(; symbols: _fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, __init__，涉及 `_fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8 (194 lines); hunks: -36,6 +36,18; -163,23 +175,81 @@ def quantize_and_insert_k_kernel(; symbols: quantize_and_insert_k_kernel, _quantize_and_insert_k_mxfp8_kernel, quantize_and_insert_k_cache
  - `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6 (76 lines); hunks: -232,10 +232,13 @@ def rope_quant_insert(; -246,8 +249,15 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_kernel, _rope_quant_insert_mxfp8_kernel, _rope_plain_insert_kernel
  - `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23 (62 lines); hunks: -23,10 +23,11; -35,17 +36,20 @@ class DequantGatherKCacheKernel(; symbols: DequantGatherKCacheKernel, CompileKey, kernel, load_g2s
  - `vllm/models/deepseek_v41/attention.py` modified +34/-9 (43 lines); hunks: -51,6 +51,7; -109,6 +110,16 @@ def _fill_short_context_topk_indices(; symbols: _fill_short_context_topk_indices, _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, __init__
  - `vllm/models/deepseek_v41/amd/dspark.py` modified +2/-0 (2 lines); hunks: -259,6 +259,8 @@ def _insert_context_kv(; symbols: _insert_context_kv
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -36,6 +36,18 @@
+# Per-token byte width of the two paged fp8 records. Both put a page's whole
+# data region ahead of its whole scale region, so ``k_cache.shape[-1]`` -- the
+# width the KV-cache spec publishes -- identifies the record:
+#   V4   (584 B): 448 fp8 e4m3 NoPE + 64 bf16 RoPE, then 7 UE8M0 scales of 64
+#                 dims each plus a pad byte.
+#   V4.1 (528 B): all 512 dims as fp8 e4m3 (RoPE included), then 16 UE8M0
diff -- vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py
@@ -232,10 +232,13 @@ def rope_quant_insert(
-    layout: ``uint8`` is the fp8_ds_mla paged layout (576 value bytes and eight
-    segregated UE8M0 scale bytes per token, including one zero padding scale);
-    ``bfloat16`` and ``float8_e4m3fn`` are the plain [448 NoPE | 64 RoPE] rows
-    read by FlashInfer, the latter scaled by the per-tensor ``fp8_scale``.
+    layout: ``uint8`` is the fp8_ds_mla paged layout, whose record the
+    per-token byte width names -- 584 B for V4 (576 value bytes and eight
diff -- vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py
@@ -23,10 +23,11 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +186/-8; `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +70/-6; `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` modified +39/-23; `vllm/models/deepseek_v41/attention.py` modified +34/-9; `vllm/models/deepseek_v41/amd/dspark.py` modified +2/-0; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-0
  - tests: `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` modified +98/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56903 - [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather

- 链接: https://github.com/vllm-project/vllm/pull/56903
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/dspark.py`；关联提交 `073f883b56a5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+23/-3，可读 patch 46 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3 (5 lines); hunks: -217,15 +217,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3 (5 lines); hunks: -217,15 +217,14 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -217,15 +217,14 @@ def forward(
-        if self.use_sequence_parallel:
-            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
-            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
+        if self.use_sequence_parallel:
+            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +2/-3
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56962 - [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1 (reopen of #56255)

- 链接: https://github.com/vllm-project/vllm/pull/56962
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`；关联提交 `bb5507741155`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+316/-41，可读 patch 471 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre，涉及 `is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`；`vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38 (72 lines); hunks: -20,7 +20,6; -79,6 +78,7; symbols: forward，涉及 `forward`；`vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38 (72 lines); hunks: -20,7 +20,6; -79,6 +78,7; symbols: forward
  - `vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import functools
+import torch
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_fused_post_pre_delayed_tilelang,
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -20,7 +20,6 @@
-    mhc_fused_post_pre_delayed_tilelang,
@@ -79,6 +78,7 @@
+from .ops.mega_mhc import mhc_shifted_post_pre
@@ -403,25 +403,23 @@ def forward(
-            residual, post_mix, res_mix, x, attn_pre, aux = (
-                mhc_fused_post_pre_delayed_tilelang(
diff -- vllm/models/deepseek_v41/nvidia/ops/__init__.py
@@ -0,0 +1,2 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` added +156/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +34/-38; `vllm/models/deepseek_v41/nvidia/ops/__init__.py` added +2/-0
- 验证与风险: diff 自带测试面 `tests/kernels/quantization/test_block_fp8.py`, `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56254 - [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer

- 链接: https://github.com/vllm-project/vllm/pull/56254
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`, `vllm/models/deepseek_v41/attention.py`；关联提交 `12575123059c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+1830/-26，可读 patch 2149 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/attention.py` modified +67/-20 (87 lines); hunks: -4,6 +4,7; -25,6 +26,7; symbols: __init__, get_kv_cache_spec, forward, get_attn_backend，涉及 `__init__, get_kv_cache_spec, forward`；`vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2 (24 lines); hunks: -309,12 +309,22 @@ def __call__(; -434,12 +444,14 @@ def dispatch( # type: ignore[override]; symbols: __call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel, CompileKey，涉及 `__call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel`；`vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2 (17 lines); hunks: -165,6 +165,7 @@ class CompileKey:; -177,6 +178,7 @@ def kernel(compile_key: CompileKey) -> Any:; symbols: CompileKey, kernel, device_kernel, host_entrypoint，涉及 `CompileKey, kernel, device_kernel`；`vllm/config/attention.py` modified +9/-0 (9 lines); hunks: -86,6 +86,15 @@ class AttentionConfig:; symbols: AttentionConfig, GPU，涉及 `AttentionConfig, GPU`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/attention.py` modified +67/-20 (87 lines); hunks: -4,6 +4,7; -25,6 +26,7; symbols: __init__, get_kv_cache_spec, forward, get_attn_backend
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2 (24 lines); hunks: -309,12 +309,22 @@ def __call__(; -434,12 +444,14 @@ def dispatch( # type: ignore[override]; symbols: __call__, _indexer_weights_out_dtypes, FusedIndexerQRopeMxFp4TritonKernel, CompileKey
  - `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2 (17 lines); hunks: -165,6 +165,7 @@ class CompileKey:; -177,6 +178,7 @@ def kernel(compile_key: CompileKey) -> Any:; symbols: CompileKey, kernel, device_kernel, host_entrypoint
  - `vllm/config/attention.py` modified +9/-0 (9 lines); hunks: -86,6 +86,15 @@ class AttentionConfig:; symbols: AttentionConfig, GPU
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -4,6 +4,7 @@
+import math
@@ -25,6 +26,7 @@
+from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
@@ -63,6 +65,9 @@
+from vllm.v1.attention.backends.mla.sparse_indexer import (
+    DeepseekV41SparseIndexerBackend,
diff -- vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
@@ -309,12 +309,22 @@ def __call__(
+def _indexer_weights_out_dtypes(vllm_config: Any) -> tuple[torch.dtype, ...]:
+    """Weights dtypes the model's indexer layers ask for: fp32 for the dense
+    scoring kernels, plus bf16 when the DeepSeek V4.1 sparse-logits indexer
+    (`SparseMQAIndexer`) is enabled."""
+    if vllm_config.attention_config.indexer_sparse_logits:
+        return (torch.float32, torch.bfloat16)
diff -- vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
@@ -165,6 +165,7 @@ class CompileKey:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/attention.py` modified +67/-20; `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` modified +22/-2; `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` modified +15/-2; `vllm/config/attention.py` modified +9/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_indexer_q_rope_quant.py`, `tests/model_executor/layers/test_mla_short_prefill_indexer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56441 - [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats

- 链接: https://github.com/vllm-project/vllm/pull/56441
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/dspark.py`；关联提交 `6ecd97f1b7fb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+234/-61，可读 patch 345 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60 (77 lines); hunks: -234,67 +234,24 @@ def _insert_context_kv(; symbols: _insert_context_kv, DSparkDeepseekV4ForCausalLM，涉及 `_insert_context_kv, DSparkDeepseekV4ForCausalLM`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60 (77 lines); hunks: -234,67 +234,24 @@ def _insert_context_kv(; symbols: _insert_context_kv, DSparkDeepseekV4ForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/dspark.py
@@ -234,67 +234,24 @@ def _insert_context_kv(
-    """RoPE + quant + paged-cache insert of (already kv_norm'd) context KV.
-    Reuses the DSV4 fused insert ops (which also process a query; we pass a dummy
-    query and discard it, since context tokens have no query). Mirrors
-    ``DeepseekV4Attention._fused_qnorm_rope_kv_insert``.
-    """
-    swa_cache = attn.swa_cache_layer.kv_cache
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/dspark.py` modified +17/-60
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56568 - [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion

- 链接: https://github.com/vllm-project/vllm/pull/56568
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `03f67b3ad1e6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+128/-23，可读 patch 259 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18 (98 lines); hunks: -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; -402,6 +406,7 @@ def fp8_fp4_mega_moe(; symbols: test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe，涉及 `test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm`；`vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5 (53 lines); hunks: -204,6 +204,7 @@ def __init__(; -386,6 +387,17 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale，涉及 `__init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18 (98 lines); hunks: -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; -402,6 +406,7 @@ def fp8_fp4_mega_moe(; symbols: test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights, FakeDeepGemm, get_symm_buffer_for_mega_moe
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5 (53 lines); hunks: -204,6 +204,7 @@ def __init__(; -386,6 +387,17 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale
- 关键代码摘录:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -367,17 +367,21 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(monkeypatch, intermediat
-        pytest.param(5120, 2304, 32, False, False, id="deepseek-v41-flash"),
+        pytest.param(5120, 2304, 32, True, False, id="deepseek-v41-flash"),
+        pytest.param(128, 2304, 128, True, False, id="padded-block128"),
+        pytest.param(5120, 2304, 32, True, True, id="deepseek-v41-flash-mxfp8"),
+        pytest.param(128, 2304, 128, False, False, id="padded-packed-scale-fallback"),
-    """V4.1-Flash's routed padding must preserve the serial shared-expert fallback."""
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -204,6 +204,7 @@ def __init__(
+        self.unpadded_intermediate_size = intermediate_size
@@ -386,6 +387,17 @@ def _finalize_shared_expert_weights(
+        unpadded_size = self.unpadded_intermediate_size * self.num_shared_experts
+        padding = self.intermediate_size * self.num_shared_experts - unpadded_size
+        pad_weights = (
+            padding > 0
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +80/-18
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +48/-5
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56935 - [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache

- 链接: https://github.com/vllm-project/vllm/pull/56935
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `benchmarks/kernels/benchmark_dsv41_mega_attn.py`, `tests/kernels/test_dsv41_mega_attn_layouts.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py` 等 14 个文件；关联提交 `d6a1677d5504`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 20 个文件，+1811/-156，可读 patch 2550 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0 (497 lines); hunks: -0,0 +1,497; symbols: is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice, DeepseekV4MegaAttnAttention，涉及 `is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice`；`vllm/models/deepseek_v41/attention.py` modified +150/-54 (204 lines); hunks: -44,6 +44,7; -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:; symbols: _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, accepts_unnormed_unroped_query，涉及 `_use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention`；`vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2 (98 lines); hunks: -43,10 +43,16; -496,6 +502,73 @@ def _dequantize_and_gather_k_mxfp8_kernel(; symbols: _dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton, dequantize_and_gather_k_cache，涉及 `_dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton`；`vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation, permute_q_to_fused，涉及 `q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0 (497 lines); hunks: -0,0 +1,497; symbols: is_flashmla_mega_attn_supported, alloc_mega_attn_output, _token_slice, DeepseekV4MegaAttnAttention
  - `vllm/models/deepseek_v41/attention.py` modified +150/-54 (204 lines); hunks: -44,6 +44,7; -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:; symbols: _use_v41_mxfp8_kv_record, _resolve_dsv4_kv_cache_dtype, DeepseekV4Attention, accepts_unnormed_unroped_query
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2 (98 lines); hunks: -43,10 +43,16; -496,6 +502,73 @@ def _dequantize_and_gather_k_mxfp8_kernel(; symbols: _dequantize_and_gather_k_mxfp8_kernel, _dequantize_and_gather_k_nvfp4_kernel, dequantize_and_gather_k_cache_triton, dequantize_and_gather_k_cache
  - `vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0 (86 lines); hunks: -0,0 +1,86; symbols: q_fused_permutation, o_fused_permutation, o_fused_chunk_permutation, permute_q_to_fused
  - `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +71/-13 (84 lines); hunks: -4,6 +4,7; -232,13 +233,14 @@ def rope_quant_insert(; symbols: rope_quant_insert, _rope_quant_insert_mxfp8_kernel, _rope_quant_insert_nvfp4_kernel, _rope_plain_insert_kernel
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
@@ -0,0 +1,497 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).
+One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
+its FP8 cast, and writes straight into the buffer ``wo_a`` consumes. So the
+layer declares ``accepts_unnormed_unroped_query`` -- the kernel RoPEs Q itself,
diff -- vllm/models/deepseek_v41/attention.py
@@ -44,6 +44,7 @@
+from vllm.config.cache import CacheDType
@@ -127,34 +128,40 @@ def _use_v41_mxfp8_kv_record() -> bool:
-    kv_cache_dtype: str,
+    kv_cache_dtype: CacheDType,
-) -> tuple[str, torch.dtype]:
+    packed_kv_cache_dtype: CacheDType = "fp8_ds_mla",
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -43,10 +43,16 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` added +497/-0; `vllm/models/deepseek_v41/attention.py` modified +150/-54; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +96/-2; `vllm/models/deepseek_v41/common/ops/fused_layout.py` added +86/-0; `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py` modified +71/-13; `vllm/models/deepseek_v41/sparse_mla.py` modified +51/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_dsv41_mega_attn_layouts.py`, `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57152 - [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1

- 链接: https://github.com/vllm-project/vllm/pull/57152
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` 等 7 个文件；关联提交 `9f9e1dac26ff`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+122/-174，可读 patch 606 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108 (139 lines); hunks: -818,16 +818,12 @@ def combine_topk_swa_indices(; -856,9 +852,6 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel，涉及 `combine_topk_swa_indices, CompileKey, kernel`；`vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19 (21 lines); hunks: -120,9 +120,7 @@ def forward_mqa(; -303,7 +301,7 @@ def _forward_prefill(; symbols: forward_mqa, _forward_prefill，涉及 `forward_mqa, _forward_prefill`；`vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17 (19 lines); hunks: -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:; -411,7 +411,7 @@ def _forward_prefill_mega(; symbols: _reserve_prefill_workspace, _forward_prefill_mega，涉及 `_reserve_prefill_workspace, _forward_prefill_mega`；`vllm/models/deepseek_v41/attention.py` modified +0/-8 (8 lines); hunks: -290,13 +290,6 @@ def __init__(; -578,7 +571,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108 (139 lines); hunks: -818,16 +818,12 @@ def combine_topk_swa_indices(; -856,9 +852,6 @@ def combine_topk_swa_indices(; symbols: combine_topk_swa_indices, CompileKey, kernel
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19 (21 lines); hunks: -120,9 +120,7 @@ def forward_mqa(; -303,7 +301,7 @@ def _forward_prefill(; symbols: forward_mqa, _forward_prefill
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17 (19 lines); hunks: -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:; -411,7 +411,7 @@ def _forward_prefill_mega(; symbols: _reserve_prefill_workspace, _forward_prefill_mega
  - `vllm/models/deepseek_v41/attention.py` modified +0/-8 (8 lines); hunks: -290,13 +290,6 @@ def __init__(; -578,7 +571,6 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-8 (8 lines); hunks: -60,11 +60,3 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -818,16 +818,12 @@ def combine_topk_swa_indices(
-    left_visible: torch.Tensor | None = None,
-    right_visible: torch.Tensor | None = None,
-    max_image_tokens: int = 0,
-    # max_image_tokens widens the SWA column region for in-image
-    # bidirectional visibility; the width is fixed per model so the
-    # caller-provided workspace stays valid on image-free batches.
diff -- vllm/models/deepseek_v41/nvidia/flashmla.py
@@ -120,9 +120,7 @@ def forward_mqa(
-            combined_topk = round_up(
-                top_k + self.window_size + self.max_image_tokens, 128
-            )
+            combined_topk = round_up(top_k + self.window_size, 128)
@@ -303,7 +301,7 @@ def _forward_prefill(
-        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
diff -- vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
@@ -311,7 +311,7 @@ def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +31/-108; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +2/-19; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +2/-17; `vllm/models/deepseek_v41/attention.py` modified +0/-8; `vllm/transformers_utils/configs/deepseek_v41.py` modified +0/-8; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +0/-4
  - tests: `tests/v1/attention/test_deepseek_v4_swa_visible.py` modified +81/-6
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_deepseek_v4_swa_visible.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57204 - [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround

- 链接: https://github.com/vllm-project/vllm/pull/57204
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/model.py`；关联提交 `30b847c1fa54`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+42/-157，可读 patch 313 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92 (129 lines); hunks: -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_...; -333,50 +335,35 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights，涉及 `test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions`；`vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65 (70 lines); hunks: -204,7 +204,6 @@ def __init__(; -387,17 +386,6 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale，涉及 `__init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92 (129 lines); hunks: -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_...; -333,50 +335,35 @@ def test_deepseek_v4_mega_moe_padding_preserves_weights(mo...; symbols: test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership, test_deepseek_v4_mega_moe_padding_preserves_weights, test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions, test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65 (70 lines); hunks: -204,7 +204,6 @@ def __init__(; -387,17 +386,6 @@ def _finalize_shared_expert_weights(; symbols: __init__, _finalize_shared_expert_weights, _prepare_shared_expert_scale
- 关键代码摘录:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -300,8 +300,10 @@ def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
-def test_deepseek_v4_mega_moe_padding_preserves_weights(monkeypatch, intermediate_size):
-    """Pad gate/up separately and zero every added down-projection column."""
+def test_deepseek_v4_mega_moe_preserves_checkpoint_dimensions(
+    monkeypatch, intermediate_size
+):
+    """Keep native widths so V4.1 routed and shared experts can fuse."""
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -204,7 +204,6 @@ def __init__(
-        self.unpadded_intermediate_size = intermediate_size
@@ -387,17 +386,6 @@ def _finalize_shared_expert_weights(
-        unpadded_size = self.unpadded_intermediate_size * self.num_shared_experts
-        padding = self.intermediate_size * self.num_shared_experts - unpadded_size
-        pad_weights = (
-            padding > 0
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +37/-92
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +5/-65
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56266 - [DSv4.1] Integrate Mega-Gate from DeepGEMM

- 链接: https://github.com/vllm-project/vllm/pull/56266
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v41/nvidia/dspark.py` 等 6 个文件；关联提交 `41f9104fa649`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+369/-35，可读 patch 782 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12 (150 lines); hunks: -15,13 +15,15; -33,6 +35,7; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table，涉及 `v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table`；`vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19 (129 lines); hunks: -98,6 +98,37; -803,6 +834,10 @@ def __init__(; symbols: MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP, __init__，涉及 `MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP`；`vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1 (21 lines); hunks: -20,6 +20,7; -51,6 +52,7; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1 (16 lines); hunks: -61,7 +61,9; -339,6 +341,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12 (150 lines); hunks: -15,13 +15,15; -33,6 +35,7; symbols: v41_moe_config, test_deepseek_v41_moe_routes_without_hash_table
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19 (129 lines); hunks: -98,6 +98,37; -803,6 +834,10 @@ def __init__(; symbols: MegaGateRoutingMetadata, prepare_mega_gate_routing_metadata, DeepseekV4MLP, __init__
  - `vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1 (21 lines); hunks: -20,6 +20,7; -51,6 +52,7; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1 (16 lines); hunks: -61,7 +61,9; -339,6 +341,7 @@ def forward(; symbols: forward
  - `vllm/models/deepseek_v4/nvidia/dspark.py` modified +14/-0 (14 lines); hunks: -17,6 +17,7; -51,12 +52,14; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -15,13 +15,15 @@
+    prepare_mega_gate_routing_metadata,
+from vllm.utils.torch_utils import set_default_torch_dtype
@@ -33,6 +35,7 @@
+            dtype=torch.bfloat16,
@@ -63,28 +66,52 @@ def v41_moe_config(dist_init):
+@pytest.mark.parametrize("use_cudagraph", [False, True])
diff -- vllm/models/deepseek_v4/nvidia/model.py
@@ -98,6 +98,37 @@
+class MegaGateRoutingMetadata(typing.NamedTuple):
+    safe_hash_input_ids: torch.Tensor | None
+    image_token_mask: torch.Tensor | None
+    hash_token_mask: torch.Tensor | None
+def prepare_mega_gate_routing_metadata(
+    input_ids: torch.Tensor,
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -20,6 +20,7 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +138/-12
  - runtime: `vllm/models/deepseek_v4/nvidia/model.py` modified +110/-19; `vllm/models/deepseek_v4/nvidia/mtp.py` modified +20/-1; `vllm/models/deepseek_v41/nvidia/model.py` modified +15/-1; `vllm/models/deepseek_v4/nvidia/dspark.py` modified +14/-0; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +14/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`, `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57432 - [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention

- 链接: https://github.com/vllm-project/vllm/pull/57432
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`；关联提交 `80447d276559`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+214/-6，可读 patch 262 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6 (68 lines); hunks: -26,7 +26,11; -183,6 +187,39 @@ class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV4...; symbols: DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build, DeepseekSparseSWAFlashInferBackend，涉及 `DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6 (68 lines); hunks: -26,7 +26,11; -183,6 +187,39 @@ class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV4...; symbols: DeepseekSparseSWAFlashInferMetadataBuilder, __init__, build, DeepseekSparseSWAFlashInferBackend
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py
@@ -26,7 +26,11 @@
-from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
+from vllm.v1.attention.backend import (
+    AttentionCGSupport,
+    CommonAttentionMetadata,
+    MultipleOf,
+)
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +62/-6
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57604 - [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers

- 链接: https://github.com/vllm-project/vllm/pull/57604
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v4_mega_moe.py`, `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`, `vllm/models/deepseek_v41/common/ops/cache_utils.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`；关联提交 `50d812b66a69`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+130/-62，可读 patch 369 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24 (67 lines); hunks: -12,7 +12,7; -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs，涉及 `_prepare_megamoe_inputs_kernel, prepare_megamoe_inputs`；`tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9 (22 lines); hunks: -686,12 +686,14 @@ def forward(self, hidden_states):; -846,6 +848,7 @@ def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout...; symbols: forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast，涉及 `forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout`；`vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1 (6 lines); hunks: -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(; symbols: dequantize_and_gather_k_cache_triton，涉及 `dequantize_and_gather_k_cache_triton`；`vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def _forward_prefill_mega(; symbols: _forward_prefill_mega，涉及 `_forward_prefill_mega`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24 (67 lines); hunks: -12,7 +12,7; -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(; symbols: _prepare_megamoe_inputs_kernel, prepare_megamoe_inputs
  - `tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9 (22 lines); hunks: -686,12 +686,14 @@ def forward(self, hidden_states):; -846,6 +848,7 @@ def test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout...; symbols: forward, test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact, test_deepseek_v4_mega_moe_stages_shared_scale_tma_layout, test_deepseek_v4_pwal_hook_finalizes_mega_moe_and_mhc_broadcast
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1 (6 lines); hunks: -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(; symbols: dequantize_and_gather_k_cache_triton
  - `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1 (2 lines); hunks: -422,7 +422,7 @@ def _forward_prefill_mega(; symbols: _forward_prefill_mega
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
@@ -12,7 +12,7 @@
-@triton.jit
+@triton.jit(do_not_specialize=["num_tokens"])
@@ -40,27 +40,32 @@ def _prepare_megamoe_inputs_kernel(
+    num_tokens,
+    BLOCK_M: tl.constexpr,
-    token_id = tl.program_id(0)
diff -- tests/models/test_deepseek_v4_mega_moe.py
@@ -686,12 +686,14 @@ def forward(self, hidden_states):
-def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
+@pytest.mark.parametrize("num_tokens", [7, 63, 64, 65, 127, 128, 135, 434, 16384])
+@pytest.mark.parametrize("hidden_size", [256, 5120])
+def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact(
+    num_tokens, hidden_size
+):
diff -- vllm/models/deepseek_v41/common/ops/cache_utils.py
@@ -584,7 +584,11 @@ def dequantize_and_gather_k_cache_triton(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` modified +43/-24; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +5/-1; `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` modified +1/-1
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +13/-9
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`, `tests/models/test_deepseek_v4_mega_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #56227 - [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash

- 链接: https://github.com/vllm-project/vllm/pull/56227
- 状态/时间: merged / 2026-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v41_replay_start.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`, `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v4/sparse_mla.py` 等 11 个文件；关联提交 `1dc2d854c120`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 30 个文件，+1067/-38，可读 patch 1949 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2 (145 lines); hunks: -2,15 +2,23; -41,14 +49,69 @@ def _gather_lookback_kernel(; symbols: _gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata, __init__，涉及 `_gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata`；`tests/models/test_deepseek_v41_replay_start.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: state, prepare_attn, _batch, _prepare，涉及 `state, prepare_attn, _batch`；`vllm/models/deepseek_v41/attention.py` modified +20/-0 (20 lines); hunks: -512,6 +512,25 @@ def __init__(; -522,6 +541,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0 (13 lines); hunks: -978,6 +978,10 @@ def kernel(; -1124,6 +1128,8 @@ def build_flashinfer_mixed_sparse_indices(; symbols: kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel，涉及 `kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2 (145 lines); hunks: -2,15 +2,23; -41,14 +49,69 @@ def _gather_lookback_kernel(; symbols: _gather_lookback_kernel, _pad_replayed_slots_kernel, ReplayAttnMetadata, __init__
  - `tests/models/test_deepseek_v41_replay_start.py` added +112/-0 (112 lines); hunks: -0,0 +1,112; symbols: state, prepare_attn, _batch, _prepare
  - `vllm/models/deepseek_v41/attention.py` modified +20/-0 (20 lines); hunks: -512,6 +512,25 @@ def __init__(; -522,6 +541,7 @@ def __init__(; symbols: __init__
  - `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0 (13 lines); hunks: -978,6 +978,10 @@ def kernel(; -1124,6 +1128,8 @@ def build_flashinfer_mixed_sparse_indices(; symbols: kernel, build_flashinfer_mixed_sparse_indices, _build_flashinfer_mixed_sparse_indices_kernel
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +6/-1 (7 lines); hunks: -200,8 +200,11 @@ def build(; -375,6 +378,7 @@ def _build_sparse_index_metadata(; symbols: build, _build_sparse_index_metadata
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model_state.py
@@ -2,15 +2,23 @@
+import numpy as np
-from vllm.config import VllmConfig
+from vllm.config import CUDAGraphMode, VllmConfig
+from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
+from vllm.v1.attention.backends.utils import PAD_SLOT_ID
+from vllm.v1.core.sched.output import NewRequestData
diff -- tests/models/test_deepseek_v41_replay_start.py
@@ -0,0 +1,112 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""SWA bounded replay in DeepseekV41ModelState: the batch's replay starts reach
+the sliding-window builders, and the replayed tokens' slots are padded in the
+prefix-cacheable groups only."""
+from dataclasses import replace
diff -- vllm/models/deepseek_v41/attention.py
@@ -512,6 +512,25 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +143/-2; `vllm/models/deepseek_v41/attention.py` modified +20/-0; `vllm/models/deepseek_v41/common/ops/cache_utils.py` modified +13/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +6/-1; `vllm/models/deepseek_v4/amd/rocm.py` modified +2/-0; `vllm/models/deepseek_v41/amd/rocm.py` modified +2/-0
  - tests: `tests/models/test_deepseek_v41_replay_start.py` added +112/-0
- 验证与风险: diff 自带测试面 `tests/models/test_deepseek_v41_replay_start.py`, `tests/v1/attention/test_deepseek_v4_swa_visible.py`, `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57434 - [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers

- 链接: https://github.com/vllm-project/vllm/pull/57434
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`；关联提交 `1596fa5f8825`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+66/-13，可读 patch 118 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13 (79 lines); hunks: -13,7 +13,10; -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__, get_padded_num_q_heads，涉及 `_copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13 (79 lines); hunks: -13,7 +13,10; -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(; symbols: _copy_ragged_to_graph_buffers, DeepseekV4ROCMAiterSparseSWAMetadata, __init__, get_padded_num_q_heads
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,7 +13,10 @@
-from vllm.models.deepseek_v41.attention import DeepseekV4Attention
+from vllm.models.deepseek_v41.attention import (
+    DeepseekV4Attention,
+    _replace_layer_index,
+)
@@ -391,6 +394,10 @@ def _copy_ragged_to_graph_buffers(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +66/-13
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/amd/rocm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #56625 - [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash

- 链接: https://github.com/vllm-project/vllm/pull/56625
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`, `vllm/models/deepseek_v41/nvidia/vl_model.py`；关联提交 `27757dde020e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+599/-63，可读 patch 809 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span, _get_grid_list，涉及 `DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span`；`vllm/models/deepseek_v4/common/vision.py` modified +156/-7 (163 lines); hunks: -40,8 +40,7; -52,6 +51,13 @@ def get_vision_cos_sin(; symbols: get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary，涉及 `get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary`；`vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image，涉及 `_make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input`；`vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image，涉及 `_make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV4VLEncoderCudaGraphMixin, _encode_image, _build_image_span, _get_grid_list
  - `vllm/models/deepseek_v4/common/vision.py` modified +156/-7 (163 lines); hunks: -40,8 +40,7; -52,6 +51,13 @@ def get_vision_cos_sin(; symbols: get_vision_cos_sin, _compute_vision_cos_sin, apply_rotary
  - `vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image
  - `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28 (40 lines); hunks: -26,6 +26,7; -44,15 +45,12; symbols: _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, _parse_and_validate_image_input, _encode_image
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/vl_cudagraph.py
@@ -0,0 +1,390 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Encoder CUDA graph support for the DeepSeek-V4.1 vision tower.
+Mixin implementing the ``SupportsEncoderCudaGraph`` protocol
+(``vllm/model_executor/models/interfaces.py``) on top of the shared
+``DeepseekV4ViT``/``DeepseekV4Aligner``. The captured graph packs a whole
diff -- vllm/models/deepseek_v4/common/vision.py
@@ -40,8 +40,7 @@
-@lru_cache(8)
-def get_vision_cos_sin(
+def _compute_vision_cos_sin(
@@ -52,6 +51,13 @@ def get_vision_cos_sin(
+@lru_cache(8)
+def get_vision_cos_sin(
diff -- vllm/models/deepseek_v41/amd/vl_model.py
@@ -26,6 +26,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/vl_cudagraph.py` added +390/-0; `vllm/models/deepseek_v4/common/vision.py` modified +156/-7; `vllm/models/deepseek_v41/amd/vl_model.py` modified +12/-28; `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +12/-28
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/models/multimodal/processing/test_tensor_schema.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57603 - [Perf][DSV4.1] Overlap mHC coefficients for small TP batches

- 链接: https://github.com/vllm-project/vllm/pull/57603
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`；关联提交 `9b49f9234431`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+357/-46，可读 patch 645 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: supports_mhc_overlap, mhc_pre_delayed_overlap，涉及 `supports_mhc_overlap, mhc_pre_delayed_overlap`；`vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16 (76 lines); hunks: -2,6 +2,7; -17,7 +18,11; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1 (30 lines); hunks: -7,6 +7,7; -106,10 +107,37 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre，涉及 `mhc_shifted_post_pre`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: supports_mhc_overlap, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16 (76 lines); hunks: -2,6 +2,7; -17,7 +18,11; symbols: __init__, forward
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1 (30 lines); hunks: -7,6 +7,7; -106,10 +107,37 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -0,0 +1,117 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""
+from functools import partial
+import torch
+from vllm.config import VllmConfig
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -2,6 +2,7 @@
+from functools import partial
@@ -17,7 +18,11 @@
-from vllm.forward_context import get_forward_context, is_forward_context_available
+from vllm.forward_context import (
+    get_forward_context,
+    in_piecewise_cudagraph,
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -7,6 +7,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` added +117/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +60/-16; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +29/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57491 - [ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm

- 链接: https://github.com/vllm-project/vllm/pull/57491
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/model.py`；关联提交 `6dfd87f59658`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+20/-10，可读 patch 96 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/model.py` modified +5/-1 (6 lines); hunks: -61,9 +61,13。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/model.py` modified +5/-1 (6 lines); hunks: -61,9 +61,13
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -61,9 +61,13 @@
-from ..common.engram import Engram, EngramLayout, NgramHashState
+from ..common.engram import EngramLayout, NgramHashState
+# Engram host offload and its prefetch stream are neither ROCm- nor
+# NVIDIA-specific, so they are imported rather than duplicated.
+from ..nvidia.engram import Engram
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +5/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_engram.py`, `tests/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57643 - [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation

- 链接: https://github.com/vllm-project/vllm/pull/57643
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`；关联提交 `9a70c233cd5d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+549/-105，可读 patch 890 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4 (160 lines); hunks: -1,15 +1,26; -29,6 +40,20 @@ def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:; symbols: supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap，涉及 `supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap`；`vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98 (98 lines); hunks: -5,10 +5,6; -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(; symbols: mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre，涉及 `mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre`；`vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2 (21 lines); hunks: -17,6 +17,7; -88,10 +89,11; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0 (3 lines); hunks: -793,6 +793,7 @@ def __init__(; -804,6 +805,7 @@ def __init__(; symbols: __init__, _init_fused_moe_experts，涉及 `__init__, _init_fused_moe_experts`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4 (160 lines); hunks: -1,15 +1,26; -29,6 +40,20 @@ def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:; symbols: supports_mhc_overlap, supports_mhc_all_reduce, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98 (98 lines); hunks: -5,10 +5,6; -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(; symbols: mhc_shifted_post_pre_deep_gemm, mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2 (21 lines); hunks: -17,6 +17,7; -88,10 +89,11; symbols: __init__, forward
  - `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0 (3 lines); hunks: -793,6 +793,7 @@ def __init__(; -804,6 +805,7 @@ def __init__(; symbols: __init__, _init_fused_moe_experts
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -1,15 +1,26 @@
-"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""
+"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""
+from typing import TYPE_CHECKING, cast
+from vllm.distributed import get_tp_group
+from vllm.model_executor.kernels.mhc.tilelang import (
+    mhc_fused_post_pre_delayed_tilelang,
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -5,10 +5,6 @@
-from vllm.model_executor.kernels.mhc.tilelang import (
-    mhc_fused_post_pre_delayed_tilelang,
-    mhc_post_tilelang,
-)
@@ -88,97 +84,3 @@ def mhc_shifted_post_pre_deep_gemm(
-def mhc_shifted_post_pre(
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -17,6 +17,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +156/-4; `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +0/-98; `vllm/models/deepseek_v41/nvidia/model.py` modified +19/-2; `vllm/models/deepseek_v4/nvidia/model.py` modified +3/-0
- 验证与风险: diff 自带测试面 `tests/distributed/test_custom_all_reduce.py`, `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57874 - [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs

- 链接: https://github.com/vllm-project/vllm/pull/57874
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`；关联提交 `67513c8b67e2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+45/-11，可读 patch 108 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0 (16 lines); hunks: -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) ->...; symbols: is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm，涉及 `is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm`；`vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8 (14 lines); hunks: -16,7 +16,10; -223,13 +226,8 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre，涉及 `mhc_shifted_post_pre`；`vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1 (3 lines); hunks: -385,8 +385,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0 (16 lines); hunks: -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) ->...; symbols: is_mega_mhc_supported, can_use_mega_mhc, mhc_shifted_post_pre_deep_gemm
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8 (14 lines); hunks: -16,7 +16,10; -223,13 +226,8 @@ def mhc_shifted_post_pre(; symbols: mhc_shifted_post_pre
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1 (3 lines); hunks: -385,8 +385,9 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py
@@ -28,6 +28,22 @@ def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
+def can_use_mega_mhc(
+    x: torch.Tensor,
+    residual: torch.Tensor,
+    pre_mix: torch.Tensor | None,
+    norm_weight: torch.Tensor | None,
+    capture_aux: bool,
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -16,7 +16,10 @@
-from .mega_mhc import is_mega_mhc_supported, mhc_shifted_post_pre_deep_gemm
+from .mega_mhc import (
+    can_use_mega_mhc,
+    mhc_shifted_post_pre_deep_gemm,
+)
@@ -223,13 +226,8 @@ def mhc_shifted_post_pre(
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -385,8 +385,9 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` modified +16/-0; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +6/-8; `vllm/models/deepseek_v41/nvidia/model.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57906 - [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm

- 链接: https://github.com/vllm-project/vllm/pull/57906
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/attention.py`；关联提交 `04c1f4a40796`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+9/-0，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/attention.py` modified +9/-0 (9 lines); hunks: -531,6 +531,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/attention.py` modified +9/-0 (9 lines); hunks: -531,6 +531,15 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/attention.py
@@ -531,6 +531,15 @@ def __init__(
+        if swa_bounded_replay and current_platform.is_rocm():
+            logger.warning_once(
+                "SWA bounded replay is off on ROCm (the sparse SWA metadata "
+                "builders forward replay_start, but the window clamp it relies "
+                "on lives in the FlashInfer and FlashMLA prefill kernels, so "
+                "the padded slots fault); the sliding-window cache takes part "
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/attention.py` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #57428 - [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter

- 链接: https://github.com/vllm-project/vllm/pull/57428
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`, `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/dspark.py`, `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` 等 7 个文件；关联提交 `f92b78f6ef9c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+1036/-623，可读 patch 2293 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44 (57 lines); hunks: -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_seq...; -276,7 +245,7 @@ def __init__(; symbols: maybe_init_gemm_rs_ar, __init__, forward，涉及 `maybe_init_gemm_rs_ar, __init__, forward`；`vllm/models/deepseek_v41/attention.py` modified +40/-0 (40 lines); hunks: -26,13 +26,15; -389,6 +391,9 @@ def __init__(; symbols: __init__, forward, bind_gemm_rs, _wo_b_proj，涉及 `__init__, forward, bind_gemm_rs`；`vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1 (38 lines); hunks: -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:; -202,6 +226,7 @@ def __init__(; symbols: _use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer, __init__，涉及 `_use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer`；`vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2 (8 lines); hunks: -1,5 +1,7; -31,7 +33,7 @@ def deep_gemm_fp8_o_proj(; symbols: deep_gemm_fp8_o_proj，涉及 `deep_gemm_fp8_o_proj`。
- 代码 diff 细节:
  - `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44 (57 lines); hunks: -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_seq...; -276,7 +245,7 @@ def __init__(; symbols: maybe_init_gemm_rs_ar, __init__, forward
  - `vllm/models/deepseek_v41/attention.py` modified +40/-0 (40 lines); hunks: -26,13 +26,15; -389,6 +391,9 @@ def __init__(; symbols: __init__, forward, bind_gemm_rs, _wo_b_proj
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1 (38 lines); hunks: -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:; -202,6 +226,7 @@ def __init__(; symbols: _use_sequence_parallel, maybe_init_gemm_rs, DeepseekV4DecoderLayer, __init__
  - `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2 (8 lines); hunks: -1,5 +1,7; -31,7 +33,7 @@ def deep_gemm_fp8_o_proj(; symbols: deep_gemm_fp8_o_proj
  - `vllm/models/deepseek_v41/nvidia/dspark.py` modified +7/-0 (7 lines); hunks: -58,6 +58,7; -113,12 +114,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/models/kimi_k3/nvidia/model.py
@@ -160,54 +160,23 @@ def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_sequence_parallel: bool)
-    enabled = envs.VLLM_KIMI_K3_GEMM_AR if all_reduce else envs.VLLM_KIMI_K3_GEMM_RS
+    enabled = envs.VLLM_KIMI_K3_GEMM_AR if all_reduce else envs.VLLM_ENABLE_GEMM_RS
-    parallel_config = vllm_config.parallel_config
-    tp_size = parallel_config.tensor_parallel_size
-    if parallel_config.use_ubatching:
-        reason = "ubatching is enabled"
diff -- vllm/models/deepseek_v41/attention.py
@@ -26,13 +26,15 @@
+from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
+    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import GemmRsAr
@@ -389,6 +391,9 @@ def __init__(
+        # Set by ``bind_gemm_rs`` when the decoder layer runs sequence
+        # parallel and the fused GEMM + reduce-scatter kernel accepts wo_b.
+        self.gemm_rs: GemmRsAr | None = None
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -193,6 +193,30 @@ def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/kimi_k3/nvidia/model.py` modified +13/-44; `vllm/models/deepseek_v41/attention.py` modified +40/-0; `vllm/models/deepseek_v41/nvidia/model.py` modified +37/-1; `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` modified +6/-2; `vllm/models/deepseek_v41/nvidia/dspark.py` modified +7/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/kernels/test_gemm_rs_ar.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57435 - [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce

- 链接: https://github.com/vllm-project/vllm/pull/57435
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`；关联提交 `ca831d1c5571`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+184/-13，可读 patch 412 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3 (25 lines); hunks: -37,6 +37,7; -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.T...; symbols: _o_proj, forward_mqa, _decode_topk_ragged, _forward_decode，涉及 `_o_proj, forward_mqa, _decode_topk_ragged`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3 (25 lines); hunks: -37,6 +37,7; -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.T...; symbols: _o_proj, forward_mqa, _decode_topk_ragged, _forward_decode
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -37,6 +37,7 @@
+    rocm_inverse_rope_rows_,
@@ -673,6 +674,7 @@ def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tens
+            inverse_rope=False,
@@ -747,15 +749,28 @@ def forward_mqa(
+        rotated = 0
-            self._forward_decode(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +22/-3
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58456 - [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM

- 链接: https://github.com/vllm-project/vllm/pull/58456
- 状态/时间: merged / 2026-09-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`；关联提交 `721d0e5c112a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+664/-29，可读 patch 957 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22 (158 lines); hunks: -13,6 +13,8; -37,7 +39,9; symbols: _split_qkv_and_norm, _o_proj, _alloc_attn_out，涉及 `_split_qkv_and_norm, _o_proj, _alloc_attn_out`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22 (158 lines); hunks: -13,6 +13,8; -37,7 +39,9; symbols: _split_qkv_and_norm, _o_proj, _alloc_attn_out
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -13,6 +13,8 @@
+from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
+from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
@@ -37,7 +39,9 @@
+    rocm_inverse_rope_mxfp8_rows,
+    rocm_mxfp8_wo_a_bmm,
@@ -663,7 +667,40 @@ def _split_qkv_and_norm(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +136/-22
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57679 - [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path

- 链接: https://github.com/vllm-project/vllm/pull/57679
- 状态/时间: merged / 2026-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/common/ops/query_quant.py`；关联提交 `609731e8f56c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+8/-3，可读 patch 40 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3 (11 lines); hunks: -3,7 +3,10; -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(; symbols: fused_q_kv_rmsnorm_quant, can_fuse_query_quant，涉及 `fused_q_kv_rmsnorm_quant, can_fuse_query_quant`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3 (11 lines); hunks: -3,7 +3,10; -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(; symbols: fused_q_kv_rmsnorm_quant, can_fuse_query_quant
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/ops/query_quant.py
@@ -3,7 +3,10 @@
-from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
+from vllm.model_executor.layers.fusion.quant_activation import (
+    QuantizedActivation,
+    get_input_quant_key,
+)
@@ -102,6 +105,7 @@ def fused_q_kv_rmsnorm_quant(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/ops/query_quant.py` modified +8/-3
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/common/ops/query_quant.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58678 - [Perf][DSv4.1] Shard the Engram wkv projection across TP ranks

- 链接: https://github.com/vllm-project/vllm/pull/58678
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/common/engram.py`；关联提交 `4bb804cc5bf0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+7/-3，可读 patch 37 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/engram.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -923,13 +923,17 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tests/kernels/test_engram.py` modified +1/-1 (2 lines); hunks: -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, back...; symbols: test_engram_constructor_honors_offload，涉及 `test_engram_constructor_honors_offload`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/engram.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -923,13 +923,17 @@ def __init__(; symbols: __init__
  - `tests/kernels/test_engram.py` modified +1/-1 (2 lines); hunks: -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, back...; symbols: test_engram_constructor_honors_offload
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/engram.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import ColumnParallelLinear
@@ -923,13 +923,17 @@ def __init__(
-        self.wkv = ReplicatedLinear(
+        # Without sequence parallelism every TP rank holds every token, so
+        # shard the output columns instead of replicating the projection.
diff -- tests/kernels/test_engram.py
@@ -827,7 +827,7 @@ def test_engram_constructor_honors_offload(monkeypatch, backend, cpu_offload):
-        engram_ops, "ReplicatedLinear", lambda *a, **k: torch.nn.Identity()
+        engram_ops, "ColumnParallelLinear", lambda *a, **k: torch.nn.Identity()
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/engram.py` modified +6/-2
  - tests: `tests/kernels/test_engram.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_engram.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57071 - [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support

- 链接: https://github.com/vllm-project/vllm/pull/57071
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/quant_config.py`, `vllm/models/deepseek_v41/amd/model.py`, `vllm/models/deepseek_v41/amd/vl_model.py`, `vllm/models/deepseek_v41/quant_config.py`；关联提交 `5840d95284fe`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+647/-102，可读 patch 1009 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8 (34 lines); hunks: -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):; -131,6 +133,22 @@ class DeepseekV41ForCausalLM(; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, attributes，涉及 `DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM`；`vllm/models/deepseek_v41/quant_config.py` modified +9/-7 (16 lines); hunks: -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method，涉及 `_is_quark_mxfp4_ocp, override_quantization_method`；`vllm/models/deepseek_v4/quant_config.py` modified +7/-7 (14 lines); hunks: -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method，涉及 `_is_quark_mxfp4_ocp, override_quantization_method`；`vllm/models/deepseek_v41/amd/model.py` modified +10/-0 (10 lines); hunks: -895,6 +895,11 @@ def _make_deepseek_v4_weights_mapper(; -908,6 +913,11 @@ def _make_deepseek_v4_weights_mapper(; symbols: _make_deepseek_v4_weights_mapper，涉及 `_make_deepseek_v4_weights_mapper`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8 (34 lines); hunks: -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):; -131,6 +133,22 @@ class DeepseekV41ForCausalLM(; symbols: DeepseekV4VLImagePixelInputs, _make_deepseek_v4_vl_weights_mapper, DeepseekV41ForCausalLM, attributes
  - `vllm/models/deepseek_v41/quant_config.py` modified +9/-7 (16 lines); hunks: -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method
  - `vllm/models/deepseek_v4/quant_config.py` modified +7/-7 (14 lines); hunks: -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:; symbols: _is_quark_mxfp4_ocp, override_quantization_method
  - `vllm/models/deepseek_v41/amd/model.py` modified +10/-0 (10 lines); hunks: -895,6 +895,11 @@ def _make_deepseek_v4_weights_mapper(; -908,6 +913,11 @@ def _make_deepseek_v4_weights_mapper(; symbols: _make_deepseek_v4_weights_mapper
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/vl_model.py
@@ -79,20 +79,22 @@ class DeepseekV4VLImagePixelInputs(TensorSchema):
+_VL_PREFIX_MAPPING: dict[str, str | None] = {
+    "layers.": "language_model.model.layers.",
+    "embed.": "language_model.model.embed.",
+    "norm.": "language_model.model.norm.",
+    "hc_head": "language_model.model.hc_head",
+    "mtp.": "language_model.model.mtp.",
diff -- vllm/models/deepseek_v41/quant_config.py
@@ -139,15 +139,17 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
+        # Quark checkpoints are always handled by QuarkConfig, which knows
+        # how to dispatch each per-layer scheme (MXFP4, MXFP8, etc.).
+        # ``from_config`` below rewrites the config into a single global FP8
+        # scheme, which would discard per-layer specs — so never claim Quark.
+        if isinstance(hf_quant_cfg, dict) and (
+            hf_quant_cfg.get("quant_method") == "quark"
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -135,15 +135,15 @@ def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/vl_model.py` modified +26/-8; `vllm/models/deepseek_v41/quant_config.py` modified +9/-7; `vllm/models/deepseek_v4/quant_config.py` modified +7/-7; `vllm/models/deepseek_v41/amd/model.py` modified +10/-0
- 验证与风险: diff 自带测试面 `tests/quantization/test_fp8.py`, `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58316 - [Bugfix][Frontend][Rust Frontend] Update DeepSeek V4.1 Flash reasoning effort mappings

- 链接: https://github.com/vllm-project/vllm/pull/58316
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`, `vllm/tokenizers/deepseek_v41_encoding.py`；关联提交 `ddd6fbca148a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+20/-20，可读 patch 119 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4 (8 lines); hunks: -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):; -138,7 +138,7 @@ def test_top_level_effort_overrides_template_effort():; symbols: test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation，涉及 `test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation`；`vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2 (4 lines); hunks: -181,8 +181,8 @@ def sort_tool_results_by_call_order(; symbols: sort_tool_results_by_call_order，涉及 `sort_tool_results_by_call_order`；`tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4；`tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1 (2 lines); hunks: -1,3 +1,3。
- 代码 diff 细节:
  - `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4 (8 lines); hunks: -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):; -138,7 +138,7 @@ def test_top_level_effort_overrides_template_effort():; symbols: test_reference_encoder_fixtures, test_top_level_effort_overrides_template_effort, test_encode_uses_one_bos_and_forwards_truncation
  - `vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2 (4 lines); hunks: -181,8 +181,8 @@ def sort_tool_results_by_call_order(; symbols: sort_tool_results_by_call_order
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4
  - `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1 (2 lines); hunks: -1,3 +1,3
  - `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_1.txt` modified +1/-1 (2 lines); hunks: -1,4 +1,4
- 关键代码摘录:

```diff
diff -- tests/tokenizers_/test_deepseek_v41.py
@@ -35,9 +35,9 @@ def test_reference_encoder_fixtures(case_id):
-        (None, 50),
-        ("low", 25),
-        ("high", 50),
+        (None, 75),
+        ("low", 50),
+        ("high", 75),
diff -- vllm/tokenizers/deepseek_v41_encoding.py
@@ -181,8 +181,8 @@ def sort_tool_results_by_call_order(
-    "low": 25,
-    "high": 50,
+    "low": 50,
+    "high": 75,
diff -- tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt
@@ -1,4 +1,4 @@
-<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)
+<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)
```

- 提取文件（未人工审阅）:
  - tests: `tests/tokenizers_/test_deepseek_v41.py` modified +4/-4; `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt` modified +1/-1; `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt` modified +1/-1
  - runtime: `vllm/tokenizers/deepseek_v41_encoding.py` modified +2/-2
  - other: `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_1.txt` modified +1/-1; `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_2.txt` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/tokenizers_/fixtures/deepseek_v41/test_output_1.txt`, `tests/tokenizers_/fixtures/deepseek_v41/test_output_2.txt`, `tests/tokenizers_/test_deepseek_v41.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58499 - [Bugfix][DSV4.1] Avoid host sync in ViT CUDA graph replay metadata

- 链接: https://github.com/vllm-project/vllm/pull/58499
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`；关联提交 `7d8c5fe9a95d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+24/-15，可读 patch 92 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8 (22 lines); hunks: -25,6 +25,7; -370,18 +371,23 @@ def postprocess_encoder_output(; symbols: postprocess_encoder_output，涉及 `postprocess_encoder_output`；`vllm/models/deepseek_v4/common/vision.py` modified +10/-7 (17 lines); hunks: -38,6 +38,7; -219,8 +220,8 @@ def forward(; symbols: _compute_vision_cos_sin, forward, build_packed_vit_metadata, build_packed_merge_metadata，涉及 `_compute_vision_cos_sin, forward, build_packed_vit_metadata`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8 (22 lines); hunks: -25,6 +25,7; -370,18 +371,23 @@ def postprocess_encoder_output(; symbols: postprocess_encoder_output
  - `vllm/models/deepseek_v4/common/vision.py` modified +10/-7 (17 lines); hunks: -38,6 +38,7; -219,8 +220,8 @@ def forward(; symbols: _compute_vision_cos_sin, forward, build_packed_vit_metadata, build_packed_merge_metadata
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/vl_cudagraph.py
@@ -25,6 +25,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -370,18 +371,23 @@ def postprocess_encoder_output(
-        types = batch_mm_kwargs["types"].to(aligner_out.device)
+        types = batch_mm_kwargs["types"]
-        # Batched span assembly: one masked fill per role across all items.
-        dtype = aligner_out.dtype
diff -- vllm/models/deepseek_v4/common/vision.py
@@ -38,6 +38,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -219,8 +220,8 @@ def forward(
-        cos = cos.to(device=x.device)
-        sin = sin.to(device=x.device)
+        cos = async_tensor_h2d(cos, x.device)
+        sin = async_tensor_h2d(sin, x.device)
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/vl_cudagraph.py` modified +14/-8; `vllm/models/deepseek_v4/common/vision.py` modified +10/-7
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/common/vision.py`, `vllm/models/deepseek_v41/common/vl_cudagraph.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58586 - [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary

- 链接: https://github.com/vllm-project/vllm/pull/58586
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`, `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py`, `vllm/models/deepseek_v41/nvidia/ops/mhc.py`；关联提交 `77871126f9b6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+1807/-309，可读 patch 2312 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0 (1134 lines); hunks: -0,0 +1,1134; symbols: _group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__, __call__，涉及 `_group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__`；`vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: _asm, load_global_u32x4, load_global_u32x2, load_global_bf16_as_f32，涉及 `_asm, load_global_u32x4, load_global_u32x2`；`vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18 (91 lines); hunks: -9,10 +9,15; -24,9 +29,17; symbols: supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce, mhc_pre_delayed_overlap，涉及 `supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce`；`vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4 (80 lines); hunks: -33,6 +33,14; -94,6 +102,7; symbols: __init__, defer_finalize, defers_finalize, forward_unfinalized，涉及 `__init__, defer_finalize, defers_finalize`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0 (1134 lines); hunks: -0,0 +1,1134; symbols: _group_leader_block_sum, _SharedOnlyPublishDeviceKernel, __init__, __call__
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: _asm, load_global_u32x4, load_global_u32x2, load_global_bf16_as_f32
  - `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18 (91 lines); hunks: -9,10 +9,15; -24,9 +29,17; symbols: supports_mhc_overlap, supports_mhc_all_reduce, init_mhc_all_reduce, mhc_pre_delayed_overlap
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4 (80 lines); hunks: -33,6 +33,14; -94,6 +102,7; symbols: __init__, defer_finalize, defers_finalize, forward_unfinalized
  - `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py
@@ -0,0 +1,1134 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from FlashInfer's low-latency MNNVL CuTe DSL all-reduce
+# (flashinfer/comm/mnnvl_cutedsl/kernel_ll, flashinfer-ai/flashinfer@139af6f6).
+"""Lamport all-reduce fused with DSV4.1's mHC post, collapse and RMSNorm.
+FlashInfer's LL kernels, copied: a publish kernel (or its MoE finalize
diff -- vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py
@@ -0,0 +1,396 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from FlashInfer's flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py
+# and runtime.py (flashinfer-ai/flashinfer@139af6f6).
+"""PTX building blocks and launch helpers for the DSV4.1 LL all-reduce."""
+import cuda.bindings.driver as cuda
diff -- vllm/models/deepseek_v41/nvidia/ops/mhc.py
@@ -9,10 +9,15 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` added +1134/-0; `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py` added +396/-0; `vllm/models/deepseek_v41/nvidia/ops/mhc.py` modified +73/-18; `vllm/models/deepseek_v41/nvidia/model.py` modified +76/-4; `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py` added +8/-0
- 验证与风险: diff 自带测试面 `tests/distributed/test_custom_all_reduce.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58634 - [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103

- 链接: https://github.com/vllm-project/vllm/pull/58634
- 状态/时间: merged / 2026-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/attention.py`, `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`, `vllm/models/deepseek_v41/nvidia/flashmla.py`, `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py`, `vllm/models/deepseek_v41/nvidia/ops/o_proj.py`；关联提交 `a9eafde59cbd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+759/-55，可读 patch 899 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _ue8m0_rp, _rope_fma, _scale, FusedWoAKernel，涉及 `_ue8m0_rp, _rope_fma, _scale`；`vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: _fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup, dsv41_o_proj，涉及 `_fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup`；`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34 (42 lines); hunks: -9,15 +9,16; -242,27 +243,14 @@ def get_padded_num_q_heads(cls, num_heads: int) -> int:; symbols: get_padded_num_q_heads, _o_proj, __init__，涉及 `get_padded_num_q_heads, _o_proj, __init__`；`vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19 (26 lines); hunks: -6,16 +6,17; -57,23 +58,10 @@ def __init__(self, *args, **kwargs) -> None:; symbols: __init__, _o_proj, get_padded_num_q_heads，涉及 `__init__, _o_proj, get_padded_num_q_heads`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0 (599 lines); hunks: -0,0 +1,599; symbols: _ue8m0_rp, _rope_fma, _scale, FusedWoAKernel
  - `vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: _fused_wo_a_max_tokens, _can_fuse_wo_a, register_dsv41_o_proj_warmup, dsv41_o_proj
  - `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34 (42 lines); hunks: -9,15 +9,16; -242,27 +243,14 @@ def get_padded_num_q_heads(cls, num_heads: int) -> int:; symbols: get_padded_num_q_heads, _o_proj, __init__
  - `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19 (26 lines); hunks: -6,16 +6,17; -57,23 +58,10 @@ def __init__(self, *args, **kwargs) -> None:; symbols: __init__, _o_proj, get_padded_num_q_heads
  - `vllm/models/deepseek_v41/attention.py` modified +2/-2 (4 lines); hunks: -697,7 +697,7 @@ def bind_gemm_rs(self) -> None:; -707,7 +707,7 @@ def _wo_b_proj(self, z: torch.Tensor) -> torch.Tensor:; symbols: bind_gemm_rs, _wo_b_proj
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py
@@ -0,0 +1,599 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSV4.1 small-batch MXFP8 WO-A with inverse RoPE and quantization."""
+from dataclasses import dataclass
+from typing import Any
+import cutlass
diff -- vllm/models/deepseek_v41/nvidia/ops/o_proj.py
@@ -0,0 +1,95 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""DSV4.1 output projection with a small-batch SM100/SM103 fusion."""
+import torch
+from torch import nn
+from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
diff -- vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py
@@ -9,15 +9,16 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py` added +599/-0; `vllm/models/deepseek_v41/nvidia/ops/o_proj.py` added +95/-0; `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` modified +8/-34; `vllm/models/deepseek_v41/nvidia/flashmla.py` modified +7/-19; `vllm/models/deepseek_v41/attention.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/kernels/test_fused_inv_rope_fp8_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57407 - [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash

- 链接: https://github.com/vllm-project/vllm/pull/57407
- 状态/时间: merged / 2026-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`；关联提交 `44af287ebe38`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+496/-20，可读 patch 592 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0 (189 lines); hunks: -29,6 +29,7; -529,6 +530,9 @@ def __init__(self, *args, **kwargs):; symbols: __init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm, _run_parallel_input_projections，涉及 `__init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm`；`vllm/models/deepseek_v41/attention.py` modified +32/-18 (50 lines); hunks: -478,7 +478,6 @@ def __init__(; -1381,6 +1380,37 @@ def _produce_k(; symbols: __init__, _produce_k, forward_q, forward，涉及 `__init__, _produce_k, forward_q`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0 (189 lines); hunks: -29,6 +29,7; -529,6 +530,9 @@ def __init__(self, *args, **kwargs):; symbols: __init__, get_padded_num_q_heads, _fused_wqa_wkv_gemm, _run_parallel_input_projections
  - `vllm/models/deepseek_v41/attention.py` modified +32/-18 (50 lines); hunks: -478,7 +478,6 @@ def __init__(; -1381,6 +1380,37 @@ def _produce_k(; symbols: __init__, _produce_k, forward_q, forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -29,6 +29,7 @@
+from vllm.utils.multi_stream_utils import execute_in_parallel
@@ -529,6 +530,9 @@ def __init__(self, *args, **kwargs):
+        if self.compressor is None and self.indexer is None:
+            # Dense layers have nothing to overlap; keep the base serial path.
+            self.aux_stream_list = None
@@ -600,8 +604,193 @@ def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
diff -- vllm/models/deepseek_v41/attention.py
@@ -478,7 +478,6 @@ def __init__(
-        # Will be None on ROCm for now.
@@ -1381,6 +1380,37 @@ def _produce_k(
+    def forward_q(
+        self,
+        qr: torch.Tensor | QuantizedActivation,
+        qr_scale: torch.Tensor | None,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +189/-0; `vllm/models/deepseek_v41/attention.py` modified +32/-18
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58983 - [ROCm][Refactor] Move DeepSeek-V4/V4.1 multi-stream overlap gate to ROCm platform

- 链接: https://github.com/vllm-project/vllm/pull/58983
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`；关联提交 `7ba3df63cbe2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+40/-37，可读 patch 126 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19 (25 lines); hunks: -649,23 +649,6 @@ def __init__(self, *args, **kwargs):; -709,7 +692,9 @@ def forward(; symbols: __init__, _enable_multi_stream_overlap, _run_sequential_pipeline, forward，涉及 `__init__, _enable_multi_stream_overlap, _run_sequential_pipeline`；`vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18 (21 lines); hunks: -615,23 +615,6 @@ def _run_parallel_input_projections(; -641,7 +624,9 @@ def forward(; symbols: _run_parallel_input_projections, _enable_multi_stream_overlap, forward，涉及 `_run_parallel_input_projections, _enable_multi_stream_overlap, forward`；`vllm/platforms/rocm.py` modified +21/-0 (21 lines); hunks: -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:; symbols: opaque_attention_op, is_navi, enable_multi_stream_overlap, get_static_graph_wrapper_cls，涉及 `opaque_attention_op, is_navi, enable_multi_stream_overlap`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19 (25 lines); hunks: -649,23 +649,6 @@ def __init__(self, *args, **kwargs):; -709,7 +692,9 @@ def forward(; symbols: __init__, _enable_multi_stream_overlap, _run_sequential_pipeline, forward
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18 (21 lines); hunks: -615,23 +615,6 @@ def _run_parallel_input_projections(; -641,7 +624,9 @@ def forward(; symbols: _run_parallel_input_projections, _enable_multi_stream_overlap, forward
  - `vllm/platforms/rocm.py` modified +21/-0 (21 lines); hunks: -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:; symbols: opaque_attention_op, is_navi, enable_multi_stream_overlap, get_static_graph_wrapper_cls
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v4/amd/rocm.py
@@ -649,23 +649,6 @@ def __init__(self, *args, **kwargs):
-    def _enable_multi_stream_overlap(self) -> bool:
-        """ROCm multi-stream gates: streams and capture region.
-        Dict metadata marks piecewise cudagraph, whose eager breaks rebuild
-        the attention inputs on the owning stream. Forking side streams
-        there would rely on runtime HIP event sync, which is unreliable in
-        this overlap on ROCm (event waits can hang), so multi-stream only
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -615,23 +615,6 @@ def _run_parallel_input_projections(
-    def _enable_multi_stream_overlap(self) -> bool:
-        """ROCm multi-stream gates: streams and capture region.
-        Dict metadata marks piecewise cudagraph, whose eager breaks rebuild
-        the attention inputs on the owning stream. Forking side streams
-        there would rely on runtime HIP event sync, which is unreliable in
-        this overlap on ROCm (event waits can hang), so multi-stream only
diff -- vllm/platforms/rocm.py
@@ -1106,6 +1106,27 @@ def opaque_attention_op(cls) -> bool:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v4/amd/rocm.py` modified +6/-19; `vllm/models/deepseek_v41/amd/rocm.py` modified +3/-18; `vllm/platforms/rocm.py` modified +21/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v4/amd/rocm.py`, `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/platforms/interface.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58655 - [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel

- 链接: https://github.com/vllm-project/vllm/pull/58655
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/model.py`；关联提交 `0af34418e99b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+324/-16，可读 patch 473 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/model.py` modified +29/-4 (33 lines); hunks: -22,7 +22,11; -251,6 +255,10 @@ def __init__(; symbols: __init__, _hc_collapse, forward，涉及 `__init__, _hc_collapse, forward`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/model.py` modified +29/-4 (33 lines); hunks: -22,7 +22,11; -251,6 +255,10 @@ def __init__(; symbols: __init__, _hc_collapse, forward
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/model.py
@@ -22,7 +22,11 @@
-from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp
+from vllm.model_executor.layers.mhc import (
+    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
+    MHCPostOp,
+    MHCPreDelayedOp,
+)
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/model.py` modified +29/-4
- 验证与风险: diff 自带测试面 `tests/kernels/mhc/test_mhc_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58671 - [ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel

- 链接: https://github.com/vllm-project/vllm/pull/58671
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`；关联提交 `63b9931b7399`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+2276/-9，可读 patch 2437 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1 (150 lines); hunks: -2,8 +2,9; -15,8 +16,14; symbols: build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store, rocm_mxfp4_indexer_q_quant，涉及 `build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store`；`vllm/models/deepseek_v41/attention.py` modified +34/-6 (40 lines); hunks: -35,6 +35,9; -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:; symbols: _uses_fp8_ds_mla_layout, _indexer_cls, __init__，涉及 `_uses_fp8_ds_mla_layout, _indexer_cls, __init__`；`vllm/config/attention.py` modified +4/-1 (5 lines); hunks: -93,7 +93,10 @@ class AttentionConfig:; symbols: AttentionConfig, GPU，涉及 `AttentionConfig, GPU`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1 (150 lines); hunks: -2,8 +2,9; -15,8 +16,14; symbols: build, _aiter_indexer_cache_ops, rocm_mxfp4_indexer_k_store, rocm_mxfp4_indexer_q_quant
  - `vllm/models/deepseek_v41/attention.py` modified +34/-6 (40 lines); hunks: -35,6 +35,9; -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:; symbols: _uses_fp8_ds_mla_layout, _indexer_cls, __init__
  - `vllm/config/attention.py` modified +4/-1 (5 lines); hunks: -93,7 +93,10 @@ class AttentionConfig:; symbols: AttentionConfig, GPU
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -2,8 +2,9 @@
+from collections.abc import Callable
-from typing import cast
+from typing import Any, cast
@@ -15,8 +16,14 @@
+from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (
+    RocmSparseAttnIndexer,
diff -- vllm/models/deepseek_v41/attention.py
@@ -35,6 +35,9 @@
+    from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (
+        RocmSparseMQAIndexer,
+    )
@@ -261,6 +264,13 @@ def _uses_fp8_ds_mla_layout(self) -> bool:
+    def _indexer_cls(
+        self, k_cache: "DeepseekV4IndexerCache | None"
diff -- vllm/config/attention.py
@@ -93,7 +93,10 @@ class AttentionConfig:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +149/-1; `vllm/models/deepseek_v41/attention.py` modified +34/-6; `vllm/config/attention.py` modified +4/-1
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py`, `tests/v1/attention/test_rocm_paged_mxfp4_indexer_plan.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #59119 - [DSv4.1] Avoid runtime recompiles of _ring_slot_mapping_kernel

- 链接: https://github.com/vllm-project/vllm/pull/59119
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/compressor.py`；关联提交 `30f5c01d4ebb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/compressor.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ class CompressorMetadata:; symbols: CompressorMetadata, _ring_slot_mapping_kernel，涉及 `CompressorMetadata, _ring_slot_mapping_kernel`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/compressor.py` modified +1/-1 (2 lines); hunks: -57,7 +57,7 @@ class CompressorMetadata:; symbols: CompressorMetadata, _ring_slot_mapping_kernel
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/compressor.py
@@ -57,7 +57,7 @@ class CompressorMetadata:
-@triton.jit
+@triton.jit(do_not_specialize=["block_table_stride", "num_actual_tokens", "num_tokens"])
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/compressor.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/compressor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58132 - [Model] Decoder-side SWA bounded replay for DeepSeek-V4.1

- 链接: https://github.com/vllm-project/vllm/pull/58132
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/test_deepseek_v41_decoder_replay_layers.py`, `tests/models/test_deepseek_v41_replay_batch.py`, `vllm/models/deepseek_v41/decoder_replay_layers.py`, `vllm/models/deepseek_v41/nvidia/model.py`, `vllm/models/deepseek_v41/nvidia/model_state.py` 等 6 个文件；关联提交 `4b2e1cfa7b80`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+1001/-60，可读 patch 1223 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2 (323 lines); hunks: -1,17 +1,24; -81,6 +88,61 @@ def _pad_replayed_slots_kernel(; symbols: _pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata, DeepseekV41ModelState，涉及 `_pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata`；`vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40 (262 lines); hunks: -83,6 +83,7; -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, _mega_gate_metadata, _run_layers，涉及 `__init__, forward, _mega_gate_metadata`；`tests/models/test_deepseek_v41_replay_batch.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: state, prepare_attn, _input_batch, _slot_mappings，涉及 `state, prepare_attn, _input_batch`；`tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _context, _set_replay_batch, _states, test_run_gathers_states_and_realigns_shared_indexer_buffers，涉及 `_context, _set_replay_batch, _states`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2 (323 lines); hunks: -1,17 +1,24; -81,6 +88,61 @@ def _pad_replayed_slots_kernel(; symbols: _pad_replayed_slots_kernel, _gather_replay_batch_kernel, ReplayAttnMetadata, DeepseekV41ModelState
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40 (262 lines); hunks: -83,6 +83,7; -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, _mega_gate_metadata, _run_layers
  - `tests/models/test_deepseek_v41_replay_batch.py` added +259/-0 (259 lines); hunks: -0,0 +1,259; symbols: state, prepare_attn, _input_batch, _slot_mappings
  - `tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _context, _set_replay_batch, _states, test_run_gathers_states_and_realigns_shared_indexer_buffers
  - `vllm/models/deepseek_v41/decoder_replay_layers.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: DecoderReplayLayers, __init__, __call__
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/nvidia/model_state.py
@@ -1,17 +1,24 @@
+from dataclasses import replace
+import torch.distributed as dist
+from vllm.distributed.parallel_state import get_dp_group
+from vllm.forward_context import DPMetadata, create_forward_context
+from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
+from vllm.v1.worker.dp_utils import should_skip_dp_coordination
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -83,6 +83,7 @@
+from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
@@ -741,6 +742,35 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Decoder-side SWA bounded replay: in eager prefill steps the layers past
+        # the last KV source run on each request's trailing window only
+        # (decoder_replay_layers.py).
+        self.decoder_replay_layers: DecoderReplayLayers | None = None
diff -- tests/models/test_deepseek_v41_replay_batch.py
@@ -0,0 +1,259 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/nvidia/model_state.py` modified +321/-2; `vllm/models/deepseek_v41/nvidia/model.py` modified +222/-40; `vllm/models/deepseek_v41/decoder_replay_layers.py` added +67/-0; `vllm/models/deepseek_v41/nvidia/vl_model.py` modified +5/-0
  - tests: `tests/models/test_deepseek_v41_replay_batch.py` added +259/-0; `tests/models/test_deepseek_v41_decoder_replay_layers.py` added +104/-0
- 验证与风险: diff 自带测试面 `tests/distributed/test_engram_dp_shard.py`, `tests/models/test_deepseek_v41_decoder_replay_layers.py`, `tests/models/test_deepseek_v41_replay_batch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57898 - [Bugfix] Profile maximum DeepSeek V4.1 vision features

- 链接: https://github.com/vllm-project/vllm/pull/57898
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/common/mm_preprocess.py`；关联提交 `ac7f3e11ea88`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+21/-8，可读 patch 37 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8 (29 lines); hunks: -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder，涉及 `get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8 (29 lines); hunks: -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:; symbols: get_image_size_with_most_features, DeepseekV4VLDummyInputsBuilder
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/mm_preprocess.py
@@ -275,15 +275,28 @@ def get_image_size_with_most_features(self) -> ImageSize:
-        # A square maximizes the ViT patch count (area) within the token
-        # budget; solve the budget-derived size directly to keep the dummy
-        # image small.
-        side = budget * patch_size * downsample_ratio
-        best_h, best_w = solve_resize_ratio(
-            side, side, patch_size, downsample_ratio, budget
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/mm_preprocess.py` modified +21/-8
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/common/mm_preprocess.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58539 - [ROCm][DSv4.1][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill

- 链接: https://github.com/vllm-project/vllm/pull/58539
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/rocm.py`；关联提交 `8ee40690331c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-10，可读 patch 58 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10 (21 lines); hunks: -1379,7 +1379,6 @@ def _forward_prefill(; -1408,21 +1407,23 @@ def _forward_prefill(; symbols: _forward_prefill，涉及 `_forward_prefill`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10 (21 lines); hunks: -1379,7 +1379,6 @@ def _forward_prefill(; -1408,21 +1407,23 @@ def _forward_prefill(; symbols: _forward_prefill
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/amd/rocm.py
@@ -1379,7 +1379,6 @@ def _forward_prefill(
-        num_prefills = swa_metadata.num_prefills
@@ -1408,21 +1407,23 @@ def _forward_prefill(
-        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
-            self.PREFILL_CHUNK_SIZE
+        chunk_plan = swa_metadata.get_prefill_chunk_plan(
+            compress_ratio=self.compress_ratio,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/amd/rocm.py` modified +11/-10
- 验证与风险: runtime 路径改动集中在 `vllm/models/deepseek_v41/amd/rocm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #59327 - [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups

- 链接: https://github.com/vllm-project/vllm/pull/59327
- 状态/时间: merged / 2026-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/amd/model.py`, `vllm/models/deepseek_v41/common/engram.py`, `vllm/models/deepseek_v41/nvidia/model.py`；关联提交 `4c2d277643e2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+547/-556，可读 patch 1396 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/common/engram.py` modified +490/-22 (512 lines); hunks: -34,30 +34,60; -583,12 +613,14 @@ def _engram_head_shard_weight_loader(; symbols: _engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader, _engram_lookup_kernel，涉及 `_engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader`；`vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497 (497 lines); hunks: -1,497 +0,0; symbols: _allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens, gather_engram_hashes，涉及 `_allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens`；`vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2 (9 lines); hunks: -98,9 +98,14；`vllm/models/deepseek_v41/amd/model.py` modified +1/-5 (6 lines); hunks: -65,13 +65,9。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/common/engram.py` modified +490/-22 (512 lines); hunks: -34,30 +34,60; -583,12 +613,14 @@ def _engram_head_shard_weight_loader(; symbols: _engram_lookup_thresholds, _is_prime, _engram_head_shard_weight_loader, _engram_lookup_kernel
  - `vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497 (497 lines); hunks: -1,497 +0,0; symbols: _allocate_huge_page_storage, engram_head_shard_rank, engram_gathered_num_tokens, gather_engram_hashes
  - `vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2 (9 lines); hunks: -98,9 +98,14
  - `vllm/models/deepseek_v41/amd/model.py` modified +1/-5 (6 lines); hunks: -65,13 +65,9
  - `tests/kernels/test_engram.py` modified +25/-25 (50 lines); hunks: -9,13 +9,10; -111,7 +108,7 @@ def test_fused_engram_post_wkv_matches_reference(; symbols: test_fused_engram_post_wkv_matches_reference, test_engram_rejects_empty_head_shards, test_engram_head_shards_reconstruct_checkpoint
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/common/engram.py
@@ -34,30 +34,60 @@
+import ctypes
+import mmap
+import os
+import tempfile
+from contextlib import ExitStack
+from vllm.compilation.breakable_cudagraph import eager_break_during_capture
diff -- vllm/models/deepseek_v41/nvidia/engram.py
@@ -1,497 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""
-import ctypes
-import mmap
-import os
diff -- vllm/models/deepseek_v41/nvidia/model.py
@@ -98,9 +98,14 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/common/engram.py` modified +490/-22; `vllm/models/deepseek_v41/nvidia/engram.py` removed +0/-497; `vllm/models/deepseek_v41/nvidia/model.py` modified +7/-2; `vllm/models/deepseek_v41/amd/model.py` modified +1/-5
  - tests: `tests/kernels/test_engram.py` modified +25/-25
- 验证与风险: diff 自带测试面 `tests/distributed/test_engram_dp_shard.py`, `tests/kernels/test_engram.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58560 - [Bugfix][DSv4.1] Keep the compressor ring out of the null block

- 链接: https://github.com/vllm-project/vllm/pull/58560
- 状态/时间: merged / 2026-10-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/deepseek_v41/compressor.py`；关联提交 `49e0f4797853`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+20/-14，可读 patch 67 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/deepseek_v41/compressor.py` modified +5/-1 (6 lines); hunks: -75,8 +75,12 @@ def _ring_slot_mapping_kernel(; symbols: _ring_slot_mapping_kernel，涉及 `_ring_slot_mapping_kernel`。
- 代码 diff 细节:
  - `vllm/models/deepseek_v41/compressor.py` modified +5/-1 (6 lines); hunks: -75,8 +75,12 @@ def _ring_slot_mapping_kernel(; symbols: _ring_slot_mapping_kernel
- 关键代码摘录:

```diff
diff -- vllm/models/deepseek_v41/compressor.py
@@ -75,8 +75,12 @@ def _ring_slot_mapping_kernel(
+    # Block 0 is the null block. A request without a ring block (dummy or padding
+    # rows, which the runners fill with the null block) must not write.
-        slot_mapping_ptr + offsets, tl.where(valid, slot, -1), mask=offsets < num_tokens
+        slot_mapping_ptr + offsets,
+        tl.where(valid & (block != 0), slot, -1),
+        mask=offsets < num_tokens,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/deepseek_v41/compressor.py` modified +5/-1
- 验证与风险: diff 自带测试面 `tests/kernels/test_compressor_kv_cache.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
