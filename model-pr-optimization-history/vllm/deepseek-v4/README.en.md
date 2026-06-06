# vllm DeepSeek V4 Model PR Optimization History

## 2026-06-05 PR Backfill Audit

Rechecked vllm upstream `origin/main@c66b19800` on 2026-06-05; 4 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2026-05-19). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-04 | [#43827](https://github.com/vllm-project/vllm/pull/43827) | [DSv4] Adding TRTLLM gen attention kernel | `test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| 2026-05-26 | [#43162](https://github.com/vllm-project/vllm/pull/43162) | [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel | `test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |
| 2026-05-25 | [#43632](https://github.com/vllm-project/vllm/pull/43632) | [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops | `test_deepseek_v4_mega_moe.py` |
| 2026-05-22 | [#42353](https://github.com/vllm-project/vllm/pull/42353) | DSv4 fused Q-norm kernel grid refactor | `test_fused_deepseek_v4_qnorm_rope_kv_insert.py` |


## 2026-05-19 PR Backfill Audit

Rechecked vllm upstream `origin/main@07beaed84` and the GitHub Pull Request files API; this pass adds timeline entries and per-PR diff audit cards for `#40392`, `#40871`, `#41255`, `#41263`, `#41428`, `#41443`, `#41522`, `#41536`, `#41694`, `#41710`, `#41778`, `#41801`, `#41812`, `#41946`, `#41957`, `#42112`, `#42169`, `#42236`, `#42320`, `#42342`, `#42541`, `#42604`, `#42810`, `#42828`, `#42899`, `#42930`, `#43004`, `#43039`, `#43073`, `#43077`.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/models/test_deepseek_v4_mega_moe.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_input_4.json` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_1.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_2.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_3.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/fixtures/deepseek_v4/test_output_4.txt` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `tests/tokenizers_/test_deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982) |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/model_executor/layers/deepseek_v4_attention.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41061](https://github.com/vllm-project/vllm/pull/41061) |
| `vllm/model_executor/models/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40950](https://github.com/vllm-project/vllm/pull/40950), [#41006](https://github.com/vllm-project/vllm/pull/41006), [#41061](https://github.com/vllm-project/vllm/pull/41061), [#41090](https://github.com/vllm-project/vllm/pull/41090), [#41148](https://github.com/vllm-project/vllm/pull/41148), [#41374](https://github.com/vllm-project/vllm/pull/41374) |
| `vllm/model_executor/models/deepseek_v4_mtp.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41006](https://github.com/vllm-project/vllm/pull/41006), [#41171](https://github.com/vllm-project/vllm/pull/41171) |
| `vllm/renderers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/tokenizers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#40982](https://github.com/vllm-project/vllm/pull/40982) |
| `vllm/tokenizers/deepseek_v4_encoding.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/transformers_utils/configs/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/v1/attention/ops/deepseek_v4_ops/__init__.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41015](https://github.com/vllm-project/vllm/pull/41015) |
| `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41015](https://github.com/vllm-project/vllm/pull/41015) |
| `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41135](https://github.com/vllm-project/vllm/pull/41135) |
| `vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |

## PR Coverage Summary

- Git-traced PRs: 11
- Extra PRs preserved from existing docs: 33
- Total PRs in this document: 44
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh` |
| 2026-04-26 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | merged | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-27 | [#40860](https://github.com/vllm-project/vllm/pull/40860) | merged | [Feat] DeepSeek V4 Rebased | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-27 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | closed | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-27 | [#40950](https://github.com/vllm-project/vllm/pull/40950) | merged | [DSV4] Add silu clamp limit to shared expert | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-28 | [#41006](https://github.com/vllm-project/vllm/pull/41006) | merged | [Model][DSV4] Support base model | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-28 | [#41061](https://github.com/vllm-project/vllm/pull/41061) | merged | [DSV4] Enable Multi-stream for Pre-Attn GEMM | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41171](https://github.com/vllm-project/vllm/pull/41171) | merged | [DSV4] Align aux stream API with DeepseekV4DecoderLayer | `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-29 | [#41090](https://github.com/vllm-project/vllm/pull/41090) | merged | [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41135](https://github.com/vllm-project/vllm/pull/41135) | merged | [Bugfix] fix inductor error for dpsk v4 | `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` |
| 2026-04-29 | [#40982](https://github.com/vllm-project/vllm/pull/40982) | merged | [DSV4] Support `max` reasoning effort | `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py` |
| 2026-04-29 | [#41148](https://github.com/vllm-project/vllm/pull/41148) | merged | [Bugfix] Fix repeated DSv4 RoPE cache initialization | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-04-29 | [#41015](https://github.com/vllm-project/vllm/pull/41015) | merged | [DSv4] Use `cvt` PTX for FP32->FP4 conversion | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` |
| 2026-04-30 | [#41374](https://github.com/vllm-project/vllm/pull/41374) | merged | [DSV4] Avoid redundant dtype conversion. | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-01 | [#41255](https://github.com/vllm-project/vllm/pull/41255) | merged | [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py` |
| 2026-05-01 | [#41443](https://github.com/vllm-project/vllm/pull/41443) | merged | [DSV4] Add knob to enable pre-attn gemm | `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py` |
| 2026-05-02 | [#41522](https://github.com/vllm-project/vllm/pull/41522) | merged | [DSV4] Guard megamoe flag with Pure TP | `vllm/model_executor/models/deepseek_v4.py` |
| 2026-05-05 | [#40871](https://github.com/vllm-project/vllm/pull/40871) | merged | [New Model][ROCm] Add AMD support for DeepSeek V4 | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-06 | [#41801](https://github.com/vllm-project/vllm/pull/41801) | merged | [Bugfix] DeepSeekV32/v4: respect string='true\|false' attribute andunwrap arguments/input wrapper | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py` |
| 2026-05-09 | [#41428](https://github.com/vllm-project/vllm/pull/41428) | merged | [DSv4] Improved fused Indexer Q quant kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py` |
| 2026-05-09 | [#41957](https://github.com/vllm-project/vllm/pull/41957) | merged | [Bugfix][PD] Fix DSv4 Disaggregated | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` |
| 2026-05-10 | [#41694](https://github.com/vllm-project/vllm/pull/41694) | merged | [DSV4] Add PP support for deepseek-v4 | `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md` |
| 2026-05-10 | [#42169](https://github.com/vllm-project/vllm/pull/42169) | merged | [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len | `csrc/topk.cu` |
| 2026-05-11 | [#40392](https://github.com/vllm-project/vllm/pull/40392) | merged | [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA | `csrc/cache_kernels_fused.cu`, `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` |
| 2026-05-11 | [#41536](https://github.com/vllm-project/vllm/pull/41536) | merged | add fused mhc_post_pre kernel | `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`, `vllm/model_executor/layers/mhc.py` |
| 2026-05-11 | [#41812](https://github.com/vllm-project/vllm/pull/41812) | merged | [ROCm][DSv4] implement flash sparse mla with triton kernels | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py` |
| 2026-05-11 | [#42236](https://github.com/vllm-project/vllm/pull/42236) | merged | [DSv4] Improved dequant gather K cache kernel | `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` |
| 2026-05-13 | [#41946](https://github.com/vllm-project/vllm/pull/41946) | merged | [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support | `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`, `vllm/model_executor/kernels/mhc/aiter.py` |
| 2026-05-13 | [#42320](https://github.com/vllm-project/vllm/pull/42320) | merged | [Bugfix] Fix DeepSeek V4 MTP HC state handling | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-14 | [#41263](https://github.com/vllm-project/vllm/pull/41263) | merged | [DSV4]   Fuse norm and router for low latency scenario | `csrc/moe/dsv4_norm_router_gemm_kernel.cu`, `benchmarks/kernels/benchmark_norm_router_gemm.py`, `csrc/moe/dsv4_norm_router_gemm_entry.cu` |
| 2026-05-14 | [#41778](https://github.com/vllm-project/vllm/pull/41778) | merged | [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell | `vllm/v1/attention/backends/mla/tokenspeed_mla.py`, `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py`, `benchmarks/attention_benchmarks/mla_runner.py` |
| 2026-05-14 | [#42112](https://github.com/vllm-project/vllm/pull/42112) | merged | [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup | `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` |
| 2026-05-14 | [#42342](https://github.com/vllm-project/vllm/pull/42342) | merged | [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'` | `requirements/cuda.txt` |
| 2026-05-15 | [#42604](https://github.com/vllm-project/vllm/pull/42604) | merged | DeepSeekV4-Pro enable cuda graph full and piecewise mode | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/mhc.py` |
| 2026-05-17 | [#42810](https://github.com/vllm-project/vllm/pull/42810) | merged | [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py` |
| 2026-05-18 | [#41710](https://github.com/vllm-project/vllm/pull/41710) | merged | fix: remove unused norm for dpskv4 | `vllm/model_executor/layers/deepseek_v4_attention.py` |
| 2026-05-18 | [#42541](https://github.com/vllm-project/vllm/pull/42541) | merged | [Bugfix] fix swiglu limit issue for humming backend + deepseek v4 | `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` |
| 2026-05-18 | [#42930](https://github.com/vllm-project/vllm/pull/42930) | merged | [Bugfix] Fix DSV4 MTP after ROCm mHC integration | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-05-19 | [#42828](https://github.com/vllm-project/vllm/pull/42828) | merged | [KVConnector][DSV4] HMA support for Mooncake store connector | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` |
| 2026-05-19 | [#42899](https://github.com/vllm-project/vllm/pull/42899) | merged | add cutedsl dsv4 indexer fp8 kernel | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` |
| 2026-05-19 | [#43004](https://github.com/vllm-project/vllm/pull/43004) | merged | [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N] | `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/model_executor/models/registry.py` |
| 2026-05-19 | [#43039](https://github.com/vllm-project/vllm/pull/43039) | merged | [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N] | `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-05-19 | [#43073](https://github.com/vllm-project/vllm/pull/43073) | merged | [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N] | `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py` |
| 2026-05-19 | [#43077](https://github.com/vllm-project/vllm/pull/43077) | merged | [Model Refactoring] Rename deepseek_v4.py to model.py [4/N] | `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/amd/deepseek_v4.py` |

## Per-PR Diff Audit Cards

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- Status/date: open / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +777/-347, 1666 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`; technical summary: Covers "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4"; the main implementation surface is `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `csrc/persistent_topk.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda, touching `sparse_attn_indexer, __init__, forward_cuda`; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...; `csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158.
- Code diff details:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0 (6 lines); hunks: -98,6 +98,7 @@ def sparse_attn_indexer(; -227,6 +228,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, __init__, forward_cuda
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0 (1 lines); hunks: -1051,6 +1051,7 @@ def __init__(; symbols: __init__, forward
  - `csrc/persistent_topk.cuh` modified +623/-232 (855 lines); hunks: -6,10 +6,12; -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...
  - `csrc/topk.cu` modified +143/-115 (258 lines); hunks: -1,5 +1,4; -13,131 +12,158
  - `vllm/utils/deep_gemm.py` modified +4/-0 (4 lines); hunks: -345,6 +345,7 @@ def fp8_fp4_mqa_logits(; -380,6 +381,7 @@ def fp8_fp4_mqa_logits(; symbols: fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/sparse_attn_indexer.py
@@ -98,6 +98,7 @@ def sparse_attn_indexer(
+    use_bf16_scores: bool = False,
@@ -227,6 +228,7 @@ def sparse_attn_indexer(
+                logits_dtype=torch.float32,
@@ -316,6 +318,7 @@ def sparse_attn_indexer(
+            logits_dtype=torch.bfloat16 if use_bf16_scores else torch.float32,
@@ -426,8 +429,10 @@ def __init__(
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -1051,6 +1051,7 @@ def __init__(
+            use_bf16_scores=True,
diff -- csrc/persistent_topk.cuh
@@ -6,10 +6,12 @@
+#include <cuda_bf16.h>
+#include <type_traits>
@@ -58,6 +60,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
+__device__ __forceinline__ auto convert_to_uint16_bf16(__nv_bfloat16 x)
+    -> uint16_t {
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +6/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-0; `vllm/utils/deep_gemm.py` modified +4/-0
  - other: `csrc/persistent_topk.cuh` modified +623/-232; `csrc/topk.cu` modified +143/-115
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/utils/deep_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +76/-23, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the DSML token leakage in DSV4/3.2"; model line: DeepSeek V4; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix the DSML token leakage in DSV4/3.2"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char, touching `test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls, touching `__init__, extract_tool_calls, _reset_streaming_state`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):
+    def test_no_marker_leak_chunked(self, parser):
+        """Chunked streaming must NOT leak DSML start-marker fragments
+        as content (GitHub #40801)."""
+        full_text = build_tool_call("fn", {"k": "v"})
+        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
+        content = "".join(d.content for d in deltas if d.content is not None)
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -26,6 +26,7 @@
+from vllm.tool_parsers.utils import partial_tag_overlap
@@ -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        self.is_tool_call_started: bool = False
+        self._sent_content_idx: int = 0
@@ -219,7 +220,7 @@ def extract_tool_calls(
-        self.is_tool_call_started = False
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40860 - [Feat] DeepSeek V4 Rebased

- Link: https://github.com/vllm-project/vllm/pull/40860
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py`, `tests/models/test_deepseek_v4_mega_moe.py`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_1.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_2.json`, `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` and 25 files; associated commits `4d51588e2381`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 150 files, +16313/-717, 20516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] DeepSeek V4 Rebased"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; technical summary: Covers "[Feat] DeepSeek V4 Rebased"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0 (1076 lines); hunks: -0,0 +1,1076; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does, touching `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0 (1076 lines); hunks: -0,0 +1,1076; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0 (483 lines); hunks: -0,0 +1,483; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `tests/tokenizers_/test_deepseek_v4.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: FakeHfTokenizer, get_added_vocab, encode, _tokenizer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1437 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -0,0 +1,1076 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+DeepseekV4 MLA Attention Layer
+"""
+from dataclasses import dataclass
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1076/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +483/-0; `vllm/tokenizers/deepseek_v4.py` added +90/-0
  - tests: `tests/tokenizers_/test_deepseek_v4.py` added +224/-0; `tests/models/test_deepseek_v4_mega_moe.py` added +184/-0; `tests/tokenizers_/fixtures/deepseek_v4/test_input_3.json` added +159/-0
- Risk and verification: The diff ships test coverage in `tests/compile/fusions_e2e/conftest.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40760 - [New Model] Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- Status/date: closed / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 158 files, +16968/-760, 21398 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model] Support DeepseekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; technical summary: Covers "[New Model] Support DeepseekV4"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does, touching `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1437/-0 (1437 lines); hunks: -0,0 +1,1437; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: CompressorBackend, __init__, get_name, get_supported_kernel_block_sizes
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1437 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -0,0 +1,1062 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+DeepseekV4 MLA Attention Layer
+"""
+from dataclasses import dataclass
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1437/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0; `vllm/model_executor/layers/mhc.py` added +436/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_use_trtllm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`, `tests/kernels/moe/test_ocp_mx_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40950 - [DSV4] Add silu clamp limit to shared expert

- Link: https://github.com/vllm-project/vllm/pull/40950
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4.py`; associated commits `706a04d34ba6`
- Diff scope read: GitHub Pull Request files API returned 7 files, +269/-29, 466 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add silu clamp limit to shared expert"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Add silu clamp limit to shared expert"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config, touching `DeepseekV4MLP, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +58/-3 (61 lines); hunks: -17,6 +17,7; -34,7 +35,10; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -17,6 +17,7 @@
+from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
@@ -34,7 +35,10 @@
-from vllm.model_executor.layers.quantization import QuantizationMethods
+from vllm.model_executor.layers.quantization import (
+    QuantizationConfig,
+    QuantizationMethods,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +58/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_activation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41006 - [Model][DSV4] Support base model

- Link: https://github.com/vllm-project/vllm/pull/41006
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; associated commits `2c8b76c5cb26`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +111/-23, 223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][DSV4] Support base model"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Model][DSV4] Support base model"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config, touching `DeepseekV4MLP, __init__, forward`; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx, touching `_find_mtp_layer_idx`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +93/-19 (112 lines); hunks: -10,7 +10,7; -65,6 +65,8; symbols: DeepseekV4MLP, __init__, forward, DeepseekV4FP8Config
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4 (22 lines); hunks: -48,9 +48,14; -326,6 +331,15 @@ def _find_mtp_layer_idx(name: str) -> int:; symbols: _find_mtp_layer_idx
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -10,7 +10,7 @@
-from vllm.config import VllmConfig
+from vllm.config import VllmConfig, get_current_vllm_config
@@ -65,6 +65,8 @@
+_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")
@@ -118,16 +120,59 @@ def forward(self, x):
-    """FP8 config that routes MoE layers to MXFP4 quantization.
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -48,9 +48,14 @@
-# MoE expert scales are fused into per-layer w13/w2 tensors; other FP8 linear
-# scales use `.weight_scale_inv`. Mirrors the regex in
-# DeepseekV4ForCausalLM.hf_to_vllm_mapper.
+# MoE expert scales are fused into per-layer w13/w2 tensors. The exact
+# parameter suffix depends on which FusedMoE method handles the experts:
+# - fp4 experts (Mxfp4MoEMethod) register ``w{1,2,3}_weight_scale``;
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +93/-19; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41061 - [DSV4] Enable Multi-stream for Pre-Attn GEMM

- Link: https://github.com/vllm-project/vllm/pull/41061
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`; associated commits `5aa371dc8e38`
- Diff scope read: GitHub Pull Request files API returned 4 files, +187/-57, 439 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Enable Multi-stream for Pre-Attn GEMM"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Enable Multi-stream for Pre-Attn GEMM"; the main implementation surface is `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward, touching `DeepseekV4MLAModules, __init__, forward`; `vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38 (149 lines); hunks: -4,8 +4,9; -16,6 +17,7; symbols: DeepseekV4MLAModules, __init__, forward
  - `vllm/model_executor/models/deepseek_v4.py` modified +10/-12 (22 lines); hunks: -54,7 +54,6; -872,7 +871,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -4,8 +4,9 @@
+from collections.abc import Callable
-from typing import TYPE_CHECKING, cast
+from typing import TYPE_CHECKING, Any, cast
@@ -16,6 +17,7 @@
+from vllm.model_executor.layers.utils import cublas_gemm_bf16_bf16_fp32
@@ -51,7 +53,10 @@
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -54,7 +54,6 @@
-from vllm.utils.multi_stream_utils import AuxStreamType
@@ -872,7 +871,7 @@ def __init__(
-        aux_stream: torch.cuda.Stream | None = None,
+        aux_stream_list: list[torch.cuda.Stream] | None = None,
@@ -1005,7 +1004,7 @@ def __init__(
-            aux_stream=aux_stream,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +111/-38; `vllm/model_executor/models/deepseek_v4.py` modified +10/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_compressor.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41171 - [DSV4] Align aux stream API with DeepseekV4DecoderLayer

- Link: https://github.com/vllm-project/vllm/pull/41171
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4_mtp.py`; associated commits `6fb3f7b46b12`
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-5, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Align aux stream API with DeepseekV4DecoderLayer"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[DSV4] Align aux stream API with DeepseekV4DecoderLayer"; the main implementation surface is `vllm/model_executor/models/deepseek_v4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5 (12 lines); hunks: -35,7 +35,6; -65,6 +64,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -35,7 +35,6 @@
-from vllm.utils.multi_stream_utils import AuxStreamType
@@ -65,6 +64,7 @@ def __init__(
+        aux_stream_list: list[torch.cuda.Stream] | None = None,
@@ -112,14 +112,11 @@ def __init__(
-        self.aux_stream_dict = {
-            AuxStreamType.Attention: torch.cuda.Stream(),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4_mtp.py` modified +7/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41090 - [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading

- Link: https://github.com/vllm-project/vllm/pull/41090
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4.py`; associated commits `803b9d7881cd`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre, touching `__init__, hc_pre`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -1098,6 +1098,11 @@ def __init__(; -1170,11 +1175,6 @@ def hc_pre(; symbols: __init__, hc_pre
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1098,6 +1098,11 @@ def __init__(
+        # Lazy import to avoid top-level tilelang dependency.
+        # Registers both torch.ops.vllm.mhc_pre and mhc_post
+        import vllm.model_executor.layers.mhc  # noqa: F401
@@ -1170,11 +1175,6 @@ def hc_pre(
-        # Lazy import to avoid top-level tilelang dependency.
-        # Registers both torch.ops.vllm.mhc_pre and mhc_post,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41135 - [Bugfix] fix inductor error for dpsk v4

- Link: https://github.com/vllm-project/vllm/pull/41135
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`; associated commits `2ae73c758cee`
- Diff scope read: GitHub Pull Request files API returned 1 files, +106/-36, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix inductor error for dpsk v4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`; technical summary: Covers "[Bugfix] fix inductor error for dpsk v4"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake, touching `fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36 (142 lines); hunks: -10,6 +10,7; -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(; symbols: fused_inv_rope_fp8_quant, _fused_inv_rope_fp8_quant_kernel_impl, _fused_inv_rope_fp8_quant_kernel_fake
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
@@ -10,6 +10,7 @@
+from vllm.utils.torch_utils import direct_register_custom_op
@@ -180,34 +181,74 @@ def fused_inv_rope_fp8_quant(
-    fp8_buf = torch.empty(
-        (n_groups, num_tokens, d),
-        dtype=fp8_dtype,
-        device=o.device,
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` modified +106/-36
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40982 - [DSV4] Support `max` reasoning effort

- Link: https://github.com/vllm-project/vllm/pull/40982
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`; associated commits `33f36d42605a`
- Diff scope read: GitHub Pull Request files API returned 6 files, +126/-6, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Support `max` reasoning effort"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`; technical summary: Covers "[DSV4] Support `max` reasoning effort"; the main implementation surface is `tests/tokenizers_/test_deepseek_v4.py`, `vllm/tokenizers/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values, touching `test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking`; `vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template, touching `apply_chat_template`.
- Code diff details:
  - `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1 (67 lines); hunks: -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():; -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values...; symbols: test_deepseek_v4_renders_parsed_history_tool_arguments, test_deepseek_v4_accepts_openai_reasoning_effort_values, test_deepseek_v4_none_reasoning_effort_disables_thinking, test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values
  - `vllm/tokenizers/deepseek_v4.py` modified +8/-2 (10 lines); hunks: -40,10 +40,16 @@ def apply_chat_template(; symbols: apply_chat_template
- Key code excerpts:

```diff
diff -- tests/tokenizers_/test_deepseek_v4.py
@@ -182,7 +182,7 @@ def test_deepseek_v4_renders_parsed_history_tool_arguments():
-@pytest.mark.parametrize("reasoning_effort", ["none", "low", "medium", "high"])
+@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
@@ -195,6 +195,58 @@ def test_deepseek_v4_accepts_openai_reasoning_effort_values(reasoning_effort):
+def test_deepseek_v4_none_reasoning_effort_disables_thinking():
+    prompt = _tokenizer().apply_chat_template(
+        [{"role": "user", "content": "Hello"}],
diff -- vllm/tokenizers/deepseek_v4.py
@@ -40,10 +40,16 @@ def apply_chat_template(
-            # The V4 reference currently accepts only "max", "high", or None.
-            if reasoning_effort not in ("max", "high"):
+            if not isinstance(reasoning_effort, str):
+            elif reasoning_effort == "none":
+                thinking_mode = "chat"
+                reasoning_effort = None
```

- Reviewed files:
  - tests: `tests/tokenizers_/test_deepseek_v4.py` modified +66/-1
  - runtime: `vllm/tokenizers/deepseek_v4.py` modified +8/-2
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/chat_completion/test_chat.py`, `tests/entrypoints/openai/parser/test_harmony_utils.py`, `tests/tokenizers_/test_deepseek_v4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41148 - [Bugfix] Fix repeated DSv4 RoPE cache initialization

- Link: https://github.com/vllm-project/vllm/pull/41148
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4.py`; associated commits `9d8ad5b408bf`
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix repeated DSv4 RoPE cache initialization"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[Bugfix] Fix repeated DSv4 RoPE cache initialization"; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +0/-1 (1 lines); hunks: -1027,7 +1027,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1027,7 +1027,6 @@ def __init__(
-            dtype=config.torch_dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41015 - [DSv4] Use `cvt` PTX for FP32->FP4 conversion

- Link: https://github.com/vllm-project/vllm/pull/41015
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`; associated commits `296741d02571`
- Diff scope read: GitHub Pull Request files API returned 4 files, +344/-62, 509 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Use `cvt` PTX for FP32->FP4 conversion"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`; technical summary: Covers "[DSv4] Use `cvt` PTX for FP32->FP4 conversion"; the main implementation surface is `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair, touching `_get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2`; `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn, touching `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35 (55 lines); hunks: -24,36 +24,22 @@ def _get_cos_sin(; -65,17 +51,16 @@ def _quantize_mxfp4_pair(x_lo, x_hi):; symbols: _get_cos_sin, _e2m1_nibble, _fp32x2_to_fp4x2, _quantize_mxfp4_pair
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6 (12 lines); hunks: -21,7 +21,7; -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(; symbols: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -24,36 +24,22 @@ def _get_cos_sin(
-def _e2m1_nibble(x):
-    """Quantize fp32 x (already scale-divided) to E2M1 4-bit nibble in uint8.
-    Matches torch.bucketize with boundaries
-    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] and right=False (each boundary
-    belongs to the lower bucket), plus sign bit."""
-    abs_x = tl.minimum(tl.abs(x), 6.0)
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py
@@ -21,7 +21,7 @@
-from .fused_indexer_q import _e2m1_nibble
+from .fused_indexer_q import _fp32x2_to_fp4x2
@@ -566,18 +566,18 @@ def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
-    amax = tl.maximum(amax, 1e-4)
+    amax = tl.maximum(amax, 6.0 * (2**-126))
-    log2_ratio = tl.ceil(tl.log2(amax / 6.0))
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +20/-35; `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` modified +6/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_compressor_kv_cache.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41374 - [DSV4] Avoid redundant dtype conversion.

- Link: https://github.com/vllm-project/vllm/pull/41374
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v4.py`; associated commits `307b17ce3316`
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-6, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Avoid redundant dtype conversion."; model line: DeepSeek V4; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Avoid redundant dtype conversion."; the main implementation surface is `vllm/model_executor/models/deepseek_v4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__, touching `_init_fused_moe_experts, forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +11/-6 (17 lines); hunks: -854,10 +854,9 @@ def _init_fused_moe_experts(; -1225,7 +1224,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: _init_fused_moe_experts, forward, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -854,10 +854,9 @@ def _init_fused_moe_experts(
-        if self.gate.tid2eid is not None:
-            if input_ids is None:
-                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
-            input_ids = input_ids.to(dtype=self.hash_indices_dtype)
+        if self.gate.tid2eid is not None and input_ids is None:
+            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +11/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.

### PR #41255 - [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4

- Link: https://github.com/vllm-project/vllm/pull/41255
- Status/date: merged / 2026-05-01
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `a9484dac7b73`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +153/-9, 180 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py`; technical summary: Covers "[Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6  @@ import regex as re; -1456,14 +1455,25  @@ def hc_head(; symbols: hc_head, touching `hc_head`；`vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137  @@ def _mhc_post_fake(; symbols: _mhc_post_fake, touching `_mhc_post_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +19/-9 (28 lines); hunks: -7,7 +7,6  @@ import regex as re; -1456,14 +1455,25  @@ def hc_head(; symbols: hc_head, touching `hc_head`
  - `vllm/model_executor/layers/mhc.py` modified +134/-0 (134 lines); hunks: -448,3 +448,137  @@ def _mhc_post_fake(; symbols: _mhc_post_fake, touching `_mhc_post_fake`
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -7,7 +7,6 @@
-import torch.nn.functional as F
@@ -1456,14 +1455,25 @@ def hc_head(
-    x = hidden_states
-    shape, dtype = x.size(), x.dtype
-    x = x.flatten(1).float()
-    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
-    mixes = F.linear(x, hc_fn) * rsqrt
-    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
diff -- vllm/model_executor/layers/mhc.py
@@ -448,3 +448,137 @@ def _mhc_post_fake(
+
+
+@tilelang.jit(
+    pass_configs={
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
+        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
+    },
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +19/-9; `vllm/model_executor/layers/mhc.py` modified +134/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41443 - [DSV4] Add knob to enable pre-attn gemm

- Link: https://github.com/vllm-project/vllm/pull/41443
- Status/date: merged / 2026-05-01
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `bcf5cac9fb95`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-3, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add knob to enable pre-attn gemm"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`; technical summary: Covers "[DSV4] Add knob to enable pre-attn gemm" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7  @@ import torch.nn.functional as F; -385,6 +386,8  @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv, touching `fused_wqa_wkv`；`vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7  @@ VLLM_DEBUG_WORKSPACE: bool = False; -1669,6 +1670,17  @@ def _get_or_set_default() -> str:; symbols: bool, _get_or_set_default, touching `bool, _get_or_set_default`；`vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7  @@ def execute_in_parallel(; -74,8 +75,9  @@ def execute_in_parallel(; symbols: execute_in_parallel, touching `execute_in_parallel`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0 (3 lines); hunks: -13,6 +13,7  @@ import torch.nn.functional as F; -385,6 +386,8  @@ def fused_wqa_wkv() -> torch.Tensor:; symbols: fused_wqa_wkv, touching `fused_wqa_wkv`
  - `vllm/envs.py` modified +12/-0 (12 lines); hunks: -245,6 +245,7  @@ VLLM_DEBUG_WORKSPACE: bool = False; -1669,6 +1670,17  @@ def _get_or_set_default() -> str:; symbols: bool, _get_or_set_default, touching `bool, _get_or_set_default`
  - `vllm/utils/multi_stream_utils.py` modified +9/-3 (12 lines); hunks: -64,6 +64,7  @@ def execute_in_parallel(; -74,8 +75,9  @@ def execute_in_parallel(; symbols: execute_in_parallel, touching `execute_in_parallel`
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -13,6 +13,7 @@
+import vllm.envs as envs
@@ -385,6 +386,8 @@ def fused_wqa_wkv() -> torch.Tensor:
+            enable=hidden_states.shape[0]
+            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
diff -- vllm/envs.py
@@ -245,6 +245,7 @@
+    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 4096
@@ -1669,6 +1670,17 @@ def _get_or_set_default() -> str:
+    # Token-count cutoff for multi-stream overlap of the attention input
+    # GEMM with auxiliary GEMMs (e.g. fused_wqa_wkv overlapped with indexer
+    # weights / kv-score projections in DeepSeek-V4). At or below this many
+    # tokens the FP8 main GEMM has idle SMs to share with the bf16 aux GEMMs
+    # and overlap is a 5-45% win; above it the FP8 GEMM saturates the device
+    # and the cross-stream sync becomes pure overhead. Set to 0 to disable
diff -- vllm/utils/multi_stream_utils.py
@@ -64,6 +64,7 @@ def execute_in_parallel(
+    enable: bool = False,
@@ -74,8 +75,9 @@ def execute_in_parallel(
-    before returning. When aux_streams is None, all aux_fns run sequentially
-    on the current stream.
+    before returning. Falls back to sequential execution on the current stream
+    when aux_streams is None or enable is False; in that case default_fn runs
+    first, then aux_fns in order.
@@ -86,13 +88,17 @@ def execute_in_parallel(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +3/-0; `vllm/envs.py` modified +12/-0; `vllm/utils/multi_stream_utils.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/envs.py`, `vllm/utils/multi_stream_utils.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41522 - [DSV4] Guard megamoe flag with Pure TP

- Link: https://github.com/vllm-project/vllm/pull/41522
- Status/date: merged / 2026-05-02
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `1c607d7b2cd4`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-10, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Guard megamoe flag with Pure TP"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`; technical summary: Covers "[DSV4] Guard megamoe flag with Pure TP" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15  @@ def __init__(; -1223,12 +1226,15  @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, str, touching `__init__, str`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +16/-10 (26 lines); hunks: -715,12 +715,15  @@ def __init__(; -1223,12 +1226,15  @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, str, touching `__init__, str`
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -715,12 +715,15 @@ def __init__(
-        if vllm_config.parallel_config.enable_expert_parallel:
-            self.use_mega_moe = (
-                vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
+        self.use_mega_moe = (
+            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
+        )
+        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
+            raise NotImplementedError(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +16/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #40871 - [New Model][ROCm] Add AMD support for DeepSeek V4

- Link: https://github.com/vllm-project/vllm/pull/40871
- Status/date: merged / 2026-05-05
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `628c43630155`.
- Diff scope read: GitHub Pull Request files API returned 22 files, +939/-134, 1657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model][ROCm] Add AMD support for DeepSeek V4"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[New Model][ROCm] Add AMD support for DeepSeek V4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +528/-60 (588 lines); hunks: -2,9 +2,11  @@ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; -13,6 +15,11  @@ from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata; symbols: fp8_paged_mqa_logits_torch, rocm_fp8_paged_mqa_logits, rocm_fp8_mqa_logits, rocm_aiter_sparse_attn_indexer_fake, touching `fp8_paged_mqa_logits_torch, rocm_fp8_paged_mqa_logits, rocm_fp8_mqa_logits`；`vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11  @@ fused_inv_rope_fp8_quant,; -53,6 +58,7  @@ from vllm.model_executor.layers.quantization.utils.quant_utils import (; symbols: __init__, forward, fused_wqa_wkv, attention_impl, touching `__init__, forward, fused_wqa_wkv`；`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7  @@ from vllm.model_executor.layers.fused_moe.config import (; -64,6 +65,8  @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format, touching `Mxfp4MoeBackend, _get_priority_backends, _return_or_raise`；`csrc/moe/topk_softplus_sqrt_kernels.cu` modified +32/-21 (53 lines); hunks: -60,15 +60,6  @@ __device__ __forceinline__ float toFloat(T value) {; -272,8 +263,14  @@ __launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__.
- Code diff details:
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +528/-60 (588 lines); hunks: -2,9 +2,11  @@ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; -13,6 +15,11  @@ from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata; symbols: fp8_paged_mqa_logits_torch, rocm_fp8_paged_mqa_logits, rocm_fp8_mqa_logits, rocm_aiter_sparse_attn_indexer_fake, touching `fp8_paged_mqa_logits_torch, rocm_fp8_paged_mqa_logits, rocm_fp8_mqa_logits`
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19 (92 lines); hunks: -28,6 +28,11  @@ fused_inv_rope_fp8_quant,; -53,6 +58,7  @@ from vllm.model_executor.layers.quantization.utils.quant_utils import (; symbols: __init__, forward, fused_wqa_wkv, attention_impl, touching `__init__, forward, fused_wqa_wkv`
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2 (81 lines); hunks: -18,6 +18,7  @@ from vllm.model_executor.layers.fused_moe.config import (; -64,6 +65,8  @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, _get_priority_backends, _return_or_raise, convert_weight_to_mxfp4_moe_kernel_format, touching `Mxfp4MoeBackend, _get_priority_backends, _return_or_raise`
  - `csrc/moe/topk_softplus_sqrt_kernels.cu` modified +32/-21 (53 lines); hunks: -60,15 +60,6  @@ __device__ __forceinline__ float toFloat(T value) {; -272,8 +263,14  @@ __launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
  - `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` modified +36/-2 (38 lines); hunks: -29,7 +29,11  @@ */; -42,7 +46,23  @@ #include "type_convert.cuh"
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -2,9 +2,11 @@
+import math
+import torch.nn.functional as F
@@ -13,6 +15,11 @@
+if current_platform.is_rocm():
+    from vllm.platforms.rocm import _ON_GFX942
+else:
+    _ON_GFX942 = False
+
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -28,6 +28,11 @@
+from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
+    rocm_forward_decode_fallback,
+    rocm_inv_rope_einsum,
+    rocm_sparse_attn_prefill,
+)
@@ -53,6 +58,7 @@
+from vllm.platforms import current_platform
@@ -198,8 +204,6 @@ def __init__(
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -18,6 +18,7 @@
+    RoutingMethodType,
@@ -64,6 +65,8 @@ class Mxfp4MoeBackend(Enum):
+    # Keep the legacy name as an alias while the ROCm split backend rename settles.
+    AITER = "AITER_MXFP4_BF16"
@@ -253,6 +256,8 @@ def _get_priority_backends() -> list[Mxfp4MoeBackend]:
+    if current_platform.is_rocm():
+        return [Mxfp4MoeBackend.AITER_MXFP4_BF16]
@@ -543,8 +548,22 @@ def _return_or_raise(
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +528/-60; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +73/-19; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +79/-2; `csrc/moe/topk_softplus_sqrt_kernels.cu` modified +32/-21
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41801 - [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper

- Link: https://github.com/vllm-project/vllm/pull/41801
- Status/date: merged / 2026-05-06
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `95582868efd4`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +224/-10, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; technical summary: Covers "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7  @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):; -101,10 +101,12  @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, None, _generate_tool_call_id, _convert_param_value, touching `__init__, None, _generate_tool_call_id`；`tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14  @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118  @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_type_conversion_in_streaming, touching `test_type_conversion_in_non_streaming, test_type_conversion_in_streaming`；`tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36  @@ def test_get_vllm_registry_structural_tag_returns_structural_tag(; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, touching `test_get_vllm_registry_structural_tag_returns_structural_tag`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7  @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):; -101,10 +101,12  @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, None, _generate_tool_call_id, _convert_param_value, touching `__init__, None, _generate_tool_call_id`
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14  @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118  @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_type_conversion_in_streaming, touching `test_type_conversion_in_non_streaming, test_type_conversion_in_streaming`
  - `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0 (33 lines); hunks: -203,3 +203,36  @@ def test_get_vllm_registry_structural_tag_returns_structural_tag(; symbols: test_get_vllm_registry_structural_tag_returns_structural_tag, touching `test_get_vllm_registry_structural_tag_returns_structural_tag`
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
+            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
@@ -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:
-    def _parse_invoke_params(self, invoke_str: str) -> dict:
-        param_dict = dict()
-        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
-            param_dict[param_name] = param_val
+    def _parse_invoke_params(self, invoke_str: str) -> dict[str, tuple[str, str]]:
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):
-        model_output = build_tool_call("toggle", {"enabled": "true", "count": "42"})
+        model_output = (
+            f"{FC_START}\n"
+            f'{INV_START}toggle">\n'
+            f'{PARAM_START}enabled" string="false">true{PARAM_END}\n'
+            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
+            f"{INV_END}\n"
+            f"{FC_END}"
diff -- tests/tool_parsers/test_deepseekv4_tool_parser.py
@@ -203,3 +203,36 @@ def test_get_vllm_registry_structural_tag_returns_structural_tag(
+
+
+def test_extract_tool_calls_arguments_wrapper():
+    mock_tokenizer = MagicMock()
+    mock_tokenizer.get_vocab.return_value = {}
+
+    tool = ChatCompletionToolsParam(
+        type="function",
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +33/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41428 - [DSv4] Improved fused Indexer Q quant kernel

- Link: https://github.com/vllm-project/vllm/pull/41428
- Status/date: merged / 2026-05-09
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `530d37130278`.
- Diff scope read: GitHub Pull Request files API returned 4 files, +474/-25, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Improved fused Indexer Q quant kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; technical summary: Covers "[DSv4] Improved fused Indexer Q quant kernel" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10  @@ # SPDX-License-Identifier: Apache-2.0; -342,30 +344,49  @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`；`tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7  @@ def _reference(; symbols: _reference, touching `_reference`；`vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8  @@ def has_mori() -> bool:; symbols: has_mori, touching `has_mori`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0 (423 lines); hunks: -0,0 +1,423  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24 (69 lines); hunks: -1,8 +1,10  @@ # SPDX-License-Identifier: Apache-2.0; -342,30 +344,49  @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7  @@ def _reference(; symbols: _reference, touching `_reference`
  - `vllm/utils/import_utils.py` modified +5/-0 (5 lines); hunks: -469,3 +469,8  @@ def has_mori() -> bool:; symbols: has_mori, touching `has_mori`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py
@@ -0,0 +1,423 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
+# to a separate file
+from functools import cache
+
+import cutlass
+import cutlass.cute as cute
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -1,8 +1,10 @@
+
+from vllm.utils.import_utils import has_cutedsl
@@ -342,30 +344,49 @@ def fused_indexer_q_rope_quant(
-        _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
-            positions,
-            index_q,
-            index_q.stride(0),
-            index_q.stride(1),
diff -- tests/kernels/test_fused_indexer_q_rope_quant.py
@@ -122,7 +122,7 @@ def _reference(
-@pytest.mark.parametrize("num_tokens", [1, 7, 32, 257])
+@pytest.mark.parametrize("num_tokens", [1, 7, 32, 257, 1023])
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` added +423/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +45/-24; `vllm/utils/import_utils.py` modified +5/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `tests/kernels/test_fused_indexer_q_rope_quant.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41957 - [Bugfix][PD] Fix DSv4 Disaggregated

- Link: https://github.com/vllm-project/vllm/pull/41957
- Status/date: merged / 2026-05-09
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `171d59ae8d1b`.
- Diff scope read: GitHub Pull Request files API returned 5 files, +49/-35, 213 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][PD] Fix DSv4 Disaggregated"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py`; technical summary: Covers "[Bugfix][PD] Fix DSv4 Disaggregated" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7  @@ ); -100,24 +101,24  @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake, touching `_compute_desc_ids, __init__, add_remote_agent`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7  @@ from vllm.distributed.kv_transfer.kv_connector.utils import (; -62,25 +63,24  @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping, touching `TPMapping, compute_tp_mapping`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7  @@ from vllm.platforms import current_platform; -46,3 +47,11  @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:; symbols: zmq_ctx, touching `zmq_ctx`；`vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -593,7 +593,7  @@ def describe(self, remote_engine_id: EngineId) -> str:; symbols: describe, touching `describe`.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23 (46 lines); hunks: -53,6 +53,7  @@ ); -100,24 +101,24  @@ def _compute_desc_ids(; symbols: _compute_desc_ids, __init__, add_remote_agent, _validate_remote_agent_handshake, touching `_compute_desc_ids, __init__, add_remote_agent`
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9 (18 lines); hunks: -10,6 +10,7  @@ from vllm.distributed.kv_transfer.kv_connector.utils import (; -62,25 +63,24  @@ class TPMapping:; symbols: TPMapping, compute_tp_mapping, touching `TPMapping, compute_tp_mapping`
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0 (9 lines); hunks: -10,6 +10,7  @@ from vllm.platforms import current_platform; -46,3 +47,11  @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:; symbols: zmq_ctx, touching `zmq_ctx`
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -593,7 +593,7  @@ def describe(self, remote_engine_id: EngineId) -> str:; symbols: describe, touching `describe`
  - `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2 (9 lines); hunks: -9,6 +9,8  @@ from __future__ import annotations; -33,12 +35,15  @@ def _compute_mapping(; symbols: _compute_mapping, touching `_compute_mapping`
- Key code excerpts:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
@@ -53,6 +53,7 @@
+    get_representative_spec_type,
@@ -100,24 +101,24 @@ def _compute_desc_ids(
-        ratio = physical_blocks_per_logical
-        logical_blocks = num_blocks // ratio
-
+            # NOTE (NickLucche) With HMA, every kv group has the same number of layers
+            # and layers from different groups share the same kv tensor.
+            # eg block_ids=[[1, 2], [3]]->blocks [1, 2] need to be
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py
@@ -10,6 +10,7 @@
+    TransferTopology,
@@ -62,25 +63,24 @@ class TPMapping:
-    tp_rank: int,
-    tp_size: int,
+    transfer_topology: TransferTopology,
-    is_mla: bool,
-    total_num_kv_heads: int,
+    tp_rank = transfer_topology.tp_rank
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py
@@ -10,6 +10,7 @@
+from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs
@@ -46,3 +47,11 @@ def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
+
+
+def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
+    if isinstance(spec, UniformTypeKVCacheSpecs):
+        # All inner specs are the same type; pick any.
+        inner = next(iter(spec.kv_cache_specs.values()))
```

- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` modified +23/-23; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` modified +9/-9; `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` modified +9/-0; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
  - tests: `tests/v1/kv_connector/unit/test_tp_mapping.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41694 - [DSV4] Add PP support for deepseek-v4

- Link: https://github.com/vllm-project/vllm/pull/41694
- Status/date: merged / 2026-05-10
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `f396bee56fb5`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +83/-22, 216 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4] Add PP support for deepseek-v4"; model line: DeepSeek V4; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`; technical summary: Covers "[DSV4] Add PP support for deepseek-v4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7  @@ from vllm.config import VllmConfig, get_current_vllm_config; -49,6 +50,7  @@ VocabParallelEmbedding,; symbols: __init__, str, forward, load_weights, touching `__init__, str, forward`；`docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7  @@ th {.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +82/-21 (103 lines); hunks: -12,6 +12,7  @@ from vllm.config import VllmConfig, get_current_vllm_config; -49,6 +50,7  @@ VocabParallelEmbedding,; symbols: __init__, str, forward, load_weights, touching `__init__, str, forward`
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: -385,7 +385,7  @@ th {
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -12,6 +12,7 @@
+    get_pp_group,
@@ -49,6 +50,7 @@
+from vllm.model_executor.models.interfaces import SupportsPP
@@ -57,8 +59,10 @@
+    PPMissingLayer,
+    is_pp_missing_parameter,
@@ -1261,12 +1265,15 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.embed_tokens = VocabParallelEmbedding(
diff -- docs/models/supported_models.md
@@ -385,7 +385,7 @@ th {
-| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, etc. | | |
+| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, etc. | | ✅︎ |
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +82/-21
  - docs: `docs/models/supported_models.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `docs/models/supported_models.md`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42169 - [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len

- Link: https://github.com/vllm-project/vllm/pull/42169
- Status/date: merged / 2026-05-10
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `986edc858a13`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len"; model line: DeepSeek V4; category: bug fix; main diff: `csrc/topk.cu`; technical summary: Covers "[Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7  @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7  @@ void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,.
- Code diff details:
  - `csrc/topk.cu` modified +2/-2 (4 lines); hunks: -20,7 +20,7  @@ void launch_persistent_topk(const torch::Tensor& logits,; -243,7 +243,7  @@ void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
- Key code excerpts:

```diff
diff -- csrc/topk.cu
@@ -20,7 +20,7 @@ void launch_persistent_topk(const torch::Tensor& logits,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
@@ -243,7 +243,7 @@ void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
-  const int64_t stride = logits.size(1);
+  const int64_t stride = logits.stride(0);
```

- Reviewed files:
  - runtime: `csrc/topk.cu` modified +2/-2
- Risk and verification: Runtime changes concentrate in `csrc/topk.cu`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #40392 - [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA

- Link: https://github.com/vllm-project/vllm/pull/40392
- Status/date: merged / 2026-05-11
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `a51376b3f05a`.
- Diff scope read: GitHub Pull Request files API returned 12 files, +966/-109, 1331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `csrc/cache_kernels_fused.cu`, `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py`; technical summary: Covers "[Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `csrc/cache_kernels_fused.cu` modified +75/-60 (135 lines); hunks: -21,28 +21,33  @@ namespace vllm {; -54,8 +59,8  @@ __global__ void concat_and_cache_mla_rope_fused_kernel(；`vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7  @@ def __init__(; -374,14 +375,21  @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake, touching `__init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake`；`vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0 (271 lines); hunks: -0,0 +1,271  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/compilation/passes/fusion/matcher_utils.py` modified +84/-0 (84 lines); hunks: -23,6 +23,9  @@ kNvfp4Dynamic,; -158,6 +161,87  @@ def forward_native(; symbols: forward_native, touching `forward_native`.
- Code diff details:
  - `csrc/cache_kernels_fused.cu` modified +75/-60 (135 lines); hunks: -21,28 +21,33  @@ namespace vllm {; -54,8 +59,8  @@ __global__ void concat_and_cache_mla_rope_fused_kernel(
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24 (43 lines); hunks: -345,6 +345,7  @@ def __init__(; -374,14 +375,21  @@ def __init__(; symbols: __init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake, touching `__init__, unified_mla_kv_cache_update, unified_mla_attention_with_output_fake`
  - `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0 (271 lines); hunks: -0,0 +1,271  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/compilation/passes/fusion/matcher_utils.py` modified +84/-0 (84 lines); hunks: -23,6 +23,9  @@ kNvfp4Dynamic,; -158,6 +161,87  @@ def forward_native(; symbols: forward_native, touching `forward_native`
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +32/-9 (41 lines); hunks: -127,29 +127,52  @@ def forward_native(; symbols: forward_native, touching `forward_native`
- Key code excerpts:

```diff
diff -- csrc/cache_kernels_fused.cu
@@ -21,28 +21,33 @@ namespace vllm {
-template <typename qk_t, bool IS_NEOX, typename raw_kv_scalar_t,
-          typename cache_t, Fp8KVCacheDataType kv_dt>
+template <typename qk_t, typename cos_sin_t, bool IS_NEOX,
+          typename raw_kv_scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
-    const qk_t* __restrict__ rope_cos_sin_cache,  // [max_position, 2,
-                                                  // rot_dim // 2]
+    const cos_sin_t* __restrict__ rope_cos_sin_cache,  // [max_position, 2,
+                                                       // rot_dim // 2]
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -345,6 +345,7 @@ def __init__(
+        attn_backend: type[AttentionBackend] | None = None,
@@ -374,14 +375,21 @@ def __init__(
-        self.attn_backend = get_attn_backend(
-            self.head_size,
-            dtype,
-            kv_cache_dtype,
-            use_mla=True,
-            use_sparse=use_sparse,
diff -- vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py
@@ -0,0 +1,271 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from torch._higher_order_ops.auto_functionalize import auto_functionalized
+
+import vllm._custom_ops as ops
+from vllm.config import VllmConfig, get_layers_from_vllm_config
+from vllm.logger import init_logger
```

- Reviewed files:
  - runtime: `csrc/cache_kernels_fused.cu` modified +75/-60; `vllm/model_executor/layers/attention/mla_attention.py` modified +19/-24; `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` added +271/-0; `vllm/compilation/passes/fusion/matcher_utils.py` modified +84/-0
- Risk and verification: Runtime changes concentrate in `csrc/cache_kernels_fused.cu`, `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41536 - add fused mhc_post_pre kernel

- Link: https://github.com/vllm-project/vllm/pull/41536
- Status/date: merged / 2026-05-11
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `171019ab1923`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +533/-11, 592 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add fused mhc_post_pre kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`, `vllm/model_executor/layers/mhc.py`; technical summary: Covers "add fused mhc_post_pre kernel" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53  @@ def forward(; -1320,12 +1350,19  @@ def forward(; symbols: forward, touching `forward`；`tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131  @@ def mhc_post_tilelang(; -427,6 +552,218  @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_post, _mhc_post_fake, touching `mhc_post_tilelang, mhc_post, _mhc_post_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +48/-11 (59 lines); hunks: -1199,23 +1199,53  @@ def forward(; -1320,12 +1350,19  @@ def forward(; symbols: forward, touching `forward`
  - `tests/kernels/test_mhc_kernels.py` added +142/-0 (142 lines); hunks: -0,0 +1,142  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/layers/mhc.py` modified +343/-0 (343 lines); hunks: -408,6 +408,131  @@ def mhc_post_tilelang(; -427,6 +552,218  @@ def mhc_post(; symbols: mhc_post_tilelang, mhc_post, _mhc_post_fake, touching `mhc_post_tilelang, mhc_post, _mhc_post_fake`
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1199,23 +1199,53 @@ def forward(
+        post_mix: torch.Tensor | None,
+        res_mix: torch.Tensor | None,
+        residual: torch.Tensor | None,
-        residual = x
-        x, post, comb = self.hc_pre(
-            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
-        )
+        if residual is None:
diff -- tests/kernels/test_mhc_kernels.py
@@ -0,0 +1,142 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+
+import vllm.model_executor.layers.mhc as mhc_ops  # noqa: F401
+from vllm.platforms import current_platform
+from vllm.utils.torch_utils import set_random_seed
diff -- vllm/model_executor/layers/mhc.py
@@ -408,6 +408,131 @@ def mhc_post_tilelang(
+@tilelang.jit(
+    pass_configs={
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
+        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
+    },
+)
+def mhc_fused_tilelang(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +48/-11; `vllm/model_executor/layers/mhc.py` modified +343/-0
  - tests: `tests/kernels/test_mhc_kernels.py` added +142/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `tests/kernels/test_mhc_kernels.py`, `vllm/model_executor/layers/mhc.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41812 - [ROCm][DSv4] implement flash sparse mla with triton kernels

- Link: https://github.com/vllm-project/vllm/pull/41812
- Status/date: merged / 2026-05-11
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `7863fff6e591`.
- Diff scope read: GitHub Pull Request files API returned 6 files, +1849/-212, 2180 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DSv4] implement flash sparse mla with triton kernels"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`; technical summary: Covers "[ROCm][DSv4] implement flash sparse mla with triton kernels" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757  @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60  @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_forward_decode_fallback, touching `rocm_inv_rope_einsum, rocm_forward_decode_fallback`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7  @@ fused_inv_rope_fp8_quant,; -725,6 +721,12  @@ def __init__(; symbols: __init__, forward, _forward_decode, _forward_prefill, touching `__init__, forward, _forward_decode`；`vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0 (6 lines); hunks: -112,6 +112,12  @@ def get_supported_head_sizes(cls) -> list[int]:; symbols: get_supported_head_sizes, touching `get_supported_head_sizes`.
- Code diff details:
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164 (922 lines); hunks: -905,185 +905,757  @@ def rocm_inv_rope_einsum(; -1092,38 +1664,60  @@ def rocm_forward_decode_fallback(; symbols: rocm_inv_rope_einsum, rocm_forward_decode_fallback, touching `rocm_inv_rope_einsum, rocm_forward_decode_fallback`
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0 (682 lines); hunks: -0,0 +1,682  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46 (70 lines); hunks: -28,11 +28,7  @@ fused_inv_rope_fp8_quant,; -725,6 +721,12  @@ def __init__(; symbols: __init__, forward, _forward_decode, _forward_prefill, touching `__init__, forward, _forward_decode`
  - `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0 (6 lines); hunks: -112,6 +112,12  @@ def get_supported_head_sizes(cls) -> list[int]:; symbols: get_supported_head_sizes, touching `get_supported_head_sizes`
  - `vllm/v1/attention/backends/mla/flashmla_sparse.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7  @@ # SPDX-License-Identifier: Apache-2.0; -112,7 +112,7  @@ def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:; symbols: get_builder_cls, touching `get_builder_cls`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -905,185 +905,757 @@ def rocm_inv_rope_einsum(
-def rocm_ref_sparse_attn_prefill(
+_DSV4_SPARSE_NOPE_DIM = 448
+_DSV4_SPARSE_ROPE_DIM = 64
+
+
+def _validate_dsv4_sparse_dims(
+    head_dim: int,
+    nope_head_dim: int,
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py
@@ -0,0 +1,682 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, cast
+
+import torch
+
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -28,11 +28,7 @@
-from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
-    rocm_forward_decode_fallback,
-    rocm_inv_rope_einsum,
-    rocm_sparse_attn_prefill,
-)
+from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum
@@ -725,6 +721,12 @@ def __init__(
+        if current_platform.is_rocm():
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +758/-164; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` added +682/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` modified +24/-46; `vllm/v1/attention/backends/mla/sparse_swa.py` modified +6/-0
  - tests: `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` added +377/-0
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42236 - [DSv4] Improved dequant gather K cache kernel

- Link: https://github.com/vllm-project/vllm/pull/42236
- Status/date: merged / 2026-05-11
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `724ed2fc352b`.
- Diff scope read: GitHub Pull Request files API returned 5 files, +658/-100, 832 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv4] Improved dequant gather K cache kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`; technical summary: Covers "[DSv4] Improved dequant gather K cache kernel" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22  @@ # SPDX-License-Identifier: Apache-2.0; -61,94 +65,6  @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl`；`vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1 (31 lines); hunks: -17,6 +17,7  @@ import torch; -303,7 +304,7  @@ def _dequantize_and_gather_k_kernel(; symbols: _dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache, touching `_dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0 (334 lines); hunks: -0,0 +1,334  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0 (145 lines); hunks: -0,0 +1,145  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92 (100 lines); hunks: -1,18 +1,22  @@ # SPDX-License-Identifier: Apache-2.0; -61,94 +65,6  @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl`
  - `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1 (31 lines); hunks: -17,6 +17,7  @@ import torch; -303,7 +304,7  @@ def _dequantize_and_gather_k_kernel(; symbols: _dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache, touching `_dequantize_and_gather_k_kernel, dequantize_and_gather_k_cache`
  - `tests/kernels/test_compressor_kv_cache.py` modified +141/-7 (148 lines); hunks: -3,11 +3,12  @@ """; -134,7 +135,140  @@ def test_deepseek_v4_attention_quant_cache_roundtrip(num_tokens: int, block_size; symbols: test_deepseek_v4_attention_quant_cache_roundtrip, test_indexer_gather_accepts_upper_bound_output, test_deepseek_v4_quant_magnitude_range, touching `test_deepseek_v4_attention_quant_cache_roundtrip, test_indexer_gather_accepts_upper_bound_output, test_deepseek_v4_quant_magnitude_range`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py
@@ -0,0 +1,334 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+from functools import cache
+
+import cutlass
+import cutlass.cute as cute
+import torch
diff -- vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py
@@ -0,0 +1,145 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+import cutlass
+import cutlass.cute as cute
+from cutlass import Float32, Uint32
+from cutlass._mlir import ir
+from cutlass._mlir.dialects import llvm, vector
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py
@@ -1,18 +1,22 @@
-# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
-# to a separate file
-from cutlass._mlir.dialects import llvm
-from cutlass.cutlass_dsl import T, dsl_user_op
+from vllm.v1.attention.ops.deepseek_v4_ops.cutedsl_utils import (
+    _bf16x2_abs,
+    _bf16x2_max,
+    _bf16x2_to_fp32,
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py` added +334/-0; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` added +145/-0; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +8/-92; `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` modified +30/-1
  - tests: `tests/kernels/test_compressor_kv_cache.py` modified +141/-7
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41946 - [Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support

- Link: https://github.com/vllm-project/vllm/pull/41946
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `a8887c208f34`.
- Diff scope read: GitHub Pull Request files API returned 12 files, +1920/-1033, 3143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`, `vllm/model_executor/kernels/mhc/aiter.py`; technical summary: Covers "[Bugfix] [ROCm] [DSV4] [Perf] Add aiter mhc support" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/kernels/mhc/torch.py` added +106/-0 (106 lines); hunks: -0,0 +1,106  @@ +# SPDX-License-Identifier: Apache-2.0.
- Code diff details:
  - `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0 (468 lines); hunks: -0,0 +1,468  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/kernels/mhc/triton.py` added +174/-0 (174 lines); hunks: -0,0 +1,174  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0 (138 lines); hunks: -0,0 +1,138  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/kernels/mhc/torch.py` added +106/-0 (106 lines); hunks: -0,0 +1,106  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/models/deepseek_v4.py` modified +59/-38 (97 lines); hunks: -35,6 +35,12  @@ RowParallelLinear,; -1168,6 +1174,9  @@ def __init__(; symbols: __init__, hc_pre, hc_post, forward, touching `__init__, hc_pre, hc_post`
- Key code excerpts:

```diff
diff -- vllm/model_executor/kernels/mhc/tilelang.py
@@ -0,0 +1,468 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+
+from vllm.utils.torch_utils import direct_register_custom_op
+
+
+def mhc_pre_tilelang(
diff -- vllm/model_executor/kernels/mhc/triton.py
@@ -0,0 +1,174 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+import torch.nn.functional as F
+from torch import Tensor
+
+from vllm.triton_utils import tl, triton
+from vllm.utils.torch_utils import direct_register_custom_op
diff -- vllm/model_executor/kernels/mhc/aiter.py
@@ -0,0 +1,138 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+
+from vllm.utils.torch_utils import direct_register_custom_op
+
+
+def mhc_pre_aiter(
```

- Reviewed files:
  - runtime: `vllm/model_executor/kernels/mhc/tilelang.py` added +468/-0; `vllm/model_executor/kernels/mhc/triton.py` added +174/-0; `vllm/model_executor/kernels/mhc/aiter.py` added +138/-0; `vllm/model_executor/kernels/mhc/torch.py` added +106/-0
  - tests: `tests/kernels/test_mhc_kernels.py` modified +58/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/kernels/mhc/tilelang.py`, `vllm/model_executor/kernels/mhc/triton.py`, `vllm/model_executor/kernels/mhc/aiter.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42320 - [Bugfix] Fix DeepSeek V4 MTP HC state handling

- Link: https://github.com/vllm-project/vllm/pull/42320
- Status/date: merged / 2026-05-13
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `f1cc7aad3c2c`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-5, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek V4 MTP HC state handling"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Bugfix] Fix DeepSeek V4 MTP HC state handling" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10  @@ def forward(; symbols: forward, touching `forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12  @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +4/-4 (8 lines); hunks: -1203,10 +1203,10  @@ def forward(; symbols: forward, touching `forward`
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1 (5 lines); hunks: -141,9 +141,12  @@ def forward(; symbols: forward, touching `forward`
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1203,10 +1203,10 @@ def forward(
-        post_mix: torch.Tensor | None,
-        res_mix: torch.Tensor | None,
-        residual: torch.Tensor | None,
-    ) -> torch.Tensor:
+        post_mix: torch.Tensor | None = None,
+        res_mix: torch.Tensor | None = None,
+        residual: torch.Tensor | None = None,
+    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -141,9 +141,12 @@ def forward(
-        hidden_states = self.mtp_block(
+        hidden_states, residual, post_mix, res_mix = self.mtp_block(
+        hidden_states = self.mtp_block.hc_post(
+            hidden_states, residual, post_mix, res_mix
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +4/-4; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41263 - [DSV4]   Fuse norm and router for low latency scenario

- Link: https://github.com/vllm-project/vllm/pull/41263
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `0a65d46628fd`.
- Diff scope read: GitHub Pull Request files API returned 11 files, +815/-43, 1013 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSV4]   Fuse norm and router for low latency scenario"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `csrc/moe/dsv4_norm_router_gemm_kernel.cu`, `benchmarks/kernels/benchmark_norm_router_gemm.py`, `csrc/moe/dsv4_norm_router_gemm_entry.cu`; technical summary: Covers "[DSV4]   Fuse norm and router for low latency scenario" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249  @@ +/*；`benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0 (183 lines); hunks: -0,0 +1,183  @@ +# SPDX-License-Identifier: Apache-2.0；`csrc/moe/dsv4_norm_router_gemm_entry.cu` added +130/-0 (130 lines); hunks: -0,0 +1,130  @@ +/*；`vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114  @@ +# SPDX-License-Identifier: Apache-2.0.
- Code diff details:
  - `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0 (249 lines); hunks: -0,0 +1,249  @@ +/*
  - `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0 (183 lines); hunks: -0,0 +1,183  @@ +# SPDX-License-Identifier: Apache-2.0
  - `csrc/moe/dsv4_norm_router_gemm_entry.cu` added +130/-0 (130 lines); hunks: -0,0 +1,130  @@ +/*
  - `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0 (114 lines); hunks: -0,0 +1,114  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/models/deepseek_v4.py` modified +44/-42 (86 lines); hunks: -23,11 +23,14  @@ DeepseekV4MLAModules,; -755,23 +758,23  @@ def __init__(; symbols: __init__, _init_fused_moe_experts, _forward_cuda, _make_deepseek_v4_weights_mapper, touching `__init__, _init_fused_moe_experts, _forward_cuda`
- Key code excerpts:

```diff
diff -- csrc/moe/dsv4_norm_router_gemm_kernel.cu
@@ -0,0 +1,249 @@
+/*
+ * Fused RMSNorm + router GEMV for DeepSeek V4 (logits are fp32; bf16
+ * output is unsupported because DSV4 hard-codes fp32 logits).  See
+ * dsv4_norm_router_gemm.h for the math.
+ *
+ * The GEMV body mirrors csrc/moe/dsv3_router_gemm_float_out.cu (warp
+ * butterfly reduction + smem cross-warp reduction, fp32 accumulation,
+ * 128-thread block, PDL on SM90+).  RMSNorm is folded into the same
diff -- benchmarks/kernels/benchmark_norm_router_gemm.py
@@ -0,0 +1,183 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Benchmark and correctness check for ``ops.dsv4_norm_router_gemm``.
+
+Two implementations are compared:
+
+  1. ``unfused``   — ``vllm_ops.rms_norm`` then ``ops.dsv3_router_gemm``,
+                     i.e. the current vLLM hot path (two kernel launches).
diff -- csrc/moe/dsv4_norm_router_gemm_entry.cu
@@ -0,0 +1,130 @@
+/*
+ * TORCH op entry for the fused RMSNorm + router GEMV kernel
+ * (DeepSeek V4 Pro).  This op is DSV4-Pro-specific: the kernel is
+ * instantiated only for ``num_experts == 384`` and ``hidden_dim ==
+ * 7168``.  Other configurations (e.g. DSV4-Flash with H=4096) must
+ * fall back to the unfused ``rms_norm`` + ``dsv3_router_gemm`` path.
+ */
+
```

- Reviewed files:
  - runtime: `csrc/moe/dsv4_norm_router_gemm_kernel.cu` added +249/-0; `benchmarks/kernels/benchmark_norm_router_gemm.py` added +183/-0; `csrc/moe/dsv4_norm_router_gemm_entry.cu` added +130/-0; `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py` added +114/-0
- Risk and verification: Runtime changes concentrate in `csrc/moe/dsv4_norm_router_gemm_kernel.cu`, `benchmarks/kernels/benchmark_norm_router_gemm.py`, `csrc/moe/dsv4_norm_router_gemm_entry.cu`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41778 - [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell

- Link: https://github.com/vllm-project/vllm/pull/41778
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `0d2732dd919b`.
- Diff scope read: GitHub Pull Request files API returned 14 files, +640/-89, 975 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/backends/mla/tokenspeed_mla.py`, `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py`, `benchmarks/attention_benchmarks/mla_runner.py`; technical summary: Covers "[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0 (180 lines); hunks: -0,0 +1,180  @@ +# SPDX-License-Identifier: Apache-2.0；`benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63 (130 lines); hunks: -179,19 +179,27  @@ def create_minimal_vllm_config(; -223,22 +231,17  @@ def create_minimal_vllm_config(; symbols: create_minimal_vllm_config, _create_backend_impl, _run_mla_benchmark_batched, touching `create_minimal_vllm_config, _create_backend_impl, _run_mla_benchmark_batched`；`vllm/v1/attention/backends/mla/prefill/registry.py` modified +4/-0 (4 lines); hunks: -43,6 +43,10  @@ class MLAPrefillBackendEnum(Enum, metaclass=_MLAPrefillBackendEnumMeta):; symbols: MLAPrefillBackendEnum, metaclass, touching `MLAPrefillBackendEnum, metaclass`.
- Code diff details:
  - `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0 (277 lines); hunks: -0,0 +1,277  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0 (180 lines); hunks: -0,0 +1,180  @@ +# SPDX-License-Identifier: Apache-2.0
  - `benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63 (130 lines); hunks: -179,19 +179,27  @@ def create_minimal_vllm_config(; -223,22 +231,17  @@ def create_minimal_vllm_config(; symbols: create_minimal_vllm_config, _create_backend_impl, _run_mla_benchmark_batched, touching `create_minimal_vllm_config, _create_backend_impl, _run_mla_benchmark_batched`
  - `vllm/v1/attention/backends/mla/prefill/registry.py` modified +4/-0 (4 lines); hunks: -43,6 +43,10  @@ class MLAPrefillBackendEnum(Enum, metaclass=_MLAPrefillBackendEnumMeta):; symbols: MLAPrefillBackendEnum, metaclass, touching `MLAPrefillBackendEnum, metaclass`
  - `vllm/v1/attention/backends/registry.py` modified +3/-0 (3 lines); hunks: -63,6 +63,9  @@ class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):; symbols: AttentionBackendEnum, metaclass, touching `AttentionBackendEnum, metaclass`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/backends/mla/tokenspeed_mla.py
@@ -0,0 +1,277 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""
+
+from typing import ClassVar
+
+import torch
+
diff -- vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py
@@ -0,0 +1,180 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""TokenSpeed CuTe DSL backend for MLA prefill."""
+
+from typing import TYPE_CHECKING
+
+import torch
+
diff -- benchmarks/attention_benchmarks/mla_runner.py
@@ -179,19 +179,27 @@ def create_minimal_vllm_config(
-        if prefill_cfg["flash_attn_version"] is not None:
-            vllm_config.attention_config.flash_attn_version = prefill_cfg[
-                "flash_attn_version"
+        if prefill_cfg.get("mla_prefill_backend_enum") is not None:
+            # Registry-based backends bypass the deprecated boolean flags.
+            from vllm.v1.attention.backends.mla.prefill import MLAPrefillBackendEnum
+
+            vllm_config.attention_config.mla_prefill_backend = MLAPrefillBackendEnum[
```

- Reviewed files:
  - runtime: `vllm/v1/attention/backends/mla/tokenspeed_mla.py` added +277/-0; `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py` added +180/-0; `benchmarks/attention_benchmarks/mla_runner.py` modified +67/-63; `vllm/v1/attention/backends/mla/prefill/registry.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/mla/tokenspeed_mla.py`, `vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py`, `benchmarks/attention_benchmarks/mla_runner.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42112 - [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup

- Link: https://github.com/vllm-project/vllm/pull/42112
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `2317682f9511`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-15, 86 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`; technical summary: Covers "[Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9  @@ def __init__(; -123,21 +126,17  @@ def prepare_metadata(; symbols: __init__, prepare_metadata, touching `__init__, prepare_metadata`；`vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12  @@ def __init__(; -89,7 +86,6  @@ def run_prefill_new_tokens(; symbols: __init__, run_prefill_new_tokens, run_prefill_context_chunk, touching `__init__, run_prefill_new_tokens, run_prefill_context_chunk`.
- Code diff details:
  - `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7 (13 lines); hunks: -77,6 +77,9  @@ def __init__(; -123,21 +126,17  @@ def prepare_metadata(; symbols: __init__, prepare_metadata, touching `__init__, prepare_metadata`
  - `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8 (11 lines); hunks: -61,15 +61,12  @@ def __init__(; -89,7 +86,6  @@ def run_prefill_new_tokens(; symbols: __init__, run_prefill_new_tokens, run_prefill_context_chunk, touching `__init__, run_prefill_new_tokens, run_prefill_context_chunk`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/backends/mla/prefill/flashinfer.py
@@ -77,6 +77,9 @@ def __init__(
+        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
+            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
+        )
@@ -123,21 +126,17 @@ def prepare_metadata(
-        (workspace_buffer,) = current_workspace_manager().get_simultaneous(
-            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
-        )
-
diff -- vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py
@@ -61,15 +61,12 @@ def __init__(
-
-    def _get_workspace_buffer(self) -> torch.Tensor:
-        (workspace_buffer,) = current_workspace_manager().get_simultaneous(
+        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
-        return workspace_buffer
@@ -89,7 +86,6 @@ def run_prefill_new_tokens(
-        workspace_buffer = self._get_workspace_buffer()
@@ -102,7 +98,7 @@ def run_prefill_new_tokens(
```

- Reviewed files:
  - runtime: `vllm/v1/attention/backends/mla/prefill/flashinfer.py` modified +6/-7; `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` modified +3/-8
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/mla/prefill/flashinfer.py`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42342 - [Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`

- Link: https://github.com/vllm-project/vllm/pull/42342
- Status/date: merged / 2026-05-14
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `23c85343fba6`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `requirements/cuda.txt`; technical summary: Covers "[Bug] Fix DeepSeek V4 `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'LoadCacheMode'`" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5  @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0.
- Code diff details:
  - `requirements/cuda.txt` modified +1/-1 (2 lines); hunks: -21,5 +21,5  @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
- Key code excerpts:

```diff
diff -- requirements/cuda.txt
@@ -21,5 +21,5 @@ nvidia-cudnn-frontend>=1.13.0,<1.19.0
-nvidia-cutlass-dsl[cu13]>=4.4.2
+nvidia-cutlass-dsl[cu13]==4.5.0
```

- Reviewed files:
  - runtime: `requirements/cuda.txt` modified +1/-1
- Risk and verification: Runtime changes concentrate in `requirements/cuda.txt`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42604 - DeepSeekV4-Pro enable cuda graph full and piecewise mode

- Link: https://github.com/vllm-project/vllm/pull/42604
- Status/date: merged / 2026-05-15
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `ccde9540bed0`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-3, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeekV4-Pro enable cuda graph full and piecewise mode"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/mhc.py`; technical summary: Covers "DeepSeekV4-Pro enable cuda graph full and piecewise mode" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30  @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23  @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):; symbols: combine_topk_swa_indices_ragged, DeepseekV4ROCMAiterSparseSWAMetadata, build, touching `combine_topk_swa_indices_ragged, DeepseekV4ROCMAiterSparseSWAMetadata, build`；`vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6  @@ # this import will also register the custom ops; -190,8 +189,6  @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`.
- Code diff details:
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0 (73 lines); hunks: -302,6 +302,30  @@ def combine_topk_swa_indices_ragged(; -317,6 +341,23  @@ class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):; symbols: combine_topk_swa_indices_ragged, DeepseekV4ROCMAiterSparseSWAMetadata, build, touching `combine_topk_swa_indices_ragged, DeepseekV4ROCMAiterSparseSWAMetadata, build`
  - `vllm/model_executor/layers/mhc.py` modified +0/-3 (3 lines); hunks: -5,7 +5,6  @@ # this import will also register the custom ops; -190,8 +189,6  @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py
@@ -302,6 +302,30 @@ def combine_topk_swa_indices_ragged(
+def _copy_ragged_to_graph_buffers(
+    ragged_indices: torch.Tensor,
+    ragged_indptr: torch.Tensor,
+    ragged_indices_buffer: torch.Tensor,
+    ragged_indptr_buffer: torch.Tensor,
+    num_rows: int,
+    max_entries_per_row: int,
+) -> tuple[torch.Tensor, torch.Tensor]:
diff -- vllm/model_executor/layers/mhc.py
@@ -5,7 +5,6 @@
-from vllm.platforms import current_platform
@@ -190,8 +189,6 @@ def forward_cuda(
-    # This @torch.compile is necessary for accuracy as well as performance.
-    @torch.compile(backend=current_platform.simple_compile_backend)
```

- Reviewed files:
  - runtime: `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +73/-0; `vllm/model_executor/layers/mhc.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/model_executor/layers/mhc.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42810 - [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy

- Link: https://github.com/vllm-project/vllm/pull/42810
- Status/date: merged / 2026-05-17
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `599e75f432e5`.
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-177, 364 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py`; technical summary: Covers "[ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11  @@ def rocm_fp8_mqa_logits(; -551,6 +555,12  @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:; symbols: rocm_fp8_mqa_logits, _topk_indices_torch, rocm_aiter_sparse_attn_indexer_fake, rocm_aiter_sparse_attn_indexer_native, touching `rocm_fp8_mqa_logits, _topk_indices_torch, rocm_aiter_sparse_attn_indexer_fake`；`vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8  @@ def _forward_rocm(; symbols: _forward_rocm, touching `_forward_rocm`；`vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35  @@ def forward_hip(; -124,21 +128,25  @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`；`vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6  @@ def forward_hip(; -541,5 +520,9  @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`.
- Code diff details:
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114 (147 lines); hunks: -542,7 +542,11  @@ def rocm_fp8_mqa_logits(; -551,6 +555,12  @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:; symbols: rocm_fp8_mqa_logits, _topk_indices_torch, rocm_aiter_sparse_attn_indexer_fake, rocm_aiter_sparse_attn_indexer_native, touching `rocm_fp8_mqa_logits, _topk_indices_torch, rocm_aiter_sparse_attn_indexer_fake`
  - `vllm/model_executor/models/deepseek_v4.py` modified +2/-1 (3 lines); hunks: -1277,7 +1277,8  @@ def _forward_rocm(; symbols: _forward_rocm, touching `_forward_rocm`
  - `vllm/model_executor/layers/mhc.py` modified +48/-40 (88 lines); hunks: -61,31 +61,35  @@ def forward_hip(; -124,21 +128,25  @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22 (27 lines); hunks: -505,27 +505,6  @@ def forward_hip(; -541,5 +520,9  @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -542,7 +542,11 @@ def rocm_fp8_mqa_logits(
-def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
+def _topk_indices_torch(
+    logits: torch.Tensor,
+    topk_tokens: int,
+    row_starts: torch.Tensor | None = None,
+) -> torch.Tensor:
@@ -551,6 +555,12 @@ def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
+    if row_starts is not None:
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1277,7 +1277,8 @@ def _forward_rocm(
-        x = self.ffn_norm(x)
+        # ffn_norm is now folded into self.ffn.norm_gate; ffn() takes
+        # the pre-norm activation directly.
diff -- vllm/model_executor/layers/mhc.py
@@ -61,31 +61,35 @@ def forward_hip(
-        hidden_size = residual.shape[-1]
-        if hidden_size % 256 == 0:
-            return torch.ops.vllm.mhc_pre_aiter(
-                residual,
-                fn,
-                hc_scale,
-                hc_base,
-                rms_eps,
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +33/-114; `vllm/model_executor/models/deepseek_v4.py` modified +2/-1; `vllm/model_executor/layers/mhc.py` modified +48/-40; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +5/-22
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/mhc.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #41710 - fix: remove unused norm for dpskv4

- Link: https://github.com/vllm-project/vllm/pull/41710
- Status/date: merged / 2026-05-18
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `df852ed503ac`.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: remove unused norm for dpskv4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/deepseek_v4_attention.py`; technical summary: Covers "fix: remove unused norm for dpskv4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7  @@ from vllm.model_executor.custom_op import PluggableLayer; -1111,7 +1111,6  @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2 (3 lines); hunks: -47,7 +47,7  @@ from vllm.model_executor.custom_op import PluggableLayer; -1111,7 +1111,6  @@ def __init__(; symbols: __init__, touching `__init__`
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
+from vllm.model_executor.layers.layernorm import RMSNorm
@@ -1111,7 +1111,6 @@ def __init__(
-        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/deepseek_v4_attention.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/deepseek_v4_attention.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42541 - [Bugfix] fix swiglu limit issue for humming backend + deepseek v4

- Link: https://github.com/vllm-project/vllm/pull/42541
- Status/date: merged / 2026-05-18
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `6859ca76159f`.
- Diff scope read: GitHub Pull Request files API returned 3 files, +34/-6, 94 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix swiglu limit issue for humming backend + deepseek v4"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; technical summary: Covers "[Bugfix] fix swiglu limit issue for humming backend + deepseek v4" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10  @@ TopKWeightAndReduceDelegate,; -422,6 +425,18  @@ def is_supported_config(; symbols: is_supported_config, main_apply, touching `is_supported_config, main_apply`；`vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12  @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):; -211,4 +216,7  @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config, touching `prepare_humming_moe_layer, get_humming_moe_quant_config`；`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12  @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config, touching `make_mxfp4_moe_quant_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4 (23 lines); hunks: -33,7 +33,10  @@ TopKWeightAndReduceDelegate,; -422,6 +425,18  @@ def is_supported_config(; symbols: is_supported_config, main_apply, touching `is_supported_config, main_apply`
  - `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1 (10 lines); hunks: -164,7 +164,12  @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):; -211,4 +216,7  @@ def get_humming_moe_quant_config(layer: RoutedExperts):; symbols: prepare_humming_moe_layer, get_humming_moe_quant_config, touching `prepare_humming_moe_layer, get_humming_moe_quant_config`
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1 (7 lines); hunks: -1567,7 +1567,12  @@ def make_mxfp4_moe_quant_config(; symbols: make_mxfp4_moe_quant_config, touching `make_mxfp4_moe_quant_config`
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py
@@ -33,7 +33,10 @@
-from vllm.model_executor.layers.fused_moe.utils import _resize_cache
+from vllm.model_executor.layers.fused_moe.utils import (
+    _resize_cache,
+    swiglu_limit_func,
+)
@@ -422,6 +425,18 @@ def is_supported_config(
+    def apply_activation(
+        self,
diff -- vllm/model_executor/layers/quantization/utils/humming_utils.py
@@ -164,7 +164,12 @@ def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):
-def get_humming_moe_quant_config(layer: RoutedExperts):
+def get_humming_moe_quant_config(
+    layer: RoutedExperts,
+    gemm1_alpha: float | None = None,
+    gemm1_beta: float | None = None,
+    gemm1_clamp_limit: float | None = None,
+):
@@ -211,4 +216,7 @@ def get_humming_moe_quant_config(layer: RoutedExperts):
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -1567,7 +1567,12 @@ def make_mxfp4_moe_quant_config(
-        return get_humming_moe_quant_config(layer)
+        return get_humming_moe_quant_config(
+            layer,
+            gemm1_alpha=gemm1_alpha,
+            gemm1_beta=gemm1_beta,
+            gemm1_clamp_limit=swiglu_limit,
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` modified +19/-4; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +9/-1; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`, `vllm/model_executor/layers/quantization/utils/humming_utils.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42930 - [Bugfix] Fix DSV4 MTP after ROCm mHC integration

- Link: https://github.com/vllm-project/vllm/pull/42930
- Status/date: merged / 2026-05-18
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `67f58ce23f46`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-12, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DSV4 MTP after ROCm mHC integration"; model line: DeepSeek V4; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; technical summary: Covers "[Bugfix] Fix DSV4 MTP after ROCm mHC integration" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12  @@ def _forward_rocm(; -1288,10 +1290,12  @@ def forward(; symbols: _forward_rocm, forward, touching `_forward_rocm, forward`；`vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10  @@ def forward(; -235,7 +236,7  @@ def compute_logits(; symbols: forward, compute_logits, touching `forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified +12/-8 (20 lines); hunks: -1261,10 +1261,12  @@ def _forward_rocm(; -1288,10 +1290,12  @@ def forward(; symbols: _forward_rocm, forward, touching `_forward_rocm, forward`
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4 (9 lines); hunks: -146,9 +146,10  @@ def forward(; -235,7 +236,7  @@ def compute_logits(; symbols: forward, compute_logits, touching `forward, compute_logits`
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -1261,10 +1261,12 @@ def _forward_rocm(
-        post_mix: torch.Tensor | None,
-        res_mix: torch.Tensor | None,
-        residual: torch.Tensor | None,
-    ) -> torch.Tensor:
+        post_mix: torch.Tensor | None = None,
+        res_mix: torch.Tensor | None = None,
+        residual: torch.Tensor | None = None,
+    ) -> tuple[
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
@@ -146,9 +146,10 @@ def forward(
-        hidden_states = self.mtp_block.hc_post(
-            hidden_states, residual, post_mix, res_mix
-        )
+        if current_platform.is_cuda():
+            hidden_states = self.mtp_block.hc_post(
+                hidden_states, residual, post_mix, res_mix
+            )
@@ -235,7 +236,7 @@ def compute_logits(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` modified +12/-8; `vllm/model_executor/models/deepseek_v4_mtp.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42828 - [KVConnector][DSV4] HMA support for Mooncake store connector

- Link: https://github.com/vllm-project/vllm/pull/42828
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `056bc2e16646`.
- Diff scope read: GitHub Pull Request files API returned 10 files, +1835/-446, 3088 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[KVConnector][DSV4] HMA support for Mooncake store connector"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py`; technical summary: Covers "[KVConnector][DSV4] HMA support for Mooncake store connector" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7  @@ and MooncakeDistributedStore integration.; -36,15 +37,25  @@ from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread, _clear_store_pressure, touching `KVTransferThread, __init__, KVCacheStoreSendingThread`；`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0 (290 lines); hunks: -0,0 +1,290  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` modified +47/-33 (80 lines); hunks: -15,7 +15,10  @@ ); -29,6 +32,7  @@ class KeyMetadata:; symbols: KeyMetadata, __hash__, to_string, set_kv_caches_base_addr, touching `KeyMetadata, __hash__, to_string`；`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` modified +50/-3 (53 lines); hunks: -26,6 +26,7  @@ KVConnectorBase_V1,; -76,7 +77,7  @@ def __repr__(self) -> str:; symbols: __repr__, prefer_cross_layer_blocks, __init__, request_finished, touching `__repr__, prefer_cross_layer_blocks, __init__`.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180 (417 lines); hunks: -10,6 +10,7  @@ and MooncakeDistributedStore integration.; -36,15 +37,25  @@ from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (; symbols: KVTransferThread, __init__, KVCacheStoreSendingThread, _clear_store_pressure, touching `KVTransferThread, __init__, KVCacheStoreSendingThread`
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0 (290 lines); hunks: -0,0 +1,290  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` modified +47/-33 (80 lines); hunks: -15,7 +15,10  @@ ); -29,6 +32,7  @@ class KeyMetadata:; symbols: KeyMetadata, __hash__, to_string, set_kv_caches_base_addr, touching `KeyMetadata, __hash__, to_string`
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` modified +50/-3 (53 lines); hunks: -26,6 +26,7  @@ KVConnectorBase_V1,; -76,7 +77,7  @@ def __repr__(self) -> str:; symbols: __repr__, prefer_cross_layer_blocks, __init__, request_finished, touching `__repr__, prefer_cross_layer_blocks, __init__`
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py` modified +27/-19 (46 lines); hunks: -22,7 +22,9  @@ ); -45,7 +47,11  @@ def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:; symbols: _new_req_prefill_tokens, __init__, update_state_after_alloc, build_connector_meta, touching `_new_req_prefill_tokens, __init__, update_state_after_alloc`
- Key code excerpts:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
@@ -10,6 +10,7 @@
+import dataclasses
@@ -36,15 +37,25 @@
-from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
+from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
+    ExternalCachedBlockPool,
+    MooncakeStoreCoordinator,
+)
+from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
diff -- vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py
@@ -0,0 +1,290 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""External-store cache-hit coordinator for MooncakeStoreConnector."""
+
+from typing import cast
+
+from vllm.v1.core.block_pool import BlockPool
+from vllm.v1.core.kv_cache_utils import (
diff -- vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py
@@ -15,7 +15,10 @@
-from vllm.v1.core.kv_cache_utils import BlockHash
+from vllm.v1.core.kv_cache_utils import (
+    BlockHash,
+    BlockHashListWithBlockSize,
+)
@@ -29,6 +32,7 @@ class KeyMetadata:
+    group_id: int = 0
@@ -46,6 +50,7 @@ def __hash__(self):
```

- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` modified +237/-180; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` added +290/-0; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` modified +47/-33; `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` modified +50/-3
  - tests: `tests/v1/kv_connector/unit/test_mooncake_store_worker.py` modified +357/-117; `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` added +342/-0; `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` added +302/-0
- Risk and verification: Runtime changes concentrate in `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #42899 - add cutedsl dsv4 indexer fp8 kernel

- Link: https://github.com/vllm-project/vllm/pull/42899
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `3ca8db2ef88e`.
- Diff scope read: GitHub Pull Request files API returned 4 files, +411/-60, 562 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add cutedsl dsv4 indexer fp8 kernel"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`; technical summary: Covers "add cutedsl dsv4 indexer fp8 kernel" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7  @@ _bf16x2_max,; -65,8 +66,48  @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, __init__, kernel, compile, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, __init__, kernel`；`vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41  @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`；`vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39  @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:; symbols: _fp8x4_to_bf16x4, loc, touching `_fp8x4_to_bf16x4, loc`；`tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17  @@ Expects bit-exact equality on both q_fp8 and weights_out.; -125,8 +129,14  @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused, touching `_reference, test_fused_indexer_q_rope_quant_matches_unfused`.
- Code diff details:
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37 (348 lines); hunks: -14,6 +14,7  @@ _bf16x2_max,; -65,8 +66,48  @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(; symbols: fused_indexer_q_rope_quant_mxfp4_cutedsl, __init__, kernel, compile, touching `fused_indexer_q_rope_quant_mxfp4_cutedsl, __init__, kernel`
  - `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20 (57 lines); hunks: -398,24 +398,41  @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`
  - `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0 (33 lines); hunks: -117,6 +117,39  @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:; symbols: _fp8x4_to_bf16x4, loc, touching `_fp8x4_to_bf16x4, loc`
  - `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3 (33 lines); hunks: -13,13 +13,17  @@ Expects bit-exact equality on both q_fp8 and weights_out.; -125,8 +129,14  @@ def _reference(; symbols: _reference, test_fused_indexer_q_rope_quant_matches_unfused, touching `_reference, test_fused_indexer_q_rope_quant_matches_unfused`
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py
@@ -14,6 +14,7 @@
+    _fp32x4_to_fp8x4,
@@ -65,8 +66,48 @@ def fused_indexer_q_rope_quant_mxfp4_cutedsl(
-class IndexerQMxFp4Kernel:
-    """Eight-thread subwarps process one ``(token, head)`` row."""
+def fused_indexer_q_rope_quant_fp8_cutedsl(
+    positions: torch.Tensor,
+    index_q: torch.Tensor,
+    index_q_cos_sin_cache: torch.Tensor,
diff -- vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py
@@ -398,24 +398,41 @@ def fused_indexer_q_rope_quant(
-    _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
-        positions,
-        index_q,
-        index_q.stride(0),
-        index_q.stride(1),
-        index_q_cos_sin_cache,
-        index_q_cos_sin_cache.stride(0),
-        index_q_cos_sin_cache.shape[-1] // 2,
diff -- vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py
@@ -117,6 +117,39 @@ def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
+@dsl_user_op
+def _fp32x4_to_fp8x4(
+    a0: Float32,
+    a1: Float32,
+    a2: Float32,
+    a3: Float32,
+    *,
+    loc=None,
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py` modified +311/-37; `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` modified +37/-20; `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py` modified +33/-0
  - tests: `tests/kernels/test_fused_indexer_q_rope_quant.py` modified +30/-3
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py`, `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #43004 - [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]

- Link: https://github.com/vllm-project/vllm/pull/43004
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `287471b99442`.
- Diff scope read: GitHub Pull Request files API returned 12 files, +189/-126, 476 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/model_executor/models/registry.py`; technical summary: Covers "[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` renamed +11/-114 (125 lines); hunks: -9,7 +9,7  @@ import torch.nn as nn; -24,7 +24,6  @@ DeepseekV4MultiHeadLatentAttentionWrapper,; symbols: forward, load_weights, __init__, str, touching `forward, load_weights, __init__`；`vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/model_executor/models/registry.py` modified +26/-5 (31 lines); hunks: -6,6 +6,7  @@ """; -97,7 +98,7  @@ "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),; symbols: _save_modelinfo_to_cache, is_transcription_only_model, touching `_save_modelinfo_to_cache, is_transcription_only_model`；`vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30  @@ +# SPDX-License-Identifier: Apache-2.0.
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` renamed +11/-114 (125 lines); hunks: -9,7 +9,7  @@ import torch.nn as nn; -24,7 +24,6  @@ DeepseekV4MultiHeadLatentAttentionWrapper,; symbols: forward, load_weights, __init__, str, touching `forward, load_weights, __init__`
  - `vllm/models/deepseek_v4/quant_config.py` added +106/-0 (106 lines); hunks: -0,0 +1,106  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/model_executor/models/registry.py` modified +26/-5 (31 lines); hunks: -6,6 +6,7  @@ """; -97,7 +98,7  @@ "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),; symbols: _save_modelinfo_to_cache, is_transcription_only_model, touching `_save_modelinfo_to_cache, is_transcription_only_model`
  - `vllm/models/deepseek_v4/__init__.py` added +30/-0 (30 lines); hunks: -0,0 +1,30  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/models/deepseek_v4/nvidia/deepseek_v4_mtp.py` renamed +6/-5 (11 lines); hunks: -34,16 +34,16  @@ VocabParallelEmbedding,; -68,6 +68,7  @@ def __init__(; symbols: __init__, _find_mtp_layer_idx, touching `__init__, _find_mtp_layer_idx`
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/deepseek_v4.py
@@ -9,7 +9,7 @@
-from vllm.config import VllmConfig, get_current_vllm_config
+from vllm.config import VllmConfig
@@ -24,7 +24,6 @@
-from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
@@ -44,29 +43,15 @@
-from vllm.model_executor.layers.quantization import (
-    QuantizationConfig,
-    QuantizationMethods,
diff -- vllm/models/deepseek_v4/quant_config.py
@@ -0,0 +1,106 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Quantization config for DeepSeek V4."""
+
+from vllm.config import get_current_vllm_config
+from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
+from vllm.model_executor.layers.quantization import QuantizationMethods
diff -- vllm/model_executor/models/registry.py
@@ -6,6 +6,7 @@
+import importlib.util
@@ -97,7 +98,7 @@
-    "DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),
+    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
@@ -611,7 +612,7 @@
-    "DeepSeekV4MTPModel": ("deepseek_v4_mtp", "DeepSeekV4MTP"),
+    "DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
@@ -870,10 +871,21 @@ def _save_modelinfo_to_cache(self, mi: _ModelInfo, module_hash: str) -> None:
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` renamed +11/-114; `vllm/models/deepseek_v4/quant_config.py` added +106/-0; `vllm/model_executor/models/registry.py` modified +26/-5; `vllm/models/deepseek_v4/__init__.py` added +30/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/models/deepseek_v4/quant_config.py`, `vllm/model_executor/models/registry.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #43039 - [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]

- Link: https://github.com/vllm-project/vllm/pull/43039
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `87b08c5f6460`.
- Diff scope read: GitHub Pull Request files API returned 5 files, +8/-11, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -18,11 +18,6  @@ ); -61,6 +56,11  @@ maybe_prefix,；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +1/-3 (4 lines); hunks: -31,9 +31,7  @@ from vllm.v1.worker.workspace import current_workspace_manager；`vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6  @@ from vllm.logger import init_logger; -55,6 +54,7  @@ from vllm.model_executor.layers.quantization.utils.quant_utils import (；`vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` modified +5/-5 (10 lines); hunks: -18,11 +18,6  @@ ); -61,6 +56,11  @@ maybe_prefix,
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +1/-3 (4 lines); hunks: -31,9 +31,7  @@ from vllm.v1.worker.workspace import current_workspace_manager
  - `vllm/models/deepseek_v4/attention.py` renamed +1/-1 (2 lines); hunks: -46,7 +46,6  @@ from vllm.logger import init_logger; -55,6 +54,7  @@ from vllm.model_executor.layers.quantization.utils.quant_utils import (
  - `vllm/models/deepseek_v4/compressor.py` renamed +0/-0 (0 lines)
  - `.github/CODEOWNERS` modified +1/-2 (3 lines); hunks: -153,9 +153,8  @@ mkdocs.yaml @hmellor
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/nvidia/deepseek_v4.py
@@ -18,11 +18,6 @@
-from vllm.model_executor.layers.deepseek_v4_attention import (
-    DeepseekV4Indexer,
-    DeepseekV4MLAModules,
-    DeepseekV4MultiHeadLatentAttentionWrapper,
-)
@@ -61,6 +56,11 @@
+from vllm.models.deepseek_v4.attention import (
+    DeepseekV4Indexer,
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py
@@ -31,9 +31,7 @@
-    from vllm.model_executor.layers.deepseek_v4_attention import (
-        DeepseekV4MLAAttention,
-    )
+    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
diff -- vllm/models/deepseek_v4/attention.py
@@ -46,7 +46,6 @@
-from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
@@ -55,6 +54,7 @@
+from vllm.models.deepseek_v4.compressor import DeepseekCompressor
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/nvidia/deepseek_v4.py` modified +5/-5; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py` modified +1/-3; `vllm/models/deepseek_v4/attention.py` renamed +1/-1; `vllm/models/deepseek_v4/compressor.py` renamed +0/-0
  - ci: `.github/CODEOWNERS` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/nvidia/deepseek_v4.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`, `vllm/models/deepseek_v4/attention.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #43073 - [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]

- Link: https://github.com/vllm-project/vllm/pull/43073
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `b14be81c1f63`.
- Diff scope read: GitHub Pull Request files API returned 20 files, +34/-29, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`; technical summary: Covers "[Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13  @@ from vllm.forward_context import get_forward_context; -23,14 +27,6  @@ CommonAttentionMetadata,；`vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8  @@ +# SPDX-License-Identifier: Apache-2.0；`vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16  @@ ReplicatedLinear,；`vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9  @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache, touching `dequantize_and_gather_k_cache`.
- Code diff details:
  - `vllm/models/deepseek_v4/compressor.py` modified +6/-10 (16 lines); hunks: -11,9 +11,13  @@ from vllm.forward_context import get_forward_context; -23,14 +27,6  @@ CommonAttentionMetadata,
  - `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0 (8 lines); hunks: -0,0 +1,8  @@ +# SPDX-License-Identifier: Apache-2.0
  - `vllm/models/deepseek_v4/attention.py` modified +3/-3 (6 lines); hunks: -19,16 +19,16  @@ ReplicatedLinear,
  - `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1 (4 lines); hunks: -366,7 +366,9  @@ def dequantize_and_gather_k_cache(; symbols: dequantize_and_gather_k_cache, touching `dequantize_and_gather_k_cache`
  - `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` renamed +2/-2 (4 lines); hunks: -346,7 +346,7  @@ def fused_indexer_q_rope_quant(; -400,7 +400,7  @@ def fused_indexer_q_rope_quant(; symbols: fused_indexer_q_rope_quant, touching `fused_indexer_q_rope_quant`
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/compressor.py
@@ -11,9 +11,13 @@
-from vllm.model_executor.layers.linear import (
-    MergedColumnParallelLinear,
+from vllm.model_executor.layers.linear import MergedColumnParallelLinear
+from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
+    _fused_kv_compress_norm_rope_insert_indexer_attn,
+    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
+    _fused_kv_compress_norm_rope_insert_sparse_attn,
+from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
diff -- vllm/models/deepseek_v4/nvidia/ops/__init__.py
@@ -0,0 +1,8 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""NVIDIA-only (cutedsl/cutlass) kernels for DeepSeek V4.
+
+These modules import ``cutlass``/``cutedsl`` at module top level, so they must
+not be imported on non-CUDA platforms. Callers should gate on
+``vllm.utils.import_utils.has_cutedsl()`` before importing from here.
+"""
diff -- vllm/models/deepseek_v4/attention.py
@@ -19,16 +19,16 @@
-from vllm.utils.deep_gemm import fp8_einsum
-from vllm.utils.torch_utils import direct_register_custom_op
-from vllm.v1.attention.ops.deepseek_v4_ops import (
+from vllm.models.deepseek_v4.common.ops import (
+from vllm.utils.deep_gemm import fp8_einsum
+from vllm.utils.torch_utils import direct_register_custom_op
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/compressor.py` modified +6/-10; `vllm/models/deepseek_v4/nvidia/ops/__init__.py` added +8/-0; `vllm/models/deepseek_v4/attention.py` modified +3/-3; `vllm/models/deepseek_v4/common/ops/cache_utils.py` renamed +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/compressor.py`, `vllm/models/deepseek_v4/nvidia/ops/__init__.py`, `vllm/models/deepseek_v4/attention.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.

### PR #43077 - [Model Refactoring] Rename deepseek_v4.py to model.py [4/N]

- Link: https://github.com/vllm-project/vllm/pull/43077
- Status/date: merged / 2026-05-19
- Trace source: 2026-05-19 PR backfill audit; traced from source-refresh notes, upstream `origin/main@07beaed84` history, and the GitHub Pull Request files API; associated commit `07beaed8422d`.
- Diff scope read: GitHub Pull Request files API returned 8 files, +8/-8, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/amd/deepseek_v4.py`; technical summary: Covers "[Model Refactoring] Rename deepseek_v4.py to model.py [4/N]" with file-level evidence, code excerpts, and validation risks below.
- Key implementation: `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11  @@ # default that mypy sees; the ROCm branch overrides it at runtime and is；`vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7  @@ from vllm.platforms import current_platform；`vllm/models/deepseek_v4/amd/deepseek_v4.py` removed +0/-1 (1 lines); hunks: -1 +0,0  @@ -../nvidia/deepseek_v4.py；`vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0  @@ -../nvidia/deepseek_v4_mtp.py.
- Code diff details:
  - `vllm/models/deepseek_v4/__init__.py` modified +4/-4 (8 lines); hunks: -17,11 +17,11  @@ # default that mypy sees; the ROCm branch overrides it at runtime and is
  - `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1 (2 lines); hunks: -40,7 +40,7  @@ from vllm.platforms import current_platform
  - `vllm/models/deepseek_v4/amd/deepseek_v4.py` removed +0/-1 (1 lines); hunks: -1 +0,0  @@ -../nvidia/deepseek_v4.py
  - `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1 (1 lines); hunks: -1 +0,0  @@ -../nvidia/deepseek_v4_mtp.py
  - `vllm/models/deepseek_v4/amd/model.py` added +1/-0 (1 lines); hunks: -0,0 +1  @@ +../nvidia/model.py
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v4/__init__.py
@@ -17,11 +17,11 @@
-    from .nvidia.deepseek_v4 import DeepseekV4ForCausalLM
-    from .nvidia.deepseek_v4_mtp import DeepSeekV4MTP
+    from .nvidia.model import DeepseekV4ForCausalLM
+    from .nvidia.mtp import DeepSeekV4MTP
-    from .amd.deepseek_v4 import DeepseekV4ForCausalLM  # type: ignore[assignment]
-    from .amd.deepseek_v4_mtp import DeepSeekV4MTP  # type: ignore[assignment]
+    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
+    from .amd.mtp import DeepSeekV4MTP  # type: ignore[assignment]
diff -- vllm/models/deepseek_v4/nvidia/mtp.py
@@ -40,7 +40,7 @@
-from .deepseek_v4 import (
+from .model import (
diff -- vllm/models/deepseek_v4/amd/deepseek_v4.py
@@ -1 +0,0 @@
-../nvidia/deepseek_v4.py
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v4/__init__.py` modified +4/-4; `vllm/models/deepseek_v4/nvidia/mtp.py` renamed +1/-1; `vllm/models/deepseek_v4/amd/deepseek_v4.py` removed +0/-1; `vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py` removed +0/-1
  - tests: `tests/models/test_deepseek_v4_mega_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v4/__init__.py`, `vllm/models/deepseek_v4/nvidia/mtp.py`, `vllm/models/deepseek_v4/amd/deepseek_v4.py`; risks are weight loading, parallel sharding, attention/MoE backend selection, quantized dtypes, and parser output, so use a real checkpoint or equivalent smoke test.
