# vLLM DeepSeek V3.2 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for DeepSeek V3.2.

- Status: supported on current mainline

## Key Conclusions

- DeepSeek V3.2 is the major vLLM fork of the DeepSeek line because it adds sparse MLA and indexer logic.
- Most regressions land in the indexer builder, tokenizer / parser, and specialized decode kernels rather than the generic V3 loader.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## Landed PRs

- [#25896](https://github.com/vllm-project/vllm/pull/25896) `Support DeepSeek-V3.2`: Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- [#25999](https://github.com/vllm-project/vllm/pull/25999) `Support indexer prefill chunking`: Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- [#26670](https://github.com/vllm-project/vllm/pull/26670) `Add AMD GPU support on DeepSeek v3.2 and SparseMLA`: Opened the ROCm SparseMLA lane for V3.2 deployments.
- [#29848](https://github.com/vllm-project/vllm/pull/29848) `Add DeepSeek-V3.2 tool parser`: Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- [#33090](https://github.com/vllm-project/vllm/pull/33090) `Fix DeepseekV32 `AssertionError: num_kv_heads == 1``: Removed a hard failure triggered by newer V3.2 attention shapes.
- [#37421](https://github.com/vllm-project/vllm/pull/37421) `Persistent TopK scheduler for DeepSeek-V3.2 decode`: Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `DeepSeek V3.2` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-29 | [#25896](https://github.com/vllm-project/vllm/pull/25896) | merged | [New Model] DeepSeek-V3.2 (Rebased to Main) | model wrapper, attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` |
| 2025-10-01 | [#25999](https://github.com/vllm-project/vllm/pull/25999) | merged | [Deepseek v3.2] Support indexer prefill chunking | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` |
| 2025-10-13 | [#26670](https://github.com/vllm-project/vllm/pull/26670) | merged | [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA | model wrapper, attention/backend, kernel, scheduler/runtime | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-02 | [#29848](https://github.com/vllm-project/vllm/pull/29848) | merged | Add DeepSeek-V3.2 tool parser. | misc | `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-01-26 | [#33090](https://github.com/vllm-project/vllm/pull/33090) | merged | [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1` | misc | `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-03-18 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu` |

### File-level PR diff reading notes

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- Link: https://github.com/vllm-project/vllm/pull/25896
- Status/date: `merged`, created 2025-09-29, merged 2025-09-30; author `zyongye`.
- Diff scope read: `71` files, `+3918/-221`; areas: model wrapper, attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, cuda, fp8, cache, kv, flash, mla, config, quant, scheduler.
- Code diff details:
  - `vllm/v1/attention/backends/mla/flashmla_sparse.py` added +544/-0 (544 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _lse2_to_lse, FlashMLASparseBackend, get_name, get_metadata_cls
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: from transformers import DeepseekV2Config, DeepseekV3Config; class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__, __init__
  - `tests/v1/attention/test_sparse_mla_backends.py` added +426/-0 (426 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _dequantize_fp8_ds_mla_entry, _quantize_dequantize_fp8_ds_mla, test_sparse_backend_metadata_registration, test_sparse_decode_metadata_filters_prefill_indices
  - `vllm/v1/attention/backends/mla/indexer.py` added +293/-0 (293 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepseekV32IndexerBackend, get_metadata_cls, get_supported_head_sizes, get_builder_cls
  - `tests/kernels/attention/test_deepgemm_attention.py` added +279/-0 (279 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: kv_cache_cast_to_fp8, per_custom_dims_cast_to_fp8, _generate_cp_test_data, _ref_fp8_mqa_logits
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`; keywords observed in patches: attention, cuda, fp8, cache, kv, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- Link: https://github.com/vllm-project/vllm/pull/25999
- Status/date: `merged`, created 2025-10-01, merged 2025-10-02; author `heheda12345`.
- Diff scope read: `3` files, `+149/-79`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks; keywords: attention, cache, kv, mla, config, cuda, flash, fp8, quant, scheduler.
- Code diff details:
  - `vllm/v1/attention/backends/mla/indexer.py` modified +90/-41 (131 lines); hunks: def get_kv_cache_stride_order() -> tuple[int, ...]:; class DeepseekV32IndexerMetadata:; symbols: get_kv_cache_stride_order, DeepseekV32IndexerPrefillMetadata:, DeepseekV32IndexerPrefillChunkMetadata:, DeepseekV32IndexerPrefillMetadata:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: def sparse_attn_indexer(; symbols: sparse_attn_indexer
  - `tests/v1/attention/test_sparse_mla_backends.py` modified +22/-0 (22 lines); hunks: from vllm.v1.attention.backends.mla.flashmla_sparse import (; def test_sparse_backend_decode_correctness(dist_init, batch_name,; symbols: test_sparse_backend_decode_correctness, test_split_prefill_chunks
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`; keywords observed in patches: attention, cache, kv, mla, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- Link: https://github.com/vllm-project/vllm/pull/26670
- Status/date: `merged`, created 2025-10-13, merged 2025-11-20; author `ganyi1996ppo`.
- Diff scope read: `9` files, `+583/-15`; areas: model wrapper, attention/backend, kernel, scheduler/runtime; keywords: attention, cache, mla, fp8, kv, quant, spec, config, cuda, flash.
- Code diff details:
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` added +325/-0 (325 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: ROCMAiterMLASparseBackend, get_name, get_metadata_cls, get_builder_cls
  - `vllm/attention/ops/rocm_aiter_mla_sparse.py` added +210/-0 (210 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: fp8_mqa_logits_torch, rocm_fp8_mqa_logits, has_mqa_logits_module, fp8_paged_mqa_logits_torch
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: def sparse_attn_indexer(; def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer
  - `vllm/v1/attention/backends/mla/indexer.py` modified +9/-6 (15 lines); hunks: ); class DeepseekV32IndexerBackend(AttentionBackend):; symbols: DeepseekV32IndexerBackend, get_supported_head_sizes, build
  - `vllm/platforms/rocm.py` modified +12/-1 (13 lines); hunks: def get_attn_backend_cls(; symbols: get_attn_backend_cls
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`; keywords observed in patches: attention, cache, mla, fp8, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #29848 - Add DeepSeek-V3.2 tool parser.

- Link: https://github.com/vllm-project/vllm/pull/29848
- Status/date: `merged`, created 2025-12-02, merged 2025-12-04; author `Xu-Wenqing`.
- Diff scope read: `2` files, `+595/-0`; areas: misc; keywords: kv, config.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: "deepseekv31_tool_parser",
- Optimization/support interpretation: The concrete diff surface is `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; keywords observed in patches: kv, config. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33090 - [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- Link: https://github.com/vllm-project/vllm/pull/33090
- Status/date: `merged`, created 2026-01-26, merged 2026-01-27; author `NickLucche`.
- Diff scope read: `1` files, `+1/-1`; areas: misc; keywords: cache, kv, mla.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: def __post_init__(self):; symbols: __post_init__
- Optimization/support interpretation: The concrete diff surface is `vllm/distributed/kv_transfer/kv_connector/utils.py`; keywords observed in patches: cache, kv, mla. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `vllm/distributed/kv_transfer/kv_connector/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- Link: https://github.com/vllm-project/vllm/pull/37421
- Status/date: `merged`, created 2026-03-18, merged 2026-04-08; author `LopezCastroRoberto`.
- Diff scope read: `9` files, `+2039/-483`; areas: model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, tests/benchmarks; keywords: topk, cuda, fp8, kv, mla, quant, spec, attention, config, scheduler.
- Code diff details:
  - `csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunks: +/*; symbols: int, int, int, size_t
  - `tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunks: def compare_top_k_results(; def test_top_k_per_row_decode_large_vocab_size(clean_logits: bool) -> None:; symbols: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk
  - `csrc/topk.cu` modified +139/-358 (497 lines); hunks: -// Portions of this file are adapted from SGLang PR:; symbols: int, int, size_t, size_t
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunks: logger = init_logger(__name__); def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer
  - `vllm/v1/attention/backends/mla/indexer.py` modified +0/-12 (12 lines); hunks: class DeepSeekV32IndexerDecodeMetadata:; def build(; symbols: DeepSeekV32IndexerDecodeMetadata:, build, build, build
- Optimization/support interpretation: The concrete diff surface is `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu`; keywords observed in patches: topk, cuda, fp8, kv, mla, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 6; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
