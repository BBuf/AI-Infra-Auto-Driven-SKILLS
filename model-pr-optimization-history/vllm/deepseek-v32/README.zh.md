# vLLM DeepSeek V3.2 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3.2 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- DeepSeek V3.2 is the major vLLM fork of the DeepSeek line because it adds sparse MLA and indexer logic.
- Most regressions land in the indexer builder, tokenizer / parser, and specialized decode kernels rather than the generic V3 loader.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## 已合入 PR

- [#25896](https://github.com/vllm-project/vllm/pull/25896) `Support DeepSeek-V3.2`：Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- [#25999](https://github.com/vllm-project/vllm/pull/25999) `Support indexer prefill chunking`：Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- [#26670](https://github.com/vllm-project/vllm/pull/26670) `Add AMD GPU support on DeepSeek v3.2 and SparseMLA`：Opened the ROCm SparseMLA lane for V3.2 deployments.
- [#29848](https://github.com/vllm-project/vllm/pull/29848) `Add DeepSeek-V3.2 tool parser`：Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- [#33090](https://github.com/vllm-project/vllm/pull/33090) `Fix DeepseekV32 `AssertionError: num_kv_heads == 1``：Removed a hard failure triggered by newer V3.2 attention shapes.
- [#37421](https://github.com/vllm-project/vllm/pull/37421) `Persistent TopK scheduler for DeepSeek-V3.2 decode`：Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3.2`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-29 | [#25896](https://github.com/vllm-project/vllm/pull/25896) | merged | [New Model] DeepSeek-V3.2 (Rebased to Main) | model wrapper, attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` |
| 2025-10-01 | [#25999](https://github.com/vllm-project/vllm/pull/25999) | merged | [Deepseek v3.2] Support indexer prefill chunking | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` |
| 2025-10-13 | [#26670](https://github.com/vllm-project/vllm/pull/26670) | merged | [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA | model wrapper, attention/backend, kernel, scheduler/runtime | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-02 | [#29848](https://github.com/vllm-project/vllm/pull/29848) | merged | Add DeepSeek-V3.2 tool parser. | misc | `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2026-01-26 | [#33090](https://github.com/vllm-project/vllm/pull/33090) | merged | [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1` | misc | `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-03-18 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu` |

### 逐 PR 代码 diff 阅读记录

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- 链接：https://github.com/vllm-project/vllm/pull/25896
- 状态/时间：`merged`，created 2025-09-29, merged 2025-09-30；作者 `zyongye`。
- 代码 diff 已读范围：`71` 个文件，`+3918/-221`；代码面：model wrapper, attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, cuda, fp8, cache, kv, flash, mla, config, quant, scheduler。
- 代码 diff 细节：
  - `vllm/v1/attention/backends/mla/flashmla_sparse.py` added +544/-0 (544 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _lse2_to_lse, FlashMLASparseBackend, get_name, get_metadata_cls
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunk: from transformers import DeepseekV2Config, DeepseekV3Config; class DeepseekV2Attention(nn.Module):; 符号: DeepseekV2MLP, DeepseekV2Attention, __init__, __init__
  - `tests/v1/attention/test_sparse_mla_backends.py` added +426/-0 (426 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _dequantize_fp8_ds_mla_entry, _quantize_dequantize_fp8_ds_mla, test_sparse_backend_metadata_registration, test_sparse_decode_metadata_filters_prefill_indices
  - `vllm/v1/attention/backends/mla/indexer.py` added +293/-0 (293 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepseekV32IndexerBackend, get_metadata_cls, get_supported_head_sizes, get_builder_cls
  - `tests/kernels/attention/test_deepgemm_attention.py` added +279/-0 (279 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: kv_cache_cast_to_fp8, per_custom_dims_cast_to_fp8, _generate_cp_test_data, _ref_fp8_mqa_logits
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`；patch 关键词为 attention, cuda, fp8, cache, kv, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- 链接：https://github.com/vllm-project/vllm/pull/25999
- 状态/时间：`merged`，created 2025-10-01, merged 2025-10-02；作者 `heheda12345`。
- 代码 diff 已读范围：`3` 个文件，`+149/-79`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：attention, cache, kv, mla, config, cuda, flash, fp8, quant, scheduler。
- 代码 diff 细节：
  - `vllm/v1/attention/backends/mla/indexer.py` modified +90/-41 (131 lines); hunk: def get_kv_cache_stride_order() -> tuple[int, ...]:; class DeepseekV32IndexerMetadata:; 符号: get_kv_cache_stride_order, DeepseekV32IndexerPrefillMetadata:, DeepseekV32IndexerPrefillChunkMetadata:, DeepseekV32IndexerPrefillMetadata:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunk: def sparse_attn_indexer(; 符号: sparse_attn_indexer
  - `tests/v1/attention/test_sparse_mla_backends.py` modified +22/-0 (22 lines); hunk: from vllm.v1.attention.backends.mla.flashmla_sparse import (; def test_sparse_backend_decode_correctness(dist_init, batch_name,; 符号: test_sparse_backend_decode_correctness, test_split_prefill_chunks
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py`；patch 关键词为 attention, cache, kv, mla, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/attention/backends/mla/indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `tests/v1/attention/test_sparse_mla_backends.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- 链接：https://github.com/vllm-project/vllm/pull/26670
- 状态/时间：`merged`，created 2025-10-13, merged 2025-11-20；作者 `ganyi1996ppo`。
- 代码 diff 已读范围：`9` 个文件，`+583/-15`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime；关键词：attention, cache, mla, fp8, kv, quant, spec, config, cuda, flash。
- 代码 diff 细节：
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` added +325/-0 (325 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: ROCMAiterMLASparseBackend, get_name, get_metadata_cls, get_builder_cls
  - `vllm/attention/ops/rocm_aiter_mla_sparse.py` added +210/-0 (210 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: fp8_mqa_logits_torch, rocm_fp8_mqa_logits, has_mqa_logits_module, fp8_paged_mqa_logits_torch
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunk: def sparse_attn_indexer(; def sparse_attn_indexer(; 符号: sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer
  - `vllm/v1/attention/backends/mla/indexer.py` modified +9/-6 (15 lines); hunk: ); class DeepseekV32IndexerBackend(AttentionBackend):; 符号: DeepseekV32IndexerBackend, get_supported_head_sizes, build
  - `vllm/platforms/rocm.py` modified +12/-1 (13 lines); hunk: def get_attn_backend_cls(; 符号: get_attn_backend_cls
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`；patch 关键词为 attention, cache, mla, fp8, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #29848 - Add DeepSeek-V3.2 tool parser.

- 链接：https://github.com/vllm-project/vllm/pull/29848
- 状态/时间：`merged`，created 2025-12-02, merged 2025-12-04；作者 `Xu-Wenqing`。
- 代码 diff 已读范围：`2` 个文件，`+595/-0`；代码面：misc；关键词：kv, config。
- 代码 diff 细节：
  - `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunk: "deepseekv31_tool_parser",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`；patch 关键词为 kv, config。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33090 - [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- 链接：https://github.com/vllm-project/vllm/pull/33090
- 状态/时间：`merged`，created 2026-01-26, merged 2026-01-27；作者 `NickLucche`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：misc；关键词：cache, kv, mla。
- 代码 diff 细节：
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunk: def __post_init__(self):; 符号: __post_init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/distributed/kv_transfer/kv_connector/utils.py`；patch 关键词为 cache, kv, mla。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `vllm/distributed/kv_transfer/kv_connector/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- 链接：https://github.com/vllm-project/vllm/pull/37421
- 状态/时间：`merged`，created 2026-03-18, merged 2026-04-08；作者 `LopezCastroRoberto`。
- 代码 diff 已读范围：`9` 个文件，`+2039/-483`；代码面：model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, tests/benchmarks；关键词：topk, cuda, fp8, kv, mla, quant, spec, attention, config, scheduler。
- 代码 diff 细节：
  - `csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunk: +/*; 符号: int, int, int, size_t
  - `tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunk: def compare_top_k_results(; def test_top_k_per_row_decode_large_vocab_size(clean_logits: bool) -> None:; 符号: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk
  - `csrc/topk.cu` modified +139/-358 (497 lines); hunk: -// Portions of this file are adapted from SGLang PR:; 符号: int, int, size_t, size_t
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunk: logger = init_logger(__name__); def sparse_attn_indexer(; 符号: sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer, sparse_attn_indexer
  - `vllm/v1/attention/backends/mla/indexer.py` modified +0/-12 (12 lines); hunk: class DeepSeekV32IndexerDecodeMetadata:; def build(; 符号: DeepSeekV32IndexerDecodeMetadata:, build, build, build
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu`；patch 关键词为 topk, cuda, fp8, kv, mla, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `csrc/persistent_topk.cuh`, `tests/kernels/test_top_k_per_row.py`, `csrc/topk.cu` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
