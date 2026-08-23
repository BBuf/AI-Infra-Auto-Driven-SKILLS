# vllm DeepSeek V3.2 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/ray_serving/elastic_ep/serve_deepseek_v2.sh` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP_MI325.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP_MI325.yaml` | no direct PR-number commit |
| `tests/kernels/test_fused_deepseek_v32_norm_rope.py` | [#46876](https://github.com/vllm-project/vllm/pull/46876), [#47207](https://github.com/vllm-project/vllm/pull/47207), [#52046](https://github.com/vllm-project/vllm/pull/52046), [#52381](https://github.com/vllm-project/vllm/pull/52381) |
| `tests/models/deepseek_v32/test_sequence_parallel.py` | [#51434](https://github.com/vllm-project/vllm/pull/51434) |
| `tests/parser/engine/test_deepseek_v32.py` | no direct PR-number commit |
| `tests/tool_parsers/test_deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#36056](https://github.com/vllm-project/vllm/pull/36056), [#41198](https://github.com/vllm-project/vllm/pull/41198), [#41801](https://github.com/vllm-project/vllm/pull/41801), [#43019](https://github.com/vllm-project/vllm/pull/43019), [#43255](https://github.com/vllm-project/vllm/pull/43255) |
| `vllm/model_executor/models/deepseek_mtp.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#38684](https://github.com/vllm-project/vllm/pull/38684), [#38870](https://github.com/vllm-project/vllm/pull/38870), [#48036](https://github.com/vllm-project/vllm/pull/48036) |
| `vllm/model_executor/models/deepseek_v2.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#25999](https://github.com/vllm-project/vllm/pull/25999), [#26456](https://github.com/vllm-project/vllm/pull/26456), [#26465](https://github.com/vllm-project/vllm/pull/26465), [#26670](https://github.com/vllm-project/vllm/pull/26670), [#26763](https://github.com/vllm-project/vllm/pull/26763), [#27532](https://github.com/vllm-project/vllm/pull/27532), [#27568](https://github.com/vllm-project/vllm/pull/27568), [#28968](https://github.com/vllm-project/vllm/pull/28968), [#29287](https://github.com/vllm-project/vllm/pull/29287), [#30841](https://github.com/vllm-project/vllm/pull/30841), [#31046](https://github.com/vllm-project/vllm/pull/31046), ... (22 total) |
| `vllm/models/deepseek_v32/__init__.py` | [#46808](https://github.com/vllm-project/vllm/pull/46808), [#47207](https://github.com/vllm-project/vllm/pull/47207) |
| `vllm/models/deepseek_v32/amd/__init__.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207) |
| `vllm/models/deepseek_v32/amd/model.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207) |
| `vllm/models/deepseek_v32/amd/mtp.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207) |
| `vllm/models/deepseek_v32/amd/rocm.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207), [#50904](https://github.com/vllm-project/vllm/pull/50904), [#51425](https://github.com/vllm-project/vllm/pull/51425) |
| `vllm/models/deepseek_v32/attention.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207), [#50904](https://github.com/vllm-project/vllm/pull/50904), [#51298](https://github.com/vllm-project/vllm/pull/51298), [#51425](https://github.com/vllm-project/vllm/pull/51425), [#52046](https://github.com/vllm-project/vllm/pull/52046), [#53021](https://github.com/vllm-project/vllm/pull/53021) |
| `vllm/models/deepseek_v32/common/__init__.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207) |
| `vllm/models/deepseek_v32/common/kernels.py` | [#47207](https://github.com/vllm-project/vllm/pull/47207), [#52046](https://github.com/vllm-project/vllm/pull/52046), [#52381](https://github.com/vllm-project/vllm/pull/52381) |
| `vllm/models/deepseek_v32/nvidia/__init__.py` | [#46808](https://github.com/vllm-project/vllm/pull/46808) |
| `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` | no direct PR-number commit |
| `vllm/models/deepseek_v32/nvidia/model.py` | [#46808](https://github.com/vllm-project/vllm/pull/46808), [#46876](https://github.com/vllm-project/vllm/pull/46876), [#47207](https://github.com/vllm-project/vllm/pull/47207), [#51434](https://github.com/vllm-project/vllm/pull/51434) |
| `vllm/models/deepseek_v32/nvidia/mtp.py` | [#46808](https://github.com/vllm-project/vllm/pull/46808), [#46876](https://github.com/vllm-project/vllm/pull/46876), [#47207](https://github.com/vllm-project/vllm/pull/47207), [#48036](https://github.com/vllm-project/vllm/pull/48036), [#51434](https://github.com/vllm-project/vllm/pull/51434) |
| `vllm/models/deepseek_v32/nvidia/ops/__init__.py` | no direct PR-number commit |
| `vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py` | no direct PR-number commit |
| `vllm/parser/deepseek_v32.py` | no direct PR-number commit |
| `vllm/renderers/deepseek_v32.py` | [#33855](https://github.com/vllm-project/vllm/pull/33855) |
| `vllm/tokenizers/deepseek_v32.py` | [#30658](https://github.com/vllm-project/vllm/pull/30658), [#33855](https://github.com/vllm-project/vllm/pull/33855), [#37004](https://github.com/vllm-project/vllm/pull/37004) |
| `vllm/tokenizers/deepseek_v32_encoding.py` | [#29837](https://github.com/vllm-project/vllm/pull/29837), [#30025](https://github.com/vllm-project/vllm/pull/30025), [#31147](https://github.com/vllm-project/vllm/pull/31147), [#32884](https://github.com/vllm-project/vllm/pull/32884) |
| `vllm/tool_parsers/deepseekv32_engine_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 47
- Extra PRs preserved from existing docs: 15
- Total PRs in this document: 62
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-09-30 | [#25896](https://github.com/vllm-project/vllm/pull/25896) | merged | [New Model] DeepSeek-V3.2 (Rebased to Main) | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2025-10-02 | [#25999](https://github.com/vllm-project/vllm/pull/25999) | merged | [Deepseek v3.2] Support indexer prefill chunking | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-15 | [#26456](https://github.com/vllm-project/vllm/pull/26456) | merged | [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26763](https://github.com/vllm-project/vllm/pull/26763) | merged | [Deepseek v3.2] Optimize top_k_per_row | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26465](https://github.com/vllm-project/vllm/pull/26465) | merged | [Deepseek v3.2] Remove extra logics in indexer | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-19 | [#28968](https://github.com/vllm-project/vllm/pull/28968) | merged | [DeepSeek] Fix DeepSeek V3.2 Rope Embedding | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-20 | [#26670](https://github.com/vllm-project/vllm/pull/26670) | merged | [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-03 | [#29837](https://github.com/vllm-project/vllm/pull/29837) | merged | [Frontend] supports deepseekv32 chat template | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2025-12-04 | [#30025](https://github.com/vllm-project/vllm/pull/30025) | merged | [Bugfix] fixed deepseekv32 tool calling error | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2025-12-04 | [#29848](https://github.com/vllm-project/vllm/pull/29848) | merged | Add DeepSeek-V3.2 tool parser. | `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2025-12-08 | [#27568](https://github.com/vllm-project/vllm/pull/27568) | merged | [DeepSeek v3.2] Make top-k work for any logit values. | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-12 | [#27532](https://github.com/vllm-project/vllm/pull/27532) | merged | [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-13 | [#30609](https://github.com/vllm-project/vllm/pull/30609) | merged | [Refactor] `TokenizerRegistry` only uses lazy imports | `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py` |
| 2025-12-15 | [#30658](https://github.com/vllm-project/vllm/pull/30658) | merged | [Bugfix] Fix deepseek_v32 tokenizer_mode | `vllm/tokenizers/deepseek_v32.py` |
| 2025-12-17 | [#30841](https://github.com/vllm-project/vllm/pull/30841) | merged | [Bugfix] deepseek-V3.2 self.weights_proj has no bias | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-19 | [#31046](https://github.com/vllm-project/vllm/pull/31046) | merged | [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-24 | [#31160](https://github.com/vllm-project/vllm/pull/31160) | merged | [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-05 | [#31147](https://github.com/vllm-project/vllm/pull/31147) | merged | Add chat prefix completion feature to DeepSeek v3.2 | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2026-01-16 | [#32175](https://github.com/vllm-project/vllm/pull/32175) | merged | [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-21 | [#29287](https://github.com/vllm-project/vllm/pull/29287) | merged | [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-22 | [#30207](https://github.com/vllm-project/vllm/pull/30207) | merged | Enable Cross layers KV cache layout at NIXL Connector | `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-01-23 | [#32884](https://github.com/vllm-project/vllm/pull/32884) | merged | [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2026-01-27 | [#33086](https://github.com/vllm-project/vllm/pull/33086) | closed | [Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1 | `vllm/v1/attention/backends/mla/indexer.py` |
| 2026-01-27 | [#33090](https://github.com/vllm-project/vllm/pull/33090) | merged | [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1` | `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-02-02 | [#33566](https://github.com/vllm-project/vllm/pull/33566) | merged | [CI] Add DeepSeek V3.2 nightly eval | `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` |
| 2026-02-06 | [#33964](https://github.com/vllm-project/vllm/pull/33964) | merged | [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32 | `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-02-08 | [#33855](https://github.com/vllm-project/vllm/pull/33855) | merged | [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used | `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py` |
| 2026-03-13 | [#37004](https://github.com/vllm-project/vllm/pull/37004) | merged | [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces | `vllm/tokenizers/deepseek_v32.py` |
| 2026-03-19 | [#36056](https://github.com/vllm-project/vllm/pull/36056) | merged | [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-03-30 | [#33703](https://github.com/vllm-project/vllm/pull/33703) | merged | [Bugfix] Support multi-type params parsing for DeepSeek v3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-02 | [#38684](https://github.com/vllm-project/vllm/pull/38684) | merged | [Perf] DSV3.2 Indexer Fused Weights Projection | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-04 | [#38870](https://github.com/vllm-project/vllm/pull/38870) | merged | [Bugfix] Fix DSV32 weight loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-08 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-04-27 | [#35968](https://github.com/vllm-project/vllm/pull/35968) | closed | [Performance] DeepSeek V3.2 multi-stream indexer overlap | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py` |
| 2026-04-29 | [#41198](https://github.com/vllm-project/vllm/pull/41198) | merged | [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-05-01 | [#41217](https://github.com/vllm-project/vllm/pull/41217) | merged | [ROCm][Deepseek] dsv3.2 further optimization | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-05-06 | [#41801](https://github.com/vllm-project/vllm/pull/41801) | merged | [Bugfix] DeepSeekV32/v4: respect string='true\|false' attribute andunwrap arguments/input wrapper | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-05-07 | [#41835](https://github.com/vllm-project/vllm/pull/41835) | merged | [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-05-14 | [#42062](https://github.com/vllm-project/vllm/pull/42062) | merged | [ROCm] Enable gluon paged MQA logits on gfx950 (MI355X) | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| 2026-05-20 | [#43019](https://github.com/vllm-project/vllm/pull/43019) | merged | [Bugfix] Use shared coerce_to_schema_type in DeepSeekV32 tool parser | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-05-21 | [#43255](https://github.com/vllm-project/vllm/pull/43255) | merged | [CI] Add composed-schema regression tests for DeepSeek V3.2/V4 parsers | `tests/tool_parsers/test_deepseekv32_tool_parser.py` |
| 2026-05-28 | [#42879](https://github.com/vllm-project/vllm/pull/42879) | merged | [Bugfix] Stream DeepSeek DSML tool-call argument deltas incrementally | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py` |
| 2026-05-28 | [#43781](https://github.com/vllm-project/vllm/pull/43781) | merged | [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950 | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| 2026-05-29 | [#42982](https://github.com/vllm-project/vllm/pull/42982) | merged | [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts) | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-06-01 | [#42944](https://github.com/vllm-project/vllm/pull/42944) | merged | fix: glm5.1 pp model loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-06-07 | [#44420](https://github.com/vllm-project/vllm/pull/44420) | merged | [feature] add index share feature for DSA MTP | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-19 | [#45895](https://github.com/vllm-project/vllm/pull/45895) | merged | [bugfix]Indexer init skip and MTP TopK share for iteration | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-06-20 | [#46199](https://github.com/vllm-project/vllm/pull/46199) | merged | [Bugfix] Move extract_layer_index back inside is_v32 guard | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-06-25 | [#46651](https://github.com/vllm-project/vllm/pull/46651) | merged | [Perf] Remove redundant clone for GLM, Deepseek etc | `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-06-26 | [#46808](https://github.com/vllm-project/vllm/pull/46808) | merged | [GLM-5] Add DSV3.2/GLM5 to `vllm/models/` | `vllm/models/deepseek_v32/nvidia/mtp.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/__init__.py` |
| 2026-06-28 | [#46600](https://github.com/vllm-project/vllm/pull/46600) | merged | [Bugfix][DSv3.2] Skip indexer weights for index-cache-skipped layers | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-06-28 | [#46876](https://github.com/vllm-project/vllm/pull/46876) | merged | [GLM5] Implement op fusion for GLM5/DSV3.2 | `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py` |
| 2026-07-14 | [#48036](https://github.com/vllm-project/vllm/pull/48036) | merged | [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel | `vllm/model_executor/models/deepseek_mtp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py` |
| 2026-07-30 | [#47207](https://github.com/vllm-project/vllm/pull/47207) | merged | [ROCm]Migrating Deepseek V3.2 to vllm/models/deepseek_v32/ | `vllm/models/deepseek_v32/amd/mtp.py`, `vllm/models/deepseek_v32/amd/model.py`, `vllm/models/deepseek_v32/amd/rocm.py` |
| 2026-08-06 | [#50904](https://github.com/vllm-project/vllm/pull/50904) | merged | [GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement | `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py` |
| 2026-08-07 | [#51298](https://github.com/vllm-project/vllm/pull/51298) | merged | [DSv32/GLM Perf] Skip short prefill topk for dense mha layer, 97.9% kernel level latency reduction | `vllm/models/deepseek_v32/attention.py` |
| 2026-08-07 | [#51425](https://github.com/vllm-project/vllm/pull/51425) | merged | [Perf] Narrow DeepSeek V3.2 eager CUDA graph region | `vllm/models/deepseek_v32/attention.py`, `vllm/models/deepseek_v32/amd/rocm.py` |
| 2026-08-07 | [#51434](https://github.com/vllm-project/vllm/pull/51434) | merged | [Perf] Optimize DeepSeek V3.2 sequence parallelism | `tests/models/deepseek_v32/test_sequence_parallel.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py` |
| 2026-08-18 | [#52381](https://github.com/vllm-project/vllm/pull/52381) | merged | Harden DeepSeek V3.2 fused kernel grids | `vllm/models/deepseek_v32/common/kernels.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py` |
| 2026-08-18 | [#52046](https://github.com/vllm-project/vllm/pull/52046) | merged | [nv] add pcp support in dsv3.2 | `vllm/models/deepseek_v32/common/kernels.py`, `vllm/models/deepseek_v32/attention.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py` |
| 2026-08-20 | [#53021](https://github.com/vllm-project/vllm/pull/53021) | merged | [Model] Remove unused DeepseekV32Indexer forward | `vllm/models/deepseek_v32/attention.py` |

## Per-PR Diff Audit Cards

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- Link: https://github.com/vllm-project/vllm/pull/25896
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `fa7e254a7f3e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 71 files, +3918/-221, 5400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[New Model] DeepSeek-V3.2 (Rebased to Main)"; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[New Model] DeepSeek-V3.2 (Rebased to Main)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__, touching `DeepseekV2MLP, DeepseekV2Attention, __init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -33,36 +33,57 @@
+from vllm.attention.backends.abstract import AttentionBackend
+from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
-from vllm.config import CacheConfig, ParallelConfig, VllmConfig
+from vllm.config import (CacheConfig, ParallelConfig, VllmConfig,
+                         get_current_vllm_config)
+from vllm.forward_context import get_forward_context
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
+        self.is_v32 = hasattr(config, "index_topk")
+        if self.is_v32:
+            topk_tokens = config.index_topk
+            topk_indices_buffer = torch.empty(
+                vllm_config.scheduler_config.max_num_batched_tokens,
+                topk_tokens,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion_attn.py`, `tests/kernels/attention/test_cache.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/attention/test_flashmla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- Link: https://github.com/vllm-project/vllm/pull/25999
- Status/date: merged / 2025-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `1e50f1be7058`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +149/-79, 324 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Support indexer prefill chunking"; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Support indexer prefill chunking"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -583,44 +583,43 @@ def sparse_attn_indexer(
-        num_prefills = attn_metadata.num_prefills
-        k_fp8 = torch.empty([prefill_metadata.total_seq_lens, head_dim],
-                            device=k.device,
-                            dtype=torch.float8_e4m3fn)
-        k_scale = torch.empty([prefill_metadata.total_seq_lens, 1],
-                              device=k.device,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_sparse_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26456 - [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather

- Link: https://github.com/vllm-project/vllm/pull/26456
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `f5ed68ef63d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-68, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer, touching `get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -75,7 +75,7 @@
-from vllm.utils import cdiv, direct_register_custom_op
+from vllm.utils import direct_register_custom_op
@@ -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:
-@torch.inference_mode()
-def cp_gather_indexer_k_quant_cache(
-    kv_cache,  # [num_blocks, block_size, head_dim + 1]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26763 - [Deepseek v3.2] Optimize top_k_per_row

- Link: https://github.com/vllm-project/vllm/pull/26763
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `80e94529845d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +13/-49, 203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Optimize top_k_per_row"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Optimize top_k_per_row"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -577,15 +577,11 @@ def sparse_attn_indexer(
-            topk_values = torch.empty(
-                num_rows, topk_tokens, dtype=logits.dtype, device=logits.device
-            )
-                topk_values,
@@ -642,15 +638,11 @@ def sparse_attn_indexer(
-        topk_values = torch.empty(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26465 - [Deepseek v3.2] Remove extra logics in indexer

- Link: https://github.com/vllm-project/vllm/pull/26465
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `09a7e6f6179b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +141/-40, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek v3.2] Remove extra logics in indexer"; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Deepseek v3.2] Remove extra logics in indexer"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -574,9 +574,9 @@ def sparse_attn_indexer(
-            topk_indices = torch.empty(
-                num_rows, topk_tokens, dtype=torch.int32, device=logits.device
-            )
+            topk_indices = topk_indices_buffer[
+                chunk.token_start : chunk.token_end, :topk_tokens
+            ]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28968 - [DeepSeek] Fix DeepSeek V3.2 Rope Embedding

- Link: https://github.com/vllm-project/vllm/pull/28968
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `88f5b19f0bc6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-3, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek] Fix DeepSeek V3.2 Rope Embedding"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek] Fix DeepSeek V3.2 Rope Embedding"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -846,8 +846,8 @@ def forward(
-        q = torch.cat([q_pe, q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)
+        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
+        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
@@ -1000,6 +1000,14 @@ def __init__(
+            self.indexer_rope_emb = get_rope(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- Link: https://github.com/vllm-project/vllm/pull/26670
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `06c20c990464`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +583/-15, 700 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA"; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -591,6 +591,7 @@ def sparse_attn_indexer(
+    fp8_dtype = current_platform.fp8_dtype()
@@ -630,7 +631,7 @@ def sparse_attn_indexer(
-                dtype=torch.float8_e4m3fn,
+                dtype=fp8_dtype,
@@ -644,7 +645,12 @@ def sparse_attn_indexer(
-            logits = fp8_mqa_logits(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29837 - [Frontend] supports deepseekv32 chat template

- Link: https://github.com/vllm-project/vllm/pull/29837
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `b78772c43351`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +616/-2, 660 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] supports deepseekv32 chat template"; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/tokenizers/deepseek_v32_encoding.py`; technical summary: Covers "[Frontend] supports deepseekv32 chat template"; the main implementation surface is `vllm/tokenizers/deepseek_v32_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -0,0 +1,456 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# copy from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
+import copy
+import json
+import re
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30025 - [Bugfix] fixed deepseekv32 tool calling error

- Link: https://github.com/vllm-project/vllm/pull/30025
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `82a64b3d8f93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-3, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fixed deepseekv32 tool calling error"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tokenizers/deepseek_v32_encoding.py`; technical summary: Covers "[Bugfix] fixed deepseekv32 tool calling error"; the main implementation surface is `vllm/tokenizers/deepseek_v32_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml, touching `tool_calls_to_openai_format, encode_arguments_to_dsml`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):
-    arguments = json.loads(tool_call["arguments"])
+    if isinstance(tool_call["arguments"], str):
+        arguments = json.loads(tool_call["arguments"])
+    else:
+        arguments = tool_call["arguments"]
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`, `vllm/tokenizers/deepseekv32.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29848 - Add DeepSeek-V3.2 tool parser.

- Link: https://github.com/vllm-project/vllm/pull/29848
- Status/date: merged / 2025-12-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +595/-0, 603 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek-V3.2 tool parser."; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; technical summary: Covers "Add DeepSeek-V3.2 tool parser."; the main implementation surface is `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id, touching `DeepSeekV32ToolParser, __init__, type`; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10
- Key code excerpts:

```diff
diff -- vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py
@@ -0,0 +1,591 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import uuid
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/entrypoints/openai/tool_parsers/__init__.py
@@ -30,6 +30,10 @@
+    "deepseek_v32": (
+        "deepseekv32_tool_parser",
+        "DeepSeekV32ToolParser",
+    ),
```

- Reviewed files:
  - runtime: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27568 - [DeepSeek v3.2] Make top-k work for any logit values.

- Link: https://github.com/vllm-project/vllm/pull/27568
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `184076c3fecf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +629/-210, 1067 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek v3.2] Make top-k work for any logit values."; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek v3.2] Make top-k work for any logit values."; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -684,18 +684,18 @@ def sparse_attn_indexer(
-            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
-            torch.ops._C.top_k_per_row(
+            torch.ops._C.top_k_per_row_prefill(
+                topk_tokens,
@@ -738,7 +738,6 @@ def sparse_attn_indexer(
-        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27532 - [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

- Link: https://github.com/vllm-project/vllm/pull/27532
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `3e41992fecdc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 30 files, +1372/-256, 2323 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -83,6 +83,7 @@
+from vllm.v1.worker.workspace import current_workspace_manager
@@ -618,8 +619,15 @@ def sparse_attn_indexer(
+        # Reserve workspace for indexer during profiling run
+        current_workspace_manager().get_simultaneous(
+            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
+            ((total_seq_lens, 4), torch.uint8),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30609 - [Refactor] `TokenizerRegistry` only uses lazy imports

- Link: https://github.com/vllm-project/vllm/pull/30609
- Status/date: merged / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +202/-176, 707 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] `TokenizerRegistry` only uses lazy imports"; model line: DeepSeek V3.2; category: docs/tests/CI; main diff: `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py`; technical summary: Covers "[Refactor] `TokenizerRegistry` only uses lazy imports"; the main implementation surface is `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/registry.py` modified +100/-100 (200 lines); hunks: -1,13 +1,13; -24,46 +24,25; symbols: TokenizerRegistry, register, _TokenizerRegistry, touching `TokenizerRegistry, register, _TokenizerRegistry`; `tests/tokenizers_/test_basic.py` modified +24/-23 (47 lines); hunks: -3,38 +3,39; symbols: _get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol, touching `_get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol`; `vllm/tokenizers/deepseekv32.py` modified +33/-14 (47 lines); hunks: -2,24 +2,18; -40,7 +34,21 @@ def from_pretrained(; symbols: DeepseekV32Tokenizer, __init__, from_pretrained, touching `DeepseekV32Tokenizer, __init__, from_pretrained`; `tests/tokenizers_/test_registry.py` modified +21/-2 (23 lines); hunks: -2,7 +2,14; -40,10 +47,22 @@ def is_fast(self) -> bool:; symbols: TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent, test_customized_tokenizer, touching `TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent`.
- Code diff details:
  - `vllm/tokenizers/registry.py` modified +100/-100 (200 lines); hunks: -1,13 +1,13; -24,46 +24,25; symbols: TokenizerRegistry, register, _TokenizerRegistry
  - `tests/tokenizers_/test_basic.py` modified +24/-23 (47 lines); hunks: -3,38 +3,39; symbols: _get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol
  - `vllm/tokenizers/deepseekv32.py` modified +33/-14 (47 lines); hunks: -2,24 +2,18; -40,7 +34,21 @@ def from_pretrained(; symbols: DeepseekV32Tokenizer, __init__, from_pretrained
  - `tests/tokenizers_/test_registry.py` modified +21/-2 (23 lines); hunks: -2,7 +2,14; -40,10 +47,22 @@ def is_fast(self) -> bool:; symbols: TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent, test_customized_tokenizer
  - `vllm/tokenizers/hf.py` modified +7/-12 (19 lines); hunks: -3,22 +3,18; -65,11 +61,10 @@ def __reduce__(self):; symbols: get_cached_tokenizer, __reduce__, HfTokenizer, CachedHfTokenizer
- Key code excerpts:

```diff
diff -- vllm/tokenizers/registry.py
@@ -1,13 +1,13 @@
-from collections.abc import Callable
+from dataclasses import dataclass, field
-from typing import TYPE_CHECKING, TypeVar, overload
+from typing import TYPE_CHECKING
-from typing_extensions import assert_never
+from typing_extensions import TypeVar, assert_never, deprecated
diff -- tests/tokenizers_/test_basic.py
@@ -3,38 +3,39 @@
-from transformers import PreTrainedTokenizerBase
+from transformers import (
+    PreTrainedTokenizer,
+    PreTrainedTokenizerBase,
+    PreTrainedTokenizerFast,
+)
diff -- vllm/tokenizers/deepseekv32.py
@@ -2,24 +2,18 @@
```

- Reviewed files:
  - runtime: `vllm/tokenizers/registry.py` modified +100/-100; `vllm/tokenizers/deepseekv32.py` modified +33/-14; `vllm/tokenizers/hf.py` modified +7/-12; `vllm/tokenizers/mistral.py` modified +2/-5; `vllm/tokenizers/__init__.py` modified +0/-6; `vllm/transformers_utils/tokenizer.py` modified +3/-3
  - tests: `tests/tokenizers_/test_basic.py` modified +24/-23; `tests/tokenizers_/test_registry.py` modified +21/-2
- Risk and verification: The diff ships test coverage in `tests/test_inputs.py`, `tests/tokenizers_/test_basic.py`, `tests/tokenizers_/test_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30658 - [Bugfix] Fix deepseek_v32 tokenizer_mode

- Link: https://github.com/vllm-project/vllm/pull/30658
- Status/date: merged / 2025-12-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32.py`; associated commits `a524d1ba0af4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix deepseek_v32 tokenizer_mode"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tokenizers/deepseek_v32.py`; technical summary: Covers "[Bugfix] Fix deepseek_v32 tokenizer_mode"; the main implementation surface is `vllm/tokenizers/deepseek_v32.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/deepseek_v32.py`, `vllm/tokenizers/registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30841 - [Bugfix] deepseek-V3.2 self.weights_proj has no bias

- Link: https://github.com/vllm-project/vllm/pull/30841
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `84896fda22d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] deepseek-V3.2 self.weights_proj has no bias"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] deepseek-V3.2 self.weights_proj has no bias"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -835,7 +835,11 @@ def __init__(
-            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
+            hidden_size,
+            self.n_head,
+            bias=False,
+            quant_config=None,
+            prefix=f"{prefix}.weights_proj",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31046 - [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31046
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `4cf9429897c1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,8 +878,11 @@ def forward(
-        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
+        # `rotary_emb` is shape-preserving; `q_pe` is already
+        # [num_tokens, n_head, rope_dim].
+        q = torch.cat([q_pe, q_nope], dim=-1)
+        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31160 - [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31160
- Status/date: merged / 2025-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `76e6a951925b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,11 +878,14 @@ def forward(
-        # `rotary_emb` is shape-preserving; `q_pe` is already
-        # [num_tokens, n_head, rope_dim].
+        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
+        # so we need to reshape back to token-flattened shapes
+        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
+        k_pe = k_pe.reshape(-1, 1, self.rope_dim)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31147 - Add chat prefix completion feature to DeepSeek v3.2

- Link: https://github.com/vllm-project/vllm/pull/31147
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `346e56455a3b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-5, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add chat prefix completion feature to DeepSeek v3.2"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tokenizers/deepseek_v32_encoding.py`; technical summary: Covers "Add chat prefix completion feature to DeepSeek v3.2"; the main implementation surface is `vllm/tokenizers/deepseek_v32_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message, touching `render_message`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -169,6 +169,7 @@ def render_message(
+    is_prefix = msg.get("prefix", False)
@@ -273,11 +274,14 @@ def render_message(
-        prompt += assistant_msg_template.format(
-            reasoning=thinking_part,
-            content=summary_content,
-            tool_calls=tool_calls_content,
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- Link: https://github.com/vllm-project/vllm/pull/32175
- Status/date: merged / 2026-01-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `5de6dd0662da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -717,13 +717,20 @@ def sparse_attn_indexer(
+            # [num_decode_tokens, n_head, head_dim] -> [bs, 1+next_n, n_head, head_dim]
+            # [num_decode_tokens, n_head] -> [bs, 1+next_n, n_head]
+            padded_weights = pack_seq_triton(weights[:num_decode_tokens], decode_lens)
+            # [bs, 1+next_n, n_head] -> [bs * next_n, n_head]
+            padded_weights = padded_weights.flatten(0, 1)
+            padded_weights = weights
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- Link: https://github.com/vllm-project/vllm/pull/29287
- Status/date: merged / 2026-01-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `6c20e89c0209`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +982/-323, 1521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer, touching `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -43,7 +43,6 @@
-from vllm.forward_context import get_forward_context
@@ -63,6 +62,7 @@
+from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
@@ -74,16 +74,11 @@
-from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
-from vllm.utils.torch_utils import direct_register_custom_op
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30207 - Enable Cross layers KV cache layout at NIXL Connector

- Link: https://github.com/vllm-project/vllm/pull/30207
- Status/date: merged / 2026-01-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +308/-89, 729 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable Cross layers KV cache layout at NIXL Connector"; model line: DeepSeek V3.2; category: docs/tests/CI; main diff: `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py`; technical summary: Covers "Enable Cross layers KV cache layout at NIXL Connector"; the main implementation surface is `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47 (225 lines); hunks: -18,8 +18,12; -48,8 +52,11; symbols: test_kv_transfer_handshake, __init__, _nixl_handshake, req_id, touching `test_kv_transfer_handshake, __init__, _nixl_handshake`; `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38 (111 lines); hunks: -54,7 +54,7; -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):; symbols: NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv, NixlConnector, touching `NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv`; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2 (41 lines); hunks: -316,27 +316,56 @@ class TpKVTopology:; -346,6 +375,14 @@ def tp_size(self) -> int:; symbols: TpKVTopology, __post_init__, is_kv_layout_blocks_first, split_k_and_v, touching `TpKVTopology, __post_init__, is_kv_layout_blocks_first`; `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2 (11 lines); hunks: -34,11 +34,18 @@ else.
- Code diff details:
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47 (225 lines); hunks: -18,8 +18,12; -48,8 +52,11; symbols: test_kv_transfer_handshake, __init__, _nixl_handshake, req_id
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38 (111 lines); hunks: -54,7 +54,7; -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):; symbols: NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv, NixlConnector
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2 (41 lines); hunks: -316,27 +316,56 @@ class TpKVTopology:; -346,6 +375,14 @@ def tp_size(self) -> int:; symbols: TpKVTopology, __post_init__, is_kv_layout_blocks_first, split_k_and_v
  - `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2 (11 lines); hunks: -34,11 +34,18 @@ else
  - `docs/features/nixl_connector_usage.md` modified +9/-0 (9 lines); hunks: -184,6 +184,15 @@ Support use case: Prefill with 'HND' and decode with 'NHD'...
- Key code excerpts:

```diff
diff -- tests/v1/kv_connector/unit/test_nixl_connector.py
@@ -18,8 +18,12 @@
-from vllm.config import KVTransferConfig
-from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
+from vllm.config import KVTransferConfig, set_current_vllm_config
+from vllm.distributed.kv_transfer.kv_connector.utils import (
+    KVOutputAggregator,
+    TpKVTopology,
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@@ -54,7 +54,7 @@
-from vllm.v1.attention.backend import AttentionMetadata
+from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
@@ -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):
-    vllm_config: VllmConfig, attn_backend_name: str
+    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
@@ -216,6 +216,7 @@ def compute_nixl_compatibility_hash(
diff -- vllm/distributed/kv_transfer/kv_connector/utils.py
@@ -316,27 +316,56 @@ class TpKVTopology:
```

- Reviewed files:
  - tests: `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47; `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2
  - docs: `docs/features/nixl_connector_usage.md` modified +9/-0
- Risk and verification: The diff ships test coverage in `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh`, `tests/v1/kv_connector/unit/test_nixl_connector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32884 - [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions

- Link: https://github.com/vllm-project/vllm/pull/32884
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `f61c9da711d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +39/-28, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tokenizers/deepseek_v32_encoding.py`; technical summary: Covers "[BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions"; the main implementation surface is `vllm/tokenizers/deepseek_v32_encoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message, touching `find_last_user_index, render_message`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -> int:
-    assert 0 <= index < len(messages)
-    assert thinking_mode in ["chat", "thinking"], (
-        f"Invalid thinking_mode `{thinking_mode}`"
-    )
+    if not (0 <= index < len(messages)):
+        raise ValueError(
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33086 - [Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1

- Link: https://github.com/vllm-project/vllm/pull/33086
- Status/date: closed / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/v1/attention/backends/mla/indexer.py`; technical summary: Covers "[Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1"; the main implementation surface is `vllm/v1/attention/backends/mla/indexer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def get_kv_cache_shape(; symbols: get_kv_cache_shape, touching `get_kv_cache_shape`.
- Code diff details:
  - `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def get_kv_cache_shape(; symbols: get_kv_cache_shape
- Key code excerpts:

```diff
diff -- vllm/v1/attention/backends/mla/indexer.py
@@ -49,7 +49,6 @@ def get_kv_cache_shape(
-        assert num_kv_heads == 1
```

- Reviewed files:
  - runtime: `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/mla/indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33090 - [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- Link: https://github.com/vllm-project/vllm/pull/33090
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/distributed/kv_transfer/kv_connector/utils.py`; technical summary: Covers "[Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`"; the main implementation surface is `vllm/distributed/kv_transfer/kv_connector/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/utils.py
@@ -322,7 +322,7 @@ def __post_init__(self):
-            num_blocks=1, block_size=16, num_kv_heads=4, head_size=1
+            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
```

- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/distributed/kv_transfer/kv_connector/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33566 - [CI] Add DeepSeek V3.2 nightly eval

- Link: https://github.com/vllm-project/vllm/pull/33566
- Status/date: merged / 2026-02-02
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/33566/files?per_page=100 --paginate Get "https://api.github.com/repos/vllm-project/vllm/pulls/33566/files?per_page=100": EOF`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`; associated commits `9f8cb81b44ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add DeepSeek V3.2 nightly eval"; model line: DeepSeek V3.2; category: docs/tests/CI; main diff: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`; technical summary: Covers "[CI] Add DeepSeek V3.2 nightly eval"; the main implementation surface is `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11.
- Code diff details:
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-V3.2"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
diff -- tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-V3.2"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0; `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`, `tests/evals/gsm8k/configs/models-h200.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33964 - [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32

- Link: https://github.com/vllm-project/vllm/pull/33964
- Status/date: merged / 2026-02-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32"; the main implementation surface is `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state, touching `_generate_tool_call_id, adjust_request, _reset_streaming_state`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:
+    def adjust_request(self, request):
+        request = super().adjust_request(request)
+        if request.tools and request.tool_choice != "none":
+            # Ensure tool call tokens
+            # (<｜DSML｜function_calls>, </｜DSML｜function_calls>)
+            # are not skippedduring decoding.
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/deepseekv32_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33855 - [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used

- Link: https://github.com/vllm-project/vllm/pull/33855
- Status/date: merged / 2026-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/renderers/deepseek_v32.py`, `vllm/tokenizers/deepseek_v32.py`; associated commits `a96197f564cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-203, 348 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py`; technical summary: Covers "[Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used"; the main implementation surface is `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer, touching `DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer`; `vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer, touching `__init__, tokenizer, get_tokenizer`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer
  - `vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32.py
@@ -1,191 +1,89 @@
+import copy
+from typing import Any
-from pathlib import Path
-from typing import Any, overload
-from transformers import BatchEncoding
+from transformers import AutoTokenizer
diff -- vllm/renderers/deepseek_v32.py
@@ -13,6 +13,7 @@
+from ..tokenizers.hf import HfTokenizer
@@ -48,10 +49,10 @@ def __init__(
-    def tokenizer(self) -> DeepseekV32Tokenizer | None:
+    def tokenizer(self) -> HfTokenizer | None:
-    def get_tokenizer(self) -> DeepseekV32Tokenizer:
+    def get_tokenizer(self) -> HfTokenizer:
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +77/-179; `vllm/renderers/deepseek_v32.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37004 - [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces

- Link: https://github.com/vllm-project/vllm/pull/37004
- Status/date: merged / 2026-03-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32.py`; associated commits `9efc4db9658a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tokenizers/deepseek_v32.py`; technical summary: Covers "[Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces"; the main implementation surface is `vllm/tokenizers/deepseek_v32.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained, touching `__reduce__, DeepseekV32Tokenizer, from_pretrained`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32.py
@@ -3,7 +3,7 @@
-from transformers import AutoTokenizer
+from transformers import PreTrainedTokenizerFast
@@ -85,5 +85,5 @@ def __reduce__(self):
-        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
+        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/tokenizers/deepseek_v32.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36056 - [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1

- Link: https://github.com/vllm-project/vllm/pull/36056
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `be12afd284f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +622/-437, 1113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call, touching `make_parser, make_tool_param, make_request`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request, touching `__init__, type, _generate_tool_call_id`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -0,0 +1,476 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for DeepSeekV32ToolParser.
+These tests use a minimal mock tokenizer so no real model weights are required.
+"""
+import json
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):
-        # Sentinel tokens
-        self.dsml_token: str = "｜DSML｜"
-        self.dsml_start_check: str = "<" + self.dsml_token
+        # Sentinel token
-        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
-        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33703 - [Bugfix] Support multi-type params parsing for DeepSeek v3.2

- Link: https://github.com/vllm-project/vllm/pull/33703
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `a6db99ba02ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +201/-18, 250 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Support multi-type params parsing for DeepSeek v3.2"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Support multi-type params parsing for DeepSeek v3.2"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types, touching `test_no_emission_while_incomplete, deepseekv32_tokenizer, parser`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked, touching `_parse_invoke_params, _convert_param_value, _convert_param_value_checked`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -11,6 +11,7 @@
+from vllm.tokenizers import get_tokenizer
@@ -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):
+@pytest.fixture(scope="module")
+def deepseekv32_tokenizer():
+    return get_tokenizer(tokenizer_name="deepseek-ai/DeepSeek-V3.2")
+@pytest.fixture
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:
-    def _convert_param_value(self, value: str, param_type: str) -> Any:
+    def _convert_param_value_checked(self, value: str, param_type: str) -> Any:
@@ -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str) -> Any:
-            try:
-                return int(value)
-            except (ValueError, TypeError):
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- Link: https://github.com/vllm-project/vllm/pull/38684
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `5f96f9aff10f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-14, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] DSV3.2 Indexer Fused Weights Projection"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Perf] DSV3.2 Indexer Fused Weights Projection"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -639,21 +639,19 @@ def __init__(
-        self.wk = ReplicatedLinear(
+        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
+        # weights_proj does not get quantized, so we run both with quant_config=None
+        # wk may be upcasted from the default quant; experiments show fusion is always
+        # faster unless WK proj is in FP4, which is not the case for all known quants.
+        self.wk_weights_proj = MergedColumnParallelLinear(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Fused indexer wk + weights_proj
+            ("wk_weights_proj", "wk", 0),
+            ("wk_weights_proj", "weights_proj", 1),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- Link: https://github.com/vllm-project/vllm/pull/38870
- Status/date: merged / 2026-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `8617f8676b5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-27, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DSV32 weight loading"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Bugfix] Fix DSV32 weight loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights, touching `DeepSeekMTP, __init__, set_moe_parameters`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -625,6 +625,11 @@ def __init__(
+        self.quant_config = quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -639,18 +644,36 @@ def __init__(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
+        self.quant_config = vllm_config.quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- Link: https://github.com/vllm-project/vllm/pull/37421
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `b55d830ec782`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +2039/-483, 2698 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -67,7 +67,9 @@
-from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
+from vllm.model_executor.layers.sparse_attn_indexer import (
+    SparseAttnIndexer,
+)
@@ -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35968 - [Performance] DeepSeek V3.2 multi-stream indexer overlap

- Link: https://github.com/vllm-project/vllm/pull/35968
- Status/date: closed / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +187/-11, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] DeepSeek V3.2 multi-stream indexer overlap"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`; technical summary: Covers "[Performance] DeepSeek V3.2 multi-stream indexer overlap"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward, touching `__init__, _compute_k, forward`; `vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward, touching `__init__, _forward_static, forward`; `tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized, touching `_indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward
  - `vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward
  - `tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -79,7 +79,8 @@
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+from vllm.utils.torch_utils import aux_stream, direct_register_custom_op
@@ -625,6 +626,11 @@ def __init__(
+        self.events = (
+            [torch.cuda.Event(), torch.cuda.Event()]
diff -- vllm/model_executor/layers/layernorm.py
@@ -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):
+    @staticmethod
+    def _forward_static(
+        weight: torch.Tensor,
+        bias: torch.Tensor,
+        dim: int,
+        eps: float,
diff -- tests/utils_/test_indexer_dual_stream.py
@@ -0,0 +1,83 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8; `vllm/model_executor/layers/layernorm.py` modified +20/-3
  - tests: `tests/utils_/test_indexer_dual_stream.py` added +83/-0
- Risk and verification: The diff ships test coverage in `tests/utils_/test_indexer_dual_stream.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41198 - [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls

- Link: https://github.com/vllm-project/vllm/pull/41198
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `762022cafb1a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-1, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming, touching `test_multiple_tools, test_type_conversion_in_non_streaming`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls, touching `extract_tool_calls`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -188,6 +188,30 @@ def test_multiple_tools(self, parser):
+    def test_type_conversion_in_non_streaming(self):
+        """Non-streaming extraction must convert params using the tool schema."""
+        tool = ChatCompletionToolsParam(
+            function=FunctionDefinition(
+                name="toggle",
+                parameters={
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -191,12 +191,13 @@ def extract_tool_calls(
+                    params = self._convert_params_with_schema(invoke_name, param_dict)
-                                arguments=json.dumps(param_dict, ensure_ascii=False),
+                                arguments=json.dumps(params, ensure_ascii=False),
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41217 - [ROCm][Deepseek] dsv3.2 further optimization

- Link: https://github.com/vllm-project/vllm/pull/41217
- Status/date: merged / 2026-05-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `bc635fad2389`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +293/-73, 605 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Deepseek] dsv3.2 further optimization"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][Deepseek] dsv3.2 further optimization"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -674,30 +674,45 @@ def forward(
-        q_pe, q_nope = torch.split(
-            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
-        )
-        # Fused wk + weights_proj: one GEMM, then split
-        kw, _ = self.wk_weights_proj(hidden_states)
-        k = kw[:, : self.head_dim]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41801 - [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper

- Link: https://github.com/vllm-project/vllm/pull/41801
- Status/date: merged / 2026-05-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `95582868efd4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +224/-10, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired, touching `test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked, touching `__init__, _generate_tool_call_id, _parse_invoke_params`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2 (157 lines); hunks: -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):; -212,6 +219,118 @@ def test_type_conversion_in_non_streaming(self):; symbols: test_type_conversion_in_non_streaming, test_string_attr_true_preserves_literal_despite_schema, test_string_attr_false_allows_schema_conversion, test_arguments_wrapper_repaired
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8 (44 lines); hunks: -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, _generate_tool_call_id, _parse_invoke_params, _convert_param_value_checked
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -203,7 +203,14 @@ def test_type_conversion_in_non_streaming(self):
-        model_output = build_tool_call("toggle", {"enabled": "true", "count": "42"})
+        model_output = (
+            f"{FC_START}\n"
+            f'{INV_START}toggle">\n'
+            f'{PARAM_START}enabled" string="false">true{PARAM_END}\n'
+            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -69,7 +69,7 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
+            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
@@ -101,10 +101,12 @@ def _generate_tool_call_id(self) -> str:
-    def _parse_invoke_params(self, invoke_str: str) -> dict:
-        param_dict = dict()
-        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +155/-2
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +36/-8
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41835 - [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA

- Link: https://github.com/vllm-project/vllm/pull/41835
- Status/date: merged / 2026-05-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `c936548ce6b0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-10, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -299,6 +299,15 @@ def __init__(
+        if (
+            self.is_rocm_aiter_moe_enabled
+            and self.gate.e_score_correction_bias is not None
+        ):
+            # AITER biased_grouped_topk requires the correction bias dtype to
+            # match the router logits. Keep DeepSeek's correction bias in fp32
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42062 - [ROCm] Enable gluon paged MQA logits on gfx950 (MI355X)

- Link: https://github.com/vllm-project/vllm/pull/42062
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Enable gluon paged MQA logits on gfx950 (MI355X)"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; technical summary: Covers "[ROCm] Enable gluon paged MQA logits on gfx950 (MI355X)"; the main implementation surface is `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +3/-2 (5 lines); hunks: -16,9 +16,10; -385,7 +386,7 @@ def rocm_fp8_paged_mqa_logits(; symbols: rocm_fp8_paged_mqa_logits, touching `rocm_fp8_paged_mqa_logits`.
- Code diff details:
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +3/-2 (5 lines); hunks: -16,9 +16,10; -385,7 +386,7 @@ def rocm_fp8_paged_mqa_logits(; symbols: rocm_fp8_paged_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -16,9 +16,10 @@
-    from vllm.platforms.rocm import _ON_GFX942
+    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
+    _ON_GFX950 = False
@@ -385,7 +386,7 @@ def rocm_fp8_paged_mqa_logits(
-        if _ON_GFX942:
+        if _ON_GFX942 or _ON_GFX950:
```

- Reviewed files:
  - runtime: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43019 - [Bugfix] Use shared coerce_to_schema_type in DeepSeekV32 tool parser

- Link: https://github.com/vllm-project/vllm/pull/43019
- Status/date: merged / 2026-05-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `a10d69116cb2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +270/-285, 615 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Use shared coerce_to_schema_type in DeepSeekV32 tool parser"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; technical summary: Covers "[Bugfix] Use shared coerce_to_schema_type in DeepSeekV32 tool parser"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +261/-233 (494 lines); hunks: -16,7 +16,6; -65,58 +64,6 @@ def build_tool_call(func_name: str, params: dict[str, str]) -...; symbols: build_tool_call, TestConvertParamValue, parser, test_null, touching `build_tool_call, TestConvertParamValue, parser`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +9/-52 (61 lines); hunks: -26,7 +26,12; -109,41 +114,6 @@ def _parse_invoke_params(self, invoke_str: str) -> dict[str...; symbols: _parse_invoke_params, _convert_param_value_checked, _convert_param_value, _repair_param_dict, touching `_parse_invoke_params, _convert_param_value_checked, _convert_param_value`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +261/-233 (494 lines); hunks: -16,7 +16,6; -65,58 +64,6 @@ def build_tool_call(func_name: str, params: dict[str, str]) -...; symbols: build_tool_call, TestConvertParamValue, parser, test_null
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +9/-52 (61 lines); hunks: -26,7 +26,12; -109,41 +114,6 @@ def _parse_invoke_params(self, invoke_str: str) -> dict[str...; symbols: _parse_invoke_params, _convert_param_value_checked, _convert_param_value, _repair_param_dict
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -16,7 +16,6 @@
-from vllm.tokenizers import get_tokenizer
@@ -65,58 +64,6 @@ def build_tool_call(func_name: str, params: dict[str, str]) -> str:
-# ---------------------------------------------------------------------------
-# Tests: DeepSeekV32ToolParser._convert_param_value
-# ---------------------------------------------------------------------------
-class TestConvertParamValue:
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -26,7 +26,12 @@
-from vllm.tool_parsers.utils import partial_tag_overlap
+from vllm.tool_parsers.utils import (
+    coerce_to_schema_type,
+    extract_types_from_schema,
+    find_tool_properties,
+    partial_tag_overlap,
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +261/-233
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +9/-52
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43255 - [CI] Add composed-schema regression tests for DeepSeek V3.2/V4 parsers

- Link: https://github.com/vllm-project/vllm/pull/43255
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`; associated commits `63ea11709bd9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +186/-0, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add composed-schema regression tests for DeepSeek V3.2/V4 parsers"; model line: DeepSeek V3.2; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`; technical summary: Covers "[CI] Add composed-schema regression tests for DeepSeek V3.2/V4 parsers"; the main implementation surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +137/-0 (137 lines); hunks: -221,6 +221,99 @@ def test_string_attr_false_allows_schema_conversion(self):; -581,6 +674,50 @@ def test_string_attr_true_preserves_literal_in_streaming(se...; symbols: test_string_attr_false_allows_schema_conversion, test_composed_schema_converts_object_and_array_params, test_string_attr_true_preserves_literal_for_composed_schema, test_arguments_wrapper_repaired, touching `test_string_attr_false_allows_schema_conversion, test_composed_schema_converts_object_and_array_params, test_string_attr_true_preserves_literal_for_composed_schema`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +137/-0 (137 lines); hunks: -221,6 +221,99 @@ def test_string_attr_false_allows_schema_conversion(self):; -581,6 +674,50 @@ def test_string_attr_true_preserves_literal_in_streaming(se...; symbols: test_string_attr_false_allows_schema_conversion, test_composed_schema_converts_object_and_array_params, test_string_attr_true_preserves_literal_for_composed_schema, test_arguments_wrapper_repaired
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -221,6 +221,99 @@ def test_string_attr_false_allows_schema_conversion(self):
+    @pytest.mark.skip_global_cleanup
+    def test_composed_schema_converts_object_and_array_params(self):
+        """Composed JSON Schema types must still drive DSML type coercion."""
+        tool = ChatCompletionToolsParam(
+            function=FunctionDefinition(
+                name="set_timer",
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +137/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42879 - [Bugfix] Stream DeepSeek DSML tool-call argument deltas incrementally

- Link: https://github.com/vllm-project/vllm/pull/42879
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +445/-63, 622 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Stream DeepSeek DSML tool-call argument deltas incrementally"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; technical summary: Covers "[Bugfix] Stream DeepSeek DSML tool-call argument deltas incrementally"; the main implementation surface is `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +313/-59 (372 lines); hunks: -4,7 +4,7; -62,6 +62,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool...; symbols: __init__, _parse_invoke_params, _repair_param_dict, _convert_params_with_schema, touching `__init__, _parse_invoke_params, _repair_param_dict`; `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +85/-4 (89 lines); hunks: -10,6 +10,7; -718,6 +719,81 @@ def test_composed_schema_conversion_in_streaming(self):; symbols: test_composed_schema_conversion_in_streaming, test_responses_function_tool_schema_in_streaming, test_streaming_matches_non_streaming_conversion_fallbacks, test_multiple_tools_streaming, touching `test_composed_schema_conversion_in_streaming, test_responses_function_tool_schema_in_streaming, test_streaming_matches_non_streaming_conversion_fallbacks`; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +47/-0 (47 lines); hunks: -14,6 +14,7; -164,11 +165,57 @@ def test_streaming_extracts_complete_invokes():; symbols: test_streaming_extracts_complete_invokes, test_streaming_emits_incremental_argument_chunks, test_get_vllm_registry_structural_tag_returns_structural_tag, touching `test_streaming_extracts_complete_invokes, test_streaming_emits_incremental_argument_chunks, test_get_vllm_registry_structural_tag_returns_structural_tag`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +313/-59 (372 lines); hunks: -4,7 +4,7; -62,6 +62,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool...; symbols: __init__, _parse_invoke_params, _repair_param_dict, _convert_params_with_schema
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +85/-4 (89 lines); hunks: -10,6 +10,7; -718,6 +719,81 @@ def test_composed_schema_conversion_in_streaming(self):; symbols: test_composed_schema_conversion_in_streaming, test_responses_function_tool_schema_in_streaming, test_streaming_matches_non_streaming_conversion_fallbacks, test_multiple_tools_streaming
  - `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +47/-0 (47 lines); hunks: -14,6 +14,7; -164,11 +165,57 @@ def test_streaming_extracts_complete_invokes():; symbols: test_streaming_extracts_complete_invokes, test_streaming_emits_incremental_argument_chunks, test_get_vllm_registry_structural_tag_returns_structural_tag
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -4,7 +4,7 @@
-from typing import Any
+from typing import Any, Literal
@@ -62,6 +62,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
+        self._buffer: str = ""
+        self._in_tool_calls: bool = False
+        self._active_tool_index: int | None = None
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -10,6 +10,7 @@
+from openai.types.responses.function_tool import FunctionTool
@@ -718,6 +719,81 @@ def test_composed_schema_conversion_in_streaming(self):
+    def test_responses_function_tool_schema_in_streaming(self):
+        """Responses API FunctionTool schemas must drive streaming conversion."""
+        tool = FunctionTool(
+            type="function",
diff -- tests/tool_parsers/test_deepseekv4_tool_parser.py
@@ -14,6 +14,7 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +313/-59
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +85/-4; `tests/tool_parsers/test_deepseekv4_tool_parser.py` modified +47/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43781 - [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950

- Link: https://github.com/vllm-project/vllm/pull/43781
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-4, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; technical summary: Covers "[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits, touching `indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -612,6 +612,7 @@ def __init__(
+        is_inplace_rope: bool = False,
@@ -673,15 +674,21 @@ def __init__(
+        self.is_inplace_rope = is_inplace_rope
-        if current_platform.is_rocm():
+        if current_platform.is_rocm() and self.is_inplace_rope:
+            # This fast path relies on rotary_emb mutating q and k inplace.
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(
+    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
@@ -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(
-        "SHUFFLE",
+        layout,
@@ -229,6 +230,7 @@ def cp_gather_indexer_k_quant_cache_triton(
+    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42982 - [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)

- Link: https://github.com/vllm-project/vllm/pull/42982
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `0b56815a24f4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +59/-29, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -304,10 +304,8 @@ def __init__(
-            # AITER biased_grouped_topk requires the correction bias dtype to
-            # match the router logits. Keep DeepSeek's correction bias in fp32
-            # by requesting fp32 router logits for this routing path.
-            self.gate.set_out_dtype(torch.float32)
+            # Accumulates in fp32; avoids bf16->fp32 cast.
+            self.gate.set_out_dtype(self.gate.weight.dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42944 - fix: glm5.1 pp model loading

- Link: https://github.com/vllm-project/vllm/pull/42944
- Status/date: merged / 2026-06-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-5, 93 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: glm5.1 pp model loading"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "fix: glm5.1 pp model loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk, touching `forward, _try_load_fp8_indexer_wk`; `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk
  - `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -105,6 +105,7 @@
+    get_pp_missing_layer_names,
@@ -742,7 +743,9 @@ def forward(
-def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
+def _try_load_fp8_indexer_wk(
+    name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
+):
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -35,7 +35,7 @@
-from .utils import maybe_prefix
+from .utils import get_pp_missing_layer_names, maybe_prefix
@@ -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        pp_missing_layer_names = get_pp_missing_layer_names(self)
@@ -282,7 +283,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                name, loaded_weight, _pending_wk_fp8, params_dict, loaded_params
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3; `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44420 - [feature] add index share feature for DSA MTP

- Link: https://github.com/vllm-project/vllm/pull/44420
- Status/date: merged / 2026-06-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +114/-25, 230 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] add index share feature for DSA MTP"; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`; technical summary: Covers "[feature] add index share feature for DSA MTP"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids, touching `forward, __init__, set_skip_topk`; `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head, touching `__init__, propose, _maybe_share_lm_head`; `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads, touching `get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids
  - `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head
  - `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads
  - `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4 (13 lines); hunks: -76,10 +76,15 @@ def load_eagle_model(target_model: nn.Module, vllm_config: V...; symbols: load_eagle_model
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1018,19 +1018,20 @@ def __init__(
-            # Enable IndexCache for DeepSeek models to reduce redundant top-k
-            # token selection computations in sparse attention.
-            use_index_cache = getattr(config, "use_index_cache", False)
-            if use_index_cache:
-                # IndexCache config
-                # Refer: https://arxiv.org/abs/2603.12201 for more details.
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -115,7 +115,9 @@ def forward(
-            positions=positions, hidden_states=hidden_states, residual=None
+            positions=positions,
+            hidden_states=hidden_states,
+            residual=None,
@@ -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+    def set_skip_topk(self, skip: bool):
diff -- vllm/v1/spec_decode/llm_base_proposer.py
@@ -70,6 +70,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15; `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2; `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3; `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1; `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/transformers_utils/model_arch_config_convertor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45003 - [Frontend] Support strict mode for tool calling

- Link: https://github.com/vllm-project/vllm/pull/45003
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +672/-1936, 3162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support strict mode for tool calling"; model line: DeepSeek V3.2; category: docs/tests/CI; main diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`; technical summary: Covers "[Frontend] Support strict mode for tool calling"; the main implementation surface is `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks, touching `StreamingXMLToolCallParser, __init__, reset_streaming_state`; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag, touching `register_model_structural_tag, register_vllm_structural_tag, decorator`; `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes, touching `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls, touching `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`.
- Code diff details:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -1,1300 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import json
-from collections.abc import Sequence
-from typing import Any
-from xml.parsers.expat import ParserCreate
diff -- vllm/tool_parsers/structural_tag_registry.py
@@ -1,14 +1,15 @@
-# Model-specific structural tag builders adapted from XGrammar's
-# builtin structural tag implementations:
-# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py
-from xgrammar import StructuralTag
+from xgrammar import StructuralTag, normalize_tool_choice
+from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
diff -- tests/tool_parsers/test_structural_tag_registry.py
@@ -0,0 +1,314 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45895 - [bugfix]Indexer init skip and MTP TopK share for iteration

- Link: https://github.com/vllm-project/vllm/pull/45895
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +69/-30, 198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix]Indexer init skip and MTP TopK share for iteration"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`; technical summary: Covers "[bugfix]Indexer init skip and MTP TopK share for iteration"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor, touching `forward, DeepSeekMultiTokenPredictor`; `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3 (10 lines); hunks: -271,7 +271,7 @@ def __init__(; -301,8 +301,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -998,8 +998,29 @@ def __init__(
+        # IndexCache config
+        # Refer: https://arxiv.org/abs/2603.12201 for more details.
-        if self.is_v32:
+        _index_topk_freq = getattr(config, "index_topk_freq", 1)
+        _index_topk_pattern = getattr(config, "index_topk_pattern", None)
+        _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -119,8 +119,12 @@ def forward(
-        hidden_states = residual + hidden_states
-        return hidden_states
+        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
+        # Recycle the post-final-norm hidden into the next draft step.
+        # compute_logits applies shared_head (== final norm) to the pre-norm
+        # element, so logits and the recycle each get exactly one final-norm.
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -349,6 +349,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17; `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0; `vllm/model_executor/layers/mla.py` modified +1/-0; `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46199 - [Bugfix] Move extract_layer_index back inside is_v32 guard

- Link: https://github.com/vllm-project/vllm/pull/46199
- Status/date: merged / 2026-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `77148992cfc9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-17, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Move extract_layer_index back inside is_v32 guard"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Move extract_layer_index back inside is_v32 guard"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1001,24 +1001,30 @@ def __init__(
-        _index_topk_freq = getattr(config, "index_topk_freq", 1)
-        _index_topk_pattern = getattr(config, "index_topk_pattern", None)
-        _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
-        layer_id = extract_layer_index(prefix)
-        if _index_topk_pattern is None:
-            _skip_topk = (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46651 - [Perf] Remove redundant clone for GLM, Deepseek etc

- Link: https://github.com/vllm-project/vllm/pull/46651
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +4/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Remove redundant clone for GLM, Deepseek etc"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[Perf] Remove redundant clone for GLM, Deepseek etc"; the main implementation surface is `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/AXK1.py
@@ -649,7 +649,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1186,7 +1186,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -184,7 +184,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/openpangu.py
@@ -935,7 +935,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/AXK1.py` modified +1/-1; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1; `vllm/model_executor/models/openpangu.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46808 - [GLM-5] Add DSV3.2/GLM5 to `vllm/models/`

- Link: https://github.com/vllm-project/vllm/pull/46808
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/__init__.py`, `vllm/models/deepseek_v32/nvidia/__init__.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; associated commits `65e655d29591`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +1170/-0, 1175 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM-5] Add DSV3.2/GLM5 to `vllm/models/`"; model line: DeepSeek V3.2; category: model support/runtime entry; main diff: `vllm/models/deepseek_v32/nvidia/mtp.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/__init__.py`; technical summary: Covers "[GLM-5] Add DSV3.2/GLM5 to `vllm/models/`"; the main implementation surface is `vllm/models/deepseek_v32/nvidia/mtp.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/nvidia/mtp.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV32MultiTokenPredictorLayer, __init__, forward, DeepseekV32MultiTokenPredictor, touching `DeepseekV32MultiTokenPredictorLayer, __init__, forward`; `vllm/models/deepseek_v32/nvidia/model.py` added +333/-0 (333 lines); hunks: -0,0 +1,333; symbols: DeepseekV32DecoderLayer, __init__, forward, DeepseekV32Model, touching `DeepseekV32DecoderLayer, __init__, forward`; `vllm/models/deepseek_v32/__init__.py` added +22/-0 (22 lines); hunks: -0,0 +1,22; `vllm/models/deepseek_v32/nvidia/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/deepseek_v32/nvidia/mtp.py` added +390/-0 (390 lines); hunks: -0,0 +1,390; symbols: DeepseekV32MultiTokenPredictorLayer, __init__, forward, DeepseekV32MultiTokenPredictor
  - `vllm/models/deepseek_v32/nvidia/model.py` added +333/-0 (333 lines); hunks: -0,0 +1,333; symbols: DeepseekV32DecoderLayer, __init__, forward, DeepseekV32Model
  - `vllm/models/deepseek_v32/__init__.py` added +22/-0 (22 lines); hunks: -0,0 +1,22
  - `vllm/models/deepseek_v32/nvidia/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/nvidia/mtp.py
@@ -0,0 +1,390 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+import torch
+import torch.nn as nn
diff -- vllm/models/deepseek_v32/nvidia/model.py
@@ -0,0 +1,333 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import torch
diff -- vllm/models/deepseek_v32/__init__.py
@@ -0,0 +1,22 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/nvidia/mtp.py` added +390/-0; `vllm/models/deepseek_v32/nvidia/model.py` added +333/-0; `vllm/models/deepseek_v32/__init__.py` added +22/-0; `vllm/models/deepseek_v32/nvidia/__init__.py` added +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v32/__init__.py`, `vllm/models/deepseek_v32/nvidia/__init__.py`, `vllm/models/deepseek_v32/nvidia/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46600 - [Bugfix][DSv3.2] Skip indexer weights for index-cache-skipped layers

- Link: https://github.com/vllm-project/vllm/pull/46600
- Status/date: merged / 2026-06-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `6eb63a1da699`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-0, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][DSv3.2] Skip indexer weights for index-cache-skipped layers"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix][DSv3.2] Skip indexer weights for index-cache-skipped layers"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -1438,6 +1438,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; -1446,6 +1451,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -1438,6 +1438,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; -1446,6 +1451,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1438,6 +1438,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        # With index_topk_freq>1 only some layers build an indexer, yet the
+        # checkpoint ships indexer weights for all of them; track the built ones.
+        indexer_present_prefixes = {
+            n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
+        }
@@ -1446,6 +1451,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46876 - [GLM5] Implement op fusion for GLM5/DSV3.2

- Link: https://github.com/vllm-project/vllm/pull/46876
- Status/date: merged / 2026-06-28
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; associated commits `89876b0c548a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1530/-91, 1825 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM5] Implement op fusion for GLM5/DSV3.2"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; technical summary: Covers "[GLM5] Implement op fusion for GLM5/DSV3.2"; the main implementation surface is `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/nvidia/model.py` modified +21/-3 (24 lines); hunks: -36,6 +36,7; -77,13 +78,18 @@ def __init__(; symbols: DeepseekV32DecoderLayer, __init__, forward, touching `DeepseekV32DecoderLayer, __init__, forward`; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +15/-6 (21 lines); hunks: -1,6 +1,5; -9,6 +8,7; symbols: forward, touching `forward`; `tests/kernels/test_fused_deepseek_v32_norm_rope.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: make_cos_sin, rms_norm, layer_norm, rope, touching `make_cos_sin, rms_norm, layer_norm`.
- Code diff details:
  - `vllm/models/deepseek_v32/nvidia/model.py` modified +21/-3 (24 lines); hunks: -36,6 +36,7; -77,13 +78,18 @@ def __init__(; symbols: DeepseekV32DecoderLayer, __init__, forward
  - `vllm/models/deepseek_v32/nvidia/mtp.py` modified +15/-6 (21 lines); hunks: -1,6 +1,5; -9,6 +8,7; symbols: forward
  - `tests/kernels/test_fused_deepseek_v32_norm_rope.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: make_cos_sin, rms_norm, layer_norm, rope
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/nvidia/model.py
@@ -36,6 +36,7 @@
+from .fused_ops import fused_allreduce_rms_norm
@@ -77,13 +78,18 @@ def __init__(
+            # Defer the MoE cross-rank all-reduce; it is fused into the next
+            # layer's input_layernorm (or the final norm) via
+            # fused_allreduce_rms_norm. self.mlp.experts is the MoERunner.
+            self.mlp.experts.moe_config.skip_final_all_reduce = True
diff -- vllm/models/deepseek_v32/nvidia/mtp.py
@@ -1,6 +1,5 @@
@@ -9,6 +8,7 @@
+from vllm.distributed import tensor_model_parallel_all_reduce
@@ -35,6 +35,7 @@
+from .kernels import fused_eh_norm
@@ -75,15 +76,23 @@ def forward(
-        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
diff -- tests/kernels/test_fused_deepseek_v32_norm_rope.py
@@ -0,0 +1,423 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/nvidia/model.py` modified +21/-3; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +15/-6
  - tests: `tests/kernels/test_fused_deepseek_v32_norm_rope.py` added +423/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48036 - [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel

- Link: https://github.com/vllm-project/vllm/pull/48036
- Status/date: merged / 2026-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; associated commits `1ff9429655f0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +34/-3, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel"; model line: DeepSeek V3.2; category: bug fix; main diff: `vllm/model_executor/models/deepseek_mtp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; technical summary: Covers "[CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel"; the main implementation surface is `vllm/model_executor/models/deepseek_mtp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_mtp.py` modified +24/-0 (24 lines); hunks: -10,6 +10,7; -40,6 +41,24; symbols: _restore_full_token_layout_if_needed, SharedHead, __init__, forward, touching `_restore_full_token_layout_if_needed, SharedHead, __init__`; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -89,7 +92,12 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +24/-0 (24 lines); hunks: -10,6 +10,7; -40,6 +41,24; symbols: _restore_full_token_layout_if_needed, SharedHead, __init__, forward
  - `vllm/models/deepseek_v32/nvidia/mtp.py` modified +10/-2 (12 lines); hunks: -21,7 +21,10; -89,7 +92,12 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -10,6 +10,7 @@
+from vllm.distributed import tensor_model_parallel_all_gather
@@ -40,6 +41,24 @@
+def _restore_full_token_layout_if_needed(
+    hidden_states: torch.Tensor,
+    residual: torch.Tensor,
+    num_tokens: int,
diff -- vllm/models/deepseek_v32/nvidia/mtp.py
@@ -21,7 +21,10 @@
-from vllm.model_executor.models.deepseek_mtp import SharedHead
+from vllm.model_executor.models.deepseek_mtp import (
+    SharedHead,
+    _restore_full_token_layout_if_needed,
+)
@@ -89,7 +92,12 @@ def forward(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +24/-0; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/parallel.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #47207 - [ROCm]Migrating Deepseek V3.2 to vllm/models/deepseek_v32/

- Link: https://github.com/vllm-project/vllm/pull/47207
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `vllm/models/deepseek_v32/__init__.py`, `vllm/models/deepseek_v32/amd/__init__.py`, `vllm/models/deepseek_v32/amd/model.py`, `vllm/models/deepseek_v32/amd/mtp.py` and 11 files; associated commits `e2efe79695e0`
- Diff scope read: GitHub Pull Request files API returned 12 files, +1031/-70, 1274 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm]Migrating Deepseek V3.2 to vllm/models/deepseek_v32/"; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/models/deepseek_v32/amd/mtp.py`, `vllm/models/deepseek_v32/amd/model.py`, `vllm/models/deepseek_v32/amd/rocm.py`; technical summary: Covers "[ROCm]Migrating Deepseek V3.2 to vllm/models/deepseek_v32/"; the main implementation surface is `vllm/models/deepseek_v32/amd/mtp.py`, `vllm/models/deepseek_v32/amd/model.py`, `vllm/models/deepseek_v32/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/amd/mtp.py` added +392/-0 (392 lines); hunks: -0,0 +1,392; symbols: DeepseekV32MultiTokenPredictorLayer, __init__, forward, DeepseekV32MultiTokenPredictor, touching `DeepseekV32MultiTokenPredictorLayer, __init__, forward`; `vllm/models/deepseek_v32/amd/model.py` added +327/-0 (327 lines); hunks: -0,0 +1,327; symbols: DeepseekV32DecoderLayer, __init__, forward, DeepseekV32Model, touching `DeepseekV32DecoderLayer, __init__, forward`; `vllm/models/deepseek_v32/amd/rocm.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: DeepseekV32MLASparseBackend, get_supported_kernel_block_sizes, DeepseekV32ROCmIndexerBackend, DeepseekV32ROCmIndexerCache, touching `DeepseekV32MLASparseBackend, get_supported_kernel_block_sizes, DeepseekV32ROCmIndexerBackend`; `vllm/models/deepseek_v32/attention.py` renamed +32/-60 (92 lines); hunks: -31,12 +31,13; -55,16 +56,14 @@ def __init__(; symbols: DeepseekV32Indexer, __init__, forward, touching `DeepseekV32Indexer, __init__, forward`.
- Code diff details:
  - `vllm/models/deepseek_v32/amd/mtp.py` added +392/-0 (392 lines); hunks: -0,0 +1,392; symbols: DeepseekV32MultiTokenPredictorLayer, __init__, forward, DeepseekV32MultiTokenPredictor
  - `vllm/models/deepseek_v32/amd/model.py` added +327/-0 (327 lines); hunks: -0,0 +1,327; symbols: DeepseekV32DecoderLayer, __init__, forward, DeepseekV32Model
  - `vllm/models/deepseek_v32/amd/rocm.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: DeepseekV32MLASparseBackend, get_supported_kernel_block_sizes, DeepseekV32ROCmIndexerBackend, DeepseekV32ROCmIndexerCache
  - `vllm/models/deepseek_v32/attention.py` renamed +32/-60 (92 lines); hunks: -31,12 +31,13; -55,16 +56,14 @@ def __init__(; symbols: DeepseekV32Indexer, __init__, forward
  - `vllm/models/deepseek_v32/__init__.py` modified +9/-5 (14 lines); hunks: -10,11 +10,15
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/amd/mtp.py
@@ -0,0 +1,392 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+import torch
+import torch.nn as nn
diff -- vllm/models/deepseek_v32/amd/model.py
@@ -0,0 +1,327 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import torch
diff -- vllm/models/deepseek_v32/amd/rocm.py
@@ -0,0 +1,263 @@
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/amd/mtp.py` added +392/-0; `vllm/models/deepseek_v32/amd/model.py` added +327/-0; `vllm/models/deepseek_v32/amd/rocm.py` added +263/-0; `vllm/models/deepseek_v32/attention.py` renamed +32/-60; `vllm/models/deepseek_v32/__init__.py` modified +9/-5; `vllm/models/deepseek_v32/nvidia/model.py` modified +2/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50904 - [GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement

- Link: https://github.com/vllm-project/vllm/pull/50904
- Status/date: merged / 2026-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`; associated commits `ad5280255ba6`
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-6, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`; technical summary: Covers "[GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement"; the main implementation surface is `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/amd/rocm.py` modified +4/-3 (7 lines); hunks: -167,7 +167,7 @@ def _fused_attention(; -222,7 +222,7 @@ def _fused_attention(; symbols: _fused_attention, touching `_fused_attention`; `vllm/models/deepseek_v32/attention.py` modified +3/-3 (6 lines); hunks: -372,7 +372,7 @@ def _fused_attention(; -430,7 +430,7 @@ def _fused_attention(; symbols: _fused_attention, touching `_fused_attention`.
- Code diff details:
  - `vllm/models/deepseek_v32/amd/rocm.py` modified +4/-3 (7 lines); hunks: -167,7 +167,7 @@ def _fused_attention(; -222,7 +222,7 @@ def _fused_attention(; symbols: _fused_attention
  - `vllm/models/deepseek_v32/attention.py` modified +3/-3 (6 lines); hunks: -372,7 +372,7 @@ def _fused_attention(; -430,7 +430,7 @@ def _fused_attention(; symbols: _fused_attention
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/amd/rocm.py
@@ -167,7 +167,7 @@ def _fused_attention(
-        if self.indexer is not None:
+        if self.indexer is not None and not self.skip_topk:
@@ -222,7 +222,7 @@ def _fused_attention(
-        if self.indexer is not None:
+        if self.indexer is not None and not self.skip_topk:
@@ -244,7 +244,8 @@ def _fused_attention(
diff -- vllm/models/deepseek_v32/attention.py
@@ -372,7 +372,7 @@ def _fused_attention(
-        if self.indexer is not None:
+        if self.indexer is not None and not self.skip_topk:
@@ -430,7 +430,7 @@ def _fused_attention(
-        if self.indexer is not None:
+        if self.indexer is not None and not self.skip_topk:
@@ -452,7 +452,7 @@ def _fused_attention(
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/amd/rocm.py` modified +4/-3; `vllm/models/deepseek_v32/attention.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51298 - [DSv32/GLM Perf] Skip short prefill topk for dense mha layer, 97.9% kernel level latency reduction

- Link: https://github.com/vllm-project/vllm/pull/51298
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/attention.py`; associated commits `4a11ed5bc4c2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-1, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32/GLM Perf] Skip short prefill topk for dense mha layer, 97.9% kernel level latency reduction"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/attention.py`; technical summary: Covers "[DSv32/GLM Perf] Skip short prefill topk for dense mha layer, 97.9% kernel level latency reduction"; the main implementation surface is `vllm/models/deepseek_v32/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/attention.py` modified +11/-1 (12 lines); hunks: -32,6 +32,7; -259,6 +260,15 @@ def __init__(; symbols: __init__, _fused_attention, touching `__init__, _fused_attention`.
- Code diff details:
  - `vllm/models/deepseek_v32/attention.py` modified +11/-1 (12 lines); hunks: -32,6 +32,7; -259,6 +260,15 @@ def __init__(; symbols: __init__, _fused_attention
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/attention.py
@@ -32,6 +32,7 @@
+from vllm.platforms import current_platform
@@ -259,6 +260,15 @@ def __init__(
+        enable_short_prefill_scoring_skip = (
+            not is_mtp_layer
+            and not skip_topk
+            and not self.use_pcp
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/attention.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v32/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51425 - [Perf] Narrow DeepSeek V3.2 eager CUDA graph region

- Link: https://github.com/vllm-project/vllm/pull/51425
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`; associated commits `46b586405412`
- Diff scope read: GitHub Pull Request files API returned 2 files, +61/-16, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Narrow DeepSeek V3.2 eager CUDA graph region"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/attention.py`, `vllm/models/deepseek_v32/amd/rocm.py`; technical summary: Covers "[Perf] Narrow DeepSeek V3.2 eager CUDA graph region"; the main implementation surface is `vllm/models/deepseek_v32/attention.py`, `vllm/models/deepseek_v32/amd/rocm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/attention.py` modified +32/-16 (48 lines); hunks: -343,22 +343,6 @@ def forward( # type: ignore[override]; -452,6 +436,26 @@ def _fused_attention(; symbols: forward, _fused_attention, _sparse_indexer_and_attn, touching `forward, _fused_attention, _sparse_indexer_and_attn`; `vllm/models/deepseek_v32/amd/rocm.py` modified +29/-0 (29 lines); hunks: -66,6 +66,35 @@ def __init__(self, vllm_config, config, prefix, topk_indices_...; symbols: __init__, forward, _compute_ql_nope, touching `__init__, forward, _compute_ql_nope`.
- Code diff details:
  - `vllm/models/deepseek_v32/attention.py` modified +32/-16 (48 lines); hunks: -343,22 +343,6 @@ def forward( # type: ignore[override]; -452,6 +436,26 @@ def _fused_attention(; symbols: forward, _fused_attention, _sparse_indexer_and_attn
  - `vllm/models/deepseek_v32/amd/rocm.py` modified +29/-0 (29 lines); hunks: -66,6 +66,35 @@ def __init__(self, vllm_config, config, prefix, topk_indices_...; symbols: __init__, forward, _compute_ql_nope
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/attention.py
@@ -343,22 +343,6 @@ def forward(  # type: ignore[override]
-        self._fused_attention(
-            positions, q_c, kv_c, k_pe, index_k, index_weights, output
-        )
-        return self.o_proj(output)[0]
-    @eager_break_during_capture
-    def _fused_attention(
diff -- vllm/models/deepseek_v32/amd/rocm.py
@@ -66,6 +66,35 @@ def __init__(self, vllm_config, config, prefix, topk_indices_buffer=None):
+    def forward(  # type: ignore[override]
+        self,
+        positions: torch.Tensor,
+        hidden_states: torch.Tensor,
+    ) -> torch.Tensor:
+        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/attention.py` modified +32/-16; `vllm/models/deepseek_v32/amd/rocm.py` modified +29/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v32/amd/rocm.py`, `vllm/models/deepseek_v32/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51434 - [Perf] Optimize DeepSeek V3.2 sequence parallelism

- Link: https://github.com/vllm-project/vllm/pull/51434
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/deepseek_v32/test_sequence_parallel.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; associated commits `e644c8cd8c73`
- Diff scope read: GitHub Pull Request files API returned 3 files, +222/-77, 444 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Optimize DeepSeek V3.2 sequence parallelism"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `tests/models/deepseek_v32/test_sequence_parallel.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`; technical summary: Covers "[Perf] Optimize DeepSeek V3.2 sequence parallelism"; the main implementation surface is `tests/models/deepseek_v32/test_sequence_parallel.py`, `vllm/models/deepseek_v32/nvidia/model.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/deepseek_v32/test_sequence_parallel.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _IdentityNorm, __init__, forward, _RecordingModule, touching `_IdentityNorm, __init__, forward`; `vllm/models/deepseek_v32/nvidia/model.py` modified +57/-70 (127 lines); hunks: -6,12 +6,10; -36,27 +34,20; symbols: _all_gather_sp_states, DeepseekV32DecoderLayer, __init__, touching `_all_gather_sp_states, DeepseekV32DecoderLayer, __init__`; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +17/-7 (24 lines); hunks: -6,12 +6,11; -38,6 +37,11; symbols: forward, touching `forward`.
- Code diff details:
  - `tests/models/deepseek_v32/test_sequence_parallel.py` added +148/-0 (148 lines); hunks: -0,0 +1,148; symbols: _IdentityNorm, __init__, forward, _RecordingModule
  - `vllm/models/deepseek_v32/nvidia/model.py` modified +57/-70 (127 lines); hunks: -6,12 +6,10; -36,27 +34,20; symbols: _all_gather_sp_states, DeepseekV32DecoderLayer, __init__
  - `vllm/models/deepseek_v32/nvidia/mtp.py` modified +17/-7 (24 lines); hunks: -6,12 +6,11; -38,6 +37,11; symbols: forward
- Key code excerpts:

```diff
diff -- tests/models/deepseek_v32/test_sequence_parallel.py
@@ -0,0 +1,148 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from types import SimpleNamespace
+from unittest.mock import Mock
+import torch
+from torch import nn
diff -- vllm/models/deepseek_v32/nvidia/model.py
@@ -6,12 +6,10 @@
+import vllm.envs as envs
-from vllm.distributed import (
-    get_pp_group,
-    tensor_model_parallel_all_gather,
-    tensor_model_parallel_reduce_scatter,
-)
diff -- vllm/models/deepseek_v32/nvidia/mtp.py
@@ -6,12 +6,11 @@
```

- Reviewed files:
  - tests: `tests/models/deepseek_v32/test_sequence_parallel.py` added +148/-0
  - runtime: `vllm/models/deepseek_v32/nvidia/model.py` modified +57/-70; `vllm/models/deepseek_v32/nvidia/mtp.py` modified +17/-7
- Risk and verification: The diff ships test coverage in `tests/models/deepseek_v32/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52381 - Harden DeepSeek V3.2 fused kernel grids

- Link: https://github.com/vllm-project/vllm/pull/52381
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `vllm/models/deepseek_v32/common/kernels.py`; associated commits `eab1cff5b0ca`
- Diff scope read: GitHub Pull Request files API returned 2 files, +79/-6, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Harden DeepSeek V3.2 fused kernel grids"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/common/kernels.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; technical summary: Covers "Harden DeepSeek V3.2 fused kernel grids"; the main implementation surface is `vllm/models/deepseek_v32/common/kernels.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/common/kernels.py` modified +6/-6 (12 lines); hunks: -152,8 +152,8 @@ def _fused_norm_rope_kernel(; -479,7 +479,7 @@ def fused_norm_rope(; symbols: _fused_norm_rope_kernel, fused_norm_rope, _fused_q_kernel, fused_q, touching `_fused_norm_rope_kernel, fused_norm_rope, _fused_q_kernel`; `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +73/-0 (73 lines); hunks: -362,6 +362,46 @@ def test_fused_norm_rope_ds_mla(num_tokens: int):; -539,6 +579,39 @@ def test_fused_q_bf16_query(num_tokens: int, has_indexer: b...; symbols: test_fused_norm_rope_ds_mla, test_fused_norm_rope_supports_large_token_count, test_fused_q_bf16_query, test_fused_q_triton_supports_large_token_count, touching `test_fused_norm_rope_ds_mla, test_fused_norm_rope_supports_large_token_count, test_fused_q_bf16_query`.
- Code diff details:
  - `vllm/models/deepseek_v32/common/kernels.py` modified +6/-6 (12 lines); hunks: -152,8 +152,8 @@ def _fused_norm_rope_kernel(; -479,7 +479,7 @@ def fused_norm_rope(; symbols: _fused_norm_rope_kernel, fused_norm_rope, _fused_q_kernel, fused_q
  - `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +73/-0 (73 lines); hunks: -362,6 +362,46 @@ def test_fused_norm_rope_ds_mla(num_tokens: int):; -539,6 +579,39 @@ def test_fused_q_bf16_query(num_tokens: int, has_indexer: b...; symbols: test_fused_norm_rope_ds_mla, test_fused_norm_rope_supports_large_token_count, test_fused_q_bf16_query, test_fused_q_triton_supports_large_token_count
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/common/kernels.py
@@ -152,8 +152,8 @@ def _fused_norm_rope_kernel(
-    pid = tl.program_id(0)
-    tok_idx = tl.program_id(1)
+    tok_idx = tl.program_id(0).to(tl.int64)
+    pid = tl.program_id(1)
@@ -479,7 +479,7 @@ def fused_norm_rope(
-    _fused_norm_rope_kernel[(4, num_tokens)](
diff -- tests/kernels/test_fused_deepseek_v32_norm_rope.py
@@ -362,6 +362,46 @@ def test_fused_norm_rope_ds_mla(num_tokens: int):
+def test_fused_norm_rope_supports_large_token_count():
+    """Keep the token count off CUDA grid-y at its 65,536-block boundary."""
+    num_tokens = 65536
+    dev = "cuda"
+    dtype = torch.bfloat16
+    positions = torch.zeros(num_tokens, device=dev, dtype=torch.int64)
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/common/kernels.py` modified +6/-6
  - tests: `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +73/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #52046 - [nv] add pcp support in dsv3.2

- Link: https://github.com/vllm-project/vllm/pull/52046
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `vllm/models/deepseek_v32/attention.py`, `vllm/models/deepseek_v32/common/kernels.py`; associated commits `5f7a20b3162e`
- Diff scope read: GitHub Pull Request files API returned 6 files, +376/-119, 819 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[nv] add pcp support in dsv3.2"; model line: DeepSeek V3.2; category: performance/backend optimization; main diff: `vllm/models/deepseek_v32/common/kernels.py`, `vllm/models/deepseek_v32/attention.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`; technical summary: Covers "[nv] add pcp support in dsv3.2"; the main implementation surface is `vllm/models/deepseek_v32/common/kernels.py`, `vllm/models/deepseek_v32/attention.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/common/kernels.py` modified +130/-85 (215 lines); hunks: -103,12 +103,16 @@ def _fused_norm_rope_kernel(; -121,6 +125,8 @@ def _fused_norm_rope_kernel(; symbols: _fused_norm_rope_kernel, touching `_fused_norm_rope_kernel`; `vllm/models/deepseek_v32/attention.py` modified +103/-20 (123 lines); hunks: -1,14 +1,22; -35,6 +43,9; symbols: DeepseekV32Indexer, forward, touching `DeepseekV32Indexer, forward`; `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +90/-6 (96 lines); hunks: -292,6 +292,81 @@ def test_fused_norm_rope_no_indexer(num_tokens: int):; -405,19 +480,29 @@ def test_fused_norm_rope_supports_large_token_count():; symbols: test_fused_norm_rope_no_indexer, test_fused_norm_rope_materializes_pcp_cache_inputs, test_fused_norm_rope_ds_mla, test_fused_norm_rope_supports_large_token_count, touching `test_fused_norm_rope_no_indexer, test_fused_norm_rope_materializes_pcp_cache_inputs, test_fused_norm_rope_ds_mla`.
- Code diff details:
  - `vllm/models/deepseek_v32/common/kernels.py` modified +130/-85 (215 lines); hunks: -103,12 +103,16 @@ def _fused_norm_rope_kernel(; -121,6 +125,8 @@ def _fused_norm_rope_kernel(; symbols: _fused_norm_rope_kernel
  - `vllm/models/deepseek_v32/attention.py` modified +103/-20 (123 lines); hunks: -1,14 +1,22; -35,6 +43,9; symbols: DeepseekV32Indexer, forward
  - `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +90/-6 (96 lines); hunks: -292,6 +292,81 @@ def test_fused_norm_rope_no_indexer(num_tokens: int):; -405,19 +480,29 @@ def test_fused_norm_rope_supports_large_token_count():; symbols: test_fused_norm_rope_no_indexer, test_fused_norm_rope_materializes_pcp_cache_inputs, test_fused_norm_rope_ds_mla, test_fused_norm_rope_supports_large_token_count
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/common/kernels.py
@@ -103,12 +103,16 @@ def _fused_norm_rope_kernel(
+    kv_out_ptr,
+    kv_out_stride,
+    kpe_out_ptr,
+    kpe_out_stride,
@@ -121,6 +125,8 @@ def _fused_norm_rope_kernel(
+    index_k_out_ptr,
diff -- vllm/models/deepseek_v32/attention.py
@@ -1,14 +1,22 @@
+from typing import TYPE_CHECKING, cast
+from vllm.distributed.parallel_state import get_tp_group
+from vllm.model_executor.layers.attention.attention import get_attention_context
+from vllm.model_executor.layers.attention.pcp import (
+    finalize_mla_pcp_decode,
+    maybe_gather_mla_latent_cache_inputs,
diff -- tests/kernels/test_fused_deepseek_v32_norm_rope.py
@@ -292,6 +292,81 @@ def test_fused_norm_rope_no_indexer(num_tokens: int):
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/common/kernels.py` modified +130/-85; `vllm/models/deepseek_v32/attention.py` modified +103/-20
  - tests: `tests/kernels/test_fused_deepseek_v32_norm_rope.py` modified +90/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `tests/model_executor/layers/test_mla_short_prefill_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53021 - [Model] Remove unused DeepseekV32Indexer forward

- Link: https://github.com/vllm-project/vllm/pull/53021
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/attention.py`; associated commits `c0233fcf0100`
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-51, 65 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove unused DeepseekV32Indexer forward"; model line: DeepSeek V3.2; category: model implementation change; main diff: `vllm/models/deepseek_v32/attention.py`; technical summary: Covers "[Model] Remove unused DeepseekV32Indexer forward"; the main implementation surface is `vllm/models/deepseek_v32/attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/deepseek_v32/attention.py` modified +0/-51 (51 lines); hunks: -25,9 +25,6; -117,54 +114,6 @@ def __init__(; symbols: __init__, forward, DeepseekV32Attention, touching `__init__, forward, DeepseekV32Attention`.
- Code diff details:
  - `vllm/models/deepseek_v32/attention.py` modified +0/-51 (51 lines); hunks: -25,9 +25,6; -117,54 +114,6 @@ def __init__(; symbols: __init__, forward, DeepseekV32Attention
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/attention.py
@@ -25,9 +25,6 @@
-from vllm.model_executor.layers.quantization.utils.fp8_utils import (
-    per_token_group_quant_fp8,
-)
@@ -117,54 +114,6 @@ def __init__(
-    def forward(
-        self,
```

- Reviewed files:
  - runtime: `vllm/models/deepseek_v32/attention.py` modified +0/-51
- Risk and verification: Runtime changes concentrate in `vllm/models/deepseek_v32/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
