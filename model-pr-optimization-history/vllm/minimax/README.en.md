# vllm MiniMax M2 Series Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/kernels/attention/test_minimax_m3.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45720](https://github.com/vllm-project/vllm/pull/45720), [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `tests/kernels/core/test_minimax_reduce_rms.py` | [#37045](https://github.com/vllm-project/vllm/pull/37045), [#43410](https://github.com/vllm-project/vllm/pull/43410), [#45935](https://github.com/vllm-project/vllm/pull/45935) |
| `tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `tests/kernels/test_minimax_m3_amd_ops.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `tests/models/multimodal/processing/test_minimax_m3.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `tests/parser/engine/test_minimax_m2.py` | [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882) |
| `tests/reasoning/test_minimax_m2_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882), [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `tests/reasoning/test_minimax_m3_reasoning_parser.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45718](https://github.com/vllm-project/vllm/pull/45718) |
| `tests/tool_parsers/test_minimax_m2_tool_parser.py` | [#35895](https://github.com/vllm-project/vllm/pull/35895), [#39599](https://github.com/vllm-project/vllm/pull/39599), [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `tests/tool_parsers/test_minimax_m3_tool_parser.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` | no direct PR-number commit |
| `vllm/model_executor/layers/minimax_rms_norm/__init__.py` | [#43410](https://github.com/vllm-project/vllm/pull/43410) |
| `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` | [#43410](https://github.com/vllm-project/vllm/pull/43410) |
| `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` | [#43410](https://github.com/vllm-project/vllm/pull/43410), [#44983](https://github.com/vllm-project/vllm/pull/44983), [#45935](https://github.com/vllm-project/vllm/pull/45935) |
| `vllm/model_executor/models/minimax_m2.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#27537](https://github.com/vllm-project/vllm/pull/27537), [#27627](https://github.com/vllm-project/vllm/pull/27627), [#30384](https://github.com/vllm-project/vllm/pull/30384), [#31493](https://github.com/vllm-project/vllm/pull/31493), [#32736](https://github.com/vllm-project/vllm/pull/32736), [#32763](https://github.com/vllm-project/vllm/pull/32763), [#36965](https://github.com/vllm-project/vllm/pull/36965), [#37045](https://github.com/vllm-project/vllm/pull/37045), [#37214](https://github.com/vllm-project/vllm/pull/37214), [#37512](https://github.com/vllm-project/vllm/pull/37512), [#38191](https://github.com/vllm-project/vllm/pull/38191), ... (14 total) |
| `vllm/model_executor/warmup/minimax_m3_msa_warmup.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/amd/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/amd/model.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45810](https://github.com/vllm-project/vllm/pull/45810), [#45892](https://github.com/vllm-project/vllm/pull/45892), [#45896](https://github.com/vllm-project/vllm/pull/45896), [#46419](https://github.com/vllm-project/vllm/pull/46419), [#46545](https://github.com/vllm-project/vllm/pull/46545) |
| `vllm/models/minimax_m3/amd/mtp.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/amd/ops/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/amd/ops/index_topk.py` | [#46546](https://github.com/vllm-project/vllm/pull/46546) |
| `vllm/models/minimax_m3/amd/ops/sparse_attn.py` | [#46546](https://github.com/vllm-project/vllm/pull/46546) |
| `vllm/models/minimax_m3/amd/ops/swiglu_oai.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/common/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/common/indexer.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45892](https://github.com/vllm-project/vllm/pull/45892), [#46546](https://github.com/vllm-project/vllm/pull/46546) |
| `vllm/models/minimax_m3/common/mm_preprocess.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/common/ops/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/common/ops/index_topk.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `vllm/models/minimax_m3/common/ops/sparse_attn.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45720](https://github.com/vllm-project/vllm/pull/45720), [#46546](https://github.com/vllm-project/vllm/pull/46546) |
| `vllm/models/minimax_m3/common/sparse_attention.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45720](https://github.com/vllm-project/vllm/pull/45720), [#45892](https://github.com/vllm-project/vllm/pull/45892), [#46546](https://github.com/vllm-project/vllm/pull/46546) |
| `vllm/models/minimax_m3/common/vision_tower.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/nvidia/__init__.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/nvidia/indexer_msa.py` | [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `vllm/models/minimax_m3/nvidia/model.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45810](https://github.com/vllm-project/vllm/pull/45810), [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `vllm/models/minimax_m3/nvidia/mtp.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/models/minimax_m3/nvidia/sparse_attention_msa.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45892](https://github.com/vllm-project/vllm/pull/45892) |
| `vllm/parser/minimax_m2.py` | [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `vllm/reasoning/minimax_m2_reasoning_parser.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#29882](https://github.com/vllm-project/vllm/pull/29882), [#35352](https://github.com/vllm-project/vllm/pull/35352), [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `vllm/reasoning/minimax_m3_reasoning_parser.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381), [#45718](https://github.com/vllm-project/vllm/pull/45718) |
| `vllm/tool_parsers/minimax_m2_tool_parser.py` | [#30555](https://github.com/vllm-project/vllm/pull/30555), [#31083](https://github.com/vllm-project/vllm/pull/31083), [#32278](https://github.com/vllm-project/vllm/pull/32278), [#32342](https://github.com/vllm-project/vllm/pull/32342), [#35895](https://github.com/vllm-project/vllm/pull/35895), [#39599](https://github.com/vllm-project/vllm/pull/39599), [#43006](https://github.com/vllm-project/vllm/pull/43006), [#43025](https://github.com/vllm-project/vllm/pull/43025), [#45701](https://github.com/vllm-project/vllm/pull/45701) |
| `vllm/tool_parsers/minimax_m3_tool_parser.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/transformers_utils/configs/minimax_m3.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |
| `vllm/transformers_utils/processors/minimax_m3.py` | [#45381](https://github.com/vllm-project/vllm/pull/45381) |

## PR Coverage Summary

- Git-traced PRs: 34
- Extra PRs preserved from existing docs: 22
- Total PRs in this document: 56
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-01 | [#13454](https://github.com/vllm-project/vllm/pull/13454) | merged | [Model][MiniMaxText01] Support MiniMaxText01 model inference | `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/models/constant_size_cache.py` |
| 2025-04-29 | [#16328](https://github.com/vllm-project/vllm/pull/16328) | merged | [Model] support MiniMax-VL-01 model | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` |
| 2025-04-29 | [#17354](https://github.com/vllm-project/vllm/pull/17354) | merged | [Bugfix] Clean up MiniMax-VL and fix processing | `vllm/model_executor/models/minimax_vl_01.py`, `docs/source/models/supported_models.md`, `tests/models/multimodal/processing/test_common.py` |
| 2025-06-13 | [#19592](https://github.com/vllm-project/vllm/pull/19592) | merged | [Model] Fix minimax model cache & lm_head precision | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-06-16 | [#19677](https://github.com/vllm-project/vllm/pull/19677) | merged | [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM) | `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` |
| 2025-06-28 | [#20199](https://github.com/vllm-project/vllm/pull/20199) | merged | [CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes | `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2025-07-03 | [#20297](https://github.com/vllm-project/vllm/pull/20297) | merged | [Feature] Support MiniMax-M1 function calls features | `tests/tool_use/test_minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2025-07-11 | [#20211](https://github.com/vllm-project/vllm/pull/20211) | merged | [Model] Support HF format of minimax | `vllm/model_executor/models/minimax_text_01.py`, `tests/models/registry.py`, `vllm/model_executor/models/registry.py` |
| 2025-08-09 | [#22151](https://github.com/vllm-project/vllm/pull/22151) | merged | [V1] [Hybrid] Support Minimax-Text-01 in V1 | `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/lightning_attn.py` |
| 2025-08-15 | [#22928](https://github.com/vllm-project/vllm/pull/22928) | merged | [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax) | `tests/models/language/generation/test_hybrid.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py` |
| 2025-08-19 | [#22116](https://github.com/vllm-project/vllm/pull/22116) | merged | [Bugfix] Fix broken Minimax-01-VL model | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/test_tensor_schema.py`, `examples/offline_inference/vision_language.py` |
| 2025-08-27 | [#22589](https://github.com/vllm-project/vllm/pull/22589) | merged | [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models | `vllm/model_executor/models/minimax_text_01.py`, `vllm/config/compilation.py` |
| 2025-08-30 | [#23831](https://github.com/vllm-project/vllm/pull/23831) | merged | [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba | `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py` |
| 2025-10-26 | [#27535](https://github.com/vllm-project/vllm/pull/27535) | merged | [Model][MiniMax-M2] Support MiniMax-M2 Model | `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2025-10-27 | [#27537](https://github.com/vllm-project/vllm/pull/27537) | merged | Fix MiniMax-M2 copyright | `vllm/model_executor/models/minimax_m2.py` |
| 2025-10-29 | [#27627](https://github.com/vllm-project/vllm/pull/27627) | merged | Fix MiniMax-M2 rmsnorm precision and remove useless code | `vllm/model_executor/models/minimax_m2.py` |
| 2025-12-10 | [#30384](https://github.com/vllm-project/vllm/pull/30384) | merged | [BugFix] Fix minimax m2 model rotary_dim | `vllm/model_executor/models/minimax_m2.py` |
| 2025-12-11 | [#29882](https://github.com/vllm-project/vllm/pull/29882) | merged | [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content. | `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2025-12-17 | [#30555](https://github.com/vllm-project/vllm/pull/30555) | merged | [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction | `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2025-12-22 | [#31083](https://github.com/vllm-project/vllm/pull/31083) | merged | Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2025-12-29 | [#31493](https://github.com/vllm-project/vllm/pull/31493) | merged | Optimize QKNorm for MiniMax-M2/M2.1 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-01-15 | [#32342](https://github.com/vllm-project/vllm/pull/32342) | merged | Fix optional parameter parsing in MiniMax M2 tool parser #32278 | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2026-01-24 | [#32763](https://github.com/vllm-project/vllm/pull/32763) | merged | feat: Complete LoRA support for MiniMaxM2 Fixes #32736 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-02-26 | [#35352](https://github.com/vllm-project/vllm/pull/35352) | merged | [Bug] Fix missing tag after tool call in MiniMax 2.1 | `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2026-03-12 | [#35895](https://github.com/vllm-project/vllm/pull/35895) | merged | [Bugfix] Fix minimax_m2 tool parser when stream interval > 1 | `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py` |
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/models/kimi_linear.py` |
| 2026-03-26 | [#37214](https://github.com/vllm-project/vllm/pull/37214) | merged | Fix minimax m2.5 nvfp4 kv scales weight loading | `vllm/model_executor/models/minimax_m2.py` |
| 2026-03-30 | [#36965](https://github.com/vllm-project/vllm/pull/36965) | merged | [Model][Quantization] Add GGUF support for MiniMax-M2.1 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-06 | [#37512](https://github.com/vllm-project/vllm/pull/37512) | merged | MiniMax-M2: add Eagle3 speculative decoding support | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-10 | [#37045](https://github.com/vllm-project/vllm/pull/37045) | merged | [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels | `vllm/model_executor/models/minimax_m2.py`, `tests/kernels/core/test_minimax_reduce_rms.py` |
| 2026-04-14 | [#39683](https://github.com/vllm-project/vllm/pull/39683) | merged | [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter | `vllm/parser/minimax_m2_parser.py` |
| 2026-04-16 | [#39861](https://github.com/vllm-project/vllm/pull/39861) | merged | [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__() | `vllm/parser/minimax_m2_parser.py` |
| 2026-04-27 | [#38191](https://github.com/vllm-project/vllm/pull/38191) | merged | [Bugfix] Fix k_norm weight sharding in MiniMaxM2Attention when total_num_kv_heads < tp_size | `vllm/model_executor/models/minimax_m2.py` |
| 2026-05-14 | [#39599](https://github.com/vllm-project/vllm/pull/39599) | merged | fix(tool-parser): preserve "none"/"nil" strings as valid enum values in minimax_m2 | `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2026-05-18 | [#42497](https://github.com/vllm-project/vllm/pull/42497) | merged | [Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2) | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` |
| 2026-05-18 | [#43006](https://github.com/vllm-project/vllm/pull/43006) | merged | [Refactor] Extract shared coerce_to_schema_type utility from Minimax M2 tool parser | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2026-05-19 | [#43025](https://github.com/vllm-project/vllm/pull/43025) | merged | [Refactor] Extract extract_types_from_schema utility from Minimax M2 tool parser | `vllm/tool_parsers/minimax_m2_tool_parser.py` |
| 2026-05-26 | [#43410](https://github.com/vllm-project/vllm/pull/43410) | merged | [Kernel] Porting fuse_minimax_qk_norm to manual fusion | `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`, `vllm/model_executor/models/minimax_m2.py` |
| 2026-05-30 | [#38445](https://github.com/vllm-project/vllm/pull/38445) | merged | [PERF]MiniMax-M2 gate kernel | `vllm/model_executor/models/minimax_m2.py` |
| 2026-06-02 | [#44279](https://github.com/vllm-project/vllm/pull/44279) | merged | [Refactor] Remove dead code from parser infrastructure | `vllm/parser/parser_manager.py`, `vllm/parser/minimax_m2_parser.py`, `vllm/parser/abstract_parser.py` |
| 2026-06-04 | [#43556](https://github.com/vllm-project/vllm/pull/43556) | merged | [Attention] Mamba attention module refactor - LINEAR | `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` |
| 2026-06-09 | [#44983](https://github.com/vllm-project/vllm/pull/44983) | merged | [Bugfix] Fix minimax_qk_norm_fusion | `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-15 | [#45381](https://github.com/vllm-project/vllm/pull/45381) | merged | [Model] Add MiniMax M3 support | `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py` |
| 2026-06-16 | [#45701](https://github.com/vllm-project/vllm/pull/45701) | merged | [Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser | `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2026-06-17 | [#45720](https://github.com/vllm-project/vllm/pull/45720) | merged | [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype | `vllm/models/minimax_m3/common/sparse_attention.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`, `tests/kernels/attention/test_minimax_m3.py` |
| 2026-06-17 | [#45896](https://github.com/vllm-project/vllm/pull/45896) | merged | [feature] MiniMax-M3-MXFP4 support added | `vllm/models/minimax_m3/amd/model.py` |
| 2026-06-18 | [#45988](https://github.com/vllm-project/vllm/pull/45988) | merged | [Perf] Remove unused loggers in `reasoning/` | `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py` |
| 2026-06-21 | [#45935](https://github.com/vllm-project/vllm/pull/45935) | merged | [Model]Fix MiniMaxM2ForCausalLM perf regression | `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `tests/kernels/core/test_minimax_reduce_rms.py` |
| 2026-06-22 | [#45993](https://github.com/vllm-project/vllm/pull/45993) | merged | [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM | `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py` |
| 2026-06-23 | [#45892](https://github.com/vllm-project/vllm/pull/45892) | merged | [Minimax-M3] BF16/FP8 Indexer using MSA | `vllm/models/minimax_m3/nvidia/indexer_msa.py`, `vllm/models/minimax_m3/common/indexer.py`, `vllm/models/minimax_m3/nvidia/model.py` |
| 2026-06-23 | [#45718](https://github.com/vllm-project/vllm/pull/45718) | merged | [Bugfix] Parse MiniMax M3 streaming reasoning by text markers | `vllm/reasoning/minimax_m3_reasoning_parser.py`, `tests/reasoning/test_minimax_m3_reasoning_parser.py` |
| 2026-06-24 | [#45810](https://github.com/vllm-project/vllm/pull/45810) | merged | [Model][MiniMax-M3] Add pipeline parallelism support | `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py` |
| 2026-06-25 | [#46546](https://github.com/vllm-project/vllm/pull/46546) | merged | [ROCm][ [Perf] sparse attention optimization on minimax-m3 | `vllm/models/minimax_m3/amd/ops/index_topk.py`, `vllm/models/minimax_m3/amd/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py` |
| 2026-06-26 | [#46419](https://github.com/vllm-project/vllm/pull/46419) | merged | [ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4 | `vllm/models/minimax_m3/amd/model.py` |
| 2026-06-26 | [#46545](https://github.com/vllm-project/vllm/pull/46545) | merged | [ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model | `vllm/models/minimax_m3/amd/model.py` |

## Per-PR Diff Audit Cards

### PR #13454 - [Model][MiniMaxText01] Support MiniMaxText01 model inference

- Link: https://github.com/vllm-project/vllm/pull/13454
- Status/date: merged / 2025-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +2440/-130, 2657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][MiniMaxText01] Support MiniMaxText01 model inference"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/models/constant_size_cache.py`; technical summary: Covers "[Model][MiniMaxText01] Support MiniMaxText01 model inference"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/models/constant_size_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func, touching `replace_weight_name, weight_loader_with_alias, wrapper`; `vllm/model_executor/layers/lightning_attn.py` added +651/-0 (651 lines); hunks: -0,0 +1,651; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel, touching `_fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce`; `vllm/model_executor/models/constant_size_cache.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: ConstantSizeCache, for, __init__, cache, touching `ConstantSizeCache, for, __init__`; `vllm/model_executor/models/mamba_cache.py` modified +21/-111 (132 lines); hunks: -1,12 +1,13; -21,7 +22,7 @@ def at_layer_idx(self, layer_idx):; symbols: at_layer_idx, MambaCacheManager, __init__, touching `at_layer_idx, MambaCacheManager, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
  - `vllm/model_executor/layers/lightning_attn.py` added +651/-0 (651 lines); hunks: -0,0 +1,651; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel
  - `vllm/model_executor/models/constant_size_cache.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: ConstantSizeCache, for, __init__, cache
  - `vllm/model_executor/models/mamba_cache.py` modified +21/-111 (132 lines); hunks: -1,12 +1,13; -21,7 +22,7 @@ def at_layer_idx(self, layer_idx):; symbols: at_layer_idx, MambaCacheManager, __init__
  - `vllm/model_executor/models/minimax_cache.py` added +35/-0 (35 lines); hunks: -0,0 +1,35; symbols: MinimaxCacheParams, at_layer_idx, MinimaxCacheManager, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -0,0 +1,1273 @@
+# SPDX-License-Identifier: Apache-2.0
+"""Inference-only MiniMaxText01 model."""
+import copy
+import math
+import re
+from typing import Dict, Iterable, List, Optional, Tuple, Union
diff -- vllm/model_executor/layers/lightning_attn.py
@@ -0,0 +1,651 @@
+# SPDX-License-Identifier: Apache-2.0
+import torch
+import triton
+import triton.language as tl
+from einops import rearrange
+@triton.jit
diff -- vllm/model_executor/models/constant_size_cache.py
@@ -0,0 +1,136 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0; `vllm/model_executor/layers/lightning_attn.py` added +651/-0; `vllm/model_executor/models/constant_size_cache.py` added +136/-0; `vllm/model_executor/models/mamba_cache.py` modified +21/-111; `vllm/model_executor/models/minimax_cache.py` added +35/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - docs: `docs/source/models/supported_models.md` modified +5/-0
  - tests: `tests/models/registry.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_lightning_attn.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16328 - [Model] support MiniMax-VL-01 model

- Link: https://github.com/vllm-project/vllm/pull/16328
- Status/date: merged / 2025-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +954/-19, 1193 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] support MiniMax-VL-01 model"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`; technical summary: Covers "[Model] support MiniMax-VL-01 model"; the main implementation surface is `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches, touching `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`; `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression, touching `test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements`; `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__, touching `MiniMaxVL01Config, __init__`; `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__, touching `MiniMaxText01Config, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
  - `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__
  - `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__
  - `vllm/model_executor/models/minimax_text_01.py` modified +53/-14 (67 lines); hunks: -3,7 +3,7; -110,7 +110,17 @@ def _forward(; symbols: _forward, forward, _prefill_and_mix_infer, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -0,0 +1,615 @@
+# SPDX-License-Identifier: Apache-2.0
+from abc import abstractmethod
+from collections.abc import Iterable, Mapping, Sequence
+from dataclasses import dataclass
+from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
+                    TypeVar, Union, cast)
diff -- tests/models/multimodal/processing/test_minimax_vl_01.py
@@ -0,0 +1,99 @@
+# SPDX-License-Identifier: Apache-2.0
+import pytest
+from PIL import Image
+from vllm.multimodal import MULTIMODAL_REGISTRY
+from vllm.multimodal.parse import ImageSize
+from vllm.multimodal.processing import BaseMultiModalProcessor
diff -- vllm/transformers_utils/configs/minimax_vl_01.py
@@ -0,0 +1,70 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0; `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0; `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0; `vllm/model_executor/models/minimax_text_01.py` modified +53/-14; `vllm/transformers_utils/configs/__init__.py` modified +4/-0
  - tests: `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0; `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py` modified +19/-0; `tests/models/decoder_only/vision_language/test_models.py` modified +13/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17354 - [Bugfix] Clean up MiniMax-VL and fix processing

- Link: https://github.com/vllm-project/vllm/pull/17354
- Status/date: merged / 2025-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +38/-283, 442 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Clean up MiniMax-VL and fix processing"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_vl_01.py`, `docs/source/models/supported_models.md`, `tests/models/multimodal/processing/test_common.py`; technical summary: Covers "[Bugfix] Clean up MiniMax-VL and fix processing"; the main implementation surface is `vllm/model_executor/models/minimax_vl_01.py`, `docs/source/models/supported_models.md`, `tests/models/multimodal/processing/test_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches, touching `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`; `docs/source/models/supported_models.md` modified +7/-0 (7 lines); hunks: -979,6 +979,13 @@ See this page for more information on how to use generativ; `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunks: -270,6 +270,7 @@ def _test_processing_correctness_mistral(; symbols: _test_processing_correctness_mistral, touching `_test_processing_correctness_mistral`; `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `docs/source/models/supported_models.md` modified +7/-0 (7 lines); hunks: -979,6 +979,13 @@ See this page for more information on how to use generativ
  - `tests/models/multimodal/processing/test_common.py` modified +1/-0 (1 lines); hunks: -270,6 +270,7 @@ def _test_processing_correctness_mistral(; symbols: _test_processing_correctness_mistral
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,52 +1,32 @@
+from collections.abc import Iterable, Mapping
+from typing import Literal, Optional, Set, Tuple, TypedDict, Union, cast
-from abc import abstractmethod
-from collections.abc import Iterable, Mapping, Sequence
-from dataclasses import dataclass
-from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
diff -- docs/source/models/supported_models.md
@@ -979,6 +979,13 @@ See [this page](#generative-models) for more information on how to use generativ
+- * `MiniMaxVL01ForConditionalGeneration`
+  * MiniMax-VL
+  * T + I<sup>E+</sup>
+  * `MiniMaxAI/MiniMax-VL-01`, etc.
+  *
+  * ✅︎
diff -- tests/models/multimodal/processing/test_common.py
@@ -270,6 +270,7 @@ def _test_processing_correctness_mistral(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282
  - docs: `docs/source/models/supported_models.md` modified +7/-0
  - tests: `tests/models/multimodal/processing/test_common.py` modified +1/-0; `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19592 - [Model] Fix minimax model cache & lm_head precision

- Link: https://github.com/vllm-project/vllm/pull/19592
- Status/date: merged / 2025-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix minimax model cache & lm_head precision"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_text_01.py`; technical summary: Covers "[Model] Fix minimax model cache & lm_head precision"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits, touching `layer_fn, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -856,7 +856,7 @@ def layer_fn(prefix):
-        self.minimax_cache = MinimaxCacheManager(dtype=self._dtype,
+        self.minimax_cache = MinimaxCacheManager(dtype=torch.float32,
@@ -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.lm_head.float()
@@ -1054,7 +1054,7 @@ def forward(self,
-        logits = self.logits_processor(self.lm_head, hidden_states,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19677 - [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)

- Link: https://github.com/vllm-project/vllm/pull/19677
- Status/date: merged / 2025-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +4/-0, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`; technical summary: Covers "[Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)"; the main implementation surface is `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name, touching `name`.
- Code diff details:
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name
- Key code excerpts:

```diff
diff -- tests/models/registry.py
@@ -205,6 +205,8 @@ def check_available_online(
+    "MiniMaxM1ForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-M1-40k",
+                                            trust_remote_code=True),
diff -- docs/models/supported_models.md
@@ -370,6 +370,7 @@ Specified using `--task generate`.
+| `MiniMaxM1ForCausalLM`                        | MiniMax-Text                                        | `MiniMaxAI/MiniMax-M1-40k`, `MiniMaxAI/MiniMax-M1-80k`etc.
diff -- vllm/model_executor/models/registry.py
@@ -36,6 +36,7 @@
+    "MiniMaxM1ForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
```

- Reviewed files:
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20199 - [CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes

- Link: https://github.com/vllm-project/vllm/pull/20199
- Status/date: merged / 2025-06-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-1, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes"; model line: MiniMax M2 Series; category: bug fix; main diff: `tests/models/registry.py`, `tests/models/test_initialization.py`; technical summary: Covers "[CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes"; the main implementation surface is `tests/models/registry.py`, `tests/models/test_initialization.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/registry.py` modified +8/-1 (9 lines); hunks: -70,6 +70,12 @@ class _HfExamplesInfo:; -207,7 +213,8 @@ def check_available_online(; symbols: _HfExamplesInfo, check_transformers_version, check_available_online, touching `_HfExamplesInfo, check_transformers_version, check_available_online`; `tests/models/test_initialization.py` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1, touching `_initialize_kv_caches_v1`.
- Code diff details:
  - `tests/models/registry.py` modified +8/-1 (9 lines); hunks: -70,6 +70,12 @@ class _HfExamplesInfo:; -207,7 +213,8 @@ def check_available_online(; symbols: _HfExamplesInfo, check_transformers_version, check_available_online
  - `tests/models/test_initialization.py` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
- Key code excerpts:

```diff
diff -- tests/models/registry.py
@@ -70,6 +70,12 @@ class _HfExamplesInfo:
+    revision: Optional[str] = None
+    """
+    The specific revision (commit hash, tag, or branch) to use for the model.
+    If not specified, the default revision will be used.
+    """
@@ -207,7 +213,8 @@ def check_available_online(
diff -- tests/models/test_initialization.py
@@ -88,6 +88,7 @@ def _initialize_kv_caches_v1(self, vllm_config):
+            revision=model_info.revision,
```

- Reviewed files:
  - tests: `tests/models/registry.py` modified +8/-1; `tests/models/test_initialization.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20297 - [Feature] Support MiniMax-M1 function calls features

- Link: https://github.com/vllm-project/vllm/pull/20297
- Status/date: merged / 2025-07-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +842/-1, 866 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support MiniMax-M1 function calls features"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `tests/tool_use/test_minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`; technical summary: Covers "[Feature] Support MiniMax-M1 function calls features"; the main implementation surface is `tests/tool_use/test_minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_use/test_minimax_tool_parser.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: minimax_tokenizer, minimax_tool_parser, assert_tool_calls, test_extract_tool_calls_no_tools, touching `minimax_tokenizer, minimax_tool_parser, assert_tool_calls`; `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think, touching `MinimaxToolParser, __init__, preprocess_model_output`; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-1 (3 lines); hunks: -10,6 +10,7; -20,5 +21,5; `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `tests/tool_use/test_minimax_tool_parser.py` added +371/-0 (371 lines); hunks: -0,0 +1,371; symbols: minimax_tokenizer, minimax_tool_parser, assert_tool_calls, test_extract_tool_calls_no_tools
  - `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py` added +369/-0 (369 lines); hunks: -0,0 +1,369; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-1 (3 lines); hunks: -10,6 +10,7; -20,5 +21,5
  - `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -256,6 +256,15 @@ For Qwen2.5, the chat template in tokenizer_config.json has...
- Key code excerpts:

```diff
diff -- tests/tool_use/test_minimax_tool_parser.py
@@ -0,0 +1,371 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+import json
+import pytest
+from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
+from vllm.entrypoints.openai.tool_parsers import MinimaxToolParser
diff -- vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py
@@ -0,0 +1,369 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+from collections.abc import Sequence
+from typing import Union
+import partial_json_parser
diff -- vllm/entrypoints/openai/tool_parsers/__init__.py
@@ -10,6 +10,7 @@
```

- Reviewed files:
  - tests: `tests/tool_use/test_minimax_tool_parser.py` added +371/-0
  - runtime: `vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py` added +369/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-1
  - docs: `examples/tool_chat_template_minimax_m1.jinja` added +91/-0; `docs/features/tool_calling.md` modified +9/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_minimax_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20211 - [Model] Support HF format of minimax

- Link: https://github.com/vllm-project/vllm/pull/20211
- Status/date: merged / 2025-07-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +36/-11, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support HF format of minimax"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/model_executor/models/minimax_text_01.py`, `tests/models/registry.py`, `vllm/model_executor/models/registry.py`; technical summary: Covers "[Model] Support HF format of minimax"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`, `tests/models/registry.py`, `vllm/model_executor/models/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer, touching `__init__, which_layer, is_linear_attn_layer`; `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -218,6 +218,8 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -35,6 +35,7; symbols: name, touching `name`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -218,6 +218,8 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -35,6 +35,7; symbols: name
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -667,16 +667,24 @@ def __init__(
-                config, 'layernorm_linear_attention_alpha', 1)
+                config, 'layernorm_linear_attention_alpha',
+                getattr(config, 'linear_attn_alpha_factor', 1))
-                config, 'layernorm_linear_attention_beta', 1)
+                config, 'layernorm_linear_attention_beta',
+                getattr(config, 'linear_attn_beta_factor', 1))
diff -- tests/models/registry.py
@@ -218,6 +218,8 @@ def check_available_online(
+    "MiniMaxForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-Text-01-hf",
+                                          min_transformers_version="4.53"),
diff -- vllm/model_executor/models/registry.py
@@ -35,6 +35,7 @@
+    "MiniMaxForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22151 - [V1] [Hybrid] Support Minimax-Text-01 in V1

- Link: https://github.com/vllm-project/vllm/pull/22151
- Status/date: merged / 2025-08-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +234/-42, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] [Hybrid] Support Minimax-Text-01 in V1"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/lightning_attn.py`; technical summary: Covers "[V1] [Hybrid] Support Minimax-Text-01 in V1"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/lightning_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape, touching `jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type`; `vllm/model_executor/layers/mamba/mamba_utils.py` modified +11/-0 (11 lines); hunks: -5,6 +5,17; symbols: MambaStateShapeCalculator, linear_attention_state_shape, mamba1_state_shape, touching `MambaStateShapeCalculator, linear_attention_state_shape, mamba1_state_shape`; `vllm/model_executor/layers/lightning_attn.py` modified +1/-1 (2 lines); hunks: -532,7 +532,7 @@ def _linear_attn_decode_kernel(; symbols: _linear_attn_decode_kernel, touching `_linear_attn_decode_kernel`; `vllm/v1/attention/backends/linear_attn.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: LinearAttentionBackend, get_builder_cls, LinearAttentionMetadata, LinearAttentionMetadataBuilder, touching `LinearAttentionBackend, get_builder_cls, LinearAttentionMetadata`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +11/-0 (11 lines); hunks: -5,6 +5,17; symbols: MambaStateShapeCalculator, linear_attention_state_shape, mamba1_state_shape
  - `vllm/model_executor/layers/lightning_attn.py` modified +1/-1 (2 lines); hunks: -532,7 +532,7 @@ def _linear_attn_decode_kernel(; symbols: _linear_attn_decode_kernel
  - `vllm/v1/attention/backends/linear_attn.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: LinearAttentionBackend, get_builder_cls, LinearAttentionMetadata, LinearAttentionMetadataBuilder
  - `vllm/v1/attention/backends/mamba_selectors.py` modified +3/-1 (4 lines); hunks: -1,16 +1,18; symbols: get_mamba_attn_backend
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -14,8 +14,9 @@
+from vllm import envs
-from vllm.config import CacheConfig, VllmConfig
+from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
@@ -33,6 +34,9 @@
+from vllm.model_executor.layers.mamba.abstract import MambaBase
+from vllm.model_executor.layers.mamba.mamba_utils import (
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -5,6 +5,17 @@
+    @classmethod
+    def linear_attention_state_shape(
+        cls,
+        num_heads: int,
+        tp_size: int,
+        head_dim: int,
diff -- vllm/model_executor/layers/lightning_attn.py
@@ -532,7 +532,7 @@ def _linear_attn_decode_kernel(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40; `vllm/model_executor/layers/mamba/mamba_utils.py` modified +11/-0; `vllm/model_executor/layers/lightning_attn.py` modified +1/-1; `vllm/v1/attention/backends/linear_attn.py` added +67/-0; `vllm/v1/attention/backends/mamba_selectors.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22928 - [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)

- Link: https://github.com/vllm-project/vllm/pull/22928
- Status/date: merged / 2025-08-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +467/-87, 1435 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `tests/models/language/generation/test_hybrid.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py`; technical summary: Covers "[V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)"; the main implementation surface is `tests/models/language/generation/test_hybrid.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/language/generation/test_hybrid.py` modified +62/-0 (62 lines); hunks: -431,3 +431,65 @@ def test_full_cuda_graph(; symbols: test_full_cuda_graph, test_fp32_state, touching `test_full_cuda_graph, test_fp32_state`; `vllm/model_executor/layers/mamba/mamba_utils.py` modified +52/-0 (52 lines); hunks: -1,6 +1,58; symbols: MambaStateDtypeCalculator, linear_attention_state_dtype, mamba1_state_dtype, mamba2_state_dtype, touching `MambaStateDtypeCalculator, linear_attention_state_dtype, mamba1_state_dtype`; `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +31/-20 (51 lines); hunks: -8,7 +8,7; -21,7 +21,7; symbols: MambaMixer2, __init__, forward_native, touching `MambaMixer2, __init__, forward_native`; `vllm/model_executor/models/zamba2.py` modified +33/-5 (38 lines); hunks: -18,7 +18,7; -33,7 +33,7; symbols: Zamba2MambaDecoderLayer, __init__, touching `Zamba2MambaDecoderLayer, __init__`.
- Code diff details:
  - `tests/models/language/generation/test_hybrid.py` modified +62/-0 (62 lines); hunks: -431,3 +431,65 @@ def test_full_cuda_graph(; symbols: test_full_cuda_graph, test_fp32_state
  - `vllm/model_executor/layers/mamba/mamba_utils.py` modified +52/-0 (52 lines); hunks: -1,6 +1,58; symbols: MambaStateDtypeCalculator, linear_attention_state_dtype, mamba1_state_dtype, mamba2_state_dtype
  - `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +31/-20 (51 lines); hunks: -8,7 +8,7; -21,7 +21,7; symbols: MambaMixer2, __init__, forward_native
  - `vllm/model_executor/models/zamba2.py` modified +33/-5 (38 lines); hunks: -18,7 +18,7; -33,7 +33,7; symbols: Zamba2MambaDecoderLayer, __init__
  - `vllm/model_executor/models/mamba2.py` modified +30/-6 (36 lines); hunks: -11,7 +11,7; -20,7 +20,7; symbols: Mamba2DecoderLayer, __init__
- Key code excerpts:

```diff
diff -- tests/models/language/generation/test_hybrid.py
@@ -431,3 +431,65 @@ def test_full_cuda_graph(
+@pytest.mark.parametrize("model", ["Zyphra/Zamba2-1.2B-instruct"])
+@pytest.mark.parametrize("max_tokens", [64])
+@pytest.mark.parametrize("num_logprobs", [5])
+def test_fp32_state(
+    hf_runner,
+    vllm_runner,
diff -- vllm/model_executor/layers/mamba/mamba_utils.py
@@ -1,6 +1,58 @@
+from typing import Union
+import torch
+from vllm.config import MambaDType, ModelDType
+from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_kv_cache_torch_dtype
+class MambaStateDtypeCalculator:
+    @classmethod
diff -- vllm/model_executor/layers/mamba/mamba_mixer2.py
@@ -8,7 +8,7 @@
```

- Reviewed files:
  - tests: `tests/models/language/generation/test_hybrid.py` modified +62/-0
  - runtime: `vllm/model_executor/layers/mamba/mamba_utils.py` modified +52/-0; `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +31/-20; `vllm/model_executor/models/zamba2.py` modified +33/-5; `vllm/model_executor/models/mamba2.py` modified +30/-6; `vllm/model_executor/models/minimax_text_01.py` modified +31/-3; `vllm/model_executor/models/nemotron_h.py` modified +28/-4
- Risk and verification: The diff ships test coverage in `tests/models/language/generation/test_hybrid.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22116 - [Bugfix] Fix broken Minimax-01-VL model

- Link: https://github.com/vllm-project/vllm/pull/22116
- Status/date: merged / 2025-08-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +123/-32, 258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix broken Minimax-01-VL model"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/test_tensor_schema.py`, `examples/offline_inference/vision_language.py`; technical summary: Covers "[Bugfix] Fix broken Minimax-01-VL model"; the main implementation surface is `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/test_tensor_schema.py`, `examples/offline_inference/vision_language.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config, touching `MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config`; `tests/models/multimodal/test_tensor_schema.py` modified +0/-1 (1 lines); hunks: -30,7 +30,6; `examples/offline_inference/vision_language.py` modified +34/-0 (34 lines); hunks: -815,6 +815,39 @@ def run_minicpmv(questions: list[str], modality: str) -> Mo...; -1463,6 +1496,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_minicpmv, run_minimax_vl_01, run_mistral3, run_tarsier2, touching `run_minicpmv, run_minimax_vl_01, run_mistral3`.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config
  - `tests/models/multimodal/test_tensor_schema.py` modified +0/-1 (1 lines); hunks: -30,7 +30,6
  - `examples/offline_inference/vision_language.py` modified +34/-0 (34 lines); hunks: -815,6 +815,39 @@ def run_minicpmv(questions: list[str], modality: str) -> Mo...; -1463,6 +1496,7 @@ def run_tarsier2(questions: list[str], modality: str) -> M...; symbols: run_minicpmv, run_minimax_vl_01, run_mistral3, run_tarsier2
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,11 +1,13 @@
-from typing import Literal, Optional, TypedDict, Union, cast
+from typing import Annotated, Literal, Optional, Union, cast
+from transformers.models.llava_next.modeling_llava_next import (
+    get_anyres_image_grid_shape, unpad_image)
@@ -17,6 +19,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
diff -- tests/models/multimodal/test_tensor_schema.py
@@ -30,7 +30,6 @@
-    "MiniMaxVL01ForConditionalGeneration": "broken model",
diff -- examples/offline_inference/vision_language.py
@@ -815,6 +815,39 @@ def run_minicpmv(questions: list[str], modality: str) -> ModelRequestData:
+def run_minimax_vl_01(questions: list[str], modality: str) -> ModelRequestData:
+    assert modality == "image"
+    model_name = "MiniMaxAI/MiniMax-VL-01"
+    engine_args = EngineArgs(
+        model=model_name,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31
  - tests: `tests/models/multimodal/test_tensor_schema.py` modified +0/-1
  - docs: `examples/offline_inference/vision_language.py` modified +34/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/test_tensor_schema.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22589 - [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models

- Link: https://github.com/vllm-project/vllm/pull/22589
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +98/-137, 387 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/model_executor/models/minimax_text_01.py`, `vllm/config/compilation.py`; technical summary: Covers "[V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`, `vllm/config/compilation.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq, touching `forward, MiniMaxText01RotaryEmbedding, __init__`; `vllm/config/compilation.py` modified +1/-0 (1 lines); hunks: -339,6 +339,7 @@ class CompilationConfig:; symbols: CompilationConfig, compute_hash, touching `CompilationConfig, compute_hash`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq
  - `vllm/config/compilation.py` modified +1/-0 (1 lines); hunks: -339,6 +339,7 @@ class CompilationConfig:; symbols: CompilationConfig, compute_hash
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,7 +1,6 @@
-import copy
@@ -19,13 +18,14 @@
+from vllm.compilation.decorators import support_torch_compile
-from vllm.forward_context import get_forward_context
+from vllm.forward_context import ForwardContext, get_forward_context
@@ -43,12 +43,15 @@
diff -- vllm/config/compilation.py
@@ -339,6 +339,7 @@ class CompilationConfig:
+        "vllm.linear_attention",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137; `vllm/config/compilation.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/config/compilation.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23831 - [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba

- Link: https://github.com/vllm-project/vllm/pull/23831
- Status/date: merged / 2025-08-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +448/-410, 917 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py`; technical summary: Covers "[V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba"; the main implementation surface is `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/mamba/linear_attn.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: MiniMaxText01RMSNormTP, __init__, weight_loader, _forward, touching `MiniMaxText01RMSNormTP, __init__, weight_loader`; `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader, touching `inner_func, MiniMaxText01RMSNormTP, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/mamba/linear_attn.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: MiniMaxText01RMSNormTP, __init__, weight_loader, _forward
  - `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/mamba/linear_attn.py
@@ -0,0 +1,442 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import math
+from typing import TYPE_CHECKING, Optional, Union
+if TYPE_CHECKING:
+    from vllm.attention.backends.abstract import AttentionBackend
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,45 +1,37 @@
-import math
-    from vllm.attention.backends.abstract import AttentionBackend
+    pass
-import torch.nn.functional as F
-from einops import rearrange
-from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/mamba/linear_attn.py` added +442/-0; `vllm/model_executor/models/minimax_text_01.py` modified +6/-410
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27535 - [Model][MiniMax-M2] Support MiniMax-M2 Model

- Link: https://github.com/vllm-project/vllm/pull/27535
- Status/date: merged / 2025-10-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `720af6ab7911`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1306/-0, 1347 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][MiniMax-M2] Support MiniMax-M2 Model"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; technical summary: Covers "[Model][MiniMax-M2] Support MiniMax-M2 Model"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward, touching `MiniMaxM2MoE, __init__, ebias_weight_loader`; `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser, touching `MiniMaxM2ReasoningParser, start_token, end_token`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -0,0 +1,585 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -0,0 +1,69 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from vllm.entrypoints.openai.protocol import (
+    ChatCompletionRequest,
+    DeltaMessage,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` added +585/-0; `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27537 - Fix MiniMax-M2 copyright

- Link: https://github.com/vllm-project/vllm/pull/27537
- Status/date: merged / 2025-10-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `5980604c44d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-3, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix MiniMax-M2 copyright"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "Fix MiniMax-M2 copyright"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -1,10 +1,9 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2025 The MiniMax AI team.
-# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27627 - Fix MiniMax-M2 rmsnorm precision and remove useless code

- Link: https://github.com/vllm-project/vllm/pull/27627
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `d6704dd099b7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +1/-19, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix MiniMax-M2 rmsnorm precision and remove useless code"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "Fix MiniMax-M2 rmsnorm precision and remove useless code"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -263,23 +263,6 @@ def __init__(
-        # TODO: support MTP
-        attn_window_size = getattr(config, "attn_window_size", None)
-        if attn_window_size is not None:
-            if isinstance(attn_window_size, list):
-                attn_window_size = attn_window_size[layer_idx]
-            elif isinstance(attn_window_size, int):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +0/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30384 - [BugFix] Fix minimax m2 model rotary_dim

- Link: https://github.com/vllm-project/vllm/pull/30384
- Status/date: merged / 2025-12-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `d017bceb08ea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix minimax m2 model rotary_dim"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "[BugFix] Fix minimax m2 model rotary_dim"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -201,7 +201,7 @@ def __init__(
-            rotary_dim=rotary_dim,
+            rotary_dim=self.head_dim,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29882 - [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content.

- Link: https://github.com/vllm-project/vllm/pull/29882
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `6299628d326f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +468/-0, 484 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content."; model line: MiniMax M2 Series; category: bug fix; main diff: `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; technical summary: Covers "[bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content."; the main implementation surface is `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning, touching `minimax_m2_tokenizer, test_reasoning`; `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning, touching `minimax_m2_tokenizer, test_reasoning`; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser, touching `MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming`.
- Code diff details:
  - `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning
  - `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser
- Key code excerpts:

```diff
diff -- tests/reasoning/test_minimax_m2_reasoning_parser.py
@@ -0,0 +1,230 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- tests/reasoning/test_minimax_m2_append_reasoning_parser.py
@@ -0,0 +1,195 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -19,6 +19,10 @@
```

- Reviewed files:
  - tests: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0; `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30555 - [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction

- Link: https://github.com/vllm-project/vllm/pull/30555
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `20fda431515d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +137/-4, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction"; model line: MiniMax M2 Series; category: bug fix; main diff: `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "[Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction"; the main implementation surface is `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser, touching `FakeTokenizer, __init__, get_vocab`; `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming, touching `_reset_streaming_state, _extract_name, extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- tests/tool_use/test_minimax_m2_tool_parser.py
@@ -0,0 +1,119 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.tool_parsers.minimax_m2_tool_parser import (
+    MinimaxM2ToolParser,
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -122,6 +122,8 @@ def _reset_streaming_state(self):
+        # Reset streamed args tracking
+        self.streamed_args_for_tool.clear()
@@ -421,9 +423,12 @@ def extract_tool_calls_streaming(
-                                "arguments": "{}",  # Placeholder, will be updated later
+                                "arguments": {},  # Placeholder, will be updated later
+                        # Initialize streamed_args_for_tool for this tool call
```

- Reviewed files:
  - tests: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31083 - Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs

- Link: https://github.com/vllm-project/vllm/pull/31083
- Status/date: merged / 2025-12-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `c02a2705f9ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +167/-48, 257 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types, touching `_extract_name, _convert_param_value, _extract_types_from_schema`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:
-        """Convert parameter value to the correct type."""
+        """Convert parameter value to the correct type (legacy single-type version)."""
+        return self._convert_param_value_with_types(value, [param_type])
+    def _extract_types_from_schema(self, schema: Any) -> list[str]:
+        """
+        Extract all possible types from a JSON schema definition.
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/minimax_m2_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31493 - Optimize QKNorm for MiniMax-M2/M2.1

- Link: https://github.com/vllm-project/vllm/pull/31493
- Status/date: merged / 2025-12-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `5bc664110f12`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize QKNorm for MiniMax-M2/M2.1"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "Optimize QKNorm for MiniMax-M2/M2.1"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -234,8 +234,9 @@ def forward(
-        q = self.q_norm(q)
-        k = self.k_norm(k)
+        q, k = MiniMaxText01RMSNormTP.forward_qk(
+            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32342 - Fix optional parameter parsing in MiniMax M2 tool parser #32278

- Link: https://github.com/vllm-project/vllm/pull/32342
- Status/date: merged / 2026-01-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `19b251fe3d26`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-5, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix optional parameter parsing in MiniMax M2 tool parser #32278"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "Fix optional parameter parsing in MiniMax M2 tool parser #32278"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types, touching `_convert_param_value_with_types`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -217,16 +217,13 @@ def _convert_param_value_with_types(
-        if value.lower() == "null":
+        # Check if the VALUE itself indicates null (not just if null is allowed)
+        if value.lower() in ("null", "none", "nil"):
-        # Try null first if it's in the list
-        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
-            return None
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/minimax_m2_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32763 - feat: Complete LoRA support for MiniMaxM2 Fixes #32736

- Link: https://github.com/vllm-project/vllm/pull/32763
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `bc0d291bfebf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Complete LoRA support for MiniMaxM2 Fixes #32736"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "feat: Complete LoRA support for MiniMaxM2 Fixes #32736"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__, touching `load_weights, MiniMaxM2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -59,7 +59,7 @@
-from .interfaces import SupportsPP
+from .interfaces import SupportsLoRA, SupportsPP
@@ -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-class MiniMaxM2ForCausalLM(nn.Module, SupportsPP):
+class MiniMaxM2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
+    packed_modules_mapping = {
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35352 - [Bug] Fix missing tag after tool call in MiniMax 2.1

- Link: https://github.com/vllm-project/vllm/pull/35352
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/minimax_m2_reasoning_parser.py`; associated commits `7fea7250a46c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix missing tag after tool call in MiniMax 2.1"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/reasoning/minimax_m2_reasoning_parser.py`; technical summary: Covers "[Bug] Fix missing tag after tool call in MiniMax 2.1"; the main implementation surface is `vllm/reasoning/minimax_m2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end`.
- Code diff details:
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids
- Key code excerpts:

```diff
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
+        self.start_token_id = self.vocab.get("<think>")
-        return any(input_id == end_token_id for input_id in reversed(input_ids))
+        start_token_id = self.start_token_id
+        for input_id in reversed(input_ids):
+            if input_id in (end_token_id, start_token_id):
+                return input_id == end_token_id
```

- Reviewed files:
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/minimax_m2_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35895 - [Bugfix] Fix minimax_m2 tool parser when stream interval > 1

- Link: https://github.com/vllm-project/vllm/pull/35895
- Status/date: merged / 2026-03-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `8647c6cf510b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +534/-532, 1119 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix minimax_m2 tool parser when stream interval > 1"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`; technical summary: Covers "[Bugfix] Fix minimax_m2 tool parser when stream interval > 1"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state, touching `__init__, type, _generate_tool_call_id`; `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser, touching `FakeTokenizer, __init__, get_vocab`; `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser, touching `FakeTokenizer, __init__, get_vocab`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser
  - `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):
-        self.invoke_start_prefix: str = "<invoke name="
-        self.invoke_end_token: str = "</invoke>"
-        self.parameter_prefix: str = "<parameter name="
-        self.parameter_end_token: str = "</parameter>"
-        # Streaming state variables
-        self.current_tool_name_sent: bool = False
diff -- tests/tool_parsers/test_minimax_m2_tool_parser.py
@@ -0,0 +1,444 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.tool_parsers.minimax_m2_tool_parser import (
+    MinimaxM2ToolParser,
diff -- tests/tool_use/test_minimax_m2_tool_parser.py
@@ -1,119 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413
  - tests: `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0; `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- Link: https://github.com/vllm-project/vllm/pull/37371
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +235/-219, 527 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/models/kimi_linear.py`; technical summary: Covers "standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01"; the main implementation surface is `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/models/kimi_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids, touching `replace_weight_name, __init__, _clear_prefill_cache`; `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids, touching `forward, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids
  - `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -52,7 +52,12 @@
-from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
+from .utils import (
+    AutoWeightsLoader,
+    PPMissingLayer,
+    is_pp_missing_parameter,
+    make_layers,
diff -- vllm/model_executor/models/kimi_linear.py
@@ -46,6 +46,7 @@
+    AutoWeightsLoader,
@@ -472,94 +473,7 @@ def forward(
-class KimiLinearForCausalLM(
-    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
-):
-    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131; `vllm/model_executor/models/kimi_linear.py` modified +97/-88
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37214 - Fix minimax m2.5 nvfp4 kv scales weight loading

- Link: https://github.com/vllm-project/vllm/pull/37214
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `74056039b776`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix minimax m2.5 nvfp4 kv scales weight loading"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "Fix minimax m2.5 nvfp4 kv scales weight loading"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                # Remap qkv_proj.[kv]_scale to attn.[kv]_scale
+                if name.endswith((".k_scale", ".v_scale")):
+                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
+                    if remapped_name is not None and remapped_name in params_dict:
+                        param = params_dict[remapped_name]
+                        weight_loader = getattr(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36965 - [Model][Quantization] Add GGUF support for MiniMax-M2.1

- Link: https://github.com/vllm-project/vllm/pull/36965
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `63babd17f1b1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +137/-10, 238 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Quantization] Add GGUF support for MiniMax-M2.1"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "[Model][Quantization] Add GGUF support for MiniMax-M2.1"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                quant_config=None,
+                quant_config=quant_config,
@@ -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                config.vocab_size, config.hidden_size, quant_config=None
+                config.vocab_size,
+                config.hidden_size,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/37512
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `f6983f01de2b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +24/-5, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "MiniMax-M2: add Eagle3 speculative decoding support"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "MiniMax-M2: add Eagle3 speculative decoding support"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__, touching `forward, MiniMaxM2Model, __init__`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -24,6 +24,7 @@
+from itertools import islice
@@ -59,7 +60,7 @@
-from .interfaces import SupportsLoRA, SupportsPP
+from .interfaces import EagleModelMixin, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -313,7 +314,7 @@ def forward(
-class MiniMaxM2Model(nn.Module):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +16/-5
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37045 - [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels

- Link: https://github.com/vllm-project/vllm/pull/37045
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/core/test_minimax_reduce_rms.py`, `vllm/model_executor/models/minimax_m2.py`; associated commits `ecd1ea13634e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +1861/-4, 1936 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Porting the TRTLLM minimax_allreduce_rms kernels"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/model_executor/models/minimax_m2.py`, `tests/kernels/core/test_minimax_reduce_rms.py`; technical summary: Covers "[Kernel] Porting the TRTLLM minimax_allreduce_rms kernels"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`, `tests/kernels/core/test_minimax_reduce_rms.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward, touching `forward`; `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk, touching `_worker_forward_qk, test_minimax_reduce_rms_qk`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward
  - `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -233,9 +233,7 @@ def forward(
-        q, k = MiniMaxText01RMSNormTP.forward_qk(
-            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
-        )
+        q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
diff -- tests/kernels/core/test_minimax_reduce_rms.py
@@ -0,0 +1,152 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for MiniMax QK RMS-norm: NCCL reference vs Lamport fused kernel."""
+import pytest
+import torch
+import torch.nn as nn
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-3
  - tests: `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_minimax_reduce_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39683 - [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter

- Link: https://github.com/vllm-project/vllm/pull/39683
- Status/date: merged / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: Fix MinimaxM2ToolParser missing tools parameter"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/parser/minimax_m2_parser.py`; technical summary: Covers "[Bugfix]: Fix MinimaxM2ToolParser missing tools parameter"; the main implementation surface is `vllm/parser/minimax_m2_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__, touching `MiniMaxM2Parser, __init__`.
- Code diff details:
  - `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- Key code excerpts:

```diff
diff -- vllm/parser/minimax_m2_parser.py
@@ -13,6 +13,9 @@
+from vllm.tool_parsers.abstract_tool_parser import (
+    Tool,
+)
@@ -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):
-    def __init__(self, tokenizer: TokenizerLike):
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
```

- Reviewed files:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/parser/minimax_m2_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39861 - [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()

- Link: https://github.com/vllm-project/vllm/pull/39861
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/parser/minimax_m2_parser.py`; technical summary: Covers "[Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()"; the main implementation surface is `vllm/parser/minimax_m2_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__, touching `MiniMaxM2Parser, __init__`.
- Code diff details:
  - `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- Key code excerpts:

```diff
diff -- vllm/parser/minimax_m2_parser.py
@@ -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):
-    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        super().__init__(tokenizer)
+    def __init__(
+        self,
+        tokenizer: TokenizerLike,
+        tools: list[Tool] | None = None,
```

- Reviewed files:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +9/-3
- Risk and verification: Runtime changes concentrate in `vllm/parser/minimax_m2_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38191 - [Bugfix] Fix k_norm weight sharding in MiniMaxM2Attention when total_num_kv_heads < tp_size

- Link: https://github.com/vllm-project/vllm/pull/38191
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `f8ac0c7cf0e3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +44/-12, 97 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix k_norm weight sharding in MiniMaxM2Attention when total_num_kv_heads < tp_size"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "[Bugfix] Fix k_norm weight sharding in MiniMaxM2Attention when total_num_kv_heads < tp_size"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +16/-3 (19 lines); hunks: -35,6 +35,7; -220,9 +221,21 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-3 (19 lines); hunks: -35,6 +35,7; -220,9 +221,21 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -35,6 +35,7 @@
+    get_tensor_model_parallel_rank,
@@ -220,9 +221,21 @@ def __init__(
-        self.k_norm = MiniMaxText01RMSNormTP(
-            self.head_dim * self.total_num_kv_heads, eps=rms_norm_eps
-        )
+        if self.total_num_kv_heads >= tp_size:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +16/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39599 - fix(tool-parser): preserve "none"/"nil" strings as valid enum values in minimax_m2

- Link: https://github.com/vllm-project/vllm/pull/39599
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `63cc8a55a97f`
- Diff scope read: GitHub Pull Request files API returned 2 files, +137/-6, 166 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(tool-parser): preserve "none"/"nil" strings as valid enum values in minimax_m2"; model line: MiniMax M2 Series; category: bug fix; main diff: `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "fix(tool-parser): preserve "none"/"nil" strings as valid enum values in minimax_m2"; the main implementation surface is `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +130/-0 (130 lines); hunks: -548,3 +548,133 @@ def test_anyof_nullable_param_object_value(self):; symbols: test_anyof_nullable_param_object_value, TestNoneStringPreservation, test_none_string_preserved_in_enum, test_none_string_preserved_plain_string, touching `test_anyof_nullable_param_object_value, TestNoneStringPreservation, test_none_string_preserved_in_enum`; `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +7/-6 (13 lines); hunks: -160,16 +160,13 @@ def _convert_param_value_with_types(; -187,7 +184,11 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types, touching `_convert_param_value_with_types`.
- Code diff details:
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +130/-0 (130 lines); hunks: -548,3 +548,133 @@ def test_anyof_nullable_param_object_value(self):; symbols: test_anyof_nullable_param_object_value, TestNoneStringPreservation, test_none_string_preserved_in_enum, test_none_string_preserved_plain_string
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +7/-6 (13 lines); hunks: -160,16 +160,13 @@ def _convert_param_value_with_types(; -187,7 +184,11 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_minimax_m2_tool_parser.py
@@ -548,3 +548,133 @@ def test_anyof_nullable_param_object_value(self):
+class TestNoneStringPreservation:
+    """Regression tests for #39567: 'none' as a string must not become None."""
+    def test_none_string_preserved_in_enum(self):
+        """'none' in an enum must stay as the string 'none', not Python None."""
+        tools = [
+            ChatCompletionToolsParam(
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -160,16 +160,13 @@ def _convert_param_value_with_types(
-        # Check if the VALUE itself indicates null (not just if null is allowed)
-        if value.lower() in ("null", "none", "nil"):
-            return None
-        # Priority: integer > number > boolean > object > array > string
+        # Priority: null > integer > number > boolean > object > array > string
+            "null",
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +130/-0
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +7/-6
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42497 - [Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2)

- Link: https://github.com/vllm-project/vllm/pull/42497
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-12, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2)"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`; technical summary: Covers "[Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2)"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +31/-12 (43 lines); hunks: -31,6 +31,9; -283,20 +286,36 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +31/-12 (43 lines); hunks: -31,6 +31,9; -283,20 +286,36 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/triton_moe.py
@@ -31,6 +31,9 @@
+from vllm.model_executor.layers.quantization.utils.fp8_utils import (
+    is_deep_gemm_e8m0_used,
+)
@@ -283,20 +286,36 @@ def apply(
-        self.activation(
-            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` modified +31/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43006 - [Refactor] Extract shared coerce_to_schema_type utility from Minimax M2 tool parser

- Link: https://github.com/vllm-project/vllm/pull/43006
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `57fef4e0bf0b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +247/-77, 353 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Extract shared coerce_to_schema_type utility from Minimax M2 tool parser"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "[Refactor] Extract shared coerce_to_schema_type utility from Minimax M2 tool parser"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-77 (79 lines); hunks: -25,6 +25,7; -146,80 +147,6 @@ def _extract_types_from_schema(self, schema: Any) -> list[s...; symbols: _extract_types_from_schema, _convert_param_value_with_types, _get_param_types_from_config, _parse_single_invoke, touching `_extract_types_from_schema, _convert_param_value_with_types, _get_param_types_from_config`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-77 (79 lines); hunks: -25,6 +25,7; -146,80 +147,6 @@ def _extract_types_from_schema(self, schema: Any) -> list[s...; symbols: _extract_types_from_schema, _convert_param_value_with_types, _get_param_types_from_config, _parse_single_invoke
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -25,6 +25,7 @@
+from vllm.tool_parsers.utils import coerce_to_schema_type
@@ -146,80 +147,6 @@ def _extract_types_from_schema(self, schema: Any) -> list[str]:
-    def _convert_param_value_with_types(
-        self, value: str, param_types: list[str]
-    ) -> Any:
-        """
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-77
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43025 - [Refactor] Extract extract_types_from_schema utility from Minimax M2 tool parser

- Link: https://github.com/vllm-project/vllm/pull/43025
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/minimax_m2_tool_parser.py`; associated commits `42b4f1fdf726`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +123/-106, 282 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Extract extract_types_from_schema utility from Minimax M2 tool parser"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`; technical summary: Covers "[Refactor] Extract extract_types_from_schema utility from Minimax M2 tool parser"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +10/-105 (115 lines); hunks: -4,7 +4,6; -25,7 +24,11; symbols: _extract_name, _extract_types_from_schema, _get_param_types_from_config, _parse_single_invoke, touching `_extract_name, _extract_types_from_schema, _get_param_types_from_config`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +10/-105 (115 lines); hunks: -4,7 +4,6; -25,7 +24,11; symbols: _extract_name, _extract_types_from_schema, _get_param_types_from_config, _parse_single_invoke
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -4,7 +4,6 @@
-from typing import Any
@@ -25,7 +24,11 @@
-from vllm.tool_parsers.utils import coerce_to_schema_type
+from vllm.tool_parsers.utils import (
+    coerce_to_schema_type,
+    extract_types_from_schema,
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +10/-105
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43410 - [Kernel] Porting fuse_minimax_qk_norm to manual fusion

- Link: https://github.com/vllm-project/vllm/pull/43410
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/core/test_minimax_reduce_rms.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`, `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py`, `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/models/minimax_m2.py`; associated commits `6e503868caa4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +262/-490, 893 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Porting fuse_minimax_qk_norm to manual fusion"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`, `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "[Kernel] Porting fuse_minimax_qk_norm to manual fusion"; the main implementation surface is `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `vllm/model_executor/layers/minimax_rms_norm/__init__.py`, `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake, MiniMaxText01RMSNormTP, touching `_minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake`; `vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10; `vllm/model_executor/models/minimax_m2.py` modified +4/-3 (7 lines); hunks: -50,7 +50,7; -243,8 +243,9 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: _minimax_qk_norm_fallback, _minimax_qk_norm_fusion, _minimax_qk_norm_fusion_fake, MiniMaxText01RMSNormTP
  - `vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `vllm/model_executor/models/minimax_m2.py` modified +4/-3 (7 lines); hunks: -50,7 +50,7; -243,8 +243,9 @@ def forward(; symbols: forward
  - `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` renamed +0/-0 (0 lines)
  - `tests/kernels/core/test_minimax_reduce_rms.py` modified +2/-2 (4 lines); hunks: -10,7 +10,7; -59,7 +59,7 @@ def _worker_forward_qk(; symbols: _worker_forward_qk
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py
@@ -0,0 +1,234 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from functools import partial
+import torch
+from torch import nn
+from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
diff -- vllm/model_executor/layers/minimax_rms_norm/__init__.py
@@ -0,0 +1,10 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
+    MiniMaxText01RMSNormTP,
+)
+__all__ = [
diff -- vllm/model_executor/models/minimax_m2.py
@@ -50,7 +50,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` added +234/-0; `vllm/model_executor/layers/minimax_rms_norm/__init__.py` added +10/-0; `vllm/model_executor/models/minimax_m2.py` modified +4/-3; `vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py` renamed +0/-0
  - tests: `tests/kernels/core/test_minimax_reduce_rms.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_minimax_reduce_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38445 - [PERF]MiniMax-M2 gate kernel

- Link: https://github.com/vllm-project/vllm/pull/38445
- Status/date: merged / 2026-05-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/minimax_m2.py`; associated commits `559d6710bf45`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +716/-23, 871 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PERF]MiniMax-M2 gate kernel"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/model_executor/models/minimax_m2.py`; technical summary: Covers "[PERF]MiniMax-M2 gate kernel"; the main implementation surface is `vllm/model_executor/models/minimax_m2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/minimax_m2.py` modified +4/-4 (8 lines); hunks: -43,10 +43,10; -113,12 +113,12 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +4/-4 (8 lines); hunks: -43,10 +43,10; -113,12 +113,12 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -43,10 +43,10 @@
+from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
-    ReplicatedLinear,
@@ -113,12 +113,12 @@ def __init__(
-        self.gate = ReplicatedLinear(
+        self.gate = GateLinear(
-            quant_config=None,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/test_fp32_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44279 - [Refactor] Remove dead code from parser infrastructure

- Link: https://github.com/vllm-project/vllm/pull/44279
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +35/-328, 459 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Remove dead code from parser infrastructure"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/parser/parser_manager.py`, `vllm/parser/minimax_m2_parser.py`, `vllm/parser/abstract_parser.py`; technical summary: Covers "[Refactor] Remove dead code from parser infrastructure"; the main implementation surface is `vllm/parser/parser_manager.py`, `vllm/parser/minimax_m2_parser.py`, `vllm/parser/abstract_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/parser/parser_manager.py` modified +14/-204 (218 lines); hunks: -3,14 +3,9; -22,170 +17,10; symbols: ParserManager, get_parser_internal, _load_lazy_parser, _register_module, touching `ParserManager, get_parser_internal, _load_lazy_parser`; `vllm/parser/minimax_m2_parser.py` removed +0/-61 (61 lines); hunks: -1,61 +0,0; symbols: MiniMaxM2Parser, __init__, touching `MiniMaxM2Parser, __init__`; `vllm/parser/abstract_parser.py` modified +15/-41 (56 lines); hunks: -37,12 +37,11; -90,19 +89,25 @@ class Parser:; symbols: Parser, __init__, vocab, parse_delta, touching `Parser, __init__, vocab`; `vllm/parser/__init__.py` modified +0/-18 (18 lines); hunks: -4,29 +4,11; symbols: register_lazy_parsers, touching `register_lazy_parsers`.
- Code diff details:
  - `vllm/parser/parser_manager.py` modified +14/-204 (218 lines); hunks: -3,14 +3,9; -22,170 +17,10; symbols: ParserManager, get_parser_internal, _load_lazy_parser, _register_module
  - `vllm/parser/minimax_m2_parser.py` removed +0/-61 (61 lines); hunks: -1,61 +0,0; symbols: MiniMaxM2Parser, __init__
  - `vllm/parser/abstract_parser.py` modified +15/-41 (56 lines); hunks: -37,12 +37,11; -90,19 +89,25 @@ class Parser:; symbols: Parser, __init__, vocab, parse_delta
  - `vllm/parser/__init__.py` modified +0/-18 (18 lines); hunks: -4,29 +4,11; symbols: register_lazy_parsers
  - `tests/parser/test_streaming.py` modified +6/-4 (10 lines); hunks: -7,7 +7,7; -45,9 +45,11 @@ def request_obj():; symbols: request_obj, make_parser, TestParser, stream_text
- Key code excerpts:

```diff
diff -- vllm/parser/parser_manager.py
@@ -3,14 +3,9 @@
-import importlib
-import os
-from collections.abc import Callable
-from vllm.utils.collection_utils import is_list_of
-from vllm.utils.import_utils import import_from_path
@@ -22,170 +17,10 @@
diff -- vllm/parser/minimax_m2_parser.py
@@ -1,61 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""
-MiniMax M2 Parser - A unified parser for MiniMax M2 models.
-This parser combines the existing MiniMaxM2ReasoningParser and
-MinimaxM2ToolParser into a single unified interface by delegating
diff -- vllm/parser/abstract_parser.py
@@ -37,12 +37,11 @@
```

- Reviewed files:
  - runtime: `vllm/parser/parser_manager.py` modified +14/-204; `vllm/parser/minimax_m2_parser.py` removed +0/-61; `vllm/parser/abstract_parser.py` modified +15/-41; `vllm/parser/__init__.py` modified +0/-18
  - tests: `tests/parser/test_streaming.py` modified +6/-4
- Risk and verification: The diff ships test coverage in `tests/parser/test_streaming.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43556 - [Attention] Mamba attention module refactor - LINEAR

- Link: https://github.com/vllm-project/vllm/pull/43556
- Status/date: merged / 2026-06-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +505/-551, 1309 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Mamba attention module refactor - LINEAR"; model line: MiniMax M2 Series; category: model implementation change; main diff: `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py`; technical summary: Covers "[Attention] Mamba attention module refactor - LINEAR"; the main implementation surface is `vllm/model_executor/models/bailing_moe_linear.py`, `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`, `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439 (452 lines); hunks: -9,19 +9,14; -30,25 +25,19; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, forward, touching `is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention`; `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: _build_rope_parameters, BailingGroupRMSNormGate, __init__, _weight_loader, touching `_build_rope_parameters, BailingGroupRMSNormGate, __init__`; `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68 (86 lines); hunks: -7,30 +7,20; -157,79 +147,39 @@ def jit_linear_forward_prefix(; symbols: clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, touching `clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention`; `vllm/model_executor/layers/mamba/linear/base.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: LinearAttention, for, __init__, mamba_type, touching `LinearAttention, for, __init__`.
- Code diff details:
  - `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439 (452 lines); hunks: -9,19 +9,14; -30,25 +25,19; symbols: is_linear_layer, _build_rope_parameters, BailingMoeV25MLAAttention, forward
  - `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: _build_rope_parameters, BailingGroupRMSNormGate, __init__, _weight_loader
  - `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68 (86 lines); hunks: -7,30 +7,20; -157,79 +147,39 @@ def jit_linear_forward_prefix(; symbols: clear_linear_attention_cache_for_new_sequences, jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type
  - `vllm/model_executor/layers/mamba/linear/base.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: LinearAttention, for, __init__, mamba_type
  - `vllm/model_executor/models/minimax_text_01.py` modified +14/-35 (49 lines); hunks: -15,7 +15,7; -35,7 +35,9; symbols: MiniMaxText01DecoderLayer, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/bailing_moe_linear.py
@@ -9,19 +9,14 @@
-from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
+from vllm.config import CacheConfig, VllmConfig
-from vllm.model_executor.custom_op import PluggableLayer
-from vllm.model_executor.layers.fla.ops.layernorm_guard import (
-    RMSNormGated,
-    layernorm_fn,
diff -- vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py
@@ -0,0 +1,384 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
+import torch
+import torch.nn.functional as F
+from transformers.configuration_utils import PretrainedConfig
diff -- vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py
@@ -7,30 +7,20 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/bailing_moe_linear.py` modified +13/-439; `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` added +384/-0; `vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py` renamed +18/-68; `vllm/model_executor/layers/mamba/linear/base.py` added +66/-0; `vllm/model_executor/models/minimax_text_01.py` modified +14/-35; `vllm/model_executor/layers/mamba/linear/__init__.py` added +0/-0
  - tests: `tests/v1/attention/test_attention_backends_selection.py` modified +10/-9
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_attention_backends_selection.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44983 - [Bugfix] Fix minimax_qk_norm_fusion

- Link: https://github.com/vllm-project/vllm/pull/44983
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`; associated commits `dc10e467a985`
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-9, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix minimax_qk_norm_fusion"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`; technical summary: Covers "[Bugfix] Fix minimax_qk_norm_fusion"; the main implementation surface is `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +41/-9 (50 lines); hunks: -12,17 +12,34; -42,7 +59,7 @@ def _minimax_qk_norm_fallback(; symbols: _all_reduce_variance, _minimax_qk_norm_fallback, __init__, weight_loader, touching `_all_reduce_variance, _minimax_qk_norm_fallback, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +41/-9 (50 lines); hunks: -12,17 +12,34; -42,7 +59,7 @@ def _minimax_qk_norm_fallback(; symbols: _all_reduce_variance, _minimax_qk_norm_fallback, __init__, weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py
@@ -12,17 +12,34 @@
+from vllm.logger import init_logger
+logger = init_logger(__name__)
+def _all_reduce_variance(var: torch.Tensor) -> torch.Tensor:
+    """All-reduce a per-token variance tensor across the TP group.
+    Variance is accumulated in fp32 for numerical stability. The FlashInfer
+    fused all-reduce caches a single global workspace keyed to the model's
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +41/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45003 - [Frontend] Support strict mode for tool calling

- Link: https://github.com/vllm-project/vllm/pull/45003
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +672/-1936, 3162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support strict mode for tool calling"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`; technical summary: Covers "[Frontend] Support strict mode for tool calling"; the main implementation surface is `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #45381 - [Model] Add MiniMax M3 support

- Link: https://github.com/vllm-project/vllm/pull/45381
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/attention/test_minimax_m3.py`, `tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py`, `tests/kernels/test_minimax_m3_amd_ops.py`, `tests/models/multimodal/processing/test_minimax_m3.py`, `tests/reasoning/test_minimax_m3_reasoning_parser.py` and 30 files; associated commits `0a1c5034f5e4`
- Diff scope read: GitHub Pull Request files API returned 108 files, +14746/-323, 16807 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MiniMax M3 support"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py`; technical summary: Covers "[Model] Add MiniMax M3 support"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`, `vllm/models/minimax_m3/common/ops/index_topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` added +1216/-0 (1216 lines); hunks: -0,0 +1,1216; symbols: _sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb, MiniMAXGemmaRMSNorm, touching `_sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb`; `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0 (1177 lines); hunks: -0,0 +1,1177; symbols: _sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm, __init__, touching `_sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm`; `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0 (898 lines); hunks: -0,0 +1,898; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel, touching `_compare_and_swap, _bitonic_merge, _index_block_score_kernel`; `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: MiniMaxVLPatchEmbed, __init__, forward, MiniMaxVLAttention, touching `MiniMaxVLPatchEmbed, __init__, forward`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` added +1216/-0 (1216 lines); hunks: -0,0 +1,1216; symbols: _sparse_attention_layer_ids, _is_moe_layer, _build_rotary_emb, MiniMAXGemmaRMSNorm
  - `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0 (1177 lines); hunks: -0,0 +1,1177; symbols: _sparse_attention_layer_ids, _is_moe_layer, MiniMAXGemmaRMSNorm, __init__
  - `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0 (898 lines); hunks: -0,0 +1,898; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel
  - `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: MiniMaxVLPatchEmbed, __init__, forward, MiniMaxVLAttention
  - `vllm/transformers_utils/processors/minimax_m3.py` added +736/-0 (736 lines); hunks: -0,0 +1,736; symbols: round_by_factor, ceil_by_factor, floor_by_factor, _smart_resize_by_long_side
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -0,0 +1,1216 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only MiniMax M3 (text backbone) model — AMD ROCm implementation.
+Self-contained per-platform impl (mirrors ``deepseek_v4/amd``). It is identical
+to ``../nvidia/model.py`` except for RMS normalization: FlashInfer's Gemma
+RMSNorm kernels are CUDA-only, so ``MiniMAXGemmaRMSNorm`` here uses a native
diff -- vllm/models/minimax_m3/nvidia/model.py
@@ -0,0 +1,1177 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only MiniMax M3 (text backbone) model.
+The MiniMax-M3-preview config selects a single set of branches:
+    * qk_norm_type == "per_head"
+    * hidden_act == "swigluoai"
diff -- vllm/models/minimax_m3/common/ops/index_topk.py
@@ -0,0 +1,898 @@
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` added +1216/-0; `vllm/models/minimax_m3/nvidia/model.py` added +1177/-0; `vllm/models/minimax_m3/common/ops/index_topk.py` added +898/-0; `vllm/models/minimax_m3/common/vision_tower.py` added +765/-0; `vllm/transformers_utils/processors/minimax_m3.py` added +736/-0; `vllm/models/minimax_m3/common/ops/sparse_attn.py` added +593/-0
- Risk and verification: The diff ships test coverage in `requirements/test/cuda.txt`, `requirements/test/rocm.txt`, `requirements/test/xpu.txt`, `tests/kernels/attention/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45701 - [Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser

- Link: https://github.com/vllm-project/vllm/pull/45701
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_minimax_m2.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/parser/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` and 6 files; associated commits `f00e163f3562`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +588/-481, 1318 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; technical summary: Covers "[Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser"; the main implementation surface is `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-278 (280 lines); hunks: -1,284 +1,8; symbols: MinimaxM2ToolParser, __init__, _generate_tool_call_id, _extract_name, touching `MinimaxM2ToolParser, __init__, _generate_tool_call_id`; `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +6/-117 (123 lines); hunks: -18,7 +18,6; -34,6 +33,10 @@ def __init__(self):; symbols: FakeTokenizer, __init__, get_vocab, decode, touching `FakeTokenizer, __init__, get_vocab`; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +2/-51 (53 lines); hunks: -8,8 +8,8; -19,7 +19,7; symbols: MiniMaxM2ReasoningParser, start_token, end_token, touching `MiniMaxM2ReasoningParser, start_token, end_token`; `tests/reasoning/test_minimax_m2_reasoning_parser.py` modified +0/-26 (26 lines); hunks: -59,14 +59,6 @@ def minimax_m2_tokenizer():; -75,14 +67,6 @@ def minimax_m2_tokenizer():; symbols: minimax_m2_tokenizer, touching `minimax_m2_tokenizer`.
- Code diff details:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-278 (280 lines); hunks: -1,284 +1,8; symbols: MinimaxM2ToolParser, __init__, _generate_tool_call_id, _extract_name
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +6/-117 (123 lines); hunks: -18,7 +18,6; -34,6 +33,10 @@ def __init__(self):; symbols: FakeTokenizer, __init__, get_vocab, decode
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +2/-51 (53 lines); hunks: -8,8 +8,8; -19,7 +19,7; symbols: MiniMaxM2ReasoningParser, start_token, end_token
  - `tests/reasoning/test_minimax_m2_reasoning_parser.py` modified +0/-26 (26 lines); hunks: -59,14 +59,6 @@ def minimax_m2_tokenizer():; -75,14 +67,6 @@ def minimax_m2_tokenizer():; symbols: minimax_m2_tokenizer
  - `tests/parser/engine/test_minimax_m2.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: mock_tokenizer, parser, make_tools, TestNonStreaming
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/minimax_m2_tool_parser.py
@@ -1,284 +1,8 @@
-import json
-import uuid
-from collections.abc import Sequence
+from vllm.parser.engine.registered_adapters import MinimaxM2ParserToolAdapter
-import regex as re
-from vllm.entrypoints.openai.chat_completion.protocol import (
diff -- tests/tool_parsers/test_minimax_m2_tool_parser.py
@@ -18,7 +18,6 @@
-EOS_ID = 99
@@ -34,6 +33,10 @@ def __init__(self):
+    def decode(self, token_ids):
+        id_to_token = {v: k for k, v in self.vocab.items()}
+        return "".join(id_to_token.get(token_id, "") for token_id in token_ids)
@@ -121,7 +124,6 @@ def test_plain_content(self, parser):
diff -- vllm/reasoning/minimax_m2_reasoning_parser.py
@@ -8,8 +8,8 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-278; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +2/-51; `vllm/parser/minimax_m2.py` added +180/-0
  - tests: `tests/tool_parsers/test_minimax_m2_tool_parser.py` modified +6/-117; `tests/reasoning/test_minimax_m2_reasoning_parser.py` modified +0/-26; `tests/parser/engine/test_minimax_m2.py` added +264/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_minimax_m2.py`, `tests/parser/engine/trace_builder.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45720 - [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype

- Link: https://github.com/vllm-project/vllm/pull/45720
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/attention/test_minimax_m3.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/sparse_attention.py`; associated commits `efd15e192a1a`
- Diff scope read: GitHub Pull Request files API returned 3 files, +60/-5, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/models/minimax_m3/common/sparse_attention.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`, `tests/kernels/attention/test_minimax_m3.py`; technical summary: Covers "[Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype"; the main implementation surface is `vllm/models/minimax_m3/common/sparse_attention.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`, `tests/kernels/attention/test_minimax_m3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/common/sparse_attention.py` modified +8/-3 (11 lines); hunks: -291,9 +291,14 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +8/-2 (10 lines); hunks: -24,6 +24,12; -456,7 +462,7 @@ def minimax_m3_sparse_attn(; symbols: minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode, touching `minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode`; `tests/kernels/attention/test_minimax_m3.py` modified +44/-0 (44 lines); hunks: -15,11 +15,13; -79,6 +81,48 @@ def _allocate_main_kv_via_contract(; symbols: _allocate_main_kv_via_contract, test_sparse_impl_uses_platform_fp8_dtype, test_sparse_kernels_recognize_fp8_dtypes, _reference_index_topk, touching `_allocate_main_kv_via_contract, test_sparse_impl_uses_platform_fp8_dtype, test_sparse_kernels_recognize_fp8_dtypes`.
- Code diff details:
  - `vllm/models/minimax_m3/common/sparse_attention.py` modified +8/-3 (11 lines); hunks: -291,9 +291,14 @@ def __init__(; symbols: __init__
  - `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +8/-2 (10 lines); hunks: -24,6 +24,12; -456,7 +462,7 @@ def minimax_m3_sparse_attn(; symbols: minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode
  - `tests/kernels/attention/test_minimax_m3.py` modified +44/-0 (44 lines); hunks: -15,11 +15,13; -79,6 +81,48 @@ def _allocate_main_kv_via_contract(; symbols: _allocate_main_kv_via_contract, test_sparse_impl_uses_platform_fp8_dtype, test_sparse_kernels_recognize_fp8_dtypes, _reference_index_topk
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/common/sparse_attention.py
@@ -291,9 +291,14 @@ def __init__(
-        self.kv_cache_fp8_dtype = (
-            torch.float8_e5m2 if "e5m2" in kv_cache_dtype else torch.float8_e4m3fn
-        )
+        if "e5m2" in kv_cache_dtype:
+            self.kv_cache_fp8_dtype = (
+                torch.float8_e5m2fnuz
diff -- vllm/models/minimax_m3/common/ops/sparse_attn.py
@@ -24,6 +24,12 @@
+_FP8_DTYPES = (
+    torch.float8_e4m3fn,
+    torch.float8_e4m3fnuz,
+    torch.float8_e5m2,
+    torch.float8_e5m2fnuz,
+)
diff -- tests/kernels/attention/test_minimax_m3.py
@@ -15,11 +15,13 @@
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/common/sparse_attention.py` modified +8/-3; `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +8/-2
  - tests: `tests/kernels/attention/test_minimax_m3.py` modified +44/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_minimax_m3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45896 - [feature] MiniMax-M3-MXFP4 support added

- Link: https://github.com/vllm-project/vllm/pull/45896
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/minimax_m3/amd/model.py`; associated commits `d112eb1ac78e`
- Diff scope read: GitHub Pull Request files API returned 4 files, +44/-3, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] MiniMax-M3-MXFP4 support added"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/models/minimax_m3/amd/model.py`; technical summary: Covers "[feature] MiniMax-M3-MXFP4 support added"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` modified +12/-2 (14 lines); hunks: -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(; symbols: load_weights, MiniMaxM3SparseForCausalLM, __init__, MiniMaxM3SparseForConditionalGeneration, touching `load_weights, MiniMaxM3SparseForCausalLM, __init__`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` modified +12/-2 (14 lines); hunks: -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(; symbols: load_weights, MiniMaxM3SparseForCausalLM, __init__, MiniMaxM3SparseForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -924,6 +924,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    packed_modules_mapping = {
+        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
@@ -987,14 +992,19 @@ class MiniMaxM3SparseForConditionalGeneration(
+    packed_modules_mapping = {
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45988 - [Perf] Remove unused loggers in `reasoning/`

- Link: https://github.com/vllm-project/vllm/pull/45988
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +0/-27, 148 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Remove unused loggers in `reasoning/`"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`; technical summary: Covers "[Perf] Remove unused loggers in `reasoning/`"; the main implementation surface is `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -6,7 +6,6; -17,8 +16,6; symbols: DeepSeekV3ReasoningParser, touching `DeepSeekV3ReasoningParser`; `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: Ernie45ReasoningParser, touching `Ernie45ReasoningParser`; `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: GraniteReasoningParser, touching `GraniteReasoningParser`; `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: HunyuanA13BReasoningParser, touching `HunyuanA13BReasoningParser`.
- Code diff details:
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -6,7 +6,6; -17,8 +16,6; symbols: DeepSeekV3ReasoningParser
  - `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: Ernie45ReasoningParser
  - `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: GraniteReasoningParser
  - `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: HunyuanA13BReasoningParser
  - `vllm/reasoning/identity_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: IdentityReasoningParser
- Key code excerpts:

```diff
diff -- vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -6,7 +6,6 @@
-from vllm.logger import init_logger
@@ -17,8 +16,6 @@
-logger = init_logger(__name__)
diff -- vllm/reasoning/ernie45_reasoning_parser.py
@@ -7,15 +7,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/granite_reasoning_parser.py
@@ -8,15 +8,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/hunyuan_a13b_reasoning_parser.py
@@ -8,15 +8,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/identity_reasoning_parser.py
```

- Reviewed files:
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3; `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3; `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3; `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3; `vllm/reasoning/identity_reasoning_parser.py` modified +0/-3; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45935 - [Model]Fix MiniMaxM2ForCausalLM perf regression

- Link: https://github.com/vllm-project/vllm/pull/45935
- Status/date: merged / 2026-06-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/core/test_minimax_reduce_rms.py`, `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`; associated commits `745bba5ea8fa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +215/-16, 305 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model]Fix MiniMaxM2ForCausalLM perf regression"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `tests/kernels/core/test_minimax_reduce_rms.py`; technical summary: Covers "[Model]Fix MiniMaxM2ForCausalLM perf regression"; the main implementation surface is `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`, `tests/kernels/core/test_minimax_reduce_rms.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +156/-13 (169 lines); hunks: -14,7 +14,7; -40,32 +40,171 @@ def _all_reduce_variance(var: torch.Tensor) -> torch.Tensor:; symbols: _all_reduce_variance, _minimax_qk_norm_fallback, _minimax_qk_var_kernel, _minimax_rms_apply_kernel, touching `_all_reduce_variance, _minimax_qk_norm_fallback, _minimax_qk_var_kernel`; `tests/kernels/core/test_minimax_reduce_rms.py` modified +59/-3 (62 lines); hunks: -10,8 +10,12; -54,8 +58,19 @@ def _worker_forward_qk(; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk, test_minimax_qk_norm_triton_fallback, touching `_worker_forward_qk, test_minimax_reduce_rms_qk, test_minimax_qk_norm_triton_fallback`.
- Code diff details:
  - `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +156/-13 (169 lines); hunks: -14,7 +14,7; -40,32 +40,171 @@ def _all_reduce_variance(var: torch.Tensor) -> torch.Tensor:; symbols: _all_reduce_variance, _minimax_qk_norm_fallback, _minimax_qk_var_kernel, _minimax_rms_apply_kernel
  - `tests/kernels/core/test_minimax_reduce_rms.py` modified +59/-3 (62 lines); hunks: -10,8 +10,12; -54,8 +58,19 @@ def _worker_forward_qk(; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk, test_minimax_qk_norm_triton_fallback
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py
@@ -14,7 +14,7 @@
-from vllm.platforms import current_platform
+from vllm.triton_utils import HAS_TRITON, tl, triton
@@ -40,32 +40,171 @@ def _all_reduce_variance(var: torch.Tensor) -> torch.Tensor:
-@torch.compile(backend=current_platform.simple_compile_backend, dynamic=True)
-def _minimax_qk_norm_fallback(
+@triton.jit
diff -- tests/kernels/core/test_minimax_reduce_rms.py
@@ -10,8 +10,12 @@
-from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
+from vllm.model_executor.layers.minimax_rms_norm import (
+    MiniMaxText01RMSNormTP,
+    rms_norm_tp,
+)
+from vllm.triton_utils import HAS_TRITON
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py` modified +156/-13
  - tests: `tests/kernels/core/test_minimax_reduce_rms.py` modified +59/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/core/test_minimax_reduce_rms.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45993 - [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/45993
- Status/date: merged / 2026-06-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +10/-3881, 4048 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM"; model line: MiniMax M2 Series; category: docs/tests/CI; main diff: `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py`; technical summary: Covers "[Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM"; the main implementation surface is `tests/tool_parsers/test_minimax_tool_parser.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/tool_parsers/minimax_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227 (1227 lines); hunks: -1,1227 +0,0; symbols: minimax_tokenizer, minimax_tool_parser, sample_tools, assert_tool_calls, touching `minimax_tokenizer, minimax_tool_parser, sample_tools`; `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000 (1000 lines); hunks: -1,1000 +0,0; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func, touching `replace_weight_name, weight_loader_with_alias, wrapper`; `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852 (852 lines); hunks: -1,852 +0,0; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think, touching `MinimaxToolParser, __init__, preprocess_model_output`; `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385 (385 lines); hunks: -1,385 +0,0; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector, __init__, touching `MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector`.
- Code diff details:
  - `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227 (1227 lines); hunks: -1,1227 +0,0; symbols: minimax_tokenizer, minimax_tool_parser, sample_tools, assert_tool_calls
  - `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000 (1000 lines); hunks: -1,1000 +0,0; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
  - `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852 (852 lines); hunks: -1,852 +0,0; symbols: MinimaxToolParser, __init__, preprocess_model_output, remove_tool_calls_from_think
  - `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385 (385 lines); hunks: -1,385 +0,0; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, MiniMaxVL01MultiModalProjector, __init__
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` removed +0/-113 (113 lines); hunks: -1,113 +0,0; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_minimax_tool_parser.py
@@ -1,1227 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# ruff: noqa: E501
-import json
-from typing import Any
-import pytest
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,1000 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""Inference-only MiniMaxText01 model."""
-from collections.abc import Iterable
-from itertools import islice
-from typing import TYPE_CHECKING
diff -- vllm/tool_parsers/minimax_tool_parser.py
@@ -1,852 +0,0 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_minimax_tool_parser.py` removed +0/-1227; `tests/models/multimodal/processing/test_minimax_vl_01.py` removed +0/-113; `tests/models/multimodal/generation/test_common.py` modified +0/-23; `tests/models/multimodal/generation/vlm_utils/model_utils.py` modified +0/-18
  - runtime: `vllm/model_executor/models/minimax_text_01.py` removed +0/-1000; `vllm/tool_parsers/minimax_tool_parser.py` removed +0/-852; `vllm/model_executor/models/minimax_vl_01.py` removed +0/-385
  - docs: `examples/generate/multimodal/vision_language_offline.py` modified +0/-34
- Risk and verification: The diff ships test coverage in `rust/src/chat/tests/templates/vllm_examples/tool_chat_template_minimax_m1.jinja`, `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/generation/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45892 - [Minimax-M3] BF16/FP8 Indexer using MSA

- Link: https://github.com/vllm-project/vllm/pull/45892
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/attention/test_minimax_m3.py`, `tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py`, `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/common/indexer.py`, `vllm/models/minimax_m3/common/ops/index_topk.py` and 9 files; associated commits `6691f087a65b`
- Diff scope read: GitHub Pull Request files API returned 13 files, +1048/-104, 1712 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Minimax-M3] BF16/FP8 Indexer using MSA"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/models/minimax_m3/nvidia/indexer_msa.py`, `vllm/models/minimax_m3/common/indexer.py`, `vllm/models/minimax_m3/nvidia/model.py`; technical summary: Covers "[Minimax-M3] BF16/FP8 Indexer using MSA"; the main implementation surface is `vllm/models/minimax_m3/nvidia/indexer_msa.py`, `vllm/models/minimax_m3/common/indexer.py`, `vllm/models/minimax_m3/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/nvidia/indexer_msa.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: MiniMaxM3IndexerMSABackend, get_builder_cls, MiniMaxM3IndexerMSAPrefillMetadata, MiniMaxM3IndexerMSAMetadata, touching `MiniMaxM3IndexerMSABackend, get_builder_cls, MiniMaxM3IndexerMSAPrefillMetadata`; `vllm/models/minimax_m3/common/indexer.py` modified +60/-8 (68 lines); hunks: -25,12 +25,14; -46,6 +48,8; symbols: MiniMaxM3IndexerBackend, __init__, forward, touching `MiniMaxM3IndexerBackend, __init__, forward`; `vllm/models/minimax_m3/nvidia/model.py` modified +40/-4 (44 lines); hunks: -402,6 +402,7 @@ def __init__(; -489,6 +490,10 @@ def __init__(; symbols: __init__, forward, _run_attention, touching `__init__, forward, _run_attention`; `vllm/models/minimax_m3/common/ops/index_topk.py` modified +30/-12 (42 lines); hunks: -373,7 +373,10 @@ def _decode_index_score_kernel(; -709,16 +712,25 @@ def minimax_m3_index_topk(; symbols: _decode_index_score_kernel, minimax_m3_index_topk, minimax_m3_index_decode, touching `_decode_index_score_kernel, minimax_m3_index_topk, minimax_m3_index_decode`.
- Code diff details:
  - `vllm/models/minimax_m3/nvidia/indexer_msa.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: MiniMaxM3IndexerMSABackend, get_builder_cls, MiniMaxM3IndexerMSAPrefillMetadata, MiniMaxM3IndexerMSAMetadata
  - `vllm/models/minimax_m3/common/indexer.py` modified +60/-8 (68 lines); hunks: -25,12 +25,14; -46,6 +48,8; symbols: MiniMaxM3IndexerBackend, __init__, forward
  - `vllm/models/minimax_m3/nvidia/model.py` modified +40/-4 (44 lines); hunks: -402,6 +402,7 @@ def __init__(; -489,6 +490,10 @@ def __init__(; symbols: __init__, forward, _run_attention
  - `vllm/models/minimax_m3/common/ops/index_topk.py` modified +30/-12 (42 lines); hunks: -373,7 +373,10 @@ def _decode_index_score_kernel(; -709,16 +712,25 @@ def minimax_m3_index_topk(; symbols: _decode_index_score_kernel, minimax_m3_index_topk, minimax_m3_index_decode
  - `vllm/models/minimax_m3/common/sparse_attention.py` modified +22/-15 (37 lines); hunks: -2,10 +2,11; -272,9 +273,10 @@ class MiniMaxM3SparseImpl(AttentionImplBase[MiniMaxM3Sparse...; symbols: MiniMaxM3SparseImpl, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/nvidia/indexer_msa.py
@@ -0,0 +1,251 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.
+Prefill scores with ``fmha_sm100``'s score-only (``OnlyScore``) path then selects
+top-k blocks with the Triton ``minimax_m3_index_topk`` kernel -- fmha is much
+faster than Triton for the wide prefill score (benchmarked ~3-5x).
diff -- vllm/models/minimax_m3/common/indexer.py
@@ -25,12 +25,14 @@
+from vllm.logger import init_logger
+from vllm.platforms import current_platform
@@ -46,6 +48,8 @@
+logger = init_logger(__name__)
@@ -120,16 +124,20 @@ def __init__(
-        if indexer_kv_dtype != "bf16":
diff -- vllm/models/minimax_m3/nvidia/model.py
@@ -402,6 +402,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/nvidia/indexer_msa.py` added +251/-0; `vllm/models/minimax_m3/common/indexer.py` modified +60/-8; `vllm/models/minimax_m3/nvidia/model.py` modified +40/-4; `vllm/models/minimax_m3/common/ops/index_topk.py` modified +30/-12; `vllm/models/minimax_m3/common/sparse_attention.py` modified +22/-15; `vllm/models/minimax_m3/amd/model.py` modified +28/-3
  - tests: `tests/kernels/attention/test_minimax_m3.py` modified +325/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_minimax_m3.py`, `tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45718 - [Bugfix] Parse MiniMax M3 streaming reasoning by text markers

- Link: https://github.com/vllm-project/vllm/pull/45718
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_minimax_m3_reasoning_parser.py`, `vllm/reasoning/minimax_m3_reasoning_parser.py`; associated commits `d8e422ccda9b`
- Diff scope read: GitHub Pull Request files API returned 2 files, +357/-67, 496 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Parse MiniMax M3 streaming reasoning by text markers"; model line: MiniMax M2 Series; category: bug fix; main diff: `vllm/reasoning/minimax_m3_reasoning_parser.py`, `tests/reasoning/test_minimax_m3_reasoning_parser.py`; technical summary: Covers "[Bugfix] Parse MiniMax M3 streaming reasoning by text markers"; the main implementation surface is `vllm/reasoning/minimax_m3_reasoning_parser.py`, `tests/reasoning/test_minimax_m3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/minimax_m3_reasoning_parser.py` modified +215/-66 (281 lines); hunks: -19,10 +19,12 @@ class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):; -35,9 +37,135 @@ def end_token(self) -> str:; symbols: MiniMaxM3ReasoningParser, end_token, __init__, _encode_text, touching `MiniMaxM3ReasoningParser, end_token, __init__`; `tests/reasoning/test_minimax_m3_reasoning_parser.py` modified +142/-1 (143 lines); hunks: -83,6 +83,20 @@ def convert_tokens_to_string(self, tokens: list[str]) -> str:; -105,7 +119,8 @@ def run_streaming(; symbols: convert_tokens_to_string, SplitMiniMaxM3Tokenizer, tokenize, RuntimeSplitMiniMaxM3Tokenizer, touching `convert_tokens_to_string, SplitMiniMaxM3Tokenizer, tokenize`.
- Code diff details:
  - `vllm/reasoning/minimax_m3_reasoning_parser.py` modified +215/-66 (281 lines); hunks: -19,10 +19,12 @@ class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):; -35,9 +37,135 @@ def end_token(self) -> str:; symbols: MiniMaxM3ReasoningParser, end_token, __init__, _encode_text
  - `tests/reasoning/test_minimax_m3_reasoning_parser.py` modified +142/-1 (143 lines); hunks: -83,6 +83,20 @@ def convert_tokens_to_string(self, tokens: list[str]) -> str:; -105,7 +119,8 @@ def run_streaming(; symbols: convert_tokens_to_string, SplitMiniMaxM3Tokenizer, tokenize, RuntimeSplitMiniMaxM3Tokenizer
- Key code excerpts:

```diff
diff -- vllm/reasoning/minimax_m3_reasoning_parser.py
@@ -19,10 +19,12 @@ class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
-    The M3 tokenizer exposes both markers as complete vocabulary tokens. The
-    chat template may also prefill the start marker when
-    ``thinking_mode="enabled"``, so generated text can begin directly inside a
-    reasoning block without emitting ``<mm:think>`` again.
+    The M3 tokenizer exposes both markers as complete vocabulary entries, but
+    generated marker text may be tokenized into smaller pieces. The streaming
diff -- tests/reasoning/test_minimax_m3_reasoning_parser.py
@@ -83,6 +83,20 @@ def convert_tokens_to_string(self, tokens: list[str]) -> str:
+class SplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
+    """Tokenizer that exposes marker vocab entries but encodes them as text."""
+    def tokenize(self, text: str) -> list[str]:
+        return list(text)
+class RuntimeSplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
+    """Tokenizer whose runtime output splits markers despite atomic encodes."""
```

- Reviewed files:
  - runtime: `vllm/reasoning/minimax_m3_reasoning_parser.py` modified +215/-66
  - tests: `tests/reasoning/test_minimax_m3_reasoning_parser.py` modified +142/-1
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_minimax_m3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45810 - [Model][MiniMax-M3] Add pipeline parallelism support

- Link: https://github.com/vllm-project/vllm/pull/45810
- Status/date: merged / 2026-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`; associated commits `d7c1821b5a31`
- Diff scope read: GitHub Pull Request files API returned 3 files, +124/-49, 397 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][MiniMax-M3] Add pipeline parallelism support"; model line: MiniMax M2 Series; category: model support/runtime entry; main diff: `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`; technical summary: Covers "[Model][MiniMax-M3] Add pipeline parallelism support"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` modified +61/-24 (85 lines); hunks: -30,7 +30,7; -64,12 +64,15; symbols: __init__, embed_input_ids, forward, touching `__init__, embed_input_ids, forward`; `vllm/models/minimax_m3/nvidia/model.py` modified +61/-24 (85 lines); hunks: -21,7 +21,7; -56,12 +56,15; symbols: __init__, embed_input_ids, forward, touching `__init__, embed_input_ids, forward`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` modified +61/-24 (85 lines); hunks: -30,7 +30,7; -64,12 +64,15; symbols: __init__, embed_input_ids, forward
  - `vllm/models/minimax_m3/nvidia/model.py` modified +61/-24 (85 lines); hunks: -21,7 +21,7; -56,12 +56,15; symbols: __init__, embed_input_ids, forward
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -30,7 +30,7 @@
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
@@ -64,12 +64,15 @@
+    SupportsPP,
+    PPMissingLayer,
+    make_empty_intermediate_tensors_factory,
diff -- vllm/models/minimax_m3/nvidia/model.py
@@ -21,7 +21,7 @@
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
@@ -56,12 +56,15 @@
+    SupportsPP,
+    PPMissingLayer,
+    make_empty_intermediate_tensors_factory,
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` modified +61/-24; `vllm/models/minimax_m3/nvidia/model.py` modified +61/-24
- Risk and verification: Runtime changes concentrate in `vllm/models/minimax_m3/amd/model.py`, `vllm/models/minimax_m3/nvidia/model.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46546 - [ROCm][ [Perf] sparse attention optimization on minimax-m3

- Link: https://github.com/vllm-project/vllm/pull/46546
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/minimax_m3/amd/ops/index_topk.py`, `vllm/models/minimax_m3/amd/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/indexer.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/sparse_attention.py`; associated commits `c63cd4906c2a`
- Diff scope read: GitHub Pull Request files API returned 5 files, +1238/-36, 1313 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][ [Perf] sparse attention optimization on minimax-m3"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/models/minimax_m3/amd/ops/index_topk.py`, `vllm/models/minimax_m3/amd/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`; technical summary: Covers "[ROCm][ [Perf] sparse attention optimization on minimax-m3"; the main implementation surface is `vllm/models/minimax_m3/amd/ops/index_topk.py`, `vllm/models/minimax_m3/amd/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/ops/sparse_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/ops/index_topk.py` added +939/-0 (939 lines); hunks: -0,0 +1,939; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel, touching `_compare_and_swap, _bitonic_merge, _index_block_score_kernel`; `vllm/models/minimax_m3/amd/ops/sparse_attn.py` added +271/-0 (271 lines); hunks: -0,0 +1,271; symbols: _sparse_attn_prefill_kwargs, _gqa_sparse_fwd_kernel, minimax_m3_sparse_attn, touching `_sparse_attn_prefill_kwargs, _gqa_sparse_fwd_kernel, minimax_m3_sparse_attn`; `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +0/-26 (26 lines); hunks: -31,30 +31,6; -498,7 +474,6 @@ def minimax_m3_sparse_attn(; symbols: _sparse_attn_num_stages_kwarg, minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode, touching `_sparse_attn_num_stages_kwarg, minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode`; `vllm/models/minimax_m3/common/indexer.py` modified +14/-5 (19 lines); hunks: -27,12 +27,21.
- Code diff details:
  - `vllm/models/minimax_m3/amd/ops/index_topk.py` added +939/-0 (939 lines); hunks: -0,0 +1,939; symbols: _compare_and_swap, _bitonic_merge, _index_block_score_kernel, _topk_index_kernel
  - `vllm/models/minimax_m3/amd/ops/sparse_attn.py` added +271/-0 (271 lines); hunks: -0,0 +1,271; symbols: _sparse_attn_prefill_kwargs, _gqa_sparse_fwd_kernel, minimax_m3_sparse_attn
  - `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +0/-26 (26 lines); hunks: -31,30 +31,6; -498,7 +474,6 @@ def minimax_m3_sparse_attn(; symbols: _sparse_attn_num_stages_kwarg, minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode
  - `vllm/models/minimax_m3/common/indexer.py` modified +14/-5 (19 lines); hunks: -27,12 +27,21
  - `vllm/models/minimax_m3/common/sparse_attention.py` modified +14/-5 (19 lines); hunks: -24,12 +24,21
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/ops/index_topk.py
@@ -0,0 +1,939 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.
+Index queries score each 128-token block of index keys (max over the block),
+then the top-k blocks (plus forced init/local blocks) are selected per query
+token. Adapted to vLLM's paged KV cache: the KV page size is forced to equal the
diff -- vllm/models/minimax_m3/amd/ops/sparse_attn.py
@@ -0,0 +1,271 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""ROCm gfx942/gfx950 block-sparse GQA prefill kernel for MiniMax-M3.
+Only the prefill path is specialized on CDNA: each 128-token KV block is split
+into SUB_K-token sub-tiles to right-size the per-block QK/PV MFMAs. Everything
+else -- the decode split-K kernels, the FP8 dtype set, the sparse block size --
diff -- vllm/models/minimax_m3/common/ops/sparse_attn.py
@@ -31,30 +31,6 @@
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/ops/index_topk.py` added +939/-0; `vllm/models/minimax_m3/amd/ops/sparse_attn.py` added +271/-0; `vllm/models/minimax_m3/common/ops/sparse_attn.py` modified +0/-26; `vllm/models/minimax_m3/common/indexer.py` modified +14/-5; `vllm/models/minimax_m3/common/sparse_attention.py` modified +14/-5
- Risk and verification: Runtime changes concentrate in `vllm/models/minimax_m3/amd/ops/index_topk.py`, `vllm/models/minimax_m3/amd/ops/sparse_attn.py`, `vllm/models/minimax_m3/common/indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46419 - [ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4

- Link: https://github.com/vllm-project/vllm/pull/46419
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/minimax_m3/amd/model.py`; associated commits `8e394244a59a`
- Diff scope read: GitHub Pull Request files API returned 4 files, +27/-8, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/models/minimax_m3/amd/model.py`; technical summary: Covers "[ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` modified +1/-0 (1 lines); hunks: -309,6 +309,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` modified +1/-0 (1 lines); hunks: -309,6 +309,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -309,6 +309,7 @@ def __init__(
+            intermediate_pad=0,
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46545 - [ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model

- Link: https://github.com/vllm-project/vllm/pull/46545
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/minimax_m3/amd/model.py`; associated commits `c2507fb2937a`
- Diff scope read: GitHub Pull Request files API returned 5 files, +110/-20, 266 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model"; model line: MiniMax M2 Series; category: performance/backend optimization; main diff: `vllm/models/minimax_m3/amd/model.py`; technical summary: Covers "[ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model"; the main implementation surface is `vllm/models/minimax_m3/amd/model.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/models/minimax_m3/amd/model.py` modified +47/-3 (50 lines); hunks: -23,6 +23,7; -95,6 +96,7; symbols: _fuse_shared_experts_enabled, _sparse_attention_layer_ids, __init__, forward, touching `_fuse_shared_experts_enabled, _sparse_attention_layer_ids, __init__`.
- Code diff details:
  - `vllm/models/minimax_m3/amd/model.py` modified +47/-3 (50 lines); hunks: -23,6 +23,7; -95,6 +96,7; symbols: _fuse_shared_experts_enabled, _sparse_attention_layer_ids, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/models/minimax_m3/amd/model.py
@@ -23,6 +23,7 @@
+import vllm.envs as envs
@@ -95,6 +96,7 @@
+from vllm.platforms import current_platform
@@ -104,6 +106,23 @@
+def _fuse_shared_experts_enabled(config: PretrainedConfig) -> bool:
+    """Whether to fuse the shared expert into the routed grouped MoE.
```

- Reviewed files:
  - runtime: `vllm/models/minimax_m3/amd/model.py` modified +47/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
