# vllm Gemma 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_gemma4.jinja` | [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39570](https://github.com/vllm-project/vllm/pull/39570), [#41459](https://github.com/vllm-project/vllm/pull/41459), [#42188](https://github.com/vllm-project/vllm/pull/42188), [#45553](https://github.com/vllm-project/vllm/pull/45553), [#45867](https://github.com/vllm-project/vllm/pull/45867) |
| `tests/kernels/moe/test_gemma4router.py` | [#39083](https://github.com/vllm-project/vllm/pull/39083) |
| `tests/models/multimodal/processing/test_gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#40796](https://github.com/vllm-project/vllm/pull/40796), [#41799](https://github.com/vllm-project/vllm/pull/41799), [#41837](https://github.com/vllm-project/vllm/pull/41837), [#42217](https://github.com/vllm-project/vllm/pull/42217), [#43296](https://github.com/vllm-project/vllm/pull/43296) |
| `tests/models/multimodal/processing/test_gemma4_unified.py` | [#44429](https://github.com/vllm-project/vllm/pull/44429) |
| `tests/parser/engine/test_gemma4_streaming_reasoning.py` | [#45588](https://github.com/vllm-project/vllm/pull/45588), [#45834](https://github.com/vllm-project/vllm/pull/45834), [#45852](https://github.com/vllm-project/vllm/pull/45852) |
| `tests/reasoning/test_gemma4_reasoning_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#45553](https://github.com/vllm-project/vllm/pull/45553), [#45588](https://github.com/vllm-project/vllm/pull/45588) |
| `tests/renderers/test_gemma4_chat_template.py` | [#39027](https://github.com/vllm-project/vllm/pull/39027), [#41459](https://github.com/vllm-project/vllm/pull/41459), [#45553](https://github.com/vllm-project/vllm/pull/45553) |
| `tests/tool_parsers/test_gemma4_tool_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38909](https://github.com/vllm-project/vllm/pull/38909), [#38992](https://github.com/vllm-project/vllm/pull/38992), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39114](https://github.com/vllm-project/vllm/pull/39114), [#39679](https://github.com/vllm-project/vllm/pull/39679), [#41991](https://github.com/vllm-project/vllm/pull/41991), [#42128](https://github.com/vllm-project/vllm/pull/42128), [#45588](https://github.com/vllm-project/vllm/pull/45588) |
| `tests/tool_use/test_gemma4_responses_adjust_request.py` | [#45588](https://github.com/vllm-project/vllm/pull/45588), [#45795](https://github.com/vllm-project/vllm/pull/45795), [#45832](https://github.com/vllm-project/vllm/pull/45832) |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/model_executor/models/gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38844](https://github.com/vllm-project/vllm/pull/38844), [#38879](https://github.com/vllm-project/vllm/pull/38879), [#39045](https://github.com/vllm-project/vllm/pull/39045), [#39083](https://github.com/vllm-project/vllm/pull/39083), [#39450](https://github.com/vllm-project/vllm/pull/39450), [#40588](https://github.com/vllm-project/vllm/pull/40588), [#40708](https://github.com/vllm-project/vllm/pull/40708), [#40786](https://github.com/vllm-project/vllm/pull/40786), [#41206](https://github.com/vllm-project/vllm/pull/41206), [#41574](https://github.com/vllm-project/vllm/pull/41574), [#42250](https://github.com/vllm-project/vllm/pull/42250), ... (13 total) |
| `vllm/model_executor/models/gemma4_mm.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38872](https://github.com/vllm-project/vllm/pull/38872), [#39234](https://github.com/vllm-project/vllm/pull/39234), [#39291](https://github.com/vllm-project/vllm/pull/39291), [#39450](https://github.com/vllm-project/vllm/pull/39450), [#39842](https://github.com/vllm-project/vllm/pull/39842), [#40411](https://github.com/vllm-project/vllm/pull/40411), [#40534](https://github.com/vllm-project/vllm/pull/40534), [#40796](https://github.com/vllm-project/vllm/pull/40796), [#41799](https://github.com/vllm-project/vllm/pull/41799), [#41837](https://github.com/vllm-project/vllm/pull/41837), [#42217](https://github.com/vllm-project/vllm/pull/42217), ... (18 total) |
| `vllm/model_executor/models/gemma4_mtp.py` | [#41745](https://github.com/vllm-project/vllm/pull/41745), [#43909](https://github.com/vllm-project/vllm/pull/43909), [#44429](https://github.com/vllm-project/vllm/pull/44429) |
| `vllm/model_executor/models/gemma4_unified.py` | [#44429](https://github.com/vllm-project/vllm/pull/44429), [#44571](https://github.com/vllm-project/vllm/pull/44571) |
| `vllm/parser/gemma4.py` | [#45553](https://github.com/vllm-project/vllm/pull/45553), [#45588](https://github.com/vllm-project/vllm/pull/45588), [#45832](https://github.com/vllm-project/vllm/pull/45832), [#45834](https://github.com/vllm-project/vllm/pull/45834), [#45852](https://github.com/vllm-project/vllm/pull/45852) |
| `vllm/reasoning/gemma4_engine_reasoning_parser.py` | [#45588](https://github.com/vllm-project/vllm/pull/45588) |
| `vllm/reasoning/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/tool_parsers/gemma4_engine_tool_parser.py` | [#45588](https://github.com/vllm-project/vllm/pull/45588), [#45795](https://github.com/vllm-project/vllm/pull/45795) |
| `vllm/tool_parsers/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#45553](https://github.com/vllm-project/vllm/pull/45553) |
| `vllm/v1/spec_decode/gemma4.py` | [#41745](https://github.com/vllm-project/vllm/pull/41745), [#43982](https://github.com/vllm-project/vllm/pull/43982) |
| `vllm/v1/worker/gpu/spec_decode/gemma4/__init__.py` | [#43241](https://github.com/vllm-project/vllm/pull/43241) |
| `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` | [#43241](https://github.com/vllm-project/vllm/pull/43241) |

## PR 覆盖总览

- git 追溯 PR 数: 49
- 原文档显式引用补充 PR 数: 5
- 当前文档总 PR 数: 54
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-02 | [#38826](https://github.com/vllm-project/vllm/pull/38826) | merged | feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use) | `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-04-02 | [#38847](https://github.com/vllm-project/vllm/pull/38847) | merged | [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter | `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-03 | [#38872](https://github.com/vllm-project/vllm/pull/38872) | merged | [Misc] Clean up Gemma4 implementation | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-05 | [#38992](https://github.com/vllm-project/vllm/pull/38992) | merged | [Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-04-06 | [#38879](https://github.com/vllm-project/vllm/pull/38879) | merged | [Gemma4] Enable Fast Prefill Optimization | `vllm/model_executor/models/gemma4.py` |
| 2026-04-08 | [#38909](https://github.com/vllm-project/vllm/pull/38909) | merged | [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-04-08 | [#39114](https://github.com/vllm-project/vllm/pull/39114) | merged | [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-04-08 | [#39027](https://github.com/vllm-project/vllm/pull/39027) | merged | [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes | `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/renderers/test_gemma4_chat_template.py` |
| 2026-04-09 | [#39045](https://github.com/vllm-project/vllm/pull/39045) | merged | [Gemma4] Support quantized MoE | `vllm/model_executor/models/gemma4.py` |
| 2026-04-10 | [#39450](https://github.com/vllm-project/vllm/pull/39450) | merged | Add Gemma4 Eagle3 support | `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-11 | [#38844](https://github.com/vllm-project/vllm/pull/38844) | merged | [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly | `vllm/model_executor/models/gemma4.py` |
| 2026-04-14 | [#39679](https://github.com/vllm-project/vllm/pull/39679) | merged | [Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"` | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-04-15 | [#39842](https://github.com/vllm-project/vllm/pull/39842) | merged | [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-17 | [#39234](https://github.com/vllm-project/vllm/pull/39234) | merged | [Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids` | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-17 | [#39291](https://github.com/vllm-project/vllm/pull/39291) | merged | feat: Add LoRA support for Gemma4ForConditionalGeneration | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-19 | [#39083](https://github.com/vllm-project/vllm/pull/39083) | merged | [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton | `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py` |
| 2026-04-21 | [#40411](https://github.com/vllm-project/vllm/pull/40411) | merged | [Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-24 | [#40534](https://github.com/vllm-project/vllm/pull/40534) | merged | [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-29 | [#40786](https://github.com/vllm-project/vllm/pull/40786) | merged | Fix PP in Gemma4 | `vllm/model_executor/models/gemma4.py` |
| 2026-04-30 | [#41206](https://github.com/vllm-project/vllm/pull/41206) | merged | Fix Gemma4 MoE expert weight remapping | `vllm/model_executor/models/gemma4.py` |
| 2026-05-02 | [#39570](https://github.com/vllm-project/vllm/pull/39570) | merged | [Fix] Sync gemma4 chat template from hf | `examples/tool_chat_template_gemma4.jinja` |
| 2026-05-02 | [#40796](https://github.com/vllm-project/vllm/pull/40796) | merged | [Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens | `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-05-05 | [#41574](https://github.com/vllm-project/vllm/pull/41574) | merged | [Model] Fix Gemma4 MoE activation mismatch | `vllm/model_executor/models/gemma4.py` |
| 2026-05-06 | [#41799](https://github.com/vllm-project/vllm/pull/41799) | merged | [MM][Gemma4] Respect max_soft_tokens in encoder budget | `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-05-06 | [#41745](https://github.com/vllm-project/vllm/pull/41745) | merged | [Spec Decode] Add Gemma4 MTP speculative decoding support | `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py` |
| 2026-05-07 | [#41837](https://github.com/vllm-project/vllm/pull/41837) | merged | [MM][Gemma4] Use video profiling hints in encoder budget | `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-05-08 | [#40588](https://github.com/vllm-project/vllm/pull/40588) | merged | [Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP | `vllm/model_executor/models/gemma4.py` |
| 2026-05-08 | [#41991](https://github.com/vllm-project/vllm/pull/41991) | merged | [Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-05-09 | [#40708](https://github.com/vllm-project/vllm/pull/40708) | merged | [BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue | `vllm/model_executor/models/gemma4.py` |
| 2026-05-11 | [#42188](https://github.com/vllm-project/vllm/pull/42188) | merged | [Bugfix] Gemma 4 chat template crash with missing tool name and tool id | `examples/tool_chat_template_gemma4.jinja` |
| 2026-05-12 | [#42217](https://github.com/vllm-project/vllm/pull/42217) | merged | [Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash | `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-05-13 | [#42250](https://github.com/vllm-project/vllm/pull/42250) | merged | [Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution | `vllm/model_executor/models/gemma4.py` |
| 2026-05-14 | [#42128](https://github.com/vllm-project/vllm/pull/42128) | merged | [Bugfix] Fix Gemma4ToolParser streaming float corruption | `tests/tool_parsers/test_gemma4_tool_parser.py` |
| 2026-05-21 | [#43169](https://github.com/vllm-project/vllm/pull/43169) | merged | [Perf][Gemma4] Batch vision encoder calls for image and video processing | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-05-22 | [#43296](https://github.com/vllm-project/vllm/pull/43296) | merged | [CI] Fix "test_awq_load[gemma4-moe-*]" failure | `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/processing/test_gemma4.py` |
| 2026-05-28 | [#41459](https://github.com/vllm-project/vllm/pull/41459) | merged | fix(frontend): Add multimodal placeholders to Gemma4 tool message template | `tests/renderers/test_gemma4_chat_template.py`, `examples/tool_chat_template_gemma4.jinja` |
| 2026-05-30 | [#43909](https://github.com/vllm-project/vllm/pull/43909) | merged | [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered` | `vllm/model_executor/models/gemma4_mtp.py` |
| 2026-06-02 | [#43798](https://github.com/vllm-project/vllm/pull/43798) | merged | [Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-06-02 | [#44232](https://github.com/vllm-project/vllm/pull/44232) | merged | [Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-06-03 | [#44429](https://github.com/vllm-project/vllm/pull/44429) | merged | [Model] Add Gemma4 Unified (encoder-free) support | `vllm/model_executor/models/gemma4_unified.py`, `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-06-04 | [#43982](https://github.com/vllm-project/vllm/pull/43982) | merged | [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load | `vllm/v1/spec_decode/gemma4.py` |
| 2026-06-04 | [#43241](https://github.com/vllm-project/vllm/pull/43241) | merged | [Model Runner V2][Spec Decode] Add Gemma4 MTP support | `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py` |
| 2026-06-04 | [#44340](https://github.com/vllm-project/vllm/pull/44340) | merged | [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings | `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-06-05 | [#44571](https://github.com/vllm-project/vllm/pull/44571) | merged | [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified | `vllm/model_executor/models/gemma4_unified.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-08 | [#44828](https://github.com/vllm-project/vllm/pull/44828) | merged | [BugFix] Use served model name in gemma4 audio-tower error message | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-12 | [#45163](https://github.com/vllm-project/vllm/pull/45163) | merged | [Model] Add DiffusionGemma Support | `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/models/config.py` |
| 2026-06-15 | [#45588](https://github.com/vllm-project/vllm/pull/45588) | merged | [Frontend] Replace legacy Gemma4 parsers with engine-based implementation | `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py` |
| 2026-06-16 | [#45553](https://github.com/vllm-project/vllm/pull/45553) | merged | [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync | `vllm/tool_parsers/gemma4_utils.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `examples/tool_chat_template_gemma4.jinja` |
| 2026-06-16 | [#45795](https://github.com/vllm-project/vllm/pull/45795) | merged | [Bugfix] Gemma4: skip forced JSON for required/named tool choice | `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py` |
| 2026-06-17 | [#45832](https://github.com/vllm-project/vllm/pull/45832) | merged | [Bugfix][Gemma4] Fix parsing when thinking is disabled | `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py` |
| 2026-06-17 | [#45852](https://github.com/vllm-project/vllm/pull/45852) | merged | [Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834) | `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py` |
| 2026-06-17 | [#45867](https://github.com/vllm-project/vllm/pull/45867) | merged | [Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls | `examples/tool_chat_template_gemma4.jinja` |

## 逐 PR diff 审计卡

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- 链接: https://github.com/vllm-project/vllm/pull/38826
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`, `vllm/model_executor/models/gemma4.py` 等 8 个文件；关联提交 `08ed2b9688b4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+5051/-1，可读 patch 5167 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo，涉及 `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs`；`vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward，涉及 `_get_text_config, Gemma4MLP, __init__`；`tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args，涉及 `mock_tokenizer, parser, mock_request`；`vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments，涉及 `parse_thinking_output, _strip_thought_label, _clean_answer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
  - `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0 (196 lines); hunks: -0,0 +1,196; symbols: generic_tokenizer, test_gemma4_reasoning, _encode
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -0,0 +1,1341 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Gemma 4 multimodal model (image + audio + video support).
+Adds vision tower, audio tower, and multimodal embedders on top of the
+text-only Gemma4ForCausalLM.  The vision/audio encoders are loaded via
+AutoModel.from_config and run in eager mode while the language model uses
diff -- vllm/model_executor/models/gemma4.py
@@ -0,0 +1,1239 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
+#
+#
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -0,0 +1,504 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0; `vllm/model_executor/models/gemma4.py` added +1239/-0; `vllm/model_executor/models/gemma4_utils.py` added +292/-0; `vllm/tool_parsers/gemma4_utils.py` added +183/-0; `vllm/reasoning/gemma4_utils.py` added +130/-0; `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` added +84/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0; `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/registry.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38847 - [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter

- 链接: https://github.com/vllm-project/vllm/pull/38847
- 状态/时间: merged / 2026-04-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter」；主要实现面是 `vllm/tool_parsers/gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3 (6 lines); hunks: -38,7 +38,7; -281,8 +281,8 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__，涉及 `Gemma4ToolParser, __init__`。
- 代码 diff 细节:
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3 (6 lines); hunks: -38,7 +38,7; -281,8 +281,8 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -38,7 +38,7 @@
-from vllm.tool_parsers.abstract_tool_parser import ToolParser
+from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
@@ -281,8 +281,8 @@ class Gemma4ToolParser(ToolParser):
-    def __init__(self, tokenizer: TokenizerLike):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
```

- 已读文件:
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/gemma4_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38872 - [Misc] Clean up Gemma4 implementation

- 链接: https://github.com/vllm-project/vllm/pull/38872
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `550643541956`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+5/-300，可读 patch 333 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Clean up Gemma4 implementation」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Misc] Clean up Gemma4 implementation」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +3/-6 (9 lines); hunks: -15,7 +15,6; -480,12 +479,10 @@ def _call_hf_processor(; symbols: _call_hf_processor，涉及 `_call_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +3/-6 (9 lines); hunks: -15,7 +15,6; -480,12 +479,10 @@ def _call_hf_processor(; symbols: _call_hf_processor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -15,7 +15,6 @@
-import sys
@@ -480,12 +479,10 @@ def _call_hf_processor(
-            logger.error(
-                "Unsupported max_soft_tokens value: %d. Valid values are %s. Exiting.",
-                val,
-                _SUPPORTED_SOFT_TOKENS,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +3/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4_utils.py`, `vllm/transformers_utils/model_arch_config_convertor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38992 - [Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters

- 链接: https://github.com/vllm-project/vllm/pull/38992
- 状态/时间: merged / 2026-04-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `f53fa26e05c4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+33/-3，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json，涉及 `test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):
+    def test_streaming_split_delimiter_no_invalid_json(self, parser, mock_request):
+        """Partial <|"|> delimiter chars must not leak into streamed JSON.
+        Reproduces the bug from https://github.com/vllm-project/vllm/issues/38946
+        where a token boundary splits the string delimiter, leaving fragments
+        like '<|' at the end of a parsed value which then corrupt the JSON.
+        """
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38879 - [Gemma4] Enable Fast Prefill Optimization

- 链接: https://github.com/vllm-project/vllm/pull/38879
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `47e605092b7f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+369/-47，可读 patch 490 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Enable Fast Prefill Optimization」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Gemma4] Enable Fast Prefill Optimization」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunks: -19,6 +19,7; -32,6 +33,7; symbols: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__，涉及 `forward, _run_decoder_layers, Gemma4SelfDecoderLayers`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunks: -19,6 +19,7; -32,6 +33,7; symbols: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -19,6 +19,7 @@
+from dataclasses import replace
@@ -32,6 +33,7 @@
+from vllm.forward_context import get_forward_context
@@ -56,6 +58,7 @@
+from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata
@@ -636,7 +639,205 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +369/-47
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38909 - [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls

- 链接: https://github.com/vllm-project/vllm/pull/38909
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `d734445fcd79`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+64/-2，可读 patch 77 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes，涉及 `test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, parser, mock_request):
+    def test_streaming_does_not_duplicate_plain_text_after_tool_call(
+        self, parser, mock_request, monkeypatch
+    ):
+        """Buffered plain text after a tool call must not corrupt current_text."""
+        captured_current_texts: list[str] = []
+        original_extract_streaming = parser._extract_streaming
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39114 - [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values

- 链接: https://github.com/vllm-project/vllm/pull/39114
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `13151a4df43d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+78/-8，可读 patch 159 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks，涉及 `test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):
+    def test_streaming_boolean_split_across_chunks(self, parser, mock_request):
+        """Boolean value split across token boundaries must not corrupt JSON."""
+        chunks = [
+            "<|tool_call>",
+            "call:search{input:{all:" + "true"[:3],
+            "e}}",
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39027 - [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes

- 链接: https://github.com/vllm-project/vllm/pull/39027
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `8477fe427d17`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+878/-16，可读 patch 1083 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/renderers/test_gemma4_chat_template.py`；技术摘要: 覆盖「[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes」；主要实现面是 `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/renderers/test_gemma4_chat_template.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode，涉及 `generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output`；`tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array，涉及 `test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld`；`tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled，涉及 `gemma4_template, _render, TestGemma4ChatTemplate`；`examples/tool_chat_template_gemma4.jinja` added +331/-0 (331 lines); hunks: -0,0 +1,331。
- 代码 diff 细节:
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array
  - `tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled
  - `examples/tool_chat_template_gemma4.jinja` added +331/-0 (331 lines); hunks: -0,0 +1,331
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_gemma4_reasoning_parser.py
@@ -4,6 +4,9 @@
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+)
@@ -100,6 +103,39 @@ def generic_tokenizer():
+THOUGHT_PREFIX = {
+    "output": "<|channel>thought\nActual reasoning here<channel|>Final answer",
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -114,6 +114,19 @@ def test_empty_value(self):
+    def test_empty_value_partial_withheld(self):
+        """Key with no value is withheld in partial mode to avoid premature emission."""
+        result = _parse_gemma4_args("key:", partial=True)
+        assert result == {}
+        # also with a space after the colon
+        result = _parse_gemma4_args("key: ", partial=True)
diff -- tests/renderers/test_gemma4_chat_template.py
@@ -0,0 +1,345 @@
```

- 已读文件:
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0; `tests/renderers/test_gemma4_chat_template.py` added +345/-0
  - docs: `examples/tool_chat_template_gemma4.jinja` added +331/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39045 - [Gemma4] Support quantized MoE

- 链接: https://github.com/vllm-project/vllm/pull/39045
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `3aecdf08b4a8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-14，可读 patch 89 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Support quantized MoE」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Gemma4] Support quantized MoE」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunks: -1248,21 +1248,27 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; -1322,9 +1328,21 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _weight_iterator，涉及 `load_weights, _weight_iterator`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunks: -1248,21 +1248,27 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; -1322,9 +1328,21 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _weight_iterator
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -1248,21 +1248,27 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        # MoE expert weight mapping: checkpoint 3D packed tensors are
-        # exploded in _weight_iterator to per-expert 2D weights like:
+        # MoE expert weight mapping: checkpoint can have either:
+        #   1. 3D packed tensors (exploded in _weight_iterator to per-expert 2D)
+        #   2. Already per-expert 2D weights (if quantized)
+        # Map to FusedMoE parameters:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +34/-14
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39450 - Add Gemma4 Eagle3 support

- 链接: https://github.com/vllm-project/vllm/pull/39450
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `e7cfd7c5b9a1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+43/-10，可读 patch 146 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Gemma4 Eagle3 support」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「Add Gemma4 Eagle3 support」；主要实现面是 `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunks: -60,7 +60,13; -838,7 +844,7 @@ def forward(; symbols: forward, Gemma4Model, __init__，涉及 `forward, Gemma4Model, __init__`；`vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunks: -64,7 +64,12; -845,7 +850,12 @@ def forward(self, inputs_embeds: torch.Tensor) -> torch.Ten...; symbols: forward, Gemma4ForConditionalGeneration，涉及 `forward, Gemma4ForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunks: -60,7 +60,13; -838,7 +844,7 @@ def forward(; symbols: forward, Gemma4Model, __init__
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunks: -64,7 +64,12; -845,7 +850,12 @@ def forward(self, inputs_embeds: torch.Tensor) -> torch.Ten...; symbols: forward, Gemma4ForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -60,7 +60,13 @@
-from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
+from .interfaces import (
+    EagleModelMixin,
+    MixtureOfExperts,
+    SupportsEagle3,
+    SupportsLoRA,
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -64,7 +64,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsEagle3,
+    SupportsMultiModal,
+    SupportsPP,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +20/-5; `vllm/model_executor/models/gemma4_mm.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/speculative.py`, `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38844 - [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly

- 链接: https://github.com/vllm-project/vllm/pull/38844
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `92feb9991d15`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+40/-0，可读 patch 66 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunks: -69,6 +69,7; -1397,6 +1398,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Gemma4ForCausalLM，涉及 `load_weights, Gemma4ForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunks: -69,6 +69,7; -1397,6 +1398,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Gemma4ForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -69,6 +69,7 @@
+    WeightsMapper,
@@ -1397,6 +1398,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    hf_to_vllm_mapper = WeightsMapper(
+        orig_to_new_prefix={
+            # Gemma4ForConditionalGeneration already loads the text stack
+            # from `model.language_model.*`. We reuse that same checkpoint
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +17/-0
- 验证与风险: diff 自带测试面 `tests/lora/test_lora_checkpoints.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39679 - [Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`

- 链接: https://github.com/vllm-project/vllm/pull/39679
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `b075604da10a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types，涉及 `test_boolean_false, test_null_value, test_mixed_types`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -85,6 +85,14 @@ def test_boolean_false(self):
+    def test_null_value(self):
+        # Bare `null` must parse as None (Python), not the string "null".
+        # Without this, tool_choice=auto would emit `{"param": "null"}`
+        # instead of `{"param": null}` for nullable tool parameters.
+        result = _parse_gemma4_args("param:null")
+        assert result == {"param": None}
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39842 - [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models

- 链接: https://github.com/vllm-project/vllm/pull/39842
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `6dc949140693`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +7/-2 (9 lines); hunks: -167,10 +167,15 @@ def get_default_tok_params(self):; symbols: get_default_tok_params, get_hf_processor，涉及 `get_default_tok_params, get_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +7/-2 (9 lines); hunks: -167,10 +167,15 @@ def get_default_tok_params(self):; symbols: get_default_tok_params, get_hf_processor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -167,10 +167,15 @@ def get_default_tok_params(self):
-        correctly.
+        correctly for IT models. For PT models (without chat template), we
+        keep the default (True) to ensure BOS is added for raw prompts.
+        tokenizer = self.ctx.get_tokenizer()
+        has_chat_template = getattr(tokenizer, "chat_template", None) is not None
-        params = params.with_kwargs(add_special_tokens=False)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39234 - [Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`

- 链接: https://github.com/vllm-project/vllm/pull/39234
- 状态/时间: merged / 2026-04-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `b1dc87a0989f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +3/-2 (5 lines); hunks: -1254,9 +1254,10 @@ def embed_input_ids(; symbols: embed_input_ids，涉及 `embed_input_ids`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +3/-2 (5 lines); hunks: -1254,9 +1254,10 @@ def embed_input_ids(; symbols: embed_input_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -1254,9 +1254,10 @@ def embed_input_ids(
-                is_multimodal = is_multimodal.to(input_ids.device)
-                    is_multimodal, torch.zeros_like(input_ids), input_ids
+                    is_multimodal.to(input_ids.device, non_blocking=True),
+                    torch.zeros_like(input_ids),
+                    input_ids,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39291 - feat: Add LoRA support for Gemma4ForConditionalGeneration

- 链接: https://github.com/vllm-project/vllm/pull/39291
- 状态/时间: merged / 2026-04-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `640cc9dd7dae`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Add LoRA support for Gemma4ForConditionalGeneration」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「feat: Add LoRA support for Gemma4ForConditionalGeneration」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +10/-2 (12 lines); hunks: -67,6 +67,7; -880,6 +881,7 @@ class Gemma4ForConditionalGeneration(; symbols: Gemma4ForConditionalGeneration, load_weights, get_mm_mapping，涉及 `Gemma4ForConditionalGeneration, load_weights, get_mm_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +10/-2 (12 lines); hunks: -67,6 +67,7; -880,6 +881,7 @@ class Gemma4ForConditionalGeneration(; symbols: Gemma4ForConditionalGeneration, load_weights, get_mm_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -67,6 +67,7 @@
+    SupportsLoRA,
@@ -880,6 +881,7 @@ class Gemma4ForConditionalGeneration(
+    SupportsLoRA,
@@ -1357,10 +1359,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        connectors = ["embed_vision"]
+        tower_models = ["vision_tower"]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39083 - [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton

- 链接: https://github.com/vllm-project/vllm/pull/39083
- 状态/时间: merged / 2026-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/moe/test_gemma4router.py`, `vllm/model_executor/models/gemma4.py`；关联提交 `45232a454e4c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+180/-16，可读 patch 226 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py`；技术摘要: 覆盖「[FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton」；主要实现面是 `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +122/-16 (138 lines); hunks: -57,7 +57,9; -79,6 +81,120; symbols: _gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch, _get_text_config，涉及 `_gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch`；`tests/kernels/moe/test_gemma4router.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: sort_by_id, test_gemma4_routing_kernel_triton，涉及 `sort_by_id, test_gemma4_routing_kernel_triton`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +122/-16 (138 lines); hunks: -57,7 +57,9; -79,6 +81,120; symbols: _gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch, _get_text_config
  - `tests/kernels/moe/test_gemma4router.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: sort_by_id, test_gemma4_routing_kernel_triton
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -57,7 +57,9 @@
+from vllm.platforms import current_platform
+from vllm.triton_utils import tl, triton
@@ -79,6 +81,120 @@
+@triton.jit
+def _gemma4_routing_kernel(
+    gating_ptr,
diff -- tests/kernels/moe/test_gemma4router.py
@@ -0,0 +1,57 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import torch
+from vllm.model_executor.models.gemma4 import (
+    gemma4_fused_routing_kernel_triton,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +122/-16
  - tests: `tests/kernels/moe/test_gemma4router.py` added +57/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_gemma4router.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40411 - [Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference

- 链接: https://github.com/vllm-project/vllm/pull/40411
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `20d37434911d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-8，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +9/-8 (17 lines); hunks: -849,22 +849,23 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +9/-8 (17 lines); hunks: -849,22 +849,23 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -849,22 +849,23 @@ def __init__(
-        self.embedding_projection = ReplicatedLinear(
+        self.embedding_pre_projection_norm = RMSNorm(
-            self.text_hidden_size,
-            bias=False,
+            eps=self.eps,
+            has_weight=False,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +9/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40534 - [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard

- 链接: https://github.com/vllm-project/vllm/pull/40534
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `512f52219240`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+73/-1，可读 patch 108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Gemma4: add bidirectional vision attention for sliding layers with window guard」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Model] Gemma4: add bidirectional vision attention for sliding layers with window guard」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +59/-0 (59 lines); hunks: -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1310,6 +1320,12 @@ def forward(; symbols: __init__, forward, compute_logits, _clear_mm_prefix_for_full_attn_layers，涉及 `__init__, forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +59/-0 (59 lines); hunks: -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1310,6 +1320,12 @@ def forward(; symbols: __init__, forward, compute_logits, _clear_mm_prefix_for_full_attn_layers
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # --- Precompute full-attention layer indices for bidi clearing ---
+        self._full_attn_layer_idxs: frozenset[int] = frozenset()
+        text_config = config.text_config
+        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
+            layer_types = getattr(text_config, "layer_types", None)
+            if layer_types:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +59/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/worker/gpu_model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40786 - Fix PP in Gemma4

- 链接: https://github.com/vllm-project/vllm/pull/40786
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `5371d6fb4023`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-16，可读 patch 49 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix PP in Gemma4」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「Fix PP in Gemma4」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +9/-16 (25 lines); hunks: -1144,11 +1144,6 @@ def _make_empty_intermediate_tensors(; -1312,13 +1307,12 @@ def forward(; symbols: _make_empty_intermediate_tensors, forward，涉及 `_make_empty_intermediate_tensors, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +9/-16 (25 lines); hunks: -1144,11 +1144,6 @@ def _make_empty_intermediate_tensors(; -1312,13 +1307,12 @@ def forward(; symbols: _make_empty_intermediate_tensors, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -1144,11 +1144,6 @@ def _make_empty_intermediate_tensors(
-                "residual": torch.zeros(
-                    (batch_size, hidden_size),
-                    dtype=dtype,
-                    device=device,
-                ),
@@ -1312,13 +1307,12 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +9/-16
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41206 - Fix Gemma4 MoE expert weight remapping

- 链接: https://github.com/vllm-project/vllm/pull/41206
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `ca97f7b9bbf2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Gemma4 MoE expert weight remapping」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「Fix Gemma4 MoE expert weight remapping」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +5/-1 (6 lines); hunks: -84,6 +84,10; -1650,7 +1654,7 @@ def _weight_iterator():; symbols: _remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator，涉及 `_remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +5/-1 (6 lines); hunks: -84,6 +84,10; -1650,7 +1654,7 @@ def _weight_iterator():; symbols: _remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -84,6 +84,10 @@
+def _remap_gemma4_expert_weight_name(name: str) -> str:
+    return re.sub(r"(?<!\.moe)\.experts\.(\d+)\.", r".moe.experts.\1.", name)
@@ -1650,7 +1654,7 @@ def _weight_iterator():
-                name = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", name)
+                name = _remap_gemma4_expert_weight_name(name)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39570 - [Fix] Sync gemma4 chat template from hf

- 链接: https://github.com/vllm-project/vllm/pull/39570
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`；关联提交 `c408fdd663af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+67/-44，可读 patch 239 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Sync gemma4 chat template from hf」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_gemma4.jinja`；技术摘要: 覆盖「[Fix] Sync gemma4 chat template from hf」；主要实现面是 `examples/tool_chat_template_gemma4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_gemma4.jinja` modified +67/-44 (111 lines); hunks: -1,44 +1,25; -71,6 +52,32。
- 代码 diff 细节:
  - `examples/tool_chat_template_gemma4.jinja` modified +67/-44 (111 lines); hunks: -1,44 +1,25; -71,6 +52,32
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_gemma4.jinja
@@ -1,44 +1,25 @@
-{%- macro format_parameters(properties, required) -%}
+{%- macro format_parameters(properties, required, filter_keys=false) -%}
-        {%- if key not in standard_keys -%}
+        {%- if not filter_keys or key not in standard_keys -%}
-            {%- if value['nullable'] %}
-                {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
```

- 已读文件:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +67/-44
- 验证与风险: 该 PR 主要落在文档/示例 `examples/tool_chat_template_gemma4.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #40796 - [Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens

- 链接: https://github.com/vllm-project/vllm/pull/40796
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `c3ad791e1a9a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+62/-1，可读 patch 77 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens」；主要实现面是 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0 (54 lines); hunks: -12,6 +12,60; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt，涉及 `test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt`；`vllm/model_executor/models/gemma4_mm.py` modified +8/-1 (9 lines); hunks: -265,7 +265,14 @@ def _compute_num_soft_tokens(; symbols: _compute_num_soft_tokens, get_image_repl，涉及 `_compute_num_soft_tokens, get_image_repl`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0 (54 lines); hunks: -12,6 +12,60; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt
  - `vllm/model_executor/models/gemma4_mm.py` modified +8/-1 (9 lines); hunks: -265,7 +265,14 @@ def _compute_num_soft_tokens(; symbols: _compute_num_soft_tokens, get_image_repl
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/processing/test_gemma4.py
@@ -12,6 +12,60 @@
+@pytest.mark.parametrize(
+    "image_width,image_height,max_soft_tokens",
+    [
+        # Production repro: a 3x900 image (extreme aspect ratio) made the
+        # prompt-side estimator return 289 while the HF Gemma 4 image
+        # processor's vision tower output capped at 280, producing the
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -265,7 +265,14 @@ def _compute_num_soft_tokens(
-        return num_patches // (pooling_kernel_size**2)
+        # Clamp to ``max_soft_tokens``: extreme aspect ratios (e.g. 3x900)
+        # cause the floor() above to round one dim up to ``unit`` while the
+        # other scales freely, which over-shoots ``max_patches``. The HF
+        # Gemma 4 image processor caps its vision-tower output at
+        # ``max_soft_tokens``, so without this clamp the prompt-side
```

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +8/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41574 - [Model] Fix Gemma4 MoE activation mismatch

- 链接: https://github.com/vllm-project/vllm/pull/41574
- 状态/时间: merged / 2026-05-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `6bb924bbf3a9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+23/-1，可读 patch 130 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix Gemma4 MoE activation mismatch」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Model] Fix Gemma4 MoE activation mismatch」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +1/-1 (2 lines); hunks: -360,7 +360,7 @@ def routing_function(; symbols: routing_function, forward，涉及 `routing_function, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +1/-1 (2 lines); hunks: -360,7 +360,7 @@ def routing_function(; symbols: routing_function, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -360,7 +360,7 @@ def routing_function(
-            activation="gelu",
+            activation="gelu_tanh",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/activation.py`, `vllm/model_executor/layers/fused_moe/fused_batched_moe.py`, `vllm/model_executor/layers/fused_moe/fused_humming_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41799 - [MM][Gemma4] Respect max_soft_tokens in encoder budget

- 链接: https://github.com/vllm-project/vllm/pull/41799
- 状态/时间: merged / 2026-05-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `242afc6bf40d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+91/-11，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Gemma4] Respect max_soft_tokens in encoder budget」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[MM][Gemma4] Respect max_soft_tokens in encoder budget」；主要实现面是 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0 (60 lines); hunks: -2,6 +2,7; -66,6 +67,65 @@ def test_compute_num_soft_tokens_does_not_exceed_max_soft_tok...; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens, test_limit_mm_per_prompt，涉及 `test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens`；`vllm/model_executor/models/gemma4_mm.py` modified +31/-11 (42 lines); hunks: -81,10 +81,26; -216,10 +232,14 @@ def get_mm_max_tokens_per_item(; symbols: _get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor, get_replacement_image，涉及 `_get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0 (60 lines); hunks: -2,6 +2,7; -66,6 +67,65 @@ def test_compute_num_soft_tokens_does_not_exceed_max_soft_tok...; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens, test_limit_mm_per_prompt
  - `vllm/model_executor/models/gemma4_mm.py` modified +31/-11 (42 lines); hunks: -81,10 +81,26; -216,10 +232,14 @@ def get_mm_max_tokens_per_item(; symbols: _get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor, get_replacement_image
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/processing/test_gemma4.py
@@ -2,6 +2,7 @@
+from PIL import Image as PILImage
@@ -66,6 +67,65 @@ def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
+@pytest.mark.parametrize(
+    ("mm_processor_kwargs", "expected_image_tokens"),
+    [
+        ({}, 280),
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -81,10 +81,26 @@
+_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
+def _get_max_soft_tokens(
+    merged_kwargs: Mapping[str, object],
+) -> tuple[object | None, bool]:
+    """Return configured image max_soft_tokens and whether it is top-level."""
+    val = merged_kwargs.get("max_soft_tokens")
```

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +31/-11
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41745 - [Spec Decode] Add Gemma4 MTP speculative decoding support

- 链接: https://github.com/vllm-project/vllm/pull/41745
- 状态/时间: merged / 2026-05-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`；关联提交 `27e0057aeda6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1121/-72，可读 patch 1390 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec Decode] Add Gemma4 MTP speculative decoding support」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`；技术摘要: 覆盖「[Spec Decode] Add Gemma4 MTP speculative decoding support」；主要实现面是 `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mtp.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: Gemma4MTPMaskedEmbedder, __init__, _select_and_score, forward，涉及 `Gemma4MTPMaskedEmbedder, __init__, _select_and_score`；`vllm/v1/spec_decode/gemma4.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Gemma4Proposer, __init__, set_per_group_block_table, model_returns_tuple，涉及 `Gemma4Proposer, __init__, set_per_group_block_table`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mtp.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: Gemma4MTPMaskedEmbedder, __init__, _select_and_score, forward
  - `vllm/v1/spec_decode/gemma4.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Gemma4Proposer, __init__, set_per_group_block_table, model_returns_tuple
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mtp.py
@@ -0,0 +1,603 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Gemma4 MTP (Multi-Token Prediction) model.
+The Gemma4 assistant model is a lightweight decoder that shares KV cache
+with the target (backbone) model.  All assistant decoder layers are
+KV-shared: they only have Q projections (no K/V projections or norms),
diff -- vllm/v1/spec_decode/gemma4.py
@@ -0,0 +1,335 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Gemma4 MTP (Multi-Token Prediction) proposer for speculative decoding.
+The Gemma4 assistant model runs all decoder layers per draft step
+(producing one token), and all its attention layers share KV cache
+with the target model via cross-model KV sharing.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mtp.py` added +603/-0; `vllm/v1/spec_decode/gemma4.py` added +335/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/v1/e2e/spec_decode/test_spec_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41837 - [MM][Gemma4] Use video profiling hints in encoder budget

- 链接: https://github.com/vllm-project/vllm/pull/41837
- 状态/时间: merged / 2026-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `f650ace6de5a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+50/-4，可读 patch 96 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Gemma4] Use video profiling hints in encoder budget」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[MM][Gemma4] Use video profiling hints in encoder budget」；主要实现面是 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0 (35 lines); hunks: -1,6 +1,8; -102,6 +104,39 @@ def test_get_mm_max_tokens_per_item_respects_configured_max...; symbols: test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens，涉及 `test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens`；`vllm/model_executor/models/gemma4_mm.py` modified +9/-1 (10 lines); hunks: -246,7 +246,15 @@ def get_mm_max_tokens_per_item(; symbols: get_mm_max_tokens_per_item, get_data_parser，涉及 `get_mm_max_tokens_per_item, get_data_parser`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0 (35 lines); hunks: -1,6 +1,8; -102,6 +104,39 @@ def test_get_mm_max_tokens_per_item_respects_configured_max...; symbols: test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens
  - `vllm/model_executor/models/gemma4_mm.py` modified +9/-1 (10 lines); hunks: -246,7 +246,15 @@ def get_mm_max_tokens_per_item(; symbols: get_mm_max_tokens_per_item, get_data_parser
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/processing/test_gemma4.py
@@ -1,6 +1,8 @@
+from collections.abc import Mapping
@@ -102,6 +104,39 @@ def test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens(
+@pytest.mark.parametrize(
+    ("limit_mm_per_prompt", "expected_video_tokens"),
+    [
+        ({"video": 1}, 32 * (70 + 2 + 6)),
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -246,7 +246,15 @@ def get_mm_max_tokens_per_item(
-        tokens["video"] = _VIDEO_MAX_FRAMES * (_VIDEO_MAX_SOFT_TOKENS + 2 + 6)
+        num_frames = _VIDEO_MAX_FRAMES
+        mm_config = self.ctx.model_config.get_multimodal_config()
+        video_opts = mm_config.limit_per_prompt.get("video")
+        if (
+            isinstance(video_opts, VideoDummyOptions)
```

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +9/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40588 - [Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP

- 链接: https://github.com/vllm-project/vllm/pull/40588
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `90f145aaf724`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+67/-16，可读 patch 127 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +2/-8 (10 lines); hunks: -35,7 +35,7; -238,13 +238,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +2/-8 (10 lines); hunks: -35,7 +35,7; -238,13 +238,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -35,7 +35,7 @@
-from vllm.model_executor.layers.activation import GeluAndMul
+from vllm.model_executor.layers.activation import get_act_and_mul_fn
@@ -238,13 +238,7 @@ def __init__(
-        if hidden_activation != "gelu_pytorch_tanh":
-            raise ValueError(
-                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation "
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +2/-8
- 验证与风险: diff 自带测试面 `tests/model_executor/test_gemma_hidden_act.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41991 - [Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser

- 链接: https://github.com/vllm-project/vllm/pull/41991
- 状态/时间: merged / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `dbd86a67e3ee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+34/-0，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0 (15 lines); hunks: -135,6 +135,11 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -149,6 +154,16 @@ def test_bare_values(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array, test_string_array，涉及 `test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0 (15 lines); hunks: -135,6 +135,11 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -149,6 +154,16 @@ def test_bare_values(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array, test_string_array
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -135,6 +135,11 @@ def test_empty_value_after_other_keys_partial_withheld(self):
+    @pytest.mark.timeout(5)
+    def test_malformed_partial_array(self):
+        result = _parse_gemma4_args(":[t:[]")
+        assert isinstance(result, dict)
@@ -149,6 +154,16 @@ def test_bare_values(self):
+    @pytest.mark.timeout(5)
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40708 - [BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue

- 链接: https://github.com/vllm-project/vllm/pull/40708
- 状态/时间: merged / 2026-05-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `25abddc1a5cb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+55/-18，可读 patch 100 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +24/-18 (42 lines); hunks: -40,6 +40,7; -1368,30 +1369,35 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +24/-18 (42 lines); hunks: -40,6 +40,7; -1368,30 +1369,35 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -40,6 +40,7 @@
+    fused_moe_make_expert_params_mapping,
@@ -1368,30 +1369,35 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        #
-        # Use prefix matching to handle both weights and
-        # quantization scale parameters. The param_name is a prefix ending
-        # in underscore, and weight_name ends with a dot, so that:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +24/-18
- 验证与风险: diff 自带测试面 `tests/models/quantization/test_awq.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42188 - [Bugfix] Gemma 4 chat template crash with missing tool name and tool id

- 链接: https://github.com/vllm-project/vllm/pull/42188
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`；关联提交 `b1687527b836`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Gemma 4 chat template crash with missing tool name and tool id」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_gemma4.jinja`；技术摘要: 覆盖「[Bugfix] Gemma 4 chat template crash with missing tool name and tool id」；主要实现面是 `examples/tool_chat_template_gemma4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -263,7 +263,7; -277,7 +277,7。
- 代码 diff 细节:
  - `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -263,7 +263,7; -277,7 +277,7
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_gemma4.jinja
@@ -263,7 +263,7 @@
-                    {{- format_tool_response_block(tool_response['name'] | default('unknown'), tool_response['response']) -}}
+                    {{- format_tool_response_block(tool_response['name'] | default('unknown', true), tool_response['response']) -}}
@@ -277,7 +277,7 @@
-                        {%- set ns_tname = namespace(name=follow.get('name') | default('unknown')) -%}
+                        {%- set ns_tname = namespace(name=follow.get('name') | default('unknown', true)) -%}
```

- 已读文件:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/tool_chat_template_gemma4.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #42217 - [Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash

- 链接: https://github.com/vllm-project/vllm/pull/42217
- 状态/时间: merged / 2026-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `630492da308e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+45/-7，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash」；主要实现面是 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0 (33 lines); hunks: -4,9 +4,12; -15,6 +18,36; symbols: test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked，涉及 `test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked`；`vllm/model_executor/models/gemma4_mm.py` modified +12/-7 (19 lines); hunks: -124,12 +124,12 @@ class Gemma4ImagePixelInputs(TensorSchema):; -1128,15 +1128,20 @@ def _process_image_input(; symbols: Gemma4ImagePixelInputs, _process_image_input，涉及 `Gemma4ImagePixelInputs, _process_image_input`。
- 代码 diff 细节:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0 (33 lines); hunks: -4,9 +4,12; -15,6 +18,36; symbols: test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-7 (19 lines); hunks: -124,12 +124,12 @@ class Gemma4ImagePixelInputs(TensorSchema):; -1128,15 +1128,20 @@ def _process_image_input(; symbols: Gemma4ImagePixelInputs, _process_image_input
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/processing/test_gemma4.py
@@ -4,9 +4,12 @@
+import torch
+from vllm.model_executor.models.gemma4_mm import Gemma4ImagePixelInputs
+from vllm.multimodal.inputs import MultiModalFieldConfig
@@ -15,6 +18,36 @@
+def test_gemma4_image_schema_accepts_variable_patch_counts():
+    Gemma4ImagePixelInputs(
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -124,12 +124,12 @@ class Gemma4ImagePixelInputs(TensorSchema):
-        torch.Tensor,
-        TensorShape("bn", "np", "pp"),
+        torch.Tensor | list[torch.Tensor],
+        TensorShape("bn", "np", "pp", dynamic_dims={"np"}),
-        torch.Tensor,
-        TensorShape("bn", "np", 2),
```

- 已读文件:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +12/-7
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42250 - [Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution

- 链接: https://github.com/vllm-project/vllm/pull/42250
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `5794c65f8c36`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-4，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；技术摘要: 覆盖「[Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution」；主要实现面是 `vllm/model_executor/models/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4.py` modified +7/-4 (11 lines); hunks: -326,8 +326,9 @@ def __init__(; -336,10 +337,12 @@ def routing_function(; symbols: __init__, routing_function，涉及 `__init__, routing_function`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4.py` modified +7/-4 (11 lines); hunks: -326,8 +326,9 @@ def __init__(; -336,10 +337,12 @@ def routing_function(; symbols: __init__, routing_function
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -326,8 +326,9 @@ def __init__(
-        per_expert_scale = self.per_expert_scale
+        # NOTE: self.per_expert_scale is read at call time (not captured into
+        # a local) so that torch.func.functional_call parameter substitution
+        # reaches the routing function correctly.
@@ -336,10 +337,12 @@ def routing_function(
-                    gating_output, topk, per_expert_scale
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +7/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42128 - [Bugfix] Fix Gemma4ToolParser streaming float corruption

- 链接: https://github.com/vllm-project/vllm/pull/42128
- 状态/时间: merged / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`；关联提交 `665f9c42535c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+41/-0，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4ToolParser streaming float corruption」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix Gemma4ToolParser streaming float corruption」；主要实现面是 `tests/tool_parsers/test_gemma4_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -135,6 +135,26 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -164,6 +184,15 @@ def test_stray_closing_bracket(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array, test_stray_closing_bracket，涉及 `test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -135,6 +135,26 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -164,6 +184,15 @@ def test_stray_closing_bracket(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array, test_stray_closing_bracket
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -135,6 +135,26 @@ def test_empty_value_after_other_keys_partial_withheld(self):
+    def test_trailing_dot_float_partial_withheld(self):
+        """Bare float ending with '.' is withheld in partial mode.
+        Regression test for #42047: float("108.") → 108.0 causes
+        streaming diff corruption (108.0 → 108.2 becomes 108.02).
+        """
+        # Single key with trailing dot — withheld entirely
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43169 - [Perf][Gemma4] Batch vision encoder calls for image and video processing

- 链接: https://github.com/vllm-project/vllm/pull/43169
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `2b75a73b8e23`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+180/-86，可读 patch 336 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][Gemma4] Batch vision encoder calls for image and video processing」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Perf][Gemma4] Batch vision encoder calls for image and video processing」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +180/-86 (266 lines); hunks: -61,6 +61,7; -960,6 +961,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _process_image_input，涉及 `__init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +180/-86 (266 lines); hunks: -61,6 +61,7; -960,6 +961,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _process_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -61,6 +61,7 @@
+from vllm.platforms import current_platform
@@ -960,6 +961,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Lazy-initialized on first encoder call (see _encoder_max_batch).
+        self._encoder_budget_bytes = 0
+        self._encoder_bytes_per_patch = 0
@@ -1100,6 +1104,19 @@ def _parse_and_validate_multimodal_inputs(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +180/-86
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43296 - [CI] Fix "test_awq_load[gemma4-moe-*]" failure

- 链接: https://github.com/vllm-project/vllm/pull/43296
- 状态/时间: merged / 2026-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；关联提交 `025d4f5cd261`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+116/-31，可读 patch 216 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Fix "test_awq_load[gemma4-moe-*]" failure」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/processing/test_gemma4.py`；技术摘要: 覆盖「[CI] Fix "test_awq_load[gemma4-moe-*]" failure」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/processing/test_gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +54/-30 (84 lines); hunks: -961,9 +961,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -1104,18 +1101,36 @@ def _parse_and_validate_multimodal_inputs(; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _encoder_chunk，涉及 `__init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch`；`tests/models/multimodal/processing/test_gemma4.py` modified +62/-1 (63 lines); hunks: -7,9 +7,13; -224,3 +228,60 @@ def test_limit_mm_per_prompt(; symbols: test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching, test_encoder_chunk_zero_patches_is_safe，涉及 `test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +54/-30 (84 lines); hunks: -961,9 +961,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -1104,18 +1101,36 @@ def _parse_and_validate_multimodal_inputs(; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _encoder_chunk
  - `tests/models/multimodal/processing/test_gemma4.py` modified +62/-1 (63 lines); hunks: -7,9 +7,13; -224,3 +228,60 @@ def test_limit_mm_per_prompt(; symbols: test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching, test_encoder_chunk_zero_patches_is_safe
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -961,9 +961,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        # Lazy-initialized on first encoder call (see _encoder_max_batch).
-        self._encoder_budget_bytes = 0
-        self._encoder_bytes_per_patch = 0
@@ -1104,18 +1101,36 @@ def _parse_and_validate_multimodal_inputs(
-    def _encoder_max_batch(self, patches_per_item: int) -> int:
-        """Max items per encoder call given per-item patch count."""
diff -- tests/models/multimodal/processing/test_gemma4.py
@@ -7,9 +7,13 @@
-from vllm.model_executor.models.gemma4_mm import Gemma4ImagePixelInputs
+from vllm.model_executor.models.gemma4_mm import (
+    Gemma4ForConditionalGeneration,
+    Gemma4ImagePixelInputs,
+)
+from vllm.utils.mem_constants import GiB_bytes
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +54/-30
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +62/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41459 - fix(frontend): Add multimodal placeholders to Gemma4 tool message template

- 链接: https://github.com/vllm-project/vllm/pull/41459
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`, `tests/renderers/test_gemma4_chat_template.py`；关联提交 `69c9f199574e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+78/-0，可读 patch 89 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(frontend): Add multimodal placeholders to Gemma4 tool message template」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/renderers/test_gemma4_chat_template.py`, `examples/tool_chat_template_gemma4.jinja`；技术摘要: 覆盖「fix(frontend): Add multimodal placeholders to Gemma4 tool message template」；主要实现面是 `tests/renderers/test_gemma4_chat_template.py`, `examples/tool_chat_template_gemma4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/renderers/test_gemma4_chat_template.py` modified +69/-0 (69 lines); hunks: -343,3 +343,72 @@ def test_format_argument_types(self, gemma4_template):; symbols: test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities，涉及 `test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities`；`examples/tool_chat_template_gemma4.jinja` modified +9/-0 (9 lines); hunks: -295,6 +295,15。
- 代码 diff 细节:
  - `tests/renderers/test_gemma4_chat_template.py` modified +69/-0 (69 lines); hunks: -343,3 +343,72 @@ def test_format_argument_types(self, gemma4_template):; symbols: test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities
  - `examples/tool_chat_template_gemma4.jinja` modified +9/-0 (9 lines); hunks: -295,6 +295,15
- 关键代码摘录:

```diff
diff -- tests/renderers/test_gemma4_chat_template.py
@@ -343,3 +343,72 @@ def test_format_argument_types(self, gemma4_template):
+    def test_tool_response_with_multimodal_content(self, gemma4_template):
+        """Multimodal placeholders in tool messages are emitted after the
+        tool_response block."""
+        messages = [
+            {"role": "user", "content": "Download the image and describe it."},
+            {
diff -- examples/tool_chat_template_gemma4.jinja
@@ -295,6 +295,15 @@
+                            {%- for part in tool_body -%}
+                                {%- if part.get('type') == 'image' -%}
+                                    {{- '<|image|>' -}}
+                                {%- elif part.get('type') == 'audio' -%}
+                                    {{- '<|audio|>' -}}
+                                {%- elif part.get('type') == 'video' -%}
```

- 已读文件:
  - tests: `tests/renderers/test_gemma4_chat_template.py` modified +69/-0
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +9/-0
- 验证与风险: diff 自带测试面 `tests/renderers/test_gemma4_chat_template.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43909 - [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`

- 链接: https://github.com/vllm-project/vllm/pull/43909
- 状态/时间: merged / 2026-05-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mtp.py`；关联提交 `e1105064b282`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 37 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mtp.py`；技术摘要: 覆盖「[Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`」；主要实现面是 `vllm/model_executor/models/gemma4_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1 (10 lines); hunks: -501,6 +501,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -567,14 +568,20 @@ def forward(; symbols: __init__, forward, _get_full_lm_head_weight, compute_logits，涉及 `__init__, forward, _get_full_lm_head_weight`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1 (10 lines); hunks: -501,6 +501,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -567,14 +568,20 @@ def forward(; symbols: __init__, forward, _get_full_lm_head_weight, compute_logits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mtp.py
@@ -501,6 +501,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self._stable_full_lm_head_weight: torch.Tensor | None = None
@@ -567,14 +568,20 @@ def forward(
+        if self._stable_full_lm_head_weight is not None:
+            return self._stable_full_lm_head_weight
-        return lm_head_weight[: self.masked_embedding.vocab_size]
+        lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43798 - [Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl

- 链接: https://github.com/vllm-project/vllm/pull/43798
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `f91fb2fcf3f1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+111/-11，可读 patch 271 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +51/-10 (61 lines); hunks: -16,7 +16,7; -41,6 +41,7; symbols: __init__, forward, Gemma4ForConditionalGeneration，涉及 `__init__, forward, Gemma4ForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +51/-10 (61 lines); hunks: -16,7 +16,7; -41,6 +41,7; symbols: __init__, forward, Gemma4ForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -16,7 +16,7 @@
-from typing import Annotated, Any, Literal
+from typing import TYPE_CHECKING, Annotated, Any, Literal
@@ -41,6 +41,7 @@
+from vllm.model_executor.models.transformers.utils import recursive_replace_linear
@@ -71,6 +72,7 @@
+    SupportsQuant,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +51/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/bitsandbytes.py`, `vllm/model_executor/model_loader/bitsandbytes_loader.py`, `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44232 - [Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor

- 链接: https://github.com/vllm-project/vllm/pull/44232
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `2fd0e52252f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+19/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +19/-0 (19 lines); hunks: -519,6 +519,25 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only, _call_hf_processor，涉及 `_get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +19/-0 (19 lines); hunks: -519,6 +519,25 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only, _call_hf_processor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -519,6 +519,25 @@ def _get_dummy_videos(
+    def _apply_hf_processor_text_only(
+        self,
+        prompt_text: str,
+        tokenization_kwargs: Mapping[str, object],
+    ) -> list[int]:
+        # Bypass the HF processor and tokenize directly.  The HF
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +19/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44429 - [Model] Add Gemma4 Unified (encoder-free) support

- 链接: https://github.com/vllm-project/vllm/pull/44429
- 状态/时间: merged / 2026-06-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4_mtp.py`, `vllm/model_executor/models/gemma4_unified.py`；关联提交 `a248b45d0548`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+791/-31，可读 patch 1039 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Gemma4 Unified (encoder-free) support」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/gemma4_unified.py`, `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[Model] Add Gemma4 Unified (encoder-free) support」；主要实现面是 `vllm/model_executor/models/gemma4_unified.py`, `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_unified.py` added +466/-0 (466 lines); hunks: -0,0 +1,466; symbols: Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb, forward，涉及 `Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb`；`tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0 (205 lines); hunks: -0,0 +1,205; symbols: test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens，涉及 `test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens`；`vllm/model_executor/models/gemma4_mm.py` modified +64/-19 (83 lines); hunks: -121,7 +121,7 @@ class Gemma4ImagePixelInputs(TensorSchema):; -341,6 +341,29 @@ def get_image_repl(; symbols: Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens, get_audio_repl，涉及 `Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens`；`vllm/model_executor/models/gemma4_mtp.py` modified +19/-3 (22 lines); hunks: -279,11 +279,19 @@ def __init__(; -545,6 +553,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_input_ids, compute_logits, get_top_tokens，涉及 `__init__, embed_input_ids, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_unified.py` added +466/-0 (466 lines); hunks: -0,0 +1,466; symbols: Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb, forward
  - `tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0 (205 lines); hunks: -0,0 +1,205; symbols: test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens
  - `vllm/model_executor/models/gemma4_mm.py` modified +64/-19 (83 lines); hunks: -121,7 +121,7 @@ class Gemma4ImagePixelInputs(TensorSchema):; -341,6 +341,29 @@ def get_image_repl(; symbols: Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens, get_audio_repl
  - `vllm/model_executor/models/gemma4_mtp.py` modified +19/-3 (22 lines); hunks: -279,11 +279,19 @@ def __init__(; -545,6 +553,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_input_ids, compute_logits, get_top_tokens
  - `vllm/model_executor/models/gemma4.py` modified +6/-3 (9 lines); hunks: -1051,11 +1051,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_unified.py
@@ -0,0 +1,466 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Gemma 4 Unified multimodal model (encoder-free image + audio + video).
+The Unified Gemma4 variant has no SigLIP vision tower and no audio tower.
+Raw pixel patches are projected directly to LM space via a Dense+LayerNorm
+pipeline with factorized 2D positional embeddings (Gemma4UnifiedVisionEmbedder),
diff -- tests/models/multimodal/processing/test_gemma4_unified.py
@@ -0,0 +1,205 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Mapping
+import pytest
+import torch
+from PIL import Image as PILImage
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -121,7 +121,7 @@ class Gemma4ImagePixelInputs(TensorSchema):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_unified.py` added +466/-0; `vllm/model_executor/models/gemma4_mm.py` modified +64/-19; `vllm/model_executor/models/gemma4_mtp.py` modified +19/-3; `vllm/model_executor/models/gemma4.py` modified +6/-3
  - tests: `tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_gemma4_unified.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43982 - [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load

- 链接: https://github.com/vllm-project/vllm/pull/43982
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/v1/spec_decode/gemma4.py`；关联提交 `128adabfe0fe`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/v1/spec_decode/gemma4.py`；技术摘要: 覆盖「[Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load」；主要实现面是 `vllm/v1/spec_decode/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/spec_decode/gemma4.py` modified +6/-1 (7 lines); hunks: -81,11 +81,16 @@ def build_per_group_and_layer_attn_metadata(; symbols: build_per_group_and_layer_attn_metadata，涉及 `build_per_group_and_layer_attn_metadata`。
- 代码 diff 细节:
  - `vllm/v1/spec_decode/gemma4.py` modified +6/-1 (7 lines); hunks: -81,11 +81,16 @@ def build_per_group_and_layer_attn_metadata(; symbols: build_per_group_and_layer_attn_metadata
- 关键代码摘录:

```diff
diff -- vllm/v1/spec_decode/gemma4.py
@@ -81,11 +81,16 @@ def build_per_group_and_layer_attn_metadata(
+        batch_size = common_attn_metadata.batch_size()
-                cm.block_table_tensor = self._per_group_block_tables[gid]
+                # Slice to actual batch size to match cu_seqlens_q dimension.
+                # The stored block tables may be padded (num_reqs_padded) from
+                # the target forward pass, but the drafter operates on the
+                # unpadded batch.
```

- 已读文件:
  - runtime: `vllm/v1/spec_decode/gemma4.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/v1/spec_decode/gemma4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43241 - [Model Runner V2][Spec Decode] Add Gemma4 MTP support

- 链接: https://github.com/vllm-project/vllm/pull/43241
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/v1/worker/gpu/spec_decode/gemma4/__init__.py`, `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py`；关联提交 `ceb0111a90ac`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1243/-942，可读 patch 2279 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Runner V2][Spec Decode] Add Gemma4 MTP support」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`；技术摘要: 覆盖「[Model Runner V2][Spec Decode] Add Gemma4 MTP support」；主要实现面是 `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893 (901 lines); hunks: -1,903 +1,18; symbols: EagleSpeculator, __init__, init_cudagraph_manager, load_model，涉及 `EagleSpeculator, __init__, init_cudagraph_manager`；`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0 (795 lines); hunks: -0,0 +1,795; symbols: AutoRegressiveSpeculator, __init__, advance_draft_positions, model_returns_tuple，涉及 `AutoRegressiveSpeculator, __init__, advance_draft_positions`；`vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: BaseSpeculator, init_cudagraph_manager, capture, propose，涉及 `BaseSpeculator, init_cudagraph_manager, capture`；`vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: Gemma4Speculator, advance_draft_positions, model_returns_tuple, load_draft_model，涉及 `Gemma4Speculator, advance_draft_positions, model_returns_tuple`。
- 代码 diff 细节:
  - `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893 (901 lines); hunks: -1,903 +1,18; symbols: EagleSpeculator, __init__, init_cudagraph_manager, load_model
  - `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0 (795 lines); hunks: -0,0 +1,795; symbols: AutoRegressiveSpeculator, __init__, advance_draft_positions, model_returns_tuple
  - `vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: BaseSpeculator, init_cudagraph_manager, capture, propose
  - `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: Gemma4Speculator, advance_draft_positions, model_returns_tuple, load_draft_model
  - `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` added +22/-0 (22 lines); hunks: -0,0 +1,22; symbols: MTPSpeculator, model_returns_tuple, load_draft_model
- 关键代码摘录:

```diff
diff -- vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
@@ -1,903 +1,18 @@
-from typing import Any
-import torch
-from vllm.config import VllmConfig, get_layers_from_vllm_config
-from vllm.config.compilation import CUDAGraphMode
-from vllm.forward_context import BatchDescriptor, set_forward_context
-from vllm.logger import init_logger
diff -- vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
@@ -0,0 +1,795 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import Any
+import torch
+from vllm.config import VllmConfig
+from vllm.config.compilation import CUDAGraphMode
diff -- vllm/v1/worker/gpu/spec_decode/speculator.py
@@ -0,0 +1,224 @@
```

- 已读文件:
  - runtime: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893; `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0; `vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0; `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0; `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` added +22/-0; `vllm/v1/worker/gpu/spec_decode/__init__.py` modified +16/-3
- 验证与风险: runtime 路径改动集中在 `vllm/v1/attention/backends/flashinfer.py`, `vllm/v1/attention/backends/triton_attn.py`, `vllm/v1/attention/backends/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44340 - [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings

- 链接: https://github.com/vllm-project/vllm/pull/44340
- 状态/时间: merged / 2026-06-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+744/-27，可读 patch 1040 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`；技术摘要: 覆盖「[Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings」；主要实现面是 `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__, get_min_capability，涉及 `fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__`；`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int, __init__，涉及 `_dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int`；`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2 (103 lines); hunks: -30,6 +30,9; -45,6 +48,7; symbols: get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel, _is_wNa8o8_int，涉及 `get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel`；`vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5 (66 lines); hunks: -1,5 +1,6; -42,16 +43,57 @@ def humming_is_layer_skipped(config: dict[str, Any], prefix:...; symbols: humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer, prepare_humming_moe_layer，涉及 `humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__, get_min_capability
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int, __init__
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2 (103 lines); hunks: -30,6 +30,9; -45,6 +48,7; symbols: get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel, _is_wNa8o8_int
  - `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5 (66 lines); hunks: -1,5 +1,6; -42,16 +43,57 @@ def humming_is_layer_skipped(config: dict[str, Any], prefix:...; symbols: humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer, prepare_humming_moe_layer
  - `vllm/model_executor/kernels/linear/mixed_precision/humming.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: HummingLinearKernel, get_min_capability, can_implement, process_weights_after_loading
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py
@@ -0,0 +1,257 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Weight N-bit INT scheme with static INT8 input/output activation quant.
+Handles compressed-tensors INT weight checkpoints that carry static per-tensor
+INT8 ``input_activations`` and/or ``output_activations``. The activation quant is
+reproduced as a float fake-quant on the layer input and output, around a
diff -- vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py
@@ -0,0 +1,170 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Quantized embedding method for compressed-tensors.
+Adds dequant-on-lookup support for a pack-quantized ``VocabParallelEmbedding``
+(2-8 bit INT, channel- or group-quantized). Only the gathered token rows are
+unpacked and dequantized, so the packed weight is never densified.
diff -- vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
@@ -30,6 +30,9 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5; `vllm/model_executor/kernels/linear/mixed_precision/humming.py` added +61/-0; `vllm/model_executor/models/gemma4_mm.py` modified +12/-12
- 验证与风险: diff 自带测试面 `requirements/test/rocm.txt`, `tests/kernels/quantization/test_quantized_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44571 - [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified

- 链接: https://github.com/vllm-project/vllm/pull/44571
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_unified.py`；关联提交 `da1daf40bf18`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Exclude vision embedder from quantization in Gemma4 Unified」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_unified.py`；技术摘要: 覆盖「[Bugfix] Exclude vision embedder from quantization in Gemma4 Unified」；主要实现面是 `vllm/model_executor/models/gemma4_unified.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_unified.py` modified +3/-1 (4 lines); hunks: -80,7 +80,7 @@ class Gemma4UnifiedVisionEmbedder(nn.Module):; -91,6 +91,7 @@ def __init__(self, config, quant_config=None):; symbols: Gemma4UnifiedVisionEmbedder, __init__，涉及 `Gemma4UnifiedVisionEmbedder, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_unified.py` modified +3/-1 (4 lines); hunks: -80,7 +80,7 @@ class Gemma4UnifiedVisionEmbedder(nn.Module):; -91,6 +91,7 @@ def __init__(self, config, quant_config=None):; symbols: Gemma4UnifiedVisionEmbedder, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_unified.py
@@ -80,7 +80,7 @@ class Gemma4UnifiedVisionEmbedder(nn.Module):
-    def __init__(self, config, quant_config=None):
+    def __init__(self, config, quant_config=None, prefix=""):
@@ -91,6 +91,7 @@ def __init__(self, config, quant_config=None):
+            prefix=f"{prefix}.patch_dense",
@@ -267,6 +268,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+                prefix=maybe_prefix(prefix, "vision_embedder"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_unified.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_unified.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- 链接: https://github.com/vllm-project/vllm/pull/43167
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+88/-731，可读 patch 1251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove KV cache scale boilerplate from model weight loading methods」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`；技术摘要: 覆盖「Remove KV cache scale boilerplate from model weight loading methods」；主要实现面是 `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name，涉及 `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`；`vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader，涉及 `_get_moe_weight_dtype, kv_cache_scale_loader`；`vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod，涉及 `KVCacheScaleParameter, __new__, weight_loader`；`vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter，涉及 `get_quant_method, get_cache_scale, get_cache_scale_mapper`。
- 代码 diff 细节:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- 验证与风险: diff 自带测试面 `tests/model_executor/test_eagle_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44828 - [BugFix] Use served model name in gemma4 audio-tower error message

- 链接: https://github.com/vllm-project/vllm/pull/44828
- 状态/时间: merged / 2026-06-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `469f3dcf1d70`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Use served model name in gemma4 audio-tower error message」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；技术摘要: 覆盖「[BugFix] Use served model name in gemma4 audio-tower error message」；主要实现面是 `vllm/model_executor/models/gemma4_mm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` modified +5/-1 (6 lines); hunks: -34,6 +34,7; -217,7 +218,10 @@ def validate_num_items(self, modality: str, num_items: int)...; symbols: validate_num_items，涉及 `validate_num_items`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` modified +5/-1 (6 lines); hunks: -34,6 +34,7; -217,7 +218,10 @@ def validate_num_items(self, modality: str, num_items: int)...; symbols: validate_num_items
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -34,6 +34,7 @@
+from vllm.config.model import get_served_model_name
@@ -217,7 +218,10 @@ def validate_num_items(self, modality: str, num_items: int) -> None:
-            model = self.ctx.model_config.model
+            model_config = self.ctx.model_config
+            model = get_served_model_name(
+                model_config.model, model_config.served_model_name
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/gemma4_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- 链接: https://github.com/vllm-project/vllm/pull/41184
- 状态/时间: merged / 2026-06-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 90 个文件，+2734/-2027，可读 patch 7329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`；技术摘要: 覆盖「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts，涉及 `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`；`vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method，涉及 `FusedMoeWeightScaleSupported, RoutedExperts, __init__`；`vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward，涉及 `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`；`vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__，涉及 `FusedMoEWithLoRA, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1,1424 +1,404 @@
-from collections.abc import Callable, Iterable
-from enum import Enum
-from typing import Literal, cast, overload
+from collections.abc import Callable
+from typing import Any
-from torch.nn.parameter import UninitializedParameter
diff -- vllm/model_executor/layers/fused_moe/routed_experts.py
@@ -0,0 +1,1144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable, Iterable
+from enum import Enum
+from typing import TYPE_CHECKING, Any, Literal, cast, overload
+import torch
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@ -1,28 +1,39 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- 验证与风险: diff 自带测试面 `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45163 - [Model] Add DiffusionGemma Support

- 链接: https://github.com/vllm-project/vllm/pull/45163
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 52 个文件，+2698/-235，可读 patch 3935 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add DiffusionGemma Support」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/models/config.py`；技术摘要: 覆盖「[Model] Add DiffusionGemma Support」；主要实现面是 `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/models/config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39 (184 lines); hunks: -20,9 +20,11; -343,6 +345,9 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__, _reset_streaming_state, adjust_request，涉及 `Gemma4ToolParser, __init__, _reset_streaming_state`；`tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0 (82 lines); hunks: -702,6 +702,88 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call, test_streaming_multi_chunk_batched_tool_calls，涉及 `test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call`；`vllm/model_executor/models/config.py` modified +55/-0 (55 lines); hunks: -105,6 +105,60 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; -591,6 +645,7 @@ def verify_and_update_model_config(model_config: "ModelConfi...; symbols: verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig, verify_and_update_model_config，涉及 `verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig`；`tests/models/registry.py` modified +4/-0 (4 lines); hunks: -901,6 +901,10 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`。
- 代码 diff 细节:
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39 (184 lines); hunks: -20,9 +20,11; -343,6 +345,9 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__, _reset_streaming_state, adjust_request
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0 (82 lines); hunks: -702,6 +702,88 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call, test_streaming_multi_chunk_batched_tool_calls
  - `vllm/model_executor/models/config.py` modified +55/-0 (55 lines); hunks: -105,6 +105,60 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; -591,6 +645,7 @@ def verify_and_update_model_config(model_config: "ModelConfi...; symbols: verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig, verify_and_update_model_config
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -901,6 +901,10 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/utils.py` modified +3/-1 (4 lines); hunks: -486,6 +486,7 @@ def dummy_hf_overrides(; -558,7 +559,8 @@ class DummyConfig:; symbols: dummy_hf_overrides, DummyConfig
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -20,9 +20,11 @@
+from openai.types.responses import ToolChoiceFunction
+    ChatCompletionNamedToolChoiceParam,
@@ -343,6 +345,9 @@ class Gemma4ToolParser(ToolParser):
+    # Gemma4 emits native special-token tool calls, not generic JSON calls.
+    supports_required_and_named = False
@@ -390,6 +395,23 @@ def _reset_streaming_state(self) -> None:
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -702,6 +702,88 @@ def test_streaming_html_argument_does_not_duplicate_tag_prefixes(
+    def _collect_tool_calls_by_index(self, results):
+        """Group streamed tool-call fragments by their ``index``.
+        Returns ``{index: {"name": str | None, "arguments": str}}`` where
+        ``arguments`` is the concatenation of every streamed argument
+        fragment for that index (which should form valid JSON once complete).
+        """
diff -- vllm/model_executor/models/config.py
@@ -105,6 +105,60 @@ def verify_and_update_config(vllm_config: "VllmConfig") -> None:
```

- 已读文件:
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39; `vllm/model_executor/models/config.py` modified +55/-0; `vllm/model_executor/models/gemma4.py` modified +1/-3; `vllm/model_executor/models/registry.py` modified +4/-0; `vllm/transformers_utils/configs/__init__.py` modified +4/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0; `tests/models/registry.py` modified +4/-0; `tests/models/utils.py` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_mixed_causal_attn.py`, `tests/models/registry.py`, `tests/models/utils.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45588 - [Frontend] Replace legacy Gemma4 parsers with engine-based implementation

- 链接: https://github.com/vllm-project/vllm/pull/45588
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py` 等 7 个文件；关联提交 `76a373eff47a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+2808/-1332，可读 patch 4822 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Replace legacy Gemma4 parsers with engine-based implementation」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`；技术摘要: 覆盖「[Frontend] Replace legacy Gemma4 parsers with engine-based implementation」；主要实现面是 `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0 (1201 lines); hunks: -0,0 +1,1201; symbols: _make_tokenizer, decode, _stream_tokens_batched, _collect_fields，涉及 `_make_tokenizer, decode, _stream_tokens_batched`；`tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51 (189 lines); hunks: -8,31 +8,105; -49,6 +123,9 @@ def mock_request():; symbols: _make_tool, mock_tokenizer, parser, mock_request，涉及 `_make_tool, mock_tokenizer, parser`；`tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4 (8 lines); hunks: -83,15 +83,15 @@ def generic_tokenizer():; -111,7 +111,7 @@ def generic_tokenizer():; symbols: generic_tokenizer，涉及 `generic_tokenizer`；`vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: Gemma4EngineToolParser，涉及 `Gemma4EngineToolParser`。
- 代码 diff 细节:
  - `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0 (1201 lines); hunks: -0,0 +1,1201; symbols: _make_tokenizer, decode, _stream_tokens_batched, _collect_fields
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51 (189 lines); hunks: -8,31 +8,105; -49,6 +123,9 @@ def mock_request():; symbols: _make_tool, mock_tokenizer, parser, mock_request
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4 (8 lines); hunks: -83,15 +83,15 @@ def generic_tokenizer():; -111,7 +111,7 @@ def generic_tokenizer():; symbols: generic_tokenizer
  - `vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: Gemma4EngineToolParser
  - `vllm/reasoning/gemma4_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
- 关键代码摘录:

```diff
diff -- tests/parser/engine/test_gemma4_streaming_reasoning.py
@@ -0,0 +1,1201 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for the unified Gemma4 parser engine."""
+import json
+from unittest.mock import MagicMock
+import pytest
diff -- tests/tool_parsers/test_gemma4_tool_parser.py
@@ -8,31 +8,105 @@
-from vllm.tool_parsers.gemma4_tool_parser import (
+from vllm.parser.gemma4 import (
-    Gemma4ToolParser,
+from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
+TOOL_CALL_START_ID = 48
+TOOL_CALL_END_ID = 49
diff -- tests/reasoning/test_gemma4_reasoning_parser.py
@@ -83,15 +83,15 @@ def generic_tokenizer():
```

- 已读文件:
  - tests: `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51; `tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4; `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +13/-6
  - runtime: `vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0; `vllm/reasoning/gemma4_engine_reasoning_parser.py` added +6/-0; `vllm/parser/gemma4.py` added +557/-0
- 验证与风险: diff 自带测试面 `tests/parser/engine/replay_harness.py`, `tests/parser/engine/test_delegating_replay.py`, `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/parser/engine/test_parser_engine.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45553 - [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync

- 链接: https://github.com/vllm-project/vllm/pull/45553
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `vllm/parser/gemma4.py`, `vllm/tool_parsers/gemma4_utils.py`；关联提交 `6607a80dabfa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+94/-74，可读 patch 343 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/gemma4_utils.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `examples/tool_chat_template_gemma4.jinja`；技术摘要: 覆盖「[Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync」；主要实现面是 `vllm/tool_parsers/gemma4_utils.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `examples/tool_chat_template_gemma4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/gemma4_utils.py` modified +7/-28 (35 lines); hunks: -35,8 +35,6; -52,42 +50,23; symbols: _parse_tool_arguments, parse_tool_calls，涉及 `_parse_tool_arguments, parse_tool_calls`；`tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7 @@ def generic_tokenizer():; symbols: generic_tokenizer，涉及 `generic_tokenizer`；`examples/tool_chat_template_gemma4.jinja` modified +64/-40 (104 lines); hunks: -116,7 +116,9; -172,18 +174,21；`vllm/parser/gemma4.py` modified +20/-3 (23 lines); hunks: -423,6 +423,8 @@ def __init__(; -437,6 +439,21 @@ def __init__(; symbols: __init__, adjust_request, _reset, is_reasoning_end，涉及 `__init__, adjust_request, _reset`。
- 代码 diff 细节:
  - `vllm/tool_parsers/gemma4_utils.py` modified +7/-28 (35 lines); hunks: -35,8 +35,6; -52,42 +50,23; symbols: _parse_tool_arguments, parse_tool_calls
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7 @@ def generic_tokenizer():; symbols: generic_tokenizer
  - `examples/tool_chat_template_gemma4.jinja` modified +64/-40 (104 lines); hunks: -116,7 +116,9; -172,18 +174,21
  - `vllm/parser/gemma4.py` modified +20/-3 (23 lines); hunks: -423,6 +423,8 @@ def __init__(; -437,6 +439,21 @@ def __init__(; symbols: __init__, adjust_request, _reset, is_reasoning_end
  - `tests/renderers/test_gemma4_chat_template.py` modified +2/-2 (4 lines); hunks: -358,7 +358,7 @@ def test_tool_response_with_multimodal_content(self, gemma4_...; -392,7 +392,7 @@ def test_tool_response_with_all_modalities(self, gemma4_temp...; symbols: test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/gemma4_utils.py
@@ -35,8 +35,6 @@
-import json
@@ -52,42 +50,23 @@
-    Handles the ``key:<|"|>value<|"|>`` format used by Gemma4, with fallback
-    to heuristic key-value extraction. Also tolerates the slightly different
-    ``key: "value"`` format (space + plain quotes) that some chat templates
-    produce.
diff -- tests/reasoning/test_gemma4_reasoning_parser.py
@@ -54,7 +54,7 @@ def generic_tokenizer():
-    "is_reasoning_end": False,
+    "is_reasoning_end": True,
diff -- examples/tool_chat_template_gemma4.jinja
@@ -116,7 +116,9 @@
-    {%- if argument is string -%}
+    {%- if argument is none -%}
+        {{- 'null' -}}
+    {%- elif argument is string -%}
```

- 已读文件:
  - runtime: `vllm/tool_parsers/gemma4_utils.py` modified +7/-28; `vllm/parser/gemma4.py` modified +20/-3
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1; `tests/renderers/test_gemma4_chat_template.py` modified +2/-2
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +64/-40
- 验证与风险: diff 自带测试面 `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45795 - [Bugfix] Gemma4: skip forced JSON for required/named tool choice

- 链接: https://github.com/vllm-project/vllm/pull/45795
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/tool_parsers/gemma4_engine_tool_parser.py`；关联提交 `b9684d99e9ba`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+120/-1，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Gemma4: skip forced JSON for required/named tool choice」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`；技术摘要: 覆盖「[Bugfix] Gemma4: skip forced JSON for required/named tool choice」；主要实现面是 `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0 (28 lines); hunks: -1,8 +1,36; symbols: Gemma4EngineToolParser, adjust_request，涉及 `Gemma4EngineToolParser, adjust_request`；`tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1 (93 lines); hunks: -20,6 +20,13; -28,6 +35,7; symbols: _get_weather_tool, _build_responses_request, _build_chat_request，涉及 `_get_weather_tool, _build_responses_request, _build_chat_request`。
- 代码 diff 细节:
  - `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0 (28 lines); hunks: -1,8 +1,36; symbols: Gemma4EngineToolParser, adjust_request
  - `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1 (93 lines); hunks: -20,6 +20,13; -28,6 +35,7; symbols: _get_weather_tool, _build_responses_request, _build_chat_request
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/gemma4_engine_tool_parser.py
@@ -1,8 +1,36 @@
+from openai.types.responses import ToolChoiceFunction
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionNamedToolChoiceParam,
+    ChatCompletionRequest,
+)
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
diff -- tests/tool_use/test_gemma4_responses_adjust_request.py
@@ -20,6 +20,13 @@
+3. :class:`Gemma4EngineToolParser` (the engine-based parser, #45588) sets
+   ``supports_required_and_named=False`` but did not skip the forced
+   ``structured_outputs`` JSON for ``required``/named tool choice. The model
+   was constrained to JSON the native parser cannot read, so the call leaked
+   as content with empty ``tool_calls``. ``adjust_request`` now skips that
+   constraint so Gemma4 emits its native ``<|tool_call>`` syntax.
```

- 已读文件:
  - runtime: `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0
  - tests: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1
- 验证与风险: diff 自带测试面 `tests/tool_use/test_gemma4_responses_adjust_request.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45832 - [Bugfix][Gemma4] Fix parsing when thinking is disabled

- 链接: https://github.com/vllm-project/vllm/pull/45832
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`；关联提交 `b831374cf1db`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+70/-28，可读 patch 120 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma4] Fix parsing when thinking is disabled」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`；技术摘要: 覆盖「[Bugfix][Gemma4] Fix parsing when thinking is disabled」；主要实现面是 `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21 (78 lines); hunks: -68,28 +68,33 @@ def _build_responses_request(*, tool_choice: str | dict[str,...; -212,3 +217,34 @@ def test_gemma4_named_skips_structured_outputs_responses()...; symbols: _build_responses_request, _build_chat_request, _StubTokenizer, test_gemma4_named_skips_structured_outputs_responses，涉及 `_build_responses_request, _build_chat_request, _StubTokenizer`；`vllm/parser/gemma4.py` modified +13/-7 (20 lines); hunks: -443,16 +443,22 @@ def adjust_request(; symbols: adjust_request, _reset，涉及 `adjust_request, _reset`。
- 代码 diff 细节:
  - `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21 (78 lines); hunks: -68,28 +68,33 @@ def _build_responses_request(*, tool_choice: str | dict[str,...; -212,3 +217,34 @@ def test_gemma4_named_skips_structured_outputs_responses()...; symbols: _build_responses_request, _build_chat_request, _StubTokenizer, test_gemma4_named_skips_structured_outputs_responses
  - `vllm/parser/gemma4.py` modified +13/-7 (20 lines); hunks: -443,16 +443,22 @@ def adjust_request(; symbols: adjust_request, _reset
- 关键代码摘录:

```diff
diff -- tests/tool_use/test_gemma4_responses_adjust_request.py
@@ -68,28 +68,33 @@ def _build_responses_request(*, tool_choice: str | dict[str, Any]) -> ResponsesR
-def _build_chat_request(*, tool_choice: str | dict[str, Any]) -> ChatCompletionRequest:
-    return ChatCompletionRequest.model_validate(
-        {
-            "model": "gemma4-test",
-            "messages": [{"role": "user", "content": "What is the weather in Hanoi?"}],
-            "tools": [
diff -- vllm/parser/gemma4.py
@@ -443,16 +443,22 @@ def adjust_request(
-        """Skip ``skip_special_tokens=False`` when thinking is disabled.
+        """Keep special tokens when thinking or tool calls need them.
-        When there are no reasoning channel tokens to preserve,
-        keeping the default prevents tool-call delimiter tokens
-        from leaking into content (e.g. with ``tool_choice="none"``).
+        ``skip_special_tokens`` must stay ``False`` when there is something to
```

- 已读文件:
  - tests: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21
  - runtime: `vllm/parser/gemma4.py` modified +13/-7
- 验证与风险: diff 自带测试面 `tests/tool_use/test_gemma4_responses_adjust_request.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45852 - [Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)

- 链接: https://github.com/vllm-project/vllm/pull/45852
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`；关联提交 `3c6084bb0d51`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+270/-0，可读 patch 353 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`；技术摘要: 覆盖「[Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)」；主要实现面是 `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0 (208 lines); hunks: -25,12 +25,14; -253,6 +255,212 @@ def test_reasoning_extracted(self, parser, mock_tokenizer,...; symbols: test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer, open_reasoning_parser，涉及 `test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer`；`vllm/parser/gemma4.py` modified +28/-0 (28 lines); hunks: -353,6 +353,14 @@ def gemma4_config() -> ParserEngineConfig:; -518,6 +526,26 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt, _events_to_delta，涉及 `gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt`。
- 代码 diff 细节:
  - `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0 (208 lines); hunks: -25,12 +25,14; -253,6 +255,212 @@ def test_reasoning_extracted(self, parser, mock_tokenizer,...; symbols: test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer, open_reasoning_parser
  - `vllm/parser/gemma4.py` modified +28/-0 (28 lines); hunks: -353,6 +353,14 @@ def gemma4_config() -> ParserEngineConfig:; -518,6 +526,26 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt, _events_to_delta
- 关键代码摘录:

```diff
diff -- tests/parser/engine/test_gemma4_streaming_reasoning.py
@@ -25,12 +25,14 @@
+NEW_TURN_ID = 53  # <|turn>
+    NEW_TURN_ID: "<|turn>",
@@ -253,6 +255,212 @@ def test_reasoning_extracted(self, parser, mock_tokenizer, request_obj):
+# ── Prompt ends inside an open <|channel>thought\n block ─────────────
+_OPEN_REASONING_GEN_SEQUENCE: list[tuple[int, str]] = [
+    (7001, "Sure"),
diff -- vllm/parser/gemma4.py
@@ -353,6 +353,14 @@ def gemma4_config() -> ParserEngineConfig:
+            # No-op: if we pre-initialised the engine to REASONING from the
+            # prompt (see ``adjust_initial_state_from_prompt``) but the model
+            # still emits its own ``<|channel>`` opener, swallow it instead
+            # of leaking it as TEXT_CHUNK.
+            (ParserState.REASONING, "THINK_START"): Transition(
+                ParserState.REASONING,
```

- 已读文件:
  - tests: `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0
  - runtime: `vllm/parser/gemma4.py` modified +28/-0
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_gemma4_streaming_reasoning.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45867 - [Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls

- 链接: https://github.com/vllm-project/vllm/pull/45867
- 状态/时间: merged / 2026-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`；关联提交 `58b2e896423f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_gemma4.jinja`；技术摘要: 覆盖「[Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls」；主要实现面是 `examples/tool_chat_template_gemma4.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -231,10 +231,10。
- 代码 diff 细节:
  - `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -231,10 +231,10
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_gemma4.jinja
@@ -231,10 +231,10 @@
-    {#- Render reasoning/reasoning_content as thinking channel (tool-call turns only) -#}
+    {#- Render reasoning/reasoning_content as thinking channel -#}
-    {%- if thinking_text and thinking_gate and message.get('tool_calls') -%}
+    {%- if thinking_text and thinking_gate -%}
```

- 已读文件:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/tool_chat_template_gemma4.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
