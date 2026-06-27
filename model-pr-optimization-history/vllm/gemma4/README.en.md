# vllm Gemma 4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
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

## PR Coverage Summary

- Git-traced PRs: 49
- Extra PRs preserved from existing docs: 5
- Total PRs in this document: 54
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- Link: https://github.com/vllm-project/vllm/pull/38826
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`, `vllm/model_executor/models/gemma4.py` and 8 files; associated commits `08ed2b9688b4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +5051/-1, 5167 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)"; model line: Gemma 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo, touching `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs`; `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward, touching `_get_text_config, Gemma4MLP, __init__`; `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args, touching `mock_tokenizer, parser, mock_request`; `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments, touching `parse_thinking_output, _strip_thought_label, _clean_answer`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
  - `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0 (196 lines); hunks: -0,0 +1,196; symbols: generic_tokenizer, test_gemma4_reasoning, _encode
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0; `vllm/model_executor/models/gemma4.py` added +1239/-0; `vllm/model_executor/models/gemma4_utils.py` added +292/-0; `vllm/tool_parsers/gemma4_utils.py` added +183/-0; `vllm/reasoning/gemma4_utils.py` added +130/-0; `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` added +84/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0; `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/registry.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38847 - [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter

- Link: https://github.com/vllm-project/vllm/pull/38847
- Status/date: merged / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter"; model line: Gemma 4; category: bug fix; main diff: `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "[Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter"; the main implementation surface is `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3 (6 lines); hunks: -38,7 +38,7; -281,8 +281,8 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__, touching `Gemma4ToolParser, __init__`.
- Code diff details:
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3 (6 lines); hunks: -38,7 +38,7; -281,8 +281,8 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/gemma4_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38872 - [Misc] Clean up Gemma4 implementation

- Link: https://github.com/vllm-project/vllm/pull/38872
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `550643541956`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +5/-300, 333 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up Gemma4 implementation"; model line: Gemma 4; category: model implementation change; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Misc] Clean up Gemma4 implementation"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +3/-6 (9 lines); hunks: -15,7 +15,6; -480,12 +479,10 @@ def _call_hf_processor(; symbols: _call_hf_processor, touching `_call_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +3/-6 (9 lines); hunks: -15,7 +15,6; -480,12 +479,10 @@ def _call_hf_processor(; symbols: _call_hf_processor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +3/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4_utils.py`, `vllm/transformers_utils/model_arch_config_convertor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38992 - [Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters

- Link: https://github.com/vllm-project/vllm/pull/38992
- Status/date: merged / 2026-04-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `f53fa26e05c4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +33/-3, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json, touching `test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38879 - [Gemma4] Enable Fast Prefill Optimization

- Link: https://github.com/vllm-project/vllm/pull/38879
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `47e605092b7f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +369/-47, 490 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Gemma4] Enable Fast Prefill Optimization"; model line: Gemma 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Gemma4] Enable Fast Prefill Optimization"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunks: -19,6 +19,7; -32,6 +33,7; symbols: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__, touching `forward, _run_decoder_layers, Gemma4SelfDecoderLayers`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunks: -19,6 +19,7; -32,6 +33,7; symbols: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +369/-47
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38909 - [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls

- Link: https://github.com/vllm-project/vllm/pull/38909
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `d734445fcd79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +64/-2, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes, touching `test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39114 - [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values

- Link: https://github.com/vllm-project/vllm/pull/39114
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `13151a4df43d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +78/-8, 159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks, touching `test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39027 - [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes

- Link: https://github.com/vllm-project/vllm/pull/39027
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `8477fe427d17`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +878/-16, 1083 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes"; model line: Gemma 4; category: bug fix; main diff: `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/renderers/test_gemma4_chat_template.py`; technical summary: Covers "[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes"; the main implementation surface is `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/renderers/test_gemma4_chat_template.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode, touching `generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output`; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array, touching `test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld`; `tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled, touching `gemma4_template, _render, TestGemma4ChatTemplate`; `examples/tool_chat_template_gemma4.jinja` added +331/-0 (331 lines); hunks: -0,0 +1,331.
- Code diff details:
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array
  - `tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled
  - `examples/tool_chat_template_gemma4.jinja` added +331/-0 (331 lines); hunks: -0,0 +1,331
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0; `tests/renderers/test_gemma4_chat_template.py` added +345/-0
  - docs: `examples/tool_chat_template_gemma4.jinja` added +331/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39045 - [Gemma4] Support quantized MoE

- Link: https://github.com/vllm-project/vllm/pull/39045
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `3aecdf08b4a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-14, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Gemma4] Support quantized MoE"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Gemma4] Support quantized MoE"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunks: -1248,21 +1248,27 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; -1322,9 +1328,21 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _weight_iterator, touching `load_weights, _weight_iterator`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunks: -1248,21 +1248,27 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; -1322,9 +1328,21 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, _weight_iterator
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +34/-14
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39450 - Add Gemma4 Eagle3 support

- Link: https://github.com/vllm-project/vllm/pull/39450
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `e7cfd7c5b9a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +43/-10, 146 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Gemma4 Eagle3 support"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "Add Gemma4 Eagle3 support"; the main implementation surface is `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunks: -60,7 +60,13; -838,7 +844,7 @@ def forward(; symbols: forward, Gemma4Model, __init__, touching `forward, Gemma4Model, __init__`; `vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunks: -64,7 +64,12; -845,7 +850,12 @@ def forward(self, inputs_embeds: torch.Tensor) -> torch.Ten...; symbols: forward, Gemma4ForConditionalGeneration, touching `forward, Gemma4ForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunks: -60,7 +60,13; -838,7 +844,7 @@ def forward(; symbols: forward, Gemma4Model, __init__
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunks: -64,7 +64,12; -845,7 +850,12 @@ def forward(self, inputs_embeds: torch.Tensor) -> torch.Ten...; symbols: forward, Gemma4ForConditionalGeneration
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +20/-5; `vllm/model_executor/models/gemma4_mm.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38844 - [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly

- Link: https://github.com/vllm-project/vllm/pull/38844
- Status/date: merged / 2026-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `92feb9991d15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +40/-0, 66 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunks: -69,6 +69,7; -1397,6 +1398,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Gemma4ForCausalLM, touching `load_weights, Gemma4ForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunks: -69,6 +69,7; -1397,6 +1398,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch...; symbols: load_weights, Gemma4ForCausalLM
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +17/-0
- Risk and verification: The diff ships test coverage in `tests/lora/test_lora_checkpoints.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39679 - [Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`

- Link: https://github.com/vllm-project/vllm/pull/39679
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `b075604da10a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types, touching `test_boolean_false, test_null_value, test_mixed_types`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39842 - [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models

- Link: https://github.com/vllm-project/vllm/pull/39842
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `6dc949140693`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +7/-2 (9 lines); hunks: -167,10 +167,15 @@ def get_default_tok_params(self):; symbols: get_default_tok_params, get_hf_processor, touching `get_default_tok_params, get_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +7/-2 (9 lines); hunks: -167,10 +167,15 @@ def get_default_tok_params(self):; symbols: get_default_tok_params, get_hf_processor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39234 - [Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`

- Link: https://github.com/vllm-project/vllm/pull/39234
- Status/date: merged / 2026-04-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `b1dc87a0989f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`"; model line: Gemma 4; category: model implementation change; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +3/-2 (5 lines); hunks: -1254,9 +1254,10 @@ def embed_input_ids(; symbols: embed_input_ids, touching `embed_input_ids`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +3/-2 (5 lines); hunks: -1254,9 +1254,10 @@ def embed_input_ids(; symbols: embed_input_ids
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gemma4_mm.py
@@ -1254,9 +1254,10 @@ def embed_input_ids(
-                is_multimodal = is_multimodal.to(input_ids.device)
-                    is_multimodal, torch.zeros_like(input_ids), input_ids
+                    is_multimodal.to(input_ids.device, non_blocking=True),
+                    torch.zeros_like(input_ids),
+                    input_ids,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39291 - feat: Add LoRA support for Gemma4ForConditionalGeneration

- Link: https://github.com/vllm-project/vllm/pull/39291
- Status/date: merged / 2026-04-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `640cc9dd7dae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-2, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Add LoRA support for Gemma4ForConditionalGeneration"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "feat: Add LoRA support for Gemma4ForConditionalGeneration"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +10/-2 (12 lines); hunks: -67,6 +67,7; -880,6 +881,7 @@ class Gemma4ForConditionalGeneration(; symbols: Gemma4ForConditionalGeneration, load_weights, get_mm_mapping, touching `Gemma4ForConditionalGeneration, load_weights, get_mm_mapping`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +10/-2 (12 lines); hunks: -67,6 +67,7; -880,6 +881,7 @@ class Gemma4ForConditionalGeneration(; symbols: Gemma4ForConditionalGeneration, load_weights, get_mm_mapping
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39083 - [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton

- Link: https://github.com/vllm-project/vllm/pull/39083
- Status/date: merged / 2026-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/kernels/moe/test_gemma4router.py`, `vllm/model_executor/models/gemma4.py`; associated commits `45232a454e4c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +180/-16, 226 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton"; model line: Gemma 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py`; technical summary: Covers "[FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton"; the main implementation surface is `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +122/-16 (138 lines); hunks: -57,7 +57,9; -79,6 +81,120; symbols: _gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch, _get_text_config, touching `_gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch`; `tests/kernels/moe/test_gemma4router.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: sort_by_id, test_gemma4_routing_kernel_triton, touching `sort_by_id, test_gemma4_routing_kernel_triton`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +122/-16 (138 lines); hunks: -57,7 +57,9; -79,6 +81,120; symbols: _gemma4_routing_kernel, gemma4_fused_routing_kernel_triton, gemma4_routing_function_torch, _get_text_config
  - `tests/kernels/moe/test_gemma4router.py` added +57/-0 (57 lines); hunks: -0,0 +1,57; symbols: sort_by_id, test_gemma4_routing_kernel_triton
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +122/-16
  - tests: `tests/kernels/moe/test_gemma4router.py` added +57/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_gemma4router.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40411 - [Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference

- Link: https://github.com/vllm-project/vllm/pull/40411
- Status/date: merged / 2026-04-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `20d37434911d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-8, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +9/-8 (17 lines); hunks: -849,22 +849,23 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +9/-8 (17 lines); hunks: -849,22 +849,23 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +9/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40534 - [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard

- Link: https://github.com/vllm-project/vllm/pull/40534
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `512f52219240`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +73/-1, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Gemma4: add bidirectional vision attention for sliding layers with window guard"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Model] Gemma4: add bidirectional vision attention for sliding layers with window guard"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +59/-0 (59 lines); hunks: -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1310,6 +1320,12 @@ def forward(; symbols: __init__, forward, compute_logits, _clear_mm_prefix_for_full_attn_layers, touching `__init__, forward, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +59/-0 (59 lines); hunks: -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; -1310,6 +1320,12 @@ def forward(; symbols: __init__, forward, compute_logits, _clear_mm_prefix_for_full_attn_layers
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +59/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/worker/gpu_model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40786 - Fix PP in Gemma4

- Link: https://github.com/vllm-project/vllm/pull/40786
- Status/date: merged / 2026-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `5371d6fb4023`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-16, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix PP in Gemma4"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "Fix PP in Gemma4"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +9/-16 (25 lines); hunks: -1144,11 +1144,6 @@ def _make_empty_intermediate_tensors(; -1312,13 +1307,12 @@ def forward(; symbols: _make_empty_intermediate_tensors, forward, touching `_make_empty_intermediate_tensors, forward`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +9/-16 (25 lines); hunks: -1144,11 +1144,6 @@ def _make_empty_intermediate_tensors(; -1312,13 +1307,12 @@ def forward(; symbols: _make_empty_intermediate_tensors, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +9/-16
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41206 - Fix Gemma4 MoE expert weight remapping

- Link: https://github.com/vllm-project/vllm/pull/41206
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `ca97f7b9bbf2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Gemma4 MoE expert weight remapping"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "Fix Gemma4 MoE expert weight remapping"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +5/-1 (6 lines); hunks: -84,6 +84,10; -1650,7 +1654,7 @@ def _weight_iterator():; symbols: _remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator, touching `_remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +5/-1 (6 lines); hunks: -84,6 +84,10; -1650,7 +1654,7 @@ def _weight_iterator():; symbols: _remap_gemma4_expert_weight_name, _gemma4_routing_kernel, _weight_iterator
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -84,6 +84,10 @@
+def _remap_gemma4_expert_weight_name(name: str) -> str:
+    return re.sub(r"(?<!\.moe)\.experts\.(\d+)\.", r".moe.experts.\1.", name)
@@ -1650,7 +1654,7 @@ def _weight_iterator():
-                name = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", name)
+                name = _remap_gemma4_expert_weight_name(name)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39570 - [Fix] Sync gemma4 chat template from hf

- Link: https://github.com/vllm-project/vllm/pull/39570
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`; associated commits `c408fdd663af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +67/-44, 239 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Sync gemma4 chat template from hf"; model line: Gemma 4; category: bug fix; main diff: `examples/tool_chat_template_gemma4.jinja`; technical summary: Covers "[Fix] Sync gemma4 chat template from hf"; the main implementation surface is `examples/tool_chat_template_gemma4.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_gemma4.jinja` modified +67/-44 (111 lines); hunks: -1,44 +1,25; -71,6 +52,32.
- Code diff details:
  - `examples/tool_chat_template_gemma4.jinja` modified +67/-44 (111 lines); hunks: -1,44 +1,25; -71,6 +52,32
- Key code excerpts:

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

- Reviewed files:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +67/-44
- Risk and verification: This is mostly docs/examples in `examples/tool_chat_template_gemma4.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #40796 - [Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens

- Link: https://github.com/vllm-project/vllm/pull/40796
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `c3ad791e1a9a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +62/-1, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens"; model line: Gemma 4; category: bug fix; main diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens"; the main implementation surface is `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0 (54 lines); hunks: -12,6 +12,60; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt, touching `test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt`; `vllm/model_executor/models/gemma4_mm.py` modified +8/-1 (9 lines); hunks: -265,7 +265,14 @@ def _compute_num_soft_tokens(; symbols: _compute_num_soft_tokens, get_image_repl, touching `_compute_num_soft_tokens, get_image_repl`.
- Code diff details:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0 (54 lines); hunks: -12,6 +12,60; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_limit_mm_per_prompt
  - `vllm/model_executor/models/gemma4_mm.py` modified +8/-1 (9 lines); hunks: -265,7 +265,14 @@ def _compute_num_soft_tokens(; symbols: _compute_num_soft_tokens, get_image_repl
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +54/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41574 - [Model] Fix Gemma4 MoE activation mismatch

- Link: https://github.com/vllm-project/vllm/pull/41574
- Status/date: merged / 2026-05-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `6bb924bbf3a9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +23/-1, 130 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix Gemma4 MoE activation mismatch"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Model] Fix Gemma4 MoE activation mismatch"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +1/-1 (2 lines); hunks: -360,7 +360,7 @@ def routing_function(; symbols: routing_function, forward, touching `routing_function, forward`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +1/-1 (2 lines); hunks: -360,7 +360,7 @@ def routing_function(; symbols: routing_function, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/gemma4.py
@@ -360,7 +360,7 @@ def routing_function(
-            activation="gelu",
+            activation="gelu_tanh",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/activation.py`, `vllm/model_executor/layers/fused_moe/fused_batched_moe.py`, `vllm/model_executor/layers/fused_moe/fused_humming_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41799 - [MM][Gemma4] Respect max_soft_tokens in encoder budget

- Link: https://github.com/vllm-project/vllm/pull/41799
- Status/date: merged / 2026-05-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `242afc6bf40d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +91/-11, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Gemma4] Respect max_soft_tokens in encoder budget"; model line: Gemma 4; category: docs/tests/CI; main diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[MM][Gemma4] Respect max_soft_tokens in encoder budget"; the main implementation surface is `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0 (60 lines); hunks: -2,6 +2,7; -66,6 +67,65 @@ def test_compute_num_soft_tokens_does_not_exceed_max_soft_tok...; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens, test_limit_mm_per_prompt, touching `test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens`; `vllm/model_executor/models/gemma4_mm.py` modified +31/-11 (42 lines); hunks: -81,10 +81,26; -216,10 +232,14 @@ def get_mm_max_tokens_per_item(; symbols: _get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor, get_replacement_image, touching `_get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor`.
- Code diff details:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0 (60 lines); hunks: -2,6 +2,7; -66,6 +67,65 @@ def test_compute_num_soft_tokens_does_not_exceed_max_soft_tok...; symbols: test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_prompt_updates_respects_nested_max_soft_tokens, test_limit_mm_per_prompt
  - `vllm/model_executor/models/gemma4_mm.py` modified +31/-11 (42 lines); hunks: -81,10 +81,26; -216,10 +232,14 @@ def get_mm_max_tokens_per_item(; symbols: _get_max_soft_tokens, get_mm_max_tokens_per_item, _call_hf_processor, get_replacement_image
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +60/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +31/-11
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41745 - [Spec Decode] Add Gemma4 MTP speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/41745
- Status/date: merged / 2026-05-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`; associated commits `27e0057aeda6`
- Diff scope read: GitHub Pull Request files API returned 9 files, +1121/-72, 1390 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec Decode] Add Gemma4 MTP speculative decoding support"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`; technical summary: Covers "[Spec Decode] Add Gemma4 MTP speculative decoding support"; the main implementation surface is `vllm/model_executor/models/gemma4_mtp.py`, `vllm/v1/spec_decode/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mtp.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: Gemma4MTPMaskedEmbedder, __init__, _select_and_score, forward, touching `Gemma4MTPMaskedEmbedder, __init__, _select_and_score`; `vllm/v1/spec_decode/gemma4.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Gemma4Proposer, __init__, set_per_group_block_table, model_returns_tuple, touching `Gemma4Proposer, __init__, set_per_group_block_table`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mtp.py` added +603/-0 (603 lines); hunks: -0,0 +1,603; symbols: Gemma4MTPMaskedEmbedder, __init__, _select_and_score, forward
  - `vllm/v1/spec_decode/gemma4.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Gemma4Proposer, __init__, set_per_group_block_table, model_returns_tuple
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mtp.py` added +603/-0; `vllm/v1/spec_decode/gemma4.py` added +335/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/v1/e2e/spec_decode/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41837 - [MM][Gemma4] Use video profiling hints in encoder budget

- Link: https://github.com/vllm-project/vllm/pull/41837
- Status/date: merged / 2026-05-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `f650ace6de5a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +50/-4, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Gemma4] Use video profiling hints in encoder budget"; model line: Gemma 4; category: docs/tests/CI; main diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[MM][Gemma4] Use video profiling hints in encoder budget"; the main implementation surface is `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0 (35 lines); hunks: -1,6 +1,8; -102,6 +104,39 @@ def test_get_mm_max_tokens_per_item_respects_configured_max...; symbols: test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens, touching `test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens`; `vllm/model_executor/models/gemma4_mm.py` modified +9/-1 (10 lines); hunks: -246,7 +246,15 @@ def get_mm_max_tokens_per_item(; symbols: get_mm_max_tokens_per_item, get_data_parser, touching `get_mm_max_tokens_per_item, get_data_parser`.
- Code diff details:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0 (35 lines); hunks: -1,6 +1,8; -102,6 +104,39 @@ def test_get_mm_max_tokens_per_item_respects_configured_max...; symbols: test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_video_num_frames, test_get_prompt_updates_respects_nested_max_soft_tokens
  - `vllm/model_executor/models/gemma4_mm.py` modified +9/-1 (10 lines); hunks: -246,7 +246,15 @@ def get_mm_max_tokens_per_item(; symbols: get_mm_max_tokens_per_item, get_data_parser
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +35/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +9/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40588 - [Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP

- Link: https://github.com/vllm-project/vllm/pull/40588
- Status/date: merged / 2026-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `90f145aaf724`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +67/-16, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Models][Gemma3/Gemma4] Support hidden_act variants in gated MLP"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +2/-8 (10 lines); hunks: -35,7 +35,7; -238,13 +238,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +2/-8 (10 lines); hunks: -35,7 +35,7; -238,13 +238,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +2/-8
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_gemma_hidden_act.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41991 - [Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser

- Link: https://github.com/vllm-project/vllm/pull/41991
- Status/date: merged / 2026-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `dbd86a67e3ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +34/-0, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0 (15 lines); hunks: -135,6 +135,11 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -149,6 +154,16 @@ def test_bare_values(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array, test_string_array, touching `test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0 (15 lines); hunks: -135,6 +135,11 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -149,6 +154,16 @@ def test_bare_values(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_malformed_partial_array, TestParseGemma4Array, test_string_array
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +15/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40708 - [BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue

- Link: https://github.com/vllm-project/vllm/pull/40708
- Status/date: merged / 2026-05-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `25abddc1a5cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +55/-18, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +24/-18 (42 lines); hunks: -40,6 +40,7; -1368,30 +1369,35 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +24/-18 (42 lines); hunks: -40,6 +40,7; -1368,30 +1369,35 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +24/-18
- Risk and verification: The diff ships test coverage in `tests/models/quantization/test_awq.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42188 - [Bugfix] Gemma 4 chat template crash with missing tool name and tool id

- Link: https://github.com/vllm-project/vllm/pull/42188
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`; associated commits `b1687527b836`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Gemma 4 chat template crash with missing tool name and tool id"; model line: Gemma 4; category: bug fix; main diff: `examples/tool_chat_template_gemma4.jinja`; technical summary: Covers "[Bugfix] Gemma 4 chat template crash with missing tool name and tool id"; the main implementation surface is `examples/tool_chat_template_gemma4.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -263,7 +263,7; -277,7 +277,7.
- Code diff details:
  - `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -263,7 +263,7; -277,7 +277,7
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_gemma4.jinja
@@ -263,7 +263,7 @@
-                    {{- format_tool_response_block(tool_response['name'] | default('unknown'), tool_response['response']) -}}
+                    {{- format_tool_response_block(tool_response['name'] | default('unknown', true), tool_response['response']) -}}
@@ -277,7 +277,7 @@
-                        {%- set ns_tname = namespace(name=follow.get('name') | default('unknown')) -%}
+                        {%- set ns_tname = namespace(name=follow.get('name') | default('unknown', true)) -%}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +2/-2
- Risk and verification: This is mostly docs/examples in `examples/tool_chat_template_gemma4.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #42217 - [Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash

- Link: https://github.com/vllm-project/vllm/pull/42217
- Status/date: merged / 2026-05-12
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `630492da308e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +45/-7, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash"; model line: Gemma 4; category: bug fix; main diff: `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Fix] Gemma4 Mixed-Resolution Image Co-Batching Crash"; the main implementation surface is `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0 (33 lines); hunks: -4,9 +4,12; -15,6 +18,36; symbols: test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked, touching `test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked`; `vllm/model_executor/models/gemma4_mm.py` modified +12/-7 (19 lines); hunks: -124,12 +124,12 @@ class Gemma4ImagePixelInputs(TensorSchema):; -1128,15 +1128,20 @@ def _process_image_input(; symbols: Gemma4ImagePixelInputs, _process_image_input, touching `Gemma4ImagePixelInputs, _process_image_input`.
- Code diff details:
  - `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0 (33 lines); hunks: -4,9 +4,12; -15,6 +18,36; symbols: test_gemma4_image_schema_accepts_variable_patch_counts, test_gemma4_image_batching_keeps_variable_patch_counts_unstacked
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-7 (19 lines); hunks: -124,12 +124,12 @@ class Gemma4ImagePixelInputs(TensorSchema):; -1128,15 +1128,20 @@ def _process_image_input(; symbols: Gemma4ImagePixelInputs, _process_image_input
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +33/-0
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +12/-7
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42250 - [Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution

- Link: https://github.com/vllm-project/vllm/pull/42250
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `5794c65f8c36`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-4, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Bugfix][Model] Gemma4 MoE routing closure captures per_expert_scale, breaking functional_call substitution"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4.py` modified +7/-4 (11 lines); hunks: -326,8 +326,9 @@ def __init__(; -336,10 +337,12 @@ def routing_function(; symbols: __init__, routing_function, touching `__init__, routing_function`.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +7/-4 (11 lines); hunks: -326,8 +326,9 @@ def __init__(; -336,10 +337,12 @@ def routing_function(; symbols: __init__, routing_function
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42128 - [Bugfix] Fix Gemma4ToolParser streaming float corruption

- Link: https://github.com/vllm-project/vllm/pull/42128
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`; associated commits `665f9c42535c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +41/-0, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4ToolParser streaming float corruption"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Gemma4ToolParser streaming float corruption"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -135,6 +135,26 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -164,6 +184,15 @@ def test_stray_closing_bracket(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array, test_stray_closing_bracket, touching `test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -135,6 +135,26 @@ def test_empty_value_after_other_keys_partial_withheld(self):; -164,6 +184,15 @@ def test_stray_closing_bracket(self):; symbols: test_empty_value_after_other_keys_partial_withheld, test_trailing_dot_float_partial_withheld, test_malformed_partial_array, test_stray_closing_bracket
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43169 - [Perf][Gemma4] Batch vision encoder calls for image and video processing

- Link: https://github.com/vllm-project/vllm/pull/43169
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `2b75a73b8e23`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +180/-86, 336 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Gemma4] Batch vision encoder calls for image and video processing"; model line: Gemma 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Perf][Gemma4] Batch vision encoder calls for image and video processing"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +180/-86 (266 lines); hunks: -61,6 +61,7; -960,6 +961,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _process_image_input, touching `__init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +180/-86 (266 lines); hunks: -61,6 +61,7; -960,6 +961,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _process_image_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +180/-86
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43296 - [CI] Fix "test_awq_load[gemma4-moe-*]" failure

- Link: https://github.com/vllm-project/vllm/pull/43296
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; associated commits `025d4f5cd261`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +116/-31, 216 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Fix "test_awq_load[gemma4-moe-*]" failure"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/processing/test_gemma4.py`; technical summary: Covers "[CI] Fix "test_awq_load[gemma4-moe-*]" failure"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`, `tests/models/multimodal/processing/test_gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +54/-30 (84 lines); hunks: -961,9 +961,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -1104,18 +1101,36 @@ def _parse_and_validate_multimodal_inputs(; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _encoder_chunk, touching `__init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch`; `tests/models/multimodal/processing/test_gemma4.py` modified +62/-1 (63 lines); hunks: -7,9 +7,13; -224,3 +228,60 @@ def test_limit_mm_per_prompt(; symbols: test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching, test_encoder_chunk_zero_patches_is_safe, touching `test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +54/-30 (84 lines); hunks: -961,9 +961,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -1104,18 +1101,36 @@ def _parse_and_validate_multimodal_inputs(; symbols: __init__, _parse_and_validate_multimodal_inputs, _encoder_max_batch, _encoder_chunk
  - `tests/models/multimodal/processing/test_gemma4.py` modified +62/-1 (63 lines); hunks: -7,9 +7,13; -224,3 +228,60 @@ def test_limit_mm_per_prompt(; symbols: test_limit_mm_per_prompt, test_encoder_chunk_tight_budget_fits_in_free, test_encoder_chunk_roomy_gpu_keeps_batching, test_encoder_chunk_zero_patches_is_safe
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +54/-30
  - tests: `tests/models/multimodal/processing/test_gemma4.py` modified +62/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41459 - fix(frontend): Add multimodal placeholders to Gemma4 tool message template

- Link: https://github.com/vllm-project/vllm/pull/41459
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`, `tests/renderers/test_gemma4_chat_template.py`; associated commits `69c9f199574e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +78/-0, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(frontend): Add multimodal placeholders to Gemma4 tool message template"; model line: Gemma 4; category: bug fix; main diff: `tests/renderers/test_gemma4_chat_template.py`, `examples/tool_chat_template_gemma4.jinja`; technical summary: Covers "fix(frontend): Add multimodal placeholders to Gemma4 tool message template"; the main implementation surface is `tests/renderers/test_gemma4_chat_template.py`, `examples/tool_chat_template_gemma4.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/renderers/test_gemma4_chat_template.py` modified +69/-0 (69 lines); hunks: -343,3 +343,72 @@ def test_format_argument_types(self, gemma4_template):; symbols: test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities, touching `test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities`; `examples/tool_chat_template_gemma4.jinja` modified +9/-0 (9 lines); hunks: -295,6 +295,15.
- Code diff details:
  - `tests/renderers/test_gemma4_chat_template.py` modified +69/-0 (69 lines); hunks: -343,3 +343,72 @@ def test_format_argument_types(self, gemma4_template):; symbols: test_format_argument_types, test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities
  - `examples/tool_chat_template_gemma4.jinja` modified +9/-0 (9 lines); hunks: -295,6 +295,15
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/renderers/test_gemma4_chat_template.py` modified +69/-0
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +9/-0
- Risk and verification: The diff ships test coverage in `tests/renderers/test_gemma4_chat_template.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43909 - [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`

- Link: https://github.com/vllm-project/vllm/pull/43909
- Status/date: merged / 2026-05-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mtp.py`; associated commits `e1105064b282`
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-1, 37 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mtp.py`; technical summary: Covers "[Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered`"; the main implementation surface is `vllm/model_executor/models/gemma4_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1 (10 lines); hunks: -501,6 +501,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -567,14 +568,20 @@ def forward(; symbols: __init__, forward, _get_full_lm_head_weight, compute_logits, touching `__init__, forward, _get_full_lm_head_weight`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1 (10 lines); hunks: -501,6 +501,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -567,14 +568,20 @@ def forward(; symbols: __init__, forward, _get_full_lm_head_weight, compute_logits
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mtp.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43798 - [Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl

- Link: https://github.com/vllm-project/vllm/pull/43798
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `f91fb2fcf3f1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +111/-11, 271 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +51/-10 (61 lines); hunks: -16,7 +16,7; -41,6 +41,7; symbols: __init__, forward, Gemma4ForConditionalGeneration, touching `__init__, forward, Gemma4ForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +51/-10 (61 lines); hunks: -16,7 +16,7; -41,6 +41,7; symbols: __init__, forward, Gemma4ForConditionalGeneration
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +51/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/bitsandbytes.py`, `vllm/model_executor/model_loader/bitsandbytes_loader.py`, `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44232 - [Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor

- Link: https://github.com/vllm-project/vllm/pull/44232
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `2fd0e52252f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +19/-0 (19 lines); hunks: -519,6 +519,25 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only, _call_hf_processor, touching `_get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +19/-0 (19 lines); hunks: -519,6 +519,25 @@ def _get_dummy_videos(; symbols: _get_dummy_videos, Gemma4MultiModalProcessor, _apply_hf_processor_text_only, _call_hf_processor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +19/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44429 - [Model] Add Gemma4 Unified (encoder-free) support

- Link: https://github.com/vllm-project/vllm/pull/44429
- Status/date: merged / 2026-06-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4_mtp.py`, `vllm/model_executor/models/gemma4_unified.py`; associated commits `a248b45d0548`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +791/-31, 1039 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Gemma4 Unified (encoder-free) support"; model line: Gemma 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/gemma4_unified.py`, `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[Model] Add Gemma4 Unified (encoder-free) support"; the main implementation surface is `vllm/model_executor/models/gemma4_unified.py`, `tests/models/multimodal/processing/test_gemma4_unified.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_unified.py` added +466/-0 (466 lines); hunks: -0,0 +1,466; symbols: Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb, forward, touching `Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb`; `tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0 (205 lines); hunks: -0,0 +1,205; symbols: test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens, touching `test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens`; `vllm/model_executor/models/gemma4_mm.py` modified +64/-19 (83 lines); hunks: -121,7 +121,7 @@ class Gemma4ImagePixelInputs(TensorSchema):; -341,6 +341,29 @@ def get_image_repl(; symbols: Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens, get_audio_repl, touching `Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens`; `vllm/model_executor/models/gemma4_mtp.py` modified +19/-3 (22 lines); hunks: -279,11 +279,19 @@ def __init__(; -545,6 +553,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_input_ids, compute_logits, get_top_tokens, touching `__init__, embed_input_ids, compute_logits`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_unified.py` added +466/-0 (466 lines); hunks: -0,0 +1,466; symbols: Gemma4UnifiedVisionEmbedder, __init__, _factorized_posemb, forward
  - `tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0 (205 lines); hunks: -0,0 +1,205; symbols: test_gemma4_unified_image_schema_accepts_variable_patch_counts, test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked, test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens, test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens
  - `vllm/model_executor/models/gemma4_mm.py` modified +64/-19 (83 lines); hunks: -121,7 +121,7 @@ class Gemma4ImagePixelInputs(TensorSchema):; -341,6 +341,29 @@ def get_image_repl(; symbols: Gemma4ImagePixelInputs, get_image_repl, _compute_audio_num_tokens, get_audio_repl
  - `vllm/model_executor/models/gemma4_mtp.py` modified +19/-3 (22 lines); hunks: -279,11 +279,19 @@ def __init__(; -545,6 +553,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, embed_input_ids, compute_logits, get_top_tokens
  - `vllm/model_executor/models/gemma4.py` modified +6/-3 (9 lines); hunks: -1051,11 +1051,14 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_unified.py` added +466/-0; `vllm/model_executor/models/gemma4_mm.py` modified +64/-19; `vllm/model_executor/models/gemma4_mtp.py` modified +19/-3; `vllm/model_executor/models/gemma4.py` modified +6/-3
  - tests: `tests/models/multimodal/processing/test_gemma4_unified.py` added +205/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_gemma4_unified.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43982 - [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load

- Link: https://github.com/vllm-project/vllm/pull/43982
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/v1/spec_decode/gemma4.py`; associated commits `128adabfe0fe`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load"; model line: Gemma 4; category: bug fix; main diff: `vllm/v1/spec_decode/gemma4.py`; technical summary: Covers "[Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load"; the main implementation surface is `vllm/v1/spec_decode/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/spec_decode/gemma4.py` modified +6/-1 (7 lines); hunks: -81,11 +81,16 @@ def build_per_group_and_layer_attn_metadata(; symbols: build_per_group_and_layer_attn_metadata, touching `build_per_group_and_layer_attn_metadata`.
- Code diff details:
  - `vllm/v1/spec_decode/gemma4.py` modified +6/-1 (7 lines); hunks: -81,11 +81,16 @@ def build_per_group_and_layer_attn_metadata(; symbols: build_per_group_and_layer_attn_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/v1/spec_decode/gemma4.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `vllm/v1/spec_decode/gemma4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43241 - [Model Runner V2][Spec Decode] Add Gemma4 MTP support

- Link: https://github.com/vllm-project/vllm/pull/43241
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/v1/worker/gpu/spec_decode/gemma4/__init__.py`, `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py`; associated commits `ceb0111a90ac`
- Diff scope read: GitHub Pull Request files API returned 14 files, +1243/-942, 2279 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Runner V2][Spec Decode] Add Gemma4 MTP support"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`; technical summary: Covers "[Model Runner V2][Spec Decode] Add Gemma4 MTP support"; the main implementation surface is `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893 (901 lines); hunks: -1,903 +1,18; symbols: EagleSpeculator, __init__, init_cudagraph_manager, load_model, touching `EagleSpeculator, __init__, init_cudagraph_manager`; `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0 (795 lines); hunks: -0,0 +1,795; symbols: AutoRegressiveSpeculator, __init__, advance_draft_positions, model_returns_tuple, touching `AutoRegressiveSpeculator, __init__, advance_draft_positions`; `vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: BaseSpeculator, init_cudagraph_manager, capture, propose, touching `BaseSpeculator, init_cudagraph_manager, capture`; `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: Gemma4Speculator, advance_draft_positions, model_returns_tuple, load_draft_model, touching `Gemma4Speculator, advance_draft_positions, model_returns_tuple`.
- Code diff details:
  - `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893 (901 lines); hunks: -1,903 +1,18; symbols: EagleSpeculator, __init__, init_cudagraph_manager, load_model
  - `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0 (795 lines); hunks: -0,0 +1,795; symbols: AutoRegressiveSpeculator, __init__, advance_draft_positions, model_returns_tuple
  - `vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: BaseSpeculator, init_cudagraph_manager, capture, propose
  - `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0 (158 lines); hunks: -0,0 +1,158; symbols: Gemma4Speculator, advance_draft_positions, model_returns_tuple, load_draft_model
  - `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` added +22/-0 (22 lines); hunks: -0,0 +1,22; symbols: MTPSpeculator, model_returns_tuple, load_draft_model
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` modified +8/-893; `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` added +795/-0; `vllm/v1/worker/gpu/spec_decode/speculator.py` added +224/-0; `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` added +158/-0; `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` added +22/-0; `vllm/v1/worker/gpu/spec_decode/__init__.py` modified +16/-3
- Risk and verification: Runtime changes concentrate in `vllm/v1/attention/backends/flashinfer.py`, `vllm/v1/attention/backends/triton_attn.py`, `vllm/v1/attention/backends/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44340 - [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings

- Link: https://github.com/vllm-project/vllm/pull/44340
- Status/date: merged / 2026-06-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +744/-27, 1040 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`; technical summary: Covers "[Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings"; the main implementation surface is `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__, get_min_capability, touching `fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int, __init__, touching `_dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2 (103 lines); hunks: -30,6 +30,9; -45,6 +48,7; symbols: get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel, _is_wNa8o8_int, touching `get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel`; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5 (66 lines); hunks: -1,5 +1,6; -42,16 +43,57 @@ def humming_is_layer_skipped(config: dict[str, Any], prefix:...; symbols: humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer, prepare_humming_moe_layer, touching `humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0 (257 lines); hunks: -0,0 +1,257; symbols: fake_quant_static_int8, CompressedTensorsWNA8O8Int, __init__, get_min_capability
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _dequant_gather_kernel, _dequant_gather_triton, CompressedTensorsEmbeddingWNA16Int, __init__
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2 (103 lines); hunks: -30,6 +30,9; -45,6 +48,7; symbols: get_quant_method, _quantization_scheme_map_from_config, _is_wNa16_group_channel, _is_wNa8o8_int
  - `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5 (66 lines); hunks: -1,5 +1,6; -42,16 +43,57 @@ def humming_is_layer_skipped(config: dict[str, Any], prefix:...; symbols: humming_is_layer_skipped, convert_linear_layer_to_humming_standard, prepare_humming_layer, prepare_humming_moe_layer
  - `vllm/model_executor/kernels/linear/mixed_precision/humming.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: HummingLinearKernel, get_min_capability, can_implement, process_weights_after_loading
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py` added +257/-0; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py` added +170/-0; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` modified +101/-2; `vllm/model_executor/layers/quantization/utils/humming_utils.py` modified +61/-5; `vllm/model_executor/kernels/linear/mixed_precision/humming.py` added +61/-0; `vllm/model_executor/models/gemma4_mm.py` modified +12/-12
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/kernels/quantization/test_quantized_embedding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44571 - [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified

- Link: https://github.com/vllm-project/vllm/pull/44571
- Status/date: merged / 2026-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_unified.py`; associated commits `da1daf40bf18`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Exclude vision embedder from quantization in Gemma4 Unified"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_unified.py`; technical summary: Covers "[Bugfix] Exclude vision embedder from quantization in Gemma4 Unified"; the main implementation surface is `vllm/model_executor/models/gemma4_unified.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_unified.py` modified +3/-1 (4 lines); hunks: -80,7 +80,7 @@ class Gemma4UnifiedVisionEmbedder(nn.Module):; -91,6 +91,7 @@ def __init__(self, config, quant_config=None):; symbols: Gemma4UnifiedVisionEmbedder, __init__, touching `Gemma4UnifiedVisionEmbedder, __init__`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_unified.py` modified +3/-1 (4 lines); hunks: -80,7 +80,7 @@ class Gemma4UnifiedVisionEmbedder(nn.Module):; -91,6 +91,7 @@ def __init__(self, config, quant_config=None):; symbols: Gemma4UnifiedVisionEmbedder, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_unified.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_unified.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- Link: https://github.com/vllm-project/vllm/pull/43167
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 56 files, +88/-731, 1251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove KV cache scale boilerplate from model weight loading methods"; model line: Gemma 4; category: docs/tests/CI; main diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`; technical summary: Covers "Remove KV cache scale boilerplate from model weight loading methods"; the main implementation surface is `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name, touching `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`; `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader, touching `_get_moe_weight_dtype, kv_cache_scale_loader`; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod, touching `KVCacheScaleParameter, __new__, weight_loader`; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter, touching `get_quant_method, get_cache_scale, get_cache_scale_mapper`.
- Code diff details:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_eagle_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44828 - [BugFix] Use served model name in gemma4 audio-tower error message

- Link: https://github.com/vllm-project/vllm/pull/44828
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4_mm.py`; associated commits `469f3dcf1d70`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Use served model name in gemma4 audio-tower error message"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "[BugFix] Use served model name in gemma4 audio-tower error message"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` modified +5/-1 (6 lines); hunks: -34,6 +34,7; -217,7 +218,10 @@ def validate_num_items(self, modality: str, num_items: int)...; symbols: validate_num_items, touching `validate_num_items`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` modified +5/-1 (6 lines); hunks: -34,6 +34,7; -217,7 +218,10 @@ def validate_num_items(self, modality: str, num_items: int)...; symbols: validate_num_items
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/gemma4_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- Link: https://github.com/vllm-project/vllm/pull/41184
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 90 files, +2734/-2027, 7329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; model line: Gemma 4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`; technical summary: Covers "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts, touching `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method, touching `FusedMoeWeightScaleSupported, RoutedExperts, __init__`; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward, touching `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`; `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__, touching `FusedMoEWithLoRA, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- Risk and verification: The diff ships test coverage in `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45163 - [Model] Add DiffusionGemma Support

- Link: https://github.com/vllm-project/vllm/pull/45163
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 52 files, +2698/-235, 3935 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add DiffusionGemma Support"; model line: Gemma 4; category: docs/tests/CI; main diff: `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/models/config.py`; technical summary: Covers "[Model] Add DiffusionGemma Support"; the main implementation surface is `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/models/config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39 (184 lines); hunks: -20,9 +20,11; -343,6 +345,9 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__, _reset_streaming_state, adjust_request, touching `Gemma4ToolParser, __init__, _reset_streaming_state`; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0 (82 lines); hunks: -702,6 +702,88 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call, test_streaming_multi_chunk_batched_tool_calls, touching `test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call`; `vllm/model_executor/models/config.py` modified +55/-0 (55 lines); hunks: -105,6 +105,60 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; -591,6 +645,7 @@ def verify_and_update_model_config(model_config: "ModelConfi...; symbols: verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig, verify_and_update_model_config, touching `verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig`; `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -901,6 +901,10 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`.
- Code diff details:
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39 (184 lines); hunks: -20,9 +20,11; -343,6 +345,9 @@ class Gemma4ToolParser(ToolParser):; symbols: Gemma4ToolParser, __init__, _reset_streaming_state, adjust_request
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0 (82 lines); hunks: -702,6 +702,88 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_streaming_html_argument_does_not_duplicate_tag_prefixes, _collect_tool_calls_by_index, test_streaming_single_chunk_complete_tool_call, test_streaming_multi_chunk_batched_tool_calls
  - `vllm/model_executor/models/config.py` modified +55/-0 (55 lines); hunks: -105,6 +105,60 @@ def verify_and_update_config(vllm_config: "VllmConfig") ->...; -591,6 +645,7 @@ def verify_and_update_model_config(model_config: "ModelConfi...; symbols: verify_and_update_config, DiffusionGemmaModelForBlockDiffusionConfig, DeepseekV4ForCausalLMConfig, verify_and_update_model_config
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -901,6 +901,10 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/utils.py` modified +3/-1 (4 lines); hunks: -486,6 +486,7 @@ def dummy_hf_overrides(; -558,7 +559,8 @@ class DummyConfig:; symbols: dummy_hf_overrides, DummyConfig
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +145/-39; `vllm/model_executor/models/config.py` modified +55/-0; `vllm/model_executor/models/gemma4.py` modified +1/-3; `vllm/model_executor/models/registry.py` modified +4/-0; `vllm/transformers_utils/configs/__init__.py` modified +4/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +82/-0; `tests/models/registry.py` modified +4/-0; `tests/models/utils.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_mixed_causal_attn.py`, `tests/models/registry.py`, `tests/models/utils.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45588 - [Frontend] Replace legacy Gemma4 parsers with engine-based implementation

- Link: https://github.com/vllm-project/vllm/pull/45588
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py` and 7 files; associated commits `76a373eff47a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +2808/-1332, 4822 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Replace legacy Gemma4 parsers with engine-based implementation"; model line: Gemma 4; category: docs/tests/CI; main diff: `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`; technical summary: Covers "[Frontend] Replace legacy Gemma4 parsers with engine-based implementation"; the main implementation surface is `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0 (1201 lines); hunks: -0,0 +1,1201; symbols: _make_tokenizer, decode, _stream_tokens_batched, _collect_fields, touching `_make_tokenizer, decode, _stream_tokens_batched`; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51 (189 lines); hunks: -8,31 +8,105; -49,6 +123,9 @@ def mock_request():; symbols: _make_tool, mock_tokenizer, parser, mock_request, touching `_make_tool, mock_tokenizer, parser`; `tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4 (8 lines); hunks: -83,15 +83,15 @@ def generic_tokenizer():; -111,7 +111,7 @@ def generic_tokenizer():; symbols: generic_tokenizer, touching `generic_tokenizer`; `vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: Gemma4EngineToolParser, touching `Gemma4EngineToolParser`.
- Code diff details:
  - `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0 (1201 lines); hunks: -0,0 +1,1201; symbols: _make_tokenizer, decode, _stream_tokens_batched, _collect_fields
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51 (189 lines); hunks: -8,31 +8,105; -49,6 +123,9 @@ def mock_request():; symbols: _make_tool, mock_tokenizer, parser, mock_request
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4 (8 lines); hunks: -83,15 +83,15 @@ def generic_tokenizer():; -111,7 +111,7 @@ def generic_tokenizer():; symbols: generic_tokenizer
  - `vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: Gemma4EngineToolParser
  - `vllm/reasoning/gemma4_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/parser/engine/test_gemma4_streaming_reasoning.py` added +1201/-0; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +138/-51; `tests/reasoning/test_gemma4_reasoning_parser.py` modified +4/-4; `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +13/-6
  - runtime: `vllm/tool_parsers/gemma4_engine_tool_parser.py` added +8/-0; `vllm/reasoning/gemma4_engine_reasoning_parser.py` added +6/-0; `vllm/parser/gemma4.py` added +557/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/replay_harness.py`, `tests/parser/engine/test_delegating_replay.py`, `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `tests/parser/engine/test_parser_engine.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45553 - [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync

- Link: https://github.com/vllm-project/vllm/pull/45553
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `vllm/parser/gemma4.py`, `vllm/tool_parsers/gemma4_utils.py`; associated commits `6607a80dabfa`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +94/-74, 343 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync"; model line: Gemma 4; category: bug fix; main diff: `vllm/tool_parsers/gemma4_utils.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `examples/tool_chat_template_gemma4.jinja`; technical summary: Covers "[Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync"; the main implementation surface is `vllm/tool_parsers/gemma4_utils.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `examples/tool_chat_template_gemma4.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/gemma4_utils.py` modified +7/-28 (35 lines); hunks: -35,8 +35,6; -52,42 +50,23; symbols: _parse_tool_arguments, parse_tool_calls, touching `_parse_tool_arguments, parse_tool_calls`; `tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7 @@ def generic_tokenizer():; symbols: generic_tokenizer, touching `generic_tokenizer`; `examples/tool_chat_template_gemma4.jinja` modified +64/-40 (104 lines); hunks: -116,7 +116,9; -172,18 +174,21; `vllm/parser/gemma4.py` modified +20/-3 (23 lines); hunks: -423,6 +423,8 @@ def __init__(; -437,6 +439,21 @@ def __init__(; symbols: __init__, adjust_request, _reset, is_reasoning_end, touching `__init__, adjust_request, _reset`.
- Code diff details:
  - `vllm/tool_parsers/gemma4_utils.py` modified +7/-28 (35 lines); hunks: -35,8 +35,6; -52,42 +50,23; symbols: _parse_tool_arguments, parse_tool_calls
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -54,7 +54,7 @@ def generic_tokenizer():; symbols: generic_tokenizer
  - `examples/tool_chat_template_gemma4.jinja` modified +64/-40 (104 lines); hunks: -116,7 +116,9; -172,18 +174,21
  - `vllm/parser/gemma4.py` modified +20/-3 (23 lines); hunks: -423,6 +423,8 @@ def __init__(; -437,6 +439,21 @@ def __init__(; symbols: __init__, adjust_request, _reset, is_reasoning_end
  - `tests/renderers/test_gemma4_chat_template.py` modified +2/-2 (4 lines); hunks: -358,7 +358,7 @@ def test_tool_response_with_multimodal_content(self, gemma4_...; -392,7 +392,7 @@ def test_tool_response_with_all_modalities(self, gemma4_temp...; symbols: test_tool_response_with_multimodal_content, test_tool_response_with_all_modalities
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/gemma4_utils.py` modified +7/-28; `vllm/parser/gemma4.py` modified +20/-3
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +1/-1; `tests/renderers/test_gemma4_chat_template.py` modified +2/-2
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +64/-40
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45795 - [Bugfix] Gemma4: skip forced JSON for required/named tool choice

- Link: https://github.com/vllm-project/vllm/pull/45795
- Status/date: merged / 2026-06-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/tool_parsers/gemma4_engine_tool_parser.py`; associated commits `b9684d99e9ba`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +120/-1, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Gemma4: skip forced JSON for required/named tool choice"; model line: Gemma 4; category: bug fix; main diff: `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`; technical summary: Covers "[Bugfix] Gemma4: skip forced JSON for required/named tool choice"; the main implementation surface is `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `tests/tool_use/test_gemma4_responses_adjust_request.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0 (28 lines); hunks: -1,8 +1,36; symbols: Gemma4EngineToolParser, adjust_request, touching `Gemma4EngineToolParser, adjust_request`; `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1 (93 lines); hunks: -20,6 +20,13; -28,6 +35,7; symbols: _get_weather_tool, _build_responses_request, _build_chat_request, touching `_get_weather_tool, _build_responses_request, _build_chat_request`.
- Code diff details:
  - `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0 (28 lines); hunks: -1,8 +1,36; symbols: Gemma4EngineToolParser, adjust_request
  - `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1 (93 lines); hunks: -20,6 +20,13; -28,6 +35,7; symbols: _get_weather_tool, _build_responses_request, _build_chat_request
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/gemma4_engine_tool_parser.py` modified +28/-0
  - tests: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +92/-1
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_gemma4_responses_adjust_request.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45832 - [Bugfix][Gemma4] Fix parsing when thinking is disabled

- Link: https://github.com/vllm-project/vllm/pull/45832
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`; associated commits `b831374cf1db`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +70/-28, 120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma4] Fix parsing when thinking is disabled"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`; technical summary: Covers "[Bugfix][Gemma4] Fix parsing when thinking is disabled"; the main implementation surface is `tests/tool_use/test_gemma4_responses_adjust_request.py`, `vllm/parser/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21 (78 lines); hunks: -68,28 +68,33 @@ def _build_responses_request(*, tool_choice: str | dict[str,...; -212,3 +217,34 @@ def test_gemma4_named_skips_structured_outputs_responses()...; symbols: _build_responses_request, _build_chat_request, _StubTokenizer, test_gemma4_named_skips_structured_outputs_responses, touching `_build_responses_request, _build_chat_request, _StubTokenizer`; `vllm/parser/gemma4.py` modified +13/-7 (20 lines); hunks: -443,16 +443,22 @@ def adjust_request(; symbols: adjust_request, _reset, touching `adjust_request, _reset`.
- Code diff details:
  - `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21 (78 lines); hunks: -68,28 +68,33 @@ def _build_responses_request(*, tool_choice: str | dict[str,...; -212,3 +217,34 @@ def test_gemma4_named_skips_structured_outputs_responses()...; symbols: _build_responses_request, _build_chat_request, _StubTokenizer, test_gemma4_named_skips_structured_outputs_responses
  - `vllm/parser/gemma4.py` modified +13/-7 (20 lines); hunks: -443,16 +443,22 @@ def adjust_request(; symbols: adjust_request, _reset
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_use/test_gemma4_responses_adjust_request.py` modified +57/-21
  - runtime: `vllm/parser/gemma4.py` modified +13/-7
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_gemma4_responses_adjust_request.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45852 - [Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)

- Link: https://github.com/vllm-project/vllm/pull/45852
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`; associated commits `3c6084bb0d51`
- Diff scope read: GitHub Pull Request files API returned 6 files, +270/-0, 353 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)"; model line: Gemma 4; category: bug fix; main diff: `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`; technical summary: Covers "[Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open ` ` (fixes #45834)"; the main implementation surface is `tests/parser/engine/test_gemma4_streaming_reasoning.py`, `vllm/parser/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0 (208 lines); hunks: -25,12 +25,14; -253,6 +255,212 @@ def test_reasoning_extracted(self, parser, mock_tokenizer,...; symbols: test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer, open_reasoning_parser, touching `test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer`; `vllm/parser/gemma4.py` modified +28/-0 (28 lines); hunks: -353,6 +353,14 @@ def gemma4_config() -> ParserEngineConfig:; -518,6 +526,26 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt, _events_to_delta, touching `gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt`.
- Code diff details:
  - `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0 (208 lines); hunks: -25,12 +25,14; -253,6 +255,212 @@ def test_reasoning_extracted(self, parser, mock_tokenizer,...; symbols: test_reasoning_extracted, TestGemma4PromptOpenReasoning, open_reasoning_tokenizer, open_reasoning_parser
  - `vllm/parser/gemma4.py` modified +28/-0 (28 lines); hunks: -353,6 +353,14 @@ def gemma4_config() -> ParserEngineConfig:; -518,6 +526,26 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: gemma4_config, is_reasoning_end, adjust_initial_state_from_prompt, _events_to_delta
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/parser/engine/test_gemma4_streaming_reasoning.py` modified +208/-0
  - runtime: `vllm/parser/gemma4.py` modified +28/-0
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_gemma4_streaming_reasoning.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45867 - [Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls

- Link: https://github.com/vllm-project/vllm/pull/45867
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`; associated commits `58b2e896423f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls"; model line: Gemma 4; category: bug fix; main diff: `examples/tool_chat_template_gemma4.jinja`; technical summary: Covers "[Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls"; the main implementation surface is `examples/tool_chat_template_gemma4.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -231,10 +231,10.
- Code diff details:
  - `examples/tool_chat_template_gemma4.jinja` modified +2/-2 (4 lines); hunks: -231,10 +231,10
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_gemma4.jinja
@@ -231,10 +231,10 @@
-    {#- Render reasoning/reasoning_content as thinking channel (tool-call turns only) -#}
+    {#- Render reasoning/reasoning_content as thinking channel -#}
-    {%- if thinking_text and thinking_gate and message.get('tool_calls') -%}
+    {%- if thinking_text and thinking_gate -%}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_gemma4.jinja` modified +2/-2
- Risk and verification: This is mostly docs/examples in `examples/tool_chat_template_gemma4.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
