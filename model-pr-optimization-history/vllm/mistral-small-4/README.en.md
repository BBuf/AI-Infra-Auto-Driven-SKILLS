# vllm Mistral Small 4 Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/generate/multimodal/mistral-small_offline.py` | no direct PR-number commit |
| `examples/tool_chat_template_mistral.jinja` | [#5649](https://github.com/vllm-project/vllm/pull/5649) |
| `examples/tool_chat_template_mistral3.jinja` | [#17195](https://github.com/vllm-project/vllm/pull/17195), [#17644](https://github.com/vllm-project/vllm/pull/17644) |
| `examples/tool_chat_template_mistral_parallel.jinja` | [#5649](https://github.com/vllm-project/vllm/pull/5649) |
| `tests/models/fixtures/ministral_3b_chat.json` | no direct PR-number commit |
| `tests/models/fixtures/mistral_small_3_chat.json` | [#14977](https://github.com/vllm-project/vllm/pull/14977) |
| `tests/models/language/generation/test_mistral.py` | [#20093](https://github.com/vllm-project/vllm/pull/20093), [#28659](https://github.com/vllm-project/vllm/pull/28659), [#29918](https://github.com/vllm-project/vllm/pull/29918) |
| `tests/reasoning/test_mistral_reasoning_parser.py` | [#26358](https://github.com/vllm-project/vllm/pull/26358), [#30391](https://github.com/vllm-project/vllm/pull/30391) |
| `tests/renderers/test_mistral.py` | no direct PR-number commit |
| `tests/tokenizers_/test_mistral.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#38150](https://github.com/vllm-project/vllm/pull/38150) |
| `tests/tool_parsers/test_mistral_tool_parser.py` | [#30724](https://github.com/vllm-project/vllm/pull/30724), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217), [#40531](https://github.com/vllm-project/vllm/pull/40531) |
| `tests/tool_use/mistral/__init__.py` | no direct PR-number commit |
| `tests/tool_use/mistral/conftest.py` | no direct PR-number commit |
| `tests/tool_use/mistral/test_mistral_tool_calls.py` | [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `tests/tool_use/mistral/utils.py` | [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `vllm/model_executor/models/mistral.py` | [#1196](https://github.com/vllm-project/vllm/pull/1196), [#1220](https://github.com/vllm-project/vllm/pull/1220), [#1254](https://github.com/vllm-project/vllm/pull/1254), [#1303](https://github.com/vllm-project/vllm/pull/1303), [#2868](https://github.com/vllm-project/vllm/pull/2868), [#32780](https://github.com/vllm-project/vllm/pull/32780), [#33095](https://github.com/vllm-project/vllm/pull/33095) |
| `vllm/model_executor/models/mistral3.py` | [#15505](https://github.com/vllm-project/vllm/pull/15505), [#15950](https://github.com/vllm-project/vllm/pull/15950), [#17270](https://github.com/vllm-project/vllm/pull/17270), [#17428](https://github.com/vllm-project/vllm/pull/17428), [#21945](https://github.com/vllm-project/vllm/pull/21945), [#33939](https://github.com/vllm-project/vllm/pull/33939), [#36928](https://github.com/vllm-project/vllm/pull/36928) |
| `vllm/model_executor/models/mistral_eagle.py` | [#41024](https://github.com/vllm-project/vllm/pull/41024) |
| `vllm/model_executor/models/mistral_large_3.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757) |
| `vllm/model_executor/models/mistral_large_3_eagle.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#36163](https://github.com/vllm-project/vllm/pull/36163), [#37232](https://github.com/vllm-project/vllm/pull/37232) |
| `vllm/reasoning/mistral_reasoning_parser.py` | [#30391](https://github.com/vllm-project/vllm/pull/30391) |
| `vllm/renderers/mistral.py` | no direct PR-number commit |
| `vllm/tokenizers/mistral.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#31138](https://github.com/vllm-project/vllm/pull/31138), [#34651](https://github.com/vllm-project/vllm/pull/34651), [#36971](https://github.com/vllm-project/vllm/pull/36971), [#37209](https://github.com/vllm-project/vllm/pull/37209), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `vllm/tool_parsers/mistral_tool_parser.py` | [#30724](https://github.com/vllm-project/vllm/pull/30724), [#34651](https://github.com/vllm-project/vllm/pull/34651), [#37209](https://github.com/vllm-project/vllm/pull/37209), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217), [#39294](https://github.com/vllm-project/vllm/pull/39294), [#40043](https://github.com/vllm-project/vllm/pull/40043), [#40531](https://github.com/vllm-project/vllm/pull/40531) |
| `vllm/transformers_utils/configs/mistral.py` | [#1196](https://github.com/vllm-project/vllm/pull/1196), [#1254](https://github.com/vllm-project/vllm/pull/1254), [#20570](https://github.com/vllm-project/vllm/pull/20570), [#28659](https://github.com/vllm-project/vllm/pull/28659), [#29172](https://github.com/vllm-project/vllm/pull/29172), [#29239](https://github.com/vllm-project/vllm/pull/29239), [#29757](https://github.com/vllm-project/vllm/pull/29757), [#33521](https://github.com/vllm-project/vllm/pull/33521), [#34028](https://github.com/vllm-project/vllm/pull/34028), [#34104](https://github.com/vllm-project/vllm/pull/34104), [#36163](https://github.com/vllm-project/vllm/pull/36163), [#37104](https://github.com/vllm-project/vllm/pull/37104), ... (13 total) |
| `vllm/utils/mistral.py` | [#34651](https://github.com/vllm-project/vllm/pull/34651), [#40043](https://github.com/vllm-project/vllm/pull/40043) |

## PR Coverage Summary

- Git-traced PRs: 44
- Extra PRs preserved from existing docs: 8
- Total PRs in this document: 52
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2023-09-28 | [#1196](https://github.com/vllm-project/vllm/pull/1196) | merged | [Mistral] Mistral-7B-v0.1 support | `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py` |
| 2023-09-28 | [#1220](https://github.com/vllm-project/vllm/pull/1220) | merged | Fix Mistral model | `vllm/model_executor/models/mistral.py` |
| 2023-10-10 | [#1303](https://github.com/vllm-project/vllm/pull/1303) | merged | [Minor] Fix comment in mistral.py | `vllm/model_executor/models/mistral.py` |
| 2023-10-13 | [#1254](https://github.com/vllm-project/vllm/pull/1254) | merged | Bump up transformers version & Remove MistralConfig | `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral.py` |
| 2024-02-22 | [#2868](https://github.com/vllm-project/vllm/pull/2868) | merged | Migrate MistralForCausalLM to LlamaForCausalLM | `vllm/model_executor/models/mistral.py` |
| 2024-09-04 | [#5649](https://github.com/vllm-project/vllm/pull/5649) | merged | [Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models | `examples/tool_chat_template_mistral_parallel.jinja`, `examples/tool_chat_template_mistral.jinja` |
| 2025-03-18 | [#14977](https://github.com/vllm-project/vllm/pull/14977) | merged | [Mistral-Small 3.1] Update docs and tests | `tests/models/fixtures/mistral_small_3_chat.json` |
| 2025-03-20 | [#15184](https://github.com/vllm-project/vllm/pull/15184) | merged | [Doc] Update Mistral Small 3.1/Pixtral example | `examples/offline_inference/mistral-small.py` |
| 2025-04-01 | [#15505](https://github.com/vllm-project/vllm/pull/15505) | merged | [Model] Support Mistral3 in the HF Transformers format | `vllm/model_executor/models/mistral3.py` |
| 2025-04-02 | [#15950](https://github.com/vllm-project/vllm/pull/15950) | merged | [V1] Support Mistral3 in V1 | `vllm/model_executor/models/mistral3.py` |
| 2025-04-07 | [#16147](https://github.com/vllm-project/vllm/pull/16147) | merged | [Misc] Update Mistral-3.1 example | `examples/offline_inference/mistral-small.py` |
| 2025-04-28 | [#17270](https://github.com/vllm-project/vllm/pull/17270) | merged | [Bugfix] Fix Mistral3 spatial merge error | `vllm/model_executor/models/mistral3.py` |
| 2025-04-29 | [#17195](https://github.com/vllm-project/vllm/pull/17195) | merged | [Misc] Add a Jinja template to support Mistral3 function calling | `examples/tool_chat_template_mistral3.jinja` |
| 2025-04-30 | [#17428](https://github.com/vllm-project/vllm/pull/17428) | merged | Support LoRA for Mistral3 | `vllm/model_executor/models/mistral3.py` |
| 2025-05-08 | [#17644](https://github.com/vllm-project/vllm/pull/17644) | merged | [Bugfix] Fix tool call template validation for Mistral models | `examples/tool_chat_template_mistral3.jinja` |
| 2025-06-05 | [#19193](https://github.com/vllm-project/vllm/pull/19193) | merged | [mistral_common] Add v11 tokenizer | `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py` |
| 2025-06-26 | [#20093](https://github.com/vllm-project/vllm/pull/20093) | merged | [Bugfix] Fix Mistral tool-parser regex for nested JSON | `tests/models/language/generation/test_mistral.py` |
| 2025-07-07 | [#20570](https://github.com/vllm-project/vllm/pull/20570) | merged | [Config] Refactor mistral configs | `vllm/transformers_utils/configs/mistral.py` |
| 2025-08-20 | [#21945](https://github.com/vllm-project/vllm/pull/21945) | merged | Migrate Mistral3ImagePixelInputs to TensorSchema | `vllm/model_executor/models/mistral3.py` |
| 2025-10-09 | [#26358](https://github.com/vllm-project/vllm/pull/26358) | merged | Refactor MistralTokenizer | `tests/reasoning/test_mistral_reasoning_parser.py` |
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | merged | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-21 | [#29172](https://github.com/vllm-project/vllm/pull/29172) | merged | Fix mistral config | `vllm/transformers_utils/configs/mistral.py` |
| 2025-11-21 | [#28659](https://github.com/vllm-project/vllm/pull/28659) | merged | Default model load/config/tokenizer to `mistral` format if relevant files exist | `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py` |
| 2025-11-22 | [#29239](https://github.com/vllm-project/vllm/pull/29239) | merged | [Bugfix] Use HF config fields as fallback when loading Mistral config | `vllm/transformers_utils/configs/mistral.py` |
| 2025-12-02 | [#29757](https://github.com/vllm-project/vllm/pull/29757) | merged | Add Mistral Large 3 and Ministral 3 | `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py` |
| 2025-12-02 | [#29918](https://github.com/vllm-project/vllm/pull/29918) | merged | [BUGFIX] Fix regex pattern for Mistral Tool Call | `tests/models/language/generation/test_mistral.py` |
| 2025-12-11 | [#30391](https://github.com/vllm-project/vllm/pull/30391) | merged | [IMPROVEMENT] Change MistralReasoningParser behavior | `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py` |
| 2025-12-15 | [#30588](https://github.com/vllm-project/vllm/pull/30588) | closed | Fix edge case Mistral tool parser | `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/qwen3_vl.py` |
| 2025-12-23 | [#30724](https://github.com/vllm-project/vllm/pull/30724) | merged | Fix edge case Mistral tool parser | `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tool_parsers/test_mistral_tool_parser.py` |
| 2025-12-26 | [#31138](https://github.com/vllm-project/vllm/pull/31138) | merged | [Mistral common] Ensure all functions are imported from the top & only use public methods | `vllm/tokenizers/mistral.py` |
| 2026-01-22 | [#32780](https://github.com/vllm-project/vllm/pull/32780) | merged | [Llama.py -> mistral.py] Extract mistral-only relevant code into separate file | `vllm/model_executor/models/mistral.py` |
| 2026-01-26 | [#33095](https://github.com/vllm-project/vllm/pull/33095) | merged | Remove unused logic in `models/mistral.py` | `vllm/model_executor/models/mistral.py` |
| 2026-01-31 | [#33174](https://github.com/vllm-project/vllm/pull/33174) | merged | Add support for Mistral Large 3 inference with Flashinfer MoE | `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` |
| 2026-02-02 | [#33521](https://github.com/vllm-project/vllm/pull/33521) | merged | Fix mistral sliding window parsing | `vllm/transformers_utils/configs/mistral.py` |
| 2026-02-07 | [#33939](https://github.com/vllm-project/vllm/pull/33939) | merged | Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3 | `vllm/model_executor/models/mistral3.py` |
| 2026-02-12 | [#34104](https://github.com/vllm-project/vllm/pull/34104) | merged | Fix Mistral config remap to accept compressed-tensors quantization #34028 | `vllm/transformers_utils/configs/mistral.py` |
| 2026-02-23 | [#34651](https://github.com/vllm-project/vllm/pull/34651) | merged | [Feature] Lazy import for the "mistral" tokenizer module. | `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/utils/mistral.py` |
| 2026-03-06 | [#36156](https://github.com/vllm-project/vllm/pull/36156) | merged | [Bugfix] Fix simple Mistral-Small example | `examples/offline_inference/mistral-small.py` |
| 2026-03-11 | [#36782](https://github.com/vllm-project/vllm/pull/36782) | merged | [Bugfix] Fix Mistral-small `--format` | `examples/offline_inference/mistral-small.py` |
| 2026-03-11 | [#36163](https://github.com/vllm-project/vllm/pull/36163) | merged | Add support to Mistral large 3 eagle with dense layers | `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral_large_3_eagle.py` |
| 2026-03-14 | [#36971](https://github.com/vllm-project/vllm/pull/36971) | merged | Mistral common v10 | `vllm/tokenizers/mistral.py` |
| 2026-03-16 | [#37104](https://github.com/vllm-project/vllm/pull/37104) | merged | Patch Mistral config | `vllm/transformers_utils/configs/mistral.py` |
| 2026-03-16 | [#37232](https://github.com/vllm-project/vllm/pull/37232) | merged | Fix EagleMistralLarge3Model initialization | `vllm/model_executor/models/mistral_large_3_eagle.py` |
| 2026-03-17 | [#37209](https://github.com/vllm-project/vllm/pull/37209) | merged | Fix some Mistral parser issues | `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py` |
| 2026-03-18 | [#36928](https://github.com/vllm-project/vllm/pull/36928) | merged | [LoRA][BugFix] Fix skipped LoRA adapters for Mistral3 | `vllm/model_executor/models/mistral3.py` |
| 2026-04-06 | [#38150](https://github.com/vllm-project/vllm/pull/38150) | merged | [Mistral Grammar] Support Grammar Factory | `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tokenizers_/test_mistral.py` |
| 2026-04-07 | [#37292](https://github.com/vllm-project/vllm/pull/37292) | merged | Fix Mistral yarn warning in Transformers v5 | `vllm/transformers_utils/configs/mistral.py` |
| 2026-04-16 | [#39217](https://github.com/vllm-project/vllm/pull/39217) | merged | [Mistral Grammar] Fix tool and reasoning parsing | `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py` |
| 2026-04-22 | [#40531](https://github.com/vllm-project/vllm/pull/40531) | merged | [Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output | `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py` |
| 2026-04-24 | [#40043](https://github.com/vllm-project/vllm/pull/40043) | merged | [Feature] Avoid eager import of the "mistral_common" package. | `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py` |
| 2026-04-24 | [#39294](https://github.com/vllm-project/vllm/pull/39294) | merged | [Bugfix][Parser] Fix Mistral tool parser for HF tokenizers | `vllm/tool_parsers/mistral_tool_parser.py` |
| 2026-04-28 | [#41024](https://github.com/vllm-project/vllm/pull/41024) | merged | [FEATURE] Add EagleMistralForCausalLM | `vllm/model_executor/models/mistral_eagle.py` |

## Per-PR Diff Audit Cards

### PR #1196 - [Mistral] Mistral-7B-v0.1 support

- Link: https://github.com/vllm-project/vllm/pull/1196
- Status/date: merged / 2023-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`; associated commits `bb1ba58f0647`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +571/-25, 795 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mistral] Mistral-7B-v0.1 support"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "[Mistral] Mistral-7B-v0.1 support"; the main implementation surface is `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` added +404/-0 (404 lines); hunks: -0,0 +1,404; symbols: MistralMLP, __init__, forward, MistralAttention, touching `MistralMLP, __init__, forward`; `vllm/transformers_utils/configs/mistral.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: MistralConfig, __init__, touching `MistralConfig, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` added +404/-0 (404 lines); hunks: -0,0 +1,404; symbols: MistralMLP, __init__, forward, MistralAttention
  - `vllm/transformers_utils/configs/mistral.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: MistralConfig, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -0,0 +1,404 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/transformers_utils/configs/mistral.py
@@ -0,0 +1,66 @@
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
+#
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` added +404/-0; `vllm/transformers_utils/configs/mistral.py` added +66/-0
- Risk and verification: Runtime changes concentrate in `vllm/config.py`, `vllm/core/block_manager.py`, `vllm/core/scheduler.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1220 - Fix Mistral model

- Link: https://github.com/vllm-project/vllm/pull/1220
- Status/date: merged / 2023-09-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`; associated commits `a8e98aee0c16`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +27/-14, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Mistral model"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral.py`; technical summary: Covers "Fix Mistral model"; the main implementation surface is `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,7 +29,6; -46,6 +45,7.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,7 +29,6; -46,6 +45,7
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -29,7 +29,6 @@
-from vllm.transformers_utils.configs.mistral import MistralConfig
@@ -46,6 +45,7 @@
+from vllm.transformers_utils.configs.mistral import MistralConfig
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1303 - [Minor] Fix comment in mistral.py

- Link: https://github.com/vllm-project/vllm/pull/1303
- Status/date: merged / 2023-10-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`; associated commits `b95ee898fe1c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Minor] Fix comment in mistral.py"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral.py`; technical summary: Covers "[Minor] Fix comment in mistral.py"; the main implementation surface is `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -20,7 +20,7 @@
-"""Inference-only LLaMA model compatible with HuggingFace weights.
+"""Inference-only Mistral model compatible with HuggingFace weights.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #1254 - Bump up transformers version & Remove MistralConfig

- Link: https://github.com/vllm-project/vllm/pull/1254
- Status/date: merged / 2023-10-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`; associated commits `e7c8555d0652`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +4/-81, 136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Bump up transformers version & Remove MistralConfig"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral.py`; technical summary: Covers "Bump up transformers version & Remove MistralConfig"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` removed +0/-66 (66 lines); hunks: -1,66 +0,0; symbols: MistralConfig, __init__, touching `MistralConfig, __init__`; `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,6 +29,7; -44,7 +45,6.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` removed +0/-66 (66 lines); hunks: -1,66 +0,0; symbols: MistralConfig, __init__
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,6 +29,7; -44,7 +45,6
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -1,66 +0,0 @@
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
-#
diff -- vllm/model_executor/models/mistral.py
@@ -29,6 +29,7 @@
+from transformers import MistralConfig
@@ -44,7 +45,6 @@
-from vllm.transformers_utils.configs.mistral import MistralConfig
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` removed +0/-66; `vllm/model_executor/models/mistral.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2868 - Migrate MistralForCausalLM to LlamaForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/2868
- Status/date: merged / 2024-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`; associated commits `344020c926ad`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +6/-379, 421 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate MistralForCausalLM to LlamaForCausalLM"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/model_executor/models/mistral.py`; technical summary: Covers "Migrate MistralForCausalLM to LlamaForCausalLM"; the main implementation surface is `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: MistralMLP, __init__, forward, MistralAttention, touching `MistralMLP, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: MistralMLP, __init__, forward, MistralAttention
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -1,377 +0,0 @@
-# coding=utf-8
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` removed +0/-377
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5649 - [Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models

- Link: https://github.com/vllm-project/vllm/pull/5649
- Status/date: merged / 2024-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_mistral.jinja`, `examples/tool_chat_template_mistral_parallel.jinja`; associated commits `e02ce498be2e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +2588/-83, 3136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `examples/tool_chat_template_mistral_parallel.jinja`, `examples/tool_chat_template_mistral.jinja`; technical summary: Covers "[Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models"; the main implementation surface is `examples/tool_chat_template_mistral_parallel.jinja`, `examples/tool_chat_template_mistral.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94; `examples/tool_chat_template_mistral.jinja` added +86/-0 (86 lines); hunks: -0,0 +1,86.
- Code diff details:
  - `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94
  - `examples/tool_chat_template_mistral.jinja` added +86/-0 (86 lines); hunks: -0,0 +1,86
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_mistral_parallel.jinja
@@ -0,0 +1,94 @@
+{%- if messages[0]["role"] == "system" %}
+    {%- set system_message = messages[0]["content"] %}
+    {%- set loop_messages = messages[1:] %}
+{%- else %}
+    {%- set loop_messages = messages %}
+{%- endif %}
diff -- examples/tool_chat_template_mistral.jinja
@@ -0,0 +1,86 @@
+{%- if messages[0]["role"] == "system" %}
+    {%- set system_message = messages[0]["content"] %}
+    {%- set loop_messages = messages[1:] %}
+{%- else %}
+    {%- set loop_messages = messages %}
+{%- endif %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0; `examples/tool_chat_template_mistral.jinja` added +86/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/__init__.py`, `tests/tool_use/conftest.py`, `tests/tool_use/test_chat_completions.py`, `tests/tool_use/test_parallel_tool_calls.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14977 - [Mistral-Small 3.1] Update docs and tests

- Link: https://github.com/vllm-project/vllm/pull/14977
- Status/date: merged / 2025-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/fixtures/mistral_small_3_chat.json`; associated commits `f863ffc96532`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +34/-60, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mistral-Small 3.1] Update docs and tests"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `tests/models/fixtures/mistral_small_3_chat.json`; technical summary: Covers "[Mistral-Small 3.1] Update docs and tests"; the main implementation surface is `tests/models/fixtures/mistral_small_3_chat.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0 (1 lines); hunks: -0,0 +1.
- Code diff details:
  - `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0 (1 lines); hunks: -0,0 +1
- Key code excerpts:

```diff
diff -- tests/models/fixtures/mistral_small_3_chat.json
@@ -0,0 +1 @@
+[[[1784, 3937, 6122, 1261, 7244, 10575, 28528, 1408, 1261, 32656, 11237, 1044, 7283, 2015, 1454, 1261, 38462, 4818, 1046, 2], "The image shows a black dog lying on a wooden floor,
```

- Reviewed files:
  - tests: `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_pixtral.py`, `tests/models/fixtures/mistral_small_3_chat.json`, `tests/models/fixtures/pixtral_chat_engine.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15184 - [Doc] Update Mistral Small 3.1/Pixtral example

- Link: https://github.com/vllm-project/vllm/pull/15184
- Status/date: merged / 2025-03-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 37 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Update Mistral Small 3.1/Pixtral example"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `examples/offline_inference/mistral-small.py`; technical summary: Covers "[Doc] Update Mistral Small 3.1/Pixtral example"; the main implementation surface is `examples/offline_inference/mistral-small.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/offline_inference/mistral-small.py` renamed +8/-2 (10 lines); hunks: -6,14 +6,16; -51,6 +53,8 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo, touching `run_simple_demo, run_advanced_demo`.
- Code diff details:
  - `examples/offline_inference/mistral-small.py` renamed +8/-2 (10 lines); hunks: -6,14 +6,16; -51,6 +53,8 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo
- Key code excerpts:

```diff
diff -- examples/offline_inference/mistral-small.py
@@ -6,14 +6,16 @@
-# This script is an offline demo for running Mistral-Small-3
+# This script is an offline demo for running Mistral-Small-3.1
-# vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer-mode mistral --limit-mm-per-prompt 'image=4' --max-model-len 16384
+# vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
+#   --tokenizer-mode mistral --config-format mistral --load-format mistral \
+#   --limit-mm-per-prompt 'image=4' --max-model-len 16384
```

- Reviewed files:
  - docs: `examples/offline_inference/mistral-small.py` renamed +8/-2
- Risk and verification: This is mostly docs/examples in `examples/offline_inference/mistral-small.py`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #15505 - [Model] Support Mistral3 in the HF Transformers format

- Link: https://github.com/vllm-project/vllm/pull/15505
- Status/date: merged / 2025-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `51d7c6a2b23e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +723/-4, 805 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Mistral3 in the HF Transformers format"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "[Model] Support Mistral3 in the HF Transformers format"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` added +656/-0 (656 lines); hunks: -0,0 +1,656; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, forward, touching `Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` added +656/-0 (656 lines); hunks: -0,0 +1,656; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -0,0 +1,656 @@
+# SPDX-License-Identifier: Apache-2.0
+from abc import abstractmethod
+from collections.abc import Iterable, Mapping, Sequence
+from functools import cached_property
+from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
+                    TypeVar, Union)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` added +656/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15950 - [V1] Support Mistral3 in V1

- Link: https://github.com/vllm-project/vllm/pull/15950
- Status/date: merged / 2025-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `f021b9799386`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-7, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V1] Support Mistral3 in V1"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "[V1] Support Mistral3 in V1"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +9/-6 (15 lines); hunks: -31,12 +31,12; -425,7 +425,7 @@ def init_vision_tower_for_llava(; symbols: Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration, _parse_and_validate_image_input, touching `Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +9/-6 (15 lines); hunks: -31,12 +31,12; -425,7 +425,7 @@ def init_vision_tower_for_llava(; symbols: Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration, _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -31,12 +31,12 @@
-from .interfaces import (MultiModalEmbeddings, SupportsMultiModal, SupportsPP,
-                         SupportsV0Only)
+from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
-from .vision import get_vision_encoder_info, select_patch_features
+from .vision import (get_vision_encoder_info, scatter_patch_features,
+                     select_patch_features)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +9/-6
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16147 - [Misc] Update Mistral-3.1 example

- Link: https://github.com/vllm-project/vllm/pull/16147
- Status/date: merged / 2025-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-8, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Update Mistral-3.1 example"; model line: Mistral Small 4; category: model implementation change; main diff: `examples/offline_inference/mistral-small.py`; technical summary: Covers "[Misc] Update Mistral-3.1 example"; the main implementation surface is `examples/offline_inference/mistral-small.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/offline_inference/mistral-small.py` modified +22/-8 (30 lines); hunks: -13,9 +13,14; -44,19 +49,22; symbols: run_simple_demo, run_advanced_demo, main, touching `run_simple_demo, run_advanced_demo, main`.
- Code diff details:
  - `examples/offline_inference/mistral-small.py` modified +22/-8 (30 lines); hunks: -13,9 +13,14; -44,19 +49,22; symbols: run_simple_demo, run_advanced_demo, main
- Key code excerpts:

```diff
diff -- examples/offline_inference/mistral-small.py
@@ -13,9 +13,14 @@
+# # Mistral format
+#
+# # HF format
+# vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
+#   --limit-mm-per-prompt 'image=4' --max-model-len 16384
@@ -44,19 +49,22 @@
```

- Reviewed files:
  - docs: `examples/offline_inference/mistral-small.py` modified +22/-8
- Risk and verification: This is mostly docs/examples in `examples/offline_inference/mistral-small.py`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17270 - [Bugfix] Fix Mistral3 spatial merge error

- Link: https://github.com/vllm-project/vllm/pull/17270
- Status/date: merged / 2025-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `cb3f2d8d10ff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-3, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Mistral3 spatial merge error"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "[Bugfix] Fix Mistral3 spatial merge error"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -272,6 +272,9 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, get_replacement, touching `_get_prompt_updates, get_replacement`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -272,6 +272,9 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, get_replacement
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -272,6 +272,9 @@ def _get_prompt_updates(
+        # Need to sneak in spatial_merge_size for Mistral3
+        vision_config.spatial_merge_size = getattr(hf_config,
+                                                   "spatial_merge_size", 1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`, `vllm/model_executor/models/pixtral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17195 - [Misc] Add a Jinja template to support Mistral3 function calling

- Link: https://github.com/vllm-project/vllm/pull/17195
- Status/date: merged / 2025-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_mistral3.jinja`; associated commits `96e06e3cb73f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +119/-0, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Add a Jinja template to support Mistral3 function calling"; model line: Mistral Small 4; category: bug fix; main diff: `examples/tool_chat_template_mistral3.jinja`; technical summary: Covers "[Misc] Add a Jinja template to support Mistral3 function calling"; the main implementation surface is `examples/tool_chat_template_mistral3.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_mistral3.jinja` added +119/-0 (119 lines); hunks: -0,0 +1,119.
- Code diff details:
  - `examples/tool_chat_template_mistral3.jinja` added +119/-0 (119 lines); hunks: -0,0 +1,119
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_mistral3.jinja
@@ -0,0 +1,119 @@
+{%- set today = strftime_now("%Y-%m-%d") %}
+{%- set default_system_message = "You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was la
+{{- bos_token }}
+{%- if messages[0]['role'] == 'system' %}
+    {%- if messages[0]['content'] is string %}
+        {%- set system_message = messages[0]['content'] %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_mistral3.jinja` added +119/-0
- Risk and verification: This is mostly docs/examples in `examples/tool_chat_template_mistral3.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17428 - Support LoRA for Mistral3

- Link: https://github.com/vllm-project/vllm/pull/17428
- Status/date: merged / 2025-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `a44c4f1d2f7c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-4, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support LoRA for Mistral3"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "Support LoRA for Mistral3"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +14/-3 (17 lines); hunks: -18,6 +18,7; -31,7 +32,8; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights, get_mm_mapping, touching `init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +14/-3 (17 lines); hunks: -18,6 +18,7; -31,7 +32,8; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights, get_mm_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -18,6 +18,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -31,7 +32,8 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
+                         SupportsMultiModal, SupportsPP)
@@ -382,8 +384,8 @@ def init_vision_tower_for_llava(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +14/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17644 - [Bugfix] Fix tool call template validation for Mistral models

- Link: https://github.com/vllm-project/vllm/pull/17644
- Status/date: merged / 2025-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_mistral3.jinja`; associated commits `ca04b97c9361`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix tool call template validation for Mistral models"; model line: Mistral Small 4; category: bug fix; main diff: `examples/tool_chat_template_mistral3.jinja`; technical summary: Covers "[Bugfix] Fix tool call template validation for Mistral models"; the main implementation surface is `examples/tool_chat_template_mistral3.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_mistral3.jinja` modified +9/-2 (11 lines); hunks: -29,7 +29,14; -116,4 +123,4.
- Code diff details:
  - `examples/tool_chat_template_mistral3.jinja` modified +9/-2 (11 lines); hunks: -29,7 +29,14; -116,4 +123,4
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_mistral3.jinja
@@ -29,7 +29,14 @@
-{%- for message in loop_messages | rejectattr("role", "equalto", "tool") | rejectattr("role", "equalto", "tool_results") | selectattr("tool_calls", "undefined") %}
+{%- set filtered_messages = [] %}
+{%- for message in loop_messages %}
+    {%- if message["role"] not in ["tool", "tool_results"] and not message.get("tool_calls") %}
+        {%- set filtered_messages = filtered_messages + [message] %}
+    {%- endif %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_mistral3.jinja` modified +9/-2
- Risk and verification: This is mostly docs/examples in `examples/tool_chat_template_mistral3.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #19193 - [mistral_common] Add v11 tokenizer

- Link: https://github.com/vllm-project/vllm/pull/19193
- Status/date: merged / 2025-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +32/-4, 70 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[mistral_common] Add v11 tokenizer"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`; technical summary: Covers "[mistral_common] Add v11 tokenizer"; the main implementation surface is `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4 (34 lines); hunks: -44,11 +44,17 @@ def is_valid_id(id: str) -> bool:; -70,6 +76,12 @@ def __init__(self, tokenizer: AnyTokenizer):; symbols: is_valid_id, _is_fn_name_regex_support, MistralToolParser, __init__, touching `is_valid_id, _is_fn_name_regex_support, MistralToolParser`; `vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -187,6 +187,8 @@ class MistralTokenizer(TokenizerBase):; symbols: MistralTokenizer, __init__, touching `MistralTokenizer, __init__`.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4 (34 lines); hunks: -44,11 +44,17 @@ def is_valid_id(id: str) -> bool:; -70,6 +76,12 @@ def __init__(self, tokenizer: AnyTokenizer):; symbols: is_valid_id, _is_fn_name_regex_support, MistralToolParser, __init__
  - `vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -187,6 +187,8 @@ class MistralTokenizer(TokenizerBase):; symbols: MistralTokenizer, __init__
- Key code excerpts:

```diff
diff -- vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py
@@ -44,11 +44,17 @@ def is_valid_id(id: str) -> bool:
+def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
+    return isinstance(model_tokenizer, MistralTokenizer) \
+        and model_tokenizer.version >= 11
-    Tool call parser for Mistral 7B Instruct v0.3, intended for use with the
-    examples/tool_chat_template_mistral.jinja template.
+    Tool call parser for Mistral 7B Instruct v0.3, intended for use with
diff -- vllm/transformers_utils/tokenizers/mistral.py
@@ -187,6 +187,8 @@ class MistralTokenizer(TokenizerBase):
+        _mistral_version_str = self.instruct.tokenizer.version.value
+        self.version: int = int(_mistral_version_str.split("v")[-1])
```

- Reviewed files:
  - runtime: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4; `vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20093 - [Bugfix] Fix Mistral tool-parser regex for nested JSON

- Link: https://github.com/vllm-project/vllm/pull/20093
- Status/date: merged / 2025-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/language/generation/test_mistral.py`; associated commits `754b00edb3fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-2, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Mistral tool-parser regex for nested JSON"; model line: Mistral Small 4; category: bug fix; main diff: `tests/models/language/generation/test_mistral.py`; technical summary: Covers "[Bugfix] Fix Mistral tool-parser regex for nested JSON"; the main implementation surface is `tests/models/language/generation/test_mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/language/generation/test_mistral.py` modified +51/-0 (51 lines); hunks: -10,6 +10,7; -318,3 +319,53 @@ def test_mistral_guided_decoding(; symbols: test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer, __init__, touching `test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer`.
- Code diff details:
  - `tests/models/language/generation/test_mistral.py` modified +51/-0 (51 lines); hunks: -10,6 +10,7; -318,3 +319,53 @@ def test_mistral_guided_decoding(; symbols: test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer, __init__
- Key code excerpts:

```diff
diff -- tests/models/language/generation/test_mistral.py
@@ -10,6 +10,7 @@
+from vllm.transformers_utils.tokenizer import MistralTokenizer
@@ -318,3 +319,53 @@ def test_mistral_guided_decoding(
+def test_mistral_function_call_nested_json():
+    """Ensure that the function-name regex captures the entire outer-most
+    JSON block, including nested braces."""
+    # Create a minimal stub tokenizer that provides the few attributes the
```

- Reviewed files:
  - tests: `tests/models/language/generation/test_mistral.py` modified +51/-0
- Risk and verification: The diff ships test coverage in `tests/models/language/generation/test_mistral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20570 - [Config] Refactor mistral configs

- Link: https://github.com/vllm-project/vllm/pull/20570
- Status/date: merged / 2025-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `14601f5fba13`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +167/-113, 320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Config] Refactor mistral configs"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "[Config] Refactor mistral configs"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args, touching `adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -0,0 +1,120 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import Any
+from transformers import PretrainedConfig
+from vllm.logger import init_logger
+logger = init_logger(__name__)
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` added +120/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama.py`, `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21945 - Migrate Mistral3ImagePixelInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/21945
- Status/date: merged / 2025-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `c4477f55e581`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-21, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate Mistral3ImagePixelInputs to TensorSchema"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "Migrate Mistral3ImagePixelInputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +17/-21 (38 lines); hunks: -3,7 +3,7; -32,6 +32,7; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, _validate_pixel_values, touching `Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +17/-21 (38 lines); hunks: -3,7 +3,7; -32,6 +32,7; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, _validate_pixel_values
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -3,7 +3,7 @@
-from typing import (Final, Literal, Optional, Protocol, TypedDict, TypeVar,
+from typing import (Annotated, Final, Literal, Optional, Protocol, TypeVar,
@@ -32,6 +32,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -42,16 +43,24 @@
-class Mistral3ImagePixelInputs(TypedDict):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +17/-21
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26358 - Refactor MistralTokenizer

- Link: https://github.com/vllm-project/vllm/pull/26358
- Status/date: merged / 2025-10-09
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_mistral_reasoning_parser.py`; associated commits `c6187f55f7c4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +2349/-461, 3215 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor MistralTokenizer"; model line: Mistral Small 4; category: model implementation change; main diff: `tests/reasoning/test_mistral_reasoning_parser.py`; technical summary: Covers "Refactor MistralTokenizer"; the main implementation surface is `tests/reasoning/test_mistral_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27 (28 lines); hunks: -2,8 +2,6; -14,33 +12,9; symbols: mistral_tokenizer, touching `mistral_tokenizer`.
- Code diff details:
  - `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27 (28 lines); hunks: -2,8 +2,6; -14,33 +12,9; symbols: mistral_tokenizer
- Key code excerpts:

```diff
diff -- tests/reasoning/test_mistral_reasoning_parser.py
@@ -2,8 +2,6 @@
-from mistral_common.tokens.tokenizers.base import SpecialTokens
-from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
@@ -14,33 +12,9 @@
-    # TODO(Julien): upon model release change to a tokenizer already configured.
-    # =================================================================
-        "mistralai/Devstral-Small-2507"
```

- Reviewed files:
  - tests: `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27
- Risk and verification: The diff ships test coverage in `tests/entrypoints/test_chat_utils.py`, `tests/models/multimodal/generation/test_pixtral.py`, `tests/models/multimodal/generation/test_voxtral.py`, `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- Link: https://github.com/vllm-project/vllm/pull/28542
- Status/date: merged / 2025-11-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 104 files, +544/-912, 4603 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope, touching `get_rope`; `vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation, touching `NemotronConfig, __init__, _rope_scaling_validation`; `vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope
  - `vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation
  - `vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/openpangu.py` modified +7/-19 (26 lines); hunks: -77,6 +77,7; -259,7 +260,6 @@ def __init__(; symbols: check_ffn_act_fn, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/__init__.py
@@ -26,23 +26,23 @@ def get_rope(
-    base: float,
-    rope_scaling: dict[str, Any] | None = None,
+    rope_parameters: dict[str, Any] | None = None,
-    if rope_scaling is not None:
+    if rope_parameters is not None:
-        rope_scaling_tuple = {
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):
-        rope_theta (`float`, *optional*, defaults to 10000.0):
-            The base period of the RoPE embeddings.
+        rope_parameters (`dict`, *optional*):
+            The parameters of the RoPE embeddings.
@@ -132,8 +132,7 @@ def __init__(
-        rope_theta=10000.0,
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -27,7 +27,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/transformers_utils/configs/nemotron.py` modified +31/-29; `vllm/model_executor/models/deepseek_v2.py` modified +13/-30; `vllm/model_executor/models/chameleon.py` modified +4/-25; `vllm/model_executor/models/openpangu.py` modified +7/-19; `vllm/model_executor/models/hunyuan_v1.py` modified +2/-23
- Risk and verification: The diff ships test coverage in `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29172 - Fix mistral config

- Link: https://github.com/vllm-project/vllm/pull/29172
- Status/date: merged / 2025-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `434f3d3eb869`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix mistral config"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Fix mistral config"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +4/-0 (4 lines); hunks: -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args, touching `_remap_mistral_yarn_args`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +4/-0 (4 lines); hunks: -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:
+    if rope_theta := config.pop("rope_theta", None):
+        config["rope_parameters"]["rope_theta"] = rope_theta
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28659 - Default model load/config/tokenizer to `mistral` format if relevant files exist

- Link: https://github.com/vllm-project/vllm/pull/28659
- Status/date: merged / 2025-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`; associated commits `57430fc95c8a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +230/-34, 497 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Default model load/config/tokenizer to `mistral` format if relevant files exist"; model line: Mistral Small 4; category: model implementation change; main diff: `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Default model load/config/tokenizer to `mistral` format if relevant files exist"; the main implementation surface is `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/language/generation/test_mistral.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def test_mistral_format(; symbols: test_mistral_format, touching `test_mistral_format`; `vllm/transformers_utils/configs/mistral.py` modified +1/-1 (2 lines); hunks: -118,7 +118,7 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: _remap_general_mistral_args, touching `_remap_general_mistral_args`.
- Code diff details:
  - `tests/models/language/generation/test_mistral.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def test_mistral_format(; symbols: test_mistral_format
  - `vllm/transformers_utils/configs/mistral.py` modified +1/-1 (2 lines); hunks: -118,7 +118,7 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: _remap_general_mistral_args
- Key code excerpts:

```diff
diff -- tests/models/language/generation/test_mistral.py
@@ -208,7 +208,7 @@ def test_mistral_format(
-        tokenizer_mode="auto",
+        tokenizer_mode="hf",
diff -- vllm/transformers_utils/configs/mistral.py
@@ -118,7 +118,7 @@ def _remap_general_mistral_args(config: dict) -> dict:
-        "max_seq_len": ("max_seq_len", 128_000),
+        "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
```

- Reviewed files:
  - tests: `tests/models/language/generation/test_mistral.py` modified +1/-1
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/language/generation/test_mistral.py`, `tests/models/multimodal/test_mapping.py`, `tests/models/quantization/test_bitsandbytes.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29239 - [Bugfix] Use HF config fields as fallback when loading Mistral config

- Link: https://github.com/vllm-project/vllm/pull/29239
- Status/date: merged / 2025-11-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `d1cf8214e523`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +25/-4, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Use HF config fields as fallback when loading Mistral config"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "[Bugfix] Use HF config fields as fallback when loading Mistral config"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +10/-3 (13 lines); hunks: -9,14 +9,18; -52,6 +56,9 @@ def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -...; symbols: adapt_config_dict, touching `adapt_config_dict`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +10/-3 (13 lines); hunks: -9,14 +9,18; -52,6 +56,9 @@ def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -...; symbols: adapt_config_dict
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -9,14 +9,18 @@
-def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -> PretrainedConfig:
-    config_dict.update(kwargs)
+def adapt_config_dict(
+    config_dict: dict[str, Any],
+    defaults: dict[str, Any],
+) -> PretrainedConfig:
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +10/-3
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29757 - Add Mistral Large 3 and Ministral 3

- Link: https://github.com/vllm-project/vllm/pull/29757
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/models/mistral_large_3.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/tokenizers/mistral.py`, `vllm/transformers_utils/configs/mistral.py`; associated commits `d8c6210eeaa7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +724/-30, 1015 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Mistral Large 3 and Ministral 3"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Add Mistral Large 3 and Ministral 3"; the main implementation surface is `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM, touching `EagleMistralLarge3Model, __init__, forward`; `tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunks: -91,6 +91,118; -1108,13 +1220,6 @@ def test_decode(; symbols: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int, touching `test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty`; `vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunks: -18,9 +18,31 @@ def adapt_config_dict(; -140,17 +162,20 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args, touching `adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args`; `vllm/model_executor/models/mistral_large_3.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds, touching `MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds`.
- Code diff details:
  - `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM
  - `tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunks: -91,6 +91,118; -1108,13 +1220,6 @@ def test_decode(; symbols: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int
  - `vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunks: -18,9 +18,31 @@ def adapt_config_dict(; -140,17 +162,20 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args
  - `vllm/model_executor/models/mistral_large_3.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds
  - `vllm/tokenizers/mistral.py` modified +36/-0 (36 lines); hunks: -97,6 +97,8 @@ def _prepare_apply_chat_template_tools_and_messages(; -139,6 +141,33 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, decode, batch_decode, convert_tokens_to_string
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral_large_3_eagle.py
@@ -0,0 +1,165 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from functools import partial
+import torch
+import torch.nn as nn
diff -- tests/tokenizers_/test_mistral.py
@@ -91,6 +91,118 @@
+        (
+            {
+                "messages": [
+                    {
+                        "role": "user",
+                        "content": "What is the current local date and time?",
diff -- vllm/transformers_utils/configs/mistral.py
@@ -18,9 +18,31 @@ def adapt_config_dict(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0; `vllm/transformers_utils/configs/mistral.py` modified +62/-12; `vllm/model_executor/models/mistral_large_3.py` added +63/-0; `vllm/tokenizers/mistral.py` modified +36/-0
  - tests: `tests/tokenizers_/test_mistral.py` modified +151/-7
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/tokenizers_/test_mistral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29918 - [BUGFIX] Fix regex pattern for Mistral Tool Call

- Link: https://github.com/vllm-project/vllm/pull/29918
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/language/generation/test_mistral.py`; associated commits `1b1e35aaf9d9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +36/-1, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BUGFIX] Fix regex pattern for Mistral Tool Call"; model line: Mistral Small 4; category: bug fix; main diff: `tests/models/language/generation/test_mistral.py`; technical summary: Covers "[BUGFIX] Fix regex pattern for Mistral Tool Call"; the main implementation surface is `tests/models/language/generation/test_mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/language/generation/test_mistral.py` modified +35/-0 (35 lines); hunks: -315,3 +315,38 @@ def get_vocab():; symbols: get_vocab, touching `get_vocab`.
- Code diff details:
  - `tests/models/language/generation/test_mistral.py` modified +35/-0 (35 lines); hunks: -315,3 +315,38 @@ def get_vocab():; symbols: get_vocab
- Key code excerpts:

```diff
diff -- tests/models/language/generation/test_mistral.py
@@ -315,3 +315,38 @@ def get_vocab():
+    # multiple calls
+    multiple_args_dict = [
+        {
+            "city": "Dallas",
+            "state": "TX",
+            "unit": "fahrenheit",
```

- Reviewed files:
  - tests: `tests/models/language/generation/test_mistral.py` modified +35/-0
- Risk and verification: The diff ships test coverage in `tests/models/language/generation/test_mistral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30391 - [IMPROVEMENT] Change MistralReasoningParser behavior

- Link: https://github.com/vllm-project/vllm/pull/30391
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`; associated commits `aa3c250c487e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +186/-64, 383 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[IMPROVEMENT] Change MistralReasoningParser behavior"; model line: Mistral Small 4; category: bug fix; main diff: `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`; technical summary: Covers "[IMPROVEMENT] Change MistralReasoningParser behavior"; the main implementation surface is `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61 (145 lines); hunks: -18,47 +18,53 @@ def mistral_tokenizer():; -78,17 +84,17 @@ def mistral_tokenizer():; symbols: mistral_tokenizer, touching `mistral_tokenizer`; `vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3 (105 lines); hunks: -3,20 +3,29; -53,3 +62,93 @@ def end_token(self) -> str:; symbols: MistralReasoningParser, __init__, end_token, is_reasoning_end, touching `MistralReasoningParser, __init__, end_token`.
- Code diff details:
  - `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61 (145 lines); hunks: -18,47 +18,53 @@ def mistral_tokenizer():; -78,17 +84,17 @@ def mistral_tokenizer():; symbols: mistral_tokenizer
  - `vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3 (105 lines); hunks: -3,20 +3,29; -53,3 +62,93 @@ def end_token(self) -> str:; symbols: MistralReasoningParser, __init__, end_token, is_reasoning_end
- Key code excerpts:

```diff
diff -- tests/reasoning/test_mistral_reasoning_parser.py
@@ -18,47 +18,53 @@ def mistral_tokenizer():
-SIMPLE_REASONING = {
+INVALID_SIMPLE_REASONING = {
-    "reasoning": "This is a reasoning section",
-    "content": "This is the rest",
-    "is_reasoning_end": True,
+    "reasoning": None,
diff -- vllm/reasoning/mistral_reasoning_parser.py
@@ -3,20 +3,29 @@
+from vllm.entrypoints.openai.protocol import (
+    ChatCompletionRequest,
+    ResponsesRequest,
+)
-from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
+from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
```

- Reviewed files:
  - tests: `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61
  - runtime: `vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_mistral_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30588 - Fix edge case Mistral tool parser

- Link: https://github.com/vllm-project/vllm/pull/30588
- Status/date: closed / 2025-12-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 241 files, +6757/-2646, 15228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix edge case Mistral tool parser"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/qwen3_vl.py`; technical summary: Covers "Fix edge case Mistral tool parser"; the main implementation surface is `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/audioflamingo3.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder, __init__, touching `AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder`; `vllm/model_executor/models/bagel.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: BagelImagePixelInputs, BagelVisionMLP, __init__, forward, touching `BagelImagePixelInputs, BagelVisionMLP, __init__`; `vllm/model_executor/models/qwen3_vl.py` modified +448/-34 (482 lines); hunks: -50,7 +50,7; -67,12 +67,19; symbols: __init__, forward, touching `__init__, forward`; `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt, touching `build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt`.
- Code diff details:
  - `vllm/model_executor/models/audioflamingo3.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder, __init__
  - `vllm/model_executor/models/bagel.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: BagelImagePixelInputs, BagelVisionMLP, __init__, forward
  - `vllm/model_executor/models/qwen3_vl.py` modified +448/-34 (482 lines); hunks: -50,7 +50,7; -67,12 +67,19; symbols: __init__, forward
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt
  - `tests/entrypoints/openai/test_sparse_tensor_validation.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _encode_tensor, _create_malicious_sparse_tensor, _create_valid_sparse_tensor, _create_valid_dense_tensor
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/audioflamingo3.py
@@ -0,0 +1,639 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
+# reserved.
+#
diff -- vllm/model_executor/models/bagel.py
@@ -0,0 +1,584 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 Bytedance Ltd. and/or its affiliates.
+"""Inference-only BAGEL model compatible with HuggingFace weights.
+BAGEL is a unified multimodal model for image understanding and generation.
+For vLLM, we focus on the image understanding (vision-to-text) capabilities.
diff -- vllm/model_executor/models/qwen3_vl.py
@@ -50,7 +50,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/audioflamingo3.py` added +639/-0; `vllm/model_executor/models/bagel.py` added +584/-0; `vllm/model_executor/models/qwen3_vl.py` modified +448/-34; `vllm/attention/layers/mm_encoder_attention.py` added +284/-0; `vllm/model_executor/layers/quantization/fp8.py` modified +157/-92
  - tests: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0; `tests/entrypoints/openai/test_sparse_tensor_validation.py` added +342/-0; `tests/models/multimodal/generation/test_whisper.py` modified +123/-111
- Risk and verification: The diff ships test coverage in `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh`, `tests/compile/distributed/test_fusions_e2e.py`, `tests/compile/test_dynamic_shapes_compilation.py`, `tests/conftest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30724 - Fix edge case Mistral tool parser

- Link: https://github.com/vllm-project/vllm/pull/30724
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `38c361f99dff`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +115/-56, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix edge case Mistral tool parser"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tool_parsers/test_mistral_tool_parser.py`; technical summary: Covers "Fix edge case Mistral tool parser"; the main implementation surface is `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tool_parsers/test_mistral_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54 (135 lines); hunks: -131,78 +131,105 @@ def extract_tool_calls(; symbols: extract_tool_calls, extract_tool_calls_streaming, touching `extract_tool_calls, extract_tool_calls_streaming`; `tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2 (36 lines); hunks: -281,6 +281,8 @@ def test_extract_tool_calls_pre_v11_tokenizer(; -326,6 +328,36 @@ def test_extract_tool_calls_pre_v11_tokenizer(; symbols: test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming, touching `test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming`.
- Code diff details:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54 (135 lines); hunks: -131,78 +131,105 @@ def extract_tool_calls(; symbols: extract_tool_calls, extract_tool_calls_streaming
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2 (36 lines); hunks: -281,6 +281,8 @@ def test_extract_tool_calls_pre_v11_tokenizer(; -326,6 +328,36 @@ def test_extract_tool_calls_pre_v11_tokenizer(; symbols: test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -131,78 +131,105 @@ def extract_tool_calls(
-        Extract the tool calls from a complete model response. Requires
-        find-and-replacing single quotes with double quotes for JSON parsing,
-        make sure your tool call arguments don't ever include quotes!
+        Extract the tool calls from a complete model response.
+        Content and tool calls formatting depends on the Mistral's tokenizer version
+        used to train the model:
diff -- tests/tool_parsers/test_mistral_tool_parser.py
@@ -281,6 +281,8 @@ def test_extract_tool_calls_pre_v11_tokenizer(
+        "complex",
+        "wrong_json",
@@ -326,6 +328,36 @@ def test_extract_tool_calls_pre_v11_tokenizer(
+        (
+            # Complex
+            """hi{hi[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')""",  # noqa: E501
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_mistral_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31138 - [Mistral common] Ensure all functions are imported from the top & only use public methods

- Link: https://github.com/vllm-project/vllm/pull/31138
- Status/date: merged / 2025-12-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/mistral.py`; associated commits `48e744976cf4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +24/-57, 181 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mistral common] Ensure all functions are imported from the top & only use public methods"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/tokenizers/mistral.py`; technical summary: Covers "[Mistral common] Ensure all functions are imported from the top & only use public methods"; the main implementation surface is `vllm/tokenizers/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/mistral.py` modified +20/-53 (73 lines); hunks: -3,17 +3,28; -101,8 +112,6 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id, from_pretrained, touching `_prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id`.
- Code diff details:
  - `vllm/tokenizers/mistral.py` modified +20/-53 (73 lines); hunks: -3,17 +3,28; -101,8 +112,6 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id, from_pretrained
- Key code excerpts:

```diff
diff -- vllm/tokenizers/mistral.py
@@ -3,17 +3,28 @@
+from mistral_common.protocol.instruct.request import (
+    ChatCompletionRequest as MistralChatCompletionRequest,
+)
+from mistral_common.protocol.instruct.tool_calls import Function, Tool
+from mistral_common.protocol.instruct.validator import ValidationMode
+from mistral_common.tokens.tokenizers.base import (
```

- Reviewed files:
  - runtime: `vllm/tokenizers/mistral.py` modified +20/-53
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32780 - [Llama.py -> mistral.py] Extract mistral-only relevant code into separate file

- Link: https://github.com/vllm-project/vllm/pull/32780
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`; associated commits `1579c9b5fd0f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +248/-115, 426 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama.py -> mistral.py] Extract mistral-only relevant code into separate file"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mistral.py`; technical summary: Covers "[Llama.py -> mistral.py] Extract mistral-only relevant code into separate file"; the main implementation surface is `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: MistralAttention, __init__, _get_llama_4_attn_scale, forward, touching `MistralAttention, __init__, _get_llama_4_attn_scale`.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: MistralAttention, __init__, _get_llama_4_attn_scale, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -0,0 +1,242 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Mistral adaptation of the LLaMA architecture."""
+from collections.abc import Iterable
+import torch
+from torch import nn
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` added +242/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/mistral.py`, `vllm/model_executor/models/registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33095 - Remove unused logic in `models/mistral.py`

- Link: https://github.com/vllm-project/vllm/pull/33095
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral.py`; associated commits `d56afd45fd4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-8, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove unused logic in `models/mistral.py`"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/model_executor/models/mistral.py`; technical summary: Covers "Remove unused logic in `models/mistral.py`"; the main implementation surface is `vllm/model_executor/models/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral.py` modified +0/-8 (8 lines); hunks: -156,16 +156,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mistral.py` modified +0/-8 (8 lines); hunks: -156,16 +156,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -156,16 +156,8 @@ def __init__(
-        quant_config = self.get_quant_config(vllm_config)
-        do_fusion = getattr(
-            quant_config, "enable_quantization_scaling_fusion", False
-        ) and vllm_config.cache_config.cache_dtype.startswith("fp8")
-        if do_fusion:
-            self.input_layernorm.quant_scaling_from = self.self_attn.qkv_proj
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral.py` modified +0/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- Link: https://github.com/vllm-project/vllm/pull/33174
- Status/date: merged / 2026-01-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1104/-31, 1278 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support for Mistral Large 3 inference with Flashinfer MoE"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`; technical summary: Covers "Add support for Mistral Large 3 inference with Flashinfer MoE"; the main implementation surface is `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=4096,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33521 - Fix mistral sliding window parsing

- Link: https://github.com/vllm-project/vllm/pull/33521
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `beb889948276`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-22, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix mistral sliding window parsing"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Fix mistral sliding window parsing"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +25/-9 (34 lines); hunks: -14,6 +14,7 @@ def adapt_config_dict(; -161,6 +162,29 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window, _remap_mistral_quantization_args, touching `adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +25/-9 (34 lines); hunks: -14,6 +14,7 @@ def adapt_config_dict(; -161,6 +162,29 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window, _remap_mistral_quantization_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -14,6 +14,7 @@ def adapt_config_dict(
+    config_dict = _remap_mistral_sliding_window(config_dict)
@@ -161,6 +162,29 @@ def _remap_general_mistral_args(config: dict) -> dict:
+def _remap_mistral_sliding_window(config: dict) -> dict:
+    # Remap sliding_window (list) -> layer_types (list) + sliding window (int)
+    # for HF compatibility
+    # Mistral configs may define sliding_window as list[int]. Convert it
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +25/-9
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33939 - Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3

- Link: https://github.com/vllm-project/vllm/pull/33939
- Status/date: merged / 2026-02-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `4df44c16ba8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-1, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3"; model line: Mistral Small 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +9/-1 (10 lines); hunks: -44,6 +44,7; -408,7 +409,7 @@ def init_vision_tower_for_llava(; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str, set_aux_hidden_state_layers, touching `init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +9/-1 (10 lines); hunks: -44,6 +44,7; -408,7 +409,7 @@ def init_vision_tower_for_llava(; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str, set_aux_hidden_state_layers
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -44,6 +44,7 @@
+    SupportsEagle3,
@@ -408,7 +409,7 @@ def init_vision_tower_for_llava(
-    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP
+    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP, SupportsEagle3
@@ -432,6 +433,13 @@ def get_placeholder_str(cls, modality: str, i: int) -> str | None:
+    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34104 - Fix Mistral config remap to accept compressed-tensors quantization #34028

- Link: https://github.com/vllm-project/vllm/pull/34104
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `f5897613fb27`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Mistral config remap to accept compressed-tensors quantization #34028"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Fix Mistral config remap to accept compressed-tensors quantization #34028"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +8/-0 (8 lines); hunks: -198,6 +198,14 @@ def _remap_mistral_quantization_args(config: dict) -> dict:; symbols: _remap_mistral_quantization_args, touching `_remap_mistral_quantization_args`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +8/-0 (8 lines); hunks: -198,6 +198,14 @@ def _remap_mistral_quantization_args(config: dict) -> dict:; symbols: _remap_mistral_quantization_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -198,6 +198,14 @@ def _remap_mistral_quantization_args(config: dict) -> dict:
+        elif (
+            str(quantization.get("quant_method", "")).lower().replace("_", "-")
+            == "compressed-tensors"
+        ):
+            # Pass through compressed-tensors config, while normalizing
+            # quant_method to the canonical community spelling.
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34651 - [Feature] Lazy import for the "mistral" tokenizer module.

- Link: https://github.com/vllm-project/vllm/pull/34651
- Status/date: merged / 2026-02-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`; associated commits `54e2f83d0a82`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +68/-48, 399 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Lazy import for the "mistral" tokenizer module."; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/utils/mistral.py`; technical summary: Covers "[Feature] Lazy import for the "mistral" tokenizer module."; the main implementation surface is `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/utils/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6 (10 lines); hunks: -25,10 +25,10; -66,9 +66,7 @@ def is_valid_id(id: str) -> bool:; symbols: is_valid_id, _is_pre_v11_tokeniser, MistralToolParser, __init__, touching `is_valid_id, _is_pre_v11_tokeniser, MistralToolParser`; `vllm/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -210,6 +210,8 @@ def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | by...; symbols: _tekken_token_to_id, MistralTokenizer, from_pretrained, touching `_tekken_token_to_id, MistralTokenizer, from_pretrained`; `vllm/utils/mistral.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: is_mistral_tokenizer, attribute, touching `is_mistral_tokenizer, attribute`.
- Code diff details:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6 (10 lines); hunks: -25,10 +25,10; -66,9 +66,7 @@ def is_valid_id(id: str) -> bool:; symbols: is_valid_id, _is_pre_v11_tokeniser, MistralToolParser, __init__
  - `vllm/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -210,6 +210,8 @@ def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | by...; symbols: _tekken_token_to_id, MistralTokenizer, from_pretrained
  - `vllm/utils/mistral.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: is_mistral_tokenizer, attribute
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -25,10 +25,10 @@
-from vllm.tokenizers.mistral import MistralTokenizer
+from vllm.utils.mistral import is_mistral_tokenizer
@@ -66,9 +66,7 @@ def is_valid_id(id: str) -> bool:
-    return not (
-        isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11
-    )
diff -- vllm/tokenizers/mistral.py
@@ -210,6 +210,8 @@ def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | bytes) -> int:
+    IS_MISTRAL_TOKENIZER = True  # used by vllm.utils.mistral
diff -- vllm/utils/mistral.py
@@ -0,0 +1,28 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Provides lazy import of the vllm.tokenizers.mistral module."""
+from __future__ import annotations
+from typing import TYPE_CHECKING, TypeGuard
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6; `vllm/tokenizers/mistral.py` modified +2/-0; `vllm/utils/mistral.py` added +28/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/reasoning/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36156 - [Bugfix] Fix simple Mistral-Small example

- Link: https://github.com/vllm-project/vllm/pull/36156
- Status/date: merged / 2026-03-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix simple Mistral-Small example"; model line: Mistral Small 4; category: bug fix; main diff: `examples/offline_inference/mistral-small.py`; technical summary: Covers "[Bugfix] Fix simple Mistral-Small example"; the main implementation surface is `examples/offline_inference/mistral-small.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/offline_inference/mistral-small.py` modified +5/-2 (7 lines); hunks: -7,6 +7,7; -79,8 +80,10 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo, touching `run_simple_demo`.
- Code diff details:
  - `examples/offline_inference/mistral-small.py` modified +5/-2 (7 lines); hunks: -7,6 +7,7; -79,8 +80,10 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo
- Key code excerpts:

```diff
diff -- examples/offline_inference/mistral-small.py
@@ -7,6 +7,7 @@
+from vllm.multimodal.utils import encode_image_url
@@ -79,8 +80,10 @@ def run_simple_demo(args: argparse.Namespace):
-                    "type": "image_pil",
-                    "image_pil": ImageAsset("cherry_blossom").pil_image,
+                    "type": "image_url",
+                    "image_url": {
```

- Reviewed files:
  - docs: `examples/offline_inference/mistral-small.py` modified +5/-2
- Risk and verification: This is mostly docs/examples in `examples/offline_inference/mistral-small.py`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36782 - [Bugfix] Fix Mistral-small `--format`

- Link: https://github.com/vllm-project/vllm/pull/36782
- Status/date: merged / 2026-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-6, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Mistral-small `--format`"; model line: Mistral Small 4; category: bug fix; main diff: `examples/offline_inference/mistral-small.py`; technical summary: Covers "[Bugfix] Fix Mistral-small `--format`"; the main implementation surface is `examples/offline_inference/mistral-small.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/offline_inference/mistral-small.py` modified +6/-6 (12 lines); hunks: -62,9 +62,9 @@ def run_simple_demo(args: argparse.Namespace):; -102,9 +102,9 @@ def run_advanced_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo, touching `run_simple_demo, run_advanced_demo`.
- Code diff details:
  - `examples/offline_inference/mistral-small.py` modified +6/-6 (12 lines); hunks: -62,9 +62,9 @@ def run_simple_demo(args: argparse.Namespace):; -102,9 +102,9 @@ def run_advanced_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo
- Key code excerpts:

```diff
diff -- examples/offline_inference/mistral-small.py
@@ -62,9 +62,9 @@ def run_simple_demo(args: argparse.Namespace):
-        tokenizer_mode="mistral" if args.format == "mistral" else "auto",
-        config_format="mistral" if args.format == "mistral" else "auto",
-        load_format="mistral" if args.format == "mistral" else "auto",
+        tokenizer_mode="mistral" if args.format == "mistral" else "hf",
+        config_format="mistral" if args.format == "mistral" else "hf",
+        load_format="mistral" if args.format == "mistral" else "hf",
```

- Reviewed files:
  - docs: `examples/offline_inference/mistral-small.py` modified +6/-6
- Risk and verification: This is mostly docs/examples in `examples/offline_inference/mistral-small.py`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #36163 - Add support to Mistral large 3 eagle with dense layers

- Link: https://github.com/vllm-project/vllm/pull/36163
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/transformers_utils/configs/mistral.py`; associated commits `afebeffbfbf2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +28/-1, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support to Mistral large 3 eagle with dense layers"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`; technical summary: Covers "Add support to Mistral large 3 eagle with dense layers"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +23/-0 (23 lines); hunks: -19,6 +19,10 @@ def adapt_config_dict(; -291,3 +295,22 @@ def _remap_moe_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args, touching `adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args`; `vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1 (6 lines); hunks: -1,6 +1,7; -33,7 +34,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +23/-0 (23 lines); hunks: -19,6 +19,10 @@ def adapt_config_dict(; -291,3 +295,22 @@ def _remap_moe_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args
  - `vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1 (6 lines); hunks: -1,6 +1,7; -33,7 +34,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -19,6 +19,10 @@ def adapt_config_dict(
+    is_mla = bool(config_dict.get("qk_nope_head_dim"))
+    if is_mla:
+        config_dict = _remap_mistral_mla_args(config_dict)
@@ -291,3 +295,22 @@ def _remap_moe_args(config: dict) -> dict:
+def _remap_mistral_mla_args(config: dict) -> dict:
+    if not config.get("moe"):
diff -- vllm/model_executor/models/mistral_large_3_eagle.py
@@ -1,6 +1,7 @@
+import copy
@@ -33,7 +34,9 @@ def __init__(
-        config = vllm_config.model_config.hf_config
+        config = copy.deepcopy(vllm_config.model_config.hf_config)
+        config.first_k_dense_replace += start_layer_id
@@ -53,6 +56,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +23/-0; `vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36971 - Mistral common v10

- Link: https://github.com/vllm-project/vllm/pull/36971
- Status/date: merged / 2026-03-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/mistral.py`; associated commits `e42b49bd69d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-3, 74 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Mistral common v10"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/tokenizers/mistral.py`; technical summary: Covers "Mistral common v10"; the main implementation surface is `vllm/tokenizers/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/mistral.py` modified +19/-0 (19 lines); hunks: -7,6 +7,9; -192,6 +195,15 @@ def validate_request_params(request: "ChatCompletionRequest"):; symbols: validate_request_params, _tekken_token_to_id, apply_chat_template, decode, touching `validate_request_params, _tekken_token_to_id, apply_chat_template`.
- Code diff details:
  - `vllm/tokenizers/mistral.py` modified +19/-0 (19 lines); hunks: -7,6 +7,9; -192,6 +195,15 @@ def validate_request_params(request: "ChatCompletionRequest"):; symbols: validate_request_params, _tekken_token_to_id, apply_chat_template, decode
- Key code excerpts:

```diff
diff -- vllm/tokenizers/mistral.py
@@ -7,6 +7,9 @@
+from mistral_common.protocol.instruct.request import (
+    ReasoningEffort,
+)
@@ -192,6 +195,15 @@ def validate_request_params(request: "ChatCompletionRequest"):
+    if request.reasoning_effort and request.reasoning_effort not in list(
+        ReasoningEffort
```

- Reviewed files:
  - runtime: `vllm/tokenizers/mistral.py` modified +19/-0
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37104 - Patch Mistral config

- Link: https://github.com/vllm-project/vllm/pull/37104
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `ffbc2e5bdbfb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +49/-30, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Patch Mistral config"; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Patch Mistral config"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +10/-7 (17 lines); hunks: -113,12 +113,13 @@ def _remap_mistral_vision_args(config: dict) -> dict:; -128,9 +129,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args, touching `_remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +10/-7 (17 lines); hunks: -113,12 +113,13 @@ def _remap_mistral_vision_args(config: dict) -> dict:; -128,9 +129,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -113,12 +113,13 @@ def _remap_mistral_vision_args(config: dict) -> dict:
-        "factor": "factor",
-        "original_max_position_embeddings": "original_max_position_embeddings",
-        "beta": "beta_fast",
-        "alpha": "beta_slow",
-        "apply_scale": "apply_yarn_scaling",
+        "factor": ("factor", float),
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +10/-7
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`, `vllm/transformers_utils/model_arch_config_convertor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37232 - Fix EagleMistralLarge3Model initialization

- Link: https://github.com/vllm-project/vllm/pull/37232
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral_large_3_eagle.py`; associated commits `7961486a9b74`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EagleMistralLarge3Model initialization"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral_large_3_eagle.py`; technical summary: Covers "Fix EagleMistralLarge3Model initialization"; the main implementation surface is `vllm/model_executor/models/mistral_large_3_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0 (1 lines); hunks: -74,6 +74,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0 (1 lines); hunks: -74,6 +74,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral_large_3_eagle.py
@@ -74,6 +74,7 @@ def __init__(
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral_large_3_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37209 - Fix some Mistral parser issues

- Link: https://github.com/vllm-project/vllm/pull/37209
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `5db91f0aaf35`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +42/-34, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix some Mistral parser issues"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`; technical summary: Covers "Fix some Mistral parser issues"; the main implementation surface is `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tokenizers/mistral.py` modified +27/-26 (53 lines); hunks: -15,8 +15,15; -26,21 +33,20; symbols: from_pretrained, __init__, convert_tokens_to_ids, touching `from_pretrained, __init__, convert_tokens_to_ids`; `vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3 (10 lines); hunks: -241,7 +241,10 @@ def extract_tool_calls_streaming(; -275,7 +278,8 @@ def _extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer, touching `extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer`.
- Code diff details:
  - `vllm/tokenizers/mistral.py` modified +27/-26 (53 lines); hunks: -15,8 +15,15; -26,21 +33,20; symbols: from_pretrained, __init__, convert_tokens_to_ids
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3 (10 lines); hunks: -241,7 +241,10 @@ def extract_tool_calls_streaming(; -275,7 +278,8 @@ def _extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer
- Key code excerpts:

```diff
diff -- vllm/tokenizers/mistral.py
@@ -15,8 +15,15 @@
+    Tokenizer,
+)
+from mistral_common.tokens.tokenizers.instruct import (
+    InstructTokenizerBase,
+    InstructTokenizerV13,
+)
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -241,7 +241,10 @@ def extract_tool_calls_streaming(
-        if self.bot_token_id not in current_token_ids:
+        has_bot_token = (
+            self.bot_token_id in current_token_ids or self.bot_token in current_text
+        )
+        if not has_bot_token:
@@ -275,7 +278,8 @@ def _extract_tool_calls_streaming(
```

- Reviewed files:
  - runtime: `vllm/tokenizers/mistral.py` modified +27/-26; `vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36928 - [LoRA][BugFix] Fix skipped LoRA adapters for Mistral3

- Link: https://github.com/vllm-project/vllm/pull/36928
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral3.py`; associated commits `5bc1da147fb0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA][BugFix] Fix skipped LoRA adapters for Mistral3"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/model_executor/models/mistral3.py`; technical summary: Covers "[LoRA][BugFix] Fix skipped LoRA adapters for Mistral3"; the main implementation surface is `vllm/model_executor/models/mistral3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(; symbols: Mistral3ForConditionalGeneration, touching `Mistral3ForConditionalGeneration`.
- Code diff details:
  - `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(; symbols: Mistral3ForConditionalGeneration
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(
+            # Some PEFT LoRAs are trained against the text submodule directly
+            # and produce names like `base_model.model.model.layers.*`.
+            "model.": "language_model.model.",
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mistral3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38150 - [Mistral Grammar] Support Grammar Factory

- Link: https://github.com/vllm-project/vllm/pull/38150
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tokenizers_/test_mistral.py`, `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `fef56c18555e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +601/-29, 816 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mistral Grammar] Support Grammar Factory"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tokenizers_/test_mistral.py`; technical summary: Covers "[Mistral Grammar] Support Grammar Factory"; the main implementation surface is `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tokenizers_/test_mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3 (347 lines); hunks: -3,19 +3,43; -40,6 +64,13 @@ def mistral_tool_parser(mistral_tokenizer):; symbols: mistral_tool_parser, non_mistral_parser, assert_tool_calls, test_fast_detokenization_text_detection_pre_v11, touching `mistral_tool_parser, non_mistral_parser, assert_tool_calls`; `vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9 (142 lines); hunks: -10,6 +10,18; -25,6 +37,7; symbols: StreamingState, MistralToolParser, __init__, adjust_request, touching `StreamingState, MistralToolParser, __init__`; `tests/tokenizers_/test_mistral.py` modified +28/-0 (28 lines); hunks: -3,8 +3,10; -2407,3 +2409,29 @@ def test_convert_ids_to_tokens(; symbols: test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer, touching `test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer`; `vllm/tokenizers/mistral.py` modified +25/-0 (25 lines); hunks: -1,9 +1,12; -45,6 +48,7; symbols: convert_ids_to_tokens, supports_grammar, grammar_factory, llg_tokenizer, touching `convert_ids_to_tokens, supports_grammar, grammar_factory`.
- Code diff details:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3 (347 lines); hunks: -3,19 +3,43; -40,6 +64,13 @@ def mistral_tool_parser(mistral_tokenizer):; symbols: mistral_tool_parser, non_mistral_parser, assert_tool_calls, test_fast_detokenization_text_detection_pre_v11
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9 (142 lines); hunks: -10,6 +10,18; -25,6 +37,7; symbols: StreamingState, MistralToolParser, __init__, adjust_request
  - `tests/tokenizers_/test_mistral.py` modified +28/-0 (28 lines); hunks: -3,8 +3,10; -2407,3 +2409,29 @@ def test_convert_ids_to_tokens(; symbols: test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer
  - `vllm/tokenizers/mistral.py` modified +25/-0 (25 lines); hunks: -1,9 +1,12; -45,6 +48,7; symbols: convert_ids_to_tokens, supports_grammar, grammar_factory, llg_tokenizer
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_mistral_tool_parser.py
@@ -3,19 +3,43 @@
+from unittest.mock import MagicMock, patch
-from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
+from mistral_common.protocol.instruct.tool_calls import (
+    FunctionCall,
+    ToolCall,
+)
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -10,6 +10,18 @@
+from mistral_common.protocol.instruct.tool_calls import (
+    NamedToolChoice as MistralNamedToolChoice,
+)
+from mistral_common.protocol.instruct.tool_calls import (
+    Tool as MistralTool,
+)
diff -- tests/tokenizers_/test_mistral.py
@@ -3,8 +3,10 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3; `tests/tokenizers_/test_mistral.py` modified +28/-0
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9; `vllm/tokenizers/mistral.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_mistral.py`, `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/v1/structured_output/test_backend_guidance.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37292 - Fix Mistral yarn warning in Transformers v5

- Link: https://github.com/vllm-project/vllm/pull/37292
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/configs/mistral.py`; associated commits `edcc37a8cee2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Mistral yarn warning in Transformers v5"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/transformers_utils/configs/mistral.py`; technical summary: Covers "Fix Mistral yarn warning in Transformers v5"; the main implementation surface is `vllm/transformers_utils/configs/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/mistral.py` modified +6/-0 (6 lines); hunks: -2,7 +2,9; -134,6 +136,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args, touching `_remap_mistral_yarn_args`.
- Code diff details:
  - `vllm/transformers_utils/configs/mistral.py` modified +6/-0 (6 lines); hunks: -2,7 +2,9; -134,6 +136,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -2,7 +2,9 @@
+from packaging.version import Version
+from transformers import __version__ as TRANSFORMERS_VERSION
@@ -134,6 +136,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:
+    # Ignore apply_yarn_scaling in Transformers > v5 RoPE validation to remove warnings
+    if Version(TRANSFORMERS_VERSION) >= Version("5.3.0.dev0"):
+        config["ignore_keys_at_rope_validation"] = {"apply_yarn_scaling"}
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/configs/mistral.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39217 - [Mistral Grammar] Fix tool and reasoning parsing

- Link: https://github.com/vllm-project/vllm/pull/39217
- Status/date: merged / 2026-04-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/tool_use/mistral/test_mistral_tool_calls.py`, `tests/tool_use/mistral/utils.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `c0722f22de71`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1601/-266, 2396 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Mistral Grammar] Fix tool and reasoning parsing"; model line: Mistral Small 4; category: bug fix; main diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`; technical summary: Covers "[Mistral Grammar] Fix tool and reasoning parsing"; the main implementation surface is `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180 (932 lines); hunks: -3,6 +3,7; -23,24 +24,33; symbols: mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools, touching `mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools`; `vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10 (188 lines); hunks: -1,12 +1,15; -37,14 +40,19; symbols: _is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult, adjust_request, touching `_is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult`; `vllm/tokenizers/mistral.py` modified +49/-38 (87 lines); hunks: -54,6 +54,50; -159,44 +203,11 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls, _prepare_apply_chat_template_tools_and_messages, touching `_pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls`; `tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3 (483 lines); hunks: -1,25 +1,198; -28,3 +201,307 @@ async def test_tool_call_with_tool_choice(client: openai.As...; symbols: _requires_tool_parser, _is_pre_v11, StreamedToolCallResult, _collect_streamed_tool_call, touching `_requires_tool_parser, _is_pre_v11, StreamedToolCallResult`.
- Code diff details:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180 (932 lines); hunks: -3,6 +3,7; -23,24 +24,33; symbols: mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10 (188 lines); hunks: -1,12 +1,15; -37,14 +40,19; symbols: _is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult, adjust_request
  - `vllm/tokenizers/mistral.py` modified +49/-38 (87 lines); hunks: -54,6 +54,50; -159,44 +203,11 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls, _prepare_apply_chat_template_tools_and_messages
  - `tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3 (483 lines); hunks: -1,25 +1,198; -28,3 +201,307 @@ async def test_tool_call_with_tool_choice(client: openai.As...; symbols: _requires_tool_parser, _is_pre_v11, StreamedToolCallResult, _collect_streamed_tool_call
  - `tests/tool_use/mistral/utils.py` modified +24/-10 (34 lines); hunks: -2,16 +2,7; -21,6 +12,11 @@ class ServerConfig(TypedDict, total=False):; symbols: ServerConfig
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_mistral_tool_parser.py
@@ -3,6 +3,7 @@
+from typing import Any
@@ -23,24 +24,33 @@
+from pydantic import ValidationError
+    DeltaFunctionCall,
+    ExtractedToolCallInformation,
+from vllm.entrypoints.openai.engine.protocol import FunctionCall as VllmFunctionCall
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -1,12 +1,15 @@
+from __future__ import annotations
+from dataclasses import dataclass
-from typing import Any
+from typing import TYPE_CHECKING, Any
@@ -37,14 +40,19 @@
+from vllm.reasoning.mistral_reasoning_parser import MistralReasoningParser
diff -- vllm/tokenizers/mistral.py
@@ -54,6 +54,50 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180; `tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3; `tests/tool_use/mistral/utils.py` modified +24/-10
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10; `vllm/tokenizers/mistral.py` modified +49/-38
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/tool_use/mistral/test_mistral_tool_calls.py`, `tests/tool_use/mistral/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40531 - [Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output

- Link: https://github.com/vllm-project/vllm/pull/40531
- Status/date: merged / 2026-04-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `cfa49213d778`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +66/-18, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output"; model line: Mistral Small 4; category: bug fix; main diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`; technical summary: Covers "[Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output"; the main implementation surface is `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9 (58 lines); hunks: -24,7 +24,6; -250,6 +249,7 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):; symbols: test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises, test_extract_tool_calls_pre_v11_regex_fallback_raises, touching `test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises`; `vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9 (26 lines); hunks: -479,21 +479,28 @@ def extract_tool_calls(; -504,7 +511,8 @@ def extract_tool_calls(; symbols: extract_tool_calls, touching `extract_tool_calls`.
- Code diff details:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9 (58 lines); hunks: -24,7 +24,6; -250,6 +249,7 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):; symbols: test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises, test_extract_tool_calls_pre_v11_regex_fallback_raises
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9 (26 lines); hunks: -479,21 +479,28 @@ def extract_tool_calls(; -504,7 +511,8 @@ def extract_tool_calls(; symbols: extract_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_mistral_tool_parser.py
@@ -24,7 +24,6 @@
-from pydantic import ValidationError
@@ -250,6 +249,7 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):
+        "trailing_data_after_json",
@@ -338,6 +338,24 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):
+        (
+            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]\nextra trailing data""",  # noqa: E501
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -479,21 +479,28 @@ def extract_tool_calls(
-                tool_calls = json.loads(stringified_tool_calls)
+                # Use raw_decode to parse the first valid JSON value,
+                # ignoring trailing tokens the model may emit after
+                # the tool call array.
+                tool_calls, _ = json.JSONDecoder().raw_decode(stringified_tool_calls)
-                # use a regex to find the part corresponding to the tool call.
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_mistral_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40043 - [Feature] Avoid eager import of the "mistral_common" package.

- Link: https://github.com/vllm-project/vllm/pull/40043
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`; associated commits `56bdf85e10b8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +47/-23, 194 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Avoid eager import of the "mistral_common" package."; model line: Mistral Small 4; category: model implementation change; main diff: `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`; technical summary: Covers "[Feature] Avoid eager import of the "mistral_common" package."; the main implementation surface is `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0 (2 lines); hunks: -118,6 +118,8 @@ class MistralToolParser(ToolParser):; symbols: MistralToolParser, touching `MistralToolParser`; `vllm/utils/mistral.py` modified +15/-0 (15 lines); hunks: -12,8 +12,10; -26,3 +28,16 @@ def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGu...; symbols: is_mistral_tokenizer, is_mistral_tool_parser, attribute, touching `is_mistral_tokenizer, is_mistral_tool_parser, attribute`.
- Code diff details:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0 (2 lines); hunks: -118,6 +118,8 @@ class MistralToolParser(ToolParser):; symbols: MistralToolParser
  - `vllm/utils/mistral.py` modified +15/-0 (15 lines); hunks: -12,8 +12,10; -26,3 +28,16 @@ def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGu...; symbols: is_mistral_tokenizer, is_mistral_tool_parser, attribute
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -118,6 +118,8 @@ class MistralToolParser(ToolParser):
+    IS_MISTRAL_TOOL_PARSER = True  # used by vllm.utils.mistral
diff -- vllm/utils/mistral.py
@@ -12,8 +12,10 @@
+    import vllm.tool_parsers.mistral_tool_parser as mtp
+    mtp = LazyLoader("mtp", globals(), "vllm.tool_parsers.mistral_tool_parser")
@@ -26,3 +28,16 @@ def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGuard[mt.MistralToken
+def is_mistral_tool_parser(cls: type | None) -> bool:
+    """Return true if *cls* is (a subclass of) MistralToolParser.
+    Uses a class attribute check so that importing
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0; `vllm/utils/mistral.py` modified +15/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/serve/render/serving.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39294 - [Bugfix][Parser] Fix Mistral tool parser for HF tokenizers

- Link: https://github.com/vllm-project/vllm/pull/39294
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/mistral_tool_parser.py`; associated commits `2ec18f5df43e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-4, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Parser] Fix Mistral tool parser for HF tokenizers"; model line: Mistral Small 4; category: bug fix; main diff: `vllm/tool_parsers/mistral_tool_parser.py`; technical summary: Covers "[Bugfix][Parser] Fix Mistral tool parser for HF tokenizers"; the main implementation surface is `vllm/tool_parsers/mistral_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4 (17 lines); hunks: -91,7 +91,12 @@ def is_valid_id(id: str) -> bool:; -137,15 +142,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: is_valid_id, _is_pre_v11_tokeniser, __init__, extract_tool_calls, touching `is_valid_id, _is_pre_v11_tokeniser, __init__`.
- Code diff details:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4 (17 lines); hunks: -91,7 +91,12 @@ def is_valid_id(id: str) -> bool:; -137,15 +142,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: is_valid_id, _is_pre_v11_tokeniser, __init__, extract_tool_calls
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/mistral_tool_parser.py
@@ -91,7 +91,12 @@ def is_valid_id(id: str) -> bool:
-    return not (is_mistral_tokenizer(model_tokenizer) and model_tokenizer.version >= 11)
+    if is_mistral_tokenizer(model_tokenizer):
+        return model_tokenizer.version < 11
+    # For HF tokenizers, check if [ARGS] token exists in vocab
+    # which indicates a v11+ equivalent tokenizer
+    vocab: dict[str, int] = getattr(model_tokenizer, "get_vocab", lambda: {})()
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/mistral_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41024 - [FEATURE] Add EagleMistralForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/41024
- Status/date: merged / 2026-04-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mistral_eagle.py`; associated commits `e9f8f31e9a4c`
- Diff scope read: GitHub Pull Request files API returned 3 files, +172/-0, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[FEATURE] Add EagleMistralForCausalLM"; model line: Mistral Small 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mistral_eagle.py`; technical summary: Covers "[FEATURE] Add EagleMistralForCausalLM"; the main implementation surface is `vllm/model_executor/models/mistral_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mistral_eagle.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: EagleMistralDecoderLayer, __init__, get_quant_config, EagleMistralModel, touching `EagleMistralDecoderLayer, __init__, get_quant_config`.
- Code diff details:
  - `vllm/model_executor/models/mistral_eagle.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: EagleMistralDecoderLayer, __init__, get_quant_config, EagleMistralModel
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mistral_eagle.py
@@ -0,0 +1,166 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+import torch
+import torch.nn as nn
+from vllm.compilation.decorators import support_torch_compile
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mistral_eagle.py` added +166/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
