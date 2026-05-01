# vllm Mistral Small 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/generate/multimodal/mistral-small_offline.py` | 无直接 PR 号提交 |
| `examples/tool_chat_template_mistral.jinja` | [#5649](https://github.com/vllm-project/vllm/pull/5649) |
| `examples/tool_chat_template_mistral3.jinja` | [#17195](https://github.com/vllm-project/vllm/pull/17195), [#17644](https://github.com/vllm-project/vllm/pull/17644) |
| `examples/tool_chat_template_mistral_parallel.jinja` | [#5649](https://github.com/vllm-project/vllm/pull/5649) |
| `tests/models/fixtures/ministral_3b_chat.json` | 无直接 PR 号提交 |
| `tests/models/fixtures/mistral_small_3_chat.json` | [#14977](https://github.com/vllm-project/vllm/pull/14977) |
| `tests/models/language/generation/test_mistral.py` | [#20093](https://github.com/vllm-project/vllm/pull/20093), [#28659](https://github.com/vllm-project/vllm/pull/28659), [#29918](https://github.com/vllm-project/vllm/pull/29918) |
| `tests/reasoning/test_mistral_reasoning_parser.py` | [#26358](https://github.com/vllm-project/vllm/pull/26358), [#30391](https://github.com/vllm-project/vllm/pull/30391) |
| `tests/renderers/test_mistral.py` | 无直接 PR 号提交 |
| `tests/tokenizers_/test_mistral.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#38150](https://github.com/vllm-project/vllm/pull/38150) |
| `tests/tool_parsers/test_mistral_tool_parser.py` | [#30724](https://github.com/vllm-project/vllm/pull/30724), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217), [#40531](https://github.com/vllm-project/vllm/pull/40531) |
| `tests/tool_use/mistral/__init__.py` | 无直接 PR 号提交 |
| `tests/tool_use/mistral/conftest.py` | 无直接 PR 号提交 |
| `tests/tool_use/mistral/test_mistral_tool_calls.py` | [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `tests/tool_use/mistral/utils.py` | [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `vllm/model_executor/models/mistral.py` | [#1196](https://github.com/vllm-project/vllm/pull/1196), [#1220](https://github.com/vllm-project/vllm/pull/1220), [#1254](https://github.com/vllm-project/vllm/pull/1254), [#1303](https://github.com/vllm-project/vllm/pull/1303), [#2868](https://github.com/vllm-project/vllm/pull/2868), [#32780](https://github.com/vllm-project/vllm/pull/32780), [#33095](https://github.com/vllm-project/vllm/pull/33095) |
| `vllm/model_executor/models/mistral3.py` | [#15505](https://github.com/vllm-project/vllm/pull/15505), [#15950](https://github.com/vllm-project/vllm/pull/15950), [#17270](https://github.com/vllm-project/vllm/pull/17270), [#17428](https://github.com/vllm-project/vllm/pull/17428), [#21945](https://github.com/vllm-project/vllm/pull/21945), [#33939](https://github.com/vllm-project/vllm/pull/33939), [#36928](https://github.com/vllm-project/vllm/pull/36928) |
| `vllm/model_executor/models/mistral_eagle.py` | [#41024](https://github.com/vllm-project/vllm/pull/41024) |
| `vllm/model_executor/models/mistral_large_3.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757) |
| `vllm/model_executor/models/mistral_large_3_eagle.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#36163](https://github.com/vllm-project/vllm/pull/36163), [#37232](https://github.com/vllm-project/vllm/pull/37232) |
| `vllm/reasoning/mistral_reasoning_parser.py` | [#30391](https://github.com/vllm-project/vllm/pull/30391) |
| `vllm/renderers/mistral.py` | 无直接 PR 号提交 |
| `vllm/tokenizers/mistral.py` | [#29757](https://github.com/vllm-project/vllm/pull/29757), [#31138](https://github.com/vllm-project/vllm/pull/31138), [#34651](https://github.com/vllm-project/vllm/pull/34651), [#36971](https://github.com/vllm-project/vllm/pull/36971), [#37209](https://github.com/vllm-project/vllm/pull/37209), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217) |
| `vllm/tool_parsers/mistral_tool_parser.py` | [#30724](https://github.com/vllm-project/vllm/pull/30724), [#34651](https://github.com/vllm-project/vllm/pull/34651), [#37209](https://github.com/vllm-project/vllm/pull/37209), [#38150](https://github.com/vllm-project/vllm/pull/38150), [#39217](https://github.com/vllm-project/vllm/pull/39217), [#39294](https://github.com/vllm-project/vllm/pull/39294), [#40043](https://github.com/vllm-project/vllm/pull/40043), [#40531](https://github.com/vllm-project/vllm/pull/40531) |
| `vllm/transformers_utils/configs/mistral.py` | [#1196](https://github.com/vllm-project/vllm/pull/1196), [#1254](https://github.com/vllm-project/vllm/pull/1254), [#20570](https://github.com/vllm-project/vllm/pull/20570), [#28659](https://github.com/vllm-project/vllm/pull/28659), [#29172](https://github.com/vllm-project/vllm/pull/29172), [#29239](https://github.com/vllm-project/vllm/pull/29239), [#29757](https://github.com/vllm-project/vllm/pull/29757), [#33521](https://github.com/vllm-project/vllm/pull/33521), [#34028](https://github.com/vllm-project/vllm/pull/34028), [#34104](https://github.com/vllm-project/vllm/pull/34104), [#36163](https://github.com/vllm-project/vllm/pull/36163), [#37104](https://github.com/vllm-project/vllm/pull/37104), ... (13 total) |
| `vllm/utils/mistral.py` | [#34651](https://github.com/vllm-project/vllm/pull/34651), [#40043](https://github.com/vllm-project/vllm/pull/40043) |

## PR 覆盖总览

- git 追溯 PR 数: 44
- 原文档显式引用补充 PR 数: 8
- 当前文档总 PR 数: 52
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #1196 - [Mistral] Mistral-7B-v0.1 support

- 链接: https://github.com/vllm-project/vllm/pull/1196
- 状态/时间: merged / 2023-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`；关联提交 `bb1ba58f0647`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+571/-25，可读 patch 795 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mistral] Mistral-7B-v0.1 support」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/mistral.py` added +404/-0 (404 lines); hunks: -0,0 +1,404; symbols: MistralMLP, __init__, forward, MistralAttention，涉及 `MistralMLP, __init__, forward`；`vllm/transformers_utils/configs/mistral.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: MistralConfig, __init__，涉及 `MistralConfig, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` added +404/-0 (404 lines); hunks: -0,0 +1,404; symbols: MistralMLP, __init__, forward, MistralAttention
  - `vllm/transformers_utils/configs/mistral.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: MistralConfig, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` added +404/-0; `vllm/transformers_utils/configs/mistral.py` added +66/-0
- 验证与风险: runtime 路径改动集中在 `vllm/config.py`, `vllm/core/block_manager.py`, `vllm/core/scheduler.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1220 - Fix Mistral model

- 链接: https://github.com/vllm-project/vllm/pull/1220
- 状态/时间: merged / 2023-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`；关联提交 `a8e98aee0c16`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+27/-14，可读 patch 124 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Mistral model」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral.py`；技术摘要: 覆盖「Fix Mistral model」；主要实现面是 `vllm/model_executor/models/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,7 +29,6; -46,6 +45,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,7 +29,6; -46,6 +45,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -29,7 +29,6 @@
-from vllm.transformers_utils.configs.mistral import MistralConfig
@@ -46,6 +45,7 @@
+from vllm.transformers_utils.configs.mistral import MistralConfig
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1303 - [Minor] Fix comment in mistral.py

- 链接: https://github.com/vllm-project/vllm/pull/1303
- 状态/时间: merged / 2023-10-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`；关联提交 `b95ee898fe1c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Minor] Fix comment in mistral.py」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mistral.py
@@ -20,7 +20,7 @@
-"""Inference-only LLaMA model compatible with HuggingFace weights.
+"""Inference-only Mistral model compatible with HuggingFace weights.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1254 - Bump up transformers version & Remove MistralConfig

- 链接: https://github.com/vllm-project/vllm/pull/1254
- 状态/时间: merged / 2023-10-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/configs/mistral.py`；关联提交 `e7c8555d0652`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+4/-81，可读 patch 136 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Bump up transformers version & Remove MistralConfig」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral.py`；技术摘要: 覆盖「Bump up transformers version & Remove MistralConfig」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` removed +0/-66 (66 lines); hunks: -1,66 +0,0; symbols: MistralConfig, __init__，涉及 `MistralConfig, __init__`；`vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,6 +29,7; -44,7 +45,6。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` removed +0/-66 (66 lines); hunks: -1,66 +0,0; symbols: MistralConfig, __init__
  - `vllm/model_executor/models/mistral.py` modified +1/-1 (2 lines); hunks: -29,6 +29,7; -44,7 +45,6
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` removed +0/-66; `vllm/model_executor/models/mistral.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/mistral.py`, `vllm/transformers_utils/config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #2868 - Migrate MistralForCausalLM to LlamaForCausalLM

- 链接: https://github.com/vllm-project/vllm/pull/2868
- 状态/时间: merged / 2024-02-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`；关联提交 `344020c926ad`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+6/-379，可读 patch 421 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate MistralForCausalLM to LlamaForCausalLM」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mistral.py`；技术摘要: 覆盖「Migrate MistralForCausalLM to LlamaForCausalLM」；主要实现面是 `vllm/model_executor/models/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: MistralMLP, __init__, forward, MistralAttention，涉及 `MistralMLP, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` removed +0/-377 (377 lines); hunks: -1,377 +0,0; symbols: MistralMLP, __init__, forward, MistralAttention
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` removed +0/-377
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/__init__.py`, `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5649 - [Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models

- 链接: https://github.com/vllm-project/vllm/pull/5649
- 状态/时间: merged / 2024-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_mistral.jinja`, `examples/tool_chat_template_mistral_parallel.jinja`；关联提交 `e02ce498be2e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+2588/-83，可读 patch 3136 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `examples/tool_chat_template_mistral_parallel.jinja`, `examples/tool_chat_template_mistral.jinja`；技术摘要: 覆盖「[Feature] OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models」；主要实现面是 `examples/tool_chat_template_mistral_parallel.jinja`, `examples/tool_chat_template_mistral.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94；`examples/tool_chat_template_mistral.jinja` added +86/-0 (86 lines); hunks: -0,0 +1,86。
- 代码 diff 细节:
  - `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94
  - `examples/tool_chat_template_mistral.jinja` added +86/-0 (86 lines); hunks: -0,0 +1,86
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_mistral_parallel.jinja` added +94/-0; `examples/tool_chat_template_mistral.jinja` added +86/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/__init__.py`, `tests/tool_use/conftest.py`, `tests/tool_use/test_chat_completions.py`, `tests/tool_use/test_parallel_tool_calls.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14977 - [Mistral-Small 3.1] Update docs and tests

- 链接: https://github.com/vllm-project/vllm/pull/14977
- 状态/时间: merged / 2025-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/fixtures/mistral_small_3_chat.json`；关联提交 `f863ffc96532`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+34/-60，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mistral-Small 3.1] Update docs and tests」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `tests/models/fixtures/mistral_small_3_chat.json`；技术摘要: 覆盖「[Mistral-Small 3.1] Update docs and tests」；主要实现面是 `tests/models/fixtures/mistral_small_3_chat.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0 (1 lines); hunks: -0,0 +1。
- 代码 diff 细节:
  - `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0 (1 lines); hunks: -0,0 +1
- 关键代码摘录:

```diff
diff -- tests/models/fixtures/mistral_small_3_chat.json
@@ -0,0 +1 @@
+[[[1784, 3937, 6122, 1261, 7244, 10575, 28528, 1408, 1261, 32656, 11237, 1044, 7283, 2015, 1454, 1261, 38462, 4818, 1046, 2], "The image shows a black dog lying on a wooden floor,
```

- 已读文件:
  - tests: `tests/models/fixtures/mistral_small_3_chat.json` added +1/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_pixtral.py`, `tests/models/fixtures/mistral_small_3_chat.json`, `tests/models/fixtures/pixtral_chat_engine.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15184 - [Doc] Update Mistral Small 3.1/Pixtral example

- 链接: https://github.com/vllm-project/vllm/pull/15184
- 状态/时间: merged / 2025-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-2，可读 patch 37 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Update Mistral Small 3.1/Pixtral example」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `examples/offline_inference/mistral-small.py`；技术摘要: 覆盖「[Doc] Update Mistral Small 3.1/Pixtral example」；主要实现面是 `examples/offline_inference/mistral-small.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/offline_inference/mistral-small.py` renamed +8/-2 (10 lines); hunks: -6,14 +6,16; -51,6 +53,8 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo，涉及 `run_simple_demo, run_advanced_demo`。
- 代码 diff 细节:
  - `examples/offline_inference/mistral-small.py` renamed +8/-2 (10 lines); hunks: -6,14 +6,16; -51,6 +53,8 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/offline_inference/mistral-small.py` renamed +8/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/offline_inference/mistral-small.py`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #15505 - [Model] Support Mistral3 in the HF Transformers format

- 链接: https://github.com/vllm-project/vllm/pull/15505
- 状态/时间: merged / 2025-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `51d7c6a2b23e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+723/-4，可读 patch 805 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Mistral3 in the HF Transformers format」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「[Model] Support Mistral3 in the HF Transformers format」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` added +656/-0 (656 lines); hunks: -0,0 +1,656; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, forward，涉及 `Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` added +656/-0 (656 lines); hunks: -0,0 +1,656; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` added +656/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15950 - [V1] Support Mistral3 in V1

- 链接: https://github.com/vllm-project/vllm/pull/15950
- 状态/时间: merged / 2025-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `f021b9799386`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-7，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] Support Mistral3 in V1」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mistral3.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +9/-6 (15 lines); hunks: -31,12 +31,12; -425,7 +425,7 @@ def init_vision_tower_for_llava(; symbols: Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration, _parse_and_validate_image_input，涉及 `Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +9/-6 (15 lines); hunks: -31,12 +31,12; -425,7 +425,7 @@ def init_vision_tower_for_llava(; symbols: Mistral3ImagePixelInputs, init_vision_tower_for_llava, Mistral3ForConditionalGeneration, _parse_and_validate_image_input
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +9/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16147 - [Misc] Update Mistral-3.1 example

- 链接: https://github.com/vllm-project/vllm/pull/16147
- 状态/时间: merged / 2025-04-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-8，可读 patch 77 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Update Mistral-3.1 example」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `examples/offline_inference/mistral-small.py`；技术摘要: 覆盖「[Misc] Update Mistral-3.1 example」；主要实现面是 `examples/offline_inference/mistral-small.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/offline_inference/mistral-small.py` modified +22/-8 (30 lines); hunks: -13,9 +13,14; -44,19 +49,22; symbols: run_simple_demo, run_advanced_demo, main，涉及 `run_simple_demo, run_advanced_demo, main`。
- 代码 diff 细节:
  - `examples/offline_inference/mistral-small.py` modified +22/-8 (30 lines); hunks: -13,9 +13,14; -44,19 +49,22; symbols: run_simple_demo, run_advanced_demo, main
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/offline_inference/mistral-small.py` modified +22/-8
- 验证与风险: 该 PR 主要落在文档/示例 `examples/offline_inference/mistral-small.py`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #17270 - [Bugfix] Fix Mistral3 spatial merge error

- 链接: https://github.com/vllm-project/vllm/pull/17270
- 状态/时间: merged / 2025-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `cb3f2d8d10ff`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+5/-3，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Mistral3 spatial merge error」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「[Bugfix] Fix Mistral3 spatial merge error」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -272,6 +272,9 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, get_replacement，涉及 `_get_prompt_updates, get_replacement`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -272,6 +272,9 @@ def _get_prompt_updates(; symbols: _get_prompt_updates, get_replacement
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -272,6 +272,9 @@ def _get_prompt_updates(
+        # Need to sneak in spatial_merge_size for Mistral3
+        vision_config.spatial_merge_size = getattr(hf_config,
+                                                   "spatial_merge_size", 1)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`, `vllm/model_executor/models/pixtral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17195 - [Misc] Add a Jinja template to support Mistral3 function calling

- 链接: https://github.com/vllm-project/vllm/pull/17195
- 状态/时间: merged / 2025-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_mistral3.jinja`；关联提交 `96e06e3cb73f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+119/-0，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Add a Jinja template to support Mistral3 function calling」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_mistral3.jinja`；技术摘要: 覆盖「[Misc] Add a Jinja template to support Mistral3 function calling」；主要实现面是 `examples/tool_chat_template_mistral3.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_mistral3.jinja` added +119/-0 (119 lines); hunks: -0,0 +1,119。
- 代码 diff 细节:
  - `examples/tool_chat_template_mistral3.jinja` added +119/-0 (119 lines); hunks: -0,0 +1,119
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_mistral3.jinja` added +119/-0
- 验证与风险: 该 PR 主要落在文档/示例 `examples/tool_chat_template_mistral3.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #17428 - Support LoRA for Mistral3

- 链接: https://github.com/vllm-project/vllm/pull/17428
- 状态/时间: merged / 2025-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `a44c4f1d2f7c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-4，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support LoRA for Mistral3」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「Support LoRA for Mistral3」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +14/-3 (17 lines); hunks: -18,6 +18,7; -31,7 +32,8; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights, get_mm_mapping，涉及 `init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +14/-3 (17 lines); hunks: -18,6 +18,7; -31,7 +32,8; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, load_weights, get_mm_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +14/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17644 - [Bugfix] Fix tool call template validation for Mistral models

- 链接: https://github.com/vllm-project/vllm/pull/17644
- 状态/时间: merged / 2025-05-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_mistral3.jinja`；关联提交 `ca04b97c9361`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix tool call template validation for Mistral models」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_mistral3.jinja`；技术摘要: 覆盖「[Bugfix] Fix tool call template validation for Mistral models」；主要实现面是 `examples/tool_chat_template_mistral3.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_mistral3.jinja` modified +9/-2 (11 lines); hunks: -29,7 +29,14; -116,4 +123,4。
- 代码 diff 细节:
  - `examples/tool_chat_template_mistral3.jinja` modified +9/-2 (11 lines); hunks: -29,7 +29,14; -116,4 +123,4
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_mistral3.jinja` modified +9/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/tool_chat_template_mistral3.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #19193 - [mistral_common] Add v11 tokenizer

- 链接: https://github.com/vllm-project/vllm/pull/19193
- 状态/时间: merged / 2025-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+32/-4，可读 patch 70 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[mistral_common] Add v11 tokenizer」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`；技术摘要: 覆盖「[mistral_common] Add v11 tokenizer」；主要实现面是 `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4 (34 lines); hunks: -44,11 +44,17 @@ def is_valid_id(id: str) -> bool:; -70,6 +76,12 @@ def __init__(self, tokenizer: AnyTokenizer):; symbols: is_valid_id, _is_fn_name_regex_support, MistralToolParser, __init__，涉及 `is_valid_id, _is_fn_name_regex_support, MistralToolParser`；`vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -187,6 +187,8 @@ class MistralTokenizer(TokenizerBase):; symbols: MistralTokenizer, __init__，涉及 `MistralTokenizer, __init__`。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4 (34 lines); hunks: -44,11 +44,17 @@ def is_valid_id(id: str) -> bool:; -70,6 +76,12 @@ def __init__(self, tokenizer: AnyTokenizer):; symbols: is_valid_id, _is_fn_name_regex_support, MistralToolParser, __init__
  - `vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -187,6 +187,8 @@ class MistralTokenizer(TokenizerBase):; symbols: MistralTokenizer, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py` modified +30/-4; `vllm/transformers_utils/tokenizers/mistral.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py`, `vllm/transformers_utils/tokenizers/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20093 - [Bugfix] Fix Mistral tool-parser regex for nested JSON

- 链接: https://github.com/vllm-project/vllm/pull/20093
- 状态/时间: merged / 2025-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/language/generation/test_mistral.py`；关联提交 `754b00edb3fd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+53/-2，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Mistral tool-parser regex for nested JSON」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `tests/models/language/generation/test_mistral.py`；技术摘要: 覆盖「[Bugfix] Fix Mistral tool-parser regex for nested JSON」；主要实现面是 `tests/models/language/generation/test_mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/language/generation/test_mistral.py` modified +51/-0 (51 lines); hunks: -10,6 +10,7; -318,3 +319,53 @@ def test_mistral_guided_decoding(; symbols: test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer, __init__，涉及 `test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer`。
- 代码 diff 细节:
  - `tests/models/language/generation/test_mistral.py` modified +51/-0 (51 lines); hunks: -10,6 +10,7; -318,3 +319,53 @@ def test_mistral_guided_decoding(; symbols: test_mistral_guided_decoding, test_mistral_function_call_nested_json, _StubMistralTokenizer, __init__
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/language/generation/test_mistral.py` modified +51/-0
- 验证与风险: diff 自带测试面 `tests/models/language/generation/test_mistral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20570 - [Config] Refactor mistral configs

- 链接: https://github.com/vllm-project/vllm/pull/20570
- 状态/时间: merged / 2025-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `14601f5fba13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+167/-113，可读 patch 320 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Config] Refactor mistral configs」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「[Config] Refactor mistral configs」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args，涉及 `adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` added +120/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama.py`, `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21945 - Migrate Mistral3ImagePixelInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/21945
- 状态/时间: merged / 2025-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `c4477f55e581`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-21，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate Mistral3ImagePixelInputs to TensorSchema」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「Migrate Mistral3ImagePixelInputs to TensorSchema」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +17/-21 (38 lines); hunks: -3,7 +3,7; -32,6 +32,7; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, _validate_pixel_values，涉及 `Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +17/-21 (38 lines); hunks: -3,7 +3,7; -32,6 +32,7; symbols: Mistral3ImagePixelInputs, Mistral3PatchMerger, __init__, _validate_pixel_values
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +17/-21
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26358 - Refactor MistralTokenizer

- 链接: https://github.com/vllm-project/vllm/pull/26358
- 状态/时间: merged / 2025-10-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_mistral_reasoning_parser.py`；关联提交 `c6187f55f7c4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+2349/-461，可读 patch 3215 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor MistralTokenizer」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `tests/reasoning/test_mistral_reasoning_parser.py`；技术摘要: 覆盖「Refactor MistralTokenizer」；主要实现面是 `tests/reasoning/test_mistral_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27 (28 lines); hunks: -2,8 +2,6; -14,33 +12,9; symbols: mistral_tokenizer，涉及 `mistral_tokenizer`。
- 代码 diff 细节:
  - `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27 (28 lines); hunks: -2,8 +2,6; -14,33 +12,9; symbols: mistral_tokenizer
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_mistral_reasoning_parser.py` modified +1/-27
- 验证与风险: diff 自带测试面 `tests/entrypoints/test_chat_utils.py`, `tests/models/multimodal/generation/test_pixtral.py`, `tests/models/multimodal/generation/test_voxtral.py`, `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28542 - Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/28542
- 状态/时间: merged / 2025-11-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 104 个文件，+544/-912，可读 patch 4603 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5」；主要实现面是 `vllm/model_executor/layers/rotary_embedding/__init__.py`, `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope，涉及 `get_rope`；`vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation，涉及 `NemotronConfig, __init__, _rope_scaling_validation`；`vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38 (76 lines); hunks: -26,23 +26,23 @@ def get_rope(; -60,15 +60,15 @@ def get_rope(; symbols: get_rope
  - `vllm/transformers_utils/configs/nemotron.py` modified +31/-29 (60 lines); hunks: -88,8 +88,8 @@ class NemotronConfig(PretrainedConfig):; -132,8 +132,7 @@ def __init__(; symbols: NemotronConfig, __init__, _rope_scaling_validation
  - `vllm/model_executor/models/deepseek_v2.py` modified +13/-30 (43 lines); hunks: -27,7 +27,6; -111,8 +110,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/chameleon.py` modified +4/-25 (29 lines); hunks: -264,8 +264,7 @@ def __init__(; -292,7 +291,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/openpangu.py` modified +7/-19 (26 lines); hunks: -77,6 +77,7; -259,7 +260,6 @@ def __init__(; symbols: check_ffn_act_fn, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/__init__.py` modified +38/-38; `vllm/transformers_utils/configs/nemotron.py` modified +31/-29; `vllm/model_executor/models/deepseek_v2.py` modified +13/-30; `vllm/model_executor/models/chameleon.py` modified +4/-25; `vllm/model_executor/models/openpangu.py` modified +7/-19; `vllm/model_executor/models/hunyuan_v1.py` modified +2/-23
- 验证与风险: diff 自带测试面 `tests/compile/test_functionalization.py`, `tests/kernels/core/test_mrope.py`, `tests/kernels/core/test_pos_encoding.py`, `tests/kernels/moe/test_gpt_oss_triton_kernels.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29172 - Fix mistral config

- 链接: https://github.com/vllm-project/vllm/pull/29172
- 状态/时间: merged / 2025-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `434f3d3eb869`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix mistral config」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Fix mistral config」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +4/-0 (4 lines); hunks: -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args，涉及 `_remap_mistral_yarn_args`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +4/-0 (4 lines); hunks: -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/mistral.py
@@ -90,6 +90,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:
+    if rope_theta := config.pop("rope_theta", None):
+        config["rope_parameters"]["rope_theta"] = rope_theta
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28659 - Default model load/config/tokenizer to `mistral` format if relevant files exist

- 链接: https://github.com/vllm-project/vllm/pull/28659
- 状态/时间: merged / 2025-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`；关联提交 `57430fc95c8a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+230/-34，可读 patch 497 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Default model load/config/tokenizer to `mistral` format if relevant files exist」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Default model load/config/tokenizer to `mistral` format if relevant files exist」；主要实现面是 `tests/models/language/generation/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/language/generation/test_mistral.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def test_mistral_format(; symbols: test_mistral_format，涉及 `test_mistral_format`；`vllm/transformers_utils/configs/mistral.py` modified +1/-1 (2 lines); hunks: -118,7 +118,7 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: _remap_general_mistral_args，涉及 `_remap_general_mistral_args`。
- 代码 diff 细节:
  - `tests/models/language/generation/test_mistral.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def test_mistral_format(; symbols: test_mistral_format
  - `vllm/transformers_utils/configs/mistral.py` modified +1/-1 (2 lines); hunks: -118,7 +118,7 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: _remap_general_mistral_args
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/language/generation/test_mistral.py` modified +1/-1
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/language/generation/test_mistral.py`, `tests/models/multimodal/test_mapping.py`, `tests/models/quantization/test_bitsandbytes.py`, `tests/tool_use/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29239 - [Bugfix] Use HF config fields as fallback when loading Mistral config

- 链接: https://github.com/vllm-project/vllm/pull/29239
- 状态/时间: merged / 2025-11-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `d1cf8214e523`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+25/-4，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Use HF config fields as fallback when loading Mistral config」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「[Bugfix] Use HF config fields as fallback when loading Mistral config」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +10/-3 (13 lines); hunks: -9,14 +9,18; -52,6 +56,9 @@ def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -...; symbols: adapt_config_dict，涉及 `adapt_config_dict`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +10/-3 (13 lines); hunks: -9,14 +9,18; -52,6 +56,9 @@ def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -...; symbols: adapt_config_dict
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +10/-3
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29757 - Add Mistral Large 3 and Ministral 3

- 链接: https://github.com/vllm-project/vllm/pull/29757
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/models/mistral_large_3.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/tokenizers/mistral.py`, `vllm/transformers_utils/configs/mistral.py`；关联提交 `d8c6210eeaa7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+724/-30，可读 patch 1015 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Large 3 and Ministral 3」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Add Mistral Large 3 and Ministral 3」；主要实现面是 `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM，涉及 `EagleMistralLarge3Model, __init__, forward`；`tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunks: -91,6 +91,118; -1108,13 +1220,6 @@ def test_decode(; symbols: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int，涉及 `test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty`；`vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunks: -18,9 +18,31 @@ def adapt_config_dict(; -140,17 +162,20 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args，涉及 `adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args`；`vllm/model_executor/models/mistral_large_3.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds，涉及 `MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunks: -0,0 +1,165; symbols: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM
  - `tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunks: -91,6 +91,118; -1108,13 +1220,6 @@ def test_decode(; symbols: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int
  - `vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunks: -18,9 +18,31 @@ def adapt_config_dict(; -140,17 +162,20 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args
  - `vllm/model_executor/models/mistral_large_3.py` added +63/-0 (63 lines); hunks: -0,0 +1,63; symbols: MistralLarge3ForCausalLM, load_weights, _remap_mistral_to_ds
  - `vllm/tokenizers/mistral.py` modified +36/-0 (36 lines); hunks: -97,6 +97,8 @@ def _prepare_apply_chat_template_tools_and_messages(; -139,6 +141,33 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, decode, batch_decode, convert_tokens_to_string
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0; `vllm/transformers_utils/configs/mistral.py` modified +62/-12; `vllm/model_executor/models/mistral_large_3.py` added +63/-0; `vllm/tokenizers/mistral.py` modified +36/-0
  - tests: `tests/tokenizers_/test_mistral.py` modified +151/-7
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/tokenizers_/test_mistral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29918 - [BUGFIX] Fix regex pattern for Mistral Tool Call

- 链接: https://github.com/vllm-project/vllm/pull/29918
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/language/generation/test_mistral.py`；关联提交 `1b1e35aaf9d9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+36/-1，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] Fix regex pattern for Mistral Tool Call」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `tests/models/language/generation/test_mistral.py`；技术摘要: 覆盖「[BUGFIX] Fix regex pattern for Mistral Tool Call」；主要实现面是 `tests/models/language/generation/test_mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/language/generation/test_mistral.py` modified +35/-0 (35 lines); hunks: -315,3 +315,38 @@ def get_vocab():; symbols: get_vocab，涉及 `get_vocab`。
- 代码 diff 细节:
  - `tests/models/language/generation/test_mistral.py` modified +35/-0 (35 lines); hunks: -315,3 +315,38 @@ def get_vocab():; symbols: get_vocab
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/language/generation/test_mistral.py` modified +35/-0
- 验证与风险: diff 自带测试面 `tests/models/language/generation/test_mistral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30391 - [IMPROVEMENT] Change MistralReasoningParser behavior

- 链接: https://github.com/vllm-project/vllm/pull/30391
- 状态/时间: merged / 2025-12-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`；关联提交 `aa3c250c487e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+186/-64，可读 patch 383 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[IMPROVEMENT] Change MistralReasoningParser behavior」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`；技术摘要: 覆盖「[IMPROVEMENT] Change MistralReasoningParser behavior」；主要实现面是 `tests/reasoning/test_mistral_reasoning_parser.py`, `vllm/reasoning/mistral_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61 (145 lines); hunks: -18,47 +18,53 @@ def mistral_tokenizer():; -78,17 +84,17 @@ def mistral_tokenizer():; symbols: mistral_tokenizer，涉及 `mistral_tokenizer`；`vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3 (105 lines); hunks: -3,20 +3,29; -53,3 +62,93 @@ def end_token(self) -> str:; symbols: MistralReasoningParser, __init__, end_token, is_reasoning_end，涉及 `MistralReasoningParser, __init__, end_token`。
- 代码 diff 细节:
  - `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61 (145 lines); hunks: -18,47 +18,53 @@ def mistral_tokenizer():; -78,17 +84,17 @@ def mistral_tokenizer():; symbols: mistral_tokenizer
  - `vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3 (105 lines); hunks: -3,20 +3,29; -53,3 +62,93 @@ def end_token(self) -> str:; symbols: MistralReasoningParser, __init__, end_token, is_reasoning_end
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_mistral_reasoning_parser.py` modified +84/-61
  - runtime: `vllm/reasoning/mistral_reasoning_parser.py` modified +102/-3
- 验证与风险: diff 自带测试面 `tests/reasoning/test_mistral_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30588 - Fix edge case Mistral tool parser

- 链接: https://github.com/vllm-project/vllm/pull/30588
- 状态/时间: closed / 2025-12-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 241 个文件，+6757/-2646，可读 patch 15228 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix edge case Mistral tool parser」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/qwen3_vl.py`；技术摘要: 覆盖「Fix edge case Mistral tool parser」；主要实现面是 `vllm/model_executor/models/audioflamingo3.py`, `vllm/model_executor/models/bagel.py`, `vllm/model_executor/models/qwen3_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/audioflamingo3.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder, __init__，涉及 `AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder`；`vllm/model_executor/models/bagel.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: BagelImagePixelInputs, BagelVisionMLP, __init__, forward，涉及 `BagelImagePixelInputs, BagelVisionMLP, __init__`；`vllm/model_executor/models/qwen3_vl.py` modified +448/-34 (482 lines); hunks: -50,7 +50,7; -67,12 +67,19; symbols: __init__, forward，涉及 `__init__, forward`；`tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt，涉及 `build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt`。
- 代码 diff 细节:
  - `vllm/model_executor/models/audioflamingo3.py` added +639/-0 (639 lines); hunks: -0,0 +1,639; symbols: AudioFlamingo3FeatureInputs, AudioFlamingo3EmbeddingInputs, AudioFlamingo3Encoder, __init__
  - `vllm/model_executor/models/bagel.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: BagelImagePixelInputs, BagelVisionMLP, __init__, forward
  - `vllm/model_executor/models/qwen3_vl.py` modified +448/-34 (482 lines); hunks: -50,7 +50,7; -67,12 +67,19; symbols: __init__, forward
  - `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0 (434 lines); hunks: -0,0 +1,434; symbols: build_dots_ocr_prompt, build_processor_prompt, build_ovis_prompt, build_qwen2_5_video_prompt
  - `tests/entrypoints/openai/test_sparse_tensor_validation.py` added +342/-0 (342 lines); hunks: -0,0 +1,342; symbols: _encode_tensor, _create_malicious_sparse_tensor, _create_valid_sparse_tensor, _create_valid_dense_tensor
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/audioflamingo3.py` added +639/-0; `vllm/model_executor/models/bagel.py` added +584/-0; `vllm/model_executor/models/qwen3_vl.py` modified +448/-34; `vllm/attention/layers/mm_encoder_attention.py` added +284/-0; `vllm/model_executor/layers/quantization/fp8.py` modified +157/-92
  - tests: `tests/models/multimodal/generation/test_vit_backend_functionality.py` added +434/-0; `tests/entrypoints/openai/test_sparse_tensor_validation.py` added +342/-0; `tests/models/multimodal/generation/test_whisper.py` modified +123/-111
- 验证与风险: diff 自带测试面 `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh`, `tests/compile/distributed/test_fusions_e2e.py`, `tests/compile/test_dynamic_shapes_compilation.py`, `tests/conftest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30724 - Fix edge case Mistral tool parser

- 链接: https://github.com/vllm-project/vllm/pull/30724
- 状态/时间: merged / 2025-12-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `38c361f99dff`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+115/-56，可读 patch 224 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix edge case Mistral tool parser」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tool_parsers/test_mistral_tool_parser.py`；技术摘要: 覆盖「Fix edge case Mistral tool parser」；主要实现面是 `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tool_parsers/test_mistral_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54 (135 lines); hunks: -131,78 +131,105 @@ def extract_tool_calls(; symbols: extract_tool_calls, extract_tool_calls_streaming，涉及 `extract_tool_calls, extract_tool_calls_streaming`；`tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2 (36 lines); hunks: -281,6 +281,8 @@ def test_extract_tool_calls_pre_v11_tokenizer(; -326,6 +328,36 @@ def test_extract_tool_calls_pre_v11_tokenizer(; symbols: test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming，涉及 `test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming`。
- 代码 diff 细节:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54 (135 lines); hunks: -131,78 +131,105 @@ def extract_tool_calls(; symbols: extract_tool_calls, extract_tool_calls_streaming
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2 (36 lines); hunks: -281,6 +281,8 @@ def test_extract_tool_calls_pre_v11_tokenizer(; -326,6 +328,36 @@ def test_extract_tool_calls_pre_v11_tokenizer(; symbols: test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls, test_extract_tool_calls_streaming
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +81/-54
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +34/-2
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_mistral_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31138 - [Mistral common] Ensure all functions are imported from the top & only use public methods

- 链接: https://github.com/vllm-project/vllm/pull/31138
- 状态/时间: merged / 2025-12-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/mistral.py`；关联提交 `48e744976cf4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+24/-57，可读 patch 181 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mistral common] Ensure all functions are imported from the top & only use public methods」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/tokenizers/mistral.py`；技术摘要: 覆盖「[Mistral common] Ensure all functions are imported from the top & only use public methods」；主要实现面是 `vllm/tokenizers/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/mistral.py` modified +20/-53 (73 lines); hunks: -3,17 +3,28; -101,8 +112,6 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id, from_pretrained，涉及 `_prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id`。
- 代码 diff 细节:
  - `vllm/tokenizers/mistral.py` modified +20/-53 (73 lines); hunks: -3,17 +3,28; -101,8 +112,6 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _prepare_apply_chat_template_tools_and_messages, validate_request_params, _tekken_token_to_id, from_pretrained
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tokenizers/mistral.py` modified +20/-53
- 验证与风险: runtime 路径改动集中在 `vllm/tokenizers/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32780 - [Llama.py -> mistral.py] Extract mistral-only relevant code into separate file

- 链接: https://github.com/vllm-project/vllm/pull/32780
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`；关联提交 `1579c9b5fd0f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+248/-115，可读 patch 426 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Llama.py -> mistral.py] Extract mistral-only relevant code into separate file」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mistral.py`；技术摘要: 覆盖「[Llama.py -> mistral.py] Extract mistral-only relevant code into separate file」；主要实现面是 `vllm/model_executor/models/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: MistralAttention, __init__, _get_llama_4_attn_scale, forward，涉及 `MistralAttention, __init__, _get_llama_4_attn_scale`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: MistralAttention, __init__, _get_llama_4_attn_scale, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` added +242/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/mistral.py`, `vllm/model_executor/models/registry.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33095 - Remove unused logic in `models/mistral.py`

- 链接: https://github.com/vllm-project/vllm/pull/33095
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral.py`；关联提交 `d56afd45fd4e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-8，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove unused logic in `models/mistral.py`」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mistral.py`；技术摘要: 覆盖「Remove unused logic in `models/mistral.py`」；主要实现面是 `vllm/model_executor/models/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral.py` modified +0/-8 (8 lines); hunks: -156,16 +156,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral.py` modified +0/-8 (8 lines); hunks: -156,16 +156,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral.py` modified +0/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- 链接: https://github.com/vllm-project/vllm/pull/33174
- 状态/时间: merged / 2026-01-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1104/-31，可读 patch 1278 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support for Mistral Large 3 inference with Flashinfer MoE」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`；技术摘要: 覆盖「Add support for Mistral Large 3 inference with Flashinfer MoE」；主要实现面是 `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=4096,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_flashinfer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33521 - Fix mistral sliding window parsing

- 链接: https://github.com/vllm-project/vllm/pull/33521
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `beb889948276`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-22，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix mistral sliding window parsing」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Fix mistral sliding window parsing」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +25/-9 (34 lines); hunks: -14,6 +14,7 @@ def adapt_config_dict(; -161,6 +162,29 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window, _remap_mistral_quantization_args，涉及 `adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +25/-9 (34 lines); hunks: -14,6 +14,7 @@ def adapt_config_dict(; -161,6 +162,29 @@ def _remap_general_mistral_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_sliding_window, _remap_mistral_quantization_args
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +25/-9
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33939 - Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3

- 链接: https://github.com/vllm-project/vllm/pull/33939
- 状态/时间: merged / 2026-02-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `4df44c16ba8c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-1，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +9/-1 (10 lines); hunks: -44,6 +44,7; -408,7 +409,7 @@ def init_vision_tower_for_llava(; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str, set_aux_hidden_state_layers，涉及 `init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +9/-1 (10 lines); hunks: -44,6 +44,7; -408,7 +409,7 @@ def init_vision_tower_for_llava(; symbols: init_vision_tower_for_llava, Mistral3ForConditionalGeneration, get_placeholder_str, set_aux_hidden_state_layers
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +9/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34104 - Fix Mistral config remap to accept compressed-tensors quantization #34028

- 链接: https://github.com/vllm-project/vllm/pull/34104
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `f5897613fb27`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Mistral config remap to accept compressed-tensors quantization #34028」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Fix Mistral config remap to accept compressed-tensors quantization #34028」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +8/-0 (8 lines); hunks: -198,6 +198,14 @@ def _remap_mistral_quantization_args(config: dict) -> dict:; symbols: _remap_mistral_quantization_args，涉及 `_remap_mistral_quantization_args`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +8/-0 (8 lines); hunks: -198,6 +198,14 @@ def _remap_mistral_quantization_args(config: dict) -> dict:; symbols: _remap_mistral_quantization_args
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34651 - [Feature] Lazy import for the "mistral" tokenizer module.

- 链接: https://github.com/vllm-project/vllm/pull/34651
- 状态/时间: merged / 2026-02-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`；关联提交 `54e2f83d0a82`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+68/-48，可读 patch 399 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Lazy import for the "mistral" tokenizer module.」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/utils/mistral.py`；技术摘要: 覆盖「[Feature] Lazy import for the "mistral" tokenizer module.」；主要实现面是 `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/utils/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6 (10 lines); hunks: -25,10 +25,10; -66,9 +66,7 @@ def is_valid_id(id: str) -> bool:; symbols: is_valid_id, _is_pre_v11_tokeniser, MistralToolParser, __init__，涉及 `is_valid_id, _is_pre_v11_tokeniser, MistralToolParser`；`vllm/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -210,6 +210,8 @@ def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | by...; symbols: _tekken_token_to_id, MistralTokenizer, from_pretrained，涉及 `_tekken_token_to_id, MistralTokenizer, from_pretrained`；`vllm/utils/mistral.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: is_mistral_tokenizer, attribute，涉及 `is_mistral_tokenizer, attribute`。
- 代码 diff 细节:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6 (10 lines); hunks: -25,10 +25,10; -66,9 +66,7 @@ def is_valid_id(id: str) -> bool:; symbols: is_valid_id, _is_pre_v11_tokeniser, MistralToolParser, __init__
  - `vllm/tokenizers/mistral.py` modified +2/-0 (2 lines); hunks: -210,6 +210,8 @@ def _tekken_token_to_id(tokenizer: "Tekkenizer", t: str | by...; symbols: _tekken_token_to_id, MistralTokenizer, from_pretrained
  - `vllm/utils/mistral.py` added +28/-0 (28 lines); hunks: -0,0 +1,28; symbols: is_mistral_tokenizer, attribute
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +4/-6; `vllm/tokenizers/mistral.py` modified +2/-0; `vllm/utils/mistral.py` added +28/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/reasoning/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36156 - [Bugfix] Fix simple Mistral-Small example

- 链接: https://github.com/vllm-project/vllm/pull/36156
- 状态/时间: merged / 2026-03-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-2，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix simple Mistral-Small example」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `examples/offline_inference/mistral-small.py`；技术摘要: 覆盖「[Bugfix] Fix simple Mistral-Small example」；主要实现面是 `examples/offline_inference/mistral-small.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/offline_inference/mistral-small.py` modified +5/-2 (7 lines); hunks: -7,6 +7,7; -79,8 +80,10 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo，涉及 `run_simple_demo`。
- 代码 diff 细节:
  - `examples/offline_inference/mistral-small.py` modified +5/-2 (7 lines); hunks: -7,6 +7,7; -79,8 +80,10 @@ def run_simple_demo(args: argparse.Namespace):; symbols: run_simple_demo
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/offline_inference/mistral-small.py` modified +5/-2
- 验证与风险: 该 PR 主要落在文档/示例 `examples/offline_inference/mistral-small.py`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36782 - [Bugfix] Fix Mistral-small `--format`

- 链接: https://github.com/vllm-project/vllm/pull/36782
- 状态/时间: merged / 2026-03-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-6，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Mistral-small `--format`」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `examples/offline_inference/mistral-small.py`；技术摘要: 覆盖「[Bugfix] Fix Mistral-small `--format`」；主要实现面是 `examples/offline_inference/mistral-small.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/offline_inference/mistral-small.py` modified +6/-6 (12 lines); hunks: -62,9 +62,9 @@ def run_simple_demo(args: argparse.Namespace):; -102,9 +102,9 @@ def run_advanced_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo，涉及 `run_simple_demo, run_advanced_demo`。
- 代码 diff 细节:
  - `examples/offline_inference/mistral-small.py` modified +6/-6 (12 lines); hunks: -62,9 +62,9 @@ def run_simple_demo(args: argparse.Namespace):; -102,9 +102,9 @@ def run_advanced_demo(args: argparse.Namespace):; symbols: run_simple_demo, run_advanced_demo
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/offline_inference/mistral-small.py` modified +6/-6
- 验证与风险: 该 PR 主要落在文档/示例 `examples/offline_inference/mistral-small.py`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36163 - Add support to Mistral large 3 eagle with dense layers

- 链接: https://github.com/vllm-project/vllm/pull/36163
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/transformers_utils/configs/mistral.py`；关联提交 `afebeffbfbf2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+28/-1，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support to Mistral large 3 eagle with dense layers」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`；技术摘要: 覆盖「Add support to Mistral large 3 eagle with dense layers」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`, `vllm/model_executor/models/mistral_large_3_eagle.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +23/-0 (23 lines); hunks: -19,6 +19,10 @@ def adapt_config_dict(; -291,3 +295,22 @@ def _remap_moe_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args，涉及 `adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args`；`vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1 (6 lines); hunks: -1,6 +1,7; -33,7 +34,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +23/-0 (23 lines); hunks: -19,6 +19,10 @@ def adapt_config_dict(; -291,3 +295,22 @@ def _remap_moe_args(config: dict) -> dict:; symbols: adapt_config_dict, _remap_moe_args, _remap_mistral_mla_args
  - `vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1 (6 lines); hunks: -1,6 +1,7; -33,7 +34,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +23/-0; `vllm/model_executor/models/mistral_large_3_eagle.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral_large_3_eagle.py`, `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36971 - Mistral common v10

- 链接: https://github.com/vllm-project/vllm/pull/36971
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/mistral.py`；关联提交 `e42b49bd69d4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+22/-3，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Mistral common v10」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/tokenizers/mistral.py`；技术摘要: 覆盖「Mistral common v10」；主要实现面是 `vllm/tokenizers/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/mistral.py` modified +19/-0 (19 lines); hunks: -7,6 +7,9; -192,6 +195,15 @@ def validate_request_params(request: "ChatCompletionRequest"):; symbols: validate_request_params, _tekken_token_to_id, apply_chat_template, decode，涉及 `validate_request_params, _tekken_token_to_id, apply_chat_template`。
- 代码 diff 细节:
  - `vllm/tokenizers/mistral.py` modified +19/-0 (19 lines); hunks: -7,6 +7,9; -192,6 +195,15 @@ def validate_request_params(request: "ChatCompletionRequest"):; symbols: validate_request_params, _tekken_token_to_id, apply_chat_template, decode
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tokenizers/mistral.py` modified +19/-0
- 验证与风险: runtime 路径改动集中在 `vllm/tokenizers/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37104 - Patch Mistral config

- 链接: https://github.com/vllm-project/vllm/pull/37104
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `ffbc2e5bdbfb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+49/-30，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Patch Mistral config」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Patch Mistral config」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +10/-7 (17 lines); hunks: -113,12 +113,13 @@ def _remap_mistral_vision_args(config: dict) -> dict:; -128,9 +129,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args，涉及 `_remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +10/-7 (17 lines); hunks: -113,12 +113,13 @@ def _remap_mistral_vision_args(config: dict) -> dict:; -128,9 +129,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +10/-7
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/mistral.py`, `vllm/transformers_utils/model_arch_config_convertor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37232 - Fix EagleMistralLarge3Model initialization

- 链接: https://github.com/vllm-project/vllm/pull/37232
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral_large_3_eagle.py`；关联提交 `7961486a9b74`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix EagleMistralLarge3Model initialization」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral_large_3_eagle.py`；技术摘要: 覆盖「Fix EagleMistralLarge3Model initialization」；主要实现面是 `vllm/model_executor/models/mistral_large_3_eagle.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0 (1 lines); hunks: -74,6 +74,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0 (1 lines); hunks: -74,6 +74,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mistral_large_3_eagle.py
@@ -74,6 +74,7 @@ def __init__(
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral_large_3_eagle.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral_large_3_eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37209 - Fix some Mistral parser issues

- 链接: https://github.com/vllm-project/vllm/pull/37209
- 状态/时间: merged / 2026-03-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `5db91f0aaf35`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+42/-34，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix some Mistral parser issues」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`；技术摘要: 覆盖「Fix some Mistral parser issues」；主要实现面是 `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/mistral.py` modified +27/-26 (53 lines); hunks: -15,8 +15,15; -26,21 +33,20; symbols: from_pretrained, __init__, convert_tokens_to_ids，涉及 `from_pretrained, __init__, convert_tokens_to_ids`；`vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3 (10 lines); hunks: -241,7 +241,10 @@ def extract_tool_calls_streaming(; -275,7 +278,8 @@ def _extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer，涉及 `extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer`。
- 代码 diff 细节:
  - `vllm/tokenizers/mistral.py` modified +27/-26 (53 lines); hunks: -15,8 +15,15; -26,21 +33,20; symbols: from_pretrained, __init__, convert_tokens_to_ids
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3 (10 lines); hunks: -241,7 +241,10 @@ def extract_tool_calls_streaming(; -275,7 +278,8 @@ def _extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming, _extract_tool_calls_streaming, _extract_tool_calls_streaming_pre_v11_tokenizer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tokenizers/mistral.py` modified +27/-26; `vllm/tool_parsers/mistral_tool_parser.py` modified +7/-3
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36928 - [LoRA][BugFix] Fix skipped LoRA adapters for Mistral3

- 链接: https://github.com/vllm-project/vllm/pull/36928
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral3.py`；关联提交 `5bc1da147fb0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA][BugFix] Fix skipped LoRA adapters for Mistral3」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mistral3.py`；技术摘要: 覆盖「[LoRA][BugFix] Fix skipped LoRA adapters for Mistral3」；主要实现面是 `vllm/model_executor/models/mistral3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(; symbols: Mistral3ForConditionalGeneration，涉及 `Mistral3ForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral3.py` modified +3/-0 (3 lines); hunks: -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(; symbols: Mistral3ForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mistral3.py
@@ -429,6 +429,9 @@ class Mistral3ForConditionalGeneration(
+            # Some PEFT LoRAs are trained against the text submodule directly
+            # and produce names like `base_model.model.model.layers.*`.
+            "model.": "language_model.model.",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral3.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mistral3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38150 - [Mistral Grammar] Support Grammar Factory

- 链接: https://github.com/vllm-project/vllm/pull/38150
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tokenizers_/test_mistral.py`, `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `fef56c18555e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+601/-29，可读 patch 816 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mistral Grammar] Support Grammar Factory」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tokenizers_/test_mistral.py`；技术摘要: 覆盖「[Mistral Grammar] Support Grammar Factory」；主要实现面是 `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `tests/tokenizers_/test_mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3 (347 lines); hunks: -3,19 +3,43; -40,6 +64,13 @@ def mistral_tool_parser(mistral_tokenizer):; symbols: mistral_tool_parser, non_mistral_parser, assert_tool_calls, test_fast_detokenization_text_detection_pre_v11，涉及 `mistral_tool_parser, non_mistral_parser, assert_tool_calls`；`vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9 (142 lines); hunks: -10,6 +10,18; -25,6 +37,7; symbols: StreamingState, MistralToolParser, __init__, adjust_request，涉及 `StreamingState, MistralToolParser, __init__`；`tests/tokenizers_/test_mistral.py` modified +28/-0 (28 lines); hunks: -3,8 +3,10; -2407,3 +2409,29 @@ def test_convert_ids_to_tokens(; symbols: test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer，涉及 `test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer`；`vllm/tokenizers/mistral.py` modified +25/-0 (25 lines); hunks: -1,9 +1,12; -45,6 +48,7; symbols: convert_ids_to_tokens, supports_grammar, grammar_factory, llg_tokenizer，涉及 `convert_ids_to_tokens, supports_grammar, grammar_factory`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3 (347 lines); hunks: -3,19 +3,43; -40,6 +64,13 @@ def mistral_tool_parser(mistral_tokenizer):; symbols: mistral_tool_parser, non_mistral_parser, assert_tool_calls, test_fast_detokenization_text_detection_pre_v11
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9 (142 lines); hunks: -10,6 +10,18; -25,6 +37,7; symbols: StreamingState, MistralToolParser, __init__, adjust_request
  - `tests/tokenizers_/test_mistral.py` modified +28/-0 (28 lines); hunks: -3,8 +3,10; -2407,3 +2409,29 @@ def test_convert_ids_to_tokens(; symbols: test_convert_ids_to_tokens, test_grammar_factory, test_llg_tokenizer
  - `vllm/tokenizers/mistral.py` modified +25/-0 (25 lines); hunks: -1,9 +1,12; -45,6 +48,7; symbols: convert_ids_to_tokens, supports_grammar, grammar_factory, llg_tokenizer
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +344/-3; `tests/tokenizers_/test_mistral.py` modified +28/-0
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +133/-9; `vllm/tokenizers/mistral.py` modified +25/-0
- 验证与风险: diff 自带测试面 `tests/tokenizers_/test_mistral.py`, `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/v1/structured_output/test_backend_guidance.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37292 - Fix Mistral yarn warning in Transformers v5

- 链接: https://github.com/vllm-project/vllm/pull/37292
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/configs/mistral.py`；关联提交 `edcc37a8cee2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Mistral yarn warning in Transformers v5」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/mistral.py`；技术摘要: 覆盖「Fix Mistral yarn warning in Transformers v5」；主要实现面是 `vllm/transformers_utils/configs/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/mistral.py` modified +6/-0 (6 lines); hunks: -2,7 +2,9; -134,6 +136,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args，涉及 `_remap_mistral_yarn_args`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/mistral.py` modified +6/-0 (6 lines); hunks: -2,7 +2,9; -134,6 +136,10 @@ def _remap_mistral_yarn_args(config: dict) -> dict:; symbols: _remap_mistral_yarn_args
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/mistral.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/configs/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39217 - [Mistral Grammar] Fix tool and reasoning parsing

- 链接: https://github.com/vllm-project/vllm/pull/39217
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/tool_use/mistral/test_mistral_tool_calls.py`, `tests/tool_use/mistral/utils.py`, `vllm/tokenizers/mistral.py`, `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `c0722f22de71`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1601/-266，可读 patch 2396 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Mistral Grammar] Fix tool and reasoning parsing」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`；技术摘要: 覆盖「[Mistral Grammar] Fix tool and reasoning parsing」；主要实现面是 `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/tokenizers/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180 (932 lines); hunks: -3,6 +3,7; -23,24 +24,33; symbols: mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools，涉及 `mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools`；`vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10 (188 lines); hunks: -1,12 +1,15; -37,14 +40,19; symbols: _is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult, adjust_request，涉及 `_is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult`；`vllm/tokenizers/mistral.py` modified +49/-38 (87 lines); hunks: -54,6 +54,50; -159,44 +203,11 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls, _prepare_apply_chat_template_tools_and_messages，涉及 `_pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls`；`tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3 (483 lines); hunks: -1,25 +1,198; -28,3 +201,307 @@ async def test_tool_call_with_tool_choice(client: openai.As...; symbols: _requires_tool_parser, _is_pre_v11, StreamedToolCallResult, _collect_streamed_tool_call，涉及 `_requires_tool_parser, _is_pre_v11, StreamedToolCallResult`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180 (932 lines); hunks: -3,6 +3,7; -23,24 +24,33; symbols: mistral_pre_v11_tokenizer, stream_delta_message_generator, test_extract_tool_calls_no_tools
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10 (188 lines); hunks: -1,12 +1,15; -37,14 +40,19; symbols: _is_pre_v11_tokeniser, MistralToolParser, MistralStreamingResult, adjust_request
  - `vllm/tokenizers/mistral.py` modified +49/-38 (87 lines); hunks: -54,6 +54,50; -159,44 +203,11 @@ def _prepare_apply_chat_template_tools_and_messages(; symbols: _pop_unallowed_keys_and_warn, adapt_inplace_to_mistral_tool, maybe_serialize_tool_calls, _prepare_apply_chat_template_tools_and_messages
  - `tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3 (483 lines); hunks: -1,25 +1,198; -28,3 +201,307 @@ async def test_tool_call_with_tool_choice(client: openai.As...; symbols: _requires_tool_parser, _is_pre_v11, StreamedToolCallResult, _collect_streamed_tool_call
  - `tests/tool_use/mistral/utils.py` modified +24/-10 (34 lines); hunks: -2,16 +2,7; -21,6 +12,11 @@ class ServerConfig(TypedDict, total=False):; symbols: ServerConfig
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +752/-180; `tests/tool_use/mistral/test_mistral_tool_calls.py` modified +480/-3; `tests/tool_use/mistral/utils.py` modified +24/-10
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +178/-10; `vllm/tokenizers/mistral.py` modified +49/-38
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_mistral_tool_parser.py`, `tests/tool_use/mistral/test_mistral_tool_calls.py`, `tests/tool_use/mistral/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40531 - [Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output

- 链接: https://github.com/vllm-project/vllm/pull/40531
- 状态/时间: merged / 2026-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `cfa49213d778`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+66/-18，可读 patch 160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`；技术摘要: 覆盖「[Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output」；主要实现面是 `tests/tool_parsers/test_mistral_tool_parser.py`, `vllm/tool_parsers/mistral_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9 (58 lines); hunks: -24,7 +24,6; -250,6 +249,7 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):; symbols: test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises, test_extract_tool_calls_pre_v11_regex_fallback_raises，涉及 `test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises`；`vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9 (26 lines); hunks: -479,21 +479,28 @@ def extract_tool_calls(; -504,7 +511,8 @@ def extract_tool_calls(; symbols: extract_tool_calls，涉及 `extract_tool_calls`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9 (58 lines); hunks: -24,7 +24,6; -250,6 +249,7 @@ def test_extract_tool_calls_no_tools(parser_fixture, request):; symbols: test_extract_tool_calls_no_tools, test_extract_tool_calls_pre_v11_tokenizer, test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises, test_extract_tool_calls_pre_v11_regex_fallback_raises
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9 (26 lines); hunks: -479,21 +479,28 @@ def extract_tool_calls(; -504,7 +511,8 @@ def extract_tool_calls(; symbols: extract_tool_calls
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_mistral_tool_parser.py` modified +49/-9
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +17/-9
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_mistral_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40043 - [Feature] Avoid eager import of the "mistral_common" package.

- 链接: https://github.com/vllm-project/vllm/pull/40043
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`；关联提交 `56bdf85e10b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+47/-23，可读 patch 194 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Avoid eager import of the "mistral_common" package.」；模型线: Mistral Small 4；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`；技术摘要: 覆盖「[Feature] Avoid eager import of the "mistral_common" package.」；主要实现面是 `vllm/tool_parsers/mistral_tool_parser.py`, `vllm/utils/mistral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0 (2 lines); hunks: -118,6 +118,8 @@ class MistralToolParser(ToolParser):; symbols: MistralToolParser，涉及 `MistralToolParser`；`vllm/utils/mistral.py` modified +15/-0 (15 lines); hunks: -12,8 +12,10; -26,3 +28,16 @@ def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGu...; symbols: is_mistral_tokenizer, is_mistral_tool_parser, attribute，涉及 `is_mistral_tokenizer, is_mistral_tool_parser, attribute`。
- 代码 diff 细节:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0 (2 lines); hunks: -118,6 +118,8 @@ class MistralToolParser(ToolParser):; symbols: MistralToolParser
  - `vllm/utils/mistral.py` modified +15/-0 (15 lines); hunks: -12,8 +12,10; -26,3 +28,16 @@ def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGu...; symbols: is_mistral_tokenizer, is_mistral_tool_parser, attribute
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +2/-0; `vllm/utils/mistral.py` modified +15/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/serve/render/serving.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39294 - [Bugfix][Parser] Fix Mistral tool parser for HF tokenizers

- 链接: https://github.com/vllm-project/vllm/pull/39294
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/mistral_tool_parser.py`；关联提交 `2ec18f5df43e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-4，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Parser] Fix Mistral tool parser for HF tokenizers」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/mistral_tool_parser.py`；技术摘要: 覆盖「[Bugfix][Parser] Fix Mistral tool parser for HF tokenizers」；主要实现面是 `vllm/tool_parsers/mistral_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4 (17 lines); hunks: -91,7 +91,12 @@ def is_valid_id(id: str) -> bool:; -137,15 +142,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: is_valid_id, _is_pre_v11_tokeniser, __init__, extract_tool_calls，涉及 `is_valid_id, _is_pre_v11_tokeniser, __init__`。
- 代码 diff 细节:
  - `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4 (17 lines); hunks: -91,7 +91,12 @@ def is_valid_id(id: str) -> bool:; -137,15 +142,15 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[T...; symbols: is_valid_id, _is_pre_v11_tokeniser, __init__, extract_tool_calls
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/mistral_tool_parser.py` modified +13/-4
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/mistral_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41024 - [FEATURE] Add EagleMistralForCausalLM

- 链接: https://github.com/vllm-project/vllm/pull/41024
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mistral_eagle.py`；关联提交 `e9f8f31e9a4c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+172/-0，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FEATURE] Add EagleMistralForCausalLM」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mistral_eagle.py`；技术摘要: 覆盖「[FEATURE] Add EagleMistralForCausalLM」；主要实现面是 `vllm/model_executor/models/mistral_eagle.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mistral_eagle.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: EagleMistralDecoderLayer, __init__, get_quant_config, EagleMistralModel，涉及 `EagleMistralDecoderLayer, __init__, get_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mistral_eagle.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: EagleMistralDecoderLayer, __init__, get_quant_config, EagleMistralModel
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/mistral_eagle.py` added +166/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
