# vllm Gemma 4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_gemma4.jinja` | [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/kernels/moe/test_gemma4router.py` | [#39083](https://github.com/vllm-project/vllm/pull/39083) |
| `tests/models/multimodal/processing/test_gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `tests/reasoning/test_gemma4_reasoning_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/renderers/test_gemma4_chat_template.py` | [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/tool_parsers/test_gemma4_tool_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38909](https://github.com/vllm-project/vllm/pull/38909), [#38992](https://github.com/vllm-project/vllm/pull/38992), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39114](https://github.com/vllm-project/vllm/pull/39114), [#39679](https://github.com/vllm-project/vllm/pull/39679) |
| `tests/tool_use/test_gemma4_responses_adjust_request.py` | 无直接 PR 号提交 |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/model_executor/models/gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38844](https://github.com/vllm-project/vllm/pull/38844), [#38879](https://github.com/vllm-project/vllm/pull/38879), [#39045](https://github.com/vllm-project/vllm/pull/39045), [#39083](https://github.com/vllm-project/vllm/pull/39083), [#39450](https://github.com/vllm-project/vllm/pull/39450) |
| `vllm/model_executor/models/gemma4_mm.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38872](https://github.com/vllm-project/vllm/pull/38872), [#39234](https://github.com/vllm-project/vllm/pull/39234), [#39291](https://github.com/vllm-project/vllm/pull/39291), [#39450](https://github.com/vllm-project/vllm/pull/39450), [#39842](https://github.com/vllm-project/vllm/pull/39842), [#40411](https://github.com/vllm-project/vllm/pull/40411), [#40534](https://github.com/vllm-project/vllm/pull/40534) |
| `vllm/reasoning/gemma4_reasoning_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `vllm/reasoning/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/tool_parsers/gemma4_tool_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38847](https://github.com/vllm-project/vllm/pull/38847), [#38909](https://github.com/vllm-project/vllm/pull/38909), [#38992](https://github.com/vllm-project/vllm/pull/38992), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39114](https://github.com/vllm-project/vllm/pull/39114), [#39679](https://github.com/vllm-project/vllm/pull/39679) |
| `vllm/tool_parsers/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |

## PR 覆盖总览

- git 追溯 PR 数: 18
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 18
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-02 | [#38826](https://github.com/vllm-project/vllm/pull/38826) | merged | feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use) | `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-02 | [#38847](https://github.com/vllm-project/vllm/pull/38847) | merged | [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter | `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-03 | [#38872](https://github.com/vllm-project/vllm/pull/38872) | merged | [Misc] Clean up Gemma4 implementation | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-05 | [#38992](https://github.com/vllm-project/vllm/pull/38992) | merged | [Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters | `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-06 | [#38879](https://github.com/vllm-project/vllm/pull/38879) | merged | [Gemma4] Enable Fast Prefill Optimization | `vllm/model_executor/models/gemma4.py` |
| 2026-04-08 | [#38909](https://github.com/vllm-project/vllm/pull/38909) | merged | [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls | `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-08 | [#39114](https://github.com/vllm-project/vllm/pull/39114) | merged | [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values | `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-08 | [#39027](https://github.com/vllm-project/vllm/pull/39027) | merged | [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes | `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py` |
| 2026-04-09 | [#39045](https://github.com/vllm-project/vllm/pull/39045) | merged | [Gemma4] Support quantized MoE | `vllm/model_executor/models/gemma4.py` |
| 2026-04-10 | [#39450](https://github.com/vllm-project/vllm/pull/39450) | merged | Add Gemma4 Eagle3 support | `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-11 | [#38844](https://github.com/vllm-project/vllm/pull/38844) | merged | [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly | `vllm/model_executor/models/gemma4.py` |
| 2026-04-14 | [#39679](https://github.com/vllm-project/vllm/pull/39679) | merged | [Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"` | `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-15 | [#39842](https://github.com/vllm-project/vllm/pull/39842) | merged | [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-17 | [#39234](https://github.com/vllm-project/vllm/pull/39234) | merged | [Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids` | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-17 | [#39291](https://github.com/vllm-project/vllm/pull/39291) | merged | feat: Add LoRA support for Gemma4ForConditionalGeneration | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-19 | [#39083](https://github.com/vllm-project/vllm/pull/39083) | merged | [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton | `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py` |
| 2026-04-21 | [#40411](https://github.com/vllm-project/vllm/pull/40411) | merged | [Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference | `vllm/model_executor/models/gemma4_mm.py` |
| 2026-04-24 | [#40534](https://github.com/vllm-project/vllm/pull/40534) | merged | [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard | `vllm/model_executor/models/gemma4_mm.py` |

## 逐 PR diff 审计卡

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- 链接: https://github.com/vllm-project/vllm/pull/38826
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_gemma4.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`, `vllm/model_executor/models/gemma4.py` 等 10 个文件；关联提交 `08ed2b9688b4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+5051/-1，可读 patch 5167 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: PR Title `feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)` PR Description **Sumary** This PR introduces comprehensive support...。
- 实现要点: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo，涉及 `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs`；`vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward，涉及 `_get_text_config, Gemma4MLP, __init__`；`vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser，涉及 `_parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array`；`tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args，涉及 `mock_tokenizer, parser, mock_request`。
- 代码 diff 细节:
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward
  - `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -0,0 +1,724 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0; `vllm/model_executor/models/gemma4.py` added +1239/-0; `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0; `vllm/model_executor/models/gemma4_utils.py` added +292/-0; `vllm/reasoning/gemma4_reasoning_parser.py` added +193/-0; `vllm/tool_parsers/gemma4_utils.py` added +183/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0; `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/registry.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38847 - [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter

- 链接: https://github.com/vllm-project/vllm/pull/38847
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/gemma4_tool_parser.py`；关联提交 `bb39382b2b28`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: Fix `Gemma4ToolParser.__init__()` to accept the `tools` parameter, matching the base `ToolParser` interface. Without this fix, enabling tool calling with `--tool-call-parser gem...。
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
- 动机: 标题「[Misc] Clean up Gemma4 implementation」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文未提供可用摘要。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；关联提交 `f53fa26e05c4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+33/-3，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: Issue #38946 Fix Gemma 4 streaming tool calls producing invalid JSON due to partial delimiter chars not being stripped cc @sallyom。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json，涉及 `test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json`；`vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3 (7 lines); hunks: -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaM...; symbols: _emit_argument_diff，涉及 `_emit_argument_diff`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3 (7 lines); hunks: -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaM...; symbols: _emit_argument_diff
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaMessage | None:
-        # tokens arrive. Strip trailing '}', '"', and ']' sequences
-        # to get the "safe prefix".
+        # tokens arrive. Strip trailing '}', '"', ']' and partial
+        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
+        # "safe prefix".
-        while safe_json and safe_json[-1] in ("}", '"', "]"):
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38879 - [Gemma4] Enable Fast Prefill Optimization

- 链接: https://github.com/vllm-project/vllm/pull/38879
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `47e605092b7f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+369/-47，可读 patch 490 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Enable Fast Prefill Optimization」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/gemma4.py`；PR 正文摘要: Add `--kv-sharing-fast-prefill` support for Gemma 4 models, porting the YOCO (You Only Cache Once) fast prefill optimization from Gemma3n. When enabled, the cross-decoder layers...。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；关联提交 `d734445fcd79`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+64/-2，可读 patch 77 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: Fix a Gemma4 streaming bug where buffered deltas are stitched back into `current_text`, which can corrupt normal text after or inside tool calls. This showed up most clearly wit...。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes，涉及 `test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming`；`vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -436,8 +436,10 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming，涉及 `extract_tool_calls_streaming`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -436,8 +436,10 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -436,8 +436,10 @@ def extract_tool_calls_streaming(
-        # Reconstruct current_text after buffering to stay in sync
-        current_text = previous_text + delta_text
+        # Keep current_text from the upstream stream state. The buffered delta
+        # is only for emission, and must not be stitched back into the
+        # accumulated model text or normal content like "<div>" can be
+        # duplicated into "<<div>" when a tool call just ended.
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39114 - [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values

- 链接: https://github.com/vllm-project/vllm/pull/39114
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；关联提交 `13151a4df43d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+78/-8，可读 patch 159 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: Fix https://github.com/vllm-project/vllm/issues/39089 Root cause: _parse_gemma4_value() misidentified partial literals as bare strings, causing a type change (string → boolean)...。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks，涉及 `test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks`；`vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8 (41 lines); hunks: -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:; -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:; symbols: _parse_gemma4_value, _parse_gemma4_args，涉及 `_parse_gemma4_value, _parse_gemma4_args`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8 (41 lines); hunks: -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:; -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:; symbols: _parse_gemma4_value, _parse_gemma4_args
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:
-def _parse_gemma4_args(args_str: str) -> dict:
+def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
@@ -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:
+    Args:
+        args_str: The raw Gemma4 argument string.
+        partial: When True (streaming), bare values at end of string are
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39027 - [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes

- 链接: https://github.com/vllm-project/vllm/pull/39027
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py` 等 6 个文件；关联提交 `8477fe427d17`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+878/-16，可读 patch 1083 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py`；PR 正文摘要: Fix multiple issues preventing Gemma4 models from working correctly with multi-turn tool calling and reasoning in vLLM: - Add new Gemma4 chat template that properly encodes tool...。
- 实现要点: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode，涉及 `generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output`；`tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array，涉及 `test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld`；`vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3 (38 lines); hunks: -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -63,6 +73,29 @@ def end_token(self) -> str:; symbols: __init__, adjust_request, start_token, end_token，涉及 `__init__, adjust_request, start_token`；`vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -122,14 +122,16 @@ def _parse_gemma4_args(args_str: str, *, partial: bool = F...; symbols: _parse_gemma4_args，涉及 `_parse_gemma4_args`。
- 代码 diff 细节:
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array
  - `vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3 (38 lines); hunks: -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -63,6 +73,29 @@ def end_token(self) -> str:; symbols: __init__, adjust_request, start_token, end_token
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -122,14 +122,16 @@ def _parse_gemma4_args(args_str: str, *, partial: bool = F...; symbols: _parse_gemma4_args
  - `tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled
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
diff -- vllm/reasoning/gemma4_reasoning_parser.py
@@ -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
```

- 已读文件:
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0; `tests/renderers/test_gemma4_chat_template.py` added +345/-0
  - runtime: `vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2
  - docs: `examples/tool_chat_template_gemma4.jinja` added +331/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39045 - [Gemma4] Support quantized MoE

- 链接: https://github.com/vllm-project/vllm/pull/39045
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4.py`；关联提交 `3aecdf08b4a8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-14，可读 patch 89 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Gemma4] Support quantized MoE」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/gemma4.py`；PR 正文摘要: - Expand gemma4 MoE weight loading to include logic for 2D quantized layers and parameters Testing - Existing MoE checkpoints(without quantization) load without issue - Quantize...。
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
- 动机: 标题「Add Gemma4 Eagle3 support」；模型线: Gemma 4；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: Enables Eagle3 style speculative decoding on Gemma4 models. Test model: RedHatAI/gemma-4-31B-it-speculator.eagle3 Served locally and verified it ran and produced reasonable acce...。
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
- 动机: 标题「[Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4.py`；PR 正文摘要: `Gemma4ForConditionalGeneration` names the text stack under `model.language_model.*`, while the text-only `Gemma4ForCausalLM` path exposes the same layers under `model.*`. That...。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；关联提交 `b075604da10a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；PR 正文摘要: When serving a Gemma4 model with `--enable-auto-tool-choice --tool-call-parser gemma4`, a tool parameter that accepts `null` (e.g. `{"type": "string", "nullable": true}`) is mis...。
- 实现要点: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types，涉及 `test_boolean_false, test_null_value, test_mixed_types`；`vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0 (4 lines); hunks: -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:; symbols: _parse_gemma4_value，涉及 `_parse_gemma4_value`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0 (4 lines); hunks: -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:; symbols: _parse_gemma4_value
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:
+    # Null
+    if value_str.lower() in ("null", "none", "nil"):
+        return None
```

- 已读文件:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_gemma4_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39842 - [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models

- 链接: https://github.com/vllm-project/vllm/pull/39842
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/gemma4_mm.py`；关联提交 `6dc949140693`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: This PR fixes the token repetition issue (e.g., "is is is is...") observed in Gemma 4 Pre-Trained (PT) models when running in completion mode (without a chat template). The issu...。
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
- 动机: 标题「[Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids`」；模型线: Gemma 4；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: This removes blocking GPU/CPU sync in Gemma4 `embed_input_ids` following up from #34246 **Before:** **After:**。
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
- 动机: 标题「feat: Add LoRA support for Gemma4ForConditionalGeneration」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: fix https://github.com/vllm-project/vllm/issues/39246 Enable LoRA for Gemma4ForConditionalGeneration > Implement: > - get_num_mm_connector_tokens > - get_num_mm_encoder_tokens >...。
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
- 动机: 标题「[FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton」；模型线: Gemma 4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/gemma4.py`, `tests/kernels/moe/test_gemma4router.py`；PR 正文摘要: Improve the performance of Gemma4 by introducing triton fused routing function. The custom routing function introduces many synchronizations point and read write to global memor...。
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
- 动机: 标题「[Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference」；模型线: Gemma 4；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: Fix the norm ordering in `Gemma4MultimodalEmbedder` to match the HF transformers reference implementation. The vLLM implementation had **post-projection** RMSNorm while the HF r...。
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
- 动机: 标题「[Model] Gemma4: add bidirectional vision attention for sliding layers with window guard」；模型线: Gemma 4；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/gemma4_mm.py`；PR 正文摘要: Implement `use_bidirectional_attention="vision"` support for Gemma4 models (gemma-4-31B-it, gemma-4-26B-A4B-it), addressing #40106. Gemma4's architecture applies bidirectional a...。
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

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
