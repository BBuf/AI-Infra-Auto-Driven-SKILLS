# vllm Gemma 4 PR Diff Audit Reference

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 0
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_gemma4.jinja` | [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/kernels/moe/test_gemma4router.py` | [#39083](https://github.com/vllm-project/vllm/pull/39083) |
| `tests/models/multimodal/processing/test_gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `tests/reasoning/test_gemma4_reasoning_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/renderers/test_gemma4_chat_template.py` | [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `tests/tool_parsers/test_gemma4_tool_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38909](https://github.com/vllm-project/vllm/pull/38909), [#38992](https://github.com/vllm-project/vllm/pull/38992), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39114](https://github.com/vllm-project/vllm/pull/39114), [#39679](https://github.com/vllm-project/vllm/pull/39679) |
| `tests/tool_use/test_gemma4_responses_adjust_request.py` | no direct PR-number commit |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/model_executor/models/gemma4.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38844](https://github.com/vllm-project/vllm/pull/38844), [#38879](https://github.com/vllm-project/vllm/pull/38879), [#39045](https://github.com/vllm-project/vllm/pull/39045), [#39083](https://github.com/vllm-project/vllm/pull/39083), [#39450](https://github.com/vllm-project/vllm/pull/39450), [#40786](https://github.com/vllm-project/vllm/pull/40786), [#41206](https://github.com/vllm-project/vllm/pull/41206) |
| `vllm/model_executor/models/gemma4_mm.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38872](https://github.com/vllm-project/vllm/pull/38872), [#39234](https://github.com/vllm-project/vllm/pull/39234), [#39291](https://github.com/vllm-project/vllm/pull/39291), [#39450](https://github.com/vllm-project/vllm/pull/39450), [#39842](https://github.com/vllm-project/vllm/pull/39842), [#40411](https://github.com/vllm-project/vllm/pull/40411), [#40534](https://github.com/vllm-project/vllm/pull/40534) |
| `vllm/reasoning/gemma4_reasoning_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#39027](https://github.com/vllm-project/vllm/pull/39027) |
| `vllm/reasoning/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |
| `vllm/tool_parsers/gemma4_tool_parser.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826), [#38847](https://github.com/vllm-project/vllm/pull/38847), [#38909](https://github.com/vllm-project/vllm/pull/38909), [#38992](https://github.com/vllm-project/vllm/pull/38992), [#39027](https://github.com/vllm-project/vllm/pull/39027), [#39114](https://github.com/vllm-project/vllm/pull/39114), [#39679](https://github.com/vllm-project/vllm/pull/39679) |
| `vllm/tool_parsers/gemma4_utils.py` | [#38826](https://github.com/vllm-project/vllm/pull/38826) |

## Timeline

| Date | PR | State | Title | Main files |
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
| 2026-04-29 | [#40786](https://github.com/vllm-project/vllm/pull/40786) | merged | Fix PP in Gemma4 | `vllm/model_executor/models/gemma4.py` |
| 2026-04-30 | [#41206](https://github.com/vllm-project/vllm/pull/41206) | merged | Fix Gemma4 MoE expert weight remapping | `vllm/model_executor/models/gemma4.py` |

## Per-PR Diff Audit Cards

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- Link: https://github.com/vllm-project/vllm/pull/38826
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_gemma4.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`, `vllm/model_executor/models/gemma4.py` and 10 files; associated commits `08ed2b9688b4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +5051/-1, 5167 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)"; model line: Gemma 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo, touching `Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs`; `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward, touching `_get_text_config, Gemma4MLP, __init__`; `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser, touching `_parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array`; `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args, touching `mock_tokenizer, parser, mock_request`.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: -0,0 +1,1341; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: -0,0 +1,1239; symbols: _get_text_config, Gemma4MLP, __init__, forward
  - `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -0,0 +1,724 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/gemma4_mm.py` added +1341/-0; `vllm/model_executor/models/gemma4.py` added +1239/-0; `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0; `vllm/model_executor/models/gemma4_utils.py` added +292/-0; `vllm/reasoning/gemma4_reasoning_parser.py` added +193/-0; `vllm/tool_parsers/gemma4_utils.py` added +183/-0
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0; `tests/reasoning/test_gemma4_reasoning_parser.py` added +196/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_common.py`, `tests/models/multimodal/processing/test_gemma4.py`, `tests/models/registry.py`, `tests/reasoning/test_gemma4_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38847 - [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter

- Link: https://github.com/vllm-project/vllm/pull/38847
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/gemma4_tool_parser.py`; associated commits `bb39382b2b28`; preserved from an explicit existing history/skill citation
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; associated commits `f53fa26e05c4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +33/-3, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json, touching `test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json`; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3 (7 lines); hunks: -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaM...; symbols: _emit_argument_diff, touching `_emit_argument_diff`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0 (29 lines); hunks: -502,3 +502,32 @@ def test_streaming_empty_args(self, parser, mock_request):; symbols: test_streaming_empty_args, test_streaming_split_delimiter_no_invalid_json
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3 (7 lines); hunks: -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaM...; symbols: _emit_argument_diff
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -675,10 +675,11 @@ def _emit_argument_diff(self, raw_args_str: str) -> DeltaMessage | None:
-        # tokens arrive. Strip trailing '}', '"', and ']' sequences
-        # to get the "safe prefix".
+        # tokens arrive. Strip trailing '}', '"', ']' and partial
+        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
+        # "safe prefix".
-        while safe_json and safe_json[-1] in ("}", '"', "]"):
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +29/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-3
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; associated commits `d734445fcd79`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +64/-2, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "[Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes, touching `test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming`; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -436,8 +436,10 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming, touching `extract_tool_calls_streaming`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0 (60 lines); hunks: -531,3 +531,63 @@ def test_streaming_split_delimiter_no_invalid_json(self, pa...; symbols: test_streaming_split_delimiter_no_invalid_json, test_streaming_does_not_duplicate_plain_text_after_tool_call, wrapped_extract_streaming, test_streaming_html_argument_does_not_duplicate_tag_prefixes
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -436,8 +436,10 @@ def extract_tool_calls_streaming(; symbols: extract_tool_calls_streaming
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -436,8 +436,10 @@ def extract_tool_calls_streaming(
-        # Reconstruct current_text after buffering to stay in sync
-        current_text = previous_text + delta_text
+        # Keep current_text from the upstream stream state. The buffered delta
+        # is only for emission, and must not be stitched back into the
+        # accumulated model text or normal content like "<div>" can be
+        # duplicated into "<<div>" when a tool call just ended.
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +60/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39114 - [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values

- Link: https://github.com/vllm-project/vllm/pull/39114
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; associated commits `13151a4df43d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +78/-8, 159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks, touching `test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks`; `vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8 (41 lines); hunks: -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:; -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:; symbols: _parse_gemma4_value, _parse_gemma4_args, touching `_parse_gemma4_value, _parse_gemma4_args`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0 (45 lines); hunks: -491,6 +491,51 @@ def test_streaming_numeric_args(self, parser, mock_request):; symbols: test_streaming_numeric_args, test_streaming_boolean_split_across_chunks, test_streaming_false_split_across_chunks, test_streaming_number_split_across_chunks
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8 (41 lines); hunks: -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:; -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:; symbols: _parse_gemma4_value, _parse_gemma4_args
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -78,7 +78,7 @@ def _parse_gemma4_value(value_str: str) -> object:
-def _parse_gemma4_args(args_str: str) -> dict:
+def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
@@ -89,6 +89,12 @@ def _parse_gemma4_args(args_str: str) -> dict:
+    Args:
+        args_str: The raw Gemma4 argument string.
+        partial: When True (streaming), bare values at end of string are
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +45/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +33/-8
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39027 - [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes

- Link: https://github.com/vllm-project/vllm/pull/39027
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_gemma4.jinja`, `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py` and 6 files; associated commits `8477fe427d17`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +878/-16, 1083 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes"; model line: Gemma 4; category: bug fix; main diff: `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py`; technical summary: Covers "[Tool] `adjust_request` to reasoning parser, and Gemma4 fixes"; the main implementation surface is `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode, touching `generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output`; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array, touching `test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld`; `vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3 (38 lines); hunks: -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -63,6 +73,29 @@ def end_token(self) -> str:; symbols: __init__, adjust_request, start_token, end_token, touching `__init__, adjust_request, start_token`; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -122,14 +122,16 @@ def _parse_gemma4_args(args_str: str, *, partial: bool = F...; symbols: _parse_gemma4_args, touching `_parse_gemma4_args`.
- Code diff details:
  - `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8 (95 lines); hunks: -4,6 +4,9; -100,6 +103,39 @@ def generic_tokenizer():; symbols: generic_tokenizer, test_gemma4_reasoning, gemma4_encode_output, _encode
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0 (40 lines); hunks: -114,6 +114,19 @@ def test_empty_value(self):; -636,3 +649,30 @@ def test_streaming_html_argument_does_not_duplicate_tag_pre...; symbols: test_empty_value, test_empty_value_partial_withheld, test_empty_value_after_other_keys_partial_withheld, TestParseGemma4Array
  - `vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3 (38 lines); hunks: -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -63,6 +73,29 @@ def end_token(self) -> str:; symbols: __init__, adjust_request, start_token, end_token
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2 (6 lines); hunks: -122,14 +122,16 @@ def _parse_gemma4_args(args_str: str, *, partial: bool = F...; symbols: _parse_gemma4_args
  - `tests/renderers/test_gemma4_chat_template.py` added +345/-0 (345 lines); hunks: -0,0 +1,345; symbols: gemma4_template, _render, TestGemma4ChatTemplate, test_basic_multiturn_thinking_disabled
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
diff -- vllm/reasoning/gemma4_reasoning_parser.py
@@ -52,6 +52,16 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
```

- Reviewed files:
  - tests: `tests/reasoning/test_gemma4_reasoning_parser.py` modified +87/-8; `tests/tool_parsers/test_gemma4_tool_parser.py` modified +40/-0; `tests/renderers/test_gemma4_chat_template.py` added +345/-0
  - runtime: `vllm/reasoning/gemma4_reasoning_parser.py` modified +35/-3; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-2
  - docs: `examples/tool_chat_template_gemma4.jinja` added +331/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_gemma4_reasoning_parser.py`, `tests/renderers/test_gemma4_chat_template.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39045 - [Gemma4] Support quantized MoE

- Link: https://github.com/vllm-project/vllm/pull/39045
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `3aecdf08b4a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-14, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Gemma4] Support quantized MoE"; model line: Gemma 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/gemma4.py`; technical summary: Covers "[Gemma4] Support quantized MoE"; the main implementation surface is `vllm/model_executor/models/gemma4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "Add Gemma4 Eagle3 support"; model line: Gemma 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "Add Gemma4 Eagle3 support"; the main implementation surface is `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; associated commits `b075604da10a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`"; model line: Gemma 4; category: bug fix; main diff: `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; technical summary: Covers "[Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"`"; the main implementation surface is `tests/tool_parsers/test_gemma4_tool_parser.py`, `vllm/tool_parsers/gemma4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types, touching `test_boolean_false, test_null_value, test_mixed_types`; `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0 (4 lines); hunks: -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:; symbols: _parse_gemma4_value, touching `_parse_gemma4_value`.
- Code diff details:
  - `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0 (8 lines); hunks: -85,6 +85,14 @@ def test_boolean_false(self):; symbols: test_boolean_false, test_null_value, test_mixed_types
  - `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0 (4 lines); hunks: -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:; symbols: _parse_gemma4_value
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
diff -- vllm/tool_parsers/gemma4_tool_parser.py
@@ -66,6 +66,10 @@ def _parse_gemma4_value(value_str: str) -> object:
+    # Null
+    if value_str.lower() in ("null", "none", "nil"):
+        return None
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_gemma4_tool_parser.py` modified +8/-0
  - runtime: `vllm/tool_parsers/gemma4_tool_parser.py` modified +4/-0
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
- Motivation: Title: "feat: Add LoRA support for Gemma4ForConditionalGeneration"; model line: Gemma 4; category: bug fix; main diff: `vllm/model_executor/models/gemma4_mm.py`; technical summary: Covers "feat: Add LoRA support for Gemma4ForConditionalGeneration"; the main implementation surface is `vllm/model_executor/models/gemma4_mm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `5371d6fb4023`
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
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/gemma4.py`; associated commits `ca97f7b9bbf2`
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
