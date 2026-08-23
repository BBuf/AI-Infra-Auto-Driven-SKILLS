# vllm Step 3.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/processing/test_step3_vl_image_embeds.py` | 无直接 PR 号提交 |
| `tests/reasoning/test_step3p5_reasoning_parser.py` | [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `tests/tool_parsers/test_step3p5_tool_parser.py` | [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/model_executor/models/step3_text.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/step3_vl.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33755](https://github.com/vllm-project/vllm/pull/33755), [#34478](https://github.com/vllm-project/vllm/pull/34478), [#41892](https://github.com/vllm-project/vllm/pull/41892) |
| `vllm/model_executor/models/step3p5_mtp.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#48883](https://github.com/vllm-project/vllm/pull/48883) |
| `vllm/reasoning/step3p5_reasoning_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `vllm/tool_parsers/step3p5_tool_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/transformers_utils/configs/step3_vl.py` | 无直接 PR 号提交 |
| `vllm/transformers_utils/configs/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/transformers_utils/processors/step3_vl.py` | 无直接 PR 号提交 |
| `vllm/v1/spec_decode/step3p5.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 6
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 10
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-02-02 | [#33523](https://github.com/vllm-project/vllm/pull/33523) | merged | [Models] Step-3.5-Flash | `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py` |
| 2026-02-05 | [#33690](https://github.com/vllm-project/vllm/pull/33690) | merged | [Bugfix] Fix step3p5 parser when using mtp | `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-02-07 | [#33755](https://github.com/vllm-project/vllm/pull/33755) | merged | [Model] Enable Step3p5ForCausalLM testing | `vllm/model_executor/models/step3p5.py` |
| 2026-02-22 | [#34478](https://github.com/vllm-project/vllm/pull/34478) | merged | [Model] Add NVFP4 quantization support for Step3.5-Flash | `vllm/model_executor/models/step3p5.py` |
| 2026-02-25 | [#34211](https://github.com/vllm-project/vllm/pull/34211) | merged | [Bugfix] Fix step3p5 reasoning with interleaved thinking | `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py` |
| 2026-05-13 | [#41892](https://github.com/vllm-project/vllm/pull/41892) | merged | [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports) | `vllm/model_executor/models/step3p5.py` |
| 2026-05-18 | [#42224](https://github.com/vllm-project/vllm/pull/42224) | merged | [MM][CG] Enable encoder Cudagraph for Step3VL | `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py` |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | merged | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |

## 逐 PR diff 审计卡

### PR #33523 - [Models] Step-3.5-Flash

- 链接: https://github.com/vllm-project/vllm/pull/33523
- 状态/时间: merged / 2026-02-02
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/33523 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py`, `vllm/reasoning/step3p5_reasoning_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/transformers_utils/configs/step3p5.py`；关联提交 `c3b40dc3e74d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+3107/-4，可读 patch 3270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Step-3.5-Flash」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py`；技术摘要: 覆盖「[Models] Step-3.5-Flash」；主要实现面是 `vllm/tool_parsers/step3p5_tool_parser.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/models/step3p5_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0 (1511 lines); hunks: -0,0 +1,1511; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks，涉及 `StreamingXMLToolCallParser, __init__, reset_streaming_state`；`vllm/model_executor/models/step3p5.py` added +894/-0 (894 lines); hunks: -0,0 +1,894; symbols: FP32ReplicatedLinear, forward, Step3p5MLP, __init__，涉及 `FP32ReplicatedLinear, forward, Step3p5MLP`；`vllm/model_executor/models/step3p5_mtp.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: SharedHead, __init__, forward, Step3p5AMultiTokenPredictorLayer，涉及 `SharedHead, __init__, forward`；`vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0 (153 lines); hunks: -0,0 +1,153; symbols: Step3p5ReasoningParser, start_token, end_token, __init__，涉及 `Step3p5ReasoningParser, start_token, end_token`。
- 代码 diff 细节:
  - `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0 (1511 lines); hunks: -0,0 +1,1511; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/model_executor/models/step3p5.py` added +894/-0 (894 lines); hunks: -0,0 +1,894; symbols: FP32ReplicatedLinear, forward, Step3p5MLP, __init__
  - `vllm/model_executor/models/step3p5_mtp.py` added +315/-0 (315 lines); hunks: -0,0 +1,315; symbols: SharedHead, __init__, forward, Step3p5AMultiTokenPredictorLayer
  - `vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0 (153 lines); hunks: -0,0 +1,153; symbols: Step3p5ReasoningParser, start_token, end_token, __init__
  - `vllm/transformers_utils/configs/step3p5.py` added +100/-0 (100 lines); hunks: -0,0 +1,100; symbols: Step3p5Config, __init__
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -0,0 +1,1511 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import ast
+import json
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/model_executor/models/step3p5.py
@@ -0,0 +1,894 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Jurassic model."""
+from collections.abc import Iterable
+from typing import Any
+import torch
diff -- vllm/model_executor/models/step3p5_mtp.py
@@ -0,0 +1,315 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/step3p5_tool_parser.py` added +1511/-0; `vllm/model_executor/models/step3p5.py` added +894/-0; `vllm/model_executor/models/step3p5_mtp.py` added +315/-0; `vllm/reasoning/step3p5_reasoning_parser.py` added +153/-0; `vllm/transformers_utils/configs/step3p5.py` added +100/-0
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_activation.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33690 - [Bugfix] Fix step3p5 parser when using mtp

- 链接: https://github.com/vllm-project/vllm/pull/33690
- 状态/时间: merged / 2026-02-05
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/33690 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；关联提交 `82914d2ae8d0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+1455/-5，可读 patch 1508 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix step3p5 parser when using mtp」；模型线: Step 3.5；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix step3p5 parser when using mtp」；主要实现面是 `tests/tool_parsers/test_step3p5_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0 (1435 lines); hunks: -0,0 +1,1435; symbols: step3p5_tokenizer, step3p5_tool_parser, sample_tools, assert_tool_calls，涉及 `step3p5_tokenizer, step3p5_tool_parser, sample_tools`；`vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5 (25 lines); hunks: -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; -110,7 +125,7 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; symbols: parse_single_streaming_chunks，涉及 `parse_single_streaming_chunks`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0 (1435 lines); hunks: -0,0 +1,1435; symbols: step3p5_tokenizer, step3p5_tool_parser, sample_tools, assert_tool_calls
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5 (25 lines); hunks: -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; -110,7 +125,7 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> D...; symbols: parse_single_streaming_chunks
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_step3p5_tool_parser.py
@@ -0,0 +1,1435 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+from collections.abc import Generator
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import (
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -97,11 +97,26 @@ def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage:
+        entry_call_id = self.current_call_id
+        entry_tool_call_index = self.tool_call_index
+        fallback_call_id = None
+        if entry_call_id is not None:
+            if (
+                self.current_call_id == entry_call_id
```

- 已读文件:
  - tests: `tests/tool_parsers/test_step3p5_tool_parser.py` added +1435/-0
  - runtime: `vllm/tool_parsers/step3p5_tool_parser.py` modified +20/-5
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_step3p5_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33755 - [Model] Enable Step3p5ForCausalLM testing

- 链接: https://github.com/vllm-project/vllm/pull/33755
- 状态/时间: merged / 2026-02-07
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/33755 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/step3p5.py`；关联提交 `db4ede974343`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+28/-32，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable Step3p5ForCausalLM testing」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/step3p5.py`；技术摘要: 覆盖「[Model] Enable Step3p5ForCausalLM testing」；主要实现面是 `vllm/model_executor/models/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunks: -36,7 +36,6; -770,37 +769,17 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunks: -36,7 +36,6; -770,37 +769,17 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -36,7 +36,6 @@
-    DEFAULT_VOCAB_PADDING_SIZE,
@@ -770,37 +769,17 @@ def __init__(
-        lora_config = vllm_config.lora_config
-        self.config = config
-        self.vllm_config = vllm_config
-        self.moe_layers: list[FusedMoEBlock] = []
```

- 已读文件:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +12/-25
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34478 - [Model] Add NVFP4 quantization support for Step3.5-Flash

- 链接: https://github.com/vllm-project/vllm/pull/34478
- 状态/时间: merged / 2026-02-22
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/34478 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/step3p5.py`；关联提交 `b7892a3beff0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+204/-4，可读 patch 291 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add NVFP4 quantization support for Step3.5-Flash」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/step3p5.py`；技术摘要: 覆盖「[Model] Add NVFP4 quantization support for Step3.5-Flash」；主要实现面是 `vllm/model_executor/models/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunks: -2,7 +2,8; -231,6 +232,7 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunks: -2,7 +2,8; -231,6 +232,7 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -2,7 +2,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -231,6 +232,7 @@ def __init__(
+                quant_config=quant_config,
@@ -640,12 +642,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +71/-1
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_nvfp4_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34211 - [Bugfix] Fix step3p5 reasoning with interleaved thinking

- 链接: https://github.com/vllm-project/vllm/pull/34211
- 状态/时间: merged / 2026-02-25
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/34211 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py`；关联提交 `af5e6afa0af2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+387/-14，可读 patch 423 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix step3p5 reasoning with interleaved thinking」；模型线: Step 3.5；类别: 缺陷修复；主要 diff: `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py`；技术摘要: 覆盖「[Bugfix] Fix step3p5 reasoning with interleaved thinking」；主要实现面是 `tests/reasoning/test_step3p5_reasoning_parser.py`, `vllm/reasoning/step3p5_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0 (341 lines); hunks: -0,0 +1,341; symbols: step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline，涉及 `step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline`；`vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14 (60 lines); hunks: -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -136,9 +171,6 @@ def extract_reasoning_streaming(; symbols: __init__, is_reasoning_end, is_reasoning_end_streaming, _is_reasoning_end_from_ids，涉及 `__init__, is_reasoning_end, is_reasoning_end_streaming`。
- 代码 diff 细节:
  - `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0 (341 lines); hunks: -0,0 +1,341; symbols: step3p5_tokenizer, test_reasoning, test_step3p5_streaming_drops_leading_newline
  - `vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14 (60 lines); hunks: -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):; -136,9 +171,6 @@ def extract_reasoning_streaming(; symbols: __init__, is_reasoning_end, is_reasoning_end_streaming, _is_reasoning_end_from_ids
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_step3p5_reasoning_parser.py
@@ -0,0 +1,341 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
diff -- vllm/reasoning/step3p5_reasoning_parser.py
@@ -39,24 +39,59 @@ def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
-        # Used to delay the reasoning end detection.
-        # This is necessary to remove the newline appears immediately after </think>,
-        # which may cause the end detection to be delayed by one round.
-        self.end_offset = 1
+        # Tracks whether we've seen </think> but are still waiting for one more
+        # token to confirm the end.
```

- 已读文件:
  - tests: `tests/reasoning/test_step3p5_reasoning_parser.py` added +341/-0
  - runtime: `vllm/reasoning/step3p5_reasoning_parser.py` modified +46/-14
- 验证与风险: diff 自带测试面 `tests/reasoning/test_step3p5_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41892 - [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)

- 链接: https://github.com/vllm-project/vllm/pull/41892
- 状态/时间: merged / 2026-05-13
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/41892 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/step3p5.py`；关联提交 `3b1ef03be4a3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+46/-4，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)」；模型线: Step 3.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/step3p5.py`；技术摘要: 覆盖「[Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports)」；主要实现面是 `vllm/model_executor/models/step3p5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/step3p5.py` modified +6/-0 (6 lines); hunks: -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, Step3p5ForCausalLM，涉及 `load_weights, Step3p5ForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/step3p5.py` modified +6/-0 (6 lines); hunks: -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, Step3p5ForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/step3p5.py
@@ -817,6 +817,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+    # Required so quantization exclude lists match fused module prefixes.
+    packed_modules_mapping = {
+        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
+    }
```

- 已读文件:
  - runtime: `vllm/model_executor/models/step3p5.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py`, `vllm/model_executor/models/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42224 - [MM][CG] Enable encoder Cudagraph for Step3VL

- 链接: https://github.com/vllm-project/vllm/pull/42224
- 状态/时间: merged / 2026-05-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+384/-22，可读 patch 534 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][CG] Enable encoder Cudagraph for Step3VL」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`；技术摘要: 覆盖「[MM][CG] Enable encoder Cudagraph for Step3VL」；主要实现面是 `vllm/model_executor/models/step3_vl.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/step3_vl.py` modified +323/-2 (325 lines); hunks: -46,7 +46,12; -487,7 +492,9 @@ def forward(; symbols: forward, Step3VLForConditionalGeneration, __init__, device，涉及 `forward, Step3VLForConditionalGeneration, __init__`；`vllm/model_executor/models/interfaces.py` modified +21/-0 (21 lines); hunks: -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(; symbols: select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs，涉及 `select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs`；`vllm/model_executor/models/utils.py` modified +16/-0 (16 lines); hunks: -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: get_layer_index, scatter_output_slices，涉及 `get_layer_index, scatter_output_slices`；`tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -41,6 +41,13 @@ def qwen_vl_chat_template(content: str) -> str:; -90,6 +97,11 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, step3_vl_chat_template，涉及 `qwen_vl_chat_template, step3_vl_chat_template`。
- 代码 diff 细节:
  - `vllm/model_executor/models/step3_vl.py` modified +323/-2 (325 lines); hunks: -46,7 +46,12; -487,7 +492,9 @@ def forward(; symbols: forward, Step3VLForConditionalGeneration, __init__, device
  - `vllm/model_executor/models/interfaces.py` modified +21/-0 (21 lines); hunks: -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(; symbols: select_encoder_cudagraph_items, postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs
  - `vllm/model_executor/models/utils.py` modified +16/-0 (16 lines); hunks: -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_la...; symbols: get_layer_index, scatter_output_slices
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0 (12 lines); hunks: -41,6 +41,13 @@ def qwen_vl_chat_template(content: str) -> str:; -90,6 +97,11 @@ def qwen_vl_chat_template(content: str) -> str:; symbols: qwen_vl_chat_template, step3_vl_chat_template
  - `docs/design/cuda_graphs_multimodal.md` modified +2/-0 (2 lines); hunks: -77,6 +77,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...; -89,6 +90,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [Supp...
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/step3_vl.py
@@ -46,7 +46,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsEncoderCudaGraph,
+    SupportsMultiModal,
+    SupportsPP,
diff -- vllm/model_executor/models/interfaces.py
@@ -1594,6 +1594,27 @@ def select_encoder_cudagraph_items(
+    def postprocess_encoder_output(
+        self,
+        output: torch.Tensor,
+        indices: list[int],
+        per_item_out_tokens: list[int],
+        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
diff -- vllm/model_executor/models/utils.py
@@ -884,3 +884,19 @@ def get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/step3_vl.py` modified +323/-2; `vllm/model_executor/models/interfaces.py` modified +21/-0; `vllm/model_executor/models/utils.py` modified +16/-0; `vllm/model_executor/models/step_vl.py` modified +1/-0; `vllm/v1/worker/encoder_cudagraph.py` modified +8/-20
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +12/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +2/-0; `examples/generate/multimodal/vision_language_offline.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44346 - [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers

- 链接: https://github.com/vllm-project/vllm/pull/44346
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+20/-15，可读 patch 178 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；模型线: Step 3.5；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`；技术摘要: 覆盖「[Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers」；主要实现面是 `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap，涉及 `safe_literal_eval, partial_tag_overlap`；`vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize，涉及 `_try_parse_wildcard_number, _deserialize`；`vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments，涉及 `_parse_arguments`；`vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element，涉及 `_end_element`。
- 代码 diff 细节:
  - `vllm/tool_parsers/utils.py` modified +7/-0 (7 lines); hunks: -3,6 +3,7; -31,6 +32,12; symbols: safe_literal_eval, partial_tag_overlap
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3 (6 lines); hunks: -1,7 +1,6; -27,6 +26,7; symbols: _try_parse_wildcard_number, _deserialize
  - `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,7 +1,6; -28,7 +27,7; symbols: _parse_arguments
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3 (5 lines); hunks: -1,6 +1,5; -26,7 +25,7; symbols: _end_element
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2 (4 lines); hunks: -11,7 +11,6; -42,6 +41,7; symbols: _deserialize
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/utils.py
@@ -3,6 +3,7 @@
+import warnings
@@ -31,6 +32,12 @@
+def safe_literal_eval(text: str):
+    with warnings.catch_warnings():
+        warnings.simplefilter("ignore", SyntaxWarning)
+        return ast.literal_eval(text)
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -1,7 +1,6 @@
-import ast
@@ -27,6 +26,7 @@
+from vllm.tool_parsers.utils import safe_literal_eval
@@ -183,13 +183,13 @@ def _try_parse_wildcard_number(value: str) -> int | float | None:
-        """Deserialize a string value using json.loads then ast.literal_eval."""
+        """Deserialize a string value using json.loads then safe_literal_eval."""
diff -- vllm/tool_parsers/minicpm5xml_tool_parser.py
@@ -1,7 +1,6 @@
```

- 已读文件:
  - runtime: `vllm/tool_parsers/utils.py` modified +7/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +3/-3; `vllm/tool_parsers/minicpm5xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +2/-3; `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +2/-2; `vllm/tool_parsers/poolside_v1_tool_parser.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/glm4_moe_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/tool_parsers/minicpm5xml_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- 链接: https://github.com/vllm-project/vllm/pull/41184
- 状态/时间: merged / 2026-06-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 90 个文件，+2734/-2027，可读 patch 7329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`；技术摘要: 覆盖「[MoE Refactor] FusedMoE/MoERunner inversion refactor」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- 链接: https://github.com/vllm-project/vllm/pull/43586
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+809/-69，可读 patch 1559 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；模型线: Step 3.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`；技术摘要: 覆盖「[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR」；主要实现面是 `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features，涉及 `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__`；`docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata，涉及 `BudgetGraphMetadata`；`tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template，涉及 `VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template`；`examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2，涉及 `run_tarsier2`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features
  - `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template
  - `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2
  - `vllm/model_executor/models/interfaces.py` modified +5/-0 (5 lines); hunks: -1623,6 +1623,7 @@ def postprocess_encoder_output(; -1643,6 +1644,7 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, prepare_encoder_cudagraph_replay_buffers, encoder_cudagraph_forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -4,7 +4,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -15,6 +15,7 @@
+    SupportsEncoderCudaGraph,
@@ -52,6 +53,7 @@
+    IMAGE_SIZE,
diff -- docs/design/cuda_graphs_multimodal.md
@@ -2,6 +2,8 @@
+For two-tower vision encoders (e.g., DeepSeek-OCR's SAM + CLIP with dynamic tiling), a **dual-path graph** mode captures two independent sets of CUDA graphs — one for the global i
@@ -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on the host side. Th
+For two-tower vision encoders such as DeepSeek-OCR (SAM + CLIP with dynamic tiling), the global image path and local patch path have independent token profiles (272 tokens per glo
@@ -37,17 +41,57 @@ class BudgetGraphMetadata:
+When `EncoderCudaGraphConfig.enable_dual_path_graph` is `True`, the manager generates two independent budget lists — `global_token_budgets` (multiples of `global_token_per_image`)
+For dual-path models, the manager routes to `_execute_local_dual_path()`, which constrains both global and local token budgets simultaneously during packing (see [Dual-Path graph
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -29,6 +29,7 @@ class VitCudagraphTestConfig:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5; `vllm/model_executor/models/interfaces.py` modified +5/-0; `vllm/model_executor/models/step3_vl.py` modified +5/-0; `vllm/model_executor/models/glm4_1v.py` modified +4/-0; `vllm/model_executor/models/internvl.py` modified +4/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +63/-16; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
