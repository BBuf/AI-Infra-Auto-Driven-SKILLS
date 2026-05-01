# vllm Step 3.5 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `7075df79b3`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_step3p5_reasoning_parser.py` | [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `tests/tool_parsers/test_step3p5_tool_parser.py` | [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/model_executor/models/step3_text.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/step3_vl.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33755](https://github.com/vllm-project/vllm/pull/33755), [#34478](https://github.com/vllm-project/vllm/pull/34478) |
| `vllm/model_executor/models/step3p5_mtp.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/reasoning/step3p5_reasoning_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#34211](https://github.com/vllm-project/vllm/pull/34211) |
| `vllm/tool_parsers/step3p5_tool_parser.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523), [#33690](https://github.com/vllm-project/vllm/pull/33690) |
| `vllm/transformers_utils/configs/step3_vl.py` | 无直接 PR 号提交 |
| `vllm/transformers_utils/configs/step3p5.py` | [#33523](https://github.com/vllm-project/vllm/pull/33523) |
| `vllm/transformers_utils/processors/step3_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 6
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
| 2026-03-20 | [#37579](https://github.com/vllm-project/vllm/pull/37579) | merged | [Model] Refactor Step3-VL processor to HF style | `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py` |

## 逐 PR diff 审计卡

### PR #33523 - [Models] Step-3.5-Flash

- 链接: https://github.com/vllm-project/vllm/pull/33523
- 状态/时间: merged / 2026-02-02
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/step3p5.py`；关联提交 `db4ede974343`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+28/-32，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Enable Step3p5ForCausalLM testing」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/step3p5.py`；未提供可用技术摘要。
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

### PR #37579 - [Model] Refactor Step3-VL processor to HF style

- 链接: https://github.com/vllm-project/vllm/pull/37579
- 状态/时间: merged / 2026-03-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+228/-160，可读 patch 511 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Refactor Step3-VL processor to HF style」；模型线: Step 3.5；类别: 文档/测试/CI；主要 diff: `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`；技术摘要: 覆盖「[Model] Refactor Step3-VL processor to HF style」；主要实现面是 `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunks: -8,13 +8,13; -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -...; symbols: Step3VisionProcessor, get_num_patches, __call__，涉及 `Step3VisionProcessor, get_num_patches, __call__`；`vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunks: -39,7 +39,11; -86,21 +90,30 @@ class Step3VLImageEmbeddingInputs(TensorSchema):; symbols: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor，涉及 `Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor`；`vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunks: -558,6 +558,7 @@ def __call__(; symbols: __call__，涉及 `__call__`；`vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunks: -19,7 +19,6 @@ def __init__(; symbols: __init__, __call__，涉及 `__init__, __call__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunks: -8,13 +8,13; -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -...; symbols: Step3VisionProcessor, get_num_patches, __call__
  - `vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunks: -39,7 +39,11; -86,21 +90,30 @@ class Step3VLImageEmbeddingInputs(TensorSchema):; symbols: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor
  - `vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunks: -558,6 +558,7 @@ def __call__(; symbols: __call__
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunks: -19,7 +19,6 @@ def __init__(; symbols: __init__, __call__
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/step3_vl.py
@@ -8,13 +8,13 @@
-from transformers import BatchFeature, PretrainedConfig, TensorType
+from transformers import BatchFeature, ProcessorMixin, TensorType
-ImageWithPatches = tuple[Image.Image, list[Image.Image], list[bool] | None]
+ImageWithPatches = tuple[Image.Image, list[Image.Image], list[bool]]
@@ -185,7 +185,7 @@ def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
-    ) -> tuple[Image.Image, list[Image.Image], list[bool] | None]:
diff -- vllm/model_executor/models/step3_vl.py
@@ -39,7 +39,11 @@
-from vllm.transformers_utils.processors.step3_vl import Step3VLProcessor
+from vllm.transformers_utils.processors.step3_vl import (
+    MAX_IMAGE_SIZE,
+    Step3VLImageProcessor,
+    Step3VLProcessor,
+)
diff -- vllm/transformers_utils/processors/internvl.py
@@ -558,6 +558,7 @@ def __call__(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127; `vllm/model_executor/models/step3_vl.py` modified +27/-29; `vllm/transformers_utils/processors/internvl.py` modified +4/-3; `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`, `vllm/transformers_utils/processors/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
