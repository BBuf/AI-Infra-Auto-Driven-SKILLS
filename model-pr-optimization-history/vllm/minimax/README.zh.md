# vllm MiniMax M2 Series 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_minimax_m1.jinja` | [#20297](https://github.com/vllm-project/vllm/pull/20297) |
| `tests/kernels/core/test_minimax_reduce_rms.py` | [#37045](https://github.com/vllm-project/vllm/pull/37045) |
| `tests/models/multimodal/processing/test_minimax_vl_01.py` | [#16328](https://github.com/vllm-project/vllm/pull/16328), [#17354](https://github.com/vllm-project/vllm/pull/17354) |
| `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882) |
| `tests/reasoning/test_minimax_m2_reasoning_parser.py` | [#29882](https://github.com/vllm-project/vllm/pull/29882) |
| `tests/tool_parsers/test_minimax_m2_tool_parser.py` | [#35895](https://github.com/vllm-project/vllm/pull/35895) |
| `tests/tool_parsers/test_minimax_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` | [#37045](https://github.com/vllm-project/vllm/pull/37045) |
| `vllm/model_executor/models/minimax_m2.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#27537](https://github.com/vllm-project/vllm/pull/27537), [#27627](https://github.com/vllm-project/vllm/pull/27627), [#30384](https://github.com/vllm-project/vllm/pull/30384), [#31493](https://github.com/vllm-project/vllm/pull/31493), [#32736](https://github.com/vllm-project/vllm/pull/32736), [#32763](https://github.com/vllm-project/vllm/pull/32763), [#36965](https://github.com/vllm-project/vllm/pull/36965), [#37045](https://github.com/vllm-project/vllm/pull/37045), [#37214](https://github.com/vllm-project/vllm/pull/37214), [#37512](https://github.com/vllm-project/vllm/pull/37512) |
| `vllm/model_executor/models/minimax_text_01.py` | [#13454](https://github.com/vllm-project/vllm/pull/13454), [#16328](https://github.com/vllm-project/vllm/pull/16328), [#19592](https://github.com/vllm-project/vllm/pull/19592), [#20211](https://github.com/vllm-project/vllm/pull/20211), [#22151](https://github.com/vllm-project/vllm/pull/22151), [#22589](https://github.com/vllm-project/vllm/pull/22589), [#22928](https://github.com/vllm-project/vllm/pull/22928), [#23831](https://github.com/vllm-project/vllm/pull/23831), [#37371](https://github.com/vllm-project/vllm/pull/37371) |
| `vllm/model_executor/models/minimax_vl_01.py` | [#16328](https://github.com/vllm-project/vllm/pull/16328), [#17354](https://github.com/vllm-project/vllm/pull/17354), [#22116](https://github.com/vllm-project/vllm/pull/22116) |
| `vllm/parser/minimax_m2_parser.py` | [#39683](https://github.com/vllm-project/vllm/pull/39683), [#39861](https://github.com/vllm-project/vllm/pull/39861) |
| `vllm/reasoning/minimax_m2_reasoning_parser.py` | [#27535](https://github.com/vllm-project/vllm/pull/27535), [#29882](https://github.com/vllm-project/vllm/pull/29882), [#35352](https://github.com/vllm-project/vllm/pull/35352) |
| `vllm/tool_parsers/minimax_m2_tool_parser.py` | [#30555](https://github.com/vllm-project/vllm/pull/30555), [#31083](https://github.com/vllm-project/vllm/pull/31083), [#32278](https://github.com/vllm-project/vllm/pull/32278), [#32342](https://github.com/vllm-project/vllm/pull/32342), [#35895](https://github.com/vllm-project/vllm/pull/35895) |
| `vllm/tool_parsers/minimax_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 30
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 32
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-01 | [#13454](https://github.com/vllm-project/vllm/pull/13454) | merged | [Model][MiniMaxText01] Support MiniMaxText01 model inference | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-04-29 | [#16328](https://github.com/vllm-project/vllm/pull/16328) | merged | [Model] support MiniMax-VL-01 model | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` |
| 2025-04-29 | [#17354](https://github.com/vllm-project/vllm/pull/17354) | merged | [Bugfix] Clean up MiniMax-VL and fix processing | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py` |
| 2025-06-13 | [#19592](https://github.com/vllm-project/vllm/pull/19592) | merged | [Model] Fix minimax model cache & lm_head precision | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-06-16 | [#19677](https://github.com/vllm-project/vllm/pull/19677) | merged | [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM) | `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` |
| 2025-06-28 | [#20199](https://github.com/vllm-project/vllm/pull/20199) | merged | [CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes | `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2025-07-03 | [#20297](https://github.com/vllm-project/vllm/pull/20297) | merged | [Feature] Support MiniMax-M1 function calls features | `examples/tool_chat_template_minimax_m1.jinja` |
| 2025-07-11 | [#20211](https://github.com/vllm-project/vllm/pull/20211) | merged | [Model] Support HF format of minimax | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-09 | [#22151](https://github.com/vllm-project/vllm/pull/22151) | merged | [V1] [Hybrid] Support Minimax-Text-01 in V1 | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-15 | [#22928](https://github.com/vllm-project/vllm/pull/22928) | merged | [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax) | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-19 | [#22116](https://github.com/vllm-project/vllm/pull/22116) | merged | [Bugfix] Fix broken Minimax-01-VL model | `vllm/model_executor/models/minimax_vl_01.py` |
| 2025-08-27 | [#22589](https://github.com/vllm-project/vllm/pull/22589) | merged | [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models | `vllm/model_executor/models/minimax_text_01.py` |
| 2025-08-30 | [#23831](https://github.com/vllm-project/vllm/pull/23831) | merged | [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba | `vllm/model_executor/models/minimax_text_01.py` |
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
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/minimax_text_01.py` |
| 2026-03-26 | [#37214](https://github.com/vllm-project/vllm/pull/37214) | merged | Fix minimax m2.5 nvfp4 kv scales weight loading | `vllm/model_executor/models/minimax_m2.py` |
| 2026-03-30 | [#36965](https://github.com/vllm-project/vllm/pull/36965) | merged | [Model][Quantization] Add GGUF support for MiniMax-M2.1 | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-06 | [#37512](https://github.com/vllm-project/vllm/pull/37512) | merged | MiniMax-M2: add Eagle3 speculative decoding support | `vllm/model_executor/models/minimax_m2.py` |
| 2026-04-10 | [#37045](https://github.com/vllm-project/vllm/pull/37045) | merged | [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels | `vllm/model_executor/models/minimax_m2.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `tests/kernels/core/test_minimax_reduce_rms.py` |
| 2026-04-14 | [#39683](https://github.com/vllm-project/vllm/pull/39683) | merged | [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter | `vllm/parser/minimax_m2_parser.py` |
| 2026-04-16 | [#39861](https://github.com/vllm-project/vllm/pull/39861) | merged | [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__() | `vllm/parser/minimax_m2_parser.py` |

## 逐 PR diff 审计卡

### PR #13454 - [Model][MiniMaxText01] Support MiniMaxText01 model inference

- 链接: https://github.com/vllm-project/vllm/pull/13454
- 状态/时间: merged / 2025-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `9ef98d527ee1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+2440/-130，可读 patch 2657 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][MiniMaxText01] Support MiniMaxText01 model inference」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: This PR is intended to support the MiniMaxText01 model inference. It can run on a single machine with 8xH800 and 8xH20, where a single H800 machine can handle a maximum context...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func，涉及 `replace_weight_name, weight_loader_with_alias, wrapper`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: -0,0 +1,1273; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -0,0 +1,1273 @@
+# SPDX-License-Identifier: Apache-2.0
+"""Inference-only MiniMaxText01 model."""
+import copy
+import math
+import re
+from typing import Dict, Iterable, List, Optional, Tuple, Union
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` added +1273/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_lightning_attn.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16328 - [Model] support MiniMax-VL-01 model

- 链接: https://github.com/vllm-project/vllm/pull/16328
- 状态/时间: merged / 2025-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/models/minimax_vl_01.py`；关联提交 `cde384cd92c8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+954/-19，可读 patch 1193 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] support MiniMax-VL-01 model」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`；PR 正文摘要: CLOSES #12073 This PR is intended to support the MiniMaxVL01 model inference. It adopts the “ViT-MLP-LLM” framework, which is a commonly used technique in the field of multimoda...。
- 实现要点: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches，涉及 `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`；`tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression，涉及 `test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements`；`vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__，涉及 `MiniMaxVL01Config, __init__`；`vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__，涉及 `MiniMaxText01Config, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: -0,0 +1,615; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
  - `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: MiniMaxVL01Config, __init__
  - `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxText01Config, __init__
  - `vllm/model_executor/models/minimax_text_01.py` modified +53/-14 (67 lines); hunks: -3,7 +3,7; -110,7 +110,17 @@ def _forward(; symbols: _forward, forward, _prefill_and_mix_infer, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` added +615/-0; `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0; `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0; `vllm/model_executor/models/minimax_text_01.py` modified +53/-14
  - tests: `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17354 - [Bugfix] Clean up MiniMax-VL and fix processing

- 链接: https://github.com/vllm-project/vllm/pull/17354
- 状态/时间: merged / 2025-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/model_executor/models/minimax_vl_01.py`；关联提交 `00ee37efa236`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+38/-283，可读 patch 442 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Clean up MiniMax-VL and fix processing」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`；PR 正文摘要: Follow-up to #16328 @qscqesze can you verify this fix? Also, can you open another PR after this one to add example scripts under `examples/offline_inference/vision_language.py`...。
- 实现要点: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches，涉及 `MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs`；`tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override，涉及 `test_processor_override`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282 (312 lines); hunks: -1,52 +1,32; -69,66 +49,8 @@ class MiniMaxVL01ImageEmbeddingInputs(TypedDict):; symbols: MaxImageTokenMeta, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1 (1 lines); hunks: -12,7 +12,6; symbols: test_processor_override
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,52 +1,32 @@
+from collections.abc import Iterable, Mapping
+from typing import Literal, Optional, Set, Tuple, TypedDict, Union, cast
-from abc import abstractmethod
-from collections.abc import Iterable, Mapping, Sequence
-from dataclasses import dataclass
-from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
diff -- tests/models/multimodal/processing/test_minimax_vl_01.py
@@ -12,7 +12,6 @@
-# yapf: enable
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +30/-282
  - tests: `tests/models/multimodal/processing/test_minimax_vl_01.py` modified +0/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19592 - [Model] Fix minimax model cache & lm_head precision

- 链接: https://github.com/vllm-project/vllm/pull/19592
- 状态/时间: merged / 2025-06-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `a24cb91600bd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix minimax model cache & lm_head precision」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: Change the precision of the MiniMax model in vLLM: update the LM head and KV cache from bfloat16 (bf16) to float32 (fp32). Purpose: To improve numerical stability and output acc...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits，涉及 `layer_fn, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +3/-3 (6 lines); hunks: -856,7 +856,7 @@ def layer_fn(prefix):; -1021,7 +1021,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: layer_fn, __init__, forward, compute_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19677 - [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)

- 链接: https://github.com/vllm-project/vllm/pull/19677
- 状态/时间: merged / 2025-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-0，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`；PR 正文摘要: We would like to propose official support for the MiniMaxM1ForCausalLM model within vLLM. This model shares the exact same architecture and inference behavior as MiniMaxText01Fo...。
- 实现要点: `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name，涉及 `name`。
- 代码 diff 细节:
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -205,6 +205,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -370,6 +370,7 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -36,6 +36,7; symbols: name
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
  - runtime: `vllm/model_executor/models/registry.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20199 - [CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes

- 链接: https://github.com/vllm-project/vllm/pull/20199
- 状态/时间: merged / 2025-06-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-1，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI Fix] Pin tests/models/registry.py MiniMaxText01ForCausalLM to revision due to model changes」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `tests/models/registry.py`, `tests/models/test_initialization.py`；PR 正文摘要: FIX https://github.com/vllm-project/vllm/issues/20198 It seems like the model definition has changed upstream this morning https://huggingface.co/MiniMaxAI/MiniMax-Text-01/commi...。
- 实现要点: `tests/models/registry.py` modified +8/-1 (9 lines); hunks: -70,6 +70,12 @@ class _HfExamplesInfo:; -207,7 +213,8 @@ def check_available_online(; symbols: _HfExamplesInfo, check_transformers_version, check_available_online，涉及 `_HfExamplesInfo, check_transformers_version, check_available_online`；`tests/models/test_initialization.py` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1，涉及 `_initialize_kv_caches_v1`。
- 代码 diff 细节:
  - `tests/models/registry.py` modified +8/-1 (9 lines); hunks: -70,6 +70,12 @@ class _HfExamplesInfo:; -207,7 +213,8 @@ def check_available_online(; symbols: _HfExamplesInfo, check_transformers_version, check_available_online
  - `tests/models/test_initialization.py` modified +1/-0 (1 lines); hunks: -88,6 +88,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/models/registry.py` modified +8/-1; `tests/models/test_initialization.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/test_initialization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20297 - [Feature] Support MiniMax-M1 function calls features

- 链接: https://github.com/vllm-project/vllm/pull/20297
- 状态/时间: merged / 2025-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_minimax_m1.jinja`；关联提交 `363528de27db`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+842/-1，可读 patch 866 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Support MiniMax-M1 function calls features」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `examples/tool_chat_template_minimax_m1.jinja`；PR 正文摘要: Support MiniMax-M1 function calls features How to use: test_scripts: Outpt: Streaming Test: Output:。
- 实现要点: `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91。
- 代码 diff 细节:
  - `examples/tool_chat_template_minimax_m1.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_minimax_m1.jinja
@@ -0,0 +1,91 @@
+{{ '<begin_of_document>' -}}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools is defined %}
+    {%- set tools = none %}
```

- 已读文件:
  - docs: `examples/tool_chat_template_minimax_m1.jinja` added +91/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/test_minimax_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20211 - [Model] Support HF format of minimax

- 链接: https://github.com/vllm-project/vllm/pull/20211
- 状态/时间: merged / 2025-07-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `922f316441ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+36/-11，可读 patch 101 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support HF format of minimax」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: https://github.com/vllm-project/vllm/pull/20199 set us up to test the older format of `MiniMaxAI/MiniMax-Text-01` we initially implemented. This PR implements the small changes...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer，涉及 `__init__, which_layer, is_linear_attn_layer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +33/-11 (44 lines); hunks: -667,16 +667,24 @@ def __init__(; -794,6 +802,18 @@ def __init__(; symbols: __init__, which_layer, is_linear_attn_layer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -667,16 +667,24 @@ def __init__(
-                config, 'layernorm_linear_attention_alpha', 1)
+                config, 'layernorm_linear_attention_alpha',
+                getattr(config, 'linear_attn_alpha_factor', 1))
-                config, 'layernorm_linear_attention_beta', 1)
+                config, 'layernorm_linear_attention_beta',
+                getattr(config, 'linear_attn_beta_factor', 1))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +33/-11
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22151 - [V1] [Hybrid] Support Minimax-Text-01 in V1

- 链接: https://github.com/vllm-project/vllm/pull/22151
- 状态/时间: merged / 2025-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `6ade99eafa37`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+234/-42，可读 patch 448 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] [Hybrid] Support Minimax-Text-01 in V1」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape，涉及 `jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +152/-40 (192 lines); hunks: -14,8 +14,9; -33,6 +34,9; symbols: jit_linear_forward_prefix, MiniMaxText01LinearAttention, mamba_type, get_state_shape
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -14,8 +14,9 @@
+from vllm import envs
-from vllm.config import CacheConfig, VllmConfig
+from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
@@ -33,6 +34,9 @@
+from vllm.model_executor.layers.mamba.abstract import MambaBase
+from vllm.model_executor.layers.mamba.mamba_utils import (
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +152/-40
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/lightning_attn.py`, `vllm/model_executor/layers/mamba/mamba_utils.py`, `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22928 - [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)

- 链接: https://github.com/vllm-project/vllm/pull/22928
- 状态/时间: merged / 2025-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `75531a6c1342`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+467/-87，可读 patch 1435 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax)」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: This PR adds support for customizing the dtype of the state stored for models based on mamba/mamba2/linear attention. It introduces a new CLI argument `mamba_ssm_cache_dtype` wh...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +31/-3 (34 lines); hunks: -16,7 +16,8; -36,7 +37,7; symbols: MiniMaxText01LinearAttention, mamba_type, get_state_dtype, get_state_shape，涉及 `MiniMaxText01LinearAttention, mamba_type, get_state_dtype`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +31/-3 (34 lines); hunks: -16,7 +16,8; -36,7 +37,7; symbols: MiniMaxText01LinearAttention, mamba_type, get_state_dtype, get_state_shape
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -16,7 +16,8 @@
-from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
+from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
+                         get_current_vllm_config)
@@ -36,7 +37,7 @@
-    MambaStateShapeCalculator)
+    MambaStateDtypeCalculator, MambaStateShapeCalculator)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +31/-3
- 验证与风险: diff 自带测试面 `tests/models/language/generation/test_hybrid.py`, `tests/v1/worker/test_gpu_model_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22116 - [Bugfix] Fix broken Minimax-01-VL model

- 链接: https://github.com/vllm-project/vllm/pull/22116
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_vl_01.py`；关联提交 `31fd3265c8b2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+123/-32，可读 patch 258 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix broken Minimax-01-VL model」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_vl_01.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config，涉及 `MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31 (120 lines); hunks: -1,11 +1,13; -17,6 +19,7; symbols: MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, _get_mm_fields_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_vl_01.py
@@ -1,11 +1,13 @@
-from typing import Literal, Optional, TypedDict, Union, cast
+from typing import Annotated, Literal, Optional, Union, cast
+from transformers.models.llava_next.modeling_llava_next import (
+    get_anyres_image_grid_shape, unpad_image)
@@ -17,6 +19,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_vl_01.py` modified +89/-31
- 验证与风险: diff 自带测试面 `tests/models/multimodal/test_tensor_schema.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22589 - [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models

- 链接: https://github.com/vllm-project/vllm/pull/22589
- 状态/时间: merged / 2025-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `dd589322801e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+98/-137，可读 patch 387 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq，涉及 `forward, MiniMaxText01RotaryEmbedding, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +97/-137 (234 lines); hunks: -1,7 +1,6; -19,13 +18,14; symbols: forward, MiniMaxText01RotaryEmbedding, __init__, _compute_inv_freq
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,7 +1,6 @@
-import copy
@@ -19,13 +18,14 @@
+from vllm.compilation.decorators import support_torch_compile
-from vllm.forward_context import get_forward_context
+from vllm.forward_context import ForwardContext, get_forward_context
@@ -43,12 +43,15 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +97/-137
- 验证与风险: runtime 路径改动集中在 `vllm/config/compilation.py`, `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23831 - [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba

- 链接: https://github.com/vllm-project/vllm/pull/23831
- 状态/时间: merged / 2025-08-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `4071c76cf3cf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+448/-410，可读 patch 917 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba」；模型线: MiniMax M2 Series；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: This PR just moves the `MinimaxLinearAttention` layer into the `layers/mamba` directory, to be consistent with the other mamba-like layers (e.g., mamba1, mamba2, short_conv). cc...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader，涉及 `inner_func, MiniMaxText01RMSNormTP, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +6/-410 (416 lines); hunks: -1,45 +1,37; -50,10 +42,7; symbols: inner_func, MiniMaxText01RMSNormTP, __init__, weight_loader
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -1,45 +1,37 @@
-import math
-    from vllm.attention.backends.abstract import AttentionBackend
+    pass
-import torch.nn.functional as F
-from einops import rearrange
-from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +6/-410
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27535 - [Model][MiniMax-M2] Support MiniMax-M2 Model

- 链接: https://github.com/vllm-project/vllm/pull/27535
- 状态/时间: merged / 2025-10-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`；关联提交 `720af6ab7911`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1306/-0，可读 patch 1347 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][MiniMax-M2] Support MiniMax-M2 Model」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`；PR 正文摘要: This PR is intended to support the MiniMax-M2 model inference. It has been tested on a single machine with both 8x96GB and 4x96GB GPU configurations. Deploy Deploy on 8 GPUs Dep...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward，涉及 `MiniMaxM2MoE, __init__, ebias_weight_loader`；`vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser，涉及 `MiniMaxM2ReasoningParser, start_token, end_token`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` added +585/-0; `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27537 - Fix MiniMax-M2 copyright

- 链接: https://github.com/vllm-project/vllm/pull/27537
- 状态/时间: merged / 2025-10-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `5980604c44d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-3，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix MiniMax-M2 copyright」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: Fix MiniMax-M2 copyright.。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +2/-3 (5 lines); hunks: -1,10 +1,9
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -1,10 +1,9 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2025 The MiniMax AI team.
-# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +2/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27627 - Fix MiniMax-M2 rmsnorm precision and remove useless code

- 链接: https://github.com/vllm-project/vllm/pull/27627
- 状态/时间: merged / 2025-10-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `d6704dd099b7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+1/-19，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix MiniMax-M2 rmsnorm precision and remove useless code」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: - Fix rmsnorm precision - Remove useless code。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +0/-18 (18 lines); hunks: -263,23 +263,6 @@ def __init__(; -288,7 +271,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +0/-18
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30384 - [BugFix] Fix minimax m2 model rotary_dim

- 链接: https://github.com/vllm-project/vllm/pull/30384
- 状态/时间: merged / 2025-12-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `d017bceb08ea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix minimax m2 model rotary_dim」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: After #29966, get_rope always reads partial_rotary_factor from the configuration and performs the multiplication again, even if rotary_dim has already been scaled. This leads to...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: -201,7 +201,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -201,7 +201,7 @@ def __init__(
-            rotary_dim=rotary_dim,
+            rotary_dim=self.head_dim,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29882 - [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content.

- 链接: https://github.com/vllm-project/vllm/pull/29882
- 状态/时间: merged / 2025-12-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`；关联提交 `6299628d326f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+468/-0，可读 patch 484 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content.」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `tests/reasoning/test_minimax_m2_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`；PR 正文摘要: Fix `minimax_m2` reasoning parser not separating `content` and `reasoning_content` in streaming mode. Problem MiniMax-M2 models don't generate ` ` start token, only ` ` end toke...。
- 实现要点: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning，涉及 `minimax_m2_tokenizer, test_reasoning`；`tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning，涉及 `minimax_m2_tokenizer, test_reasoning`；`vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser，涉及 `MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming`。
- 代码 diff 细节:
  - `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0 (230 lines); hunks: -0,0 +1,230; symbols: minimax_m2_tokenizer, test_reasoning
  - `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: minimax_m2_tokenizer, test_reasoning
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0 (43 lines); hunks: -19,6 +19,10; -31,6 +35,45 @@ def end_token(self) -> str:; symbols: MiniMaxM2ReasoningParser, end_token, extract_reasoning_streaming, MiniMaxM2AppendThinkReasoningParser
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_minimax_m2_reasoning_parser.py` added +230/-0; `tests/reasoning/test_minimax_m2_append_reasoning_parser.py` added +195/-0
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +43/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_minimax_m2_append_reasoning_parser.py`, `tests/reasoning/test_minimax_m2_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30555 - [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction

- 链接: https://github.com/vllm-project/vllm/pull/30555
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/minimax_m2_tool_parser.py`；关联提交 `20fda431515d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+137/-4，可读 patch 186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `tests/tool_use/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`；PR 正文摘要: Fix https://github.com/vllm-project/vllm/issues/30554. Prevent IndexError in MiniMax M2 tool parser during streaming extraction. Add args str to streamed_args_for_tool field.。
- 实现要点: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser，涉及 `FakeTokenizer, __init__, get_vocab`；`vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming，涉及 `_reset_streaming_state, _extract_name, extract_tool_calls_streaming`。
- 代码 diff 细节:
  - `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4 (22 lines); hunks: -122,6 +122,8 @@ def _reset_streaming_state(self):; -421,9 +423,12 @@ def extract_tool_calls_streaming(; symbols: _reset_streaming_state, _extract_name, extract_tool_calls_streaming
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_use/test_minimax_m2_tool_parser.py` added +119/-0
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +18/-4
- 验证与风险: diff 自带测试面 `tests/tool_use/test_minimax_m2_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31083 - Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs

- 链接: https://github.com/vllm-project/vllm/pull/31083
- 状态/时间: merged / 2025-12-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/minimax_m2_tool_parser.py`；关联提交 `c02a2705f9ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+167/-48，可读 patch 257 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs」；模型线: MiniMax M2 Series；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`；PR 正文摘要: The MiniMax-M2 tool call parser supports `anyOf`, `oneOf`, `allOf`, type arrays, and enum fields. And add MiniMax-M2.1 to the documentation.。
- 实现要点: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types，涉及 `_extract_name, _convert_param_value, _extract_types_from_schema`。
- 代码 diff 细节:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47 (213 lines); hunks: -138,37 +138,167 @@ def _extract_name(self, name_str: str) -> str:; -207,17 +337,11 @@ def _parse_single_invoke(; symbols: _extract_name, _convert_param_value, _extract_types_from_schema, _convert_param_value_with_types
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +166/-47
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/minimax_m2_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31493 - Optimize QKNorm for MiniMax-M2/M2.1

- 链接: https://github.com/vllm-project/vllm/pull/31493
- 状态/时间: merged / 2025-12-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `5bc664110f12`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-2，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize QKNorm for MiniMax-M2/M2.1」；模型线: MiniMax M2 Series；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: Fuses QKNorm in tensor-parallel mode so that only a single all_reduce is launched, reducing communication overhead. Part of #31478. Benchmarks were conducted on MiniMax-M2.1 wit...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +3/-2 (5 lines); hunks: -234,8 +234,9 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -234,8 +234,9 @@ def forward(
-        q = self.q_norm(q)
-        k = self.k_norm(k)
+        q, k = MiniMaxText01RMSNormTP.forward_qk(
+            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
+        )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mamba/linear_attn.py`, `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32342 - Fix optional parameter parsing in MiniMax M2 tool parser #32278

- 链接: https://github.com/vllm-project/vllm/pull/32342
- 状态/时间: merged / 2026-01-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/minimax_m2_tool_parser.py`；关联提交 `19b251fe3d26`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-5，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix optional parameter parsing in MiniMax M2 tool parser #32278」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`；PR 正文摘要: 32278 main branch: "logprobs":null, \"logfile\": \"/tmp/foo\"}"。
- 实现要点: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types，涉及 `_convert_param_value_with_types`。
- 代码 diff 细节:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5 (7 lines); hunks: -217,16 +217,13 @@ def _convert_param_value_with_types(; symbols: _convert_param_value_with_types
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +2/-5
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/minimax_m2_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32763 - feat: Complete LoRA support for MiniMaxM2 Fixes #32736

- 链接: https://github.com/vllm-project/vllm/pull/32763
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `bc0d291bfebf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Complete LoRA support for MiniMaxM2 Fixes #32736」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: FIX #32736 This PR builds upon #31703 by adding the missing components to enable full LoRA support for MiniMaxM2ForCausalLM. Changes: - Add SupportsLoRA interface to MiniMaxM2Fo...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__，涉及 `load_weights, MiniMaxM2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: -59,7 +59,7; -487,7 +487,15 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiniMaxM2ForCausalLM, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35352 - [Bug] Fix missing tag after tool call in MiniMax 2.1

- 链接: https://github.com/vllm-project/vllm/pull/35352
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/reasoning/minimax_m2_reasoning_parser.py`；关联提交 `7fea7250a46c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix missing tag after tool call in MiniMax 2.1」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/reasoning/minimax_m2_reasoning_parser.py`；PR 正文摘要: Fix: #35349 This PR improves the reasoning end detection logic in MiniMaxM2AppendThinkReasoningParser.is_reasoning_end. After a successful tool call, the following line incorrec...。
- 实现要点: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids，涉及 `MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end`。
- 代码 diff 细节:
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1 (7 lines); hunks: -87,10 +87,15 @@ class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):; symbols: MiniMaxM2AppendThinkReasoningParser, __init__, is_reasoning_end, extract_content_ids
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/reasoning/minimax_m2_reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35895 - [Bugfix] Fix minimax_m2 tool parser when stream interval > 1

- 链接: https://github.com/vllm-project/vllm/pull/35895
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `vllm/tool_parsers/minimax_m2_tool_parser.py`；关联提交 `8647c6cf510b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+534/-532，可读 patch 1119 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix minimax_m2 tool parser when stream interval > 1」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/minimax_m2_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`；PR 正文摘要: Fix MiniMax M2 tool parser dropping arguments when stream_interval > 1. Background MiniMax M2 uses XML-based tool call format: The old parser used an incremental state machine t...。
- 实现要点: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state，涉及 `__init__, type, _generate_tool_call_id`；`tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser，涉及 `FakeTokenizer, __init__, get_vocab`；`tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser，涉及 `FakeTokenizer, __init__, get_vocab`。
- 代码 diff 细节:
  - `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413 (503 lines); hunks: -37,37 +37,10 @@ def __init__(self, tokenizer: TokenizerLike):; -103,46 +76,15 @@ def _generate_tool_call_id(self) -> str:; symbols: __init__, type, _generate_tool_call_id, _reset_streaming_state
  - `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: FakeTokenizer, __init__, get_vocab, parser
  - `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119 (119 lines); hunks: -1,119 +0,0; symbols: FakeTokenizer, __init__, get_vocab, minimax_m2_tool_parser
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/minimax_m2_tool_parser.py` modified +90/-413
  - tests: `tests/tool_parsers/test_minimax_m2_tool_parser.py` added +444/-0; `tests/tool_use/test_minimax_m2_tool_parser.py` removed +0/-119
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_use/test_minimax_m2_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- 链接: https://github.com/vllm-project/vllm/pull/37371
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_text_01.py`；关联提交 `17808394bc48`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+235/-219，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_text_01.py`；PR 正文摘要: FIX (partial) #15697 Verified the refactor with a mock-weight loading script using a "Tiny Model Hack" (reducing layers to 1 for fast validation): Mock Weights: Generated fake t...。
- 实现要点: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids，涉及 `replace_weight_name, __init__, _clear_prefill_cache`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_text_01.py` modified +138/-131 (269 lines); hunks: -52,7 +52,12; -494,6 +499,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: replace_weight_name, __init__, _clear_prefill_cache, embed_input_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_text_01.py
@@ -52,7 +52,12 @@
-from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
+from .utils import (
+    AutoWeightsLoader,
+    PPMissingLayer,
+    is_pp_missing_parameter,
+    make_layers,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_text_01.py` modified +138/-131
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37214 - Fix minimax m2.5 nvfp4 kv scales weight loading

- 链接: https://github.com/vllm-project/vllm/pull/37214
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `74056039b776`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-0，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix minimax m2.5 nvfp4 kv scales weight loading」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: Fix kv scale weight loading issue in minimax m2.5 nvfp4:。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +11/-0 (11 lines); hunks: -439,6 +439,17 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/minimax_m2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36965 - [Model][Quantization] Add GGUF support for MiniMax-M2.1

- 链接: https://github.com/vllm-project/vllm/pull/36965
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `63babd17f1b1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+137/-10，可读 patch 238 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Quantization] Add GGUF support for MiniMax-M2.1」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: FIX #28724 Add GGUF loading support for MiniMax-M2.1 (456B MoE, 45.9B active, 256 experts, 8 active per token). This enables serving GGUF-quantized MiniMax-M2.1 checkpoints (e.g...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +5/-2 (7 lines); hunks: -331,7 +331,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -518,7 +518,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/model.py`, `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- 链接: https://github.com/vllm-project/vllm/pull/37512
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/minimax_m2.py`；关联提交 `f6983f01de2b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+24/-5，可读 patch 99 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「MiniMax-M2: add Eagle3 speculative decoding support」；模型线: MiniMax M2 Series；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/minimax_m2.py`；PR 正文摘要: - Add SupportsEagle3 interface to MiniMaxM2ForCausalLM with aux hidden state collection in MiniMaxM2Model.forward() - Register Eagle3MiniMaxM2ForCausalLM in speculative decoding...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__，涉及 `forward, MiniMaxM2Model, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: -24,6 +24,7; -59,7 +60,7; symbols: forward, MiniMaxM2Model, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +16/-5
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37045 - [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels

- 链接: https://github.com/vllm-project/vllm/pull/37045
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/kernels/core/test_minimax_reduce_rms.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `vllm/model_executor/models/minimax_m2.py`；关联提交 `ecd1ea13634e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1861/-4，可读 patch 1936 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Porting the TRTLLM minimax_allreduce_rms kernels」；模型线: MiniMax M2 Series；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/minimax_m2.py`, `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`, `tests/kernels/core/test_minimax_reduce_rms.py`；PR 正文摘要: See: https://github.com/NVIDIA/TensorRT-LLM/pull/12163 Plan Accuracy Verification(https://github.com/vllm-project/vllm/pull/37045/commits/69f231c7b211a9088bf591679fcdced35d4e919...。
- 实现要点: `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: _minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern, __init__，涉及 `_minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern`；`tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk，涉及 `_worker_forward_qk, test_minimax_reduce_rms_qk`。
- 代码 diff 细节:
  - `vllm/model_executor/models/minimax_m2.py` modified +1/-3 (4 lines); hunks: -233,9 +233,7 @@ def forward(; symbols: forward
  - `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0 (340 lines); hunks: -0,0 +1,340; symbols: _minimax_qk_norm_fused, _minimax_qk_norm_fused_fake, MiniMaxQKNormPattern, __init__
  - `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: _worker_forward_qk, test_minimax_reduce_rms_qk
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/minimax_m2.py
@@ -233,9 +233,7 @@ def forward(
-        q, k = MiniMaxText01RMSNormTP.forward_qk(
-            self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
-        )
+        q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
diff -- vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py
@@ -0,0 +1,340 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Fusion pass: replace MiniMax QK allreduce + RMS norm with the Lamport
+fused kernel (minimax_allreduce_rms_qk) for decode-size batches.
+Pattern (inlined forward_qk in compiled graph):
diff -- tests/kernels/core/test_minimax_reduce_rms.py
@@ -0,0 +1,152 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- 已读文件:
  - runtime: `vllm/model_executor/models/minimax_m2.py` modified +1/-3; `vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py` added +340/-0
  - tests: `tests/kernels/core/test_minimax_reduce_rms.py` added +152/-0
- 验证与风险: diff 自带测试面 `tests/kernels/core/test_minimax_reduce_rms.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39683 - [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter

- 链接: https://github.com/vllm-project/vllm/pull/39683
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/parser/minimax_m2_parser.py`；关联提交 `d2130a47bb97`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-2，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: Fix MinimaxM2ToolParser missing tools parameter」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/parser/minimax_m2_parser.py`；PR 正文摘要: [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter before。
- 实现要点: `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__，涉及 `MiniMaxM2Parser, __init__`。
- 代码 diff 细节:
  - `vllm/parser/minimax_m2_parser.py` modified +5/-2 (7 lines); hunks: -13,6 +13,9; -40,12 +43,12 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/parser/minimax_m2_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39861 - [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()

- 链接: https://github.com/vllm-project/vllm/pull/39861
- 状态/时间: merged / 2026-04-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/parser/minimax_m2_parser.py`；关联提交 `8d7c9628337a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-3，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__()」；模型线: MiniMax M2 Series；类别: 缺陷修复；主要 diff: `vllm/parser/minimax_m2_parser.py`；PR 正文摘要: Fixes #39847. `MiniMaxM2Parser.__init__()` does not accept `**kwargs`, causing a `TypeError` on every streaming chat completion request when serving MiniMax M2 models: The error...。
- 实现要点: `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__，涉及 `MiniMaxM2Parser, __init__`。
- 代码 diff 细节:
  - `vllm/parser/minimax_m2_parser.py` modified +9/-3 (12 lines); hunks: -43,11 +43,17 @@ class MiniMaxM2Parser(DelegatingParser):; symbols: MiniMaxM2Parser, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/parser/minimax_m2_parser.py` modified +9/-3
- 验证与风险: runtime 路径改动集中在 `vllm/parser/minimax_m2_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
