# vllm Intern-S1 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_internlm2_tool.jinja` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_internlm2_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/internlm2.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/internlm2_ve.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/interns1.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#21671](https://github.com/vllm-project/vllm/pull/21671), [#22417](https://github.com/vllm-project/vllm/pull/22417), [#23510](https://github.com/vllm-project/vllm/pull/23510), [#25644](https://github.com/vllm-project/vllm/pull/25644) |
| `vllm/model_executor/models/interns1_pro.py` | [#33636](https://github.com/vllm-project/vllm/pull/33636), [#33793](https://github.com/vllm-project/vllm/pull/33793) |
| `vllm/model_executor/models/interns1_vit.py` | [#21628](https://github.com/vllm-project/vllm/pull/21628), [#27480](https://github.com/vllm-project/vllm/pull/27480) |
| `vllm/tool_parsers/internlm2_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-26 | [#21628](https://github.com/vllm-project/vllm/pull/21628) | merged | Support Intern-S1 | `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py` |
| 2025-07-27 | [#21671](https://github.com/vllm-project/vllm/pull/21671) | merged | [VLM] Add video support for Intern-S1 | `vllm/model_executor/models/interns1.py` |
| 2025-08-07 | [#22417](https://github.com/vllm-project/vllm/pull/22417) | merged | [Bugfix] Fix wrong method name in Intern-S1 image processor | `vllm/model_executor/models/interns1.py` |
| 2025-09-02 | [#23510](https://github.com/vllm-project/vllm/pull/23510) | merged | Migrate Interns1 inputs to TensorSchema | `vllm/model_executor/models/interns1.py` |
| 2025-09-25 | [#25644](https://github.com/vllm-project/vllm/pull/25644) | merged | [Bugfix] Fix InternS1 video processing after Transformers v4.56 | `vllm/model_executor/models/interns1.py` |
| 2025-10-24 | [#27480](https://github.com/vllm-project/vllm/pull/27480) | merged | [Bugfix] Fix interns1-vit qk norm code path | `vllm/model_executor/models/interns1_vit.py` |
| 2026-02-03 | [#33636](https://github.com/vllm-project/vllm/pull/33636) | merged | [Models] Intern-S1-Pro | `vllm/model_executor/models/interns1_pro.py` |
| 2026-02-04 | [#33793](https://github.com/vllm-project/vllm/pull/33793) | merged | [Bugfix] Fix interns1-pro initialization and PP | `vllm/model_executor/models/interns1_pro.py` |

## 逐 PR diff 审计卡

### PR #21628 - Support Intern-S1

- 链接: https://github.com/vllm-project/vllm/pull/21628
- 状态/时间: merged / 2025-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`；关联提交 `875af38e0121`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+1196/-0，可读 patch 1247 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Intern-S1」；模型线: Intern-S1；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/interns1.py`, `vllm/model_executor/models/interns1_vit.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunks: -0,0 +1,711; symbols: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs，涉及 `InternS1MultiModalProjector, __init__, forward`；`vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings，涉及 `InternS1VisionPatchEmbeddings, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1.py` added +711/-0 (711 lines); hunks: -0,0 +1,711; symbols: InternS1MultiModalProjector, __init__, forward, InternS1ImagePixelInputs
  - `vllm/model_executor/models/interns1_vit.py` added +421/-0 (421 lines); hunks: -0,0 +1,421; symbols: InternS1VisionPatchEmbeddings, __init__, forward, InternS1VisionEmbeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -0,0 +1,711 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# --------------------------------------------------------
+# InternS1
+# Copyright (c) 2025 Shanghai AI Lab
+# Licensed under The MIT License [see LICENSE for details]
diff -- vllm/model_executor/models/interns1_vit.py
@@ -0,0 +1,421 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1.py` added +711/-0; `vllm/model_executor/models/interns1_vit.py` added +421/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21671 - [VLM] Add video support for Intern-S1

- 链接: https://github.com/vllm-project/vllm/pull/21671
- 状态/时间: merged / 2025-07-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1.py`；关联提交 `3d847a3125cd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+173/-50，可读 patch 375 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Add video support for Intern-S1」；模型线: Intern-S1；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/interns1.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunks: -9,9 +9,10; -139,13 +140,13 @@ def get_interns1_target_ratios(; symbols: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits，涉及 `get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1.py` modified +166/-45 (211 lines); hunks: -9,9 +9,10; -139,13 +140,13 @@ def get_interns1_target_ratios(; symbols: get_interns1_target_ratios, InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -9,9 +9,10 @@
+import regex as re
-from transformers import InternVLProcessor, PretrainedConfig
+from transformers import BatchFeature, InternVLProcessor, PretrainedConfig
@@ -139,13 +140,13 @@ def get_interns1_target_ratios(
-    """Basic image-only ProcessingInfo for InternS1-style models."""
+    """ProcessingInfo for InternS1-style models."""
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1.py` modified +166/-45
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22417 - [Bugfix] Fix wrong method name in Intern-S1 image processor

- 链接: https://github.com/vllm-project/vllm/pull/22417
- 状态/时间: merged / 2025-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1.py`；关联提交 `04cf435d95fe`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix wrong method name in Intern-S1 image processor」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/interns1.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens，涉及 `get_num_image_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1.py` modified +1/-1 (2 lines); hunks: -161,7 +161,7 @@ def get_num_image_tokens(; symbols: get_num_image_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -161,7 +161,7 @@ def get_num_image_tokens(
-        num_image_patches = processor.get_number_of_image_tokens(
+        num_image_patches = processor.get_number_of_image_patches(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/interns1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23510 - Migrate Interns1 inputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/23510
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1.py`；关联提交 `56d04089ef50`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+50/-51，可读 patch 167 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate Interns1 inputs to TensorSchema」；模型线: Intern-S1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/interns1.py`；PR 正文摘要: This PR migrates Interns1 inputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings it in line with recent changes t...。
- 实现要点: `vllm/model_executor/models/interns1.py` modified +50/-51 (101 lines); hunks: -7,7 +7,7; -32,6 +32,7; symbols: forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs，涉及 `forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1.py` modified +50/-51 (101 lines); hunks: -7,7 +7,7; -32,6 +32,7; symbols: forward, InternS1ImagePixelInputs, InternS1ImageEmbeddingInputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -7,7 +7,7 @@
-from typing import Literal, Optional, TypedDict, Union
+from typing import Annotated, Literal, Optional, Union
@@ -32,6 +32,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -62,51 +63,60 @@ def forward(self, image_features):
-class InternS1ImagePixelInputs(TypedDict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1.py` modified +50/-51
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/interns1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25644 - [Bugfix] Fix InternS1 video processing after Transformers v4.56

- 链接: https://github.com/vllm-project/vllm/pull/25644
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1.py`；关联提交 `03858e6d1c85`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+68/-3，可读 patch 128 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix InternS1 video processing after Transformers v4.56」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/interns1.py`；PR 正文摘要: - FIX #25451。
- 实现要点: `vllm/model_executor/models/interns1.py` modified +10/-1 (11 lines); hunks: -16,6 +16,8; -31,6 +33,8; symbols: InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits，涉及 `InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1.py` modified +10/-1 (11 lines); hunks: -16,6 +16,8; -31,6 +33,8; symbols: InternS1ProcessingInfo, get_hf_processor, get_supported_mm_limits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1.py
@@ -16,6 +16,8 @@
+from transformers.models.internvl.video_processing_internvl import (
+    InternVLVideoProcessor)
@@ -31,6 +33,8 @@
+from vllm.transformers_utils.processor import (
+    cached_video_processor_from_config)
@@ -152,7 +156,12 @@ class InternS1ProcessingInfo(BaseProcessingInfo):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1.py` modified +10/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27480 - [Bugfix] Fix interns1-vit qk norm code path

- 链接: https://github.com/vllm-project/vllm/pull/27480
- 状态/时间: merged / 2025-10-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1_vit.py`；关联提交 `acc78aeb88c8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-4，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix interns1-vit qk norm code path」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/interns1_vit.py`；PR 正文摘要: - Fix https://github.com/InternLM/Intern-S1/issues/29。
- 实现要点: `vllm/model_executor/models/interns1_vit.py` modified +3/-4 (7 lines); hunks: -217,16 +217,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1_vit.py` modified +3/-4 (7 lines); hunks: -217,16 +217,15 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1_vit.py
@@ -217,16 +217,15 @@ def __init__(
-        B, N, C = x.shape
+        """x shape: (B, N, C)"""
-            B_, N_, H_, D_ = q.shape
-            q = self.q_norm(q.flatten(-2, -1)).view(B_, N_, H_, D_)
-            k = self.k_norm(k.flatten(-2, -1)).view(B_, N_, H_, D_)
+            q = self.q_norm(q)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1_vit.py` modified +3/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/interns1_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33636 - [Models] Intern-S1-Pro

- 链接: https://github.com/vllm-project/vllm/pull/33636
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1_pro.py`；关联提交 `a3acfa10719a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+942/-11，可读 patch 1062 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Intern-S1-Pro」；模型线: Intern-S1；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/interns1_pro.py`；PR 正文摘要: Intern-S1-Pro model support.。
- 实现要点: `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunks: -0,0 +1,633; symbols: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP，涉及 `InternS1ProProcessingInfo, get_hf_config, get_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1_pro.py` added +633/-0 (633 lines); hunks: -0,0 +1,633; symbols: InternS1ProProcessingInfo, get_hf_config, get_hf_processor, InternS1ProMoeMLP
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1_pro.py
@@ -0,0 +1,633 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The vLLM team.
+# Copyright 2025 The Qwen Team.
+# Copyright 2025 The HuggingFace Inc. team.
+# All rights reserved.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1_pro.py` added +633/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33793 - [Bugfix] Fix interns1-pro initialization and PP

- 链接: https://github.com/vllm-project/vllm/pull/33793
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/interns1_pro.py`；关联提交 `192ad4648b20`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+43/-22，可读 patch 163 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix interns1-pro initialization and PP」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/interns1_pro.py`；PR 正文摘要: - Fix broken InternS1-PRO intialization because the previous PR is drafted on an old version vLLM.  - Also fix PP for InternS1-PRO。
- 实现要点: `vllm/model_executor/models/interns1_pro.py` modified +26/-12 (38 lines); hunks: -32,7 +32,6; -41,8 +40,8; symbols: __init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration，涉及 `__init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interns1_pro.py` modified +26/-12 (38 lines); hunks: -32,7 +32,6; -41,8 +40,8; symbols: __init__, InternS1ProMoeLLMForCausalLM, InternS1ProForConditionalGeneration
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interns1_pro.py
@@ -32,7 +32,6 @@
-from vllm.attention.layer import Attention
@@ -41,8 +40,8 @@
+from vllm.model_executor.layers.attention import Attention
-from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
@@ -188,7 +187,6 @@ def __init__(
-            routing_method_type=RoutingMethodType.Renormalize,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interns1_pro.py` modified +26/-12
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_tensor_schema.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
