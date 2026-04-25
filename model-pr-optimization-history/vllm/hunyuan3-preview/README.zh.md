# vllm Hunyuan3 Preview 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `tests/tool_parsers/test_hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/model_executor/models/hy_v3_mtp.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/reasoning/hy_v3_reasoning_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/tool_parsers/hy_v3_tool_parser.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |
| `vllm/transformers_utils/configs/hy_v3.py` | [#40681](https://github.com/vllm-project/vllm/pull/40681) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-23 | [#21368](https://github.com/vllm-project/vllm/pull/21368) | merged | [Model] add Hunyuan V1 Dense Model support. | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py` |
| 2025-11-25 | [#29327](https://github.com/vllm-project/vllm/pull/29327) | merged | [Model] Add HunyuanOCR support | `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py` |
| 2026-01-27 | [#33035](https://github.com/vllm-project/vllm/pull/33035) | merged | feature: support eagle3 for HunyuanVL & Hunyuan | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`, `vllm/v1/spec_decode/eagle.py` |
| 2026-04-23 | [#40681](https://github.com/vllm-project/vllm/pull/40681) | merged | [Model] Support Hy3 preview | `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py` |

## 逐 PR diff 审计卡

### PR #21368 - [Model] add Hunyuan V1 Dense Model support.

- 链接: https://github.com/vllm-project/vllm/pull/21368
- 状态/时间: merged / 2025-07-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+57/-19，可读 patch 136 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] add Hunyuan V1 Dense Model support.」；模型线: Hunyuan3 Preview；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunks: -61,6 +61,19; -140,8 +153,8 @@ def __init__(; symbols: _is_moe, _get_cla_factor, __init__, _split_qkv_weight，涉及 `_is_moe, _get_cla_factor, __init__`；`vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunks: -79,7 +79,8；`tests/models/registry.py` modified +2/-0 (2 lines); hunks: -199,6 +199,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -363,6 +363,7 @@ th {。
- 代码 diff 细节:
  - `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunks: -61,6 +61,19; -140,8 +153,8 @@ def __init__(; symbols: _is_moe, _get_cla_factor, __init__, _split_qkv_weight
  - `vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunks: -79,7 +79,8
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -199,6 +199,8 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -363,6 +363,7 @@ th {
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -61,6 +61,19 @@
+def _is_moe(config: PretrainedConfig) -> bool:
+    num_experts = getattr(config, "num_experts", None)
+    if isinstance(num_experts, int):
+        return num_experts > 1
+    if isinstance(num_experts, list) and num_experts:
+        # Ensure all elements are integers before calling max.
diff -- vllm/model_executor/models/registry.py
@@ -79,7 +79,8 @@
-    "HunYuanMoEV1ForCausalLM": ("hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM"),
+    "HunYuanMoEV1ForCausalLM": ("hunyuan_v1", "HunYuanMoEV1ForCausalLM"),
+    "HunYuanDenseV1ForCausalLM": ("hunyuan_v1", "HunYuanDenseV1ForCausalLM"),
diff -- tests/models/registry.py
@@ -199,6 +199,8 @@ def check_available_online(
+    "HunYuanDenseV1ForCausalLM":_HfExamplesInfo("tencent/Hunyuan-7B-Instruct-0124",
+                                               trust_remote_code=True),
diff -- docs/models/supported_models.md
```

- 已读文件:
  - runtime: `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18; `vllm/model_executor/models/registry.py` modified +2/-1
  - tests: `tests/models/registry.py` modified +2/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29327 - [Model] Add HunyuanOCR support

- 链接: https://github.com/vllm-project/vllm/pull/29327
- 状态/时间: merged / 2025-11-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+2415/-4，可读 patch 2653 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add HunyuanOCR support」；模型线: Hunyuan3 Preview；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`；PR 正文摘要: - Add HunyuanOCR support。
- 实现要点: `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunks: -0,0 +1,1028; symbols: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__，涉及 `HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP`；`vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunks: -0,0 +1,477; symbols: for, smart_resize, HunYuanVLImageProcessor, __init__，涉及 `for, smart_resize, HunYuanVLImageProcessor`；`vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to，涉及 `HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig`；`vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunks: -0,0 +1,233; symbols: HunYuanVLProcessor, __init__, __call__, batch_decode，涉及 `HunYuanVLProcessor, __init__, __call__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunks: -0,0 +1,1028; symbols: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__
  - `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunks: -0,0 +1,477; symbols: for, smart_resize, HunYuanVLImageProcessor, __init__
  - `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to
  - `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunks: -0,0 +1,233; symbols: HunYuanVLProcessor, __init__, __call__, batch_decode
  - `vllm/model_executor/layers/rotary_embedding/xdrope.py` added +102/-0 (102 lines); hunks: -0,0 +1,102; symbols: XDRotaryEmbedding, __init__, forward, get_next_input_positions
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/hunyuan_vision.py
@@ -0,0 +1,1028 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# coding=utf-8
+# Copyright 2025 The HunYuan team.
+# Copyright 2025 The vLLM team.
+# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/transformers_utils/processors/hunyuan_vl_image.py
@@ -0,0 +1,477 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://github.com/ManaEstras/transformers/blob/v4.57.1.hyvl/src/transformers/models/hunyuan_vl/image_processing_hunyuan_vl.py
+"""Image processor class for HunYuanVL."""
+# isort conflicts with ruff for transformers imports
+# isort: skip_file
diff -- vllm/transformers_utils/configs/hunyuan_vl.py
@@ -0,0 +1,322 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0; `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0; `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0; `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0; `vllm/model_executor/layers/rotary_embedding/xdrope.py` added +102/-0; `vllm/model_executor/models/interfaces.py` modified +50/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33035 - feature: support eagle3 for HunyuanVL & Hunyuan

- 链接: https://github.com/vllm-project/vllm/pull/33035
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+49/-3，可读 patch 165 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feature: support eagle3 for HunyuanVL & Hunyuan」；模型线: Hunyuan3 Preview；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`, `vllm/v1/spec_decode/eagle.py`；PR 正文摘要: Eagle3 models: https://huggingface.co/collections/AngelSlim/eagle3 详细结果见：AngelSlim Speculative-Decoding HunyuanOCR Model | Model | Method | OmniDocBench | | |-------------|-----...。
- 实现要点: `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunks: -66,7 +66,7; -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, forward, _split_qkv_weight，涉及 `__init__, embed_input_ids, forward`；`vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunks: -83,6 +83,7; -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(; symbols: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers，涉及 `HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers`；`vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunks: -115,6 +115,8 @@ def __init__(; -129,6 +131,12 @@ def __init__(; symbols: __init__, _get_positions, _set_positions，涉及 `__init__, _get_positions, _set_positions`；`vllm/config/speculative.py` modified +8/-1 (9 lines); hunks: -675,7 +675,14 @@ def _verify_args(self) -> Self:; symbols: _verify_args，涉及 `_verify_args`。
- 代码 diff 细节:
  - `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunks: -66,7 +66,7; -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, forward, _split_qkv_weight
  - `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunks: -83,6 +83,7; -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(; symbols: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers
  - `vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunks: -115,6 +115,8 @@ def __init__(; -129,6 +131,12 @@ def __init__(; symbols: __init__, _get_positions, _set_positions
  - `vllm/config/speculative.py` modified +8/-1 (9 lines); hunks: -675,7 +675,14 @@ def _verify_args(self) -> Self:; symbols: _verify_args
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -66,7 +66,7 @@
-from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
+from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -630,6 +630,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.aux_hidden_state_layers = tuple[int, ...]()
@@ -654,9 +655,13 @@ def forward(
+        aux_hidden_states = []
diff -- vllm/model_executor/models/hunyuan_vision.py
@@ -83,6 +83,7 @@
+    SupportsEagle3,
@@ -780,6 +781,7 @@ class HunYuanVLForConditionalGeneration(
+    SupportsEagle3,
@@ -966,6 +968,13 @@ def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
+    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        self.language_model.model.aux_hidden_state_layers = layers
diff -- vllm/v1/spec_decode/eagle.py
@@ -115,6 +115,8 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2; `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0; `vllm/v1/spec_decode/eagle.py` modified +15/-0; `vllm/config/speculative.py` modified +8/-1
- 验证与风险: runtime 路径改动集中在 `vllm/config/speculative.py`, `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/hunyuan_vision.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40681 - [Model] Support Hy3 preview

- 链接: https://github.com/vllm-project/vllm/pull/40681
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3.py`, `vllm/model_executor/models/hy_v3_mtp.py`, `vllm/reasoning/hy_v3_reasoning_parser.py` 等 7 个文件；关联提交 `d0009ddb0b96`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2696/-0，可读 patch 2801 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Hy3 preview」；模型线: Hunyuan3 Preview；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/hy_v3.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`, `vllm/model_executor/models/hy_v3_mtp.py`；PR 正文摘要: Support Hy3-preview model Test model, reasoning parser and tool parser. All pass. Hy3-preview model Hy3 preview is a Mixture-of-Experts model with integrated fast and slow think...。
- 实现要点: `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused，涉及 `HYV3FeedForward, __init__, forward`；`vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options，涉及 `HYV3ToolParser, _normalize_type, _get_arg_schema`；`vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__，涉及 `_is_moe, _get_cla_factor, HYV3SharedHead`；`tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls，涉及 `hy_v3_tokenizer, hy_v3_tool_parser, mock_request`。
- 代码 diff 细节:
  - `vllm/model_executor/models/hy_v3.py` added +707/-0 (707 lines); hunks: -0,0 +1,707; symbols: HYV3FeedForward, __init__, forward, HYV3MoEFused
  - `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0 (645 lines); hunks: -0,0 +1,645; symbols: HYV3ToolParser, _normalize_type, _get_arg_schema, _get_schema_options
  - `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0 (470 lines); hunks: -0,0 +1,470; symbols: _is_moe, _get_cla_factor, HYV3SharedHead, __init__
  - `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0 (274 lines); hunks: -0,0 +1,274; symbols: hy_v3_tokenizer, hy_v3_tool_parser, mock_request, TestHYV3ExtractToolCalls
  - `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: hy_v3_tokenizer, test_reasoning, test_is_reasoning_end_full_prompt
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/hy_v3.py
@@ -0,0 +1,707 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# coding=utf-8
+# Copyright 2026 The HY team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -0,0 +1,645 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import ast
+import json
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/model_executor/models/hy_v3_mtp.py
@@ -0,0 +1,470 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/hy_v3.py` added +707/-0; `vllm/tool_parsers/hy_v3_tool_parser.py` added +645/-0; `vllm/model_executor/models/hy_v3_mtp.py` added +470/-0; `vllm/transformers_utils/configs/hy_v3.py` added +185/-0; `vllm/reasoning/hy_v3_reasoning_parser.py` added +137/-0
  - tests: `tests/tool_parsers/test_hy_v3_tool_parser.py` added +274/-0; `tests/reasoning/test_hy_v3_reasoning_parser.py` added +243/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/reasoning/test_hy_v3_reasoning_parser.py`, `tests/tool_parsers/test_hy_v3_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
