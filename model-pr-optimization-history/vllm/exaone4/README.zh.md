# vLLM EXAONE 4/4.5/K-EXAONE 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/exaone4.py` | [#21060](https://github.com/vllm-project/vllm/pull/21060), [#23918](https://github.com/vllm-project/vllm/pull/23918), [#31621](https://github.com/vllm-project/vllm/pull/31621), [#39388](https://github.com/vllm-project/vllm/pull/39388), [#50524](https://github.com/vllm-project/vllm/pull/50524) |
| `vllm/model_executor/models/exaone4_5.py` | [#39388](https://github.com/vllm-project/vllm/pull/39388), [#42246](https://github.com/vllm-project/vllm/pull/42246), [#45073](https://github.com/vllm-project/vllm/pull/45073) |
| `vllm/model_executor/models/exaone4_5_mtp.py` | [#39388](https://github.com/vllm-project/vllm/pull/39388), [#39526](https://github.com/vllm-project/vllm/pull/39526), [#42246](https://github.com/vllm-project/vllm/pull/42246) |
| `vllm/model_executor/models/exaone_moe.py` | [#31621](https://github.com/vllm-project/vllm/pull/31621), [#32196](https://github.com/vllm-project/vllm/pull/32196), [#50524](https://github.com/vllm-project/vllm/pull/50524) |
| `vllm/model_executor/models/exaone_moe_mtp.py` | [#31621](https://github.com/vllm-project/vllm/pull/31621), [#39388](https://github.com/vllm-project/vllm/pull/39388), [#50524](https://github.com/vllm-project/vllm/pull/50524) |

## PR 覆盖总览

- git 追溯 PR 数: 9
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 9
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-19 | [#21060](https://github.com/vllm-project/vllm/pull/21060) | merged | [Model] EXAONE 4.0 model support | `vllm/model_executor/models/exaone4.py`, `vllm/transformers_utils/configs/exaone4.py` |
| 2025-09-02 | [#23918](https://github.com/vllm-project/vllm/pull/23918) | merged | [BugFix] Fix EXAONE4 rotary embeddings | `vllm/model_executor/models/exaone4.py` |
| 2026-01-12 | [#31621](https://github.com/vllm-project/vllm/pull/31621) | merged | Add K-EXAONE-236B-A23B | `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`, `vllm/model_executor/models/exaone4.py` |
| 2026-01-12 | [#32196](https://github.com/vllm-project/vllm/pull/32196) | merged | [BugFix] fix FusedMoE.make_expert_params_mapping in EXAONE-MoE | `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-10 | [#39388](https://github.com/vllm-project/vllm/pull/39388) | merged | Add EXAONE-4.5 | `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone_moe_mtp.py` |
| 2026-04-11 | [#39526](https://github.com/vllm-project/vllm/pull/39526) | merged | [Bugfix] add SupportsMultiModal to Exaone4_5_MTP | `vllm/model_executor/models/exaone4_5_mtp.py` |
| 2026-05-11 | [#42246](https://github.com/vllm-project/vllm/pull/42246) | merged | Fix EXAONE-4.5 to align with Transformers update | `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone4_5.py` |
| 2026-06-10 | [#45073](https://github.com/vllm-project/vllm/pull/45073) | merged | [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder | `vllm/model_executor/models/exaone4_5.py` |
| 2026-08-03 | [#50524](https://github.com/vllm-project/vllm/pull/50524) | merged | [Model] Add K-EXAONE-2.0-750B-A37B | `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe_mtp.py` |

## 逐 PR diff 审计卡

### PR #21060 - [Model] EXAONE 4.0 model support

- 链接: https://github.com/vllm-project/vllm/pull/21060
- 状态/时间: merged / 2025-07-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4.py`；关联提交 `3e04107d97ae`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+809/-3，可读 patch 863 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4.py` added +547/-0 (547 lines); hunks: -0,0 +1,547; symbols: Exaone4GatedMLP, __init__, forward, Exaone4Attention，涉及 `Exaone4GatedMLP, __init__, forward`；`vllm/transformers_utils/configs/exaone4.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: check_is_sliding, Exaone4Config, to, __init__，涉及 `check_is_sliding, Exaone4Config, to`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4.py` added +547/-0 (547 lines); hunks: -0,0 +1,547; symbols: Exaone4GatedMLP, __init__, forward, Exaone4Attention
  - `vllm/transformers_utils/configs/exaone4.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: check_is_sliding, Exaone4Config, to, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4.py
@@ -0,0 +1,547 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Adapted from
+# https://github.com/lgai-exaone/transformers/blob/add-exaone4/src/transformers/models/exaone4/modeling_exaone4.py
+# Copyright 2025 The LG CNS Gen AI Solution Delivery Team.
diff -- vllm/transformers_utils/configs/exaone4.py
@@ -0,0 +1,252 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Copied from
+# https://github.com/lgai-exaone/transformers/blob/add-exaone4/src/transformers/models/exaone4/configuration_exaone4.py
+# Copyright 2025 The LG CNS Gen AI Solution Delivery Team.
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4.py` added +547/-0; `vllm/transformers_utils/configs/exaone4.py` added +252/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23918 - [BugFix] Fix EXAONE4 rotary embeddings

- 链接: https://github.com/vllm-project/vllm/pull/23918
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4.py`；关联提交 `38ba061f6f44`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 20 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4.py` modified +3/-3 (6 lines); hunks: -164,8 +164,8 @@ def __init__(; -201,7 +201,7 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4.py` modified +3/-3 (6 lines); hunks: -164,8 +164,8 @@ def __init__(; -201,7 +201,7 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4.py
@@ -164,8 +164,8 @@ def __init__(
-        # apply rotary embeddings to every layer
-        self.apply_all_layers = not is_sliding
+        # apply rotary embeddings to every layer in full attention models
+        self.apply_rope_all_layers = "sliding_attention" not in config.layer_types
@@ -201,7 +201,7 @@ def forward(
-        if self.sliding_window or self.apply_all_layers:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/exaone4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31621 - Add K-EXAONE-236B-A23B

- 链接: https://github.com/vllm-project/vllm/pull/31621
- 状态/时间: merged / 2026-01-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`；关联提交 `63ed2409e8cf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+856/-0，可读 patch 921 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone_moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: ExaoneMoe, __init__, forward, ExaoneMoeDecoderLayer，涉及 `ExaoneMoe, __init__, forward`；`vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings, forward，涉及 `ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings`；`vllm/model_executor/models/exaone4.py` modified +2/-0 (2 lines); hunks: -72,6 +72,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone_moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: ExaoneMoe, __init__, forward, ExaoneMoeDecoderLayer
  - `vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: ExaoneMoeMultiTokenPredictor, __init__, get_input_embeddings, forward
  - `vllm/model_executor/models/exaone4.py` modified +2/-0 (2 lines); hunks: -72,6 +72,7 @@ def __init__(; -88,6 +89,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -0,0 +1,578 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -0,0 +1,255 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only ExaoneMoe MTP model."""
+from collections.abc import Iterable
+import torch
+from torch import nn
diff -- vllm/model_executor/models/exaone4.py
@@ -72,6 +72,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone_moe.py` added +578/-0; `vllm/model_executor/models/exaone_moe_mtp.py` added +255/-0; `vllm/model_executor/models/exaone4.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32196 - [BugFix] fix FusedMoE.make_expert_params_mapping in EXAONE-MoE

- 链接: https://github.com/vllm-project/vllm/pull/32196
- 状态/时间: merged / 2026-01-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone_moe.py`；关联提交 `3d962d72ab01`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone_moe.py` modified +1/-0 (1 lines); hunks: -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping，涉及 `get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone_moe.py` modified +1/-0 (1 lines); hunks: -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -338,6 +338,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/exaone_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39388 - Add EXAONE-4.5

- 链接: https://github.com/vllm-project/vllm/pull/39388
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`, `vllm/model_executor/models/exaone_moe_mtp.py`；关联提交 `e7a1387e7380`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+600/-10，可读 patch 727 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4_5.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: EXAONE4_5_VisionAttention, __init__, split_qkv, forward，涉及 `EXAONE4_5_VisionAttention, __init__, split_qkv`；`vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP, load_weights，涉及 `Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP`；`vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5 (5 lines); hunks: -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):; symbols: ExaoneMoeMTP, __init__，涉及 `ExaoneMoeMTP, __init__`；`vllm/model_executor/models/exaone4.py` modified +3/-0 (3 lines); hunks: -75,6 +75,7 @@ def __init__(; -83,6 +84,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4_5.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: EXAONE4_5_VisionAttention, __init__, split_qkv, forward
  - `vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: Exaone4_5MultiTokenPredictor, __init__, Exaone4_5_MTP, load_weights
  - `vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5 (5 lines); hunks: -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):; symbols: ExaoneMoeMTP, __init__
  - `vllm/model_executor/models/exaone4.py` modified +3/-0 (3 lines); hunks: -75,6 +75,7 @@ def __init__(; -83,6 +84,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4_5.py
@@ -0,0 +1,366 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -0,0 +1,129 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only EXAONE-4_5 MTP model."""
+from collections.abc import Iterable
+import torch
+from torch import nn
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -184,11 +184,6 @@ class ExaoneMoeMTP(nn.Module):
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4_5.py` added +366/-0; `vllm/model_executor/models/exaone4_5_mtp.py` added +129/-0; `vllm/model_executor/models/exaone_moe_mtp.py` modified +0/-5; `vllm/model_executor/models/exaone4.py` modified +3/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39526 - [Bugfix] add SupportsMultiModal to Exaone4_5_MTP

- 链接: https://github.com/vllm-project/vllm/pull/39526
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4_5_mtp.py`；关联提交 `da72daced2a8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+36/-1，可读 patch 62 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1 (37 lines); hunks: -23,8 +23,14; -85,9 +91,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, Exaone4_5_MTP，涉及 `__init__, embed_input_ids, Exaone4_5_MTP`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1 (37 lines); hunks: -23,8 +23,14; -85,9 +91,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, embed_input_ids, Exaone4_5_MTP
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -23,8 +23,14 @@
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsMultiModal,
+    _require_is_multimodal,
+)
+    _merge_multimodal_embeddings,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4_5_mtp.py` modified +36/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/exaone4_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42246 - Fix EXAONE-4.5 to align with Transformers update

- 链接: https://github.com/vllm-project/vllm/pull/42246
- 状态/时间: merged / 2026-05-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`；关联提交 `27ae67636479`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+53/-13，可读 patch 147 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9 (62 lines); hunks: -9,6 +9,7; -22,6 +23,7; symbols: __init__, embed_input_ids, forward，涉及 `__init__, embed_input_ids, forward`；`vllm/model_executor/models/exaone4_5.py` modified +0/-4 (4 lines); hunks: -23,7 +23,6; -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Pr...; symbols: get_hf_processor, get_image_processor，涉及 `get_hf_processor, get_image_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9 (62 lines); hunks: -9,6 +9,7; -22,6 +23,7; symbols: __init__, embed_input_ids, forward
  - `vllm/model_executor/models/exaone4_5.py` modified +0/-4 (4 lines); hunks: -23,7 +23,6; -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Pr...; symbols: get_hf_processor, get_image_processor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4_5_mtp.py
@@ -9,6 +9,7 @@
+from vllm.distributed.parallel_state import get_pp_group
@@ -22,6 +23,7 @@
+from vllm.sequence import IntermediateTensors
@@ -48,6 +50,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        text_config = config.text_config
@@ -58,18 +61,18 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
diff -- vllm/model_executor/models/exaone4_5.py
@@ -23,7 +23,6 @@
-    Exaone4_5_ImageProcessor,
@@ -304,9 +303,6 @@ def get_hf_processor(self, **kwargs: object) -> Exaone4_5_Processor:
-    def get_image_processor(self, **kwargs: object) -> Exaone4_5_ImageProcessor:
-        return Exaone4_5_ImageProcessor(**kwargs)
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4_5_mtp.py` modified +53/-9; `vllm/model_executor/models/exaone4_5.py` modified +0/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/exaone4_5.py`, `vllm/model_executor/models/exaone4_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45073 - [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/45073
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4_5.py`；关联提交 `ccc05de03888`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 42 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone4_5.py` modified +7/-0 (7 lines); hunks: -152,6 +152,8 @@ def forward(; -176,6 +178,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone4_5.py` modified +7/-0 (7 lines); hunks: -152,6 +152,8 @@ def forward(; -176,6 +178,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone4_5.py
@@ -152,6 +152,8 @@ def forward(
+        sequence_lengths: torch.Tensor
+        | None = None,  # Only used for FlashInfer CuDNN backend
@@ -176,6 +178,7 @@ def forward(
+            sequence_lengths=sequence_lengths,
@@ -190,6 +193,7 @@ def forward(
+        "sequence_lengths": 0,
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone4_5.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/exaone4_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #50524 - [Model] Add K-EXAONE-2.0-750B-A37B

- 链接: https://github.com/vllm-project/vllm/pull/50524
- 状态/时间: merged / 2026-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/exaone4.py`, `vllm/model_executor/models/exaone_moe.py`, `vllm/model_executor/models/exaone_moe_mtp.py`；关联提交 `9ae11a6b895d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+161/-11，可读 patch 289 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/exaone_moe.py` modified +152/-5 (157 lines); hunks: -29,19 +29,25; -59,6 +65,7 @@ class ExaoneMoe(nn.Module):; symbols: ExaoneMoe, __init__, forward，涉及 `ExaoneMoe, __init__, forward`；`vllm/model_executor/models/exaone4.py` modified +5/-2 (7 lines); hunks: -31,7 +31,7; -70,6 +70,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/exaone_moe.py` modified +152/-5 (157 lines); hunks: -29,19 +29,25; -59,6 +65,7 @@ class ExaoneMoe(nn.Module):; symbols: ExaoneMoe, __init__, forward
  - `vllm/model_executor/models/exaone4.py` modified +5/-2 (7 lines); hunks: -31,7 +31,7; -70,6 +70,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1 (2 lines); hunks: -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/exaone_moe.py
@@ -29,19 +29,25 @@
+from vllm.model_executor.layers.attention import Attention
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
+    QKVParallelLinear,
+    ReplicatedLinear,
+    RowParallelLinear,
diff -- vllm/model_executor/models/exaone4.py
@@ -31,7 +31,7 @@
-from vllm.model_executor.layers.activation import SiluAndMul
+from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
@@ -70,6 +70,7 @@ def __init__(
+        swiglu_limit: float | None = None,
@@ -95,7 +96,9 @@ def __init__(
-        self.act_fn = SiluAndMul()
diff -- vllm/model_executor/models/exaone_moe_mtp.py
@@ -81,7 +81,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/exaone_moe.py` modified +152/-5; `vllm/model_executor/models/exaone4.py` modified +5/-2; `vllm/model_executor/models/exaone_moe_mtp.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
