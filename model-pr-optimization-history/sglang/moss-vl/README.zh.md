# sglang MOSS-VL 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/models/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454), [#23932](https://github.com/sgl-project/sglang/pull/23932) |
| `python/sglang/srt/multimodal/processors/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py` |
| 2026-04-30 | [#23932](https://github.com/sgl-project/sglang/pull/23932) | merged | [moss-vl] use Conv3dLayer and remove no-op flat_encoder_result | `python/sglang/srt/models/moss_vl.py` |

## 逐 PR diff 审计卡

### PR #23454 - [srt] Add Moss-VL Python runtime support

- 链接: https://github.com/sgl-project/sglang/pull/23454
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`；关联提交 `59724e90a9b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+2401/-6，可读 patch 2611 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[srt] Add Moss-VL Python runtime support」；模型线: MOSS-VL；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`；技术摘要: 覆盖「[srt] Add Moss-VL Python runtime support」；主要实现面是 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunks: -0,0 +1,1643; symbols: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed，涉及 `MossVLVisionMLP, __init__, forward`；`python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunks: -0,0 +1,612; symbols: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info，涉及 `MossVLImageProcessor, __init__, _build_mm_items`。
- 代码 diff 细节:
  - `python/sglang/srt/models/moss_vl.py` added +1643/-0 (1643 lines); hunks: -0,0 +1,1643; symbols: MossVLVisionMLP, __init__, forward, MossVLVisionPatchEmbed
  - `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0 (612 lines); hunks: -0,0 +1,612; symbols: MossVLImageProcessor, __init__, _build_mm_items, _build_vision_token_info
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/moss_vl.py
@@ -0,0 +1,1643 @@
+"""PyTorch Moss-VL model for SGLang - Qwen3VL Vision + Text with Cross Attention."""
+from __future__ import annotations
+import logging
+from functools import partial
+from typing import Iterable, List, Optional, Tuple
+import torch
diff -- python/sglang/srt/multimodal/processors/moss_vl.py
@@ -0,0 +1,612 @@
+import asyncio
+import os
+import re
+import tempfile
+from typing import Dict, List, Optional, Tuple, Union
+from urllib.parse import unquote, urlparse
```

- 已读文件:
  - runtime: `python/sglang/srt/models/moss_vl.py` added +1643/-0; `python/sglang/srt/multimodal/processors/moss_vl.py` added +612/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/managers/schedule_batch.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23932 - [moss-vl] use Conv3dLayer and remove no-op flat_encoder_result

- 链接: https://github.com/sgl-project/sglang/pull/23932
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/moss_vl.py`；关联提交 `4f0b44c5c666`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-60，可读 patch 146 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[moss-vl] use Conv3dLayer and remove no-op flat_encoder_result」；模型线: MOSS-VL；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/moss_vl.py`；技术摘要: 覆盖「[moss-vl] use Conv3dLayer and remove no-op flat_encoder_result」；主要实现面是 `python/sglang/srt/models/moss_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/moss_vl.py` modified +12/-60 (72 lines); hunks: -20,6 +20,7; -96,7 +97,7 @@ def __init__(self, config) -> None:; symbols: __init__, pad_input_ids, _collect_mm_data, _get_vision_features，涉及 `__init__, pad_input_ids, _collect_mm_data`。
- 代码 diff 细节:
  - `python/sglang/srt/models/moss_vl.py` modified +12/-60 (72 lines); hunks: -20,6 +20,7; -96,7 +97,7 @@ def __init__(self, config) -> None:; symbols: __init__, pad_input_ids, _collect_mm_data, _get_vision_features
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/moss_vl.py
@@ -20,6 +20,7 @@
+from sglang.srt.layers.conv import Conv3dLayer
@@ -96,7 +97,7 @@ def __init__(self, config) -> None:
-        self.proj = nn.Conv3d(
+        self.proj = Conv3dLayer(
@@ -1142,11 +1143,10 @@ def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
-            return None, None, None, None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/moss_vl.py` modified +12/-60
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/moss_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
