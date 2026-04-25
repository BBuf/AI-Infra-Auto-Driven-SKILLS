# sglang MOSS-VL 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/models/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |
| `python/sglang/srt/multimodal/processors/moss_vl.py` | [#23454](https://github.com/sgl-project/sglang/pull/23454) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 1
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#23454](https://github.com/sgl-project/sglang/pull/23454) | merged | [srt] Add Moss-VL Python runtime support | `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py` |

## 逐 PR diff 审计卡

### PR #23454 - [srt] Add Moss-VL Python runtime support

- 链接: https://github.com/sgl-project/sglang/pull/23454
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`；关联提交 `59724e90a9b8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+2401/-6，可读 patch 2611 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MOSS-VL 补齐模型支持入口或运行时能力，标题为「[srt] Add Moss-VL Python runtime support」，变更集中在 `python/sglang/srt/models/moss_vl.py`, `python/sglang/srt/multimodal/processors/moss_vl.py`。PR 描述补充为：## Summary This PR adds Python-side runtime support for Moss-VL in SRT. The changes include: - add `MossVLForConditionalGeneration` model support - add a Moss-VL multimodal proc...
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

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
