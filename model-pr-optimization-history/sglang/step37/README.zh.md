# SGLang Step 3.7 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/StepFun/Step-3.7-Flash.mdx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/step3p7.py` | [#26565](https://github.com/sgl-project/sglang/pull/26565) |
| `python/sglang/srt/models/step3p7.py` | [#26565](https://github.com/sgl-project/sglang/pull/26565) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 1
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-29 | [#26565](https://github.com/sgl-project/sglang/pull/26565) | merged | model: Step-3.7-Flash Support | `python/sglang/srt/models/step3p7.py`, `python/sglang/srt/configs/step3p7.py` |

## 逐 PR diff 审计卡

### PR #26565 - model: Step-3.7-Flash Support

- 链接: https://github.com/sgl-project/sglang/pull/26565
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/step3p7.py`, `python/sglang/srt/models/step3p7.py`；关联提交 `3bdea78ad11d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+1094/-7，可读 patch 1284 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output，涉及 `Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__`；`python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config，涉及 `Step3p7VisionEncoderConfig, __init__, Step3p7Config`。
- 代码 diff 细节:
  - `python/sglang/srt/models/step3p7.py` added +200/-0 (200 lines); hunks: -0,0 +1,200; symbols: Step3p7ForConditionalGeneration, get_model_config_for_expert_location, __init__, _get_vision_model_output
  - `python/sglang/srt/configs/step3p7.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: Step3p7VisionEncoderConfig, __init__, Step3p7Config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/step3p7.py
@@ -0,0 +1,200 @@
+from typing import Iterable, List, Optional, Tuple
+import torch
+from torch import nn
+from transformers.activations import ACT2FN
+from sglang.srt.configs.step3p7 import Step3p7Config
+from sglang.srt.layers.linear import ColumnParallelLinear
diff -- python/sglang/srt/configs/step3p7.py
@@ -0,0 +1,97 @@
+from typing import Optional, Union
+from transformers.configuration_utils import PretrainedConfig
+class Step3p7VisionEncoderConfig(PretrainedConfig):
+    model_type = "perception_encoder"
+    def __init__(
+        self,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/step3p7.py` added +200/-0; `python/sglang/srt/configs/step3p7.py` added +97/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/step3p5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
