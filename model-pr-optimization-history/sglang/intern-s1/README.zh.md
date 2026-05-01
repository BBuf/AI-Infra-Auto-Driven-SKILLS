# sglang Intern-S1 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/internlm_detector.py` | [#14866](https://github.com/sgl-project/sglang/pull/14866) |
| `python/sglang/srt/models/interns1.py` | [#8350](https://github.com/sgl-project/sglang/pull/8350), [#9299](https://github.com/sgl-project/sglang/pull/9299), [#12367](https://github.com/sgl-project/sglang/pull/12367) |
| `python/sglang/srt/models/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |
| `python/sglang/srt/multimodal/processors/interns1pro.py` | [#18145](https://github.com/sgl-project/sglang/pull/18145) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-26 | [#8350](https://github.com/sgl-project/sglang/pull/8350) | merged | model: support intern-s1 | `python/sglang/srt/models/interns1.py` |
| 2025-08-19 | [#9299](https://github.com/sgl-project/sglang/pull/9299) | merged | support for interns1-mini | `python/sglang/srt/models/interns1.py` |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py` |
| 2025-11-03 | [#12367](https://github.com/sgl-project/sglang/pull/12367) | merged | [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids | `python/sglang/srt/models/interns1.py` |
| 2025-12-16 | [#14866](https://github.com/sgl-project/sglang/pull/14866) | merged | Adding tool calling and reasoning parser support for Intern-S1 | `python/sglang/srt/function_call/internlm_detector.py` |
| 2026-01-26 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2026-02-04 | [#18145](https://github.com/sgl-project/sglang/pull/18145) | merged | support interns1-pro | `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py` |

## 逐 PR diff 审计卡

### PR #8350 - model: support intern-s1

- 链接: https://github.com/sgl-project/sglang/pull/8350
- 状态/时间: merged / 2025-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/interns1.py`；关联提交 `b7094a5ef197`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+616/-63，可读 patch 986 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support intern-s1」；模型线: Intern-S1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/interns1.py`；技术摘要: 覆盖「model: support intern-s1」；主要实现面是 `python/sglang/srt/models/interns1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/interns1.py` added +328/-0 (328 lines); hunks: -0,0 +1,328; symbols: InternS1ForConditionalGeneration, __init__, _update_hf_config, pixel_shuffle，涉及 `InternS1ForConditionalGeneration, __init__, _update_hf_config`。
- 代码 diff 细节:
  - `python/sglang/srt/models/interns1.py` added +328/-0 (328 lines); hunks: -0,0 +1,328; symbols: InternS1ForConditionalGeneration, __init__, _update_hf_config, pixel_shuffle
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -0,0 +1,328 @@
+from typing import Iterable, List, Optional, Set, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
+from sglang.srt.distributed import parallel_state
+from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
```

- 已读文件:
  - runtime: `python/sglang/srt/models/interns1.py` added +328/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/lang/chat_template.py`, `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/configs/model_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9299 - support for interns1-mini

- 链接: https://github.com/sgl-project/sglang/pull/9299
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/interns1.py`；关联提交 `a31ea4482436`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support for interns1-mini」；模型线: Intern-S1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/interns1.py`；技术摘要: 覆盖「support for interns1-mini」；主要实现面是 `python/sglang/srt/models/interns1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/interns1.py` modified +5/-0 (5 lines); hunks: -21,6 +21,7; -70,6 +71,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/interns1.py` modified +5/-0 (5 lines); hunks: -21,6 +21,7; -70,6 +71,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -21,6 +21,7 @@
+from sglang.srt.models.qwen3 import Qwen3ForCausalLM
@@ -70,6 +71,10 @@ def __init__(
+        elif config.text_config.architectures[0] == "Qwen3ForCausalLM":
+            self.language_model = Qwen3ForCausalLM(
+                config=config.text_config, quant_config=quant_config
+            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/interns1.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- 链接: https://github.com/sgl-project/sglang/pull/9381
- 状态/时间: merged / 2025-08-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-17，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: InternS1 don't recognize image, updates image token for InternVL processor」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py`；技术摘要: 覆盖「fix: InternS1 don't recognize image, updates image token for InternVL processor」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/conversation.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl，涉及 `__init__, process_image_internvl`；`python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunks: -625,7 +625,7 @@ def generate_chat_conv(; -817,20 +817,7 @@ def generate_chat_conv(; symbols: generate_chat_conv，涉及 `generate_chat_conv`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl
  - `python/sglang/srt/conversation.py` modified +2/-15 (17 lines); hunks: -625,7 +625,7 @@ def generate_chat_conv(; -817,20 +817,7 @@ def generate_chat_conv(; symbols: generate_chat_conv
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-            image_token="<image>",
+            image_token="<IMG_CONTEXT>",
@@ -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=12):
+        original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"
+        input_text = input_text.replace(self.IMG_CONTEXT_TOKEN, original_placeholder)
-            input_text = input_text.replace("<image>", image_tokens, 1)
diff -- python/sglang/srt/conversation.py
@@ -625,7 +625,7 @@ def generate_chat_conv(
-                        if conv.name in ["internvl-2-5", "interns1"]:
+                        if conv.name in ["internvl-2-5"]:
@@ -817,20 +817,7 @@ def generate_chat_conv(
-        image_token="<image>",
-    )
-)
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2; `python/sglang/srt/conversation.py` modified +2/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12367 - [Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids

- 链接: https://github.com/sgl-project/sglang/pull/12367
- 状态/时间: merged / 2025-11-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/interns1.py`；关联提交 `65f1d065c5cf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+8/-41，可读 patch 110 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/interns1.py`；技术摘要: 覆盖「[Bug] Fix Intern-S1 model accuracy and support /generate interface with input_ids」；主要实现面是 `python/sglang/srt/models/interns1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunks: -1,4 +1,4; -50,16 +50,13 @@ def __init__(; symbols: __init__, pixel_shuffle, extract_feature, load_weights，涉及 `__init__, pixel_shuffle, extract_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/models/interns1.py` modified +3/-21 (24 lines); hunks: -1,4 +1,4; -50,16 +50,13 @@ def __init__(; symbols: __init__, pixel_shuffle, extract_feature, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/interns1.py
@@ -1,4 +1,4 @@
-from typing import Iterable, List, Optional, Set, Tuple
+from typing import Iterable, List, Optional, Tuple
@@ -50,16 +50,13 @@ def __init__(
-        self.ps_version = getattr(config, "ps_version", "v1")
-        # self.template = getattr(config, 'template', 'internvl2_5')
-        logger.info(f"ps_version: {self.ps_version}")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/interns1.py` modified +3/-21
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/interns1.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14866 - Adding tool calling and reasoning parser support for Intern-S1

- 链接: https://github.com/sgl-project/sglang/pull/14866
- 状态/时间: merged / 2025-12-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/internlm_detector.py`；关联提交 `5e96beb3e559`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+290/-14，可读 patch 361 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Adding tool calling and reasoning parser support for Intern-S1」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/internlm_detector.py`；技术摘要: 覆盖「Adding tool calling and reasoning parser support for Intern-S1」；主要实现面是 `python/sglang/srt/function_call/internlm_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: InternlmDetector, __init__, has_tool_call, get_arguments，涉及 `InternlmDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/internlm_detector.py` added +248/-0 (248 lines); hunks: -0,0 +1,248; symbols: InternlmDetector, __init__, has_tool_call, get_arguments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/internlm_detector.py
@@ -0,0 +1,248 @@
+# modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parser/internlm2_parser.py
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/internlm_detector.py` added +248/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/constrained/base_grammar_backend.py`, `python/sglang/srt/constrained/xgrammar_backend.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- 链接: https://github.com/sgl-project/sglang/pull/17040
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(processor): support InternS1 text_config in InternVL processor」；模型线: Intern-S1；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`；技术摘要: 覆盖「fix(processor): support InternS1 text_config in InternVL processor」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunks: -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor,...; -121,9 +131,7 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4 (16 lines); hunks: -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor,...; -121,9 +131,7 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -72,7 +72,17 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-        llm_arch = hf_config.llm_config.architectures[0]
+        # Support both InternVL (llm_config) and InternS1 (text_config).
+        # Different multimodal models use different field names for the text backbone:
+        # - InternVL uses: hf_config.llm_config
+        # - InternS1 uses: hf_config.text_config
+        # - Some store architectures at top-level
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +12/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18145 - support interns1-pro

- 链接: https://github.com/sgl-project/sglang/pull/18145
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`；关联提交 `3e7ecb78a60f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+586/-2，可读 patch 647 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support interns1-pro」；模型线: Intern-S1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`；技术摘要: 覆盖「support interns1-pro」；主要实现面是 `python/sglang/srt/models/interns1pro.py`, `python/sglang/srt/multimodal/processors/interns1pro.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/interns1pro.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: InternS1ProTextAttention, __init__, forward_prepare_npu, InternS1ProTextDecoderLayer，涉及 `InternS1ProTextAttention, __init__, forward_prepare_npu`；`python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: InternS1_1ImageProcessor, get_mm_data, process_mm_data_async，涉及 `InternS1_1ImageProcessor, get_mm_data, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/models/interns1pro.py` added +252/-0 (252 lines); hunks: -0,0 +1,252; symbols: InternS1ProTextAttention, __init__, forward_prepare_npu, InternS1ProTextDecoderLayer
  - `python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: InternS1_1ImageProcessor, get_mm_data, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/interns1pro.py
@@ -0,0 +1,252 @@
+import functools
+import logging
+from typing import Any, Dict, Iterable, Optional, Tuple
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
diff -- python/sglang/srt/multimodal/processors/interns1pro.py
@@ -0,0 +1,118 @@
+import time
+from typing import List, Union
+from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.models.interns1pro import InternS1ProForConditionalGeneration
+from sglang.srt.multimodal.processors.qwen_vl import (
+    QwenVLImageProcessor,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/interns1pro.py` added +252/-0; `python/sglang/srt/multimodal/processors/interns1pro.py` added +118/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
