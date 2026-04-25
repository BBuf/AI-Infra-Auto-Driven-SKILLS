# sglang InternVL 3.5 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/InternVL/InternVL3.5.mdx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#8067](https://github.com/sgl-project/sglang/pull/8067), [#9705](https://github.com/sgl-project/sglang/pull/9705) |
| `python/sglang/srt/models/internvl.py` | [#5350](https://github.com/sgl-project/sglang/pull/5350), [#6870](https://github.com/sgl-project/sglang/pull/6870), [#9705](https://github.com/sgl-project/sglang/pull/9705), [#13640](https://github.com/sgl-project/sglang/pull/13640), [#13925](https://github.com/sgl-project/sglang/pull/13925), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#16732](https://github.com/sgl-project/sglang/pull/16732), [#19127](https://github.com/sgl-project/sglang/pull/19127) |
| `python/sglang/srt/multimodal/internvl_utils.py` | 无直接 PR 号提交 |
| `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` | [#16732](https://github.com/sgl-project/sglang/pull/16732) |
| `python/sglang/srt/multimodal/processors/internvl.py` | [#9381](https://github.com/sgl-project/sglang/pull/9381), [#9795](https://github.com/sgl-project/sglang/pull/9795), [#10375](https://github.com/sgl-project/sglang/pull/10375), [#15942](https://github.com/sgl-project/sglang/pull/15942), [#17040](https://github.com/sgl-project/sglang/pull/17040), [#19127](https://github.com/sgl-project/sglang/pull/19127), [#19997](https://github.com/sgl-project/sglang/pull/19997) |

## PR 覆盖总览

- git 追溯 PR 数: 14
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 14
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-02 | [#5350](https://github.com/sgl-project/sglang/pull/5350) | merged | Support InternVL3 | `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` |
| 2025-06-11 | [#6870](https://github.com/sgl-project/sglang/pull/6870) | merged | vlm: adapt internvl to VisionAttention | `python/sglang/srt/models/internvl.py` |
| 2025-07-20 | [#8067](https://github.com/sgl-project/sglang/pull/8067) | merged | fix: fix the bug of loading Internvl3 | `python/sglang/srt/configs/internvl.py` |
| 2025-08-20 | [#9381](https://github.com/sgl-project/sglang/pull/9381) | merged | fix: InternS1 don't recognize image, updates image token for InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-09-02 | [#9705](https://github.com/sgl-project/sglang/pull/9705) | merged | Support the internvl3.5 family models in sglang | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py` |
| 2025-09-10 | [#9795](https://github.com/sgl-project/sglang/pull/9795) | merged | refactor(InternVL): Use gpu to preprocess the input image | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-09-15 | [#10375](https://github.com/sgl-project/sglang/pull/10375) | merged | fix(internvl): fix accuracy issue of normalization | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2025-11-21 | [#13640](https://github.com/sgl-project/sglang/pull/13640) | merged | [VLM] Support Piecewise CUDA Graph for InternVL | `python/sglang/srt/models/internvl.py` |
| 2025-11-26 | [#13925](https://github.com/sgl-project/sglang/pull/13925) | merged | [VLM] Support InternVL Vision Encoder Data Parallelism | `python/sglang/srt/models/internvl.py` |
| 2025-12-30 | [#15942](https://github.com/sgl-project/sglang/pull/15942) | merged | [VLM] Support Video for InternVL3_5 | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-01-14 | [#16732](https://github.com/sgl-project/sglang/pull/16732) | merged | [VLM] Support ViT CUDA Graph for InternVL | `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py` |
| 2026-01-26 | [#17040](https://github.com/sgl-project/sglang/pull/17040) | merged | fix(processor): support InternS1 text_config in InternVL processor | `python/sglang/srt/multimodal/processors/internvl.py` |
| 2026-02-27 | [#19127](https://github.com/sgl-project/sglang/pull/19127) | merged | [vlm][internVL] Support processor and embedding inputs for InternVL | `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py` |
| 2026-03-15 | [#19997](https://github.com/sgl-project/sglang/pull/19997) | merged | Fix InternVL and vision attention for non-CUDA backends (e.g. XPU) | `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py` |

## 逐 PR diff 审计卡

### PR #5350 - Support InternVL3

- 链接: https://github.com/sgl-project/sglang/pull/5350
- 状态/时间: merged / 2025-05-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`；关联提交 `3409aaab32c6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1728/-9，可读 patch 1901 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「Support InternVL3」，变更集中在 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`。PR 描述补充为：## Motivation Support InternVL3 ## Modifications Based on PR https://github.com/sgl-project/sglang/pull/3351. and https://github.com/sgl-project/sglang/pull/4433 Support both In...
- 实现要点: `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunks: -0,0 +1,696; symbols: InternLM2Config, to, __init__, _rope_scaling_validation，涉及 `InternLM2Config, to, __init__`；`python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunks: -0,0 +1,670; symbols: FlashAttention, __init__, forward, InternAttention，涉及 `FlashAttention, __init__, forward`；`python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: InternVLImageProcessor, __init__, build_transform, resize_image，涉及 `InternVLImageProcessor, __init__, build_transform`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/internvl.py` added +696/-0 (696 lines); hunks: -0,0 +1,696; symbols: InternLM2Config, to, __init__, _rope_scaling_validation
  - `python/sglang/srt/models/internvl.py` added +670/-0 (670 lines); hunks: -0,0 +1,670; symbols: FlashAttention, __init__, forward, InternAttention
  - `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0 (232 lines); hunks: -0,0 +1,232; symbols: InternVLImageProcessor, __init__, build_transform, resize_image
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/internvl.py
@@ -0,0 +1,696 @@
+import copy
+import os
+from shutil import copyfile
+from typing import Any, Dict, List, Optional, Tuple, Union
+import sentencepiece as spm
+from transformers import (
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,670 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/multimodal_processors/internvl.py
@@ -0,0 +1,232 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/internvl.py` added +696/-0; `python/sglang/srt/models/internvl.py` added +670/-0; `python/sglang/srt/managers/multimodal_processors/internvl.py` added +232/-0
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6870 - vlm: adapt internvl to VisionAttention

- 链接: https://github.com/sgl-project/sglang/pull/6870
- 状态/时间: merged / 2025-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`；关联提交 `83d87685c531`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+103/-126，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「vlm: adapt internvl to VisionAttention」，变更集中在 `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation ## Modifications ## Checklist - [ ] Format your code according to the Code Formatting with Pre-Commit. - [ ] Add unit tests as outlined in the Running Unit Tests....
- 实现要点: `python/sglang/srt/models/internvl.py` modified +46/-102 (148 lines); hunks: -11,21 +11,19; -40,83 +38,32; symbols: FlashAttention, InternAttention, __init__, forward，涉及 `FlashAttention, InternAttention, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/internvl.py` modified +46/-102 (148 lines); hunks: -11,21 +11,19; -40,83 +38,32; symbols: FlashAttention, InternAttention, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -11,21 +11,19 @@
-from typing import Iterable, List, Optional, Tuple, Union
+from typing import Iterable, List, Optional, Set, Tuple, Union
-from einops import rearrange, repeat
-from sgl_kernel.flash_attn import flash_attn_varlen_func
+from sglang.srt.layers.attention.vision import SingletonCache, VisionAttention
@@ -40,83 +38,32 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/internvl.py` modified +46/-102
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8067 - fix: fix the bug of loading Internvl3

- 链接: https://github.com/sgl-project/sglang/pull/8067
- 状态/时间: merged / 2025-07-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`；关联提交 `750838adc4f9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「fix: fix the bug of loading Internvl3」，变更集中在 `python/sglang/srt/configs/internvl.py`。PR 描述补充为：## Motivation When trying to use the InternVL3 model, you will get an error: > Unsupported architecture: Qwen2ForCausalLM This is because Internvl3 uses the llm structure of qwe...
- 实现要点: `python/sglang/srt/configs/internvl.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -311,6 +312,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/internvl.py` modified +3/-0 (3 lines); hunks: -9,6 +9,7; -311,6 +312,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/internvl.py
@@ -9,6 +9,7 @@
+    Qwen2Config,
@@ -311,6 +312,8 @@ def __init__(
+        elif llm_config.get("architectures")[0] == "Qwen2ForCausalLM":
+            self.llm_config = Qwen2Config(**llm_config)
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/internvl.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9381 - fix: InternS1 don't recognize image, updates image token for InternVL processor

- 链接: https://github.com/sgl-project/sglang/pull/9381
- 状态/时间: merged / 2025-08-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `84719b527a2d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-17，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「fix: InternS1 don't recognize image, updates image token for InternVL processor」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`。PR 描述补充为：Updates the image token for InternVL to ` `. This change aligns the image token with the updated template and improves consistency in image processing. Also removes the `interns...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl，涉及 `__init__, process_image_internvl`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2 (9 lines); hunks: -44,7 +44,7 @@ def __init__(self, hf_config, server_args, _image_processor, *...; -218,13 +218,18 @@ def process_image_internvl(image, input_size=448, max_num=...; symbols: __init__, process_image_internvl
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
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/conversation.py`, `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9705 - Support the internvl3.5 family models in sglang

- 链接: https://github.com/sgl-project/sglang/pull/9705
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`；关联提交 `f64b8e3e4e13`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+34/-0，可读 patch 84 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「Support the internvl3.5 family models in sglang」，变更集中在 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`。PR 描述补充为：## Motivation This PR provides support for all models in the internvl3.5 family ## Modification Registers different backbone models for all internvl3.5 models. Also fixes the qu...
- 实现要点: `python/sglang/srt/models/internvl.py` modified +28/-0 (28 lines); hunks: -26,8 +26,10; -445,6 +447,14 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/configs/internvl.py` modified +6/-0 (6 lines); hunks: -6,11 +6,13; -316,7 +318,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/internvl.py` modified +28/-0 (28 lines); hunks: -26,8 +26,10; -445,6 +447,14 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/configs/internvl.py` modified +6/-0 (6 lines); hunks: -6,11 +6,13; -316,7 +318,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -26,8 +26,10 @@
+from sglang.srt.models.gpt_oss import GptOssForCausalLM
+from sglang.srt.models.qwen3 import Qwen3ForCausalLM
@@ -445,6 +447,14 @@ def __init__(
+        elif config.llm_config.architectures[0] == "GptOssForCausalLM":
+            self.language_model = GptOssForCausalLM(
+                config=config.llm_config, quant_config=quant_config
diff -- python/sglang/srt/configs/internvl.py
@@ -6,11 +6,13 @@
+    GptOssConfig,
+    Qwen3MoeConfig,
@@ -316,7 +318,11 @@ def __init__(
+            self.llm_config = Qwen3MoeConfig(**llm_config)
+        elif llm_config.get("architectures")[0] == "Qwen3ForCausalLM":
+        elif llm_config.get("architectures")[0] == "GptOssForCausalLM":
```

- 已读文件:
  - runtime: `python/sglang/srt/models/internvl.py` modified +28/-0; `python/sglang/srt/configs/internvl.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9795 - refactor(InternVL): Use gpu to preprocess the input image

- 链接: https://github.com/sgl-project/sglang/pull/9795
- 状态/时间: merged / 2025-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `15f993472c58`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+141/-129，可读 patch 340 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 优化关键推理路径或后端选择，标题为「refactor(InternVL): Use gpu to preprocess the input image」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`。PR 描述补充为：## Motivation Now ,internvl preprocesses images on the CPU. Putting it on the GPU for preprocessing will improve performance. ## Modifications image operations such as resizing,...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129 (270 lines); hunks: -2,8 +2,10; -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, build_transform, resize_image, to_tensor，涉及 `__init__, build_transform, resize_image`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129 (270 lines); hunks: -2,8 +2,10; -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, build_transform, resize_image, to_tensor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -2,8 +2,10 @@
-from decord import VideoReader, cpu
+import torchvision.transforms as T
+from decord import VideoReader, cpu, gpu
+from torchvision.transforms import InterpolationMode
@@ -48,99 +50,6 @@ def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
-    @staticmethod
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +141/-129
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10375 - fix(internvl): fix accuracy issue of normalization

- 链接: https://github.com/sgl-project/sglang/pull/10375
- 状态/时间: merged / 2025-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `1fcccda4b2b3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-8，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「fix(internvl): fix accuracy issue of normalization」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`。PR 描述补充为：## Motivation - Found precision issues when testing internvl3 8b, but no issues on 1b and 38b. - The reason for the modification at that time was that we considered using a fixe...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8 (28 lines); hunks: -1,5 +1,7; -19,6 +21,20; symbols: InternVLImageProcessor, _get_normalize_tensors, __init__, load_video，涉及 `InternVLImageProcessor, _get_normalize_tensors, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8 (28 lines); hunks: -1,5 +1,7; -19,6 +21,20; symbols: InternVLImageProcessor, _get_normalize_tensors, __init__, load_video
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -1,5 +1,7 @@
+from functools import lru_cache
@@ -19,6 +21,20 @@
+    IMAGENET_MEAN = [0.485, 0.456, 0.406]
+    IMAGENET_STD = [0.229, 0.224, 0.225]
+    @staticmethod
+    @lru_cache(maxsize=1)
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +20/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13640 - [VLM] Support Piecewise CUDA Graph for InternVL

- 链接: https://github.com/sgl-project/sglang/pull/13640
- 状态/时间: merged / 2025-11-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`；关联提交 `475962a139d1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+103/-13，可读 patch 183 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Support Piecewise CUDA Graph for InternVL」，变更集中在 `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation Address https://github.com/sgl-project/sglang/pull/12838 This PR is to support Piecewise CUDA Graph for InternVL. It is a follow up of #13055 . The accuracy result...
- 实现要点: `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunks: -14,10 +14,7; -471,6 +468,12 @@ def __init__(; symbols: __init__, pixel_shuffle, forward, pad_input_ids，涉及 `__init__, pixel_shuffle, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/internvl.py` modified +21/-10 (31 lines); hunks: -14,10 +14,7; -471,6 +468,12 @@ def __init__(; symbols: __init__, pixel_shuffle, forward, pad_input_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -14,10 +14,7 @@
-from sglang.srt.managers.mm_utils import (
-    MultiModalityDataPaddingPatternTokenPairs,
-    general_mm_embed_routine,
-)
+from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternTokenPairs
@@ -471,6 +468,12 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/internvl.py` modified +21/-10
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_piecewise_cuda_graph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13925 - [VLM] Support InternVL Vision Encoder Data Parallelism

- 链接: https://github.com/sgl-project/sglang/pull/13925
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`；关联提交 `ca5c8b16f67d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+118/-25，可读 patch 266 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Support InternVL Vision Encoder Data Parallelism」，变更集中在 `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation This PR is to support InternVL Vision-Encoder data parallelism. https://github.com/sgl-project/sglang/issues/12971 From profiling we can see InternVL Vision-Encode...
- 实现要点: `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunks: -7,11 +7,16; -28,6 +33,8; symbols: __init__, forward, InternMLP，涉及 `__init__, forward, InternMLP`。
- 代码 diff 细节:
  - `python/sglang/srt/models/internvl.py` modified +83/-25 (108 lines); hunks: -7,11 +7,16; -28,6 +33,8; symbols: __init__, forward, InternMLP
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -7,11 +7,16 @@
-from transformers.activations import ACT2FN
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
+from sglang.srt.layers.activation import get_act_fn
```

- 已读文件:
  - runtime: `python/sglang/srt/models/internvl.py` modified +83/-25
- 验证与风险: diff 自带测试面 `test/nightly/test_encoder_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15942 - [VLM] Support Video for InternVL3_5

- 链接: https://github.com/sgl-project/sglang/pull/15942
- 状态/时间: merged / 2025-12-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `94bcc19bcef6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+426/-118，可读 patch 658 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Support Video for InternVL3_5」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation This PR is to enable SGLang supporting InternVL3_5 video input. InternVL3.5-8B demonstrated speedup 2.5x than Qwen2.5-VL-7B video inference. Prompt script: Server...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunks: -1,6 +1,8; -15,26 +17,44; symbols: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__，涉及 `InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors`；`python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunks: -539,6 +539,7 @@ def __init__(; -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: __init__, get_image_feature, get_video_feature, forward，涉及 `__init__, get_image_feature, get_video_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118 (536 lines); hunks: -1,6 +1,8; -15,26 +17,44; symbols: InternVLImageProcessor, InternVLProcessor, _get_normalize_tensors, __init__
  - `python/sglang/srt/models/internvl.py` modified +8/-0 (8 lines); hunks: -539,6 +539,7 @@ def __init__(; -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: __init__, get_image_feature, get_video_feature, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -1,6 +1,8 @@
+import logging
+from typing import List
@@ -15,26 +17,44 @@
+logger = logging.getLogger(__name__)
-class InternVLImageProcessor(BaseMultimodalProcessor):
+class InternVLProcessor(BaseMultimodalProcessor):
diff -- python/sglang/srt/models/internvl.py
@@ -539,6 +539,7 @@ def __init__(
+            Modality.VIDEO: self.get_video_feature,
@@ -594,6 +595,13 @@ def get_image_feature(self, items: List[MultimodalDataItem]):
+    def get_video_feature(self, items: List[MultimodalDataItem]):
+        # items: each item corresponds to one video (recommended)
+        # item.feature shape: [num_frames, 3, 448, 448]  (or [num_tiles, 3, 448, 448])
+        pixel_values = torch.cat([item.feature for item in items], dim=0)
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +418/-118; `python/sglang/srt/models/internvl.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16732 - [VLM] Support ViT CUDA Graph for InternVL

- 链接: https://github.com/sgl-project/sglang/pull/16732
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`；关联提交 `feae615b1146`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+219/-6，可读 patch 304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[VLM] Support ViT CUDA Graph for InternVL」，变更集中在 `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation This PR is to support ViT CUDA Graph for InternVL family. InternVL model is different from Qwen2.5-VL and Qwen3-VL which setting flatten_batch=True, InternVL model...
- 实现要点: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: InternViTCudaGraphRunner, __init__, device, dtype，涉及 `InternViTCudaGraphRunner, __init__, device`；`python/sglang/srt/models/internvl.py` modified +27/-3 (30 lines); hunks: -13,6 +13,7; -36,6 +37,9; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: InternViTCudaGraphRunner, __init__, device, dtype
  - `python/sglang/srt/models/internvl.py` modified +27/-3 (30 lines); hunks: -13,6 +13,7; -36,6 +37,9; symbols: forward, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py
@@ -0,0 +1,183 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/internvl.py
@@ -13,6 +13,7 @@
+from sglang.srt.environ import envs
@@ -36,6 +37,9 @@
+from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
+    InternViTCudaGraphRunner,
+)
@@ -82,8 +86,9 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py` added +183/-0; `python/sglang/srt/models/internvl.py` modified +27/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17040 - fix(processor): support InternS1 text_config in InternVL processor

- 链接: https://github.com/sgl-project/sglang/pull/17040
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `539924037fbc`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「fix(processor): support InternS1 text_config in InternVL processor」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`。PR 描述补充为：## Motivation InternS1 models use `text_config` instead of `llm_config` for the text backbone configuration. When attempting to launch InternS1 models (e.g., `internlm/Intern-S1...
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

### PR #19127 - [vlm][internVL] Support processor and embedding inputs for InternVL

- 链接: https://github.com/sgl-project/sglang/pull/19127
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `f0c208959794`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+282/-7，可读 patch 379 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 补齐模型支持入口或运行时能力，标题为「[vlm][internVL] Support processor and embedding inputs for InternVL」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`。PR 描述补充为：## Motivation This PR helps the issue: https://github.com/sgl-project/sglang/issues/6483 For InternVL, it adds support to the processor and embedding inputs. Specifically, Huggi...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunks: -9,11 +9,15; -255,9 +259,113 @@ def _resolve_video_num_frames(; symbols: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data，涉及 `_resolve_video_num_frames, _has_special_format, _process_special_format`；`python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunks: -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, get_video_feature，涉及 `get_image_feature, get_video_feature`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1 (110 lines); hunks: -9,11 +9,15; -255,9 +259,113 @@ def _resolve_video_num_frames(; symbols: _resolve_video_num_frames, _has_special_format, _process_special_format, process_and_combine_mm_data
  - `python/sglang/srt/models/internvl.py` modified +7/-0 (7 lines); hunks: -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):; symbols: get_image_feature, get_video_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -9,11 +9,15 @@
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+)
+    BaseMultiModalProcessorOutput,
diff -- python/sglang/srt/models/internvl.py
@@ -616,13 +616,20 @@ def get_image_feature(self, items: List[MultimodalDataItem]):
+        # If already precomputed embeddings (not raw pixel values), skip vision encoder.
+        # Normal pixel_values are 4D [N, C, H, W]; precomputed embeddings are 2D or 3D.
+        if pixel_values.dim() != 4:
+            return pixel_values
+        # If already precomputed embeddings, skip vision encoder.
+        if pixel_values.dim() != 4:
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +109/-1; `python/sglang/srt/models/internvl.py` modified +7/-0
- 验证与风险: diff 自带测试面 `test/registered/vlm/test_vlm_input_format.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19997 - Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)

- 链接: https://github.com/sgl-project/sglang/pull/19997
- 状态/时间: merged / 2026-03-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `7458407437ca`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+184/-14，可读 patch 324 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 InternVL 3.5 修复已暴露的启动、加载、解析或数值问题，标题为「Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)」，变更集中在 `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py`。PR 描述补充为：## Motivation InternVL and vision attention currently assume CUDA: they use hardcoded `"cuda"` and `.cuda()`, and the vision attention backend selection does not handle XPU. Thi...
- 实现要点: `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10 (27 lines); hunks: -19,6 +19,7; -434,7 +435,7 @@ async def process_qwen_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `process_qwen_mm_data_async, process_internlm2_mm_data_async`；`test/srt/xpu/test_internvl.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: InternVLXPUServerBase, setUpClass, tearDownClass, TestInternVL25Server，涉及 `InternVLXPUServerBase, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10 (27 lines); hunks: -19,6 +19,7; -434,7 +435,7 @@ async def process_qwen_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `test/srt/xpu/test_internvl.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: InternVLXPUServerBase, setUpClass, tearDownClass, TestInternVL25Server
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -19,6 +19,7 @@
+from sglang.srt.utils import get_device
@@ -434,7 +435,7 @@ async def process_qwen_mm_data_async(
-        mean, std = self._get_normalize_tensors(device="cuda")
+        mean, std = self._get_normalize_tensors(device=get_device())
@@ -444,10 +445,11 @@ async def process_qwen_mm_data_async(
-                    torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
diff -- test/srt/xpu/test_internvl.py
@@ -0,0 +1,147 @@
+"""
+XPU tests for InternVL models (InternVL2.5-2B, InternVL3.5-2B).
+Uses the same structure as test_vision_openai_server_a.py: OpenAI /v1 chat API
+and ImageOpenAITestMixin. An XPU-specific base injects --device xpu and
+--attention-backend intel_xpu.
+Usage (pick module path to match your cwd):
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/internvl.py` modified +17/-10
  - tests: `test/srt/xpu/test_internvl.py` added +147/-0
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/xpu/test_internvl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
