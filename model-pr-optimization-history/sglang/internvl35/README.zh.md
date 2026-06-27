# sglang InternVL 3.5 模型 PR 优化历史

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
- 原文档显式引用补充 PR 数: 12
- 当前文档总 PR 数: 26
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-03-21 | [#3351](https://github.com/sgl-project/sglang/pull/3351) | closed | model: Intern vl 2.5 | `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py` |
| 2025-05-02 | [#5350](https://github.com/sgl-project/sglang/pull/5350) | merged | Support InternVL3 | `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py` |
| 2025-05-30 | [#4433](https://github.com/sgl-project/sglang/pull/4433) | closed | Support InternVL2.5 | `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py` |
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
| 2026-03-15 | [#20282](https://github.com/sgl-project/sglang/pull/20282) | merged | Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug | `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-29 | [#19749](https://github.com/sgl-project/sglang/pull/19749) | merged | [Feature] Optimizations for JPEG input on NVIDIA GPU | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` |
| 2026-04-03 | [#21899](https://github.com/sgl-project/sglang/pull/21899) | merged | [VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality | `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py` |
| 2026-04-03 | [#21738](https://github.com/sgl-project/sglang/pull/21738) | merged | refactor: replace mm_inputs dict with MultimodalProcessorOutput | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-25 | [#23568](https://github.com/sgl-project/sglang/pull/23568) | merged | Parakeet nemotron encoder | `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | merged | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |

## 逐 PR diff 审计卡

### PR #3351 - model: Intern vl 2.5

- 链接: https://github.com/sgl-project/sglang/pull/3351
- 状态/时间: closed / 2025-03-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 24 个文件，+4538/-163，可读 patch 5186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: Intern vl 2.5」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`；技术摘要: 覆盖「model: Intern vl 2.5」；主要实现面是 `python/sglang/srt/models/deepseek_janus_pro.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/tokenizers/lmtokenizer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0 (2174 lines); hunks: -0,0 +1,2174; symbols: VQ_16, ModelArgs, _ntuple, parse，涉及 `VQ_16, ModelArgs, _ntuple`；`python/sglang/srt/models/internvl.py` added +622/-0 (622 lines); hunks: -0,0 +1,622; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward，涉及 `InternVisionEmbeddings, __init__, _get_pos_embed`；`python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: InternLM2Tokenizer, __init__, no_prefix_space_tokens, vocab_size，涉及 `InternLM2Tokenizer, __init__, no_prefix_space_tokens`；`python/sglang/srt/configs/janus.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: DictToObject, __init__, VisionConfig, GenAlignerConfig，涉及 `DictToObject, __init__, VisionConfig`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0 (2174 lines); hunks: -0,0 +1,2174; symbols: VQ_16, ModelArgs, _ntuple, parse
  - `python/sglang/srt/models/internvl.py` added +622/-0 (622 lines); hunks: -0,0 +1,622; symbols: InternVisionEmbeddings, __init__, _get_pos_embed, forward
  - `python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: InternLM2Tokenizer, __init__, no_prefix_space_tokens, vocab_size
  - `python/sglang/srt/configs/janus.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: DictToObject, __init__, VisionConfig, GenAlignerConfig
  - `python/sglang/srt/models/minicpmv.py` modified +11/-73 (84 lines); hunks: -41,7 +41,6; -51,7 +50,7; symbols: __init__, pad_input_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_janus_pro.py
@@ -0,0 +1,2174 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,622 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/tokenizers/lmtokenizer.py
@@ -0,0 +1,242 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_janus_pro.py` added +2174/-0; `python/sglang/srt/models/internvl.py` added +622/-0; `python/sglang/srt/tokenizers/lmtokenizer.py` added +242/-0; `python/sglang/srt/configs/janus.py` added +155/-0; `python/sglang/srt/models/minicpmv.py` modified +11/-73; `python/sglang/srt/models/qwen2_vl.py` modified +12/-35
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5350 - Support InternVL3

- 链接: https://github.com/sgl-project/sglang/pull/5350
- 状态/时间: merged / 2025-05-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`；关联提交 `3409aaab32c6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1728/-9，可读 patch 1901 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support InternVL3」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`；技术摘要: 覆盖「Support InternVL3」；主要实现面是 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/multimodal_processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #4433 - Support InternVL2.5

- 链接: https://github.com/sgl-project/sglang/pull/4433
- 状态/时间: closed / 2025-05-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1210/-16，可读 patch 1464 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support InternVL2.5」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`；技术摘要: 覆盖「Support InternVL2.5」；主要实现面是 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/configs/model_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/internvl.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: FlashAttention, __init__, forward, InternVisionEmbeddings，涉及 `FlashAttention, __init__, forward`；`python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2 (9 lines); hunks: -49,7 +49,11; -187,7 +191,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -318,7 +318,10 @@ def _verify_quantization(self) -> None:; -472,6 +475,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: _verify_quantization, get_hf_eos_token_id, is_generation_model，涉及 `_verify_quantization, get_hf_eos_token_id, is_generation_model`；`python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1984,7 +1984,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/internvl.py` added +733/-0 (733 lines); hunks: -0,0 +1,733; symbols: FlashAttention, __init__, forward, InternVisionEmbeddings
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2 (9 lines); hunks: -49,7 +49,11; -187,7 +191,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -318,7 +318,10 @@ def _verify_quantization(self) -> None:; -472,6 +475,7 @@ def is_generation_model(model_architectures: List[str], is_e...; symbols: _verify_quantization, get_hf_eos_token_id, is_generation_model
  - `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: -1984,7 +1984,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/internlm2.py` modified +1/-1 (2 lines); hunks: -114,7 +114,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/internvl.py
@@ -0,0 +1,733 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -49,7 +49,11 @@
-from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
+from sglang.srt.hf_transformers_utils import (
+    get_processor,
+    get_tokenizer,
+    get_tokenizer_from_processor,
+)
diff -- python/sglang/srt/configs/model_config.py
@@ -318,7 +318,10 @@ def _verify_quantization(self) -> None:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/internvl.py` added +733/-0; `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-2; `python/sglang/srt/configs/model_config.py` modified +5/-1; `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1; `python/sglang/srt/models/internlm2.py` modified +1/-1; `python/sglang/srt/managers/image_processors/intern_vl.py` added +230/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6870 - vlm: adapt internvl to VisionAttention

- 链接: https://github.com/sgl-project/sglang/pull/6870
- 状态/时间: merged / 2025-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`；关联提交 `83d87685c531`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+103/-126，可读 patch 408 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「vlm: adapt internvl to VisionAttention」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「vlm: adapt internvl to VisionAttention」；主要实现面是 `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`；关联提交 `750838adc4f9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: fix the bug of loading Internvl3」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/internvl.py`；技术摘要: 覆盖「fix: fix the bug of loading Internvl3」；主要实现面是 `python/sglang/srt/configs/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `84719b527a2d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+9/-17，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: InternS1 don't recognize image, updates image token for InternVL processor」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`；技术摘要: 覆盖「fix: InternS1 don't recognize image, updates image token for InternVL processor」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/internvl.py`, `python/sglang/srt/models/internvl.py`；关联提交 `f64b8e3e4e13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+34/-0，可读 patch 84 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support the internvl3.5 family models in sglang」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`；技术摘要: 覆盖「Support the internvl3.5 family models in sglang」；主要实现面是 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/configs/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `15f993472c58`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+141/-129，可读 patch 340 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor(InternVL): Use gpu to preprocess the input image」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`；技术摘要: 覆盖「refactor(InternVL): Use gpu to preprocess the input image」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `1fcccda4b2b3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-8，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(internvl): fix accuracy issue of normalization」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`；技术摘要: 覆盖「fix(internvl): fix accuracy issue of normalization」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[VLM] Support Piecewise CUDA Graph for InternVL」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「[VLM] Support Piecewise CUDA Graph for InternVL」；主要实现面是 `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[VLM] Support InternVL Vision Encoder Data Parallelism」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「[VLM] Support InternVL Vision Encoder Data Parallelism」；主要实现面是 `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[VLM] Support Video for InternVL3_5」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「[VLM] Support Video for InternVL3_5」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/internvl.py`, `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`；关联提交 `feae615b1146`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+219/-6，可读 patch 304 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support ViT CUDA Graph for InternVL」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「[VLM] Support ViT CUDA Graph for InternVL」；主要实现面是 `python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py`, `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `539924037fbc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-4，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(processor): support InternS1 text_config in InternVL processor」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`；技术摘要: 覆盖「fix(processor): support InternS1 text_config in InternVL processor」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[vlm][internVL] Support processor and embedding inputs for InternVL」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`；技术摘要: 覆盖「[vlm][internVL] Support processor and embedding inputs for InternVL」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/models/internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/internvl.py`；关联提交 `7458407437ca`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+184/-14，可读 patch 324 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py`；技术摘要: 覆盖「Fix InternVL and vision attention for non-CUDA backends (e.g. XPU)」；主要实现面是 `python/sglang/srt/multimodal/processors/internvl.py`, `test/srt/xpu/test_internvl.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- 链接: https://github.com/sgl-project/sglang/pull/20282
- 状态/时间: merged / 2026-03-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+704/-90，可读 patch 1053 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`；技术摘要: 覆盖「Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug」；主要实现面是 `python/sglang/srt/layers/conv.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/pixtral.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad，涉及 `_tuplify, _check_enable_linear, _reverse_repeat_tuple`；`python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger，涉及 `__init__, copy_conv3d_weight_to_linear, forward`；`python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__，涉及 `VisionTransformer, __init__`；`python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: -0,0 +1,300; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad
  - `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: -35,6 +35,7; -203,34 +204,25 @@ def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger
  - `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: -35,6 +35,7; -328,7 +329,7 @@ class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__
  - `python/sglang/srt/models/clip.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -32,7 +33,7 @@ def __init__(self, config: CLIPVisionConfig):; symbols: __init__
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1 (3 lines); hunks: -11,6 +11,7; -113,7 +114,7 @@ def __init__(self, config, quant_config: Optional[Quantizati...; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/conv.py
@@ -0,0 +1,300 @@
+"""
+Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.
+When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
+is equivalent to unfold + F.linear, which is significantly faster on CUDA and
+also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
+(https://github.com/pytorch/pytorch/issues/168167).
diff -- python/sglang/srt/models/glm4v.py
@@ -35,6 +35,7 @@
+from sglang.srt.layers.conv import Conv3dLayer
@@ -203,34 +204,25 @@ def __init__(
-        self.proj = nn.Conv3d(
+        self.proj = Conv3dLayer(
-        k = self.in_channels * self.temporal_patch_size * self.patch_size**2
-        self.linear = nn.Linear(
diff -- python/sglang/srt/models/pixtral.py
@@ -35,6 +35,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/conv.py` added +300/-0; `python/sglang/srt/models/glm4v.py` modified +12/-27; `python/sglang/srt/models/pixtral.py` modified +3/-2; `python/sglang/srt/models/clip.py` modified +2/-1; `python/sglang/srt/models/dots_vlm_vit.py` modified +2/-1; `python/sglang/srt/models/idefics2.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/unit/test_conv_layer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17784 - Upgrade transformers==5.3.0

- 链接: https://github.com/sgl-project/sglang/pull/17784
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 95 个文件，+1136/-343，可读 patch 2752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers==5.3.0」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`；技术摘要: 覆盖「Upgrade transformers==5.3.0」；主要实现面是 `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update，涉及 `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`；`python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope，涉及 `_get_rope_param, get_rope`；`python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes，涉及 `ModelImpl, is_deepseek_nsa, _derive_model_shapes`；`python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__，涉及 `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19749 - [Feature] Optimizations for JPEG input on NVIDIA GPU

- 链接: https://github.com/sgl-project/sglang/pull/19749
- 状态/时间: merged / 2026-03-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+114/-46，可读 patch 301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Optimizations for JPEG input on NVIDIA GPU」；模型线: InternVL 3.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`；技术摘要: 覆盖「[Feature] Optimizations for JPEG input on NVIDIA GPU」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/llava.py`, `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6 (20 lines); hunks: -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:; -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):; symbols: get_combined_regex, BaseMultimodalProcessor, __init__, get_estimated_frames_list，涉及 `get_combined_regex, BaseMultimodalProcessor, __init__`；`python/sglang/srt/multimodal/processors/llava.py` modified +2/-1 (3 lines); hunks: -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):; -49,7 +50,7 @@ def _process_single_image_task(; symbols: LlavaImageProcessor, __init__, _process_single_image_task，涉及 `LlavaImageProcessor, __init__, _process_single_image_task`；`python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0 (3 lines); hunks: -35,6 +35,9; symbols: NanoNemotronVLImageProcessor, __init__，涉及 `NanoNemotronVLImageProcessor, __init__`；`python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: InternVLProcessor，涉及 `InternVLProcessor`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6 (20 lines); hunks: -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:; -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):; symbols: get_combined_regex, BaseMultimodalProcessor, __init__, get_estimated_frames_list
  - `python/sglang/srt/multimodal/processors/llava.py` modified +2/-1 (3 lines); hunks: -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):; -49,7 +50,7 @@ def _process_single_image_task(; symbols: LlavaImageProcessor, __init__, _process_single_image_task
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0 (3 lines); hunks: -35,6 +35,9; symbols: NanoNemotronVLImageProcessor, __init__
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7; symbols: InternVLProcessor
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +1/-0 (1 lines); hunks: -16,6 +16,7; symbols: KimiK2_5VLImageProcessor, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -173,6 +173,7 @@ def get_combined_regex(self) -> re.Pattern:
+    gpu_image_decode = True  # Enable GPU decoding by default
@@ -468,8 +469,9 @@ def get_estimated_frames_list(self, image_data):
-    @staticmethod
+    @classmethod
+        cls,
@@ -481,7 +483,8 @@ def _load_single_item(
diff -- python/sglang/srt/multimodal/processors/llava.py
@@ -33,6 +33,7 @@ class LlavaImageProcessor(BaseMultimodalProcessor):
+    gpu_image_decode = False  # Llava processes loaded image as PIL image explicitly
@@ -49,7 +50,7 @@ def _process_single_image_task(
-            image, image_size = load_image(url)
+            image, image_size = load_image(url, False)
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -35,6 +35,9 @@
+    gpu_image_decode = (
+        False  # NanoNemotronVL processes loaded image as PIL image explicitly
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +14/-6; `python/sglang/srt/multimodal/processors/llava.py` modified +2/-1; `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +3/-0; `python/sglang/srt/multimodal/processors/internvl.py` modified +1/-0; `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +1/-0; `python/sglang/srt/multimodal/processors/kimi_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21899 - [VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality

- 链接: https://github.com/sgl-project/sglang/pull/21899
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+217/-136，可读 patch 647 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py`；技术摘要: 覆盖「[VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES modality」；主要实现面是 `python/sglang/srt/models/llava.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/llava.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llava.py` modified +47/-34 (81 lines); hunks: -55,6 +55,21; -63,13 +78,8 @@ def pad_input_ids(self, input_ids: List[int], image_inputs: M...; symbols: LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids, forward，涉及 `LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids`；`python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8 (36 lines); hunks: -588,11 +588,21 @@ async def process_qwen_mm_data_async(; -702,11 +712,21 @@ async def process_internlm2_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `process_qwen_mm_data_async, process_internlm2_mm_data_async`；`python/sglang/srt/multimodal/processors/llava.py` modified +16/-11 (27 lines); hunks: -187,34 +187,39 @@ async def process_mm_data_async(; symbols: process_mm_data_async，涉及 `process_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7 (26 lines); hunks: -223,6 +223,8 @@ async def process_mm_data_async(; -231,6 +233,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async，涉及 `process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llava.py` modified +47/-34 (81 lines); hunks: -55,6 +55,21; -63,13 +78,8 @@ def pad_input_ids(self, input_ids: List[int], image_inputs: M...; symbols: LlavaBaseForCausalLM, _infer_image_aspect_ratio, pad_input_ids, forward
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8 (36 lines); hunks: -588,11 +588,21 @@ async def process_qwen_mm_data_async(; -702,11 +712,21 @@ async def process_internlm2_mm_data_async(; symbols: process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/llava.py` modified +16/-11 (27 lines); hunks: -187,34 +187,39 @@ async def process_mm_data_async(; symbols: process_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7 (26 lines); hunks: -223,6 +223,8 @@ async def process_mm_data_async(; -231,6 +233,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async
  - `python/sglang/srt/models/minicpmv.py` modified +15/-3 (18 lines); hunks: -993,7 +993,11 @@ def pad_input_ids(self, input_ids: List[int], image_inputs:...; -1155,7 +1159,11 @@ def pad_input_ids(self, input_ids: List[int], image_input...; symbols: pad_input_ids
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llava.py
@@ -55,6 +55,21 @@
+    @staticmethod
+    def _infer_image_aspect_ratio(mm_items):
+        """Determine image_aspect_ratio from processor metadata or item count."""
+        # Check if processor stored the aspect_ratio it used
+        for item in mm_items:
+            ar = item.model_specific_data.get("image_aspect_ratio")
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -588,11 +588,21 @@ async def process_qwen_mm_data_async(
-            items.append(
-                MultimodalDataItem(
-                    feature=image_tensor, modality=Modality.IMAGE, offsets=image_offsets
-                )
+            # Split per-image for better cache granularity
+            assert len(num_patches_list) == len(image_offsets), (
diff -- python/sglang/srt/multimodal/processors/llava.py
@@ -187,34 +187,39 @@ async def process_mm_data_async(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llava.py` modified +47/-34; `python/sglang/srt/multimodal/processors/internvl.py` modified +28/-8; `python/sglang/srt/multimodal/processors/llava.py` modified +16/-11; `python/sglang/srt/multimodal/processors/minicpm.py` modified +19/-7; `python/sglang/srt/models/minicpmv.py` modified +15/-3; `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-3
- 验证与风险: diff 自带测试面 `python/sglang/test/test_mm_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- 链接: https://github.com/sgl-project/sglang/pull/21738
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 40 个文件，+408/-314，可读 patch 1321 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor: replace mm_inputs dict with MultimodalProcessorOutput」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「refactor: replace mm_inputs dict with MultimodalProcessorOutput」；主要实现面是 `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async，涉及 `get_mm_data, process_mm_data_async`；`python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_mm_data_async, process_qwen_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async，涉及 `get_mm_data, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23 (50 lines); hunks: -12,7 +12,11; -474,17 +478,17 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24 (49 lines); hunks: -11,6 +11,7; -337,14 +338,14 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22 (45 lines); hunks: -5,6 +5,7; -158,17 +159,17 @@ async def _process_special_format(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19 (42 lines); hunks: -1,7 +1,11; -26,15 +30,15 @@ def get_mm_data(self, prompt, embeddings, img_grid_thw):; symbols: get_mm_data, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15 (34 lines); hunks: -1,6 +1,10; -69,13 +73,13 @@ def get_mm_data(self, prompt, embeddings, **kwargs):; symbols: get_mm_data, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/qwen_vl.py
@@ -12,7 +12,11 @@
-from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalProcessorOutput,
+)
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -11,6 +11,7 @@
+    MultimodalProcessorOutput,
@@ -337,14 +338,14 @@ async def _process_special_format(
-        return {
-            "input_ids": input_ids_tensor.flatten().tolist(),
-            "mm_items": mm_items,
-            "im_start_id": self.img_start_token_id,
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -5,6 +5,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +27/-23; `python/sglang/srt/multimodal/processors/internvl.py` modified +25/-24; `python/sglang/srt/multimodal/processors/minicpm.py` modified +23/-22; `python/sglang/srt/multimodal/processors/interns1pro.py` modified +23/-19; `python/sglang/srt/multimodal/processors/qwen_audio.py` modified +19/-15; `python/sglang/srt/multimodal/processors/transformers_auto.py` modified +18/-14
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 330 个文件，+80364/-0，可读 patch 68714 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: InternVL 3.5；类别: 文档/测试/CI；主要 diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`；技术摘要: 覆盖「Add new Mintlify documentation site (docs_new/)」；主要实现面是 `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in，涉及 `get_messages, get_current_weather, convert_dict_to_tool`；`docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages，涉及 `CapitalInfo, get_messages`；`docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317；`docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)。
- 代码 diff 细节:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- 关键代码摘录:

```diff
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -0,0 +1,740 @@
+---
+title: "Tool Parser"
+metatags:
+    description: "SGLang function calling: tool parsers for DeepSeek, Llama, Qwen, Mistral, GLM, Kimi K2. OpenAI-compatible tool use API."
+---
+This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -0,0 +1,663 @@
+---
+title: "Structured Outputs For Reasoning Models"
+metatags:
+    description: "SGLang structured outputs for reasoning models: free-form thinking with constrained final output for DeepSeek R1, QwQ models."
+---
+When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -0,0 +1,317 @@
```

- 已读文件:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23568 - Parakeet nemotron encoder

- 链接: https://github.com/sgl-project/sglang/pull/23568
- 状态/时间: merged / 2026-04-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1289/-116，可读 patch 1817 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Parakeet nemotron encoder」；模型线: InternVL 3.5；类别: 模型实现调整；主要 diff: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py`；技术摘要: 覆盖「Parakeet nemotron encoder」；主要实现面是 `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`, `python/sglang/srt/multimodal/internvl_utils.py`, `python/sglang/srt/models/radio.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic，涉及 `__init__, preprocess_image, render_image`；`python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0 (239 lines); hunks: -1,4 +1,6; -113,3 +115,240 @@ def image_to_pixel_values(; symbols: image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image, resize_image_to_pixels，涉及 `image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image`；`python/sglang/srt/models/radio.py` modified +135/-57 (192 lines); hunks: -13,6 +13,7; -33,6 +34,8; symbols: forward, ViTPatchGenerator, __init__，涉及 `forward, ViTPatchGenerator, __init__`；`python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle，涉及 `__init__, pad_input_ids, pixel_shuffle`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36 (358 lines); hunks: -11,23 +11,39; -63,18 +79,62 @@ def __init__(self, hf_config, server_args, _image_processor,...; symbols: __init__, preprocess_image, render_image, render_image_dynamic
  - `python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0 (239 lines); hunks: -1,4 +1,6; -113,3 +115,240 @@ def image_to_pixel_values(; symbols: image_to_pixel_values, compute_dynamic_image_size, dynamic_resize_image, resize_image_to_pixels
  - `python/sglang/srt/models/radio.py` modified +135/-57 (192 lines); hunks: -13,6 +13,7; -33,6 +34,8; symbols: forward, ViTPatchGenerator, __init__
  - `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20 (191 lines); hunks: -35,8 +35,10; -66,9 +68,13 @@ def __init__(; symbols: __init__, pad_input_ids, pixel_shuffle
  - `python/sglang/srt/models/parakeet.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: ParakeetProjection, __init__, forward, ProjectedParakeet
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/nano_nemotron_vl.py
@@ -11,23 +11,39 @@
+import logging
+import math
-from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
diff -- python/sglang/srt/multimodal/internvl_utils.py
@@ -1,4 +1,6 @@
+import math
@@ -113,3 +115,240 @@ def image_to_pixel_values(
+def compute_dynamic_image_size(
+    orig_w: int,
+    orig_h: int,
+    patch_size: int,
diff -- python/sglang/srt/models/radio.py
@@ -13,6 +13,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/nano_nemotron_vl.py` modified +322/-36; `python/sglang/srt/multimodal/internvl_utils.py` modified +239/-0; `python/sglang/srt/models/radio.py` modified +135/-57; `python/sglang/srt/models/nano_nemotron_vl.py` modified +171/-20; `python/sglang/srt/models/parakeet.py` added +182/-0; `python/sglang/srt/multimodal/audio_from_video.py` added +89/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/nano_nemotron_vl.py`, `python/sglang/srt/configs/parakeet.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- 链接: https://github.com/sgl-project/sglang/pull/25182
- 状态/时间: merged / 2026-05-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 136 个文件，+255/-0，可读 patch 872 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: add vLLM SPDX copyright headers to ported files」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`；技术摘要: 覆盖「chore: add vLLM SPDX copyright headers to ported files」；主要实现面是 `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7；`python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6；`python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6。
- 代码 diff 细节:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- 已读文件:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24751 - fix(mm): make multimodal data loading non-blocking to prevent health check stalls

- 链接: https://github.com/sgl-project/sglang/pull/24751
- 状态/时间: merged / 2026-05-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+45/-44，可读 patch 401 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；模型线: InternVL 3.5；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`；技术摘要: 覆盖「fix(mm): make multimodal data loading non-blocking to prevent health check stalls」；主要实现面是 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/internvl.py`, `python/sglang/srt/multimodal/processors/minicpm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data，涉及 `_process_loaded_mm_data, load_mm_data`；`python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async，涉及 `_process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async`；`python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async，涉及 `_process_special_format, process_mm_data_async`；`python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async，涉及 `__init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7 (15 lines); hunks: -1,3 +1,4; -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):; symbols: _process_loaded_mm_data, load_mm_data
  - `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3 (6 lines); hunks: -310,7 +310,7 @@ async def _process_special_format(; -423,7 +423,7 @@ async def process_qwen_mm_data_async(; symbols: _process_special_format, process_qwen_mm_data_async, process_internlm2_mm_data_async
  - `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2 (4 lines); hunks: -118,7 +118,7 @@ async def _process_special_format(; -190,7 +190,7 @@ async def process_mm_data_async(; symbols: _process_special_format, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1 (2 lines); hunks: -20,7 +20,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
  - `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1 (2 lines); hunks: -29,7 +29,7 @@ def __init__(self, hf_config, server_args, _processor, *args,...; symbols: __init__, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -1,3 +1,4 @@
+import asyncio
@@ -729,7 +730,7 @@ def _process_loaded_mm_data(self, modality, raw_data, result):
-    def load_mm_data(
+    async def load_mm_data(
@@ -772,7 +773,7 @@ def load_mm_data(
-            return self.legacy_load_mm_data(
diff -- python/sglang/srt/multimodal/processors/internvl.py
@@ -310,7 +310,7 @@ async def _process_special_format(
-            base_output = self.load_mm_data(
+            base_output = await self.load_mm_data(
@@ -423,7 +423,7 @@ async def process_qwen_mm_data_async(
-        base_output = self.load_mm_data(
+        base_output = await self.load_mm_data(
@@ -644,7 +644,7 @@ async def process_internlm2_mm_data_async(
diff -- python/sglang/srt/multimodal/processors/minicpm.py
@@ -118,7 +118,7 @@ async def _process_special_format(
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/base_processor.py` modified +8/-7; `python/sglang/srt/multimodal/processors/internvl.py` modified +3/-3; `python/sglang/srt/multimodal/processors/minicpm.py` modified +2/-2; `python/sglang/srt/multimodal/processors/clip.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +1/-1; `python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/clip.py`, `python/sglang/srt/multimodal/processors/deepseek_ocr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- 链接: https://github.com/sgl-project/sglang/pull/28567
- 状态/时间: merged / 2026-06-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 184 个文件，+1865/-1727，可读 patch 8932 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_parallel(): a structured accessor for parallel-topology state」；模型线: InternVL 3.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`；技术摘要: 覆盖「Add get_parallel(): a structured accessor for parallel-topology state」；主要实现面是 `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention，涉及 `ApertusMLP, __init__, forward`；`python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales，涉及 `__init__, forward, load_kv_cache_scales`；`python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__，涉及 `_resolve_moe_input_pad_multiple, __init__`；`python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- 验证与风险: diff 自带测试面 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
