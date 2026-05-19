# sglang DeepSeek OCR 模型 PR 优化历史

## 2026-05-19 新增覆盖

按 sglang 上游 `origin/main@5073c82a37`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `DeepSeek OCR` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/deepseek_ocr.md` | [#17897](https://github.com/sgl-project/sglang/pull/17897) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` | [#23337](https://github.com/sgl-project/sglang/pull/23337), [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `docs_new/docs/basic_usage/deepseek_ocr.mdx` | [#23337](https://github.com/sgl-project/sglang/pull/23337) |
| `docs_new/src/snippets/autoregressive/deepseek-ocr-deployment.jsx` | [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `python/sglang/srt/configs/deepseek_ocr.py` | [#20708](https://github.com/sgl-project/sglang/pull/20708), [#17897](https://github.com/sgl-project/sglang/pull/17897), [#12470](https://github.com/sgl-project/sglang/pull/12470), [#12415](https://github.com/sgl-project/sglang/pull/12415), [#12384](https://github.com/sgl-project/sglang/pull/12384), [#11891](https://github.com/sgl-project/sglang/pull/11891) |
| `python/sglang/srt/models/deepseek_ocr.py` | [#25182](https://github.com/sgl-project/sglang/pull/25182), [#12555](https://github.com/sgl-project/sglang/pull/12555), [#19732](https://github.com/sgl-project/sglang/pull/19732), [#18774](https://github.com/sgl-project/sglang/pull/18774), [#18860](https://github.com/sgl-project/sglang/pull/18860), [#17897](https://github.com/sgl-project/sglang/pull/17897), [#11891](https://github.com/sgl-project/sglang/pull/11891) |
| `python/sglang/srt/multimodal/processors/deepseek_ocr.py` | [#21738](https://github.com/sgl-project/sglang/pull/21738), [#17897](https://github.com/sgl-project/sglang/pull/17897), [#11891](https://github.com/sgl-project/sglang/pull/11891) |
| `test/srt/xpu/test_deepseek_ocr.py` | [#23820](https://github.com/sgl-project/sglang/pull/23820), [#23044](https://github.com/sgl-project/sglang/pull/23044), [#21735](https://github.com/sgl-project/sglang/pull/21735), [#13561](https://github.com/sgl-project/sglang/pull/13561) |
| `test/srt/xpu/test_deepseek_ocr_triton.py` | [#23820](https://github.com/sgl-project/sglang/pull/23820), [#23044](https://github.com/sgl-project/sglang/pull/23044), [#21735](https://github.com/sgl-project/sglang/pull/21735) |

## PR 覆盖总览

- git 追溯 PR 数: 18
- 关键词/补充 PR 数: 7
- 当前文档总 PR 数: 25
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-23 | [#11891](https://github.com/sgl-project/sglang/pull/11891) | merged | model: support deepseek-ocr | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-31 | [#12384](https://github.com/sgl-project/sglang/pull/12384) | merged | [Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p… | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py` |
| 2025-10-31 | [#12415](https://github.com/sgl-project/sglang/pull/12415) | merged | Feat: deepseek-ocr logits processor | `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py` |
| 2025-10-31 | [#12470](https://github.com/sgl-project/sglang/pull/12470) | merged | Fix lint in deepseek-ocr | `python/sglang/srt/configs/deepseek_ocr.py` |
| 2025-11-04 | [#12619](https://github.com/sgl-project/sglang/pull/12619) | open | [NPU] supports ds-ocr model on ascend | `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-01-30 | [#17897](https://github.com/sgl-project/sglang/pull/17897) | merged | Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833 | `docs/basic_usage/deepseek_ocr.md`, `docs/index.rst`, `docs/supported_models/multimodal_language_models.md` |
| 2026-02-05 | [#13561](https://github.com/sgl-project/sglang/pull/13561) | merged | [XPU] Integrate MoE and minor improvements in XPU attention backend | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py` |
| 2026-02-17 | [#18774](https://github.com/sgl-project/sglang/pull/18774) | merged | Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1 | `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-03-02 | [#19722](https://github.com/sgl-project/sglang/pull/19722) | open | fix: align DeepSeek OCR vision dtypes | `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-03-11 | [#19732](https://github.com/sgl-project/sglang/pull/19732) | merged | [AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test | `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` |
| 2026-03-18 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | `benchmark/mmmu/bench_sglang.py`, `benchmark/mmmu/eval_utils.py`, `python/sglang/srt/configs/deepseek_ocr.py` |
| 2026-03-19 | [#12555](https://github.com/sgl-project/sglang/pull/12555) | merged | [CPU] Fix MoE layer support for DeepSeek-OCR models | `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py` |
| 2026-04-03 | [#21738](https://github.com/sgl-project/sglang/pull/21738) | merged | refactor: replace mm_inputs dict with MultimodalProcessorOutput | `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py` |
| 2026-04-04 | [#21735](https://github.com/sgl-project/sglang/pull/21735) | merged | fix ut test_moe | `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore` |
| 2026-04-21 | [#23044](https://github.com/sgl-project/sglang/pull/23044) | merged | [XPU] Fix DeepSeek-OCR tests under transformers 5.x | `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` |
| 2026-04-29 | [#23820](https://github.com/sgl-project/sglang/pull/23820) | merged | Update XPU Docker runtime stack & hf_home config | `.github/workflows/pr-test-xpu.yml`, `docker/xpu.Dockerfile`, `test/srt/xpu/test_deepseek_ocr.py` |
| 2026-05-08 | [#24701](https://github.com/sgl-project/sglang/pull/24701) | open | [FIX][1/2] fix step3-vl/deepseek-ocr image processor error | `python/sglang/srt/multimodal/processors/step3_vl.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `benchmark/hicache/bench_serving.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` |
| 2026-05-14 | [#25257](https://github.com/sgl-project/sglang/pull/25257) | open | [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2 | `python/sglang/srt/models/deepseek.py` |
| 2026-05-15 | [#25364](https://github.com/sgl-project/sglang/pull/25364) | open | Add Accuracy Benchmark for OCR models | `.codespellrc`, `benchmark/ocr/README.md`, `benchmark/ocr/bench_sglang.py` |
| 2026-05-15 | [#25403](https://github.com/sgl-project/sglang/pull/25403) | open | [FIX][2/2] fix step3-vl/deepseek-ocr image processor error | `python/sglang/srt/configs/deepseek_ocr.py` |
| 2026-05-18 | [#25589](https://github.com/sgl-project/sglang/pull/25589) | open | Use hf_transformers_utils.get_processor to load model | `python/sglang/benchmark/utils.py` |

## 逐 PR diff 审计卡

### PR #11891 - model: support deepseek-ocr

- 链接: https://github.com/sgl-project/sglang/pull/11891
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+2125/-117，可读 patch 2504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support deepseek-ocr」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/model_config.py`。
- 实现要点:
  - `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0；symbols: ImageTransform, __init__, __call__, VisionEncoderConfig
  - `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95；symbols: find_closest_aspect_ratio, dynamic_preprocess, format_messages_v2
  - `python/sglang/srt/configs/model_config.py` modified +1/-0
  - `python/sglang/srt/model_loader/utils.py` modified +0/-1
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0
  - `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95
  - `python/sglang/srt/configs/model_config.py` modified +1/-0
  - `python/sglang/srt/model_loader/utils.py` modified +0/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -0,0 +1,262 @@
+from typing import Tuple
+
+import torchvision.transforms as T
+from PIL import Image
+from transformers import PretrainedConfig
+
+BASE_SIZE = 1024
+IMAGE_SIZE = 640
+CROP_MODE = True
+MIN_CROPS = 2
+MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.
+MAX_CONCURRENCY = 100  # If you have limited GPU memory, lower the concurrency count.
+NUM_WORKERS = 64  # image pre-process (resize/padding) workers
diff -- python/sglang/srt/configs/deepseekvl2.py
@@ -11,6 +11,8 @@
     ProcessorMixin,
 )

+from sglang.srt.configs.deepseek_ocr import BASE_SIZE, IMAGE_SIZE, MAX_CROPS, MIN_CROPS
+

 def select_best_resolution(image_size, candidate_resolutions):
     # used for cropping
@@ -61,6 +63,7 @@ def __setitem__(self, key, value):
 class VLChatProcessorOutput(DictOutput):
     input_ids: torch.LongTensor
     target_ids: torch.LongTensor
+    images_crop: torch.LongTensor
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` added +262/-0; `python/sglang/srt/configs/deepseekvl2.py` modified +194/-95; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/model_loader/utils.py` modified +0/-1; `python/sglang/srt/models/deepseek_ocr.py` added +1516/-0; `python/sglang/srt/models/deepseek_v2.py` modified +0/-1; `python/sglang/srt/multimodal/processors/base_processor.py` modified +1/-0; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` added +37/-0
  - tests: `test/srt/test_vision_openai_server_a.py` modified +56/-0; `test/srt/test_vision_openai_server_common.py` modified +6/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #12384 - [Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…

- 链接: https://github.com/sgl-project/sglang/pull/12384
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+683/-216，可读 patch 1133 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix]: distinguish processors for deepseek_vl2 and deepseek_ocr to p…」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/multimodal/customized_mm_processor_utils.py`。
- 实现要点:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10；symbols: DictOutput, items, keys, __getitem__
  - `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194；symbols: format_messages_v2
  - `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0；symbols: register_customized_processor, MyModelConfig, decorator
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12；symbols: _is_deepseek_ocr_model
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10
  - `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194
  - `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -1,8 +1,19 @@
-from typing import Tuple
-
-import torchvision.transforms as T
-from PIL import Image
-from transformers import PretrainedConfig
+import math
+from dataclasses import dataclass
+from typing import Any, Dict, List, Optional, Tuple
+
+import torch
+from PIL import Image, ImageOps
+from transformers import (
+    AutoProcessor,
diff -- python/sglang/srt/configs/deepseekvl2.py
@@ -11,8 +11,6 @@
     ProcessorMixin,
 )

-from sglang.srt.configs.deepseek_ocr import BASE_SIZE, IMAGE_SIZE, MAX_CROPS, MIN_CROPS
-

 def select_best_resolution(image_size, candidate_resolutions):
     # used for cropping
@@ -63,7 +61,6 @@ def __setitem__(self, key, value):
 class VLChatProcessorOutput(DictOutput):
     input_ids: torch.LongTensor
     target_ids: torch.LongTensor
-    images_crop: torch.LongTensor
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +521/-10; `python/sglang/srt/configs/deepseekvl2.py` modified +95/-194; `python/sglang/srt/multimodal/customized_mm_processor_utils.py` added +35/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +32/-12
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #12415 - Feat: deepseek-ocr logits processor

- 链接: https://github.com/sgl-project/sglang/pull/12415
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+89/-1，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat: deepseek-ocr logits processor」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/sampling/custom_logit_processor.py`。
- 实现要点:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0；symbols: get_default_ngram_custom_params
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1；symbols: DeepseekOCRNoRepeatNGramLogitProcessor, __call__
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -15,6 +15,10 @@
     register_customized_processor,
 )

+from sglang.srt.sampling.custom_logit_processor import (
+    DeepseekOCRNoRepeatNGramLogitProcessor,
+)
+
 BASE_SIZE = 1024
 IMAGE_SIZE = 640
 CROP_MODE = True
@@ -26,6 +30,24 @@
 SKIP_REPEAT = True
 MODEL_PATH = "deepseek-ai/DeepSeek-OCR"  # change to your model path
diff -- python/sglang/srt/sampling/custom_logit_processor.py
@@ -1,7 +1,7 @@
 import json
 from abc import ABC, abstractmethod
 from functools import lru_cache
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

 import dill
 import orjson
@@ -126,3 +126,69 @@ class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
     THINKING_START_TOKEN_ID: int = 128798
     THINKING_END_TOKEN_ID: int = 128799
     NEW_LINE_TOKEN_ID: int = 201
+
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +22/-0; `python/sglang/srt/sampling/custom_logit_processor.py` modified +67/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #12470 - Fix lint in deepseek-ocr

- 链接: https://github.com/sgl-project/sglang/pull/12470
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix lint in deepseek-ocr」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -14,7 +14,6 @@
 from sglang.srt.multimodal.customized_mm_processor_utils import (
     register_customized_processor,
 )
-
 from sglang.srt.sampling.custom_logit_processor import (
     DeepseekOCRNoRepeatNGramLogitProcessor,
 )
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +0/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #12619 - [NPU] supports ds-ocr model on ascend

- 链接: https://github.com/sgl-project/sglang/pull/12619
- 状态/时间: open / 2025-11-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+200/-60，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] supports ds-ocr model on ascend」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek.py` modified +142/-49；symbols: get_moe_weights, start_layer, end_layer, get_model_config_for_expert_location
  - `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +142/-49
  - `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -23,11 +23,14 @@
 from transformers import PretrainedConfig

 from sglang.srt.distributed import (
+    get_pp_group,
     get_tensor_model_parallel_rank,
     get_tensor_model_parallel_world_size,
     tensor_model_parallel_all_reduce,
 )
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
 from sglang.srt.layers.activation import SiluAndMul
+from sglang.srt.layers.dp_attention import is_dp_attention_enabled
 from sglang.srt.layers.layernorm import RMSNorm
 from sglang.srt.layers.linear import (
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -30,6 +30,7 @@
 from transformers.models.vitdet.modeling_vitdet import get_rel_pos

 from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
 from sglang.srt.layers.quantization import QuantizationConfig
 from sglang.srt.managers.mm_utils import (
     MultiModalityDataPaddingPatternMultimodalTokens,
@@ -1770,6 +1771,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
             (".gate_up_proj", ".up_proj", 1),
         ]

+        expert_params_mapping = FusedMoE.make_expert_params_mapping(
+            ckpt_gate_proj_name="gate_proj",
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +142/-49; `python/sglang/srt/models/deepseek_ocr.py` modified +58/-11
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #17897 - Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833

- 链接: https://github.com/sgl-project/sglang/pull/17897
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+618/-140，可读 patch 1057 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek-OCR-2 in SGLang (OCR2 vision pipeline, tokenization alignment, and weight loading fixes)#17833」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `docs/basic_usage/deepseek_ocr.md`, `docs/index.rst`, `docs/supported_models/multimodal_language_models.md`。
- 实现要点:
  - `docs/basic_usage/deepseek_ocr.md` added +54/-0
  - `docs/index.rst` modified +1/-0
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9
- 代码 diff 细节:
  - `docs/basic_usage/deepseek_ocr.md` added +54/-0
  - `docs/index.rst` modified +1/-0
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9
- 关键代码摘录:

```diff
diff -- docs/basic_usage/deepseek_ocr.md
@@ -0,0 +1,54 @@
+# DeepSeek OCR (OCR-1 / OCR-2)
+
+DeepSeek OCR models are multimodal (image + text) models for OCR and document understanding.
+
+## Launch server
+
+```shell
+python -m sglang.launch_server \
+  --model-path deepseek-ai/DeepSeek-OCR-2 \
+  --trust-remote-code \
+  --host 0.0.0.0 \
+  --port 30000
+```
diff -- docs/index.rst
@@ -33,6 +33,7 @@ Its core features include:
    basic_usage/ollama_api.md
    basic_usage/offline_engine_api.ipynb
    basic_usage/native_api.ipynb
+   basic_usage/deepseek_ocr.md
    basic_usage/sampling_params.md
    basic_usage/popular_model_usage.rst

```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +32/-9; `python/sglang/srt/configs/model_config.py` modified +9/-4; `python/sglang/srt/model_loader/utils.py` modified +6/-2; `python/sglang/srt/models/deepseek_ocr.py` modified +446/-116; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +8/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +61/-9
  - docs/bench: `docs/basic_usage/deepseek_ocr.md` added +54/-0; `docs/index.rst` modified +1/-0; `docs/supported_models/multimodal_language_models.md` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #13561 - [XPU] Integrate MoE and minor improvements in XPU attention backend

- 链接: https://github.com/sgl-project/sglang/pull/13561
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+233/-7，可读 patch 372 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Integrate MoE and minor improvements in XPU attention backend」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`。
- 实现要点:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2
  - `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-0
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2
  - `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -20,6 +20,8 @@
     is_cpu,
     is_cuda,
     is_hip,
+    is_xpu,
+    use_intel_xpu_backend,
 )
 from sglang.srt.utils.custom_op import register_custom_op

@@ -40,6 +42,8 @@
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_cpu = is_cpu()
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+_is_xpu = is_xpu()
diff -- python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py
@@ -5,12 +5,13 @@
 import torch
 import triton

-from sglang.srt.utils import is_cuda, is_hip
+from sglang.srt.utils import is_cuda, is_hip, is_xpu

 _is_cuda = is_cuda()
 _is_hip = is_hip()
+_is_xpu = is_xpu()

-if _is_cuda or _is_hip:
+if _is_cuda or _is_hip or _is_xpu:
     from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
```
- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +34/-1; `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +3/-2; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +13/-3; `python/sglang/srt/layers/moe/topk.py` modified +1/-0; `python/sglang/srt/layers/quantization/unquant.py` modified +49/-0; `python/sglang/srt/utils/common.py` modified +11/-1
  - tests: `test/srt/run_suite.py` modified +1/-0; `test/srt/xpu/test_deepseek_ocr.py` added +121/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18860 - update pre-commit config

- 链接: https://github.com/sgl-project/sglang/pull/18860
- 状态/时间: merged / 2026-02-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+170/-159，可读 patch 1254 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update pre-commit config」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py`。
- 实现要点:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- 代码 diff 细节:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/lint.yml
@@ -26,9 +26,9 @@ jobs:
         run: SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure

       - name: Run sgl-kernel clang-format checks
-        uses: DoozyX/clang-format-lint-action@v0.18.1
+        uses: DoozyX/clang-format-lint-action@v0.20
         with:
           source: sgl-kernel
           extensions: h,c,cpp,hpp,cu,cuh,cc
-          clangFormatVersion: 18
+          clangFormatVersion: 20
           style: file
diff -- .pre-commit-config.yaml
@@ -3,7 +3,7 @@ exclude: ^(python/sglang/multimodal_gen/csrc|python/sglang/jit_kernel/flash_atte

 repos:
   - repo: https://github.com/pre-commit/pre-commit-hooks
-    rev: v5.0.0
+    rev: v6.0.0
     hooks:
       - id: check-symlinks
       - id: destroyed-symlinks
@@ -21,12 +21,12 @@ repos:
       - id: debug-statements
       - id: no-commit-to-branch
   - repo: https://github.com/PyCQA/isort
-    rev: 5.13.2
```
- 已读文件:
  - runtime: `python/sglang/jit_kernel/include/sgl_kernel/type.cuh` modified +16/-15; `python/sglang/multimodal_gen/apps/webui/main.py` modified +2/-4; `python/sglang/multimodal_gen/configs/models/encoders/qwen3.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/cache/__init__.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py` modified +2/-1; `python/sglang/multimodal_gen/runtime/layers/activation.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/layernorm.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/rotary_embedding.py` modified +1/-0
  - docs/bench: `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1; `benchmark/tip_suggestion/bench_other.py` modified +2/-8; `benchmark/tip_suggestion/bench_sglang.py` modified +2/-8; `benchmark/tip_suggestion/lmql_funcs.py` modified +2/-8; `docs/advanced_features/lora.ipynb` modified +10/-20; `docs/advanced_features/structured_outputs.ipynb` modified +0/-1; `docs/advanced_features/structured_outputs_for_reasoning_models.ipynb` modified +0/-1; `docs/advanced_features/vlm_query.ipynb` modified +0/-1
  - other: `.github/workflows/lint.yml` modified +2/-2; `.pre-commit-config.yaml` modified +6/-6; `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18774 - Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1

- 链接: https://github.com/sgl-project/sglang/pull/18774
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-1，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Adapt the Qwen2Model._update_causal_mask for transformers==4.57.1」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1216,9 +1216,18 @@ def forward(
                 cache_position=None,
             ):
                 self._current_token_type_ids = token_type_ids
+                causal_mask_mapping = {
+                    "full_attention": self._update_causal_mask(
+                        attention_mask,
+                        inputs_embeds,
+                        cache_position,
+                        past_key_values,
+                        output_attentions,
+                    )
+                }
                 return super().forward(
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +10/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #19722 - fix: align DeepSeek OCR vision dtypes

- 链接: https://github.com/sgl-project/sglang/pull/19722
- 状态/时间: open / 2026-03-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-6，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: align DeepSeek OCR vision dtypes」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -1509,18 +1509,26 @@ def _collect_mm_flag(
         return values

     def _encode_ocr2_features(self, images: torch.Tensor) -> torch.Tensor:
+        sam_dtype = next(self.sam_model.parameters()).dtype
+        projector_dtype = next(self.projector.parameters()).dtype
+        images = images.to(dtype=sam_dtype)
         features = self.sam_model(images)
         features = self.qwen2_model(features)
+        features = features.to(dtype=projector_dtype)
         features = self.projector(features)
         return features.view(-1, features.shape[-1])

     def _encode_ocr1_features(self, images: torch.Tensor) -> torch.Tensor:
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +14/-6
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #19732 - [AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test

- 链接: https://github.com/sgl-project/sglang/pull/19732
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+23/-5，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [DeepSeek-OCR-2 Day 0] Enable DeepSeek-OCR-2 on AMD GPUs and add nightly test」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_ocr.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +2/-0
  - `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` modified +17/-2
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +2/-0
  - `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` modified +17/-2
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -125,8 +125,9 @@ def isin_list(
     elements: torch.Tensor,
     test_elements_list: list[int],
 ) -> torch.Tensor:
-    test_elements = torch.tensor(test_elements_list, pin_memory=True).to(
-        device=elements.device, non_blocking=True
+    use_pin = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
+    test_elements = torch.tensor(test_elements_list, pin_memory=use_pin).to(
+        device=elements.device, non_blocking=use_pin
     )

     return torch.isin(elements, test_elements)
@@ -1685,7 +1686,7 @@ def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tens

diff -- python/sglang/srt/multimodal/processors/base_processor.py
@@ -210,6 +210,8 @@ def __init__(
             "image_emb_mask": Modality.IMAGE,
             "images_spatial_crop": Modality.IMAGE,
             "images_crop": Modality.IMAGE,
+            "has_local_crops": Modality.IMAGE,
+            "has_images": Modality.IMAGE,
             "tgt_size": Modality.IMAGE,
             "image_grid_hws": Modality.IMAGE,
             "aspect_ratio_ids": Modality.IMAGE,
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_ocr.py` modified +4/-3; `python/sglang/srt/multimodal/processors/base_processor.py` modified +2/-0
  - tests: `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` modified +17/-2
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- 链接: https://github.com/sgl-project/sglang/pull/20708
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+360/-124，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Small 4 (Pixtral) support」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `benchmark/mmmu/bench_sglang.py`, `benchmark/mmmu/eval_utils.py`, `python/sglang/srt/configs/deepseek_ocr.py`。
- 实现要点:
  - `benchmark/mmmu/bench_sglang.py` modified +49/-10
  - `benchmark/mmmu/eval_utils.py` modified +8/-0
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +2/-2
  - `python/sglang/srt/configs/deepseekvl2.py` modified +3/-3
- 代码 diff 细节:
  - `benchmark/mmmu/bench_sglang.py` modified +49/-10
  - `benchmark/mmmu/eval_utils.py` modified +8/-0
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +2/-2
  - `python/sglang/srt/configs/deepseekvl2.py` modified +3/-3
- 关键代码摘录:

```diff
diff -- benchmark/mmmu/bench_sglang.py
@@ -11,11 +11,14 @@

 import argparse
 import asyncio
+import base64
+import mimetypes
 import re
 import sys
 import time
 import traceback
 from dataclasses import dataclass, field
+from pathlib import Path
 from typing import Any, List, Optional, Tuple

diff -- benchmark/mmmu/eval_utils.py
@@ -40,6 +40,7 @@ class EvalArgs:
     temperature: Optional[float] = None
     response_answer_regex: str = "(.*)"
     lora_path: Optional[str] = None
+    reasoning_effort: Optional[str] = None

     @staticmethod
     def add_cli_args(parser: argparse.ArgumentParser):
@@ -120,6 +121,13 @@ def add_cli_args(parser: argparse.ArgumentParser):
             default=EvalArgs.lora_path,
             help="Specify the LoRA path to use for evaluation. If specified, the value will be specified in the body of every request as `lora-path`.",
         )
+        parser.add_argument(
+            "--reasoning-effort",
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +2/-2; `python/sglang/srt/configs/deepseekvl2.py` modified +3/-3; `python/sglang/srt/configs/janus_pro.py` modified +12/-12; `python/sglang/srt/configs/jet_nemotron.py` modified +12/-12; `python/sglang/srt/entrypoints/openai/protocol.py` modified +1/-1; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13; `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +1/-1
  - docs/bench: `benchmark/mmmu/bench_sglang.py` modified +49/-10; `benchmark/mmmu/eval_utils.py` modified +8/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #12555 - [CPU] Fix MoE layer support for DeepSeek-OCR models

- 链接: https://github.com/sgl-project/sglang/pull/12555
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+65/-10，可读 patch 110 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Fix MoE layer support for DeepSeek-OCR models」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek.py` modified +31/-9
  - `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1；symbols: post_load_weights
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +31/-9
  - `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -48,7 +48,12 @@
 )
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.model_loader.weight_utils import default_weight_loader
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
+
+_is_cpu_amx_available = cpu_has_amx_support()
+_is_cpu = is_cpu()
+if _is_cpu and _is_cpu_amx_available:
+    import sgl_kernel  # noqa: F401


 class DeepseekMLP(nn.Module):
diff -- python/sglang/srt/models/deepseek_ocr.py
@@ -41,6 +41,10 @@
 from sglang.srt.models.deepseek import DeepseekForCausalLM
 from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
 from sglang.srt.models.transformers import maybe_prefix
+from sglang.srt.utils import cpu_has_amx_support, is_cpu
+
+_is_cpu_amx_available = cpu_has_amx_support()
+_is_cpu = is_cpu()

 NestedTensors: TypeAlias = Union[
     list["NestedTensors"],
@@ -1772,7 +1776,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

         params_dict = dict(self.named_parameters())
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +31/-9; `python/sglang/srt/models/deepseek_ocr.py` modified +34/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #21738 - refactor: replace mm_inputs dict with MultimodalProcessorOutput

- 链接: https://github.com/sgl-project/sglang/pull/21738
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 40 个文件，+408/-314，可读 patch 1321 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor: replace mm_inputs dict with MultimodalProcessorOutput」；模型线: DeepSeek OCR；类别: 模型实现调整；主要 diff: `python/sglang/srt/disaggregation/encode_receiver.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/managers/io_struct.py`。
- 实现要点:
  - `python/sglang/srt/disaggregation/encode_receiver.py` modified +1/-1
  - `python/sglang/srt/disaggregation/encode_server.py` modified +1/-3
  - `python/sglang/srt/managers/io_struct.py` modified +1/-1
  - `python/sglang/srt/managers/mm_utils.py` modified +3/-4
- 代码 diff 细节:
  - `python/sglang/srt/disaggregation/encode_receiver.py` modified +1/-1
  - `python/sglang/srt/disaggregation/encode_server.py` modified +1/-3
  - `python/sglang/srt/managers/io_struct.py` modified +1/-1
  - `python/sglang/srt/managers/mm_utils.py` modified +3/-4
- 关键代码摘录:

```diff
diff -- python/sglang/srt/disaggregation/encode_receiver.py
@@ -532,7 +532,7 @@ def _try_recv_mm_data(self):
             **self.recv_embedding_data.get_mm_extra_meta(),
         )
         self.recv_req.mm_inputs = mm_inputs
-        self.recv_req.input_ids = mm_inputs["input_ids"]
+        self.recv_req.input_ids = mm_inputs.input_ids
         self.status = WaitingImageRequestStatus.SUCCESS
         self.recv_socket.close()

diff -- python/sglang/srt/disaggregation/encode_server.py
@@ -673,6 +673,7 @@ async def encode_with_global_cache(
         part_idx: int,
         hashes: Optional[List[str]] = None,
     ) -> torch.Tensor:
+        # mm_inputs: dict
         mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
         grid_thw = _get_mm_grid_dim(mm_inputs, modality)
         mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
@@ -853,7 +854,6 @@ async def _process_mm_items(self, mm_items, modality):
             images = await self._flatten_and_load_images(mm_items)
             image_config = self.vision_config.get("image", {})
             processor_input = self.image_processor(images=images, **image_config)
-            feature = processor_input["pixel_values"]
             if hasattr(self.model, "thinker"):  # for omni models
```
- 已读文件:
  - runtime: `python/sglang/srt/disaggregation/encode_receiver.py` modified +1/-1; `python/sglang/srt/disaggregation/encode_server.py` modified +1/-3; `python/sglang/srt/managers/io_struct.py` modified +1/-1; `python/sglang/srt/managers/mm_utils.py` modified +3/-4; `python/sglang/srt/managers/schedule_batch.py` modified +58/-4; `python/sglang/srt/managers/scheduler.py` modified +7/-7; `python/sglang/srt/managers/session_controller.py` modified +1/-1; `python/sglang/srt/managers/tokenizer_manager.py` modified +9/-9
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #21735 - fix ut test_moe

- 链接: https://github.com/sgl-project/sglang/pull/21735
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+105/-32，可读 patch 214 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix ut test_moe」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `test/srt/run_suite.py`, `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`。
- 实现要点:
  - `test/srt/run_suite.py` modified +2/-1
  - `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26；symbols: _cleanup_xpu_memory
  - `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0；symbols: TestDeepSeekOCRTriton, setUpClass
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5；symbols: _cleanup_xpu_memory
- 代码 diff 细节:
  - `test/srt/run_suite.py` modified +2/-1
  - `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26
  - `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0
  - `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5
- 关键代码摘录:

```diff
diff -- test/srt/run_suite.py
@@ -77,7 +77,8 @@
 # NOTE: please sort the test cases alphabetically by the test file name
 suite_xpu = {
     "per-commit-xpu": [
-        TestFile("xpu/test_deepseek_ocr.py"),
+        TestFile("xpu/test_deepseek_ocr.py", 360),
+        TestFile("xpu/test_deepseek_ocr_triton.py", 360),
         # TestFile("xpu/test_internvl.py"),
         TestFile("xpu/test_intel_xpu_backend.py"),
     ],
diff -- test/srt/xpu/test_deepseek_ocr.py
@@ -2,9 +2,11 @@
 python3 -m unittest test_deepseek_ocr.py
 """

+import gc
 import json
 import os
 import unittest
+from pathlib import Path

 import requests
 from transformers import AutoTokenizer
@@ -19,11 +21,32 @@

```
- 已读文件:
  - tests: `test/srt/run_suite.py` modified +2/-1; `test/srt/xpu/test_deepseek_ocr.py` modified +28/-26; `test/srt/xpu/test_deepseek_ocr_triton.py` added +51/-0; `test/srt/xpu/test_intel_xpu_backend.py` modified +24/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+19458/-0，可读 patch 19508 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`。
- 实现要点:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 代码 diff 细节:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 关键代码摘录:

```diff
diff -- .gitignore
@@ -192,6 +192,7 @@ work_dirs/
 *.csv

 !logo.png
+!docs_new/images/*.png

 # Prerequisites
 *.d
diff -- docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml
@@ -0,0 +1,39 @@
+name: Sync LMSYS SGLang blogs
+
+on:
+  workflow_dispatch:
+  schedule:
+    - cron: "0 */12 * * *"
+
+permissions:
+  contents: write
+
+jobs:
+  sync:
+    runs-on: ubuntu-latest
```
- 已读文件:
  - docs/bench: `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0; `docs_new/.gitignore` added +30/-0; `docs_new/.mintignore` added +7/-0; `docs_new/AGENTS.md` added +381/-0; `docs_new/CONTRIBUTING.md` added +34/-0; `docs_new/LICENSE` added +201/-0; `docs_new/README.md` added +126/-0; `docs_new/cards/Autoregressive-benchmark-card.png` added +0/-0
  - other: `.gitignore` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23044 - [XPU] Fix DeepSeek-OCR tests under transformers 5.x

- 链接: https://github.com/sgl-project/sglang/pull/23044
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-8，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[XPU] Fix DeepSeek-OCR tests under transformers 5.x」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `test/srt/xpu/test_deepseek_ocr.py`, `test/srt/xpu/test_deepseek_ocr_triton.py`。
- 实现要点:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4
- 代码 diff 细节:
  - `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4
- 关键代码摘录:

```diff
diff -- test/srt/xpu/test_deepseek_ocr.py
@@ -9,9 +9,9 @@
 from pathlib import Path

 import requests
-from transformers import AutoTokenizer

 from sglang.srt.utils import kill_process_tree
+from sglang.srt.utils.hf_transformers import get_tokenizer
 from sglang.test.test_utils import (
     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
     DEFAULT_URL_FOR_TEST,
@@ -38,9 +38,7 @@ def _cleanup_xpu_memory(cls):
     def setUpClass(cls):
         cls._cleanup_xpu_memory()
diff -- test/srt/xpu/test_deepseek_ocr_triton.py
@@ -7,8 +7,8 @@
 from pathlib import Path

 import test_deepseek_ocr as deepseek_ocr
-from transformers import AutoTokenizer

+from sglang.srt.utils.hf_transformers import get_tokenizer
 from sglang.test.test_utils import (
     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
     DEFAULT_URL_FOR_TEST,
@@ -21,9 +21,7 @@ class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):
     def setUpClass(cls):
         cls._cleanup_xpu_memory()
         cls.model = "deepseek-ai/DeepSeek-OCR"
```
- 已读文件:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +2/-4; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +2/-4
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- 链接: https://github.com/sgl-project/sglang/pull/23337
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+3881/-1454，可读 patch 7195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`。
- 实现要点:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 代码 diff 细节:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 关键代码摘录:

```diff
diff -- .pre-commit-config.yaml
@@ -93,6 +93,13 @@ repos:
         language: system
         files: ^test/registered/.*\.py$
         pass_filenames: false
+      - id: check-no-docs-changes
+        name: reject changes under legacy docs/
+        entry: python3 scripts/ci/check_no_docs_changes.py
+        language: system
+        pass_filenames: false
+        always_run: true
+        stages: [pre-commit]
   - repo: https://github.com/lycheeverse/lychee.git
     rev: lychee-v0.22.0
     hooks:
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx
@@ -26,7 +26,7 @@ To use DeepSeek-Math-V2, you must agree to DeepSeek's Community License. See [LI

 ## 2. SGLang Installation

-Please refer to the [official SGLang installation guide](../../../docs/get-started/installation) for installation instructions.
+Please refer to the [official SGLang installation guide](../../../docs/get-started/install) for installation instructions.

 ## 3. Model Deployment

```
- 已读文件:
  - docs/bench: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/Ernie/Ernie4.5.mdx` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +7/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23820 - Update XPU Docker runtime stack & hf_home config

- 链接: https://github.com/sgl-project/sglang/pull/23820
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+38/-54，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update XPU Docker runtime stack & hf_home config」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `.github/workflows/pr-test-xpu.yml`, `docker/xpu.Dockerfile`, `test/srt/xpu/test_deepseek_ocr.py`。
- 实现要点:
  - `.github/workflows/pr-test-xpu.yml` modified +3/-3
  - `docker/xpu.Dockerfile` modified +10/-6
  - `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3；symbols: TestDeepSeekOCRTriton
- 代码 diff 细节:
  - `.github/workflows/pr-test-xpu.yml` modified +3/-3
  - `docker/xpu.Dockerfile` modified +10/-6
  - `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17
  - `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3
- 关键代码摘录:

```diff
diff -- .github/workflows/pr-test-xpu.yml
@@ -72,8 +72,6 @@ jobs:
     needs: [check-changes, pr-gate]
     if: needs.check-changes.outputs.main_package == 'true'
     runs-on: intel-bmg
-    env:
-      HF_HOME: /home/sdp/.cache/huggingface
     steps:
       - name: Checkout code
         uses: actions/checkout@v4
@@ -99,8 +97,10 @@ jobs:
           container_id=$(docker run -dt \
             --group-add 992 \
             --group-add $(getent group video | cut -d: -f3) \
-            -v ${HF_HOME}:/root/.cache/huggingface \
diff -- docker/xpu.Dockerfile
@@ -20,6 +20,16 @@ ARG SG_LANG_KERNEL_BRANCH=main
 RUN useradd -m -d /home/sdp -s /bin/bash sdp && \
     chown -R sdp:sdp /home/sdp

+USER root
+
+# Install the latest UMD driver for SYCL-TLA
+RUN apt-get install -y software-properties-common && \
+    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
+    apt-get update  && \
+    apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc && \
+    apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo && \
+    apt-get install -y libze-dev intel-ocloc
+
```
- 已读文件:
  - tests: `test/srt/xpu/test_deepseek_ocr.py` modified +6/-17; `test/srt/xpu/test_deepseek_ocr_triton.py` modified +7/-3; `test/srt/xpu/test_intel_xpu_backend.py` modified +12/-25
  - other: `.github/workflows/pr-test-xpu.yml` modified +3/-3; `docker/xpu.Dockerfile` modified +10/-6
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #24701 - [FIX][1/2] fix step3-vl/deepseek-ocr image processor error

- 链接: https://github.com/sgl-project/sglang/pull/24701
- 状态/时间: open / 2026-05-08
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+67/-20，可读 patch 160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FIX][1/2] fix step3-vl/deepseek-ocr image processor error」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/step3_vl.py`。
- 实现要点:
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20；symbols: forward, get_image_size, square_pad, resize
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/step3_vl.py
@@ -8,6 +8,7 @@
 from PIL import Image
 from torchvision import transforms
 from torchvision.transforms import InterpolationMode
+from torchvision.transforms import functional as F
 from transformers import BatchFeature, ProcessorMixin, TensorType

 from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
@@ -20,14 +21,37 @@
     MultimodalSpecialTokens,
 )

-ImageWithPatches = tuple[Image.Image, list[Image.Image], list[int] | None]
+Step3Image = Union[Image.Image, torch.Tensor]
```
- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/step3_vl.py` modified +67/-20
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- 链接: https://github.com/sgl-project/sglang/pull/25182
- 状态/时间: merged / 2026-05-13
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+185/-0，可读 patch 649 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「chore: add vLLM SPDX copyright headers to ported files」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `benchmark/hicache/bench_serving.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`。
- 实现要点:
  - `benchmark/hicache/bench_serving.py` modified +2/-0
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-0
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +2/-0
  - `python/sglang/bench_serving.py` modified +2/-0
- 代码 diff 细节:
  - `benchmark/hicache/bench_serving.py` modified +2/-0
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-0
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +2/-0
  - `python/sglang/bench_serving.py` modified +2/-0
- 关键代码摘录:

```diff
diff -- benchmark/hicache/bench_serving.py
@@ -1,3 +1,5 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 # Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
 # Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
@@ -1,3 +1,5 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
 import argparse
 import time
```
- 已读文件:
  - runtime: `python/sglang/bench_serving.py` modified +2/-0; `python/sglang/multimodal_gen/runtime/distributed/communication_op.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/device_communicators/base_device_communicator.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/device_communicators/cpu_communicator.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/device_communicators/cuda_communicator.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/device_communicators/pynccl.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/device_communicators/pynccl_wrapper.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py` modified +2/-0
  - docs/bench: `benchmark/hicache/bench_serving.py` modified +2/-0; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +2/-0; `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +2/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25257 - [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2

- 链接: https://github.com/sgl-project/sglang/pull/25257
- 状态/时间: open / 2026-05-14
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-3，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek.py`。
- 实现要点:
  - `python/sglang/srt/models/deepseek.py` modified +11/-3
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek.py` modified +11/-3
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek.py
@@ -39,7 +39,6 @@
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
 from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
-from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
 from sglang.srt.layers.moe.topk import TopK
 from sglang.srt.layers.quantization.base_config import QuantizationConfig
 from sglang.srt.layers.radix_attention import RadixAttention
@@ -50,14 +49,23 @@
 )
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.model_loader.weight_utils import default_weight_loader
-from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
+from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu, is_npu
```
- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek.py` modified +11/-3
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #25364 - Add Accuracy Benchmark for OCR models

- 链接: https://github.com/sgl-project/sglang/pull/25364
- 状态/时间: open / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1961/-2，可读 patch 2017 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Accuracy Benchmark for OCR models」；模型线: DeepSeek OCR；类别: 模型支持/运行时入口；主要 diff: `.codespellrc`, `benchmark/ocr/README.md`, `benchmark/ocr/bench_sglang.py`。
- 实现要点:
  - `.codespellrc` modified +1/-1
  - `benchmark/ocr/README.md` added +179/-0
  - `benchmark/ocr/bench_sglang.py` added +735/-0；symbols: BenchArgs, add_cli_args, preflight_check, pdf_page_to_base64_png
  - `benchmark/ocr/eval_utils.py` added +631/-0；symbols: normalize_text, strip_markdown, fuzzy_contains, exact_contains
- 代码 diff 细节:
  - `.codespellrc` modified +1/-1
  - `benchmark/ocr/README.md` added +179/-0
  - `benchmark/ocr/bench_sglang.py` added +735/-0
  - `benchmark/ocr/eval_utils.py` added +631/-0
- 关键代码摘录:

```diff
diff -- .codespellrc
@@ -1,3 +1,3 @@
 [codespell]
-ignore-words-list = ans, als, hel, boostrap, childs, te, vas, hsa, ment, cann, thi, makro, wil, rouge, PRIS, ather, MIS, medias, allready, inout, nd, fo, visibles, nothink, renderD, ond, tbe
+ignore-words-list = ans, als, hel, boostrap, childs, te, vas, hsa, ment, cann, thi, makro, wil, rouge, PRIS, ather, MIS, medias, allready, inout, nd, fo, visibles, nothink, renderD, ond, tbe, NED
 skip = *.json, *.jsonl, *.patch, *.txt, *.lock
diff -- benchmark/ocr/README.md
@@ -0,0 +1,179 @@
+# OCR Accuracy Benchmark
+
+Evaluates `deepseek-ai/DeepSeek-OCR-2` (and any compatible OCR VLM) on
+**olmOCR-bench** (AllenAI), the benchmark explicitly used in DeepSeek-OCR-2
+official evaluations.
+
+Targets **olmOCR-bench** because:
+- Public HuggingFace dataset with 7,010 deterministic unit tests
+- Explicitly cited by DeepSeek-OCR-2 authors
+- Clear pass/fail semantics — no heavy CDM/TEDS/LaTeXML dependencies
+- Covers 7 challenging document types across 1,403 PDF pages
+
+---
```
- 已读文件:
  - runtime: `python/pyproject.toml` modified +1/-0; `python/pyproject_xpu.toml` modified +1/-0; `python/sglang/srt/configs/deepseekvl2.py` modified +15/-1; `python/sglang/srt/multimodal/processors/deepseek_ocr.py` modified +16/-0
  - docs/bench: `benchmark/ocr/README.md` added +179/-0; `benchmark/ocr/bench_sglang.py` added +735/-0; `benchmark/ocr/eval_utils.py` added +631/-0; `benchmark/ocr/generate_report.py` added +382/-0
  - other: `.codespellrc` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25403 - [FIX][2/2] fix step3-vl/deepseek-ocr image processor error

- 链接: https://github.com/sgl-project/sglang/pull/25403
- 状态/时间: open / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+99/-12，可读 patch 183 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[FIX][2/2] fix step3-vl/deepseek-ocr image processor error」；模型线: DeepSeek OCR；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/deepseek_ocr.py`。
- 实现要点:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +99/-12；symbols: get_image_size, resize_image, crop_image, pad_image
- 代码 diff 细节:
  - `python/sglang/srt/configs/deepseek_ocr.py` modified +99/-12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/deepseek_ocr.py
@@ -1,9 +1,11 @@
 import math
 from dataclasses import dataclass
-from typing import Any, Dict, List, Optional, Tuple
+from typing import Any, Dict, List, Optional, Tuple, Union

 import torch
 from PIL import Image, ImageOps
+from torchvision.transforms import InterpolationMode
+from torchvision.transforms import functional as TF
 from transformers import (
     AutoProcessor,
     LlamaTokenizerFast,
@@ -18,6 +20,8 @@
```
- 已读文件:
  - runtime: `python/sglang/srt/configs/deepseek_ocr.py` modified +99/-12
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #25589 - Use hf_transformers_utils.get_processor to load model

- 链接: https://github.com/sgl-project/sglang/pull/25589
- 状态/时间: open / 2026-05-18
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-13，可读 patch 42 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use hf_transformers_utils.get_processor to load model」；模型线: DeepSeek OCR；类别: 文档/测试/CI；主要 diff: `python/sglang/benchmark/utils.py`。
- 实现要点:
  - `python/sglang/benchmark/utils.py` modified +8/-13
- 代码 diff 细节:
  - `python/sglang/benchmark/utils.py` modified +8/-13
- 关键代码摘录:

```diff
diff -- python/sglang/benchmark/utils.py
@@ -7,7 +7,6 @@
 import requests
 from tqdm.asyncio import tqdm
 from transformers import (
-    AutoProcessor,
     AutoTokenizer,
     PreTrainedTokenizer,
     PreTrainedTokenizerFast,
@@ -66,25 +65,21 @@ def get_tokenizer(

 def get_processor(
     pretrained_model_name_or_path: str,
-) -> AutoProcessor:
+):
```
- 已读文件:
  - runtime: `python/sglang/benchmark/utils.py` modified +8/-13
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。
