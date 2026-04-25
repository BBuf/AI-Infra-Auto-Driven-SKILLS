# sglang Mistral Small 4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Mistral/Devstral-2.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Mistral/Ministral-3.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Mistral/Mistral-Small-4.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/ministral-3-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/mistral-small-4-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/mistral_detector.py` | [#6597](https://github.com/sgl-project/sglang/pull/6597), [#14921](https://github.com/sgl-project/sglang/pull/14921), [#20708](https://github.com/sgl-project/sglang/pull/20708) |
| `python/sglang/srt/models/ministral3.py` | [#14251](https://github.com/sgl-project/sglang/pull/14251) |
| `python/sglang/srt/models/mistral.py` | [#108](https://github.com/sgl-project/sglang/pull/108), [#5099](https://github.com/sgl-project/sglang/pull/5099) |
| `python/sglang/srt/models/mistral_large_3.py` | [#14213](https://github.com/sgl-project/sglang/pull/14213), [#14466](https://github.com/sgl-project/sglang/pull/14466), [#14485](https://github.com/sgl-project/sglang/pull/14485) |
| `python/sglang/srt/models/mistral_large_3_eagle.py` | [#14466](https://github.com/sgl-project/sglang/pull/14466), [#14485](https://github.com/sgl-project/sglang/pull/14485), [#20708](https://github.com/sgl-project/sglang/pull/20708) |
| `python/sglang/srt/utils/hf_transformers/mistral_utils.py` | 无直接 PR 号提交 |
| `test/manual/models/test_mistral_large3_basic.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_mistral_large3.py` | [#15422](https://github.com/sgl-project/sglang/pull/15422), [#18065](https://github.com/sgl-project/sglang/pull/18065), [#19402](https://github.com/sgl-project/sglang/pull/19402) |
| `test/registered/ascend/llm_models/test_npu_mistral_7b.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_mistral_small_3_1_24b_instruct_2503.py` | 无直接 PR 号提交 |
| `test/registered/models/test_ministral3_models.py` | 无直接 PR 号提交 |
| `test/registered/models/test_ministral4_models.py` | [#21620](https://github.com/sgl-project/sglang/pull/21620) |
| `test/registered/unit/function_call/test_mistral_detector.py` | [#21399](https://github.com/sgl-project/sglang/pull/21399) |

## PR 覆盖总览

- git 追溯 PR 数: 14
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 15
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-01-26 | [#108](https://github.com/sgl-project/sglang/pull/108) | merged | Fix Mistral model loading | `python/sglang/srt/models/mistral.py` |
| 2025-05-17 | [#5099](https://github.com/sgl-project/sglang/pull/5099) | merged | model(vlm): mistral 3.1 | `python/sglang/srt/models/mistral.py` |
| 2025-05-26 | [#6597](https://github.com/sgl-project/sglang/pull/6597) | merged | feat: Improve Mistral and Qwen25 function call parsing | `python/sglang/srt/function_call/mistral_detector.py` |
| 2025-12-04 | [#14213](https://github.com/sgl-project/sglang/pull/14213) | merged | Add Mistral Large 3 support. | `python/sglang/srt/models/mistral_large_3.py` |
| 2025-12-04 | [#14251](https://github.com/sgl-project/sglang/pull/14251) | merged | ministral3 | `python/sglang/srt/models/ministral3.py` |
| 2025-12-05 | [#14466](https://github.com/sgl-project/sglang/pull/14466) | merged | Add Mistral Large 3 Eagle Support | `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/mistral_large_3.py` |
| 2025-12-12 | [#14921](https://github.com/sgl-project/sglang/pull/14921) | merged | update mistral detector | `python/sglang/srt/function_call/mistral_detector.py` |
| 2025-12-13 | [#14485](https://github.com/sgl-project/sglang/pull/14485) | merged | Mistral Large 3 NVFP4 support | `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py` |
| 2025-12-18 | [#15049](https://github.com/sgl-project/sglang/pull/15049) | merged | Mistral Large 3 NVFP4 TRTLLM MoE support | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-02-03 | [#18065](https://github.com/sgl-project/sglang/pull/18065) | merged | [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-02-25 | [#15422](https://github.com/sgl-project/sglang/pull/15422) | merged | Flashinfer MOE FP8 support for Mistral Large 3. | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-02-26 | [#19402](https://github.com/sgl-project/sglang/pull/19402) | merged | Fix nightly Mistral-Large-3 NVFP4 accuracy threshold | `test/registered/8-gpu-models/test_mistral_large3.py` |
| 2026-03-18 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py` |
| 2026-03-30 | [#21620](https://github.com/sgl-project/sglang/pull/21620) | merged | fix: Mistral Small 4 fails to start due to config/weight format mismatch | `test/registered/models/test_ministral4_models.py` |
| 2026-04-06 | [#21399](https://github.com/sgl-project/sglang/pull/21399) | merged | [CI] Add unit tests for function_call detectors (hermes, llama32, mistral) | `test/registered/unit/function_call/test_mistral_detector.py` |

## 逐 PR diff 审计卡

### PR #108 - Fix Mistral model loading

- 链接: https://github.com/sgl-project/sglang/pull/108
- 状态/时间: merged / 2024-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mistral.py`；关联提交 `cd6872334e9e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Mistral model loading」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mistral.py`；PR 正文摘要: Close #107 Co-authored with @johndun。
- 实现要点: `python/sglang/srt/models/mistral.py` added +10/-0 (10 lines); hunks: -0,0 +1,10; symbols: MistralForCausalLM, __init__，涉及 `MistralForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mistral.py` added +10/-0 (10 lines); hunks: -0,0 +1,10; symbols: MistralForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mistral.py
@@ -0,0 +1,10 @@
+"""Inference-only Mistral model."""
+from sglang.srt.models.llama2 import LlamaForCausalLM
+class MistralForCausalLM(LlamaForCausalLM):
+    def __init__(self, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+EntryClass = MistralForCausalLM
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mistral.py` added +10/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mistral.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5099 - model(vlm): mistral 3.1

- 链接: https://github.com/sgl-project/sglang/pull/5099
- 状态/时间: merged / 2025-05-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mistral.py`；关联提交 `64825b839521`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+152/-21，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model(vlm): mistral 3.1」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/mistral.py`；PR 正文摘要: Support Mistral Small 3.1 VLM (#4518). This is an extension to #5084 (#2351) by reusing the same `LlavaForConditionalGeneration` backbone and `Pixtral` vision encoder.。
- 实现要点: `python/sglang/srt/models/mistral.py` modified +71/-1 (72 lines); hunks: -13,11 +13,81; symbols: MistralForCausalLM, Mistral3ForConditionalGeneration, __init__, get_image_feature，涉及 `MistralForCausalLM, Mistral3ForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mistral.py` modified +71/-1 (72 lines); hunks: -13,11 +13,81; symbols: MistralForCausalLM, Mistral3ForConditionalGeneration, __init__, get_image_feature
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mistral.py
@@ -13,11 +13,81 @@
+from typing import List, Union
+import torch
+from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
+from sglang.srt.managers.schedule_batch import MultimodalDataItem
-EntryClass = MistralForCausalLM
+class Mistral3ForConditionalGeneration:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mistral.py` modified +71/-1
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6597 - feat: Improve Mistral and Qwen25 function call parsing

- 链接: https://github.com/sgl-project/sglang/pull/6597
- 状态/时间: merged / 2025-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/mistral_detector.py`；关联提交 `16f69b1f65c6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+318/-61，可读 patch 529 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: Improve Mistral and Qwen25 function call parsing」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/mistral_detector.py`；PR 正文摘要: This PR focuses to resolve the **parallel tool calls parsing** for `MistralDetector` and `Qwen25Detector` See Multiple Tool Call Support for MistralDetector and Qwen25Detector f...。
- 实现要点: `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26 (98 lines); hunks: -1,4 +1,5; -11,12 +12,14; symbols: MistralDetector, __init__, has_tool_call, _clean_text，涉及 `MistralDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26 (98 lines); hunks: -1,4 +1,5; -11,12 +12,14; symbols: MistralDetector, __init__, has_tool_call, _clean_text
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -1,4 +1,5 @@
+import logging
@@ -11,12 +12,14 @@
+logger = logging.getLogger(__name__)
-      [TOOL_CALLS] [{"name":"xxx", "arguments":{...}}]
+      [TOOL_CALLS] [{"name":"func1", "arguments":{...}}, {"name":"func2", "arguments":{...}}]
@@ -32,21 +35,6 @@ def has_tool_call(self, text: str) -> bool:
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +72/-26
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14213 - Add Mistral Large 3 support.

- 链接: https://github.com/sgl-project/sglang/pull/14213
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mistral_large_3.py`；关联提交 `842807843671`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1400/-120，可读 patch 2012 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Large 3 support.」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/mistral_large_3.py`；PR 正文摘要: This PR introduces support for model Mistral Large 3. To enable the model, several key modifications were made. * Two new models are supported: MistralLarge3ForCausalLM and Pixt...。
- 实现要点: `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds，涉及 `MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunks: -0,0 +1,81; symbols: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -0,0 +1,81 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+import regex as re
+import torch
+from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mistral_large_3.py` added +81/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14251 - ministral3

- 链接: https://github.com/sgl-project/sglang/pull/14251
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ministral3.py`；关联提交 `6d37e7088337`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+245/-26，可读 patch 405 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ministral3」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/ministral3.py`；PR 正文摘要: Usage Assume this PR is merged: Or you could install transformers from source: Now you can launch the server with: Then run the following MMMU benchmark script:。
- 实现要点: `python/sglang/srt/models/ministral3.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _get_llama_4_attn_scale, Ministral3Attention, __init__, forward，涉及 `_get_llama_4_attn_scale, Ministral3Attention, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ministral3.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _get_llama_4_attn_scale, Ministral3Attention, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ministral3.py
@@ -0,0 +1,157 @@
+from typing import Any, Dict, Optional
+import torch
+from transformers import PretrainedConfig
+from sglang.srt.layers.quantization.base_config import QuantizationConfig
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.models.llama import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/ministral3.py` added +157/-0
- 验证与风险: diff 自带测试面 `test/srt/models/test_ministral3_models.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14466 - Add Mistral Large 3 Eagle Support

- 链接: https://github.com/sgl-project/sglang/pull/14466
- 状态/时间: merged / 2025-12-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`；关联提交 `205f041e9619`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+313/-62，可读 patch 550 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Large 3 Eagle Support」；模型线: Mistral Small 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/mistral_large_3.py`；PR 正文摘要: Support Mistral Large 3 Eagle. The eagle checkpoint `mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle` is using the FP8 per-tensor quantization while the standard FP8/NVFP4 ch...。
- 实现要点: `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle，涉及 `MistralLarge3Model, __init__, forward`；`python/sglang/srt/models/mistral_large_3.py` modified +0/-3 (3 lines); hunks: -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(; symbols: _iterable_remap_mistral_to_ds，涉及 `_iterable_remap_mistral_to_ds`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle
  - `python/sglang/srt/models/mistral_large_3.py` modified +0/-3 (3 lines); hunks: -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(; symbols: _iterable_remap_mistral_to_ds
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -0,0 +1,105 @@
+from typing import Optional
+import torch
+from torch import nn
+from transformers import PretrainedConfig
+from python.sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
+from sglang.srt.distributed import get_pp_group
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -72,9 +72,6 @@ def _iterable_remap_mistral_to_ds(
-            if name.endswith(".weight_scale") and ".experts." not in name:
-                name = re.sub(r"\.weight_scale$", ".weight_scale_inv", name)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0; `python/sglang/srt/models/mistral_large_3.py` modified +0/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14921 - update mistral detector

- 链接: https://github.com/sgl-project/sglang/pull/14921
- 状态/时间: merged / 2025-12-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/mistral_detector.py`；关联提交 `fd1ebbb0d614`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+274/-34，可读 patch 361 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update mistral detector」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/mistral_detector.py`；PR 正文摘要: This PR updates `MistralDetector` to recognize and parse an additional “legacy compact” tool-call syntax emitted by some templates/models: - **Canonical (newly supported)**: `[T...。
- 实现要点: `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34 (274 lines); hunks: -1,47 +1,49; -51,31 +53,235 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: MistralDetector, __init__, has_tool_call, detect_and_parse，涉及 `MistralDetector, __init__, has_tool_call`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34 (274 lines); hunks: -1,47 +1,49; -51,31 +53,235 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: MistralDetector, __init__, has_tool_call, detect_and_parse
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -1,47 +1,49 @@
-import re
-from typing import List
+from typing import Any, List, Optional, Tuple
+    ToolCallItem,
+from sglang.srt.function_call.utils import _is_complete_json
-    Detector for Mistral model function call format.
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +240/-34
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14485 - Mistral Large 3 NVFP4 support

- 链接: https://github.com/sgl-project/sglang/pull/14485
- 状态/时间: merged / 2025-12-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`；关联提交 `f6031adf0875`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+502/-36，可读 patch 707 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Mistral Large 3 NVFP4 support」；模型线: Mistral Small 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mistral_large_3.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`；PR 正文摘要: Support Mistral Large 3 NVFP4. Depends on https://github.com/sgl-project/sglang/pull/14466. * GSM8K test results:。
- 实现要点: `python/sglang/srt/models/mistral_large_3.py` modified +1/-1 (2 lines); hunks: -1,5 +1,5；`python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0 (2 lines); hunks: -1,3 +1,5。
- 代码 diff 细节:
  - `python/sglang/srt/models/mistral_large_3.py` modified +1/-1 (2 lines); hunks: -1,5 +1,5
  - `python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0 (2 lines); hunks: -1,3 +1,5
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mistral_large_3.py
@@ -1,5 +1,5 @@
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3.py
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -1,3 +1,5 @@
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral_large_3_eagle.py
+# SPDX-License-Identifier: Apache-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mistral_large_3.py` modified +1/-1; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15049 - Mistral Large 3 NVFP4 TRTLLM MoE support

- 链接: https://github.com/sgl-project/sglang/pull/15049
- 状态/时间: merged / 2025-12-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+340/-151，可读 patch 624 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Mistral Large 3 NVFP4 TRTLLM MoE support」；模型线: Mistral Small 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；PR 正文摘要: Support Mistral Large 3 NVFP4 TRTLLM MoE. Tested with cmd from #14485 + `--moe-runner-backend flashinfer_trtllm`: Accuracy TRTLLM MoE: Default cutlass MoE: Perf TRTLLM MoE: Defa...。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunks: -11,10 +11,15; -29,10 +34,18; symbols: __init__, create_weights, process_weights_after_loading，涉及 `__init__, create_weights, process_weights_after_loading`；`python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunks: -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):; symbols: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe，涉及 `swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe`；`python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunks: -42,6 +42,7; -1398,130 +1399,6 @@ def create_weights(; symbols: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale，涉及 `create_weights, prepare_static_weights_for_kernel, process_weights_after_loading`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunks: -548,8 +548,9 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: get_moe_impl_class，涉及 `get_moe_impl_class`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunks: -11,10 +11,15; -29,10 +34,18; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunks: -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):; symbols: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunks: -42,6 +42,7; -1398,130 +1399,6 @@ def create_weights(; symbols: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunks: -548,8 +548,9 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: get_moe_impl_class
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-1 (1 lines); hunks: -1093,7 +1093,6 @@ def forward(self, hidden_states: torch.Tensor, topk_output...; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -11,10 +11,15 @@
-from sglang.srt.distributed import get_tensor_model_parallel_world_size
+from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
+)
+from sglang.srt.layers.dp_attention import is_allocation_symmetric
diff -- python/sglang/srt/layers/quantization/utils.py
@@ -592,3 +592,143 @@ def swizzle_blockscale(scale: torch.Tensor):
+def reorder_w1w3_to_w3w1(
+    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
+    size = weight.size(dim)
+    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -42,6 +42,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21; `python/sglang/srt/layers/quantization/utils.py` modified +140/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-1; `python/sglang/srt/server_args.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18065 - [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE

- 链接: https://github.com/sgl-project/sglang/pull/18065
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_mistral_large3.py`；关联提交 `99fab2ce673e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+115/-111，可读 patch 282 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_mistral_large3.py`；PR 正文摘要: TRTLLM MoE refactoring PR(#15151) broke Mistral Large 3 NVFP4 MoE support(#15049), this PR is trying to fix the issue. Same with results in #15049。
- 实现要点: `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8 (29 lines); hunks: -9,19 +9,21; -56,22 +58,33 @@ def test_mistral_large3_all_variants(self):; symbols: TestMistralLarge3, for, test_mistral_large3_all_variants，涉及 `TestMistralLarge3, for, test_mistral_large3_all_variants`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8 (29 lines); hunks: -9,19 +9,21; -56,22 +58,33 @@ def test_mistral_large3_all_variants(self):; symbols: TestMistralLarge3, for, test_mistral_large3_all_variants
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -9,19 +9,21 @@
-register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)
+register_cuda_ci(est_time=3000, suite="nightly-8-gpu-common", nightly=True)
-MISTRAL_LARGE3_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
+MISTRAL_LARGE3_FP8_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
+MISTRAL_LARGE3_NVFP4_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"
-    Two variants:
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +21/-8
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_mistral_large3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15422 - Flashinfer MOE FP8 support for Mistral Large 3.

- 链接: https://github.com/sgl-project/sglang/pull/15422
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_mistral_large3.py`；关联提交 `350190487be4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+60/-17，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Flashinfer MOE FP8 support for Mistral Large 3.」；模型线: Mistral Small 4；类别: 性能/后端优化；主要 diff: `test/registered/8-gpu-models/test_mistral_large3.py`；PR 正文摘要: This PR brings in Flashinfer MOE FP8 support for Mistral Large 3. It requires an upcoming release of flashinfer to work. Without EP8: With EP8:。
- 实现要点: `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5 (7 lines); hunks: -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):; -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants，涉及 `test_mistral_large3_all_variants`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5 (7 lines); hunks: -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):; -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -46,6 +46,7 @@ def test_mistral_large3_all_variants(self):
+            "--moe-runner-backend=flashinfer_trtllm",
@@ -58,10 +59,6 @@ def test_mistral_large3_all_variants(self):
-        # TODO: add this to base args when FP8 TRTLLM moe is supported
-        nvfp4_args = [
-            "--moe-runner-backend=flashinfer_trtllm",
-        ]
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +2/-5
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_mistral_large3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19402 - Fix nightly Mistral-Large-3 NVFP4 accuracy threshold

- 链接: https://github.com/sgl-project/sglang/pull/19402
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_mistral_large3.py`；关联提交 `e14fd4accb43`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix nightly Mistral-Large-3 NVFP4 accuracy threshold」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_mistral_large3.py`；PR 正文摘要: - Lower gsm8k baseline accuracy from 0.90 to 0.85 for the Mistral-Large-3 nightly test - The NVFP4 quantized variant (`Mistral-Large-3-675B-Instruct-2512-NVFP4`) has slightly un...。
- 实现要点: `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1 (2 lines); hunks: -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants，涉及 `test_mistral_large3_all_variants`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1 (2 lines); hunks: -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):; symbols: test_mistral_large3_all_variants
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_mistral_large3.py
@@ -88,7 +88,7 @@ def test_mistral_large3_all_variants(self):
-            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.90),
+            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.85),
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_mistral_large3.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_mistral_large3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- 链接: https://github.com/sgl-project/sglang/pull/20708
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`；关联提交 `6b8a6545b231`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+360/-124，可读 patch 868 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Mistral Small 4 (Pixtral) support」；模型线: Mistral Small 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/mistral_detector.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`；PR 正文摘要: - Add Mistral Small 4 (119B) model support, reusing the MistralLarge3/DeepSeekV3 backend with Pixtral vision encoder - Handle Mistral-native config format (`params.json`) for Mi...。
- 实现要点: `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment，涉及 `detect_and_parse, parse_streaming_increment`；`python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3 (14 lines); hunks: -18,7 +18,10; -99,9 +102,14 @@ def __init__(; symbols: MistralLarge3Model, MistralLarge3EagleModel, __init__，涉及 `MistralLarge3Model, MistralLarge3EagleModel, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9 (26 lines); hunks: -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: detect_and_parse, parse_streaming_increment
  - `python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3 (14 lines); hunks: -18,7 +18,10; -99,9 +102,14 @@ def __init__(; symbols: MistralLarge3Model, MistralLarge3EagleModel, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/mistral_detector.py
@@ -90,19 +90,27 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
-        parsed = self._try_parse_compact_args_format(tool_part)
-        if not parsed:
+        # Loop to extract all consecutive compact tool calls.
+        all_calls: list = []
+        remaining = tool_part
+        while remaining:
diff -- python/sglang/srt/models/mistral_large_3_eagle.py
@@ -18,7 +18,10 @@
-class MistralLarge3Model(DeepseekV2Model):
+class MistralLarge3EagleModel(DeepseekV2Model):
+    """EAGLE draft model with an fc layer that fuses token embeddings and
+    target-model hidden states before passing through transformer layers."""
@@ -99,9 +102,14 @@ def __init__(
-        config.quant_config = quant_config
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/mistral_detector.py` modified +17/-9; `python/sglang/srt/models/mistral_large_3_eagle.py` modified +11/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/deepseek_ocr.py`, `python/sglang/srt/configs/deepseekvl2.py`, `python/sglang/srt/configs/janus_pro.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21620 - fix: Mistral Small 4 fails to start due to config/weight format mismatch

- 链接: https://github.com/sgl-project/sglang/pull/21620
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/models/test_ministral4_models.py`；关联提交 `1d6424d5ad2d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+59/-7，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Mistral Small 4 fails to start due to config/weight format mismatch」；模型线: Mistral Small 4；类别: 缺陷修复；主要 diff: `test/registered/models/test_ministral4_models.py`；PR 正文摘要: Fixes #21611 `mistralai/Mistral-Small-4-119B-2603` fails to start with `AttributeError` because `w_kc` is `None`. Root Cause Mistral Small 4 ships with **both** `params.json` an...。
- 实现要点: `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: TestMistralSmall4TextOnly, TestMistralSmall4MMMU，涉及 `TestMistralSmall4TextOnly, TestMistralSmall4MMMU`。
- 代码 diff 细节:
  - `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: TestMistralSmall4TextOnly, TestMistralSmall4MMMU
- 关键代码摘录:

```diff
diff -- test/registered/models/test_ministral4_models.py
@@ -0,0 +1,32 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
+from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
+from sglang.test.server_fixtures.default_fixture import DefaultServerBase
+from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase
```

- 已读文件:
  - tests: `test/registered/models/test_ministral4_models.py` added +32/-0
- 验证与风险: diff 自带测试面 `test/registered/models/test_ministral4_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21399 - [CI] Add unit tests for function_call detectors (hermes, llama32, mistral)

- 链接: https://github.com/sgl-project/sglang/pull/21399
- 状态/时间: merged / 2026-04-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/unit/function_call/test_mistral_detector.py`；关联提交 `30f5b8760851`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+595/-0，可读 patch 598 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add unit tests for function_call detectors (hermes, llama32, mistral)」；模型线: Mistral Small 4；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_mistral_detector.py`；PR 正文摘要: - Add 39 unit tests for three previously untested function call format detectors - Contributes to #20865 (Improve Unit Test Coverage) - Tests are CPU-only, no server or model lo...。
- 实现要点: `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: TestMistralDetector, setUp, test_has_tool_call_json_array_format, test_has_tool_call_compact_format，涉及 `TestMistralDetector, setUp, test_has_tool_call_json_array_format`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: TestMistralDetector, setUp, test_has_tool_call_json_array_format, test_has_tool_call_compact_format
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_mistral_detector.py
@@ -0,0 +1,224 @@
+"""Unit tests for MistralDetector — no server, no model loading."""
+import json
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.mistral_detector import MistralDetector
+from sglang.test.ci.ci_register import register_cpu_ci
+from sglang.test.test_utils import CustomTestCase
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_mistral_detector.py` added +224/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_hermes_detector.py`, `test/registered/unit/function_call/test_llama32_detector.py`, `test/registered/unit/function_call/test_mistral_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
