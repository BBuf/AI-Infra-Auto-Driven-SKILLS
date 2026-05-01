# sglang Llama 4 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/basic_usage/llama4.md` | [#13421](https://github.com/sgl-project/sglang/pull/13421) |
| `docs_new/cookbook/autoregressive/Llama/Llama4.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/llama4.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/llama4-maverick-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/llama4-scout-deployment.jsx` | 无直接 PR 号提交 |
| `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` | [#6679](https://github.com/sgl-project/sglang/pull/6679) |
| `python/sglang/srt/models/llama4.py` | [#5092](https://github.com/sgl-project/sglang/pull/5092), [#5127](https://github.com/sgl-project/sglang/pull/5127), [#5128](https://github.com/sgl-project/sglang/pull/5128), [#5144](https://github.com/sgl-project/sglang/pull/5144), [#5159](https://github.com/sgl-project/sglang/pull/5159), [#5194](https://github.com/sgl-project/sglang/pull/5194), [#6752](https://github.com/sgl-project/sglang/pull/6752), [#7729](https://github.com/sgl-project/sglang/pull/7729), [#8512](https://github.com/sgl-project/sglang/pull/8512), [#8683](https://github.com/sgl-project/sglang/pull/8683), [#9101](https://github.com/sgl-project/sglang/pull/9101), [#10047](https://github.com/sgl-project/sglang/pull/10047), ... (16 total) |
| `python/sglang/srt/models/mllama4.py` | [#5092](https://github.com/sgl-project/sglang/pull/5092), [#5144](https://github.com/sgl-project/sglang/pull/5144), [#5194](https://github.com/sgl-project/sglang/pull/5194), [#6985](https://github.com/sgl-project/sglang/pull/6985), [#7129](https://github.com/sgl-project/sglang/pull/7129), [#8272](https://github.com/sgl-project/sglang/pull/8272), [#8512](https://github.com/sgl-project/sglang/pull/8512), [#10042](https://github.com/sgl-project/sglang/pull/10042), [#10047](https://github.com/sgl-project/sglang/pull/10047), [#10611](https://github.com/sgl-project/sglang/pull/10611), [#11282](https://github.com/sgl-project/sglang/pull/11282) |
| `python/sglang/srt/multimodal/processors/mllama4.py` | [#7840](https://github.com/sgl-project/sglang/pull/7840), [#8156](https://github.com/sgl-project/sglang/pull/8156) |
| `test/manual/lora/test_lora_llama4.py` | 无直接 PR 号提交 |
| `test/manual/models/test_llama4_models.py` | 无直接 PR 号提交 |
| `test/registered/8-gpu-models/test_llama4.py` | [#16599](https://github.com/sgl-project/sglang/pull/16599), [#16971](https://github.com/sgl-project/sglang/pull/16971) |
| `test/registered/ascend/llm_models/test_npu_llama4_scount_17b_16e.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 28
- 原文档显式引用补充 PR 数: 1
- 当前文档总 PR 数: 29
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-07 | [#5092](https://github.com/sgl-project/sglang/pull/5092) | merged | Add Llama4 support | `python/sglang/srt/models/llama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/mllama4.py` |
| 2025-04-08 | [#5159](https://github.com/sgl-project/sglang/pull/5159) | merged | Support 2x8xH100 for Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5128](https://github.com/sgl-project/sglang/pull/5128) | merged | Optimize topk operation in llama4 | `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5194](https://github.com/sgl-project/sglang/pull/5194) | merged | Support Llama4 fp8 inference | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-04-09 | [#5144](https://github.com/sgl-project/sglang/pull/5144) | merged | model: support mllama4 | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-04-11 | [#5127](https://github.com/sgl-project/sglang/pull/5127) | merged | Optimize attention in llama4 | `python/sglang/srt/models/llama4.py` |
| 2025-05-09 | [#6162](https://github.com/sgl-project/sglang/pull/6162) | merged | [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2025-05-31 | [#6679](https://github.com/sgl-project/sglang/pull/6679) | merged | update llama4 chat template and pythonic parser | `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` |
| 2025-07-01 | [#6985](https://github.com/sgl-project/sglang/pull/6985) | merged | support llama4 eagle3 | `python/sglang/srt/models/mllama4.py` |
| 2025-07-04 | [#7729](https://github.com/sgl-project/sglang/pull/7729) | merged | refactor llama4 dp attention logic | `python/sglang/srt/models/llama4.py` |
| 2025-07-08 | [#7129](https://github.com/sgl-project/sglang/pull/7129) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang | `python/sglang/srt/models/mllama4.py` |
| 2025-07-08 | [#7840](https://github.com/sgl-project/sglang/pull/7840) | merged | Fix llama4 vision | `python/sglang/srt/multimodal/processors/mllama4.py` |
| 2025-07-23 | [#8272](https://github.com/sgl-project/sglang/pull/8272) | merged | Skip llama4 vision module loading when multimodal disabled | `python/sglang/srt/models/mllama4.py` |
| 2025-07-27 | [#8156](https://github.com/sgl-project/sglang/pull/8156) | merged | Support precomputed_embeddings for Llama 4 | `python/sglang/srt/multimodal/processors/mllama4.py` |
| 2025-08-02 | [#8512](https://github.com/sgl-project/sglang/pull/8512) | merged | model: adapt mllama4 to VisionAttention | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py` |
| 2025-08-03 | [#8683](https://github.com/sgl-project/sglang/pull/8683) | merged | [fix] Fix divide by zero error for llama4. | `python/sglang/srt/models/llama4.py` |
| 2025-08-09 | [#6752](https://github.com/sgl-project/sglang/pull/6752) | merged | Tiny Llama4 type error in constructor | `python/sglang/srt/models/llama4.py` |
| 2025-08-14 | [#9101](https://github.com/sgl-project/sglang/pull/9101) | merged | Feature: support qwen and llama4 reducescatter for dp attention padding | `python/sglang/srt/models/llama4.py` |
| 2025-09-06 | [#10047](https://github.com/sgl-project/sglang/pull/10047) | merged | support Llama4 with non uniformed intermediate size across layers for… | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`, `test/srt/lora/test_lora_llama4.py` |
| 2025-09-29 | [#10611](https://github.com/sgl-project/sglang/pull/10611) | merged | fix: fp8 mllama4 without vision modules being quantized | `python/sglang/srt/models/mllama4.py` |
| 2025-10-06 | [#10042](https://github.com/sgl-project/sglang/pull/10042) | merged | [quantization] Fix scale remapping for mllama4 | `python/sglang/srt/models/mllama4.py` |
| 2025-10-07 | [#11282](https://github.com/sgl-project/sglang/pull/11282) | merged | fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration | `python/sglang/srt/models/mllama4.py` |
| 2025-10-31 | [#12405](https://github.com/sgl-project/sglang/pull/12405) | merged | Fix the shared expert & routed expert overlap in Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2025-11-08 | [#12811](https://github.com/sgl-project/sglang/pull/12811) | merged | use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap | `python/sglang/srt/models/llama4.py` |
| 2025-11-25 | [#13421](https://github.com/sgl-project/sglang/pull/13421) | merged | Add Llama4 attention backend auto-selection | `docs/basic_usage/llama4.md` |
| 2026-01-07 | [#16599](https://github.com/sgl-project/sglang/pull/16599) | merged | ci: adding llama4 placeholder test to nightly | `test/registered/8-gpu-models/test_llama4.py` |
| 2026-01-14 | [#16971](https://github.com/sgl-project/sglang/pull/16971) | merged | fix: renaming test file and job names + skip blocking llama4 nightly | `test/registered/8-gpu-models/test_llama4.py` |
| 2026-01-30 | [#12813](https://github.com/sgl-project/sglang/pull/12813) | merged | add weightless qk norm to RMSNorm interface for Llama 4 | `python/sglang/srt/models/llama4.py` |
| 2026-02-27 | [#17123](https://github.com/sgl-project/sglang/pull/17123) | merged | llama4 npu adapt | `python/sglang/srt/models/llama4.py` |

## 逐 PR diff 审计卡

### PR #5092 - Add Llama4 support

- 链接: https://github.com/sgl-project/sglang/pull/5092
- 状态/时间: merged / 2025-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`；关联提交 `f04c80dc42be`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+2213/-21，可读 patch 2640 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Llama4 support」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/llama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「Add Llama4 support」；主要实现面是 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunks: -0,0 +1,420; symbols: Llama4MoE, custom_routing_function, __init__, forward，涉及 `Llama4MoE, custom_routing_function, __init__`；`python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk，涉及 `Mllama4ImageProcessor, __init__, process_mm_data_async`；`python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary，涉及 `Llama4ForConditionalGeneration, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunks: -0,0 +1,420; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk
  - `python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -0,0 +1,420 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/managers/multimodal_processors/mllama4.py
@@ -0,0 +1,161 @@
+from typing import List, Mapping, Optional, Tuple, Union
+import torch
+from PIL import Image
+from transformers import Llama4Processor
+from transformers.image_utils import SizeDict
+from transformers.models.llama4.image_processing_llama4 import (
diff -- python/sglang/srt/models/mllama4.py
@@ -0,0 +1,154 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` added +420/-0; `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0; `python/sglang/srt/models/mllama4.py` added +154/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/lang/chat_template.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/conversation.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5159 - Support 2x8xH100 for Llama 4

- 链接: https://github.com/sgl-project/sglang/pull/5159
- 状态/时间: merged / 2025-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `5039d547724c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+77/-19，可读 patch 178 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support 2x8xH100 for Llama 4」；模型线: Llama 4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Support 2x8xH100 for Llama 4」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +77/-19 (96 lines); hunks: -27,6 +27,13; -38,6 +45,7; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +77/-19 (96 lines); hunks: -27,6 +27,13; -38,6 +45,7; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -27,6 +27,13 @@
+from sglang.srt.layers.dp_attention import (
+    dp_gather_partial,
+    dp_scatter,
+    get_attention_dp_size,
+    get_attention_tp_rank,
+    get_attention_tp_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +77/-19
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5128 - Optimize topk operation in llama4

- 链接: https://github.com/sgl-project/sglang/pull/5128
- 状态/时间: merged / 2025-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `86a876d883a7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+18/-15，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize topk operation in llama4」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Optimize topk operation in llama4」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +2/-2 (4 lines); hunks: -48,7 +48,7; -63,7 +63,7 @@ def custom_routing_function(; symbols: custom_routing_function，涉及 `custom_routing_function`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +2/-2 (4 lines); hunks: -48,7 +48,7; -63,7 +63,7 @@ def custom_routing_function(; symbols: custom_routing_function
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -48,7 +48,7 @@
-from sglang.srt.utils import add_prefix, get_compiler_backend, make_layers
+from sglang.srt.utils import add_prefix, fast_topk, get_compiler_backend, make_layers
@@ -63,7 +63,7 @@ def custom_routing_function(
-        router_scores_aK, router_indices_aK = torch.topk(gating_output, topk, dim=-1)
+        router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/speculative/eagle_utils.py`, `python/sglang/srt/speculative/eagle_worker.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5194 - Support Llama4 fp8 inference

- 链接: https://github.com/sgl-project/sglang/pull/5194
- 状态/时间: merged / 2025-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`；关联提交 `406524821457`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+537/-106，可读 patch 1026 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Llama4 fp8 inference」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Support Llama4 fp8 inference」；主要实现面是 `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunks: -7,6 +7,7; -16,6 +17,7; symbols: Llama4ForConditionalGeneration, __init__, load_weights，涉及 `Llama4ForConditionalGeneration, __init__, load_weights`；`python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -414,7 +414,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunks: -7,6 +7,7; -16,6 +17,7; symbols: Llama4ForConditionalGeneration, __init__, load_weights
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -414,7 +414,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -7,6 +7,7 @@
+from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
@@ -16,6 +17,7 @@
+        "gate_up_proj": ["gate_proj", "up_proj"],
@@ -96,6 +98,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
+        # Params for weights, fp8 weight scales, fp8 activation scales
+        # (param_name, weight_name, expert_id, shard_id)
diff -- python/sglang/srt/models/llama4.py
@@ -414,7 +414,7 @@ def __init__(
-            prefix="model.layers",
+            prefix=add_prefix("layers", prefix),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +52/-18; `python/sglang/srt/models/llama4.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_int8_kernel.py`, `test/srt/test_triton_moe_channel_fp8_kernel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5144 - model: support mllama4

- 链接: https://github.com/sgl-project/sglang/pull/5144
- 状态/时间: merged / 2025-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`；关联提交 `fbebcb7aa4aa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+145/-65，可读 patch 403 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: support mllama4」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`, `python/sglang/srt/models/llama4.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +50/-11 (61 lines); hunks: -1,14 +1,19; -30,6 +35,9 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, forward，涉及 `__init__, pad_input_ids, get_image_feature`；`python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36 (57 lines); hunks: -1,10 +1,8; -15,7 +13,6; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async，涉及 `Mllama4ImageProcessor, __init__, process_mm_data_async`；`python/sglang/srt/models/llama4.py` modified +3/-0 (3 lines); hunks: -466,6 +466,9 @@ def __init__(; symbols: __init__, get_input_embeddings, _init_model，涉及 `__init__, get_input_embeddings, _init_model`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +50/-11 (61 lines); hunks: -1,14 +1,19; -30,6 +35,9 @@ def __init__(; symbols: __init__, pad_input_ids, get_image_feature, forward
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36 (57 lines); hunks: -1,10 +1,8; -15,7 +13,6; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async
  - `python/sglang/srt/models/llama4.py` modified +3/-0 (3 lines); hunks: -466,6 +466,9 @@ def __init__(; symbols: __init__, get_input_embeddings, _init_model
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,14 +1,19 @@
-# TODO: add Aapted from vllm/mllama4.py
-from typing import Optional, Set, Tuple
+from typing import List, Optional, Set, Tuple
-from transformers import Llama4Config
+from transformers import Llama4Config, Llama4VisionModel
+from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector
diff -- python/sglang/srt/managers/multimodal_processors/mllama4.py
@@ -1,10 +1,8 @@
-from typing import List, Mapping, Optional, Tuple, Union
+from typing import List, Union
-from PIL import Image
-from transformers import Llama4Processor
-from transformers.models.llama4.image_processing_llama4 import (
+from transformers.models.llama4.image_processing_llama4_fast import (
diff -- python/sglang/srt/models/llama4.py
@@ -466,6 +466,9 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +50/-11; `python/sglang/srt/managers/multimodal_processors/mllama4.py` modified +21/-36; `python/sglang/srt/models/llama4.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5127 - Optimize attention in llama4

- 链接: https://github.com/sgl-project/sglang/pull/5127
- 状态/时间: merged / 2025-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `cd7e32e2cb15`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-12，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize attention in llama4」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Optimize attention in llama4」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +18/-12 (30 lines); hunks: -240,37 +240,43 @@ def __init__(; symbols: __init__, _get_attn_scale, _mul_attn_scale, forward，涉及 `__init__, _get_attn_scale, _mul_attn_scale`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +18/-12 (30 lines); hunks: -240,37 +240,43 @@ def __init__(; symbols: __init__, _get_attn_scale, _mul_attn_scale, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -240,37 +240,43 @@ def __init__(
+    @torch.compile(dynamic=True, backend=get_compiler_backend())
+    def _mul_attn_scale(self, positions, q):
+        attn_scale = self._get_attn_scale(positions)
+        return (q * attn_scale).to(q.dtype)
-        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
+        qk, v = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +18/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6162 - [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph

- 链接: https://github.com/sgl-project/sglang/pull/6162
- 状态/时间: merged / 2025-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+125/-8，可读 patch 203 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Llama4 gibberish output with long context and CUDA graph」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/flashattention_backend.py`；技术摘要: 覆盖「[Bugfix] Fix Llama4 gibberish output with long context and CUDA graph」；主要实现面是 `python/sglang/srt/layers/attention/flashattention_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunks: -913,8 +913,10 @@ def forward_decode(; -970,7 +972,7 @@ def forward_decode(; symbols: forward_decode, init_cuda_graph_state，涉及 `forward_decode, init_cuda_graph_state`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunks: -913,8 +913,10 @@ def forward_decode(; -970,7 +972,7 @@ def forward_decode(; symbols: forward_decode, init_cuda_graph_state
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -913,8 +913,10 @@ def forward_decode(
-        use_local_attention = (
-            self.attention_chunk_size is not None and local_attn_metadata is not None
+        use_local_attn = (
+            self.attention_chunk_size is not None
+            and local_attn_metadata is not None
+            and (hasattr(layer, "use_irope") and layer.use_irope)
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/flashattention_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6679 - update llama4 chat template and pythonic parser

- 链接: https://github.com/sgl-project/sglang/pull/6679
- 状态/时间: merged / 2025-05-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/chat_template/tool_chat_template_llama4_pythonic.jinja`；关联提交 `4fac524b14a0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+165/-73，可读 patch 350 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update llama4 chat template and pythonic parser」；模型线: Llama 4；类别: 文档/测试/CI；主要 diff: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja`；技术摘要: 覆盖「update llama4 chat template and pythonic parser」；主要实现面是 `examples/chat_template/tool_chat_template_llama4_pythonic.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63 (98 lines); hunks: -1,86 +1,52; -92,10 +58,12。
- 代码 diff 细节:
  - `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63 (98 lines); hunks: -1,86 +1,52; -92,10 +58,12
- 关键代码摘录:

```diff
diff -- examples/chat_template/tool_chat_template_llama4_pythonic.jinja
@@ -1,86 +1,52 @@
-{# Copied from https://github.com/yeqcharlotte/vllm/blob/4fcf68a948bbe0498dc8a98feafa102cfb1dd210/examples/tool_chat_template_llama4_pythonic.jinja to enable better model response
+{# Copied from https://github.com/wukaixingxp/vllm/blob/8a32e2a6e452a03c0e8222e3876ad6086cbf581f/examples/tool_chat_template_llama4_pythonic.jinja to enable better model response.
-{%- if custom_tools is defined %}
+{%- if custom_tools is defined and custom_tools %}
-{%- if not tools_in_user_message is defined %}
-    {%- set tools_in_user_message = false %}
```

- 已读文件:
  - docs: `examples/chat_template/tool_chat_template_llama4_pythonic.jinja` modified +35/-63
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6985 - support llama4 eagle3

- 链接: https://github.com/sgl-project/sglang/pull/6985
- 状态/时间: merged / 2025-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `886d34496475`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+139/-19，可读 patch 237 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support llama4 eagle3」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「support llama4 eagle3」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +29/-0 (29 lines); hunks: -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head，涉及 `load_weights, set_eagle3_layers_to_capture, get_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +29/-0 (29 lines); hunks: -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -223,5 +223,34 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
+    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
+        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):
+            self.language_model.set_eagle3_layers_to_capture(layer_ids)
+    def get_embed_and_head(self):
+        # For EAGLE3, we delegate to the language model which should have this method
+        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/weight_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7729 - refactor llama4 dp attention logic

- 链接: https://github.com/sgl-project/sglang/pull/7729
- 状态/时间: merged / 2025-07-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `8c298031d57e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-45，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor llama4 dp attention logic」；模型线: Llama 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「refactor llama4 dp attention logic」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +32/-45 (77 lines); hunks: -27,9 +27,8; -367,7 +366,10 @@ def __init__(; symbols: __init__, _is_moe_layer, forward，涉及 `__init__, _is_moe_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +32/-45 (77 lines); hunks: -27,9 +27,8; -367,7 +366,10 @@ def __init__(; symbols: __init__, _is_moe_layer, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -27,9 +27,8 @@
+from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
-    dp_gather_partial,
-    dp_scatter,
@@ -367,7 +366,10 @@ def __init__(
-        is_moe_layer = (layer_id + 1) % config.interleave_moe_layer_step == 0
+        self.config = config
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +32/-45
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7129 - Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang

- 链接: https://github.com/sgl-project/sglang/pull/7129
- 状态/时间: merged / 2025-07-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `659907e32b95`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+643/-81，可读 patch 847 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunks: -1,3 +1,6; -19,6 +22,13; symbols: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index，涉及 `Llama4ForConditionalGeneration, __init__, _has_vision_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunks: -1,3 +1,6; -19,6 +22,13; symbols: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,3 +1,6 @@
+import json as json_lib
+import logging
+import os
@@ -19,6 +22,13 @@
+from sglang.srt.model_loader.weight_utils import (
+    default_weight_loader,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +360/-79
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7840 - Fix llama4 vision

- 链接: https://github.com/sgl-project/sglang/pull/7840
- 状态/时间: merged / 2025-07-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/mllama4.py`；关联提交 `4bab50a6b580`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+82/-63，可读 patch 186 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix llama4 vision」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/multimodal/processors/mllama4.py`；技术摘要: 覆盖「Fix llama4 vision」；主要实现面是 `python/sglang/srt/multimodal/processors/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60 (122 lines); hunks: -60,70 +60,72 @@ async def process_mm_data_async(; symbols: process_mm_data_async，涉及 `process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60 (122 lines); hunks: -60,70 +60,72 @@ async def process_mm_data_async(; symbols: process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/mllama4.py
@@ -60,70 +60,72 @@ async def process_mm_data_async(
-        if "pixel_values" in processor_output:
-            image_processor = processor.image_processor
-            tokenizer = self._processor.tokenizer
+        if "pixel_values" not in processor_output:  # no image processed
+            return None
-            # Calculate tile size and find supported resolutions
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/mllama4.py` modified +62/-60
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/conversation.py`, `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/multimodal/processors/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8272 - Skip llama4 vision module loading when multimodal disabled

- 链接: https://github.com/sgl-project/sglang/pull/8272
- 状态/时间: merged / 2025-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `e2d66f60c8f8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-3，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Skip llama4 vision module loading when multimodal disabled」；模型线: Llama 4；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「Skip llama4 vision module loading when multimodal disabled」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +10/-3 (13 lines); hunks: -23,6 +23,7; -55,13 +56,17 @@ def __init__(; symbols: __init__, load_weights, _should_skip_weight, _transform_weight_name，涉及 `__init__, load_weights, _should_skip_weight`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +10/-3 (13 lines); hunks: -23,6 +23,7; -55,13 +56,17 @@ def __init__(; symbols: __init__, load_weights, _should_skip_weight, _transform_weight_name
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -23,6 +23,7 @@
+    global_server_args_dict,
@@ -55,13 +56,17 @@ def __init__(
-        self.has_vision = self._has_vision_weights(config)
-        if not self.has_vision:
+        self.has_vision_weights = self._has_vision_weights(config)
+        if not self.has_vision_weights:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +10/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8156 - Support precomputed_embeddings for Llama 4

- 链接: https://github.com/sgl-project/sglang/pull/8156
- 状态/时间: merged / 2025-07-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/multimodal/processors/mllama4.py`；关联提交 `44d600cd675f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+449/-123，可读 patch 235 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support precomputed_embeddings for Llama 4」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/mllama4.py`；技术摘要: 覆盖「Support precomputed_embeddings for Llama 4」；主要实现面是 `python/sglang/srt/multimodal/processors/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109 (125 lines); hunks: -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args...; -37,114 +37,21 @@ async def process_mm_data_async(; symbols: __init__, process_mm_data_async，涉及 `__init__, process_mm_data_async`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109 (125 lines); hunks: -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args...; -37,114 +37,21 @@ async def process_mm_data_async(; symbols: __init__, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/mllama4.py
@@ -22,12 +22,12 @@ def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
-        self.boi_token_index = hf_config.boi_token_index
-        self.eoi_token_index = hf_config.eoi_token_index
-        self.image_token_index = hf_config.image_token_index
-        self.multimodal_tokens = MultimodalSpecialTokens(
+        self.IM_START_TOKEN_ID = hf_config.boi_token_index
+        self.IM_END_TOKEN_ID = hf_config.eoi_token_index
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/mllama4.py` modified +16/-109
- 验证与风险: diff 自带测试面 `test/srt/test_vlm_input_format.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8512 - model: adapt mllama4 to VisionAttention

- 链接: https://github.com/sgl-project/sglang/pull/8512
- 状态/时间: merged / 2025-08-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`；关联提交 `ea93079b3038`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+518/-52，可读 patch 903 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: adapt mllama4 to VisionAttention」；模型线: Llama 4；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「model: adapt mllama4 to VisionAttention」；主要实现面是 `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +428/-19 (447 lines); hunks: -1,17 +1,24; -26,10 +33,10; symbols: Llama4VisionMLP, __init__, forward, pixel_shuffle，涉及 `Llama4VisionMLP, __init__, forward`；`python/sglang/srt/models/llama4.py` modified +11/-2 (13 lines); hunks: -241,13 +241,22 @@ def __init__(; -257,7 +266,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +428/-19 (447 lines); hunks: -1,17 +1,24; -26,10 +33,10; symbols: Llama4VisionMLP, __init__, forward, pixel_shuffle
  - `python/sglang/srt/models/llama4.py` modified +11/-2 (13 lines); hunks: -241,13 +241,22 @@ def __init__(; -257,7 +266,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -1,17 +1,24 @@
+import math
-from transformers import Llama4Config
+from transformers import Llama4Config, Llama4VisionConfig
-    Llama4VisionModel,
+    vision_apply_rotary_emb,
+from sglang.srt.layers.attention.vision import VisionAttention
diff -- python/sglang/srt/models/llama4.py
@@ -241,13 +241,22 @@ def __init__(
+        qkv_quant_config = quant_config
+        o_quant_config = quant_config
+        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:
+            if add_prefix("q_proj", prefix) in quant_config.ignore:
+                qkv_quant_config = None
+            if add_prefix("o_proj", prefix) in quant_config.ignore:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +428/-19; `python/sglang/srt/models/llama4.py` modified +11/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hf_transformers_utils.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/managers/tokenizer_manager.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8683 - [fix] Fix divide by zero error for llama4.

- 链接: https://github.com/sgl-project/sglang/pull/8683
- 状态/时间: merged / 2025-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `7ed8e51bc31e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Fix divide by zero error for llama4.」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「[fix] Fix divide by zero error for llama4.」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +2/-0 (2 lines); hunks: -406,6 +406,8 @@ def __init__(; symbols: __init__, _is_moe_layer, forward，涉及 `__init__, _is_moe_layer, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +2/-0 (2 lines); hunks: -406,6 +406,8 @@ def __init__(; symbols: __init__, _is_moe_layer, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -406,6 +406,8 @@ def __init__(
+        if self.config.interleave_moe_layer_step == 0:
+            return self.config.num_local_experts > 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6752 - Tiny Llama4 type error in constructor

- 链接: https://github.com/sgl-project/sglang/pull/6752
- 状态/时间: merged / 2025-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `4a9f3eef90a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Tiny Llama4 type error in constructor」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Tiny Llama4 type error in constructor」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -203,7 +203,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -203,7 +203,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -203,7 +203,7 @@ def __init__(
-        self.use_rope = int((layer_id + 1) % 4 != 0)
+        self.use_rope = (layer_id + 1) % 4 != 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- 链接: https://github.com/sgl-project/sglang/pull/9101
- 状态/时间: merged / 2025-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `4c22897a66ab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+68/-16，可读 patch 210 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feature: support qwen and llama4 reducescatter for dp attention padding」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Feature: support qwen and llama4 reducescatter for dp attention padding」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer，涉及 `__init__, forward, _is_moe_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -131,14 +131,19 @@ def __init__(
-    def forward(self, hidden_states, forward_batch: ForwardBatch):
+    def forward(
+        self,
+        hidden_states,
+        forward_batch: ForwardBatch,
+        use_reduce_scatter: bool = False,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +16/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/lora/layers.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10047 - support Llama4 with non uniformed intermediate size across layers for…

- 链接: https://github.com/sgl-project/sglang/pull/10047
- 状态/时间: merged / 2025-09-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/models/mllama4.py`；关联提交 `ab62b135c18a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+123/-13，可读 patch 220 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support Llama4 with non uniformed intermediate size across layers for…」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`, `test/srt/lora/test_lora_llama4.py`；技术摘要: 覆盖「support Llama4 with non uniformed intermediate size across layers for…」；主要实现面是 `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/models/llama4.py`, `test/srt/lora/test_lora_llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +25/-0 (25 lines); hunks: -961,5 +961,30 @@ def get_embed(self):; symbols: get_embed, set_embed, get_hidden_dim，涉及 `get_embed, set_embed, get_hidden_dim`；`python/sglang/srt/models/llama4.py` modified +9/-0 (9 lines); hunks: -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:; -540,6 +546,9 @@ def __init__(; symbols: _is_moe_layer, get_intermediate_size, forward, __init__，涉及 `_is_moe_layer, get_intermediate_size, forward`；`test/srt/lora/test_lora_llama4.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestLlama4LoRA, setUpClass, test_bringup，涉及 `TestLlama4LoRA, setUpClass, test_bringup`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +25/-0 (25 lines); hunks: -961,5 +961,30 @@ def get_embed(self):; symbols: get_embed, set_embed, get_hidden_dim
  - `python/sglang/srt/models/llama4.py` modified +9/-0 (9 lines); hunks: -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:; -540,6 +546,9 @@ def __init__(; symbols: _is_moe_layer, get_intermediate_size, forward, __init__
  - `test/srt/lora/test_lora_llama4.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestLlama4LoRA, setUpClass, test_bringup
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -961,5 +961,30 @@ def get_embed(self):
+    def get_hidden_dim(self, module_name, layer_idx):
+        # return input_dim, output_dim
+        if module_name == "qkv_proj":
+            return (
+                self.config.hidden_size,
+                self.config.head_dim
diff -- python/sglang/srt/models/llama4.py
@@ -423,6 +423,12 @@ def _is_moe_layer(self, layer_id: int) -> bool:
+    def get_intermediate_size(self) -> int:
+        if isinstance(self.feed_forward, Llama4MoE):
+            return self.config.intermediate_size
+        else:
+            return self.config.intermediate_size_mlp
@@ -540,6 +546,9 @@ def __init__(
diff -- test/srt/lora/test_lora_llama4.py
@@ -0,0 +1,61 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +25/-0; `python/sglang/srt/models/llama4.py` modified +9/-0
  - tests: `test/srt/lora/test_lora_llama4.py` added +61/-0
- 验证与风险: diff 自带测试面 `test/srt/lora/test_lora_llama4.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10611 - fix: fp8 mllama4 without vision modules being quantized

- 链接: https://github.com/sgl-project/sglang/pull/10611
- 状态/时间: merged / 2025-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `9de1320b637c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-3，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: fp8 mllama4 without vision modules being quantized」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「fix: fp8 mllama4 without vision modules being quantized」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +14/-3 (17 lines); hunks: -291,7 +291,7 @@ def __init__(; -446,9 +446,20 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +14/-3 (17 lines); hunks: -291,7 +291,7 @@ def __init__(; -446,9 +446,20 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -291,7 +291,7 @@ def __init__(
-        hidden_states = hidden_states.permute(0, 2, 1)
+        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
@@ -446,9 +446,20 @@ def __init__(
+            # TODO: make this more general
+            ignore_quant_layers = getattr(config, "quantization_config", {}).get(
+                "ignore", {}
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +14/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10042 - [quantization] Fix scale remapping for mllama4

- 链接: https://github.com/sgl-project/sglang/pull/10042
- 状态/时间: merged / 2025-10-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `c7a104c12bda`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[quantization] Fix scale remapping for mllama4」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「[quantization] Fix scale remapping for mllama4」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params，涉及 `_handle_scale_remapping, _handle_stacked_params`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
-            return remapped_name is None
+            return remapped_name is not None and remapped_name != name
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11282 - fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration

- 链接: https://github.com/sgl-project/sglang/pull/11282
- 状态/时间: merged / 2025-10-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mllama4.py`；关联提交 `fd8a0b29c044`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mllama4.py`；技术摘要: 覆盖「fix: correct scale parameter remapping logic in Llama4ForConditionalGeneration」；主要实现面是 `python/sglang/srt/models/mllama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params，涉及 `_handle_scale_remapping, _handle_stacked_params`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mllama4.py` modified +1/-1 (2 lines); hunks: -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: di...; symbols: _handle_scale_remapping, _handle_stacked_params
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mllama4.py
@@ -700,7 +700,7 @@ def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
-            return remapped_name is not None and remapped_name != name
+            return remapped_name != name
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mllama4.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12405 - Fix the shared expert & routed expert overlap in Llama 4

- 链接: https://github.com/sgl-project/sglang/pull/12405
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `34c286b8115e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix the shared expert & routed expert overlap in Llama 4」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「Fix the shared expert & routed expert overlap in Llama 4」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def forward(; symbols: forward, _forward_core，涉及 `forward, _forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +1/-1 (2 lines); hunks: -148,7 +148,7 @@ def forward(; symbols: forward, _forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -148,7 +148,7 @@ def forward(
-        if hidden_states.shape[0] < 4 and _is_cuda:
+        if _is_cuda:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12811 - use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap

- 链接: https://github.com/sgl-project/sglang/pull/12811
- 状态/时间: merged / 2025-11-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `49653c88964f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap」；模型线: Llama 4；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「use fast stream instead of torch.cuda.current_stream in llama 4 shared experts overlap」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +3/-2 (5 lines); hunks: -58,6 +58,7; -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):; symbols: _forward_core_normal, _forward_core_shared_routed_overlap，涉及 `_forward_core_normal, _forward_core_shared_routed_overlap`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +3/-2 (5 lines); hunks: -58,6 +58,7; -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):; symbols: _forward_core_normal, _forward_core_shared_routed_overlap
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -58,6 +58,7 @@
+from sglang.srt.utils.common import get_current_device_stream_fast
@@ -164,7 +165,7 @@ def _forward_core_normal(self, hidden_states):
-        alt_stream.wait_stream(self.device_module.current_stream())
+        alt_stream.wait_stream(get_current_device_stream_fast())
@@ -173,7 +174,7 @@ def _forward_core_shared_routed_overlap(self, hidden_states):
-        self.device_module.current_stream().wait_stream(alt_stream)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13421 - Add Llama4 attention backend auto-selection

- 链接: https://github.com/sgl-project/sglang/pull/13421
- 状态/时间: merged / 2025-11-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/basic_usage/llama4.md`；关联提交 `fcccaf9001ab`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+24/-5，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add Llama4 attention backend auto-selection」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `docs/basic_usage/llama4.md`；技术摘要: 覆盖「Add Llama4 attention backend auto-selection」；主要实现面是 `docs/basic_usage/llama4.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunks: -21,6 +21,15 @@ python3 -m sglang.launch_server \。
- 代码 diff 细节:
  - `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunks: -21,6 +21,15 @@ python3 -m sglang.launch_server \
- 关键代码摘录:

```diff
diff -- docs/basic_usage/llama4.md
@@ -21,6 +21,15 @@ python3 -m sglang.launch_server \
+- **Attention Backend Auto-Selection**: SGLang automatically selects the optimal attention backend for Llama 4 based on your hardware. You typically don't need to specify `--atten
+  - **Blackwell GPUs (B200/GB200)**: `trtllm_mha`
+  - **Hopper GPUs (H100/H200)**: `fa3`
+  - **AMD GPUs**: `aiter`
+  - **Intel XPU**: `intel_xpu`
+  - **Other platforms**: `triton` (fallback)
```

- 已读文件:
  - docs: `docs/basic_usage/llama4.md` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16599 - ci: adding llama4 placeholder test to nightly

- 链接: https://github.com/sgl-project/sglang/pull/16599
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_llama4.py`；关联提交 `2ff872311b4d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+55/-0，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「ci: adding llama4 placeholder test to nightly」；模型线: Llama 4；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_llama4.py`；技术摘要: 覆盖「ci: adding llama4 placeholder test to nightly」；主要实现面是 `test/registered/8-gpu-models/test_llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_llama4.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: TestLlama4Unified, for, test_llama4，涉及 `TestLlama4Unified, for, test_llama4`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_llama4.py` added +55/-0 (55 lines); hunks: -0,0 +1,55; symbols: TestLlama4Unified, for, test_llama4
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_llama4.py
@@ -0,0 +1,55 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_llama4.py` added +55/-0
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_llama4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16971 - fix: renaming test file and job names + skip blocking llama4 nightly

- 链接: https://github.com/sgl-project/sglang/pull/16971
- 状态/时间: merged / 2026-01-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_llama4.py`；关联提交 `aa2b4f766179`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+22/-21，可读 patch 142 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: renaming test file and job names + skip blocking llama4 nightly」；模型线: Llama 4；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_llama4.py`；技术摘要: 覆盖「fix: renaming test file and job names + skip blocking llama4 nightly」；主要实现面是 `test/registered/8-gpu-models/test_llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_llama4.py` modified +2/-1 (3 lines); hunks: -12,7 +12,8; symbols: TestLlama4Unified, TestLlama4, for，涉及 `TestLlama4Unified, TestLlama4, for`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_llama4.py` modified +2/-1 (3 lines); hunks: -12,7 +12,8; symbols: TestLlama4Unified, TestLlama4, for
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_llama4.py
@@ -12,7 +12,8 @@
-class TestLlama4Unified(unittest.TestCase):
+@unittest.skip("Blocked: Missing HF token permission for Llama 4 model")
+class TestLlama4(unittest.TestCase):
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_llama4.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_deepseek_v31.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`, `test/registered/8-gpu-models/test_glm_46.py`, `test/registered/8-gpu-models/test_glm_46_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12813 - add weightless qk norm to RMSNorm interface for Llama 4

- 链接: https://github.com/sgl-project/sglang/pull/12813
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `22df62d5862b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-2，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add weightless qk norm to RMSNorm interface for Llama 4」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「add weightless qk norm to RMSNorm interface for Llama 4」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -242,6 +242,7 @@ def __init__(
+                has_weight=False,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17123 - llama4 npu adapt

- 链接: https://github.com/sgl-project/sglang/pull/17123
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/llama4.py`；关联提交 `bc9190435b3f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+37/-1，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「llama4 npu adapt」；模型线: Llama 4；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/llama4.py`；技术摘要: 覆盖「llama4 npu adapt」；主要实现面是 `python/sglang/srt/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/llama4.py` modified +4/-0 (4 lines); hunks: -56,11 +56,13; -329,6 +331,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/llama4.py` modified +4/-0 (4 lines); hunks: -56,11 +56,13; -329,6 +331,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/llama4.py
@@ -56,11 +56,13 @@
+    is_npu,
+_is_npu = is_npu()
@@ -329,6 +331,8 @@ def forward(
+            if _is_npu:
+                qk = torch.cat([q_out_unused, k_out_unused], dim=-1)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/llama4.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/ascend/llm_models/test_ascend_llama4_scount_17b_16e.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
