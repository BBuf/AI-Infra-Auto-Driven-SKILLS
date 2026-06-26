# sglang MiMo V2 Flash 模型 PR 优化历史

## 2026-06-26 最新源码扫描

已按 SGLang 上游 `sgl-project/sglang@8524678889485801e7a4a12d62015be0c68f7a90` 重新扫描本文下方列出的 tracked files。
文件级匹配使用 GitHub mirror 的 `git log --name-only`；PR 标题、链接和合并时间通过 GitHub GraphQL Pull Request API 批量复核。上一时效锚点：`2026-06-05`。

结果：发现 7 个额外 PR-numbered merge 触及 tracked files，但尚未提升为下方完整逐 PR diff audit card。此节只作为 freshness index；需要引用实现细节时，仍应先人工阅读 PR diff 再补完整卡片。

| 合并日期 | PR | 标题 | 命中的 tracked files |
| --- | --- | --- | --- |
| 2026-06-25 | [#29253](https://github.com/sgl-project/sglang/pull/29253) | Add MiMo V2.5 Blackwell vision FA4 recipe | `MiMo-V2.5.mdx`, `mimo-v25-deployment.jsx` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | Add get_parallel(): a structured accessor for parallel-topology state | `mimo_mtp.py`, `mimo_v2.py`, `mimo_v2_nextn.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | [Spec] Retire Spec V1 | `MiMo-V2.5.mdx` |
| 2026-06-11 | [#26278](https://github.com/sgl-project/sglang/pull/26278) | Support MiMo v2 ASR | `mimo_audio.py`, `mimo_v2.py` |
| 2026-06-10 | [#27668](https://github.com/sgl-project/sglang/pull/27668) | Fix MiMo-V2.5-Pro DP-attention dp size in cookbook deployment snippet | `mimo-v25-deployment.jsx` |
| 2026-06-10 | [#25455](https://github.com/sgl-project/sglang/pull/25455) | [NPU] MiMo-V2-Flash Adaptation | `mimo_v2.py` |
| 2026-06-08 | [#27512](https://github.com/sgl-project/sglang/pull/27512) | [Spec] Clamp multimodal pad sentinels in spec-v2 draft prefill embedding | `mimo_v2_nextn.py` |

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 sglang 上游 `origin/main@6cfdc1858` 复核；自上次时效基准（2026-05-19）以来，共有 5 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-05-29 | [#26673](https://github.com/sgl-project/sglang/pull/26673) | [refactor] remove unused op_mlp | `mimo_v2.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test_mimo_models.py` |
| 2026-05-26 | [#25964](https://github.com/sgl-project/sglang/pull/25964) | [EPD] Cross-request batching for image/audio encoder | `mimo_v2.py` |
| 2026-05-22 | [#24751](https://github.com/sgl-project/sglang/pull/24751) | fix(mm): make multimodal data loading non-blocking to prevent health check stalls | `mimo_v2.py` |
| 2026-05-19 | [#25359](https://github.com/sgl-project/sglang/pull/25359) | [Docs] MiMo-V2.5 cookbook: B200 benchmarks + multi-layer EAGLE acceptance profile + long-context reference | `MiMo-V2.5.mdx`, `mimo-v25-deployment.jsx` |


## 2026-05-19 PR 补漏复核

已按 sglang 上游 `origin/main@78cb38ed5` 和 GitHub Pull Request files API 复核；本轮补齐 `#24931`, `#25588` 的时间线与逐 PR diff 审计卡。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2-Flash.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945) |
| `docs_new/src/snippets/autoregressive/mimo-v2-flash-deployment.jsx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` | [#23851](https://github.com/sgl-project/sglang/pull/23851), [#23936](https://github.com/sgl-project/sglang/pull/23936), [#23945](https://github.com/sgl-project/sglang/pull/23945) |
| `python/sglang/srt/function_call/mimo_detector.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `python/sglang/srt/models/mimo.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059) |
| `python/sglang/srt/models/mimo_audio.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_mtp.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059), [#7370](https://github.com/sgl-project/sglang/pull/7370) |
| `python/sglang/srt/models/mimo_v2.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_v2_nextn.py` | [#23808](https://github.com/sgl-project/sglang/pull/23808), [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/models/mimo_vl.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `python/sglang/srt/multimodal/processors/mimo_v2.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811) |
| `test/registered/8-gpu-models/test_mimo_models.py` | [#23811](https://github.com/sgl-project/sglang/pull/23811), [#24118](https://github.com/sgl-project/sglang/pull/24118) |
| `test/registered/ascend/llm_models/test_npu_mimo_7b_rl.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_mimo_vl_7b_rl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 9
- 原文档显式引用补充 PR 数: 7
- 当前文档总 PR 数: 16
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-22 | [#6059](https://github.com/sgl-project/sglang/pull/6059) | merged | Support XiaomiMiMo inference with mtp | `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py` |
| 2025-06-20 | [#7370](https://github.com/sgl-project/sglang/pull/7370) | merged | Clean unused import for mimo mtp model | `python/sglang/srt/models/mimo_mtp.py` |
| 2025-12-19 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | `python/sglang/srt/function_call/mimo_detector.py` |
| 2025-12-20 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-25 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-02-01 | [#18051](https://github.com/sgl-project/sglang/pull/18051) | merged | [Fix] Remove no use code in MiMo-V2-Flash | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-02-02 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` |
| 2026-04-01 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-27 | [#23851](https://github.com/sgl-project/sglang/pull/23851) | merged | [Docs] add cookbook for MiMo-V2.5 family | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-04-28 | [#23808](https://github.com/sgl-project/sglang/pull/23808) | merged | [Feature] Xiaomi MiMo-V2.5-Pro day0 support | `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py` |
| 2026-04-28 | [#23945](https://github.com/sgl-project/sglang/pull/23945) | merged | docs: enable MiMo V2.5 MTP cookbook path | `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` |
| 2026-04-29 | [#23936](https://github.com/sgl-project/sglang/pull/23936) | merged | mimo v2.5 pro sglang-jax cookbook | `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` |
| 2026-04-30 | [#24118](https://github.com/sgl-project/sglang/pull/24118) | merged | fix: rename mimo spec threshold attr to num_accepted_drafts_thres | `test/registered/8-gpu-models/test_mimo_models.py` |
| 2026-04-30 | [#23811](https://github.com/sgl-project/sglang/pull/23811) | merged | [Feature] Xiaomi MiMo-V2.5 day0 support | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py` |
| 2026-05-18 | [#24931](https://github.com/sgl-project/sglang/pull/24931) | merged | feat(mimo-v2): add EPD disaggregation support | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py` |
| 2026-05-19 | [#25588](https://github.com/sgl-project/sglang/pull/25588) | merged | perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode | `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/utils/video_decoder.py` |

## 逐 PR diff 审计卡

### PR #6059 - Support XiaomiMiMo inference with mtp

- 链接: https://github.com/sgl-project/sglang/pull/6059
- 状态/时间: merged / 2025-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo.py`, `python/sglang/srt/models/mimo_mtp.py`；关联提交 `a6ae3af15e84`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+344/-6，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support XiaomiMiMo inference with mtp」；模型线: MiMo V2 Flash；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`；技术摘要: 覆盖「Support XiaomiMiMo inference with mtp」；主要实现面是 `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_mtp.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMTP，涉及 `MiMoMultiTokenPredictorLayer, __init__, forward`；`python/sglang/srt/models/mimo.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_mtp.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMTP
  - `python/sglang/srt/models/mimo.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_mtp.py
@@ -0,0 +1,220 @@
+# Adapted from https://github.com/vllm-project/vllm/pull/17433/files  and deepseek_nextn.py
+from functools import partial
+from typing import Any, Dict, Iterable, Optional, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_mtp.py` added +220/-0; `python/sglang/srt/models/mimo.py` renamed +0/-0
- 验证与风险: diff 自带测试面 `test/srt/models/test_mtp_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7370 - Clean unused import for mimo mtp model

- 链接: https://github.com/sgl-project/sglang/pull/7370
- 状态/时间: merged / 2025-06-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_mtp.py`；关联提交 `dea8aa7ab8e8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-18，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Clean unused import for mimo mtp model」；模型线: MiMo V2 Flash；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/mimo_mtp.py`；技术摘要: 覆盖「Clean unused import for mimo mtp model」；主要实现面是 `python/sglang/srt/models/mimo_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_mtp.py` modified +2/-18 (20 lines); hunks: -7,33 +7,17; symbols: MiMoMultiTokenPredictorLayer，涉及 `MiMoMultiTokenPredictorLayer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_mtp.py` modified +2/-18 (20 lines); hunks: -7,33 +7,17; symbols: MiMoMultiTokenPredictorLayer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_mtp.py
@@ -7,33 +7,17 @@
-from sglang.srt.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
-    split_tensor_along_last_dim,
-    tensor_model_parallel_all_gather,
-)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_mtp.py` modified +2/-18
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mimo_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15207 - [Feature] Xiaomi `MiMo-V2-Flash` day0 support

- 链接: https://github.com/sgl-project/sglang/pull/15207
- 状态/时间: merged / 2025-12-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/mimo_detector.py`；关联提交 `160a06cab23f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+5396/-169，可读 patch 6509 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Xiaomi `MiMo-V2-Flash` day0 support」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `python/sglang/srt/function_call/mimo_detector.py`；技术摘要: 覆盖「[Feature] Xiaomi `MiMo-V2-Flash` day0 support」；主要实现面是 `python/sglang/srt/function_call/mimo_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__，涉及 `_get_param_type, _convert_param_value, MiMoDetector`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/mimo_detector.py
@@ -0,0 +1,281 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/mimo_detector.py` added +281/-0
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- 链接: https://github.com/sgl-project/sglang/pull/15464
- 状态/时间: merged / 2025-12-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+66/-10，可读 patch 175 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize MiMo-V2-Flash by flashinfer fused allreduce」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mimo_v2_flash.py`；技术摘要: 覆盖「Optimize MiMo-V2-Flash by flashinfer fused allreduce」；主要实现面是 `python/sglang/srt/models/mimo_v2_flash.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunks: -13,7 +13,7; -45,7 +45,11; symbols: __init__, forward, forward_normal，涉及 `__init__, forward, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunks: -13,7 +13,7; -45,7 +45,11; symbols: __init__, forward, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -13,7 +13,7 @@
-from typing import Any, Dict, Iterable, Optional, Tuple, Union
+from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
@@ -45,7 +45,11 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
+from sglang.srt.layers.moe import (
+    get_moe_a2a_backend,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mimo_v2_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15488 - [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg

- 链接: https://github.com/sgl-project/sglang/pull/15488
- 状态/时间: merged / 2025-12-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+16/-16，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunks: -334,7 +334,6 @@ def __init__(; -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid，涉及 `__init__, profile_max_num_token, handle_max_mamba_cache`；`python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunks: -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):; -2263,6 +2263,8 @@ def _handle_cache_compatibility(self):; symbols: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference，涉及 `_handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunks: -334,7 +334,6 @@ def __init__(; -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid
  - `python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunks: -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):; -2263,6 +2263,8 @@ def _handle_cache_compatibility(self):; symbols: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -334,7 +334,6 @@ def __init__(
-        self.kv_cache_memory = 0
@@ -1582,10 +1581,9 @@ def profile_max_num_token(self, total_gpu_memory: int):
-        self.kv_cache_memory = int(rest_memory * (1 << 30))
-        max_num_token = int(self.kv_cache_memory // cell_size)
-        return max_num_token
+        return int(rest_memory * (1 << 30)) // cell_size
diff -- python/sglang/srt/server_args.py
@@ -1203,11 +1203,11 @@ def _handle_model_specific_adjustments(self):
-            self.swa_full_tokens_ratio = 1.0
-            logger.warning(
-                "Reset swa_full_tokens_ratio to 1.0 for MiMoV2FlashForCausalLM model"
-            )
+                self.swa_full_tokens_ratio = 1.0
+                logger.warning(
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +10/-12; `python/sglang/srt/server_args.py` modified +6/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18051 - [Fix] Remove no use code in MiMo-V2-Flash

- 链接: https://github.com/sgl-project/sglang/pull/18051
- 状态/时间: merged / 2026-02-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-20，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Remove no use code in MiMo-V2-Flash」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/mimo_v2_flash.py`；技术摘要: 覆盖「[Fix] Remove no use code in MiMo-V2-Flash」；主要实现面是 `python/sglang/srt/models/mimo_v2_flash.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20 (23 lines); hunks: -13,7 +13,7; -557,16 +557,10 @@ def forward(; symbols: forward, get_input_embedding, get_input_embeddings, set_eagle3_layers_to_capture，涉及 `forward, get_input_embedding, get_input_embeddings`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20 (23 lines); hunks: -13,7 +13,7; -557,16 +557,10 @@ def forward(; symbols: forward, get_input_embedding, get_input_embeddings, set_eagle3_layers_to_capture
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -13,7 +13,7 @@
-from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
+from typing import Any, Dict, Iterable, Optional, Tuple, Union
@@ -557,16 +557,10 @@ def forward(
-        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
-        hidden_states, residual = (
-            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +3/-20
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/mimo_v2_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17634 - [MiMoV2Flash] [feat]: support two batch overlap

- 链接: https://github.com/sgl-project/sglang/pull/17634
- 状态/时间: merged / 2026-02-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+292/-8，可读 patch 366 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MiMoV2Flash] [feat]: support two batch overlap」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`；技术摘要: 覆盖「[MiMoV2Flash] [feat]: support two batch overlap」；主要实现面是 `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a，涉及 `forward_deepep, op_gate, op_select_experts`；`python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunks: -51,6 +51,15 @@ def init_new_tbo(; -209,3 +218,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill，涉及 `init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunks: -51,6 +51,15 @@ def init_new_tbo(; -209,3 +218,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -19,18 +19,21 @@
+from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+    ScatterMode,
@@ -66,7 +69,12 @@
-from sglang.srt.utils import LazyValue, add_prefix, make_layers
+from sglang.srt.utils import (
diff -- python/sglang/srt/batch_overlap/operations_strategy.py
@@ -51,6 +51,15 @@ def init_new_tbo(
+        elif layer_name == "MiMoV2DecoderLayer":
+            return OperationsStrategy.concat(
+                [
+                    _compute_moe_mimov2_layer_operations_strategy_tbo(
+                        layer, forward_mode
+                    )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/batch_overlap/operations_strategy.py`, `python/sglang/srt/models/mimo_v2_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- 链接: https://github.com/sgl-project/sglang/pull/21414
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(MiMo-V2-Flash): add mimo reasoning parser」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`；技术摘要: 覆盖「fix(MiMo-V2-Flash): add mimo reasoning parser」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunks: -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatComple...; symbols: _get_reasoning_from_request，涉及 `_get_reasoning_from_request`；`python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -495,6 +495,7 @@ class ReasoningParser:; symbols: ReasoningParser，涉及 `ReasoningParser`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunks: -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatComple...; symbols: _get_reasoning_from_request
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -495,6 +495,7 @@ class ReasoningParser:; symbols: ReasoningParser
- 关键代码摘录:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -1268,6 +1268,12 @@ def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
+        if self.reasoning_parser in ["mimo"]:
+            # Models that require explicit enable thinking (enable_thinking=True)
+            return (
+                request.chat_template_kwargs is not None
+                and request.chat_template_kwargs.get("enable_thinking") is True
+            )
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -495,6 +495,7 @@ class ReasoningParser:
+        "mimo": Qwen3Detector,
```

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0; `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23851 - [Docs] add cookbook for MiMo-V2.5 family

- 链接: https://github.com/sgl-project/sglang/pull/23851
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；关联提交 `f34222da1b22`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1025/-1，可读 patch 1042 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] add cookbook for MiMo-V2.5 family」；模型线: MiMo V2 Flash；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；技术摘要: 覆盖「[Docs] add cookbook for MiMo-V2.5 family」；主要实现面是 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0 (626 lines); hunks: -0,0 +1,626；`docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0 (397 lines); hunks: -0,0 +1,397。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0 (626 lines); hunks: -0,0 +1,626
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0 (397 lines); hunks: -0,0 +1,397
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -0,0 +1,626 @@
+---
+title: MiMo-V2.5
+metatags:
+    description: "Deploy XiaomiMiMo MiMo-V2.5-Pro (1.02T MoE, text) and MiMo-V2.5 (310B MoE, multimodal) with SGLang — EAGLE speculative decoding, hybrid attention, and 1M-token c
+tag: NEW
+---
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -0,0 +1,397 @@
+export const MiMoV25Deployment = () => {
+  // MiMo-V2.5 family deployment matrix:
+  //   Variant × Hardware → slug, tp, multinode, blackwell
+  //
+  //   V2.5-Pro (1.02T / 42B active) — text-only:
+  //     H200  → tp=16, 2 nodes,     FP8 (Hopper: fa3 + DeepEP)
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` added +626/-0; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` added +397/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/docs.json`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23808 - [Feature] Xiaomi MiMo-V2.5-Pro day0 support

- 链接: https://github.com/sgl-project/sglang/pull/23808
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`；关联提交 `1a55646dcdf0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+80/-23，可读 patch 280 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Xiaomi MiMo-V2.5-Pro day0 support」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`；技术摘要: 覆盖「[Feature] Xiaomi MiMo-V2.5-Pro day0 support」；主要实现面是 `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/mimo_v2.py` renamed +27/-8 (35 lines); hunks: -76,7 +76,7; -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):; symbols: MiMoV2MoE, __init__, forward, MiMoV2DecoderLayer，涉及 `MiMoV2MoE, __init__, forward`；`python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6 (27 lines); hunks: -28,6 +28,7; -39,23 +40,23; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2MTP，涉及 `MiMoV2MTPLayer, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2.py` renamed +27/-8 (35 lines); hunks: -76,7 +76,7; -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):; symbols: MiMoV2MoE, __init__, forward, MiMoV2DecoderLayer
  - `python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6 (27 lines); hunks: -28,6 +28,7; -39,23 +40,23; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2MTP
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_v2.py
@@ -76,7 +76,7 @@
-MiMoV2FlashConfig = None
+MiMoV2Config = None
@@ -178,7 +178,7 @@ class MiMoV2MoE(nn.Module):
-        config: MiMoV2FlashConfig,
+        config: MiMoV2Config,
@@ -562,7 +562,7 @@ def forward(
diff -- python/sglang/srt/models/mimo_v2_nextn.py
@@ -28,6 +28,7 @@
+    get_attention_tp_size,
@@ -39,23 +40,23 @@
-from sglang.srt.models.mimo_v2_flash import (
+from sglang.srt.models.mimo_v2 import (
-    MiMoV2FlashForCausalLM,
+    MiMoV2ForCausalLM,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2.py` renamed +27/-8; `python/sglang/srt/models/mimo_v2_nextn.py` renamed +21/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23945 - docs: enable MiMo V2.5 MTP cookbook path

- 链接: https://github.com/sgl-project/sglang/pull/23945
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；关联提交 `e458a9248fef`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+90/-88，可读 patch 308 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「docs: enable MiMo V2.5 MTP cookbook path」；模型线: MiMo V2 Flash；类别: 文档/测试/CI；主要 diff: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；技术摘要: 覆盖「docs: enable MiMo V2.5 MTP cookbook path」；主要实现面是 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80 (164 lines); hunks: -43,7 +43,7 @@ tag: NEW; -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...；`docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8 (14 lines); hunks: -15,7 +15,7 @@ export const MiMoV25Deployment = () => {; -44,7 +44,7 @@ export const MiMoV25Deployment = () => {。
- 代码 diff 细节:
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80 (164 lines); hunks: -43,7 +43,7 @@ tag: NEW; -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8 (14 lines); hunks: -15,7 +15,7 @@ export const MiMoV25Deployment = () => {; -44,7 +44,7 @@ export const MiMoV25Deployment = () => {
- 关键代码摘录:

```diff
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -43,7 +43,7 @@ tag: NEW
-- **Multi-Token Prediction (MTP)**: 3-layer MTP module accelerates decoding (329M params on V2.5; V2.5-Pro supports EAGLE speculative decoding on top of MTP).
+- **Multi-Token Prediction (MTP)**: 3-layer MTP module accelerates decoding. Both variants support EAGLE speculative decoding with MTP weights.
@@ -84,9 +84,10 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
-- The checkpoint has a TP=4-interleaved fused `qkv_proj`; attention-TP per DP group **must** be 4. So DP-attention is always required (`--dp = TP / 4`), and total GPUs must be a m
+- The checkpoint has a TP=4-interleaved fused `qkv_proj`; attention-TP per DP group **must** be 4. Use `--dp = TP / 4`; for TP > 4 this also requires DP-attention. Total GPUs must
+- EAGLE MTP uses the checkpoint's MTP weights. For H100/H200, enable `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, and `--enable-multi-layer-eagle`.
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -15,7 +15,7 @@ export const MiMoV25Deployment = () => {
-  //     EAGLE MTP — Pro only. Adds --speculative-* flags + SGLANG_ENABLE_SPEC_V2=1.
+  //     EAGLE MTP — adds --speculative-* flags + SGLANG_ENABLE_SPEC_V2=1.
@@ -44,7 +44,7 @@ export const MiMoV25Deployment = () => {
-        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "Pro only" },
+        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "EAGLE" },
@@ -68,8 +68,8 @@ export const MiMoV25Deployment = () => {
```

- 已读文件:
  - docs: `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +84/-80; `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +6/-8
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #23936 - mimo v2.5 pro sglang-jax cookbook

- 链接: https://github.com/sgl-project/sglang/pull/23936
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；关联提交 `6c7b2421816c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+114/-16，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「mimo v2.5 pro sglang-jax cookbook」；模型线: MiMo V2 Flash；类别: 文档/测试/CI；主要 diff: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`；技术摘要: 覆盖「mimo v2.5 pro sglang-jax cookbook」；主要实现面是 `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`, `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16 (94 lines); hunks: -34,10 +34,12 @@ export const MiMoV25Deployment = () => {; -93,15 +95,19 @@ export const MiMoV25Deployment = () => {；`docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0 (36 lines); hunks: -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../doc...; -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...。
- 代码 diff 细节:
  - `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16 (94 lines); hunks: -34,10 +34,12 @@ export const MiMoV25Deployment = () => {; -93,15 +95,19 @@ export const MiMoV25Deployment = () => {
  - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0 (36 lines); hunks: -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../doc...; -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressi...
- 关键代码摘录:

```diff
diff -- docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx
@@ -34,10 +34,12 @@ export const MiMoV25Deployment = () => {
-        { id: "h200",  label: "H200",  default: true  },
-        { id: "h100",  label: "H100",  default: false },
-        { id: "b200",  label: "B200",  default: false },
-        { id: "gb300", label: "GB300", default: false },
+        { id: "h200",     label: "H200",     default: true  },
+        { id: "h100",     label: "H100",     default: false },
diff -- docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx
@@ -65,6 +65,8 @@ Refer to the [official SGLang installation guide](../../../docs/get-started/inst
+**TPU (sgl-jax):** MiMo-V2.5-Pro can also be served on TPU via the JAX-based [sgl-jax](https://github.com/sgl-project/sglang-jax) runtime. The container image and `pip install` st
@@ -95,6 +97,40 @@ import { MiMoV25Deployment } from '/src/snippets/autoregressive/mimo-v25-deploym
+### 3.3 TPU Deployment (MiMo-V2.5-Pro, sgl-jax)
+MiMo-V2.5-Pro can also be served on TPU via [sgl-jax](https://github.com/sgl-project/sglang-jax). The runtime is a separate JAX-based stack (`sgl_jax.launch_server`); pick **TPU v
+| TPU Type | Topology | Chips/Node | Nodes | Total Chips | JAX Devices/Chip | Total JAX Devices (= `--tp-size`) |
+| --- | --- | --- | --- | --- | --- | --- |
```

- 已读文件:
  - docs: `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx` modified +78/-16; `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx` modified +36/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`, `docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24118 - fix: rename mimo spec threshold attr to num_accepted_drafts_thres

- 链接: https://github.com/sgl-project/sglang/pull/24118
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/8-gpu-models/test_mimo_models.py`；关联提交 `c54ada994bd5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: rename mimo spec threshold attr to num_accepted_drafts_thres」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `test/registered/8-gpu-models/test_mimo_models.py`；技术摘要: 覆盖「fix: rename mimo spec threshold attr to num_accepted_drafts_thres」；主要实现面是 `test/registered/8-gpu-models/test_mimo_models.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultSe...; symbols: TestMiMoV2Flash，涉及 `TestMiMoV2Flash`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultSe...; symbols: TestMiMoV2Flash
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_mimo_models.py
@@ -45,7 +45,7 @@ class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
-    accept_length_thres = 3.2
+    num_accepted_drafts_thres = 3.2
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_mimo_models.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_mimo_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23811 - [Feature] Xiaomi MiMo-V2.5 day0 support

- 链接: https://github.com/sgl-project/sglang/pull/23811
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/models/mimo_v2_nextn.py`, `python/sglang/srt/models/mimo_vl.py`, `python/sglang/srt/multimodal/processors/mimo_v2.py` 等 6 个文件；关联提交 `651af06a0b5e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+4369/-87，可读 patch 4885 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Xiaomi MiMo-V2.5 day0 support」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`；技术摘要: 覆盖「[Feature] Xiaomi MiMo-V2.5 day0 support」；主要实现面是 `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_audio.py`, `python/sglang/srt/models/mimo_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0 (2039 lines); hunks: -0,0 +1,2039; symbols: ImageInput, __post_init__, VideoInput, AudioInput，涉及 `ImageInput, __post_init__, VideoInput`；`python/sglang/srt/models/mimo_audio.py` added +1350/-0 (1350 lines); hunks: -0,0 +1,1350; symbols: flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update, longrope_frequency_update，涉及 `flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update`；`python/sglang/srt/models/mimo_vl.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed, sync_proj_weight_linear_format，涉及 `MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed`；`python/sglang/srt/models/mimo_v2.py` modified +222/-13 (235 lines); hunks: -13,13 +13,14; -63,11 +64,18; symbols: load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__, routed_experts_weights_of_layer，涉及 `load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0 (2039 lines); hunks: -0,0 +1,2039; symbols: ImageInput, __post_init__, VideoInput, AudioInput
  - `python/sglang/srt/models/mimo_audio.py` added +1350/-0 (1350 lines); hunks: -0,0 +1,1350; symbols: flash_attn_varlen_func, _compute_default_rope_parameters, _dynamic_rope_update, longrope_frequency_update
  - `python/sglang/srt/models/mimo_vl.py` added +507/-0 (507 lines); hunks: -0,0 +1,507; symbols: MiMoVLVisionConfig, __init__, MiMoVisionPatchEmbed, sync_proj_weight_linear_format
  - `python/sglang/srt/models/mimo_v2.py` modified +222/-13 (235 lines); hunks: -13,13 +13,14; -63,11 +64,18; symbols: load_mimo_v2_qkv_proj_weight, MiMoV2MLP, __init__, routed_experts_weights_of_layer
  - `python/sglang/srt/models/mimo_v2_nextn.py` modified +12/-7 (19 lines); hunks: -19,6 +19,7; -28,7 +29,6; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -0,0 +1,2039 @@
+"""MiMoV2 multimodal processor -- protocol, utilities, and processor."""
+import asyncio
+import base64
+import copy
+import io
+import json
diff -- python/sglang/srt/models/mimo_audio.py
@@ -0,0 +1,1350 @@
+"""MiMo audio: tokenizer, encoding utilities, and audio encoder."""
+# Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
+import logging
+import math
+import os
+import typing as tp
diff -- python/sglang/srt/models/mimo_vl.py
@@ -0,0 +1,507 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` added +2039/-0; `python/sglang/srt/models/mimo_audio.py` added +1350/-0; `python/sglang/srt/models/mimo_vl.py` added +507/-0; `python/sglang/srt/models/mimo_v2.py` modified +222/-13; `python/sglang/srt/models/mimo_v2_nextn.py` modified +12/-7
  - tests: `test/registered/8-gpu-models/test_mimo_models.py` modified +38/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/server_fixtures/mmmu_fixture.py`, `test/registered/8-gpu-models/test_mimo_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24931 - feat(mimo-v2): add EPD disaggregation support

- 链接: https://github.com/sgl-project/sglang/pull/24931
- 状态/时间: merged / 2026-05-18
- 反查来源: 2026-05-19 PR 补漏审计；从源码复核补记、上游 `origin/main@78cb38ed5` 提交历史和 GitHub Pull Request files API 反查；关联提交 `784fe7e99b80`。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+961/-289，可读 patch 1710 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(mimo-v2): add EPD disaggregation support」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py`；技术摘要: 覆盖「feat(mimo-v2): add EPD disaggregation support」，下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110 (669 lines); hunks: -268,6 +268,49  @@ def __post_init__(self):; -445,6 +488,207  @@ def __init__(; symbols: __post_init__, __init__, _process_audio_content, _process_video_audio_content，涉及 `__post_init__, __init__, _process_audio_content`；`python/sglang/srt/models/mimo_v2.py` modified +148/-28 (176 lines); hunks: -68,7 +68,11  @@ MultiModalityDataPaddingPatternMultimodalTokens,; -1007,6 +1011,11  @@ class MiMoV2ForCausalLM(nn.Module):; symbols: MiMoV2ForCausalLM, __init__, get_video_feature, forward，涉及 `MiMoV2ForCausalLM, __init__, get_video_feature`；`python/sglang/srt/managers/tokenizer_manager.py` modified +5/-0 (5 lines); hunks: -773,6 +773,11  @@ async def _tokenize_one_request(; symbols: _tokenize_one_request，涉及 `_tokenize_one_request`；`python/sglang/srt/disaggregation/encode_server.py` modified +136/-102 (238 lines); hunks: -24,7 +24,10  @@ from sglang.srt.configs.load_config import LoadConfig; -45,6 +48,7  @@ set_global_server_args_for_scheduler,; symbols: _get_mm_feature, _background_insert, wrap_one, _encode，涉及 `_get_mm_feature, _background_insert, wrap_one`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110 (669 lines); hunks: -268,6 +268,49  @@ def __post_init__(self):; -445,6 +488,207  @@ def __init__(; symbols: __post_init__, __init__, _process_audio_content, _process_video_audio_content，涉及 `__post_init__, __init__, _process_audio_content`
  - `python/sglang/srt/models/mimo_v2.py` modified +148/-28 (176 lines); hunks: -68,7 +68,11  @@ MultiModalityDataPaddingPatternMultimodalTokens,; -1007,6 +1011,11  @@ class MiMoV2ForCausalLM(nn.Module):; symbols: MiMoV2ForCausalLM, __init__, get_video_feature, forward，涉及 `MiMoV2ForCausalLM, __init__, get_video_feature`
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-0 (5 lines); hunks: -773,6 +773,11  @@ async def _tokenize_one_request(; symbols: _tokenize_one_request，涉及 `_tokenize_one_request`
  - `python/sglang/srt/disaggregation/encode_server.py` modified +136/-102 (238 lines); hunks: -24,7 +24,10  @@ from sglang.srt.configs.load_config import LoadConfig; -45,6 +48,7  @@ set_global_server_args_for_scheduler,; symbols: _get_mm_feature, _background_insert, wrap_one, _encode，涉及 `_get_mm_feature, _background_insert, wrap_one`
  - `python/sglang/srt/disaggregation/encode_receiver.py` modified +64/-18 (82 lines); hunks: -193,7 +193,29  @@ def copy_without_embedding(self):; -231,6 +253,7  @@ def __init__(; symbols: copy_without_embedding, __init__, _set_part_grid, _set_video_meta_for_part，涉及 `copy_without_embedding, __init__, _set_part_grid`
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -268,6 +268,49 @@ def __post_init__(self):
+def _decode_frames_and_timestamps(vdw, ele):
+    # Shared E/D frame-sampling recipe: smart_nframes + linspace + permute.
+    total_frames, video_fps = len(vdw), vdw.avg_fps
+    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
+    idx = list(np.unique(np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)))
+    video_tensor = vdw.get_frames_as_tensor(idx).permute(0, 3, 1, 2).float()
+    timestamps = torch.as_tensor(idx, dtype=torch.float32) / video_fps
+    return video_tensor, timestamps
diff -- python/sglang/srt/models/mimo_v2.py
@@ -68,7 +68,11 @@
-from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
+from sglang.srt.managers.schedule_batch import (
+    Modality,
+    MultimodalDataItem,
+    MultimodalInputs,
+)
@@ -1007,6 +1011,11 @@ class MiMoV2ForCausalLM(nn.Module):
+    # Prefixes for weight routing in encoder_only/language_only modes
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -773,6 +773,11 @@ async def _tokenize_one_request(
+                    if self.server_args.language_only:
+                        logger.warning(
+                            "Encoder embedding not available, "
+                            "falling back to local mm processing"
+                        )
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +559/-110; `python/sglang/srt/models/mimo_v2.py` modified +148/-28; `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-0; `python/sglang/srt/disaggregation/encode_server.py` modified +136/-102
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/models/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py`；风险点是权重加载、并行切分、attention/MoE 后端选择、量化 dtype 和 parser 输出，需要至少做一次真实 checkpoint 或等价 smoke。

### PR #25588 - perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode

- 链接: https://github.com/sgl-project/sglang/pull/25588
- 状态/时间: merged / 2026-05-19
- 反查来源: 2026-05-19 PR 补漏审计；从源码复核补记、上游 `origin/main@78cb38ed5` 提交历史和 GitHub Pull Request files API 反查；关联提交 `f0763859edbb`。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+75/-8，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/utils/video_decoder.py`；技术摘要: 覆盖「perf(mimo-v2-epd): enable GPU image preprocess and parallel video decode」，下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2 (28 lines); hunks: -359,11 +359,13  @@ def __init__(; -546,6 +548,20  @@ def _as_dict(obj):; symbols: __init__, _as_dict, _load_video_for_encoder, preprocess_for_encoder，涉及 `__init__, _as_dict, _load_video_for_encoder`；`python/sglang/srt/managers/tokenizer_manager.py` modified +1/-0 (1 lines); hunks: -453,6 +453,7  @@ def init_disaggregation(self):; symbols: init_disaggregation，涉及 `init_disaggregation`；`python/sglang/srt/utils/video_decoder.py` modified +48/-6 (54 lines); hunks: -1,6 +1,7  @@ """Unified video decoder: torchcodec preferred, decord as fallback."""; -38,31 +39,38  @@ class VideoDecoderWrapper:; symbols: VideoDecoderWrapper, get_frames_as_tensor, source_bytes，涉及 `VideoDecoderWrapper, get_frames_as_tensor, source_bytes`。
- 代码 diff 细节:
  - `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2 (28 lines); hunks: -359,11 +359,13  @@ def __init__(; -546,6 +548,20  @@ def _as_dict(obj):; symbols: __init__, _as_dict, _load_video_for_encoder, preprocess_for_encoder，涉及 `__init__, _as_dict, _load_video_for_encoder`
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +1/-0 (1 lines); hunks: -453,6 +453,7  @@ def init_disaggregation(self):; symbols: init_disaggregation，涉及 `init_disaggregation`
  - `python/sglang/srt/utils/video_decoder.py` modified +48/-6 (54 lines); hunks: -1,6 +1,7  @@ """Unified video decoder: torchcodec preferred, decord as fallback."""; -38,31 +39,38  @@ class VideoDecoderWrapper:; symbols: VideoDecoderWrapper, get_frames_as_tensor, source_bytes，涉及 `VideoDecoderWrapper, get_frames_as_tensor, source_bytes`
- 关键代码摘录:

```diff
diff -- python/sglang/srt/multimodal/processors/mimo_v2.py
@@ -359,11 +359,13 @@ def __init__(
+        video_decode_num_threads=0,
+        self.video_decode_num_threads = video_decode_num_threads
@@ -546,6 +548,20 @@ def _as_dict(obj):
+        image_cfg = (mm_config or {}).get("image", {})
+        if "device" in image_cfg:
+            kwargs["device"] = image_cfg["device"]
+
+        video_cfg = (mm_config or {}).get("video", {})
diff -- python/sglang/srt/managers/tokenizer_manager.py
@@ -453,6 +453,7 @@ def init_disaggregation(self):
+                hf_config=self.model_config.hf_config,
diff -- python/sglang/srt/utils/video_decoder.py
@@ -1,6 +1,7 @@
+import os
@@ -38,31 +39,38 @@ class VideoDecoderWrapper:
-    def __init__(self, source, device: str = "cpu"):
+    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):
+        num_decode_threads: number of parallel decoder instances for frame
+            extraction (torchcodec only). 0 = auto (capped at 16),
+            1 = single decoder. Set > 1 to split frame indices across
+            multiple decoders in parallel threads.
```

- 已读文件:
  - runtime: `python/sglang/srt/multimodal/processors/mimo_v2.py` modified +26/-2; `python/sglang/srt/managers/tokenizer_manager.py` modified +1/-0; `python/sglang/srt/utils/video_decoder.py` modified +48/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/multimodal/processors/mimo_v2.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/utils/video_decoder.py`；风险点是权重加载、并行切分、attention/MoE 后端选择、量化 dtype 和 parser 输出，需要至少做一次真实 checkpoint 或等价 smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
