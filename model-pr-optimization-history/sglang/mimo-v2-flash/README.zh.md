# sglang MiMo V2 Flash 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2-Flash.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/mimo-v2-flash-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/mimo_detector.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `python/sglang/srt/models/mimo.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059) |
| `python/sglang/srt/models/mimo_mtp.py` | [#6059](https://github.com/sgl-project/sglang/pull/6059), [#7370](https://github.com/sgl-project/sglang/pull/7370) |
| `python/sglang/srt/models/mimo_v2_flash.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207), [#15464](https://github.com/sgl-project/sglang/pull/15464), [#17634](https://github.com/sgl-project/sglang/pull/17634), [#18051](https://github.com/sgl-project/sglang/pull/18051) |
| `python/sglang/srt/models/mimo_v2_flash_nextn.py` | [#15207](https://github.com/sgl-project/sglang/pull/15207) |
| `test/registered/8-gpu-models/test_mimo_models.py` | 无直接 PR 号提交 |
| `test/registered/ascend/llm_models/test_npu_mimo_7b_rl.py` | 无直接 PR 号提交 |
| `test/registered/ascend/vlm_models/test_npu_mimo_vl_7b_rl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 6
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-22 | [#6059](https://github.com/sgl-project/sglang/pull/6059) | merged | Support XiaomiMiMo inference with mtp | `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py` |
| 2025-06-20 | [#7370](https://github.com/sgl-project/sglang/pull/7370) | merged | Clean unused import for mimo mtp model | `python/sglang/srt/models/mimo_mtp.py` |
| 2025-12-19 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`, `python/sglang/srt/function_call/mimo_detector.py` |
| 2025-12-20 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-25 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-02-01 | [#18051](https://github.com/sgl-project/sglang/pull/18051) | merged | [Fix] Remove no use code in MiMo-V2-Flash | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-02-02 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2026-04-01 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |

## 逐 PR diff 审计卡

### PR #6059 - Support XiaomiMiMo inference with mtp

- 链接: https://github.com/sgl-project/sglang/pull/6059
- 状态/时间: merged / 2025-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo.py`, `python/sglang/srt/models/mimo_mtp.py`；关联提交 `a6ae3af15e84`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+344/-6，可读 patch 388 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「Support XiaomiMiMo inference with mtp」，变更集中在 `python/sglang/srt/models/mimo_mtp.py`, `python/sglang/srt/models/mimo.py`。PR 描述补充为：## Motivation Support XiaomiMiMo inference with mtp ## Modifications Add new model support. Add corresponding MTP accuracy & latency test ### How to start server ## Test Result...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_mtp.py`；关联提交 `dea8aa7ab8e8`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-18，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「Clean unused import for mimo mtp model」，变更集中在 `python/sglang/srt/models/mimo_mtp.py`。PR 描述补充为：## Motivation ## Modifications Clean unused import for mimo mtp model. ## Checklist - [x] Format your code according to the Code Formatting with Pre-Commit. - [ ] Add unit tests...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/mimo_detector.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`；关联提交 `160a06cab23f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+5396/-169，可读 patch 6509 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「[Feature] Xiaomi `MiMo-V2-Flash` day0 support」，变更集中在 `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/models/mimo_v2_flash_nextn.py`, `python/sglang/srt/function_call/mimo_detector.py`。PR 描述补充为：## Motivation MiMo-V2-Flash is a Mixture-of-Experts (MoE) language model with 309B total parameters and 15B active parameters. Designed for high-speed reasoning and agentic work...
- 实现要点: `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunks: -0,0 +1,927; symbols: MiMoV2MLP, __init__, forward, MoEGate，涉及 `MiMoV2MLP, __init__, forward`；`python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2ModelNextN，涉及 `MiMoV2MTPLayer, __init__, forward`；`python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__，涉及 `_get_param_type, _convert_param_value, MiMoDetector`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunks: -0,0 +1,927; symbols: MiMoV2MLP, __init__, forward, MoEGate
  - `python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0 (366 lines); hunks: -0,0 +1,366; symbols: MiMoV2MTPLayer, __init__, forward, MiMoV2ModelNextN
  - `python/sglang/srt/function_call/mimo_detector.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: _get_param_type, _convert_param_value, MiMoDetector, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/mimo_v2_flash.py
@@ -0,0 +1,927 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/mimo_v2_flash_nextn.py
@@ -0,0 +1,366 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/function_call/mimo_detector.py
@@ -0,0 +1,281 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0; `python/sglang/srt/models/mimo_v2_flash_nextn.py` added +366/-0; `python/sglang/srt/function_call/mimo_detector.py` added +281/-0
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- 链接: https://github.com/sgl-project/sglang/pull/15464
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_v2_flash.py`；关联提交 `165f5c04cbc2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+66/-10，可读 patch 175 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 优化关键推理路径或后端选择，标题为「Optimize MiMo-V2-Flash by flashinfer fused allreduce」，变更集中在 `python/sglang/srt/models/mimo_v2_flash.py`。PR 描述补充为：## Motivation This PR is to make MiMo-V2-Flash model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add. The E2E TTFT reduce 5.1%. ## Modifications ## Ac...
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
- 动机: 该 PR 围绕 MiMo V2 Flash 修复已暴露的启动、加载、解析或数值问题，标题为「[MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg」，变更集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`。PR 描述补充为：## Motivation MiMoV2Flash uses SWA, it should respect the argument `--swa-full-tokens-ratio` to allocate corresponding size of KV cache. ## Modifications ## Accuracy Tests ## Be...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_v2_flash.py`；关联提交 `9227d4f74883`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-20，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「[Fix] Remove no use code in MiMo-V2-Flash」，变更集中在 `python/sglang/srt/models/mimo_v2_flash.py`。PR 描述补充为：## Motivation https://github.com/sgl-project/sglang/pull/15464 introduced some code piece unrelated with flashinfer fused allreduce which had no usage. This PR is to wipe them o...
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/mimo_v2_flash.py`；关联提交 `cbf150039037`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+292/-8，可读 patch 366 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「[MiMoV2Flash] [feat]: support two batch overlap」，变更集中在 `python/sglang/srt/models/mimo_v2_flash.py`。PR 描述补充为：## Motivation support mimo_v2_flash two batch overlap: p: d: lb: ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according t...
- 实现要点: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a，涉及 `forward_deepep, op_gate, op_select_experts`。
- 代码 diff 细节:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: -19,18 +19,21; -66,7 +69,12; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a
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
```

- 已读文件:
  - runtime: `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/batch_overlap/operations_strategy.py`, `python/sglang/srt/models/mimo_v2_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- 链接: https://github.com/sgl-project/sglang/pull/21414
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「fix(MiMo-V2-Flash): add mimo reasoning parser」，变更集中在 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`。PR 描述补充为：## Motivation MiMo-V2-Flash and similar models default enable_thinking to false in their chat templates, but qwen3-family requests without the flag are currently treated as reas...
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

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
