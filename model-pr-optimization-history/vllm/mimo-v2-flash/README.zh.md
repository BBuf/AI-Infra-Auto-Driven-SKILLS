# vllm MiMo V2 Flash 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/mimo.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433) |
| `vllm/model_executor/models/mimo_mtp.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433), [#25136](https://github.com/vllm-project/vllm/pull/25136) |
| `vllm/model_executor/models/mimo_v2_flash.py` | [#30836](https://github.com/vllm-project/vllm/pull/30836), [#31175](https://github.com/vllm-project/vllm/pull/31175), [#40045](https://github.com/vllm-project/vllm/pull/40045) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-12 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | `vllm/model_executor/models/mimo_mtp.py` |
| 2025-12-19 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-01-05 | [#31175](https://github.com/vllm-project/vllm/pull/31175) | merged | [Bugfix] Properly apply v_scale for mimo_v2_flash | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-04-24 | [#40045](https://github.com/vllm-project/vllm/pull/40045) | merged | [Attention] use diff kv backend for mimo v2 flash | `vllm/model_executor/models/mimo_v2_flash.py` |

## 逐 PR diff 审计卡

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- 链接: https://github.com/vllm-project/vllm/pull/17433
- 状态/时间: merged / 2025-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo.py`, `vllm/model_executor/models/mimo_mtp.py`；关联提交 `acee8f48aa9c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+507/-4，可读 patch 576 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「[Model] Support MiMo-7B inference with MTP」，变更集中在 `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`。PR 描述补充为：MiMo-7B is a decoder-only Transformer LM with MTP layers that exhibits extraordinary reasoning potential. The model and technical report are open-sourced in https://github.com/X...
- 实现要点: `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunks: -0,0 +1,283; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor，涉及 `MiMoMultiTokenPredictorLayer, __init__, forward`；`vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunks: -0,0 +1,190; symbols: MiMoModel, forward, load_weights, MiMoForCausalLM，涉及 `MiMoModel, forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunks: -0,0 +1,283; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor
  - `vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunks: -0,0 +1,190; symbols: MiMoModel, forward, load_weights, MiMoForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_mtp.py
@@ -0,0 +1,283 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
+# Copyright 2025 Xiaomi Corporation.
+# Copyright 2023 The vLLM team.
+# Copyright 2024 DeepSeek-AI team.
diff -- vllm/model_executor/models/mimo.py
@@ -0,0 +1,190 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
+# Copyright 2025 Xiaomi Corporation.
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_mtp.py` added +283/-0; `vllm/model_executor/models/mimo.py` added +190/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25136 - [spec decode] Fix MTP inference path for MiMo-7B model

- 链接: https://github.com/vllm-project/vllm/pull/25136
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_mtp.py`；关联提交 `c4cb0af98a8e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+20/-6，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 修复已暴露的启动、加载、解析或数值问题，标题为「[spec decode] Fix MTP inference path for MiMo-7B model」，变更集中在 `vllm/model_executor/models/mimo_mtp.py`。PR 描述补充为：## Purpose Fix MiMo-7B MTP inference path ## Test Plan Acceptance rate test + lm_eval ## Test Result - acceptance rate - lm_eval baseline - lm_eval with mtp --- Essential Elemen...
- 实现要点: `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunks: -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name，涉及 `load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunks: -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_mtp.py
@@ -241,17 +241,27 @@ def load_weights(self, weights: Iterable[tuple[str,
+        # append mtp_start_layer_idx
+        pattern = r"(model\.mtp_layers\.)(\d+)(\.)"
+        match = re.match(pattern, name)
+        if match:
+            original_num = int(match.group(2))
+            new_num = original_num + self.config.num_hidden_layers
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_mtp.py` modified +14/-4
- 验证与风险: runtime 路径改动集中在 `vllm/config/speculative.py`, `vllm/model_executor/models/mimo_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30836 - [Model] Add MiMo-V2-Flash support

- 链接: https://github.com/vllm-project/vllm/pull/30836
- 状态/时间: merged / 2025-12-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2_flash.py`；关联提交 `969bbc7c6166`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+789/-13，可读 patch 946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 补齐模型支持入口或运行时能力，标题为「[Model] Add MiMo-V2-Flash support」，变更集中在 `vllm/model_executor/models/mimo_v2_flash.py`。PR 描述补充为：## Purpose Add support for MiMo-V2-Flash. ## Examples ### Example 1 ### Example 2 ## Accuracy GSM8K
- 实现要点: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE，涉及 `MiMoV2MLP, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -0,0 +1,720 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from itertools import islice
+import torch
+from torch import nn
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31175 - [Bugfix] Properly apply v_scale for mimo_v2_flash

- 链接: https://github.com/vllm-project/vllm/pull/31175
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2_flash.py`；关联提交 `951302989814`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-13，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Properly apply v_scale for mimo_v2_flash」，变更集中在 `vllm/model_executor/models/mimo_v2_flash.py`。PR 描述补充为：## Purpose Noticed this when comparing with the Transformers implementation Before: After: ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Chec...
- 实现要点: `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13 (23 lines); hunks: -211,6 +211,7 @@ def __init__(; -241,6 +242,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13 (23 lines); hunks: -211,6 +211,7 @@ def __init__(; -241,6 +242,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -211,6 +211,7 @@ def __init__(
+        v_scale: float | None = None,
@@ -241,6 +242,7 @@ def __init__(
+        self.v_scale = v_scale
@@ -304,6 +306,10 @@ def forward(
+        # Apply v_scale before attention
+        if self.v_scale is not None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +10/-13
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mimo_v2_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40045 - [Attention] use diff kv backend for mimo v2 flash

- 链接: https://github.com/vllm-project/vllm/pull/40045
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2_flash.py`；关联提交 `e8ee2a78dbc0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+112/-24，可读 patch 270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 MiMo V2 Flash 优化关键推理路径或后端选择，标题为「[Attention] use diff kv backend for mimo v2 flash」，变更集中在 `vllm/model_executor/models/mimo_v2_flash.py`。PR 描述补充为：## Purpose ### Diff kv A key characteristic of mimo v2 flash architecture is that the attention layer uses different head dimensions for keys and values (v_head_dim != head_dim)...
- 实现要点: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_flash.py
@@ -46,6 +46,9 @@
+from vllm.v1.attention.backends.flash_attn_diffkv import (
+    FlashAttentionDiffKVBackend,
+)
@@ -287,6 +290,15 @@ def __init__(
+        # Use DiffKV backend when V has a different head dim than K
+        if self.v_head_dim != self.head_dim:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/v1/attention/backends/fa_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
