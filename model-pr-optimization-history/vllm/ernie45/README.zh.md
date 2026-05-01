# vllm ERNIE 4.5 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `7075df79b3`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/model_executor/test_ernie45_vl_mrope.py` | [#39753](https://github.com/vllm-project/vllm/pull/39753) |
| `tests/reasoning/test_ernie45_reasoning_parser.py` | [#25027](https://github.com/vllm-project/vllm/pull/25027) |
| `tests/tool_parsers/test_ernie45_moe_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/ernie45.py` | [#21735](https://github.com/vllm-project/vllm/pull/21735) |
| `vllm/model_executor/models/ernie45_moe.py` | [#25936](https://github.com/vllm-project/vllm/pull/25936), [#26684](https://github.com/vllm-project/vllm/pull/26684), [#27316](https://github.com/vllm-project/vllm/pull/27316) |
| `vllm/model_executor/models/ernie45_vl.py` | [#39753](https://github.com/vllm-project/vllm/pull/39753) |
| `vllm/model_executor/models/ernie45_vl_moe.py` | [#25936](https://github.com/vllm-project/vllm/pull/25936), [#26885](https://github.com/vllm-project/vllm/pull/26885) |
| `vllm/model_executor/models/ernie_mtp.py` | 无直接 PR 号提交 |
| `vllm/reasoning/ernie45_reasoning_parser.py` | [#25027](https://github.com/vllm-project/vllm/pull/25027), [#27973](https://github.com/vllm-project/vllm/pull/27973) |
| `vllm/tool_parsers/ernie45_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 6
- 当前文档总 PR 数: 14
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-02 | [#20220](https://github.com/vllm-project/vllm/pull/20220) | merged | [Model] Add Ernie4.5 and Ernie4.5MoE Model Support | `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py` |
| 2025-07-28 | [#21717](https://github.com/vllm-project/vllm/pull/21717) | merged | [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-07-28 | [#21735](https://github.com/vllm-project/vllm/pull/21735) | merged | [`Ernie 4.5`] Name Change for Base 0.3B Model | `vllm/model_executor/models/ernie45.py` |
| 2025-08-27 | [#22514](https://github.com/vllm-project/vllm/pull/22514) | merged | [Model] Add Ernie4.5 VL Model Support | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py` |
| 2025-09-09 | [#24074](https://github.com/vllm-project/vllm/pull/24074) | merged | [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs | `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py` |
| 2025-09-30 | [#25936](https://github.com/vllm-project/vllm/pull/25936) | merged | [Bugfix][Model]fix ernie45 moe gate&bias dtype to float32 | `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/models/ernie45_moe.py` |
| 2025-10-12 | [#22100](https://github.com/vllm-project/vllm/pull/22100) | merged | [EPLB] Support ernie4.5-moe | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-10-13 | [#25027](https://github.com/vllm-project/vllm/pull/25027) | merged | [Model] Add reasoning_parser and tool_parser for Ernie45 thinking | `vllm/reasoning/ernie45_reasoning_parser.py`, `tests/reasoning/test_ernie45_reasoning_parser.py` |
| 2025-10-14 | [#26684](https://github.com/vllm-project/vllm/pull/26684) | merged | [Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-10-16 | [#26885](https://github.com/vllm-project/vllm/pull/26885) | merged | [Model][Bugfix] fix ernie45 vl run failed from shared experts optimization | `vllm/model_executor/models/ernie45_vl_moe.py` |
| 2025-10-27 | [#27316](https://github.com/vllm-project/vllm/pull/27316) | merged | [Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple | `vllm/model_executor/models/ernie45_moe.py` |
| 2025-11-04 | [#27973](https://github.com/vllm-project/vllm/pull/27973) | merged | [Model] fix ernie45 reasoning_parser | `vllm/reasoning/ernie45_reasoning_parser.py` |
| 2025-12-25 | [#31274](https://github.com/vllm-project/vllm/pull/31274) | merged | [Model][Ernie4.5-VL] Support video metadata for timestamp rendering | `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py` |
| 2026-04-14 | [#39753](https://github.com/vllm-project/vllm/pull/39753) | merged | [Model] Use mm_features for Ernie-4.5 VL M-RoPE | `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py` |

## 逐 PR diff 审计卡

### PR #20220 - [Model] Add Ernie4.5 and Ernie4.5MoE Model Support

- 链接: https://github.com/vllm-project/vllm/pull/20220
- 状态/时间: merged / 2025-07-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+634/-0，可读 patch 657 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Ernie4.5 and Ernie4.5MoE Model Support」；模型线: ERNIE 4.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`；技术摘要: 覆盖「[Model] Add Ernie4.5 and Ernie4.5MoE Model Support」；主要实现面是 `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunks: -0,0 +1,583; symbols: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE，涉及 `Ernie4_5_MoeMLP, __init__, forward`；`vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: Ernie4_5_ForCausalLM, __init__，涉及 `Ernie4_5_ForCausalLM, __init__`；`tests/models/registry.py` modified +4/-0 (4 lines); hunks: -162,6 +162,10 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`docs/models/supported_models.md` modified +2/-0 (2 lines); hunks: -330,6 +330,8 @@ Specified using `--task generate`.。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunks: -0,0 +1,583; symbols: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE
  - `vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: Ernie4_5_ForCausalLM, __init__
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -162,6 +162,10 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +2/-0 (2 lines); hunks: -330,6 +330,8 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: -53,6 +53,8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -0,0 +1,583 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Baidu team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/ernie45.py
@@ -0,0 +1,43 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Baidu team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- tests/models/registry.py
@@ -162,6 +162,10 @@ def check_available_online(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` added +583/-0; `vllm/model_executor/models/ernie45.py` added +43/-0; `vllm/model_executor/models/registry.py` modified +2/-0
  - tests: `tests/models/registry.py` modified +4/-0
  - docs: `docs/models/supported_models.md` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21717 - [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts

- 链接: https://github.com/vllm-project/vllm/pull/21717
- 状态/时间: merged / 2025-07-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-5，可读 patch 39 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_moe.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunks: -109,8 +109,8 @@ def __init__(; -137,7 +137,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunks: -109,8 +109,8 @@ def __init__(; -137,7 +137,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -109,8 +109,8 @@ def __init__(
-        self.moe_num_shared_experts = getattr(config, "moe_num_shared_experts",
-                                              None)
+        self.has_shared_experts = (getattr(config, "moe_num_shared_experts", 0)
+                                   > 0)
@@ -137,7 +137,7 @@ def __init__(
-        if self.moe_num_shared_experts is not None:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +6/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21735 - [`Ernie 4.5`] Name Change for Base 0.3B Model

- 链接: https://github.com/vllm-project/vllm/pull/21735
- 状态/时间: merged / 2025-07-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/ernie45.py`；关联提交 `656c24f1b5d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+8/-8，可读 patch 51 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[`Ernie 4.5`] Name Change for Base 0.3B Model」；模型线: ERNIE 4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/ernie45.py`；技术摘要: 覆盖「[`Ernie 4.5`] Name Change for Base 0.3B Model」；主要实现面是 `vllm/model_executor/models/ernie45.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7; symbols: Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__，涉及 `Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7; symbols: Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45.py
@@ -28,7 +28,7 @@
-class Ernie4_5_ForCausalLM(LlamaForCausalLM):
+class Ernie4_5ForCausalLM(LlamaForCausalLM):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22514 - [Model] Add Ernie4.5 VL Model Support

- 链接: https://github.com/vllm-project/vllm/pull/22514
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+2463/-0，可读 patch 2540 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Ernie4.5 VL Model Support」；模型线: ERNIE 4.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`；技术摘要: 覆盖「[Model] Add Ernie4.5 VL Model Support」；主要实现面是 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunks: -0,0 +1,1504; symbols: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave，涉及 `rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision`；`vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunks: -0,0 +1,723; symbols: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward，涉及 `Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__`；`vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunks: -393,6 +393,15 @@ def get_input_positions_tensor(; -513,6 +522,120 @@ def _glm4v_get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor，涉及 `get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor`；`vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: Ernie4_5_VLRotaryEmbedding, forward，涉及 `Ernie4_5_VLRotaryEmbedding, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunks: -0,0 +1,1504; symbols: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave
  - `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunks: -0,0 +1,723; symbols: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunks: -393,6 +393,15 @@ def get_input_positions_tensor(; -513,6 +522,120 @@ def _glm4v_get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: Ernie4_5_VLRotaryEmbedding, forward
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -396,6 +396,8 @@ def check_available_online(; symbols: check_available_online
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl.py
@@ -0,0 +1,1504 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Baidu team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/ernie45_vl_moe.py
@@ -0,0 +1,723 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 The Baidu team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/layers/rotary_embedding/mrope.py
@@ -393,6 +393,15 @@ def get_input_positions_tensor(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` added +1504/-0; `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0; `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0; `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24074 - [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs

- 链接: https://github.com/vllm-project/vllm/pull/24074
- 状态/时间: merged / 2025-09-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-7，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix][Model] Fix Ernie4.5-VL hanging on long inputs」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；技术摘要: 覆盖「[BugFix][Model] Fix Ernie4.5-VL hanging on long inputs」；主要实现面是 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunks: -66,8 +66,6; -839,6 +837,15 @@ def get_image_processor(self, **kwargs: object):; symbols: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info，涉及 `get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item`；`vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunks: -287,8 +287,13 @@ def forward(; -310,7 +315,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunks: -66,8 +66,6; -839,6 +837,15 @@ def get_image_processor(self, **kwargs: object):; symbols: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunks: -287,8 +287,13 @@ def forward(; -310,7 +315,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl.py
@@ -66,8 +66,6 @@
-_MAX_FRAMES_PER_VIDEO = 16
@@ -839,6 +837,15 @@ def get_image_processor(self, **kwargs: object):
+    def get_mm_max_tokens_per_item(
+        self,
+        seq_len: int,
+        mm_counts: Mapping[str, int],
diff -- vllm/model_executor/models/ernie45_vl_moe.py
@@ -287,8 +287,13 @@ def forward(
-        if visual_token_mask is not None and visual_token_mask.any():
-            # assert visual_token_mask.shape[0] != hidden_states.shape[0]
+        if visual_token_mask is not None and visual_token_mask.all():
+            # only vision modal input
+            router_logits, _ = self.vision_experts_gate(hidden_states)
+            final_hidden_states = self.vision_experts(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +10/-4; `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25936 - [Bugfix][Model]fix ernie45 moe gate&bias dtype to float32

- 链接: https://github.com/vllm-project/vllm/pull/25936
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；关联提交 `ef6e0e7132ec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+13/-7，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model]fix ernie45 moe gate&bias dtype to float32」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/models/ernie45_moe.py`；技术摘要: 覆盖「[Bugfix][Model]fix ernie45 moe gate&bias dtype to float32」；主要实现面是 `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/models/ernie45_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5 (15 lines); hunks: -199,7 +199,7 @@ def __init__(; -209,6 +209,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/ernie45_moe.py` modified +3/-2 (5 lines); hunks: -120,11 +120,12 @@ def __init__(; -157,7 +158,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5 (15 lines); hunks: -199,7 +199,7 @@ def __init__(; -209,6 +209,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/ernie45_moe.py` modified +3/-2 (5 lines); hunks: -120,11 +120,12 @@ def __init__(; -157,7 +158,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl_moe.py
@@ -199,7 +199,7 @@ def __init__(
-            torch.empty(2, config.moe_num_experts[0]))
+            torch.empty(2, config.moe_num_experts[0], dtype=torch.float32))
@@ -209,6 +209,7 @@ def __init__(
+                params_dtype=torch.float32,
@@ -238,6 +239,7 @@ def __init__(
+                params_dtype=torch.float32,
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -120,11 +120,12 @@ def __init__(
+                                     params_dtype=torch.float32,
-            torch.empty(config.moe_num_experts))
+            torch.empty(config.moe_num_experts, dtype=torch.float32))
@@ -157,7 +158,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states)
+        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5; `vllm/model_executor/models/ernie45_moe.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22100 - [EPLB] Support ernie4.5-moe

- 链接: https://github.com/vllm-project/vllm/pull/22100
- 状态/时间: merged / 2025-10-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+132/-7，可读 patch 243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[EPLB] Support ernie4.5-moe」；模型线: ERNIE 4.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/ernie45_moe.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/ernie45_moe.py` modified +132/-7 (139 lines); hunks: -33,8 +33,12; -58,7 +62,7; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_moe.py` modified +132/-7 (139 lines); hunks: -33,8 +33,12; -58,7 +62,7; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -33,8 +33,12 @@
-from vllm.config import CacheConfig, VllmConfig
-from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
+from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
+from vllm.distributed import (
+    get_ep_group,
+    get_pp_group,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +132/-7
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25027 - [Model] Add reasoning_parser and tool_parser for Ernie45 thinking

- 链接: https://github.com/vllm-project/vllm/pull/25027
- 状态/时间: merged / 2025-10-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_ernie45_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`；关联提交 `782505ed8eb4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+870/-0，可读 patch 909 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add reasoning_parser and tool_parser for Ernie45 thinking」；模型线: ERNIE 4.5；类别: 文档/测试/CI；主要 diff: `vllm/reasoning/ernie45_reasoning_parser.py`, `tests/reasoning/test_ernie45_reasoning_parser.py`；技术摘要: 覆盖「[Model] Add reasoning_parser and tool_parser for Ernie45 thinking」；主要实现面是 `vllm/reasoning/ernie45_reasoning_parser.py`, `tests/reasoning/test_ernie45_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: Ernie45ReasoningParser, start_token, end_token, __init__，涉及 `Ernie45ReasoningParser, start_token, end_token`；`tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: ernie45_tokenizer, test_reasoning，涉及 `ernie45_tokenizer, test_reasoning`。
- 代码 diff 细节:
  - `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: Ernie45ReasoningParser, start_token, end_token, __init__
  - `tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: ernie45_tokenizer, test_reasoning
- 关键代码摘录:

```diff
diff -- vllm/reasoning/ernie45_reasoning_parser.py
@@ -0,0 +1,169 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.logger import init_logger
diff -- tests/reasoning/test_ernie45_reasoning_parser.py
@@ -0,0 +1,124 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from tests.reasoning.utils import run_reasoning_extraction
+from vllm.reasoning import ReasoningParser, ReasoningParserManager
```

- 已读文件:
  - runtime: `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0
  - tests: `tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_ernie45_reasoning_parser.py`, `tests/tool_use/test_ernie45_moe_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26684 - [Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code

- 链接: https://github.com/vllm-project/vllm/pull/26684
- 状态/时间: merged / 2025-10-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/ernie45_moe.py`；关联提交 `01ad27faff35`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-12，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_moe.py`；技术摘要: 覆盖「[Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code」；主要实现面是 `vllm/model_executor/models/ernie45_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_moe.py` modified +22/-12 (34 lines); hunks: -23,7 +23,8; -139,10 +140,10 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_moe.py` modified +22/-12 (34 lines); hunks: -23,7 +23,8; -139,10 +140,10 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -23,7 +23,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -139,10 +140,10 @@ def __init__(
-        parallel_config = vllm_config.parallel_config
+        eplb_config = vllm_config.parallel_config.eplb_config
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +22/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26885 - [Model][Bugfix] fix ernie45 vl run failed from shared experts optimization

- 链接: https://github.com/vllm-project/vllm/pull/26885
- 状态/时间: merged / 2025-10-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/ernie45_vl_moe.py`；关联提交 `e51928793e10`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-5，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Bugfix] fix ernie45 vl run failed from shared experts optimization」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_vl_moe.py`；技术摘要: 覆盖「[Model][Bugfix] fix ernie45 vl run failed from shared experts optimization」；主要实现面是 `vllm/model_executor/models/ernie45_vl_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5 (27 lines); hunks: -341,7 +341,10 @@ def forward(; -353,16 +356,26 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5 (27 lines); hunks: -341,7 +341,10 @@ def forward(; -353,16 +356,26 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl_moe.py
@@ -341,7 +341,10 @@ def forward(
-            final_hidden_states = torch.zeros_like(hidden_states)
+            final_experts_hidden_states = torch.zeros_like(hidden_states)
+            final_shared_ouput = (
+                torch.zeros_like(hidden_states) if self.has_shared_experts else None
+            )
@@ -353,16 +356,26 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_vl_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27316 - [Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple

- 链接: https://github.com/vllm-project/vllm/pull/27316
- 状态/时间: merged / 2025-10-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/ernie45_moe.py`；关联提交 `63b22e0dbb90`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/ernie45_moe.py`；技术摘要: 覆盖「[Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple」；主要实现面是 `vllm/model_executor/models/ernie45_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_moe.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_moe.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        else:
+            final_hidden_states = final_hidden_states[1]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/ernie45_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27973 - [Model] fix ernie45 reasoning_parser

- 链接: https://github.com/vllm-project/vllm/pull/27973
- 状态/时间: merged / 2025-11-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/reasoning/ernie45_reasoning_parser.py`；关联提交 `43a6acfb7de8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] fix ernie45 reasoning_parser」；模型线: ERNIE 4.5；类别: 缺陷修复；主要 diff: `vllm/reasoning/ernie45_reasoning_parser.py`；技术摘要: 覆盖「[Model] fix ernie45 reasoning_parser」；主要实现面是 `vllm/reasoning/ernie45_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2 (4 lines); hunks: -36,8 +36,8 @@ def end_token(self) -> str:; symbols: end_token, __init__，涉及 `end_token, __init__`。
- 代码 diff 细节:
  - `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2 (4 lines); hunks: -36,8 +36,8 @@ def end_token(self) -> str:; symbols: end_token, __init__
- 关键代码摘录:

```diff
diff -- vllm/reasoning/ernie45_reasoning_parser.py
@@ -36,8 +36,8 @@ def end_token(self) -> str:
-    def __init__(self, tokenizer: PreTrainedTokenizerBase):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
+        super().__init__(tokenizer, *args, **kwargs)
```

- 已读文件:
  - runtime: `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/reasoning/ernie45_reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31274 - [Model][Ernie4.5-VL] Support video metadata for timestamp rendering

- 链接: https://github.com/vllm-project/vllm/pull/31274
- 状态/时间: merged / 2025-12-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+82/-5，可读 patch 137 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Ernie4.5-VL] Support video metadata for timestamp rendering」；模型线: ERNIE 4.5；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`；技术摘要: 覆盖「[Model][Ernie4.5-VL] Support video metadata for timestamp rendering」；主要实现面是 `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunks: -21,7 +21,7; -41,7 +41,7; symbols: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm，涉及 `get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser`；`tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunks: -104,7 +104,8 @@ def create_metadata(frames: np.ndarray):; symbols: create_metadata，涉及 `create_metadata`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunks: -21,7 +21,7; -41,7 +41,7; symbols: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm
  - `tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunks: -104,7 +104,8 @@ def create_metadata(frames: np.ndarray):; symbols: create_metadata
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl.py
@@ -21,7 +21,7 @@
-"""Inference-only Erine VL model compatible with HuggingFace weights."""
+"""Inference-only Ernie VL model compatible with HuggingFace weights."""
@@ -41,7 +41,7 @@
-from vllm.config.multimodal import BaseDummyOptions
+from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
@@ -64,7 +64,7 @@
diff -- tests/models/multimodal/processing/test_common.py
@@ -104,7 +104,8 @@ def create_metadata(frames: np.ndarray):
-    # GLM4.1V and Qwen3-VL requires video metadata to be included in the input
+    # Ernie4.5-VL, GLM4.1V and Qwen3-VL requires video metadata
+    "ernie4_5_moe_vl": qwen3_vl_patch_mm_data,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +80/-4
  - tests: `tests/models/multimodal/processing/test_common.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39753 - [Model] Use mm_features for Ernie-4.5 VL M-RoPE

- 链接: https://github.com/vllm-project/vllm/pull/39753
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/model_executor/test_ernie45_vl_mrope.py`, `vllm/model_executor/models/ernie45_vl.py`；关联提交 `0008729abfbd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+196/-123，可读 patch 339 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use mm_features for Ernie-4.5 VL M-RoPE」；模型线: ERNIE 4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`；技术摘要: 覆盖「[Model] Use mm_features for Ernie-4.5 VL M-RoPE」；主要实现面是 `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/ernie45_vl.py` modified +53/-123 (176 lines); hunks: -23,9 +23,8; -1401,131 +1400,62 @@ def get_mrope_input_positions(; symbols: get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input，涉及 `get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input`；`tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: _force_cpu_default_device, DummyConfig, make_model, make_mm_feature，涉及 `_force_cpu_default_device, DummyConfig, make_model`。
- 代码 diff 细节:
  - `vllm/model_executor/models/ernie45_vl.py` modified +53/-123 (176 lines); hunks: -23,9 +23,8; -1401,131 +1400,62 @@ def get_mrope_input_positions(; symbols: get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input
  - `tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: _force_cpu_default_device, DummyConfig, make_model, make_mm_feature
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/ernie45_vl.py
@@ -23,9 +23,8 @@
-import itertools
-from collections.abc import Callable, Iterable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
@@ -1401,131 +1400,62 @@ def get_mrope_input_positions(
-        kwargs = MultiModalFeatureSpec.gather_kwargs(
-            mm_features,
diff -- tests/model_executor/test_ernie45_vl_mrope.py
@@ -0,0 +1,143 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from dataclasses import dataclass
+import pytest
+import torch
+from vllm.model_executor.models.ernie45_vl import (
```

- 已读文件:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +53/-123
  - tests: `tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0
- 验证与风险: diff 自带测试面 `tests/model_executor/test_ernie45_vl_mrope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
