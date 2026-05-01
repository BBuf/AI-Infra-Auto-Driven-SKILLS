# vllm MiMo V2 Flash 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `vllm/model_executor/models/mimo.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433) |
| `vllm/model_executor/models/mimo_audio.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/model_executor/models/mimo_mtp.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433), [#25136](https://github.com/vllm-project/vllm/pull/25136) |
| `vllm/model_executor/models/mimo_v2.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#41029](https://github.com/vllm-project/vllm/pull/41029) |
| `vllm/model_executor/models/mimo_v2_mtp.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/model_executor/models/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/transformers_utils/configs/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/transformers_utils/processors/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-12 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | `vllm/model_executor/models/mimo_mtp.py` |
| 2025-12-19 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |
| 2026-01-05 | [#31175](https://github.com/vllm-project/vllm/pull/31175) | merged | [Bugfix] Properly apply v_scale for mimo_v2_flash | `vllm/model_executor/models/mimo_v2_flash.py` |
| 2026-04-24 | [#40045](https://github.com/vllm-project/vllm/pull/40045) | merged | [Attention] use diff kv backend for mimo v2 flash | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py` |
| 2026-04-27 | [#40967](https://github.com/vllm-project/vllm/pull/40967) | merged | [Model] Add MiMo-V2.5 support | `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py` |
| 2026-04-28 | [#41029](https://github.com/vllm-project/vllm/pull/41029) | merged | [Model] update for mimo v25 | `vllm/model_executor/models/mimo_v2.py` |

## 逐 PR diff 审计卡

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- 链接: https://github.com/vllm-project/vllm/pull/17433
- 状态/时间: merged / 2025-05-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo.py`, `vllm/model_executor/models/mimo_mtp.py`；关联提交 `acee8f48aa9c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+507/-4，可读 patch 576 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support MiMo-7B inference with MTP」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`；技术摘要: 覆盖「[Model] Support MiMo-7B inference with MTP」；主要实现面是 `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[spec decode] Fix MTP inference path for MiMo-7B model」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mimo_mtp.py`；技术摘要: 覆盖「[spec decode] Fix MTP inference path for MiMo-7B model」；主要实现面是 `vllm/model_executor/models/mimo_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+789/-13，可读 patch 946 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add MiMo-V2-Flash support」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`；技术摘要: 覆盖「[Model] Add MiMo-V2-Flash support」；主要实现面是 `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE，涉及 `MiMoV2MLP, __init__, forward`；`vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunks: -277,6 +277,7 @@ def __init__(; -475,6 +476,7 @@ def __init__(; symbols: __init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader，涉及 `__init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader`；`vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunks: -1252,6 +1252,14 @@ def validate_fp8_block_shape(; symbols: validate_fp8_block_shape，涉及 `validate_fp8_block_shape`；`tests/models/registry.py` modified +3/-0 (3 lines); hunks: -459,6 +459,9 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: -0,0 +1,720; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE
  - `vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunks: -277,6 +277,7 @@ def __init__(; -475,6 +476,7 @@ def __init__(; symbols: __init__, _maybe_allow_fp8_block_shape_mismatch, weight_loader
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunks: -1252,6 +1252,14 @@ def validate_fp8_block_shape(; symbols: validate_fp8_block_shape
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -459,6 +459,9 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: -415,6 +415,7 @@ th {
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
diff -- vllm/model_executor/layers/linear.py
@@ -277,6 +277,7 @@ def __init__(
+        self.allow_fp8_block_shape_mismatch = False
@@ -475,6 +476,7 @@ def __init__(
+        self._maybe_allow_fp8_block_shape_mismatch()
@@ -509,6 +511,33 @@ def __init__(
+    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
+        quant_config = getattr(self, "quant_config", None)
diff -- vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -1252,6 +1252,14 @@ def validate_fp8_block_shape(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0; `vllm/model_executor/layers/linear.py` modified +49/-13; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/model.py` modified +5/-0; `vllm/config/__init__.py` modified +2/-0
  - tests: `tests/models/registry.py` modified +3/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31175 - [Bugfix] Properly apply v_scale for mimo_v2_flash

- 链接: https://github.com/vllm-project/vllm/pull/31175
- 状态/时间: merged / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-13，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Properly apply v_scale for mimo_v2_flash」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mimo_v2_flash.py`；技术摘要: 覆盖「[Bugfix] Properly apply v_scale for mimo_v2_flash」；主要实现面是 `vllm/model_executor/models/mimo_v2_flash.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+112/-24，可读 patch 270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] use diff kv backend for mimo v2 flash」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py`；技术摘要: 覆盖「[Attention] use diff kv backend for mimo v2 flash」；主要实现面是 `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/attention/attention.py`, `tools/pre_commit/generate_attention_backend_docs.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/layers/attention/attention.py` modified +1/-0 (1 lines); hunks: -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec，涉及 `get_kv_cache_spec`；`tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8 (49 lines); hunks: -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; symbols: parse_flash_attn_features，涉及 `parse_flash_attn_features`；`vllm/v1/attention/backends/fa_utils.py` modified +22/-3 (25 lines); hunks: -54,7 +54,10 @@ def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None...; -112,6 +115,23 @@ def get_flash_attn_version(; symbols: get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input, flash_attn_supports_sinks，涉及 `get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8 (22 lines); hunks: -46,6 +46,9; -287,6 +290,15 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/attention/attention.py` modified +1/-0 (1 lines); hunks: -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCa...; symbols: get_kv_cache_spec
  - `tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8 (49 lines); hunks: -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:; symbols: parse_flash_attn_features
  - `vllm/v1/attention/backends/fa_utils.py` modified +22/-3 (25 lines); hunks: -54,7 +54,10 @@ def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None...; -112,6 +115,23 @@ def get_flash_attn_version(; symbols: get_scheduler_metadata, get_flash_attn_version, flash_attn_supports_quant_query_input, flash_attn_supports_sinks
  - `vllm/v1/attention/backends/flash_attn_diffkv.py` modified +18/-4 (22 lines); hunks: -6,14 +6,16; -23,8 +25,6; symbols: FlashAttentionDiffKVBackend, get_kv_cache_stride_order, FlashAttentionDiffKVImpl, __init__
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
diff -- vllm/model_executor/layers/attention/attention.py
@@ -597,6 +597,7 @@ def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
+                head_size_v=self.head_size_v,
diff -- tools/pre_commit/generate_attention_backend_docs.py
@@ -634,9 +634,10 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:
-    # Analyze the functions to determine FA3-specific features
+    # Analyze the functions to determine FA3/FA4-specific features
+    fa4_supports_sinks = False
@@ -656,17 +657,49 @@ def parse_flash_attn_features() -> dict[str, dict[str, Any]]:
-        # Check flash_attn_supports_sinks - looks for `get_flash_attn_version() == 3`
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_flash.py` modified +14/-8; `vllm/model_executor/layers/attention/attention.py` modified +1/-0; `vllm/v1/attention/backends/fa_utils.py` modified +22/-3; `vllm/v1/attention/backends/flash_attn_diffkv.py` modified +18/-4; `vllm/v1/kv_cache_interface.py` modified +14/-0; `vllm/vllm_flash_attn/flash_attn_interface.py` modified +1/-0
  - other: `tools/pre_commit/generate_attention_backend_docs.py` modified +41/-8
  - docs: `docs/design/attention_backends.md` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/v1/attention/backends/fa_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40967 - [Model] Add MiMo-V2.5 support

- 链接: https://github.com/vllm-project/vllm/pull/40967
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_audio.py`, `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`, `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/transformers_utils/configs/mimo_v2_omni.py` 等 6 个文件；关联提交 `c245d35ff467`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+4737/-5，可读 patch 4920 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add MiMo-V2.5 support」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__，涉及 `MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger`；`vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init，涉及 `_vq_default, _ema_inplace, _laplace_smoothing`；`vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput，涉及 `ImageInput, VideoInput, AudioInput`；`vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers，涉及 `MiMoV2MTPLayer, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0 (1488 lines); hunks: -0,0 +1,1488; symbols: MiMoVisionMLP, MiMoVisionPatchEmbed, MiMoVisionPatchMerger, __init__
  - `vllm/model_executor/models/mimo_audio.py` added +1389/-0 (1389 lines); hunks: -0,0 +1,1389; symbols: _vq_default, _ema_inplace, _laplace_smoothing, _uniform_init
  - `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0 (1285 lines); hunks: -0,0 +1,1285; symbols: ImageInput, VideoInput, AudioInput, VideoAudioInput
  - `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0 (373 lines); hunks: -0,0 +1,373; symbols: MiMoV2MTPLayer, __init__, forward, _MiMoV2MTPLayers
  - `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0 (65 lines); hunks: -0,0 +1,65; symbols: Mimo_VLVisionConfig, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -0,0 +1,1488 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import math
+from collections.abc import Callable, Iterable, Mapping, Sequence
+from functools import partial
+from typing import Any
diff -- vllm/model_executor/models/mimo_audio.py
@@ -0,0 +1,1389 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MiMo audio: tokenizer, encoding utilities, and audio encoder.
+Ported from SGLang's mimo_audio.py.
+Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
+"""
diff -- vllm/transformers_utils/processors/mimo_v2_omni.py
@@ -0,0 +1,1285 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_omni.py` added +1488/-0; `vllm/model_executor/models/mimo_audio.py` added +1389/-0; `vllm/transformers_utils/processors/mimo_v2_omni.py` added +1285/-0; `vllm/model_executor/models/mimo_v2_mtp.py` added +373/-0; `vllm/transformers_utils/configs/mimo_v2_omni.py` added +65/-0; `vllm/model_executor/models/mimo_v2.py` renamed +22/-2
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41029 - [Model] update for mimo v25

- 链接: https://github.com/vllm-project/vllm/pull/41029
- 状态/时间: merged / 2026-04-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`；关联提交 `7a1eb8ac2ec4`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+10/-8，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] update for mimo v25」；模型线: MiMo V2 Flash；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mimo_v2.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/mimo_v2.py` modified +1/-1 (2 lines); hunks: -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM，涉及 `load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +1/-1 (2 lines); hunks: -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, MiMoV2ProForCausalLM, MiMoV2ForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -733,7 +733,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-class MiMoV2ProForCausalLM(MiMoV2FlashForCausalLM):
+class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
