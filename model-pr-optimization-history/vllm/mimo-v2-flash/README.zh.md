# vLLM MiMo V2 Flash 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/test_mimo_v2_omni.py` | [#49815](https://github.com/vllm-project/vllm/pull/49815), [#58900](https://github.com/vllm-project/vllm/pull/58900) |
| `tests/models/quantization/test_mimo_v2_qkv_shard.py` | [#57508](https://github.com/vllm-project/vllm/pull/57508) |
| `tests/models/quantization/test_mimo_v2_w4a16_routing.py` | [#58262](https://github.com/vllm-project/vllm/pull/58262) |
| `tests/v1/e2e/spec_decode/mtp/mimo/__init__.py` | 无直接 PR 号提交 |
| `tests/v1/e2e/spec_decode/mtp/mimo/test_mtp.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/mimo.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433) |
| `vllm/model_executor/models/mimo_audio.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/model_executor/models/mimo_mtp.py` | [#17433](https://github.com/vllm-project/vllm/pull/17433), [#25136](https://github.com/vllm-project/vllm/pull/25136) |
| `vllm/model_executor/models/mimo_v2.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#41029](https://github.com/vllm-project/vllm/pull/41029), [#41797](https://github.com/vllm-project/vllm/pull/41797), [#45200](https://github.com/vllm-project/vllm/pull/45200), [#46104](https://github.com/vllm-project/vllm/pull/46104), [#57508](https://github.com/vllm-project/vllm/pull/57508), [#57784](https://github.com/vllm-project/vllm/pull/57784), [#58142](https://github.com/vllm-project/vllm/pull/58142) |
| `vllm/model_executor/models/mimo_v2_mtp.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#41905](https://github.com/vllm-project/vllm/pull/41905), [#57508](https://github.com/vllm-project/vllm/pull/57508), [#58142](https://github.com/vllm-project/vllm/pull/58142) |
| `vllm/model_executor/models/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#49815](https://github.com/vllm-project/vllm/pull/49815), [#57784](https://github.com/vllm-project/vllm/pull/57784), [#58938](https://github.com/vllm-project/vllm/pull/58938) |
| `vllm/parser/mimo.py` | [#58019](https://github.com/vllm-project/vllm/pull/58019) |
| `vllm/reasoning/mimo_engine_reasoning_parser.py` | [#58019](https://github.com/vllm-project/vllm/pull/58019) |
| `vllm/tool_parsers/mimo_tool_parser.py` | [#58019](https://github.com/vllm-project/vllm/pull/58019) |
| `vllm/transformers_utils/configs/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967) |
| `vllm/transformers_utils/processors/mimo_v2_omni.py` | [#40967](https://github.com/vllm-project/vllm/pull/40967), [#43117](https://github.com/vllm-project/vllm/pull/43117) |

## PR 覆盖总览

- git 追溯 PR 数: 17
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 21
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
| 2026-05-09 | [#41905](https://github.com/vllm-project/vllm/pull/41905) | merged | [SpecDecoding] extend mtp support for mimo 2.5 | `vllm/model_executor/models/mimo_v2_mtp.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-11 | [#41797](https://github.com/vllm-project/vllm/pull/41797) | merged | [Attention] add triton diff-kv backend for mimo | `vllm/model_executor/models/mimo_v2.py` |
| 2026-06-15 | [#45200](https://github.com/vllm-project/vllm/pull/45200) | merged | [Models] Fix MiMo v2.x QKV TP sharding + FP4 support | `vllm/model_executor/models/mimo_v2.py` |
| 2026-07-01 | [#46104](https://github.com/vllm-project/vllm/pull/46104) | merged | [Spec Decode] Support SWA + DFlash for MiMo | `vllm/model_executor/models/mimo_v2.py` |
| 2026-07-11 | [#43117](https://github.com/vllm-project/vllm/pull/43117) | merged | fix(processor): route MiMo-V2-Omni media fetch through MediaConnector | `vllm/transformers_utils/processors/mimo_v2_omni.py` |
| 2026-08-10 | [#49815](https://github.com/vllm-project/vllm/pull/49815) | merged | [Bugfix][MiMo] Apply vision attention sinks in the window attention path | `tests/models/multimodal/test_mimo_v2_omni.py`, `vllm/model_executor/models/mimo_v2_omni.py` |
| 2026-09-19 | [#57508](https://github.com/vllm-project/vllm/pull/57508) | merged | [Bugfix][Model] Fix MiMo-V2.5 fused fp8 qkv_proj sharding (pre-shard count is num_key_value_heads; MTP path too) | `tests/models/quantization/test_mimo_v2_qkv_shard.py`, `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py` |
| 2026-09-20 | [#57784](https://github.com/vllm-project/vllm/pull/57784) | merged | [Feature] support bf16 MoE router and mxfp4 MoE for MiMo V2 | `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_omni.py` |
| 2026-09-28 | [#58900](https://github.com/vllm-project/vllm/pull/58900) | merged | [Bugfix] Fix the two multimodal root tests that fail on main (OpenPangu-VL embed merge, MiMo sink test fixture) | `tests/models/multimodal/test_mimo_v2_omni.py`, `vllm/model_executor/models/openpangu_vl.py` |
| 2026-09-29 | [#58142](https://github.com/vllm-project/vllm/pull/58142) | merged | [Bugfix][Model] MiMo: keep fused fp8 qkv_proj pairing state across weight-loading calls | `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py` |
| 2026-09-29 | [#58019](https://github.com/vllm-project/vllm/pull/58019) | merged | [Frontend] Support strict MiMo-V2.6 tool calling | `vllm/tool_parsers/mimo_tool_parser.py`, `vllm/reasoning/mimo_engine_reasoning_parser.py`, `vllm/parser/mimo.py` |
| 2026-09-29 | [#58938](https://github.com/vllm-project/vllm/pull/58938) | merged | [Bugfix][MiMo] Declare embedding_fields so an EPD pair can serve images | `vllm/model_executor/models/mimo_v2_omni.py` |
| 2026-09-30 | [#58262](https://github.com/vllm-project/vllm/pull/58262) | merged | [ROCm][MoE] Support MiMo-V2.6 MXFP4 on gfx942 | `tests/models/quantization/test_mimo_v2_w4a16_routing.py`, `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py` |

## 逐 PR diff 审计卡

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- 链接: https://github.com/vllm-project/vllm/pull/17433
- 状态/时间: merged / 2025-05-12
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/17433 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/25136 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/30836 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/31175 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/40045 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/40967 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_audio.py`, `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`, `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/transformers_utils/configs/mimo_v2_omni.py` 等 6 个文件；关联提交 `c245d35ff467`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+4737/-5，可读 patch 4920 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add MiMo-V2.5 support」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`；技术摘要: 覆盖「[Model] Add MiMo-V2.5 support」；主要实现面是 `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/mimo_audio.py`, `vllm/transformers_utils/processors/mimo_v2_omni.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/41029 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`；关联提交 `7a1eb8ac2ec4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+10/-8，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] update for mimo v25」；模型线: MiMo V2 Flash；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/mimo_v2.py`；技术摘要: 覆盖「[Model] update for mimo v25」；主要实现面是 `vllm/model_executor/models/mimo_v2.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #41905 - [SpecDecoding] extend mtp support for mimo 2.5

- 链接: https://github.com/vllm-project/vllm/pull/41905
- 状态/时间: merged / 2026-05-09
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/41905 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2_mtp.py`；关联提交 `2ee8c2a56e41`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-10，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SpecDecoding] extend mtp support for mimo 2.5」；模型线: MiMo V2 Flash；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/mimo_v2_mtp.py`；技术摘要: 覆盖「[SpecDecoding] extend mtp support for mimo 2.5」；主要实现面是 `vllm/model_executor/models/mimo_v2_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2_mtp.py` modified +3/-10 (13 lines); hunks: -49,7 +49,7; -170,10 +170,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, compute_logits，涉及 `__init__, forward, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_mtp.py` modified +3/-10 (13 lines); hunks: -49,7 +49,7; -170,10 +170,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, forward, compute_logits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_mtp.py
@@ -49,7 +49,7 @@
-# only the first layer and only one speculative token.
+# only the first layer
@@ -170,10 +170,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
-        if spec_cfg.num_speculative_tokens != 1:
-            raise ValueError(
-                "MiMo-V2 MTP in vLLM only supports num_speculative_tokens=1."
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2_mtp.py` modified +3/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mimo_v2_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- 链接: https://github.com/vllm-project/vllm/pull/43167
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+88/-731，可读 patch 1251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove KV cache scale boilerplate from model weight loading methods」；模型线: MiMo V2 Flash；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`；技术摘要: 覆盖「Remove KV cache scale boilerplate from model weight loading methods」；主要实现面是 `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name，涉及 `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`；`vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader，涉及 `_get_moe_weight_dtype, kv_cache_scale_loader`；`vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod，涉及 `KVCacheScaleParameter, __new__, weight_loader`；`vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter，涉及 `get_quant_method, get_cache_scale, get_cache_scale_mapper`。
- 代码 diff 细节:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- 验证与风险: diff 自带测试面 `tests/model_executor/test_eagle_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41797 - [Attention] add triton diff-kv backend for mimo

- 链接: https://github.com/vllm-project/vllm/pull/41797
- 状态/时间: merged / 2026-06-11
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/41797 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`；关联提交 `f81daf888063`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+1041/-9，可读 patch 1103 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] add triton diff-kv backend for mimo」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mimo_v2.py`；技术摘要: 覆盖「[Attention] add triton diff-kv backend for mimo」；主要实现面是 `vllm/model_executor/models/mimo_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2.py` modified +21/-7 (28 lines); hunks: -47,9 +47,7; -292,11 +290,27 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +21/-7 (28 lines); hunks: -47,9 +47,7; -292,11 +290,27 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -47,9 +47,7 @@
-from vllm.v1.attention.backends.flash_attn_diffkv import (
-    FlashAttentionDiffKVBackend,
-)
+from vllm.v1.attention.backends.registry import AttentionBackendEnum
@@ -292,11 +290,27 @@ def __init__(
-        # Use DiffKV backend when V has a different head dim than K
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +21/-7
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_triton_unified_attention_diffkv.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45200 - [Models] Fix MiMo v2.x QKV TP sharding + FP4 support

- 链接: https://github.com/vllm-project/vllm/pull/45200
- 状态/时间: merged / 2026-06-15
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/45200 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`；关联提交 `b5adb027ad03`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+170/-5，可读 patch 245 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Fix MiMo v2.x QKV TP sharding + FP4 support」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/mimo_v2.py`；技术摘要: 覆盖「[Models] Fix MiMo v2.x QKV TP sharding + FP4 support」；主要实现面是 `vllm/model_executor/models/mimo_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2.py` modified +160/-5 (165 lines); hunks: -35,6 +35,10; -455,6 +459,85 @@ def is_compressed_softmax_layer(self) -> bool:; symbols: is_compressed_softmax_layer, _shard_fp8_qkv_proj, MiMoV2Model, __init__，涉及 `is_compressed_softmax_layer, _shard_fp8_qkv_proj, MiMoV2Model`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +160/-5 (165 lines); hunks: -35,6 +35,10; -455,6 +459,85 @@ def is_compressed_softmax_layer(self) -> bool:; symbols: is_compressed_softmax_layer, _shard_fp8_qkv_proj, MiMoV2Model, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -35,6 +35,10 @@
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    GroupShape,
+    scaled_quantize,
+)
@@ -455,6 +459,85 @@ def is_compressed_softmax_layer(self) -> bool:
+def _shard_fp8_qkv_proj(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +160/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/models/mimo_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #46104 - [Spec Decode] Support SWA + DFlash for MiMo

- 链接: https://github.com/vllm-project/vllm/pull/46104
- 状态/时间: merged / 2026-07-01
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/46104 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`；关联提交 `9969466a5978`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+243/-25，可读 patch 500 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Spec Decode] Support SWA + DFlash for MiMo」；模型线: MiMo V2 Flash；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/mimo_v2.py`；技术摘要: 覆盖「[Spec Decode] Support SWA + DFlash for MiMo」；主要实现面是 `vllm/model_executor/models/mimo_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/mimo_v2.py` modified +16/-3 (19 lines); hunks: -53,7 +53,12; -539,7 +544,7 @@ def _shard_fp8_qkv_proj(; symbols: _shard_fp8_qkv_proj, MiMoV2Model, __init__, forward，涉及 `_shard_fp8_qkv_proj, MiMoV2Model, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +16/-3 (19 lines); hunks: -53,7 +53,12; -539,7 +544,7 @@ def _shard_fp8_qkv_proj(; symbols: _shard_fp8_qkv_proj, MiMoV2Model, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -53,7 +53,12 @@
-from .interfaces import MixtureOfExperts, SupportsPP
+from .interfaces import (
+    EagleModelMixin,
+    MixtureOfExperts,
+    SupportsEagle3,
+    SupportsPP,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +16/-3
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43117 - fix(processor): route MiMo-V2-Omni media fetch through MediaConnector

- 链接: https://github.com/vllm-project/vllm/pull/43117
- 状态/时间: merged / 2026-07-11
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/43117 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/processors/mimo_v2_omni.py`；关联提交 `54503ecec0f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-76，可读 patch 190 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix(processor): route MiMo-V2-Omni media fetch through MediaConnector」；模型线: MiMo V2 Flash；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/processors/mimo_v2_omni.py`；技术摘要: 覆盖「fix(processor): route MiMo-V2-Omni media fetch through MediaConnector」；主要实现面是 `vllm/transformers_utils/processors/mimo_v2_omni.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/mimo_v2_omni.py` modified +22/-76 (98 lines); hunks: -7,33 +7,21; -62,7 +50,7; symbols: ImageInput, VideoInput, AudioInput, _smart_resize，涉及 `ImageInput, VideoInput, AudioInput`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/mimo_v2_omni.py` modified +22/-76 (98 lines); hunks: -7,33 +7,21; -62,7 +50,7; symbols: ImageInput, VideoInput, AudioInput, _smart_resize
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/mimo_v2_omni.py
@@ -7,33 +7,21 @@
-import copy
-import io
-from io import BytesIO
-import requests
-try:
-    from torchcodec.decoders import AudioDecoder
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/mimo_v2_omni.py` modified +22/-76
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/processors/mimo_v2_omni.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #49815 - [Bugfix][MiMo] Apply vision attention sinks in the window attention path

- 链接: https://github.com/vllm-project/vllm/pull/49815
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/test_mimo_v2_omni.py`, `vllm/model_executor/models/mimo_v2_omni.py`；关联提交 `c3cac8c63d91`；保留自原 history/skill 显式引用
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+132/-12，可读 patch 200 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/multimodal/test_mimo_v2_omni.py` added +91/-0 (91 lines); hunks: -0,0 +1,91; symbols: vision_attn_env, _reference, test_window_attention_applies_sinks，涉及 `vision_attn_env, _reference, test_window_attention_applies_sinks`；`vllm/model_executor/models/mimo_v2_omni.py` modified +27/-10 (37 lines); hunks: -193,7 +193,7 @@ def __init__(; -214,21 +214,38 @@ def _forward_window_attn(; symbols: __init__, _forward_window_attn，涉及 `__init__, _forward_window_attn`。
- 代码 diff 细节:
  - `tests/models/multimodal/test_mimo_v2_omni.py` added +91/-0 (91 lines); hunks: -0,0 +1,91; symbols: vision_attn_env, _reference, test_window_attention_applies_sinks
  - `vllm/model_executor/models/mimo_v2_omni.py` modified +27/-10 (37 lines); hunks: -193,7 +193,7 @@ def __init__(; -214,21 +214,38 @@ def _forward_window_attn(; symbols: __init__, _forward_window_attn
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/test_mimo_v2_omni.py
@@ -0,0 +1,91 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MiMo-V2 vision window attention has to apply the per-head sink logits."""
+import pytest
+import torch
+from tests.utils import ensure_current_vllm_config
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -193,7 +193,7 @@ def __init__(
-        # Sink attention weights (loaded but not used in vLLM flash_attn)
+        # Per-head sink logits, applied in the window attention path.
@@ -214,21 +214,38 @@ def _forward_window_attn(
-        """Window attention via flash_attn_varlen_func with window_size."""
-        from vllm.vllm_flash_attn import flash_attn_varlen_func
+        """Window attention with the per-head sink applied to key 0.
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/multimodal/test_mimo_v2_omni.py` added +91/-0
  - runtime: `vllm/model_executor/models/mimo_v2_omni.py` modified +27/-10
- 验证与风险: diff 自带测试面 `tests/models/multimodal/test_mimo_v2_omni.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57508 - [Bugfix][Model] Fix MiMo-V2.5 fused fp8 qkv_proj sharding (pre-shard count is num_key_value_heads; MTP path too)

- 链接: https://github.com/vllm-project/vllm/pull/57508
- 状态/时间: merged / 2026-09-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/quantization/test_mimo_v2_qkv_shard.py`, `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`；关联提交 `211e252d0b4f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+415/-63，可读 patch 543 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/quantization/test_mimo_v2_qkv_shard.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: _chunk_rows, _quantize_chunks, _owned_rows, _dequantize，涉及 `_chunk_rows, _quantize_chunks, _owned_rows`；`vllm/model_executor/models/mimo_v2.py` modified +141/-55 (196 lines); hunks: -50,6 +50,7; -460,6 +461,28 @@ def is_compressed_softmax_layer(self) -> bool:; symbols: is_compressed_softmax_layer, _requantize_fp8, _shard_fp8_qkv_proj, MiMoV2Model，涉及 `is_compressed_softmax_layer, _requantize_fp8, _shard_fp8_qkv_proj`；`vllm/model_executor/models/mimo_v2_mtp.py` modified +49/-8 (57 lines); hunks: -45,7 +45,7; -266,6 +266,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `tests/models/quantization/test_mimo_v2_qkv_shard.py` added +225/-0 (225 lines); hunks: -0,0 +1,225; symbols: _chunk_rows, _quantize_chunks, _owned_rows, _dequantize
  - `vllm/model_executor/models/mimo_v2.py` modified +141/-55 (196 lines); hunks: -50,6 +50,7; -460,6 +461,28 @@ def is_compressed_softmax_layer(self) -> bool:; symbols: is_compressed_softmax_layer, _requantize_fp8, _shard_fp8_qkv_proj, MiMoV2Model
  - `vllm/model_executor/models/mimo_v2_mtp.py` modified +49/-8 (57 lines); hunks: -45,7 +45,7; -266,6 +266,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- tests/models/quantization/test_mimo_v2_qkv_shard.py
@@ -0,0 +1,225 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit test for MiMo-V2 fused fp8 ``qkv_proj`` sharding.
+The checkpoint stores the fused QKV pre-sharded at ``num_key_value_heads``
+chunks (``ckpt_tp``), each holding that slice's ``[Q_c | K_c | V_c]`` rows with
+its own block scale grid. ``ckpt_tp`` is not the layer's KV-head count: a
diff -- vllm/model_executor/models/mimo_v2.py
@@ -50,6 +50,7 @@
+from vllm.utils.math_utils import cdiv
@@ -460,6 +461,28 @@ def is_compressed_softmax_layer(self) -> bool:
+def _requantize_fp8(
+    grouped: torch.Tensor, rows_rank: int, block: int, dtype: torch.dtype
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Block-quantize a rank's ``[Q | K | V]`` rows back to fp8.
diff -- vllm/model_executor/models/mimo_v2_mtp.py
@@ -45,7 +45,7 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/quantization/test_mimo_v2_qkv_shard.py` added +225/-0
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +141/-55; `vllm/model_executor/models/mimo_v2_mtp.py` modified +49/-8
- 验证与风险: diff 自带测试面 `tests/models/quantization/test_mimo_v2_qkv_shard.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57784 - [Feature] support bf16 MoE router and mxfp4 MoE for MiMo V2

- 链接: https://github.com/vllm-project/vllm/pull/57784
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_omni.py`；关联提交 `9b2f34cad446`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+21/-6，可读 patch 109 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/mimo_v2.py` modified +8/-5 (13 lines); hunks: -27,6 +27,7; -156,14 +157,16 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/mimo_v2_omni.py` modified +4/-1 (5 lines); hunks: -52,6 +52,7; -1209,7 +1210,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, MiMoV2OmniForCausalLM，涉及 `get_dummy_mm_data, MiMoV2OmniForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +8/-5 (13 lines); hunks: -27,6 +27,7; -156,14 +157,16 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/mimo_v2_omni.py` modified +4/-1 (5 lines); hunks: -52,6 +52,7; -1209,7 +1210,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, MiMoV2OmniForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -27,6 +27,7 @@
+from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
@@ -156,14 +157,16 @@ def __init__(
-        self.gate = nn.Linear(
+        self.gate = GateLinear(
-            dtype=self.gate_dtype,
+            params_dtype=self.gate_dtype,
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -52,6 +52,7 @@
+    SupportsEagle3,
@@ -1209,7 +1210,9 @@ def get_dummy_mm_data(
-class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):
+class MiMoV2OmniForCausalLM(
+    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, SupportsEagle3
+):
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +8/-5; `vllm/model_executor/models/mimo_v2_omni.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_omni.py`, `vllm/model_executor/models/qwen3_dflash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58900 - [Bugfix] Fix the two multimodal root tests that fail on main (OpenPangu-VL embed merge, MiMo sink test fixture)

- 链接: https://github.com/vllm-project/vllm/pull/58900
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/test_mimo_v2_omni.py`；关联提交 `0da126676e38`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+15/-27，可读 patch 66 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/multimodal/test_mimo_v2_omni.py` modified +3/-18 (21 lines); hunks: -5,13 +5,7; -21,20 +15,11; symbols: vision_attn_env，涉及 `vision_attn_env`；`vllm/model_executor/models/openpangu_vl.py` modified +12/-9 (21 lines); hunks: -1055,15 +1055,18 @@ def get_input_embeddings(; symbols: get_input_embeddings, _process_image_input，涉及 `get_input_embeddings, _process_image_input`。
- 代码 diff 细节:
  - `tests/models/multimodal/test_mimo_v2_omni.py` modified +3/-18 (21 lines); hunks: -5,13 +5,7; -21,20 +15,11; symbols: vision_attn_env
  - `vllm/model_executor/models/openpangu_vl.py` modified +12/-9 (21 lines); hunks: -1055,15 +1055,18 @@ def get_input_embeddings(; symbols: get_input_embeddings, _process_image_input
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/test_mimo_v2_omni.py
@@ -5,13 +5,7 @@
-from tests.utils import ensure_current_vllm_config
-from vllm.distributed.parallel_state import (
-    init_distributed_environment,
-    initialize_model_parallel,
-)
-from vllm.utils.network_utils import get_open_port
diff -- vllm/model_executor/models/openpangu_vl.py
@@ -1055,15 +1055,18 @@ def get_input_embeddings(
-        inputs_embeds = self.language_model.embed_input_ids(input_ids)
-        if multimodal_embeddings is not None:
-            inputs_embeds = self.embed_input_ids(  # type: ignore[call-overload]
-                input_ids,
-                inputs_embeds,
-                multimodal_embeddings,
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/multimodal/test_mimo_v2_omni.py` modified +3/-18
  - runtime: `vllm/model_executor/models/openpangu_vl.py` modified +12/-9
- 验证与风险: diff 自带测试面 `tests/models/multimodal/test_mimo_v2_omni.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58142 - [Bugfix][Model] MiMo: keep fused fp8 qkv_proj pairing state across weight-loading calls

- 链接: https://github.com/vllm-project/vllm/pull/58142
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`；关联提交 `77e52645e9ba`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+12/-3，可读 patch 29 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/mimo_v2.py` modified +6/-2 (8 lines); hunks: -742,8 +742,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`；`vllm/model_executor/models/mimo_v2_mtp.py` modified +6/-1 (7 lines); hunks: -266,7 +266,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2.py` modified +6/-2 (8 lines); hunks: -742,8 +742,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
  - `vllm/model_executor/models/mimo_v2_mtp.py` modified +6/-1 (7 lines); hunks: -266,7 +266,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2.py
@@ -742,8 +742,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        # sharded together.
-        pending_fp8_qkv_proj: dict[str, dict[str, torch.Tensor]] = {}
+        # sharded together. The state must outlive this call: AutoWeightsLoader
+        # delegates per contiguous group of names, so a pair can straddle two
+        # calls and would otherwise be dropped silently.
+        pending_fp8_qkv_proj = getattr(self, "_pending_fp8_qkv_proj", None)
diff -- vllm/model_executor/models/mimo_v2_mtp.py
@@ -266,7 +266,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        pending_qkv_proj: dict[str, dict[str, torch.Tensor]] = {}
+        # The pairing state must outlive this call: AutoWeightsLoader delegates
+        # per contiguous group of names, so a pair can straddle two calls and
+        # would otherwise be dropped silently.
+        pending_qkv_proj = getattr(self, "_pending_qkv_proj", None)
+        if pending_qkv_proj is None:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/mimo_v2.py` modified +6/-2; `vllm/model_executor/models/mimo_v2_mtp.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mimo_v2.py`, `vllm/model_executor/models/mimo_v2_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58019 - [Frontend] Support strict MiMo-V2.6 tool calling

- 链接: https://github.com/vllm-project/vllm/pull/58019
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/parser/mimo.py`, `vllm/reasoning/mimo_engine_reasoning_parser.py`, `vllm/tool_parsers/mimo_tool_parser.py`；关联提交 `dfc8e0f3e2ad`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+243/-15，可读 patch 437 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/tool_parsers/mimo_tool_parser.py` added +25/-0 (25 lines); hunks: -0,0 +1,25; symbols: MiMoToolParser, get_structural_tag，涉及 `MiMoToolParser, get_structural_tag`；`vllm/reasoning/mimo_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6；`vllm/parser/mimo.py` added +42/-0 (42 lines); hunks: -0,0 +1,42; symbols: _mimo_arg_converter, MiMoParser, __init__，涉及 `_mimo_arg_converter, MiMoParser, __init__`。
- 代码 diff 细节:
  - `vllm/tool_parsers/mimo_tool_parser.py` added +25/-0 (25 lines); hunks: -0,0 +1,25; symbols: MiMoToolParser, get_structural_tag
  - `vllm/reasoning/mimo_engine_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `vllm/parser/mimo.py` added +42/-0 (42 lines); hunks: -0,0 +1,42; symbols: _mimo_arg_converter, MiMoParser, __init__
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/mimo_tool_parser.py
@@ -0,0 +1,25 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from xgrammar.structural_tag import TriggeredTagsFormat
+from vllm.parser.engine.registered_adapters import MiMoParserToolAdapter
+from vllm.tool_parsers.tool_strict_level import ToolStrictLevel
+class MiMoToolParser(MiMoParserToolAdapter):  # type: ignore[valid-type, misc]
diff -- vllm/reasoning/mimo_engine_reasoning_parser.py
@@ -0,0 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from vllm.parser.engine.registered_adapters import MiMoParserReasoningAdapter
+__all__ = ["MiMoParserReasoningAdapter"]
diff -- vllm/parser/mimo.py
@@ -0,0 +1,42 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/tool_parsers/mimo_tool_parser.py` added +25/-0; `vllm/reasoning/mimo_engine_reasoning_parser.py` added +6/-0; `vllm/parser/mimo.py` added +42/-0
- 验证与风险: diff 自带测试面 `requirements/test/cpu.txt`, `requirements/test/cuda.txt`, `requirements/test/rocm.txt`, `requirements/test/xpu.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #58938 - [Bugfix][MiMo] Declare embedding_fields so an EPD pair can serve images

- 链接: https://github.com/vllm-project/vllm/pull/58938
- 状态/时间: merged / 2026-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mimo_v2_omni.py`；关联提交 `8aaeef343a10`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+20/-4，可读 patch 45 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/mimo_v2_omni.py` modified +20/-4 (24 lines); hunks: -69,7 +69,10; -670,6 +673,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiMoV2OmniMultiModalDataParser, MiMoV2OmniProcessingInfo, get_supported_mm_limits，涉及 `load_weights, MiMoV2OmniMultiModalDataParser, MiMoV2OmniProcessingInfo`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mimo_v2_omni.py` modified +20/-4 (24 lines); hunks: -69,7 +69,10; -670,6 +673,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, MiMoV2OmniMultiModalDataParser, MiMoV2OmniProcessingInfo, get_supported_mm_limits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mimo_v2_omni.py
@@ -69,7 +69,10 @@
-from .qwen2_vl import _create_qwen2vl_field_factory
+from .qwen2_vl import (
+    Qwen2VLMultiModalDataParser,
+    _create_qwen2vl_field_factory,
+)
@@ -670,6 +673,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/mimo_v2_omni.py` modified +20/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mimo_v2_omni.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #58262 - [ROCm][MoE] Support MiMo-V2.6 MXFP4 on gfx942

- 链接: https://github.com/vllm-project/vllm/pull/58262
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/quantization/test_mimo_v2_w4a16_routing.py`；关联提交 `16d4ac4ba3a0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+192/-5，可读 patch 219 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/models/quantization/test_mimo_v2_w4a16_routing.py` added +176/-0 (176 lines); hunks: -0,0 +1,176; symbols: mimo_routing_method, mimo_moe_config, test_ungrouped_sigmoid_router_is_deepseekv3, test_aiter_w4a16_supports_mimo_routing，涉及 `mimo_routing_method, mimo_moe_config, test_ungrouped_sigmoid_router_is_deepseekv3`；`vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py` modified +16/-5 (21 lines); hunks: -295,6 +295,7 @@ def __init__(; -344,6 +345,7 @@ def _supports_routing_method(; symbols: __init__, _supports_routing_method, apply，涉及 `__init__, _supports_routing_method, apply`。
- 代码 diff 细节:
  - `tests/models/quantization/test_mimo_v2_w4a16_routing.py` added +176/-0 (176 lines); hunks: -0,0 +1,176; symbols: mimo_routing_method, mimo_moe_config, test_ungrouped_sigmoid_router_is_deepseekv3, test_aiter_w4a16_supports_mimo_routing
  - `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py` modified +16/-5 (21 lines); hunks: -295,6 +295,7 @@ def __init__(; -344,6 +345,7 @@ def _supports_routing_method(; symbols: __init__, _supports_routing_method, apply
- 关键代码摘录:

```diff
diff -- tests/models/quantization/test_mimo_v2_w4a16_routing.py
@@ -0,0 +1,176 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Routing support for the AITER Triton MXFP4 W4A16 MoE backend.
+MiMo-V2.6 uses an ungrouped sigmoid router with a per-expert correction bias
+(``scoring_func=sigmoid``, ``topk_method=noaux_tc``, ``n_group == 1``), which
+``get_routing_method_type`` classifies as ``RoutingMethodType.DeepSeekV3``.
diff -- vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py
@@ -295,6 +295,7 @@ def __init__(
+            RoutingMethodType.DeepSeekV3,
@@ -344,6 +345,7 @@ def _supports_routing_method(
+            RoutingMethodType.DeepSeekV3,
@@ -373,11 +375,20 @@ def apply(
-        score_mode = (
-            "sqrtsoftplus"
```

- 提取文件（未人工审阅）:
  - tests: `tests/models/quantization/test_mimo_v2_w4a16_routing.py` added +176/-0
  - runtime: `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py` modified +16/-5
- 验证与风险: diff 自带测试面 `tests/models/quantization/test_mimo_v2_w4a16_routing.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
