# vLLM Ling 3.0 (BailingMoeV3) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/processing/test_bailing_moe_v3_vl.py` | 无直接 PR 号提交 |
| `tests/parser/engine/test_ling3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `tests/transformers_utils/test_bailing_moe_v3_vl_config.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/bailing_moe_v3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045), [#51265](https://github.com/vllm-project/vllm/pull/51265), [#53593](https://github.com/vllm-project/vllm/pull/53593) |
| `vllm/model_executor/models/bailing_moe_v3_mtp.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045), [#51265](https://github.com/vllm-project/vllm/pull/51265) |
| `vllm/model_executor/models/bailing_moe_v3_vl.py` | 无直接 PR 号提交 |
| `vllm/parser/ling3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/reasoning/ling3_reasoning_parser.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/tool_parsers/ling3_tool_parser.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/transformers_utils/configs/bailing_moe_v3_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-05 | [#51045](https://github.com/vllm-project/vllm/pull/51045) | merged | [Model][Frontend] Add Ling 3.0 Flash BF16, MTP, and parser support | `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`, `vllm/tool_parsers/ling3_tool_parser.py` |
| 2026-08-10 | [#51265](https://github.com/vllm-project/vllm/pull/51265) | merged | `[Model][Quantization] Add Ling-3.0-flash-fp8 support` | `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py` |
| 2026-08-25 | [#53593](https://github.com/vllm-project/vllm/pull/53593) | merged | [Bugfix] BailingMoeV3 KDA: skip absent metadata during CUDA graph profiling | `vllm/model_executor/models/bailing_moe_v3.py` |

## 逐 PR diff 审计卡

### PR #51045 - [Model][Frontend] Add Ling 3.0 Flash BF16, MTP, and parser support

- 链接: https://github.com/vllm-project/vllm/pull/51045
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_ling3.py`, `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`, `vllm/parser/ling3.py`, `vllm/reasoning/ling3_reasoning_parser.py` 等 6 个文件；关联提交 `d4da0c55af3a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+2070/-1，可读 patch 2176 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0 (1289 lines); hunks: -0,0 +1,1289; symbols: bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer, _get_kda_state_shape_for_config，涉及 `bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer`；`vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: _get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__, forward，涉及 `_get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__`；`vllm/tool_parsers/ling3_tool_parser.py` added +11/-0 (11 lines); hunks: -0,0 +1,11; symbols: Ling3ToolParser，涉及 `Ling3ToolParser`；`vllm/reasoning/ling3_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0 (1289 lines); hunks: -0,0 +1,1289; symbols: bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer, _get_kda_state_shape_for_config
  - `vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: _get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__, forward
  - `vllm/tool_parsers/ling3_tool_parser.py` added +11/-0 (11 lines); hunks: -0,0 +1,11; symbols: Ling3ToolParser
  - `vllm/reasoning/ling3_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/parser/engine/test_ling3.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: _tokenizer, test_ling3_registered, test_ling3_defaults_thinking_on, test_ling3_disable_thinking_keeps_reasoning_as_content
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_v3.py
@@ -0,0 +1,1289 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""vLLM implementation for BailingMoeV3ForCausalLM.
+The HuggingFace reference model mixes MLA full-attention layers with Kimi
+Delta Attention linear layers and Bailing MoE blocks.  This file keeps the V3
+module/weight names aligned with the reference implementation while reusing
diff -- vllm/model_executor/models/bailing_moe_v3_mtp.py
@@ -0,0 +1,389 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Bailing MoE v3 MTP model."""
+from collections.abc import Iterable
+import torch
+import torch.nn as nn
diff -- vllm/tool_parsers/ling3_tool_parser.py
@@ -0,0 +1,11 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0; `vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0; `vllm/tool_parsers/ling3_tool_parser.py` added +11/-0; `vllm/reasoning/ling3_reasoning_parser.py` added +6/-0; `vllm/parser/ling3.py` added +83/-0
  - tests: `tests/parser/engine/test_ling3.py` added +105/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/parser/engine/test_ling3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #51265 - `[Model][Quantization] Add Ling-3.0-flash-fp8 support`

- 链接: https://github.com/vllm-project/vllm/pull/51265
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`；关联提交 `ba1cdcfcf05f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+298/-36，可读 patch 572 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21 (245 lines); hunks: -10,6 +10,8; -55,6 +57,8; symbols: bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config, _configure_ling_fp8_quant_config，涉及 `bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config`；`vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2 (25 lines); hunks: -15,6 +15,7; -25,8 +26,11; symbols: __init__, compute_logits, BailingMoeV3MTPModel, get_expert_mapping，涉及 `__init__, compute_logits, BailingMoeV3MTPModel`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21 (245 lines); hunks: -10,6 +10,8; -55,6 +57,8; symbols: bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config, _configure_ling_fp8_quant_config
  - `vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2 (25 lines); hunks: -15,6 +15,7; -25,8 +26,11; symbols: __init__, compute_logits, BailingMoeV3MTPModel, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_v3.py
@@ -10,6 +10,8 @@
+from math import lcm
+from typing import TypeGuard
@@ -55,6 +57,8 @@
+from vllm.model_executor.layers.quantization.fp8 import Fp8Config
+from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
@@ -79,7 +83,13 @@
diff -- vllm/model_executor/models/bailing_moe_v3_mtp.py
@@ -15,6 +15,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -25,8 +26,11 @@
+    BailingMoeV3ForCausalLM,
+    _configure_ling_fp8_quant_config,
+    _maybe_pad_block_fp8_shared_expert_checkpoint_tensor,
@@ -79,7 +83,14 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21; `vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2
- 验证与风险: diff 自带测试面 `tests/quantization/test_quark.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #53593 - [Bugfix] BailingMoeV3 KDA: skip absent metadata during CUDA graph profiling

- 链接: https://github.com/vllm-project/vllm/pull/53593
- 状态/时间: merged / 2026-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/bailing_moe_v3.py`；关联提交 `09fddeb4ce7e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 12 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1 (5 lines); hunks: -813,7 +813,10 @@ def _forward(; symbols: _forward，涉及 `_forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1 (5 lines); hunks: -813,7 +813,10 @@ def _forward(; symbols: _forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/bailing_moe_v3.py
@@ -813,7 +813,10 @@ def _forward(
-        attn_metadata = attn_metadata_map[self.prefix]
+        attn_metadata = attn_metadata_map.get(self.prefix)
+        if attn_metadata is None:
+            # Profile/warmup dummy runs skip mamba-family metadata.
+            return
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/bailing_moe_v3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
