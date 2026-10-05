# vLLM Hunyuan V4 (Hy4) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `benchmarks/kernels/benchmark_hy_v4_ihc.py` | 无直接 PR 号提交 |
| `vllm/models/hy_v4/__init__.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160), [#54405](https://github.com/vllm-project/vllm/pull/54405), [#57526](https://github.com/vllm-project/vllm/pull/57526) |
| `vllm/models/hy_v4/amd/__init__.py` | [#57526](https://github.com/vllm-project/vllm/pull/57526) |
| `vllm/models/hy_v4/amd/model.py` | [#57526](https://github.com/vllm-project/vllm/pull/57526) |
| `vllm/models/hy_v4/nvidia/__init__.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/attention.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160), [#57811](https://github.com/vllm-project/vllm/pull/57811) |
| `vllm/models/hy_v4/nvidia/flashmla_sparse.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/hc.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/model.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/moe.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/mtp.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/models/hy_v4/nvidia/triton_ihc.py` | 无直接 PR 号提交 |
| `vllm/reasoning/hy_v4_reasoning_parser.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/tool_parsers/hy_v4_tool_parser.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/transformers_utils/configs/hy_v4.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-29 | [#54160](https://github.com/vllm-project/vllm/pull/54160) | merged | [Hy4] support Hy4-preview model | `vllm/tool_parsers/hy_v4_tool_parser.py`, `vllm/models/hy_v4/nvidia/mtp.py`, `vllm/models/hy_v4/nvidia/attention.py` |
| 2026-09-08 | [#54405](https://github.com/vllm-project/vllm/pull/54405) | merged | [ROCm][CI] Enable HY-V4 model initialization on ROCm | `vllm/models/hy_v4/__init__.py` |
| 2026-09-21 | [#57811](https://github.com/vllm-project/vllm/pull/57811) | merged | [BUGFIX][HY4] Record indexer completion event for full CUDA graph capture | `vllm/models/hy_v4/nvidia/attention.py` |
| 2026-09-21 | [#57526](https://github.com/vllm-project/vllm/pull/57526) | merged | [Perf][ROCm] Add a ROCm path for Hy4 and compile the backbone | `vllm/models/hy_v4/amd/model.py`, `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/amd/__init__.py` |

## 逐 PR diff 审计卡

### PR #54160 - [Hy4] support Hy4-preview model

- 链接: https://github.com/vllm-project/vllm/pull/54160
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/nvidia/__init__.py`, `vllm/models/hy_v4/nvidia/attention.py`, `vllm/models/hy_v4/nvidia/flashmla_sparse.py`, `vllm/models/hy_v4/nvidia/hc.py` 等 11 个文件；关联提交 `b2f685834a64`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 47 个文件，+6260/-217，可读 patch 7355 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0 (1166 lines); hunks: -0,0 +1,1166; symbols: ToolSchema, ToolCallDict, ExtractResult, StreamToolCall，涉及 `ToolSchema, ToolCallDict, ExtractResult`；`vllm/models/hy_v4/nvidia/mtp.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param, _prepare_mtp_fp8_expert_scale，涉及 `_get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param`；`vllm/models/hy_v4/nvidia/attention.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer, __init__，涉及 `compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer`；`vllm/models/hy_v4/nvidia/model.py` added +716/-0 (716 lines); hunks: -0,0 +1,716; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward，涉及 `_normalize_hyv4_config, HYV4DecoderLayer, __init__`。
- 代码 diff 细节:
  - `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0 (1166 lines); hunks: -0,0 +1,1166; symbols: ToolSchema, ToolCallDict, ExtractResult, StreamToolCall
  - `vllm/models/hy_v4/nvidia/mtp.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param, _prepare_mtp_fp8_expert_scale
  - `vllm/models/hy_v4/nvidia/attention.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer, __init__
  - `vllm/models/hy_v4/nvidia/model.py` added +716/-0 (716 lines); hunks: -0,0 +1,716; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward
  - `vllm/models/hy_v4/nvidia/hc.py` added +370/-0 (370 lines); hunks: -0,0 +1,370; symbols: HYV4HCPreLayer, __init__, reset_parameters, forward
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/hy_v4_tool_parser.py
@@ -0,0 +1,1166 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from __future__ import annotations
+import ast
+import json
+from typing import Any, TypedDict
diff -- vllm/models/hy_v4/nvidia/mtp.py
@@ -0,0 +1,937 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Multi-token prediction (MTP) head for HY V4 (NVIDIA)."""
+import copy
+import typing
+from collections.abc import Callable, Iterable
diff -- vllm/models/hy_v4/nvidia/attention.py
@@ -0,0 +1,765 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0; `vllm/models/hy_v4/nvidia/mtp.py` added +937/-0; `vllm/models/hy_v4/nvidia/attention.py` added +765/-0; `vllm/models/hy_v4/nvidia/model.py` added +716/-0; `vllm/models/hy_v4/nvidia/hc.py` added +370/-0; `vllm/reasoning/hy_v4_reasoning_parser.py` added +324/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/test_registry.py`, `tests/models/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #54405 - [ROCm][CI] Enable HY-V4 model initialization on ROCm

- 链接: https://github.com/vllm-project/vllm/pull/54405
- 状态/时间: merged / 2026-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/hy_v4/__init__.py`；关联提交 `6b5a12c0f843`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+6/-7，可读 patch 36 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/hy_v4/__init__.py` modified +4/-5 (9 lines); hunks: -14,19 +14,18。
- 代码 diff 细节:
  - `vllm/models/hy_v4/__init__.py` modified +4/-5 (9 lines); hunks: -14,19 +14,18
- 关键代码摘录:

```diff
diff -- vllm/models/hy_v4/__init__.py
@@ -14,19 +14,18 @@
-Only NVIDIA is supported for now. The port also drops the reference
+NVIDIA and ROCm are supported. The port also drops the reference
-if current_platform.is_rocm():
-    raise NotImplementedError("hy_v4 does not yet support ROCm.")
-elif current_platform.is_xpu():
+if current_platform.is_xpu():
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/hy_v4/__init__.py` modified +4/-5
- 验证与风险: diff 自带测试面 `tests/models/test_registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #57811 - [BUGFIX][HY4] Record indexer completion event for full CUDA graph capture

- 链接: https://github.com/vllm-project/vllm/pull/57811
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/hy_v4/nvidia/attention.py`；关联提交 `86ce4d10e290`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0 (2 lines); hunks: -758,6 +758,8 @@ def _indexer_and_attn(; symbols: _indexer_and_attn，涉及 `_indexer_and_attn`。
- 代码 diff 细节:
  - `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0 (2 lines); hunks: -758,6 +758,8 @@ def _indexer_and_attn(; symbols: _indexer_and_attn
- 关键代码摘录:

```diff
diff -- vllm/models/hy_v4/nvidia/attention.py
@@ -758,6 +758,8 @@ def _indexer_and_attn(
+        if self.is_sparse:
+            self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/models/hy_v4/nvidia/attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #57526 - [Perf][ROCm] Add a ROCm path for Hy4 and compile the backbone

- 链接: https://github.com/vllm-project/vllm/pull/57526
- 状态/时间: merged / 2026-09-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/amd/__init__.py`, `vllm/models/hy_v4/amd/model.py`；关联提交 `0ff0477ff955`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+782/-7，可读 patch 816 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/models/hy_v4/amd/model.py` added +718/-0 (718 lines); hunks: -0,0 +1,718; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward，涉及 `_normalize_hyv4_config, HYV4DecoderLayer, __init__`；`vllm/models/hy_v4/__init__.py` modified +14/-7 (21 lines); hunks: -14,20 +14,27；`vllm/models/hy_v4/amd/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2。
- 代码 diff 细节:
  - `vllm/models/hy_v4/amd/model.py` added +718/-0 (718 lines); hunks: -0,0 +1,718; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward
  - `vllm/models/hy_v4/__init__.py` modified +14/-7 (21 lines); hunks: -14,20 +14,27
  - `vllm/models/hy_v4/amd/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- 关键代码摘录:

```diff
diff -- vllm/models/hy_v4/amd/model.py
@@ -0,0 +1,718 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/models/hy_v4/__init__.py
@@ -14,20 +14,27 @@
-NVIDIA and ROCm are supported. The port also drops the reference
-implementation's HPC/TPCP fusion paths, which depend on infrastructure that
-does not exist in this tree.
+NVIDIA and ROCm are supported, the latter through `amd/`. The port also drops
+the reference implementation's HPC/TPCP fusion paths, which depend on
+infrastructure that does not exist in this tree.
diff -- vllm/models/hy_v4/amd/__init__.py
@@ -0,0 +1,2 @@
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/models/hy_v4/amd/model.py` added +718/-0; `vllm/models/hy_v4/__init__.py` modified +14/-7; `vllm/models/hy_v4/amd/__init__.py` added +2/-0
- 验证与风险: diff 自带测试面 `tests/models/test_hyv4_rocm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
