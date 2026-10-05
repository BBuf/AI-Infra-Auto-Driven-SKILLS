# vLLM Hunyuan V4 (Hy4) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmarks/kernels/benchmark_hy_v4_ihc.py` | no direct PR-number commit |
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
| `vllm/models/hy_v4/nvidia/triton_ihc.py` | no direct PR-number commit |
| `vllm/reasoning/hy_v4_reasoning_parser.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/tool_parsers/hy_v4_tool_parser.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |
| `vllm/transformers_utils/configs/hy_v4.py` | [#54160](https://github.com/vllm-project/vllm/pull/54160) |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-29 | [#54160](https://github.com/vllm-project/vllm/pull/54160) | merged | [Hy4] support Hy4-preview model | `vllm/tool_parsers/hy_v4_tool_parser.py`, `vllm/models/hy_v4/nvidia/mtp.py`, `vllm/models/hy_v4/nvidia/attention.py` |
| 2026-09-08 | [#54405](https://github.com/vllm-project/vllm/pull/54405) | merged | [ROCm][CI] Enable HY-V4 model initialization on ROCm | `vllm/models/hy_v4/__init__.py` |
| 2026-09-21 | [#57811](https://github.com/vllm-project/vllm/pull/57811) | merged | [BUGFIX][HY4] Record indexer completion event for full CUDA graph capture | `vllm/models/hy_v4/nvidia/attention.py` |
| 2026-09-21 | [#57526](https://github.com/vllm-project/vllm/pull/57526) | merged | [Perf][ROCm] Add a ROCm path for Hy4 and compile the backbone | `vllm/models/hy_v4/amd/model.py`, `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/amd/__init__.py` |

## Per-PR Diff Audit Cards

### PR #54160 - [Hy4] support Hy4-preview model

- Link: https://github.com/vllm-project/vllm/pull/54160
- Status/date: merged / 2026-08-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/nvidia/__init__.py`, `vllm/models/hy_v4/nvidia/attention.py`, `vllm/models/hy_v4/nvidia/flashmla_sparse.py`, `vllm/models/hy_v4/nvidia/hc.py` and 11 files; associated commits `b2f685834a64`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 47 files, +6260/-217, 7355 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0 (1166 lines); hunks: -0,0 +1,1166; symbols: ToolSchema, ToolCallDict, ExtractResult, StreamToolCall, touching `ToolSchema, ToolCallDict, ExtractResult`; `vllm/models/hy_v4/nvidia/mtp.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param, _prepare_mtp_fp8_expert_scale, touching `_get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param`; `vllm/models/hy_v4/nvidia/attention.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer, __init__, touching `compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer`; `vllm/models/hy_v4/nvidia/model.py` added +716/-0 (716 lines); hunks: -0,0 +1,716; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward, touching `_normalize_hyv4_config, HYV4DecoderLayer, __init__`.
- Code diff details:
  - `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0 (1166 lines); hunks: -0,0 +1,1166; symbols: ToolSchema, ToolCallDict, ExtractResult, StreamToolCall
  - `vllm/models/hy_v4/nvidia/mtp.py` added +937/-0 (937 lines); hunks: -0,0 +1,937; symbols: _get_spec_layer_idx_from_weight_name, _should_skip_missing_mtp_scale_param, _resolve_fused_expert_param, _prepare_mtp_fp8_expert_scale
  - `vllm/models/hy_v4/nvidia/attention.py` added +765/-0 (765 lines); hunks: -0,0 +1,765; symbols: compute_skip_topk_layers, is_skip_topk_indexer_weight, Indexer, __init__
  - `vllm/models/hy_v4/nvidia/model.py` added +716/-0 (716 lines); hunks: -0,0 +1,716; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward
  - `vllm/models/hy_v4/nvidia/hc.py` added +370/-0 (370 lines); hunks: -0,0 +1,370; symbols: HYV4HCPreLayer, __init__, reset_parameters, forward
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `vllm/tool_parsers/hy_v4_tool_parser.py` added +1166/-0; `vllm/models/hy_v4/nvidia/mtp.py` added +937/-0; `vllm/models/hy_v4/nvidia/attention.py` added +765/-0; `vllm/models/hy_v4/nvidia/model.py` added +716/-0; `vllm/models/hy_v4/nvidia/hc.py` added +370/-0; `vllm/reasoning/hy_v4_reasoning_parser.py` added +324/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_registry.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #54405 - [ROCm][CI] Enable HY-V4 model initialization on ROCm

- Link: https://github.com/vllm-project/vllm/pull/54405
- Status/date: merged / 2026-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/hy_v4/__init__.py`; associated commits `6b5a12c0f843`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +6/-7, 36 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/hy_v4/__init__.py` modified +4/-5 (9 lines); hunks: -14,19 +14,18.
- Code diff details:
  - `vllm/models/hy_v4/__init__.py` modified +4/-5 (9 lines); hunks: -14,19 +14,18
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/hy_v4/__init__.py` modified +4/-5
- Risk and verification: The diff ships test coverage in `tests/models/test_registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57811 - [BUGFIX][HY4] Record indexer completion event for full CUDA graph capture

- Link: https://github.com/vllm-project/vllm/pull/57811
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/hy_v4/nvidia/attention.py`; associated commits `86ce4d10e290`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0 (2 lines); hunks: -758,6 +758,8 @@ def _indexer_and_attn(; symbols: _indexer_and_attn, touching `_indexer_and_attn`.
- Code diff details:
  - `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0 (2 lines); hunks: -758,6 +758,8 @@ def _indexer_and_attn(; symbols: _indexer_and_attn
- Key code excerpts:

```diff
diff -- vllm/models/hy_v4/nvidia/attention.py
@@ -758,6 +758,8 @@ def _indexer_and_attn(
+        if self.is_sparse:
+            self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/hy_v4/nvidia/attention.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/hy_v4/nvidia/attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57526 - [Perf][ROCm] Add a ROCm path for Hy4 and compile the backbone

- Link: https://github.com/vllm-project/vllm/pull/57526
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/hy_v4/__init__.py`, `vllm/models/hy_v4/amd/__init__.py`, `vllm/models/hy_v4/amd/model.py`; associated commits `0ff0477ff955`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +782/-7, 816 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/hy_v4/amd/model.py` added +718/-0 (718 lines); hunks: -0,0 +1,718; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward, touching `_normalize_hyv4_config, HYV4DecoderLayer, __init__`; `vllm/models/hy_v4/__init__.py` modified +14/-7 (21 lines); hunks: -14,20 +14,27; `vllm/models/hy_v4/amd/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/hy_v4/amd/model.py` added +718/-0 (718 lines); hunks: -0,0 +1,718; symbols: _normalize_hyv4_config, HYV4DecoderLayer, __init__, forward
  - `vllm/models/hy_v4/__init__.py` modified +14/-7 (21 lines); hunks: -14,20 +14,27
  - `vllm/models/hy_v4/amd/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/hy_v4/amd/model.py` added +718/-0; `vllm/models/hy_v4/__init__.py` modified +14/-7; `vllm/models/hy_v4/amd/__init__.py` added +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/test_hyv4_rocm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
