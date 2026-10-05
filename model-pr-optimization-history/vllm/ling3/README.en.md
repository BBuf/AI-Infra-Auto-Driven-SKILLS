# vLLM Ling 3.0 (BailingMoeV3) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/models/multimodal/processing/test_bailing_moe_v3_vl.py` | no direct PR-number commit |
| `tests/parser/engine/test_ling3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `tests/transformers_utils/test_bailing_moe_v3_vl_config.py` | no direct PR-number commit |
| `vllm/model_executor/models/bailing_moe_v3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045), [#51265](https://github.com/vllm-project/vllm/pull/51265), [#53593](https://github.com/vllm-project/vllm/pull/53593) |
| `vllm/model_executor/models/bailing_moe_v3_mtp.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045), [#51265](https://github.com/vllm-project/vllm/pull/51265) |
| `vllm/model_executor/models/bailing_moe_v3_vl.py` | no direct PR-number commit |
| `vllm/parser/ling3.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/reasoning/ling3_reasoning_parser.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/tool_parsers/ling3_tool_parser.py` | [#51045](https://github.com/vllm-project/vllm/pull/51045) |
| `vllm/transformers_utils/configs/bailing_moe_v3_vl.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-05 | [#51045](https://github.com/vllm-project/vllm/pull/51045) | merged | [Model][Frontend] Add Ling 3.0 Flash BF16, MTP, and parser support | `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`, `vllm/tool_parsers/ling3_tool_parser.py` |
| 2026-08-10 | [#51265](https://github.com/vllm-project/vllm/pull/51265) | merged | `[Model][Quantization] Add Ling-3.0-flash-fp8 support` | `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py` |
| 2026-08-25 | [#53593](https://github.com/vllm-project/vllm/pull/53593) | merged | [Bugfix] BailingMoeV3 KDA: skip absent metadata during CUDA graph profiling | `vllm/model_executor/models/bailing_moe_v3.py` |

## Per-PR Diff Audit Cards

### PR #51045 - [Model][Frontend] Add Ling 3.0 Flash BF16, MTP, and parser support

- Link: https://github.com/vllm-project/vllm/pull/51045
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_ling3.py`, `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`, `vllm/parser/ling3.py`, `vllm/reasoning/ling3_reasoning_parser.py` and 6 files; associated commits `d4da0c55af3a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +2070/-1, 2176 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0 (1289 lines); hunks: -0,0 +1,1289; symbols: bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer, _get_kda_state_shape_for_config, touching `bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer`; `vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: _get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__, forward, touching `_get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__`; `vllm/tool_parsers/ling3_tool_parser.py` added +11/-0 (11 lines); hunks: -0,0 +1,11; symbols: Ling3ToolParser, touching `Ling3ToolParser`; `vllm/reasoning/ling3_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6.
- Code diff details:
  - `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0 (1289 lines); hunks: -0,0 +1,1289; symbols: bailing_v3_kda_attention, bailing_v3_kda_attention_fake, _is_kda_layer, _get_kda_state_shape_for_config
  - `vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0 (389 lines); hunks: -0,0 +1,389; symbols: _get_draft_hf_config, BailingMoeV3MTPSharedHead, __init__, forward
  - `vllm/tool_parsers/ling3_tool_parser.py` added +11/-0 (11 lines); hunks: -0,0 +1,11; symbols: Ling3ToolParser
  - `vllm/reasoning/ling3_reasoning_parser.py` added +6/-0 (6 lines); hunks: -0,0 +1,6
  - `tests/parser/engine/test_ling3.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: _tokenizer, test_ling3_registered, test_ling3_defaults_thinking_on, test_ling3_disable_thinking_keeps_reasoning_as_content
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` added +1289/-0; `vllm/model_executor/models/bailing_moe_v3_mtp.py` added +389/-0; `vllm/tool_parsers/ling3_tool_parser.py` added +11/-0; `vllm/reasoning/ling3_reasoning_parser.py` added +6/-0; `vllm/parser/ling3.py` added +83/-0
  - tests: `tests/parser/engine/test_ling3.py` added +105/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/parser/engine/test_ling3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51265 - `[Model][Quantization] Add Ling-3.0-flash-fp8 support`

- Link: https://github.com/vllm-project/vllm/pull/51265
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/bailing_moe_v3.py`, `vllm/model_executor/models/bailing_moe_v3_mtp.py`; associated commits `ba1cdcfcf05f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +298/-36, 572 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21 (245 lines); hunks: -10,6 +10,8; -55,6 +57,8; symbols: bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config, _configure_ling_fp8_quant_config, touching `bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config`; `vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2 (25 lines); hunks: -15,6 +15,7; -25,8 +26,11; symbols: __init__, compute_logits, BailingMoeV3MTPModel, get_expert_mapping, touching `__init__, compute_logits, BailingMoeV3MTPModel`.
- Code diff details:
  - `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21 (245 lines); hunks: -10,6 +10,8; -55,6 +57,8; symbols: bailing_v3_kda_attention, _load_a_log, _is_block_fp8_config, _configure_ling_fp8_quant_config
  - `vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2 (25 lines); hunks: -15,6 +15,7; -25,8 +26,11; symbols: __init__, compute_logits, BailingMoeV3MTPModel, get_expert_mapping
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` modified +224/-21; `vllm/model_executor/models/bailing_moe_v3_mtp.py` modified +23/-2
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53593 - [Bugfix] BailingMoeV3 KDA: skip absent metadata during CUDA graph profiling

- Link: https://github.com/vllm-project/vllm/pull/53593
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/bailing_moe_v3.py`; associated commits `09fddeb4ce7e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +4/-1, 12 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1 (5 lines); hunks: -813,7 +813,10 @@ def _forward(; symbols: _forward, touching `_forward`.
- Code diff details:
  - `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1 (5 lines); hunks: -813,7 +813,10 @@ def _forward(; symbols: _forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/bailing_moe_v3.py
@@ -813,7 +813,10 @@ def _forward(
-        attn_metadata = attn_metadata_map[self.prefix]
+        attn_metadata = attn_metadata_map.get(self.prefix)
+        if attn_metadata is None:
+            # Profile/warmup dummy runs skip mamba-family metadata.
+            return
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/model_executor/models/bailing_moe_v3.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/bailing_moe_v3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
