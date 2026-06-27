# vllm ERNIE 4.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/model_executor/test_ernie45_vl_mrope.py` | [#39753](https://github.com/vllm-project/vllm/pull/39753) |
| `tests/reasoning/test_ernie45_reasoning_parser.py` | [#25027](https://github.com/vllm-project/vllm/pull/25027) |
| `tests/tool_parsers/test_ernie45_moe_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` | no direct PR-number commit |
| `vllm/model_executor/models/ernie45.py` | [#21735](https://github.com/vllm-project/vllm/pull/21735) |
| `vllm/model_executor/models/ernie45_moe.py` | [#25936](https://github.com/vllm-project/vllm/pull/25936), [#26684](https://github.com/vllm-project/vllm/pull/26684), [#27316](https://github.com/vllm-project/vllm/pull/27316) |
| `vllm/model_executor/models/ernie45_vl.py` | [#39753](https://github.com/vllm-project/vllm/pull/39753) |
| `vllm/model_executor/models/ernie45_vl_moe.py` | [#25936](https://github.com/vllm-project/vllm/pull/25936), [#26885](https://github.com/vllm-project/vllm/pull/26885) |
| `vllm/model_executor/models/ernie_mtp.py` | no direct PR-number commit |
| `vllm/reasoning/ernie45_reasoning_parser.py` | [#25027](https://github.com/vllm-project/vllm/pull/25027), [#27973](https://github.com/vllm-project/vllm/pull/27973) |
| `vllm/tool_parsers/ernie45_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 13
- Total PRs in this document: 21
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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
| 2026-04-16 | [#39780](https://github.com/vllm-project/vllm/pull/39780) | merged | [Bugfix] Reject empty tools array with HTTP 400 | `vllm/entrypoints/openai/chat_completion/protocol.py`, `tests/tool_parsers/test_ernie45_moe_tool_parser.py`, `tests/tool_parsers/test_xlam_tool_parser.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-30 | [#43997](https://github.com/vllm-project/vllm/pull/43997) | merged | [Refactor] Remove dead current_tool_name_sent assignments from tool parsers | `vllm/tool_parsers/hunyuan_a13b_tool_parser.py`, `vllm/tool_parsers/ernie45_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py` |
| 2026-06-08 | [#41184](https://github.com/vllm-project/vllm/pull/41184) | merged | [MoE Refactor] FusedMoE/MoERunner inversion refactor | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 2026-06-18 | [#45988](https://github.com/vllm-project/vllm/pull/45988) | merged | [Perf] Remove unused loggers in `reasoning/` | `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py` |

## Per-PR Diff Audit Cards

### PR #20220 - [Model] Add Ernie4.5 and Ernie4.5MoE Model Support

- Link: https://github.com/vllm-project/vllm/pull/20220
- Status/date: merged / 2025-07-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +634/-0, 657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Ernie4.5 and Ernie4.5MoE Model Support"; model line: ERNIE 4.5; category: docs/tests/CI; main diff: `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`; technical summary: Covers "[Model] Add Ernie4.5 and Ernie4.5MoE Model Support"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunks: -0,0 +1,583; symbols: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE, touching `Ernie4_5_MoeMLP, __init__, forward`; `vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: Ernie4_5_ForCausalLM, __init__, touching `Ernie4_5_ForCausalLM, __init__`; `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -162,6 +162,10 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `docs/models/supported_models.md` modified +2/-0 (2 lines); hunks: -330,6 +330,8 @@ Specified using `--task generate`..
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` added +583/-0 (583 lines); hunks: -0,0 +1,583; symbols: Ernie4_5_MoeMLP, __init__, forward, Ernie4_5_MoeMoE
  - `vllm/model_executor/models/ernie45.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: Ernie4_5_ForCausalLM, __init__
  - `tests/models/registry.py` modified +4/-0 (4 lines); hunks: -162,6 +162,10 @@ def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +2/-0 (2 lines); hunks: -330,6 +330,8 @@ Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: -53,6 +53,8
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` added +583/-0; `vllm/model_executor/models/ernie45.py` added +43/-0; `vllm/model_executor/models/registry.py` modified +2/-0
  - tests: `tests/models/registry.py` modified +4/-0
  - docs: `docs/models/supported_models.md` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21717 - [Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts

- Link: https://github.com/vllm-project/vllm/pull/21717
- Status/date: merged / 2025-07-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-5, 39 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_moe.py`; technical summary: Covers "[Bugfix] Fix Ernie4_5_MoeForCausalLM shared experts"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunks: -109,8 +109,8 @@ def __init__(; -137,7 +137,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` modified +6/-5 (11 lines); hunks: -109,8 +109,8 @@ def __init__(; -137,7 +137,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +6/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21735 - [`Ernie 4.5`] Name Change for Base 0.3B Model

- Link: https://github.com/vllm-project/vllm/pull/21735
- Status/date: merged / 2025-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/ernie45.py`; associated commits `656c24f1b5d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +8/-8, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[`Ernie 4.5`] Name Change for Base 0.3B Model"; model line: ERNIE 4.5; category: model implementation change; main diff: `vllm/model_executor/models/ernie45.py`; technical summary: Covers "[`Ernie 4.5`] Name Change for Base 0.3B Model"; the main implementation surface is `vllm/model_executor/models/ernie45.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7; symbols: Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__, touching `Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/ernie45.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7; symbols: Ernie4_5_ForCausalLM, Ernie4_5ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/ernie45.py
@@ -28,7 +28,7 @@
-class Ernie4_5_ForCausalLM(LlamaForCausalLM):
+class Ernie4_5ForCausalLM(LlamaForCausalLM):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22514 - [Model] Add Ernie4.5 VL Model Support

- Link: https://github.com/vllm-project/vllm/pull/22514
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +2463/-0, 2540 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Ernie4.5 VL Model Support"; model line: ERNIE 4.5; category: model support/runtime entry; main diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`; technical summary: Covers "[Model] Add Ernie4.5 VL Model Support"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/layers/rotary_embedding/mrope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunks: -0,0 +1,1504; symbols: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave, touching `rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision`; `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunks: -0,0 +1,723; symbols: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward, touching `Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__`; `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunks: -393,6 +393,15 @@ def get_input_positions_tensor(; -513,6 +522,120 @@ def _glm4v_get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor, touching `get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor`; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: Ernie4_5_VLRotaryEmbedding, forward, touching `Ernie4_5_VLRotaryEmbedding, forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` added +1504/-0 (1504 lines); hunks: -0,0 +1,1504; symbols: rotate_half, apply_rotary_emb_torch, apply_rotary_pos_emb_vision, all_gather_interleave
  - `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0 (723 lines); hunks: -0,0 +1,723; symbols: Ernie4_5_VLMoeMLP, Ernie4_5_VLMoeAttention, __init__, forward
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0 (123 lines); hunks: -393,6 +393,15 @@ def get_input_positions_tensor(; -513,6 +522,120 @@ def _glm4v_get_input_positions_tensor(; symbols: get_input_positions_tensor, _glm4v_get_input_positions_tensor, _ernie_get_input_positions_tensor, _vl_get_input_positions_tensor
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: Ernie4_5_VLRotaryEmbedding, forward
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: -396,6 +396,8 @@ def check_available_online(; symbols: check_available_online
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` added +1504/-0; `vllm/model_executor/models/ernie45_vl_moe.py` added +723/-0; `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +123/-0; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` added +72/-0; `vllm/model_executor/models/registry.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +2/-0; `tests/models/multimodal/processing/test_common.py` modified +1/-0
  - docs: `docs/models/supported_models.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24074 - [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs

- Link: https://github.com/vllm-project/vllm/pull/24074
- Status/date: merged / 2025-09-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-7, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix][Model] Fix Ernie4.5-VL hanging on long inputs"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; technical summary: Covers "[BugFix][Model] Fix Ernie4.5-VL hanging on long inputs"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunks: -66,8 +66,6; -839,6 +837,15 @@ def get_image_processor(self, **kwargs: object):; symbols: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info, touching `get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item`; `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunks: -287,8 +287,13 @@ def forward(; -310,7 +315,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +10/-4 (14 lines); hunks: -66,8 +66,6; -839,6 +837,15 @@ def get_image_processor(self, **kwargs: object):; symbols: get_image_processor, get_supported_mm_limits, get_mm_max_tokens_per_item, _get_vision_info
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3 (11 lines); hunks: -287,8 +287,13 @@ def forward(; -310,7 +315,7 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +10/-4; `vllm/model_executor/models/ernie45_vl_moe.py` modified +8/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_vl.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25936 - [Bugfix][Model]fix ernie45 moe gate&bias dtype to float32

- Link: https://github.com/vllm-project/vllm/pull/25936
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; associated commits `ef6e0e7132ec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +13/-7, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Model]fix ernie45 moe gate&bias dtype to float32"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/models/ernie45_moe.py`; technical summary: Covers "[Bugfix][Model]fix ernie45 moe gate&bias dtype to float32"; the main implementation surface is `vllm/model_executor/models/ernie45_vl_moe.py`, `vllm/model_executor/models/ernie45_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5 (15 lines); hunks: -199,7 +199,7 @@ def __init__(; -209,6 +209,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/ernie45_moe.py` modified +3/-2 (5 lines); hunks: -120,11 +120,12 @@ def __init__(; -157,7 +158,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5 (15 lines); hunks: -199,7 +199,7 @@ def __init__(; -209,6 +209,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/ernie45_moe.py` modified +3/-2 (5 lines); hunks: -120,11 +120,12 @@ def __init__(; -157,7 +158,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl_moe.py` modified +10/-5; `vllm/model_executor/models/ernie45_moe.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22100 - [EPLB] Support ernie4.5-moe

- Link: https://github.com/vllm-project/vllm/pull/22100
- Status/date: merged / 2025-10-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +132/-7, 243 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[EPLB] Support ernie4.5-moe"; model line: ERNIE 4.5; category: model support/runtime entry; main diff: `vllm/model_executor/models/ernie45_moe.py`; technical summary: Covers "[EPLB] Support ernie4.5-moe"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_moe.py` modified +132/-7 (139 lines); hunks: -33,8 +33,12; -58,7 +62,7; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` modified +132/-7 (139 lines); hunks: -33,8 +33,12; -58,7 +62,7; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +132/-7
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25027 - [Model] Add reasoning_parser and tool_parser for Ernie45 thinking

- Link: https://github.com/vllm-project/vllm/pull/25027
- Status/date: merged / 2025-10-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_ernie45_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`; associated commits `782505ed8eb4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +870/-0, 909 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add reasoning_parser and tool_parser for Ernie45 thinking"; model line: ERNIE 4.5; category: docs/tests/CI; main diff: `vllm/reasoning/ernie45_reasoning_parser.py`, `tests/reasoning/test_ernie45_reasoning_parser.py`; technical summary: Covers "[Model] Add reasoning_parser and tool_parser for Ernie45 thinking"; the main implementation surface is `vllm/reasoning/ernie45_reasoning_parser.py`, `tests/reasoning/test_ernie45_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: Ernie45ReasoningParser, start_token, end_token, __init__, touching `Ernie45ReasoningParser, start_token, end_token`; `tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: ernie45_tokenizer, test_reasoning, touching `ernie45_tokenizer, test_reasoning`.
- Code diff details:
  - `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0 (169 lines); hunks: -0,0 +1,169; symbols: Ernie45ReasoningParser, start_token, end_token, __init__
  - `tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: ernie45_tokenizer, test_reasoning
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/reasoning/ernie45_reasoning_parser.py` added +169/-0
  - tests: `tests/reasoning/test_ernie45_reasoning_parser.py` added +124/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_ernie45_reasoning_parser.py`, `tests/tool_use/test_ernie45_moe_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26684 - [Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code

- Link: https://github.com/vllm-project/vllm/pull/26684
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/ernie45_moe.py`; associated commits `01ad27faff35`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-12, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_moe.py`; technical summary: Covers "[Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_moe.py` modified +22/-12 (34 lines); hunks: -23,7 +23,8; -139,10 +140,10 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` modified +22/-12 (34 lines); hunks: -23,7 +23,8; -139,10 +140,10 @@ def __init__(; symbols: __init__, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +22/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26885 - [Model][Bugfix] fix ernie45 vl run failed from shared experts optimization

- Link: https://github.com/vllm-project/vllm/pull/26885
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/ernie45_vl_moe.py`; associated commits `e51928793e10`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-5, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Bugfix] fix ernie45 vl run failed from shared experts optimization"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_vl_moe.py`; technical summary: Covers "[Model][Bugfix] fix ernie45 vl run failed from shared experts optimization"; the main implementation surface is `vllm/model_executor/models/ernie45_vl_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5 (27 lines); hunks: -341,7 +341,10 @@ def forward(; -353,16 +356,26 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5 (27 lines); hunks: -341,7 +341,10 @@ def forward(; -353,16 +356,26 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl_moe.py` modified +22/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27316 - [Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple

- Link: https://github.com/vllm-project/vllm/pull/27316
- Status/date: merged / 2025-10-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/ernie45_moe.py`; associated commits `63b22e0dbb90`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/model_executor/models/ernie45_moe.py`; technical summary: Covers "[Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_moe.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_moe.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/ernie45_moe.py
@@ -215,6 +215,8 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        else:
+            final_hidden_states = final_hidden_states[1]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_moe.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/ernie45_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27973 - [Model] fix ernie45 reasoning_parser

- Link: https://github.com/vllm-project/vllm/pull/27973
- Status/date: merged / 2025-11-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/reasoning/ernie45_reasoning_parser.py`; associated commits `43a6acfb7de8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] fix ernie45 reasoning_parser"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/reasoning/ernie45_reasoning_parser.py`; technical summary: Covers "[Model] fix ernie45 reasoning_parser"; the main implementation surface is `vllm/reasoning/ernie45_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2 (4 lines); hunks: -36,8 +36,8 @@ def end_token(self) -> str:; symbols: end_token, __init__, touching `end_token, __init__`.
- Code diff details:
  - `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2 (4 lines); hunks: -36,8 +36,8 @@ def end_token(self) -> str:; symbols: end_token, __init__
- Key code excerpts:

```diff
diff -- vllm/reasoning/ernie45_reasoning_parser.py
@@ -36,8 +36,8 @@ def end_token(self) -> str:
-    def __init__(self, tokenizer: PreTrainedTokenizerBase):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
+        super().__init__(tokenizer, *args, **kwargs)
```

- Reviewed files:
  - runtime: `vllm/reasoning/ernie45_reasoning_parser.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/ernie45_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31274 - [Model][Ernie4.5-VL] Support video metadata for timestamp rendering

- Link: https://github.com/vllm-project/vllm/pull/31274
- Status/date: merged / 2025-12-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +82/-5, 137 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Ernie4.5-VL] Support video metadata for timestamp rendering"; model line: ERNIE 4.5; category: docs/tests/CI; main diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`; technical summary: Covers "[Model][Ernie4.5-VL] Support video metadata for timestamp rendering"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunks: -21,7 +21,7; -41,7 +41,7; symbols: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm, touching `get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser`; `tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunks: -104,7 +104,8 @@ def create_metadata(frames: np.ndarray):; symbols: create_metadata, touching `create_metadata`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +80/-4 (84 lines); hunks: -21,7 +21,7; -41,7 +41,7; symbols: get_max_video_tokens, Ernie4_5VLMultiModalProcessor, _get_data_parser, _pixel_values_norm
  - `tests/models/multimodal/processing/test_common.py` modified +2/-1 (3 lines); hunks: -104,7 +104,8 @@ def create_metadata(frames: np.ndarray):; symbols: create_metadata
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +80/-4
  - tests: `tests/models/multimodal/processing/test_common.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39753 - [Model] Use mm_features for Ernie-4.5 VL M-RoPE

- Link: https://github.com/vllm-project/vllm/pull/39753
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/model_executor/test_ernie45_vl_mrope.py`, `vllm/model_executor/models/ernie45_vl.py`; associated commits `0008729abfbd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +196/-123, 339 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use mm_features for Ernie-4.5 VL M-RoPE"; model line: ERNIE 4.5; category: docs/tests/CI; main diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`; technical summary: Covers "[Model] Use mm_features for Ernie-4.5 VL M-RoPE"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/ernie45_vl.py` modified +53/-123 (176 lines); hunks: -23,9 +23,8; -1401,131 +1400,62 @@ def get_mrope_input_positions(; symbols: get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input, touching `get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input`; `tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: _force_cpu_default_device, DummyConfig, make_model, make_mm_feature, touching `_force_cpu_default_device, DummyConfig, make_model`.
- Code diff details:
  - `vllm/model_executor/models/ernie45_vl.py` modified +53/-123 (176 lines); hunks: -23,9 +23,8; -1401,131 +1400,62 @@ def get_mrope_input_positions(; symbols: get_mrope_input_positions, iter_mm_grid_thw, _parse_and_validate_image_input
  - `tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0 (143 lines); hunks: -0,0 +1,143; symbols: _force_cpu_default_device, DummyConfig, make_model, make_mm_feature
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/ernie45_vl.py` modified +53/-123
  - tests: `tests/model_executor/test_ernie45_vl_mrope.py` added +143/-0
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_ernie45_vl_mrope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39780 - [Bugfix] Reject empty tools array with HTTP 400

- Link: https://github.com/vllm-project/vllm/pull/39780
- Status/date: merged / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +23/-23, 81 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Reject empty tools array with HTTP 400"; model line: ERNIE 4.5; category: bug fix; main diff: `vllm/entrypoints/openai/chat_completion/protocol.py`, `tests/tool_parsers/test_ernie45_moe_tool_parser.py`, `tests/tool_parsers/test_xlam_tool_parser.py`; technical summary: Covers "[Bugfix] Reject empty tools array with HTTP 400"; the main implementation surface is `vllm/entrypoints/openai/chat_completion/protocol.py`, `tests/tool_parsers/test_ernie45_moe_tool_parser.py`, `tests/tool_parsers/test_xlam_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/entrypoints/openai/chat_completion/protocol.py` modified +12/-12 (24 lines); hunks: -678,6 +678,18 @@ def check_structured_outputs_count(cls, data):; -704,18 +716,6 @@ def check_tool_usage(cls, data):; symbols: check_structured_outputs_count, check_tool_usage, touching `check_structured_outputs_count, check_tool_usage`; `tests/tool_parsers/test_ernie45_moe_tool_parser.py` modified +1/-1 (2 lines); hunks: -328,7 +328,7 @@ def test_extract_tool_calls_streaming_incremental(; symbols: test_extract_tool_calls_streaming_incremental, touching `test_extract_tool_calls_streaming_incremental`; `tests/tool_parsers/test_xlam_tool_parser.py` modified +1/-1 (2 lines); hunks: -484,7 +484,7 @@ def test_extract_tool_calls_streaming_incremental(; symbols: test_extract_tool_calls_streaming_incremental, touching `test_extract_tool_calls_streaming_incremental`; `tests/tool_use/test_chat_completion_request_validations.py` modified +9/-9 (18 lines); hunks: -26,15 +26,15 @@ def test_chat_completion_request_with_no_tools():; symbols: test_chat_completion_request_with_no_tools, touching `test_chat_completion_request_with_no_tools`.
- Code diff details:
  - `vllm/entrypoints/openai/chat_completion/protocol.py` modified +12/-12 (24 lines); hunks: -678,6 +678,18 @@ def check_structured_outputs_count(cls, data):; -704,18 +716,6 @@ def check_tool_usage(cls, data):; symbols: check_structured_outputs_count, check_tool_usage
  - `tests/tool_parsers/test_ernie45_moe_tool_parser.py` modified +1/-1 (2 lines); hunks: -328,7 +328,7 @@ def test_extract_tool_calls_streaming_incremental(; symbols: test_extract_tool_calls_streaming_incremental
  - `tests/tool_parsers/test_xlam_tool_parser.py` modified +1/-1 (2 lines); hunks: -484,7 +484,7 @@ def test_extract_tool_calls_streaming_incremental(; symbols: test_extract_tool_calls_streaming_incremental
  - `tests/tool_use/test_chat_completion_request_validations.py` modified +9/-9 (18 lines); hunks: -26,15 +26,15 @@ def test_chat_completion_request_with_no_tools():; symbols: test_chat_completion_request_with_no_tools
- Key code excerpts:

```diff
diff -- vllm/entrypoints/openai/chat_completion/protocol.py
@@ -678,6 +678,18 @@ def check_structured_outputs_count(cls, data):
+        if isinstance(data, ValueError):
+            raise data
+        if not isinstance(data, dict):
+            return data
+        # Reject empty tools array, matching OpenAI API behavior
+        if data.get("tools") == []:
diff -- tests/tool_parsers/test_ernie45_moe_tool_parser.py
@@ -328,7 +328,7 @@ def test_extract_tool_calls_streaming_incremental(
-    request = ChatCompletionRequest(model=MODEL, messages=[], tools=[])
+    request = ChatCompletionRequest(model=MODEL, messages=[])
diff -- tests/tool_parsers/test_xlam_tool_parser.py
@@ -484,7 +484,7 @@ def test_extract_tool_calls_streaming_incremental(
-    request = ChatCompletionRequest(model=MODEL, messages=[], tools=[])
+    request = ChatCompletionRequest(model=MODEL, messages=[])
diff -- tests/tool_use/test_chat_completion_request_validations.py
@@ -26,15 +26,15 @@ def test_chat_completion_request_with_no_tools():
```

- Reviewed files:
  - runtime: `vllm/entrypoints/openai/chat_completion/protocol.py` modified +12/-12
  - tests: `tests/tool_parsers/test_ernie45_moe_tool_parser.py` modified +1/-1; `tests/tool_parsers/test_xlam_tool_parser.py` modified +1/-1; `tests/tool_use/test_chat_completion_request_validations.py` modified +9/-9
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_ernie45_moe_tool_parser.py`, `tests/tool_parsers/test_xlam_tool_parser.py`, `tests/tool_use/test_chat_completion_request_validations.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: ERNIE 4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake, touching `_resolve_layer_name, _moe_forward, _moe_forward_shared`; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__, touching `FusedMoE, __init__`; `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- Link: https://github.com/vllm-project/vllm/pull/35782
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 33 files, +112/-141, 926 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove SharedFusedMoE class"; model line: ERNIE 4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[MoE Refactor] Remove SharedFusedMoE class"; the main implementation surface is `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward, touching `SharedFusedMoE, forward`; `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping, touching `__init__, make_empty_intermediate_tensors, get_expert_mapping`; `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights, touching `__init__, load_moe_expert_weights, load_weights`; `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights, touching `__init__, compute_logits, get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward
  - `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -48,9 +48,9; -311,7 +311,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -1,25 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import torch
-from vllm.model_executor.layers.fused_moe.layer import FusedMoE
-# TODO(bnell): Remove this entirely
-class SharedFusedMoE(FusedMoE):
diff -- vllm/model_executor/models/afmoe.py
@@ -18,7 +18,7 @@
-from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -124,8 +124,8 @@ def __init__(
-        # Routed experts using SharedFusedMoE
-        self.experts = SharedFusedMoE(
+        # Routed experts using FusedMoE
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/llama4.py` modified +5/-5; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/deepseek_v2.py` modified +4/-4; `vllm/model_executor/models/ernie45_moe.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: ERNIE 4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43997 - [Refactor] Remove dead current_tool_name_sent assignments from tool parsers

- Link: https://github.com/vllm-project/vllm/pull/43997
- Status/date: merged / 2026-05-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +0/-6, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Remove dead current_tool_name_sent assignments from tool parsers"; model line: ERNIE 4.5; category: model implementation change; main diff: `vllm/tool_parsers/hunyuan_a13b_tool_parser.py`, `vllm/tool_parsers/ernie45_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`; technical summary: Covers "[Refactor] Remove dead current_tool_name_sent assignments from tool parsers"; the main implementation surface is `vllm/tool_parsers/hunyuan_a13b_tool_parser.py`, `vllm/tool_parsers/ernie45_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/hunyuan_a13b_tool_parser.py` modified +0/-3 (3 lines); hunks: -38,7 +38,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -262,7 +261,6 @@ def _handle_test_compatibility(self, current_text: str):; symbols: __init__, _handle_test_compatibility, _handle_tool_name_streaming, touching `__init__, _handle_test_compatibility, _handle_tool_name_streaming`; `vllm/tool_parsers/ernie45_tool_parser.py` modified +0/-1 (1 lines); hunks: -34,7 +34,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, touching `__init__`; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +0/-1 (1 lines); hunks: -246,7 +246,6 @@ def _parse_value(; symbols: _parse_value, __init__, touching `_parse_value, __init__`; `vllm/tool_parsers/phi4mini_tool_parser.py` modified +0/-1 (1 lines); hunks: -47,7 +47,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/tool_parsers/hunyuan_a13b_tool_parser.py` modified +0/-3 (3 lines); hunks: -38,7 +38,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; -262,7 +261,6 @@ def _handle_test_compatibility(self, current_text: str):; symbols: __init__, _handle_test_compatibility, _handle_tool_name_streaming
  - `vllm/tool_parsers/ernie45_tool_parser.py` modified +0/-1 (1 lines); hunks: -34,7 +34,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__
  - `vllm/tool_parsers/hy_v3_tool_parser.py` modified +0/-1 (1 lines); hunks: -246,7 +246,6 @@ def _parse_value(; symbols: _parse_value, __init__
  - `vllm/tool_parsers/phi4mini_tool_parser.py` modified +0/-1 (1 lines); hunks: -47,7 +47,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/hunyuan_a13b_tool_parser.py
@@ -38,7 +38,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        self.current_tool_name_sent = False
@@ -262,7 +261,6 @@ def _handle_test_compatibility(self, current_text: str):
-                    self.current_tool_name_sent = True
@@ -306,7 +304,6 @@ def _handle_tool_name_streaming(
-                self.current_tool_name_sent = True
diff -- vllm/tool_parsers/ernie45_tool_parser.py
@@ -34,7 +34,6 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        self.current_tool_name_sent = False
diff -- vllm/tool_parsers/hy_v3_tool_parser.py
@@ -246,7 +246,6 @@ def _parse_value(
-        self.current_tool_name_sent: bool = False
diff -- vllm/tool_parsers/phi4mini_tool_parser.py
@@ -47,7 +47,6 @@ def __init__(
-        self.current_tool_name_sent: bool = False
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/hunyuan_a13b_tool_parser.py` modified +0/-3; `vllm/tool_parsers/ernie45_tool_parser.py` modified +0/-1; `vllm/tool_parsers/hy_v3_tool_parser.py` modified +0/-1; `vllm/tool_parsers/phi4mini_tool_parser.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/ernie45_tool_parser.py`, `vllm/tool_parsers/hunyuan_a13b_tool_parser.py`, `vllm/tool_parsers/hy_v3_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41184 - [MoE Refactor] FusedMoE/MoERunner inversion refactor

- Link: https://github.com/vllm-project/vllm/pull/41184
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 90 files, +2734/-2027, 7329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; model line: ERNIE 4.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`; technical summary: Covers "[MoE Refactor] FusedMoE/MoERunner inversion refactor"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts, touching `FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE`; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method, touching `FusedMoeWeightScaleSupported, RoutedExperts, __init__`; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward, touching `register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward`; `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__, touching `FusedMoEWithLoRA, __init__`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334 (1648 lines); hunks: -1,1424 +1,404; symbols: FusedMoeWeightScaleSupported, make_parallel_config, FusedMoE, determine_expert_counts
  - `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0 (1144 lines); hunks: -0,0 +1,1144; symbols: FusedMoeWeightScaleSupported, RoutedExperts, __init__, _replace_quant_method
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82 (339 lines); hunks: -1,28 +1,39; -43,8 +54,23; symbols: register_layer_for_moe_forward_op, get_layer_from_name, _moe_forward
  - `vllm/lora/layers/fused_moe.py` modified +76/-43 (119 lines); hunks: -10,7 +10,7; -25,15 +25,24; symbols: FusedMoEWithLoRA, __init__
  - `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1 (107 lines); hunks: -13,7 +13,7; -1633,3 +1633,108 @@ def maybe_remap_kv_scale_name(name: str, params_dict: di...; symbols: maybe_remap_kv_scale_name, maybe_remap_moe_expert_param_name, remap_moe_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1,1424 +1,404 @@
-from collections.abc import Callable, Iterable
-from enum import Enum
-from typing import Literal, cast, overload
+from collections.abc import Callable
+from typing import Any
-from torch.nn.parameter import UninitializedParameter
diff -- vllm/model_executor/layers/fused_moe/routed_experts.py
@@ -0,0 +1,1144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Callable, Iterable
+from enum import Enum
+from typing import TYPE_CHECKING, Any, Literal, cast, overload
+import torch
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@ -1,28 +1,39 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +314/-1334; `vllm/model_executor/layers/fused_moe/routed_experts.py` added +1144/-0; `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` modified +257/-82; `vllm/lora/layers/fused_moe.py` modified +76/-43; `vllm/model_executor/model_loader/weight_utils.py` modified +106/-1; `vllm/model_executor/layers/fused_moe/runner/moe_runner_interface.py` modified +102/-2
- Risk and verification: The diff ships test coverage in `tests/distributed/test_eplb_fused_moe_layer.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45988 - [Perf] Remove unused loggers in `reasoning/`

- Link: https://github.com/vllm-project/vllm/pull/45988
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +0/-27, 148 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Remove unused loggers in `reasoning/`"; model line: ERNIE 4.5; category: performance/backend optimization; main diff: `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`; technical summary: Covers "[Perf] Remove unused loggers in `reasoning/`"; the main implementation surface is `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -6,7 +6,6; -17,8 +16,6; symbols: DeepSeekV3ReasoningParser, touching `DeepSeekV3ReasoningParser`; `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: Ernie45ReasoningParser, touching `Ernie45ReasoningParser`; `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: GraniteReasoningParser, touching `GraniteReasoningParser`; `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: HunyuanA13BReasoningParser, touching `HunyuanA13BReasoningParser`.
- Code diff details:
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -6,7 +6,6; -17,8 +16,6; symbols: DeepSeekV3ReasoningParser
  - `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: Ernie45ReasoningParser
  - `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: GraniteReasoningParser
  - `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -8,15 +8,12; symbols: HunyuanA13BReasoningParser
  - `vllm/reasoning/identity_reasoning_parser.py` modified +0/-3 (3 lines); hunks: -7,15 +7,12; symbols: IdentityReasoningParser
- Key code excerpts:

```diff
diff -- vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -6,7 +6,6 @@
-from vllm.logger import init_logger
@@ -17,8 +16,6 @@
-logger = init_logger(__name__)
diff -- vllm/reasoning/ernie45_reasoning_parser.py
@@ -7,15 +7,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/granite_reasoning_parser.py
@@ -8,15 +8,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/hunyuan_a13b_reasoning_parser.py
@@ -8,15 +8,12 @@
-from vllm.logger import init_logger
-logger = init_logger(__name__)
diff -- vllm/reasoning/identity_reasoning_parser.py
```

- Reviewed files:
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` modified +0/-3; `vllm/reasoning/ernie45_reasoning_parser.py` modified +0/-3; `vllm/reasoning/granite_reasoning_parser.py` modified +0/-3; `vllm/reasoning/hunyuan_a13b_reasoning_parser.py` modified +0/-3; `vllm/reasoning/identity_reasoning_parser.py` modified +0/-3; `vllm/reasoning/minimax_m2_reasoning_parser.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/ernie45_reasoning_parser.py`, `vllm/reasoning/granite_reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
