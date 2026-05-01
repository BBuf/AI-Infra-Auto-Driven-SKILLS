# vllm ERNIE 4.5 Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

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
- Extra PRs preserved from existing docs: 6
- Total PRs in this document: 14
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

## Per-PR Diff Audit Cards

### PR #20220 - [Model] Add Ernie4.5 and Ernie4.5MoE Model Support

- Link: https://github.com/vllm-project/vllm/pull/20220
- Status/date: merged / 2025-07-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +634/-0, 657 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Ernie4.5 and Ernie4.5MoE Model Support"; model line: ERNIE 4.5; category: model support/runtime entry; main diff: `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`; technical summary: Covers "[Model] Add Ernie4.5 and Ernie4.5MoE Model Support"; the main implementation surface is `vllm/model_executor/models/ernie45_moe.py`, `vllm/model_executor/models/ernie45.py`, `tests/models/registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model][Ernie4.5-VL] Support video metadata for timestamp rendering"; model line: ERNIE 4.5; category: model support/runtime entry; main diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`; technical summary: Covers "[Model][Ernie4.5-VL] Support video metadata for timestamp rendering"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `tests/models/multimodal/processing/test_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Model] Use mm_features for Ernie-4.5 VL M-RoPE"; model line: ERNIE 4.5; category: model implementation change; main diff: `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`; technical summary: Covers "[Model] Use mm_features for Ernie-4.5 VL M-RoPE"; the main implementation surface is `vllm/model_executor/models/ernie45_vl.py`, `tests/model_executor/test_ernie45_vl_mrope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
