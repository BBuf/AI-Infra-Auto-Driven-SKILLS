# TensorRT-LLM Step 3.7 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tensorrt_llm/_torch/models/modeling_step3p7.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tensorrt_llm/_torch/models/modeling_step3p7vl.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tests/unittest/_torch/modeling/test_modeling_step3p7.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-06-04 | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711) | merged | [None][feat] Support Step-3.7-Flash model | `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py` |
| 2026-06-10 | [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) | merged | [None][feat] Enable MTP for Step-3.7 NVFP4 and port Step-3.7VL vision tower to TRT-LLM modules | `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` |

## Per-PR Diff Audit Cards

### PR #14711 - [None][feat] Support Step-3.7-Flash model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14711
- Status/date: merged / 2026-06-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py`; associated commits `6dc60cb2a78e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 22 files, +4276/-51, 4494 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0 (1689 lines); hunks: -0,0 +1,1689; symbols: _fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched, _nvfp4_stack_dequant, touching `_fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched`; `tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0 (1020 lines); hunks: -0,0 +1,1020; symbols: _rotate_half, _apply_rotary_emb, Step3VisionRope2D, __init__, touching `_rotate_half, _apply_rotary_emb, Step3VisionRope2D`; `tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0 (715 lines); hunks: -0,0 +1,715; symbols: _load_config, _load_safetensors_keys, TestStep3p7Helpers, test_split_stacked_moe_weights_expands_per_expert_keys, touching `_load_config, _load_safetensors_keys, TestStep3p7Helpers`; `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config, TestStep3p7VisionTower, touching `_load_config, _make_tiny_vision_config, _make_tiny_vision_model_config`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0 (1689 lines); hunks: -0,0 +1,1689; symbols: _fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched, _nvfp4_stack_dequant
  - `tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0 (1020 lines); hunks: -0,0 +1,1020; symbols: _rotate_half, _apply_rotary_emb, Step3VisionRope2D, __init__
  - `tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0 (715 lines); hunks: -0,0 +1,715; symbols: _load_config, _load_safetensors_keys, TestStep3p7Helpers, test_split_stacked_moe_weights_expands_per_expert_keys
  - `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config, TestStep3p7VisionTower
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_step3p7.py
@@ -0,0 +1,1689 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_step3p7vl.py
@@ -0,0 +1,1020 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modeling/test_modeling_step3p7.py
@@ -0,0 +1,715 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0; `tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0; `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14926 - [None][feat] Enable MTP for Step-3.7 NVFP4 and port Step-3.7VL vision tower to TRT-LLM modules

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14926
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py`; associated commits `62c65216b667`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +977/-283, 1880 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149 (662 lines); hunks: -18,17 +18,11; -40,8 +34,9; symbols: _rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin, Step3VisionRope2D, touching `_rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin`; `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4 (13 lines); hunks: -1473,6 +1473,7 @@ def forward(; -1498,18 +1499,22 @@ def forward(; symbols: forward, touching `forward`; `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77 (445 lines); hunks: -17,11 +17,14; -53,6 +56,18; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config, touching `_load_config, _make_tiny_vision_config, _make_tiny_vision_model_config`; `tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31 (68 lines); hunks: -52,9 +52,11; -417,10 +419,11 @@ class TestStep3p7Checkpoint(unittest.TestCase):; symbols: TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting, touching `TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149 (662 lines); hunks: -18,17 +18,11; -40,8 +34,9; symbols: _rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin, Step3VisionRope2D
  - `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4 (13 lines); hunks: -1473,6 +1473,7 @@ def forward(; -1498,18 +1499,22 @@ def forward(; symbols: forward
  - `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77 (445 lines); hunks: -17,11 +17,14; -53,6 +56,18; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config
  - `tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31 (68 lines); hunks: -52,9 +52,11; -417,10 +419,11 @@ class TestStep3p7Checkpoint(unittest.TestCase):; symbols: TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_step3p7vl.py
@@ -18,17 +18,11 @@
-The vision tower is intentionally kept in raw torch (SDPA for non-causal
-attention) instead of TensorRT-LLM's ``Attention`` module.  ``Attention``
-specialises for causal/MLA text decoders; a faithful port of the HF code
-keeps weight names trivially compatible (``vision_model.transformer.resblocks.<i>
-.attn.{in_proj_weight,in_proj_bias,out_proj.{weight,bias}}``) and avoids
-plumbing the vision attention through the text attention metadata.
diff -- tensorrt_llm/_torch/models/modeling_step3p7.py
@@ -1473,6 +1473,7 @@ def forward(
+        spec_input_ids: Optional[torch.LongTensor] = None,
@@ -1498,18 +1499,22 @@ def forward(
-            spec_input_ids = input_ids
+            # The MTP/spec worker always needs the real token ids. On the
+            # multimodal path the main model consumes fused ``inputs_embeds``
+            # and ``input_ids`` is None, so the VLM wrapper forwards the
diff -- tests/unittest/_torch/modeling/test_modeling_step3p7vl.py
@@ -17,11 +17,14 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149; `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4
  - tests: `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77; `tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
