# TensorRT-LLM Step 3.7 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tensorrt_llm/_torch/models/modeling_step3p7.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tensorrt_llm/_torch/models/modeling_step3p7vl.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tests/unittest/_torch/modeling/test_modeling_step3p7.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |
| `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711), [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-06-04 | [#14711](https://github.com/NVIDIA/TensorRT-LLM/pull/14711) | merged | [None][feat] Support Step-3.7-Flash model | `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py` |
| 2026-06-10 | [#14926](https://github.com/NVIDIA/TensorRT-LLM/pull/14926) | merged | [None][feat] Enable MTP for Step-3.7 NVFP4 and port Step-3.7VL vision tower to TRT-LLM modules | `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` |

## 逐 PR diff 审计卡

### PR #14711 - [None][feat] Support Step-3.7-Flash model

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14711
- 状态/时间: merged / 2026-06-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py`；关联提交 `6dc60cb2a78e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 22 个文件，+4276/-51，可读 patch 4494 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0 (1689 lines); hunks: -0,0 +1,1689; symbols: _fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched, _nvfp4_stack_dequant，涉及 `_fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched`；`tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0 (1020 lines); hunks: -0,0 +1,1020; symbols: _rotate_half, _apply_rotary_emb, Step3VisionRope2D, __init__，涉及 `_rotate_half, _apply_rotary_emb, Step3VisionRope2D`；`tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0 (715 lines); hunks: -0,0 +1,715; symbols: _load_config, _load_safetensors_keys, TestStep3p7Helpers, test_split_stacked_moe_weights_expands_per_expert_keys，涉及 `_load_config, _load_safetensors_keys, TestStep3p7Helpers`；`tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config, TestStep3p7VisionTower，涉及 `_load_config, _make_tiny_vision_config, _make_tiny_vision_model_config`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0 (1689 lines); hunks: -0,0 +1,1689; symbols: _fp8_block_dequant_3d, _ceil_div, _nvfp4_dequant_batched, _nvfp4_stack_dequant
  - `tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0 (1020 lines); hunks: -0,0 +1,1020; symbols: _rotate_half, _apply_rotary_emb, Step3VisionRope2D, __init__
  - `tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0 (715 lines); hunks: -0,0 +1,715; symbols: _load_config, _load_safetensors_keys, TestStep3p7Helpers, test_split_stacked_moe_weights_expands_per_expert_keys
  - `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0 (529 lines); hunks: -0,0 +1,529; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config, TestStep3p7VisionTower
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_step3p7.py` added +1689/-0; `tensorrt_llm/_torch/models/modeling_step3p7vl.py` added +1020/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_step3p7.py` added +715/-0; `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` added +529/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14926 - [None][feat] Enable MTP for Step-3.7 NVFP4 and port Step-3.7VL vision tower to TRT-LLM modules

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14926
- 状态/时间: merged / 2026-06-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_step3p7.py`, `tensorrt_llm/_torch/models/modeling_step3p7vl.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7.py`, `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py`；关联提交 `62c65216b667`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+977/-283，可读 patch 1880 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149 (662 lines); hunks: -18,17 +18,11; -40,8 +34,9; symbols: _rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin, Step3VisionRope2D，涉及 `_rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin`；`tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4 (13 lines); hunks: -1473,6 +1473,7 @@ def forward(; -1498,18 +1499,22 @@ def forward(; symbols: forward，涉及 `forward`；`tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77 (445 lines); hunks: -17,11 +17,14; -53,6 +56,18; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config，涉及 `_load_config, _make_tiny_vision_config, _make_tiny_vision_model_config`；`tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31 (68 lines); hunks: -52,9 +52,11; -417,10 +419,11 @@ class TestStep3p7Checkpoint(unittest.TestCase):; symbols: TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting，涉及 `TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149 (662 lines); hunks: -18,17 +18,11; -40,8 +34,9; symbols: _rotate_half, _apply_rotary_emb, _apply_rotary_emb_cos_sin, Step3VisionRope2D
  - `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4 (13 lines); hunks: -1473,6 +1473,7 @@ def forward(; -1498,18 +1499,22 @@ def forward(; symbols: forward
  - `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77 (445 lines); hunks: -17,11 +17,14; -53,6 +56,18; symbols: _load_config, _make_tiny_vision_config, _make_tiny_vision_model_config
  - `tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31 (68 lines); hunks: -52,9 +52,11; -417,10 +419,11 @@ class TestStep3p7Checkpoint(unittest.TestCase):; symbols: TestStep3p7Checkpoint, _check_text_config, test_config_and_weight_accounting
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_step3p7vl.py` modified +513/-149; `tensorrt_llm/_torch/models/modeling_step3p7.py` modified +9/-4
  - tests: `tests/unittest/_torch/modeling/test_modeling_step3p7vl.py` modified +368/-77; `tests/unittest/_torch/modeling/test_modeling_step3p7.py` modified +37/-31
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
