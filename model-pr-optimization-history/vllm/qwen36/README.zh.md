# vllm Qwen3.6 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` | [#41652](https://github.com/vllm-project/vllm/pull/41652) |
| `tests/lora/test_qwen36_moe_lora.py` | [#42242](https://github.com/vllm-project/vllm/pull/42242) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-18 | [#42242](https://github.com/vllm-project/vllm/pull/42242) | merged | [LoRA] Support 2D and 3D MoE LoRA adapter at the same time | `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py` |
| 2026-07-06 | [#41652](https://github.com/vllm-project/vllm/pull/41652) | merged | [Quantization] add humming moe backend to all dense/moe oracles | `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` |

## 逐 PR diff 审计卡

### PR #42242 - [LoRA] Support 2D and 3D MoE LoRA adapter at the same time

- 链接: https://github.com/vllm-project/vllm/pull/42242
- 状态/时间: merged / 2026-05-18
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/42242 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_qwen36_moe_lora.py`；关联提交 `7d5b03378268`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+391/-9，可读 patch 607 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA] Support 2D and 3D MoE LoRA adapter at the same time」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py`；技术摘要: 覆盖「[LoRA] Support 2D and 3D MoE LoRA adapter at the same time」；主要实现面是 `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2，涉及 `_build_prompts, _generate, _run_mixed_2d_3d_lora_test`；`vllm/entrypoints/openai/models/serving.py` modified +4/-1 (5 lines); hunks: -121,7 +121,9 @@ async def init_static_loras(self):; -177,6 +179,7 @@ async def load_lora_adapter(; symbols: init_static_loras, load_lora_adapter，涉及 `init_static_loras, load_lora_adapter`；`vllm/lora/layers/fused_moe.py` modified +4/-0 (4 lines); hunks: -37,6 +37,10 @@ def __init__(self, base_layer: FusedMoE) -> None:; symbols: __init__，涉及 `__init__`；`vllm/entrypoints/openai/models/protocol.py` modified +1/-0 (1 lines); hunks: -16,3 +16,4 @@ class LoRAModulePath:; symbols: LoRAModulePath，涉及 `LoRAModulePath`。
- 代码 diff 细节:
  - `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2
  - `vllm/entrypoints/openai/models/serving.py` modified +4/-1 (5 lines); hunks: -121,7 +121,9 @@ async def init_static_loras(self):; -177,6 +179,7 @@ async def load_lora_adapter(; symbols: init_static_loras, load_lora_adapter
  - `vllm/lora/layers/fused_moe.py` modified +4/-0 (4 lines); hunks: -37,6 +37,10 @@ def __init__(self, base_layer: FusedMoE) -> None:; symbols: __init__
  - `vllm/entrypoints/openai/models/protocol.py` modified +1/-0 (1 lines); hunks: -16,3 +16,4 @@ class LoRAModulePath:; symbols: LoRAModulePath
  - `vllm/entrypoints/serve/lora/api_router.py` modified +1/-0 (1 lines); hunks: -37,6 +37,7 @@ def attach_router(app: FastAPI):; symbols: attach_router
- 关键代码摘录:

```diff
diff -- tests/lora/test_qwen36_moe_lora.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+import vllm
+import vllm.config
+from vllm.assets.image import ImageAsset
diff -- vllm/entrypoints/openai/models/serving.py
@@ -121,7 +121,9 @@ async def init_static_loras(self):
-                lora_path=lora.path, lora_name=lora.name
+                lora_path=lora.path,
+                lora_name=lora.name,
+                is_3d_lora_weight=lora.is_3d_lora_weight,
@@ -177,6 +179,7 @@ async def load_lora_adapter(
+                is_3d_lora_weight=request.is_3d_lora_weight,
diff -- vllm/lora/layers/fused_moe.py
@@ -37,6 +37,10 @@ def __init__(self, base_layer: FusedMoE) -> None:
```

- 已读文件:
  - tests: `tests/lora/test_qwen36_moe_lora.py` added +156/-0
  - runtime: `vllm/entrypoints/openai/models/serving.py` modified +4/-1; `vllm/lora/layers/fused_moe.py` modified +4/-0; `vllm/entrypoints/openai/models/protocol.py` modified +1/-0; `vllm/entrypoints/serve/lora/api_router.py` modified +1/-0; `vllm/entrypoints/serve/lora/protocol.py` modified +1/-0; `vllm/lora/model_manager.py` modified +117/-3
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_qwen36_moe_lora.py`, `tests/lora/test_qwen3moe_tp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41652 - [Quantization] add humming moe backend to all dense/moe oracles

- 链接: https://github.com/vllm-project/vllm/pull/41652
- 状态/时间: merged / 2026-07-06
- 元数据刷新说明: 当前 GitHub API 查询失败（`command failed: gh api repos/vllm-project/vllm/pulls/41652 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`）；保留此前已审计卡片，避免丢弃不可变提交与 diff 证据。
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`；关联提交 `d891b9bd51ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 73 个文件，+1336/-122，可读 patch 2552 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] add humming moe backend to all dense/moe oracles」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`；技术摘要: 覆盖「[Quantization] add humming moe backend to all dense/moe oracles」；主要实现面是 `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10；`vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: HummingFP8ScaledMMLinearKernel, is_supported, can_implement, process_weights_after_loading，涉及 `HummingFP8ScaledMMLinearKernel, is_supported, can_implement`；`vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21 (132 lines); hunks: -11,6 +11,7; -46,6 +47,7; symbols: WNA16MoEBackend, backend_to_kernel_cls, for, _get_priority_backends，涉及 `WNA16MoEBackend, backend_to_kernel_cls, for`；`vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1 (83 lines); hunks: -2,6 +2,7; -22,13 +23,15; symbols: Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_int8_backend，涉及 `Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls`。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: HummingFP8ScaledMMLinearKernel, is_supported, can_implement, process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21 (132 lines); hunks: -11,6 +11,7; -46,6 +47,7; symbols: WNA16MoEBackend, backend_to_kernel_cls, for, _get_priority_backends
  - `vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1 (83 lines); hunks: -2,6 +2,7; -22,13 +23,15; symbols: Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_int8_backend
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +81/-0 (81 lines); hunks: -1,6 +1,7; -45,6 +46,7 @@ class Fp8MoeBackend(Enum):; symbols: Fp8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_fp8_backend
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml
@@ -0,0 +1,10 @@
+model_name: "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
+accuracy_threshold: 0.91
+num_questions: 1319
+num_fewshot: 5
+gen_prefix: " <think>\n\n</think>\n"
+server_args: >-
diff -- vllm/model_executor/kernels/linear/scaled_mm/humming.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.logger import init_logger
+from vllm.model_executor.layers.quantization.utils.humming_utils import (
+    convert_linear_layer_to_humming_standard,
diff -- vllm/model_executor/layers/fused_moe/oracle/int_wna16.py
@@ -11,6 +11,7 @@
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0
  - runtime: `vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0; `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21; `vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +81/-0; `vllm/model_executor/kernels/linear/nvfp4/humming.py` added +73/-0; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +69/-1
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
