# vllm Qwen3.6 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` | [#41652](https://github.com/vllm-project/vllm/pull/41652) |
| `tests/lora/test_qwen36_moe_lora.py` | [#42242](https://github.com/vllm-project/vllm/pull/42242) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-18 | [#42242](https://github.com/vllm-project/vllm/pull/42242) | merged | [LoRA] Support 2D and 3D MoE LoRA adapter at the same time | `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py` |
| 2026-07-06 | [#41652](https://github.com/vllm-project/vllm/pull/41652) | merged | [Quantization] add humming moe backend to all dense/moe oracles | `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` |

## Per-PR Diff Audit Cards

### PR #42242 - [LoRA] Support 2D and 3D MoE LoRA adapter at the same time

- Link: https://github.com/vllm-project/vllm/pull/42242
- Status/date: merged / 2026-05-18
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/42242 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_qwen36_moe_lora.py`; associated commits `7d5b03378268`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +391/-9, 607 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] Support 2D and 3D MoE LoRA adapter at the same time"; model line: Qwen3.6; category: performance/backend optimization; main diff: `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py`; technical summary: Covers "[LoRA] Support 2D and 3D MoE LoRA adapter at the same time"; the main implementation surface is `tests/lora/test_qwen36_moe_lora.py`, `vllm/entrypoints/openai/models/serving.py`, `vllm/lora/layers/fused_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2, touching `_build_prompts, _generate, _run_mixed_2d_3d_lora_test`; `vllm/entrypoints/openai/models/serving.py` modified +4/-1 (5 lines); hunks: -121,7 +121,9 @@ async def init_static_loras(self):; -177,6 +179,7 @@ async def load_lora_adapter(; symbols: init_static_loras, load_lora_adapter, touching `init_static_loras, load_lora_adapter`; `vllm/lora/layers/fused_moe.py` modified +4/-0 (4 lines); hunks: -37,6 +37,10 @@ def __init__(self, base_layer: FusedMoE) -> None:; symbols: __init__, touching `__init__`; `vllm/entrypoints/openai/models/protocol.py` modified +1/-0 (1 lines); hunks: -16,3 +16,4 @@ class LoRAModulePath:; symbols: LoRAModulePath, touching `LoRAModulePath`.
- Code diff details:
  - `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2
  - `vllm/entrypoints/openai/models/serving.py` modified +4/-1 (5 lines); hunks: -121,7 +121,9 @@ async def init_static_loras(self):; -177,6 +179,7 @@ async def load_lora_adapter(; symbols: init_static_loras, load_lora_adapter
  - `vllm/lora/layers/fused_moe.py` modified +4/-0 (4 lines); hunks: -37,6 +37,10 @@ def __init__(self, base_layer: FusedMoE) -> None:; symbols: __init__
  - `vllm/entrypoints/openai/models/protocol.py` modified +1/-0 (1 lines); hunks: -16,3 +16,4 @@ class LoRAModulePath:; symbols: LoRAModulePath
  - `vllm/entrypoints/serve/lora/api_router.py` modified +1/-0 (1 lines); hunks: -37,6 +37,7 @@ def attach_router(app: FastAPI):; symbols: attach_router
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/lora/test_qwen36_moe_lora.py` added +156/-0
  - runtime: `vllm/entrypoints/openai/models/serving.py` modified +4/-1; `vllm/lora/layers/fused_moe.py` modified +4/-0; `vllm/entrypoints/openai/models/protocol.py` modified +1/-0; `vllm/entrypoints/serve/lora/api_router.py` modified +1/-0; `vllm/entrypoints/serve/lora/protocol.py` modified +1/-0; `vllm/lora/model_manager.py` modified +117/-3
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_qwen36_moe_lora.py`, `tests/lora/test_qwen3moe_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41652 - [Quantization] add humming moe backend to all dense/moe oracles

- Link: https://github.com/vllm-project/vllm/pull/41652
- Status/date: merged / 2026-07-06
- Metadata refresh note: the current GitHub API lookup failed (`command failed: gh api repos/vllm-project/vllm/pulls/41652 gh: API rate limit exceeded for user ID 35585791. If you reach out to GitHub Support for help, please include the requ...`); this previously audited card is retained instead of discarding immutable commit and diff evidence.
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`; associated commits `d891b9bd51ce`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 73 files, +1336/-122, 2552 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] add humming moe backend to all dense/moe oracles"; model line: Qwen3.6; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`; technical summary: Covers "[Quantization] add humming moe backend to all dense/moe oracles"; the main implementation surface is `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`, `vllm/model_executor/kernels/linear/scaled_mm/humming.py`, `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10; `vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: HummingFP8ScaledMMLinearKernel, is_supported, can_implement, process_weights_after_loading, touching `HummingFP8ScaledMMLinearKernel, is_supported, can_implement`; `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21 (132 lines); hunks: -11,6 +11,7; -46,6 +47,7; symbols: WNA16MoEBackend, backend_to_kernel_cls, for, _get_priority_backends, touching `WNA16MoEBackend, backend_to_kernel_cls, for`; `vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1 (83 lines); hunks: -2,6 +2,7; -22,13 +23,15; symbols: Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_int8_backend, touching `Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls`.
- Code diff details:
  - `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10
  - `vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: HummingFP8ScaledMMLinearKernel, is_supported, can_implement, process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21 (132 lines); hunks: -11,6 +11,7; -46,6 +47,7; symbols: WNA16MoEBackend, backend_to_kernel_cls, for, _get_priority_backends
  - `vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1 (83 lines); hunks: -2,6 +2,7; -22,13 +23,15; symbols: Int8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_int8_backend
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +81/-0 (81 lines); hunks: -1,6 +1,7; -45,6 +46,7 @@ class Fp8MoeBackend(Enum):; symbols: Fp8MoeBackend, _get_priority_backends, backend_to_kernel_cls, map_fp8_backend
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0
  - runtime: `vllm/model_executor/kernels/linear/scaled_mm/humming.py` added +156/-0; `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` modified +111/-21; `vllm/model_executor/layers/fused_moe/oracle/int8.py` modified +82/-1; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +81/-0; `vllm/model_executor/kernels/linear/nvfp4/humming.py` added +73/-0; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +69/-1
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
