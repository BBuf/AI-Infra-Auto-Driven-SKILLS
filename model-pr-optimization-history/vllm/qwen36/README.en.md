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
| 2026-05-18 | [#42242](https://github.com/vllm-project/vllm/pull/42242) | merged | [LoRA] Support 2D and 3D MoE LoRA adapter at the same time | `tests/lora/test_qwen36_moe_lora.py` |
| 2026-07-06 | [#41652](https://github.com/vllm-project/vllm/pull/41652) | merged | [Quantization] add humming moe backend to all dense/moe oracles | `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` |

## Per-PR Diff Audit Cards

### PR #42242 - [LoRA] Support 2D and 3D MoE LoRA adapter at the same time

- Link: https://github.com/vllm-project/vllm/pull/42242
- Status/date: merged / 2026-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/lora/test_qwen36_moe_lora.py`; associated commits `7d5b03378268`
- Diff scope read: GitHub Pull Request files API returned 16 files, +391/-9, 607 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] Support 2D and 3D MoE LoRA adapter at the same time"; model line: Qwen3.6; category: docs/tests/CI; main diff: `tests/lora/test_qwen36_moe_lora.py`; technical summary: Covers "[LoRA] Support 2D and 3D MoE LoRA adapter at the same time"; the main implementation surface is `tests/lora/test_qwen36_moe_lora.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2, touching `_build_prompts, _generate, _run_mixed_2d_3d_lora_test`.
- Code diff details:
  - `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2
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
```

- Reviewed files:
  - tests: `tests/lora/test_qwen36_moe_lora.py` added +156/-0
- Risk and verification: The diff ships test coverage in `tests/lora/conftest.py`, `tests/lora/test_qwen36_moe_lora.py`, `tests/lora/test_qwen3moe_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41652 - [Quantization] add humming moe backend to all dense/moe oracles

- Link: https://github.com/vllm-project/vllm/pull/41652
- Status/date: merged / 2026-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`; associated commits `d891b9bd51ce`
- Diff scope read: GitHub Pull Request files API returned 73 files, +1336/-122, 2552 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] add humming moe backend to all dense/moe oracles"; model line: Qwen3.6; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`; technical summary: Covers "[Quantization] add humming moe backend to all dense/moe oracles"; the main implementation surface is `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10.
- Code diff details:
  - `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10
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
```

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
