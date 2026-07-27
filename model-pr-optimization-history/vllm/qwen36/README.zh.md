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
| 2026-05-18 | [#42242](https://github.com/vllm-project/vllm/pull/42242) | merged | [LoRA] Support 2D and 3D MoE LoRA adapter at the same time | `tests/lora/test_qwen36_moe_lora.py` |
| 2026-07-06 | [#41652](https://github.com/vllm-project/vllm/pull/41652) | merged | [Quantization] add humming moe backend to all dense/moe oracles | `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` |

## 逐 PR diff 审计卡

### PR #42242 - [LoRA] Support 2D and 3D MoE LoRA adapter at the same time

- 链接: https://github.com/vllm-project/vllm/pull/42242
- 状态/时间: merged / 2026-05-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/lora/test_qwen36_moe_lora.py`；关联提交 `7d5b03378268`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+391/-9，可读 patch 607 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA] Support 2D and 3D MoE LoRA adapter at the same time」；模型线: Qwen3.6；类别: 文档/测试/CI；主要 diff: `tests/lora/test_qwen36_moe_lora.py`；技术摘要: 覆盖「[LoRA] Support 2D and 3D MoE LoRA adapter at the same time」；主要实现面是 `tests/lora/test_qwen36_moe_lora.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2，涉及 `_build_prompts, _generate, _run_mixed_2d_3d_lora_test`。
- 代码 diff 细节:
  - `tests/lora/test_qwen36_moe_lora.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _build_prompts, _generate, _run_mixed_2d_3d_lora_test, test_qwen36_moe_mixed_2d_3d_lora_tp2
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
```

- 已读文件:
  - tests: `tests/lora/test_qwen36_moe_lora.py` added +156/-0
- 验证与风险: diff 自带测试面 `tests/lora/conftest.py`, `tests/lora/test_qwen36_moe_lora.py`, `tests/lora/test_qwen3moe_tp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41652 - [Quantization] add humming moe backend to all dense/moe oracles

- 链接: https://github.com/vllm-project/vllm/pull/41652
- 状态/时间: merged / 2026-07-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`；关联提交 `d891b9bd51ce`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 73 个文件，+1336/-122，可读 patch 2552 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] add humming moe backend to all dense/moe oracles」；模型线: Qwen3.6；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`；技术摘要: 覆盖「[Quantization] add humming moe backend to all dense/moe oracles」；主要实现面是 `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0 (10 lines); hunks: -0,0 +1,10
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
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/humming/Qwen3.6-35B-A3B-NVFP4-humming.yaml` added +10/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen2-1.5B-Instruct-FP8W8-humming.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming-act-fp8.yaml`, `tests/evals/gsm8k/configs/humming/Qwen3-0.6B-MXFP8-humming.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
