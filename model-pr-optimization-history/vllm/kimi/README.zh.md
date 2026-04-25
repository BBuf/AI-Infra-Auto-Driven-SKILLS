# vllm Kimi K2/K2.5/Linear/VL 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_kimi_k2_reasoning_parser.py` | [#37438](https://github.com/vllm-project/vllm/pull/37438) |
| `tests/tool_parsers/test_kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/model_executor/models/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127), [#36903](https://github.com/vllm-project/vllm/pull/36903) |
| `vllm/model_executor/models/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33320](https://github.com/vllm-project/vllm/pull/33320), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#33562](https://github.com/vllm-project/vllm/pull/33562), [#33876](https://github.com/vllm-project/vllm/pull/33876), [#34427](https://github.com/vllm-project/vllm/pull/34427), [#34501](https://github.com/vllm-project/vllm/pull/34501), [#36192](https://github.com/vllm-project/vllm/pull/36192), [#36361](https://github.com/vllm-project/vllm/pull/36361), [#37693](https://github.com/vllm-project/vllm/pull/37693), [#39344](https://github.com/vllm-project/vllm/pull/39344) |
| `vllm/model_executor/models/kimi_k25_vit.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33346](https://github.com/vllm-project/vllm/pull/33346), [#34501](https://github.com/vllm-project/vllm/pull/34501) |
| `vllm/model_executor/models/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809), [#27834](https://github.com/vllm-project/vllm/pull/27834), [#27885](https://github.com/vllm-project/vllm/pull/27885), [#37371](https://github.com/vllm-project/vllm/pull/37371) |
| `vllm/model_executor/models/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#16833](https://github.com/vllm-project/vllm/pull/16833), [#17156](https://github.com/vllm-project/vllm/pull/17156), [#21769](https://github.com/vllm-project/vllm/pull/21769), [#23114](https://github.com/vllm-project/vllm/pull/23114), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#31738](https://github.com/vllm-project/vllm/pull/31738) |
| `vllm/model_executor/models/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387), [#23817](https://github.com/vllm-project/vllm/pull/23817), [#29309](https://github.com/vllm-project/vllm/pull/29309), [#31738](https://github.com/vllm-project/vllm/pull/31738) |
| `vllm/reasoning/kimi_k2_reasoning_parser.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131), [#33646](https://github.com/vllm-project/vllm/pull/33646) |
| `vllm/tokenizers/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/tool_parsers/kimi_k2_tool_parser.py` | [#31207](https://github.com/vllm-project/vllm/pull/31207), [#38579](https://github.com/vllm-project/vllm/pull/38579) |
| `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/configs/kimi_k25.py` | [#33131](https://github.com/vllm-project/vllm/pull/33131) |
| `vllm/transformers_utils/configs/kimi_linear.py` | [#27809](https://github.com/vllm-project/vllm/pull/27809) |
| `vllm/transformers_utils/configs/kimi_vl.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/configs/moonvit.py` | [#16387](https://github.com/vllm-project/vllm/pull/16387) |
| `vllm/transformers_utils/processors/kimi_audio.py` | [#36127](https://github.com/vllm-project/vllm/pull/36127) |
| `vllm/transformers_utils/processors/kimi_k25.py` | [#37693](https://github.com/vllm-project/vllm/pull/37693) |

## PR 覆盖总览

- git 追溯 PR 数: 29
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 29
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-14 | [#16387](https://github.com/vllm-project/vllm/pull/16387) | merged | [Model][VLM] Add Kimi-VL model support | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py` |
| 2025-04-18 | [#16833](https://github.com/vllm-project/vllm/pull/16833) | merged | [Misc] Clean up Kimi-VL | `vllm/model_executor/models/kimi_vl.py` |
| 2025-04-25 | [#17156](https://github.com/vllm-project/vllm/pull/17156) | merged | fix float16 support for kimi-vl | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-05 | [#21769](https://github.com/vllm-project/vllm/pull/21769) | merged | Migrate KimiVLImagePixelInputs to TensorSchema | `vllm/model_executor/models/kimi_vl.py` |
| 2025-08-19 | [#23114](https://github.com/vllm-project/vllm/pull/23114) | merged | [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/kimi_vl.py` |
| 2025-09-01 | [#23817](https://github.com/vllm-project/vllm/pull/23817) | merged | [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506 | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2025-10-30 | [#27809](https://github.com/vllm-project/vllm/pull/27809) | merged | [Model] Introduce Kimi Linear to vLLM | `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py` |
| 2025-10-31 | [#27834](https://github.com/vllm-project/vllm/pull/27834) | merged | [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants | `vllm/model_executor/models/kimi_linear.py` |
| 2025-10-31 | [#27885](https://github.com/vllm-project/vllm/pull/27885) | merged | fix incorrect type annotation in KimiMLP | `vllm/model_executor/models/kimi_linear.py` |
| 2025-11-24 | [#29309](https://github.com/vllm-project/vllm/pull/29309) | merged | [XPU]fix Kimi-VL-A3B-thinking on xpu | `vllm/model_executor/models/moonvit.py` |
| 2025-12-30 | [#31207](https://github.com/vllm-project/vllm/pull/31207) | merged | fix: update kimi k2 tool parser logic | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |
| 2026-01-06 | [#31738](https://github.com/vllm-project/vllm/pull/31738) | merged | [Models]: Use `MMEncoderAttention` for MoonViT | `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py` |
| 2026-01-27 | [#33131](https://github.com/vllm-project/vllm/pull/33131) | merged | [Models] Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py` |
| 2026-01-29 | [#33320](https://github.com/vllm-project/vllm/pull/33320) | merged | [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d… | `vllm/model_executor/models/kimi_k25.py` |
| 2026-01-30 | [#33346](https://github.com/vllm-project/vllm/pull/33346) | merged | [Models] Refactor Kimi-K2.5 weight loading | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py` |
| 2026-02-02 | [#33562](https://github.com/vllm-project/vllm/pull/33562) | merged | [Bugfix] Enable Kimi k25 processor test | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34427](https://github.com/vllm-project/vllm/pull/34427) | merged | [Bugfix] Delete unused redundant code in Kimi-K2.5 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-13 | [#34501](https://github.com/vllm-project/vllm/pull/34501) | merged | [Bugfix] Add quant_config in ViT of Kimi-K2.5 | `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-02-27 | [#33646](https://github.com/vllm-project/vllm/pull/33646) | merged | [Bugfix] Handle case when kimi ends reasoning with a tool call | `vllm/reasoning/kimi_k2_reasoning_parser.py` |
| 2026-03-06 | [#36192](https://github.com/vllm-project/vllm/pull/36192) | merged | [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-11 | [#36127](https://github.com/vllm-project/vllm/pull/36127) | merged | [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct | `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py` |
| 2026-03-11 | [#36361](https://github.com/vllm-project/vllm/pull/36361) | merged | Kimi k2.5 MLA based eagle3 | `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-14 | [#36903](https://github.com/vllm-project/vllm/pull/36903) | merged | [Misc] Clean up Kimi-audio whisper encoder loading | `vllm/model_executor/models/kimi_audio.py` |
| 2026-03-18 | [#37371](https://github.com/vllm-project/vllm/pull/37371) | merged | standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 | `vllm/model_executor/models/kimi_linear.py` |
| 2026-03-19 | [#37438](https://github.com/vllm-project/vllm/pull/37438) | merged | [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support | `tests/reasoning/test_kimi_k2_reasoning_parser.py` |
| 2026-03-20 | [#37693](https://github.com/vllm-project/vllm/pull/37693) | merged | [Model] Update Kimi-K25 and Isaac processors to fit HF-style | `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-12 | [#39344](https://github.com/vllm-project/vllm/pull/39344) | merged | fix(kimi_k25): resolve media_placeholder_token_id from tokenizer | `vllm/model_executor/models/kimi_k25.py` |
| 2026-04-19 | [#38579](https://github.com/vllm-project/vllm/pull/38579) | merged | [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping | `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py` |

## 逐 PR diff 审计卡

### PR #16387 - [Model][VLM] Add Kimi-VL model support

- 链接: https://github.com/vllm-project/vllm/pull/16387
- 状态/时间: merged / 2025-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`, `vllm/transformers_utils/configs/kimi_vl.py`, `vllm/transformers_utils/configs/moonvit.py`；关联提交 `b1308b84a3a6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+1436/-14，可读 patch 1618 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Model][VLM] Add Kimi-VL model support」，变更集中在 `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`, `vllm/transformers_utils/configs/kimi_vl.py`。PR 描述补充为：CLOSES #16387 # Feature * Added support for Kimi-VL * https://github.com/MoonshotAI/Kimi-VL/ * https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct * https://huggingface.co/mo...
- 实现要点: `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope，涉及 `multihead_attention, sdpa_attention, _apply_rope_input_validation`；`vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward，涉及 `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`；`vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__，涉及 `KimiVLConfig, __init__`；`vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__，涉及 `MoonViTConfig, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/moonvit.py` added +628/-0 (628 lines); hunks: -0,0 +1,628; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, apply_rope
  - `vllm/model_executor/models/kimi_vl.py` added +608/-0 (608 lines); hunks: -0,0 +1,608; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
  - `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0 (36 lines); hunks: -0,0 +1,36; symbols: KimiVLConfig, __init__
  - `vllm/transformers_utils/configs/moonvit.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: MoonViTConfig, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -0,0 +1,628 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# This file is meant to be used in kimi_vl.py only
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/model_executor/models/kimi_vl.py
@@ -0,0 +1,608 @@
+# SPDX-License-Identifier: Apache-2.0
+# ruff: noqa: E501
+# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
+# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
+#
+# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
diff -- vllm/transformers_utils/configs/kimi_vl.py
@@ -0,0 +1,36 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/moonvit.py` added +628/-0; `vllm/model_executor/models/kimi_vl.py` added +608/-0; `vllm/transformers_utils/configs/kimi_vl.py` added +36/-0; `vllm/transformers_utils/configs/moonvit.py` added +32/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/vlm_utils/model_utils.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16833 - [Misc] Clean up Kimi-VL

- 链接: https://github.com/vllm-project/vllm/pull/16833
- 状态/时间: merged / 2025-04-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`；关联提交 `aadb6565628c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+20/-44，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 调整模型相关实现，标题为「[Misc] Clean up Kimi-VL」，变更集中在 `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：Apply #15799 and #16416 to Kimi-VL model. cc @courage17340
- 实现要点: `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens，涉及 `KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-40 (57 lines); hunks: -56,7 +56,6; -70,22 +69,20; symbols: KimiVLProcessingInfo, get_hf_config, get_supported_mm_limits, get_num_image_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,7 +56,6 @@
-from vllm.logger import init_logger
@@ -70,22 +69,20 @@
-from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
-                                    NestedTensors)
+from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
+                                    MultiModalKwargs, NestedTensors)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-40
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17156 - fix float16 support for kimi-vl

- 链接: https://github.com/vllm-project/vllm/pull/17156
- 状态/时间: merged / 2025-04-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`；关联提交 `69bff9bc8934`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「fix float16 support for kimi-vl」，变更集中在 `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：FIX https://github.com/MoonshotAI/Kimi-VL/issues/41 inference with float16: output:
- 实现要点: `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input，涉及 `_parse_and_validate_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-2 (3 lines); hunks: -340,8 +340,7 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -340,8 +340,7 @@ def _parse_and_validate_image_input(
-        # fp32 -> bf16
-        pixel_values = pixel_values.to(torch.bfloat16)
+        pixel_values = pixel_values.to(self.vision_tower.dtype)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21769 - Migrate KimiVLImagePixelInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/21769
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`；关联提交 `05fae021750b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+15/-9，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 调整模型相关实现，标题为「Migrate KimiVLImagePixelInputs to TensorSchema」，变更集中在 `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：## Purpose This PR migrates KimiVLImagePixelInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings it in line wit...
- 实现要点: `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input，涉及 `forward, KimiVLImagePixelInputs, _parse_and_validate_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_vl.py` modified +15/-9 (24 lines); hunks: -46,7 +46,7; -79,6 +79,7; symbols: forward, KimiVLImagePixelInputs, _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -46,7 +46,7 @@
-from typing import Any, Literal, Optional, TypedDict, Union
+from typing import Annotated, Any, Literal, Optional, Union
@@ -79,6 +79,7 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
@@ -118,15 +119,22 @@ def forward(self, image_features: torch.Tensor) -> torch.Tensor:
-class KimiVLImagePixelInputs(TypedDict):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +15/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23114 - [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506

- 链接: https://github.com/vllm-project/vllm/pull/23114
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`；关联提交 `fda9537c5e61`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-13，可读 patch 77 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506」，变更集中在 `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：## Purpose Fixes https://github.com/vllm-project/vllm/issues/23077 Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506 ## Test Plan ## Test Result ## (Optional...
- 实现要点: `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__，涉及 `get_replacement, KimiVLForConditionalGeneration, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_vl.py` modified +17/-12 (29 lines); hunks: -54,16 +54,16; -81,7 +81,7; symbols: get_replacement, KimiVLForConditionalGeneration, get_placeholder_str, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_vl.py
@@ -54,16 +54,16 @@
-from vllm.distributed import (get_tensor_model_parallel_rank,
-                              get_tensor_model_parallel_world_size)
+from vllm.distributed import get_pp_group
-from vllm.model_executor.models.interfaces import SupportsMultiModal
+from vllm.model_executor.models.interfaces import (SupportsMultiModal,
+                                                   SupportsPP)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_vl.py` modified +17/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23817 - [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506

- 链接: https://github.com/vllm-project/vllm/pull/23817
- 状态/时间: merged / 2025-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`；关联提交 `a0e0efd6bdcf`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+157/-62，可读 patch 478 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506」，变更集中在 `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：## Purpose Add option to run MiniCPM-V-4 vision encoder in data parallel manner while the main model is in TP. Can be enabled by flag: mm_encoder_tp_mode="data" ## Test Plan ##...
- 实现要点: `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer，涉及 `MLP2, __init__, forward`；`vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward，涉及 `MaxImageTokenMeta, KimiVLMultiModalProjector, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/moonvit.py` modified +55/-22 (77 lines); hunks: -42,7 +42,6; -55,6 +54,8; symbols: MLP2, __init__, forward, MoonVitEncoderLayer
  - `vllm/model_executor/models/kimi_vl.py` modified +39/-15 (54 lines); hunks: -56,6 +56,7; -76,6 +77,7; symbols: MaxImageTokenMeta, KimiVLMultiModalProjector, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -42,7 +42,6 @@
-import math
@@ -55,6 +54,8 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.models.utils import maybe_prefix
@@ -383,21 +384,30 @@ class MLP2(nn.Module):
-    def __init__(self, dims: list[int], activation, bias=True):
diff -- vllm/model_executor/models/kimi_vl.py
@@ -56,6 +56,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -76,6 +77,7 @@
+from vllm.multimodal.utils import run_dp_sharded_mrope_vision_model
@@ -93,29 +95,35 @@ class MaxImageTokenMeta:
-    def __init__(self, config: KimiVLConfig):
+    def __init__(self, config: KimiVLConfig, \
```

- 已读文件:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +55/-22; `vllm/model_executor/models/kimi_vl.py` modified +39/-15
- 验证与风险: diff 自带测试面 `tests/multimodal/test_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27809 - [Model] Introduce Kimi Linear to vLLM

- 链接: https://github.com/vllm-project/vllm/pull/27809
- 状态/时间: merged / 2025-10-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`；关联提交 `4e68cc9b6aa2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+1326/-49，可读 patch 1510 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Model] Introduce Kimi Linear to vLLM」，变更集中在 `vllm/model_executor/models/kimi_linear.py`, `vllm/transformers_utils/configs/kimi_linear.py`。PR 描述补充为：## Purpose Introducing Kimi Linear, an advanced hybrid attention model that combines the efficiency of Kimi Delta Attention (KDA), a refined version of Gated DeltaNet, with redu...
- 实现要点: `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE，涉及 `KimiMLP, __init__, forward`；`vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe，涉及 `KimiLinearConfig, __init__, is_mla`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_linear.py` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: KimiMLP, __init__, forward, KimiMoE
  - `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiLinearConfig, __init__, is_mla, is_moe
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -0,0 +1,663 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Iterable
+from typing import Any
+import torch
+from torch import nn
diff -- vllm/transformers_utils/configs/kimi_linear.py
@@ -0,0 +1,144 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers.configuration_utils import PretrainedConfig
+from vllm.logger import init_logger
+logger = init_logger(__name__)
+class KimiLinearConfig(PretrainedConfig):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_linear.py` added +663/-0; `vllm/transformers_utils/configs/kimi_linear.py` added +144/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27834 - [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants

- 链接: https://github.com/vllm-project/vllm/pull/27834
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_linear.py`；关联提交 `e5ef4dfc11ab`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Kimi-Linear] Correct prefixes and add compatibility to AWQ quants」，变更集中在 `vllm/model_executor/models/kimi_linear.py`。PR 描述补充为：## Purpose This PR purpose is to add prefix to shared_experts params and correct block_sparse_moe prefix from "mlp" to "block_sparse_moe", which ultimately allows vllm to initia...
- 实现要点: `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_linear.py` modified +2/-1 (3 lines); hunks: -155,6 +155,7 @@ def __init__(; -340,7 +341,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -155,6 +155,7 @@ def __init__(
+                prefix=f"{prefix}.shared_experts",
@@ -340,7 +341,7 @@ def __init__(
-                prefix=f"{prefix}.mlp",
+                prefix=f"{prefix}.block_sparse_moe",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27885 - fix incorrect type annotation in KimiMLP

- 链接: https://github.com/vllm-project/vllm/pull/27885
- 状态/时间: merged / 2025-10-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_linear.py`；关联提交 `bc306fe5e978`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「fix incorrect type annotation in KimiMLP」，变更集中在 `vllm/model_executor/models/kimi_linear.py`。PR 描述补充为：Description: ## Summary Fixed incorrect type annotation for `quant_config` parameter in `KimiMLP.__init__()`. ## Changes - Changed `quant_config` type from `QKVParallelLinear |...
- 实现要点: `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_linear.py` modified +1/-2 (3 lines); hunks: -22,7 +22,6; -61,7 +60,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -22,7 +22,6 @@
-    QKVParallelLinear,
@@ -61,7 +60,7 @@ def __init__(
-        quant_config: QKVParallelLinear | None = None,
+        quant_config: QuantizationConfig | None = None,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29309 - [XPU]fix Kimi-VL-A3B-thinking on xpu

- 链接: https://github.com/vllm-project/vllm/pull/29309
- 状态/时间: merged / 2025-11-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/moonvit.py`；关联提交 `3cfa63ad9916`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-6，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[XPU]fix Kimi-VL-A3B-thinking on xpu」，变更集中在 `vllm/model_executor/models/moonvit.py`。PR 描述补充为：## Purpose enable Kimi-VL-A3B-thinking text/image support on xpu. For image processing, we route to `flash_attn` backend and use `varlen_attention`. `torch.SDPA` path can't work...
- 实现要点: `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__，涉及 `multihead_attention, Rope2DPosEmb, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/moonvit.py` modified +14/-6 (20 lines); hunks: -56,10 +56,13; -106,10 +109,10 @@ def multihead_attention(; symbols: multihead_attention, Rope2DPosEmb, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -56,10 +56,13 @@
+from vllm.platforms import current_platform
+elif current_platform.is_xpu():
+    from vllm.attention.utils.fa_utils import flash_attn_varlen_func
@@ -106,10 +109,10 @@ def multihead_attention(
-        q_cu_seqlens,
-        k_cu_seqlens,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +14/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/moonvit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31207 - fix: update kimi k2 tool parser logic

- 链接: https://github.com/vllm-project/vllm/pull/31207
- 状态/时间: merged / 2025-12-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`；关联提交 `358bfd315cad`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+211/-202，可读 patch 511 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「fix: update kimi k2 tool parser logic」，变更集中在 `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`。PR 描述补充为：Purpose Fix streaming content leakage in Kimi-K2 tool parser. During streaming tool calls, the content field incorrectly contained tool call markers and content (e.g., functions...
- 实现要点: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin，涉及 `assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools`；`vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming，涉及 `_check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191 (383 lines); hunks: -44,6 +44,33 @@ def assert_tool_calls(; -346,61 +373,32 @@ def test_token_leak_between_section_and_tool_begin(kimi_k2...; symbols: assert_tool_calls, run_streaming_sequence, test_extract_tool_calls_no_tools, test_token_leak_between_section_and_tool_begin
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11 (30 lines); hunks: -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str,...; -238,6 +237,7 @@ def extract_tool_calls_streaming(; symbols: _check_and_strip_markers, _reset_section_state, extract_tool_calls_streaming
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -44,6 +44,33 @@ def assert_tool_calls(
+def run_streaming_sequence(parser, deltas):
+    """Helper to simulate a streaming sequence and return results."""
+    previous_text = ""
+    previous_token_ids: list[int] = []
+    results = []
+    for delta_text, delta_token_ids in deltas:
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -122,7 +122,6 @@ def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
@@ -238,6 +237,7 @@ def extract_tool_calls_streaming(
@@ -252,13 +252,18 @@ def extract_tool_calls_streaming(
-                remaining = buffered_text
-                # Return remaining text as reasoning content if non-empty
-                if remaining.strip():
-                    return DeltaMessage(content=remaining)
```

- 已读文件:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +192/-191
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +19/-11
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_kimi_k2_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31738 - [Models]: Use `MMEncoderAttention` for MoonViT

- 链接: https://github.com/vllm-project/vllm/pull/31738
- 状态/时间: merged / 2026-01-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`；关联提交 `7101e0851f73`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+72/-158，可读 patch 345 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 调整模型相关实现，标题为「[Models]: Use `MMEncoderAttention` for MoonViT」，变更集中在 `vllm/model_executor/models/moonvit.py`, `vllm/model_executor/models/kimi_vl.py`。PR 描述补充为：## Purpose - We missed MoonViT for Kimi-VL in https://github.com/vllm-project/vllm/pull/30125https://github.com/vllm-project/vllm/pull/30125 - This PR updates its attention inte...
- 实现要点: `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__，涉及 `multihead_attention, sdpa_attention, _apply_rope_input_validation`；`vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/moonvit.py` modified +71/-157 (228 lines); hunks: -51,118 +51,20; -411,11 +313,19 @@ def __init__(; symbols: multihead_attention, sdpa_attention, _apply_rope_input_validation, __init__
  - `vllm/model_executor/models/kimi_vl.py` modified +1/-1 (2 lines); hunks: -325,7 +325,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/moonvit.py
@@ -51,118 +51,20 @@
-from transformers.utils import is_flash_attn_2_available
+from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
+from vllm.config import MultiModalConfig
+from vllm.distributed import divide, get_tensor_model_parallel_world_size
-from vllm.model_executor.layers.linear import ReplicatedLinear
+from vllm.model_executor.layers.linear import (
diff -- vllm/model_executor/models/kimi_vl.py
@@ -325,7 +325,7 @@ def __init__(
-            self.use_data_parallel,
+            multimodal_config=model_config.multimodal_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/moonvit.py` modified +71/-157; `vllm/model_executor/models/kimi_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_vl.py`, `vllm/model_executor/models/moonvit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33131 - [Models] Kimi-K2.5

- 链接: https://github.com/vllm-project/vllm/pull/33131
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`, `vllm/transformers_utils/configs/kimi_k25.py`；关联提交 `b539f988e1ee`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1799/-8，可读 patch 2011 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Models] Kimi-K2.5」，变更集中在 `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/configs/kimi_k25.py`。PR 描述补充为：## Purpose Kimi-K2.5 model support - see recipe at https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html ## Test Plan ## Test Result --- Essential Elements o...
- 实现要点: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape，涉及 `_apply_rope_input_validation, get_rope_shape_decorate, wrapper`；`vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__，涉及 `MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor`；`vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size，涉及 `KimiK25VisionConfig, __init__, KimiK25Config`；`vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming，涉及 `KimiK2ReasoningParser, __init__, is_reasoning_end`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _apply_rope_input_validation, get_rope_shape_decorate, wrapper, get_rope_shape
  - `vllm/model_executor/models/kimi_k25.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: MaxImageTokenMeta, KimiK25MediaPixelInputs, MoonshotKimiVAutoProcessor, __init__
  - `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0 (129 lines); hunks: -0,0 +1,129; symbols: KimiK25VisionConfig, __init__, KimiK25Config, hidden_size
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: KimiK2ReasoningParser, __init__, is_reasoning_end, is_reasoning_end_streaming
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -0,0 +1,678 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Vision tower implementation for Kimi-K2.5 model.
+This module provides the vision encoder components for Kimi-K2.5,
+including 3D patch embedding, RoPE position embedding, and
diff -- vllm/model_executor/models/kimi_k25.py
@@ -0,0 +1,581 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# ruff: noqa: E501
+"""
+Kimi-K2.5 Model Implementation for vLLM.
+Kimi-K2.5 extends Kimi-K2 with vision support
diff -- vllm/transformers_utils/configs/kimi_k25.py
@@ -0,0 +1,129 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` added +678/-0; `vllm/model_executor/models/kimi_k25.py` added +581/-0; `vllm/transformers_utils/configs/kimi_k25.py` added +129/-0; `vllm/reasoning/kimi_k2_reasoning_parser.py` added +80/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33320 - [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…

- 链接: https://github.com/vllm-project/vllm/pull/33320
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `17b17c068453`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d…」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：commit msg: Replaced the hardcoded `torch.cuda.current_device()` with `current_platform.current_device()` in the `KimiK25ForConditionalGeneration` initialization. This change en...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -58,6 +58,7; -320,7 +321,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -58,6 +58,7 @@
+from vllm.platforms import current_platform
@@ -320,7 +321,7 @@ def __init__(
-        self.device = torch.cuda.current_device()
+        self.device = current_platform.current_device()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33346 - [Models] Refactor Kimi-K2.5 weight loading

- 链接: https://github.com/vllm-project/vllm/pull/33346
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`；关联提交 `8bfc8d5600ed`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+40/-176，可读 patch 282 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补强部署文档、测试或 CI 验证面，标题为「[Models] Refactor Kimi-K2.5 weight loading」，变更集中在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`。PR 描述补充为：## Purpose - Refactor Kim-K2.5 model interface usage to catch up previous refactoring ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input，涉及 `KimiK25ForConditionalGeneration, get_placeholder_str, __init__`；`vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +38/-174 (212 lines); hunks: -23,16 +23,7; -64,7 +55,12; symbols: KimiK25ForConditionalGeneration, get_placeholder_str, __init__, _parse_and_validate_media_input
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2 (4 lines); hunks: -660,13 +660,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,16 +23,7 @@
-from vllm.distributed import get_pp_group
-from vllm.model_executor.layers.fused_moe import SharedFusedMoE
-from vllm.model_executor.layers.logits_processor import LogitsProcessor
-from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -660,13 +660,13 @@ def __init__(
-            prefix=maybe_prefix(prefix, "linear_1"),
+            prefix=f"{prefix}.linear_1",
-            prefix=maybe_prefix(prefix, "linear_2"),
+            prefix=f"{prefix}.linear_2",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +38/-174; `vllm/model_executor/models/kimi_k25_vit.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33562 - [Bugfix] Enable Kimi k25 processor test

- 链接: https://github.com/vllm-project/vllm/pull/33562
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `4061dcf4c51a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+96/-12，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Bugfix] Enable Kimi k25 processor test」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose - Enable Kimi-K2.5 processor test - Add vision chunk to vision example ## Test Plan ## Test Result Test should pass --- Essential Elements of an Effective PR Descript...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__，涉及 `MoonshotKimiVAutoProcessor, __init__, __call__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +27/-5 (32 lines); hunks: -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):; -122,13 +126,30 @@ def __call__(; symbols: MoonshotKimiVAutoProcessor, __init__, __call__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -96,16 +96,20 @@ class MoonshotKimiVAutoProcessor(ProcessorMixin):
-    def __init__(self, media_processor=None, tokenizer=None):
+    def __init__(
+        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
+    ):
+        self.media_token_id = media_token_id
+        assert self.media_token_id is not None
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +27/-5
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- 链接: https://github.com/vllm-project/vllm/pull/33876
- 状态/时间: merged / 2026-02-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `a2522839d87d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-5，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose - `nvidia/Kimi-K2.5-NVFP4` is quantized based on legacy model layout (`language_model.layers.*`), which was refactored at #33346 - Since v0.15.0 has released, this PR...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits，涉及 `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -24,7 +24,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsMultiModal,
+    SupportsPP,
+    SupportsQuant,
+)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34427 - [Bugfix] Delete unused redundant code in Kimi-K2.5

- 链接: https://github.com/vllm-project/vllm/pull/34427
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `62788f99a4d0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-5，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Delete unused redundant code in Kimi-K2.5」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose ## Test Plan ## Test Result --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "Fix some issue (link existing issues...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +0/-5 (5 lines); hunks: -11,7 +11,6; -378,10 +377,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -11,7 +11,6 @@
-import copy
@@ -378,10 +377,6 @@ def __init__(
-        sub_vllm_config = copy.deepcopy(vllm_config)
-        sub_vllm_config.model_config.hf_config = (
-            sub_vllm_config.model_config.hf_config.text_config
-        )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +0/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34501 - [Bugfix] Add quant_config in ViT of Kimi-K2.5

- 链接: https://github.com/vllm-project/vllm/pull/34501
- 状态/时间: merged / 2026-02-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`；关联提交 `4a9952ec1b15`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-0，可读 patch 158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Bugfix] Add quant_config in ViT of Kimi-K2.5」，变更集中在 `vllm/model_executor/models/kimi_k25_vit.py`, `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose - In Kimi-K2.5, if the ViT is quantized, we need to transfer `quant_config` to the ViT module. ## Test Plan - We test the w4a8 weights including ViT quant. - The weig...
- 实现要点: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input，涉及 `__init__, _maybe_ignore_quant_config, _parse_and_validate_media_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0 (15 lines); hunks: -28,6 +28,7; -304,6 +305,7 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/kimi_k25.py` modified +11/-0 (11 lines); hunks: -23,6 +23,10; -361,6 +365,7 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, _parse_and_validate_media_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25_vit.py
@@ -28,6 +28,7 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
@@ -304,6 +305,7 @@ def __init__(
+        quant_config: QuantizationConfig | None = None,
@@ -314,13 +316,15 @@ def __init__(
+            quant_config=quant_config,
+            quant_config=quant_config,
diff -- vllm/model_executor/models/kimi_k25.py
@@ -23,6 +23,10 @@
+from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
+    CompressedTensorsConfig,
+)
@@ -361,6 +365,7 @@ def __init__(
+                quant_config=self._maybe_ignore_quant_config(quant_config),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25_vit.py` modified +15/-0; `vllm/model_executor/models/kimi_k25.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/kimi_k25_vit.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33646 - [Bugfix] Handle case when kimi ends reasoning with a tool call

- 链接: https://github.com/vllm-project/vllm/pull/33646
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/reasoning/kimi_k2_reasoning_parser.py`；关联提交 `9251ed5c4fc6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+230/-2，可读 patch 240 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Handle case when kimi ends reasoning with a tool call」，变更集中在 `vllm/reasoning/kimi_k2_reasoning_parser.py`。PR 描述补充为：## Purpose Kimi reasoning parser is currently based off Deepseeks. However, Kimi may start a tool call without using a think end token. When kimi ends with a tool call inside re...
- 实现要点: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end，涉及 `KimiK2ReasoningParser, __init__, _is_identity_mode`。
- 代码 diff 细节:
  - `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0 (228 lines); hunks: -0,0 +1,228; symbols: KimiK2ReasoningParser, __init__, _is_identity_mode, is_reasoning_end
- 关键代码摘录:

```diff
diff -- vllm/reasoning/kimi_k2_reasoning_parser.py
@@ -0,0 +1,228 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
```

- 已读文件:
  - runtime: `vllm/reasoning/kimi_k2_reasoning_parser.py` added +228/-0
- 验证与风险: runtime 路径改动集中在 `vllm/reasoning/__init__.py`, `vllm/reasoning/kimi_k2_reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- 链接: https://github.com/vllm-project/vllm/pull/36192
- 状态/时间: merged / 2026-03-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `00bd08edeee5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 调整模型相关实现，标题为「[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：Replace hardcoded trust_remote_code=True with the user's configured trust_remote_code setting from model_config in both nemotron_vl.py and kimi_k25.py. This prevents bypassing t...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +2/-1 (3 lines); hunks: -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -174,7 +174,8 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-            self.ctx.model_config.model, trust_remote_code=True
+            self.ctx.model_config.model,
+            trust_remote_code=self.ctx.model_config.trust_remote_code,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36127 - [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct

- 链接: https://github.com/vllm-project/vllm/pull/36127
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja`, `vllm/transformers_utils/processors/kimi_audio.py`；关联提交 `42fadebecb79`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+1446/-29，可读 patch 1583 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Model] Add support for moonshotai/Kimi-Audio-7B-Instruct」，变更集中在 `vllm/model_executor/models/kimi_audio.py`, `vllm/tokenizers/kimi_audio.py`, `vllm/transformers_utils/processors/kimi_audio.py`。PR 描述补充为：This PR adds support for the Kimi-Audio-7B-Instruct model from Moonshot AI, which is a state-of-the-art speech-to-text model combining Whisper encoder with Qwen2 LLM. ## Purpose...
- 实现要点: `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo，涉及 `_get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__`；`vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__，涉及 `_load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained`；`vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class，涉及 `_get_feat_extract_output_lengths, KimiAudioProcessor, __init__`；`vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config，涉及 `KimiAudioRenderer, from_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_audio.py` added +725/-0 (725 lines); hunks: -0,0 +1,725; symbols: _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__, KimiAudioProcessingInfo
  - `vllm/tokenizers/kimi_audio.py` added +410/-0 (410 lines); hunks: -0,0 +1,410; symbols: _load_tiktoken_encoding, KimiAudioTokenizer, from_pretrained, __init__
  - `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `vllm/renderers/kimi_audio.py` added +49/-0 (49 lines); hunks: -0,0 +1,49; symbols: KimiAudioRenderer, from_config
  - `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -0,0 +1,725 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
+import os
+from collections.abc import Iterable, Mapping, Sequence
+from typing import Any, ClassVar, Literal
diff -- vllm/tokenizers/kimi_audio.py
@@ -0,0 +1,410 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tokenizer for Kimi-Audio using TikToken."""
+import contextlib
+import json
+from pathlib import Path
diff -- vllm/transformers_utils/processors/kimi_audio.py
@@ -0,0 +1,163 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_audio.py` added +725/-0; `vllm/tokenizers/kimi_audio.py` added +410/-0; `vllm/transformers_utils/processors/kimi_audio.py` added +163/-0; `vllm/renderers/kimi_audio.py` added +49/-0; `vllm/transformers_utils/chat_templates/template_kimi_audio.jinja` added +13/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36361 - Kimi k2.5 MLA based eagle3

- 链接: https://github.com/vllm-project/vllm/pull/36361
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `557389473755`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+499/-8，可读 patch 649 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 调整模型相关实现，标题为「Kimi k2.5 MLA based eagle3」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose @IzzyPutterman is original author. This allows for Eagles that share MLA instead of GQA for attention, so one can train Eagle3s for Kimi and Deepseek and use them acr...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers，涉及 `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -28,6 +28,8 @@
+    SupportsEagle,
+    SupportsEagle3,
@@ -311,7 +313,12 @@ def split_video_chunks(self, video):
-    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant
+    nn.Module,
+    SupportsMultiModal,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36903 - [Misc] Clean up Kimi-audio whisper encoder loading

- 链接: https://github.com/vllm-project/vllm/pull/36903
- 状态/时间: merged / 2026-03-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_audio.py`；关联提交 `a8e8d62dd80f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+89/-116，可读 patch 382 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Misc] Clean up Kimi-audio whisper encoder loading」，变更集中在 `vllm/model_executor/models/kimi_audio.py`。PR 描述补充为：## Purpose - Add `subfolder` to `DefaultModelLoader.Source`, which can allow us to load model components from subfolder similar to vLLM-Omni's diffusion loader: https://github.c...
- 实现要点: `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__，涉及 `_get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_audio.py` modified +61/-111 (172 lines); hunks: -3,25 +3,21; -64,15 +60,6; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, KimiAudioWhisperEncoder, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_audio.py
@@ -3,25 +3,21 @@
-import os
-from huggingface_hub import snapshot_download
-from safetensors import safe_open
-from vllm.model_executor.model_loader.weight_utils import (
-    default_weight_loader,
-)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_audio.py` modified +61/-111
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/default_loader.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/kimi_audio.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37371 - standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01

- 链接: https://github.com/vllm-project/vllm/pull/37371
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_linear.py`；关联提交 `17808394bc48`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+235/-219，可读 patch 527 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01」，变更集中在 `vllm/model_executor/models/kimi_linear.py`。PR 描述补充为：FIX (partial) #15697 ## Test Plan Verified the refactor with a mock-weight loading script using a "Tiny Model Hack" (reducing layers to 1 for fast validation): Mock Weights: Gen...
- 实现要点: `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids，涉及 `forward, KimiLinearForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_linear.py` modified +97/-88 (185 lines); hunks: -46,6 +46,7; -472,94 +473,7 @@ def forward(; symbols: forward, KimiLinearForCausalLM, __init__, embed_input_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_linear.py
@@ -46,6 +46,7 @@
+    AutoWeightsLoader,
@@ -472,94 +473,7 @@ def forward(
-class KimiLinearForCausalLM(
-    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
-):
-    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_linear.py` modified +97/-88
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_linear.py`, `vllm/model_executor/models/minimax_text_01.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37438 - [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support

- 链接: https://github.com/vllm-project/vllm/pull/37438
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_kimi_k2_reasoning_parser.py`；关联提交 `c63ca2b2e696`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+173/-18，可读 patch 227 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补齐模型支持入口或运行时能力，标题为「[Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support」，变更集中在 `tests/reasoning/test_kimi_k2_reasoning_parser.py`。PR 描述补充为：## Summary Fixes https://github.com/vllm-project/vllm/issues/37397 Kimi-K2.5 (`model_type: kimi_k25`) reuses the same ` `/` ` reasoning format as Kimi-K2, but vLLM had several g...
- 实现要点: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags，涉及 `kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled`。
- 代码 diff 细节:
  - `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0 (155 lines); hunks: -0,0 +1,155; symbols: kimi_k2_tokenizer, test_parser_selection_thinking_enabled, test_parser_selection_thinking_disabled, test_extract_reasoning_with_think_tags
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_kimi_k2_reasoning_parser.py
@@ -0,0 +1,155 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.engine.protocol import DeltaMessage
+from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
```

- 已读文件:
  - tests: `tests/reasoning/test_kimi_k2_reasoning_parser.py` added +155/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_kimi_k2_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37693 - [Model] Update Kimi-K25 and Isaac processors to fit HF-style

- 链接: https://github.com/vllm-project/vllm/pull/37693
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/kimi_k25.py`；关联提交 `37aadf623786`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+128/-95，可读 patch 366 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 补强部署文档、测试或 CI 验证面，标题为「[Model] Update Kimi-K25 and Isaac processors to fit HF-style」，变更集中在 `vllm/transformers_utils/processors/kimi_k25.py`, `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Purpose Refactor processor logic to fit HF calling style. ## Test Plan `tests/models/multimodal/generation/test_common.py` passes for Isaac 0.2. (Isaac 0.1 is failing on main...
- 实现要点: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__，涉及 `KimiK25Processor, __init__, __call__`；`vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits，涉及 `KimiK25ProcessingInfo, __init__, get_hf_processor`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38 (92 lines); hunks: -1,38 +1,41; -42,31 +45,44 @@ def __call__(; symbols: KimiK25Processor, __init__, __call__
  - `vllm/model_executor/models/kimi_k25.py` modified +16/-18 (34 lines); hunks: -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):; -132,20 +138,15 @@ def get_supported_mm_limits(self) -> Mapping[str, int | No...; symbols: KimiK25ProcessingInfo, __init__, get_hf_processor, get_supported_mm_limits
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/kimi_k25.py
@@ -1,38 +1,41 @@
-import torch
-from transformers import BatchFeature
+from transformers import BaseImageProcessor, BatchFeature, TensorType
+from vllm.tokenizers.hf import HfTokenizer
-    attributes = ["tokenizer"]
-    tokenizer_class = "AutoTokenizer"
diff -- vllm/model_executor/models/kimi_k25.py
@@ -104,19 +104,25 @@ class KimiK25ProcessingInfo(BaseProcessingInfo):
-        self.hf_config = self.get_hf_config()
-        self.media_token_id = self.hf_config.media_placeholder_token_id
-        media_processor = cached_get_image_processor(
+        self.hf_config = hf_config = self.get_hf_config()
+        tokenizer = self.get_tokenizer()
+        image_processor = cached_get_image_processor(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/kimi_k25.py` modified +54/-38; `vllm/model_executor/models/kimi_k25.py` modified +16/-18
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/isaac.py`, `vllm/model_executor/models/kimi_k25.py`, `vllm/transformers_utils/processors/isaac.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39344 - fix(kimi_k25): resolve media_placeholder_token_id from tokenizer

- 链接: https://github.com/vllm-project/vllm/pull/39344
- 状态/时间: merged / 2026-04-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/kimi_k25.py`；关联提交 `17e787a7792b`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+24/-3，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「fix(kimi_k25): resolve media_placeholder_token_id from tokenizer」，变更集中在 `vllm/model_executor/models/kimi_k25.py`。PR 描述补充为：## Summary Kimi-K2.5 multimodal inference (images/video) is completely broken because `KimiK25Config.media_placeholder_token_id` (163605) disagrees with the tokenizer's actual m...
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement，涉及 `__init__, _get_prompt_updates, get_replacement`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +24/-3 (27 lines); hunks: -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:; -232,8 +254,7 @@ def _get_prompt_updates(; symbols: __init__, _get_prompt_updates, get_replacement
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -113,7 +113,29 @@ def __init__(self, ctx: InputProcessingContext) -> None:
-        self.media_token_id = media_token_id = hf_config.media_placeholder_token_id
+        # Resolve token ID from the tokenizer because transformers v5
+        # may remap token IDs vs config.json.
+        config_token_id = hf_config.media_placeholder_token_id
+        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
+        is_valid_resolved = isinstance(resolved_token_id, int) and (
```

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +24/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38579 - [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping

- 链接: https://github.com/vllm-project/vllm/pull/38579
- 状态/时间: merged / 2026-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`；关联提交 `03ce1c6ed908`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+684/-1405，可读 patch 2206 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Kimi K2/K2.5/Linear/VL 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping」，变更集中在 `tests/tool_parsers/test_kimi_k2_tool_parser.py`, `vllm/tool_parsers/kimi_k2_tool_parser.py`。PR 描述补充为：## Purpose Rewrites KimiK2ToolParser streaming to re-parse current_text on each streaming delta instead of tracking incremental token-ID counts. This eliminates the fragile stat...
- 实现要点: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls，涉及 `kimi_k2_tokenizer, kimi_k2_tool_parser, parser`；`vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state，涉及 `KimiK2ToolParser, __init__, _check_and_strip_markers`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921 (1446 lines); hunks: -3,14 +3,20; -20,959 +26,557 @@ def kimi_k2_tokenizer():; symbols: kimi_k2_tokenizer, kimi_k2_tool_parser, parser, assert_tool_calls
  - `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484 (643 lines); hunks: -1,6 +1,5; -17,137 +16,59; symbols: KimiK2ToolParser, __init__, _check_and_strip_markers, _reset_section_state
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_kimi_k2_tool_parser.py
@@ -3,14 +3,20 @@
+from unittest.mock import MagicMock
-from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
+from tests.tool_parsers.utils import (
+    run_tool_extraction,
+    run_tool_extraction_streaming,
+)
diff -- vllm/tool_parsers/kimi_k2_tool_parser.py
@@ -1,6 +1,5 @@
-# code modified from deepseekv3_tool_parser.py
@@ -17,137 +16,59 @@
+from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
+from vllm.tool_parsers.utils import partial_tag_overlap
-        self.current_tool_name_sent: bool = False
+        # Streaming state
```

- 已读文件:
  - tests: `tests/tool_parsers/test_kimi_k2_tool_parser.py` modified +525/-921
  - runtime: `vllm/tool_parsers/kimi_k2_tool_parser.py` modified +159/-484
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_kimi_k2_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
