# vllm Llama 4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。
- diffusion 相关模型已从本目录剔除，不再纳入模型优化 skill/history。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_llama4_json.jinja` | [#16428](https://github.com/vllm-project/vllm/pull/16428) |
| `examples/tool_chat_template_llama4_pythonic.jinja` | [#16463](https://github.com/vllm-project/vllm/pull/16463), [#17917](https://github.com/vllm-project/vllm/pull/17917) |
| `tests/models/multimodal/processing/test_llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113) |
| `tests/models/multimodal/processing/test_mllama4.py` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_llama4_pythonic_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` | [#25145](https://github.com/vllm-project/vllm/pull/25145), [#25889](https://github.com/vllm-project/vllm/pull/25889), [#26790](https://github.com/vllm-project/vllm/pull/26790), [#30709](https://github.com/vllm-project/vllm/pull/30709) |
| `vllm/model_executor/models/llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16311](https://github.com/vllm-project/vllm/pull/16311), [#16439](https://github.com/vllm-project/vllm/pull/16439), [#16512](https://github.com/vllm-project/vllm/pull/16512), [#16801](https://github.com/vllm-project/vllm/pull/16801), [#17315](https://github.com/vllm-project/vllm/pull/17315), [#19997](https://github.com/vllm-project/vllm/pull/19997), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#21499](https://github.com/vllm-project/vllm/pull/21499), [#22691](https://github.com/vllm-project/vllm/pull/22691), [#22701](https://github.com/vllm-project/vllm/pull/22701), ... (18 total) |
| `vllm/model_executor/models/llama4_eagle.py` | [#20591](https://github.com/vllm-project/vllm/pull/20591), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#27136](https://github.com/vllm-project/vllm/pull/27136), [#29926](https://github.com/vllm-project/vllm/pull/29926) |
| `vllm/model_executor/models/mllama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16201](https://github.com/vllm-project/vllm/pull/16201), [#16365](https://github.com/vllm-project/vllm/pull/16365), [#16746](https://github.com/vllm-project/vllm/pull/16746), [#18368](https://github.com/vllm-project/vllm/pull/18368), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#22021](https://github.com/vllm-project/vllm/pull/22021), [#22107](https://github.com/vllm-project/vllm/pull/22107), [#25961](https://github.com/vllm-project/vllm/pull/25961), [#28602](https://github.com/vllm-project/vllm/pull/28602), [#30709](https://github.com/vllm-project/vllm/pull/30709), [#35147](https://github.com/vllm-project/vllm/pull/35147) |
| `vllm/tool_parsers/llama4_pythonic_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 37
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 39
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-06 | [#16104](https://github.com/vllm-project/vllm/pull/16104) | merged | [Model] Support Llama4 in vLLM | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` |
| 2025-04-07 | [#16113](https://github.com/vllm-project/vllm/pull/16113) | merged | Upstream Llama4 Support to Main | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py` |
| 2025-04-07 | [#16201](https://github.com/vllm-project/vllm/pull/16201) | merged | [Misc] Move Llama 4 projector call into encoder execution | `vllm/model_executor/models/mllama4.py` |
| 2025-04-09 | [#16311](https://github.com/vllm-project/vllm/pull/16311) | merged | [BugFix] llama4 qknorm should be not shared across head | `vllm/model_executor/models/llama4.py` |
| 2025-04-10 | [#16365](https://github.com/vllm-project/vllm/pull/16365) | merged | [Model] Remove image mm limit for LLaMa4 | `vllm/model_executor/models/mllama4.py` |
| 2025-04-11 | [#16439](https://github.com/vllm-project/vllm/pull/16439) | merged | [Llama4] Enable attention temperature tuning by default for long context (>32k) | `vllm/model_executor/models/llama4.py` |
| 2025-04-11 | [#16463](https://github.com/vllm-project/vllm/pull/16463) | merged | [Frontend] Added chat templates for LLaMa4 pythonic tool calling | `examples/tool_chat_template_llama4_pythonic.jinja` |
| 2025-04-12 | [#16512](https://github.com/vllm-project/vllm/pull/16512) | merged | Optimized topk for topk=1 (Llama-4) | `vllm/model_executor/models/llama4.py` |
| 2025-04-18 | [#16801](https://github.com/vllm-project/vllm/pull/16801) | merged | [BugFix] Accuracy fix for llama4 int4 - improperly casted scales | `vllm/model_executor/models/llama4.py` |
| 2025-04-18 | [#16746](https://github.com/vllm-project/vllm/pull/16746) | merged | [Bugfix] fix pp for llama4 | `vllm/model_executor/models/mllama4.py` |
| 2025-04-24 | [#16428](https://github.com/vllm-project/vllm/pull/16428) | merged | Add chat template for Llama 4 models | `examples/tool_chat_template_llama4_json.jinja` |
| 2025-04-29 | [#17315](https://github.com/vllm-project/vllm/pull/17315) | merged | [model] make llama4 compatible with pure dense layers | `vllm/model_executor/models/llama4.py` |
| 2025-05-22 | [#17917](https://github.com/vllm-project/vllm/pull/17917) | merged | [Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser | `examples/tool_chat_template_llama4_pythonic.jinja` |
| 2025-06-02 | [#18368](https://github.com/vllm-project/vllm/pull/18368) | merged | [Model] enable data parallel for Llama4 vision encoder | `vllm/model_executor/models/mllama4.py` |
| 2025-06-25 | [#19997](https://github.com/vllm-project/vllm/pull/19997) | merged | [Llama4] Update `attn_temperature_tuning` | `vllm/model_executor/models/llama4.py` |
| 2025-07-12 | [#20419](https://github.com/vllm-project/vllm/pull/20419) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2025-07-16 | [#20591](https://github.com/vllm-project/vllm/pull/20591) | merged | [Meta] Llama4 EAGLE Support | `vllm/model_executor/models/llama4_eagle.py` |
| 2025-07-30 | [#21499](https://github.com/vllm-project/vllm/pull/21499) | merged | [NVIDIA] Fix Llama4 Scout FP4 functionality issues | `vllm/model_executor/models/llama4.py` |
| 2025-07-31 | [#20788](https://github.com/vllm-project/vllm/pull/20788) | merged | [Meta] Official Eagle mm support, first enablement on llama4 | `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py` |
| 2025-08-03 | [#22107](https://github.com/vllm-project/vllm/pull/22107) | merged | [Fix] Fix llama4 modelopt weight loading error | `vllm/model_executor/models/mllama4.py` |
| 2025-08-12 | [#22511](https://github.com/vllm-project/vllm/pull/22511) | merged | Fix Llama4 FlashInfer FP4 MoE issues | `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` |
| 2025-08-13 | [#22701](https://github.com/vllm-project/vllm/pull/22701) | merged | Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op | `vllm/model_executor/models/llama4.py` |
| 2025-08-19 | [#22691](https://github.com/vllm-project/vllm/pull/22691) | merged | [bug fix] Fix llama4 spec decoding | `vllm/model_executor/models/llama4.py` |
| 2025-08-28 | [#22021](https://github.com/vllm-project/vllm/pull/22021) | merged | Migrate Llama4ImagePatchInputs to TensorSchema | `vllm/model_executor/models/mllama4.py` |
| 2025-09-30 | [#25889](https://github.com/vllm-project/vllm/pull/25889) | merged | [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding` | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-10-06 | [#25961](https://github.com/vllm-project/vllm/pull/25961) | merged | Support llama3 eagle3 head with llama4 verifier | `vllm/model_executor/models/mllama4.py` |
| 2025-10-14 | [#26790](https://github.com/vllm-project/vllm/pull/26790) | merged | llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-10-21 | [#27136](https://github.com/vllm-project/vllm/pull/27136) | merged | [Fix][Spec Decode] Fix llama4 draft loading with different quantization | `vllm/model_executor/models/llama4_eagle.py` |
| 2025-10-30 | [#25145](https://github.com/vllm-project/vllm/pull/25145) | merged | [XPU][bugfix] fix rope for llama4 and deepseek | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-11-14 | [#28602](https://github.com/vllm-project/vllm/pull/28602) | merged | LLaMA4 LoRA Adapter Enablement | `vllm/model_executor/models/mllama4.py` |
| 2025-11-20 | [#28577](https://github.com/vllm-project/vllm/pull/28577) | merged | [BugFix] Fix Llama4 Pipeline Parallelism Assert Error | `vllm/model_executor/models/llama4.py` |
| 2025-12-05 | [#29926](https://github.com/vllm-project/vllm/pull/29926) | merged | [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute | `vllm/model_executor/models/llama4_eagle.py` |
| 2026-01-10 | [#30709](https://github.com/vllm-project/vllm/pull/30709) | merged | [Misc][LLaMa4] Compile LLaMa Vision Encoder | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2026-01-23 | [#32886](https://github.com/vllm-project/vllm/pull/32886) | merged | [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4 | `vllm/model_executor/models/llama4.py` |
| 2026-02-11 | [#34243](https://github.com/vllm-project/vllm/pull/34243) | merged | [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8) | `vllm/model_executor/models/llama4.py` |
| 2026-02-19 | [#34471](https://github.com/vllm-project/vllm/pull/34471) | merged | [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers | `vllm/model_executor/models/llama4.py` |
| 2026-02-21 | [#34997](https://github.com/vllm-project/vllm/pull/34997) | merged | Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers " | `vllm/model_executor/models/llama4.py` |
| 2026-02-23 | [#35033](https://github.com/vllm-project/vllm/pull/35033) | merged | [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests | `vllm/model_executor/models/llama4.py` |
| 2026-02-24 | [#35147](https://github.com/vllm-project/vllm/pull/35147) | merged | [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4) | `vllm/model_executor/models/mllama4.py` |

## 逐 PR diff 审计卡

### PR #16104 - [Model] Support Llama4 in vLLM

- 链接: https://github.com/vllm-project/vllm/pull/16104
- 状态/时间: merged / 2025-04-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+2369/-142，可读 patch 3141 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Model] Support Llama4 in vLLM」，变更集中在 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json`。PR 描述补充为：Add the support for Llama4 Scout (17B x 16 Experts) and Maverick (17B x 128 Experts) in vLLM. Using 8xH100, vLLM can serve Scout with 1M context and Maverick with about 430K. Us...
- 实现要点: `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward，涉及 `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`；`vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward，涉及 `Llama4MoE, custom_routing_function, __init__`；`vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200；`tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override，涉及 `test_processor_override`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
  - `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0 (68 lines); hunks: -851,6 +851,70 @@ def _compute_inv_freq(self, base: Union[int, float]) -> tor...; -1130,6 +1194,10 @@ def get_rope(; symbols: _compute_inv_freq, Llama4VisionRotaryEmbedding, __init__, _compute_cos_sin_cache
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -0,0 +1,886 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/models/llama4.py
@@ -0,0 +1,530 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json
@@ -0,0 +1,200 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` added +886/-0; `vllm/model_executor/models/llama4.py` added +530/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0; `vllm/model_executor/layers/fused_moe/layer.py` modified +41/-24; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +24/-14
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_llama4.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16113 - Upstream Llama4 Support to Main

- 链接: https://github.com/vllm-project/vllm/pull/16113
- 状态/时间: merged / 2025-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_llama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`；关联提交 `55dcce91df15`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 43 个文件，+2436/-155，可读 patch 3350 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Upstream Llama4 Support to Main」，变更集中在 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py`。PR 描述补充为：As a follow up of https://github.com/vllm-project/vllm/pull/16104, we upstream the Llama4 support to the main branch. The goal of this PR: - Support the llama4 - Clean up some h...
- 实现要点: `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward，涉及 `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`；`vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward，涉及 `Llama4MoE, custom_routing_function, __init__`；`tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override，涉及 `test_processor_override`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -0,0 +1,895 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/models/llama4.py
@@ -0,0 +1,531 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- tests/models/multimodal/processing/test_llama4.py
@@ -0,0 +1,99 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` added +895/-0; `vllm/model_executor/models/llama4.py` added +531/-0
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- 验证与风险: diff 自带测试面 `tests/models/decoder_only/audio_language/test_ultravox.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/test_phi3v.py`, `tests/models/decoder_only/vision_language/test_pixtral.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16201 - [Misc] Move Llama 4 projector call into encoder execution

- 链接: https://github.com/vllm-project/vllm/pull/16201
- 状态/时间: merged / 2025-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `ed636d99caa0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-3，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 调整模型相关实现，标题为「[Misc] Move Llama 4 projector call into encoder execution」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：On vLLM we consider encoder execution to generate multimodal embeddings ready to be merged into text embeddings, thus it often includes the projector call even though it's not p...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings，涉及 `_process_image_input, get_multimodal_embeddings, get_input_embeddings`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -760,6 +760,8 @@ def _process_image_input(
+        vision_embeddings_flat = self.multi_modal_projector(
+            vision_embeddings_flat)
@@ -791,10 +793,9 @@ def get_input_embeddings(
-            multimodal_embeddings = torch.cat(multimodal_embeddings)
-            mm_embeddings = self.multi_modal_projector(multimodal_embeddings)
-                input_ids, inputs_embeds, select_patch_features(mm_embeddings),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16311 - [BugFix] llama4 qknorm should be not shared across head

- 链接: https://github.com/vllm-project/vllm/pull/16311
- 状态/时间: merged / 2025-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `ec7da6fcf32f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-12，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[BugFix] llama4 qknorm should be not shared across head」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：QKNorm should not be across head, we need to do qknorm per head, score goes up from 0.6858 to 0.7153. This also resovles TP=4 Accuracy Issue https://github.com/vllm-project/vllm...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -155,14 +155,8 @@ def __init__(self,
-        self.q_norm = RMSNorm(
-            hidden_size=self.q_size,
-            eps=config.rms_norm_eps,
-            has_weight=False,
-            dtype=torch.float32,
-        ) if self.use_qk_norm else None
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16365 - [Model] Remove image mm limit for LLaMa4

- 链接: https://github.com/vllm-project/vllm/pull/16365
- 状态/时间: merged / 2025-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `61de3ef74b9c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-7，可读 patch 84 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补强部署文档、测试或 CI 验证面，标题为「[Model] Remove image mm limit for LLaMa4」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：Follow up in https://github.com/vllm-project/vllm/pull/16104 to remove the hard limit. Added some tests to allow testing more images in vision_language_multi_image.py. Default d...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk，涉及 `get_hf_processor, get_supported_mm_limits, get_patch_per_chunk`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Processor:
-        return {"image": 10}
+        # Although vLLM can support more images from an infra capability
+        # perspective, we do not recommend using >10 images in practice.
+        return {"image": None}
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16439 - [Llama4] Enable attention temperature tuning by default for long context (>32k)

- 链接: https://github.com/vllm-project/vllm/pull/16439
- 状态/时间: merged / 2025-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `99ef59cf7f93`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Llama4] Enable attention temperature tuning by default for long context (>32k)」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Attention temperature tuning (on nope layers) improves accuracy on long context (>32k) tasks. Enabling it by default for long context unless explicitly disabled by the user. Eva...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__，涉及 `Llama4ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):
-        # Update temperature tuning config from generation config
+        # update temperature tuning config from generation config
+        # enable temperature tuning by default when max_model_len > 32K
+        default_attn_temperature_tuning = \
+            vllm_config.model_config.max_model_len > 32768
-            = gen_config.get("attn_temperature_tuning", False)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16463 - [Frontend] Added chat templates for LLaMa4 pythonic tool calling

- 链接: https://github.com/vllm-project/vllm/pull/16463
- 状态/时间: merged / 2025-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_llama4_pythonic.jinja`；关联提交 `16eda8c43a9d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+182/-2，可读 patch 223 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Frontend] Added chat templates for LLaMa4 pythonic tool calling」，变更集中在 `examples/tool_chat_template_llama4_pythonic.jinja`。PR 描述补充为：# Summary Fixing tool calling related chat templates for llama4. LLaMa4 followed similar pythonic format. Co-authored with @wukaixingxp. Tool call messages and tool response mes...
- 实现要点: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139。
- 代码 diff 细节:
  - `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_llama4_pythonic.jinja
@@ -0,0 +1,139 @@
+{{- bos_token }}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools_in_user_message is defined %}
+    {%- set tools_in_user_message = false %}
```

- 已读文件:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/conftest.py`, `tests/tool_use/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16512 - Optimized topk for topk=1 (Llama-4)

- 链接: https://github.com/vllm-project/vllm/pull/16512
- 状态/时间: merged / 2025-04-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `bd6028d6b0bb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-2，可读 patch 31 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 优化关键推理路径或后端选择，标题为「Optimized topk for topk=1 (Llama-4)」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Clear speedup for latency case, adapted from https://github.com/sgl-project/sglang/commit/86a876d883a7c7a0e2b0fca5ef86e20ab92c0694 (thank you!) Llama Scout FP8 on 2xH100, input/...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function，涉及 `custom_routing_function`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -37,7 +37,7 @@
-from .utils import (AutoWeightsLoader, extract_layer_index,
+from .utils import (AutoWeightsLoader, extract_layer_index, fast_topk,
@@ -50,7 +50,7 @@ def custom_routing_function(
-        router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
+        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16801 - [BugFix] Accuracy fix for llama4 int4 - improperly casted scales

- 链接: https://github.com/vllm-project/vllm/pull/16801
- 状态/时间: merged / 2025-04-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `7eb42556281d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+6/-9，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[BugFix] Accuracy fix for llama4 int4 - improperly casted scales」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：A few MoE fixes: 1) Scales where fp16/bf16 but were being casted to float unsafely. Add a check by using `.data_ptr ()` 2) Fix `AttributeError: 'FusedMoE' object has no attribut...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__，涉及 `custom_routing_function, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -51,8 +51,8 @@ def custom_routing_function(
-        router_scores = torch.sigmoid(router_scores.float()).to(
-            hidden_states.dtype)
+        # psuedo-standard is that the router scores are floats
+        router_scores = torch.sigmoid(router_scores.float())
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16746 - [Bugfix] fix pp for llama4

- 链接: https://github.com/vllm-project/vllm/pull/16746
- 状态/时间: merged / 2025-04-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `e31045f95ca0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Bugfix] fix pp for llama4」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：This PR fixes PP for llama4 (https://github.com/vllm-project/vllm/issues/16385) Loading language model with architecture to support PP verification and correct the prefix to sep...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config.with_hf_config(config.text_config),
+            vllm_config=vllm_config.with_hf_config(config.text_config,
+                                                   ["LlamaForCausalLM"]),
@@ -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,
-            weights, prefix="language_model.model.")
+            weights, prefix="language_model.")
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16428 - Add chat template for Llama 4 models

- 链接: https://github.com/vllm-project/vllm/pull/16428
- 状态/时间: merged / 2025-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_llama4_json.jinja`；关联提交 `05e1fbfc52ca`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+139/-1，可读 patch 172 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Add chat template for Llama 4 models」，变更集中在 `examples/tool_chat_template_llama4_json.jinja`。PR 描述补充为：The main differences are that some token have changed, e.g. start_header_id was changed to header_start. Now the models also support multiple tool calls although one of our para...
- 实现要点: `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116。
- 代码 diff 细节:
  - `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_llama4_json.jinja
@@ -0,0 +1,116 @@
+{%- macro is_array_of_type_objects(var) -%}
+    {%- if var is iterable and var is not string -%}
+        {%- set valid = true -%}
+        {%- for item in var -%}
+            {%- if 'type' not in item -%}
+                {%- set valid = false -%}
```

- 已读文件:
  - docs: `examples/tool_chat_template_llama4_json.jinja` added +116/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17315 - [model] make llama4 compatible with pure dense layers

- 链接: https://github.com/vllm-project/vllm/pull/17315
- 状态/时间: merged / 2025-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `b4ac4fa04da1`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 调整模型相关实现，标题为「[model] make llama4 compatible with pure dense layers」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：make llama4 compatible with pure dense layers when interleave_moe_layer_step == 0 (no moe layers)
- 实现要点: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -273,8 +273,8 @@ def __init__(
-        is_moe_layer = (self.layer_idx +
-                        1) % config.interleave_moe_layer_step == 0
+        is_moe_layer = config.interleave_moe_layer_step > 0 and (
+            self.layer_idx + 1) % config.interleave_moe_layer_step == 0
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17917 - [Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser

- 链接: https://github.com/vllm-project/vllm/pull/17917
- 状态/时间: merged / 2025-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_llama4_pythonic.jinja`；关联提交 `c91fe7b1b9c4`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+541/-72，可读 patch 720 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser」，变更集中在 `examples/tool_chat_template_llama4_pythonic.jinja`。PR 描述补充为：Change the llama4 pythonic template and small fix on the edge case where llama4 model may output ` ` unexpectedly. BFCL test result: | Name | reported | Base_vllm | Pythonic_vll...
- 实现要点: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12。
- 代码 diff 细节:
  - `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_llama4_pythonic.jinja
@@ -1,85 +1,51 @@
-{%- if custom_tools is defined %}
+{%- if custom_tools is defined and custom_tools%}
-{%- if not tools_in_user_message is defined %}
-    {%- set tools_in_user_message = false %}
-{%- endif %}
-{%- if not tools is defined %}
```

- 已读文件:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/tool_use/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18368 - [Model] enable data parallel for Llama4 vision encoder

- 链接: https://github.com/vllm-project/vllm/pull/18368
- 状态/时间: merged / 2025-06-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `ebb1ec931871`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+214/-68，可读 patch 496 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Model] enable data parallel for Llama4 vision encoder」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：**Summary:** Llama4 vision encoder in dp8 is ~3x as fast as in tp8, especially when handling a large number of input images (eg. 9 images per request). Add an enable_vision_enco...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle，涉及 `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -34,6 +34,7 @@
+                                               ReplicatedLinear,
@@ -49,6 +50,7 @@
+from vllm.multimodal.utils import run_dp_sharded_vision_model
@@ -84,23 +86,29 @@ class Llama4ImagePatchInputs(TypedDict):
-    def __init__(self,
-                 input_size: int,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +167/-68
- 验证与风险: runtime 路径改动集中在 `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19997 - [Llama4] Update `attn_temperature_tuning`

- 链接: https://github.com/vllm-project/vllm/pull/19997
- 状态/时间: merged / 2025-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `1afa9948f593`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补强部署文档、测试或 CI 验证面，标题为「[Llama4] Update `attn_temperature_tuning`」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：## Purpose Since https://github.com/huggingface/transformers/pull/37501 landed It's the below on HF now: So we no longer need this comment ## Test Plan Start the model on TP=8 o...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -148,9 +148,8 @@ def __init__(self,
-        # TODO: attn_temperature_tuning should be a bool in huggingface
-            config.attn_temperature_tuning > 0
+            config.attn_temperature_tuning
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +1/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- 链接: https://github.com/vllm-project/vllm/pull/20419
- 状态/时间: merged / 2025-07-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`；关联提交 `4afe687a8291`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+501/-35，可读 patch 693 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Enable ModelOpt Llama4 fp8 checkpoint deployment」，变更集中在 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`。PR 描述补充为：## Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint，涉及 `Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights`；`vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint
  - `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMultiModal,
+        "gate_up_proj": ["gate_proj", "up_proj"],
@@ -902,32 +903,109 @@ def _consolidate_qkv_weights(
-    def load_weights(self, weights: Iterable[tuple[str,
-                                                   torch.Tensor]]) -> set[str]:
+    def _rename_weight_for_modelopt_checkpoint(self, name: str) -> str:
+        """Rename weights from ModelOpt llama4 fp8 checkpoints to vLLM
diff -- vllm/model_executor/models/llama4.py
@@ -35,7 +35,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader, maybe_remap_kv_scale_name)
@@ -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,
-                name = name.replace(weight_name, param_name)
+                # This check is for ModelOpt ckpts with kv cache quant enabled
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +144/-20; `vllm/model_executor/models/llama4.py` modified +55/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/model_loader/weight_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20591 - [Meta] Llama4 EAGLE Support

- 链接: https://github.com/vllm-project/vllm/pull/20591
- 状态/时间: merged / 2025-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4_eagle.py`；关联提交 `c11013db8b76`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+258/-18，可读 patch 363 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Meta] Llama4 EAGLE Support」，变更集中在 `vllm/model_executor/models/llama4_eagle.py`。PR 描述补充为：## Purpose Support EAGLE speculative decoding with dense-only draft model for Llama4, using official Meta based support Original Author: @zixi-qi ## Test Plan Ran with a uploade...
- 实现要点: `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights，涉及 `LlamaModel, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -0,0 +1,214 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` added +214/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/test_initialization.py`, `tests/v1/e2e/test_spec_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21499 - [NVIDIA] Fix Llama4 Scout FP4 functionality issues

- 链接: https://github.com/vllm-project/vllm/pull/21499
- 状态/时间: merged / 2025-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `ff08e51940a7`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+219/-70，可读 patch 432 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[NVIDIA] Fix Llama4 Scout FP4 functionality issues」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Fix the weight loading issues and accuray issues when using the NVIDIA ModelOpt Llama4 Scout FP4 model. ## Essential Elements of an Effective PR Description Checklist - [x] The...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights，涉及 `load_moe_expert_weights, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -342,34 +342,94 @@ def load_moe_expert_weights(
+        """
+        Load MoE expert weights.
+        Args:
+            name: The name of the weight to load.
+            loaded_weight: The weight to load.
+            params_dict: The dictionary of module parameters.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +205/-67
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20788 - [Meta] Official Eagle mm support, first enablement on llama4

- 链接: https://github.com/vllm-project/vllm/pull/20788
- 状态/时间: merged / 2025-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/llama4_eagle.py`；关联提交 `9e0726e5bfd2`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+206/-37，可读 patch 487 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Meta] Official Eagle mm support, first enablement on llama4」，变更集中在 `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py`。PR 描述补充为：## Purpose Enable MM inference for EAGLE, targeting mllama4 in this PR but generally easy to extend to other models Issue with this PR: MM chunked prefill needs to be disabled,...
- 实现要点: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights，涉及 `__init__, get_input_embeddings, forward`；`vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights
  - `vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -37,8 +37,9 @@
+from vllm.multimodal.inputs import NestedTensors
-from .utils import AutoWeightsLoader, maybe_prefix
+from .utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings
@@ -78,15 +79,23 @@ def __init__(
+    def get_input_embeddings(
+        self,
diff -- vllm/model_executor/models/llama4.py
@@ -256,6 +256,7 @@ def __init__(
+        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4; `vllm/model_executor/models/llama4.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/v1/e2e/test_spec_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22107 - [Fix] Fix llama4 modelopt weight loading error

- 链接: https://github.com/vllm-project/vllm/pull/22107
- 状态/时间: merged / 2025-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `337eb23bcca6`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-4，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Fix] Fix llama4 modelopt weight loading error」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：# Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [ ] The test p...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint，涉及 `_consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -906,11 +906,13 @@ def _consolidate_qkv_weights(
-        if name.startswith("model."):
+        if name.startswith("model.") or name.startswith(
+                "language_model.model."):
+            renamed = name.replace("model.", "language_model.model.",
+                                   1) if name.startswith("model.") else name
-                renamed = name.replace("model.", "language_model.model.", 1)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- 链接: https://github.com/vllm-project/vllm/pull/22511
- 状态/时间: merged / 2025-08-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+9/-5，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Fix Llama4 FlashInfer FP4 MoE issues」，变更集中在 `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`。PR 描述补充为：FlashInfer cutlass FP4 MoE already support Llama4, so remove unncessary asserts and set group to 0 when not used. # Essential Elements of an Effective PR Description Checklist -...
- 实现要点: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare，涉及 `prepare`；`vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply，涉及 `apply`；`vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply，涉及 `apply`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py
@@ -60,7 +60,12 @@ def prepare(
-        assert not apply_router_weight_on_input
+        if apply_router_weight_on_input:
+            topk = topk_ids.size(1)
+            # TODO: this only works for topK=1, will need to update for topK>1
+            assert topk == 1, \
+                "apply_router_weight_on_input is only implemented for topk=1"
diff -- vllm/model_executor/layers/quantization/modelopt.py
@@ -1299,8 +1299,9 @@ def apply(
-                n_group=num_expert_group,
-                topk_group=topk_group,
+                n_group=num_expert_group
+                if num_expert_group is not None else 0,
+                topk_group=topk_group if topk_group is not None else 0,
diff -- vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
@@ -170,8 +170,6 @@ def apply(
-        assert not apply_router_weight_on_input
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1; `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2; `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22701 - Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op

- 链接: https://github.com/vllm-project/vllm/pull/22701
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `4f0f844b1675`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：The rms_norm custom op cuda kernel supports non-contiguous striding on the "num_tokens" dim, but it assumes that all the num_tokens dims (if there are more than one num_tokens d...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -224,10 +224,14 @@ def forward(
-            q = q.reshape(-1, self.num_heads, self.head_dim)
+            # Normalization is applied on the head_dim dimension. The rest of
+            # the dimensions are collapsed into a single dimension to support
+            # custom rms_norm cuda kernel.
+            q = q.reshape(-1, self.head_dim)
-            k = k.reshape(-1, self.num_kv_heads, self.head_dim)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22691 - [bug fix] Fix llama4 spec decoding

- 链接: https://github.com/vllm-project/vllm/pull/22691
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `5bfe0dea7a34`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-2，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[bug fix] Fix llama4 spec decoding」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：# Essential Elements of an Effective PR Description Checklist - [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". - [x] The test p...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale，涉及 `__init__, _get_attn_scale`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -195,7 +195,9 @@ def __init__(self,
-        attn_cls = Attention if self.nope else ChunkedLocalAttention
+        use_chunked_local_attn = not self.nope and config.attention_chunk_size
+        attn_cls = (ChunkedLocalAttention
+                    if use_chunked_local_attn else Attention)
@@ -206,7 +208,7 @@ def __init__(self,
-            } if not self.nope else {}))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +4/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22021 - Migrate Llama4ImagePatchInputs to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/22021
- 状态/时间: merged / 2025-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `f32a5bc5058a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-18，可读 patch 87 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 调整模型相关实现，标题为「Migrate Llama4ImagePatchInputs to TensorSchema」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：## Purpose This PR migrates Llama4ImagePatchInputs from a TypedDict-based definition to a structured TensorSchema model with runtime shape validation. This brings it in line wit...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input，涉及 `Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -19,7 +19,7 @@
-from typing import Literal, Optional, TypedDict, Union
+from typing import Annotated, Literal, Optional, Union
@@ -53,35 +53,42 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
-class Llama4ImagePatchInputs(TypedDict):
-    type: Literal["pixel_values"]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-18
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25889 - [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`

- 链接: https://github.com/vllm-project/vllm/pull/25889
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；关联提交 `43b752c325d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`」，变更集中在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`。PR 描述补充为：## Purpose Fix #25888. Especially, this PR roll back the misplaced dtype cast introduced by #21126 which deteriorates the vision understanding performance of llama 4 family. Not...
- 实现要点: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native，涉及 `forward_native`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -59,7 +59,9 @@ def forward_native(  # type: ignore[override]
-        self._match_cos_sin_cache_dtype(query)
+        # self.cos_sin_cache here is complex tensor so we cannot cast into
+        # query's dtype directly with self._match_cos_sin_cache_dtype
+        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25961 - Support llama3 eagle3 head with llama4 verifier

- 链接: https://github.com/vllm-project/vllm/pull/25961
- 状态/时间: merged / 2025-10-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `05f6846ede18`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+83/-8，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「Support llama3 eagle3 head with llama4 verifier」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：## Summary This PR enables Eagle3 speculative decoding with Llama3 drafter and Llama4 multimodal verifier support, with configurable auxiliary hidden state layers. ## Key Featur...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers，涉及 `get_dummy_mm_data, Llama4ForConditionalGeneration, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -64,7 +64,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsEagle3,
+    SupportsMultiModal,
+    SupportsPP,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +25/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/llama_eagle3.py`, `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26790 - llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch

- 链接: https://github.com/vllm-project/vllm/pull/26790
- 状态/时间: merged / 2025-10-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；关联提交 `87efc681dbd5`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch」，变更集中在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`。PR 描述补充为：## Purpose On ROCm/HIP, the base RoPE path expects `(positions, query, key)` and, in the fallback branch, calls `forward_cuda(positions, query, key)`. The Llama4‑Vision RoPE imp...
- 实现要点: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip，涉及 `forward_cuda, forward_hip`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -78,3 +78,10 @@ def forward_cuda(  # type: ignore[override]
+    def forward_hip(  # type: ignore[override]
+        self,
+        query: torch.Tensor,
+        key: torch.Tensor | None = None,
+    ) -> tuple[torch.Tensor, torch.Tensor | None]:
+        return self.forward_native(query, key)
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27136 - [Fix][Spec Decode] Fix llama4 draft loading with different quantization

- 链接: https://github.com/vllm-project/vllm/pull/27136
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4_eagle.py`；关联提交 `be4445072c4e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-10，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Fix][Spec Decode] Fix llama4 draft loading with different quantization」，变更集中在 `vllm/model_executor/models/llama4_eagle.py`。PR 描述补充为：## Purpose This PR takes care of the case where draft model and target model may use different quantization configs. `replace` doesn't work with pydantic dataclass `deepcopy` fa...
- 实现要点: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -60,16 +60,23 @@ def __init__(
-        self.layers = nn.ModuleList(
-            [
-                Llama4DecoderLayer(
-                    vllm_config=vllm_config,
-                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
-                    config=self.config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4_eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25145 - [XPU][bugfix] fix rope for llama4 and deepseek

- 链接: https://github.com/vllm-project/vllm/pull/25145
- 状态/时间: merged / 2025-10-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；关联提交 `b798e39f931a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+22/-32，可读 patch 116 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[XPU][bugfix] fix rope for llama4 and deepseek」，变更集中在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`。PR 描述补充为：## Purpose Fix more dispatch issue on xpu introduced in https://github.com/vllm-project/vllm/pull/24444 ## Test Plan ## Test Result --- Essential Elements of an Effective PR Des...
- 实现要点: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip，涉及 `Llama4VisionRotaryEmbedding, __init__, forward_cuda`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -5,10 +5,10 @@
-from .base import RotaryEmbedding
+from .base import RotaryEmbeddingBase
-class Llama4VisionRotaryEmbedding(RotaryEmbedding):
+class Llama4VisionRotaryEmbedding(RotaryEmbeddingBase):
@@ -78,10 +78,3 @@ def forward_cuda(  # type: ignore[override]
-    def forward_hip(  # type: ignore[override]
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/rotary_embedding/base.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28602 - LLaMA4 LoRA Adapter Enablement

- 链接: https://github.com/vllm-project/vllm/pull/28602
- 状态/时间: merged / 2025-11-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `964d65deedb9`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-2，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「LLaMA4 LoRA Adapter Enablement」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：## Purpose Adds LoRA Adapter Support for LLaMA 4 Architecture. ## Test Plan Tested with custom LoRA and with vLLM Bench. ## Test Result Tests showed some performance degradation...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping，涉及 `get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -35,6 +35,7 @@
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -45,6 +46,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -68,11 +70,15 @@
+    SupportsLoRA,
-from .utils import AutoWeightsLoader, maybe_prefix
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +34/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28577 - [BugFix] Fix Llama4 Pipeline Parallelism Assert Error

- 链接: https://github.com/vllm-project/vllm/pull/28577
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `dc45efc8ef7f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[BugFix] Fix Llama4 Pipeline Parallelism Assert Error」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Summary: When enabling PP for Llama4 model, each rank has a few `PPMissingLayer()`. The line `assert isinstance(layer, Llama4DecoderLayer)` raises assert error when encountering...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata，涉及 `set_moe_parameters, update_physical_experts_metadata`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -54,6 +54,7 @@
+    PPMissingLayer,
@@ -738,6 +739,9 @@ def set_moe_parameters(self):
+            if isinstance(layer, PPMissingLayer):
+                continue
@@ -774,6 +778,9 @@ def update_physical_experts_metadata(
+            if isinstance(layer, PPMissingLayer):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29926 - [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute

- 链接: https://github.com/vllm-project/vllm/pull/29926
- 状态/时间: merged / 2025-12-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4_eagle.py`；关联提交 `962d703818c0`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+16/-3，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Bugfix][llama4_eagle] Fix missing 'lm_head' attribute」，变更集中在 `vllm/model_executor/models/llama4_eagle.py`。PR 描述补充为：This PR: 1. Fixes llama4_eagle model by adding the missing `lm_head` attribute to the model class. 2. Fixes `test_spec_decode.py::test_eagle_correctness[FLASH_ATTN-llama4_eagle]...
- 实现要点: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform，涉及 `__init__, transform`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -28,7 +28,10 @@
-from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
+from vllm.model_executor.layers.vocab_parallel_embedding import (
+    ParallelLMHead,
+    VocabParallelEmbedding,
+)
@@ -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2
- 验证与风险: diff 自带测试面 `tests/v1/e2e/test_spec_decode.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30709 - [Misc][LLaMa4] Compile LLaMa Vision Encoder

- 链接: https://github.com/vllm-project/vllm/pull/30709
- 状态/时间: merged / 2026-01-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`, `vllm/model_executor/models/mllama4.py`；关联提交 `ea6d067a2aeb`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+85/-20，可读 patch 202 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 调整模型相关实现，标题为「[Misc][LLaMa4] Compile LLaMa Vision Encoder」，变更集中在 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`。PR 描述补充为：## Purpose We want to speedup up inference for mllama4 by applying `torch.compile` to the intensive workload, similar to what is done in #23207. We start by enabling the VisionE...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__，涉及 `forward, Llama4VisionModel, __init__`；`vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native，涉及 `forward_native`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -31,9 +31,11 @@
-from vllm.config import VllmConfig
+from vllm.compilation.decorators import support_torch_compile
+from vllm.config import VllmConfig, set_current_vllm_config
+from vllm.forward_context import set_forward_context
@@ -47,6 +49,7 @@
+from vllm.model_executor.models.vision import should_torch_compile_mm_vit
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -60,14 +60,17 @@ def forward_native(  # type: ignore[override]
-        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
+        # NOTE: by not storing cos_sin_cache in self, we can avoid
+        # memory buffer update which is costly to runtime
+        cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
-        freqs_ci = self.cos_sin_cache.view(*broadcast_shape)
+        freqs_ci = cos_sin_cache.view(*broadcast_shape)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +29/-9; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2
- 验证与风险: diff 自带测试面 `tests/compile/fullgraph/test_multimodal_compile.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32886 - [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4

- 链接: https://github.com/vllm-project/vllm/pull/32886
- 状态/时间: merged / 2026-01-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `1fb648bf107e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+21/-1，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：## Purpose #32862 Add a version-guarded fallback in Llama4 MoE weight loading to avoid CPU FP8 indexing on older PyTorch releases. For torch Essential Elements of an Effective P...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights，涉及 `load_moe_expert_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -51,6 +51,8 @@
+from vllm.platforms import current_platform
+from vllm.utils.torch_utils import is_torch_equal_or_newer
@@ -504,7 +506,25 @@ def load_moe_expert_weights(
-                    new_loaded_weight = new_loaded_weight[local_expert_indices]
+                    # Workaround for FP8 CPU indexing on older PyTorch:
+                    # https://github.com/vllm-project/vllm/issues/32862
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +21/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34243 - [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)

- 链接: https://github.com/vllm-project/vllm/pull/34243
- 状态/时间: merged / 2026-02-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `11c7ace34061`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-5，可读 patch 76 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Llama-4 weights of `q/k_proj` are permuted during model loading to prepare the model for interleaved/gpt-neox rope. The same permutation needs to be applied on quantization weig...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute，涉及 `permute_qk_weight_for_rotary, permute`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,6 +44,9 @@
+from vllm.model_executor.layers.quantization.compressed_tensors import (
+    compressed_tensors as ct,
+)
@@ -829,11 +832,20 @@ def permute_qk_weight_for_rotary(
-        def permute(w: torch.Tensor, n_heads: int, is_weight_scale: bool):
+        def permute(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34471 - [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers

- 链接: https://github.com/vllm-project/vllm/pull/34471
- 状态/时间: merged / 2026-02-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `ee1d25f199ee`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-68，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：This PR resolves a couple of issues in loading of Llama-4 models with quantized self-attention layers. Current logic hardcodes support only for NVFP4 modelopt ckpts, and for INT...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute，涉及 `permute_qk_weight_for_rotary, permute`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,9 +44,6 @@
-from vllm.model_executor.layers.quantization.compressed_tensors import (
-    compressed_tensors as ct,
-)
@@ -831,74 +828,38 @@ def permute_qk_weight_for_rotary(
-        # Helper function to permute the weight's channels
-        def permute(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34997 - Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers "

- 链接: https://github.com/vllm-project/vllm/pull/34997
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `0e22cd618b5d`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+68/-29，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers "」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Reverts vllm-project/vllm#34471 to fix CI failures: https://github.com/vllm-project/vllm/issues/34995 FIX: https://github.com/vllm-project/vllm/issues/34995 cc @eldarkurtic
- 实现要点: `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute，涉及 `permute_qk_weight_for_rotary, permute`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,6 +44,9 @@
+from vllm.model_executor.layers.quantization.compressed_tensors import (
+    compressed_tensors as ct,
+)
@@ -828,38 +831,74 @@ def permute_qk_weight_for_rotary(
-        modules = name.split(".")
-        # Permute Q/K weights and corresponding scales for rotary embedding.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +68/-29
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35033 - [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests

- 链接: https://github.com/vllm-project/vllm/pull/35033
- 状态/时间: merged / 2026-02-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/llama4.py`；关联提交 `1e8438a89a64`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+31/-70，可读 patch 127 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 修复已暴露的启动、加载、解析或数值问题，标题为「[Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests」，变更集中在 `vllm/model_executor/models/llama4.py`。PR 描述补充为：Bring back reverted Llama-4 bug fix changes from https://github.com/vllm-project/vllm/pull/34471 (reverted with https://github.com/vllm-project/vllm/pull/34997) and also fix the...
- 实现要点: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute，涉及 `permute_qk_weight_for_rotary, permute`。
- 代码 diff 细节:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,9 +44,6 @@
-from vllm.model_executor.layers.quantization.compressed_tensors import (
-    compressed_tensors as ct,
-)
@@ -831,74 +828,38 @@ def permute_qk_weight_for_rotary(
-        # Helper function to permute the weight's channels
-        def permute(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_maverick.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35147 - [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)

- 链接: https://github.com/vllm-project/vllm/pull/35147
- 状态/时间: merged / 2026-02-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/mllama4.py`；关联提交 `012dee92331c`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-1，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 该 PR 围绕 Llama 4 补齐模型支持入口或运行时能力，标题为「[Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)」，变更集中在 `vllm/model_executor/models/mllama4.py`。PR 描述补充为：## Purpose Enable LoRA adapters for the vision tower and connector of Llama 4 Vision (`Llama4ForConditionalGeneration` / `mllama4.py`), as part of #31479. Previously, LoRA could...
- 实现要点: `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens，涉及 `get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens`。
- 代码 diff 细节:
  - `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:
-            connector="multi_modal_projector.",
+            connector=[
+                "multi_modal_projector.",
+                "vision_model.vision_adapter.",
+            ],
+    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/mllama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 本版不再接受只列 PR 标题的写法；每个 PR 必须有反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
